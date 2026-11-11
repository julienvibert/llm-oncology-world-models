#!/usr/bin/env python3
"""N14 (exploratory): does the gain from matched examples depend on how closely they match?

Specification: manuscript/specifications/2026-11-10_N14_support_gradient_spec.md. Nothing here chooses a threshold, measure or horizon.

Primary: per-patient paired gain in the 12-month whole-course joint score (energy score on 17
summaries, primary convention, equal-size fans; the quantity of Fig. 5d, 233 patients),
gain = score(generic examples) - score(matched examples); positive = matching helped beyond cohort
examples. Spearman correlation with support_boundary (the clinical agreement score of the 20th,
weakest admitted matched donor) ranked within cancer (average ranks / cancer size), pooled.
95% interval from 2,000 patient bootstrap resamples stratified by cancer (seed 20261110);
two-sided P from 2,000 permutations of the support ranks within cancer.
Secondary (one): gain = score(record alone) - score(matched examples).
Display: mean gain with bootstrap interval in within-cancer thirds of support_boundary, formed so
that tied scores stay together; per third the share of the cohort's matching criteria that are
unknown for the patient (from the generation-time matcher cards, reconstructed and checked).

Before any result, two verifications are run and recorded in the manifest:
  1. queries.jsonl is the file the four-arm prompts were built from (SHA-256 vs manifests);
  2. how unknown values and the molecular term enter the generation-time similarity (code at the
     generation commit), checked by recomputing every query's support_boundary and card mean.

GOVERNANCE. GENIE-BPC under DUA. Patient-level rows are read in memory only; every output is an
aggregate, and no reported cell has 1-4 patients (asserted).

Run: uv run --offline python scripts/explore/n14_support_gradient_v1.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

UNIVERSE = ROOT / "outputs" / "exploratory_query_universe"
QUERIES = UNIVERSE / "queries.jsonl"
FINAL = ROOT / "outputs" / "four_arm_preprint_20260928"
ROWS = FINAL / "private" / "joint_patient_rows_scores_20261012.csv"
PUBLISHED = FINAL / "joint_scores_scores_20261012.csv"
OUT = ROOT / "outputs" / "exploratory_n14_support_gradient"
GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE", "CRC": "CRC_GENIE"}
CANCERS = ("NSCLC", "BRCA", "CRC")

HORIZON = 12.0
CONVENTION = "primary"
QUANTITY = "equal_size"
N_EXPECTED = 233
SEED = 20261110
N_BOOT = 2000
N_PERM = 2000
MIN_CELL = 5
MATCHED, GENERIC, RECORD = "patients_like_me", "generic_cohort", "patient_only"
CONTRASTS = {
    "primary": {"label": "generic minus matched", "minus": GENERIC, "published_row": "contrast:4-3"},
    "secondary": {"label": "record alone minus matched", "minus": RECORD, "published_row": "contrast:4-1"},
}
SCRIPTS_AT_GENERATION = ("scripts/explore/clinical_match_v1.py",
                         "scripts/explore/build_query_universe_v1.py",
                         "scripts/explore/disease_extent_v1.py",
                         "scripts/explore/genie_operator_registry_v2.py")


# ---------------------------------------------------------------- pure helpers (unit-tested)

def within_group_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average rank of each value within its group, divided by the group size (ties share a rank)."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    out = np.full(len(values), np.nan)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = rankdata(values[idx], method="average") / len(idx)
    return out


def within_group_thirds(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Thirds (1 = lowest values) within each group, from average ranks so ties stay together.

    A value with average rank r among n goes to third floor(3 (r - 1/2) / n) + 1. Without ties
    this splits n into near-equal thirds; with ties every patient with the same value has the
    same r and therefore the same third, so group sizes may be unequal.
    """
    values, groups = np.asarray(values, float), np.asarray(groups)
    out = np.zeros(len(values), int)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        r = rankdata(values[idx], method="average")
        out[idx] = np.minimum(3, np.floor(3.0 * (r - 0.5) / len(idx)).astype(int) + 1)
    return out


def paired_gain(scores: pd.DataFrame, minus_arm: str, ref_arm: str = MATCHED) -> pd.Series:
    """score(minus_arm) - score(ref_arm) per patient (wide frame, one column per arm).

    Scores are losses (lower is better), so a positive gain means the reference arm did better.
    """
    return scores[minus_arm] - scores[ref_arm]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties."""
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    return float(np.corrcoef(rx, ry)[0, 1])


def pooled_rho(gain: np.ndarray, support: np.ndarray, cancer: np.ndarray) -> float:
    """Spearman of the gain with support ranked within cancer, pooled over cancers."""
    return spearman(gain, within_group_rank(support, cancer))


# ---------------------------------------------------------------- provenance

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def verify_universe_hash() -> dict:
    """Verification 1: queries.jsonl is the file the four-arm prompts were built from."""
    digest = sha256(QUERIES)
    universe = json.loads((UNIVERSE / "manifest.json").read_text())
    prompts = json.loads((ROOT / "outputs" / "exploratory_four_arm_prompts"
                          / "provenance.json").read_text())
    scores = json.loads((FINAL / "manifest_scores_20261012.json").read_text())
    run = json.loads((ROOT / "runs" / "four_arm_v1_definitive_p40b20_20260826"
                      / "manifest.json").read_text())
    checks = {
        "queries_jsonl_sha256": digest,
        "universe_builder_manifest_queries_sha256_truncated":
            universe["queries_sha256"],
        "universe_builder_commit": universe["git_commit"],
        "prompt_emitter_provenance_query_universe_sha256": prompts["query_universe_sha256"],
        "prompt_emitter_commit": prompts["git_commit"],
        "generation_run": "runs/four_arm_v1_definitive_p40b20_20260826",
        "generation_run_universe_checksum_truncated": run["universe_checksum"],
        "generation_run_commit": run["git_commit"],
        "scoring_manifest_query_universe_digest":
            scores["input_checks"]["query_universe_digest"],
        "scoring_manifest_generation_commit": scores["generation_commit"],
    }
    ok = (digest == prompts["query_universe_sha256"]
          == scores["input_checks"]["query_universe_digest"]
          and digest.startswith(universe["queries_sha256"])
          and digest.startswith(run["universe_checksum"])
          and universe["git_commit"] == prompts["git_commit"] == run["git_commit"]
          == scores["generation_commit"])
    checks["verified"] = bool(ok)
    return checks


def verify_matcher_code(commit: str) -> dict:
    """Verification 2a: the matcher code at the generation commit (read from git, not comments)."""
    out: dict = {"generation_commit": commit, "files": {}}
    for f in SCRIPTS_AT_GENERATION:
        at_gen = git("show", f"{commit}:{f}")
        now = (ROOT / f).read_text()
        out["files"][f] = {
            "sha256_at_generation_commit": hashlib.sha256(at_gen.encode()).hexdigest(),
            "identical_to_working_copy": at_gen == now}
    builder = git("show", f"{commit}:scripts/explore/build_query_universe_v1.py")
    matcher = git("show", f"{commit}:scripts/explore/clinical_match_v1.py")
    out["molecular_offset_calls_in_builder"] = builder.count("molecular_offset(") - 1
    out["similarity_called_with_cohort_keep_set"] = (
        builder.count("CM.similarity(records[qi][\"card\"], records[j][\"card\"], keep)") == 2)
    out["fixed_denominator_rule_present"] = (
        "fixed = len(keep) if keep else len(detail)" in matcher)
    out["molecular_driver_is_a_criterion"] = '"molecular driver": (None if "unknown"' in matcher
    out["registry_drift_note"] = (
        "genie_operator_registry_v2.py differs from the generation commit only in "
        "future_outcomes (n_new_metastatic_sites_future NaN where sites are not collected), "
        "which the matcher card does not use; the reconstruction check below confirms that "
        "every support_boundary and card mean is reproduced exactly.")
    out["reading"] = (
        "At the generation commit, similarity(a, b, keep) = sum over the comparable criteria in "
        "the cohort's fixed keep set of (+1 agree, -1 disagree) divided by len(keep). A criterion "
        "unknown on either side is dropped from the numerator only, so it contributes 0 and does "
        "not shrink the denominator. The molecular driver is one ordinary criterion with the same "
        "+1/-1/0 rule (unknown never matches unknown). molecular_offset (the additive +/-1 "
        "adjustment described in the builder's support_rule/molecular_rule manifest text) is "
        "defined but never called, so that text is stale. Within a cancer every patient is "
        "scored on the same criteria with the same denominator (NSCLC 11, CRC 10, BRCA 8), so "
        "support_boundary is on one scale within a cancer; across cancers the criteria and "
        "denominators differ, hence the within-cancer ranks. Within a cancer, a patient with "
        "more unknown criteria has fewer criteria that can score +1, so a lower boundary can "
        "partly reflect sparser documentation.")
    return out


# ---------------------------------------------------------------- cards (generation-time matcher)

def reconstruct_cards(queries: list[dict], keep: dict[str, frozenset[str]]) -> pd.DataFrame:
    """Rebuild each query's matcher card and check support_boundary / card mean against the file.

    Returns, per query, the share of the cohort's criteria unknown for the patient and the
    recomputed boundary and mean. Patient-level, kept in memory only.
    """
    import clinical_match_v1 as CM
    import disease_extent_v1 as DE
    import genie_operator_registry_v2 as GV2

    rows = []
    for cancer, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        patients = {p.pid: p for p in GV2.load(genie)}
        mets = DE.load_metastasis(genie)

        def card(pid: str, tau: float, driver: str):
            return CM.describe(patients[pid], tau, driver, mets.get(pid))

        for q in (q for q in queries if q["cancer"] == cancer):
            c = card(q["patient"], q["tau_months"], q["molecular_driver"])
            sims = np.array([CM.similarity(c, card(d["patient"], d["tau_months"],
                                                   d["molecular_driver"]), keep[cancer])
                             for d in q["relevant_donors"]])
            unknown = query_side_unknown(c)
            rows.append({"query_id": q["query_id"], "cancer": cancer,
                         "boundary_recomputed": float(sims.min()),
                         "mean_recomputed": float(sims.mean()),
                         "share_unknown": float(np.mean([unknown[k] for k in keep[cancer]]))})
    return pd.DataFrame(rows)


def query_side_unknown(c) -> dict[str, bool]:
    """Which of the matcher's criteria cannot be compared because the patient's own value is unknown
    (the None rules of clinical_match_v1.agreements, read from the patient side only)."""
    return {
        "cancer present at the landmark": c.presence in (None, "equivocal"),
        "stage at diagnosis": c.stage_iv is None,
        "metastatic disease by landmark": c.mets_documented == "unknown",
        "molecular driver": c.driver == "unknown",
        "disease state at landmark": c.status is None,
        "regimen sequence": c.depth == "unknown",
        "time to this treatment": not c.tempo,
        "sex": c.female is None,
        "age within a decade": c.age is None,
        "metastatic sites overlap": c.sites is None,
        "histology": c.histology is None,
    }


# ---------------------------------------------------------------- statistics

def stratified_indices(cancer: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for g in CANCERS:
        here = np.flatnonzero(cancer == g)
        idx.append(rng.choice(here, size=len(here), replace=True))
    return np.concatenate(idx)


def correlation(gain: np.ndarray, support: np.ndarray, cancer: np.ndarray) -> dict:
    rho = pooled_rho(gain, support, cancer)
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = stratified_indices(cancer, rng)
        boot[b] = pooled_rho(gain[i], support[i], cancer[i])
    rng = np.random.default_rng(SEED)
    ranks = within_group_rank(support, cancer)
    perm = np.empty(N_PERM)
    for b in range(N_PERM):
        r = ranks.copy()
        for g in CANCERS:
            here = np.flatnonzero(cancer == g)
            r[here] = rng.permutation(r[here])
        perm[b] = spearman(gain, r)
    p = (1 + int(np.sum(np.abs(perm) >= abs(rho) - 1e-12))) / (N_PERM + 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi), "p_permutation": p,
            "n_boot_finite": int(np.isfinite(boot).sum())}


def third_means(gain: np.ndarray, third: np.ndarray, cancer: np.ndarray) -> list[dict]:
    """Mean gain per third, interval from resampling patients within cancer x third (fixed n)."""
    rng = np.random.default_rng(SEED)
    out = []
    for t in (1, 2, 3):
        cells = [np.flatnonzero((third == t) & (cancer == g)) for g in CANCERS]
        idx = np.concatenate(cells)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            boot[b] = gain[np.concatenate([rng.choice(c, len(c), replace=True)
                                           for c in cells if len(c)])].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"third": t, "n": int(len(idx)), "mean_gain": float(gain[idx].mean()),
                    "ci_low": float(lo), "ci_high": float(hi),
                    **{f"n_{g}": int(len(c)) for g, c in zip(CANCERS, cells, strict=True)}})
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", "scripts/explore/n14_support_gradient_v1.py").strip())

    # --- verifications, before any outcome is read
    v1 = verify_universe_hash()
    assert v1["verified"], "queries.jsonl is not the file recorded by the four-arm manifests"
    v2 = verify_matcher_code(v1["generation_run_commit"])
    queries = [json.loads(line) for line in QUERIES.open()]
    uni = json.loads((UNIVERSE / "manifest.json").read_text())
    keep = {c: frozenset(uni["cohorts"][c]["match_criteria"]) for c in CANCERS}
    q = pd.DataFrame([{k: r[k] for k in ("query_id", "cancer", "support_boundary",
                                         "mean_agreement_relevant", "n_candidates")}
                      for r in queries])
    cards = reconstruct_cards(queries, keep)
    q = q.merge(cards, on=["query_id", "cancer"], validate="one_to_one")
    v2["reconstruction"] = {
        "queries": int(len(q)),
        "support_boundary_reproduced": int(np.isclose(q.boundary_recomputed,
                                                      q.support_boundary, atol=1e-12).sum()),
        "mean_agreement_relevant_reproduced": int(np.isclose(q.mean_recomputed,
                                                             q.mean_agreement_relevant,
                                                             atol=1e-12).sum()),
        "criteria_per_cancer": {c: sorted(keep[c]) for c in CANCERS},
        "denominator_per_cancer": {c: len(keep[c]) for c in CANCERS},
        "distinct_boundary_values_per_cancer": {
            c: int(q.loc[q.cancer == c, "support_boundary"].nunique()) for c in CANCERS},
    }
    assert v2["reconstruction"]["support_boundary_reproduced"] == len(q), \
        "generation-time matcher could not reproduce support_boundary"
    share_unknown_ok = v2["reconstruction"]["mean_agreement_relevant_reproduced"] == len(q)

    # --- outcome: per-patient joint score, Fig. 5d population
    d = pd.read_csv(ROWS, float_precision="round_trip")
    d = d[(d.horizon_months == HORIZON) & (d.convention == CONVENTION)]
    wide = d.pivot(index="query_id", columns="arm", values=QUANTITY)
    wide = wide[[RECORD, "setting", GENERIC, MATCHED]].dropna()
    assert len(wide) == N_EXPECTED, len(wide)
    pub = pd.read_csv(PUBLISHED, float_precision="round_trip")
    pub = pub[(pub.quantity == QUANTITY) & (pub.convention == CONVENTION)
              & (pub.horizon_months == HORIZON) & (pub.cancer == "all")].set_index("row")
    repro = {}
    for name, spec in CONTRASTS.items():
        later_minus_earlier = -paired_gain(wide, spec["minus"])   # matched - other, as published
        published = pub.loc[spec["published_row"]]
        repro[name] = {"published_row": spec["published_row"],
                       "published_mean": float(published["mean"]),
                       "published_n": int(published["n"]),
                       "recomputed_mean": float(later_minus_earlier.mean()),
                       "recomputed_n": int(len(later_minus_earlier)),
                       "reproduced": bool(np.isclose(later_minus_earlier.mean(),
                                                     published["mean"], rtol=0, atol=1e-12)
                                          and len(later_minus_earlier) == published["n"])}
        assert repro[name]["reproduced"], repro[name]

    a = wide.join(q.set_index("query_id"), how="left")
    assert a.support_boundary.notna().all() and len(a) == N_EXPECTED
    cancer = a.cancer.to_numpy()
    support = a.support_boundary.to_numpy(float)
    third = within_group_thirds(support, cancer)

    stats, summary = {}, []
    for name, spec in CONTRASTS.items():
        gain = paired_gain(a, spec["minus"]).to_numpy(float)
        stats[name] = {"contrast": spec["label"], "n": int(len(gain)),
                       "mean_gain": float(gain.mean()), **correlation(gain, support, cancer)}
        for row in third_means(gain, third, cancer):
            sel = third == row["third"]
            extra = {"share_unknown_mean": float(a.share_unknown[sel].mean())}
            for g in CANCERS:
                cell = sel & (cancer == g)
                extra[f"support_boundary_median_{g}"] = float(np.median(support[cell]))
            summary.append({"analysis": name, "contrast": spec["label"], **row, **extra})
    s = pd.DataFrame(summary)
    cell_cols = [f"n_{g}" for g in CANCERS] + ["n"]
    assert (s[cell_cols] >= MIN_CELL).all().all(), "a reported cell has fewer than 5 patients"
    s.to_csv(OUT / "n14_summary_v1.csv", index=False)

    result = {
        "what": "N14 exploratory, post hoc: paired gain from matched examples vs clinical "
                "agreement of the weakest matched donor (support_boundary)",
        "population": {"n": N_EXPECTED, **{g: int((cancer == g).sum()) for g in CANCERS},
                       "definition": "patients scored at 12 months, primary convention, "
                                     "equal-size fans (Fig. 5d population)"},
        "outcome": "whole-course joint score at 12 months (energy score, 17 summaries, lower is "
                   "better), equal_size, primary convention",
        "measure": "support_boundary ranked within cancer (average ranks / cancer size)",
        "thirds_rule": "within cancer, third = floor(3 (r - 1/2) / n) + 1 from average ranks; "
                       "tied scores share a third; third 1 = weakest agreement",
        "thirds_interval": "2,000 bootstrap resamples within cancer x third (fixed n)",
        "correlation_interval": "2,000 patient bootstrap resamples stratified by cancer, "
                                "within-cancer ranks recomputed in each resample",
        "permutation": "2,000 permutations of the support ranks within cancer; two-sided, "
                       "P = (1 + #|rho_perm| >= |rho_obs|) / 2001",
        "seed": SEED,
        "share_unknown_definition": "share of the cancer's fixed matching criteria whose value "
                                    "is unknown for the patient (query side), from the "
                                    "reconstructed generation-time matcher card",
        "share_unknown_reconstruction_checked": bool(share_unknown_ok),
        "primary": stats["primary"],
        "secondary": stats["secondary"],
        "n_candidates_min_population": int(a.n_candidates.min()),
        "n_candidates_min_universe": int(q.n_candidates.min()),
    }
    (OUT / "n14_statistics_v1.json").write_text(json.dumps(result, indent=2) + "\n")

    manifest = {
        "script": "scripts/explore/n14_support_gradient_v1.py",
        "specification": "manuscript/specifications/2026-11-10_N14_support_gradient_spec.md",
        "specification_sha256": sha256(ROOT / "manuscript" / "specifications"
                                       / "2026-11-10_N14_support_gradient_spec.md"),
        "git_commit": commit,
        "script_uncommitted": dirty,
        "script_sha256": sha256(Path(__file__)),
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p)
                          for p in (QUERIES, UNIVERSE / "manifest.json", ROWS, PUBLISHED)},
        "genie_root": "~/GenieBPC (restricted; read in memory, aggregates only)",
        "verification_1_query_universe": v1,
        "verification_2_similarity": v2,
        "fig5d_reproduction": repro,
        "outputs": ["n14_summary_v1.csv", "n14_statistics_v1.json", "manifest.json"],
        "governance": "aggregate only; no patient identifiers; every reported cell >= 5",
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for name in CONTRASTS:
        r = stats[name]
        print(f"{name:9s} {r['contrast']}: mean {r['mean_gain']:+.4f}  rho {r['rho']:+.3f} "
              f"({r['ci_low']:+.3f} to {r['ci_high']:+.3f}), P = {r['p_permutation']:.3f}")
        for _, row in s[s.analysis == name].iterrows():
            print(f"   third {row.third}: n {row.n} ({row.n_NSCLC}/{row.n_BRCA}/{row.n_CRC}) "
                  f"mean {row.mean_gain:+.4f} ({row.ci_low:+.4f} to {row.ci_high:+.4f}) "
                  f"unknown {row.share_unknown_mean:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
