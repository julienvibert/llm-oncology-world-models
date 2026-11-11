#!/usr/bin/env python3
"""The eight-arm PFS harness: computes what needs no model, emits prompts for what does.

One landmark per real treatment start. One endpoint: imaging-confirmed PFS from that start,
scored as a DISTRIBUTION -- proper scoring rule, so an arm cannot win by hedging.

    R0   population / action median          deterministic, computed here
    R1   raw k-nearest-neighbour             deterministic, computed here
    R2   local contrast transport            deterministic, computed here
    R3   LLM, no anchors                     prompt emitted
    R4   LLM, same-action anchors            prompt emitted
    R4b  LLM, same-action anchors, outcomes withheld    prompt emitted
    R5   LLM, paired A1/A2 anchors           prompt emitted
    R6   LLM, shuffled anchors               prompt emitted

WHAT "FAIR" MEANS HERE, EXACTLY. Two numbers, and they are not the same number.

    --k  (default 40)              the EVIDENCE SET. Retrieval returns up to this many
                                   anchors; the prompt summarises all of them. The
                                   deterministic k-NN comparator R1 uses the same k but is
                                   an out-of-fold same-action reference, not the identical
                                   hard-match/relaxation set shown to the model.
    --render-target (default 10)   how many of those k are printed row by row. A display
                                   budget, not an evidence budget.

So the default contract is "40 retrieved anchors, 10 of the 40 shown in detail" -- NOT ten
anchors. R1 is a classical reference at k=40; it does not receive the exact R4 set. The exact
"does the LLM add beyond what it was shown?" comparator is the anchor Kaplan-Meier curve,
sampled to the same finite draw count by the scorer. A k=10 run moves both R1 and retrieval
to the same neighbourhood size, but does not make their eligibility rules identical.

R2 is a LOCAL CONTRAST TRANSPORT, never a local ATE. Treatment here was chosen for reasons
nobody recorded, so the difference between two local groups is confounded by indication. The
word ATE is reserved for randomised assignment.

R4b is the arm that separates two very different explanations of any R4 win: the model is
reasoning from similar clinical stories, or it is copying neighbours' outcomes. If R4b
matches R4, the outcomes were decoration and what the anchors supplied was context -- era,
institution, practice pattern -- which would change what we render.

EXECUTION BOUNDARY. This script never calls a model, never reads a key, never spends. It
writes prompt payloads and a manifest; generation is launched separately. Everything it can
compute without a model, it computes, so the baselines are settled before a token is bought.

METRIC DISCIPLINE. The retrieval distance is fixed, unsupervised standardised Euclidean.
Deterministic fitted comparators are grouped by patient -- several lines belong to one person
and a row-level split leaks. Prognostic weights are used only to choose display strata for an
optional aggregate renderer; they never enter retrieval distance and are recorded separately.

GOVERNANCE: GENIE-BPC is under a data use agreement. The manifest and scores are aggregate.
Prompt payloads contain de-identified per-patient rows by necessity and are written to a
gitignored directory; they are inputs to a generation run, not analysis outputs.

Run:  uv run python scripts/explore/build_anchored_pfs_arms_v1.py --self-test
      uv run python scripts/explore/build_anchored_pfs_arms_v1.py --limit 200
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthonccases.anchored import (  # noqa: E402
    AnchorPolicy, Corpus, provenance, render, retrieve, retrieve_pair,
)
from synthonccases.anchored import query as Q  # noqa: E402
from synthonccases.anchored import summary as S  # noqa: E402

import measure_anchor_operating_window as W  # noqa: E402
import measure_pfs_distribution_bar as B  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_anchored_pfs_v1"
SEED = 20260810
N_FOLDS = 5

# ACTION GRANULARITY. The action is the regimen as recorded -- "Carboplatin, Pemetrexed
# Disodium", not "platinum". A class label pools carboplatin/pemetrexed with FOLFOX and then
# asks the model to anchor on patients who received a different drug.
#
# Measured before choosing: 80% of decisions still have at least 40 patients on the SAME
# regimen within their own cancer, and 90% have at least 10. Regimens below the floor fall
# back to their class, and the fallback is recorded per decision rather than hidden, so the
# two can be read apart afterwards.
#
# Applied here rather than in the shared loader, because the loader also feeds the frozen
# benchmarks and changing the action there would move the bar this experiment is measured
# against. Within this script every arm -- R0 through R6 -- uses the same definition.
MIN_REGIMEN_SUPPORT = 20

CONTEXT_FIELDS = ("cancer", "action", "line_number", "treatment_start_year")
#: Shown for every anchor, in this order, in EVERY arm. Fixed rather than ranked per anchor
#: set: a column set that depends on which anchors were retrieved makes the shuffled control
#: differ from the treatment arm in its columns too, and the comparison stops being clean.
#:
#: Deliberately excludes n_prior_scans, n_prior_visits and visit_rate. Those measure how
#: closely a patient was watched, not what their disease was doing; a latent-trajectory
#: generator handed them would be reading care intensity as biology.
DISPLAY_FIELDS = ("cancer", "action", "driver_bucket", "line_number",
                  "treatment_start_year", "age_at_start", "stage_iv", "dmets",
                  "n_disease_sites", "liver_met", "brain_met", "bone_met",
                  "last_scan_progressing", "months_on_prior_line", "prior_lines")
#: The indicators do not get their own columns -- they decide how the columns they govern
#: are printed. A row reading "not assessed" says it better than a 0/1 column beside a
#: number the reader then has to reconcile with it.
#: Channels GENIE-BPC never records. Not zero — absent. Rendering must keep them apart.
UNOBSERVED = ("ECOG performance status", "laboratory values", "weight change")


def base_policy(k: int, render_target: int) -> AnchorPolicy:
    return AnchorPolicy(
        k=k, render_target=render_target, min_support=10, metric="euclidean",
        # Hard filters, not distance terms. Cancer and stage are categorical facts a
        # clinician would never trade off against a small numeric similarity: a stage-II
        # colorectal patient is not "like" a stage-IV lung patient at any distance.
        # Measured before adding stage: 98% of decisions keep at least 10 anchors and 71%
        # keep at least 40. Line number was tested too and is NOT a filter -- it drops the
        # 40-anchor share to 20%, so it stays a rendered column and a distance term.
        # HARD CLINICAL ELIGIBILITY, then distance within it. Cancer, stage, line bin and
        # regimen are the axes where mixing is clinically nonsensical; burden, history and
        # age are where a difference is a matter of degree and belong in the ranking.
        # Biomarker belongs in this list and is NOT here: only the lung loader exists, and
        # matching on a marker for one cohort while ignoring it for two would encode the
        # cohort rather than the biology.
        match_exact=("cancer", "stage_iv", "line_group"),
        # measured tier sizes, share of decisions retaining >=10 anchors:
        #   strict (cancer+stage+line+regimen)  94%
        #   drop the line bin                   98%
        #   drop the stage                      98%
        # Each tier states the axes it holds FIXED, and every one of them appears in its
        # field list. The action is not carried over from `same_action` here: the tier
        # filter is exactly what is written, so a label reading "same regimen" has to name
        # the action column or it would be describing a filter that was never applied.
        tiers=(("same regimen, a different line of treatment",
                ("cancer", "stage_iv", "action")),
               ("same regimen and line, a different stage",
                ("cancer", "line_group", "action")),
               ("same cancer, stage and line, a different regimen",
                ("cancer", "line_group", "stage_iv"))),
        context_fields=CONTEXT_FIELDS, fields=DISPLAY_FIELDS,
        include_residual=True, include_aggregate=True, include_outcomes=True,
        label="anchored-pfs-v1")


ARMS: dict[str, dict] = {
    "R3_no_anchors": dict(renderer="none", kind="llm"),
    # NAMED FOR ITS RETRIEVAL POLICY, not its rendering. It was `R4_table` while the runner and
    # the scorer both keyed on `R4_same_action`, so every headline comparison in the readout
    # would have been silently absent -- the same failure mode as a config declaring the wrong
    # response contract: nothing errors, the answer just is not there. The check below makes
    # the two files agree by construction.
    "R4_same_action": dict(renderer="compact", kind="llm"),
    # The evidence-block ladder. Same retrieved cohort in all three; only what is said
    # about it changes, so a difference between them is attributable to the FORM of the
    # evidence and nothing else.
    "R4_aggregate_marginal": dict(renderer="aggregate_marginal", kind="llm"),
    "R4_aggregate_conditional": dict(renderer="aggregate_conditional", kind="llm"),
    "R4b_no_outcomes": dict(renderer="compact", kind="llm",
                            policy=dict(include_outcomes=False)),
    "R5_paired": dict(renderer="compact", kind="llm", paired=True),
    # THE CONTROL, BY DERANGEMENT. Each query receives ANOTHER QUERY'S R4 evidence block,
    # whole and unaltered. The earlier version scrambled anchors across the whole corpus, which
    # produced a visibly incoherent mixture of cancers and regimens -- a model can see at a
    # glance that such evidence is nonsense and ignore it, so what it controlled for was "text
    # is present" and not "plausible evidence is present". Borrowing a real block keeps the
    # volume, the internal coherence and the outcome scale identical, and changes only WHOSE
    # patients they are, which is the one thing the comparison is about.
    "R6_shuffled": dict(renderer="compact", kind="llm", borrow_from="R4_same_action"),
    # the control for the aggregate arms: same FORM, wrong evidence. Without it, an
    # aggregate arm that scores well cannot be told apart from one that reports a number
    # the model would have produced anyway.
    "R6_shuffled_aggregate": dict(renderer="aggregate_conditional", kind="llm",
                                  borrow_from="R4_aggregate_conditional"),
}


# --------------------------------------------------------------------------------------
# THE ARM NAMES ARE A CONTRACT WITH THE READOUT, so they are checked rather than trusted.
# `score_anchored_pfs_v1` names the comparisons it exists to produce; an arm this builder emits
# under a different name is not scored and nothing says so.
# --------------------------------------------------------------------------------------
import score_anchored_pfs_v1 as _S  # noqa: E402
import score_anchored_pfs_v1 as S_score  # noqa: E402

_unknown = set(ARMS) - set(_S.KNOWN_ARMS)
if _unknown:
    raise SystemExit(f"[fatal] these arms are not names the scorer can consume, so they would "
                     f"be emitted and never scored: {sorted(_unknown)}")
_needed = {a for pair in _S.PRIMARY_COMPARISONS for a in pair[:2]}
_missing_primary = set(_S.CORE_ARMS) - set(ARMS)
if _missing_primary:
    raise SystemExit(f"[fatal] the readout's headline comparisons need {sorted(_missing_primary)} "
                     f"and this builder does not emit them")


def assign_actions(d: pd.DataFrame, regimens: pd.Series) -> pd.DataFrame:
    """Regimen where it can carry anchors, the coarse class where it cannot."""
    reg = regimens.astype(str).str.strip()
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([d["cancer"], reg])).groupby(
        level=[0, 1]).sum()
    enough = np.array([counts.get((c, r), 0) >= MIN_REGIMEN_SUPPORT
                       for c, r in zip(d["cancer"], reg, strict=True)])
    out = d.copy()
    out["action"] = np.where(enough, reg, d["action"])
    out["action_granularity"] = np.where(enough, "regimen", "class")
    return out


def prognostic_weights(d: pd.DataFrame, channels: list[str]) -> dict[str, float]:
    """How much each channel moves the outcome, for choosing what to stratify on.

    A penalised Cox fit on the corpus, used ONLY to rank axes for display. It never touches
    the distance metric and never reads a query's own outcome. Fitted once on all landmarks
    rather than per fold: the influence of any single patient on a one-coefficient-per-
    channel summary is negligible, the query's own rows are excluded from its anchors
    anyway, and a per-query refit would cost more than the choice is worth. Recorded in the
    manifest so the ranking can be reproduced or challenged.
    """
    from lifelines import CoxPHFitter
    x = d[channels].astype(float)
    x = x.loc[:, x.std() > 1e-9]
    x = (x - x.mean()) / x.std()
    frame = x.assign(T=d["pfs_mos"].to_numpy(float), E=d["pfs_event"].to_numpy(float))
    try:
        fit = CoxPHFitter(penalizer=0.5).fit(frame, "T", "E")
    except Exception as exc:                                   # noqa: BLE001
        print(f"   prognostic weights unavailable ({type(exc).__name__}); "
              "strata will be ranked by residual size alone")
        return {}
    return {c: float(abs(v)) for c, v in fit.params_.items()}


def build_corpus(d: pd.DataFrame, channels: list[str],
                 weights: dict[str, float] | None = None) -> Corpus:
    return Corpus(d, channels=channels, subject_col="record_id", action_col="action",
                  prognostic_weights=weights or {},
                  # "never scanned" must not be numerically identical to "scanned, clean"
                  observation_flags={
                      f: tuple(c for c in cols if c in channels or c in DISPLAY_FIELDS)
                      for f, cols in {**W.OBSERVATION_FLAGS,
                                      "panel_reported": ("driver_bucket", "n_drivers",
                                                         "alterations")}.items()
                      if f in channels},
                  outcome_cols=["pfs_mos", "pfs_event", "os_mos", "os_event",
                                "ttnt_mos", "ttnt_event"],
                  unobserved=list(UNOBSERVED),
                  outcome_time_col="pfs_mos", outcome_event_col="pfs_event",
                  # a generator asked for a whole realistic course must see what happened
                  # after progression, or it is calibrated on the first segment only
                  extra_outcomes=[("overall survival", "os_mos", "os_event"),
                                  ("next treatment", "ttnt_mos", "ttnt_event")])


def decision_table(cancers: list[str] | None = None) -> pd.DataFrame:
    """THE canonical landmark table, with `query_id`. One implementation, used by everyone.

    query_id is `record_id@<positional index>`, so it is only meaningful relative to the
    exact row sequence that produced it. `attach_regimens` ends in drop_duplicates() and
    reset_index(), which renumbers every row after the first drop -- so a table built
    BEFORE that join and one built AFTER share the same id space while disagreeing about
    what the ids mean.

    That is not a hypothetical. The scorer built its ids before the join and the builder
    after: of the definitive run's 500 queries, 134 failed to match at all and 70 of the 366
    that DID match resolved to a different decision for the same patient, silently attaching
    the wrong recorded outcome to a generation. A complete-looking report, 19% of it wrong.

    Anything that needs to turn a query_id back into an outcome must call this.
    """
    d = W.load_decisions()
    if cancers:
        d = d[d["cancer"].isin(cancers)]
    d = d[d["pfs_mos"].gt(0) & d["pfs_event"].notna()].reset_index(drop=True)
    d = attach_regimens(d)
    d = attach_identity(d)
    d = attach_biomarkers(d)
    d = d.reset_index(drop=True)
    d["query_id"] = d["record_id"].astype(str) + "@" + d.index.astype(str)
    return d


def attach_regimens(d: pd.DataFrame) -> pd.DataFrame:
    """Re-read the recorded regimen string and refine the action to it where support allows."""
    reg = []
    for cohort in W.COHORTS:
        r = pd.read_csv(W.GENIE_ROOT / cohort / "regimen_cancer_level_dataset.csv",
                        low_memory=False)
        r["record_id"] = cohort.split("_")[0] + "::" + r["record_id"].astype(str)
        r["months_since_dx"] = pd.to_numeric(r["dx_reg_start_int_mos"], errors="coerce")
        r["os_mos"] = pd.to_numeric(r["tt_os_g_mos"], errors="coerce")
        r["os_event"] = pd.to_numeric(r["os_g_status"], errors="coerce")
        reg.append(r[["record_id", "months_since_dx", "regimen_drugs",
                      "os_mos", "os_event"]])
    reg = pd.concat(reg, ignore_index=True).dropna()
    merged = d.merge(reg, on=["record_id", "months_since_dx"], how="left")
    merged = merged.drop_duplicates(subset=["record_id", "months_since_dx"]).reset_index(
        drop=True)
    merged["regimen_drugs"] = merged["regimen_drugs"].fillna("unknown")
    return assign_actions(merged, merged["regimen_drugs"])


def recorded_path(path: Path) -> str:
    """How a written file is named in the manifest.

    Repo-relative when the output lives in the repo, which is what makes a manifest
    portable between checkouts; absolute otherwise. `relative_to` RAISES for a path outside
    the root, so an --out-dir under /tmp used to kill the build after all the work was done
    and before anything was written.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def attach_identity(d: pd.DataFrame) -> pd.DataFrame:
    """Histology, primary site and the YEAR of diagnosis, for the baseline seed.

    Read here rather than in the shared loader for the same reason the regimen string is:
    the loader feeds the frozen benchmarks, and this experiment must not move their inputs.
    """
    parts = []
    for cohort in W.COHORTS:
        idx = pd.read_csv(W.GENIE_ROOT / cohort / "cancer_level_dataset_index.csv",
                          low_memory=False)
        pat = pd.read_csv(W.GENIE_ROOT / cohort / "patient_level_dataset.csv",
                          low_memory=False)
        idx = idx.merge(pat[["record_id", "birth_year"]], on="record_id", how="left")
        # the finest histology the cohort records: breast has its own field, the others
        # carry only the adeno/squamous split
        hist = "ca_hist_brca" if "ca_hist_brca" in idx.columns else "ca_hist_adeno_squamous"
        idx["histology"] = idx[hist].astype(str)
        idx["primary_site_code"] = idx["ca_d_site"].astype(str)
        idx["dx_year"] = (pd.to_numeric(idx["birth_year"], errors="coerce")
                          + pd.to_numeric(idx["dob_ca_dx_yrs"], errors="coerce"))
        idx["record_id"] = cohort.split("_")[0] + "::" + idx["record_id"].astype(str)
        parts.append(idx[["record_id", "histology", "primary_site_code", "dx_year"]]
                     .drop_duplicates(subset=["record_id"]))
    return d.merge(pd.concat(parts, ignore_index=True), on="record_id", how="left")


#: Genes reported per cancer, chosen for what changes management rather than for what is
#: recurrent. A panel reports hundreds; putting all of them in front of the model would bury
#: the driver in passengers and invite it to build a story out of noise.
DRIVER_GENES: dict[str, tuple[str, ...]] = {
    "NSCLC": ("EGFR", "ALK", "ROS1", "BRAF", "KRAS", "MET", "RET", "ERBB2", "NTRK1"),
    "BRCA": ("ERBB2", "PIK3CA", "BRCA1", "BRCA2", "ESR1", "AKT1", "PTEN"),
    "CRC": ("KRAS", "NRAS", "BRAF", "PIK3CA", "ERBB2", "MLH1", "MSH2", "MSH6", "PMS2"),
}
#: Variant classes that plausibly change a protein. A silent mutation is a sequencing fact,
#: not a clinical one.
CODING_CLASSES = frozenset({
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site",
    "Nonstop_Mutation",
})


def attach_biomarkers(d: pd.DataFrame) -> pd.DataFrame:
    """Sequencing results KNOWN AT THE LANDMARK, per decision.

    Two rules, and the experiment is wrong without either.

    * **Report date, not sample date.** A panel resulted after this treatment started was
      not available when the decision was taken. Including it would hand the model a fact
      from the future of the very decision it is predicting -- the most expensive kind of
      leak, because it looks like good conditioning.
    * **Sequenced-and-negative is not unsequenced.** "EGFR wild type" and "never tested"
      are different clinical states, and the second is much more common early. They are
      separated by `panel_reported`, the same missing-indicator discipline as imaging.
    """
    out = []
    for cohort in W.COHORTS:
        cancer = cohort.split("_")[0]
        prefix = cancer + "::"
        genes = set(DRIVER_GENES.get(cancer, ()))
        panel = pd.read_csv(W.GENIE_ROOT / cohort / "cancer_panel_test_level_dataset.csv",
                            low_memory=False)
        panel["record_id"] = prefix + panel["record_id"].astype(str)
        panel["report_mos"] = pd.to_numeric(panel["dx_cpt_rep_mos"], errors="coerce")
        panel = panel.dropna(subset=["report_mos", "cpt_genie_sample_id"])

        maf = pd.read_csv(W.GENIE_ROOT / cohort / "data_mutations_extended.txt",
                          sep="	", low_memory=False)
        maf = maf[maf["Hugo_Symbol"].isin(genes)
                  & maf["Variant_Classification"].isin(CODING_CLASSES)]
        by_sample: dict[str, list[str]] = {}
        for sid, g in maf.groupby("Tumor_Sample_Barcode"):
            calls = []
            for _, r in g.iterrows():
                aa = str(r.get("HGVSp_Short") or "").lstrip("p.")
                calls.append(f"{r['Hugo_Symbol']} {aa}" if aa and aa != "nan"
                             else str(r["Hugo_Symbol"]))
            by_sample[str(sid)] = sorted(dict.fromkeys(calls))

        tests = {rec: g.sort_values("report_mos")
                 for rec, g in panel.groupby("record_id")}
        rows = d.index[d["cancer"] == cancer]
        for i in rows:
            t0 = float(d.at[i, "months_since_dx"])
            g = tests.get(str(d.at[i, "record_id"]))
            g = g[g["report_mos"] <= t0] if g is not None else None
            if g is None or g.empty:
                out.append({"i": i, "panel_reported": 0.0, "alterations": "",
                            "driver_bucket": "", "n_drivers": 0.0})
                continue
            calls: list[str] = []
            for sid in g["cpt_genie_sample_id"].astype(str):
                calls.extend(by_sample.get(sid, []))
            calls = sorted(dict.fromkeys(calls))
            out.append({"i": i, "panel_reported": 1.0, "alterations": "; ".join(calls),
                        "driver_bucket": (calls[0].split()[0] if calls else "none found"),
                        "n_drivers": float(len(calls))})
    bio = pd.DataFrame(out).set_index("i").sort_index()
    return pd.concat([d, bio], axis=1)


def landmark_histories(d: pd.DataFrame, *, imaging: bool = True) -> dict[str, list[dict]]:
    """Every DATED fact the release records about a patient, keyed by patient.

    Treatment starts come from the regimen table and imaging reads from the imaging table,
    both carrying a real interval from diagnosis. Counts without dates -- how many prior
    progressions, how many prior scans -- are deliberately absent: a prefix asserts that
    these events happened on these days, and a plausible date is still a fabricated one.

    `imaging` exists because the two CONTRACTS disagree about what a prefix event must
    carry. Under `continuation` a prefix event is a conditioning fact and "progression seen
    on this date" is a complete one. Under the inherited `latent_completion` contract the
    prefix is replayed, and replay needs the disease and host deltas that this release does
    not record for an imaging read -- which is why these events were removed. Excluding them
    is not free: the anchor table shows every neighbour's `last_scan_progressing` while the
    index patient's own progression history stays silent, and that asymmetry has no defence
    once the validator no longer forces it.
    """
    facts: dict[str, list[dict]] = {}
    for cohort in W.COHORTS:
        prefix = cohort.split("_")[0] + "::"
        r = pd.read_csv(W.GENIE_ROOT / cohort / "regimen_cancer_level_dataset.csv",
                        low_memory=False)
        r["record_id"] = prefix + r["record_id"].astype(str)
        r["months"] = pd.to_numeric(r["dx_reg_start_int_mos"], errors="coerce")
        for rec, g in r.dropna(subset=["months"]).groupby("record_id"):
            g = g.sort_values("months")
            out = facts.setdefault(rec, [])
            for j, (_, row) in enumerate(g.iterrows()):
                label = str(row["regimen_drugs"])
                out.append({
                    "months": float(row["months"]),
                    "type": "systemic_treatment_start" if j == 0
                            else "systemic_treatment_switch",
                    "regimen": {"label": label,
                                "components": [c.strip() for c in label.split(",")
                                               if c.strip()]}})
        if not imaging:
            continue
        im, _ = W.prior_state(W.GENIE_ROOT / cohort, prefix)
        first_tx = {rec: min(f["months"] for f in fs) for rec, fs in facts.items()}
        for rec, g in im[im["progressing"] > 0].groupby("record_id"):
            out = facts.setdefault(rec, [])
            t0 = first_tx.get(rec, float("inf"))
            for m in g["months"]:
                # before any systemic therapy this is baseline disease tempo, not a PFS
                # progression -- the same rule the hybrid pipeline applies
                out.append({"months": float(m),
                            "type": "progression" if float(m) > t0
                                    else "patient_state_update",
                            "rationale": "imaging read as progressive disease"})
    return facts


def deterministic_arms(d: pd.DataFrame, x: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """R0-R2: survival curves per query, out-of-fold, needing no model.

    R2 transports the local contrast between the query's action and the most common
    alternative in the same neighbourhood -- observational, so a contrast and not an effect.
    """
    from sklearn.model_selection import GroupKFold
    t = d["pfs_mos"].to_numpy(float)
    e = d["pfs_event"].to_numpy(float)
    act = d["action"].to_numpy()
    pid = d["record_id"].to_numpy()
    grid = B.GRID
    out = {n: np.ones((len(d), len(grid))) for n in ("R0_action_median", "R1_knn",
                                                     "R2_local_contrast")}
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(x, e, groups=pid):
        marg = B.km_curve(t[tr], e[tr], grid)
        for a in np.unique(act[te]):
            m = act[tr] == a
            out["R0_action_median"][te[act[te] == a]] = (
                B.km_curve(t[tr][m], e[tr][m], grid) if m.sum() >= 20 else marg)
        for j, ti in enumerate(te):
            same = act[tr] == act[ti]
            pool = same if same.sum() >= k else np.ones(len(tr), bool)
            dist = np.linalg.norm(x[tr][pool] - x[ti], axis=1)
            take = np.argsort(dist)[:k]
            out["R1_knn"][ti] = B.km_curve(t[tr][pool][take], e[tr][pool][take], grid)

            # R2: local contrast transport. Take a WIDER neighbourhood ignoring action
            # (more support), then correct it for the fact that this patient's action
            # differs from the neighbourhood's mix.
            #
            # The correction is applied to the CUMULATIVE HAZARD, not to survival. Adding a
            # contrast to S(t) can push the curve upward, which is not a survival function
            # at all -- the self-test caught exactly that. Scaling the hazard keeps the
            # result monotone by construction, and a hazard ratio is how a local treatment
            # contrast is expressed anyway.
            near = np.argsort(np.linalg.norm(x[tr] - x[ti], axis=1))[: 3 * k]
            own = near[act[tr][near] == act[ti]]
            alt = near[act[tr][near] != act[ti]]
            s_all = B.km_curve(t[tr][near], e[tr][near], grid)
            if len(own) >= 10 and len(alt) >= 10:
                h_own = -np.log(np.clip(B.km_curve(t[tr][own], e[tr][own], grid), 1e-6, 1))
                h_alt = -np.log(np.clip(B.km_curve(t[tr][alt], e[tr][alt], grid), 1e-6, 1))
                # one ratio for the whole curve: proportional hazards, stated as such, and
                # clipped so a near-zero denominator cannot produce an absurd shift
                ratio = float(np.clip(h_own.sum() / max(h_alt.sum(), 1e-6), 0.25, 4.0))
                h_all = -np.log(np.clip(s_all, 1e-6, 1.0))
                out["R2_local_contrast"][ti] = np.clip(np.exp(-ratio * h_all), 1e-6, 1.0)
            else:
                out["R2_local_contrast"][ti] = out["R1_knn"][ti]
    return out


def emit_queries(d: pd.DataFrame, queries: list[int], facts: dict[str, list[dict]],
                 out_dir: Path, *, imaging: bool = True) -> dict:
    """The query side, written ONCE and shared by every arm.

    One file, not a copy per arm: the experimental contract is that the patient and their
    fixed history are identical across R3/R4/R4b/R5/R6 and only the anchor block moves.
    Writing them per arm would make that a convention someone has to check; writing them
    once makes it structural.

    Every payload is lowered through the frozen `lower_conditioning_cell` and checked with
    the frozen `validate_observed_prefix` before it is written, so a prefix that the runner
    would reject is caught here and not after a spend.
    """
    from synthonccases.oncollm.condition_lower import lower_conditioning_cell
    from synthonccases.oncollm.conditioning_audit import validate_observed_prefix

    rows, no_date, n_events = [], 0, []
    audit = collections.Counter()
    unmatched: list[str] = []
    for i in queries:
        r = d.iloc[i]
        origin = Q.diagnosis_date(r.get("dx_year"))
        if origin is None:
            # no diagnosis year means no calendar to hang the history on. The landmark is
            # emitted without a prefix rather than with one anchored to a made-up year.
            no_date += 1
        seed, pinned = Q.baseline_seed(r)
        fixed = ({} if origin is None else
                 Q.prefix_events(facts.get(str(r["record_id"]), []), origin=origin,
                                 landmark_months=float(r["months_since_dx"]),
                                 landmark_action=str(r["regimen_drugs"])))
        for key in ("n_dropped_pre_diagnosis", "n_duplicates_collapsed",
                    "n_merged_within_window", "n_observations_at_landmark_dropped"):
            audit[key] += int(fixed.pop(key, 0)) if fixed else 0
        if fixed:
            n_cut = int(fixed.pop("n_treatment_events_at_cutoff"))
            audit["n_with_concurrent_treatment_at_cutoff"] += int(n_cut > 1)
            # THE LANDMARK MUST BE UNAMBIGUOUS. The endpoint is PFS from one named
            # treatment start; a prefix whose cutoff day does not carry that treatment
            # leaves the model to infer which regimen it is continuing from.
            if not fixed.pop("index_treatment_matched"):
                unmatched.append(f"{r['record_id']}@{i}")
        payload = Q.conditioning_payload(seed, pinned, fixed) if fixed else {
            "baseline_seed": seed, "pinned_paths": list(pinned)}
        spec = lower_conditioning_cell(conditioning_payload=payload)
        validate_observed_prefix(spec, baseline_diagnosis_date=origin)
        n_events.append(len(fixed.get("events", [])))
        rows.append({"query_id": f"{r['record_id']}@{i}", "cancer": r["cancer"],
                     "oncotree_code": seed["identity_card"]["diagnosis"]["oncotree_code"],
                     "landmark_months": float(r["months_since_dx"]),
                     "conditioning_payload": payload})
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} landmarks have no index treatment on the cutoff date, so "
            f"the regimen the trajectory continues from is ambiguous: {unmatched[:5]}")
    path = out_dir / "queries.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"path": recorded_path(path), "n_queries": len(rows),
            "n_without_diagnosis_year": no_date,
            # which contract these payloads are usable under. The runner checks it rather
            # than discovering it as a wall of replay rejections.
            "prefix_includes_imaging": imaging,
            "usable_contracts": (["continuation"] if imaging
                                 else ["continuation", "latent_completion"]),
            **dict(audit),
            "median_prefix_events": int(np.median(n_events)) if n_events else 0,
            "max_prefix_events": int(np.max(n_events)) if n_events else 0}


#: Draw counts at which a deterministic arm is ALSO scored, so it pays the same discretisation
#: cost the generative arms pay. 10 is the registered futures-per-landmark.
SAMPLED_AT: tuple[int, ...] = (5, 10, 20)


def _sampled_skill(surv: np.ndarray, t: np.ndarray, e: np.ndarray, null: float,
                   *, n_gen: int, seed: int, reps: int = 40) -> dict:
    """Skill of a smooth arm after converting each patient's curve into `n_gen` draws.

    The generative arms are scored from a finite sample; this is what the SAME arm scores once
    it is handed the same handicap. Reported as a median and interval over `reps` redraws,
    because a single redraw is itself noisy and a point estimate would invite over-reading.
    """
    rng = np.random.default_rng(seed)
    skills = []
    for _ in range(reps):
        emp = np.empty_like(surv)
        for i in range(len(surv)):
            draws = S_score.sample_from_survival_curve(surv[i], B.GRID, n_patients=1,
                                                       n_samples=n_gen, rng=rng)[0]
            emp[i] = (draws[:, None] > B.GRID[None, :]).mean(axis=0)
        ibs = B.integrated_brier(np.clip(emp, 1e-6, 1.0), t, e, B.GRID, t, e)
        skills.append(1 - ibs / null)
    a = np.array(skills)
    return {"n_gen": n_gen, "skill_median": float(np.median(a)),
            "skill_lo": float(np.percentile(a, 2.5)), "skill_hi": float(np.percentile(a, 97.5))}


def derangement(n: int, seed: int) -> list[int]:
    """A permutation with NO fixed point, so no query can be given its own evidence back.

    Deterministic from `seed` and recorded in the manifest: the control has to be reproducible,
    and "we shuffled it" is not a description anyone can check. A plain shuffle leaves about one
    query in every e holding its own block, which would quietly turn part of the control arm
    back into the treated arm.
    """
    if n < 2:
        return list(range(n))
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return [int(i) for i in perm]
    # deterministic fallback: a single cycle has no fixed point by construction
    return [(i + 1) % n for i in range(n)]


def emit_prompts(d: pd.DataFrame, corpus: Corpus, policy: AnchorPolicy, queries: list[int],
                 out_dir: Path, *, seed: int = SEED) -> dict[str, dict]:
    """One payload per (arm, query). No model call; this is the input a run consumes."""
    summary: dict[str, dict] = {}
    built: dict[str, dict[str, dict]] = {}          # arm -> query_id -> row, for borrowing
    _borrow_map: dict[str, dict[str, str]] = {}     # source arm -> query_id -> donor query_id
    # borrowing arms run LAST, so the arm they borrow from is already built
    order = sorted(ARMS.items(), key=lambda kv: bool(kv[1].get("borrow_from")))
    for arm, spec in order:
        pol = policy.variant(label=f"{policy.label}:{arm}", **spec.get("policy", {}))
        rows, supported, refused, not_constructible = [], 0, 0, 0
        for i in queries:
            q = d.iloc[i].copy()
            q["query_id"] = f"{d.iloc[i]['record_id']}@{i}"
            source_arm = spec.get("borrow_from")
            borrowed_from = ""
            evidence_row: dict | None = None
            if source_arm:
                # Borrowed wholesale from another query's block. Retrieval is still run for
                # THIS query so the row carries its own support diagnostics -- what was
                # available to it matters even though it was given someone else's evidence.
                res = retrieve(corpus, q, pol)
                donors = built.get(source_arm, {})
                if not donors:
                    not_constructible += 1
                    continue
                keys = sorted(donors)
                mapping = _borrow_map.setdefault(
                    source_arm, dict(zip(keys, (keys[j] for j in derangement(len(keys), seed)),
                                         strict=True)))
                donor_key = mapping.get(q["query_id"])
                if donor_key is None or donor_key not in donors:
                    not_constructible += 1
                    continue
                donor = donors[donor_key]
                block = donor["anchor_block"]
                borrowed_from = donor_key
                evidence_row = donor
            elif spec.get("paired"):
                others = d.loc[d["action"] != q["action"], "action"].value_counts()
                if others.empty:
                    refused += 1
                    continue
                ra, rb = retrieve_pair(corpus, q, pol, q["action"], others.index[0])
                block = (render(spec["renderer"], corpus, ra, pol) + "\n\n"
                         + render(spec["renderer"], corpus, rb, pol))
                res = ra
            else:
                res = retrieve(corpus, q, pol)
                block = render(spec["renderer"], corpus, res, pol)
            # A shuffle control that could not be built is not a control with no anchors —
            # it is an absent control. Emitting it as R6 would put "no comparable patients"
            # in front of the model and then score that as though anchors had been shuffled.
            # has_anchors, NOT supported: a sham result deliberately reports supported
            # False, so testing that here silently dropped every control row.
            if source_arm and not res.has_anchors:
                not_constructible += 1
                continue
            evidence_has_anchors = bool(
                evidence_row["n_anchors"] if evidence_row is not None else res.n_anchors)
            if source_arm and not evidence_has_anchors:
                not_constructible += 1
                continue
            supported += int(evidence_has_anchors)
            refused += int(not evidence_has_anchors)
            # THE ECHO BASELINE. What the evidence block itself says the median is. A
            # generator that simply repeats it has transported nothing, and that failure
            # looks exactly like success unless the number travels with the prompt.
            if evidence_row is not None:
                # R6 must be audited against the evidence it ACTUALLY received. The old
                # implementation borrowed another query's block but kept this query's KM
                # curve and support fields, so the echo/adaptation diagnostics compared the
                # generation to an invisible reference. Query-side support is retained below
                # under explicit query_* names.
                km_median = evidence_row["anchor_km_median"]
                km_curve = list(evidence_row["anchor_km_curve"])
                support_label = evidence_row["support_label"]
                n_anchors = int(evidence_row["n_anchors"])
                anchor_spread = evidence_row["anchor_spread"]
                outcome_dispersion = evidence_row["outcome_dispersion"]
                n_eligible = int(evidence_row["n_eligible"])
                n_strict = int(evidence_row["n_strict"])
                tier_counts = list(evidence_row["tier_counts"])
                censored_fraction = evidence_row["censored_fraction"]
                evidence_policy_hash = evidence_row["policy_hash"]
                evidence_corpus_id = evidence_row["corpus_id"]
            else:
                km_median = float("nan")
                km_curve: list[float] = []
                if res.has_anchors and corpus.outcome_time_col:
                    sub = corpus.frame.iloc[list(res.anchor_index)]
                    km = S.km_summary(sub[corpus.outcome_time_col],
                                      sub[corpus.outcome_event_col])
                    km_median = km.median if km.median is not None else float("nan")
                    # THE FULL CURVE, not only its median. The scorer needs it to sample the
                    # anchor distribution the same number of times the model was sampled.
                    km_curve = [round(float(v), 6) for v in B.km_curve(
                        pd.to_numeric(sub[corpus.outcome_time_col], errors="coerce")
                          .fillna(0.0).to_numpy(float),
                        pd.to_numeric(sub[corpus.outcome_event_col], errors="coerce")
                          .fillna(0.0).to_numpy(float), B.GRID)]
                support_label = res.support_label
                n_anchors = res.n_anchors
                anchor_spread = res.anchor_spread
                outcome_dispersion = res.outcome_dispersion
                n_eligible = res.n_eligible
                n_strict = res.n_strict
                tier_counts = list(res.tier_counts)
                censored_fraction = res.censored_fraction
                evidence_policy_hash = res.policy_hash
                evidence_corpus_id = res.corpus_id
            row = {"query_id": q["query_id"], "arm": arm,
                   "anchor_km_median": km_median,
                   "anchor_km_curve": km_curve,
                   "support_label": support_label, "n_anchors": n_anchors,
                   "anchor_spread": anchor_spread,
                   "outcome_dispersion": outcome_dispersion,
                         # how the set was assembled travels with it: a ten-anchor set that
                         # is six strict and four relaxed must be separable at scoring time
                         # from ten strict ones, and the block alone cannot be parsed for it
                   "n_eligible": n_eligible, "n_strict": n_strict,
                   "tier_counts": tier_counts,
                   "censored_fraction": censored_fraction,
                   "policy_hash": res.policy_hash, "corpus_id": res.corpus_id,
                   "evidence_policy_hash": evidence_policy_hash,
                   "evidence_corpus_id": evidence_corpus_id,
                         # Support around the QUERY is a different object from support in a
                         # borrowed evidence block. Keep both so the control remains auditable.
                   "query_support_label": res.support_label,
                   "query_n_anchors": res.n_anchors,
                   "query_anchor_spread": res.anchor_spread,
                   "query_outcome_dispersion": res.outcome_dispersion,
                   "query_n_eligible": res.n_eligible,
                   "query_n_strict": res.n_strict,
                         # whose evidence this actually is: empty for the real arms, the donor's
                         # query_id for a borrowed control, so the derangement is auditable
                   "borrowed_from": borrowed_from,
                   "anchor_block": block}
            if evidence_row is not None:
                for key in ("anchor_block", "anchor_km_median", "anchor_km_curve",
                            "support_label", "n_anchors", "anchor_spread",
                            "outcome_dispersion", "n_eligible", "n_strict",
                            "tier_counts", "censored_fraction"):
                    left, right = row[key], evidence_row[key]
                    both_missing = bool(
                        np.isscalar(left) and np.isscalar(right)
                        and pd.isna(left) and pd.isna(right)
                    )
                    if not both_missing and left != right:
                        raise AssertionError(
                            f"borrowed control {q['query_id']} describes evidence other than "
                            f"its donor {borrowed_from}: field {key} differs"
                        )
            rows.append(row)
        built[arm] = {r["query_id"]: r for r in rows}
        frame = pd.DataFrame(rows)
        path = out_dir / f"prompts_{arm}.jsonl"
        frame.to_json(path, orient="records", lines=True)
        summary[arm] = {"policy_hash": pol.hash, "renderer": spec["renderer"],
                        "n_queries": len(rows), "n_supported": supported,
                        "n_refused": refused,
                        "n_control_not_constructible": not_constructible,
                        "borrowed_from_arm": spec.get("borrow_from", ""),
                        "median_block_chars": int(frame["anchor_block"].str.len().median())
                        if len(frame) else 0,
                        "path": recorded_path(path)}
    return summary


def self_test() -> int:
    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({
        "record_id": [f"p{i // 2}" for i in range(n)],
        "action": rng.choice(["A", "B"], n),
        "cancer": rng.choice(["BRCA", "CRC"], n),
        "line_number": rng.integers(1, 4, n).astype(float),
        "era": rng.normal(2018, 2, n),
        "burden": rng.normal(size=n),
        "ecog": rng.normal(size=n),
        "pfs_mos": rng.exponential(5.0, n) + 0.1,
        "pfs_event": rng.choice([0.0, 1.0], n, p=[0.2, 0.8]),
    })
    # the columns the policy filters and renders on. The self-test exercises the real
    # policy, so its frame has to carry every column that policy names, or the check
    # passes only because retrieval never ran.
    d["stage_iv"] = rng.choice([0.0, 1.0], n)
    d["line_group"] = np.where(d["line_number"] <= 1, "1L", "2L+")
    d["treatment_start_year"] = d["era"].round().astype(int)
    d["os_mos"] = d["pfs_mos"] + rng.exponential(6.0, n)
    d["os_event"] = rng.choice([0.0, 1.0], n, p=[0.5, 0.5])
    d["ttnt_mos"] = d["pfs_mos"] + rng.exponential(1.0, n)
    d["ttnt_event"] = rng.choice([0.0, 1.0], n, p=[0.4, 0.6])
    x = d[["burden", "ecog", "line_number", "era"]].to_numpy(float)
    x = (x - x.mean(0)) / x.std(0)
    det = deterministic_arms(d, x, k=20)
    for name, surv in det.items():
        assert surv.shape == (n, len(B.GRID)), name
        assert np.all((surv >= 0) & (surv <= 1)), f"{name} left [0,1]"
        # a survival curve must never increase with time
        assert np.all(np.diff(surv, axis=1) <= 1e-9), f"{name} is not monotone"

    corpus = build_corpus(d, ["burden", "ecog", "line_number", "era"])
    pol = base_policy(k=20, render_target=4)
    q = d.iloc[0].copy()
    q["query_id"] = "t0"
    res = retrieve(corpus, q, pol)
    block = render("compact", corpus, res, pol)
    if res.supported:
        # the unobserved channels are named as absent, never rendered as a zero column
        assert "Not recorded by this source" in block
        assert "ECOG performance status" in block
        assert "| ECOG performance status |" not in block
    print("self-test OK — deterministic arms produce monotone curves in [0,1], and the "
          "compact renderer names unobserved channels instead of zeroing them")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=40,
                    help="EVIDENCE SET: anchors retrieved and summarised in the prompt, and "
                         "the neighbourhood size of the deterministic k-NN reference R1. "
                         "The counts match; their eligibility rules do not.")
    ap.add_argument("--render-target", type=int, default=10,
                    help="DISPLAY BUDGET: how many of the k anchors are printed row by row "
                         "for the model. Changes what is shown, not what either arm knows; "
                         "it is NOT the comparator size.")
    ap.add_argument("--limit", type=int, default=200,
                    help="landmarks to emit prompts for; the decisive run needs 500+")
    ap.add_argument("--cancers", nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-prefix-imaging", action="store_true",
                    help="leave dated imaging progressions out of the query prefix. Needed "
                         "only for the latent_completion ablation, whose acceptance replays "
                         "the prefix and requires deltas this release does not record.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    # THE SAME FUNCTION THE SCORER CALLS. Duplicating this sequence is what produced the
    # id-space split; it is defined once and used by both.
    d = decision_table(args.cancers)
    # a fractional year reads as false precision and invites the model to over-interpret it;
    # the calendar year is the context axis that actually carries practice pattern
    d["treatment_start_year"] = d["era"].round().astype("Int64")
    # 1L versus later is the bin a clinician reasons in; the exact line number stays a
    # rendered column and a distance term
    d["line_group"] = np.where(d["line_number"] <= 1, "1L", "2L+")
    # `panel_reported` and `n_drivers` join the state: a sequenced patient with no driver
    # and an unsequenced one are different patients, exactly as with imaging.
    channels = [c for c in (*W.STATE_RICH, *W.HISTORY_RICH, "era", "age_at_start",
                            "panel_reported", "n_drivers")
                if c in d.columns]
    x = W.feature_matrix(d, [*W.CONTEXT, *W.STATE_RICH, *W.HISTORY_RICH])
    t, e = d["pfs_mos"].to_numpy(float), d["pfs_event"].to_numpy(float)

    print(f"GENIE-BPC {'+'.join(sorted(d['cancer'].unique()))}: {len(d)} landmarks, "
          f"{d['record_id'].nunique()} patients")
    print(f"endpoint: imaging-confirmed PFS, median {np.median(t):.1f} months, "
          f"{e.mean():.0%} progressed\n")

    det = deterministic_arms(d, x, args.k)
    null = B.integrated_brier(np.tile(B.km_curve(t, e, B.GRID), (len(d), 1)), t, e,
                              B.GRID, t, e)
    print("DETERMINISTIC ARMS — computed here, no model, no spend")
    print(f"   {'arm':22s} {'IBS':>9} {'skill':>8} {'C-index':>9} {'skill@10':>11}")
    print("   " + "-" * 65)
    scores = {}
    from lifelines.utils import concordance_index
    for name, surv in det.items():
        sv = np.clip(surv, 1e-6, 1.0)
        ibs = B.integrated_brier(sv, t, e, B.GRID, t, e)
        ci = float(concordance_index(t, -(1.0 - sv[:, 5]), e))
        # AND THE SAME ARM SAMPLED LIKE THE MODEL. A deterministic arm emits a smooth survival
        # curve; an LLM arm emits `n_gen` draws, and turning any distribution into n draws costs
        # skill however good the distribution is. Scoring the smooth curve against the model's
        # empirical one charges a penalty to one side only, and the gap would be read as the
        # model losing. Both numbers are reported so the comparison can be made like with like.
        sampled = {str(n): _sampled_skill(sv, t, e, null, n_gen=n, seed=SEED)
                   for n in SAMPLED_AT}
        scores[name] = {"ibs": ibs, "skill": 1 - ibs / null, "c_index": ci,
                        "skill_sampled": sampled}
        at10 = sampled.get("10", {}).get("skill_median", float("nan"))
        print(f"   {name:22s} {ibs:9.4f} {1 - ibs / null:8.3f} {ci:9.3f} {at10:11.3f}")
    # refit under the SAME representation as the arms above, after missingness indicators
    # were added: 0.160 -> 0.161, C-index unchanged. A bar fitted on a different feature
    # space is not a bar.
    print("\n   bar to beat: discrete-time RF, skill 0.161, C-index 0.681 "
          "(refit with missing indicators)")

    rng = np.random.default_rng(SEED)
    queries = sorted(rng.choice(len(d), size=min(args.limit, len(d)), replace=False).tolist())
    facts = landmark_histories(d, imaging=not args.no_prefix_imaging)
    prog = prognostic_weights(d, channels)
    corpus = build_corpus(d, channels, prog)
    pol = base_policy(args.k, args.render_target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    qinfo = emit_queries(d, queries, facts, args.out_dir,
                         imaging=not args.no_prefix_imaging)
    emitted = emit_prompts(d, corpus, pol, queries, args.out_dir)
    print("\nQUERY SIDE — one payload per landmark, shared by every arm")
    print(f"   {qinfo['n_queries']} queries, median {qinfo['median_prefix_events']} fixed "
          f"prefix events (max {qinfo['max_prefix_events']}); every landmark carries its "
          "index treatment at the cutoff")
    print(f"   collapsed {qinfo.get('n_duplicates_collapsed', 0)} duplicate and "
          f"{qinfo.get('n_merged_within_window', 0)} within-window repeat events, dropped "
          f"{qinfo.get('n_observations_at_landmark_dropped', 0)} observations dated ON the "
          f"landmark and {qinfo.get('n_dropped_pre_diagnosis', 0)} dated before diagnosis; "
          f"{qinfo.get('n_with_concurrent_treatment_at_cutoff', 0)} landmarks have a "
          "concurrent regimen that day, labelled as such")

    print(f"\nLLM ARMS — prompts written, nothing spent ({len(queries)} landmarks)")
    print(f"   {'arm':22s} {'renderer':12s} {'supported':>10} {'refused':>8} "
          f"{'no control':>11} {'median chars':>13}")
    print("   " + "-" * 84)
    for arm, info in emitted.items():
        print(f"   {arm:22s} {info['renderer']:12s} {info['n_supported']:10d} "
              f"{info['n_refused']:8d} {info['n_control_not_constructible']:11d} "
              f"{info['median_block_chars']:13d}")

    # the prompt lineage travels with the run. Without it a returned generation cannot be
    # attributed to a template, and the parent-hash guard protects only the machine that
    # composed the prompt, not the record of what was composed.
    manifest = {
        "prompt_provenance": provenance(),
        "seed": SEED,
        # recorded apart because they are different quantities and were being read as one
        "k_evidence_set": args.k, "k_knn_comparator": args.k,
        "render_target_displayed": args.render_target,
        "fairness_contract": ("R1 is a patient-grouped out-of-fold same-action k-NN "
                              "reference at the same k, not the identical R4 anchor set. "
                              "The exact evidence comparator is the anchor KM curve sampled "
                              "to the LLM draw count; render_target controls only row display."),
        "n_landmarks": len(d), "n_patients": int(d["record_id"].nunique()),
        "n_queries": len(queries), "cancers": sorted(d["cancer"].unique().tolist()),
        "endpoint": "imaging-confirmed PFS from line start, distributional",
        "grid_months": B.GRID.tolist(),
        "channels": channels, "unobserved": list(UNOBSERVED),
        "prognostic_weights": prog,
        "policy": pol.to_dict(), "policy_hash": pol.hash, "corpus_id": corpus.corpus_id,
        "feature_matrix_sha256": hashlib.sha256(x.tobytes()).hexdigest()[:16],
        "deterministic_scores": scores,
        "bar": {"model": "discrete-time RF", "skill": 0.161, "c_index": 0.681,
                "representation": "context+rich state+history with missingness indicators",
                "refit_commit": "same feature list as the arms in this run"},
        "llm_arms": emitted, "queries": qinfo,
        "note": "R2 is a local outcome contrast, not an ATE: assignment is observational.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}")
    print("   manifest.json + one prompts_*.jsonl per LLM arm. No model was called.")
    if args.limit < 500:
        print(f"\n   NOTE: {args.limit} landmarks resolves ~"
              f"{0.013 * (3465 / args.limit) ** 0.5:.3f} skill. This is a plumbing run; the "
              "decisive test needs 500+.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
