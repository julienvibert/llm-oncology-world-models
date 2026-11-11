#!/usr/bin/env python3
"""Close the loop: returned generations in, calibrated comparison out.

Takes whatever a generation run produced, turns each set of sampled futures into a PFS
distribution, and scores it against the same proper rule every deterministic arm was scored
with -- so an LLM arm and a Kaplan-Meier arm are directly comparable and neither can win by
hedging.

INPUT CONTRACT. One JSONL row per (arm, query):

    {"query_id": "...", "arm": "R4_same_action", "pfs_samples": [3.2, 5.1, ...],
     "refused": false, "model": "...", "policy_hash": "..."}

`pfs_samples` are months to progression, one per generated future. A row with `refused` true
contributes to the abstention rate and to no score -- an arm that declines is not the same
as an arm that guesses, and averaging them would hide exactly that difference.

THE FOUR QUADRANTS, which is where the answer actually lives:

    high support, high roughness   many neighbours who disagree. Naive averaging fails and
                                   the evidence to reason from is present. THE TARGET.
    high support, low roughness    neighbours agree; k-NN should already suffice.
    low support, high roughness    little evidence and it conflicts. Risky extrapolation,
                                   weakly testable, and not a place to claim a win.
    low support, low roughness     little evidence but a flat gradient. Simple extrapolation
                                   or aggregate evidence should carry it.

A pooled improvement that does not show up in the high-support rough quadrant is not the
effect this framework predicts, and should not be reported as though it were.

THE DECISIVE COMPARISON is not against the baselines. It is R4 against R6: identical prompt
shape, anchors drawn from the wrong region. If shuffling does not hurt, the model is reacting
to prompt volume rather than to evidence, and every other number is uninterpretable.

This script calls no model and spends nothing. It reads what a run returned.

GOVERNANCE: GENIE-BPC is under a data use agreement. Scores and counts only, to a gitignored
directory. No patient-level rows are written.

Run:  uv run python scripts/explore/score_anchored_pfs_v1.py --self-test
      uv run python scripts/explore/score_anchored_pfs_v1.py --generations run/gens.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_pfs_distribution_bar as B  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_anchored_pfs_v1"
SEED = 20260810
BOOTS = 600
COVERAGE_LEVELS = (0.50, 0.80, 0.90)
#: Anything outside this is a parse failure, not a prediction. Flagged, never clipped.
PLAUSIBLE_MONTHS = (0.0, 240.0)


def samples_to_survival(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical survival from generated futures: the share still progression-free at t."""
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        return np.full(len(grid), np.nan)
    return np.array([float((s > t).mean()) for t in grid])


def coverage(samples: list[np.ndarray], truth: np.ndarray, event: np.ndarray,
             level: float) -> float:
    """Share of patients whose observed PFS falls inside the central interval.

    A censored patient counts as covered when their censoring time is below the upper bound:
    all we know is that they had not progressed yet, which is consistent with any larger
    value. Scoring them as misses would punish an arm for the follow-up we happen to have.
    """
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    hit, n = 0, 0
    for s, t, e in zip(samples, truth, event, strict=True):
        s = s[np.isfinite(s)]
        if len(s) < 2:
            continue
        lo, hi = np.quantile(s, [lo_q, hi_q])
        n += 1
        hit += int(lo <= t <= hi) if e == 1 else int(t <= hi)
    return hit / n if n else float("nan")


def matrix_coverage(samples: np.ndarray, truth: np.ndarray, event: np.ndarray,
                    level: float) -> float:
    """Vectorized coverage for a finite-sample benchmark."""
    if samples.size == 0 or samples.shape[1] < 2:
        return float("nan")
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lo, hi = np.quantile(samples, [lo_q, hi_q], axis=1)
    event_hit = (event == 1) & (lo <= truth) & (truth <= hi)
    cens_hit = (event == 0) & (truth <= hi)
    return float((event_hit | cens_hit).mean())


def sample_from_survival_curve(surv: np.ndarray, grid: np.ndarray, *,
                               n_patients: int, n_samples: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Draw finite PFS samples from a discrete survival curve.

    The final mass is placed just beyond the scoring grid, so samples in the tail stay
    progression-free throughout the evaluated horizon.
    """
    start = np.concatenate([[1.0], np.clip(surv, 0.0, 1.0)])
    probs = np.maximum(start[:-1] - start[1:], 0.0)
    tail = max(float(surv[-1]), 0.0)
    values = np.concatenate([grid, [grid[-1] + 1.0]])
    probs = np.concatenate([probs, [tail]])
    probs = probs / probs.sum()
    return values[rng.choice(len(values), size=(n_patients, n_samples), p=probs)]


def finite_sample_null_benchmark(time: np.ndarray, event: np.ndarray,
                                 n_gen_values: list[int], *,
                                 reps: int = 100,
                                 seed: int = SEED) -> dict[str, dict]:
    """Monte-Carlo penalty for scoring finite samples from the marginal KM curve.

    This is the correct no-patient-information null *after* converting only ``n_gen``
    samples into an empirical survival curve. With few samples it scores below the true
    marginal KM even though the generating distribution is the marginal KM.
    """
    rng = np.random.default_rng(seed)
    marginal = B.km_curve(time, event, B.GRID)
    null_ibs = B.integrated_brier(np.tile(marginal, (len(time), 1)), time, event,
                                  B.GRID, time, event)
    out: dict[str, dict] = {}
    for n_gen in sorted(set(int(n) for n in n_gen_values if int(n) >= 2)):
        skills, covs = [], {lv: [] for lv in COVERAGE_LEVELS}
        for _ in range(reps):
            samples = sample_from_survival_curve(
                marginal, B.GRID, n_patients=len(time), n_samples=n_gen, rng=rng)
            surv = (samples[:, :, None] > B.GRID[None, None, :]).mean(axis=1)
            ibs = B.integrated_brier(surv, time, event, B.GRID, time, event)
            skills.append(1 - ibs / null_ibs)
            for lv in COVERAGE_LEVELS:
                covs[lv].append(matrix_coverage(samples, time, event, lv))
        row = {
            "n_gen": n_gen,
            "reps": reps,
            "skill_median": float(np.median(skills)),
            "skill_lo": float(np.percentile(skills, 2.5)),
            "skill_hi": float(np.percentile(skills, 97.5)),
        }
        for lv in COVERAGE_LEVELS:
            vals = np.asarray(covs[lv], float)
            row[f"cover_{int(lv * 100)}_median"] = float(np.median(vals))
            row[f"cover_{int(lv * 100)}_lo"] = float(np.percentile(vals, 2.5))
            row[f"cover_{int(lv * 100)}_hi"] = float(np.percentile(vals, 97.5))
        out[str(n_gen)] = row
    return out


def load_generations(path: Path) -> pd.DataFrame:
    """Read and validate. Malformed rows are reported, never silently dropped."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    g = pd.DataFrame(rows)
    for col in ("query_id", "arm"):
        if col not in g.columns:
            raise ValueError(f"generations are missing required column {col!r}")
    if "refused" not in g.columns:
        g["refused"] = False
    if "pfs_samples" not in g.columns:
        raise ValueError("generations must carry pfs_samples")
    g["pfs_samples"] = g["pfs_samples"].apply(
        lambda v: np.asarray(v, float) if isinstance(v, list) else np.array([]))
    lo, hi = PLAUSIBLE_MONTHS
    g["n_samples"] = g["pfs_samples"].apply(len)
    g["n_implausible"] = g["pfs_samples"].apply(
        lambda s: int(((s < lo) | (s > hi) | ~np.isfinite(s)).sum()))
    return g


def validate_run_matrix(g: pd.DataFrame, manifest: dict, *, definitive: bool) -> None:
    """Refuse to score anything but the exact registered matrix.

    This scorer once analysed an interrupted, breast-only prefix of 23 of 500 queries and
    reported arm contrasts from it without complaint. Producing a clean-looking readout from
    4% of a run, in query order, is the failure mode most likely to be believed: the numbers
    are well formed and the cohort is a systematic subset. Under --definitive the matrix must
    be complete AND match the manifest; otherwise the shortfall is reported and scoring
    continues, clearly labelled.
    """
    want_queries = [str(q) for q in manifest.get("query_ids") or []]
    want_arms = [str(a) for a in manifest.get("arms") or []]
    want_repeats = int(manifest.get("repeats") or 0)
    if not (want_queries and want_arms and want_repeats):
        raise SystemExit("[fatal] manifest does not declare query_ids, arms and repeats; "
                         "the run matrix cannot be validated")

    problems: list[str] = []
    got_queries = {str(q) for q in g["query_id"]}
    got_arms = {str(a) for a in g["arm"]}
    missing_q = [q for q in want_queries if q not in got_queries]
    extra_q = sorted(got_queries - set(want_queries))
    if missing_q:
        problems.append(f"{len(missing_q)} of {len(want_queries)} registered queries absent "
                        f"(first: {missing_q[:3]})")
    if extra_q:
        problems.append(f"queries not in the manifest: {extra_q[:3]}")
    if set(want_arms) != got_arms:
        problems.append(f"arms: registered {sorted(want_arms)}, present {sorted(got_arms)}")

    # REQUESTED, not accepted. A generation can be legitimately rejected -- a contract
    # violation, a truncation -- and the design allocates a fixed number of ATTEMPTS per
    # cell, not a guaranteed number of survivors. Checking accepted samples here would fail
    # almost every cell of a healthy run and then be relaxed, which is worse than not
    # checking. Unequal surviving fans are a real problem, and match_fan_sizes handles it.
    cells = g.groupby(["query_id", "arm"])
    counted = (cells["n_requested"].sum() if "n_requested" in g.columns
               else cells["arm"].size() * 0 + want_repeats)
    expected_cells = len(want_queries) * len(want_arms)
    short = int((counted < want_repeats).sum())
    if len(counted) != expected_cells:
        problems.append(f"{len(counted)} query-by-arm cells present, "
                        f"{expected_cells} registered")
    if short:
        problems.append(f"{short} cells REQUESTED fewer than the registered {want_repeats} "
                        "futures")
    if "n_requested" not in g.columns:
        problems.append("generations carry no n_requested column, so requested-count "
                        "completeness could not be verified")
    empty = int((cells["n_samples"].sum() == 0).sum())
    if empty:
        problems.append(f"{empty} cells have NO accepted futures at all")

    if problems:
        head = "[fatal] this is not the registered run matrix:" if definitive else \
            "[warn] PARTIAL RUN -- results below are not the registered matrix:"
        message = head + "\n  - " + "\n  - ".join(problems)
        if definitive:
            raise SystemExit(message)
        print(message + "\n")


def decision_table() -> pd.DataFrame:
    """The canonical landmark table with `query_id`, from the builder that defines it.

    Imported lazily: the builder imports THIS module at import time to cross-check arm
    names, so a top-level import here is a cycle. Delegating rather than re-deriving is the
    whole point -- query_id is `record_id@<positional index>` and only means anything
    relative to the exact row sequence that produced it.
    """
    import build_anchored_pfs_arms_v1 as Bld
    return Bld.decision_table()


def comparators_on_scored_landmarks(query_ids: set[str], *, n_gen: int, k: int = 40,
                                    reps: int = 40, seed: int = SEED) -> dict | None:
    """The deterministic arms, scored on the SAME landmarks and at the SAME draw count.

    The builder computes these on all 11,196 landmarks because the neighbour search needs
    them; the LLM arms are scored on the 500 that were sampled. Printing the two beside each
    other compares different cohorts under different null denominators, and the gap it
    implies is not real: k-NN reads 0.118 on the full cohort, 0.093 on the exact 500, and
    +0.012 once discretised to ten draws the way a generative arm is. That last number is
    the only one a ten-draw LLM arm can be compared with.
    """
    import build_anchored_pfs_arms_v1 as Bld

    d = Bld.decision_table()
    sel = d["query_id"].isin(query_ids).to_numpy()
    if sel.sum() == 0:
        return None
    x = W_FEATURES(d)
    t = d["pfs_mos"].to_numpy(float)
    e = d["pfs_event"].to_numpy(float)
    det = Bld.deterministic_arms(d, x, k)
    sub_t, sub_e = t[sel], e[sel]
    null = B.integrated_brier(np.tile(B.km_curve(sub_t, sub_e, B.GRID), (int(sel.sum()), 1)),
                              sub_t, sub_e, B.GRID, sub_t, sub_e)
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for name, surv in det.items():
        smooth = 1 - B.integrated_brier(np.clip(surv[sel], 1e-6, 1.0), sub_t, sub_e,
                                        B.GRID, sub_t, sub_e) / null
        draws = []
        for _ in range(reps):
            samples = np.stack([
                sample_from_survival_curve(surv[i], B.GRID, n_patients=1, n_samples=n_gen,
                                           rng=rng)[0]
                for i in np.where(sel)[0]])
            emp = (samples[:, :, None] > B.GRID[None, None, :]).mean(axis=1)
            draws.append(1 - B.integrated_brier(np.clip(emp, 1e-6, 1), sub_t, sub_e,
                                                B.GRID, sub_t, sub_e) / null)
        out[name] = {"skill_smooth": float(smooth),
                     "skill_at_n_gen": float(np.median(draws)),
                     "lo": float(np.percentile(draws, 2.5)),
                     "hi": float(np.percentile(draws, 97.5))}
    return {"n_landmarks": int(sel.sum()), "n_gen": n_gen, "arms": out}


def W_FEATURES(d):
    import measure_anchor_operating_window as W
    return W.feature_matrix(d, [*W.CONTEXT, *W.STATE_RICH, *W.HISTORY_RICH])


def provider_sensitivity(attempts_path: Path, curves: dict, comparisons: list,
                         seed: int = SEED, perms: int = 2000) -> dict | None:
    """A3, materialised: is the arm contrast an artefact of which backend served which arm?

    Two prespecified outputs. MASD with a QUERY-CLUSTERED permutation null -- arm labels are
    permuted within a query, since every query contributes all three arms, so the null keeps
    each query's own set of backends and asks only whether their assignment to arms matters.
    And a standardised contrast, reweighting each arm to the pooled backend composition, so
    an arm cannot benefit from having drawn a better-performing backend more often.
    """
    if not attempts_path.exists():
        return None
    import build_anchored_pfs_arms_v1 as Bld  # noqa: F401  (kept for symmetry of imports)

    rows = []
    for line in attempts_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted"):
            continue
        details = r.get("attempts_detail") or []
        winner = next((d for d in details if d.get("accepted")), details[-1] if details else {})
        name = str((winner.get("usage") or {}).get("served_provider") or "").strip().lower()
        if name:
            rows.append({"query_id": r["query_id"], "arm": r["arm"], "provider": name})
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["query_id", "arm"])

    def masd(frame: pd.DataFrame) -> float:
        share = (frame.groupby(["arm", "provider"]).size()
                 / frame.groupby("arm").size()).unstack(fill_value=0.0)
        return float((share.max(axis=0) - share.min(axis=0)).max()) if len(share) else 0.0

    observed = masd(df)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(perms):
        shuffled = df.copy()
        shuffled["arm"] = (shuffled.groupby("query_id")["arm"]
                           .transform(lambda a: rng.permutation(a.to_numpy())))
        null.append(masd(shuffled))
    null = np.asarray(null)
    p_value = float((null >= observed).mean())

    # standardisation: weight each (arm, provider) cell to the POOLED provider composition
    pooled = df["provider"].value_counts(normalize=True)
    standardised = {}
    for a, b, _why in comparisons:
        if a not in curves or b not in curves:
            continue
        qa, sa, ta, ea, pa = curves[a]
        qb, sb, _tb, _eb, _pb = curves[b]
        shared = np.intersect1d(qa, qb)
        if len(shared) < 30:
            continue
        ia = np.array([np.where(qa == q)[0][0] for q in shared])
        ib = np.array([np.where(qb == q)[0][0] for q in shared])
        prov = dict(zip(df[df.arm == a]["query_id"], df[df.arm == a]["provider"],
                        strict=False))
        who = np.array([prov.get(q, "") for q in shared])
        raw = _skill_difference(sa[ia], sb[ib], ta[ia], ea[ia], np.arange(len(shared)))
        parts, weights = [], []
        for name, weight in pooled.items():
            mask = who == name
            if mask.sum() < 10:
                continue
            value = _skill_difference(sa[ia], sb[ib], ta[ia], ea[ia], np.where(mask)[0])
            if value is not None:
                parts.append(value)
                weights.append(weight)
        std = (float(np.average(parts, weights=weights)) if parts else None)
        standardised[f"{a} - {b}"] = {"raw": raw, "provider_standardised": std}
    return {"masd": observed, "masd_permutation_p": p_value, "n_permutations": perms,
            "providers": sorted(df["provider"].unique()),
            "standardised_contrasts": standardised}


def validate_against_prereg(run_manifest: dict, prereg_path: Path) -> None:
    """Under --definitive, the RUN ITSELF must be the registered design.

    validate_run_matrix only checks the generations against the run manifest, which proves
    internal consistency and nothing else: a 10-query qualification is perfectly consistent
    with its own manifest and passed --definitive scoring without complaint. The runner
    enforces the design before spending; the scorer has to enforce it again before reporting,
    or a smoke can be scored under the same flag as the definitive run.
    """
    if not prereg_path.exists():
        raise SystemExit(f"[fatal] --definitive needs the preregistration; {prereg_path} "
                         "is missing")
    import yaml
    alloc = (yaml.safe_load(prereg_path.read_text()) or {}).get("allocation", {})
    problems = []
    got_landmarks = len(run_manifest.get("query_ids") or [])
    if int(alloc.get("landmarks", -1)) != got_landmarks:
        problems.append(f"landmarks: registered {alloc.get('landmarks')}, "
                        f"run has {got_landmarks}")
    if int(alloc.get("futures_per_landmark", -1)) != int(run_manifest.get("repeats") or 0):
        problems.append(f"futures per landmark: registered "
                        f"{alloc.get('futures_per_landmark')}, run has "
                        f"{run_manifest.get('repeats')}")
    if sorted(alloc.get("arms") or []) != sorted(run_manifest.get("arms") or []):
        problems.append(f"arms: registered {sorted(alloc.get('arms') or [])}, run has "
                        f"{sorted(run_manifest.get('arms') or [])}")
    want_model = str(alloc.get("model", ""))
    norm = lambda x: "".join(c for c in str(x).lower() if c.isalnum())     # noqa: E731
    got_model = str(run_manifest.get("model_id") or run_manifest.get("model_short") or "")
    if want_model and norm(want_model) not in norm(got_model):
        problems.append(f"model: registered {want_model}, run used {got_model}")
    if problems:
        raise SystemExit("[fatal] --definitive was requested but this run is not the "
                         "registered design:\n  - " + "\n  - ".join(problems))


def match_fan_sizes(g: pd.DataFrame, *, seed: int = SEED) -> pd.DataFrame:
    """Cut every arm of a query down to that query's smallest surviving fan.

    Arms are compared per patient, and turning a distribution into M draws costs skill in a
    way that depends on M -- so an arm scored at 10 futures against an arm scored at 6 is
    partly a comparison of M. Failures are not arm-neutral either: on the partial, R3 had the
    lowest acceptance, which would have handed the baseline arm the smaller fan systematically.

    Subsampling is DETERMINISTIC RANDOM, never the first M. Samples are stored in completion
    order, which is latency-correlated: taking the head would select the fastest generations,
    and generation latency is not independent of what was generated.
    """
    rng = np.random.default_rng(seed)
    out = g.copy()
    keep: list[np.ndarray] = []
    per_query = out.groupby("query_id")["n_samples"].min()
    for row in out.itertuples(index=False):
        target = int(per_query[row.query_id])
        samples = np.asarray(row.pfs_samples, float)
        if target <= 0 or len(samples) <= target:
            keep.append(samples)
            continue
        # Seeded per (query, arm) so a rerun reproduces the identical subsample, and so
        # adding a query cannot renumber another query's draw. SHA-256, NOT Python hash():
        # str hashing is salted per interpreter, so hash() would return a different value in
        # every process and the "deterministic" subsample would silently differ between runs.
        # The unit test that claimed to check determinism ran inside one process and could
        # not see it.
        digest = hashlib.sha256(f"{seed}|{row.query_id}|{row.arm}".encode()).digest()
        local = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        keep.append(samples[np.sort(local.choice(len(samples), target, replace=False))])
    out["pfs_samples"] = keep
    out["n_samples"] = out["pfs_samples"].apply(len)
    assert rng is not None
    return out


def anchor_reference(curves: list[np.ndarray], t: np.ndarray, e: np.ndarray,
                     null_ibs: float, *, n_gen: int, reps: int = 200,
                     seed: int = SEED) -> dict:
    """Score the ANCHOR evidence itself, sampled the same number of times the model was.

    This is the comparator the adaptation question needs. The anchor Kaplan-Meier curve is
    smooth; the model's output is an empirical curve from `n_gen` draws, and converting a
    distribution into `n_gen` draws costs skill regardless of how good the distribution is --
    about 0.05 at twenty draws and 0.49 at two, measured. Comparing the model against the
    SMOOTH anchor curve therefore charges the model a penalty the anchor never pays, and the
    difference would be read as the model failing to adapt.

    So the anchor curve is drawn from `n_gen` times per query, scored exactly like the model's
    samples, and repeated. The median and 95% interval of that distribution are the reference,
    and the interval is reported: a difference means nothing without the spread of the thing
    it is measured against.
    """
    rng = np.random.default_rng(seed)
    usable = [(i, c) for i, c in enumerate(curves)
              if c is not None and len(c) == len(B.GRID) and np.all(np.isfinite(c))]
    if len(usable) < 5:
        return {}
    idx = np.array([i for i, _ in usable])
    stack = np.array([c for _, c in usable], float)
    skills = []
    for _ in range(reps):
        surv = np.empty((len(idx), len(B.GRID)))
        for row, curve in enumerate(stack):
            draws = sample_from_survival_curve(curve, B.GRID, n_patients=1,
                                               n_samples=n_gen, rng=rng)[0]
            surv[row] = samples_to_survival(draws, B.GRID)
        ibs = B.integrated_brier(np.clip(surv, 1e-6, 1.0), t[idx], e[idx], B.GRID, t, e)
        skills.append(1 - ibs / null_ibs)
    skills = np.array(skills)
    return {"n_queries": int(len(idx)), "n_gen": int(n_gen), "reps": int(reps),
            "skill_median": float(np.median(skills)),
            "skill_lo": float(np.percentile(skills, 2.5)),
            "skill_hi": float(np.percentile(skills, 97.5)),
            "queries_used": idx.tolist()}


def randomised_pit_from_draws(samples: list, time: np.ndarray, event: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Where reality falls in each fan, in [0,1]. ONE implementation, from the trust map.

    Two randomisations are needed and an earlier version here did only the second. With an
    m-member fan the RANK ITSELF is discrete -- S(T) can only take m+1 values -- so without a
    jitter inside the rank the histogram is spiky for reasons that have nothing to do with the
    model, and the KS statistic is inflated by the discreteness alone. That is visible as
    deciles landing on exact multiples of 0.1. A CENSORED patient is then drawn from the part
    of the fan still compatible with T > C rather than dropped.
    """
    import trust_map_pit_v1 as T
    return np.array([T.randomised_pit(np.asarray(s_, float), float(t_), bool(e_), rng)
                     for s_, t_, e_ in zip(samples, time, event, strict=True)])


def pit_arm_contrasts(gens: pd.DataFrame, arms: list[str], *, reps: int = 400,
                      boots: int = 1200, seed: int = SEED) -> dict:
    """Pairwise PIT-uniformity contrasts, integrating over the jitter.

    THE JITTER IS A NUISANCE PARAMETER, NOT A RESULT. The finite-rank randomisation is a
    device for un-discretising the rank, so a single draw of it is one realisation of
    something we do not care about -- and the realisations are wide enough to flip a
    conclusion: across seeds the R4-R3 contrast ranged -0.046 to -0.085, and reporting the
    -0.046 draw made a real effect look absent.

    The point estimate is therefore the MEDIAN observed contrast over `reps` randomisations,
    and each bootstrap replicate resamples patients AND redraws the jitter, so both variance
    sources are in the interval. Lower KS is better, so a negative contrast favours the
    first arm.
    """
    import trust_map_pit_v1 as T
    from scipy import stats

    ref = gens[gens.arm == arms[0]].sort_values("query_id")
    order = ref["query_id"].to_numpy()
    t = ref["pfs_mos"].to_numpy(float)
    e = ref["pfs_event"].to_numpy(float)
    patient = ref["record_id"].to_numpy()
    draws = {a: gens[gens.arm == a].set_index("query_id").loc[order, "pfs_samples"].tolist()
             for a in arms if (gens.arm == a).any()}
    if len(draws) < 2:
        return {}

    def pit_of(arm: str, rs: int) -> np.ndarray:
        rng = np.random.default_rng(rs)
        return np.array([T.randomised_pit(np.asarray(x, float), float(ti), bool(ei), rng)
                         for x, ti, ei in zip(draws[arm], t, e, strict=True)])

    def ks(values: np.ndarray) -> float:
        return float(stats.kstest(values, "uniform").statistic)

    pairs = [(a, b) for i, a in enumerate(draws) for b in list(draws)[i + 1:]]
    point: dict[tuple, list] = {pr: [] for pr in pairs}
    per_arm: dict[str, list] = {a: [] for a in draws}
    for r in range(reps):
        cur = {a: pit_of(a, seed + r) for a in draws}
        stat = {a: ks(cur[a]) for a in draws}
        for a in draws:
            per_arm[a].append(stat[a])
        for a, b in pairs:
            point[(a, b)].append(stat[a] - stat[b])

    rng = np.random.default_rng(seed)
    pats = np.unique(patient)
    by = {q: np.where(patient == q)[0] for q in pats}
    boot: dict[tuple, list] = {pr: [] for pr in pairs}
    for r in range(boots):
        cur = {a: pit_of(a, seed + 100_000 + r) for a in draws}
        take = np.concatenate([by[q] for q in rng.choice(pats, len(pats), replace=True)])
        stat = {a: ks(cur[a][take]) for a in draws}
        for a, b in pairs:
            boot[(a, b)].append(stat[a] - stat[b])

    out = {"per_arm_ks": {a: {"median": float(np.median(v)),
                              "lo": float(np.percentile(v, 2.5)),
                              "hi": float(np.percentile(v, 97.5))}
                          for a, v in per_arm.items()},
           "reps": reps, "boots": boots, "contrasts": {}}
    for a, b in pairs:
        v = np.asarray(boot[(a, b)])
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        out["contrasts"][f"{a} - {b}"] = {
            "delta_ks": float(np.median(point[(a, b)])),
            "jitter_only_lo": float(np.percentile(point[(a, b)], 2.5)),
            "jitter_only_hi": float(np.percentile(point[(a, b)], 97.5)),
            "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def population_registration(gens: pd.DataFrame, *, arms: list[str], boots: int = 600,
                           min_stratum: int = 25, seed: int = SEED) -> dict | None:
    """Does the generated world keep the SEPARATION between clinical populations?

    Two failure modes have to be told apart, and a spread or a rank correlation over three
    cancers cannot do it -- on this run the Spearman sign FLIPS between medians (-0.50) and
    RMST-24 (+0.50), which is what three groups buys.

    So the readout is a registration line across prespecified strata,

        mu_stratum(generated) = alpha + beta * mu_stratum(reference)

    where alpha is a global translation -- the calibration shift anchoring is supposed to
    deliver -- and beta is what happens to the structure:

        beta = 1   separation preserved
        0<beta<1   attenuated: populations pushed together
        beta = 0   flattened: one distribution for every population
        beta < 0   topology inverted

    Fitted against reality AND against the anchor evidence, because those answer different
    questions: the first asks whether the generated world matches the real one, the second
    whether the model transmitted the structure it was actually shown.
    """
    need = {"cancer", "line_number"}
    if not need <= set(gens.columns):
        return None
    g = gens.copy()
    g["line_group"] = np.where(g["line_number"] <= 1, "1L", "2L+")
    g["stratum"] = g["cancer"].astype(str) + "|" + g["line_group"]

    def rmst(values: np.ndarray) -> float:
        return float(np.mean(np.minimum(np.asarray(values, float), 24.0)))

    def summarise(frame: pd.DataFrame, arm: str) -> dict[str, float]:
        sub = frame[frame.arm == arm]
        out = {}
        for stratum, part in sub.groupby("stratum"):
            if len(part) < min_stratum:
                continue
            pooled = np.concatenate([np.asarray(x, float) for x in part["pfs_samples"]])
            out[stratum] = rmst(pooled)
        return out

    def real_summary(frame: pd.DataFrame) -> dict[str, float]:
        ref = frame[frame.arm == arms[0]]
        out = {}
        for stratum, part in ref.groupby("stratum"):
            if len(part) < min_stratum:
                continue
            km = B.km_curve(part["pfs_mos"].to_numpy(float),
                            part["pfs_event"].to_numpy(float), B.GRID)
            out[stratum] = float(np.trapezoid(km, B.GRID))
        return out

    def fit(x: dict, y: dict) -> tuple[float, float] | None:
        keys = sorted(set(x) & set(y))
        if len(keys) < 3:
            return None
        a = np.array([x[k] for k in keys])
        b = np.array([y[k] for k in keys])
        beta, alpha = np.polyfit(a, b, 1)
        return float(alpha), float(beta)

    rng = np.random.default_rng(seed)
    patients = gens[gens.arm == arms[0]]["record_id"].unique()
    out: dict[str, dict] = {}
    for arm in arms:
        base = fit(real_summary(g), summarise(g, arm))
        if base is None:
            continue
        betas, alphas = [], []
        for _ in range(boots):
            take = set(rng.choice(patients, len(patients), replace=True))
            sub = g[g.record_id.isin(take)]
            got = fit(real_summary(sub), summarise(sub, arm))
            if got is not None:
                alphas.append(got[0])
                betas.append(got[1])
        out[arm] = {
            "n_strata": len(set(real_summary(g)) & set(summarise(g, arm))),
            "alpha": base[0], "beta": base[1],
            "beta_lo": float(np.percentile(betas, 2.5)) if betas else float("nan"),
            "beta_hi": float(np.percentile(betas, 97.5)) if betas else float("nan"),
            "alpha_lo": float(np.percentile(alphas, 2.5)) if alphas else float("nan"),
            "alpha_hi": float(np.percentile(alphas, 97.5)) if alphas else float("nan"),
        }
    return {"statistic": "RMST-24 by cancer x line group", "arms": out,
            "min_stratum": min_stratum, "boots": boots}


def adaptation_contrast(model_surv: np.ndarray, anchor_curves: list, t: np.ndarray,
                        e: np.ndarray, patient: np.ndarray, *, n_gen: int,
                        boots: int = 400, seed: int = SEED) -> dict | None:
    """Model minus anchor-only, paired, with a PATIENT-CLUSTERED interval.

    The previous readout compared the model's point skill against the Monte-Carlo spread of
    sampling the anchor curve. That interval describes how precisely the REFERENCE was
    estimated; it says nothing about whether this model beats it on these patients, and it
    made a point estimate BELOW the interval print as "within".

    Each bootstrap replicate resamples patients and redraws the anchor sample, so both the
    between-patient and the finite-ensemble variance land in the same interval.
    """
    usable = [i for i, c in enumerate(anchor_curves)
              if isinstance(c, (list, np.ndarray)) and np.size(c) == len(B.GRID)]
    if len(usable) < 30:
        return None
    idx = np.array(usable, int)
    curves = np.array([np.asarray(anchor_curves[i], float) for i in idx])
    tt, ee, pp = t[idx], e[idx], patient[idx]
    rng = np.random.default_rng(seed)
    pats = np.unique(pp)
    by = {q: np.where(pp == q)[0] for q in pats}

    def one(take: np.ndarray, draw: np.random.Generator) -> float | None:
        sub_t, sub_e = tt[take], ee[take]
        null = B.integrated_brier(np.tile(B.km_curve(sub_t, sub_e, B.GRID), (len(take), 1)),
                                  sub_t, sub_e, B.GRID, sub_t, sub_e)
        if null <= 0:
            return None
        samples = np.stack([sample_from_survival_curve(curves[i], B.GRID, n_patients=1,
                                                       n_samples=n_gen, rng=draw)[0]
                            for i in take])
        anchor_emp = (samples[:, :, None] > B.GRID[None, None, :]).mean(axis=1)
        m = 1 - B.integrated_brier(np.clip(model_surv[idx][take], 1e-6, 1), sub_t, sub_e,
                                   B.GRID, sub_t, sub_e) / null
        a = 1 - B.integrated_brier(np.clip(anchor_emp, 1e-6, 1), sub_t, sub_e,
                                   B.GRID, sub_t, sub_e) / null
        return float(m - a)

    observed = one(np.arange(len(idx)), np.random.default_rng(seed))
    diffs = []
    for _ in range(boots):
        take = np.concatenate([by[q] for q in rng.choice(pats, len(pats), replace=True)])
        value = one(take, rng)
        if value is not None:
            diffs.append(value)
    d = np.array(diffs)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    verdict = "beats" if lo > 0 else ("below" if hi < 0 else "within")
    return {"n_queries": len(idx), "n_gen": n_gen, "delta_adapt": observed,
            "lo": lo, "hi": hi, "verdict": verdict, "n_boot": int(len(d))}


def score_arm(surv: np.ndarray, samples: list[np.ndarray], t: np.ndarray, e: np.ndarray,
              null_ibs: float) -> dict:
    from lifelines.utils import concordance_index
    s = np.clip(surv, 1e-6, 1.0)
    ibs = B.integrated_brier(s, t, e, B.GRID, t, e)
    # PREREGISTERED: randomised over BOTH the finite rank and the censoring, all patients
    # included. B.pit_values drops censored ones, which at 71% events discards 29% of the
    # cohort and selects for fast progressors.
    pit = randomised_pit_from_draws(samples, t, e, np.random.default_rng(SEED))
    ks = (float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit), endpoint=False))))
          if len(pit) else float("nan"))
    out = {"n": int(len(t)), "ibs": ibs, "skill": 1 - ibs / null_ibs,
           "c_index": float(concordance_index(t, -(1.0 - s[:, 5]), e)), "pit_ks": ks}
    for lv in COVERAGE_LEVELS:
        out[f"cover_{int(lv * 100)}"] = coverage(samples, t, e, lv)
    return out


def _skill_difference(a: np.ndarray, b: np.ndarray, t: np.ndarray, e: np.ndarray,
                      take: np.ndarray) -> float | None:
    """Skill(a) - skill(b) on one index set, or None when the null model is degenerate."""
    tt, ee = t[take], e[take]
    null = B.integrated_brier(np.tile(B.km_curve(tt, ee, B.GRID), (len(take), 1)),
                              tt, ee, B.GRID, tt, ee)
    if null <= 0:
        return None
    sa = 1 - B.integrated_brier(np.clip(a[take], 1e-6, 1), tt, ee, B.GRID, tt, ee) / null
    sb = 1 - B.integrated_brier(np.clip(b[take], 1e-6, 1), tt, ee, B.GRID, tt, ee) / null
    return float(sa - sb)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, t: np.ndarray, e: np.ndarray,
                     patient: np.ndarray, boots: int = BOOTS) -> dict:
    """Skill difference a - b, resampled BY PATIENT, on the queries both arms answered.

    THE POINT ESTIMATE IS THE OBSERVED CONTRAST, not the mean of the bootstrap replicates.
    An earlier version reported `diffs.mean()`, which is a different quantity: the resampling
    distribution is centred on the observed value plus a bias term, so its mean answers
    "what would this statistic average to under resampling" rather than "what did we see".
    The bootstrap's job here is the interval. The bias it does reveal is reported rather than
    silently absorbed, because a large one is itself a finding about the estimator.
    """
    rng = np.random.default_rng(SEED)
    pats = np.unique(patient)
    idx_by = {p: np.where(patient == p)[0] for p in pats}
    observed = _skill_difference(a, b, t, e, np.arange(len(t)))
    diffs = []
    for _ in range(boots):
        take = np.concatenate([idx_by[p] for p in rng.choice(pats, len(pats), replace=True)])
        value = _skill_difference(a, b, t, e, take)
        if value is not None:
            diffs.append(value)
    d = np.array(diffs)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta_skill": observed,
            "bootstrap_mean": float(d.mean()),
            "bootstrap_bias": (None if observed is None else float(d.mean() - observed)),
            "lo": lo, "hi": hi, "n_boot": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def quadrant(support: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """The four regions, split at medians. Names say what each one means for the method."""
    hi_s = support >= np.median(support)
    hi_r = roughness >= np.nanmedian(roughness)
    out = np.full(len(support), "", dtype=object)
    out[hi_s & hi_r] = "high support, rough — TARGET"
    out[hi_s & ~hi_r] = "high support, flat — kNN suffices"
    out[~hi_s & hi_r] = "low support, rough — risky"
    out[~hi_s & ~hi_r] = "low support, flat — extrapolate"
    return out


def compare_arm_curves(curves: dict[str, tuple], a: str, b: str, *,
                       query_subset: set[str] | None = None,
                       min_shared: int = 30) -> dict | None:
    """Paired, patient-clustered comparison on shared queries, optionally in one stratum."""
    if a not in curves or b not in curves:
        return None
    qa, sa, ta, ea, pa = curves[a]
    qb, sb, _, _, _ = curves[b]
    ia_by = {str(q): i for i, q in enumerate(qa)}
    ib_by = {str(q): i for i, q in enumerate(qb)}
    shared = sorted(set(ia_by) & set(ib_by))
    if query_subset is not None:
        shared = [q for q in shared if q in query_subset]
    if len(shared) < min_shared:
        return {"n_shared": int(len(shared)), "scored": False,
                "minimum_required": int(min_shared)}
    ia = np.array([ia_by[q] for q in shared], int)
    ib = np.array([ib_by[q] for q in shared], int)
    res = paired_bootstrap(sa[ia], sb[ib], ta[ia], ea[ia], pa[ia])
    return {**res, "n_shared": int(len(shared)), "scored": True}


TARGET_QUADRANT = "high support, rough — TARGET"


def score_quadrants(curves: dict[str, tuple], r4: pd.DataFrame,
                    comparisons: tuple[tuple[str, str, str], ...], *,
                    min_shared: int = 20) -> dict:
    """Evaluate registered paired effects within R4-defined support/roughness quadrants."""
    needed = {"query_id", "support", "roughness"}
    if not needed.issubset(r4.columns):
        return {}
    base = r4.loc[
        np.isfinite(pd.to_numeric(r4["support"], errors="coerce"))
        & np.isfinite(pd.to_numeric(r4["roughness"], errors="coerce"))
    ].copy()
    if base.empty:
        return {}
    base["quadrant"] = quadrant(base["support"].to_numpy(float),
                                base["roughness"].to_numpy(float))
    by_quadrant = {
        str(name): set(group["query_id"].astype(str))
        for name, group in base.groupby("quadrant")
    }
    out: dict[str, dict] = {}
    for a, b, why in comparisons:
        key = f"{a} - {b}"
        cells = {
            name: compare_arm_curves(curves, a, b, query_subset=ids,
                                     min_shared=min_shared)
            for name, ids in by_quadrant.items()
        }
        cells = {name: value for name, value in cells.items() if value is not None}
        scored = {name: value for name, value in cells.items() if value.get("scored")}
        largest = (max(scored, key=lambda name: scored[name]["delta_skill"])
                   if scored else None)
        out[key] = {
            "why": why,
            "quadrants": cells,
            "largest_point_estimate_quadrant": largest,
            "target_is_largest": bool(largest == TARGET_QUADRANT) if largest else None,
        }
    return out


def self_test() -> int:
    grid = B.GRID
    rng = np.random.default_rng(0)
    n = 300
    truth = rng.exponential(6.0, n)
    event = np.ones(n)

    # a generator that samples the true distribution beats one that does not, and the
    # scorer must rank them that way or nothing downstream means anything
    good = [rng.exponential(6.0, 30) for _ in range(n)]
    bad = [rng.exponential(30.0, 30) for _ in range(n)]
    sg = np.array([samples_to_survival(s, grid) for s in good])
    sb = np.array([samples_to_survival(s, grid) for s in bad])
    null = B.integrated_brier(np.tile(B.km_curve(truth, event, grid), (n, 1)),
                              truth, event, grid, truth, event)
    assert score_arm(sg, good, truth, event, null)["skill"] > \
        score_arm(sb, bad, truth, event, null)["skill"]

    # a wider forecast covers more but must not thereby score better
    wide = [np.concatenate([s, rng.exponential(40.0, 30)]) for s in good]
    sw = np.array([samples_to_survival(s, grid) for s in wide])
    assert coverage(wide, truth, event, 0.8) >= coverage(good, truth, event, 0.8)
    assert score_arm(sw, wide, truth, event, null)["skill"] < \
        score_arm(sg, good, truth, event, null)["skill"]

    # empirical survival is monotone and starts at or below 1
    s = samples_to_survival(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5, 3.5]))
    assert np.all(np.diff(s) <= 0) and s[0] <= 1.0
    assert np.isnan(samples_to_survival(np.array([]), grid)).all()

    # a censored patient counts as covered when their censoring time is below the top of
    # the interval: they had not progressed, which is consistent with any larger value
    late = [np.array([10.0, 12.0, 14.0])]
    assert coverage(late, np.array([2.0]), np.array([0.0]), 0.8) == 1.0
    assert coverage(late, np.array([2.0]), np.array([1.0]), 0.8) == 0.0
    assert matrix_coverage(np.asarray(late), np.array([2.0]), np.array([0.0]), 0.8) == 1.0

    # finite samples drawn from the correct marginal curve score below the curve itself;
    # the penalty shrinks as the number of generated futures grows
    bench = finite_sample_null_benchmark(truth, event, [2, 20], reps=12, seed=1)
    assert bench["2"]["skill_median"] < bench["20"]["skill_median"] <= 0.05
    assert bench["2"]["cover_80_median"] < bench["20"]["cover_80_median"]

    # quadrants split at the medians and name all four
    q = quadrant(np.arange(100.0), np.arange(100.0)[::-1])
    assert len(set(q)) == 2 and all("support" in v for v in set(q))

    # A primary file missing one arm must fail instead of silently omitting its headline.
    validate_arm_set(set(CORE_ARMS))
    try:
        validate_arm_set({"R3_no_anchors", "R4_same_action"})
    except ValueError as exc:
        assert "R6_shuffled" in str(exc)
    else:
        raise AssertionError("an incomplete primary arm set must be refused")

    # The preregistered quadrant prediction is computed, not reduced to a count table.
    rng = np.random.default_rng(11)
    nq = 80
    qids = np.array([f"q{i}" for i in range(nq)])
    patient = np.array([f"p{i}" for i in range(nq)])
    qt = rng.exponential(7.0, nq) + 0.5
    qe = np.ones(nq)
    support = np.r_[np.ones(40), np.full(40, 10.0)]
    roughness = np.tile(np.r_[np.ones(20), np.full(20, 10.0)], 2)
    labels = quadrant(support, roughness)
    r3 = np.array([samples_to_survival(np.array([18.0]), B.GRID) for _ in range(nq)])
    r4 = r3.copy()
    target = labels == TARGET_QUADRANT
    r4[target] = np.array([
        samples_to_survival(np.array([value]), B.GRID) for value in qt[target]
    ])
    curves = {
        "R4_same_action": (qids, r4, qt, qe, patient),
        "R3_no_anchors": (qids, r3, qt, qe, patient),
        "R6_shuffled": (qids, r3.copy(), qt, qe, patient),
    }
    qframe = pd.DataFrame({"query_id": qids, "support": support,
                           "roughness": roughness})
    qscore = score_quadrants(curves, qframe, PRIMARY_COMPARISONS, min_shared=10)
    key = "R4_same_action - R3_no_anchors"
    assert qscore[key]["target_is_largest"] is True
    assert len(qscore[key]["quadrants"]) == 4
    # THE ADAPTATION REFERENCE. A model that reproduces its anchor distribution exactly
    # must score at the reference, not above or below it: the whole point of sampling the
    # anchor curve n_gen times is that both sides pay the same discretisation cost.
    rng = np.random.default_rng(7)
    n = 120
    curve = B.km_curve(rng.exponential(8.0, 400), np.ones(400), B.GRID)
    truth = sample_from_survival_curve(curve, B.GRID, n_patients=n, n_samples=1,
                                       rng=rng)[:, 0]
    ev = np.ones(n)
    null = B.integrated_brier(np.tile(B.km_curve(truth, ev, B.GRID), (n, 1)),
                              truth, ev, B.GRID, truth, ev)
    curves = [curve] * n
    ref = anchor_reference(curves, truth, ev, null, n_gen=20, reps=40, seed=7)
    assert ref, "the reference must be computable from valid curves"
    used = np.array(ref["queries_used"], int)
    drawn = sample_from_survival_curve(curve, B.GRID, n_patients=n, n_samples=20, rng=rng)
    mimic = np.array([samples_to_survival(row, B.GRID) for row in drawn])
    ibs = B.integrated_brier(np.clip(mimic[used], 1e-6, 1.0), truth[used], ev[used],
                             B.GRID, truth, ev)
    delta = (1 - ibs / null) - ref["skill_median"]
    assert ref["skill_lo"] < (1 - ibs / null) < ref["skill_hi"], (
        f"a generator that mimics its anchor curve must land inside the reference interval; "
        f"got skill {1 - ibs / null:.3f} against [{ref['skill_lo']:.3f}, "
        f"{ref['skill_hi']:.3f}]")
    assert abs(delta) < 0.05, f"delta_adapt should be near zero for a mimic, got {delta:+.3f}"

    print("self-test OK — a truthful generator outscores a wrong one, a generator that "
          "mimics\n   its anchor curve lands inside the finite-sampled reference interval, "
          "widening does\n   not help, censored coverage is one-sided, and empirical "
          "survival is monotone")
    return 0


def echo_diagnostic(gens: pd.DataFrame) -> pd.DataFrame:
    """Is the generator transporting the evidence, or reading it back?

    An aggregate evidence block hands the model a number. If the generated median is that
    number, the arm scores well and has added nothing -- a null result wearing the clothes
    of a success, and the failure mode that gets easier the more legible the evidence block
    becomes. So it is measured directly rather than argued about.

    Slope near 1 with a tight fit means the evidence is being copied. A slope well below 1,
    or a wide scatter that still improves the score, means the model is starting from the
    evidence and moving -- which is the whole claim.
    """
    rows = []
    for arm, g in gens.groupby("arm"):
        g = g[~g["refused"].astype(bool) & g["anchor_km_median"].notna()]
        if len(g) < 5:
            continue
        pred = np.array([float(np.median(s)) for s in g["pfs_samples"]])
        anch = g["anchor_km_median"].to_numpy(float)
        ok = np.isfinite(pred) & np.isfinite(anch)
        pred, anch = pred[ok], anch[ok]
        if len(pred) < 5 or np.std(anch) < 1e-9:
            continue
        slope = float(np.polyfit(anch, pred, 1)[0])
        r = float(np.corrcoef(anch, pred)[0, 1])
        rows.append({"arm": arm, "n": len(pred), "slope": slope, "pearson_r": r,
                     "median_abs_gap": float(np.median(np.abs(pred - anch))),
                     "within_half_month": float(np.mean(np.abs(pred - anch) < 0.5)),
                     "median_predicted": float(np.median(pred)),
                     "median_evidence": float(np.median(anch))})
    return pd.DataFrame(rows)


#: The two comparisons the locked pass-1 claim requires. The builder imports these names and
#: refuses to emit a primary design under a different spelling.
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("R4_same_action", "R6_shuffled", "DECISIVE: anchors vs wrong anchors"),
    ("R4_same_action", "R3_no_anchors", "does anchoring help at all"),
)
EXPLORATORY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("R4_same_action", "R4b_no_outcomes", "outcomes vs context alone"),
)
HEADLINE_COMPARISONS = PRIMARY_COMPARISONS + EXPLORATORY_COMPARISONS
CORE_ARMS: frozenset[str] = frozenset({"R3_no_anchors", "R4_same_action", "R6_shuffled"})


#: Every arm name the readout can consume. Anything outside this set is silently unscored.
KNOWN_ARMS: frozenset[str] = frozenset(
    {"R3_no_anchors", "R4_same_action", "R4b_no_outcomes", "R6_shuffled",
     "R4_aggregate_marginal", "R4_aggregate_conditional", "R6_shuffled_aggregate",
     "R5_paired", "R4_same_action_k8", "R6_shuffled_k8"}
)
#: Arms whose PROMPT actually contained the retrieved evidence. R3 is the unanchored control:
#: its rows still carry `anchor_km_curve` because the builder attaches retrieval metadata to
#: every row, but that curve was never rendered into R3's prompt. Scoring R3's "improvement
#: over the evidence it was given" therefore states something false about evidence it never
#: saw -- the primary contrast is unaffected, the mechanism sentence is not.
ARMS_WITH_SUPPLIED_EVIDENCE: frozenset[str] = frozenset(KNOWN_ARMS - {"R3_no_anchors"})


def validate_arm_set(present_arms: set[str], *, allow_nonprimary: bool = False) -> None:
    """Refuse misspelled arms and incomplete pass-1 files before computing any headline."""
    unknown = present_arms - set(KNOWN_ARMS)
    if unknown:
        raise ValueError(f"generation file contains unknown arms: {sorted(unknown)}")
    touched_core = present_arms & set(CORE_ARMS)
    missing_core = set(CORE_ARMS) - present_arms
    if touched_core and missing_core and not allow_nonprimary:
        raise ValueError(
            "primary GENIE readout is incomplete: found core arms "
            f"{sorted(touched_core)} but is missing {sorted(missing_core)}. "
            "Use --allow-nonprimary only for a declared conditional analysis."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=Path, required=False,
                    help="JSONL returned by a generation run")
    ap.add_argument("--manifest", type=Path, default=OUT / "manifest.json",
                    help="the INPUT/scope manifest written by the builder. Source of the "
                         "frozen deterministic bar and the cohort metadata.")
    ap.add_argument("--run-manifest", type=Path, default=None,
                    help="the RUN manifest written by the runner beside attempts.jsonl. "
                         "Source of query_ids/arms/repeats for run-matrix validation. "
                         "Defaults to manifest.json next to --generations. These are two "
                         "different files with two different jobs: the builder's manifest "
                         "knows the comparator bars and nothing about what was actually "
                         "requested, and the runner's manifest knows the requested matrix "
                         "and nothing about the bars.")
    ap.add_argument("--scope-dir", type=Path, default=None,
                    help="directory containing manifest/prompts_*.jsonl (default: manifest parent)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--mc-null-reps", type=int, default=100,
                    help="Monte-Carlo replicates for the finite-sample marginal-KM null")
    ap.add_argument("--allow-nonprimary", action="store_true",
                    help="allow a conditional/exploratory file that contains only part of "
                         "the locked three-arm pass-1 design")
    ap.add_argument("--prereg", type=Path,
                    default=ROOT / "manuscript" / "genie_anchored_pfs_prereg_v1.yaml",
                    help="checked under --definitive: the run manifest must BE the "
                         "registered design, not merely consistent with itself")
    ap.add_argument("--definitive", action="store_true",
                    help="refuse to score anything but the exact registered run matrix. "
                         "This scorer once analysed a breast-only 23-of-500 prefix without "
                         "complaint, and a well-formed readout from a systematic 4% subset "
                         "is the failure mode most likely to be believed.")
    ap.add_argument("--no-fan-matching", action="store_true",
                    help="score arms at their raw fan sizes. Only for diagnostics: turning a "
                         "distribution into M draws costs skill as a function of M, so unequal "
                         "fans make an arm contrast partly a comparison of M.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.generations is None:
        ap.error("--generations is required unless --self-test")

    man = json.loads(args.manifest.read_text())
    scope_dir = args.scope_dir or args.manifest.parent
    # THE BUILDER'S OWN PIPELINE, not a re-implementation of it. query_id is
    # `record_id@<positional index>`, so it only means anything relative to the exact row
    # sequence that produced it -- and this scorer used to build that sequence BEFORE the
    # regimen join while the builder built it AFTER. attach_regimens ends in
    # drop_duplicates() + reset_index(), which renumbers every row past the first drop.
    # On the definitive run that silently dropped 134 of 500 queries AND resolved 70 of the
    # 366 survivors to a different decision for the same patient.
    d = decision_table()

    gens = load_generations(args.generations)
    present_arms = set(gens["arm"].astype(str).unique())
    validate_arm_set(present_arms, allow_nonprimary=args.allow_nonprimary)
    # TWO MANIFESTS, TWO JOBS. Reading the matrix out of the builder's manifest fails
    # validation (it declares no requested matrix), and reading the bars out of the runner's
    # manifest prints None. One --manifest argument was doing both and could only ever
    # satisfy one of them.
    run_manifest_path = args.run_manifest or (args.generations.parent / "manifest.json")
    if run_manifest_path.exists():
        run_man = json.loads(run_manifest_path.read_text())
        if not (run_man.get("query_ids") and run_man.get("arms") and run_man.get("repeats")):
            raise SystemExit(
                f"[fatal] {run_manifest_path} is not a RUN manifest -- it declares no "
                "query_ids/arms/repeats. Pass --run-manifest pointing at the manifest.json "
                "the runner wrote beside attempts.jsonl.")
        print(f"run matrix from   {run_manifest_path}")
        print(f"comparator bars from {args.manifest}")
        if args.definitive:
            validate_against_prereg(run_man, args.prereg)
        validate_run_matrix(gens, run_man, definitive=args.definitive)
    elif args.definitive:
        raise SystemExit(f"[fatal] --definitive needs the run manifest; {run_manifest_path} "
                         "is missing. Pass --run-manifest explicitly if it lives elsewhere.")
    if not args.no_fan_matching:
        before = gens["n_samples"].to_numpy().copy()
        gens = match_fan_sizes(gens)
        dropped = int(before.sum() - gens["n_samples"].sum())
        if dropped:
            print(f"fan matching: {dropped} futures dropped so every arm of a query is "
                  f"scored at that query's smallest fan "
                  f"(median {gens['n_samples'].median():.0f}); deterministic random "
                  "subsample, not the first M")
    before_join = gens["query_id"].nunique()
    # cancer and line_number ride along for the population-registration readout; they are
    # stratification keys, never scored.
    join_cols = [c for c in ("query_id", "pfs_mos", "pfs_event", "record_id",
                             "cancer", "line_number") if c in d.columns]
    gens = gens.merge(d[join_cols], on="query_id", how="inner")
    after_join = gens["query_id"].nunique()
    if after_join != before_join:
        lost = before_join - after_join
        message = (f"{lost} of {before_join} generated queries did not join to a landmark. "
                   "A query_id that does not resolve means the generations and this table "
                   "were built from different row sequences, and the ones that DO resolve "
                   "cannot be trusted either -- the same drift that loses a row shifts the "
                   "rows after it onto their neighbours' outcomes.")
        if args.definitive:
            raise SystemExit(f"[fatal] {message}")
        print(f"[warn] {message}\n")
    # the support audit lives with the PROMPTS, not with the generations, so a run cannot
    # be stratified without joining it back. n_anchors is support; outcome_dispersion is
    # the roughness of the neighbourhood the anchors came from.
    prompt_files = sorted(scope_dir.glob("prompts_*.jsonl"))
    if prompt_files:
        audit = pd.concat([pd.read_json(f, lines=True) for f in prompt_files],
                          ignore_index=True)
        keep = ["query_id", "arm", "n_anchors", "outcome_dispersion", "anchor_spread",
                "anchor_km_median", "anchor_km_curve"]
        gens = gens.merge(audit[[c for c in keep if c in audit.columns]],
                          on=["query_id", "arm"], how="left")
        gens = gens.rename(columns={"n_anchors": "support",
                                    "outcome_dispersion": "roughness"})
    print(f"generations: {len(gens)} rows, {gens['arm'].nunique()} arms, "
          f"{gens['query_id'].nunique()} queries matched to landmarks")
    bad = int(gens["n_implausible"].sum())
    if bad:
        print(f"   [warn] {bad} samples outside {PLAUSIBLE_MONTHS} months — parse failures, "
              "reported not clipped")

    curves, scores, per_arm = {}, {}, {}
    for arm, g in gens.groupby("arm"):
        ok = g[~g["refused"].astype(bool) & (g["n_samples"] > 1)]
        if ok.empty:
            print(f"   [skip] {arm}: no scorable rows")
            continue
        t = ok["pfs_mos"].to_numpy(float)
        e = ok["pfs_event"].to_numpy(float)
        samples = list(ok["pfs_samples"])
        surv = np.array([samples_to_survival(s, B.GRID) for s in samples])
        null = B.integrated_brier(np.tile(B.km_curve(t, e, B.GRID), (len(t), 1)), t, e,
                                  B.GRID, t, e)
        sc = score_arm(surv, samples, t, e, null)
        sc["abstention"] = float(g["refused"].astype(bool).mean())
        sc["median_samples"] = float(ok["n_samples"].median())
        # THE ADAPTATION TEST. Does the model improve on the evidence it was handed, once
        # both are scored as finite samples? Reported with the reference's own interval.
        if "anchor_km_curve" in ok.columns and arm in ARMS_WITH_SUPPLIED_EVIDENCE:
            n_gen = int(round(float(ok["n_samples"].median())))
            ref = anchor_reference(
                [np.asarray(c, float) if isinstance(c, (list, np.ndarray)) else None
                 for c in ok["anchor_km_curve"]], t, e, null, n_gen=n_gen)
            if ref:
                sub = np.array(ref.pop("queries_used"), int)
                own = B.integrated_brier(np.clip(surv[sub], 1e-6, 1.0), t[sub], e[sub],
                                         B.GRID, t, e)
                sc["anchor_reference"] = ref
                sc["skill_on_reference_queries"] = float(1 - own / null)
                contrast = adaptation_contrast(
                    surv, list(ok["anchor_km_curve"]), t, e,
                    ok["record_id"].to_numpy(), n_gen=n_gen)
                if contrast:
                    sc["adaptation"] = contrast
                    sc["delta_adapt"] = contrast["delta_adapt"]
        scores[arm] = sc
        curves[arm] = (ok["query_id"].to_numpy(), surv, t, e,
                       ok["record_id"].to_numpy())
        per_arm[arm] = ok

    print(f"\n{'arm':22s} {'n':>6} {'skill':>8} {'C-index':>9} {'PIT KS':>8} "
          f"{'cov50':>7} {'cov80':>7} {'cov90':>7} {'abstain':>8}")
    print("-" * 92)
    for arm, s in sorted(scores.items(), key=lambda kv: -kv[1]["skill"]):
        print(f"{arm:22s} {s['n']:6d} {s['skill']:8.3f} {s['c_index']:9.3f} "
              f"{s['pit_ks']:8.3f} {s['cover_50']:7.0%} {s['cover_80']:7.0%} "
              f"{s['cover_90']:7.0%} {s['abstention']:8.0%}")
    adapt = {a: s for a, s in scores.items() if "adaptation" in s}
    if adapt:
        print("\nadaptation: does the model improve on the anchor evidence it was given,")
        print("with both scored as the same number of finite samples?")
        print("R3_no_anchors is EXCLUDED: it was given no evidence, so it has nothing to "
              "improve on.")
        print(f"{'arm':26s} {'n':>5} {'model':>8} {'anchor':>8} "
              f"{'delta [95% clustered]':>28} {'':>7}")
        print("-" * 88)
        for arm, s in sorted(adapt.items(), key=lambda kv: -kv[1]["adaptation"]["delta_adapt"]):
            a, r = s["adaptation"], s["anchor_reference"]
            print(f"{arm:26s} {a['n_queries']:5d} {s['skill_on_reference_queries']:8.3f} "
                  f"{r['skill_median']:8.3f} "
                  f"{a['delta_adapt']:+8.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}]  "
                  f"{a['verdict']:>7}")
        print("   THREE states, not two. 'below' means the model scored WORSE than the")
        print("   anchor-only Kaplan-Meier estimate; an earlier version printed that as")
        print("   'within' because it only tested whether the model beat the upper bound.")
        print("   The interval is a PATIENT-CLUSTERED bootstrap of the paired difference,")
        print("   not the Monte-Carlo spread of the reference. Note the anchor curve is the")
        print("   retrieved cohort's KM, which is not literally what the model was shown --")
        print("   it received summaries and ten detailed rows drawn from that cohort.")

    if "anchor_km_median" in gens.columns:
        echo = echo_diagnostic(gens)
        if not echo.empty:
            print("\necho check: is the generated median just the evidence median read "
                  "back?")
            print(f"{'arm':26s} {'n':>5} {'slope':>7} {'r':>6} {'|gap| med':>10} "
                  f"{'within 0.5mo':>13} {'pred/evid median':>18}")
            print("-" * 92)
            for _, r in echo.sort_values("slope", ascending=False).iterrows():
                print(f"{r['arm']:26s} {int(r['n']):5d} {r['slope']:7.2f} "
                      f"{r['pearson_r']:6.2f} {r['median_abs_gap']:10.2f} "
                      f"{r['within_half_month']:13.0%} "
                      f"{r['median_predicted']:8.1f} /{r['median_evidence']:8.1f}")
            print("   slope ~1 with a tight fit = the evidence copied back, not transported")

    n_gen_values = sorted({
        int(round(s["median_samples"])) for s in scores.values()
        if np.isfinite(s.get("median_samples", np.nan)) and s.get("median_samples", 0) >= 2
    })
    mc_null = finite_sample_null_benchmark(
        d["pfs_mos"].to_numpy(float), d["pfs_event"].to_numpy(float),
        n_gen_values, reps=args.mc_null_reps,
    ) if n_gen_values else {}
    if mc_null:
        print("\nfinite-sample null: marginal KM sampled into n_gen futures, then scored "
              f"like the LLM samples ({len(d):,} landmarks, {args.mc_null_reps} reps)")
        print(f"{'n_gen':>6} {'skill median [95%]':>26} {'cov80 median [95%]':>26}")
        for row in mc_null.values():
            print(
                f"{row['n_gen']:6d} "
                f"{row['skill_median']:+.3f} "
                f"[{row['skill_lo']:+.3f}, {row['skill_hi']:+.3f}] "
                f"{row['cover_80_median']:.0%} "
                f"[{row['cover_80_lo']:.0%}, {row['cover_80_hi']:.0%}]"
            )
    bar = man.get("frozen_bar") or man.get("bar", {})
    print(f"\nfrozen statistical bar, FULL COHORT ({man.get('n_landmarks')} landmarks): "
          f"skill {bar.get('skill')}, C-index {bar.get('c_index')}")
    for name, sc in man.get("deterministic_scores", {}).items():
        print(f"   {name:22s} skill {sc['skill']:.3f}")
    print("   NOT COMPARABLE TO THE ARMS ABOVE: different cohort, different null "
          "denominator,\n   and a smooth curve rather than a finite sample.")

    # The only comparison an LLM arm can legitimately be held to: same landmarks, same
    # number of draws.
    n_gen_here = int(round(float(np.median([s["median_samples"] for s in scores.values()]))))
    exact = comparators_on_scored_landmarks(set(gens["query_id"]), n_gen=n_gen_here)
    if exact:
        print(f"\nSAME LANDMARKS, SAME DRAW COUNT — the comparison that is actually fair "
              f"({exact['n_landmarks']} landmarks, {exact['n_gen']} draws)")
        rows = [(f"{k} (deterministic)", v["skill_at_n_gen"], v["skill_smooth"])
                for k, v in exact["arms"].items()]
        rows += [(f"{a} (LLM)", sc["skill"], float("nan")) for a, sc in scores.items()]
        print(f"   {'arm':30s} {'skill @ draws':>14} {'skill smooth':>14}")
        print("   " + "-" * 60)
        for name, at_n, smooth in sorted(rows, key=lambda r: -r[1]):
            smooth_txt = "     —" if not np.isfinite(smooth) else f"{smooth:14.3f}"
            print(f"   {name:30s} {at_n:14.3f} {smooth_txt}")
        payload_exact = exact

    # --- population registration: is between-population structure preserved? ---------------
    reg = population_registration(gens, arms=sorted(scores))
    if reg:
        print(f"\nPOPULATION REGISTRATION — {reg['statistic']}, generated on real")
        print("   beta = 1 separation preserved | 0<beta<1 attenuated | 0 flattened | <0 inverted")
        print(f"   {'arm':24s}{'strata':>7}{'alpha (shift)':>26}{'beta (structure)':>26}")
        for arm, v in sorted(reg["arms"].items(), key=lambda kv: -kv[1]["beta"]):
            print(f"   {arm:24s}{v['n_strata']:>7}"
                  f"{v['alpha']:>13.2f} [{v['alpha_lo']:+.2f},{v['alpha_hi']:+.2f}]"
                  f"{v['beta']:>13.2f} [{v['beta_lo']:+.2f},{v['beta_hi']:+.2f}]")

    # --- PIT uniformity contrasts, integrated over the jitter -------------------------------
    pit_contrasts = pit_arm_contrasts(gens, sorted(scores))
    if pit_contrasts:
        print("\nCALIBRATION (PIT uniformity) — lower KS is better, jitter integrated out")
        for arm, v in sorted(pit_contrasts["per_arm_ks"].items(), key=lambda kv: kv[1]["median"]):
            print(f"   {arm:24s} KS {v['median']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        print(f"   contrasts, patient-clustered with fresh randomisation per replicate "
              f"({pit_contrasts['boots']} replicates):")
        for name, v in pit_contrasts["contrasts"].items():
            tag = "  DISTINGUISHABLE" if v["excludes_zero"] else "  not distinguishable"
            print(f"   {name:38s} {v['delta_ks']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{tag}")
        print("   A single draw of the jitter is not the result: across seeds the R4-R3")
        print("   contrast ranged -0.046 to -0.085, and one draw made a real effect vanish.")

    # --- A3 provider sensitivity, always reported ------------------------------------------
    prov = provider_sensitivity(run_manifest_path.parent / "attempts.jsonl",
                                curves, HEADLINE_COMPARISONS)
    if prov:
        print(f"\nPROVIDER SENSITIVITY (A3) — backends: {', '.join(prov['providers'])}")
        print(f"   MASD {prov['masd']:.4f}   query-clustered permutation p = "
              f"{prov['masd_permutation_p']:.3f} ({prov['n_permutations']} permutations)")
        band = ("green, unstandardised primary" if prov["masd"] <= 0.05 else
                "AMBER, standardised becomes primary" if prov["masd"] <= 0.15 else
                "RED, pinned rerun required")
        print(f"   registered band: {band}")
        for name, v in prov["standardised_contrasts"].items():
            std = "n/a" if v["provider_standardised"] is None else \
                f"{v['provider_standardised']:+.4f}"
            print(f"   {name:34s} raw {v['raw']:+.4f}   standardised {std}")

    # --- the decisive control -------------------------------------------------------------
    comparisons = {}
    for a, b, why in HEADLINE_COMPARISONS:
        result = compare_arm_curves(curves, a, b)
        if result is None:
            continue
        if not result["scored"]:
            if (a, b, why) in PRIMARY_COMPARISONS:
                print(f"\n{why}\n   unscored: only {result['n_shared']} shared queries "
                      f"(need {result['minimum_required']})")
            continue
        comparisons[f"{a} - {b}"] = {**result, "why": why}
        verdict = "distinguishable" if result["excludes_zero"] else "NOT distinguishable"
        print(f"\n{why}\n   {a} - {b}: {result['delta_skill']:+.4f} "
              f"[{result['lo']:+.4f}, {result['hi']:+.4f}] on "
              f"{result['n_shared']} shared — {verdict}")

    if "R4_same_action - R6_shuffled" in comparisons:
        c = comparisons["R4_same_action - R6_shuffled"]
        if not c["excludes_zero"]:
            print("\n   Shuffling the anchors did not measurably hurt. The model is "
                  "reacting to prompt volume rather than to the evidence, and every other "
                  "number here is uninterpretable.")

    # --- where the effect lives ------------------------------------------------------------
    quadrant_comparisons = {}
    if "R4_same_action" in per_arm and "support" in per_arm["R4_same_action"].columns:
        g = per_arm["R4_same_action"]
        g = g.assign(quadrant=quadrant(g["support"].to_numpy(float),
                                       g["roughness"].to_numpy(float)))
        print("\nWHERE THE EFFECT LIVES — a pooled win that misses the target quadrant is "
              "not this framework's effect")
        print(g.groupby("quadrant").size().to_string())
        quadrant_comparisons = score_quadrants(curves, per_arm["R4_same_action"],
                                               PRIMARY_COMPARISONS)
        for comparison, result in quadrant_comparisons.items():
            print(f"\n   {comparison}")
            for name, cell in result["quadrants"].items():
                if not cell.get("scored"):
                    print(f"      {name}: unscored, n={cell['n_shared']}")
                    continue
                print(f"      {name}: {cell['delta_skill']:+.4f} "
                      f"[{cell['lo']:+.4f}, {cell['hi']:+.4f}], n={cell['n_shared']}")
            print("      largest point estimate: "
                  f"{result['largest_point_estimate_quadrant'] or 'none scorable'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scores.json").write_text(json.dumps(
        {"arms": scores, "comparisons": comparisons,
         "quadrant_comparisons": quadrant_comparisons, "frozen_bar": bar,
         "deterministic_full_cohort": man.get("deterministic_scores", {}),
         "deterministic_on_scored_landmarks": locals().get("payload_exact"),
         "provider_sensitivity": prov,
         "pit_contrasts": pit_contrasts,
         "population_registration": reg,
         "finite_sample_null": mc_null,
         "n_implausible_samples": bad}, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}/scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
