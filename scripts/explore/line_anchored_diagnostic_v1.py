#!/usr/bin/env python3
"""Line-anchored landmarks: eligible counts, strata, and whether history depth matters.

A DIAGNOSTIC, not a launch gate and not a result. Its job is to answer three questions
before any generation: how many patients actually remain once one landmark is assigned per
patient, how the line-depth strata are populated, and whether the history signal varies
with depth on real data.

Corrections this implements:

  * ONE LANDMARK PER PATIENT. A patient reaching line 3 also has a line-1 and a line-2
    start, so counting line starts overstates availability. Each patient is assigned a
    single landmark by a seeded draw among their own line starts, and the eligible counts
    are computed after that assignment.
  * LINE 3+ IS NOT ONE THING. The exact line number is kept and line 3 is reported
    separately from line 4+.
  * NO POOLED ESTIMATE. Strata are reported separately. A balanced 50/50/50 design
    estimates a balanced population, not the natural GENIE one, so pooling would answer a
    question nobody asked.
  * SUPPORT IS THE CONFOUND. Line depth and reference-class size move together, so both
    are reported and neither is credited with the other's effect.

Landmark = the start of the assigned treatment line. The future window runs from there, so
regimen-level PFS is conventional again and the whole-course outcomes remain well defined.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only, gitignored.

Run:  uv run python scripts/explore/line_anchored_diagnostic_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genie_operator_registry_v2 as GV2  # noqa: E402
import trajectory_feature_poc_v1 as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE", "CRC": "CRC_GENIE"}
OUT = ROOT / "outputs" / "exploratory_line_anchored_diagnostic"
H, ALPHA, SEED, FOLDS, N_BOOT = 12.0, 100.0, 20260824, 5, 399
MIN_STRATUM = 40


def depth_label(number: float) -> str:
    if not np.isfinite(number):
        return "unknown"
    n = int(number)
    return {1: "line 1", 2: "line 2", 3: "line 3"}.get(n, "line 4+") if n >= 1 else "unknown"


def assign_landmarks(patients: list, seed: int) -> list[dict]:
    """One landmark per patient, drawn from that patient's own line starts."""
    rng = np.random.default_rng(seed)
    out = []
    for p in sorted(patients, key=lambda q: q.pid):
        starts = [ln for ln in p.lines if np.isfinite(ln.get("number", np.nan))
                  and ln["start"] > 0.0]
        if not starts:
            continue
        chosen = starts[int(rng.integers(len(starts)))]
        out.append({"patient": p, "tau": float(chosen["start"]),
                    "line": float(chosen["number"]),
                    "depth": depth_label(chosen["number"]),
                    "bucket": chosen["bucket"]})
    return out


def eligible(record: dict) -> bool:
    p, tau = record["patient"], record["tau"]
    if p.death <= tau:
        return False
    if not (p.last_contact >= tau + H or p.death <= tau + H):
        return False
    if not any(w <= tau for w, _ in p.status):
        return False
    return GV2.observed_span(p, tau, tau + H) > 0 or p.death <= tau + H


def support_count(records: list[dict]) -> dict[str, int]:
    """Crude reference-class size: same line depth, same starting treatment class, and
    the same current disease state. Crude on purpose -- it is the confound indicator, not
    an estimand."""
    keys = {}
    for r in records:
        state = next((s for w, s in reversed(r["patient"].status) if w <= r["tau"]), "none")
        keys[r["patient"].pid] = (r["depth"], r["bucket"], state)
    counts = pd.Series(list(keys.values())).value_counts().to_dict()
    return {pid: int(counts[k]) for pid, k in keys.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for cohort, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        if not genie.exists():
            continue
        patients = GV2.load(genie)
        records = assign_landmarks(patients, SEED)
        keep = [r for r in records if eligible(r)]
        support = support_count(keep)
        counts = pd.Series([r["depth"] for r in keep]).value_counts().to_dict()
        assigned = pd.Series([r["depth"] for r in records]).value_counts().to_dict()
        print(f"\n=== {cohort}: {len(patients)} patients, {len(records)} with a line "
              f"start, {len(keep)} eligible after one-landmark assignment")
        for depth in ("line 1", "line 2", "line 3", "line 4+"):
            med = np.median([support[r["patient"].pid] for r in keep
                             if r["depth"] == depth]) if counts.get(depth) else float("nan")
            taus = [r["tau"] for r in keep if r["depth"] == depth]
            print(f"    {depth:8s} assigned {assigned.get(depth, 0):4d} -> eligible "
                  f"{counts.get(depth, 0):4d}   median months from diagnosis "
                  f"{np.median(taus) if taus else float('nan'):5.1f}   median "
                  f"reference-class size {med:.0f}")

        rows_x, rows_y, meta = [], [], []
        for r in keep:
            rows_x.append(GV2.prefix_features(r["patient"], r["tau"]))
            rows_y.append(GV2.future_outcomes(r["patient"], r["tau"], H))
            meta.append(r)
        if len(rows_x) < 3 * MIN_STRATUM:
            print("    too few eligible patients for the depth ladder; counts only")
            report[cohort] = {"assigned": assigned, "eligible": counts}
            continue
        X, Y = pd.DataFrame(rows_x), pd.DataFrame(rows_y)[GV2.OUTCOMES]
        groups = np.array([r["patient"].pid for r in meta])
        depths = np.array([r["depth"] for r in meta])
        rungs = list(GV2.LADDER)
        cum, cumulative = {}, []
        for rung, feats in GV2.LADDER.items():
            cumulative = cumulative + [c for c in feats if c in X.columns]
            cum[rung] = list(cumulative)
        base_cols, full_cols = cum[rungs[2]], cum[rungs[-1]]

        blocks = [[GV2.OUTCOMES.index(o) for o in names]
                  for names in GV2.OUTCOME_BLOCKS.values()]
        stratum_gain = {}
        for depth in ("line 1", "line 2", "line 3", "line 4+"):
            sel = depths == depth
            if sel.sum() < MIN_STRATUM:
                continue
            ymat = Y[sel].to_numpy(float)
            ok = np.isfinite(ymat).all(axis=1)
            sd = np.where(ymat[ok].std(axis=0) > 1e-9, ymat[ok].std(axis=0), 1.0)
            ystd = (ymat[ok] - ymat[ok].mean(axis=0)) / sd
            g = groups[sel][ok]
            folds = M.group_folds(g, min(FOLDS, len(set(g))), SEED)
            base = np.nan_to_num(X.loc[sel, base_cols].to_numpy(float)[ok])
            full = np.nan_to_num(X.loc[sel, full_cols].to_numpy(float)[ok])
            alphas = [ALPHA] * ystd.shape[1]
            s_base = M.block_joint_score(base, ystd, blocks, folds, g, SEED, alphas)
            s_full = M.block_joint_score(full, ystd, blocks, folds, g, SEED, alphas)
            preds_b = {k: M.oof_predict(base, ystd[:, k], folds, g, SEED, ALPHA)
                       for k in range(ystd.shape[1])}
            preds_f = {k: M.oof_predict(full, ystd[:, k], folds, g, SEED, ALPHA)
                       for k in range(ystd.shape[1])}
            rng = np.random.default_rng(SEED)
            draws = []
            for _ in range(N_BOOT):
                rows = rng.integers(0, len(g), len(g))
                tot = sum(float((ystd[rows, k] ** 2).sum()) for k in preds_b)
                sb = sum(float(((ystd[rows, k] - preds_b[k][rows]) ** 2).sum())
                         for k in preds_b)
                sf = sum(float(((ystd[rows, k] - preds_f[k][rows]) ** 2).sum())
                         for k in preds_f)
                draws.append((1 - sf / tot) - (1 - sb / tot) if tot > 0 else np.nan)
            lo, hi = np.nanquantile(draws, [0.025, 0.975])
            med_support = float(np.median([support[p] for p in g]))
            stratum_gain[depth] = {"n": int(ok.sum()), "base": s_base, "full": s_full,
                                   "gain": s_full - s_base, "ci": [float(lo), float(hi)],
                                   "median_support": med_support}
            print(f"    {depth:8s} n={int(ok.sum()):4d}  base {s_base:6.3f}  full "
                  f"{s_full:6.3f}  history gain {s_full - s_base:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  median support {med_support:.0f}")
        report[cohort] = {"assigned": assigned, "eligible": counts,
                          "by_depth": stratum_gain}

    depths = ["line 1", "line 2", "line 3", "line 4+"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6))
    ax = axes[0]
    width = 0.8 / max(1, len(report))
    for i, (cohort, r) in enumerate(report.items()):
        vals = [r["eligible"].get(d, 0) for d in depths]
        ax.bar(np.arange(len(depths)) + (i - len(report) / 2 + 0.5) * width, vals, width,
               label=f"{cohort} (n={sum(vals)})")
    ax.axhline(50, color="#B00020", ls="--", lw=1.4)
    ax.text(len(depths) - 0.5, 52, "50 needed per stratum", ha="right", fontsize=8.5,
            color="#B00020")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_ylabel("eligible patients (one landmark each)")
    ax.set_title("A. Eligible counts after one-landmark assignment", fontsize=11,
                 loc="left")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[1]
    for i, (cohort, r) in enumerate(report.items()):
        by = r.get("by_depth", {})
        xs = [j for j, d in enumerate(depths) if d in by]
        if not xs:
            continue
        vals = [by[depths[j]]["gain"] for j in xs]
        lo = [vals[k] - by[depths[j]]["ci"][0] for k, j in enumerate(xs)]
        hi = [by[depths[j]]["ci"][1] - vals[k] for k, j in enumerate(xs)]
        ax.errorbar(np.array(xs) + (i - len(report) / 2 + 0.5) * 0.12, vals,
                    yerr=[lo, hi], fmt="o-", capsize=3, lw=1.6, ms=6, label=cohort)
    ax.axhline(0, color="#333333", lw=1.1)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_ylabel("history gain over context + observation + current state")
    ax.set_title("B. Does history matter more at greater depth?\n"
                 "stratified only; support is confounded with depth", fontsize=11,
                 loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Line-anchored landmarks: feasibility and the history-depth axis",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "line_anchored_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (OUT / "readout.json").write_text(json.dumps(
        {"purpose": "diagnostic for the line-anchored design; not a launch gate",
         "horizon_months": H, "alpha": ALPHA, "seed": SEED,
         "min_stratum": MIN_STRATUM, "cohorts": report}, indent=2) + "\n")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
