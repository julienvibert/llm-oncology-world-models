#!/usr/bin/env python3
"""Can a method give ONE patient the right PFS distribution? The bar, before any generation.

A generative model does not produce a number, it produces samples. So the question is not
whether a predicted PFS equals the observed one -- it never will, and demanding that would
be incoherent -- but whether the observed PFS looks like a draw from the predicted
distribution, and whether that distribution is tight enough to be worth having.

Those are two different requirements and both are needed:

    calibration   is the real patient inside the predicted distribution, at the right rate?
                  Measured by the probability-integral transform: S_i(T_i) should be uniform
                  across patients if the predicted survival functions are right.
    sharpness     a method that hands every patient the whole cohort's PFS curve is PERFECTLY
                  calibrated and useless. Sharpness is what separates a forecast from a
                  disclaimer.

The integrated Brier score with inverse-probability-of-censoring weights scores both at once
and is proper, so it cannot be gamed by widening. Skill is versus the marginal Kaplan-Meier
-- the "same curve for everyone" forecast -- which is exactly the useless-but-calibrated
baseline, so a positive skill means information about THIS patient.

THE LADDER
    marginal KM        one curve for everyone. Calibrated by construction, zero sharpness.
    arm KM             one curve per treatment. What a textbook would quote.
    archetype KM       the patient's cluster, on their arm -- "patients like me" as a curve.
    supervised k-NN KM the k nearest on the same arm, metric learned from outcomes.
    Cox                a fitted individual curve.
    discrete-time RF   per-interval hazard from a forest, multiplied into a survival curve.
                       Handles censoring natively and is the strongest conventional learner.

SCOPE, deliberately narrow. One landmark: the start of one treatment line. One endpoint:
imaging-confirmed PFS from that start. Not OS, not events after later lines -- both compound
the choices made after the landmark, which no model at the landmark can be asked to know.

GOVERNANCE: GENIE-BPC is under a data use agreement. Aggregate scores only, into a
gitignored directory. No patient-level rows are written anywhere.

Run:  uv run python scripts/explore/measure_pfs_distribution_bar.py --self-test
      uv run python scripts/explore/measure_pfs_distribution_bar.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_anchor_operating_window as W  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_pfs_distribution_bar"
SEED = 20260810
GRID = np.arange(1.0, 24.1, 1.0)      # months at which the survival curve is evaluated
K_NN = 80
K_CLUSTERS = 8


def km_curve(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival evaluated on the grid. Plain, so it can be called per cell."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    n = len(t)
    surv, s, i = [], 1.0, 0
    at_risk = n
    uniq = np.unique(t)
    curve_t, curve_s = [], []
    for u in uniq:
        d = int(((t == u) & (e == 1)).sum())
        c = int((t == u).sum())
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
        at_risk -= c
        curve_t.append(u)
        curve_s.append(s)
    curve_t, curve_s = np.asarray(curve_t), np.asarray(curve_s)
    idx = np.searchsorted(curve_t, grid, side="right") - 1
    out = np.where(idx >= 0, curve_s[np.clip(idx, 0, len(curve_s) - 1)], 1.0)
    del surv, i
    return out


def censoring_curve(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier of the CENSORING distribution, for the IPCW weights."""
    return km_curve(time, 1 - event, grid)


def integrated_brier(surv: np.ndarray, time: np.ndarray, event: np.ndarray,
                     grid: np.ndarray, g_train_time: np.ndarray,
                     g_train_event: np.ndarray) -> float:
    """IPCW Brier score integrated over the grid. Proper, so widening cannot help.

    Censoring weights come from the TRAINING fold, so a held-out patient's own censoring
    time never enters their own weight.
    """
    g_grid = np.clip(censoring_curve(g_train_time, g_train_event, grid), 1e-3, None)
    g_at_t = np.clip(censoring_curve(g_train_time, g_train_event, np.clip(time, 1e-6, None)),
                     1e-3, None)
    bs = np.zeros(len(grid))
    for j, tau in enumerate(grid):
        died = (time <= tau) & (event == 1)
        alive = time > tau
        term = np.zeros(len(time))
        term[died] = (surv[died, j] ** 2) / g_at_t[died]
        term[alive] = ((1.0 - surv[alive, j]) ** 2) / g_grid[j]
        bs[j] = term.mean()
    return float(np.trapezoid(bs, grid) / (grid[-1] - grid[0]))


def pit_values(surv: np.ndarray, time: np.ndarray, event: np.ndarray,
               grid: np.ndarray) -> np.ndarray:
    """S_i(T_i) for patients with an observed event — uniform if the forecast is calibrated.

    Censored patients are excluded rather than partially credited. With 80%+ events here the
    loss is small, and the alternative needs an assumption the comparison does not need.
    """
    j = np.clip(np.searchsorted(grid, time, side="right") - 1, 0, len(grid) - 1)
    return surv[np.arange(len(time)), j][event == 1]


def discrete_time_rf(xtr: np.ndarray, ttr: np.ndarray, etr: np.ndarray, xte: np.ndarray,
                     grid: np.ndarray) -> np.ndarray:
    """Per-interval hazard from a forest, multiplied into a survival curve.

    Person-period expansion: each patient contributes one row per interval they are still at
    risk in, labelled with whether the event happened in it. Censoring is handled by the
    patient simply stopping contributing rows, which is what makes this usable without an
    extra assumption.
    """
    from sklearn.ensemble import RandomForestClassifier
    rows, labels = [], []
    for i in range(len(xtr)):
        last = int(np.searchsorted(grid, ttr[i], side="right"))
        for j in range(min(last + 1, len(grid))):
            rows.append(np.concatenate([xtr[i], [grid[j]]]))
            labels.append(1.0 if (etr[i] == 1 and j == min(last, len(grid) - 1)) else 0.0)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED,
                                n_jobs=-1).fit(np.asarray(rows), np.asarray(labels))
    surv = np.ones((len(xte), len(grid)))
    running = np.ones(len(xte))
    for j, tau in enumerate(grid):
        feats = np.hstack([xte, np.full((len(xte), 1), tau)])
        running = running * (1.0 - rf.predict_proba(feats)[:, 1])
        surv[:, j] = running
    return surv


def run_ladder(d: pd.DataFrame, x: np.ndarray, xm: np.ndarray) -> tuple[pd.DataFrame, dict]:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    t = d["pfs_mos"].to_numpy(float)
    e = d["pfs_event"].to_numpy(float)
    act = d["action"].to_numpy()
    pid = d["record_id"].to_numpy()
    names = ["marginal KM", "arm KM", f"archetype KM, k={K_CLUSTERS}",
             f"supervised k-NN KM, k={K_NN}", "Cox", "discrete-time RF"]
    surv = {n: np.ones((len(d), len(GRID))) for n in names}

    for tr, te in GroupKFold(n_splits=5).split(x, e, groups=pid):
        # metric learned on the training fold only, targeting early progression
        early = (t[tr] <= np.median(t[tr])).astype(float)
        imp = RandomForestClassifier(n_estimators=300, min_samples_leaf=8, random_state=SEED,
                                     n_jobs=-1).fit(x[tr], early).feature_importances_
        w = np.sqrt(np.maximum(imp / max(imp.max(), 1e-12), 0.02))
        xtr_w, xte_w = x[tr] * w, x[te] * w

        marg = km_curve(t[tr], e[tr], GRID)
        surv["marginal KM"][te] = marg
        for a in np.unique(act[te]):
            m = act[tr] == a
            c = km_curve(t[tr][m], e[tr][m], GRID) if m.sum() >= 20 else marg
            surv["arm KM"][te[act[te] == a]] = c

        km = KMeans(K_CLUSTERS, n_init=10, random_state=SEED).fit(xtr_w)
        lab_tr, lab_te = km.labels_, km.predict(xte_w)
        for j, ti in enumerate(te):
            cell = (lab_tr == lab_te[j]) & (act[tr] == act[ti])
            src = cell if cell.sum() >= 20 else (lab_tr == lab_te[j])
            surv[f"archetype KM, k={K_CLUSTERS}"][ti] = (
                km_curve(t[tr][src], e[tr][src], GRID) if src.sum() >= 20 else marg)
            same = act[tr] == act[ti]
            pool = same if same.sum() >= K_NN else np.ones(len(tr), bool)
            take = np.argsort(np.linalg.norm(xtr_w[pool] - xte_w[j], axis=1))[:K_NN]
            surv[f"supervised k-NN KM, k={K_NN}"][ti] = km_curve(
                t[tr][pool][take], e[tr][pool][take], GRID)

        cols = [f"f{i}" for i in range(xm.shape[1])]
        fr = pd.DataFrame(xm[tr], columns=cols)
        fr["T"], fr["E"] = np.clip(t[tr], 1e-3, None), e[tr]
        cph = CoxPHFitter(penalizer=0.5).fit(fr, "T", "E")
        sf = cph.predict_survival_function(pd.DataFrame(xm[te], columns=cols), times=GRID)
        surv["Cox"][te] = sf.to_numpy().T
        surv["discrete-time RF"][te] = discrete_time_rf(xtr_w, t[tr], e[tr], xte_w, GRID)

    rows = []
    ibs_marg = integrated_brier(surv["marginal KM"], t, e, GRID, t, e)
    for n in names:
        s = np.clip(surv[n], 1e-6, 1.0)
        ibs = integrated_brier(s, t, e, GRID, t, e)
        pit = pit_values(s, t, e, GRID)
        # KS distance from uniform: 0 is perfectly calibrated
        ks = float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit), endpoint=False))))
        risk = 1.0 - s[:, min(5, len(GRID) - 1)]
        rows.append({"model": n, "ibs": ibs, "ibs_skill": 1 - ibs / ibs_marg,
                     "c_index": float(concordance_index(t, -risk, e)),
                     "pit_ks": ks, "median_6mo_surv": float(np.median(s[:, 5]))})
    return pd.DataFrame(rows), {"ibs_marginal": ibs_marg}


def self_test() -> int:
    # KM on complete data with no censoring equals the empirical survival function
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.ones(4)
    s = km_curve(t, e, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(s, [0.75, 0.5, 0.25, 0.0]), s

    # a sharper, correct forecast must score better than the marginal one
    rng = np.random.default_rng(0)
    n = 600
    fast = rng.random(n) < 0.5
    tt = np.where(fast, rng.exponential(2.0, n), rng.exponential(12.0, n))
    ee = np.ones(n)
    grid = GRID
    marg = np.tile(km_curve(tt, ee, grid), (n, 1))
    grp = np.where(fast[:, None], km_curve(tt[fast], ee[fast], grid),
                   km_curve(tt[~fast], ee[~fast], grid))
    ibs_m = integrated_brier(marg, tt, ee, grid, tt, ee)
    ibs_g = integrated_brier(grp, tt, ee, grid, tt, ee)
    assert ibs_g < ibs_m, (ibs_g, ibs_m)

    # widening a forecast cannot improve a proper score
    flat = np.tile(np.linspace(1.0, 0.5, len(grid)), (n, 1))
    assert integrated_brier(flat, tt, ee, grid, tt, ee) > ibs_g

    # PIT of a correct forecast is roughly uniform
    pit = pit_values(grp, tt, ee, grid)
    assert 0.35 < pit.mean() < 0.65, pit.mean()
    print("self-test OK — KM matches the empirical curve, a sharper correct forecast wins, "
          "widening does not, and PIT of a correct forecast is near-uniform")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cancers", nargs="*", default=None,
                    help="restrict to these cohorts (default: all three pooled)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    d = W.load_decisions()
    if args.cancers:
        d = d[d["cancer"].isin(args.cancers)]
    d = d[d["pfs_mos"].gt(0) & d["pfs_event"].notna()].reset_index(drop=True)
    fcols = [*W.CONTEXT, *W.STATE_RICH, *W.HISTORY_RICH]
    x = W.feature_matrix(d, fcols)
    xm = W.feature_matrix(d, fcols, drop_first=True)
    t, e = d["pfs_mos"].to_numpy(float), d["pfs_event"].to_numpy(float)
    print(f"GENIE-BPC {'+'.join(sorted(d['cancer'].unique()))}: {len(d)} line "
          f"starts, {d['record_id'].nunique()} patients")
    print(f"imaging-confirmed PFS from the line start — median {np.median(t):.1f} months, "
          f"{e.mean():.0%} progressed, {1 - e.mean():.0%} censored")
    print(f"survival evaluated on {GRID[0]:.0f}-{GRID[-1]:.0f} months\n")

    res, meta = run_ladder(d, x, xm)
    print(f"   {'model':30s} {'IBS':>8} {'skill':>8} {'C-index':>9} {'PIT KS':>8} "
          f"{'6-mo S':>8}")
    print("   " + "-" * 78)
    for _, r in res.iterrows():
        print(f"   {r['model']:30s} {r['ibs']:8.4f} {r['ibs_skill']:8.3f} "
              f"{r['c_index']:9.3f} {r['pit_ks']:8.3f} {r['median_6mo_surv']:8.2f}")
    best = res.loc[res["ibs_skill"].idxmax()]
    print(f"\n   bar for an anchored generator: {best['model']} — IBS skill "
          f"{best['ibs_skill']:.3f}, C-index {best['c_index']:.3f}")
    print("   skill is versus the marginal KM, the calibrated-but-useless forecast.")
    print("   PIT KS is distance from uniform: lower is better calibrated, and a flat")
    print("   forecast can be well calibrated while carrying no information — read both.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pfs_distribution_bar.json").write_text(json.dumps(
        {"n_decisions": int(len(d)), "n_patients": int(d["record_id"].nunique()),
         "cancers": sorted(d["cancer"].unique().tolist()),
         "median_pfs_months": float(np.median(t)), "event_rate": float(e.mean()),
         "grid_months": GRID.tolist(), **meta,
         "results": res.to_dict("records")}, indent=2) + "\n")
    print(f"\nwrote {args.out_dir} (aggregate only — no patient-level rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
