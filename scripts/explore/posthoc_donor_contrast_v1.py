#!/usr/bin/env python3
"""Post hoc contrast: matched versus generic donors used directly, 12-month survival.

Rule fixed before computing (14 Oct 2026).
Reuses the scorer's own functions (four_arm_scores_v1: ipcw_weights, bootstrap_cohorts,
wilcoxon_p) on its saved per-patient predictions, with the same patient order (sorted query
ids), the same 450-patient cohort for censoring weights and the same bootstrap seed, as in
`scripts/four_arm_extract_v1.py` (`boot_brier`).

Two checks run first and must pass: the rebuilt times and events equal the saved rows, and the
two donor contrasts already in the final survival table are reproduced exactly.

GENIE-BPC under DUA: reads gitignored per-patient files; writes aggregate rows only.
Run: uv run python scripts/explore/posthoc_donor_contrast_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import four_arm_scores_v1 as S  # noqa: E402

BASE = ROOT / "outputs" / "four_arm_preprint_20260928"
PRIVATE = BASE / "private" / "survival_patient_rows_scores_20261012.csv"
TABLE = BASE / "survival_scores_20261012.csv"
FRAME = ROOT / "outputs" / "exploratory_four_arm_deep" / "baseline_landmark_frame.csv"
OUT = BASE / "posthoc_donor_contrast_20261014.csv"
H = 12.0
CHECK_PAIRS = [("vs:generic_cohort-donor_generic", "generic_cohort", "donor_generic"),
               ("vs:patients_like_me-donor_matched", "patients_like_me", "donor_matched")]
NEW_PAIR = ("posthoc:donor_matched-donor_generic", "donor_matched", "donor_generic")


def main() -> None:
    frame = pd.read_csv(FRAME, float_precision="round_trip").set_index("query_id")
    rows = pd.read_csv(PRIVATE, float_precision="round_trip")
    rows = rows[rows["horizon_months"] == H].set_index("query_id")
    qall = sorted(frame.index)
    assert len(qall) == 450
    t_all = frame.loc[qall, "time"].to_numpy(float)
    e_all = frame.loc[qall, "event"].to_numpy(int)
    st_all = frame.loc[qall, "cancer"].to_numpy(str)
    common = rows.index
    assert np.allclose(frame.loc[common, "time"].to_numpy(float), rows["time"].to_numpy(float),
                       atol=1e-3), "times differ from the saved rows"
    assert (frame.loc[common, "event"].to_numpy(int) == rows["event"].to_numpy(int)).all()
    in_pop = np.isin(qall, common)
    forecasters = ["generic_cohort", "patients_like_me", "donor_generic", "donor_matched"]
    pred = {f: np.full(len(qall), np.nan) for f in forecasters}
    pos = {q: i for i, q in enumerate(qall)}
    for q, r in rows.iterrows():
        for f in forecasters:
            pred[f][pos[q]] = r[f"p_{f}"]
    w, y, _ = S.ipcw_weights(t_all, e_all, st_all, H)
    resampled = [(idx, *S.ipcw_weights(t_all[idx], e_all[idx], st_all[idx], H)[:2])
                 for idx in S.bootstrap_cohorts(st_all.tolist(), "survival", H)]

    def val(f, idx, wv, yv, mask):
        loss = np.where(np.isnan(yv), 0.0, (pred[f][idx] - np.nan_to_num(yv)) ** 2)
        sel = mask[idx]
        return float((wv[sel] * loss[sel]).sum() / max(sel.sum(), 1))

    def contrast(fa, fb, mask):
        full = np.arange(len(qall))
        est = val(fa, full, w, y, mask) - val(fb, full, w, y, mask)
        draws = [val(fa, i, wb, yb, mask) - val(fb, i, wb, yb, mask) for i, wb, yb in resampled]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        sub = np.flatnonzero(mask)
        d = np.where(np.isnan(y[sub]), np.nan, (pred[fa][sub] - np.nan_to_num(y[sub])) ** 2
                     - (pred[fb][sub] - np.nan_to_num(y[sub])) ** 2)
        return est, float(lo), float(hi), S.wilcoxon_p(d[~np.isnan(d)]), len(sub)

    table = pd.read_csv(TABLE, float_precision="round_trip")
    saved = table[(table.horizon_months == H) & (table.cancer == "all")].set_index("row")
    for name, fa, fb in CHECK_PAIRS:
        est, lo, hi, _p, _n = contrast(fa, fb, in_pop)
        ref = saved.loc[name]
        got = (round(est, 6), round(lo, 6), round(hi, 6))
        want = (ref.brier_ipcw_difference, ref.ci_low, ref.ci_high)
        assert np.allclose(got, want, atol=2e-6), f"{name}: {got} != {want}"
        print(f"check reproduced: {name} {got}")

    out = []
    for cancer in ("all", *sorted(set(st_all))):
        mask = in_pop & ((st_all == cancer) if cancer != "all" else True)
        est, lo, hi, p, n = contrast(NEW_PAIR[1], NEW_PAIR[2], mask)
        out.append({"horizon_months": H, "cancer": cancer, "row": NEW_PAIR[0], "n": n,
                    "brier_ipcw_difference": round(est, 6), "ci_low": round(lo, 6),
                    "ci_high": round(hi, 6), "p_wilcoxon_evaluable": p,
                    "status": "post hoc, unadjusted; interval conditional on fixed donor forecasts"})
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
