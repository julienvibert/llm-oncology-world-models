#!/usr/bin/env python3
"""How does anchoring align generation? A decomposition, not a single score.

A pooled score can improve for three different reasons, and the paper needs to say which:

    calibration      the population-level bias shrinks (everyone moves toward truth)
    discrimination   patients move toward THEIR OWN truth: the across-patient ranking of
                     generated values tracks the ranking of real ones
    compression      the error shrinks because the generated cohort collapses toward a
                     common course -- an improvement in MAE that DESTROYS heterogeneity

For each arm and channel this reports bias, MAE, Spearman rho (within cancer, so the
trivial between-cancer signal cannot masquerade as discrimination), and the SD ratio of
generated per-patient means to real values -- the compression measure. 1.0 preserves real
heterogeneity; below 1.0 the model regresses patients toward a common course.

Also: the four pre-registered contrasts on the occupancy geometry (masked five-state
total-variation distance), fan coverage of the real outcome, progression-entry agreement,
and the regimen-depth stratification. Patient-level bootstrap CIs throughout.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only.

Run:  uv run python scripts/explore/four_arm_deep_readout_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEEP = ROOT / "outputs" / "exploratory_four_arm_deep"
ARMS = ("patient_only", "setting", "generic_cohort", "patients_like_me")
CONTRASTS = (("setting", "patient_only", "2-1 setting"),
             ("generic_cohort", "setting", "3-2 generic cohort"),
             ("patients_like_me", "generic_cohort", "4-3 matched vs generic"),
             ("patients_like_me", "patient_only", "4-1 complete method"))
B = 2000
RNG = np.random.default_rng(20260828)


def boot_ci(values: np.ndarray, stat=np.mean) -> tuple[float, float, float]:
    """Patient-level bootstrap: point estimate and 95% interval."""
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(values), (B, len(values)))
    draws = stat(values[idx], axis=1)
    return float(stat(values)), float(np.quantile(draws, 0.025)), \
        float(np.quantile(draws, 0.975))


def paired(df: pd.DataFrame, col: str, hi: str, lo: str) -> dict:
    """hi minus lo on |col|, paired within patient, bootstrap CI + Wilcoxon p."""
    from scipy.stats import wilcoxon
    wide = df.pivot_table(index="query_id", columns="arm", values=col)
    wide = wide.dropna(subset=[hi, lo])
    delta = (wide[hi] - wide[lo]).to_numpy()
    est, lo_ci, hi_ci = boot_ci(delta)
    p = wilcoxon(wide[hi], wide[lo]).pvalue if len(wide) >= 10 else float("nan")
    return {"delta": round(est, 4), "ci": [round(lo_ci, 4), round(hi_ci, 4)],
            "p": float(f"{p:.3g}"), "n": int(len(wide))}


def within_cancer_rho(merged: pd.DataFrame, gen_col: str, real_col: str) -> float:
    """Spearman rho computed inside each cancer, weighted by n.

    Pooled rho would be inflated by the between-cancer outcome differences, which every
    arm reproduces trivially.
    """
    from scipy.stats import spearmanr
    rhos, ns = [], []
    for _c, grp in merged.groupby("cancer"):
        if len(grp) >= 10:
            rhos.append(spearmanr(grp[gen_col], grp[real_col]).statistic)
            ns.append(len(grp))
    return float(np.average(rhos, weights=ns)) if rhos else float("nan")


def main() -> int:
    gen = pd.read_csv(DEEP / "generated_side.csv")
    real = pd.read_csv(DEEP / "real_side.csv")
    per = (gen.groupby(["query_id", "arm"], as_index=False)
              .agg(cancer=("cancer", "first"), depth=("depth", "first"),
                   gen_os=("gen_os", "mean"), months_on_tx=("months_on_tx", "mean"),
                   n_sub_lines=("n_sub_lines", "mean"),
                   n_new_sites=("n_new_sites", "mean"), occ_tv=("occ_tv", "mean"),
                   occ_tv_realtime=("occ_tv_realtime", "mean"),
                   gen_unresolved_frac=("gen_unresolved_frac", "mean"),
                   t_first_prog=("t_first_prog", "mean"),
                   prog_frac=("t_first_prog", lambda v: np.mean(np.isfinite(v)))))
    m = per.merge(real[["query_id", "died", "real_os", "months_on_tx", "n_sub_lines",
                        "n_new_sites", "t_first_prog"]],
                  on="query_id", suffixes=("", "_real"))
    report: dict = {}

    # ---- 1. the decomposition table ---------------------------------------------------
    channels = (("gen_os", "real_os", "survival (died only)", True),
                ("months_on_tx", "months_on_tx_real", "months on treatment", False),
                ("n_sub_lines", "n_sub_lines_real", "subsequent regimens", False),
                ("n_new_sites", "n_new_sites_real", "new distant sites", False))
    decomp: dict = {}
    for gen_col, real_col, label, died_only in channels:
        sub = m[m.died] if died_only else m
        rows = {}
        for arm in ARMS:
            g = sub[sub.arm == arm].dropna(subset=[gen_col, real_col])
            err = (g[gen_col] - g[real_col]).to_numpy()
            bias, b_lo, b_hi = boot_ci(err)
            mae, m_lo, m_hi = boot_ci(np.abs(err))
            rows[arm] = {
                "n": int(len(g)),
                "bias": [round(bias, 2), round(b_lo, 2), round(b_hi, 2)],
                "mae": [round(mae, 2), round(m_lo, 2), round(m_hi, 2)],
                "rho_within_cancer": round(within_cancer_rho(g, gen_col, real_col), 3),
                # COMPRESSION: the heterogeneity the generated cohort keeps
                "sd_ratio": round(float(g[gen_col].std() / g[real_col].std()), 3),
            }
        decomp[label] = rows
    report["decomposition"] = decomp

    # ---- 2. occupancy geometry contrasts ----------------------------------------------
    occ = {"mean_tv_by_arm": {}}
    for arm in ARMS:
        est, lo, hi = boot_ci(per[per.arm == arm].occ_tv.to_numpy())
        occ["mean_tv_by_arm"][arm] = [round(est, 4), round(lo, 4), round(hi, 4)]
    occ["contrasts"] = {label: paired(per, "occ_tv", hi, lo)
                        for hi, lo, label in CONTRASTS}
    # THE SAFEGUARDED DISTANCE beside the intersection one, and the unresolved generated
    # time per arm. A prediction that goes silent over a period it was asked about is
    # charged for it here; the contrasts must agree under both definitions.
    occ["contrasts_realtime"] = {label: paired(per, "occ_tv_realtime", hi, lo)
                                 for hi, lo, label in CONTRASTS}
    occ["generated_unresolved_fraction_by_arm"] = {
        a: round(float(per[per.arm == a].gen_unresolved_frac.mean()), 4) for a in ARMS}
    report["occupancy_geometry"] = occ

    # ---- 3. progression-entry agreement -----------------------------------------------
    prog = {}
    real_has = real.set_index("query_id").t_first_prog.apply(np.isfinite)
    for arm in ARMS:
        g = per[per.arm == arm].set_index("query_id")
        both = g.join(real_has.rename("real_has"), how="inner")
        # the fan votes: an entry is predicted when most repeats produce one
        pred = both.prog_frac > 0.5
        tp = int((pred & both.real_has).sum())
        tn = int((~pred & ~both.real_has).sum())
        fp = int((pred & ~both.real_has).sum())
        fn = int((~pred & both.real_has).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        timing = (g.t_first_prog
                  - real.set_index("query_id").t_first_prog).abs().dropna().to_numpy()
        t_est, t_lo, t_hi = boot_ci(timing)
        prog[arm] = {"sensitivity": round(sens, 3), "specificity": round(spec, 3),
                     "balanced_accuracy": round((sens + spec) / 2, 3),
                     "timing_mae_months": [round(t_est, 2), round(t_lo, 2),
                                           round(t_hi, 2)],
                     "n_timing": int(len(timing))}
    report["progression_entry"] = prog

    # ---- 4. fan coverage of the real outcome ------------------------------------------
    cov = {}
    died_ids = set(real[real.died].query_id)
    fans = gen[gen.query_id.isin(died_ids)].dropna(subset=["gen_os"])
    for arm in ARMS:
        rows_ = []
        for qid, grp in fans[fans.arm == arm].groupby("query_id"):
            ro = float(real.loc[real.query_id == qid, "real_os"].iloc[0])
            v = grp.gen_os.to_numpy()
            if len(v) < 5:
                continue
            q10, q90 = np.quantile(v, [0.1, 0.9])
            rows_.append((float(v.min()) <= ro <= float(v.max()),
                          q10 <= ro <= q90, q90 - q10))
        arr = np.array(rows_, dtype=float)
        full, f_lo, f_hi = boot_ci(arr[:, 0])
        central, c_lo, c_hi = boot_ci(arr[:, 1])
        cov[arm] = {"n": int(len(arr)),
                    "covers_full_range": [round(full, 3), round(f_lo, 3),
                                          round(f_hi, 3)],
                    "covers_central_80": [round(central, 3), round(c_lo, 3),
                                          round(c_hi, 3)],
                    "median_width_q10_q90_months": round(float(np.median(arr[:, 2])), 1)}
    report["fan_coverage"] = cov

    # ---- 5. depth stratification ------------------------------------------------------
    depth = {}
    for d in ("regimen 1", "regimen 2", "regimen 3+"):
        sub = per[per.depth == d]
        sub_os = m[(m.depth == d) & m.died].copy()
        sub_os["abs_os_err"] = (sub_os.gen_os - sub_os.real_os).abs()
        depth[d] = {
            "occ_tv 4-3": paired(sub, "occ_tv", "patients_like_me", "generic_cohort"),
            "occ_tv 4-1": paired(sub, "occ_tv", "patients_like_me", "patient_only"),
            "os_err 4-1": paired(sub_os, "abs_os_err", "patients_like_me",
                                 "patient_only"),
        }
    report["by_depth"] = depth

    (DEEP / "deep_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
