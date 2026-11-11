"""Panel f4c_survival_12m: twelve-month survival: mean predicted survival and Brier score.

Message: matched examples move predicted 12-month survival towards the observed value and lower the
Brier error, but direct donor predictions and a Cox model are more accurate.
Data: final survival table (survival_scores_20261012.csv), horizon 12 months, all cancers, rows
forecaster:<f> (mean_predicted_alive, observed_alive_ipcw, brier_ipcw with ci_low/ci_high) and the paired
Brier contrasts (contrast:*, vs:*; brier_ipcw_difference, ci, p_wilcoxon_evaluable) for the legend.
Comparison population: 449 patients with all seven forecasters available.
The panel shares its rows with f4f_c_index, which sits to its right in the assembled figure.
Run: uv run python scripts/figures/panels/f4c_survival_12m.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ROWS, F_REL, INK, LIGHT, dot, point, final_table, ptxt, write_aggregates, R2  # noqa: E402
from common import panel, save_panel  # noqa: E402

t = final_table("survival_scores_20261012.csv")
t = t[(t.horizon_months == 12) & (t.cancer == "all")].set_index("row")
fc = {k: t.loc[f"forecaster:{k}"] for k, *_ in ROWS}
N = int(fc["patient_only"].n)
assert all(int(r.n) == N == 449 for r in fc.values())
obs = float(fc["patient_only"].observed_alive_ipcw)
assert all(abs(float(r.observed_alive_ipcw) - obs) < 1e-12 for r in fc.values())

write_aggregates("f4c_survival_12m",
                 ["forecaster", "n_patients", "mean_pred_12m_pct", "observed_ipcw_12m_pct", "brier_ipcw", "brier_lo", "brier_hi", "ece"],
                 [[k, N, round(100 * fc[k].mean_predicted_alive, 2), round(100 * obs, 2), fc[k].brier_ipcw,
                   fc[k].ci_low, fc[k].ci_high, fc[k].ece] for k, *_ in ROWS])

W = R2["b_width"]
fig, (a1, a2) = panel(W, R2["height"], ncols=2, sharey=True)
fig.set_layout_engine("none")
H = R2["height"]
L, gap, R = R2["label_col"], 4.0, 1.5
w = (W - L - gap - R) / 2
for ax, x0 in ((a1, L), (a2, L + w + gap)):
    ax.set_position([x0 / W, R2["bottom"] / H, w / W, (H - R2["bottom"] - R2["top"]) / H])

y = np.arange(len(ROWS))[::-1]
a1.axvline(100 * obs, color=INK, lw=0.75, zorder=1)
a1.text(100 * obs + 1.0, len(ROWS) - 0.15, f"Observed {100*obs:.1f}%", ha="left", va="center", fontsize=6)
for yi, (k, lab, col, mk) in zip(y, ROWS):
    m = 100 * fc[k].mean_predicted_alive
    point(a1, yi, m, col, mk)
    if m > 100 * obs:
        a1.text(m + 2.0, yi, f"{m:.1f}%", va="center", ha="left", fontsize=6)
    else:
        a1.text(m - 2.0, yi, f"{m:.1f}%", va="center", ha="right", fontsize=6)
    dot(a2, yi, fc[k].brier_ipcw, fc[k].ci_low, fc[k].ci_high, col, mk)
    a2.text(fc[k].ci_high + 0.005, yi, f"{fc[k].brier_ipcw:.3f}", va="center", ha="left", fontsize=6)
for ax, xl in ((a1, (55, 100)), (a2, (0.13, 0.28))):
    ax.plot(xl, [y[3] - 0.5] * 2, color=LIGHT, lw=0.5)
    ax.set_xlim(*xl)
a1.set_yticks(y)
a1.set_yticklabels([r[1] for r in ROWS])
a1.set_ylim(-0.6, len(ROWS) + 0.35)
a1.set_xticks([60, 70, 80, 90, 100])
a1.set_xlabel(f"Mean predicted survival at\n12 months (%), n = {N} patients")
a2.set_xticks([0.15, 0.20, 0.25])
a2.set_xticklabels(["0.15", "0.20", "0.25"])
a2.set_xlabel("Brier score at 12 months\n(prediction error; 0 = perfect)")
a2.tick_params(axis="y", left=False)


def contrast(row):
    r = t.loc[row]
    return f"{r.brier_ipcw_difference:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}), {ptxt(r.p_wilcoxon_evaluable)}"


lab = {k: lb for k, lb, *_ in ROWS}
arms = "; ".join(f"{lab[k]} {100*fc[k].mean_predicted_alive:.1f}%" for k, *_ in ROWS)
briers = "; ".join(f"{lab[k]} {fc[k].brier_ipcw:.3f} ({fc[k].ci_low:.3f} to {fc[k].ci_high:.3f})" for k, *_ in ROWS)
pairs = [("contrast:2-1", "+ care setting minus patient record"), ("contrast:3-2", "+ generic minus + care setting"),
         ("contrast:4-3", "+ matched minus + generic"), ("contrast:4-1", "+ matched minus patient record"),
         ("vs:generic_cohort-donor_generic", "+ generic examples minus generic donors"),
         ("vs:patients_like_me-donor_matched", "+ matched examples minus matched donors"),
         ("vs:patients_like_me-cox_cv", "+ matched examples minus Cox model")]
legend = (
    f"Survival at 12 months predicted for {N} of the 450 GENIE-BPC patients (150 each with lung, breast and colorectal cancer; "
    "the comparison is restricted to the patients for whom all seven forecasters gave a prediction). Left, mean predicted "
    f"probability of being alive at 12 months: {arms}. Black line, observed survival {100*obs:.1f}% (inverse-probability-of-"
    "censoring weighted). No interval is saved for these means. Right, Brier score of the individual 12-month predictions, the "
    "mean squared difference between each patient's predicted probability and what happened, with censoring weights "
    f"(0 = perfect; 95% bootstrap interval over patients): {briers}. Paired differences in Brier score (95% interval; two-sided "
    "Wilcoxon signed-rank test on evaluable patients, not adjusted for multiplicity; negative = first named better): "
    + "; ".join(f"{name} {contrast(r)}" for r, name in pairs)
    + ". Coloured circles, language-model arms (light amber to dark wine = increasing patient evidence; each patient's "
    "probability is the share of up to ten generated futures alive at 12 months); open downward triangle and open square, observed "
    "survival of the twenty generic or twenty matched donors used directly; open diamond, Cox regression fitted on the other "
    "patients of the cohort (10-fold cross-validation)."
)
save_panel(fig, "f4c_survival_12m", legend=legend,
           sources=[f"{F_REL}/survival_scores_20261012.csv"],
           confirm=["Final table has no interval for the mean predicted survival nor for the observed (IPCW) survival: points are drawn without intervals and the legend says so.",
                    "Observed survival is the IPCW estimate (72.2%) on 449 patients.",
                    "Brier P values are unadjusted secondary tests (p_wilcoxon_evaluable); they are in the legend, not on the panel.",
                    "Generic donors used directly: open triangle.",
                    "Brier now printed to 3 decimals (table precision is higher)."],
           message="Matched examples moved 12-month survival towards observed (about 90% to 79% vs 72%) and lowered Brier; matched donors and Cox were more accurate.")
