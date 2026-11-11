"""Figure 4 (file f4i_treatment), predicted documented systemic treatment in the first 12 months.

Message: every arm predicted more documented systemic treatment than was recorded (about 10 vs 6.7 months;
about 1.4 vs 0.54 subsequent regimens); the donors used directly matched the documented means and had lower
absolute errors.
Data: treatment_scalars_scores_20261012.csv, horizon 12 months, quantities months_any and
n_subsequent_regimens, rows forecaster:<f> (mean_forecast, mean_observed, mean_abs_error with ci_low/ci_high),
and paired absolute-error comparisons (contrast:*, vs:*; abs_error_difference, ci, p_wilcoxon) for the legend.
n = 317 patients with evaluable treatment records.
Run: uv run python scripts/figures/panels/f4i_treatment.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ROWS, F_REL, INK, LIGHT, point, final_table, ptxt, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

TX = [k for k in ROWS if k[0] != "cox_cv"]
t = final_table("treatment_scalars_scores_20261012.csv")
t = t[t.horizon_months == 12]
Q = [("months_any", (0, 15.5), [0, 4, 8, 12], "Months on systemic therapy\nin the first 12 months", "{:.1f}"),
     ("n_subsequent_regimens", (0, 2.05), [0, 0.5, 1, 1.5], "Subsequent regimens\nin the first 12 months", "{:.2f}")]
v = {(q, k): t[(t.quantity == q) & (t.row == f"forecaster:{k}")].iloc[0] for q, *_ in Q for k, *_ in TX}
N = int(v[("months_any", "patient_only")].n)
assert N == 317 and all(int(r.n) == N for r in v.values())
obs = {q: float(v[(q, "patient_only")].mean_observed) for q, *_ in Q}
assert all(abs(float(v[(q, k)].mean_observed) - obs[q]) < 1e-12 for q, *_ in Q for k, *_ in TX)
# guard against confusing the forecast mean with its absolute error (matched donors, regimens)
assert round(v[("n_subsequent_regimens", "donor_matched")].mean_forecast, 2) == 0.53
assert round(v[("n_subsequent_regimens", "donor_matched")].mean_abs_error, 2) == 0.61

write_aggregates("f4i_treatment",
                 ["quantity", "forecaster", "n_patients", "mean_forecast", "mean_documented", "mean_signed_error",
                  "mean_abs_error", "abs_error_lo", "abs_error_hi"],
                 [[q, k, N, r.mean_forecast, r.mean_observed, r.mean_signed_error, r.mean_abs_error, r.ci_low, r.ci_high]
                  for (q, k), r in v.items()])

W, H = 84, 33
fig, axs = panel(W, H, ncols=2, sharey=True)
fig.set_layout_engine("none")
L, w1, gap, R, B, T = 23.5, 26.0, 4.0, 1.0, 10.5, 1.0
w2 = W - L - w1 - gap - R
for ax, x0, w in ((axs[0], L, w1), (axs[1], L + w1 + gap, w2)):
    ax.set_position([x0 / W, B / H, w / W, (H - B - T) / H])
ys = np.arange(len(TX))[::-1]
for ax, (q, xl, ticks, xlab, fmt) in zip(axs, Q):
    ax.axvline(obs[q], color=INK, lw=0.75, zorder=1)
    for yi, (k, _, col, mk) in zip(ys, TX):
        point(ax, yi, v[(q, k)].mean_forecast, col, mk)
        ax.text(xl[1], yi, fmt.format(v[(q, k)].mean_abs_error), fontsize=6, ha="right", va="center")
    top = len(TX) - 0.15
    ax.text(obs[q], top, "Documented", fontsize=6, ha="center", va="center", zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(xl[1], top, "Error", fontsize=6, ha="right", va="center")
    ax.plot(xl, [ys[3] - 0.5] * 2, color=LIGHT, lw=0.5)
    ax.set_xlim(*xl)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.set_xlabel(xlab)
axs[0].set_yticks(ys)
axs[0].set_yticklabels([lab for _, lab, *_ in TX])
axs[0].set_ylim(-0.6, len(TX) + 0.35)
axs[1].tick_params(axis="y", left=False)

lab = {k: lb for k, lb, *_ in TX}


def pair(q, row):
    r = t[(t.quantity == q) & (t.row == row)].iloc[0]
    return f"{r.abs_error_difference:+.2f} ({r.ci_low:+.2f} to {r.ci_high:+.2f}), {ptxt(r.p_wilcoxon)}"


def block(q, unit, fmt):
    means = "; ".join(f"{lab[k]} {fmt.format(v[(q, k)].mean_forecast)}" for k, *_ in TX)
    errs = "; ".join(f"{lab[k]} {fmt.format(v[(q, k)].mean_abs_error)} ({fmt.format(v[(q, k)].ci_low)} to "
                     f"{fmt.format(v[(q, k)].ci_high)})" for k, *_ in TX)
    pairs = "; ".join(f"{name} {pair(q, row)}" for row, name in [
        ("contrast:4-1", "+ matched examples minus patient record"), ("contrast:4-3", "+ matched minus + generic examples"),
        ("vs:generic_cohort-donor_generic", "+ generic examples minus generic donors"),
        ("vs:patients_like_me-donor_matched", "+ matched examples minus matched donors")])
    return (f"documented mean {fmt.format(obs[q])} {unit}; mean forecast: {means}. Mean absolute error per patient (95% "
            f"bootstrap interval): {errs}. Paired differences in absolute error (95% interval; two-sided Wilcoxon signed-rank "
            f"test, not adjusted for multiplicity): {pairs}")


legend = (
    f"Predicted documented systemic treatment in the first 12 months after the start of the recorded regimen, {N} of the 450 "
    "GENIE-BPC patients (those with evaluable treatment records through 12 months; 150 each with lung, breast and colorectal "
    "cancer at entry). Left, months with any systemic therapy; right, number of subsequent regimens. Points, mean over patients "
    "of each forecaster's prediction (for the language-model arms, each patient's value is the mean of up to ten generated "
    "futures; for the donors used directly, the mean over the eligible members of the card, those whose courses are resolved "
    "through 12 months or death, at least seven of the twenty); no interval is saved for these "
    "means. Black line, documented mean. Right-hand column (Error), mean absolute difference between each patient's "
    "prediction and the documented value, in the same units. Months on therapy: " + block("months_any", "months", "{:.1f}")
    + ". Subsequent regimens: " + block("n_subsequent_regimens", "regimens", "{:.2f}")
    + ". Coloured circles, language-model arms (light amber to dark wine = increasing patient evidence); open downward triangle "
    "and open square, generic and matched donors used directly. The comparison concerns treatment documented in the registry, "
    "which may miss treatment given elsewhere. DeepSeek V4 Flash."
)
save_panel(fig, "f4i_treatment", legend=legend,
           sources=[f"{F_REL}/treatment_scalars_scores_20261012.csv"],
           confirm=["No interval for mean forecasts in the final table: points without intervals; intervals given for the absolute errors (legend).",
                    "Matched donors: forecast mean 0.53 regimens (point) and mean absolute error 0.61 (Error column); asserted in the script.",
                    "Wording 'predicted documented systemic treatment'; not 'over-treated'.",
                    "Donor forecast: mean over eligible card members (courses resolved through 12 months or death, at least seven of twenty), not all twenty."],
           message="All arms predicted more documented systemic treatment than recorded (about 10 vs 6.7 months; 1.3-1.4 vs 0.54 regimens); donors used directly matched the means and had lower errors.")
