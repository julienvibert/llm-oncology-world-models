"""f4f_c_index and f4g_calibration: risk ranking and calibration of the survival forecasts (final tables).

f4f  ranking: Harrell's C-index of each forecaster (24-month restricted mean survival as risk score,
     n = 447) and the paired difference from the matched-examples arm (comparator minus matched) with its
     95% interval, where the planned comparisons provide one. Same rows as f4c_survival_12m;
     it carries its own row labels.
f4g  calibration: predicted versus observed 12-month survival for all seven forecasters (bins as saved);
     key gives the average gap (expected calibration error, percentage points). Placed in Extended Data.
Data: survival_scores_20261012.csv (rows cindex:<f>, cindex_contrast:4-1, cindex_contrast:4-3,
cindex_vs:patients_like_me-donor_matched, cindex_vs:patients_like_me-cox_cv at 24 months; rows
forecaster:<f> at 12 months with ece and ece_bins).
Run: uv run python scripts/figures/panels/f4f_ranking_calibration.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ROWS, F_REL, INK, GREY, LIGHT, dot, point, final_table, write_aggregates, R2  # noqa: E402
from common import panel, save_panel  # noqa: E402

SOURCES = [f"{F_REL}/survival_scores_20261012.csv"]
t = final_table("survival_scores_20261012.csv")
t = t[t.cancer == "all"]
c24 = t[t.horizon_months == 24].set_index("row")
lab = {k: lb for k, lb, *_ in ROWS}

ci = {k: float(c24.loc[f"cindex:{k}"].c_index) for k, *_ in ROWS}
NC = int(c24.loc["cindex:patients_like_me"].n)
assert NC == 447 and all(int(c24.loc[f"cindex:{k}"].n) == NC for k, *_ in ROWS)

# comparator minus matched examples = minus (matched minus comparator) as saved; percentile interval negated
SAVED = {"patient_only": "cindex_contrast:4-1", "generic_cohort": "cindex_contrast:4-3",
         "donor_matched": "cindex_vs:patients_like_me-donor_matched", "cox_cv": "cindex_vs:patients_like_me-cox_cv"}
delta = {}
for k, row in SAVED.items():
    r = c24.loc[row]
    delta[k] = (-float(r.c_index_difference), -float(r.ci_high), -float(r.ci_low))
    assert abs(delta[k][0] - (ci[k] - ci["patients_like_me"])) < 1e-9, k

write_aggregates("f4f_c_index",
                 ["forecaster", "n_patients", "c_index", "delta_vs_matched", "delta_lo", "delta_hi", "source_row"],
                 [[k, NC, ci[k], *(delta[k] if k in delta else ("", "", "")), SAVED.get(k, "")] for k, *_ in ROWS])

# ------------------------------------------------------------------ f4f figure (rows aligned with panel b)
W, H = 180 - R2["b_width"] + 25, R2["height"]  # own row labels
fig, (a1, a2) = panel(W, H, ncols=2, sharey=True)
fig.set_layout_engine("none")
L, w1, gap, R = 28.0, 27.0, 5.0, 2.0
w2 = W - L - w1 - gap - R
for ax, x0, w in ((a1, L, w1), (a2, L + w1 + gap, w2)):
    ax.set_position([x0 / W, R2["bottom"] / H, w / W, (H - R2["bottom"] - R2["top"]) / H])
ys = np.arange(len(ROWS))[::-1]
for yi, (k, _, col, mk) in zip(ys, ROWS):
    point(a1, yi, ci[k], col, mk)
    a1.text(ci[k] + 0.016, yi, f"{ci[k]:.3f}", va="center", ha="left", fontsize=6)
    if k in delta:
        dot(a2, yi, *delta[k], col, mk)
    elif k == "patients_like_me":
        a2.text(0, yi, "reference", va="center", ha="center", fontsize=6, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.6))
    else:
        a2.text(0, yi, "not compared", va="center", ha="center", fontsize=6, color=GREY,
                bbox=dict(fc="white", ec="none", pad=0.6))
a2.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
a1.set_yticks(ys)
a1.set_yticklabels([lab[k] for k, *_ in ROWS])
a1.tick_params(axis="y", length=0)
a1.spines["left"].set_visible(False)
a1.set_ylim(-0.6, len(ROWS) + 0.35)
a1.set_xlim(0.55, 0.80)
a1.set_xticks([0.6, 0.7, 0.8])
a1.set_xticklabels(["0.6", "0.7", "0.8"])
a1.set_xlabel(f"C-index (ranking),\nn = {NC} patients")
a2.set_xlim(-0.065, 0.065)
a2.set_xticks([-0.05, 0, 0.05])
a2.set_xticklabels(["−0.05", "0", "+0.05"])
a2.set_xlabel("C-index minus\nmatched examples")
a2.tick_params(axis="y", left=False)
a2.text(-0.063, len(ROWS) - 0.15, "← matched better", fontsize=6, va="center", ha="left")
a2.text(0.063, len(ROWS) - 0.15, "other better →", fontsize=6, va="center", ha="right")
for ax, xl in ((a1, (0.55, 0.80)), (a2, (-0.065, 0.065))):
    ax.plot(xl, [ys[3] - 0.5] * 2, color=LIGHT, lw=0.5)

dl = "; ".join(f"{lab[k]} {delta[k][0]:+.3f} ({delta[k][1]:+.3f} to {delta[k][2]:+.3f})" for k, *_ in ROWS if k in delta)
save_panel(
    fig, "f4f_c_index", sources=SOURCES,
    legend=("How well each forecaster ranks patients by survival time (Harrell's C-index: the probability that, of two "
            "patients, the one who died earlier was given the shorter predicted survival; 0.5 = chance, 1 = perfect). The ranking "
            "score is each forecaster's predicted restricted mean survival to 24 months. "
            f"n = {NC} of the 450 GENIE-BPC patients (150 each with lung, breast and colorectal cancer), those with all seven "
            "forecasters available at 24 months. Rows as in Fig. 4d. Left, C-index (no interval is saved for single forecasters): "
            + "; ".join(f"{lab[k]} {ci[k]:.3f}" for k, *_ in ROWS)
            + ". Right, paired difference from the matched-examples arm (comparator minus matched examples; positive values mean "
            "the comparator ranked patients better), with the 95% bootstrap interval over patients, for the planned "
            f"comparisons: {dl}. The care-setting arm and the generic donors have no planned paired comparison with the "
            "matched arm ('not compared'); the generic-examples arm ranked patients better than the generic donors used "
            "directly (difference "
            + "{:+.3f}; {:+.3f} to {:+.3f}).".format(*(float(c24.loc['cindex_vs:generic_cohort-donor_generic'][c])
                                                       for c in ("c_index_difference", "ci_low", "ci_high")))
            + " No P values are saved for C-index differences. Colours and symbols as in Fig. 4d."),
    message="The matched examples rank patients about as well as the cohort methods; the gain over the patient record is not clearly established.",
    confirm=["Final table gives no interval for single C-indexes and no P value for C-index differences: points without intervals, differences with saved 95% intervals only.",
             "Differences are the saved 'matched minus comparator' values with sign reversed (interval bounds swapped and negated) to keep the 'comparator minus matched' design.",
             "Population changed from 450 to 447 patients (all seven forecasters available at 24 months).",
             "Generic donors' C-index (0.575) is low because the twenty generic donors are shared by patients of the same cancer and era; it widens the x axis."],
)

# ------------------------------------------------------------------ f4g calibration (Extended Data)
c12 = t[t.horizon_months == 12].set_index("row")
cal = {k: c12.loc[f"forecaster:{k}"] for k, *_ in ROWS}
NB = int(cal["patients_like_me"].n)
assert all(int(r.ece_n_evaluable) == NB == 449 for r in cal.values())
bins = {k: json.loads(cal[k].ece_bins) for k in cal}
write_aggregates("f4g_calibration", ["forecaster", "bin", "n_in_bin", "predicted_pct", "observed_pct", "ece_points"],
                 [[k, b["bin"] + 1, b["n"], round(100 * b["mean_predicted"], 2), round(100 * b["mean_observed"], 2),
                   round(100 * cal[k].ece, 2)] for k, *_ in ROWS for b in bins[k]])
W, H = 72, 44
fig, ax = panel(W, H)
ax.plot([0, 100], [0, 100], color=GREY, lw=0.5, ls=(0, (3, 3)), zorder=0)
for k, _, col, mk in ROWS:
    b = bins[k]
    ls = "--" if k.startswith("donor") else "-"
    lc = INK if mk != "o" else col
    xs, yv = [x["mean_predicted"] * 100 for x in b], [x["mean_observed"] * 100 for x in b]
    ax.plot(xs, yv, ls=ls, color=lc, lw=0.9, zorder=2)
    ax.plot(xs, yv, mk, ms=3.6 if mk == "o" else 3.4, mfc=col, mec=INK, mew=0.5, zorder=3)
ax.set_xlim(0, 103)
ax.set_ylim(0, 103)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("Predicted 12-month\nsurvival (%)")
ax.set_ylabel("Observed 12-month\nsurvival (%)")
fig.set_layout_engine("none")
AX_L, AX_W = 11.5, 29.5
ax.set_position([AX_L / W, 10.5 / H, AX_W / W, (H - 11.5) / H])
# key to the right of the plot: marker, name, average gap
kx = AX_L + AX_W + 2.5           # mm
ytop, dy = H - 3.0, 4.1
fig.text(kx / W, ytop / H, "Average gap (points)", fontsize=6, va="center", ha="left")
for j, (k, lb, col, mk) in enumerate(ROWS):
    yy = (ytop - 4.6 - j * dy) / H
    fig.add_artist(Line2D([(kx + 1.0) / W], [yy], marker=mk, ls="none",
                     ms=3.4 if mk == "o" else 3.2, mfc=col, mec=INK, mew=0.5, transform=fig.transFigure))
    fig.text((kx + 2.6) / W, yy, lb.replace(", direct", ""), fontsize=6, va="center", ha="left")
    fig.text((W - 0.5) / W, yy, f"{100*cal[k].ece:.1f}", fontsize=6, va="center", ha="right")
save_panel(
    fig, "f4g_calibration", sources=SOURCES,
    legend=(f"Calibration of 12-month survival probabilities for all seven forecasters, {NB} GENIE-BPC patients (all seven "
            "forecasters available). Patients were grouped by predicted probability (bins as saved; the language-model arms put "
            "many patients at a predicted 100%, so they have fewer, unequal bins). Each marker is one group: mean predicted "
            "survival against the share of the same patients observed alive at 12 months (no patient was censored before 12 "
            "months in this population). Dashed diagonal, perfect agreement; points below it mean survival was overestimated. "
            "Coloured circles and solid lines, language-model arms (light amber to dark wine = increasing patient evidence); open "
            "triangle and open square with dashed lines, generic and matched donors used directly; open diamond, Cox model. Key, "
            "average gap between predicted and observed survival across the groups, in percentage points (expected calibration "
            "error): " + "; ".join(f"{lab[k]} {100*cal[k].ece:.1f}" for k, *_ in ROWS) + ". No interval is saved for the gap."),
    message="LLM probabilities cluster near certain survival and overestimate it; Cox and the donors follow the diagonal.",
    confirm=["Rebuilt from the final survival table (ece, ece_bins at 12 months, 449 patients).",
             "Observed per bin = mean_observed as saved; 'no patient censored before 12 months' rests on population_composition (unresolved_at_h = 0 at 12 months).",
             "No interval for the gap in the final table."],
)
