"""Panel f5d_course_contrasts: paired differences in disagreement over evaluable time, matched arm vs others.

Evaluable time includes the absorbing post-death state; per-patient fan means, averaged over patients.

Message: matched examples reduce the time in disagreement by about 2 percentage points at 12 months and 4 at
24 months against the patient record (smaller against generic examples); exploratory, unadjusted P.
Data: course_diagnostics_scores_20261012.csv, quantity disagreement, convention primary, cancer all, rows
contrast:4-1 and contrast:4-3 (mean paired difference, ci_low, ci_high, p_wilcoxon; unadjusted, no Holm).
contrast:2-1 and contrast:3-2 are saved in the aggregates and quoted in the legend.
Run: uv run python scripts/figures/panels/f5d_course_contrasts.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import F_REL, INK, PAL, dot, final_table, ptxt, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

c = final_table("course_diagnostics_scores_20261012.csv")
c = c[(c.convention == "primary") & (c.cancer == "all") & (c.quantity == "disagreement")]
HS = [3, 6, 12, 24]
ALL = ["contrast:2-1", "contrast:3-2", "contrast:4-3", "contrast:4-1"]
r = {(h, row): c[(c.horizon_months == h) & (c.row == row)].iloc[0] for h in HS for row in ALL}
n = {h: int(r[(h, "contrast:4-1")].n) for h in HS}
assert [n[h] for h in HS] == [386, 412, 426, 437]
assert all(int(x.n) == n[h] for (h, _), x in r.items())

write_aggregates("f5d_course_contrasts",
                 ["horizon_months", "row", "n_patients", "mean", "ci_low", "ci_high", "p_wilcoxon_unadjusted"],
                 [[h, row, n[h], r[(h, row)]["mean"], r[(h, row)].ci_low, r[(h, row)].ci_high, r[(h, row)].p_wilcoxon]
                  for h in HS for row in ALL])

COLS = [("contrast:4-1", "+ matched examples\nvs patient record", "o"),
        ("contrast:4-3", "+ matched examples\nvs + generic examples", "s")]
W, H = 90, 34
fig, axs = panel(W, H, ncols=2, sharey=True)
fig.set_layout_engine("none")
L, PW, PC, GAP, B, T = 12.0, 28.5, 9.5, 1.5, 10.0, 7.0
ys = np.arange(len(HS))[::-1]
XL = (-5.6, 1.0)
for i, (ax, (row, title, mk)) in enumerate(zip(axs, COLS)):
    x0 = L + i * (PW + PC + GAP)
    ax.set_position([x0 / W, B / H, PW / W, (H - B - T) / H])
    ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yi, h in zip(ys, HS):
        x = r[(h, row)]
        dot(ax, yi, 100 * x["mean"], 100 * x.ci_low, 100 * x.ci_high, PAL["arm_matched"], mk=mk, ms=4.8)
        ax.text((x0 + PW + PC - 0.3) / W, yi, ptxt(x.p_wilcoxon), transform=tr, fontsize=6, ha="right", va="center")
    ax.set_xlim(*XL)
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1])
    ax.set_xticklabels(["", "−4", "", "−2", "", "0", ""])
    ax.set_ylim(-0.55, len(HS) - 0.45)
    ax.text(0.5, 1.0 + 1.2 / (H - B - T), title, transform=ax.transAxes, fontsize=6, ha="center", va="bottom",
            linespacing=1.1)
    fig.text((x0 + PW + PC - 0.3) / W, (H - T + 1.2) / H, "P*", fontsize=6, ha="right", va="bottom")
    if i:
        ax.tick_params(axis="y", left=False)
axs[0].set_yticks(ys)
axs[0].set_yticklabels([f"{h} months" for h in HS])
fig.supxlabel("Difference in evaluable time in disagreement\n(percentage points; negative = matched closer)",
              x=(L + (2 * PW + PC + GAP) / 2) / W, y=0.6 / H, fontsize=7, ha="center", va="bottom")
fig.text((W - 0.3) / W, 0.6 / H, "*unadjusted,\nexploratory", fontsize=6, ha="right", va="bottom")


def txt(row):
    return "; ".join(f"{h} months {100 * r[(h, row)]['mean']:+.1f} ({100 * r[(h, row)].ci_low:+.1f} to "
                     f"{100 * r[(h, row)].ci_high:+.1f}), {ptxt(r[(h, row)].p_wilcoxon)}" for h in HS)


legend = (
    "Paired differences in disagreement over evaluable time (panel c; recorded follow-up plus time after a recorded death, "
    "death counted as a state) between the arm with matched examples and two other arms, by horizon. For each patient, the "
    "disagreement of the matched arm (mean over its futures) minus that of the other arm, in percentage points of evaluable "
    "time; negative values mean the futures generated with matched examples were in the recorded state for longer. Points, "
    "mean paired difference over patients (each patient weighs the same; circle, against the patient record only; square, "
    "against generic examples); lines, 95% bootstrap interval over patients (2,000 paired resamples, stratified by cancer); "
    "P, two-sided Wilcoxon signed-rank test, unadjusted. This diagnostic is exploratory and outside the primary comparison "
    "(panel b), so no correction for multiple comparisons was applied. n, patients with evaluable time at the horizon, as in panel c. "
    f"Matched examples vs patient record: {txt('contrast:4-1')}. Matched vs generic examples: {txt('contrast:4-3')}. "
    f"Not shown: care setting vs patient record, {txt('contrast:2-1')}; generic examples vs care setting, "
    f"{txt('contrast:3-2')}. DeepSeek V4 Flash."
)
save_panel(fig, "f5d_course_contrasts", legend=legend,
           sources=[f"{F_REL}/course_diagnostics_scores_20261012.csv"],
           confirm=["P values are unadjusted p_wilcoxon (exploratory diagnostic, no Holm), as specified; the figure marks them with an asterisk.",
                    "Contrasts 2-1 and 3-2 are in the legend and aggregates only (3-2 at 24 months: -1.8 points, P = 0.004)."],
           message="Against the patient record, matched examples cut the time with a different state by about 2 points at 12 "
                   "months and 4 at 24 months; against generic examples by 1.5 and 2.6 points (exploratory).")
