"""Extended Data (file ed07e_donor_agreement): documented-window disagreement, forecasts minus the donors used directly.

Message: forecasts with generic examples agree with the documented course better than the generic donors at
every horizon; forecasts with matched examples do not agree better than the matched donors (donors slightly
better at 12 and 24 months, not significant after adjustment).
Data: donor_diagnostics_scores_20261012.csv, quantity disagreement_llm_minus_donor, convention primary,
cancer all, card generic (arm generic_cohort) and matched (arm patients_like_me): mean, ci_low, ci_high,
llm_mean, donor_mean, p_holm_donor_family, n.
Run: uv run python scripts/figures/panels/ed07e_donor_agreement.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import PAL, F_REL, INK, dot, final_table, ptxt, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

d = final_table("donor_diagnostics_scores_20261012.csv")
d = d[(d.quantity == "disagreement_llm_minus_donor") & (d.convention == "primary") & (d.cancer == "all")]
HS = [3, 6, 12, 24]
CARDS = [("generic", "generic_cohort", "Generic", PAL["arm_generic"], +0.19),
         ("matched", "patients_like_me", "Matched", PAL["arm_matched"], -0.19)]
r = {(c, h): d[(d.card == c) & (d.arm == a) & (d.horizon_months == h)].iloc[0] for c, a, *_ in CARDS for h in HS}
assert len(d) == 8

write_aggregates("ed07e_donor_agreement",
                 ["card", "horizon_months", "n_patients", "llm_mean", "donor_mean", "mean_llm_minus_donor", "ci_low",
                  "ci_high", "p_wilcoxon", "p_holm_donor_family"],
                 [[c, h, int(x.n), x.llm_mean, x.donor_mean, x["mean"], x.ci_low, x.ci_high, x.p_wilcoxon,
                   x.p_holm_donor_family] for (c, h), x in r.items()])

W, H = 58, 44
fig, ax = panel(W, H)
fig.set_layout_engine("none")
L, AW, B, T = 12.5, 32.5, 10.5, 5.0
ax.set_position([L / W, B / H, AW / W, (H - B - T) / H])
ys = np.arange(len(HS))[::-1]
ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
for yi, h in zip(ys, HS):
    for c, _, _, col, off in CARDS:
        x = r[(c, h)]
        dot(ax, yi + off, x["mean"], x.ci_low, x.ci_high, col, ms=4.2)
        p = x.p_holm_donor_family
        ax.text((W - 1.5) / W, yi + off, ptxt(p), transform=blended_transform_factory(fig.transFigure, ax.transData),
                fontsize=6, ha="right", va="center", clip_on=False, fontweight="bold" if p < 0.05 else "normal")
ax.set_ylim(-0.6, len(HS) - 0.4)
ax.set_xlim(-0.36, 0.10)
ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1])
ax.set_xticklabels(["−0.3", "", "−0.1", "0", "+0.1"])
ax.set_yticks(ys)
ax.set_yticklabels([f"{h} months" for h in HS])
ax.set_xlabel("Forecast minus donors: time in\ndisagreement (>0, donors closer)")
# key above the plot: the two comparisons
ky = (H - 2.2) / H
for j, (c, _, name, col, _) in enumerate(CARDS):
    x0 = (L + 1 + j * 16) / W
    fig.add_artist(Line2D([x0], [ky], marker="o", ls="none", ms=4.0, mfc=col, mec=INK, mew=0.6, transform=fig.transFigure))
    fig.text(x0 + 1.6 / W, ky, name, fontsize=6, ha="left", va="center")
fig.text((W - 1.5) / W, ky, "Holm P", fontsize=6, ha="right", va="center")


def row(c):
    return "; ".join(f"{h} months {r[(c, h)]['mean']:+.3f} ({r[(c, h)].ci_low:+.3f} to {r[(c, h)].ci_high:+.3f}; forecast "
                     f"{r[(c, h)].llm_mean:.3f}, donors {r[(c, h)].donor_mean:.3f}; n = {int(r[(c, h)].n)}; "
                     f"{ptxt(r[(c, h)].p_holm_donor_family)})" for h in HS)


legend = (
    "Agreement of forecast courses with the documented course, compared with the donors used directly. For each patient, the "
    "documented window is the follow-up with a recorded disease state, up to the horizon; disagreement is the share of that "
    "time in which the forecast disease state differs from the documented one (averaged over the patient's futures, or over the "
    "twenty donors' courses re-observed on the same window). Points, mean over patients of the forecast's disagreement minus "
    "the donors' disagreement (positive, the donors' own courses agree better with what happened); lines, 95% bootstrap "
    "interval over patients; P, two-sided Wilcoxon signed-rank test, Holm-adjusted within the donor family (bold, below 0.05). "
    f"Generic (orange-brown), forecasts with generic examples minus the twenty generic donors: {row('generic')}. "
    f"Matched (wine), forecasts with matched examples minus the twenty matched donors: {row('matched')}. Patients with a "
    "documented window at each horizon, from 450 GENIE-BPC patients (150 each with lung, breast and colorectal cancer); "
    "populations differ between horizons. DeepSeek V4 Flash."
)
save_panel(fig, "ed07e_donor_agreement", legend=legend,
           sources=[f"{F_REL}/donor_diagnostics_scores_20261012.csv"],
           confirm=["Definition of disagreement and of the donor comparator follows the table names and the scoring specification.",
                    "Holm family size not stated in the legend (p_holm_donor_family as saved).",
                    "Occupancy (total-variation) version of the same comparison is in the aggregates source table, not shown.",
                    "Placed in Extended Data Figure 7 because Figure 4 has no room without squeezing."],
           message="Generic-example forecasts agree with documented courses better than generic donors at every horizon; matched-example forecasts do not beat the matched donors.")
