"""Extended Data (file ed08b_cancer_contrasts): primary whole-course contrasts by cancer.

Message: the matched-arm advantage on the whole-course score points the same way in lung, breast and
colorectal cancer at 12 and 24 months; per-cancer intervals are wide.
Data: joint_scores_scores_20261012.csv, quantity equal_size, convention primary, cancer NSCLC / BRCA / CRC,
rows contrast:4-1 and contrast:4-3 (mean, ci_low, ci_high, p_wilcoxon unadjusted, below_floor).
Rows flagged below_floor (< 20 patients) are drawn open and labelled descriptive, with no P.
Run: uv run python scripts/figures/panels/ed08b_cancer_contrasts.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import F_REL, INK, PAL, dot, final_table, ptxt, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

j = final_table("joint_scores_scores_20261012.csv")
j = j[(j.convention == "primary") & (j.quantity == "equal_size")]
HS = [3, 6, 12, 24]
CANCERS = [("NSCLC", "Lung cancer"), ("BRCA", "Breast cancer"), ("CRC", "Colorectal cancer")]
CON = [("contrast:4-1", "o", +0.17), ("contrast:4-3", "s", -0.17)]
r = {(ca, h, c): j[(j.cancer == ca) & (j.horizon_months == h) & (j.row == c)].iloc[0]
     for ca, _ in CANCERS for h in HS for c, *_ in CON}
n = {(ca, h): int(r[(ca, h, "contrast:4-1")].n) for ca, _ in CANCERS for h in HS}
assert all(int(x.n) == n[(ca, h)] for (ca, h, c), x in r.items())
pop = final_table("population_composition_scores_20261012.csv")
pop = pop[pop.population == "joint:primary"].set_index("horizon_months")
assert all(int(pop.loc[h][f"n_{ca}"]) == n[(ca, h)] for ca, _ in CANCERS for h in HS)
floor = {k: bool(x.below_floor) for k, x in r.items()}

write_aggregates("ed08b_cancer_contrasts",
                 ["cancer", "horizon_months", "row", "n_patients", "mean", "ci_low", "ci_high", "p_wilcoxon_unadjusted",
                  "below_floor"],
                 [[ca, h, c, n[(ca, h)], x["mean"], x.ci_low, x.ci_high, x.p_wilcoxon, floor[(ca, h, c)]]
                  for (ca, h, c), x in r.items()])

W, H = 180, 46
fig, axs = panel(W, H, ncols=3, sharex=True)
fig.set_layout_engine("none")
LB, PW, PC, GAP, B, T = 13.5, 35.6, 9.0, 3.0, 8.5, 5.0
ys = np.arange(len(HS))[::-1]
XL = (-0.06, 0.015)
for i, (ax, (ca, title)) in enumerate(zip(axs, CANCERS)):
    x0 = LB + i * (LB + PW + PC + GAP)
    ax.set_position([x0 / W, B / H, PW / W, (H - B - T) / H])
    ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yi, h in zip(ys, HS):
        for c, mk, off in CON:
            x = r[(ca, h, c)]
            fl = floor[(ca, h, c)]
            dot(ax, yi + off, x["mean"], x.ci_low, x.ci_high, "white" if fl else PAL["arm_matched"], mk=mk, ms=4.2)
            ax.text((x0 + PW + PC - 0.3) / W, yi + off, "descriptive" if fl else ptxt(x.p_wilcoxon), transform=tr,
                    fontsize=6, ha="right", va="center")
    ax.set_xlim(*XL)
    ax.set_xticks([-0.06, -0.04, -0.02, 0])
    ax.set_xticklabels(["−0.06", "−0.04", "−0.02", "0"])
    ax.set_ylim(-0.55, len(HS) - 0.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{h} months\nn = {n[(ca, h)]}" for h in HS])
    ax.text(0.5, 1.0 + 0.8 / (H - B - T), title, transform=ax.transAxes, fontsize=6, ha="center", va="bottom")
    fig.text((x0 + PW + PC - 0.3) / W, (H - T + 0.8) / H, "P*", fontsize=6, ha="right", va="bottom")
fig.supxlabel("Difference in whole-course score (matched minus other arm; negative = matched closer)",
              x=0.5, y=0.4 / H, fontsize=7, ha="center", va="bottom")
fig.text(0.5 / W, 0.4 / H, "*unadjusted", fontsize=6, ha="left", va="bottom")

n_floor = sum(floor.values())


def txt(ca, c):
    return "; ".join(f"{h} months {r[(ca, h, c)]['mean']:+.4f} ({r[(ca, h, c)].ci_low:+.4f} to {r[(ca, h, c)].ci_high:+.4f}), "
                     f"n = {n[(ca, h)]}, {ptxt(r[(ca, h, c)].p_wilcoxon)}" for h in HS)


legend = (
    "Primary whole-course contrasts (equal-size convention) within each cancer. Symbols as in panel a: circles, matched "
    "examples minus patient record; squares, matched examples minus generic examples; negative values favour the matched arm. "
    "Points, mean paired difference over patients of that cancer; lines, 95% bootstrap interval over patients of that cancer (2,000 paired resamples); P, two-sided Wilcoxon signed-rank test, unadjusted; these "
    "per-cancer contrasts are descriptive subgroups of the primary comparison (Figure 4), not separate tests of it. "
    "n, patients of that cancer with complete documentation up to the horizon. "
    + ("Rows with fewer than 20 patients would be drawn open and labelled descriptive; none fell below this floor. "
       if n_floor == 0 else f"{n_floor} rows with fewer than 20 patients are drawn open and labelled descriptive (no P). ")
    + "The x axis differs from panel a. "
    + " ".join(f"{title}: matched vs patient record, {txt(ca, 'contrast:4-1')}; matched vs generic, {txt(ca, 'contrast:4-3')}."
               for ca, title in CANCERS)
)
save_panel(fig, "ed08b_cancer_contrasts", legend=legend,
           sources=[f"{F_REL}/joint_scores_scores_20261012.csv", f"{F_REL}/population_composition_scores_20261012.csv"],
           confirm=[f"below_floor rows: {n_floor} (none at the primary convention; code draws them open with 'descriptive').",
                    "Per-cancer P values are unadjusted p_wilcoxon; wording 'descriptive subgroups of the primary family' to be checked against SCORING_SPEC.",
                    "Contrasts 2-1 and 3-2 by cancer are not shown (saved in the table)."],
           message="The matched-arm advantage points the same way in lung, breast and colorectal cancer at 12 and 24 months; "
                   "per-cancer intervals are wide.")
