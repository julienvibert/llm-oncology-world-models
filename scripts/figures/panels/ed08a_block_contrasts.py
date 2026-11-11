"""Extended Data (file ed08a_block_contrasts): whole-course score by block, matched arm vs patient record and vs generic.

Message: the matched-arm advantage in the whole-course score comes mainly from time in each disease state and
treatment received at 12 and 24 months; survival and new events show no consistent change. Blocks are for
interpretation and are not summed.
Data: joint_scores_scores_20261012.csv, quantity block_survival / block_disease / block_events / block_treatment,
convention primary, cancer all, rows contrast:4-1 and contrast:4-3 (mean, ci_low, ci_high, p_wilcoxon, unadjusted).
Run: uv run python scripts/figures/panels/ed08a_block_contrasts.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import F_REL, INK, PAL, dot, final_table, ptxt, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

j = final_table("joint_scores_scores_20261012.csv")
j = j[(j.convention == "primary") & (j.cancer == "all")]
HS = [3, 6, 12, 24]
BLOCKS = [("block_survival", "Survival"), ("block_disease", "Time in each disease state"),
          ("block_events", "Responses and progressions"), ("block_treatment", "Treatment received")]
CON = [("contrast:4-1", "o", +0.17, "Matched vs patient record"), ("contrast:4-3", "s", -0.17, "Matched vs generic examples")]
r = {(b, h, c): j[(j.quantity == b) & (j.horizon_months == h) & (j.row == c)].iloc[0]
     for b, _ in BLOCKS for h in HS for c, *_ in CON}
n = {h: int(r[("block_survival", h, "contrast:4-1")].n) for h in HS}
assert [n[h] for h in HS] == [286, 269, 233, 181]
assert all(int(x.n) == n[h] for (b, h, c), x in r.items())

write_aggregates("ed08a_block_contrasts",
                 ["block", "horizon_months", "row", "n_patients", "mean", "ci_low", "ci_high", "p_wilcoxon_unadjusted"],
                 [[b, h, c, n[h], x["mean"], x.ci_low, x.ci_high, x.p_wilcoxon] for (b, h, c), x in r.items()])

W, H = 180, 50
fig, axs = panel(W, H, ncols=4, sharey=True, sharex=True)
fig.set_layout_engine("none")
L, PW, PC, GAP, B, T = 23.0, 28.2, 9.0, 2.0, 8.5, 6.0
ys = np.arange(len(HS))[::-1]
XL = (-0.025, 0.012)
for i, (ax, (b, title)) in enumerate(zip(axs, BLOCKS)):
    x0 = L + i * (PW + PC + GAP)
    ax.set_position([x0 / W, B / H, PW / W, (H - B - T) / H])
    ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yi, h in zip(ys, HS):
        for c, mk, off, _ in CON:
            x = r[(b, h, c)]
            dot(ax, yi + off, x["mean"], x.ci_low, x.ci_high, PAL["arm_matched"], mk=mk, ms=4.2)
            ax.text((x0 + PW + PC - 0.3) / W, yi + off, ptxt(x.p_wilcoxon), transform=tr, fontsize=6, ha="right",
                    va="center")
    ax.set_xlim(*XL)
    ax.set_xticks([-0.02, -0.01, 0, 0.01])
    ax.set_xticklabels(["−0.02", "", "0", ""])
    ax.set_ylim(-0.55, len(HS) - 0.45)
    ax.text(0.5, 1.0 + 1.0 / (H - B - T), title, transform=ax.transAxes, fontsize=6, ha="center", va="bottom")
    fig.text((x0 + PW + PC - 0.3) / W, (H - T + 1.0) / H, "P*", fontsize=6, ha="right", va="bottom")
    if i:
        ax.tick_params(axis="y", left=False)
axs[0].set_yticks(ys)
axs[0].set_yticklabels([f"{h} months\nn = {n[h]}" for h in HS])
fig.supxlabel("Difference in block score (matched minus other arm; negative = matched closer)",
              x=(L + (W - L) / 2) / W, y=0.4 / H, fontsize=7, ha="center", va="bottom")
fig.text(0.5 / W, 0.4 / H, "*unadjusted", fontsize=6, ha="left", va="bottom")
for k, (_, mk, _, lab) in enumerate(CON):
    yk = (H - 1.6 - k * 2.6) / H
    fig.add_artist(Line2D([1.6 / W], [yk], marker=mk, ls="none", ms=4.2, mfc=PAL["arm_matched"], mec=INK, mew=0.6,
                          transform=fig.transFigure))
    fig.text(3.2 / W, yk, lab.replace("Matched vs ", "vs "), fontsize=6, ha="left", va="center")


def txt(b, c):
    return "; ".join(f"{h} months {r[(b, h, c)]['mean']:+.4f} ({r[(b, h, c)].ci_low:+.4f} to {r[(b, h, c)].ci_high:+.4f}), "
                     f"{ptxt(r[(b, h, c)].p_wilcoxon)}" for h in HS)


legend = (
    "Per-block contrasts of the whole-course score (equal-size convention, Figure 4). The 17 trajectory summaries fall into "
    "four blocks: survival (2 summaries: alive at the horizon, months alive); time in each disease state (4); new responses and "
    "progressions after the start of the regimen (2); and treatment received (9: time on treatment by class, regimen changes). Each block "
    "is scored separately (energy score on that block's summaries only, with the same equal-size futures) to show where the arms differ; block scores are for "
    "interpretation and are not summed (the whole-course score is not their sum). Points, mean paired difference over "
    "patients, matched-examples arm minus the patient-record arm (circles) or minus the generic-examples arm (squares); "
    "negative values mean the matched arm forecast that block of the documented course better. Lines, 95%"
    " bootstrap interval over patients (2,000 paired resamples, stratified by cancer); P, two-sided Wilcoxon signed-rank test, unadjusted (exploratory). n, patients with complete documentation "
    "up to the horizon, as in Figure 4. "
    + " ".join(f"{title}: matched vs patient record, {txt(b, 'contrast:4-1')}; matched vs generic, {txt(b, 'contrast:4-3')}."
               for b, title in BLOCKS)
)
save_panel(fig, "ed08a_block_contrasts", legend=legend,
           sources=[f"{F_REL}/joint_scores_scores_20261012.csv"],
           confirm=[                    "Block P values are unadjusted p_wilcoxon (no Holm family for blocks in the table).",
                                        "x axis shared across blocks (same score units)."],
           message="The matched-arm advantage comes from time in each disease state and treatment received at 12 and 24 "
                   "months; survival and new events show no consistent change. Blocks are not summed.")
