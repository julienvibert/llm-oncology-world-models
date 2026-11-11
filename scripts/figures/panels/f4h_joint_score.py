"""Panel f4h_joint_score: primary comparison (fixed after exploratory analyses, before the final recomputation; post hoc, not preregistered): whole-course score, four contrasts at four horizons.

Message: matched examples improve the whole-course score over the patient record at 12 and 24 months and
over generic examples at 24 months; no step is detectable at 3 or 6 months after adjustment.
Data: joint_scores_scores_20261012.csv, quantity equal_size, convention primary, cancer all; rows
arm:<arm> (mean energy score, CI) and contrast:2-1, 3-2, 4-3, 4-1 (mean paired difference, ci_low, ci_high,
p_wilcoxon, p_holm_primary_family; negative favours the later arm). n per horizon: patients with complete
documentation (selected populations, not comparable across horizons); deaths by horizon from
population_composition_scores_20261012.csv (joint:primary).
Run: uv run python scripts/figures/panels/f4h_joint_score.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, F_REL, INK, dot, final_table, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

j = final_table("joint_scores_scores_20261012.csv")
j = j[(j.quantity == "equal_size") & (j.convention == "primary") & (j.cancer == "all")]
pop = final_table("population_composition_scores_20261012.csv")
pop = pop[pop.population == "joint:primary"].set_index("horizon_months")
HS = [3, 6, 12, 24]
CONTRASTS = [("contrast:2-1", "+ care setting\nvs record alone", 1),
             ("contrast:3-2", "+ generic examples\nvs + care setting", 2),
             ("contrast:4-3", "+ matched examples\nvs + generic examples", 3),
             ("contrast:4-1", "+ matched examples\nvs record alone", 3)]
lab = {k: lb for k, lb, *_ in ARMS}
r = {(h, row): j[(j.horizon_months == h) & (j.row == row)].iloc[0] for h in HS
     for row in [c[0] for c in CONTRASTS] + [f"arm:{k}" for k, *_ in ARMS]}
n = {h: int(r[(h, "contrast:2-1")].n) for h in HS}
assert [n[h] for h in HS] == [286, 269, 233, 181]
assert all(int(r[(h, row)].n) == n[h] for (h, row) in r)
assert all(int(pop.loc[h].n) == n[h] for h in HS)
deaths = {h: int(pop.loc[h].deaths_by_h) for h in HS}


def ptxt(p):
    if p >= 0.995:
        return "P = 1.00"
    if p < 0.0001:
        return "P < 0.0001"
    if p < 0.001:
        return f"P = {p:.4f}"
    if p < 0.01:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


write_aggregates("f4h_joint_score",
                 ["horizon_months", "row", "n_patients", "deaths_by_horizon", "mean", "ci_low", "ci_high", "p_wilcoxon",
                  "p_holm_primary_family"],
                 [[h, row, n[h], deaths[h], r[(h, row)]["mean"], r[(h, row)].ci_low, r[(h, row)].ci_high,
                   r[(h, row)].p_wilcoxon, r[(h, row)].p_holm_primary_family]
                  for h in HS for row in [c[0] for c in CONTRASTS] + [f"arm:{k}" for k, *_ in ARMS]])

W, H = 180, 34
fig, axs = panel(W, H, ncols=4, sharey=True, sharex=True)
fig.set_layout_engine("none")
L, R, gap, B, T = 21.0, 1.0, 3.0, 8.5, 6.5
w = (W - L - R - 3 * gap) / 4
XL = (-0.032, 0.034)
ys = np.arange(len(HS))[::-1]
cols = {i: ARMS[i][2] for i in range(4)}
for i, (ax, (row, title, later)) in enumerate(zip(axs, CONTRASTS)):
    ax.set_position([(L + i * (w + gap)) / W, B / H, w / W, (H - B - T) / H])
    ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    for yi, h in zip(ys, HS):
        x = r[(h, row)]
        dot(ax, yi, x["mean"], x.ci_low, x.ci_high, cols[later])
        p = x.p_holm_primary_family
        ax.text(XL[1], yi, ptxt(p), fontsize=6, ha="right", va="center",
                fontweight="bold" if p < 0.05 else "normal")
    ax.set_xlim(*XL)
    ax.set_xticks([-0.03, -0.02, -0.01, 0, 0.01])
    ax.set_xticklabels(["", "−0.02", "", "0", ""])
    ax.set_ylim(-0.55, len(HS) - 0.1)
    ax.text(0.5, 1.0 + 1.0 / (H - B - T), title, transform=ax.transAxes, fontsize=6, ha="center", va="bottom",
            linespacing=1.1)
    if i:
        ax.tick_params(axis="y", left=False)
axs[0].set_yticks(ys)
axs[0].set_yticklabels([f"{h} months (n = {n[h]})" for h in HS])
fig.supxlabel("Difference in whole-course score (first named minus second; negative, first closer)",
              x=(L + (W - L - R) / 2) / W, y=0.6 / H, fontsize=7, ha="center", va="bottom")

rows = []
for row, title, _ in CONTRASTS:
    name = title.replace("\n", " ")
    rows.append(name + ": " + "; ".join(
        f"{h} months {r[(h, row)]['mean']:+.4f} ({r[(h, row)].ci_low:+.4f} to {r[(h, row)].ci_high:+.4f}), "
        f"Holm-adjusted {ptxt(r[(h, row)].p_holm_primary_family)}" for h in HS))
arm_means = "; ".join(f"{h} months " + ", ".join(f"{r[(h, f'arm:{k}')]['mean']:.3f}" for k, *_ in ARMS) for h in HS)
legend = (
    "The primary comparison of whole simulated courses, fixed after exploratory analyses and before the final "
    "recomputation (post hoc, not preregistered). For each patient, the ten generated futures of an arm were "
    "compared with the documented course by an energy score computed on 17 summaries of the trajectory over a fixed horizon "
    "(survival, time in each disease state, new responses and progressions, and treatment exposure), each scaled to the "
    "observed range, with the same fan size in every arm (equal-size convention); lower is better. The score summarizes "
    "courses; it is not a test of the full ordered path of events. Only patients with complete documentation up to the "
    "horizon are scored, so each horizon has its own selected population, and the four horizons are not comparable: "
    + "; ".join(f"{h} months, {n[h]} patients ({deaths[h]} deaths by the horizon)" for h in HS)
    + " (of 450 GENIE-BPC patients, 150 each with lung, breast and colorectal cancer). Columns, the four paired "
    "contrasts of the primary comparison between arms (the setting named first minus the setting named after \u201cvs\u201d; negative values mean the first, with more patient evidence, "
    "forecast the documented course better). Points, mean paired difference over patients, coloured by the later arm (light "
    "amber to dark wine = increasing patient evidence); lines, 95% bootstrap interval over patients; P, two-sided Wilcoxon "
    "signed-rank test, Holm-adjusted across the 16 contrasts of the primary family (bold, below 0.05). "
    + " ".join(r_ + "." for r_ in rows)
    + f" Mean score per arm (patient record, + care setting, + generic examples, + matched examples): {arm_means}. "
    "DeepSeek V4 Flash."
)
save_panel(fig, "f4h_joint_score", legend=legend,
           sources=[f"{F_REL}/joint_scores_scores_20261012.csv", f"{F_REL}/population_composition_scores_20261012.csv"],
           confirm=["Holm family: the legend says 16 contrasts (4 contrasts x 4 horizons), inferred from p_holm_primary_family; confirm against SCORING_SPEC.",
                    "Bootstrap interval described as 'over patients'; confirm resampling unit and count from the scoring spec.",
                    "Unadjusted Wilcoxon P values (p_wilcoxon) are saved in the aggregates CSV, not shown.",
                    "Holm P values of 1.00 are printed as P = 1.00 (Holm cap).",
                    "The 4-3 interval at 12 months excludes zero while Holm P = 0.06: interval and test differ in method; the panel shows both as saved."],
           message="Matched examples improve the whole-course score over the patient record at 12 and 24 months and over generic examples at 24 months; nothing detectable at 3 or 6 months.")
