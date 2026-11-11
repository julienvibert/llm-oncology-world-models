"""Extended Data (file ed08c_sensitivity): whole-course contrasts under two sensitivity analyses (selected sets).

Message: on the same patients, first-attempt futures give the same matched-arm contrasts as the full pipeline,
and the primary and sensitivity readings of the documented course agree on their common set.
Data: sensitivity_scores_20261012.csv, analysis first_attempt vs full_pipeline_same_patients (same patients), and
convention_common_set (family joint_common, convention primary vs sensitivity); rows contrast:4-1 and contrast:4-3
(n, mean, ci_low, ci_high, p_wilcoxon unadjusted).
Run: uv run python scripts/figures/panels/ed08c_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import F_REL, INK, PAL, dot, final_table, ptxt, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

s = final_table("sensitivity_scores_20261012.csv")
s = s[s.row.astype(str).str.startswith("contrast:")]
HS = [3, 6, 12, 24]
CON = [("contrast:4-1", "o"), ("contrast:4-3", "s")]
# (column title, [(selector, key label, filled)])
SETS = [("Retries: first-attempt futures\nvs with retries (same patients)",
         [(("full_pipeline_same_patients", "primary"), "With retries", True),
          (("first_attempt", "primary"), "First attempt", False)]),
        ("Observation gaps: last state kept\nvs unknown after 4 months",
         [(("convention_common_set", "primary"), "Kept", True),
          (("convention_common_set", "sensitivity"), "Unknown after 4 months", False)])]


def get(an, conv, h, c):
    q = s[(s.analysis == an) & (s.convention == conv) & (s.horizon_months == h) & (s.row == c)]
    assert len(q) == 1, (an, conv, h, c)
    return q.iloc[0]


r = {(an, conv, h, c): get(an, conv, h, c) for _, vs in SETS for (an, conv), *_ in vs for h in HS for c, _ in CON}
n = {}
for ci, (_, vs) in enumerate(SETS):
    for h in HS:
        ns = {int(r[(an, conv, h, c)].n) for (an, conv), *_ in vs for c, _ in CON}
        assert len(ns) == 1, ns  # each comparison is on one set of patients
        n[(ci, h)] = ns.pop()
pop = final_table("population_composition_scores_20261012.csv").set_index(["population", "horizon_months"])
assert all(n[(0, h)] == int(pop.loc[("joint:first_attempt", h)].n) for h in HS)
assert all(n[(1, h)] == int(pop.loc[("joint:both_conventions", h)].n) for h in HS)

write_aggregates("ed08c_sensitivity",
                 ["analysis", "convention", "horizon_months", "row", "n_patients", "mean", "ci_low", "ci_high",
                  "p_wilcoxon_unadjusted"],
                 [[an, conv, h, c, int(x.n), x["mean"], x.ci_low, x.ci_high, x.p_wilcoxon]
                  for (an, conv, h, c), x in r.items()])

W, H = 100, 60
fig, axs = panel(W, H, ncols=2, sharex=True)
fig.set_layout_engine("none")
LB, PW, GAP, B, T = 13.0, 35.5, 3.0, 8.5, 15.0
ys = np.arange(len(HS))[::-1]
OFF = {("contrast:4-1", True): 0.30, ("contrast:4-1", False): 0.12,
       ("contrast:4-3", True): -0.12, ("contrast:4-3", False): -0.30}
XL = (-0.045, 0.015)
col = PAL["arm_matched"]
for i, (ax, (title, vs)) in enumerate(zip(axs, SETS)):
    x0 = LB + i * (LB + PW + GAP)
    ax.set_position([x0 / W, B / H, PW / W, (H - B - T) / H])
    ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    for yi, h in zip(ys, HS):
        for (an, conv), _, filled in vs:
            for c, mk in CON:
                x = r[(an, conv, h, c)]
                dot(ax, yi + OFF[(c, filled)], x["mean"], x.ci_low, x.ci_high, col if filled else "white", mk=mk, ms=3.8)
    ax.set_xlim(*XL)
    ax.set_xticks([-0.04, -0.02, 0])
    ax.set_xticklabels(["−0.04", "−0.02", "0"])
    ax.set_ylim(-0.55, len(HS) - 0.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{h} months\nn = {n[(i, h)]}" for h in HS])
    ax.text(0.5, 1.0 + 4.6 / (H - B - T), title, transform=ax.transAxes, fontsize=6, ha="center", va="bottom",
            linespacing=1.1)
    # key: fill = which version of the analysis
    for k, (_, lab, filled) in enumerate(vs):
        xk = (x0 + 1.0 + k * (17.0 if i == 0 else 9.0)) / W
        yk = (H - T + 1.7) / H
        fig.add_artist(Line2D([xk], [yk], marker="o", ls="none", ms=3.8, mfc=col if filled else "white", mec=INK,
                              mew=0.6, transform=fig.transFigure))
        fig.text(xk + 1.5 / W, yk, lab, fontsize=6, ha="left", va="center")
# key: shape = which contrast (restated here, as in panel a)
for k, (mk, lab) in enumerate((("o", "Matched vs patient record"), ("s", "Matched vs generic examples"))):
    xk, yk = (LB + 1.0 + k * 34.0) / W, (H - 1.8) / H
    fig.add_artist(Line2D([xk], [yk], marker=mk, ls="none", ms=3.8, mfc=col, mec=INK, mew=0.6,
                          transform=fig.transFigure))
    fig.text(xk + 1.5 / W, yk, lab, fontsize=6, ha="left", va="center")
fig.supxlabel("Difference in whole-course score (negative = matched closer)", x=0.5, y=0.4 / H, fontsize=7,
              ha="center", va="bottom")


def txt(an, conv, c):
    return "; ".join(f"{h} months {r[(an, conv, h, c)]['mean']:+.4f} ({r[(an, conv, h, c)].ci_low:+.4f} to "
                     f"{r[(an, conv, h, c)].ci_high:+.4f}), {ptxt(r[(an, conv, h, c)].p_wilcoxon)}" for h in HS)


legend = (
    "Whole-course contrasts (equal-size convention) under two robustness checks of analytical choices. Shape, the contrast (key above the "
    "panels): circles, matched examples minus patient record; squares, matched minus generic examples. Fill, the version "
    "of the analysis (key above each panel). Both analyses use selected sets of patients, "
    "so their n are smaller than in Figure 4 and the estimates are not comparable with it. Left, retries: only the futures accepted at the "
    "first generation attempt (open; still accepted responses, not every first response) against the futures accepted after up to three attempts (filled), on the same "
    "patients: those with at least seven futures valid at the first attempt in every arm and complete documentation "
    f"(n = {', '.join(str(n[(0, h)]) for h in HS)} at 3, 6, 12 and 24 months). Right, observation gaps: the documented "
    "course read with the last recorded state kept until the next update (primary reading; filled) against assessments that become unknown after four "
    "months, with conflicts read conservatively (open), on the patients scored under both "
    f"(n = {', '.join(str(n[(1, h)]) for h in HS)}); on this common set the mean paired differences are identical under the "
    "two readings and only the bootstrap intervals differ. Points, mean paired difference over patients; lines, 95% bootstrap interval over patients (2,000 paired "
    "resamples, stratified by cancer); unadjusted two-sided Wilcoxon P in the text below, not shown in the panel. "
    f"First attempt: matched vs patient record, {txt('first_attempt', 'primary', 'contrast:4-1')}; matched vs generic, "
    f"{txt('first_attempt', 'primary', 'contrast:4-3')}. With retries, same patients: matched vs patient record, "
    f"{txt('full_pipeline_same_patients', 'primary', 'contrast:4-1')}; matched vs generic, "
    f"{txt('full_pipeline_same_patients', 'primary', 'contrast:4-3')}. Common set: matched vs patient record, "
    f"{txt('convention_common_set', 'primary', 'contrast:4-1')}; matched vs generic, "
    f"{txt('convention_common_set', 'primary', 'contrast:4-3')}."
)
same = all(abs(r[("convention_common_set", "primary", h, c)]["mean"] - r[("convention_common_set", "sensitivity", h, c)]["mean"]) < 1e-12
           for h in HS for c, _ in CON)
assert same, "legend states identical means on the common set"
save_panel(fig, "ed08c_sensitivity", legend=legend,
           sources=[f"{F_REL}/sensitivity_scores_20261012.csv", f"{F_REL}/population_composition_scores_20261012.csv"],
           confirm=["On the common set the primary and sensitivity means (and Wilcoxon P) are identical in the table; only the CIs differ (same documented summaries for patients scorable under both readings).",
                                        "P values are in the legend only (the panel shows agreement between analyses, not tests).",
                    "Contrasts 2-1 and 3-2 are saved in the table but not shown."],
           message="First-attempt futures and the full pipeline give similar matched-arm contrasts on the same patients; the "
                   "two readings agree on their common set. Both are selected sets.")
