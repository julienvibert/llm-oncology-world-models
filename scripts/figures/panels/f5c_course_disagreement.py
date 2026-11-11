"""Panel f5c_course_disagreement: disagreement over evaluable time by horizon, four arms.

Quantity (four_arm_scores_v1.labelled + disagreement): evaluable time = documented living time on the registry
calendar plus the absorbing post-death state (death is a state); per patient, the mean over that patient's futures;
then the mean over patients (not pooled person-time).

Message: matched examples bring the generated disease state slightly closer to the documented one, but at
12 months the generated state still differs for more than half of the documented time in every arm.
Data: course_diagnostics_scores_20261012.csv (final four-arm tables), quantity
disagreement (primary loss), convention primary, cancer all, rows arm:<arm> (mean over patients, 95% CI).
unresolved_fraction (same convention) is quoted in the legend only.
Run: uv run python scripts/figures/panels/f5c_course_disagreement.py
"""
import sys
from pathlib import Path

from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, F_REL, GREY, INK, final_table, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

c = final_table("course_diagnostics_scores_20261012.csv")
c = c[(c.convention == "primary") & (c.cancer == "all")]
HS = [3, 6, 12, 24]
r = {(q, h, k): c[(c.quantity == q) & (c.horizon_months == h) & (c.row == f"arm:{k}")].iloc[0]
     for q in ("disagreement", "unresolved_fraction", "occupancy_tv") for h in HS for k, *_ in ARMS}
n = {h: int(r[("disagreement", h, "patient_only")].n) for h in HS}
assert [n[h] for h in HS] == [386, 412, 426, 437]
assert all(int(x.n) == n[h] for (q, h, k), x in r.items())
pop = final_table("population_composition_scores_20261012.csv")
pop = pop[pop.population == "course_four_arm:primary"].set_index("horizon_months")
assert all(int(pop.loc[h].n) == n[h] for h in HS)
deaths = {h: int(pop.loc[h].deaths_by_h) for h in HS}

write_aggregates("f5c_course_disagreement",
                 ["horizon_months", "arm", "n_patients", "disagreement_mean", "ci_low", "ci_high",
                  "unresolved_fraction_mean", "occupancy_tv_mean"],
                 [[h, k, n[h], r[("disagreement", h, k)]["mean"], r[("disagreement", h, k)].ci_low,
                   r[("disagreement", h, k)].ci_high, r[("unresolved_fraction", h, k)]["mean"],
                   r[("occupancy_tv", h, k)]["mean"]] for h in HS for k, *_ in ARMS])

W, H = 90, 34
fig, ax = panel(W, H)
fig.set_layout_engine("none")
L, R, B, T = 10.0, 23.5, 7.0, 1.5
ax.set_position([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
OFF = [-0.27, -0.09, 0.09, 0.27]
ax.axhline(50, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.text(3.45, 48.8, "Half of the evaluable time", fontsize=6, ha="right", va="top", color=INK)
for xi, h in enumerate(HS):
    for (k, _, col, mk), off in zip(ARMS, OFF):
        x = r[("disagreement", h, k)]
        # vertical point-interval (dot() draws horizontal intervals; swap axes by hand)
        ax.plot([xi + off] * 2, [100 * x.ci_low, 100 * x.ci_high], color=INK, lw=0.9, zorder=4,
                solid_capstyle="butt")
        ax.plot(xi + off, 100 * x["mean"], mk, ms=3.2, mfc=col, mec=INK, mew=0.5, zorder=3)
ax.set_xlim(-0.5, 3.5)
ax.set_xticks(range(len(HS)))
ax.set_xticklabels([f"{h} months\nn = {n[h]}" for h in HS])
ax.set_ylim(0, 75)
ax.set_yticks([0, 25, 50, 75])
ax.set_ylabel("Time in disagreement\n(% of evaluable time)")
ax.text(3.45, 1.5, "0 = same state throughout", fontsize=6, ha="right", va="bottom")
hs = [Line2D([], [], marker=mk, ls="none", ms=3.2, mfc=col, mec=INK, mew=0.5, label=lab) for _, lab, col, mk in ARMS]
ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6, frameon=False, handletextpad=0.2,
          borderaxespad=0.2, labelspacing=0.35)

arms_txt = "; ".join(
    f"{h} months, " + ", ".join(f"{100 * r[('disagreement', h, k)]['mean']:.1f}% "
                                 f"({100 * r[('disagreement', h, k)].ci_low:.1f} to {100 * r[('disagreement', h, k)].ci_high:.1f})"
                                 for k, *_ in ARMS) for h in HS)
unres = [100 * r[("unresolved_fraction", h, k)]["mean"] for h in HS for k, *_ in ARMS]
legend = (
    "How often the generated course is in a different state from the recorded one, by horizon and arm. For each patient, "
    "the evaluable time runs from the start of the recorded regimen to the horizon: the follow-up with a disease state "
    "recorded in GENIE-BPC (read on the registry assessment calendar), plus any time after a recorded death. Death is "
    "treated as a state of its own (absorbing; not a disease state): after a recorded death, a generated future that is "
    "still alive disagrees, and a generated death before a recorded one disagrees until the recorded death. Disagreement "
    "is the share of that evaluable time in which a generated future and the recorded course are in different states, "
    "first averaged over the patient's futures, then over patients, so every patient weighs the same (not pooled "
    "person-time; 0 = the same state throughout; 100% = a different state throughout). Generated time without an "
    f"asserted state counts as disagreement ({min(unres):.1f}% to {max(unres):.1f}% of evaluable time, by arm and horizon). "
    "Points, mean over patients, coloured by arm (light amber to deep wine = increasing patient evidence: patient record "
    "only; + care setting; + generic examples; + matched examples); lines, 95% bootstrap interval over patients (2,000 "
    "paired resamples, stratified by cancer); dashed line, half of the evaluable time. n, patients with evaluable time at "
    "the horizon (450 patients, 150 each with lung, breast and colorectal cancer; "
    + ", ".join(f"{deaths[h]} deaths by {h} months" for h in HS)
    + "); the populations differ between horizons. Mean (95% interval), arms in the order above: " + arms_txt
    + ". Paired differences between arms, panel d. DeepSeek V4 Flash."
)
save_panel(fig, "f5c_course_disagreement", legend=legend,
           sources=[f"{F_REL}/course_diagnostics_scores_20261012.csv",
                    f"{F_REL}/population_composition_scores_20261012.csv"],
           confirm=["Deaths by horizon from population_composition (course_four_arm:primary); post-death time is part of the evaluable window, death an absorbing state (four_arm_scores_v1.labelled).",
                    "Occupancy total variation (order ignored) is saved in the aggregates CSV, not shown."],
           message="Matched examples bring generated courses slightly closer to the recorded ones, but at 12 months they still "
                   "disagree for more than half of the evaluable time in every arm (mean over patients).")
