"""Extended Data (file ed08d_regimen_timing): new regimen before any NEW entry into progression, documented vs generated.

Quantity (four_arm_scores_v1.progression_to_regimen; aggregation four_arm_extract_v1 section 8): for each course, a
subsequent regimen starts after the landmark and before the first NEW post-landmark entry into a progressing state
(courses with no such entry count when a new regimen starts; progression already present at the landmark is not an
entry). The share is computed within each patient's fan, then averaged over patients (mean within-patient
fraction), so it is NOT a pooled share of futures.
Message: documented courses start a new regimen before any new progression in about 1 in 5 patients; generated
courses do so in about 7 to 8 in 10 per patient on average, in every arm, at 12 and 24 months. Descriptive only.
Data: timing_scores_20261012.csv, horizons 12 and 24 months, who in {real, four arms}:
fraction_regimen_precedes_any_progression, n_patients, n_courses.
Run: uv run python scripts/figures/panels/ed08d_regimen_timing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, F_REL, INK, PAL, final_table, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

t = final_table("timing_scores_20261012.csv").set_index(["horizon_months", "who"])
HS = [12, 24]
WHO = [("real", "Documented", PAL["observed"])] + [(k, lab, col) for k, lab, col, _ in ARMS]
r = {(h, w): t.loc[(h, w)] for h in HS for w, *_ in WHO}
for h in HS:
    assert len({int(r[(h, w)].n_patients) for w, *_ in WHO}) == 1
    assert int(r[(h, "real")].n_courses) == int(r[(h, "real")].n_patients)

write_aggregates("ed08d_regimen_timing",
                 ["horizon_months", "who", "n_patients", "n_courses", "fraction_regimen_precedes_any_progression",
                  "fraction_progression", "fraction_both"],
                 [[h, w, int(x.n_patients), int(x.n_courses), x.fraction_regimen_precedes_any_progression,
                   x.fraction_progression, x.fraction_both] for (h, w), x in r.items()])

W, H = 80, 56
fig, ax = panel(W, H)
fig.set_layout_engine("none")
L, R, B, T = 21.0, 3.0, 9.0, 1.0
ax.set_position([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
y, ticks, labels = 0.0, [], []
for h in HS:
    ax.text(-0.5, y, f"{h} months, {int(r[(h, 'real')].n_patients)} patients", fontsize=6, ha="left", va="center")
    ax.text(100, y, "Courses", fontsize=6, ha="right", va="center")
    y += 1
    for w, lab, col in WHO:
        x = r[(h, w)]
        v = 100 * x.fraction_regimen_precedes_any_progression
        ax.barh(y, v, height=0.72, color=col, edgecolor=INK, linewidth=0.5)
        ax.text(v + 1.2, y, f"{v:.0f}%", fontsize=6, ha="left", va="center")
        ax.text(100, y, f"{int(x.n_courses):,}", fontsize=6, ha="right", va="center")
        ticks.append(y)
        labels.append(lab)
        y += 1
    y += 0.3
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.set_ylim(y - 0.6, -0.6)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Courses with a new regimen before any new\nprogression (%; mean over patients)")


def txt(h):
    return ", ".join(f"{lab.lower() if w == 'real' else lab} {100 * r[(h, w)].fraction_regimen_precedes_any_progression:.1f}% "
                     f"({int(r[(h, w)].n_courses):,} courses)" for w, lab, _ in WHO)


legend = (
    "Order of regimen change and new progression, a descriptive diagnostic. For each course, whether a new systemic regimen "
    "started after the start of the recorded regimen and before any NEW entry into a progressing state (documented for the "
    "observed course; asserted by the model for generated futures, read on the same assessment calendar). Courses without "
    "any such entry count when a new regimen starts; progression already present at the start of the regimen is not an "
    "entry. The percentage is computed within each patient (the share of that patient's courses) and then averaged over "
    "patients, so every patient weighs the same; it is not a pooled share of futures. Bars and numbers, mean within-patient "
    "percentage; Courses, number of courses behind it (documented, black: one per patient; generated, arm colours as in "
    "Figure 4: up to ten futures per patient in each arm). Patients as in Figure 4 (complete documentation up to the "
    f"horizon). 12 months, {int(r[(12, 'real')].n_patients)} patients: {txt(12)}. 24 months, "
    f"{int(r[(24, 'real')].n_patients)} patients: {txt(24)}. Descriptive: no interval or test. A regimen change is not "
    "evidence of progression, nor of inappropriate treatment: regimens also change for toxicity, completion of planned "
    "treatment or patient choice, and the diagnostic does not assess the reason for a switch."
)
save_panel(fig, "ed08d_regimen_timing", legend=legend,
           sources=[f"{F_REL}/timing_scores_20261012.csv"],
           confirm=["Mean within-patient fraction; course counts shown separately; 'new entry into progression' excludes progression at the landmark.",
                    "No interval or test in the table for this quantity; shares are descriptive. Courses are futures nested in patients (n patients stated).",
                    "Patients at 12 and 24 months (233, 181) match the joint primary populations of the whole-course score.",
                    "Fractions with any progression and with both events are saved in the aggregates CSV, not shown."],
           message="On average per patient, about 1 in 5 documented courses and 7 to 8 in 10 generated courses start a new "
                   "regimen before any new progression, in every arm (descriptive).")
