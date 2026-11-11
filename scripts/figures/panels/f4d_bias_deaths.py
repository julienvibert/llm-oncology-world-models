"""Panel f4d_bias_deaths: survival overestimated among observed deaths: examples shrink the excess.

Message: among the 260 patients whose death was observed, predicted survival exceeded observed
survival by 12.19 months (9.18 to 16.24) with the patient record alone and 2.74 (0.58 to 4.85) with matched examples.
Data: deep_results.json, decomposition["survival (died only)"]. Intervals as saved.
Run: uv run python scripts/figures/panels/f4d_bias_deaths.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, S, INK, dot, fix_axes, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

dec = json.loads((S / "deep_results.json").read_text())["decomposition"]["survival (died only)"]
write_aggregates("f4d_bias_deaths", ["arm", "n_patients_died", "mean_predicted_minus_observed_months", "lo", "hi"],
                 [[k, dec[k]["n"], *dec[k]["bias"]] for k, *_ in ARMS])

fig, ax = panel(66, 46)  # own row labels
y = np.arange(len(ARMS))[::-1]
ax.axvline(0, color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
for yi, (k, lab, col, _) in zip(y, ARMS):
    b, lo, hi = dec[k]["bias"]
    assert dec[k]["n"] == 260
    dot(ax, yi, b, lo, hi, col)
    ax.text(hi + 0.6, yi, f"{b:.1f} ({lo:.1f} to {hi:.1f})", va="center", ha="left", fontsize=6)
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab, *_ in ARMS])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-1.5, 36)
ax.set_xticks([0, 10, 20, 30])
ax.set_ylim(-0.6, len(ARMS) + 0.6)
ax.text(1, len(ARMS), "model predicted longer survival →", fontsize=6, va="center", ha="left")
ax.set_xlabel("Predicted minus observed\nsurvival (months), 260 deaths")
ax.spines["left"].set_visible(False)
fix_axes(fig, ax, left=26.0, right=1.5, bottom=10.5, top=1.0)

rows = "; ".join(f"{lab} {dec[k]['bias'][0]:.2f} ({dec[k]['bias'][1]:.2f} to {dec[k]['bias'][2]:.2f})" for k, lab, *_ in ARMS)
legend = (
    "Survival overestimated among patients who died, and examples reduced the excess. Mean difference between predicted and "
    "observed survival time (months) in the 260 patients whose death was observed (of 450 GENIE-BPC patients with lung, "
    "breast or colorectal cancer); positive values mean the model predicted longer survival than the patient had. Each "
    "patient's prediction is the mean of up to ten generated futures. Rows as in the panel to the left (light amber to dark wine = "
    f"increasing patient evidence). Points, mean difference; lines, 95% bootstrap interval over patients. Mean (95% interval): {rows}. "
    "Dashed line, no difference. The subset is selected on death and is not a population-wide estimate."
)
save_panel(fig, "f4d_bias_deaths", legend=legend, sources=["outputs/exploratory_four_arm_deep/deep_results.json"],
           confirm=["Numbers match the manuscript (12.19 [9.18, 16.24]; 2.74 [0.58, 4.85]; setting 9.01, generic 3.69).",
                    "Interval is the saved patient-level bootstrap; resampling count not recorded.",
                    "Row labels are omitted and shared with the panel on the left in the assembled figure; the standalone panel is unlabelled by design."],
           message="Among 260 observed deaths, predicted survival exceeded observed by 12.19 months (patient record) vs 2.74 (matched examples).")
