"""Panel si04a_input_matrix: full factorial matrix, which input changes which part of the course.

Data: outputs/perturbation_matrix_decision_aligned/readout.json (effect_null_corrected,
effect_raw; same file as f3a_input_effects; built by scripts/build_perturbation_matrix_decision_aligned.py). The cell outlined in black is examined in si04b and si04c.
Generated data only (360 Sonnet futures). Run: uv run python scripts/figures/panels/si04a_input_matrix.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import INK, ROOT, save_panel  # noqa: E402

SRC = "outputs/perturbation_matrix_decision_aligned/readout.json"
d = json.loads((ROOT / SRC).read_text())
M = np.array(d["effect_null_corrected"], float)
R = np.array(d["effect_raw"], float)
assert d["inputs"] == ["C", "X", "H", "A"] and d["readouts"][0] == "d_severity" and d["n_futures"] == 360
rows = ["EGFR status (2)", "Current condition (5)", "Growth before treatment (3)", "Treatment (4)"]
cols = ["State-map\nPC1", "Disease\nextent", "ECOG\nstatus", "Toxicity\n(encoded)",
        "First-line\ncontrol time", "Death by\n12 months", "Overall\nsurvival"]
fig = new_fig(180, 45)
ax = axes_mm(fig, 30, 13, 148, 24)
bound = float(np.nanmax(np.abs(M)))
im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
ax.set_xticks(range(7), cols)
ax.set_yticks(range(4), rows)
ax.xaxis.tick_top()
ax.tick_params(length=0, labelsize=6)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_linespacing(1.0)
for sp in ax.spines.values():
    sp.set_visible(False)
for r, c in np.ndindex(M.shape):
    col = "white" if abs(M[r, c]) > 0.6 * bound else INK
    ax.text(c, r, f"{M[r, c]:.1f}".replace("-", "−"), fontsize=6, ha="center", va="center", color=col)
for e in np.arange(-0.5, 7, 1):
    ax.axvline(e, color="white", lw=0.5)
ax.axvline(3.5, color="white", lw=2.5)
for e in np.arange(-0.5, 4, 1):
    ax.axhline(e, color="white", lw=0.5)
tr = ax.get_xaxis_transform()
for a, b, lab in ((-0.4, 3.4, "Change from the treatment decision to 3 months later"), (3.6, 6.4, "Outcomes from the decision")):
    ax.plot([a, b], [-0.02, -0.02], transform=tr, color=INK, lw=0.5, clip_on=False)
    ax.text((a + b) / 2, -0.06, lab, transform=tr, fontsize=6, ha="center", va="top")
cb = fig.colorbar(im, cax=axes_mm(fig, 112, 3.6, 40, 1.6), orientation="horizontal")
cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cb.set_ticklabels(["−3", "−2", "−1", "0", "1", "2", "3"])
cb.ax.set_ylabel("Effect beyond chance", fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
save_panel(
    fig, "si04a_input_matrix",
    legend=("Which input changes which part of the course, in the full factorial experiment. Claude Sonnet generated "
            "360 futures from 120 input combinations (EGFR status, 2 levels; current clinical condition, 5 starting "
            "states from low-burden fit to high-burden frail; prior history, 3 levels: no prior record, or progression "
            "recorded 1 or 4 months after diagnosis followed by the treatment decision; assigned treatment, 4: "
            "osimertinib, chemotherapy, pembrolizumab, supportive care; three futures each). Each cell is the spread "
            "of one readout across the levels of one input, with the other three held fixed. Large number, the spread "
            "in units of the variation between identical starting patients, minus the spread obtained when the level "
            "labels are shuffled (400 permutations); zero means no effect beyond chance. Raw values in the source data. Every readout starts at the treatment decision of each future (day 0, 32 or 123 after "
            "diagnosis). Left block: change from the decision to 3 months later (last recorded state; "
            f"{d['n_exited_by_3_months']} futures that died within 3 months excluded) in state-map PC1 (first axis of "
            "the frozen state map; largest loadings hepatic disease, disease extent, symptoms, ECOG and toxicity; not "
            "a validated scale), number of involved site classes, ECOG performance status and worst active toxicity "
            "grade (encoded scale). Right block: first-line progression-free time (active treatments; the line starts "
            "at the decision), death within 12 months of the decision and survival from the decision. The values describe the model, "
            "not patients; futures in a cell are not independent patients."),
    sources=[SRC, "scripts/build_perturbation_matrix_decision_aligned.py"],
    confirm=["Same matrix as f3a_input_effects plus the raw values.",
             "Decision-aligned readout: history x PC1 raw "
             + f"{R[2, 0]:.2f}" + ", corrected " + f"{M[2, 0]:.2f}" + "."],
    message="Treatment and EGFR status move first-line control and survival; treatment also moves early state change; "
            "prior history moves no readout beyond chance.")
