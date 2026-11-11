"""Panel f3a_input_effects: which input moves which readout (heatmap).

Message: treatment and EGFR status had the largest effects on first-line control and survival;
prior history moved no readout beyond chance once every readout is measured from the decision.
Data: outputs/perturbation_matrix_decision_aligned/readout.json (effect_null_corrected; rows C, X, H, A;
readouts d_severity, d_burden, d_host, tox, pfs1, exit12, os). Built by
scripts/build_perturbation_matrix_decision_aligned.py (windows and outcomes start at the persisted
treatment-decision cutoff): 360 Sonnet futures in a 2 x 5 x 3 x 4 design.
Run: uv run python scripts/figures/panels/f3a_input_effects.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, INK, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

SRC = "outputs/perturbation_matrix_decision_aligned/readout.json"
d = json.loads((ROOT / SRC).read_text())
M = np.array(d["effect_null_corrected"], float)
assert d["inputs"] == ["C", "X", "H", "A"]
assert d["readouts"] == ["d_severity", "d_burden", "d_host", "tox", "pfs1", "exit12", "os"]
rows = ["EGFR status", "Current condition", "Prior history", "Treatment"]
cols = ["State-map\nPC1", "Disease\nextent", "ECOG\nstatus", "Toxicity\n(encoded)",
        "First-line\ncontrol time", "Death by\n12 months", "Overall\nsurvival"]

W, H = 112.0, 50.0
fig = new_fig(W, H)
ax = axes_mm(fig, 24, 18, 86, 24)
bound = float(np.nanmax(np.abs(M)))
im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
ax.set_xticks(range(7), cols)
ax.set_yticks(range(4), rows)
ax.xaxis.tick_top()
ax.tick_params(length=0, labelsize=6)
for t in ax.get_xticklabels():
    t.set_linespacing(1.0)
for sp in ax.spines.values():
    sp.set_visible(False)
for r, c in np.ndindex(M.shape):
    v = M[r, c]
    ax.text(c, r, f"{v:.2f}".replace("-", "−"), fontsize=6, ha="center", va="center",
            color="white" if abs(v) > 0.6 * bound else INK)
for e in np.arange(-0.5, 7, 1):
    ax.axvline(e, color="white", lw=0.5)
for e in (3.5,):
    ax.axvline(e, color="white", lw=2.5)
for e in np.arange(-0.5, 4, 1):
    ax.axhline(e, color="white", lw=0.5)
tr = ax.get_xaxis_transform()
for a, b, lab in ((-0.4, 3.4, "Change by 3 months after decision"), (3.6, 6.4, "Outcomes from decision")):
    ax.plot([a, b], [-0.07, -0.07], transform=tr, color=INK, lw=0.5, clip_on=False)
    ax.text((a + b) / 2, -0.12, lab, transform=tr, fontsize=6, ha="center", va="top")
cb = fig.colorbar(im, cax=axes_mm(fig, 37, 6.0, 60, 2.2), orientation="horizontal")
cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cb.set_ticklabels(["−3", "−2", "−1", "0", "1", "2", "3"])
cb.set_label("Effect size (null-corrected)", fontsize=6, labelpad=1)
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
save_panel(
    fig, "f3a_input_effects",
    legend=("Which input moves which output. Claude Sonnet generated 360 futures from 120 input combinations "
            "(EGFR status, 2 levels; current clinical condition, 5; prior history, 3: no prior record, or "
            "progression recorded 1 or 4 months after diagnosis followed by the treatment decision; treatment, 4; "
            "three futures each). Each cell is the spread of one readout across the levels of one input, with the "
            "other inputs fixed, in units of the variation between identical starting patients, minus the spread "
            "obtained when level labels are shuffled (400 permutations). Zero means no effect beyond chance; colour "
            "is symmetric around zero. Every readout starts at the treatment decision of each future (day 0, 32 or "
            "123 after diagnosis). Left block, change from the decision to 3 months later (last recorded state; "
            f"futures that died within 3 months excluded, {d['n_exited_by_3_months']} of 360): state-map PC1 (first "
            "axis of the frozen clinical state map; largest loadings hepatic disease, disease extent, symptoms, ECOG "
            "and toxicity; not a validated severity scale), number of involved anatomical classes, ECOG performance "
            "status, and highest active toxicity grade on the encoded scale. Right block: first-line progression-free "
            "time (active treatments only; the first line starts at the decision), death within 12 months of the "
            "decision and survival from the decision. The values describe the model, not patients."),
    sources=[SRC, "scripts/build_perturbation_matrix_decision_aligned.py"],
    confirm=["Decision-aligned readout: history x PC1 = " + f"{M[2, 0]:.2f}" + ".",
             "Treatment now moves early state change (PC1 " + f"{M[3, 0]:.2f}" + ", toxicity " + f"{M[3, 3]:.2f}"
             + ") as well as first-line control and survival.",
             "Column 'State-map PC1'; row 'Prior history'."],
    message="Treatment and EGFR status had the largest effects on first-line control and survival; prior history "
            "moved no readout beyond chance once measured from the decision.")
