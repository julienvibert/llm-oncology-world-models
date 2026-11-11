"""Panel ed02e_blocks: cancer specificity by component of the ruler, per model."""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import ROOT  # noqa: E402
from common import MODEL_ORDER, INK, panel, save_panel  # noqa: E402

r = json.loads((ROOT / "outputs/exploratory_native_atlas_geometry/readout.json").read_text())
KEY = {"Sonnet": "claude_sonnet5", "Gemini 3.1 L": "gemini31_lite", "Gemma 4": "gemma4", "GLM-5.2": "glm52_fp8",
       "Kimi K3": "kimi_k3_off", "Qwen 3.6": "qwen36_flash"}
BLOCKS = ["clock", "occupancy", "richness", "grammar", "treatment"]
LAB = ["Time\ncourse", "Time in\neach state", "Number of\nevents", "Event\nsequence", "Treatment"]
mat = np.array([[r["models"][KEY[m]]["blocks"][b]["specificity_ratio"] for b in BLOCKS] for m in MODEL_ORDER])
fig, ax = panel(98, 42)
im = ax.imshow(mat, cmap="YlGnBu", norm=Normalize(1.0, 2.0), aspect="auto")
ax.set_xticks(range(5), LAB)
ax.set_yticks(range(6), MODEL_ORDER)
for i in range(6):
    for j in range(5):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] > 1.58 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Between / within distance")
cb.outline.set_visible(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save_panel(
    fig, "ed02e_blocks",
    legend=("What drives the cancer specificity. The ratio of panel d (between-cancer over within-cancer distance; 1 "
            "means no cancer-specific structure) computed separately for each of the five components of the ruler: the "
            "time course of disease state, the share of time in each state, the number of events, the sequence of "
            "event types and the treatment received. Each cell is one model and one component; the value is printed in "
            "the cell. Descriptive; no interval is stored for the component values."),
    sources=["outputs/exploratory_native_atlas_geometry/readout.json"],
    confirm=["Component names are plain-word renames of clock, occupancy, richness, grammar and "
             "treatment."],
    message="Cancer specificity is highest in the time course of disease state in every model.")
