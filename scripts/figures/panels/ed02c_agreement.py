"""Panel ed02c_agreement: agreement between models in how they organise the cancers."""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import ROOT  # noqa: E402
from common import MODEL_ORDER, INK, panel, save_panel  # noqa: E402

J = ROOT / "outputs/exploratory_native_atlas_geometry/readout.json"
r = json.loads(J.read_text())
KEY = {"Sonnet": "claude_sonnet5", "Gemini 3.1 L": "gemini31_lite", "Gemma 4": "gemma4", "GLM-5.2": "glm52_fp8",
       "Kimi K3": "kimi_k3_off", "Qwen 3.6": "qwen36_flash"}
idx = {KEY[m]: i for i, m in enumerate(MODEL_ORDER)}
rho = np.full((6, 6), np.nan)
for e in r["cross_model_topology"]["composite"]:
    i, j = idx[e["model_a"]], idx[e["model_b"]]
    rho[i, j] = rho[j, i] = e["spearman_rho"]
fig, ax = panel(88, 66)
im = ax.imshow(rho, cmap="YlGnBu", norm=Normalize(0, 0.6))
ax.set_xticks(range(6), MODEL_ORDER, rotation=90)
ax.set_yticks(range(6), MODEL_ORDER)
for i in range(6):
    for j in range(6):
        if i != j:
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if rho[i, j] > 0.38 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("Rank correlation of\ncancer-to-cancer distances")
cb.outline.set_visible(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
n = r["split_audit"]["n_matched_cancers"]
save_panel(
    fig, "ed02c_agreement",
    legend=("Do models agree on which cancers resemble each other? For each pair of models, the Spearman rank "
            f"correlation between the two models' tables of distances between cancers ({n} cancers present in all six "
            "models; distances measured on held-out generated paths over 24 months with a ruler combining time course, "
            "state occupancy, richness, event grammar and treatment). 1 would mean identical organisation; 0, none. "
            "All pairs are positive but moderate (0.26 to 0.55), so models share part of the organisation of cancers "
            "and differ in the rest. Descriptive, not a measure of clinical fidelity."),
    sources=["outputs/exploratory_native_atlas_geometry/readout.json"],
    confirm=["Number range 0.26 to 0.55 in the legend (readout.json)."],
    message="Models agree only moderately on how cancers relate to each other.")
