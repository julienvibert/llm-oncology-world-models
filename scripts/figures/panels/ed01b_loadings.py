"""Panel ed01b_loadings: which findings build PC1 and PC2, in clinical words."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import D, HOST, TUMOUR  # noqa: E402
from common import panel, save_panel  # noqa: E402

f = pd.read_csv(D / "feature_loadings.csv")
tum = f[f.block == "tumour"].sort_values("pc1_coefficient", ascending=True)
host = f[f.block == "host"].sort_values("pc1_coefficient", ascending=True)
f = pd.concat([host, tum]).reset_index(drop=True)  # plotted bottom to top: host below, tumour above
col = [TUMOUR if b == "tumour" else HOST for b in f.block]
y = np.arange(len(f)) + (f.block == "tumour") * 0.8
fig, axs = panel(92, 74, 1, 2, sharey=True)
for ax, c, name in ((axs[0], "pc1_coefficient", "PC1 weight"),
                    (axs[1], "pc2_coefficient", "PC2 weight")):
    ax.barh(y, f[c], 0.75, color=col)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(name)
    ax.set_xlim(-0.75, 0.75)
axs[0].set_xlim(-1.2, 0.75)
axs[0].set_yticks(y, f["label"])
axs[0].tick_params(axis="y", length=0)
axs[1].spines["left"].set_visible(False)
axs[1].tick_params(axis="y", length=0)
axs[0].legend(handles=[Patch(color=TUMOUR, label="Tumour finding"), Patch(color=HOST, label="Host finding")],
              loc="lower left", bbox_to_anchor=(0.0, 0.3), frameon=False, borderaxespad=0.1, handlelength=1.0, fontsize=6)
save_panel(
    fig, "ed01b_loadings",
    legend=("What builds the two axes. Each bar is the weight of one of the 20 clinical findings in a composite score, "
            "in encoded units (every finding scaled to a fixed 0 to 1 clinical range, then centred). Purple, tumour "
            "findings; orange, host findings. PC1 (left) is dominated by liver and lung disease, disease extent, symptoms "
            "and performance status, so a higher PC1 means more of these findings. PC2 (right) mainly contrasts marrow "
            "disease and marrow reserve (positive) against liver, thoracic, other anatomical and peritoneal disease (negative), "
            "plus host terms (symptoms, ECOG positive). A higher PC2 can therefore mean less lung or liver disease; it is not a host-health score. Both axes combine tumour and host findings. "
            "Source: 105,839 living states from 4,434 generated trajectories, six models, equal weight per model, "
            "trajectory and snapshot. AE, adverse event."),
    sources=["outputs/paper_state_map/feature_loadings.csv"],
    confirm=["Legend sentences describing which findings dominate each axis were written "
             "from feature_loadings.csv."],
    message="PC1 is mainly liver and lung disease, extent, symptoms and ECOG; PC2 contrasts marrow disease against hepatic, thoracic and other disease, plus host terms.")
