"""Panel ed01c_embeddings: the same states in four coordinate systems (why PC1/PC2)."""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import CMAP, ROOT  # noqa: E402
from common import panel, save_panel  # noqa: E402

D = ROOT / "outputs" / "paper_state_embedding_bakeoff"
METHODS = {"pca": "PCA (the state map)", "umap": "UMAP", "phate": "PHATE", "diffusion": "Diffusion map"}
TUMOURS = {"PAAD": ("#0072B2", "Pancreatic"), "LUAD": ("#000000", "Lung adenocarcinoma"),
           "HGSOC": ("#009E73", "Ovarian (high-grade serous)")}
TUMOURS = {"PAAD": "#0B3C8C", "LUAD": "#D55E00", "HGSOC": "#E6A800"}
NAMES = {"PAAD": "Pancreatic adenocarcinoma", "LUAD": "Lung adenocarcinoma", "HGSOC": "High-grade serous ovarian"}
coords = pd.read_csv(D / "embedding_coordinates.csv.gz")
land = pd.read_csv(D / "landmark_states_and_pcs.csv.gz", usecols=["state_key", "pc1"])
rows = pd.read_csv(D / "target_tumour_trajectory_states.csv.gz",
                   usecols=["_gid", "tumor_oncotree_code", "snapshot_index", "state_key"])
rep = pd.read_csv(D / "representative_trajectories.csv")
rows = rows[rows["_gid"].isin(set(rep["_gid"]))]
tumour_of = rep.set_index("_gid")["tumor_oncotree_code"].to_dict()
norm = Normalize(vmin=float(land["pc1"].quantile(0.02)), vmax=float(land["pc1"].quantile(0.98)), clip=True)

fig, axs = panel(180, 52, 1, 4)
for ax, (m, name) in zip(axs, METHODS.items()):
    c = coords[coords["method"] == m]
    pts = land.merge(c, on="state_key", validate="one_to_one")
    sc = ax.scatter(pts["axis1"], pts["axis2"], c=pts["pc1"], cmap=CMAP, norm=norm, s=2.5, alpha=0.5,
                    linewidths=0, rasterized=True)
    paths = rows.merge(c, on="state_key", how="left", validate="many_to_one")
    for gid, g in paths.groupby("_gid"):
        g = g.sort_values("snapshot_index")
        col = TUMOURS[tumour_of[gid]]
        ax.plot(g["axis1"], g["axis2"], color=col, lw=1.1, zorder=5,
                path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
        ax.scatter(g["axis1"], g["axis2"], s=5, color=col, edgecolor="white", linewidth=0.25, zorder=6)
        ax.scatter(g["axis1"].iloc[0], g["axis2"].iloc[0], s=24, marker="o", color="white", edgecolor=col, linewidth=1.0, zorder=7)
        ax.scatter(g["axis1"].iloc[-1], g["axis2"].iloc[-1], s=22, marker="D", color=col, edgecolor="white", linewidth=0.4, zorder=7)
    ax.set_xlabel(f"{name} axis 1")
    ax.set_ylabel(f"{name} axis 2")
    ax.set_xticks([]); ax.set_yticks([])  # noqa: E702
cb = fig.colorbar(sc, ax=list(axs), location="bottom", fraction=0.07, pad=0.03, aspect=40, shrink=0.35, anchor=(0.05, 0.5))
cb.set_label("Position along state-map PC1")
cb.outline.set_linewidth(0.5)
handles = [Line2D([0], [0], color=TUMOURS[t], lw=1.2, label=NAMES[t]) for t in TUMOURS]
handles += [Line2D([0], [0], color="black", marker="o", mfc="white", ls="none", ms=4, label="First state"),
            Line2D([0], [0], color="black", marker="D", ls="none", ms=3.5, label="Last state")]
fig.legend(handles=handles, loc="lower right", ncols=3, frameon=False, bbox_to_anchor=(1.0, -0.005))
save_panel(
    fig, "ed01c_embeddings",
    legend=("Why a linear map. The same 6,000 generated clinical states are laid out four ways: the linear map used in "
            "the paper (PCA, first two components), UMAP, PHATE and a diffusion map. Dots, states, coloured by their "
            "position along PC1 of the state map. Lines, one representative generated trajectory for "
            "each of three cancers (pancreatic adenocarcinoma, lung adenocarcinoma, high-grade serous ovarian), chosen as "
            "the trajectory closest to the median on five display-only summaries; open circle, first state; diamond, "
            "last state. In the linear map, position has a fixed meaning (PC1 increases to the right) and trajectories "
            "move smoothly. In the nonlinear layouts the PC1 gradient is broken into islands and distances between "
            "islands carry no clinical meaning. Nonlinear axes have no units. This comparison justifies the choice of "
            "display and is not evidence that any projection predicts better. Source: generated atlas, six models."),
    sources=["outputs/paper_state_embedding_bakeoff/embedding_coordinates.csv.gz",
             "outputs/paper_state_embedding_bakeoff/landmark_states_and_pcs.csv.gz",
             "outputs/paper_state_embedding_bakeoff/target_tumour_trajectory_states.csv.gz",
             "outputs/paper_state_embedding_bakeoff/representative_trajectories.csv"],
    confirm=["Local-trust and path-rho numbers are not drawn; they are in embedding_diagnostics.csv.",
             "Tumour line colours (navy, vermilion, gold) avoid the treatment and model colours."],
    message="The same states in four layouts: only the linear map keeps a fixed clinical meaning of its axes.")
