"""Panel ed02b_cancer_map: where six cancer types sit on the state map (all models pooled)."""
import sys
from pathlib import Path

from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import AX1, AX2, projection  # noqa: E402
from common import panel, save_panel  # noqa: E402

CANC = {"LUAD": ("Lung adenocarcinoma", "#D55E00"), "PAAD": ("Pancreatic adenocarcinoma", "#117733"),
        "HGSOC": ("High-grade serous ovarian", "#E6C700"), "DLBCLNOS": ("Diffuse large B-cell lymphoma", "#56B4E9"),
        "AML": ("Acute myeloid leukaemia", "#6B3FA0"), "GB": ("Glioblastoma", "#CC79A7")}
p = projection()
bg = p.sample(8000, random_state=1)
fig, ax = panel(88, 68)
ax.scatter(bg.pc1, bg.pc2, s=1.0, c="#E3E7EA", linewidths=0, rasterized=True)
for code, (name, col) in CANC.items():
    g = p[p.tumor_oncotree_code == code]
    ax.scatter(g.pc1, g.pc2, s=1.6, c=col, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter([g.pc1.mean()], [g.pc2.mean()], s=26, marker="o", c=col, edgecolor="black", linewidth=0.8, zorder=5)
ax.set_xlabel(AX1)
ax.set_ylabel(AX2)
ax.legend([Line2D([], [], marker="o", ls="", color=c, ms=4) for _, c in CANC.values()],
          [f"{n} (n = {int((p.tumor_oncotree_code == k).sum()):,})" for k, (n, _) in CANC.items()],
          loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=2, frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
save_panel(
    fig, "ed02b_cancer_map",
    legend=("How cancer types are organised on the state map. Each coloured dot is one generated living state of the "
            "named cancer (all six models pooled; n states in the key); black-edged circles, the mean position of each "
            "cancer; pale dots, 8,000 states from all cancers for reference. Cancers are distinguished by where they sit: "
            "marrow and lymphoid cancers towards the top (high PC2, driven by marrow disease and marrow reserve), solid tumours with "
            "liver and thoracic disease towards the right (high PC1). Axes as in Extended Data Fig. 1a. Six of 148 cancer types are shown, "
            "chosen to span solid, haematological and central nervous system disease."),
    sources=["outputs/paper_state_map/state_projection.csv.gz"],
    confirm=["The six cancers were chosen by hand to span disease groups, not by a rule.",
             "Legend sentence on where cancers sit is a reading of the figure."],
    message="Cancer types occupy different regions of the same state map.")
