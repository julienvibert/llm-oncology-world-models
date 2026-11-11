"""Panels ed06c_worst_cell and ed06d_short_arrows: when the average arrow of a map cell misleads.

Recomputes the field-aggregation audit (importing the audit's functions) and checks the medians against outputs/exploratory_field_aggregation_audit/readout.json.
Note: this audit uses its own tumour x host indices (the 'prespecified plane'), not the state map; panels are named accordingly.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, save_panel, INK, GREY  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
import build_field_aggregation_audit as A  # noqa: E402

plt.switch_backend("Agg")
P = "prespecified tumour × host"
s, z = A.build()
d = A.displacements(s, z)
xc, yc = A.PLANES[P]
d["cell"] = A.cells(d, xc, yc)
canc = A.cancellation(d, xc, yc, A.FCOL["tumour family"])
saved = json.loads((ROOT / "outputs/exploratory_field_aggregation_audit/readout.json").read_text())
assert abs(float(canc["R"].median()) - saved["cancellation"][P]["tumour family"]["median_R"]) < 1e-6
SRC = ["scripts/explore/build_field_aggregation_audit.py", "outputs/exploratory_field_aggregation_audit/readout.json"]
CONF = ["Recomputed by importing the audit's functions; median R for tumour-family splits asserted equal to readout.json.",
        "This audit uses its own tumour x host indices (3-month horizon, 7 x 7 grid), named as such on the axes and in the legends; it is not the frozen state map of panels a and d."]
FAMCOL = {"Lung": "#D55E00", "Skin / melanoma": "#8C6D31", "Sarcoma": "#882255", "Gastrointestinal": "#009E73",
          "Genitourinary": "#E69F00", "Head & neck": "#332288", "Breast": "#CC79A7", "Gynaecological": "#56B4E9"}

# ------------------------------------------------------------------ c: worst cell
worst = canc.loc[canc["R"].idxmin()]
W, H = 82.0, 44.0
fig = new_fig(W, H)
ax = axes_mm(fig, 14, 11, 64, 30)
OFF = {"Lung": (3, -1, "left", "center"), "Skin / melanoma": (-2, 4, "left", "bottom"), "Sarcoma": (3, 2, "left", "bottom"),
       "Gastrointestinal": (2, -9, "left", "top"), "Genitourinary": (3, 2, "left", "bottom"), "Head & neck": (4, -2, "left", "center")}
pts = [[0, 0], [worst["pooled_x"], worst["pooled_y"]]]
for k, vx, vy, n in worst["subs"]:
    col = FAMCOL.get(k, GREY)
    ax.annotate("", xy=(vx, vy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=6), zorder=3)
    ox, oy, ha, va = OFF.get(k, (3, 3, "left", "bottom"))
    ax.annotate(f"{k} ({n})", (vx, vy), fontsize=6, textcoords="offset points", xytext=(ox, oy), ha=ha, va=va, color=INK)
    pts.append([vx, vy])
ax.annotate("", xy=(worst["pooled_x"], worst["pooled_y"]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, mutation_scale=8), zorder=4)
ax.scatter([0], [0], s=14, facecolor="white", edgecolor="black", lw=0.8, zorder=6)
pts = np.array(pts)
ax.plot([pts[:, 0].min() - 0.05, pts[:, 0].max() + 0.2], [0, 0], color="#D8DEE6", lw=0.5, zorder=0)
ax.plot([0, 0], [pts[:, 1].min() - 0.03, pts[:, 1].max() + 0.05], color="#D8DEE6", lw=0.5, zorder=0)
ax.set_xlim(pts[:, 0].min() - 0.05, pts[:, 0].max() + 0.2)
ax.set_ylim(pts[:, 1].min() - 0.03, pts[:, 1].max() + 0.11)
ax.set_xlabel("Change in tumor index over 3 months", fontsize=6, labelpad=2)
ax.set_ylabel("Change in host index\nover 3 months", fontsize=6, labelpad=2)
ax.legend(handles=[Line2D([0], [0], color="black", lw=2.2, label="Average of all patients\nin the cell")],
          loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6, handlelength=1.6, borderaxespad=0.1)
save_panel(
    fig, "ed06c_worst_cell",
    legend=("An average arrow that describes nobody. One cell of the map (the cell with the most opposed subgroups; "
            f"{int(worst['n'])} generated visits) in which patients of different tumour families move in different directions. "
            "Each coloured arrow is the average 3-month movement of one tumour family in the cell (number of visits in brackets); "
            "the thick black arrow is the average of all of them, which is the arrow a pooled field would draw. The black arrow "
            f"is {100 * (1 - worst['R']):.0f}% shorter than the average length of the subgroup arrows, the length it would "
            f"have if all pointed the same way (cancellation ratio R = {worst['R']:.2f}); the subgroup directions span {worst['max_angle']:.0f} degrees. Axes "
            "are the movement of the tumour index and of the host index of the audit's own tumour x host plane (these indices are not the state-map PC1 and PC2); origin is the cell's start. "
            "Units are index units of the generated atlas (six models). Visits from the same trajectory are not independent. "
            "Where subgroups disagree, a pooled arrow should be hidden or shown with its support."),
    sources=SRC, confirm=CONF, message="Pooled arrows can average over opposing subgroup flows.")

# ------------------------------------------------------------------ d: short arrows disagree
cc = canc.copy()
cc["mag"] = np.hypot(cc["pooled_x"], cc["pooled_y"])
med = float(cc["mag"].median())
W, H = 82.0, 44.0
fig = new_fig(W, H)
ax = axes_mm(fig, 14, 11, 52, 30)
sc = ax.scatter(cc["mag"], cc["max_angle"], s=3 + cc["n"] / 40, c=cc["R"], cmap="viridis", vmin=0.5, vmax=1.0,
                edgecolor="black", lw=0.3, zorder=3)
ax.plot([0, 0.425], [90, 90], color="#C0392B", ls="--", lw=0.8, zorder=2)
ax.plot([med, med], [-8, 185], color="black", ls=":", lw=0.8, zorder=2)
ax.text(0.42, 93, "90°", fontsize=6, ha="right", va="bottom", color="#C0392B")
ax.text(med + 0.004, 181, "median length", fontsize=6, ha="left", va="top")
ax.set_xlim(0, 0.425)
ax.set_ylim(-8, 185)
ax.set_xlabel("Length of the average arrow\n(index units, 3 months)", fontsize=6, labelpad=2)
ax.set_ylabel("Spread of subgroup\ndirections (degrees)", fontsize=6, labelpad=2)
cax = axes_mm(fig, 70, 11, 2.5, 30)
cb = fig.colorbar(sc, cax=cax)
cb.set_label("Cancellation ratio R", fontsize=6, labelpad=2, rotation=270, va="bottom")
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
strong, weak = cc[cc["mag"] >= med], cc[cc["mag"] < med]
sh, lo = 100 * (weak["max_angle"] > 90).mean(), 100 * (strong["max_angle"] > 90).mean()
save_panel(
    fig, "ed06d_short_arrows",
    legend=("Where the average arrow is short, subgroups disagree. Each dot is one cell of the audit's own tumour x host plane (not the state map) (cells with "
            f"at least {A.MIN_CELL} visits, {len(cc)} cells), split by tumour family. Horizontal, length of the average (pooled) "
            "arrow; vertical, the angle spanned by the subgroup arrows in the cell (above the dashed line, some subgroups move "
            "in nearly opposite directions); colour, cancellation ratio R (1, all subgroups move the same way; lower, "
            f"opposing movements cancel); dot size, visits in the cell. Dotted line, median length. Among cells with a short arrow "
            f"{sh:.0f}% have subgroups spanning more than 90 degrees; among cells with a long arrow, {lo:.0f}%. Generated atlas, "
            "six models; visits from the same trajectory are not independent."),
    sources=SRC, confirm=CONF + ["The two percentages in the legend are computed here from the recomputed cells (as in the audit)."],
    message="Subgroups disagree mostly in cells where the average arrow is short.")
