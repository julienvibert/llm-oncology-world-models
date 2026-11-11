"""Panel ed03a_same_state_map: generated leiomyosarcoma patients passing through one similar state, on the state map.
Data path: _ed03.same_state. Run: uv run python scripts/figures/panels/ed03a_same_state_map.py"""
import sys
from pathlib import Path

from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed03 import same_state  # noqa: E402
from _f4 import fix_axes  # noqa: E402
from common import panel, save_panel  # noqa: E402

DIED, ALIVE, PAST = "#B2182B", "#1B7F9E", "#E69F00"
entries, box, bymodel = same_state()
fig, ax = panel(60, 62)
for e in entries:
    ax.plot(e["trail"]["fz1"].dropna(), e["trail"]["fz2"].dropna(), color=PAST, lw=0.7, ls=(0, (3, 1.5)), alpha=0.9, zorder=3)
    ax.plot(e["future"]["fz1"].dropna(), e["future"]["fz2"].dropna(), color=DIED if e["died"] else ALIVE, lw=0.9, alpha=0.85, zorder=4)
    fu = e["future"].dropna(subset=["fz1"])
    ax.scatter(fu["fz1"].iloc[-1], fu["fz2"].iloc[-1], s=5, color=DIED if e["died"] else ALIVE, zorder=5, lw=0)
    tr = e["trail"].dropna(subset=["fz1"]).iloc[-1]
    ax.scatter(tr["fz1"], tr["fz2"], s=14, facecolor="white", edgecolor="black", lw=0.9, zorder=6)
ax.set_xlabel("State-map PC1 (a.u.)")
ax.set_ylabel("State-map PC2 (a.u.)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="similar-state visit"),
                   Line2D([], [], color=PAST, lw=0.9, ls=(0, (3, 1.5)), label="9 months before"),
                   Line2D([], [], color=DIED, lw=0.9, label="12 months after: died"),
                   Line2D([], [], color=ALIVE, lw=0.9, label="12 months after: alive")],
          loc="upper left", frameon=False, fontsize=6, handlelength=1.4, borderaxespad=0.1)
fix_axes(fig, ax, left=9.5, right=1.5, bottom=6.5, top=1.5)
n = len(entries); nd = sum(e["died"] for e in entries)  # noqa: E702
LEGEND = (f"Generated leiomyosarcoma patients (six models) who pass through one similar state. Each line is one generated patient "
          f"(n = {n}; {nd} died within 12 months, {n - nd} alive at 12 months). The patients were selected, as before, because each has a visit inside the central band (10th-90th percentile) of the two coordinates of "
          "an older combined tumour-host axis at a similar time since diagnosis; that selection is unchanged. They are drawn here on the frozen state map "
          "(Extended Data Fig. 1), where the open circle marks each patient's selected visit, so the visits are close but not identical on this map. Amber dashed lines show the 9 months before the visit; solid lines show the 12 months after it (dots mark their end). "
          "Axes are state-map PC1 and PC2 in arbitrary units (loadings in Extended Data Fig. 1b); neither is a clinical scale. Death is outside the map, so paths end at the last living state. "
          "Patients are similar in this projection, not identical. Data are model-generated, not real patients.")
save_panel(fig, "ed03a_same_state_map", legend=LEGEND,
           sources=["outputs/exploratory_fig1_same_dot/readout_v2.json", "scripts/explore/build_fig1_same_dot_v2.py (data functions)"],
           confirm=[
                    "Display reprojected onto the frozen 20-coordinate state map (display change only). The selection band (similar state) is still defined in the older combined axis of the same-dot builder, so selected visits are not a box on this map.",
                    "Colours (died crimson, alive teal, past amber) are local to this page."],
           message="Generated patients who look similar at one visit have different pasts and futures.")
