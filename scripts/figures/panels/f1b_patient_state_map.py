"""f1b_patient_state_map: the same patient as a path through the clinical state space.

Message: the trajectory of panel a is a path on the frozen state map: response moves the patient
towards lower burden, progression and deterioration move it towards more burden and poorer
condition, and the time spent at each position is visible.

Data: the Figure 1 example patient (_exemplar.py) on the frozen state map; background, all
105,839 snapshots of the six-model atlas. Generated data.

Run: uv run python scripts/figures/panels/f1b_patient_state_map.py
"""

import sys
from pathlib import Path

import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _exemplar as E  # noqa: E402
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402

ex, snaps = E.load()
atlas = pd.read_csv(E.PROJECTION, usecols=["pc1", "pc2"])
T_END = E.death_month(ex)
segs = E.disease_segments(ex)


def state_at(t: float) -> str:
    return next((s for a, b, s in segs if a <= t < b), segs[-1][2])


# collapse consecutive snapshots at the same position; dwell = time until the next move
pts = []
for i, r in enumerate(snaps.itertuples()):
    t_next = snaps["month"].iloc[i + 1] if i + 1 < len(snaps) else T_END
    if pts and abs(pts[-1]["x"] - r.pc1) < 1e-9 and abs(pts[-1]["y"] - r.pc2) < 1e-9:
        pts[-1]["dwell"] += t_next - r.month
        continue
    pts.append({"x": r.pc1, "y": r.pc2, "t": r.month, "dwell": t_next - r.month})
P = pd.DataFrame(pts)

fig = fk.figure(width=98, height_mm=58)
fig.set_layout_engine(None)
ax = fig.add_axes([5.0 / 98, 0.09, 67.0 / 98, 0.89])
dens = LinearSegmentedColormap.from_list("atlas", ["#FFFFFF", "#F3E3C3", "#D9B77A", "#A87C3A"])
hb = ax.hexbin(atlas.pc1, atlas.pc2, gridsize=45, bins="log", cmap=dens, mincnt=1, linewidths=0,
               rasterized=True, zorder=0)
# one continuous path coloured by months since diagnosis (the course revisits nearby positions between
# months 15 and 25, so separate shrunken arrows were unreadable); arrowheads only on the longer moves
TIME = LinearSegmentedColormap.from_list("time", ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
norm = Normalize(0, T_END)
seg = [[(P.x.iloc[i], P.y.iloc[i]), (P.x.iloc[i + 1], P.y.iloc[i + 1])] for i in range(len(P) - 1)]
lc = LineCollection(seg, colors=[TIME(norm(P.t.iloc[i + 1])) for i in range(len(P) - 1)], linewidths=1.3,
                    zorder=2, capstyle="round")
ax.add_collection(lc)
for i in range(len(P) - 1):
    a, b = P.iloc[i], P.iloc[i + 1]
    if ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5 > 0.15:
        mx, my = a.x + 0.62 * (b.x - a.x), a.y + 0.62 * (b.y - a.y)
        ax.annotate("", xy=(mx, my), xytext=(a.x + 0.5 * (b.x - a.x), a.y + 0.5 * (b.y - a.y)), zorder=2.5,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.38", lw=0,
                                    color=TIME(norm(b.t)), shrinkA=0, shrinkB=0))
for r in P.itertuples():
    st = state_at(r.t)
    col = STATE_COLOURS.get(st, STATE_COLOURS["unresolved"])
    ax.scatter(r.x, r.y, s=8 + 9 * r.dwell, color=col, edgecolor=INK, linewidth=0.5, zorder=3)
ax.plot(P.x.iloc[-1], P.y.iloc[-1], marker="x", color=INK, ms=6, mew=1.2, zorder=4)


def note(i, text, dx, dy, ha="left"):
    ax.annotate(text, xy=(P.x.iloc[i], P.y.iloc[i]), xytext=(P.x.iloc[i] + dx, P.y.iloc[i] + dy),
                fontsize=6, ha=ha, va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK, shrinkA=0, shrinkB=3))


first_resp = int(P.index[P.t >= next(a for a, b, s in segs if s == "response")][0])
first_prog = int(P.index[P.t >= next(a for a, b, s in segs if s == "progression")][0])
note(0, "Diagnosis", 0.12, 0.17)
note(first_resp, f"Response on osimertinib\n({P.dwell.iloc[first_resp]:.0f} months here)", -0.12, 0.2, ha="left")
note(first_prog, "Progression", 0.1, -0.06)
note(len(P) - 1, f"Death, {T_END:.1f} months", -0.15, 0.06, ha="right")

pad1, pad2 = 0.18, 0.14
ax.set_xlim(P.x.min() - pad1, P.x.max() + pad1 / 2)
ax.set_ylim(P.y.min() - pad2, P.y.max() + pad2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("State-map PC1 (a.u.)")
ax.set_ylabel("State-map PC2 (a.u.)")
handles = [Line2D([], [], marker="o", ls="none", mfc=STATE_COLOURS[s], mec=INK, mew=0.5, ms=4, label=lab)
           for s, lab in (("unresolved", "Not yet assessed"), ("response", "Response"),
                          ("progression", "Progression"))]
handles.append(Line2D([], [], marker="h", ls="none", mfc="#D9B77A", mec="none", ms=5,
                      label="Density of all\ngenerated patients"))
handles.append(Line2D([], [], ls="none", label="Time at a position:"))
for d in (1, 6, 12):
    handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=0.5,
                          ms=(8 + 9 * d) ** 0.5, label=f"{d} month{'s' if d > 1 else ''}"))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(72.5 / 98, 0.99), ncol=1, fontsize=6,
           frameon=False, handletextpad=0.9, columnspacing=1.0, labelspacing=0.45)
cax = fig.add_axes([74.0 / 98, 0.11, 2.2 / 98, 0.22])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=TIME), cax=cax)
cb.set_ticks([0, 12, 24, 32])
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
cb.set_label("Months since\ndiagnosis", fontsize=6, labelpad=2)

legend = (
    "The patient of Fig. 1c as a path through the clinical state space. Each point is one position "
    "on the frozen state map, coloured by the disease state at that time; point area is proportional "
    "to the time spent there (key); the line joins successive positions and its colour gives the months since "
    "diagnosis (colour bar), with arrowheads on the longer moves; cross, death. Between months 15 and 25 the "
    "course moves back and forth between nearby positions. Shading, density of all "
    "105,839 positions of the 4,434 trajectories generated by six models (log scale), showing where "
    "generated patients usually are. The map axes are the first two principal components of encoded "
    "tumour burden by organ and host condition, fixed before any analysis; PC1 rises mainly with overall tumour burden; PC2 contrasts marrow disease with liver, lung and other "
    "disease and also carries host terms; neither is a validated clinical scale "
    "(Extended Data Fig. 2). Generated data (one trajectory of the atlas), unlike the constructed "
    "illustrations in panels b-d."
)
save_panel(fig, "f1b_patient_state_map", legend=legend,
           sources=["outputs/paper_state_map/state_projection.csv.gz",
                    "outputs/exploratory_figure_panels/fig1/data/exemplar.json"],
           message="The same trajectory is a path on the state map, with time spent at each position.",
           confirm=[])
