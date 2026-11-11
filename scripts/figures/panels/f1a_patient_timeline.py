"""f1a_patient_timeline: one generated patient as a rich timeline.

Message: a generated record is a whole trajectory: treatment lines, disease state, tumour burden
in each organ, physical condition and side effects, molecular and clinical events, and the
patient's position on the state map, all on one time axis.

Data: the Figure 1 example patient (_exemplar.py): events from exemplar.json, state snapshots
from the frozen state map. Generated data (Claude Sonnet 5).

Run: uv run python scripts/figures/panels/f1a_patient_timeline.py
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _exemplar as E  # noqa: E402
from common import INK, LIGHT, MODALITY_COLOURS, STATE_COLOURS, fk, save_panel  # noqa: E402

ex, snaps = E.load()
T_END = E.death_month(ex)
STATE_NAME = {"unresolved": "Not yet assessed", "response": "Response", "progression": "Progression"}

TUMOUR_CMAP = ListedColormap(["#FFF7EC", "#FDD49E", "#FDBB84", "#FC8D59", "#E34A33", "#B30000"])
TUMOUR_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), 6)
HOST_CMAP = ListedColormap(["#F1EEF6", "#BDC9E1", "#74A9CF", "#2B8CBE", "#045A8D"])
HOST_NORM = BoundaryNorm(np.linspace(0, 1, 6), 5)

# rows from top: treatment, disease, 5 tumour sites, 3 host rows, events, PC1 trace
ROWS = (["Treatment line", "Disease state"] + [s[1] for s in E.SITES] + [h[1] for h in E.HOST]
        + ["Clinical events", "State-map PC1"])
Y = {name: len(ROWS) - 1 - i for i, name in enumerate(ROWS)}
H = 0.78

fig = fk.figure(width=180, height_mm=58)
fig.set_layout_engine(None)
ax = fig.add_axes([0.15, 0.13, 0.64, 0.82])
ax.set_xlim(0, T_END + 2.6)
ax.set_ylim(-0.6, len(ROWS) + 0.05)

# treatment lines, coloured by class, drug names on the bars
for start, stop, name, cls in E.treatment_lines(ex):
    y = Y["Treatment line"]
    ax.add_patch(Rectangle((start, y - H / 2), stop - start, H, facecolor=MODALITY_COLOURS[cls],
                           edgecolor="white", lw=0.6))
    last = stop >= max(q[1] for q in E.treatment_lines(ex))  # the last label ends before the death line
    ax.text(stop if last else start + 0.2, y + H / 2 + 0.08, name, ha="right" if last else "left", va="bottom", fontsize=6, color=INK)

# disease state
for start, stop, state in E.disease_segments(ex):
    key = state if state in STATE_COLOURS else "unresolved"
    ax.add_patch(Rectangle((start, Y["Disease state"] - H / 2), stop - start, H,
                           facecolor=STATE_COLOURS[key], edgecolor="white", lw=0.6))

# tumour burden by site and host condition, piecewise constant between snapshots
times = list(snaps["month"]) + [T_END]
for i in range(len(snaps)):
    t0, t1 = times[i], times[i + 1]
    if t1 <= t0:
        continue
    row = snaps.iloc[i]
    for col, name in E.SITES:
        band = int(round(row[col] * 5))
        ax.add_patch(Rectangle((t0, Y[name] - H / 2), t1 - t0, H,
                               facecolor=TUMOUR_CMAP(TUMOUR_NORM(band)), edgecolor="none"))
    for col, name, _top in E.HOST:
        ax.add_patch(Rectangle((t0, Y[name] - H / 2), t1 - t0, H,
                               facecolor=HOST_CMAP(HOST_NORM(min(row[col], 0.999))), edgecolor="none"))

# clinical events
EVENTS = {"systemic_treatment_modification": ("$↧$", "Dose change"),
          "clonal_evolution_event": ("D", "Clonal evolution"), "local_intervention": ("s", "Local treatment")}
ye = Y["Clinical events"]
placed: list[float] = []
for e in sorted(ex["events"], key=lambda q: q["month"]):
    if e["type"] in EVENTS:
        mk = EVENTS[e["type"]][0]
        dx = 0.9 if placed and e["month"] - placed[-1] < 0.9 else 0.0
        placed.append(e["month"] + dx)
        ax.plot(e["month"] + dx, ye, marker=mk, ms=3.6, color=INK, mfc=INK if mk not in ("o",) else "white",
                mew=0.6, ls="none")
ax.axvline(T_END, color=INK, lw=0.8)
ax.text(T_END + 0.3, Y["Disease state"], "Death", fontsize=6, va="center", ha="left")

# position on the state map (composite PC1), coloured by the disease state at that time
yp, span = Y["State-map PC1"], 0.42
pc1 = snaps["pc1"].to_numpy()
lo, hi = pc1.min(), pc1.max()


def ypos(v: float) -> float:
    return yp - span + 2 * span * (v - lo) / (hi - lo)


segs = E.disease_segments(ex)


def state_at(t: float) -> str:
    return next((s for a, b, s in segs if a <= t < b), segs[-1][2])


for i in range(len(snaps)):
    t0, t1 = times[i], times[i + 1]
    if t1 <= t0:
        continue
    c = STATE_COLOURS.get(state_at(t0), STATE_COLOURS["unresolved"])
    ax.plot([t0, t1], [ypos(pc1[i])] * 2, color=INK, lw=1.2,
            solid_capstyle="butt")
    if i + 1 < len(snaps):
        ax.plot([t1, t1], [ypos(pc1[i]), ypos(pc1[i + 1])], color=LIGHT, lw=0.5)

# row labels and group brackets
ax.set_yticks([Y[r] for r in ROWS])
ax.set_yticklabels(["Treatment line", "Disease state"] + [s[1] for s in E.SITES]
                   + [h[1] for h in E.HOST] + ["Clinical events", "State-map PC1\n(up = more burden)"], fontsize=6,
                   linespacing=0.9)
ax.tick_params(axis="y", length=0, pad=2)
for spine in ("left",):
    ax.spines[spine].set_visible(False)
ax.set_xticks(range(0, 36, 6))
ax.set_xlabel("Months from diagnosis")
for name, rows in (("Tumour burden", [s[1] for s in E.SITES]), ("Condition", [h[1] for h in E.HOST])):
    top, bot = Y[rows[0]] + H / 2, Y[rows[-1]] - H / 2
    ax.annotate("", xy=(-0.135, top), xytext=(-0.135, bot), xycoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax.text(-0.15, (top + bot) / 2, name, transform=ax.get_yaxis_transform(), rotation=90,
            fontsize=6.5, ha="center", va="center")

# keys on the right
kx = 0.805
k = fig.add_axes([kx, 0.13, 0.19, 0.82])
k.set_xlim(0, 1)
k.set_ylim(0, 1)
k.axis("off")


def swatch_row(y, title, items, w=0.075):
    k.text(0, y, title, fontsize=6.5, va="bottom", ha="left")
    for j, (col, lab) in enumerate(items):
        yy = y - 0.058 * (j + 1)
        k.add_patch(Rectangle((0, yy - 0.018), w, 0.036, facecolor=col, edgecolor="none"))
        k.text(w + 0.04, yy, lab, fontsize=6, va="center")
    return y - 0.058 * (len(items) + 1) - 0.03


y = 0.95
y = swatch_row(y, "Disease state", [(STATE_COLOURS[s], STATE_NAME[s]) for s in ("response", "progression", "unresolved")])
y = swatch_row(y, "Treatment", [(MODALITY_COLOURS["targeted"], "Targeted therapy"),
                                (MODALITY_COLOURS["chemotherapy"], "Chemotherapy")])
k.text(0, y, "Tumour burden", fontsize=6.5, va="bottom")
for j, band in enumerate(E.BURDEN_BANDS):
    k.add_patch(Rectangle((j * 0.16, y - 0.06), 0.16, 0.04, facecolor=TUMOUR_CMAP(j), edgecolor="none"))
k.text(0, y - 0.075, "absent", fontsize=6, va="top")
k.text(0.96, y - 0.075, "diffuse", fontsize=6, va="top", ha="right")
y -= 0.2
k.text(0, y, "Condition, side effects", fontsize=6.5, va="bottom")
for j in range(5):
    k.add_patch(Rectangle((j * 0.192, y - 0.06), 0.192, 0.04, facecolor=HOST_CMAP(j), edgecolor="none"))
k.text(0, y - 0.075, "best", fontsize=6, va="top")
k.text(0.96, y - 0.075, "worst", fontsize=6, va="top", ha="right")
y -= 0.19
handles = [Line2D([], [], marker=m, ls="none", color=INK, mfc=INK if m != "o" else "white", ms=3.6, mew=0.6,
                  label=lab) for m, lab in EVENTS.values()]
k.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.04, y + 0.02), fontsize=6, frameon=False,
         handletextpad=0.3, labelspacing=0.25, borderaxespad=0)

legend = (
    "One generated patient as a trajectory. Metastatic EGFR-mutant lung adenocarcinoma generated by Claude "
    "Sonnet 5, selected by a fixed rule (at most 30 recorded events, a progression, a second treatment line "
    "and death; lowest identifier hash); an illustration, not a representative patient. Rows from top: "
    "treatment lines coloured by class; disease state from the generated assessments; tumour burden in each "
    "organ (absent, microscopic, small, moderate, bulky, diffuse); physical condition (ECOG performance status "
    "0-4, symptom burden 0-3) and the highest active side-effect grade, darker meaning worse; clinical events; "
    "and the patient's position along the first axis of the state map (composite PC1, which rises with overall "
    "disease burden), coloured by disease state. Each value holds until the next generated assessment. "
    "Vertical line, death at 32.7 months. Generated data."
)
save_panel(fig, "f1a_patient_timeline", legend=legend,
           sources=["outputs/exploratory_figure_panels/fig1/data/exemplar.json",
                    "outputs/paper_state_map/state_projection.csv.gz"],
           message="A generated record is a whole trajectory: treatments, disease state, organ burden, "
                   "condition, events and state-map position on one time axis.",
           confirm=["Tumour sites shown are the six ever involved in this patient; marrow, peritoneal and "
                    "soft-tissue sites were absent throughout.",
                    "The switch to carboplatin + etoposide after clonal evolution reads as a small-cell "
                    "transformation; the record does not name it, so the figure does not."])
