"""n11_landscapes: event-conditioned landscapes, concept and measurement.

Message: the conceptual model (top) says that treatment, response and progression switch the field a patient moves in;
the measured row shows how the 4,434 generated trajectories actually move in the corresponding event-conditioned views.
Top: conceptual row. An illustrative field on disease
burden and host impairment drifts everywhere towards more burden and worse condition, into an absorbing death edge;
treatment adds a temporary push back. All strengths and positions are illustrative, not estimated from data.
Bottom: outputs/exploratory_n11_event_fields/ (scripts/explore/n11_event_conditioned_fields_v1.py): mean 3-month
displacement of survivors per supported map cell (>= 30 distinct courses starting and >= 30 contributing), the 50%
highest-density region of person-time, and the example generated patient. Same window and arrow scale in all six maps.
Run: uv run python scripts/figures/panels/n11_landscapes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INK, ROOT, STATE_COLOURS, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

F = ROOT / "outputs" / "exploratory_n11_event_fields"
fields = pd.read_csv(F / "n11_fields_v1.csv")
phases = pd.read_csv(F / "n11_phases_v1.csv").set_index("phase")
dens = np.load(F / "n11_density_v1.npz")
expos = pd.read_csv(F / "n11_exemplar_v1.csv").set_index("phase")
epath = pd.read_csv(F / "n11_exemplar_path_v1.csv")

REM, CAN = STATE_COLOURS["response"], STATE_COLOURS["progression"]
ACC = "#BFDCD6"           # accumulation region fill
ARROW = INK
PT = "#2C6FB7"            # the Fig. 1c patient

# (title, short label, patient position, treatment weight w, controlled state of the treatment); illustrative values.
# Death is a general drift towards more burden and worse condition, not a point;
# an active treatment pulls patients towards a controlled state (point attractor) that is not cure and weakens near death.
CTRL1, CTRL2 = (0.22, 0.24), (0.33, 0.33)
CONCEPT = [("1 Before treatment", "drift towards death", (0.45, 0.38), 0.00, None),
           ("2 Treatment starts", "pull to a controlled state", (0.45, 0.38), 0.85, CTRL1),
           ("3 Response", "moving towards it", (0.32, 0.30), 0.85, CTRL1),
           ("4 Maintained response", "held near it", CTRL1, 0.85, CTRL1),
           ("5 Progression", "drift resumes", (0.26, 0.28), 0.00, None),
           ("6 Treatment switch", "weaker pull, worse state", (0.42, 0.37), 0.70, CTRL2)]
MEASURED = {1: "Before the first\nsystemic treatment",
            2: "First 3 months on first\ntreatment, no progression",
            3: "After first response,\nbefore next progression",
            4: "≥3 months after first\nresponse, no progression",
            5: "First 6 months after\nfirst progression",
            6: "First 3 months after\nnext start or switch"}

W, H = 180.0, 77.5
CW, GAP, X0 = 27.5, 2.0, 4.0
fig = new_fig(W, H)

# ---------------------------------------------------------------- top row: concept
# Illustrative field on (disease burden, host impairment), both 0-1. Without treatment the disease drift points
# everywhere towards more burden and worse condition and grows as both rise, into an absorbing death edge. An active
# treatment adds a pull towards its controlled state (a point attractor, not the origin: treatment controls without
# curing); the pull weakens as the patient nears death, where the drift wins.
TOP_B, TOP_H = 48.0, 20.0
DEATH = 1.55                                     # x + y >= DEATH: death (absorbing edge)


def drift(x, y):
    d = np.hypot(1.0, 0.45 + 0.6 * x)
    mag = 0.3 + 0.9 * (x + y) / 2
    return mag * 1.0 / d, mag * (0.45 + 0.6 * x) / d


def field(x, y, w, c):
    dx, dy = drift(x, y)
    if c is None or w == 0:
        return dx, dy
    we = w * np.clip(1 - (x + y) / 1.7, 0, None)   # the pull weakens as the patient nears death
    # the pull aims at a point chosen so that the field rests exactly at the controlled state c
    wc = w * max(0.0, 1 - (c[0] + c[1]) / 1.7)
    cx, cy = drift(*c)
    tx, ty = c[0] - (1 - wc) * cx / (wc * 4.0), c[1] - (1 - wc) * cy / (wc * 4.0)
    px, py = 4.0 * (tx - x), 4.0 * (ty - y)
    return (1 - we) * dx + we * px, (1 - we) * dy + we * py


gx, gy = np.meshgrid(np.linspace(0.06, 0.94, 10), np.linspace(0.08, 0.92, 7))
sx, sy = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120))
surface = -(sx + 0.8 * sy)                       # tilted surface, falling towards the death edge
dead_x = [DEATH - 1.0, 1.0, 1.0]
dead_y = [1.0, 1.0, DEATH - 1.0]
for k, (title, sub, dot, w, c) in enumerate(CONCEPT):
    ax = axes_mm(fig, X0 + k * (CW + GAP), TOP_B, CW, TOP_H)
    ax.contourf(sx, sy, surface, levels=10, cmap="Blues_r", alpha=0.30, zorder=0)
    ax.fill(dead_x, dead_y, color=STATE_COLOURS["death"], alpha=0.30, lw=0, zorder=1)
    ax.text(0.97, 0.95, "Death", fontsize=6, ha="right", va="top", zorder=6)
    alive = gx + gy < DEATH - 0.06
    vx, vy = field(gx[alive], gy[alive], w, c)
    ax.quiver(gx[alive], gy[alive], vx * 0.11, vy * 0.11, color="#7B8794", angles="xy", scale_units="xy", scale=1,
              width=0.006, headwidth=3.5, headlength=3.5, zorder=2)
    if c is not None:
        ax.plot(*c, marker="D", ms=3.0, color=REM, mec=INK, mew=0.4, zorder=4)
    v = np.array(field(dot[0], dot[1], w, c))
    v = v / max(np.hypot(*v), 1e-9) * float(np.clip(np.hypot(*v) * 0.5, 0.06 if k == 3 else 0.15, 0.19))
    if k != 3:  # maintained response: held at the controlled state, no net movement
        ax.annotate("", xy=(dot[0] + v[0], dot[1] + v[1]), xytext=dot, zorder=5,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.32", lw=1.3, color=PT))
    ax.plot(*dot, "o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.0, 1.03, f"{title}\n{sub}", transform=ax.transAxes, fontsize=6, ha="left", va="bottom", linespacing=1.0)
    if k == 0:
        ax.set_xlabel("Disease burden →", fontsize=6, labelpad=1, loc="left")
        ax.set_ylabel("Host impairment →", fontsize=6, labelpad=1, loc="bottom")
fig.text(X0 / W, (TOP_B + TOP_H + 6.3) / H, "Conceptual model", fontsize=6.5, ha="left", va="bottom")

# ---------------------------------------------------------------- bottom row: measured
BOT_B, BOT_H = 6.0, 26.0
pad = 0.06
xs = np.r_[fields.cx - 0.06, fields.cx + 0.06, fields.cx + fields.dx]
ys = np.r_[fields.cy - 0.06, fields.cy + 0.06, fields.cy + fields.dy]
xl = (xs.min() - pad, xs.max() + pad)
yl = (ys.min() - pad, ys.max() + pad)
# equal scale in x and y within the map axes
ratio = (yl[1] - yl[0]) / (xl[1] - xl[0])
want = BOT_H / CW
if ratio < want:
    mid, half = sum(yl) / 2, (xl[1] - xl[0]) * want / 2
    yl = (mid - half, mid + half)
else:
    mid, half = sum(xl) / 2, (yl[1] - yl[0]) / want / 2
    xl = (mid - half, mid + half)
xe, ye = dens["x_edges"], dens["y_edges"]
xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
for k, p in enumerate(sorted(MEASURED)):
    ax = axes_mm(fig, X0 + k * (CW + GAP), BOT_B, CW, BOT_H)
    d = dens[f"phase{p}"].T
    lvl = phases.loc[p, "hdr50_level"]
    ax.contourf(xc, yc, d, levels=[lvl, d.max() + 1], colors=[ACC], zorder=0)
    ax.contour(xc, yc, d, levels=[lvl], colors=["#6AA89B"], linewidths=0.5, zorder=1)
    f = fields[fields.phase.eq(p)]
    ax.quiver(f.cx, f.cy, f.dx, f.dy, color=ARROW, angles="xy", scale_units="xy", scale=1, width=0.006,
              headwidth=3.2, headlength=3.4, headaxislength=3.0, zorder=3)
    inside = epath.pc1.between(*xl) & epath.pc2.between(*yl)
    k_out = int(np.argmax(~inside.to_numpy())) if (~inside).any() else None
    shown = epath.iloc[:k_out] if k_out is not None else epath
    ax.plot(shown.pc1, shown.pc2, color=PT, lw=0.5, alpha=0.45, zorder=2)
    if k_out:  # the path leaves the window towards more disease: mark the exit at the edge, not hidden points
        a, b = epath.iloc[k_out - 1], epath.iloc[k_out]
        t = min(1.0, (xl[1] - 0.01 - a.pc1) / (b.pc1 - a.pc1)) if b.pc1 > a.pc1 else 1.0
        ax.annotate("", xy=(a.pc1 + t * (b.pc1 - a.pc1), a.pc2 + t * (b.pc2 - a.pc2)), xytext=(a.pc1, a.pc2), zorder=2,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.12,head_length=0.22", lw=0.5, color=PT, alpha=0.6))
    if p in expos.index:
        e = expos.loc[p]
        ax.plot(e.x, e.y, "o", ms=3.2, mfc="white", mec=PT, mew=1.0, zorder=5)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.0, 1.03, f"{p} {MEASURED[p]}", transform=ax.transAxes, fontsize=6, ha="left", va="bottom",
            linespacing=1.0)
    r = phases.loc[p]
    ax.text(0.0, -0.04, f"{int(r.trajectories):,} courses; {int(r.supported_cells)} cells\n"
            f"died within 3 months: {int(r.distinct_deaths_within_3_months):,}", transform=ax.transAxes, fontsize=6,
            ha="left", va="top", linespacing=1.0)
fig.text(X0 / W, (BOT_B + BOT_H + 6.3) / H, "Measured on 4,434 generated trajectories", fontsize=6.5, ha="left", va="bottom")

legend = (
    "Event-conditioned landscapes: a conceptual model and its measured counterpart in generated data. Top, conceptual "
    "model, drawn by hand and not estimated from data: at six moments of one course, the patient (open circle) sits "
    "on a map of disease burden (horizontal) and host impairment (vertical). Without active treatment, the grey arrows "
    "drift towards more disease and worse condition, faster as both rise, and end in death, an absorbing outcome (dark "
    "corner, drawn as a region for display; death is not a position on the map). An active treatment pulls patients towards a controlled state (diamond) with less disease and better "
    "condition but not cure; the pull weakens near death, and a later treatment pulls less, towards a worse state. The "
    "blue arrow is the net movement of the patient. The controlled state is a conceptual hypothesis, not an attractor "
    "established in data. "
    "Bottom, the corresponding event-conditioned views of the 4,434 generated "
    "trajectories of the atlas (six models) on the frozen state map. Each month at which a trajectory is alive is a "
    "start visit, assigned to the phases defined by that trajectory's own events (labels literal; phases overlap and are "
    "not successive stages of one course; phase 4 continued response is not verified; phase 6 is the next treatment start "
    "or switch, not a verified clinical line). Black arrows, mean 3-month displacement among start visits still alive "
    "3 months later, in each cell of the Extended Data Fig. 4a grid with at least 30 distinct courses starting and at "
    "least 30 distinct courses contributing; drawn at their real length, same scale in every map; deaths are not arrow "
    "destinations and are counted below each map as distinct trajectories that died within 3 months of a start visit in "
    "that phase. Green regions, the 50% highest-density region of person-time in the phase (where generated patients "
    "accumulate; smoothed, display only); they are not attractors or potentials. Blue line and open circle, the generated "
    "patient of Fig. 1c and its position at its first month in each phase; after month 27 its path leaves the window towards more disease (arrow at the edge) before death at 32.7 months. Average movements differed across event-defined phases. The measured arrows summarise survivors of "
    "generated courses; they do not show forces, stability, resistance mechanisms or causal effects of treatment, and do "
    "not show that real patients move this way. Generated data."
)
save_panel(fig, "n11_landscapes", legend=legend,
           sources=["outputs/exploratory_n11_event_fields/n11_fields_v1.csv",
                    "outputs/exploratory_n11_event_fields/n11_phases_v1.csv",
                    "outputs/exploratory_n11_event_fields/n11_density_v1.npz",
                    "concept row: illustrative field defined in this script (no data)"],
           confirm=["Concept row: drift to an absorbing death edge without treatment; active "
                    "treatment pulls towards a controlled state (conceptual point attractor); values illustrative (CONCEPT).",
                    "Measured arrows average visits within a cell (visit-weighted), bootstrap by whole trajectory saved "
                    "in n11_fields_v1.csv (not drawn)."],
           message="Concept: events switch the field; measured: how generated trajectories move in each event-conditioned "
                   "view.")
