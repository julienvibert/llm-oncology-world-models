"""f1c_dynamics: treatment and clinical events change the distribution of the next movement.

Message: at six moments of patient 1's course (diagnosis, treatment starts, response, stable
remission, resistance, treatment switch), the possible 3-month moves from the current state differ
in direction, size, spread and outcome: untreated disease mostly grows, treatment shifts the moves
towards less disease, a maintained response holds them close, resistance turns them back towards progression,
and a new treatment opens a wide, mixed distribution.

Design (six moments; no basins, potentials or forces): each
moment shows a cloud of possible 3-month moves from the current state, the end of each move coloured
by the disease state at the next assessment, deaths as terminal crosses, and the mean move (black
arrow) computed from the drawn moves. Constructed illustration, no data.

Run: uv run python scripts/figures/panels/f1c_dynamics.py
"""

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402
from _f1_world import HIST, P1, XLABEL, assess, death_mark, history_curve, smooth  # noqa: E402

XL, YL = (0.0, 9.4), (0.15, 3.75)
N = 24
# moment: (label, treatment, month, reference burden of the current line, components)
# component: (weight, mean move (dburden, dcondition), sd (burden, condition))
MOMENTS = [
    ("Diagnosis", "No treatment yet", 0.0, None, [(1.0, (1.45, 0.42), (0.4, 0.2))]),
    ("Treatment starts", "Osimertinib", 0.0, 0.0, [(0.92, (-3.4, -0.75), (0.9, 0.25)), (0.08, (1.2, 0.4), (0.3, 0.1))]),
    ("Response", "Osimertinib", 3.0, 0.0, [(1.0, (-2.0, -0.28), (0.45, 0.1))]),
    ("Maintained response", "Osimertinib", 9.0, 0.0, [(0.86, (0.0, 0.0), (0.3, 0.09)), (0.14, (0.65, 0.15), (0.15, 0.05))]),
    ("Resistance", "Osimertinib", 13.5, 0.0, [(0.85, (1.3, 0.38), (0.5, 0.15)), (0.15, (0.1, 0.05), (0.15, 0.05))]),
    ("Treatment switch", "Strategy 1", 16.0, 16.0, [(0.62, (-1.7, 0.45), (0.55, 0.22)), (0.28, (1.3, 0.6), (0.5, 0.25)),
                                                   (0.10, (2.0, 0.8), (0.25, 0.1))]),
]
NED_MOVES = {3: 1}   # one move reaches no evidence of disease during the maintained response


def at(month):
    return (float(smooth(P1["months"], P1["burden"], np.array([month]))[0]),
            float(smooth(P1["months"], P1["cond"], np.array([month]))[0]))


W, H = 102.0, 57.5
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)
AW, AH, X0, GX, Y0, GY = 30.0, 21.05, 6.5, 2.0, 0.5, 2.2
rng = np.random.default_rng(2026)
for i, (label, tx, month, ref_m, comps) in enumerate(MOMENTS):
    r, c = divmod(i, 3)
    ax = fig.add_axes([(X0 + c * (AW + GX)) / W, 1 - (Y0 + r * (AH + GY) + AH) / H, AW / W, AH / H])
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_xticks([])
    ax.set_yticks([])
    cur = np.array(at(month))
    # path so far
    if month > 0:
        t, hb, hc = history_curve(P1)
        sel = t <= month + 1e-9
        ax.plot(hb[sel], hc[sel], color=HIST, lw=1.1, zorder=3, solid_capstyle="round")
    # possible 3-month moves
    w = np.array([cp[0] for cp in comps])
    which = rng.choice(len(comps), size=N, p=w / w.sum())
    ends, deaths = [], []
    for j, k in enumerate(which):
        _, mu, sd = comps[k]
        e = cur + np.array(mu) + rng.normal(0, 1, 2) * np.array(sd)
        e[0] = max(e[0], 0.35)
        e[1] = max(e[1], 0.15)
        if label == "Treatment switch" and k == 2:
            deaths.append(e)
        else:
            ends.append(e)
    for _ in range(NED_MOVES.get(i, 0)):
        ends[-1] = np.array([0.08, cur[1] - 0.08])
    ends = np.array(ends)
    ref = at(ref_m)[0] if ref_m is not None else cur[0]
    nadir = min(P1["burden"][: int(np.searchsorted(P1["months"], month, side="right"))]) if ref_m == 0.0 else ref
    states = [assess(np.array([e[0]]), ref=ref, nadir0=nadir)[0] for e in ends]
    for e, s in zip(ends, states):
        mid = (cur + e) / 2 + np.array([0, 0.06])
        xs = np.array([cur[0], mid[0], e[0]])
        ys = np.array([cur[1], mid[1], e[1]])
        ax.plot(xs, ys, color=STATE_COLOURS[s], lw=0.55, alpha=0.75, zorder=2)
        ax.plot(*e, "o", ms=2.6, mfc=STATE_COLOURS[s], mec="white", mew=0.3, zorder=4)
    for e in deaths:
        ax.plot([cur[0], e[0]], [cur[1], e[1]], color=INK, lw=0.5, alpha=0.6, zorder=2)
        death_mark(ax, *e, ms=3.4, mew=0.9)
    mean = ends.mean(axis=0)
    if np.linalg.norm(mean - cur) > 0.15:
        ax.add_patch(FancyArrowPatch(tuple(cur), tuple(mean), arrowstyle="-|>", mutation_scale=8, color=INK,
                                     lw=1.6, zorder=6, shrinkA=2, shrinkB=0,
                                     path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]))
    ax.plot(*cur, "o", ms=4.6, mfc="white", mec=INK, mew=1.1, zorder=7)
    ax.text(0.03, 0.95, f"{i + 1}  {label}", transform=ax.transAxes, fontsize=6.5, fontweight="bold",
            va="top", ha="left", zorder=9)
    ax.text(0.03, 0.80, tx + (f", month {month:g}" if month else ", month 0"), transform=ax.transAxes,
            fontsize=6, va="top", ha="left", zorder=9)

fig.text(X0 / W + (3 * AW + 2 * GX) / 2 / W, 1 - (Y0 + 2 * AH + GY + 0.8) / H, XLABEL, fontsize=6.5,
         ha="center", va="top")
fig.text(1.2 / W, 1 - (Y0 + AH + GY / 2) / H, "State dimension 2: poorer condition  →", fontsize=6.5,
         rotation=90, ha="left", va="center")

# key along the bottom (two lines)
k = fig.add_axes([0, 0, 1, 8.6 / H])
k.set_xlim(0, W)
k.set_ylim(0, 8.6)
k.axis("off")
y1, y2 = 6.3, 2.0
k.plot([X0 + 0.5, X0 + 4.5], [y1, y1], color=STATE_COLOURS["progression"], lw=0.7)
k.plot(X0 + 4.5, y1, "o", ms=2.6, mfc=STATE_COLOURS["progression"], mec="white", mew=0.3)
k.text(X0 + 5.6, y1, "Possible 3-month move to:", fontsize=6, va="center")
xx = X0 + 33.5
for sk, lab in (("no evidence of disease", "NED"), ("response", "Response"), ("stable", "Stable"),
                ("progression", "Progression")):
    k.plot(xx, y1, "o", ms=3.2, mfc=STATE_COLOURS[sk], mec="white", mew=0.3)
    k.text(xx + 1.4, y1, lab, fontsize=6, va="center")
    xx += 4.0 + 1.25 * len(lab)
death_mark(k, xx, y1, ms=3.2, mew=0.9)
k.text(xx + 1.6, y1, "Death", fontsize=6, va="center")
k.add_patch(FancyArrowPatch((X0 + 0.5, y2), (X0 + 5.0, y2), arrowstyle="-|>", mutation_scale=7, color=INK,
                            lw=1.5))
k.text(X0 + 5.6, y2, "Mean move", fontsize=6, va="center")
k.plot(X0 + 22.0, y2, "o", ms=4.0, mfc="white", mec=INK, mew=1.0)
k.text(X0 + 23.6, y2, "Patient now", fontsize=6, va="center")
k.plot([X0 + 39.0, X0 + 43.0], [y2, y2], color=HIST, lw=1.1)
k.text(X0 + 44.2, y2, "Course so far", fontsize=6, va="center")
k.text(W - 1.0, y2, "Constructed course, not data", fontsize=6, va="center", ha="right")

legend = (
    "Treatment and clinical events change the distribution of the next movement. Six moments in the course of "
    "the constructed patient 1 of Extended Data Fig. 2a. At each moment, thin lines show possible 3-month moves from the current state "
    "(open circle); the end of each move is coloured by the disease state documented at the next assessment (no evidence of "
    "disease, response (new or maintained), stable disease or progression), and crosses mark deaths within the 3 months. "
    "Black arrow, mean of the drawn moves among patients alive at 3 months. Before treatment most moves lead to "
    "more disease (1); starting osimertinib shifts them towards less disease (2, 3); during the maintained response they stay "
    "close, with an occasional complete response or early regrowth (4); with resistance they turn back towards "
    "progression (5); at the switch to strategy 1 they spread widely, with responses, progression and deaths (6). "
    "No arrow is drawn when the mean move is negligible (4). The mean move summarises a distribution; it is not a force and does not show what will happen to this "
    "patient. 24 moves per moment, constructed illustration, not data or model output."
)
save_panel(fig, "f1c_dynamics", legend=legend, sources=["schematic; scripts/figures/panels/_f1_world.py"],
           message="Treatment, response and resistance change the distribution of possible 3-month moves from "
                   "the current state.",
           confirm=["Six-moment storyline drawn as distributions of 3-month moves, without basins or forces.",
                    "One complete response during the maintained response (panel 4) is shown to make no evidence of disease "
                    "a documented state; uncommon on osimertinib in metastatic disease."])
