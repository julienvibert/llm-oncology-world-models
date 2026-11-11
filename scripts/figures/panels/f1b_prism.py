"""f1b_prism: the conditioning prism.

Message: the same model serves the study's three experiments depending on which inputs the query
fixes: (1) unconstrained generation from the cancer type alone gives a population; (2) probing varies
one input at a time from the same starting point; (3) forecasting fixes a real patient's context,
condition and history and compares futures under alternative treatments. Patient examples (E) are a
separate, optional input.

Design: coloured inputs C, X, H, A as rays entering a prism; three output beams, one per
experiment of the study (unconstrained, probing with varying inputs, prediction of real patients).
Schematic, no data.

Run: uv run python scripts/figures/panels/f1b_prism.py
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INK, PAL, fk, save_panel  # noqa: E402

INPUT = {"C": ("#E69F00", "Context"), "X": ("#009E73", "Current condition"),
         "H": ("#882255", "History"), "A": ("#0072B2", "Treatment")}
E_COL = "#8C6D31"
OSI = PAL["osimertinib"]

fig = fk.figure(width=98, height_mm=48)
fig.set_layout_engine(None)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 98)
ax.set_ylim(0, 58)
ax.axis("off")

# inputs: coloured rays converging on the prism's left face
PL, PT, PB, APEX = 31.0, 44.0, 14.0, (45.0, 29.0)
ys = [40.5, 33.5, 26.5, 19.5]
for (key, (col, name)), y in zip(INPUT.items(), ys):
    ax.add_patch(FancyBboxPatch((0.6, y - 2.1), 4.2, 4.2, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=col, edgecolor="none"))
    ax.text(2.7, y, key, fontsize=7, color="white", ha="center", va="center", fontweight="bold")
    ax.text(5.6, y, name, fontsize=6, ha="left", va="center")
    ax.plot([22.5, PL], [y, 29 + (y - 30) * 0.45], color=col, lw=1.6, solid_capstyle="round")

# prism with faint inner dispersion bands
ax.add_patch(Polygon([(PL, PT), (PL, PB), APEX], closed=True, facecolor="#F4F1FA", edgecolor=INK, lw=0.9,
                     zorder=3))
for k, (key, (col, _)) in enumerate(INPUT.items()):
    f0, f1 = k / 4, (k + 1) / 4
    top0, top1 = PT - (PT - PB) * f0, PT - (PT - PB) * f1
    ax.add_patch(Polygon([(PL, top0), (PL, top1), APEX], closed=True, facecolor=col, alpha=0.13,
                         edgecolor="none", zorder=3))
ax.text(PL + 1.2, 29, "LLM\nclinical\nquery", fontsize=6.5, ha="left", va="center", zorder=4, linespacing=1.0)

# patient examples enter from above
ax.add_patch(FancyArrowPatch((37.0, 54.5), (37.0, 37.8), arrowstyle="-|>", mutation_scale=6, color=E_COL,
                             lw=1.2, ls=(0, (2.5, 1.5)), zorder=4))
ax.add_patch(FancyBboxPatch((34.9, 53.6), 4.2, 4.0, boxstyle="round,pad=0,rounding_size=0.8",
                            facecolor=E_COL, edgecolor="none"))
ax.text(37.0, 55.6, "E", fontsize=7, color="white", ha="center", va="center", fontweight="bold")
ax.text(34.0, 55.6, "Patient examples\n(optional)", fontsize=6, ha="right", va="center", linespacing=1.0)

# three output beams, widening with less specific queries
LAYERS = [
    {"y": 46.5, "half": 8.0, "keys": "C", "label": "1 Unconstrained generation", "col": INPUT["C"][0]},
    {"y": 29.0, "half": 4.6, "keys": "CXHA", "label": "2 Probing", "col": INPUT["X"][0]},
    {"y": 10.5, "half": 6.0, "keys": "CXHA", "label": "3 Forecasting", "col": INPUT["A"][0]},
]
X0, X1 = 56.0, 97.0
for L in LAYERS:
    ax.add_patch(Polygon([APEX, (X0, L["y"] + L["half"] * 0.55), (X0, L["y"] - L["half"] * 0.55)],
                         closed=True, facecolor=L["col"], alpha=0.16, edgecolor="none", zorder=1))
    cx = X0
    for key in L["keys"]:
        if key == "/":
            ax.text(cx + 0.2, L["y"] + L["half"] + 2.1, "or", fontsize=6, ha="left", va="center")
            cx += 3.2
            continue
        ax.add_patch(FancyBboxPatch((cx, L["y"] + L["half"] + 0.6), 3.0, 3.0,
                                    boxstyle="round,pad=0,rounding_size=0.6", facecolor=INPUT[key][0],
                                    edgecolor="none"))
        ax.text(cx + 1.5, L["y"] + L["half"] + 2.1, key, fontsize=6, color="white", ha="center", va="center",
                fontweight="bold")
        cx += 3.5
    ax.text(cx + 0.6, L["y"] + L["half"] + 2.1, L["label"], fontsize=6.5, ha="left", va="center")

rng = np.random.default_rng(4)
t = np.linspace(0, 1, 40)
# population: many starting points, wide spread
lay = LAYERS[0]
for k in range(9):
    y0 = lay["y"] + rng.uniform(-lay["half"] * 0.75, lay["half"] * 0.75)
    x0 = X0 + rng.uniform(0, 4)
    ax.plot(x0 + (X1 - x0) * t, y0 + rng.normal(0, 1.4) * t + 0.6 * np.sin(6 * t + k), color=lay["col"], lw=0.7,
            alpha=0.9)
    ax.plot(x0, y0, "o", ms=1.8, color=lay["col"])
# probing: same starting point, futures with the reference input (grey) and with one input changed
lay = LAYERS[1]
xs0, ys0 = X0 + 3, lay["y"]
SLOPE, SPREAD = 0.075, 0.02
for k in range(5):
    tt = t * (X1 - xs0)
    ax.plot(xs0 + tt, ys0 + (SLOPE + SPREAD * (k - 2)) * tt + 0.2 * np.sin(tt / 3 + k), color="#9A9A9A", lw=0.7)
    ax.plot(xs0 + tt, ys0 - (SLOPE + SPREAD * (k - 2)) * tt + 0.2 * np.sin(tt / 3 + k), color=lay["col"], lw=0.7)
ax.plot(xs0, ys0, "o", ms=3, color=INK, zorder=5)
reach = (SLOPE + 2 * SPREAD) * (X1 - xs0) + 0.6
ax.text(X1, ys0 + reach + 0.6, "Reference", fontsize=6, ha="right", va="bottom")
ax.text(X1, ys0 - reach - 0.6, "One input changed", fontsize=6, ha="right", va="top")
# one patient: history to a decision point, then two narrow fans
lay = LAYERS[2]
hx = np.linspace(X0, X0 + 9, 20)
ax.plot(hx, lay["y"] + 0.9 * np.sin((hx - X0) / 1.6), color=INPUT["H"][0], lw=1.2)
xd, yd = hx[-1], lay["y"] + 0.9 * np.sin(9 / 1.6)
for k in range(6):
    tt = t * (X1 - xd)
    ax.plot(xd + tt, yd + (0.05 * (k - 2.5)) * tt + 0.25 * np.sin(tt / 2 + k), color=INPUT["A"][0], lw=0.7)
tt = t * (X1 - xd)
ax.plot(xd + tt, yd - 0.16 * tt + 0.5 * np.sin(tt / 2.5), color=INK, lw=1.1)
ax.plot(xd, yd, "o", ms=3, color=INK, zorder=5)
ax.text(xd + 3, yd + 3.0, "Generated futures", fontsize=6, ha="left", va="bottom")
ax.text(X1, yd - 0.16 * (X1 - xd) - 1.4, "Observed course", fontsize=6, ha="right", va="top")

legend = (
    "The conditioning prism and the three experiments of the study. A clinical query fixes some of the "
    "patient's context (C; cancer type, stage, molecular profile), current condition (X), history (H) and "
    "treatment (A). (1) Unconstrained generation: the cancer type alone; the model returns a population of "
    "patients with different starting points (six models, 148 cancer types). (2) Probing: from the same starting "
    "point, one input is changed at a time and the futures are compared with the reference (grey). (3) "
    "Forecasting: a real patient's context, condition and history are fixed and the model gives that patient's "
    "possible futures, here under two alternative treatments (GENIE-BPC). Patient examples (E), such as "
    "comparable registry patients, are a separate, optional input that informs the answer without changing the "
    "question. Beam width illustrates the spread of futures; more information does not guarantee narrower, "
    "better calibrated or causally valid futures. Schematic."
)
save_panel(fig, "f1b_prism", legend=legend, sources=["schematic"],
           message="The same model serves three experiments depending on which of C, X, H and A the query "
                   "fixes and varies; examples are a separate input.",
           confirm=["Beams: the three experiments of the study (unconstrained, probing, forecasting).",
                    "Forecasting shows two alternative treatments, but in GENIE-BPC only the observed treatment "
                    "is compared with outcomes; counterfactual arms are untested there (as in the text)."])
