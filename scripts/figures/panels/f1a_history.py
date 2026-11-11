"""f1a_history: a patient is a course; a snapshot hides how the patient arrived.

Message: two constructed patients with metastatic EGFR-mutant lung adenocarcinoma reach the same
displayed state by different courses (patient 1: deep, durable response to osimertinib, then
progression at 16 months; patient 2: primary resistance with deterioration, then a partial response
to chemotherapy at 9 months). The same position carries different histories, disease states and
elapsed times.

Design: linear timelines (treatment, assessments every 3 months
coloured by disease state, landmarks) beside the same two courses in the state space; no background
field or density. Constructed illustration, no data.

Run: uv run python scripts/figures/panels/f1a_history.py
"""

import sys
from pathlib import Path

import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402
from _f1_world import (ANCHOR, HIST, NOT_ASSESSED, P1, P2, first_events, history_assessments,  # noqa: E402
                       history_curve, state_bands, style_state_axes)

W, H = 180.0, 36.0
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)
STYLE = {"Patient 1": "-", "Patient 2": (0, (3.0, 1.4))}

# ------------------------------------------------------------------ timelines
tl = fig.add_axes([11 / W, 8.0 / H, 76 / W, 27.5 / H])
tl.set_xlim(-0.3, 16.6)
tl.set_ylim(-0.2, 2.35)
tl.spines["left"].set_visible(False)
tl.set_yticks([])
tl.set_xticks([0, 3, 6, 9, 12, 15])
tl.set_xlabel("Months since diagnosis", labelpad=1.5)
TX_COL = {"Osimertinib": HIST, "Platinum chemotherapy": "#B5577F"}
rows = {"Patient 1": (P1, 1.3), "Patient 2": (P2, 0.1)}
for name, (p, y) in rows.items():
    m, _, _, st = history_assessments(p)
    end = p["months"][-1]
    # treatment bars with drug names
    for a, b, drug in p["tx"]:
        tl.add_patch(Rectangle((a, y + 0.5), b - a, 0.28, facecolor=TX_COL[drug], edgecolor="white", lw=0.5))
        tl.text((a + b) / 2, y + 0.64, drug if b - a > 4.5 else "Osim.", fontsize=6, color="white",
                ha="center", va="center")
    # continuous disease-state band (last documented state carried forward); assessments as ticks
    for a, b, s in state_bands(m, st, 0.0, end):
        tl.add_patch(Rectangle((a, y + 0.08), b - a, 0.28, facecolor=STATE_COLOURS[s], edgecolor="none"))
    for mm in m:
        tl.plot([mm, mm], [y + 0.0, y + 0.08], color=INK, lw=0.6)
    tl.plot(0, y + 0.22, "o", ms=4.0, mfc="white", mec=HIST, mew=1.0, zorder=5)
    tl.plot(end, y + 0.22, "o", ms=5.0, color=INK, zorder=6)
    tl.plot(end, y + 0.22, "o", ms=8.0, mfc="none", mec=INK, mew=0.7, zorder=6)
    tl.text(-0.6, y + 0.4, name, fontsize=6.5, ha="right", va="center")
    tl.text(end + 0.45, y + 0.22, f"{end:g} months", fontsize=6, ha="left", va="center")
# event labels, once each, under the band
for name, (p, y) in rows.items():
    m, _, _, st = history_assessments(p)
    for ev, mm in first_events(m, st).items():
        tl.text(mm, y - 0.04, ev.capitalize(), fontsize=6, ha="center", va="top")

# ------------------------------------------------------------------ state space
ax = fig.add_axes([116 / W, 8.0 / H, 63.5 / W, 27.5 / H])
style_state_axes(ax, ylabel=False, fs=6)
ax.text(0.15, 3.8, "↑ State dimension 2: poorer condition", fontsize=6, ha="left", va="top")
for name, (p, _) in rows.items():
    t, hb, hc = history_curve(p)
    ax.plot(hb, hc, color=HIST, lw=1.4, ls=STYLE[name], zorder=3, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
    m, b, c, st = history_assessments(p)
    for ev, mm in first_events(m, st).items():
        j = list(m).index(mm)
        ax.plot(b[j], c[j], "o", ms=3.8, mfc=STATE_COLOURS[ev], mec="white", mew=0.4, zorder=5)
    ax.plot(hb[0], hc[0], "o", ms=4.0, mfc="white", mec=HIST, mew=1.0, zorder=5)
ax.text(P1["burden"][0], P1["cond"][0] + 0.2, "Patient 1", fontsize=6, ha="center", va="bottom")
ax.text(P2["burden"][0] - 0.2, P2["cond"][0] + 0.0, "Patient 2", fontsize=6, ha="right", va="center")
ax.plot(*ANCHOR, "o", ms=5.0, color=INK, zorder=7)
ax.plot(*ANCHOR, "o", ms=8.0, mfc="none", mec=INK, mew=0.7, zorder=7)
ax.annotate("Same displayed state", xy=tuple(ANCHOR), xytext=(ANCHOR[0] + 2.9, 0.62), fontsize=6, ha="center",
            va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkB=4), zorder=8)

# key between the two views
k = fig.add_axes([96.5 / W, 8.0 / H, 17 / W, 27.5 / H])
k.set_xlim(0, 1)
k.set_ylim(0, 1)
k.axis("off")
items = [("response", "Response"), ("stable", "Stable"), ("progression", "Progression"),
         (NOT_ASSESSED, "Not yet\nassessed")]
dy, EXTRA = 0.112, 0.05  # extra space after the two-line entry
for i, (sk, lab) in enumerate(items):
    k.add_patch(Rectangle((0.0, 0.95 - i * dy - 0.03), 0.15, 0.06, facecolor=STATE_COLOURS[sk], edgecolor="none"))
    k.text(0.2, 0.95 - i * dy, lab, fontsize=6, va="center", linespacing=0.9)
k.plot(0.075, 0.95 - 4 * dy - EXTRA, "o", ms=3.6, mfc="white", mec=HIST, mew=0.9)
k.text(0.2, 0.95 - 4 * dy - EXTRA, "Diagnosis", fontsize=6, va="center")
k.plot(0.075, 0.95 - 5 * dy - EXTRA, "o", ms=4.0, color=INK)
k.plot(0.075, 0.95 - 5 * dy - EXTRA, "o", ms=6.5, mfc="none", mec=INK, mew=0.6)
k.text(0.2, 0.95 - 5 * dy - EXTRA, "Now", fontsize=6, va="center")
for i, name in enumerate(rows):
    yy = 0.95 - (6 + i) * dy - EXTRA
    k.plot([0.0, 0.15], [yy, yy], color=HIST, lw=1.3, ls=STYLE[name])
    k.text(0.2, yy, name, fontsize=6, va="center")

legend = (
    "A patient is a course, not a snapshot. Two constructed patients with metastatic EGFR-mutant lung "
    "adenocarcinoma, shown as timelines (left) and as paths in the same two-dimensional clinical state space "
    "(right). Patient 1 (solid) had bulky disease at diagnosis, a deep and durable response to osimertinib and "
    "progression at month 15; patient 2 (dashed) had limited disease, progressed within 3 months with clinical "
    "deterioration and responded to platinum chemotherapy. Both reach the same displayed state (black ring), "
    "after 16 and 10 months, with different histories and different documented disease states (progression and "
    "partial response). Timeline bands, documented disease state, carried forward from each assessment (ticks, "
    "every 3 months) to the next: response, at least 30% below the burden at the start of the current "
    "treatment, maintained until progression, at least 20% above the lowest burden; grey, before the first "
    "assessment. Coloured dots on the paths, first documented response and first progression; open circles, "
    "diagnosis. Bars, treatment. The state dimensions are illustrative (for "
    "example, disease burden and physical condition); the paper's state map uses composite axes oriented the "
    "same way, which combine tumour and host features and are not validated clinical scales. Constructed "
    "illustration, not data."
)
save_panel(fig, "f1a_history", legend=legend, sources=["schematic; scripts/figures/panels/_f1_world.py"],
           message="Two patients reach the same displayed state by different courses; a snapshot hides the "
                   "history, the documented state and the elapsed time.",
           confirm=["Constructed EGFR-mutant scenario (shared by panels a-d): "
                    "both histories, including chemotherapy at month 4 for primary resistance to osimertinib."])
