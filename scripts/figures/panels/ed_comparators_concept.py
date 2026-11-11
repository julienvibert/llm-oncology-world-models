"""ed_comparators_concept: who is comparable depends on the question, and the comparison is updated over time.

Message: comparability depends on information the two-dimensional map does not show. For one patient,
the comparable patients for prognosis are those in a similar state with a similar relevant history; for
tolerance of a specified treatment they are those with similar physical condition and no prior severe
toxicity; six months later the patient's state AND history have changed, so the comparison is updated:
some comparators stay, some leave, some join.

Design: each view pairs a map of current
states (dots only; comparators ringed in the question's colour and linked to patient 1; others pale)
with a timeline of the course so far of patient 1, its comparators and one instructive exclusion, as in
Figure 1a. Letters are kept across views. Criteria are illustrative proposals, not validated prediction
rules. Constructed illustration: no data and no model output.

Run: uv run python scripts/figures/panels/ed_comparators_concept.py
"""

import sys
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402
from _f1_world import XLABEL, YLABEL  # noqa: E402

PROG = "#2F6DB5"      # comparable for prognosis (blue)
TOL = "#6A3D9A"       # comparable for tolerance of platinum chemotherapy (violet)
OTHER = "#CFC8DC"     # not comparable for this question (pale lavender)
ST = {"N": STATE_COLOURS["unresolved"], "R": STATE_COLOURS["response"], "S": STATE_COLOURS["stable"], "P": STATE_COLOURS["progression"]}
XL, YL = (0.0, 10.0), (0.0, 4.2)

# Constructed patients, all metastatic EGFR-mutant lung adenocarcinoma treated first with osimertinib.
# history: documented state in successive 6-month periods since diagnosis (R response, S stable,
# P progression); tox: prior severe toxicity. "1" is patient 1 of Figure 1.
M16 = {"1": (5.0, 1.8, "RRP", False),
       "a": (4.2, 1.35, "RRP", False), "b": (5.9, 2.25, "RRP", True), "c": (5.75, 1.45, "PRS", False),
       "d": (2.7, 2.0, "RRP", False), "e": (7.7, 1.9, "PPP", False), "f": (2.0, 3.4, "RSP", True),
       "g": (8.3, 3.35, "PSP", False), "h": (4.5, 2.65, "RRP", False)}
M22 = {"1": (3.0, 1.2, "RRPR", False),
       "a": (3.45, 0.85, "RRPR", False), "b": (6.9, 2.8, "RRPP", True), "c": (6.6, 1.95, "PRSP", False),
       "d": (2.0, 1.55, "RRPR", False), "e": (8.5, 2.5, "PPPP", False), "f": (2.6, 3.7, "RSPP", True),
       "g": (8.9, 3.75, "PSPP", False), "h": (3.55, 2.4, "RRPS", False)}
PROG16 = {"a", "b", "h"}          # similar state and similar history (durable response, then progression)
TOL16 = {"a", "c", "d", "e"}      # similar physical condition and no prior severe toxicity
PROG22 = {"a", "d", "h"}          # similar updated state and history: a and h stay, b leaves, d joins

W, H = 180.0, 76.0
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)
AW, GAP, X0 = 52.0, 7.5, 7.0
MAP_Y0, MAP_H = 40.0, 31.0          # map axes (mm from bottom)
TL_Y0, TL_H = 14.5, 19.0            # timeline axes
views = [("i", "Prognosis on chemotherapy, month 16", PROG, M16, PROG16, ["1", "a", "b", "h", "c"], 16),
         ("ii", "Tolerance of platinum chemotherapy, month 16", TOL, M16, TOL16, ["1", "a", "c", "d", "e", "b"], 16),
         ("iii", "Prognosis, month 22: updated", PROG, M22, PROG22, ["1", "a", "d", "h", "b"], 22)]
NOTE = {"i": {"c": "close, but early\nprogression"},
        "ii": {"b": "earlier kidney\ninjury"},
        "iii": {"b": "leaves", "d": "joins"}}


def map_axes(k: int):
    ax = fig.add_axes([(X0 + k * (AW + GAP)) / W, MAP_Y0 / H, AW / W, MAP_H / H])
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


for k, (num, title, col, pos, chosen, rows, month) in enumerate(views):
    ax = map_axes(k)
    ax.set_xlabel(XLABEL, fontsize=6, labelpad=1.5)
    if k == 0:
        ax.set_ylabel(YLABEL, fontsize=6, labelpad=2)
    fig.text((X0 + k * (AW + GAP)) / W, (MAP_Y0 + MAP_H + 1.6) / H, f"{num}  {title}", fontsize=6.5,
             fontweight="bold", ha="left", va="bottom")
    xi, yi = pos["1"][:2]
    for key in chosen:
        x, y = pos[key][:2]
        ax.add_patch(FancyArrowPatch((xi, yi), (x, y), arrowstyle="-", color=col, lw=0.9, alpha=0.6,
                                     connectionstyle="arc3,rad=0.15", zorder=1))
    if month == 22:
        ax.plot(*M16["1"][:2], "o", ms=6.0, mfc="white", mec=INK, mew=0.8, zorder=6)
        ax.add_patch(FancyArrowPatch(M16["1"][:2], M22["1"][:2], arrowstyle="-|>", mutation_scale=7,
                                     color=INK, lw=0.9, shrinkA=4, shrinkB=5, zorder=6))
    for key, (x, y, _hist, _tox) in pos.items():
        if key == "1":
            ax.plot(x, y, "o", ms=7.5, color=INK, zorder=8)
            ax.text(x, y - 0.33, "1", fontsize=6.5, fontweight="bold", ha="center", va="top", zorder=9)
        elif key in chosen:
            ax.plot(x, y, "o", ms=6.4, mfc="white", mec=col, mew=1.6, zorder=7)
            ax.text(x, y - 0.3, key, fontsize=6, ha="center", va="top", zorder=9)
        else:
            ax.plot(x, y, "o", ms=4.6, mfc=OTHER, mec="none", zorder=5)
            ax.text(x, y - 0.27, key, fontsize=6, ha="center", va="top", color="#6F6880", zorder=9)

    # timeline: the course so far of patient 1, its comparators and the instructive exclusion
    tl = fig.add_axes([(X0 + k * (AW + GAP)) / W, TL_Y0 / H, AW / W, TL_H / H])
    tl.set_xlim(-3.2, 29.0)
    tl.set_ylim(-0.6, len(rows) - 0.3)
    tl.set_yticks([])
    tl.set_xticks([0, 6, 12, 16, 22] if month == 22 else [0, 6, 12, 16])
    tl.tick_params(axis="x", labelsize=6, length=2, pad=1)
    tl.spines["left"].set_visible(False)
    tl.set_xlabel("Months since diagnosis", fontsize=6, labelpad=1)
    for r, key in enumerate(rows):
        y = len(rows) - 1 - r
        hist, tox = pos[key][2], pos[key][3]
        edges = [3, 6, 12, 16, 22][: len(hist) + 1]          # first assessment at month 3
        bands: list[list] = [[0, 3, "N"]]
        for (a0, a1), st in zip(zip(edges, edges[1:]), hist, strict=True):
            if bands and bands[-1][2] == st:
                bands[-1][1] = a1
            else:
                bands.append([a0, a1, st])
        for a0, a1, st in bands:
            tl.add_patch(Rectangle((a0, y - 0.28), a1 - a0, 0.56, facecolor=ST[st], edgecolor="white", lw=0.6))
        if tox:
            tl.plot(3.0, y + 0.42, marker="v", ms=3.0, color=INK)
        if key == "1":
            tl.plot(-1.6, y, "o", ms=5.0, color=INK)
        elif key in chosen:
            tl.plot(-1.6, y, "o", ms=4.6, mfc="white", mec=col, mew=1.3)
        else:
            tl.plot(-1.6, y, "o", ms=3.8, mfc=OTHER, mec="none")
        tl.text(-2.6, y, key, fontsize=6, ha="right", va="center", fontweight="bold" if key == "1" else "normal")
        note = NOTE[num].get(key)
        if note:
            tl.text(edges[len(hist)] + 0.6, y, note, fontsize=6, ha="left", va="center", color=INK,
                    linespacing=0.95)
    tl.axvline(month, color=INK, lw=0.6, ls=(0, (2, 1.5)), ymin=0.02, ymax=0.98)

# key
k1 = [Line2D([], [], marker="o", ls="none", ms=6.0, color=INK, label="Patient 1"),
      Line2D([], [], marker="o", ls="none", ms=5.0, mfc="white", mec=INK, label="Patient 1 at month 16"),
      Line2D([], [], marker="o", ls="none", ms=5.5, mfc="white", mec=PROG, mew=1.5,
             label="Selected comparators for prognosis"),
      Line2D([], [], marker="o", ls="none", ms=5.5, mfc="white", mec=TOL, mew=1.5,
             label="Selected comparators for tolerance of platinum"),
      Line2D([], [], marker="o", ls="none", ms=4.6, mfc=OTHER, mec="none", label="Other patients")]
fig.legend(handles=k1, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=5, fontsize=6, frameon=False,
           handletextpad=0.4, columnspacing=1.2)
kx = fig.add_axes([0, 0, 1, 3.6 / H])
kx.set_xlim(0, W)
kx.set_ylim(0, 3.6)
kx.axis("off")
kx.text(14.0, 1.8, "Course so far, documented state:", fontsize=6, va="center")
xx = 52.0
for st, lab in (("N", "Before first assessment"), ("R", "Response"), ("S", "Stable"),
                ("P", "Progression")):
    kx.add_patch(Rectangle((xx, 1.2), 3.0, 1.3, facecolor=ST[st], edgecolor="none"))
    kx.text(xx + 3.8, 1.8, lab, fontsize=6, va="center")
    xx += 33.0 if st == "N" else 19.0
kx.plot(xx + 1.0, 1.8, marker="v", ms=3.2, color=INK)
kx.text(xx + 2.6, 1.8, "Severe kidney injury", fontsize=6, va="center")

legend = (
    "Which patients are selected as comparators depends on the clinical question, and the selection is "
    "updated as the course evolves. Constructed patients, all with metastatic EGFR-mutant lung "
    "adenocarcinoma first treated with osimertinib. Top, current state in the clinical state space of "
    "Fig. 1 (axes illustrative); letters identify the same patients in every view; thin lines join "
    "patient 1 to the comparators selected for that question, and the black arrow in iii shows patient "
    "1's own movement. Bottom, the course so far of patient 1, its selected comparators and one "
    "instructive exclusion: documented disease state since diagnosis, as in Fig. 1a, grey before the "
    "first assessment (triangles, severe kidney injury; dashed line, decision time). i, For prognosis on "
    "second-line chemotherapy at month 16, comparators are selected for a similar state and a similar "
    "history (durable response, then progression: a, b, h); c is close on the map but progressed early. "
    "ii, For tolerance of platinum chemotherapy at the same time, comparators are selected for similar "
    "physical condition and no earlier severe kidney injury, relevant to platinum (a, c, d, e); b is not "
    "selected. iii, Six months later patient 1 has responded to second-line chemotherapy (open circle, "
    "position at month 16): its state and its history have changed. For prognosis, a and h remain "
    "selected, d joins and b leaves. These selection criteria are illustrative proposals, not validated "
    "or sufficient conditions, and closeness on a two-dimensional map does not make two patients "
    "clinically interchangeable. Constructed illustration, not data."
)
save_panel(fig, "ed_comparators_concept", legend=legend, sources=["schematic"],
           message="Comparable patients depend on the question and on history, and are updated over time.",
           confirm=["Illustrative criteria: prognosis = similar state and similar relevant history; tolerance "
                    "of platinum chemotherapy = similar physical condition and no prior severe toxicity.",
                    "Membership changes at month 22: a and h stay, b leaves, d joins (constructed).",
                    "Host: Extended Data Fig. 3, as a conceptual panel distinct from the empirical history "
                    "analysis."])
