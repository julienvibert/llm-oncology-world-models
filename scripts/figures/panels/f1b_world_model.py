"""f1b_world_model: one fixed patient record enters a candidate oncology world model, which returns
possible courses under two specified strategies, from the same decision anchor.

Message: given this patient's context, current state and prior course at one decision time, a world
model forecasts distributions of whole courses under each strategy; the two fans start at the same
anchor and show trade-offs (under A, deeper disease control with poorer condition during treatment and
some early failures; under B, condition preserved but shorter control and more deaths).

Design: record card -> prism -> two small
multiples with identical axes, one per strategy, each with the recorded history and the same decision
ring; assessment dots only at 6, 12, 18 and 24 months; deaths as crosses at path ends; per-fan counts of
drawn courses under each plot. Constructed illustration, no data.

Run: uv run python scripts/figures/panels/f1b_world_model.py
"""

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402
from _f1_world import (ANCHOR, EXAMPLES, FAN_KINDS, HIST, NOT_ASSESSED, P1, SNAME, STRAT, XLABEL, assess,  # noqa: E402
                       death_mark, fan, first_events, history_assessments, history_curve, state_bands,
                       style_state_axes)

W, H = 180.0, 76.0
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)

cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(0, H)
cv.axis("off")
cv.set_zorder(0)


def badge(x: float, y: float, letter: str, col: str = INK) -> None:
    """Input letter of the C, X, H, A, E notation (Methods), drawn as a small ring."""
    cv.add_patch(Circle((x, y), 1.55, facecolor="white", edgecolor=col, lw=0.8, zorder=6))
    cv.text(x, y - 0.05, letter, fontsize=6.5, ha="center", va="center", fontweight="bold", zorder=7)

# patient record card ----------------------------------------------------------------------------
CX0, CX1, CY0, CY1 = 0.6, 44.0, 31.0, 75.4
cv.add_patch(FancyBboxPatch((CX0, CY0), CX1 - CX0, CY1 - CY0, boxstyle="round,pad=0,rounding_size=1.6",
                            facecolor="#FBF6F8", edgecolor=HIST, lw=0.8))
cv.text(CX0 + 2.0, CY1 - 2.4, "Patient record", fontsize=7, fontweight="bold", va="top")
rows = [("Context", "Lung adenocarcinoma, EGFR\nexon 19 deletion, metastatic"),
        ("Current state", "Progression in lung and liver,\nECOG performance status 1")]
y = CY1 - 8.0
for (head, body), letter in zip(rows, "CX"):
    cv.text(CX0 + 2.0, y, head, fontsize=6, va="top", fontweight="bold")
    badge(CX1 - 3.0, y - 1.0, letter, HIST)
    cv.text(CX0 + 2.0, y - 2.8, body, fontsize=6, va="top", linespacing=1.0)
    y -= 10.2
cv.text(CX0 + 2.0, y, "Prior course", fontsize=6, va="top", fontweight="bold")
badge(CX1 - 3.0, y - 1.0, "H", HIST)
TL0, TL1, TLY = CX0 + 3.0, CX1 - 4.0, CY0 + 6.2
mx = lambda m: TL0 + (TL1 - TL0) * m / 16.0  # noqa: E731
cv.add_patch(Rectangle((mx(0), TLY + 2.6), mx(16) - mx(0), 2.4, facecolor=HIST, edgecolor="none"))
cv.text(mx(8), TLY + 3.8, "Osimertinib", fontsize=6, color="white", ha="center", va="center")
cv.plot([mx(0), mx(16)], [TLY, TLY], color=INK, lw=0.5)
m, _, _, st = history_assessments(P1)
for a, b, s in state_bands(m, st, 0.0, 16.0):
    cv.add_patch(Rectangle((mx(a), TLY + 0.4), mx(b) - mx(a), 1.4, facecolor=STATE_COLOURS[s], edgecolor="none"))
for mm in m:
    cv.plot([mx(mm), mx(mm)], [TLY, TLY + 0.4], color=INK, lw=0.5)
cv.plot(mx(0), TLY + 1.1, "o", ms=3.0, mfc="white", mec=HIST, mew=0.8)
for mm in (0, 6, 12):
    cv.plot([mx(mm), mx(mm)], [TLY, TLY - 0.7], color=INK, lw=0.5)
    cv.text(mx(mm), TLY - 1.1, f"{mm}", fontsize=6, ha="center", va="top")
cv.plot([mx(16), mx(16)], [TLY - 0.9, TLY + 6.0], color=INK, lw=0.9)
cv.plot(mx(16), TLY + 1.1, "o", ms=4.6, color=INK)
cv.text(mx(16) + 0.4, TLY - 1.1, "16", fontsize=6, ha="center", va="top")
cv.text(mx(9.0), TLY - 4.2, "Months since diagnosis", fontsize=6, ha="center", va="center")

# prism: candidate oncology world model ----------------------------------------------------------
PL, PT, PB = 52.0, 65.0, 33.0
APEX = (71.5, 49.0)
cv.add_patch(Polygon([(PL, PT), (PL, PB), APEX], closed=True, facecolor="#F3F0F8", edgecolor=INK, lw=0.9,
                     zorder=3))
bands = ["#882255", "#2A6FBF", "#6DB3A6", "#DFC782", "#D55E00", "#CC79A7"]
for k, col in enumerate(bands):
    f0, f1 = k / len(bands), (k + 1) / len(bands)
    cv.add_patch(Polygon([(PL, PT - (PT - PB) * f0), (PL, PT - (PT - PB) * f1), APEX], closed=True,
                         facecolor=col, alpha=0.10, edgecolor="none", zorder=3))
cv.text(PL + 0.9, 49.0, "Candidate\noncology\nworld model", fontsize=6.5, ha="left", va="center", zorder=4,
        linespacing=1.0)
cv.add_patch(FancyArrowPatch((CX1 + 0.4, 53.0), (PL - 0.3, 53.0), arrowstyle="-|>", mutation_scale=7,
                             color=HIST, lw=1.8, zorder=2))
# strategies enter from below
for key, sy, tgt in (("A", 23.5, (60.0, 37.6)), ("B", 16.5, (65.0, 42.4))):
    cv.add_patch(FancyBboxPatch((44.5, sy - 2.2), 15.5, 4.4, boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor="white", edgecolor=STRAT[key], lw=1.2, zorder=4))
    cv.add_patch(Rectangle((45.6, sy - 1.0), 2.0, 2.0, facecolor=STRAT[key], edgecolor="none", zorder=5))
    cv.text(48.4, sy, f"Strategy {SNAME[key]}", fontsize=6.5, ha="left", va="center", zorder=5)
    cv.add_patch(FancyArrowPatch((60.1, sy), tgt, arrowstyle="-|>", mutation_scale=6, color=STRAT[key], lw=1.1,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=2", zorder=2))
badge(42.2, 20.0, "A")
# optional patient examples from above
cv.add_patch(FancyArrowPatch((60.0, 74.5), (60.0, 61.5), arrowstyle="-|>", mutation_scale=6, color=EXAMPLES,
                             lw=1.1, ls=(0, (2.5, 1.5)), zorder=4))
cv.text(61.4, 74.6, "Patient examples\n(optional)", fontsize=6, ha="left", va="top", linespacing=1.0)
badge(57.6, 72.6, "E", EXAMPLES)
# output arrow to the two forecasts
cv.add_patch(FancyArrowPatch((APEX[0] + 0.6, APEX[1]), (78.0, APEX[1]), arrowstyle="-|>", mutation_scale=8,
                             color=INK, lw=1.4, zorder=2))

# key (bottom left)
KY = 24.5
for i, (sk, lab) in enumerate((("response", "First response"), ("stable", "Stable disease"),
                                ("progression", "First progression"))):
    cv.add_patch(Rectangle((1.0, KY - i * 4.0 - 0.8), 3.0, 1.6, facecolor=STATE_COLOURS[sk], edgecolor="none"))
    if sk != "stable":
        cv.plot(2.5, KY - i * 4.0, "o", ms=3.4, mfc=STATE_COLOURS[sk], mec="white", mew=0.4)
    cv.text(4.6, KY - i * 4.0, lab, fontsize=6, va="center")
cv.add_patch(Rectangle((1.0, KY - 3 * 4.0 - 0.8), 3.0, 1.6, facecolor=STATE_COLOURS[NOT_ASSESSED], edgecolor="none"))
cv.text(4.6, KY - 3 * 4.0, "Not yet assessed", fontsize=6, va="center")
cv.plot([1.0, 4.0], [KY - 4 * 4.0] * 2, color=HIST, lw=1.4)
cv.text(4.6, KY - 4 * 4.0, "Recorded course", fontsize=6, va="center")
death_mark(cv, 25.0, KY - 2 * 4.0, ms=3.6, mew=0.9)
cv.text(27.6, KY - 2 * 4.0, "Death", fontsize=6, va="center")
cv.plot(25.0, KY - 3 * 4.0, "o", ms=3.6, mfc="white", mec=HIST, mew=0.9)
cv.text(27.6, KY - 3 * 4.0, "Diagnosis", fontsize=6, va="center")
cv.plot(25.0, KY - 4 * 4.0, "o", ms=4.4, color=INK)
cv.plot(25.0, KY - 4 * 4.0, "o", ms=6.8, mfc="none", mec=INK, mew=0.7)
cv.text(27.6, KY - 4 * 4.0, "Decision time", fontsize=6, va="center")

# ---------------------------------------------------------------- two forecasts, identical axes
FANS = {"A": fan("A", FAN_KINDS["A"], seed=10), "B": fan("B", FAN_KINDS["B"], seed=71)}
PX = {"A": 80.5, "B": 130.5}
PW, PY0, PY1 = 48.5, 22.0, 70.5
t_h, hb, hc = history_curve(P1)


def summary(courses: list[dict]) -> tuple[int, float, int]:
    """Counts over the drawn courses: deaths by 24 months; median month of first progression or death
    (assessments every 3 months; courses without either counted at 24); courses with condition at least
    0.3 units poorer at 3 months than at the decision."""
    deaths, events, worse = 0, [], 0
    for c in courses:
        m = np.arange(3, c["t"][-1] + 1e-9, 3.0)
        st = assess(np.interp(m, c["t"], c["burden"]), ref=float(ANCHOR[0]))
        prog = next((mm for mm, s in zip(m, st) if s == "progression"), None)
        cand = [v for v in (prog, c["death_t"]) if v is not None]
        events.append(min(cand) if cand else 24.0)
        deaths += int(c["died"])
        worse += int(np.interp(3.0, c["t"], c["cond"]) - ANCHOR[1] >= 0.3)
    return deaths, float(np.median(events)), worse


for key in ("A", "B"):
    ax = fig.add_axes([PX[key] / W, PY0 / H, PW / W, (PY1 - PY0) / H])
    style_state_axes(ax, xlabel=False, ylabel=False)
    ax.patch.set_alpha(0)
    if key == "A":
        ax.text(0.15, 3.8, "↑ State dimension 2:\n   poorer condition", fontsize=6, ha="left", va="top",
                linespacing=1.0)
    ax.plot(hb, hc, color=HIST, lw=1.3, zorder=2, solid_capstyle="round")
    ax.plot(hb[0], hc[0], "o", ms=3.6, mfc="white", mec=HIST, mew=0.9, zorder=3)
    for cr in FANS[key]:
        ax.plot(cr["burden"], cr["cond"], color=STRAT[key], lw=1.05, alpha=0.9, zorder=4, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=1.9, foreground="white", alpha=0.7), pe.Normal()])
        all_m = np.arange(3, cr["t"][-1] + 1e-9, 3.0)
        if cr["died"]:
            all_m = all_m[all_m < cr["t"][-1] - 0.4]
        st_all = assess(np.interp(all_m, cr["t"], cr["burden"]), ref=float(ANCHOR[0]))
        for ev, mm in first_events(all_m, st_all).items():
            ax.plot(np.interp(mm, cr["t"], cr["burden"]), np.interp(mm, cr["t"], cr["cond"]), "o", ms=3.4,
                    mfc=STATE_COLOURS[ev], mec="white", mew=0.4, zorder=6)
        if cr["died"]:
            death_mark(ax, cr["burden"][-1], cr["cond"][-1], ms=3.8, mew=1.0)
    ax.plot(*ANCHOR, "o", ms=5.5, color=INK, zorder=8)
    ax.plot(*ANCHOR, "o", ms=8.5, mfc="none", mec=INK, mew=0.8, zorder=8)
    # strategy label above the plot
    cv.add_patch(Rectangle((PX[key] + 0.5, PY1 + 1.6), 2.4, 2.4, facecolor=STRAT[key], edgecolor="none"))
    cv.text(PX[key] + 3.8, PY1 + 2.8, f"Strategy {SNAME[key]}: possible courses", fontsize=6.5, va="center")
    # per-fan counts of drawn courses
    d, med, worse = summary(FANS[key])
    n = len(FANS[key])
    lines = [("Deaths by 24 months", f"{d} of {n}"),
             ("Median months to progression or death", f"{med:g}"),
             ("Early decline in condition (3 months)", f"{worse} of {n}")]
    for i, (lab, val) in enumerate(lines):
        yy = 11.6 - i * 3.6
        cv.text(PX[key] + 0.5, yy, lab, fontsize=6, va="center", ha="left")
        cv.text(PX[key] + PW, yy, val, fontsize=6.5, va="center", ha="right", fontweight="bold")
    cv.plot([PX[key] + 0.5, PX[key] + PW], [14.2, 14.2], color=STRAT[key], lw=0.8)
cv.text((PX["A"] + PX["B"] + PW) / 2, 18.2, XLABEL, fontsize=6.5, ha="center", va="center")

legend = (
    "A candidate oncology world model forecasts whole courses under specified strategies. The record of patient 1 "
    "(panel c) at one decision time (month 16: context C, current state X and prior course H) enters the model "
    "together with one specified treatment strategy A at a time; optional patient examples E, such as comparable "
    "registry patients, are a separate input (letters as in Methods). Right, 10 possible courses under strategy 1 "
    "(blue) and under strategy 2 (pink), drawn on "
    "identical axes from the same decision state (black ring), with the recorded course (wine; open circle, "
    "diagnosis), followed for 24 months or until death (crosses). Dots, first documented response (at least 30% "
    "below the burden at the decision) and first progression (at least 20% above the lowest burden) on each "
    "course, from assessments every 3 months; a response is maintained until progression. In the record, the "
    "band shows the documented disease state carried forward between assessments (ticks). Under strategy 1 most courses respond, with poorer "
    "physical condition during treatment, and two fail early; under strategy 2 condition is preserved but progression "
    "comes earlier and more courses end in death. Below each plot, counts over the 10 drawn courses (progression "
    "from 3-monthly assessments; courses with neither event counted at 24 months; poorer condition, at least 0.3 "
    "units higher on state dimension 2 than at the decision); they describe the drawing, not a treatment effect. "
    "Axes as in panel c. The panel does not imply that current LLMs can be used to choose a treatment; whether "
    "their forecasts can be trusted is what Figs 3-6 test. Constructed illustration, not data or model output."
)
save_panel(fig, "f1b_world_model", legend=legend, sources=["schematic; scripts/figures/panels/_f1_world.py"],
           message="One fixed patient record at one decision time; a world model returns distributions of whole "
                   "courses under strategies 1 and 2 from the same anchor, with trade-offs and deaths.",
           confirm=["Clinical scenario: constructed metastatic EGFR exon 19-deleted lung adenocarcinoma at first "
                    "progression on osimertinib (month 16); strategies kept neutral and displayed as 1 and 2 (A is the treatment input in the notation); "
                    "naming them would invite reading the fans as effect estimates.",
                    "Strategy colours 1 blue, 2 pink follow the schematic convention in common.py (CHEMO = "
                    "treatment B); the prior osimertinib is drawn in the history colour (wine).",
                    "Course mix is declared (1: 2 early failures of 10; 2: 2 deeper responses of 10; fixed seeds). "
                    "Per-fan counts are computed in this script from the drawn courses: deaths 5/10 vs 7/10, median "
                    "months to progression or death 15 vs 13.5, poorer condition at 3 months 6/10 vs 0/10 (1 vs 2); "
                    "the 0.3-unit threshold for poorer condition is a toy-scale choice."])
