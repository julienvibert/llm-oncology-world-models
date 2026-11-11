"""f1d_whole_course: choosing a strategy concerns the whole course, not survival alone.

Message: two possible courses of patient 1 from the same decision state, one under each strategy, end in
death at the same time, so survival is identical; they differ in time with disease controlled, time to
first progression, treatment exposure, severe side effects and time with poorer physical condition.

Design: both courses start from the same documented state (progression at the
decision); tracks grouped by measure (disease state, treatment, physical condition) with strategy rows
inside each; every readout is computed from the drawn intervals. Constructed illustration, no data.

Run: uv run python scripts/figures/panels/f1d_whole_course.py
"""

import sys
from pathlib import Path

from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402
from _f1_world import SNAME, STRAT, death_mark  # noqa: E402

DEATH = 20.0
# disease state between assessments (every 3 months) is the last documented state, carried forward; both
# courses start from the same documented state at the decision (progression), carried forward until the
# first re-assessment at month 3. One response, maintained until progression.
COURSES = {
    "A": {"state": [(0, 3, "progression"), (3, 15, "response"), (15, 20, "progression")],
          "tx": [(0, 10, "strategy"), (15.5, 18.5, "later")], "tox": [1.5, 4.0, 8.0],
          "ecog": [(0, 1, 1), (1, 10, 2), (10, 15, 1), (15, 18, 2), (18, 20, 3)]},
    "B": {"state": [(0, 3, "progression"), (3, 9, "stable"), (9, 20, "progression")],
          "tx": [(0, 9, "strategy"), (10, 17, "later")], "tox": [13.0],
          "ecog": [(0, 9, 1), (9, 12, 2), (12, 16, 1), (16, 18, 2), (18, 20, 3)]},
}


def readouts(c: dict) -> list[str]:
    controlled = sum(b - a for a, b, s in c["state"] if s in ("response", "stable", "no evidence of disease"))
    on_tx = sum(b - a for a, b, _ in c["tx"])
    poor = sum(b - a for a, b, e in c["ecog"] if e >= 2)
    first_prog = next(a for a, b, s in c["state"] if s == "progression" and a > 0)
    return [f"{DEATH:g}", f"{controlled:g}", f"{first_prog:g}", f"{on_tx:g}", f"{poor:g}", f"{len(c['tox'])}"]


NAMES = {"response": "Response", "stable": "Stable", "progression": "Progression"}
W, H = 78.0, 57.5
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)
TX0, TX1 = 19.0, 70.0          # time axis span in mm
ax = fig.add_axes([TX0 / W, 26.0 / H, (TX1 - TX0) / W, 31.1 / H])
ax.set_xlim(0, 21.0)
ax.set_ylim(0, 10.0)
ax.spines["left"].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0, 6, 12, 18])
ax.set_xlabel("Months after the decision", labelpad=1)

# rows: (measure, strategy) -> y
Y = {("state", "A"): 8.6, ("state", "B"): 7.1, ("tx", "A"): 5.2, ("tx", "B"): 4.2,
     ("ecog", "A"): 1.7, ("ecog", "B"): 0.2}
for key, c in COURSES.items():
    y = Y[("state", key)]
    for a, b, s in c["state"]:
        ax.add_patch(Rectangle((a, y), b - a, 1.2, facecolor=STATE_COLOURS[s], edgecolor="white", lw=0.4))
        if s in NAMES and b - a >= 6:
            ax.text((a + b) / 2, y + 0.6, NAMES[s], fontsize=6, ha="center", va="center",
                    color="white" if s == "progression" else INK)
    for mm in range(3, int(DEATH), 3):
        ax.plot([mm, mm], [y - 0.25, y], color=INK, lw=0.5)
    death_mark(ax, DEATH, y + 0.6, ms=3.8, mew=1.0)
    y = Y[("tx", key)]
    for a, b, kind in c["tx"]:
        if kind == "strategy":
            ax.add_patch(Rectangle((a, y), b - a, 0.6, facecolor=STRAT[key], edgecolor="none"))
        else:
            ax.add_patch(Rectangle((a, y + 0.04), b - a, 0.52, facecolor="white", edgecolor=STRAT[key], lw=0.7,
                                   hatch="//////"))
    for t in c["tox"]:
        ax.plot(t, y + 0.95, "v", ms=3.0, color=INK)
    # physical condition: ECOG as a step line within a band (0 bottom to 3 top)
    y = Y[("ecog", key)]
    xs, ys = [], []
    for a, b, e in c["ecog"]:
        xs += [a, b]
        ys += [y + 0.33 * e, y + 0.33 * e]
    ax.plot(xs, ys, color=STRAT[key], lw=1.2, solid_capstyle="butt")
    for a, b, e in c["ecog"]:
        ax.text((a + b) / 2, y + 0.33 * e + 0.12, f"{e}", fontsize=6, ha="center", va="bottom")
ax.plot([0, 0], [0, 10], color=INK, lw=0.5, ls=(0, (2, 1.5)))

# measure and strategy labels at the left (once)
lab = fig.add_axes([0, 26.0 / H, TX0 / W, 31.1 / H])
lab.set_xlim(0, TX0)
lab.set_ylim(0, 10.0)
lab.axis("off")
for measure, text in (("state", "Disease\nstate"), ("tx", "Treatment"), ("ecog", "ECOG")):
    ya, yb = Y[(measure, "A")], Y[(measure, "B")]
    lab.text(0.3, (ya + yb) / 2 + 0.5, text, fontsize=6, va="center", ha="left", linespacing=0.95)
    for key in "AB":
        lab.add_patch(Rectangle((TX0 - 3.2, Y[(measure, key)] + 0.1), 2.2, 0.75 if measure != "ecog" else 0.9,
                                facecolor=STRAT[key], edgecolor="none"))
        lab.text(TX0 - 2.1, Y[(measure, key)] + (0.47 if measure != "ecog" else 0.55), SNAME[key], fontsize=6,
                 color="white", ha="center", va="center", fontweight="bold")

# readout table
t = fig.add_axes([0, 0, 1, 16.5 / H])
t.set_xlim(0, W)
t.set_ylim(0, 16.5)
t.axis("off")
cols = ["Survival", "Disease\ncontrolled", "Next\nprogression", "On\ntreatment", "ECOG ≥ 2",
        "Severe side\neffects (n)"]
x0, dx = 10.4, 11.2
for j, cname in enumerate(cols):
    t.text(x0 + j * dx + dx / 2, 10.6, cname, fontsize=6, ha="center", va="bottom", linespacing=0.95,
           fontweight="bold" if j == 0 else "normal")
t.plot([x0, x0 + 6 * dx], [10.0, 10.0], color=INK, lw=0.5)
t.text(0.3, 10.6, "Months", fontsize=6, ha="left", va="bottom")
t.add_patch(Rectangle((x0 + 0.8, 0.6), dx - 1.6, 9.1, facecolor="#EFEFEF", edgecolor="none", zorder=0))
for i, key in enumerate("AB"):
    y = 7.4 - i * 4.6
    t.add_patch(Rectangle((x0 - 4.6, y - 1.1), 3.4, 2.2, facecolor=STRAT[key], edgecolor="none"))
    t.text(x0 - 2.9, y, SNAME[key], fontsize=6, color="white", ha="center", va="center", fontweight="bold")
    for j, v in enumerate(readouts(COURSES[key])):
        t.text(x0 + j * dx + dx / 2, y, v, fontsize=6.5, ha="center", va="center",
               fontweight="bold" if j == 0 else "normal")

# key (top right of the time axis area)
k = fig.add_axes([0, 16.2 / H, 1, 3.6 / H])
k.set_xlim(0, W)
k.set_ylim(0, 3.6)
k.axis("off")
k.plot(2.0, 1.6, "v", ms=3.0, color=INK)
k.text(3.2, 1.6, "Severe side effect", fontsize=6, va="center")
k.add_patch(Rectangle((25.0, 1.0), 4.0, 1.2, facecolor="white", edgecolor=INK, lw=0.6, hatch="//////"))
k.text(30.0, 1.6, "Later line", fontsize=6, va="center")
death_mark(k, 45.0, 1.6, ms=3.4, mew=0.9)
k.text(46.6, 1.6, "Death", fontsize=6, va="center")
for i, key in enumerate("AB"):
    k.add_patch(Rectangle((55.0 + i * 3.4, 0.5), 2.6, 2.2, facecolor=STRAT[key], edgecolor="none"))
    k.text(56.3 + i * 3.4, 1.6, SNAME[key], fontsize=6, color="white", ha="center", va="center", fontweight="bold")
k.text(62.4, 1.6, "Strategy", fontsize=6, va="center")


legend = (
    "Choosing a strategy concerns the whole course, not survival alone. One possible course of patient 1 under "
    "each strategy, of the kind shown in panel d. Both start from the same documented state at the decision "
    "(progression; dashed line, month 0) and end in death 20 months later, so survival is identical. Rows marked 1 "
    "and 2, strategies 1 and 2 of panel d. Disease "
    "state, last documented state carried forward from assessments every 3 months (ticks); the "
    "progression documented at the decision is carried forward until the first re-assessment at month 3; course 1 has one response, maintained until progression at month 15; treatment, strategy 1 "
    "(blue) or 2 (pink) and later lines (hatched); triangles, severe side effects; ECOG, Eastern Cooperative "
    "Oncology Group performance status, written on each step (higher is poorer function). Table, readouts computed from the drawn courses, "
    "in months except side effects: survival from the decision; time with disease controlled (response or "
    "stable disease); month of the next progression after the decision; time on treatment; time with ECOG 2 or "
    "more; number of severe side effects. Course 1 has longer disease control with more toxicity and more time "
    "with poorer function; course 2 the reverse. A survival comparison alone would not separate them. "
    "Constructed illustration, not data."
)
save_panel(fig, "f1d_whole_course", legend=legend, sources=["schematic; constructed intervals in this script"],
           message="Two courses from the same decision state with identical survival differ in disease control, "
                   "progression, treatment, toxicity and function.",
           confirm=["Readouts computed from the drawn intervals: survival 20/20, disease controlled "
                    "12/6, next progression 15/9, on treatment 13/16, ECOG >= 2 14/7, severe side effects 3/1 "
                    "months (A/B); check after any edit to COURSES.",
                    "Later treatment lines drawn hatched in the strategy colour (no extra treatment-class colour, "
                    "so that strategy colours are not read as drug classes)."])
