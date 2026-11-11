"""Panel f4a_design: design of the four-arm experiment (arm colours, one text size).

Message: the same 450 patients and treatment starts are given to the model in four arms that add
care setting, then 20 generic or 20 matched donors; ten futures per patient and arm, compared with
the observed future.
Data: none plotted; n from population_composition_scores_20261012.csv (cohort_n); wording follows manuscript/DRAFT.md.
Run: uv run python scripts/figures/panels/f4a_design.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, F_REL, INK, GREY, final_table  # noqa: E402
from common import panel, save_panel  # noqa: E402
from matplotlib.patches import FancyBboxPatch

pc = final_table("population_composition_scores_20261012.csv")
n = int(pc.cohort_n.iloc[0])
assert n == 450 and set(pc.cohort_n) == {450} and set(pc.cohort_n_NSCLC) == {150}

W, H = 180, 36
fig, ax = panel(W, H)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
FS = 7

desc = ["Record and treatment only",
        "Record, treatment and\ncare setting",
        "Also 20 donors: same\ncancer, comparable era",
        "Also 20 donors: same treatment\nmodality, similar characteristics"]
txt_col = [INK, INK, "white", "white"]
ys = [H - 4.6 - i * 8.6 for i in range(4)]
ymid = sum(ys) / 4

# left block
ax.text(0, ymid + 8.5, "450 GENIE-BPC patients", fontsize=FS, va="center", ha="left")
ax.text(0, ymid + 2.5, "150 lung, 150 breast,\n150 colorectal", fontsize=FS, va="center", ha="left", linespacing=1.15)
ax.text(0, ymid - 5.5, "Same patient record and\ntreatment being started", fontsize=FS, va="center", ha="left", linespacing=1.15)
ax.text(0, ymid - 12, "Later records withheld", fontsize=FS, va="center", ha="left")

xs, xp, wp = 44, 50, 38       # trunk x, pill x, pill width
xt, xv = 138, 142             # right ticks, vertical join
ax.plot([40, xs], [ymid, ymid], color=GREY, lw=0.6)
for (k, lab, col, _), y, d, tc in zip(ARMS, ys, desc, txt_col):
    ax.plot([xs, xp], [ymid, y], color=col, lw=1.0, solid_capstyle="butt")
    ax.add_patch(FancyBboxPatch((xp, y - 3.3), wp, 6.6, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=col, ec=INK, lw=0.5))
    ax.text(xp + 2, y, lab, fontsize=FS, va="center", ha="left", color=tc)
    ax.text(xp + wp + 3, y, d, fontsize=FS, va="center", ha="left", linespacing=1.1)
    ax.plot([xt, xv], [y, y], color=col, lw=1.0, solid_capstyle="butt")
ax.plot([xv, xv], [ys[-1], ys[0]], color=GREY, lw=0.6)
ax.annotate("", (xv + 6, ymid), (xv, ymid),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=5))
ax.text(xv + 8, ymid, "10 futures per\npatient and arm,\ncompared with\nthe observed\nfuture", fontsize=FS,
        va="center", ha="left", linespacing=1.15)

legend = (
    "Design of the four-arm experiment. For each of 450 GENIE-BPC patients (150 with lung, 150 with breast and 150 with "
    "colorectal cancer), the language model (DeepSeek V4 Flash) received the patient's history up to the start of a "
    "recorded systemic treatment and the treatment being started, and generated what followed; all later records were "
    "withheld. Four arms gave increasing patient evidence, shown from light amber to dark wine in every panel of this "
    "figure: the patient record alone; the record with a description of the care setting; the setting with twenty generic "
    "patients of the same cancer type and a comparable era; and the setting with twenty matched patients, who also "
    "received the same treatment modality and resembled the patient in recorded characteristics. Each patient shown "
    "to the model carried his or her history before a treatment start and the observed future. Ten futures were "
    "requested per patient and arm and compared with the observed future."
)
save_panel(
    fig, "f4a_design", legend=legend,
    sources=["manuscript/DRAFT.md (Results, patient examples)", f"{F_REL}/population_composition_scores_20261012.csv"],
    confirm=["Schematic: wording follows the manuscript.",
             "Model name (DeepSeek V4 Flash) is in the legend only."],
    message="Four arms share one patient record and treatment start and add care setting, 20 generic or 20 matched donors; 10 futures per patient and arm; 450 patients.")
