"""si01c_fingerprint: the course of panel b summarised in seven numbers, placed among the other courses of the same run.

Message: a course can be described by its length, number and variety of events, response-progression
switches and side-effect load; the selected course is long and eventful, with few side effects, relative to
the other 71 courses of the same run. Data: Kimi atlas run; selected course as in si01a and si01b.
Run: uv run python scripts/figures/panels/si01c_fingerprint.py
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "explore"))
import _si  # noqa: E402,F401
import build_trajectory_geometry_gallery_v1 as M  # noqa: E402
import _si01  # noqa: E402
from common import INK, MODEL_COLOURS, STATE_COLOURS, fk, save_panel  # noqa: E402

fk.use("nrj")

course, TID, N_POOL = _si01.load_course()
feat = M.trajectory_features(course.all_events)
feat["cancer"] = feat["trajectory_id"].str.split("_").str[0]
IS_IDC = feat["trajectory_id"].str.startswith("IDC_")
n_courses = len(feat)
sel = feat[feat["trajectory_id"].eq(TID)].iloc[0]

METRICS = [("duration", "Follow-up (months)", "{:.0f}"),
           ("event_count", "Recorded events (n)", "{:.0f}"),
           ("event_richness", "Event types used (n)", "{:.0f}"),
           ("composition_entropy", "Evenness of event mix (0–1)", "{:.2f}"),
           ("grammar_diversity", "Variety of event sequences (0–1)", "{:.2f}"),
           ("recurrence", "Response–progression switches (n)", "{:.0f}"),
           ("toxicity_density", "Side-effect load (grade per month)", "{:.2f}")]

W, H = 180, 59
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)
left, wfp, top = 40, 60, H - 6
row_h, gap = 4.2, 2.9
rng = np.random.default_rng(20261012)
IDC_COL = "#E69F00"
axes = []
for i, (key, lab, fmt) in enumerate(METRICS):
    y0 = top - (i + 1) * row_h - i * gap
    ax = fig.add_axes([left / W, y0 / H, wfp / W, row_h / H])
    vals = feat[key].to_numpy(float)
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * 0.05 or 0.5
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-1.2, 1.2)
    q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
    ax.axvspan(q25, q75, color="#F1E2C0", lw=0, zorder=0)
    ax.vlines(q50, -1.0, 1.0, color=INK, lw=0.8, zorder=1)
    for is_idc, colour in ((False, MODEL_COLOURS["Kimi K3"]), (True, IDC_COL)):
        sub = feat[(IS_IDC == is_idc) & ~feat["trajectory_id"].eq(TID)]
        ax.scatter(sub[key], rng.uniform(-0.8, 0.8, len(sub)), s=4, color=colour, alpha=0.8,
                   linewidth=0, zorder=2)
    ax.scatter(sel[key], 0, s=26, marker="D", facecolor="white", edgecolor=INK, linewidth=1.0, zorder=4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ticks = sorted({round(lo, 2), round(hi, 2)})
    ax.set_xticks(ticks, [fmt.format(t) for t in ticks])
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_xlabel(lab, ha="right", va="center", fontsize=6)
    ax.xaxis.set_label_coords(-0.03, 0.5)
    ax.text(1.02, 0.5, fmt.format(sel[key]), transform=ax.transAxes, ha="left", va="center", fontsize=6.5,
            fontweight="bold")
    axes.append(ax)
hs = [Line2D([], [], marker="o", ls="none", color=MODEL_COLOURS["Kimi K3"], ms=3.2, label="Other Kimi K3 courses"),
      Line2D([], [], marker="o", ls="none", color=IDC_COL, ms=3.2, label="Other breast (IDC) courses")]
hs += [Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, mew=1.0, ms=4.5, label="Course of panels a and c"),
       Line2D([], [], color=INK, lw=0.8, label="Median"),
       Line2D([], [], color="#F1E2C0", lw=4, label="Middle half")]
fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=5, frameon=False, fontsize=6,
           handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1)

# event-type composition of the selected course
TREAT, ASSESS, TOX, OTHER = "#2A6FBF", None, "#045A8D", "#7A4E2D"
EV = {"toxicity": ("Side effect", TOX), "response": ("Response", STATE_COLOURS["response"]),
      "progression": ("Progression", STATE_COLOURS["progression"]),
      "patient_state_update": ("Condition reassessed", OTHER),
      "systemic_treatment_start": ("Treatment start", TREAT),
      "systemic_treatment_switch": ("Treatment switch", TREAT),
      "systemic_treatment_modification": ("Dose change", TREAT),
      "systemic_treatment_stop": ("Treatment stop", TREAT),
      "clonal_evolution_event": ("Clonal change", OTHER),
      "local_intervention": ("Local treatment", OTHER),
      "death": ("Death", OTHER)}
counts = course.events["event_type"].astype(str).value_counts()
n_ev = int(counts.sum())
bx = fig.add_axes([139 / W, 8 / H, 38 / W, (top - 8 + 1) / H])
order = list(counts.index)[::-1]
bx.barh(range(len(order)), [counts[o] / n_ev * 100 for o in order], color=[EV[o][1] for o in order], height=0.7)
bx.set_yticks(range(len(order)), [EV[o][0] for o in order])
for j, o in enumerate(order):
    bx.text(counts[o] / n_ev * 100 + 0.6, j, f"{counts[o]}", va="center", fontsize=6)
bx.set_xlim(0, counts.max() / n_ev * 100 * 1.2)
bx.set_xlabel("Share of the course's events (%)")
bx.tick_params(axis="y", length=0, pad=2)

n_idc = int(IS_IDC.sum())
legend = (
    "The course of panels a and c summarised in seven numbers and placed among generated courses (panel b of "
    f"three views of one course, {TID}). Left, each row is one summary of a course; every dot is one of the "
    f"{n_courses - 1} other accepted courses of the same Kimi K3 atlas run (all cancers, wine; the {n_idc - 1} "
    "other invasive ductal carcinoma courses, orange); the diamond is the course of panels a and c, with its value at "
    "the right. Each row has its own scale (end ticks show the smallest and largest value in the run). Vertical "
    "line, median; shaded band, middle half of the courses. Rows: months of follow-up; number of recorded "
    "events; number of different event types; evenness of the event mix (0, one type only; 1, all types equally "
    "frequent); variety of event sequences (distinct pairs of consecutive event types divided by all such pairs); "
    "number of switches between response and progression assessments; side-effect load (sum of CTCAE grades per "
    f"month of follow-up). Right, the {n_ev} recorded events of the course by type (numbers beside the bars, count "
    "of events; blue, treatment events; dark blue, side effects; brown, other). The seven numbers are the "
    "coordinates used to compare whole courses; they are descriptive and not a score. Generated data; no "
    "inferential statistics."
)
save_panel(fig, "si01c_fingerprint", legend=legend,
           sources=["runs/kimi_k3_off_full_atlas_n5_complete/derivation (snapshot_rows.csv, event_rows.csv)",
                    "scripts/explore/build_trajectory_geometry_gallery_v1.py (trajectory_features)"],
           message="A whole course can be described by seven numbers and located among other generated courses.",
           confirm=["Every course is shown at its raw value on its own scale (no data altered).",
                                        "Comparison set is the Kimi K3 atlas run only (one model); the colour of other courses is the Kimi colour, IDC courses orange (local).",
                    "Event-type colours other than response and progression are local to this panel: blue, treatment events; dark blue, side effects; brown, other events."])
