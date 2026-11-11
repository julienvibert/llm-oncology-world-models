"""si01b_event_ribbon: one generated course as stacked tracks on one time axis, every track named.

Message: the same course is read as treatment, disease state, tumour burden, condition score, side effects and
other recorded events. Same course as si01a and si01c.
Data: stored generated course IDC_tumor_only__kimi_k3_off__f02 (Kimi atlas run).
Run: uv run python scripts/figures/panels/si01b_event_ribbon.py
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "explore"))
import _si  # noqa: E402
import build_trajectory_geometry_gallery_v1 as src  # noqa: E402
import _si01  # noqa: E402
from common import INK, LIGHT, STATE_COLOURS, fk, save_panel  # noqa: E402

fk.use("nrj")

course, TID, N_POOL = _si01.load_course()
s, e = course.snapshots, course.events
T_END = float(e.loc[e.event_type.eq("death"), "month"].max())
TUMOUR_COL, HOST_COL, SIDE_COL = "#E34A33", "#2B8CBE", "#045A8D"

W, H = 180, 72
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)
tracks = [("treat", 10), ("disease", 4.5), ("tumour", 10), ("host", 9), ("side", 10), ("events", 10.5)]
gap, left, right, bottom = 1.2, 37, 1.5, 9.5
axes, y = {}, H - 2.5
for name, h in tracks:
    y -= h
    axes[name] = fig.add_axes([left / W, y / H, (W - left - right) / W, h / H])
    y -= gap
for a in axes.values():
    a.set_xlim(-0.5, T_END + 1.0)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="x", labelbottom=False)
    a.tick_params(axis="y", length=0)
    a.spines["bottom"].set_color(LIGHT)

# treatment lines
a = axes["treat"]
a.set_ylim(0, 1.3)
a.set_yticks([0.27], ["Treatment line"])
segs = src.treatment_segments(e)
MPM = (T_END + 1.5) / (W - left - right)  # months per millimetre
n_narrow = 0
for start, end, label in segs:
    name = _si.regimen_name(label.replace(" ", "_"))
    cls = _si.regimen_class(label)
    a.add_patch(Rectangle((start, 0.04), end - start, 0.46, facecolor=_si.CLASS_COLOURS[cls], edgecolor="white", lw=0.6))
    if len(name) * 1.05 * MPM < end - start - 0.5:
        a.text((start + end) / 2, 0.27, name, ha="center", va="center", fontsize=6, color="white")
    else:
        late = start > 0.8 * T_END
        a.text(end if late else start, 0.62 + 0.3 * (n_narrow % 2), name, ha="right" if late else "left", va="bottom", fontsize=6, color=INK)
        n_narrow += 1
used_small = [(st, en, lb) for st, en, lb in segs if en - st <= 6.5]
# narrow lines: names are given in the key below the tracks

# disease state
a = axes["disease"]
a.set_ylim(0, 1)
a.set_yticks([0.5], ["Disease state"])
st_events = e[e.event_type.isin(["response", "progression", "recurrence"])].sort_values("month")
cur, t0 = "unresolved", 0.0
bands = []
for r in st_events.itertuples():
    new = "progression" if r.event_type in ("progression", "recurrence") else "response"
    bands.append((t0, r.month, cur))
    cur, t0 = new, r.month
bands.append((t0, T_END, cur))
for b0, b1, c in bands:
    a.add_patch(Rectangle((b0, 0.1), b1 - b0, 0.8, facecolor=STATE_COLOURS[c], edgecolor="white", lw=0.5))

# tumour burden and condition score: values held until the next assessment
x_all = np.r_[s["month"].to_numpy(float), T_END]
for name, col, colour, ticks, tl, lab, top in [
    ("tumour", "tumour", TUMOUR_COL, [0, 1, 2, 3, 4], ["absent", "small", "moderate", "bulky", "diffuse"],
     "Tumour burden\n(all sites)", 4.2),
    ("host", "host", HOST_COL, [0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"], "Condition score\n(0 = best)", 4.4),
]:
    a = axes[name]
    y_ = np.r_[s[col].to_numpy(float), s[col].iloc[-1]]
    a.step(x_all, y_, where="post", color=colour, lw=1.1)
    a.fill_between(x_all, 0, y_, step="post", color=colour, alpha=0.22, lw=0)
    a.set_ylim(-0.05, top)
    a.set_yticks(ticks, tl)
    a.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=26 if name == "tumour" else 26)
    a.yaxis.set_label_coords(-0.13, 0.5)
    a.tick_params(axis="y", length=2, pad=1)
    a.spines["left"].set_visible(True)
    a.spines["left"].set_color(LIGHT)

# side effects
a = axes["side"]
tox = e[e.event_type.eq("toxicity")]
for r in tox.itertuples():
    g = float(r.ctcae_v5_grade) if np.isfinite(r.ctcae_v5_grade) else 1.0
    a.vlines(r.month, 0, g, color=SIDE_COL, lw=0.8)
    a.scatter(r.month, g, s=7 + 4 * g, color=SIDE_COL, edgecolor="none", zorder=3)
a.set_ylim(-0.1, 4.6)
a.set_yticks([1, 2, 3, 4], ["grade 1", "grade 2", "grade 3", "grade 4"])
a.tick_params(axis="y", length=2, pad=1)
a.set_ylabel("Side effects\n(CTCAE grade)", rotation=0, ha="right", va="center")
a.yaxis.set_label_coords(-0.13, 0.5)
a.spines["left"].set_visible(True)
a.spines["left"].set_color(LIGHT)

# other recorded events: one named row each
a = axes["events"]
rows = [("systemic_treatment_modification", "Dose change", "P"), ("local_intervention", "Local treatment", "s"),
        ("clonal_evolution_event", "Clonal (molecular) change", "D"), ("patient_state_update", "Condition reassessed", "o")]
a.set_ylim(-0.6, len(rows) - 0.4)
ypos = {r[0]: len(rows) - 1 - i for i, r in enumerate(rows)}
a.set_yticks([ypos[r[0]] for r in rows], [r[1] for r in rows])
for et, _, mk in rows:
    sub = e[e.event_type.eq(et)]
    a.scatter(sub["month"], [ypos[et]] * len(sub), marker=mk, s=16, facecolor=INK if mk != "o" else "white",
              edgecolor=INK, linewidth=0.7, zorder=3)
    a.axhline(ypos[et], color=LIGHT, lw=0.4, zorder=0)
a.tick_params(axis="x", labelbottom=True)
a.set_xlabel("Months from diagnosis")
a.set_xticks(range(0, int(T_END) + 1, 12))

# death line across all tracks
for a in axes.values():
    a.axvline(T_END, color=INK, lw=0.8, zorder=5)
axes["disease"].text(T_END - 0.4, 0.5, "Death", fontsize=6, ha="right", va="center", color="white")

# treatment key for lines too narrow to name in the bar, and class key
a = axes["treat"]
cls_used = []
for st, en, lb in segs:
    c = _si.regimen_class(lb)
    if c not in cls_used:
        cls_used.append(c)
a.set_ylim(0, 1.3)
a.set_clip_on(False)

n_courses = int(course.all_events["trajectory_id"].nunique())
legend = (
    "The same generated course as panel a, as tracks on one time axis (panel b of three views of one course). "
    f"{TID}: hormone-receptor-positive (ER positive 95%, PR positive 80% in the stored baseline), HER2-negative "
    f"invasive ductal carcinoma generated by Kimi K3, chosen as {_si01.RULE}; n = 1, followed {T_END:.0f} months "
    "to death. Treatment line: bars coloured by class, named on the bar when wide enough and above it otherwise. "
    "Disease state: from the generated assessments (response, progression; grey before the first assessment). "
    "Tumour burden: the overall burden band across all sites, from absent to diffuse. Condition score: a display "
    "composite of performance status (ECOG), symptom burden, number of active side effects and worst "
    "organ-function grade, weighted 0.58, 0.20, 0.14 and 0.08 (0 = best); it is not a coordinate of the state "
    "map. Side effects: one stem per recorded event, height = CTCAE grade. Lower rows: each recorded event of that "
    "type (dose change; local treatment; clonal, that is molecular, change; reassessment of the patient's "
    "condition). Values hold until the next generated assessment. Vertical line, death. Main Fig. 1a shows a "
    "different patient with the tracks by organ. Generated data."
)
save_panel(fig, "si01b_event_ribbon", legend=legend,
           sources=["runs/kimi_k3_off_full_atlas_n5_complete/derivation/snapshot_rows.csv",
                    "runs/kimi_k3_off_full_atlas_n5_complete/derivation/event_rows.csv",
                    "scripts/explore/build_trajectory_geometry_gallery_v1.py"],
           message="The same generated course read as treatment, disease state, burden, condition, side effects and events.",
           confirm=["Condition-score weights (0.58, 0.20, 0.14, 0.08) are those of add_state_coordinates in the source "
                    "script; the score is a display composite defined there, not a state-map axis.",
                    "Tumour-burden ticks name bands 0, 1, 2, 3, 4 of the source mapping (absent, small, moderate, "
                    "bulky, diffuse); the microscopic band (0.4) falls between absent and small.",
                    "Same course as si01a, chosen by the declared rule in _si01.py; regimens classed by keyword (_si.py)."])
