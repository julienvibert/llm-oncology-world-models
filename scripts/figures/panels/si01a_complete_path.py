"""si01a_complete_path: a complete generated course as a path on the state map, with a zoom on its dense start.

Message: over 8 years a generated breast cancer patient moves through nine treatment lines; on the state
map each line is a stretch of the path, each assessment a dot coloured by disease state.
Different patient from Fig. 1 (a lung cancer course). Data: stored generated course
IDC_tumor_only__kimi_k3_off__f02 on the frozen state map (scripts/build_paper_individual_state_trajectory_views.py).
Run: uv run python scripts/figures/panels/si01a_complete_path.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import build_paper_individual_state_trajectory_views as src  # noqa: E402
import _si  # noqa: E402
import _si01  # noqa: E402
from common import INK, STATE_COLOURS, fk, save_panel  # noqa: E402

fk.use("nrj")

_course, TID, N_POOL = _si01.load_course()
RECEPT = _si01.baseline_receptors(TID)
projection = src.load_projection()
selected = projection[projection["trajectory_id"].eq(TID)].copy()
run_id = str(selected["_run"].iloc[0])
events = src.load_events(run_id)
path = src.attach_events_and_residence(selected, events)
tx = src.treatment_events(events, TID)
D = 30.44
path["month"] = path["days_from_diagnosis"] / D
death_day = float(events[(events.trajectory_id == TID) & (events.event_type == "death")].days_from_diagnosis.min())
T_END = death_day / D

# disease state at each snapshot: carried forward from the last response or progression assessment
state, states = "unresolved", []
for et in path["event_type"]:
    if et in ("response", "progression"):
        state = et
    states.append(state)
path["state"] = states

# treatment lines
lines = []
for i, r in enumerate(tx.itertuples()):
    start = r.days_from_diagnosis / D
    end = tx.days_from_diagnosis.iloc[i + 1] / D if i + 1 < len(tx) else None
    lines.append({"n": i + 1, "name": _si.regimen_name(r.regimen_clean), "cls": _si.regimen_class(r.regimen_clean),
                  "start": start, "end": end})
stops = events[(events.trajectory_id == TID) & (events.event_type == "systemic_treatment_stop")]
for i, ln in enumerate(lines):
    if ln["end"] is None:
        ln["end"] = float(stops.days_from_diagnosis.max() / D)
    elif i + 1 < len(lines):
        # a recorded stop before the next start closes the line
        st = stops[(stops.days_from_diagnosis / D > ln["start"]) & (stops.days_from_diagnosis / D < ln["end"])]
        if len(st):
            ln["end"] = float(st.days_from_diagnosis.min() / D)


def class_at(month: float) -> str:
    for ln in lines:
        if ln["start"] <= month < ln["end"]:
            return ln["cls"]
    return "other"


pts = path[["pc1", "pc2"]].to_numpy(float)
seg_cls = [class_at(m) for m in path["month"].iloc[:-1]]
seg_col = [_si.CLASS_COLOURS[c] for c in seg_cls]
atlas = pd.read_csv(src.STATE_PATH, usecols=["pc1", "pc2"])
dens = LinearSegmentedColormap.from_list("atlas", ["#FFFFFF", "#FAF3E3", "#F1E2C0", "#E6CE9F"])

W, H = 180, 80
fig = fk.figure(width=W, height_mm=H)
fig.set_layout_engine(None)


def axbox(l, b, w, h):  # noqa: E741
    return fig.add_axes([l / W, b / H, w / W, h / H])


ax = axbox(13, 24, 104, 54)
zx = axbox(125, 24, 52, 54)
ZOOM = (-0.62, 0.32, -0.42, 0.26)
for a, lim in ((ax, (pts[:, 0].min() - 0.08, pts[:, 0].max() + 0.08, pts[:, 1].min() - 0.08, pts[:, 1].max() + 0.08)),
               (zx, ZOOM)):
    a.hexbin(atlas.pc1, atlas.pc2, gridsize=45, bins="log", cmap=dens, mincnt=1, linewidths=0, rasterized=True,
             zorder=0)
    a.set_xlim(lim[0], lim[1])
    a.set_ylim(lim[2], lim[3])
    inside = ((pts[:, 0] >= lim[0]) & (pts[:, 0] <= lim[1]) & (pts[:, 1] >= lim[2]) & (pts[:, 1] <= lim[3]))
    if a is zx:
        n_inside = int(inside.sum())
    for i in range(len(pts) - 1):
        if np.hypot(*(pts[i + 1] - pts[i])) < 0.012 or not (inside[i] and inside[i + 1]):
            continue
        a.annotate("", xy=pts[i + 1], xytext=pts[i], zorder=2,
                   arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.3", lw=0.9, color=seg_col[i],
                                   shrinkA=1.5, shrinkB=1.5))
    scale = 1.0 if a is ax else 1.6
    sizes = np.clip(src.residence_to_marker_area(path["residence_months"].to_numpy(float)) * 0.30 * scale, 3, 70)
    cols = [STATE_COLOURS[s] for s in path["state"]]
    a.scatter(pts[inside, 0], pts[inside, 1], s=sizes[inside], c=np.array(cols)[inside], edgecolor=INK,
              linewidth=0.4, zorder=3)
    a.scatter(*pts[0], marker="D", s=22, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=5)
    if inside[-1]:
        a.scatter(*pts[-1], marker="X", s=40, facecolor=INK, edgecolor="white", linewidth=0.5, zorder=6)
    # line numbers at the first assessment of each treatment line
    for ln in lines:
        j = int(np.argmax(path["month"].to_numpy() >= ln["start"]))
        if not inside[j]:
            continue
        a.annotate(str(ln["n"]), xy=pts[j], xytext=(0, 5.5) if a is ax else (0, 7), textcoords="offset points",
                   ha="center", va="center", fontsize=6, color="white", zorder=7,
                   bbox=dict(boxstyle="circle,pad=0.12", fc=_si.CLASS_COLOURS[ln["cls"]], ec="white", lw=0.4))
    a.set_xlabel("State-map PC1 (a.u.)")
for a in (ax, zx):
    a.tick_params(labelbottom=False, labelleft=False, length=0)
ax.set_ylabel("State-map PC2 (a.u.)")
# zoom frame on the full map
ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False, ec=INK, lw=0.7, ls=":",
                       zorder=8))
zx.set_xlabel(f"Close-up of the dotted box ({n_inside} of {len(pts)} assessments)")

# key strip
k = axbox(0, 0, W, 22)
k.set_xlim(0, W)
k.set_ylim(0, 22)
k.axis("off")
cols_x = [6, 66, 126]
for idx, ln in enumerate(lines):
    c, r = divmod(idx, 3)
    x, y = cols_x[c], 17.3 - 3.4 * r
    k.text(x, y, str(ln["n"]), fontsize=6, color="white", ha="center", va="center",
           bbox=dict(boxstyle="circle,pad=0.12", fc=_si.CLASS_COLOURS[ln["cls"]], ec="none"))
    k.text(x + 3, y, f"{ln['name']}, months {ln['start']:.0f}–{ln['end']:.0f}", fontsize=6, va="center")
hs = [Line2D([], [], marker="o", ls="none", mfc=STATE_COLOURS[s_], mec=INK, mew=0.4, ms=4, label=lab)
      for s_, lab in (("unresolved", "Not yet assessed"), ("response", "Response"), ("progression", "Progression"))]
hs += [Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, ms=4, label="Diagnosis"),
       Line2D([], [], marker="X", ls="none", mfc=INK, mec="white", ms=5, label="Death")]
l1 = k.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=5, frameon=False, fontsize=6,
              handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
k.add_artist(l1)
used = []
for ln in lines:
    if ln["cls"] not in used:
        used.append(ln["cls"])
if "other" in seg_cls:
    used.append("other")
lab_ = dict(_si.CLASS_LABEL, other="No treatment")
hc = [Line2D([], [], color=_si.CLASS_COLOURS[c], lw=2, label=lab_[c]) for c in used]
k.legend(handles=hc, loc="lower left", bbox_to_anchor=(0.02, 0.17), ncol=5, frameon=False, fontsize=6,
         handlelength=1.4, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)

n_snap = len(path)
legend = (
    "One generated course as a path on the state map (panel a of three views of the same course; b and c below). "
    f"The course is {TID} (invasive ductal carcinoma generated by Kimi K3), chosen by a declared rule: "
    f"{_si01.RULE}. The stored baseline lists ER {RECEPT['ER']}, PR {RECEPT['PR']} and HER2 IHC {RECEPT['HER2_IHC']}, "
    f"so the patient is hormone-receptor-positive, HER2-negative; stage at diagnosis, locoregional. Followed for "
    f"{T_END:.0f} months to death, with {len(lines)} lines of systemic treatment. n = 1; not chosen to be typical. "
    f"Each dot is one generated assessment (n = {n_snap}), coloured by the disease state at that time; dot area, "
    "time spent before the next assessment (capped for display); arrows, moves between assessments, coloured by "
    "the treatment class active at the start of the move; numbered discs, start of each line (listed below with "
    "the months it ran). Diamond, diagnosis; cross, death. Shading, density of all generated positions of the "
    "six-model atlas. Right, close-up of the dotted box, showing only the assessments inside it. Axes are the first "
    "two components of the frozen state map; PC1 mainly contrasts tumour extent and PC2 mainly contrasts marrow "
    "disease with hepatic, thoracic and other disease, with host terms also contributing; neither is a validated "
    "clinical scale (Extended Data Fig. 1). Main Fig. 1 shows a different patient; panels b and c show this same course. "
    "Generated data."
)
save_panel(fig, "si01a_complete_path", legend=legend,
           sources=["outputs/paper_state_map/state_projection.csv.gz",
                    "scripts/build_paper_individual_state_trajectory_views.py"],
           message="A whole generated course is a path on the state map: each treatment line a stretch, each assessment a dot.",
           confirm=["Treatment classes are assigned from regimen names by keyword (_si.py); ribociclib + letrozole and "
                    "alpelisib + fulvestrant are classed as targeted therapy, sacituzumab govitecan as an "
                    "antibody-drug conjugate (a local colour not in the paper plan).",
                    "Line end months come from the next start or a recorded treatment stop.",
                    "Hormone-receptor status is read from the stored baseline biomarker table (ER 95%, PR 80%, HER2 IHC 0).",
                    "Course chosen by a declared rule (longest follow-up of the five Kimi IDC courses); same course in si01b and si01c.",
                    "Extended Data Fig. 1 is the proposed numbering of the state-map definition page."])
