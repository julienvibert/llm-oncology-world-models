"""Panel ed06a_pan_atlas_field: flow of generated trajectories through the state map (whole atlas)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed06 import *  # noqa: F401,F403,E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import ROOT, save_panel  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

mem, proj = load_members()  # noqa: F405
import hashlib, json  # noqa: E401,E402
from matplotlib.lines import Line2D  # noqa: E402
n_traj = mem.trajectory_id.nunique()
xl0 = (np.quantile(mem.x0, 0.003), np.quantile(mem.x0, 0.997))  # noqa: F405
yl0 = (np.quantile(mem.y0, 0.003), np.quantile(mem.y0, 0.997))  # noqa: F405
CELL, MIN_COURSES, MIN_SURV = 0.12, 30, 30
W, H = 98.0, 82.0
AX_W, AX_H = 70.0, 70.0
xl, yl = equal_limits(xl0, yl0, AX_W, AX_H)  # noqa: F405  (same units on both axes, so arrow angles are comparable)
fld = finite_field(mem, xl, yl, CELL, MIN_COURSES, MIN_SURV)  # noqa: F405
OUTD = ROOT / "outputs" / "paper_atlas_finite_displacement_field"
OUTD.mkdir(exist_ok=True)
fld.to_csv(OUTD / "cells.csv", index=False)
(OUTD / "manifest.json").write_text(json.dumps({
    "purpose": "ED6a: supported finite six-month displacement arrows of the generated atlas on the frozen state map",
    "cell_side_map_units": CELL, "min_unique_courses": MIN_COURSES, "min_survivor_endpoints": MIN_SURV,
    "start_months": list(FRAMES_MONTHS), "horizon_months": HORIZON_MONTHS,  # noqa: F405
    "window_x": list(map(float, xl)), "window_y": list(map(float, yl)), "cells_drawn": int(len(fld)),
    "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("outputs/paper_state_map/state_projection.csv.gz",  # noqa: F405
                                                                                       "outputs/paper_state_map/state_map_contract.json")}}, indent=1))
fig = new_fig(W, H)
ax = axes_mm(fig, 12, 10, AX_W, AX_H)
inside = mem.x0.between(*xl) & mem.y0.between(*yl)
n_out = int((~inside).sum())
ax.scatter(mem.x0[inside], mem.y0[inside], s=1.0, color="#B9C4CE", alpha=0.5, linewidths=0, zorder=0, rasterized=True)
TIERS = ((30, 80, 0.0028), (80, 200, 0.0045), (200, 10**9, 0.0068))
for lo, hi, wd in TIERS:
    f = fld[(fld.n_courses >= lo) & (fld.n_courses < hi)]
    ax.quiver(f.cx, f.cy, f.dx, f.dy, f.dead_share, cmap=DEATH_CMAP, norm=DEATH_NORM, angles="xy", scale_units="xy", scale=1,  # noqa: F405
              width=wd, headwidth=3.2, headlength=3.6, headaxislength=3.2, zorder=4, pivot="tail")
ax.set_xlim(*xl)
ax.set_ylim(*yl)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(XLAB, fontsize=6, labelpad=2)  # noqa: F405
ax.set_ylabel(YLAB, fontsize=6, labelpad=2)  # noqa: F405
ax.legend(handles=[Line2D([0], [0], color="black", lw=lw, label=lab) for lw, lab in
                   ((0.6, "30-79 courses"), (1.0, "80-199"), (1.5, "200 or more"))],
          title="Unique courses in the cell", title_fontsize=6, loc="upper left", frameon=False, fontsize=6, handlelength=1.6,
          borderaxespad=0.1)
cax = axes_mm(fig, 87, 10, 3, AX_H)
cb = fig.colorbar(ScalarMappable(norm=DEATH_NORM, cmap=DEATH_CMAP), cax=cax)  # noqa: F405
cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
cb.set_ticklabels(["0", "10", "20", "30", "40", "≥50"])
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
cb.set_label("Start visits that died within 6 months (%)", fontsize=6, labelpad=2)
save_panel(
    fig, "ed06a_pan_atlas_field",
    legend=("Averaged six-month displacement of generated patients through the state map; not continuous dynamics. Source: the "
            f"generated atlas (six models, 148 cancer types, {n_traj:,} generated trajectories). Every trajectory alive at 3, 6, "
            f"12, 18 and 24 months contributes one start visit per time ({len(mem):,} start visits, grey dots; {n_out} fall outside "
            f"the window shown). The map is cut into square cells of {CELL:g} map units. Each arrow runs from the mean start "
            "position in a cell to the mean position 6 months later among the start visits still alive then (survivors); arrows are drawn at their "
            f"real length. Cells are drawn only with at least {MIN_COURSES} unique courses and {MIN_SURV} survivor endpoints; "
            f"{len(fld)} cells qualify (median {fld.n_courses.median():.0f} courses and {fld.n_survivor_courses.median():.0f} courses with a survivor "
            "endpoint per cell); arrow thickness gives the number of unique courses. Colour is the share of the cell's start "
            "visits that died within the 6 months after that visit (blue, few; crimson, many), so it is conditional on the start visit, "
            "not on diagnosis. Arrows average only courses still followed, so they omit those who died. Start visits from one course are not independent. "
            "Axes are the frozen state map (Extended Data Fig. 1), in arbitrary units with the same scale on both axes; PC2 contrasts "
            "marrow disease against hepatic, thoracic and other disease and also carries host terms, so neither axis is a clinical scale. "
            "The arrows describe the generated atlas; they are not a causal or history-free law, a patient path or a test of any rollout."),
    sources=[SRC_PROJ, "runs/<six atlas runs>/derivation/event_rows.csv", "scripts/build_paper_group_state_vector_field_views.py",  # noqa: F405
             "outputs/paper_atlas_finite_displacement_field/cells.csv"],
    confirm=["Arrows are finite six-month displacements per cell; saved with parameters and source hashes in "
             "outputs/paper_atlas_finite_displacement_field.",
             "Cell side 0.12 map units, minimum 30 unique courses and 30 survivor endpoints (declared display choices).",
             "Deaths from event_rows.csv of the six runs; trajectories whose record ends alive before 6 months are left out.",
             "Atlas = 148 cancer types x 30 generated trajectories; start visits at 3-24 months over-represent long-lived trajectories."],
    message="Averaged over generated patients, six-month displacements through the state map are coherent, with deaths concentrated at high PC1.")
