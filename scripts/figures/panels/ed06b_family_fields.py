"""Panel ed06b_family_fields: the same field drawn separately for eight tumour families."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed06 import *  # noqa: F401,F403,E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import save_panel  # noqa: E402

mem, proj = load_members()  # noqa: F405
xl = (np.quantile(mem.x0, 0.003), np.quantile(mem.x0, 0.997))  # noqa: F405
yl = (np.quantile(mem.y0, 0.003), np.quantile(mem.y0, 0.97))  # noqa: F405
NXB, NYB, MINS = 8, 6, 10
xe, ye = np.linspace(*xl, NXB + 1), np.linspace(*yl, NYB + 1)  # noqa: F405
W, H = 180.0, 86.0
fig = new_fig(W, H)
AW, AH = 39.0, 28.0
FAMS = list(FAMILY)  # noqa: F405
info = {}
for k, fam in enumerate(FAMS):
    r, c = divmod(k, 4)
    ax = axes_mm(fig, 9 + c * 43.0, (53.5 if r == 0 else 17.0), AW, AH)
    m = mem[mem.family == fam]
    f = grid_field(m, xe, ye, MINS)  # noqa: F405
    f = f[f.cx.between(*xl) & f.cy.between(*yl) & (f.n_traj >= 10) & (f.n_follow >= 5)]
    inside = m.x0.between(*xl) & m.y0.between(*yl)
    ax.scatter(m.x0[inside], m.y0[inside], s=0.8, color="#B9C4CE", alpha=0.5, linewidths=0, zorder=0, rasterized=True)
    ax.quiver(f.cx, f.cy, f.dx, f.dy, f.dead, cmap=DEATH_CMAP, norm=DEATH_NORM, angles="xy", scale_units="xy", scale=1,  # noqa: F405
              width=0.0085, headwidth=3.6, headlength=4, headaxislength=3.4, zorder=4, pivot="tail")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([])
    ax.set_yticks([])
    nt = m.trajectory_id.nunique()
    ax.text(0.03, 1.02, f"{fam}, n = {nt}", transform=ax.transAxes, fontsize=6, fontweight="bold", ha="left", va="bottom")
    if c == 0:
        ax.set_ylabel("State-map PC2 (a.u.)", fontsize=6, labelpad=1)
    if r == 1:
        ax.set_xlabel("State-map PC1 (a.u.)", fontsize=6, labelpad=1)
    info[fam] = (nt, len(m), len(f))
# colour key in the figure margin
from matplotlib.cm import ScalarMappable  # noqa: E402
cax = axes_mm(fig, 62, 7.5, 56, 2.2)
cb = fig.colorbar(ScalarMappable(norm=DEATH_NORM, cmap=DEATH_CMAP), cax=cax, orientation="horizontal")  # noqa: F405
cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
cb.set_ticklabels(["0", "10", "20", "30", "40", "≥50"])
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
cb.set_label("Died within 6 months (%)", fontsize=6, labelpad=1)
print(info)
save_panel(
    fig, "ed06b_family_fields",
    legend=("The same field drawn separately for eight tumour families of the generated atlas (n = generated trajectories in "
            "the family; 30 per cancer type, six models). Start states at 3, 6, 12, 18 and 24 months are cut into the cells of "
            f"a {NXB} by {NYB} grid of the state map; in each cell with at least {MINS} start states from at least 10 unique courses and at least 5 survivor endpoints, one arrow runs from the "
            "average start position to the average position 6 months later, among those still alive; arrow length is the real "
            "average movement (not enlarged). Arrow colour is the share of the cell's start states that died within 6 months "
            "(blue, few; crimson, many); cells with less support are blank. Grey dots, the start states of the family inside the window shown (start states above the window, 3% of the atlas, are not drawn). "
            "Families group cancer types as in the aggregation audit (Supplementary); about 40% of the atlas falls in these eight "
            "families and cancer types outside them are not shown. Start states from the same trajectory are not independent. "
            "Axes are the frozen state map (Extended Data Fig. 1), arbitrary units; neither is a clinical scale. Arrows are averaged six-month displacements of courses still followed, not continuous dynamics. Generated "
            "trajectories, not patients; the field is descriptive."),
    sources=[SRC_PROJ, "runs/<six atlas runs>/derivation/event_rows.csv", "scripts/explore/build_field_trellis.py (family mapping)"],  # noqa: F405
    confirm=["Fields are NEW computations from saved tables (grid 8x6, min 10 start states, 10 unique courses and 5 survivor endpoints, 6-month horizon, frames 3/6/12/18/24 months); "
             "not read from a saved field output.",
             "Family membership copied from scripts/explore/build_field_trellis.py (FAMILY); only about 40% of trajectories belong to a family.",
             "Family names (e.g. Gynaecological) taken from that mapping."],
    message="The direction of flow through the state map depends on the tumour family.")
