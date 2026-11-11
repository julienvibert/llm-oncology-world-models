"""f3c_forest, f3d_occupancy_map, f3e_balance: one forest estimate and its state-space split.

Data handling is the builder's, imported unchanged
(scripts/build_paper_survival_in_state_space.py): complete pre-progression paths of the 45 paired
EGFR-mutant futures per arm (Claude Sonnet), time allocated to a 28 x 28 grid on the frozen state map,
display smoothing only for the image. The three panels share one computation, so one script writes all
three, at the same height with aligned plot areas so that c (one number) and d (the same 45 pairs on the
state map) read as a pair; e checks that the map adds back to c.

  f3c  one number: months gained before first-line progression with osimertinib
  f3d  where in clinical state that time was gained or lost
  f3e  the gains and losses across the map add back to the same number

Run: uv run python scripts/figures/panels/f3c_forest_occupancy.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHEMO, GREY, INK, LIGHT, PAL, save_panel  # noqa: E402
from _f3util import axes_mm, minus, new_fig  # noqa: E402
import build_paper_survival_in_state_space as SS  # noqa: E402

OSI = PAL["osimertinib"]
SOURCES = ["runs/interventional_phase1_sonnet_full_v1c (via outputs/paper_state_map)",
           "outputs/paper_state_map/state_map_contract.json",
           "outputs/paper_survival_in_state_space/readout.json",
           "scripts/build_paper_survival_in_state_space.py (functions imported unchanged)"]
STATE_WORDS = {"S1": "Low burden, fit", "S2": "Standard metastatic", "S3": "Brain metastases, fit",
               "S4": "High burden, fit", "S5": "High burden, frail"}

states, endpoints, atlas = SS.load_trial()
intervals, trajectories = SS.build_pre_exit_intervals(states, endpoints, horizon_months=SS.HORIZON_MONTHS)
contrast = SS.paired_rmst_contrast(trajectories)
limits = SS.map_limits(atlas, intervals)
exact, smooth, x_edges, y_edges, _grid = SS.occupancy_grids(intervals, trajectories, limits=limits)
est, lo, hi = contrast
n_pairs = int(trajectories["pair_id"].nunique())
d_exact = (exact["osimertinib"] - exact["chemo"]).ravel()
pos = float(d_exact[d_exact > 1e-12].sum())
neg = float(d_exact[d_exact < -1e-12].sum())
assert abs(pos + neg - est) < 1e-6, (pos, neg, est)

# common vertical frame for c, d, e: plot area from B to B + AH (mm), figure height HF
HF, B, AH = 51.0, 11.0, 38.0

# ---------------------------------------------------------------- f3c forest
# one row per starting state (same names and same open-circle marker as the dots on the map) and the
# overall estimate; per-state intervals use the builder's own paired bootstrap on that state's pairs
WC = 64.0
fig = new_fig(WC, HF)
ax = axes_mm(fig, 21, B, 24, AH)
rows = []
for st in ("S1", "S2", "S3", "S4", "S5"):
    sub = trajectories[trajectories["state"].eq(st)]
    e_, l_, h_ = SS.paired_rmst_contrast(sub)
    rows.append((STATE_WORDS[st], e_, l_, h_, int(sub["pair_id"].nunique())))
# the five state means (9 pairs each) average to the overall estimate
assert abs(np.mean([r[1] for r in rows]) - est) < 1e-6
ys = [6.0, 5.0, 4.0, 3.0, 2.0]
ax.axvline(0, color=GREY, lw=0.5)
for y, (name, e_, l_, h_, n_) in zip(ys, rows):
    ax.plot([l_, h_], [y, y], color=INK, lw=0.9, solid_capstyle="butt")
    ax.plot(e_, y, "o", ms=4.2, mfc="white", mec=INK, mew=0.8, zorder=3)
    ax.text(1.02, y, f"{minus(e_)} [{minus(l_)}, {minus(h_)}]", transform=ax.get_yaxis_transform() if False else ax.transData,
            fontsize=6, va="center", ha="left", clip_on=False) if False else None
ax.plot([lo, hi], [0.3, 0.3], color=INK, lw=1.1, solid_capstyle="butt")
ax.plot(est, 0.3, "D", ms=5, color=INK, zorder=3)
ax.axhline(1.15, color=LIGHT, lw=0.5)
XL = (-4, 17)
ax.set_xlim(*XL)
ax.set_ylim(-0.5, 6.7)
ax.set_xticks([0, 5, 10, 15])
ax.set_yticks(ys + [0.3])
ax.set_yticklabels([r[0] for r in rows] + [f"All {n_pairs} pairs"])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Months gained with osimertinib\nbefore first-line progression")
vx = 1.06  # value column, axes fraction
for y, (name, e_, l_, h_, n_) in zip(ys + [0.3], rows + [("", est, lo, hi, n_pairs)]):
    ax.text(vx, y, f"{minus(e_, '.1f')} ({minus(l_, '.1f')} to {minus(h_, '.1f')})", transform=ax.get_yaxis_transform(), fontsize=6,
            va="center", ha="left", clip_on=False)
save_panel(
    fig, "f3c_forest", sources=SOURCES,
    legend=("Forest-plot view of the controlled EGFR experiment: difference in time before first-line progression over "
            "24 months (restricted mean), osimertinib minus chemotherapy, in months. Diamond, all 45 paired futures "
            "(15 combinations of clinical state and pretreatment history, three repeats each; Claude Sonnet). Open circles, "
            "the same difference for each of the five starting clinical states (9 pairs each); these are the same starting "
            "points, with the same names, as the open circles on the state map in the next panel. Bars, 95% bootstrap "
            "intervals (5,000 resamples of pairs); numbers on the right are the estimate and its interval, in months. The intervals "
            "describe the generated set, not uncertainty about patients."),
    message="The forest plot gives one number for the treatment difference, and the same number per starting state.",
    confirm=["Per-state rows are NEW computations in this script: the builder's paired bootstrap (seed 20260731 default of "
             "SS.paired_rmst_contrast) applied to the 9 pairs of each starting state; the five state means average exactly to "
             "the overall estimate (asserted). Intervals from 9 pairs are wide.",
             "Overall estimate and interval read from the saved builder outputs."],
)

# ---------------------------------------------------------------- f3d occupancy map
WD = 114.0
fig = new_fig(WD, HF)
ax = axes_mm(fig, 11, B, 72, AH)
cmap = LinearSegmentedColormap.from_list("chemo_white_osi", (CHEMO, "#FFFFFF", OSI))
diff = smooth["osimertinib"] - smooth["chemo"]
bound = float(np.abs(diff).max())
im = ax.imshow(diff.T, origin="lower", extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
               cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
               interpolation="bilinear", aspect="auto", zorder=1)
anchors = trajectories[trajectories["arm"].eq("chemo")].drop_duplicates("state").sort_values("state")
ax.scatter(anchors["anchor_pc1"], anchors["anchor_pc2"], s=14, facecolor="white", edgecolor=INK,
           linewidth=0.7, zorder=3)
x0, x1 = intervals["pc1"].min(), intervals["pc1"].max()
y0, y1 = intervals["pc2"].min(), intervals["pc2"].max()
px, py = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
ax.set_xlim(x0 - px, x1 + px)
ax.set_ylim(y0 - py, y1 + py)
ax.set_xlabel("Composite state PC1 (a.u.)\n← less disease burden        more →")
ax.set_ylabel("Composite state PC2 (a.u.)\n← better condition      poorer →")
ax.set_xticks([])
ax.set_yticks([])
LABEL_OFFSETS = {}  # filled below once the geometry is known
xr, yr = (x1 - x0 + 2 * px), (y1 - y0 + 2 * py)
placements = {"S1": (0.04, 0.93, "left"), "S2": (0.30, 0.07, "center"), "S3": (0.55, 0.93, "center"),
              "S4": (0.75, 0.07, "center"), "S5": (0.98, 0.66, "right")}
for _, r in anchors.iterrows():
    tx, ty, ha = placements[r["state"]]
    ax.annotate(STATE_WORDS[r["state"]], (r["anchor_pc1"], r["anchor_pc2"]), xytext=(tx, ty),
                textcoords="axes fraction", ha=ha, va="center", fontsize=6,
                color=INK, zorder=4, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1, shrinkB=2))
cax = axes_mm(fig, 91, B, 3, AH)
cb = fig.colorbar(im, cax=cax)
step = 0.25 if bound >= 0.5 else 0.1
ticks = np.arange(-np.floor(bound / step) * step, bound + 1e-9, step)
cb.set_ticks(ticks)
cb.set_ticklabels([minus(t, "+.2f") if abs(t) > 1e-9 else "0" for t in ticks])
cb.ax.yaxis.set_ticks_position("left")
cb.ax.tick_params(labelsize=6, length=2, width=0.5)
cb.outline.set_linewidth(0.5)
cb.set_label("Months per future", fontsize=6, rotation=270, labelpad=7)
cb.ax.yaxis.set_label_position("right")
fig.text(98.5 / WD, (B + AH) / HF, "More time with\nosimertinib", fontsize=6, ha="left", va="top", linespacing=1.1)
fig.text(98.5 / WD, B / HF, "More time with\nchemotherapy", fontsize=6, ha="left", va="bottom", linespacing=1.1)
save_panel(
    fig, "f3d_occupancy_map", sources=SOURCES,
    legend=("Where in clinical state the difference was gained, for the same 45 pairs. Each future's time before first-line "
            "progression (0-24 months) is spread over the clinical states it passed through, on a 28 × 28 grid of the "
            "fixed state map. Colour gives, for each region, the months per future spent there with osimertinib minus "
            "chemotherapy: blue, more time with osimertinib; purple, more time with chemotherapy; white, no difference. "
            "Both axes combine encoded tumour and host features and are not validated clinical scales; units are arbitrary. "
            "Along the horizontal axis disease burden increases to the right; along the vertical axis the host's condition "
            "gets worse upwards, so the best region (lowest burden, best condition) is the lower left. Open circles, the five "
            "starting clinical states (the three pretreatment histories of each state share one position). On the "
            f"unsmoothed grid, regions with more time on osimertinib add {minus(pos)} months per future and regions with more "
            f"time on chemotherapy {minus(neg)}; the net, {minus(est)} months, is the forest-plot estimate (Extended Data "
            "Fig. 6e). Smoothing (Gaussian, 0.85 cells) is for display only. The view is "
            "cropped to the states the generated futures visited. Progression ends the time counted and is not a place on the map."),
    message="The map shows where in clinical state the time difference was gained or lost.",
    confirm=["Starting-state words (S1-S5: low burden fit; standard metastatic; brain metastases fit; high burden fit; "
             "high burden frail) are plain renderings of the run's state names low_burden_fit, standard_metastatic, "
             "cns_met_fit, high_burden_fit, high_burden_frail.",
             "Axis arrows follow the state-map contract: PC1 towards greater encoded disease burden, PC2 towards "
             "greater host burden (poorer condition); 'lower left = best region'.",
             "Colours of the paper plan (blue osimertinib, purple chemotherapy)."],
)

# ---------------------------------------------------------------- f3e balance
WE, HE, BE, AHE = 56.0, 43.0, 11.0, 30.0
fig = new_fig(WE, HE)
ax = axes_mm(fig, 3, BE, 50, AHE)
vals = [pos, neg, est]
labs = ["Time gained: osimertinib\nfutures stayed longer", "Time lost: chemotherapy\nfutures stayed longer",
        "Net difference"]
cols = [OSI, CHEMO, INK]
ys = [2.0, 1.0, 0.0]
ax.set_xlim(-9, 16)
ax.set_ylim(-0.4, 2.95)
for y, v, c, lab in zip(ys, vals, cols, labs):
    ax.barh(y, v, color=c, height=0.36, edgecolor="none")
    ax.plot([0, 0], [y - 0.2, y + 0.2], color=GREY, lw=0.5)
    ax.text(-9, y + 0.27, lab, fontsize=6, ha="left", va="bottom", linespacing=1.05)
    ax.text(v + (0.4 if v >= 0 else -0.4), y, minus(v), va="center", ha="left" if v >= 0 else "right", fontsize=6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([-5, 0, 5, 10, 15])
ax.set_xticklabels(["−5", "0", "5", "10", "15"])
ax.set_xlabel("Months per future")
save_panel(
    fig, "f3e_balance", sources=SOURCES,
    legend=(f"Adding up the map. Regions where osimertinib futures stayed longer contribute {minus(pos)} months per future "
            f"(time gained), regions where chemotherapy futures stayed longer {minus(neg)} (time lost), and the net is "
            f"{minus(est)}, the number in the forest plot. Unsmoothed grid; the identity is exact by construction (each "
            "future's time is allocated once), so no interval is shown."),
    message="Gains and losses across the map add back to the forest-plot number.",
    confirm=["Bar names in clinical language.",
             "The +6.8 link is stated in the legend of the state-map panel."],
)
