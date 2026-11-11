"""Panels ed05c_direction_mutant and ed05d_direction_wildtype: mean movement at 6 months per starting state."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed05 import *  # noqa: F401,F403,E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import save_panel  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

D = load()  # noqa: F405
A = anchors(D)  # noqa: F405
LEG = ("Mean direction of movement in the first 6 months for each of the five starting states ({ctx}). Every arrow starts at "
       "the origin (the open circle, standing for the starting state) and ends at the average change in position; numbers name the starting states (1, low burden, fit; 2, standard metastatic; 3, brain metastases, fit; 4, high burden, fit; 5, high "
       "burden, frail). The arrow shows the average change at 6 months of the futures that are "
       "before the first-line endpoint then (chemotherapy, purple; osimertinib, blue), averaged over three histories and three "
       "futures per history (up to 9 futures per state and treatment). Arrow thickness grows with the share of those 9 futures "
       "that are before the first-line endpoint at 6 months; when not all 9 are, the count is given next to the state number (for example 2/9), and no arrow is drawn when none is (stated in the panel). The arrows describe movement only: futures that "
       "have reached the first-line endpoint are not in the average, and how long futures stay is shown in the last panel. Axes "
       "are changes in the frozen state map (Extended Data Fig. 1), in arbitrary units and with the same scale on both axes, so the arrow angle "
       "and length can be compared; neither axis is a clinical scale. Claude Sonnet; generated futures, not patients.")
import numpy as np  # noqa: E402
from common import GREY  # noqa: E402
_d = [(m.x - A.loc[m.index, "map_x"], m.y - A.loc[m.index, "map_y"]) for c, _ in CONTEXTS for arm, _, _ in ARMS  # noqa: F405
      for m in [mean_moves(D, c, arm)]]  # noqa: F405
_dx = np.concatenate([np.asarray(a) for a, _ in _d]); _dy = np.concatenate([np.asarray(b) for _, b in _d])  # noqa: E702
_dx, _dy = _dx[~np.isnan(_dx)], _dy[~np.isnan(_dy)]
_mx, _my = 0.1 * (_dx.max() - _dx.min()), 0.1 * (_dy.max() - _dy.min())
LIM_X = (min(_dx.min(), 0) - _mx, max(_dx.max(), 0) + _mx)
LIM_Y = (min(_dy.min(), 0) - _my, max(_dy.max(), 0) + _my)
LIM_X, LIM_Y = equal_limits(LIM_X, LIM_Y, 47, 40)  # noqa: F405


def _lab(dx, dy, st, extra=0.0):
    n = np.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    rad, perp = {"S2": (3.0, -4.5), "S4": (7.0, 3.0)}.get(st, (4.5, 0.0))
    if extra and st == "S1":           # counted label at the right edge: place it above the tip
        return (-4.0, 6.0)
    rad += extra
    return (rad * ux - perp * uy, rad * uy + perp * ux)


for ctx, name, pid in (("EGFR_1L", "EGFR-mutant", "ed05c_direction_mutant"), ("EGFR_WT_1L", "EGFR-wild-type", "ed05d_direction_wildtype")):
    W, H = 60.0, 60.0
    fig = new_fig(W, H)
    ax = axes_mm(fig, 11, 10, 47, 40)
    mm = {arm: mean_moves(D, ctx, arm) for arm, _, _ in ARMS}  # noqa: F405
    ax.plot(LIM_X, [0, 0], color=GREY, lw=0.4, zorder=1)
    ax.plot([0, 0], LIM_Y, color=GREY, lw=0.4, zorder=1)
    for arm, an, col in ARMS:  # noqa: F405
        missing = []
        for st, r in mm[arm].iterrows():
            if r.n_ret == 0:
                missing.append(st[1])
                continue
            dx, dy = r.x - A.loc[st, "map_x"], r.y - A.loc[st, "map_y"]
            ax.add_patch(FancyArrowPatch((0, 0), (dx, dy), arrowstyle="-|>", mutation_scale=5, lw=0.4 + 1.4 * r.n_ret / r.n,
                                         color=col, shrinkA=0, shrinkB=0, zorder=4))
            lab = st[1] if r.n_ret == r.n else f"{st[1]} ({int(r.n_ret)}/{int(r.n)})"
            ax.annotate(lab, (dx, dy), xytext=_lab(dx, dy, st, 0.0 if lab == st[1] else 7.0), textcoords="offset points",
                        fontsize=6, ha="center", va="center", color=col if arm == "osimertinib" else "#8F3A72",
                        fontweight="bold", zorder=7)
        if missing:
            fig.text(11 / W, 1 - (1.2 + 2.6 * 2) / H, f"{an}, state {', '.join(missing)}: 0/{int(mm[arm].n.max())}, no arrow",
                     ha="left", va="top", fontsize=6, color=col)
    ax.scatter([0], [0], s=10, facecolor="white", edgecolor=INK, linewidth=0.7, zorder=6)  # noqa: F405
    ax.set_xlim(*LIM_X)
    ax.set_ylim(*LIM_Y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Change in state-map PC1 (a.u.)", fontsize=6, labelpad=2)
    ax.set_ylabel("Change in state-map PC2 (a.u.)", fontsize=6, labelpad=2)
    ret = {arm: int(mm[arm].n_ret.sum()) for arm, _, _ in ARMS}  # noqa: F405
    ntot = int(mm["chemo"].n.sum())
    for i, (arm, an, col) in enumerate(ARMS):  # noqa: F405
        fig.text(11 / W, 1 - (1.2 + 2.6 * i) / H, f"{an}: {ret[arm]}/{ntot} before endpoint at 6 mo", ha="left", va="top",
                 fontsize=6, color="#8F3A72" if arm == "chemo" else col)
    save_panel(fig, pid, legend=LEG.format(ctx=name), sources=[SRC],  # noqa: F405
               confirm=["Means are over futures with no first-line exit by 6 months, recomputed from the saved per-future table "
                        "(they are not read from arm_movement_summary.csv, which pools states).",
                        "Reprojected onto the frozen state map as in ed05a (display change only)."],
               message=f"Direction of movement per starting state for the two treatments ({name}).")
