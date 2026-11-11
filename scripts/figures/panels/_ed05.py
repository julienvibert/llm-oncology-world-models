"""Shared loading and drawing helpers for Extended Data Fig. 5 (EGFR futures in the state map).

Data: outputs/paper_egfr_remission_attractor/trajectory_horizon_attractor.csv (per future, every 0.25 month to 18
months: map position, first-line exit flag, clinical-remission flag) and attractor_occupancy_landmarks.csv.
Only chemotherapy and osimertinib arms are drawn (the same two arms as main Fig. 3b-e).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PAL, CHEMO, INK, STATE_COLOURS  # noqa: E402,F401  (ROOT re-exported to ed05e)

SRC = "outputs/paper_egfr_frozen_map_reprojection/trajectory_horizon_frozen_map.csv"
SRC_ORIG = "outputs/paper_egfr_remission_attractor/trajectory_horizon_attractor.csv"
SRC_LM = "outputs/paper_egfr_remission_attractor/attractor_occupancy_landmarks.csv"
OSI = PAL["osimertinib"]
ARMS = (("chemo", "Chemotherapy", CHEMO), ("osimertinib", "Osimertinib", OSI))
CONTEXTS = (("EGFR_1L", "EGFR-mutant"), ("EGFR_WT_1L", "EGFR-wild-type"))
EXIT_COL = STATE_COLOURS["progression"]
STATE_WORDS = {"S1": "low burden, fit", "S2": "standard metastatic", "S3": "brain metastases, fit",
               "S4": "high burden, fit", "S5": "high burden, frail"}
XLAB = "State-map PC1 (a.u.)"
YLAB = "State-map PC2 (a.u.)"
FAN_H = 12.0  # months drawn
ARROW_H = 6.0  # months over which the mean movement is taken


def load() -> pd.DataFrame:
    from _reproject import reprojected
    d = reprojected()  # frozen-map positions; every other column as saved
    return d[d.arm.isin([a for a, _, _ in ARMS])].copy()


def equal_limits(xl, yl, w_mm: float, h_mm: float):
    """Expand the two limits about their centres so one data unit has the same length on both axes."""
    s = max((xl[1] - xl[0]) / w_mm, (yl[1] - yl[0]) / h_mm)
    cx, cy = sum(xl) / 2, sum(yl) / 2
    return (cx - s * w_mm / 2, cx + s * w_mm / 2), (cy - s * h_mm / 2, cy + s * h_mm / 2)


def paths(d: pd.DataFrame, context: str, arm: str, horizon: float = FAN_H):
    """Yield per future: (xs, ys, exited_within_horizon, state). A path stops at its last position before exit."""
    sub = d[(d.context == context) & (d.arm == arm) & (d.horizon_months <= horizon + 1e-9)]
    for _, g in sub.groupby("trajectory_id"):
        g = g.sort_values("horizon_months")
        live = g[~g.exit_by_horizon]
        exited = bool(g.exit_by_horizon.iloc[-1])
        yield live.map_x.to_numpy(), live.map_y.to_numpy(), exited, g.state.iloc[0]


def anchors(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d.horizon_months == 0)].groupby("state")[["map_x", "map_y"]].mean()


def limits(d: pd.DataFrame, context: str, pad: float = 0.06):
    s = d[(d.context == context) & (d.horizon_months <= FAN_H)]
    s = s[~s.exit_by_horizon | (s.horizon_months == 0)]
    x0, x1, y0, y1 = s.map_x.min(), s.map_x.max(), s.map_y.min(), s.map_y.max()
    return (x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)), (y0 - pad * (y1 - y0), y1 + pad * (y1 - y0))


def mean_moves(d: pd.DataFrame, context: str, arm: str, horizon: float = ARROW_H) -> pd.DataFrame:
    """Per starting state: mean position among futures still on first line at the horizon (retained)."""
    h = d[(d.context == context) & (d.arm == arm) & np.isclose(d.horizon_months, horizon)]
    out = []
    for st, g in h.groupby("state"):
        r = g[~g.exit_by_horizon]
        out.append(dict(state=st, n=len(g), n_ret=len(r),
                        x=r.map_x.mean() if len(r) else np.nan, y=r.map_y.mean() if len(r) else np.nan))
    return pd.DataFrame(out).set_index("state")


def draw_fans(ax, d, context, arm, col, xl, yl, show_y=True, label_anchors=True):
    """All 45 futures of one arm from the five starting states as paths in the state map."""
    n = ex = 0
    for xs, ys, exited, _ in paths(d, context, arm):
        if len(xs) < 2:
            if exited:
                ex += 1
            n += 1
            continue
        ax.plot(xs, ys, color=col, lw=0.45, alpha=0.55, solid_capstyle="round", zorder=2)
        if exited:
            ax.plot(xs[-1], ys[-1], marker="x", ms=2.6, mew=0.7, color=EXIT_COL, zorder=4)
            ex += 1
        else:
            ax.plot(xs[-1], ys[-1], marker="o", ms=1.8, mfc=col, mec="white", mew=0.3, zorder=3)
        n += 1
    a = anchors(d)
    ax.scatter(a.map_x, a.map_y, s=16, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=6)
    if label_anchors:
        off = {"S1": (-3, 3, "right"), "S2": (-1, 5.5, "center"), "S3": (4, 1, "left"),
               "S4": (4, -5, "left"), "S5": (-4, 0, "right")}
        for st, (dx, dy, ha) in off.items():
            ax.annotate(st[1], (a.loc[st, "map_x"], a.loc[st, "map_y"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, ha=ha, va="center", color=INK, fontweight="bold", zorder=7)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(XLAB, fontsize=6, labelpad=2)
    if show_y:
        ax.set_ylabel(YLAB, fontsize=6, labelpad=2)
    return n, ex
