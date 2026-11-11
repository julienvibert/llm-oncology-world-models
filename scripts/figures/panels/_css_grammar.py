"""One visual grammar for every Clinical State Space figure: only the patients and the coordinates change.

cloud      smoothed density of a population in a frame, with 50% and 90% contours
flow       landscape of where patients stay (-log time spent) with mean movement arrows on top
ribbon     clinical variables across time, one colour family per kind of variable
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INK  # noqa: E402

TEAL, BLUE, PURPLE, OCHRE, ORANGE = "#1F6F66", "#2A6FBF", "#7B3294", "#B5761A", "#D55E00"


def bare(ax, xl, yl) -> None:
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([])
    ax.set_yticks([])


def peak(xy: np.ndarray, xl, yl, sigma: float = 3.0, weights=None) -> float:
    """Peak smoothed density of a population, the common reference for its later frames."""
    h, *_ = np.histogram2d(xy[:, 0], xy[:, 1], bins=64, range=[xl, yl], weights=weights)
    return float(gaussian_filter(h, sigma).max())


def cloud(ax, xy: np.ndarray, col: str, xl, yl, sigma: float = 3.0, weights=None, min_n: float = 20, ref: float | None = None) -> None:
    """Shading and 50% and 90% contours; nothing below min_n. ref: shading relative to this density (default, own peak)."""
    n = len(xy) if weights is None else float(np.sum(weights))
    if n < min_n:
        return
    h, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=64, range=[xl, yl], weights=weights)
    d = gaussian_filter(h, sigma)
    rgba = np.zeros((*d.T.shape, 4))
    rgba[..., :3] = to_rgb(col)
    rgba[..., 3] = 0.7 * np.minimum(d.T / (ref or d.max()), 1.0)
    ax.imshow(rgba, origin="lower", extent=(*xl, *yl), aspect="auto", zorder=1)
    flat = np.sort(d.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    lv = sorted({float(flat[np.searchsorted(cum, q)]) for q in (0.9, 0.5)})
    ax.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, d.T, levels=lv, colors=[col],
               linewidths=[0.4, 0.7][: len(lv)], zorder=2)


def flow(ax, x, y, dx, dy, xl, yl, bins: int = 56, sigma: float = 1.6, arrows: int = 13, min_arrow: int = 12,
         arrow_col: str = INK):
    """Landscape of time spent, with mean movement on a coarser grid (cells with at least min_arrow samples)."""
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xl, yl])
    n = gaussian_filter(h, sigma).T
    u = np.where(n > 0.15, -np.log(np.maximum(n, 1e-9) / n.max()), np.nan)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    cf = ax.contourf(xc, yc, u, levels=np.arange(0, 5.01, 0.5), cmap="YlGnBu_r", extend="max", alpha=0.9)
    ok = np.isfinite(dx) & np.isfinite(dy)
    ge = np.linspace(*xl, arrows + 1), np.linspace(*yl, arrows + 1)
    i = np.clip(np.digitize(x[ok], ge[0]) - 1, 0, arrows - 1)
    j = np.clip(np.digitize(y[ok], ge[1]) - 1, 0, arrows - 1)
    px, py, ux, uy = [], [], [], []
    for a in range(arrows):
        for b in range(arrows):
            m = (i == a) & (j == b)
            if m.sum() >= min_arrow:
                px.append(x[ok][m].mean())
                py.append(y[ok][m].mean())
                ux.append(dx[ok][m].mean())
                uy.append(dy[ok][m].mean())
    if px:
        ax.quiver(px, py, ux, uy, angles="xy", scale_units="xy", scale=1, color=arrow_col, width=0.008, headwidth=4,
                  headlength=4.5, zorder=4)
    return cf


def ribbon(ax, m: np.ndarray, labels: list[str], times: list, colours: list[str], show_labels: bool = True,
           numbers: bool = True, title: str = "") -> None:
    """Rows, clinical variables (%); columns, times. Blank when not estimable (nan)."""
    rgb = np.ones((*m.shape, 3))
    for i in range(m.shape[0]):
        v = np.nan_to_num(m[i], nan=0.0)[:, None] / 100
        rgb[i] = 1 - v * (1 - np.array(to_rgb(colours[i])))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    if numbers:
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                if np.isfinite(m[a, b]):
                    ax.text(b, a, f"{m[a, b]:.0f}", fontsize=6, ha="center", va="center",
                            color="white" if m[a, b] > 60 else INK)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([str(t) for t in times], fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if show_labels else [], fontsize=6)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=6.5, pad=2)


def fan(ax, origin, pts: np.ndarray, col: str, lines: bool = True, max_lines: int = 60) -> None:
    """One patient: thin lines from the patient to where similar futures are, and a bold mean arrow."""
    if not len(pts):
        return
    if lines:
        for q in pts[:: max(1, len(pts) // max_lines)]:
            ax.plot([origin[0], q[0]], [origin[1], q[1]], color=col, lw=0.25, alpha=0.45, zorder=2)
    m = pts.mean(axis=0)
    ax.annotate("", xy=m, xytext=origin, zorder=7, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0))
