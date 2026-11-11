"""One template for every population figure of the Clinical State Space.

Same starting patients, same follow-up: classical summary -> starting-state outcome map -> follow-up frames -> clinical ribbon.
The population is fixed at a clinical landmark and never replaced by other patients who later visit a region. Every quantity states its clock:
state ("position at month t"), recent events ("new in the interval"), outcome from the start ("alive at month h").
Group: dict(label, col, base, track, events=None).
  base   one row per patient or future, index id: x, y (start), t_os, e_os, optional t_pf, e_pf, w (default 1).
  track  rows id, t, x, y (NaN when not located) and any ribbon columns.
  events rows id, t, kind, x, y.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _css_grammar as G  # noqa: E402
from common import GREY, INK, PANELS, STATE_COLOURS, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

EVENT = {"progression": ("o", STATE_COLOURS["progression"], "Progression"), "toxicity": ("D", "#7B3294", "Grade ≥3 AE"), "death": ("X", "#000000", "Death")}
KNOWN = 0.8  # share of local patients whose status at the horizon must be known
COMMON = (" Frames: shading is relative to the peak density at the start, so it fades as patients die or go unassessed; contours enclose 50% and 90% of the positions located in that frame."
          " Ribbon: survival is Kaplan-Meier from the start; other rows are percentages among living patients with a known value in the interval, so denominators differ by row and month.")
ALIVE_MAP = LinearSegmentedColormap.from_list("alive", ["#FFFFFF", G.TEAL])


def _km_terms(t: np.ndarray, e: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique event times and, per row of w, weighted deaths over the risk set at each. Censoring at a time follows its deaths."""
    o = np.argsort(t, kind="stable")
    t, e, w = t[o], e[o], w[:, o]
    u = np.unique(t[e > 0])
    risk = np.concatenate([np.cumsum(w[:, ::-1], axis=1)[:, ::-1], np.zeros((len(w), 1))], axis=1)[:, np.searchsorted(t, u, "left")]
    dc = np.concatenate([np.zeros((len(w), 1)), np.cumsum(w * (e > 0), axis=1)], axis=1)
    d = dc[:, np.searchsorted(t, u, "right")] - dc[:, np.searchsorted(t, u, "left")]
    return u, np.where(risk > 0, d / np.maximum(risk, 1e-12), 0.0)


def wkm(t: np.ndarray, e: np.ndarray, w: np.ndarray, h: float) -> np.ndarray:
    """Weighted Kaplan-Meier at horizon h, one estimate per row of w: (rows, patients)."""
    u, f = _km_terms(np.asarray(t, float), np.asarray(e), np.asarray(w, float))
    return np.prod(1 - f[:, u <= h], axis=1)


def km_curve(t, e, w, grid) -> np.ndarray:
    u, f = _km_terms(np.asarray(t, float), np.asarray(e), np.asarray(w, float)[None, :])
    s = np.concatenate([[1.0], np.cumprod(1 - f[0])])
    return s[np.searchsorted(u, np.asarray(grid, float), "right")]


def km_band(b: pd.DataFrame, grid, n: int = 400, seed: int = 20261106, col: tuple[str, str] = ("t_os", "e_os")) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile band, resampling independent units: b.cluster (patient or starting record), else rows."""
    t, e = b[col[0]].to_numpy(float), b[col[1]].to_numpy(int)
    w = b.w.to_numpy(float) if "w" in b else np.ones(len(b))
    codes = pd.factorize(b.cluster if "cluster" in b else b.index)[0]
    m = codes.max() + 1
    rng = np.random.default_rng(seed)
    bs = np.array([km_curve(t, e, w * np.bincount(rng.integers(0, m, m), minlength=m)[codes], grid) for _ in range(n)])
    return tuple(np.percentile(bs, [2.5, 97.5], axis=0))


def endpoint_progression(b: pd.DataFrame) -> pd.DataFrame:
    """PFS events shown as progression: a PFS exit on the date of an actual death is shown only as that death; OS censoring does not count."""
    return b[(b.e_pf > 0) & ~((b.e_os > 0) & np.isclose(b.t_pf, b.t_os))]


def kernel(gx: np.ndarray, gy: np.ndarray, x: np.ndarray, y: np.ndarray, wt: np.ndarray, bx: float, by: float) -> np.ndarray:
    d = ((gx[:, None] - x[None, :]) / bx) ** 2 + ((gy[:, None] - y[None, :]) / by) ** 2
    return np.exp(-0.5 * d) * wt[None, :]


def outcome_map(ax, g: dict, h: float, xl, yl, bw: float, min_eff: float, cmap=ALIVE_MAP, vmin=0, vmax=100, n=40, discrete=False):
    """Alive at h by starting position: kernel-weighted KM, blank where support is thin. Returns the mappable."""
    b = g["base"].dropna(subset=["x", "y"])
    x, y, t, e = b.x.to_numpy(float), b.y.to_numpy(float), b.t_os.to_numpy(float), b.e_os.to_numpy(int)
    w = b.w.to_numpy(float) if "w" in b else np.ones(len(b))
    if discrete:
        pts = b.groupby(["x", "y"]).size().reset_index()
        val = [100 * wkm(t, e, ((x == r.x) & (y == r.y))[None, :] * w[None, :], h)[0] for r in pts.itertuples()]
        return ax.scatter(pts.x, pts.y, c=val, s=60, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=INK, lw=0.4, zorder=4)
    xs, ys = np.linspace(*xl, n), np.linspace(*yl, n)
    gx, gy = [a.ravel() for a in np.meshgrid(xs, ys)]
    bx, by = bw * (xl[1] - xl[0]), bw * (yl[1] - yl[0])
    K = kernel(gx, gy, x, y, w, bx, by)
    v = 100 * wkm(t, e, K, h)
    known = ((e > 0) & (t <= h)) | (t >= h)  # status at the horizon is known, not censored before it
    kk = (K * known[None, :]).sum(axis=1)
    v[(kk < min_eff) | (kk < KNOWN * K.sum(axis=1))] = np.nan
    return ax.contourf(xs, ys, v.reshape(n, n), levels=np.linspace(vmin, vmax, 11), cmap=cmap, vmin=vmin, vmax=vmax, extend="both")


def position_at(g: dict, t: float, window: float) -> pd.DataFrame:
    """Last located record of each patient in (t - window, t], among those alive and followed at t."""
    b = g["base"]
    tr = g["track"]
    q = tr[(tr.t <= t + 1e-9) & (tr.t > t - window)].dropna(subset=["x"]).groupby("id").tail(1).set_index("id")
    return q[q.index.isin(b.index[b.t_os > t])]


def displacement(ax, g: dict, h: float, window: float, xl, yl, bw: float, min_eff: float, cells=7, discrete=False) -> None:
    """Baseline-anchored displacement: mean move from baseline to month h among survivors located at h."""
    b = g["base"].dropna(subset=["x", "y"])
    p = position_at(g, h, window)
    m = b.join(p[["x", "y"]], rsuffix="_h", how="inner")
    w = m.w.to_numpy(float) if "w" in m else np.ones(len(m))
    G.cloud(ax, b[["x", "y"]].to_numpy(float), "#9AA3AD", xl, yl, sigma=3.0, weights=b.w.to_numpy(float) if "w" in b else None)
    if discrete:
        for (x0, y0), d in m.groupby(["x", "y"]):
            ax.annotate("", xy=(d.x_h.mean(), d.y_h.mean()), xytext=(x0, y0), zorder=5, arrowprops=dict(arrowstyle="-|>", color=g["col"], lw=1.0, mutation_scale=6, shrinkA=0, shrinkB=0))
        return
    xs, ys = np.linspace(xl[0], xl[1], cells + 2)[1:-1], np.linspace(yl[0], yl[1], cells + 2)[1:-1]
    gx, gy = [a.ravel() for a in np.meshgrid(xs, ys)]
    K = kernel(gx, gy, m.x.to_numpy(float), m.y.to_numpy(float), w, bw * (xl[1] - xl[0]), bw * (yl[1] - yl[0]))
    dx = (K * (m.x_h - m.x).to_numpy()).sum(axis=1) / np.maximum(K.sum(axis=1), 1e-12)
    dy = (K * (m.y_h - m.y).to_numpy()).sum(axis=1) / np.maximum(K.sum(axis=1), 1e-12)
    k = K.sum(axis=1) >= min_eff
    ax.quiver(gx[k], gy[k], dx[k], dy[k], angles="xy", scale_units="xy", scale=1, color=g["col"], width=0.012, headwidth=3.6, headlength=4, zorder=5)


def frame(ax, g: dict, t: float, prev: float, window: float, xl, yl, points: bool, sigma: float, two_lines: bool = False) -> None:
    b = g["base"]
    p = position_at(g, t, window)
    w = b.w.reindex(p.index).to_numpy(float) if "w" in b else None
    G.cloud(ax, p[["x", "y"]].to_numpy(float), g["col"], xl, yl, sigma=sigma, weights=w, min_n=8, ref=g.get("ref"))
    ev = g.get("events")
    if ev is not None and points:
        for k in ev.kind.unique():
            mk, col, _ = EVENT[k]
            q = ev[ev.kind.eq(k) & ev.t.le(t)]
            old = q[q.t.le(prev)]
            nw = q[q.t.gt(prev)]
            ax.scatter(old.x, old.y, marker=mk, s=4, color=col, alpha=0.18, lw=0, zorder=3)
            ax.scatter(nw.x, nw.y, marker=mk, s=9, color=col, edgecolor="white", linewidth=0.2, zorder=6)
    if points:
        ax.scatter(p.x, p.y, s=1.4, color=INK, lw=0, zorder=4)
    G.bare(ax, xl, yl)
    ww = b.w if "w" in b else pd.Series(1.0, index=b.index)
    alive = float(ww[(b.t_os > t)].sum())
    died = float(ww[(b.t_os <= t) & (b.e_os > 0)].sum())
    cens = float(ww[(b.t_os <= t) & (b.e_os == 0)].sum())
    loc = float(ww.reindex(p.index).sum())
    v = {"alive": alive, "dead": died, "cens": cens, "loc": loc}
    hide = set()  # counts not printed exactly
    if g.get("small_cells"):
        def small(n: float) -> bool:
            return 0.5 <= n < 4.5
        # alive + dead + censored = the group total, which the legend gives: a count of 1 to 4 must not be recoverable
        # by subtraction, including the censored count when it is not printed (one-line caption)
        shown = {"alive", "dead", "cens"} if two_lines else {"alive", "dead"}
        hide = {k for k in ("alive", "dead", "cens") if small(v[k])}
        unknown = hide | ({"alive", "dead", "cens"} - shown)
        if len(unknown) == 1 and small(v[next(iter(unknown))]):
            rest = [k for k in shown - hide if v[k] >= 0.5]
            if rest:
                hide.add(min(rest, key=lambda k: v[k]))
        # located + not located = alive: neither may be 1 to 4 when alive is printed
        if two_lines and (small(v["loc"]) or small(alive - loc)):
            hide.add("loc")

    def c(k: str) -> str:
        n = v[k]
        if k not in hide:
            return f"{n:.0f}"
        return "<5" if 0.5 <= n < 4.5 else "≥5"
    txt = f"{c('alive')} alive, {c('dead')} dead"
    if two_lines:
        txt = f"{c('alive')} alive ({c('loc')} located)\n{c('dead')} dead, {c('cens')} censored"
    ax.text(0.5, -0.03, txt, transform=ax.transAxes, fontsize=6, ha="center", va="top", clip_on=False)


def ribbon_matrix(g: dict, rows: list, times: list, window: float) -> tuple[np.ndarray, np.ndarray]:
    b = g["base"]
    w = b.w.to_numpy(float) if "w" in b else np.ones(len(b))
    M = np.full((len(rows), len(times)), np.nan)
    N = np.full(M.shape, np.nan)
    for j, t in enumerate(times):
        tr = g["track"]
        c = tr[(tr.t <= t + 1e-9) & (tr.t > t - window)].groupby("id").tail(1).set_index("id")
        c = c[c.index.isin(b.index[b.t_os > t])]
        ww = b.w.reindex(c.index).to_numpy(float) if "w" in b else np.ones(len(c))
        for i, (lab, col, _) in enumerate(rows):
            if col == "ALIVE":
                M[i, j] = 100 * km_curve(b.t_os.to_numpy(float), b.e_os.to_numpy(int), w, [t])[0]
            elif col == "PF":
                M[i, j] = 100 * km_curve(b.t_pf.to_numpy(float), b.e_pf.to_numpy(int), w, [t])[0]
            elif ww.sum() >= 5:
                v = pd.to_numeric(c[col], errors="coerce").to_numpy(float)
                k = np.isfinite(v)
                num = (v[k] * ww[k]).sum()
                if ww[k].sum() >= 5 and not (g.get("small_cells") and (0 < num < 5 or 0 < ww[k].sum() - num < 5)):  # count or complement 1-4
                    M[i, j] = 100 * num / ww[k].sum()
                    N[i, j] = ww[k].sum()
                elif ww[k].sum() >= 5:
                    N[i, j] = ww[k].sum()  # masked small cell: denominator kept, share blank
        # rows that partition the same denominator (RECIST categories among evaluable assessments): when exactly one
        # is masked, also mask the smallest other, so the masked share cannot be recovered by subtraction
        part = [i for i, (_l, col, _c) in enumerate(rows) if col in PARTITION_ROWS]
        if g.get("small_cells") and len(part) == len(PARTITION_ROWS):
            masked = [i for i in part if np.isnan(M[i, j]) and np.isfinite(N[i, j])]
            shown = [i for i in part if np.isfinite(M[i, j])]
            if len(masked) == 1 and shown:
                M[min(shown, key=lambda i: M[i, j]), j] = np.nan
    return M, N


PARTITION_ROWS = ("crpr", "sd", "pd")


def page(panel: str, groups: list[dict], start: str, axes: str, frames: list, h: float, hd: float, window: float, xl, yl, rows: list, rib_times: list, km_max: float,
         forest: list | None, forest_label: str, legend: str, sources: list, confirm: list, message: str, bw=0.14, min_eff=12.0, points=True, sigma=3.0, discrete=False,
         pf=True, height=185.0, gap=3.6, tlabel="Months from start", xlab="", ylab="", ha=30.0, hb=29.0, fmax=26.0, rib_h=34.0, key_pad=12.0,
         frame_groups=None, frame_w=165.0, side=None, klabel="Patients (%)", axes_note="", paired=False, key_title=None,
         parts: str = "abcd", letters: str | None = None) -> None:
    """parts: which rows to draw (a summary, b starting-state maps, c frames, d course); letters: their panel letters."""
    letters = dict(zip("abcd", letters or "abcd"))
    W, H = 180.0, height
    fig = new_fig(W, H)
    ng = len(groups)
    legend = legend + " " + axes_note + COMMON
    for g in groups:
        b0 = g["base"].dropna(subset=["x", "y"])
        g["ref"] = G.peak(b0[["x", "y"]].to_numpy(float), xl, yl, sigma, b0.w.to_numpy(float) if "w" in b0 else None)
    # a, classical summary
    ya = H - 6 - ha if "a" in parts else H - 6
    grid = np.linspace(0, km_max, 121)
    kms = [] if "a" not in parts else [axes_mm(fig, 12 + i * 31, ya, 28, ha) for i in range(2)] if paired else [axes_mm(fig, 12, ya, 58, ha)]
    ax = kms[0] if kms else None
    for gi, g in enumerate(groups if kms else []):
        ax = kms[gi // 2] if paired else ax
        b = g["base"]
        w = b.w.to_numpy(float) if "w" in b else np.ones(len(b))
        ax.step(grid, km_curve(b.t_os.to_numpy(float), b.e_os.to_numpy(int), w, grid), where="post", color=g["col"], lw=1.0)
        lo, hi = km_band(b, grid)
        ax.fill_between(grid, lo, hi, step="post", color=g["col"], alpha=0.15, lw=0)
        if pf and "t_pf" in b:
            ax.step(grid, km_curve(b.t_pf.to_numpy(float), b.e_pf.to_numpy(int), w, grid), where="post", color=g["col"], lw=1.0, ls=(0, (3, 1.5)))
    for i, ax in enumerate(kms):
        ax.set_xlim(0, km_max)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "50", "100"] if i == 0 else [])
        ax.set_xlabel(tlabel, labelpad=1, fontsize=6)
        if i == 0:
            ax.set_ylabel(klabel, labelpad=1, fontsize=6)
        if paired:
            ax.set_title(groups[2 * i]["block"], fontsize=6, pad=1.5)
    hs = [Line2D([], [], color=g["col"], lw=1.2, label=f'{g.get("arm", g["label"])} (n = {g["base"].w.sum() if "w" in g["base"] else len(g["base"]):.0f})') for g in (groups[:2] if paired else groups)]
    if key_title:
        hs.insert(0, Line2D([], [], color="none", label=key_title))
    if pf and "t_pf" in groups[0]["base"] and "a" in parts:
        hs.append(Line2D([], [], color=INK, lw=1, ls=(0, (3, 1.5)), label="Solid, OS; dashed, PFS"))
    if "a" in parts:
        fig.legend(handles=hs, fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(134 / W, (H - 7) / H), handlelength=1.5, labelspacing=0.15, borderaxespad=0, borderpad=0)
    else:
        fig.legend(handles=hs, fontsize=6, frameon=False, loc="upper left", ncol=len(hs), bbox_to_anchor=(60 / W, (H - 0.6) / H), handlelength=1.5, columnspacing=1.2, borderaxespad=0, borderpad=0)
    # forest
    if forest and "a" in parts:
        fx = axes_mm(fig, 104, ya, 26, ha)
        for i, (lab, e, lo, hi, *ref) in enumerate(forest):
            fx.plot([lo, hi], [-i, -i], color=INK, lw=0.8)
            fx.plot(e, -i, "s", ms=3, color=INK)
            if ref:
                fx.plot([ref[1], ref[2]], [-i - 0.3] * 2, color=GREY, lw=0.6)
                fx.plot(ref[0], -i - 0.3, "o", ms=2.6, mfc="white", mec=GREY, mew=0.6)
        if min(f[2] for f in forest) < 0:
            fx.plot([0, 0], [-len(forest) + 0.3, 0.6], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
        fx.set_yticks([-i for i in range(len(forest))])
        fx.set_yticklabels([f[0] for f in forest], fontsize=6)
        fx.tick_params(axis="y", length=0)
        fx.spines["left"].set_visible(False)
        fx.set_ylim(-len(forest) + 0.3, 0.6)
        fx.set_xlabel(forest_label, labelpad=1, fontsize=6)
    # b, starting-state outcome maps
    y2, h2 = (ya - (15.0 if paired else 12.5) - hb, hb) if "b" in parts else (ya + 7.0, 0.0)
    n = 2 * ng
    pw = min(40.0, (165 - 1.5 * (n - 1)) / n)
    x = 12.0
    for g in (groups if "b" in parts else []):
        a1 = axes_mm(fig, x, y2, pw, h2)
        im = outcome_map(a1, g, h, xl, yl, bw, min_eff, discrete=discrete)
        G.bare(a1, xl, yl)
        if x == 12.0 and xlab:
            a1.set_xlabel(xlab, labelpad=0.5, fontsize=6)
            a1.set_ylabel(ylab, labelpad=0.5, fontsize=6)
        a1.set_title(f"Alive at {h:.0f} mo" if paired else f"{g.get('short', g['label'])}: alive at {h:.0f} mo" if ng < 4 else f"{g['label']}: alive {h:.0f} mo", fontsize=6, pad=1.5)
        a2 = axes_mm(fig, x + pw + 1.5, y2, pw, h2)
        displacement(a2, g, hd, window, xl, yl, bw, min_eff, discrete=discrete)
        G.bare(a2, xl, yl)
        a2.set_title(f"Survivors, to {hd:.0f} mo" if paired else f"Move to {hd:.0f} mo, survivors" if ng < 3 else f"Move to {hd:.0f} mo", fontsize=6, pad=1.5)
        if paired:
            fig.text((x + pw + 0.75) / W, (y2 + h2 + 3.2) / H, f"{g['block'].replace('EGFR ', '').replace('EGFR-', '').capitalize()}, {g['arm'].lower()}", fontsize=6, ha="center", va="bottom")
        x += 2 * (pw + 1.5)
    if "b" in parts:
        cb = fig.add_axes([(82 if paired else 24) / W, (y2 - 5.6) / H, 28 / W, 1.3 / H])
        fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[0, 50, 100])
        cb.tick_params(labelsize=6, length=1.2, pad=0.5)
        cb.set_ylabel("Alive (%)", fontsize=6, rotation=0, ha="right", va="center", labelpad=5)
    # c, follow-up frames (skipped when "c" is not in parts)
    y3top = y2 - (13.4 if "b" in parts else 12.0)
    if "c" in parts:
        y3top = y2 - (13.4 if "b" in parts else 12.0)
        fg = [g for g in groups if frame_groups is None or g["label"] in frame_groups]
        nf = 2 if paired else len(fg)
        key_pad = max(key_pad, gap + 6.0)
        fh = min(fmax, (y3top - rib_h - key_pad - gap * (nf - 1)) / nf)
        cols = 2 * len(frames) if paired else len(frames)
        fw = (frame_w - 1.2 * (cols - 1) - (2.0 if paired else 0.0)) / cols
        cx = [12 + c * (fw + 1.2) + (2.0 if paired and c >= len(frames) else 0.0) for c in range(cols)]
        for c in range(cols):
            fig.text((cx[c] + fw / 2) / W, (y3top + 0.4) / H, f"Month {frames[c % len(frames)]}", fontsize=6.5, ha="center", va="bottom")
        if paired:
            for b in range(2):
                fig.text((cx[b * len(frames)] + (len(frames) * (fw + 1.2)) / 2) / W, (y3top + 3.6) / H, groups[2 * b]["block"], fontsize=6.5, ha="center", va="bottom")
        for r, g in enumerate(fg):
            row, blk = (r % 2, r // 2) if paired else (r, 0)
            y = y3top - (row + 1) * fh - row * gap
            # without the starting-position row, the frames carry the shared axis labels (bottom-left frame)
            shared = "b" not in parts and bool(xlab)
            if blk == 0:
                fig.text((7.6 if shared else 10.8) / W, (y + fh / 2) / H, g.get("arm", g.get("rowlab", g["label"])), fontsize=6, rotation=90, ha="right", va="center")
            for j, t in enumerate(frames):
                prev = frames[j - 1] if j else -1e9
                a = axes_mm(fig, cx[blk * len(frames) + j], y, fw, fh)
                frame(a, g, t, prev, window, xl, yl, points, sigma, gap > 4)
                if shared and j == 0 and blk == 0 and r == len(fg) - 1:
                    a.set_ylabel(ylab, labelpad=0.5, fontsize=6, loc="bottom")
                    # below the alive/dead counts printed under each frame
                    fig.text(cx[0] / W, (y - 7.4) / H, xlab, fontsize=6, ha="left", va="center")
        if side:
            side(fig, 12 + frame_w + 8.0, y3top - nf * fh - (nf - 1) * gap, 168 - frame_w - 8.0 - 2.0, nf * fh + (nf - 1) * gap)
        kinds = sorted({k for g in groups if g.get("events") is not None for k in g["events"].kind.unique()})
        xk = 12.0 + (4.0 + 1.3 * len(xlab) if "b" not in parts and xlab else 0.0)
        yk = y3top - nf * fh - (nf - 1) * gap - gap - 1.6
        if points and kinds:
            for k in kinds:
                mk, col, lab = EVENT[k]
                fig.add_artist(Line2D([xk / W], [yk / H], marker=mk, ms=3, color=col, ls="", transform=fig.transFigure))
                fig.text((xk + 1.8) / W, yk / H, lab, fontsize=6, va="center")
                xk += 5 + 1.2 * len(lab)
            pass
        else:
            pass
    # d, clinical ribbon
    rw = (153 - 1.2 * (ng - 1)) / ng
    dn, shown = [], []
    for i, g in enumerate(groups):
        M, N = ribbon_matrix(g, rows, rib_times, window)
        dn.append(pd.DataFrame(N, index=[q[0] for q in rows], columns=rib_times).assign(group=g["label"]))
        shown.append(pd.DataFrame(M, index=[q[0] for q in rows], columns=rib_times).assign(group=g["label"], observed=bool(g.get("small_cells"))))
        rb = axes_mm(fig, 24 + i * (rw + 1.2), 7, rw, rib_h - 8)
        G.ribbon(rb, M, [q[0] for q in rows], rib_times, [q[2] for q in rows], show_labels=i == 0, title=g["label"])
        rb.set_xlabel("Months", labelpad=1, fontsize=6)
    for lab, yy, head in (("a", H - 1, "Classical summary"), ("b", y2 + h2 + (9.0 if paired else 6.4), "Outcome by starting position"), ("c", y3top + 5.5, "Position over time"), ("d", rib_h + 2, "Clinical course")):
        if lab not in parts:
            continue
        lab = letters[lab]
        fig.text(1.5 / W, yy / H, lab, fontsize=8, fontweight="bold", va="top")
        fig.text(5.5 / W, (yy - 0.35) / H, head, fontsize=6.5, va="top")
    save_panel(fig, panel, legend=legend, sources=sources, confirm=confirm, message=message, strict=False)
    pd.concat(dn).round(1).rename_axis("row").to_csv(PANELS / panel / f"{panel}_ribbon_denominators.csv")
    # shown percentages with their denominators, for the small-cell check (scripts/release/check_small_cells.py)
    pd.concat(shown).round(3).rename_axis("row").to_csv(PANELS / panel / f"{panel}_ribbon_shown.csv")
