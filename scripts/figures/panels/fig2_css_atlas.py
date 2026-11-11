"""Panel fig2_css_atlas: the Clinical State Space, from one generated course to the measured atlas.

a, one generated course (Claude Sonnet 5, EGFR-mutant lung adenocarcinoma on osimertinib; the future with the median survival
of its group) as a disease-state band and as frames zoomed on the patient: gray path so far, bold arrow from the position at
the previous frame, events on the path. b, the generated atlas of four cancer families, six models pooled: patients alive and
dead at successive months, and the Kaplan-Meier curve of the same patients. c, the same four families on the disease and host
axes at 0, 12, 24 and 36 months: density contours of living patients and, in every region with enough patients, the mean move
since the previous frame of the patients alive at both times. Model-generated data.
Run: uv run python scripts/figures/panels/fig2_css_atlas.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _css_axes as A  # noqa: E402
import _css_grammar as G  # noqa: E402
import _css_template as T  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import FAMILY_COLOURS, INK, PAL, ROOT, STATE_COLOURS, save_panel  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "scripts" / "explore" / "figpanels"))
import build_paper_precision_ladder_paired_views as paired  # noqa: E402

M = 30.4375
FAMS = [("Lung", "lung"), ("Gastrointestinal", "gi"), ("Sarcoma", "sarcoma"), ("Breast", "breast")]
FR_A = [0, 12, 24, 36, 48]
FR_C = [0, 12, 24, 36]
CELLS, MIN_CELL = 6, 8

# ---------------------------------------------------------------- a: one generated course
st, en, _ = paired.load_phase1()
st = st[st.context.eq("EGFR_1L") & st.arm.eq("osimertinib")].copy()
en = en[en._gid.isin(st._gid.unique())].set_index("_gid")
st["x"], st["y"] = A.project(st)
st["t"] = (st.days_from_diagnosis - st.cutoff_days) / M
st = st.sort_values(["_gid", "t", "snapshot_index"])
anchor = st.groupby("_gid").cutoff_days.first()
os_m = (en.os_days - anchor) / M
one = (os_m - os_m.median()).abs().idxmin()
T_END = float(os_m[one])
path = st[st._gid.eq(one) & st.t.ge(0)]
ev = pd.read_csv(ROOT / "runs/interventional_phase1_sonnet_full_v1c/derivation/event_rows.csv", low_memory=False)
ev = ev[ev.trajectory_id.eq(path.trajectory_id.iloc[0])].copy()
ev["t"] = (ev.days_from_diagnosis - anchor[one]) / M


def at(t: float) -> np.ndarray:
    c = path[path.t.le(t + 1e-9)]
    return c[["x", "y"]].iloc[-1].to_numpy(float)


# ---------------------------------------------------------------- b, c: the atlas
from build_field_trellis import CODE2FAM  # noqa: E402
import fig1_data as F1  # noqa: E402

sn = F1.enrich_snapshots(F1.load_tables())
end = pd.DataFrame({"t": sn.groupby("_gid").days_from_diagnosis.max() / M,
                    "e": sn.groupby("_gid").outcome_type.first().eq("death").astype(int)})
a = pd.read_csv(ROOT / "outputs/paper_state_map/state_projection.csv.gz", low_memory=False)
a["fam"] = a.tumor_oncotree_code.astype(str).map(CODE2FAM)
a = a[a.fam.isin([f for f, _ in FAMS])].copy()
a["x"], a["y"] = A.project(a)
a["t"] = a.days_from_diagnosis / M
a = a.sort_values(["_gid", "t", "snapshot_index"])
XL = (float(a.x.min()) - 0.1, float(a.x.max()) + 0.1)
YL = (float(a.y.min()) - 0.1, float(a.y.max()) + 0.1)
ids = a._gid.unique()
t_os, e_os = end.t.reindex(ids), end.e.reindex(ids)


def positions(q: pd.DataFrame, t: float, alive: pd.Index) -> pd.DataFrame:
    c = q[q.t.le(t + 1e-9) & q._gid.isin(alive)]
    return c.groupby("_gid")[["x", "y"]].last()


W, H = 180.0, 183.0
fig = new_fig(W, H)

# a
ta = axes_mm(fig, 12, H - 22, 48, 7)
cuts = [(0.0, "unresolved")]
for _, r in ev[ev.event_type.isin(["response", "progression"]) & ev.t.gt(0)].sort_values("t").iterrows():
    if r.event_type != cuts[-1][1]:
        cuts.append((float(r.t), r.event_type))
for k, (t0, s_) in enumerate(cuts):
    t1 = cuts[k + 1][0] if k + 1 < len(cuts) else T_END
    ta.barh(0, t1 - t0, left=t0, height=0.8, color=STATE_COLOURS[s_], lw=0)
ta.scatter([T_END], [0], marker="X", s=16, color="#000000", zorder=4, clip_on=False)
ta.set_xlim(0, 48)
ta.set_ylim(-0.5, 0.5)
ta.set_yticks([])
ta.spines["left"].set_visible(False)
ta.set_xticks([0, 12, 24, 36, 48])
ta.set_xlabel("Months from treatment start", labelpad=1, fontsize=6)
px, py = path.x.to_numpy(float), path.y.to_numpy(float)
ZX = (px.min() - 0.15 * np.ptp(px) - 0.02, px.max() + 0.15 * np.ptp(px) + 0.02)
ZY = (py.min() - 0.15 * np.ptp(py) - 0.02, py.max() + 0.15 * np.ptp(py) + 0.02)
ev = ev.assign(px=[at(max(q, 0))[0] for q in ev.t], py=[at(max(q, 0))[1] for q in ev.t])
for j, t in enumerate(FR_A):
    ax = axes_mm(fig, 66 + j * 22.6, H - 30, 21.6, 22)
    tr = path[path.t.le(min(t, T_END) + 1e-9)]
    ax.plot(tr.x, tr.y, color="#9AA3AD", lw=0.6, zorder=2)
    for kind in ("response", "progression"):
        e = ev[ev.event_type.eq(kind) & ev.t.gt(0) & ev.t.le(t)]
        new = e.t.gt(FR_A[j - 1]) if j else e.t.ge(0)
        ax.scatter(e.px[~new], e.py[~new], s=8, color=STATE_COLOURS[kind], alpha=0.35, lw=0, zorder=3)
        ax.scatter(e.px[new], e.py[new], s=16, color=STATE_COLOURS[kind], lw=0, zorder=4)
    cur = at(min(t, T_END))
    if j:
        prev = at(FR_A[j - 1])
        if np.hypot(*(cur - prev)) > 1e-6:
            ax.annotate("", xy=cur, xytext=prev, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=7,
                                                                  shrinkA=0, shrinkB=2), zorder=5)
    if t < T_END:
        ax.scatter(*cur, s=16, color=INK, zorder=6)
    else:
        ax.scatter(*cur, marker="X", s=18, color="#000000", zorder=6)
    G.bare(ax, ZX, ZY)
    ax.set_title(f"Month {t}", fontsize=6.5, pad=1.5)
    if j == 0:
        ax.set_xlabel("Disease →", labelpad=1, fontsize=6)
        ax.set_ylabel("Host →", labelpad=0, fontsize=6)

# b
months = [0, 12, 24, 36, 48]
for j, t in enumerate(months):
    bx = axes_mm(fig, 30 + j * 22.6, H - 58, 21.6, 10)
    alive = int((t_os > t).sum())
    for i, (n, col) in enumerate(((alive, PAL["osimertinib"]), (len(ids) - alive, "#000000"))):
        if n:
            bx.scatter([i], [0], s=0.12 * n, color=col, alpha=0.8, lw=0)
            bx.text(i + 0.36, 0, f"{n:,}", fontsize=6, va="center")
    bx.set_xlim(-0.6, 1.9)
    bx.set_ylim(-1, 1)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["Alive", "Dead"], fontsize=6)
    bx.set_yticks([])
    bx.tick_params(length=0, pad=1)
    bx.spines["left"].set_visible(False)
    bx.set_title(f"Month {t}", fontsize=6.5, pad=1.5)
fig.text(28 / W, (H - 53) / H, f"{len(ids):,} generated\npatients", fontsize=6, ha="right", va="center")
km = axes_mm(fig, 148, H - 64, 30, 22)
grid = np.linspace(0, 48, 241)
km.step(grid, T.km_curve(t_os.to_numpy(float), e_os.to_numpy(int), np.ones(len(ids)), grid), where="post", color=INK, lw=1.0)
lo, hi = T.km_band(pd.DataFrame({"t_os": t_os.to_numpy(float), "e_os": e_os.to_numpy(int)}), grid)
km.fill_between(grid, lo, hi, step="post", color=INK, alpha=0.15, lw=0)
for t in months[1:]:
    km.plot([t, t], [0, 1.02], color="#9AA3AD", lw=0.4, ls=":", zorder=0)
km.set_xlim(0, 48)
km.set_ylim(0, 1.02)
km.set_xticks([0, 12, 24, 36, 48])
km.set_yticks([0, 0.5, 1])
km.set_yticklabels(["0", "50", "100"])
km.set_xlabel("Months from diagnosis", labelpad=1, fontsize=6)
km.set_ylabel("Alive (%)", labelpad=1, fontsize=6)

# c
fw, fh, top = 41.0, 22.5, H - 76
for r, (fam, key) in enumerate(FAMS):
    q = a[a.fam.eq(fam)]
    g_ids = q._gid.unique()
    tos = t_os.reindex(g_ids)
    col = FAMILY_COLOURS[key]
    p0 = positions(q, 0, pd.Index(g_ids))
    ref = G.peak(p0.to_numpy(float), XL, YL, 3.0)
    y = top - (r + 1) * fh - r * 2.0
    fig.text(10.5 / W, (y + fh / 2) / H, f"{'GI' if fam == 'Gastrointestinal' else fam}\n(n = {len(g_ids)})", fontsize=6,
             rotation=90, ha="right", va="center", multialignment="center")
    for j, t in enumerate(FR_C):
        ax = axes_mm(fig, 12 + j * (fw + 1.6), y, fw, fh)
        alive = tos.index[tos > t]
        pos = positions(q, t, alive)
        G.cloud(ax, pos.to_numpy(float), col, XL, YL, sigma=3.0, ref=ref, min_n=20)
        if 0 < len(pos) < 20:  # too few for a contour: the patients themselves
            ax.scatter(pos.x, pos.y, s=6, color=col, lw=0, zorder=4)
        if j:
            pr = positions(q, FR_C[j - 1], alive).reindex(pos.index).dropna()
            cur = pos.reindex(pr.index)
            ex = np.linspace(*XL, CELLS + 1)
            ey = np.linspace(*YL, CELLS + 1)
            ci = np.clip(np.digitize(pr.x, ex) - 1, 0, CELLS - 1)
            cj = np.clip(np.digitize(pr.y, ey) - 1, 0, CELLS - 1)
            for (u, v), idx in pd.Series(range(len(pr))).groupby([ci, cj]):
                if len(idx) < MIN_CELL:
                    continue
                s0 = pr.iloc[idx.to_numpy()].mean().to_numpy(float)
                s1 = cur.iloc[idx.to_numpy()].mean().to_numpy(float)
                if np.hypot(*(s1 - s0)) < 0.01:
                    ax.scatter(*s0, s=4, color=INK, zorder=5)
                    continue
                ax.annotate("", xy=s1, xytext=s0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6,
                                                                  shrinkA=0, shrinkB=0), zorder=5)
        G.bare(ax, XL, YL)
        dead = int((tos <= t).sum())
        ax.text(0.98, 0.03, f"{len(alive)} alive, {dead} dead", transform=ax.transAxes, fontsize=6, ha="right", va="bottom")
        if r == 0:
            ax.set_title(f"Month {t}", fontsize=6.5, pad=1.5)
        if r == len(FAMS) - 1 and j == 0:
            ax.set_xlabel("Disease →  (vertical: host impairment ↑)", labelpad=1, fontsize=6)

# keys and letters
yk = H - 3
items = [("o", STATE_COLOURS["response"], "Response"), ("o", STATE_COLOURS["progression"], "Progression"), ("X", "#000000", "Death")]
x0 = 12
for mk, c_, lab in items:
    fig.add_artist(Line2D([x0 / W], [yk / H], marker=mk, ms=3, color=c_, ls="", transform=fig.transFigure))
    fig.text((x0 + 1.6) / W, yk / H, lab, fontsize=6, va="center")
    x0 += 4 + 1.15 * len(lab)
fig.add_artist(Rectangle((x0 / W, (yk - 0.9) / H), 2.4 / W, 1.8 / H, transform=fig.transFigure, facecolor=STATE_COLOURS["unresolved"], edgecolor="none"))
fig.text((x0 + 3.2) / W, yk / H, "Not yet assessed", fontsize=6, va="center")
x0 += 24
fig.add_artist(Line2D([x0 / W, (x0 + 6) / W], [yk / H, yk / H], color=INK, lw=0.9, transform=fig.transFigure))
fig.add_artist(Line2D([(x0 + 6) / W], [yk / H], marker=">", ms=2.6, color=INK, ls="", transform=fig.transFigure))
fig.text((x0 + 7.5) / W, yk / H, "Move since the previous frame", fontsize=6, va="center")
for lab, yy in (("a", H - 1), ("b", H - 40), ("c", top + 9)):
    fig.text(1.5 / W, yy / H, lab, fontsize=8, fontweight="bold", va="top")

legend = ("a, one generated course; b, alive and dead patients of the atlas families and their Kaplan-Meier "
          "curve; c, the same families on the disease and host axes, with density contours and the mean move since the previous "
          "frame in each region with at least 8 patients alive at both times; with fewer than 20 patients alive, the patients are shown as points.")
save_panel(fig, "fig2_css_atlas", legend=legend,
           sources=["outputs/paper_state_map/state_projection.csv.gz", "runs/interventional_phase1_sonnet_full_v1c",
                    "atlas run folders (via fig1_data.load_tables)"],
           confirm=[],
           message="The CSS shows evolving populations: from one course to measured flows in four cancer families.", strict=False)
