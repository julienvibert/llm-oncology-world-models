"""Panel fig2_families_km: four cancer families, classical survival beside the Clinical State Space.

Message: the Kaplan-Meier curve gives one number per month (alive or dead); the CSS frames of the same patients show where the
living patients are in disease burden and host condition, and how they move.

Left, Kaplan-Meier survival of each family (six models pooled) with dotted lines at the frame times. Right, the same patients on
the disease and host axes at 0, 12, 24 and 36 months: density contours of living patients and, in each region of a 6 x 6 grid
with at least 8 patients alive 12 months later, the mean move over the next 12 months, drawn from the frame shown (dots, moves too
short to draw).
Same data and rules as fig2_css_atlas.py part c. Model-generated data.
Run: uv run python scripts/figures/panels/fig2_families_km.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _css_axes as A  # noqa: E402
import _css_grammar as G  # noqa: E402
import _css_template as T  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from common import FAMILY_COLOURS, INK, ROOT, save_panel  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "scripts" / "explore" / "figpanels"))
from build_field_trellis import CODE2FAM  # noqa: E402
import fig1_data as F1  # noqa: E402

M = 30.4375
FAMS = [("Lung", "lung"), ("Gastrointestinal", "gi"), ("Sarcoma", "sarcoma"), ("Breast", "breast")]
FR = [0, 12, 24, 36]
CELLS, MIN_CELL = 6, 8

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


W, H = 180.0, 96.0
fig = new_fig(W, H)
kw, fw, fh, gap = 26.0, 31.6, 19.0, 1.4
x_frames = 16 + kw + 5
top = H - 6
grid = np.linspace(0, 48, 241)
aggs = []
for r, (fam, key) in enumerate(FAMS):
    q = a[a.fam.eq(fam)]
    g_ids = q._gid.unique()
    tos = t_os.reindex(g_ids)
    eos = e_os.reindex(g_ids)
    col = FAMILY_COLOURS[key]
    y = top - (r + 1) * fh - r * 2.2
    fig.text(3 / W, (y + fh / 2) / H, f"{'GI' if fam == 'Gastrointestinal' else fam}\n(n = {len(g_ids)})", fontsize=6,
             rotation=90, ha="center", va="center", multialignment="center")
    # classical view
    km = axes_mm(fig, 16, y, kw, fh)
    s = T.km_curve(tos.to_numpy(float), eos.to_numpy(int), np.ones(len(g_ids)), grid)
    lo, hi = T.km_band(pd.DataFrame({"t_os": tos.to_numpy(float), "e_os": eos.to_numpy(int)}), grid)
    km.fill_between(grid, lo, hi, step="post", color=col, alpha=0.2, lw=0)
    km.step(grid, s, where="post", color=col, lw=1.0)
    for t in FR[1:]:
        km.plot([t, t], [0, 1.02], color="#9AA3AD", lw=0.4, ls=":", zorder=0)
    km.set_xlim(0, 48)
    km.set_ylim(0, 1.02)
    km.set_xticks([0, 12, 24, 36, 48])
    km.set_yticks([0, 0.5, 1])
    km.set_yticklabels(["0", "50", "100"])
    km.tick_params(labelsize=6, length=2, pad=1)
    if r < len(FAMS) - 1:
        km.set_xticklabels([])
    else:
        km.set_xlabel("Months from diagnosis", labelpad=1, fontsize=6)
    km.set_ylabel("Alive (%)", fontsize=6, labelpad=1)
    # CSS view
    p0 = positions(q, 0, pd.Index(g_ids))
    ref = G.peak(p0.to_numpy(float), XL, YL, 3.0)
    for j, t in enumerate(FR):
        ax = axes_mm(fig, x_frames + j * (fw + gap), y, fw, fh)
        alive = tos.index[tos > t]
        pos = positions(q, t, alive)
        G.cloud(ax, pos.to_numpy(float), col, XL, YL, sigma=3.0, ref=ref, min_n=20)
        if 0 < len(pos) < 20:
            ax.scatter(pos.x, pos.y, s=6, color=col, lw=0, zorder=4)
        n_arrows = 0
        if True:  # mean move over the next 12 months, from the positions of this frame
            alive_next = tos.index[tos > t + 12]
            pr = pos.reindex(positions(q, t, alive_next).index).dropna()
            cur = positions(q, t + 12, alive_next).reindex(pr.index)
            ex = np.linspace(*XL, CELLS + 1)
            ey = np.linspace(*YL, CELLS + 1)
            ci = np.clip(np.digitize(pr.x, ex) - 1, 0, CELLS - 1)
            cj = np.clip(np.digitize(pr.y, ey) - 1, 0, CELLS - 1)
            for (u, v), idx in pd.Series(range(len(pr))).groupby([ci, cj]):
                if len(idx) < MIN_CELL:
                    continue
                s0 = pr.iloc[idx.to_numpy()].mean().to_numpy(float)
                s1 = cur.iloc[idx.to_numpy()].mean().to_numpy(float)
                n_arrows += 1
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
        if r == len(FAMS) - 1:
            ax.set_xlabel("Disease burden →", labelpad=1, fontsize=6)
        if j == 0:
            fig.text((x_frames - 1.6) / W, (y + fh / 2) / H, "Host impairment →", fontsize=6, rotation=90, ha="center",
                     va="center")
        aggs.append({"family": fam, "month": t, "alive": len(alive), "dead": dead, "regions_with_move": n_arrows})

yk = H - 2.2
fig.add_artist(Line2D([x_frames / W, (x_frames + 6) / W], [yk / H, yk / H], color=INK, lw=0.9, transform=fig.transFigure))
fig.add_artist(Line2D([(x_frames + 6) / W], [yk / H], marker=">", ms=2.6, color=INK, ls="", transform=fig.transFigure))
fig.text((x_frames + 7.5) / W, yk / H, "Mean move over the next 12 months", fontsize=6, va="center")

out = ROOT / "outputs/figures/panels/fig2_families_km"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(aggs).to_csv(out / "fig2_families_km_aggregates.csv", index=False)
legend = ("Four cancer families generated by six models: Kaplan-Meier survival (band, 95% interval; dotted lines, frame times) beside "
          "the same patients on the disease and host axes at 0, 12, 24 and 36 months, with density contours of living patients and the "
          "mean move over the next 12 months in regions with at least 8 patients alive 12 months later.")
save_panel(fig, "fig2_families_km", legend=legend,
           sources=["outputs/paper_state_map/state_projection.csv.gz", "atlas run folders (via fig1_data.load_tables)"],
           confirm=[],
           message="Survival curves give alive or dead; the CSS shows where the living patients are and how they move.", strict=False)
