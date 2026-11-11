"""Panel ed03c_prior_progression: death within 12 months by earlier progression, within each model, at similar state and time.
Run: uv run python scripts/figures/panels/ed03c_prior_progression.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed03 import NAMES, same_state  # noqa: E402
from _f4 import fix_axes  # noqa: E402
from common import MODEL_COLOURS, panel, save_panel  # noqa: E402

_, _, bm = same_state()
bm = bm.reset_index(drop=True)
fig, ax = panel(82, 62)
import numpy as np  # noqa: E402


def wilson(pc: float, n: int) -> tuple[float, float]:
    p_, z = pc / 100, 1.96
    c = (p_ + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def diff_ci(r) -> tuple[float, float]:
    a, b = r.prior / 100, r.no_prior / 100
    se = np.sqrt(a * (1 - a) / r.n_prior + b * (1 - b) / r.n_no)
    return 100 * (a - b - 1.96 * se), 100 * (a - b + 1.96 * se)

# label positions pushed apart
ys = sorted(range(len(bm)), key=lambda i: bm.prior[i])
pos = {i: bm.prior[i] for i in ys}
for _ in range(200):
    for a, b in zip(ys, ys[1:]):
        if pos[b] - pos[a] < 5.2:
            pos[a] -= 0.2; pos[b] += 0.2  # noqa: E702
for i, r in bm.iterrows():
    col = MODEL_COLOURS[NAMES[r.model]]
    jx = (i - (len(bm) - 1) / 2) * 0.035
    ax.plot([0 + jx, 1 + jx], [r.no_prior, r.prior], color=col, lw=0.9, zorder=2)
    for xx, pc, n in ((0 + jx, r.no_prior, r.n_no), (1 + jx, r.prior, r.n_prior)):
        ax.plot([xx, xx], wilson(pc, n), color=col, lw=0.6, zorder=2)
    ax.scatter([0 + jx], [r.no_prior], s=16, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
    ax.scatter([1 + jx], [r.prior], s=16, color=col, zorder=3, lw=0)
    lo_, hi_ = diff_ci(r)
    ax.text(1.17, pos[i], f"{NAMES[r.model]}  +{r.delta:.1f} ({lo_:+.1f} to {hi_:+.1f})", color="black", va="center", fontsize=6)
    ax.plot([1.12, 1.16], [r.prior, pos[i]], color=col, lw=0.5)
ax.set_xlim(-0.2, 2.85)
ax.set_xticks([0, 1], ["none", "earlier\nprogression"])
ax.set_xlabel("History before the visit")
ax.set_ylabel("Died within 12 months (%)")
fix_axes(fig, ax, left=8, right=1.5, bottom=9, top=1.5)
nn = f"{bm.n_no.min()}-{bm.n_no.max()}"; npr = f"{bm.n_prior.min()}-{bm.n_prior.max()}"  # noqa: E702
LEGEND = ("Death within 12 months among generated patients at similar visits (central bands of the two coordinates of the older combined tumour-host axis, unchanged from the original analysis and not the frozen state map, and a common "
          "time since diagnosis; one visit per generated patient), without (open circle) and with (filled circle) progression before the "
          "visit, computed within each of six models. Labels give the difference in percentage points. "
          f"Visits per model and group: {nn} without and {npr} with earlier progression. Bars, Wilson 95% intervals; in brackets, 95% interval of the difference. Within every model "
          "death is more frequent after earlier progression. Pooling the models without stratifying attenuates this difference "
          "(73.4% versus 75.6%) because the models differ in baseline mortality; it is not reversed. Data are model-generated, not real patients.")
save_panel(fig, "ed03c_prior_progression", legend=LEGEND,
           sources=["outputs/exploratory_fig1_same_dot/readout_v2.json", "scripts/explore/build_fig1_same_dot_v2.py (per_model_contrast)"],
           confirm=[
                    "Similar-state bands use the older combined axis (analysis not redefined; display-only reprojection applies to panels a, b and d).",
                    "Pooled 73.4% vs 75.6% quoted in the legend from readout_v2.json (n = 1034 and 1995).",
                    "No P value: none saved."],
           message="Within each of six models, earlier progression raises the 12-month death rate at a similar state.")
