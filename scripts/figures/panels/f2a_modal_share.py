"""Panel f2a_modal_share.

Message: within each cancer type, most generated patients share the same baseline values
(81.0% to 93.2% across the six models).
Data: outputs/exploratory_figure_panels/fig1/data/modal_share.csv (from
scripts/explore/figpanels/fig1_data.py; mean modal share of five baseline features over
148 cancer codes, 95% bootstrap interval over codes).
Run: uv run python scripts/figures/panels/f2a_modal_share.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f2 import MODEL_COLOURS, MODEL_ORDER, axes_mm, canvas, table  # noqa: E402
from common import INK, save_panel  # noqa: E402

t = table("modal_share.csv").set_index("model").loc[MODEL_ORDER]
fig = canvas(80, 48)
ax = axes_mm(fig, 16.5, 7, 61, 40)
n = len(t)
for i, (m, r) in enumerate(t.iterrows()):
    y = n - 1 - i
    est, lo, hi = 100 * r.archetype_share, 100 * r.ci_low, 100 * r.ci_high
    assert lo <= est <= hi
    c = MODEL_COLOURS[m]
    ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="butt")
    ax.plot(est, y, "o", color=c, ms=4.5, mec="none")
    ax.text(hi + 0.3, y, f"{est:.1f}", va="center", ha="left", fontsize=6, color=INK)
ax.set_yticks(range(n))
ax.set_yticklabels(MODEL_ORDER[::-1])
ax.set_ylim(-0.6, n - 0.2)
ax.set_xlim(78, 96)
ax.set_xticks([80, 85, 90, 95])
ax.set_xlabel("Patients sharing the most frequent value (%)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
n_traj = int(t.n_trajectories.sum())
save_panel(
    fig, "f2a_modal_share",
    legend=("Concentration of the generated baseline within each cancer type. For each baseline feature (primary "
            "driver, sex, stage, smoking status, frailty) we took the share of generated patients with the most "
            "frequent value inside one cancer code, then averaged the five features and the 148 cancer codes. "
            "Dots, mean per model (value above the dot); lines, 95% bootstrap interval obtained by resampling the "
            "148 cancer codes. The intervals are of similar, small width because each estimate averages the same "
            f"148 codes and five features. Each code has about five generated patients per model ({n_traj:,} in total), "
            "so the share cannot fall below about 20% by construction. Generated data."),
    sources=["outputs/exploratory_figure_panels/fig1/data/modal_share.csv"],
    confirm=["Fixed model order (same as panel d), not sorted by value.",
             "Modal share is computed from about five patients per code, so it cannot fall below 20% by construction.",
             "These univariate shares do not prove joint-archetype or whole-trajectory mode collapse.",
             "The mean is written once above each dot."],
    message="Within each cancer type, most generated patients share the same baseline values.")
