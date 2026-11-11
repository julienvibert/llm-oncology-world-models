"""Panel ed04b_scenario_counts: counts of scenarios right, partly right, wrong, per model, for the two scenario groups.
Run: uv run python scripts/figures/panels/ed04b_scenario_counts.py"""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ed04 import MODEL_NAME, MODELS, SCORE, SCORE_TEXT, SENT  # noqa: E402
from common import panel, save_panel  # noqa: E402

groups = [("static", "Treatment or\nbiomarker", 9), ("dynamic", "History or\nhost", 3)]
W, H = 80, 62
fig, axes = panel(W, H, 1, 2, gridspec_kw={"width_ratios": [9, 3]})
fig.set_layout_engine("none")
x0s = [22, 65]; ws = [38, 12.7]  # noqa: E702
counts = {}
for ax, (g, lab, n), x0, w in zip(axes, groups, x0s, ws):
    sub = SENT[SENT["axis"].eq(g)]
    assert len(sub) == n
    for i, m in enumerate(MODELS):
        v = sub[f"{m}_primary_pct"].to_numpy(float)
        left = 0
        for score in (100, 50, 0):
            k = int((v == score).sum()); counts[(m, g, score)] = k  # noqa: E702
            if k:
                ax.barh(i, k, left=left, color=SCORE[score], edgecolor="white", lw=0.6, height=0.7)
                ax.text(left + k / 2, i, str(k), ha="center", va="center", fontsize=6, color=SCORE_TEXT[score])
            left += k
    ax.set_xlim(0, n); ax.set_ylim(3.5, -0.5)  # noqa: E702
    ax.set_xticks(range(0, n + 1, 3))
    ax.set_xlabel(f"{lab} (n = {n})")
    ax.set_yticks(range(4), [MODEL_NAME[m] for m in MODELS] if g == "static" else [])
    ax.tick_params(length=2)
    ax.set_position([x0 / W, 12 / H, w / W, 43 / H])
    if g != "static":
        ax.spines["left"].set_visible(True)
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(color=SCORE[100], label="all comparisons right"), Patch(color=SCORE[50], label="half right"),
                    Patch(color=SCORE[0], label="none right")], loc="upper center", ncol=3, frameon=False, fontsize=6,
           bbox_to_anchor=(0.5, 1.0), handlelength=1, columnspacing=1.2, handletextpad=0.3)
LEGEND = ("The scenario scores of panel a, counted. For each model, the number of scenarios in which all specified comparisons were ranked "
          "correctly (teal), half (gold) or none (red). Left, the nine treatment or biomarker scenarios; right, the three that also need "
          "previous treatment or the patient's physical condition. Bar length is the number of scenarios, so the left axis spans 9 and the right 3. "
          "Counts rather than means are shown because each scenario scores only 0%, 50% or 100%. No intervals are given. "
          "Model-implied benchmark; not an observed treatment effect.")
save_panel(fig, "ed04b_scenario_counts", legend=LEGEND,
           sources=["outputs/exploratory_sentinel_benchmark/sentinel_trust_map_table.csv"],
           confirm=["Counts of scenarios (scores 0/50/100)."],
           message="Every model gets most biomarker scenarios fully right; the three history or host scenarios are rarely fully right.")
