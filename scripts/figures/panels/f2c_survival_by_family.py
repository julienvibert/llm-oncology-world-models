"""Panel f2c_survival_by_family: eight family facets, every model named, model colours.

Message: median generated survival differs by model; Kimi K3 is longest in every cancer family.
Data: outputs/exploratory_figure_panels/fig1/data/survival_by_family.csv (median months from
diagnosis to death, 95% bootstrap interval over generated patients). The producer
(scripts/explore/figpanels/fig1_data.py) writes months at 30.4 days; the manuscript uses 30.4375, so
every value is rescaled by 30.4 / 30.4375 here. This is exact: medians, maxima and percentile-bootstrap
quantiles are linear in a positive scale factor and the bootstrap indices do not depend on the values
(checked against a recomputation from days_from_diagnosis / 30.4375: max difference 1e-14).
Run: uv run python scripts/figures/panels/f2c_survival_by_family.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f2 import MODEL_COLOURS, MODEL_ORDER, axes_mm, canvas, table  # noqa: E402
import pandas as pd  # noqa: E402
from common import INK, ROOT, save_panel  # noqa: E402

t = table("survival_by_family.csv")
kw = pd.read_csv(ROOT / "outputs/exploratory_figure_panels/fig1/data/survival_by_family_test.csv").set_index("family")  # Kruskal-Wallis, Holm (fig1d_survival_test.py)
MONTH_FIX = 30.4 / 30.4375  # producer months (30.4 d) -> manuscript months (30.4375 d)
for c in ("median", "ci_low", "ci_high"):
    t[c] = t[c] * MONTH_FIX
assert len(t) == 48 and t.family.nunique() == 8
kimi = t[t.model == "Kimi K3"].set_index("family")
fams = list(kimi.sort_values("median", ascending=False).index)
assert all(kimi.loc[f, "median"] > t[(t.family == f) & (t.model != "Kimi K3")]["median"].max() for f in fams)
W, H = 180, 52
fig = canvas(W, H)
first = None
for i, f in enumerate(fams):
    col, row = i % 4, i // 4
    left, bottom = 17 + 41 * col, 29.5 if row == 0 else 7.5
    ax = axes_mm(fig, left, bottom, 38, 14)
    if first is None:
        first = ax
    else:
        ax.sharex(first)
    cell = t[t.family == f].set_index("model").loc[MODEL_ORDER]
    for j, (m, v) in enumerate(cell.iterrows()):
        y = 5 - j
        c = MODEL_COLOURS[m]
        assert v.ci_low <= v["median"] <= v.ci_high
        ax.plot([v.ci_low, v.ci_high], [y, y], color=c, lw=0.9, solid_capstyle="butt")
        ax.plot(v["median"], y, "o", color=c, ms=3.8, mec="none")
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.6, 5.6)
    ax.set_xticks([0, 30, 60])
    if row == 0:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Median survival (months)")
    ax.set_yticks(range(6))
    ax.set_yticklabels(MODEL_ORDER[::-1] if col == 0 else [""] * 6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    lo, hi = int(cell.n.min()), int(cell.n.max())
    cnt = str(lo) if lo == hi else f"{lo}-{hi}"
    name = "Head and neck" if f == "Head & neck" else f
    pa = kw.loc[f, "p_holm"]
    ptxt = "P < 0.001" if pa < 0.001 else f"P = {pa:.3f}"
    fig.text((left + 19) / W, (bottom + 14.6) / H, f"{name} (n = {cnt} per model)\n{ptxt}", ha="center",
             va="bottom", fontsize=6.5, color=INK, linespacing=1.05)
save_panel(
    fig, "f2c_survival_by_family",
    legend=("Median generated survival from diagnosis to death for each of the six models (rows, colour = model) in "
            "eight cancer families (panels, ordered by the Kimi K3 median, longest first). Dots, median; lines, 95% "
            "bootstrap interval (2,000 resamples of the generated patients in that model and family). n per model "
            "and family is given above each panel: 20 to 60 generated patients; Kimi K3 has 42 head-and-neck patients "
            "against 45 for the other models, and GLM-5.2 has 52 gastrointestinal patients against 55. All panels share "
            "the 0 to 70 month scale. Families group the 148 cancer codes. P, Kruskal-Wallis test of generated survival "
            "across the six models within each family (every course ends in death, so there is no censoring), "
            "Holm-adjusted across the eight families. Generated data."),
    sources=["outputs/exploratory_figure_panels/fig1/data/survival_by_family.csv",
             "outputs/exploratory_figure_panels/fig1/data/survival_by_family_test.csv"],
    confirm=["Kruskal-Wallis across models per family, Holm over 8 families: all P < 0.001 "
             "(largest, lung, 0.0002); computed by scripts/explore/figpanels/fig1d_survival_test.py.",
             "Kimi K3 has the longest median in all eight families (asserted in the script); the intervals do not all exclude the next model.",
             "Kimi K3 has 42 head-and-neck patients vs 45; GLM-5.2 has 52 gastrointestinal vs 55 (header gives the range).",
             "Months at 30.4375 days (producer table at 30.4, rescaled exactly)."],
    message="Median generated survival differs by model; Kimi K3 is longest in every cancer family.")
