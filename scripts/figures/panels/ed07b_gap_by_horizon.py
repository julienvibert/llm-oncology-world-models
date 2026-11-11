"""Panel ed07b_gap_by_horizon: predicted minus observed survival at 3, 6, 12 and 24 months, for each forecaster.

Reads the final scored table outputs/four_arm_preprint_20260928/survival_scores_20261012.csv (rows forecaster:<f>,
cancer "all"): mean_predicted_alive minus observed_alive_ipcw (censoring-weighted). No interval is saved for these
two means, so the gaps are drawn without one.
Run: uv run python scripts/figures/panels/ed07b_gap_by_horizon.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import F_REL, INK, ROWS, final_table, write_aggregates  # noqa: E402
from _f3util import axes_mm, minus, new_fig  # noqa: E402
from common import save_panel  # noqa: E402

TABLE = "survival_scores_20261012.csv"
HOR = [3, 6, 12, 24]
d = final_table(TABLE)
d = d[(d["cancer"].astype(str) == "all") & d["row"].astype(str).str.startswith("forecaster:")].copy()
d["key"] = d["row"].str.removeprefix("forecaster:")
missing = {k for k, *_ in ROWS} - set(d["key"])
if missing:
    raise SystemExit(f"forecasters missing from {TABLE}: {sorted(missing)}")


def cell(h: int, k: str):
    r = d[(d["horizon_months"] == h) & (d["key"] == k)]
    if len(r) != 1:
        raise SystemExit(f"expected one row for {k} at {h} months, found {len(r)}")
    return r.iloc[0]


W, H = 180.0, 50.0
LEFT, RIGHT, GAP, BOTTOM, TOP = 23.5, 1.5, 6.0, 12.0, 5.0
aw = (W - LEFT - RIGHT - GAP * (len(HOR) - 1)) / len(HOR)
fig = new_fig(W, H)
agg, ns = [], {}
for i, h in enumerate(HOR):
    ax = axes_mm(fig, LEFT + i * (aw + GAP), BOTTOM, aw, H - BOTTOM - TOP)
    obs = {cell(h, k)["observed_alive_ipcw"] for k, *_ in ROWS}
    if len(obs) != 1:
        raise SystemExit(f"observed proportion differs between forecasters at {h} months")
    obs = obs.pop()
    ns[h] = sorted({int(cell(h, k)["n"]) for k, *_ in ROWS})
    for j, (k, lab, c, mk) in enumerate(ROWS):
        r = cell(h, k)
        gap = 100 * (r["mean_predicted_alive"] - r["observed_alive_ipcw"])
        agg.append([h, k, int(r["n"]), r["mean_predicted_alive"], r["observed_alive_ipcw"], gap])
        ax.plot(gap, j, mk, ms=5 if mk == "o" else 4.5, mfc=c, mec=INK, mew=0.6, zorder=3)
        ax.annotate(minus(gap), (gap, j), xytext=(5 if gap >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if gap >= 0 else "right", va="center", fontsize=6)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    ax.set_xlim(-30, 30)
    ax.set_xticks([-20, 0, 20])
    ax.set_xticklabels(["−20", "0", "20"])
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS] if i == 0 else [])
    if i:
        ax.tick_params(axis="y", left=False)
    ax.set_title(f"{h} months: {100 * obs:.1f}% alive", fontsize=7)
mid = (LEFT + (W - RIGHT)) / 2 / W
fig.supxlabel("Predicted minus observed alive (percentage points)", x=mid, y=4.3 / H, ha="center", va="bottom",
              fontsize=7)
# two separate strings: runs of spaces collapse when the panel is placed as SVG in the assembled figure
fig.text(mid - 3 / W, 1.0 / H, "\u2190 survival underestimated", ha="right", va="bottom", fontsize=7)
fig.text(mid + 3 / W, 1.0 / H, "survival overestimated \u2192", ha="left", va="bottom", fontsize=7)
write_aggregates("ed07b_gap_by_horizon",
                 ["horizon_months", "forecaster", "n", "mean_predicted_alive", "observed_alive_ipcw", "gap_points"], agg)
legend = (
    "Whether forecast survival sits above or below what was observed, as follow-up lengthens. Each symbol is one forecaster; "
    "the value is its mean predicted probability of being alive at 3, 6, 12 or 24 months minus the observed proportion "
    "alive (censoring-weighted, shown in each heading), in percentage points, over the same GENIE-BPC patients (breast, "
    f"colorectal, non-small cell lung; n = {ns[3][0]} at 3, 6 and 12 months and {ns[24][0]} at 24 months). Right of the black "
    "line, survival is overestimated; left, underestimated. Coloured dots, the four evidence conditions of the generated "
    "forecasts (amber to wine as in the main figures); open symbols, simple comparators as in Fig. 4b: the generic and the "
    "matched donors used directly, and a cross-validated Cox model. No interval is saved for these means, so none is shown."
)
save_panel(
    fig, "ed07b_gap_by_horizon", legend=legend,
    sources=[f"{F_REL}/{TABLE}"],
    confirm=[
        "Gap = mean_predicted_alive minus observed_alive_ipcw (new arithmetic on two saved numbers); no interval is saved for it.",
        "The comparator km_cancer_cv is not in the final table and is not drawn.",
        "At 24 months the observed proportion is censoring-weighted over the evaluable patients (n_evaluable 404 of 447).",
    ],
    message="Generated forecasts overestimate survival up to 12 months; with examples they underestimate it at 24 months.",
)
