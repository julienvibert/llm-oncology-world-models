"""Panel f3b_egfr_fans: osimertinib versus chemotherapy as survival-style curves with numbers at risk.

Message: from the same starting records, osimertinib futures stay in first-line control longer than
chemotherapy futures in EGFR-mutant disease.
Data: outputs/exploratory_interventional_phase1/phase1_endpoint_long.csv (per-future first-line
progression time in days, endpoint_days, converted at 30.4375 days per month), EGFR-mutant context, 45 futures per arm (15 starting state-history
combinations x 3 repeats). Medians are checked against the saved day-level medians
(phase1_context_arm_medians.csv); 12-month fractions against outputs/paper_egfr_dynamics_summary/readout.json. Numbers at risk are counted from the same table.
Run: uv run python scripts/figures/panels/f3b_egfr_fans.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, GREY, INK, CHEMO, PAL, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402
from _egfr_days import MONTH_DAYS, context_medians, value  # noqa: E402

OSI = PAL["osimertinib"]
SRC = "outputs/exploratory_interventional_phase1/phase1_endpoint_long.csv"
e = pd.read_csv(ROOT / SRC)
e = e[e.context.eq("EGFR_1L") & e.arm_key.isin(["chemo", "osimertinib"])]
S = json.loads((ROOT / "outputs/paper_egfr_dynamics_summary/readout.json").read_text())["egfr_mutant"]
H = 24.0
TICKS = [0, 6, 12, 18, 24]
W, HT = 66.0, 49.0
fig = new_fig(W, HT)
ax = axes_mm(fig, 19, 21, 43, 26)
t = np.linspace(0, H, 4000)
arms = (("osimertinib", "Osimertinib", OSI), ("chemo", "Chemotherapy", CHEMO))
info = {}
for key, name, col in arms:
    v = np.sort(e[e.arm_key.eq(key)].endpoint_days.to_numpy(float)) / MONTH_DAYS
    assert len(v) == 45
    ax.step(t, [(v > x).mean() * 100 for x in t], where="post", color=col, lw=1.1,
            label=f"{name}, median {np.median(v):.1f} months")
    info[key] = dict(v=v, med=float(np.median(v)), f12=float((v > 12).mean()),
                     atrisk=[int((v >= x).sum()) for x in TICKS], events=int((v <= H).sum()))
MED = context_medians()
for key in ("chemo", "osimertinib"):
    assert abs(info[key]["med"] - value(MED, "claude_sonnet5", "mut", key)) < 1e-9
assert abs(info["chemo"]["f12"] - S["chemo"]["remain_at_12_months"]) < 1e-6
ax.axhline(50, color=GREY, lw=0.5, ls=":")
ax.set_xlim(0, H)
ax.set_ylim(0, 103)
ax.set_xticks(TICKS)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("Months after treatment decision")
ax.set_ylabel("Futures free of first-line\nprogression (%)")
ax.legend(loc="lower left", fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0.2)
# numbers at risk
AXW = 43.0
ax.set_position([19 / W, 21 / HT, AXW / W, 26 / HT])
tab = axes_mm(fig, 19, 1.5, AXW, 9)
tab.set_xlim(0, H)
tab.set_ylim(-0.5, 1.5)
tr = tab.get_yaxis_transform()
mm_ax = lambda mm: mm / AXW * H  # noqa: E731  (mm from the axes edge in data units)
for i, (key, name, col) in enumerate(arms):
    y = 1 - i
    for x, n in zip(TICKS, info[key]["atrisk"]):
        tab.text(x, y, str(n), ha="center", va="center", fontsize=6, color=INK)
    tab.plot([-mm_ax(18.8), -mm_ax(17.3)], [y, y], color=col, lw=1.4, clip_on=False)
    tab.text(-mm_ax(16.6), y, name, ha="left", va="center", fontsize=6, color=INK, clip_on=False)
tab.text(-mm_ax(18.5), 1.5, "Number at risk (futures)", ha="left", va="bottom", fontsize=6, color=INK, clip_on=False)
tab.axis("off")
save_panel(
    fig, "f3b_egfr_fans",
    legend=("Share of generated futures still free of first-line progression over 24 months in EGFR-mutant lung cancer "
            "after assignment of osimertinib (blue) or chemotherapy (purple), starting from the same records. Step curves "
            "show all 45 futures per arm (15 combinations of clinical state and pretreatment history, three repeats each; "
            "these are not 45 independent patients); medians are in the key. The table gives the number of futures still at "
            "risk (not yet progressed) at 0, 6, 12, 18 and 24 months; every future has an observed endpoint, so there is no "
            "censoring. The dotted line marks 50%. "
            "Fractions describe the generated set and are not clinical probabilities. Claude Sonnet."),
    sources=[SRC, "outputs/exploratory_interventional_phase1/phase1_context_arm_medians.csv",
             "outputs/paper_egfr_dynamics_summary/readout.json"],
    confirm=["Numbers at risk are counted from the per-future table (endpoint at or after the time); no test is shown because "
             "the 45 futures per arm are repeats from 15 starts, not independent patients; the paired difference with interval "
             "is in the next panel.",
             "Time axis starts at the treatment decision; months are endpoint_days / 30.4375 (manuscript convention); "
             "medians 23.43 (osimertinib) and 16.10 (chemotherapy) months, as in outputs/paper_survival_in_state_space. "
             "The scorer's endpoint_mo (30.4 days) is not used. Numbers at risk are identical under both conventions.",
             "'No censoring' rests on every future having an endpoint_mo (forced-line progression time, with a death fallback flag)."],
    message="Osimertinib futures stay in first-line control longer than chemotherapy futures in EGFR-mutant disease.")
