"""Panel n8_prime_time_in_state: PRIME, time in each state over the first 24 months.

Message: over the first 24 months of PRIME, panitumumab added progression-free time in KRAS exon 2 wild-type
disease, and in mutant disease the time gained was time dead.
Data: outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv (aggregate; restricted means from
Kaplan-Meier curves, 95% bootstrap intervals over patients within arm and KRAS group, the 2,000 resamples shared with f6a_prime_rmst).
Exploratory reanalysis of the shared PRIME subset.
Run: uv run python scripts/figures/panels/n8_prime_time_in_state.py
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GREY, INK, ROOT, STATE_COLOURS, save_panel  # noqa: E402
from _f3util import axes_mm, minus, new_fig  # noqa: E402

SRC = ROOT / "outputs" / "exploratory_n8_prime_time_in_state" / "n8_time_in_state_v1.csv"
d = pd.read_csv(SRC)
TAU = float(d.tau_months.iloc[0])
PARTS = [("alive_progression_free", "Alive, progression-free", STATE_COLOURS["response"]),
         ("alive_after_progression", "Alive after progression", STATE_COLOURS["progression"]),
         ("dead", "Dead", STATE_COLOURS["death"])]
KRAS = [("Wild-type", "KRAS wild-type"), ("Mutant", "KRAS mutant")]
ARMS = [("Panitumumab + FOLFOX", "Panitumumab + FOLFOX"), ("FOLFOX alone", "FOLFOX alone")]
DIFF = "difference (panitumumab + FOLFOX minus FOLFOX)"


def val(kras: str, arm: str, part: str) -> pd.Series:
    return d[d.kras.eq(kras) & d.arm.eq(arm) & d.part.eq(part)].iloc[0]


for k, _ in KRAS:
    for a, _ in ARMS:
        assert abs(sum(val(k, a, p).months for p, *_ in PARTS) - TAU) < 1e-2, (k, a)

W, H = 180.0, 44.0
fig = new_fig(W, H)
# left: stacked time per arm
ax = axes_mm(fig, 35.0, 9.0, 72.0, 31.0)
rows = []
y = 0.0
for k, _ in KRAS:
    for a, alab in ARMS:
        rows.append((y, k, a, f"{alab} (n = {val(k, a, PARTS[0][0]).n})"))
        y -= 1.0
    y -= 0.55
for yy, k, a, _ in rows:
    left = 0.0
    for p, _, col in PARTS:
        m = val(k, a, p).months
        ax.barh(yy, m, left=left, height=0.72, color=col, edgecolor="white", lw=0.4)
        ax.text(left + m / 2, yy, f"{m:.1f}", fontsize=6, ha="center", va="center",
                color="white" if p != "alive_progression_free" else INK)
        left += m
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[3] for r in rows])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, TAU)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xlabel("Months of the first 24 after randomization", labelpad=1)
for k, klab in KRAS:
    ys = [r[0] for r in rows if r[1] == k]
    ax.text(-0.02, (max(ys) + 0.5), klab, transform=ax.get_yaxis_transform(), fontsize=6.5, ha="right",
            va="bottom", fontweight="bold")
ax.set_ylim(rows[-1][0] - 0.6, 0.95)
# key above the bars
for x0, (_, lab, col) in zip((35.0, 66.0, 99.0), PARTS):
    fig.add_artist(Rectangle((x0 / W, 41.2 / H), 2.4 / W, 2.0 / H, transform=fig.transFigure, facecolor=col,
                             edgecolor="none"))
    fig.text((x0 + 3.2) / W, 42.2 / H, lab, fontsize=6, va="center", ha="left")

# right: difference per part, panitumumab + FOLFOX minus FOLFOX
bx = axes_mm(fig, 138.0, 9.0, 40.0, 31.0)
ys, labs = [], []
yy = 0.0
for p, plab, col in PARTS:
    for k, klab in KRAS:
        r = val(k, DIFF, p)
        bx.plot([r.lo, r.hi], [yy, yy], color=INK, lw=0.8, solid_capstyle="butt", zorder=2)
        bx.plot(r.months, yy, "o" if k == "Wild-type" else "^", ms=4.2, mfc=col, mec=INK, mew=0.5, zorder=3)
        ys.append(yy)
        short = {"alive_progression_free": "Progression-free", "alive_after_progression": "After progression",
                 "dead": "Dead"}[p]
        labs.append(f"{short}, {'WT' if k == 'Wild-type' else 'mutant'}")
        yy -= 1.0
    yy -= 0.5
bx.set_yticks(ys)
bx.set_yticklabels(labs)
bx.tick_params(axis="y", length=0)
bx.spines["left"].set_visible(False)
bx.set_xlim(-3.4, 3.9)
bx.set_xticks([-2, 0, 2])
bx.set_xticklabels([minus(-2, "+.0f"), "0", minus(2, "+.0f")])
bx.set_xlabel("Panitumumab + FOLFOX\nminus FOLFOX (months)", labelpad=1)
bx.set_ylim(ys[-1] - 0.6, 0.6)
bx.plot([0, 0], [ys[-1] - 0.6, 0.6], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)


def txt(k: str, p: str) -> str:
    r = val(k, DIFF, p)
    return f"{minus(r.months, '+.2f')} ({minus(r.lo, '+.2f')} to {minus(r.hi, '+.2f')})"


legend = (
    "How each randomized arm of PRIME spent the first 24 months, by KRAS exon 2 status (patients of the shared "
    "Project Data Sphere subset with a known KRAS result, analysed as randomized). Left, restricted mean months alive "
    "without progression, alive after progression and dead, which add up to 24; from Kaplan-Meier curves of "
    "progression-free survival (central review; progression or death) and overall survival. Overall survival follows "
    "the protocol definition (time from randomization to death, censored at the last contact date; protocol "
    "Amendment 2, section 10.2.1.2); the dataset field used for it (DTHDY) is labelled only 'Death Day' in the data "
    "documentation and is interpreted as that endpoint. Right, panitumumab + FOLFOX minus FOLFOX alone for each part "
    "(circles, wild-type, WT; triangles, mutant); lines, 95% bootstrap intervals (patients resampled within arm and KRAS "
    "group; 2,000 resamples, seed 20261028, shared with Fig. 6a, so the progression-free difference and its interval "
    "equal Fig. 6a's KRAS rows; months = days / 30.4375). Wild-type: alive progression-free " + txt("Wild-type", "alive_progression_free")
    + ", after progression " + txt("Wild-type", "alive_after_progression") + ", dead " + txt("Wild-type", "dead")
    + " months. Mutant: alive progression-free " + txt("Mutant", "alive_progression_free") + ", after progression "
    + txt("Mutant", "alive_after_progression") + ", dead " + txt("Mutant", "dead") + " months. Intervals that include "
    "zero show uncertainty, not equivalence. The parts are clinical states, not regions of the state map. Exploratory "
    "reanalysis of the shared trial data, not a confirmatory test. Observed trial data, no language model."
)
save_panel(fig, "n8_prime_time_in_state", legend=legend,
           sources=["outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv",
                    "outputs/exploratory_n8_prime_time_in_state/manifest.json"],
           confirm=["Progression-free part drawn in the paper's 'response' colour for readability; it covers response "
                    "and stable disease (alive without progression), as the key says."],
           message="Panitumumab added progression-free time in KRAS wild-type disease; in mutant disease the "
                   "difference was time dead (exploratory).")
