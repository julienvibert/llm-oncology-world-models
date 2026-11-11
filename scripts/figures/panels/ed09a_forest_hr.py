"""Panel ed09a_forest_hr: PRIME progression-free survival hazard ratios by subgroup (the forest view).

Source: outputs/exploratory_prime_corrections_2026_10_12/prime_forest_hr_itt935.csv (all 935
randomised patients, KRAS exon 2 = BMMTR1; written by scripts/explore/build_prime_itt_forest_local_km.py).
Run:  uv run python scripts/figures/panels/ed09a_forest_hr.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BLUE, CHEMO, INK, ROOT, fk, save_panel  # noqa: E402

SRC = "outputs/exploratory_prime_corrections_2026_10_12/prime_forest_hr_itt935.csv"
f = pd.read_csv(ROOT / SRC)
assert int(f.n.iloc[0]) == 935 and len(f) == 7
NAME = {k: k for k in f.label}
W, H = 180.0, 40.0
fig = fk.figure(width=W, height_mm=H)
ax = fig.add_axes([48 / W, 10 / H, 72 / W, 25 / H])
y = np.arange(len(f))[::-1]
col = [BLUE if h < 1 else CHEMO for h in f.hr]
ax.axvline(1, color=INK, lw=0.5)
for yi, r, c in zip(y, f.itertuples(), col):
    ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.2)
    ax.plot(r.hr, yi, "o", color=c, ms=3.8, mec="white", mew=0.3, zorder=3)
ax.set_xscale("log")
ax.set_xlim(0.5, 1.7)
ax.set_xticks([0.5, 0.75, 1, 1.5])
ax.set_xticklabels(["0.5", "0.75", "1", "1.5"])
ax.xaxis.set_minor_locator(__import__("matplotlib.ticker", fromlist=["NullLocator"]).NullLocator())
ax.set_yticks(y)
ax.set_yticklabels([NAME[r.label] for r in f.itertuples()])
ax.set_ylim(-0.6, len(f) - 0.4)
ax.set_xlabel("Hazard ratio for progression or death")
ax.text(0.97, 1.03, "← panitumumab + FOLFOX better", transform=ax.get_xaxis_transform(), fontsize=6, ha="right", va="bottom")
ax.text(1.03, 1.03, "FOLFOX alone better →", transform=ax.get_xaxis_transform(), fontsize=6, ha="left", va="bottom")
mn = lambda v: f"{v:.2f}".replace("-", "−")  # noqa: E731
fig.text(124 / W, 36.2 / H, "Hazard ratio (95% CI)", fontsize=6, fontweight="bold", va="bottom")
fig.text(151 / W, 36.2 / H, "Patients (events)", fontsize=6, fontweight="bold", va="bottom")
fig.text(172 / W, 36.2 / H, "P", fontsize=6, fontweight="bold", fontstyle="italic", va="bottom")
for yi, r in zip(y, f.itertuples()):
    py = (10 + 25 * (yi + 0.6) / (len(f) + 0.2)) / H
    fig.text(124 / W, py, f"{mn(r.hr)} ({mn(r.lo)} to {mn(r.hi)})", fontsize=6, va="center")
    fig.text(151 / W, py, f"{int(r.n)} ({int(r.events)})", fontsize=6, va="center")
    fig.text(172 / W, py, f"{r.p:.2f}", fontsize=6, va="center")
assert bool(((f.lo < 1) & (f.hi > 1)).all())  # every interval includes 1
nz = int(f.n_zero_day.iloc[0])
legend = ("Forest-plot view of the PRIME trial (NCT00364013; Project Data Sphere; randomised, first-line metastatic colorectal "
          "cancer; observed patients, not generated data). Each row is a separate Cox model of progression-free survival by "
          "central review (progression or death), panitumumab + FOLFOX versus FOLFOX alone, in all 935 randomised patients of "
          "the public dataset (intention to treat; 469 versus 466). Dots, hazard ratio; lines, 95% confidence interval; vertical "
          "line, no difference. Blue, point estimate below 1 (favours panitumumab + FOLFOX); purple, above 1 (favours FOLFOX "
          "alone). Patients, number randomised in the subgroup (events, progressions or deaths); P, Wald test of the arm "
          "coefficient. KRAS exon 2 (codons 12/13) status as recorded by the trial; 69 patients with a failed or missing test "
          "are in the all-patient row only. ECOG, Eastern Cooperative Oncology Group performance status (one patient with "
          f"missing status is in the all-patient row only). {nz} patients have a progression-free survival time of 0 days, all "
          "censored; they are kept at time 0, where they contribute no information to the estimate, so the hazard ratios equal "
          "those obtained without them. Every interval includes 1. A row gives one number for the whole subgroup; panels b and c "
          "ask the same question across the restricted PRIME state map.")
save_panel(fig, "ed09a_forest_hr", legend=legend,
           sources=[SRC, "outputs/exploratory_prime_corrections_2026_10_12/manifest.json"],
           confirm=["Denominators are all 935 randomised patients, including the 16 zero-day censored records (KRAS "
                    "wild-type 514, mutant 352).",
                    "KRAS exon 2 = BMMTR1 (label column BMMTNM1 'KRAS exon 2 (c12/13)'), failed/missing = unknown, excluded from "
                    "KRAS rows.",
                    "P values are the Cox Wald P of each separate model, computed by the producer; no interaction test."],
           message="Hazard-ratio forest on all 935 randomised patients: one number per subgroup; every interval includes 1.")
