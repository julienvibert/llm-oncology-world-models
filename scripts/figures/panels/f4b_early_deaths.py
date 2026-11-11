"""Figure 4c (file f4b_early_deaths), early deaths: far fewer deaths within six months in generated futures.

Message: 16.2% of the 450 patients died within 6 months (73/450); only 1.2-3.6% of generated futures did.
Data: real_side.csv (died, real_os), generated_side.csv (arm, gen_os, query_id). Wilson interval for the
observed patients; bootstrap over patients (seed 0, 2,000 resamples) for generated futures; saved to
f4b_early_deaths_aggregates.csv.
Run: uv run python scripts/figures/panels/f4b_early_deaths.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, S, LIGHT, dot, fix_axes, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

real = pd.read_csv(S / "real_side.csv")
gen = pd.read_csv(S / "generated_side.csv")
H = 6.0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


k_obs = int((real.died & (real.real_os <= H)).sum())
n_obs = len(real)
rows = [("Observed", "Observed patients", *wilson(k_obs, n_obs), k_obs, n_obs, "black", f"Observed ({k_obs}/{n_obs})")]
rng = np.random.default_rng(0)
ids = real.query_id.unique()
res = {}
for arm, lab, col, _ in ARMS:
    s = gen[gen.arm == arm]
    per = pd.DataFrame({"q": s.query_id, "e": (s.gen_os <= H).astype(int)}).groupby("q").agg(
        k=("e", "sum"), n=("e", "size")).reindex(ids).fillna(0)
    k, n = per.k.values, per.n.values
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(ids), len(ids))
        boots.append(k[i].sum() / n[i].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res[arm] = (k.sum() / n.sum(), lo, hi, int(k.sum()), int(n.sum()))
    rows.append((arm, lab, *res[arm], col, lab))

write_aggregates("f4b_early_deaths",
                 ["group", "unit", "n_patients", "deaths_within_6m_pct", "lo_pct", "hi_pct", "events", "denominator"],
                 [[r[0], "patients" if r[0] == "Observed" else "generated futures", n_obs,
                   round(100 * r[2], 2), round(100 * r[3], 2), round(100 * r[4], 2), r[5], r[6]] for r in rows])

fig, ax = panel(56, 46)
y = np.arange(len(rows))[::-1]
for yi, (g, lab, p, lo, hi, k, n, col, ylab) in zip(y, rows):
    dot(ax, yi, 100 * p, 100 * lo, 100 * hi, col)
    ax.text(100 * hi + 0.7, yi, f"{100*p:.1f}%", va="center", ha="left", fontsize=6)
ax.set_yticks(y)
ax.set_yticklabels([r[8] for r in rows])
ax.set_xlim(0, 24)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("Patients dying within\n6 months (%)")
fix_axes(fig, ax, left=21.5, right=2, bottom=10.5, top=1.0)
ax.plot([0, 24], [y[0] - 0.5] * 2, color=LIGHT, lw=0.5)
# the observed row shows its count; generated rows are futures, counts in the legend

legend = (
    "Deaths within 6 months of the start of the recorded treatment. Black, observed: "
    f"{k_obs} of {n_obs} GENIE-BPC patients ({100*k_obs/n_obs:.1f}%; 150 each with lung, breast and colorectal cancer; all with "
    "sufficient follow-up). Coloured (light amber to dark wine = increasing patient evidence), generated: share of accepted futures "
    "with death before 6 months in each arm: "
    + "; ".join(f"{lab} {100*res[a][0]:.1f}% ({res[a][3]} of {res[a][4]} futures)" for a, lab, *_ in ARMS)
    + ". Up to ten futures were generated per patient and arm, so futures are not independent. Points, percentage; lines, 95% "
    "interval (Wilson interval for observed patients; percentile interval from 2,000 bootstrap resamples of the 450 patients for "
    "generated futures)."
)
save_panel(fig, "f4b_early_deaths", legend=legend,
           sources=["outputs/exploratory_four_arm_deep/real_side.csv", "outputs/exploratory_four_arm_deep/generated_side.csv"],
           confirm=["Generated-futures interval is a patient-level bootstrap (computed here, seed 0, 2,000 resamples); the manuscript gives only point percentages. Saved in f4b_early_deaths_aggregates.csv.",
                    "Counts 56/4416, 52/4407, 60/4426, 160/4430 match the manuscript percentages.",
                    "No P value: no test of observed versus generated proportions is saved (futures are nested in patients)."],
           message="Early deaths (6 months) were strongly under-represented in every arm: 16.2% observed vs 1.2-3.6% generated.")
print({a: [round(x, 4) if isinstance(x, float) else x for x in v] for a, v in res.items()}, k_obs, n_obs)
