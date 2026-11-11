"""Panel ed10a_error_vs_random: prediction error of the peer median falls with clinical matching, not with random grouping.

Source: outputs/exploratory_precision_ladder/readout.json (median_err_fixed; null_fixed_targets: random
comparators for the SAME fixed targets, same comparator count per target).
Run:  uv run python scripts/figures/panels/ed10a_error_vs_random.py
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import INK, ORANGE, ROOT, fk, save_panel  # noqa: E402

src = json.loads((ROOT / "outputs/exploratory_precision_ladder/readout.json").read_text())
rows = src["per_rung"]
clin = np.array(src["median_err_fixed"], float)
NF = src["null_fixed_targets"]
rnd = np.array(NF["median_err"], float)
rlo, rhi = np.array(NF["draw_lo"], float), np.array(NF["draw_hi"], float)
assert src.get("ladder_definition", "").startswith("cumulative")
x = np.arange(len(rows))
LAB = ["All", "+ site\nclass", "+ stage", "+ driver", "+ condi-\ntion", "+ age,\nsex", "+ exact\nhistology"]
assert len(LAB) == len(rows)
cols = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, len(rows)))
W, H = 88.0, 54.0
fig = fk.figure(width=W, height_mm=H)
ax = fig.add_axes([14 / W, 14 / H, 72 / W, 38 / H])
ax.plot(x, rnd, color=ORANGE, lw=1.2, ls="--", marker="s", ms=3.5, zorder=2)
ax.vlines(x, rlo, rhi, color=ORANGE, lw=0.9, zorder=2)
ax.plot(x, clin, color=INK, lw=0.8, zorder=2)
ax.scatter(x, clin, c=cols, s=24, edgecolor=INK, linewidth=0.5, zorder=3)
ax.fill_between(x, clin, rnd, color=ORANGE, alpha=0.12, lw=0)
ax.set_ylim(0, rnd.max() + 1.2)
ax.set_xlim(-0.4, len(rows) - 0.6)
ax.set_xticks(x)
ax.set_xticklabels(LAB)
ax.set_ylabel("Error of the comparators' median\nsurvival (months)")
ax.set_xlabel("Matching criteria added")
ax.text(x[-1], clin[-1] - 0.45, f"{clin[-1]:.1f}", fontsize=6, ha="center", va="top")
ax.text(x[0], clin[0] - 0.45, f"{clin[0]:.1f}", fontsize=6, ha="center", va="top")
ax.text(x[-1], rnd[-1] + 0.3, f"{rnd[-1]:.1f}", fontsize=6, ha="center", va="bottom", color=INK)
ax.text(1.2, rnd.max() + 0.7, "same trajectories, random comparators", fontsize=6, ha="left", va="center")
ax.text(1.2, clin[3] - 1.7, "comparators matched on clinical criteria", fontsize=6, ha="left", va="center")
nn = src["n_fixed_cohort"]
legend = ("Does matching on clinical criteria help, or does any grouping of the same size give the same result? Generated "
          f"trajectories from the six-model atlas (not observed patients); the {nn:,} trajectories that have a comparator at "
          "every step (the same target trajectories on both lines). For each trajectory, the comparators are the other generated "
          "trajectories that share all criteria added so far (the same steps as in Fig. 6d, left to right; each step keeps every "
          "earlier criterion). Black line with coloured dots (colours as in Fig. 6d), median over "
          "trajectories of the absolute difference, in months, between the trajectory's survival and the median survival of its "
          "comparators. Orange dashed line, the same quantity for the same trajectories when each keeps the same number of "
          f"comparators but they are drawn at random from all other trajectories ({src['n_null_draws']:,} draws; median over "
          "draws, vertical lines the range of the central 95% of draws). The shaded gap is the value of the clinical criteria. Error falls with matching and rises "
          "with random grouping, so the gain comes from the clinical information, not from group size. This describes this atlas and "
          "this prediction task; it is not a rule for the best matching level. Survival here is overall survival in the atlas.")
save_panel(fig, "ed10a_error_vs_random", legend=legend,
           sources=["outputs/exploratory_precision_ladder/readout.json"],
           confirm=["Ladder cumulative (last rung keeps primary-site class); random control scored on the same fixed targets "
                    "(each keeps its clinical comparator count, comparators drawn at random). "
                    f"Fixed cohort {nn:,}. Finest step: clinical {clin[-1]:.2f} vs random {rnd[-1]:.2f} months.",
                    "The partition-based null restricted to the same targets is in the readout with its coverage (79% at the "
                    "finest step); it gives the same picture.",
                                        "Survival in months uses 30.4 days per month in this producer (manuscript: 30.4375)."],
           message="Clinical matching lowers the comparators' error; random groups of the same size raise it.")
