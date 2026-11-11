"""Panel f4e_fan_coverage: observed outcomes outside the generated range (patients who died).

Message: for only about one third of patients who died did the actual survival fall inside the 10th-90th
percentile range of their generated futures, in every arm.
Data: deep_results.json, fan_coverage[arm].covers_central_80 (proportion, bootstrap lower, upper); n = 260.
Run: uv run python scripts/figures/panels/f4e_fan_coverage.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f4 import ARMS, S, dot, fix_axes, write_aggregates  # noqa: E402
from common import panel, save_panel  # noqa: E402

cov = json.loads((S / "deep_results.json").read_text())["fan_coverage"]
assert {cov[k]["n"] for k, *_ in ARMS} == {260}
write_aggregates("f4e_fan_coverage", ["arm", "n_patients_died", "share_within_p10_p90_pct", "lo_pct", "hi_pct"],
                 [[k, 260, *(round(100 * v, 2) for v in cov[k]["covers_central_80"])] for k, *_ in ARMS])

fig, ax = panel(50, 44)
y = np.arange(len(ARMS))[::-1]
for yi, (k, lab, col, _) in zip(y, ARMS):
    p, lo, hi = (100 * v for v in cov[k]["covers_central_80"])
    dot(ax, yi, p, lo, hi, col)
    ax.text(hi + 2, yi, f"{p:.1f}%", va="center", ha="left", fontsize=6)
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab, *_ in ARMS])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_ylim(-0.6, len(ARMS) + 0.6)
ax.text(4, len(ARMS) - 0.1, "n = 260 patients who died", fontsize=6, va="center", ha="left")
ax.set_xlabel("Survival within the futures'\n10th-90th percentile (%)")
fix_axes(fig, ax, left=22.5, right=3.5, bottom=10.5, top=1.0)

rows = "; ".join(f"{lab} {100*cov[k]['covers_central_80'][0]:.1f}% ({100*cov[k]['covers_central_80'][1]:.1f} to {100*cov[k]['covers_central_80'][2]:.1f})" for k, lab, *_ in ARMS)
legend = (
    "Whether real outcomes lay inside the range of the generated futures. Among the 260 patients whose death was observed, "
    "share of patients whose actual survival time lay between the 10th and 90th percentile of the patient's generated futures "
    "(at least five and usually ten futures per patient and arm). Light amber to dark wine = "
    "increasing patient evidence). The measure is restricted to observed deaths and uses percentiles of a few futures; it does not "
    "establish population coverage and cannot separate survival bias from narrow spread. Points, share of patients; lines, 95% "
    f"bootstrap interval over patients. Share (95% interval): {rows}."
)
save_panel(fig, "f4e_fan_coverage", legend=legend, sources=["outputs/exploratory_four_arm_deep/deep_results.json",
           "outputs/exploratory_four_arm_deep/FINDINGS.md"],
           confirm=["Exploratory analysis of survival times (deep_results.json).",
                    "Values from deep_results.json fan_coverage (32.3-33.5%, 'about one third'); no nominal line drawn."],
           message="Actual survival fell within the 10th-90th percentile range of generated futures in about one third of patients who died, in every arm.")
