"""Panel n14_support_gradient (exploratory, post hoc): gain from matched examples by clinical agreement of the weakest matched donor.

Message: the 12-month whole-course gain from matched examples over generic examples does not change
detectably with how closely the weakest matched donor agreed with the patient.
Reads only the aggregate outputs of scripts/explore/n14_support_gradient_v1.py
(n14_summary_v1.csv, n14_statistics_v1.json).
Run: uv run --offline python scripts/figures/panels/n14_support_gradient.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _f3util import axes_mm, new_fig  # noqa: E402
from _f4 import PAL  # noqa: E402
from common import INK, ROOT, save_panel  # noqa: E402

D = ROOT / "outputs" / "exploratory_n14_support_gradient"
D_REL = "outputs/exploratory_n14_support_gradient"
PANEL = "n14_support_gradient"
S = pd.read_csv(D / "n14_summary_v1.csv", float_precision="round_trip")
J = json.loads((D / "n14_statistics_v1.json").read_text())
assert J["population"]["n"] == 233
COLS = [("primary", "+ matched examples\nvs + generic examples", "s"),
        ("secondary", "+ matched examples\nvs record alone", "o")]
THIRD = {1: "Weakest", 2: "Middle", 3: "Strongest"}

W, H = 88.0, 45.0
fig = new_fig(W, H)
L, PW, GAP, B, PH = 11.5, 36.0, 3.5, 14.5, 26.0
x = np.array([1, 2, 3])
ylim = (-0.012, 0.048)
for i, (name, title, mk) in enumerate(COLS):
    ax = axes_mm(fig, L + i * (PW + GAP), B, PW, PH)
    s = S[S.analysis == name].sort_values("third")
    assert s.third.tolist() == [1, 2, 3]
    ax.plot([0.5, 3.5], [0, 0], color=INK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    for xi, (_, r) in zip(x, s.iterrows()):
        ax.plot([xi, xi], [r.ci_low, r.ci_high], color=INK, lw=0.75, zorder=2, solid_capstyle="butt")
        ax.plot(xi, r.mean_gain, mk, ms=4.5 if mk == "s" else 5.0, mfc=PAL["arm_matched"], mec=INK, mew=0.6, zorder=3)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{THIRD[t]}\n(n = {n})" for t, n in zip(s.third, s.n)], fontsize=6)
    ax.set_yticks([0.0, 0.02, 0.04])
    ax.set_yticklabels(["0", "0.02", "0.04"] if i == 0 else [])
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=6, pad=2.0, linespacing=1.1)
    if i == 0:
        ax.set_ylabel("Gain in whole-course score\n(other arm minus matched)", fontsize=6.5)
fig.text((L + PW + GAP / 2) / W, 0.8 / H,
         "Clinical agreement of the weakest matched donor\n(third within cancer)",
         fontsize=6.5, ha="center", va="bottom")


def p_txt(p: float) -> str:
    return f"P = {p:.2f}" if p >= 0.01 else f"P = {p:.3f}"


def rho_txt(r: dict) -> str:
    return (f"Spearman ρ = {r['rho']:+.2f} (95% interval {r['ci_low']:+.2f} to {r['ci_high']:+.2f}), "
            f"permutation {p_txt(r['p_permutation'])}").replace("+", "").replace("-", "−")


def thirds_txt(name: str) -> str:
    s = S[S.analysis == name].sort_values("third")
    return "; ".join(f"{THIRD[r.third].lower()} third {r.mean_gain:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}), "
                     f"n = {r.n}" for _, r in s.iterrows()).replace("-", "−")


s1 = S[S.analysis == "primary"].sort_values("third")
unk = ", ".join(f"{100 * v:.0f}%" for v in s1.share_unknown_mean)
legend = (
    "Exploratory, post hoc analysis of the GENIE-BPC experiment (233 patients scored at 12 months: 87 lung, 59 breast "
    "and 87 colorectal cancer; the population of Fig. 5d). For each patient, the gain is the whole-course score at "
    "12 months (energy score on 17 trajectory summaries, equal fan sizes; lower is better) of the generic-examples arm "
    "(left) or the record-alone arm (right) minus that of the matched-examples arm; above 0, the futures generated with "
    "matched examples were closer to the documented course. Patients are grouped in thirds, within each cancer, of the "
    "clinical agreement between the patient and the 20th, least concordant matched example patient (the matcher's "
    "signed agreement over its fixed criteria, computed when the examples were selected, before generation); tied "
    "agreement values were kept in the same third, so group sizes differ. Points, mean gain; lines, 95% bootstrap "
    "interval (2,000 resamples within cancer and third). "
    f"Matched vs generic examples: {thirds_txt('primary')}; {rho_txt(J['primary'])} for the gain against agreement "
    f"ranked within cancer (2,000 bootstrap resamples stratified by cancer; 2,000 permutations within cancer). "
    f"Matched examples vs record alone: {thirds_txt('secondary')}; {rho_txt(J['secondary'])}. "
    f"Share of the matching criteria unknown for the patient, weakest to strongest third: {unk}. "
    "Every patient had at least 43 candidate example patients of the same cancer, treatment modality and era, so this "
    "analysis concerns weaker clinical agreement, not absence of comparable patients; weaker agreement may partly "
    "reflect sparser documentation. DeepSeek V4 Flash."
)
save_panel(fig, PANEL, legend=legend,
           sources=[f"{D_REL}/n14_summary_v1.csv", f"{D_REL}/n14_statistics_v1.json"],
           confirm=["Exploratory and post hoc.",
                    "Thirds are unequal because agreement scores are discrete and ties are kept together."],
           message="The gain from matched over generic examples at 12 months shows no detectable trend with the clinical "
                   "agreement of the weakest matched donor (exploratory).")
