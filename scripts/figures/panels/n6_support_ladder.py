"""Panel n6_support_ladder: comparable real patients as a treatment decision is described more precisely.

Message: as a treatment decision is described more precisely, the number of comparable real
patients collapses in all three cancers.
Data: outputs/exploratory_n6_support_ladder/n6_support_ladder_v1.csv (corrected run, counts only;
unit one_decision_per_patient), patient numbers from its manifest.json.
Left: median (IQR) number of other patients sharing every criterion up to each step (primary:
a missing value matches only the same missing value), on log10(count + 1). Right: share of
decisions with no comparator; solid primary, dashed sensitivity in which a missing value matches
nothing. Cancer colours as the tumour families of Extended Data Fig. 4.
Run: uv run python scripts/figures/panels/n6_support_ladder.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FAMILY_COLOURS, INK, ROOT, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

SRC = ROOT / "outputs" / "exploratory_n6_support_ladder"
d = pd.read_csv(SRC / "n6_support_ladder_v1.csv")
d = d[d.unit.eq("one_decision_per_patient")]
man = json.loads((SRC / "manifest.json").read_text())
CANCERS = [("NSCLC", "Lung", FAMILY_COLOURS["lung"]), ("CRC", "Colorectal", FAMILY_COLOURS["colorectal"]), ("BRCA", "Breast", FAMILY_COLOURS["breast"])]
MARK = {"NSCLC": "o", "CRC": "s", "BRCA": "^"}
STEPS = ["Cancer", "+ stage", "+ driver", "+ regimen\ndepth", "+ treatment\nclass", "+ latest\nstatus",
         "+ previous\nregimen", "+ age,\nsex"]
n_pat = {c: man["counts"][c]["patients_with_decision"] for c, *_ in CANCERS}
assert sorted(d.step.unique()) == list(range(1, 9))

W, H = 180.0, 37.0
fig = new_fig(W, H)
a1 = axes_mm(fig, 12.0, 8.6, 99.0, 27.4)
a1.text(0.02, 0.04, "GENIE-BPC registry", transform=a1.transAxes, fontsize=6.5, ha="left", va="bottom")
a2 = axes_mm(fig, 127.0, 8.6, 51.0, 27.4)
x = np.arange(1, 9)
for k, (c, name, col) in enumerate(CANCERS):
    off = (k - 1) * 0.12
    p = d[d.cancer.eq(c) & d.variant.eq("primary")].sort_values("step")
    a1.plot(x + off, p["median"] + 1, "-", color=col, lw=1.0, zorder=2)
    a1.vlines(x + off, p["q1"] + 1, p["q3"] + 1, color=col, lw=0.8, zorder=2)
    a1.plot(x + off, p["median"] + 1, MARK[c], color=col, ms=3.4, mec="white", mew=0.4, zorder=3)
    s = d[d.cancer.eq(c) & d.variant.eq("missing_matches_nothing")].sort_values("step")
    a2.plot(x, 100 * p["share_zero"], "-", marker=MARK[c], color=col, lw=1.0, ms=2.8, mec="white", mew=0.3)
    a2.plot(x, 100 * s["share_zero"], "--", color=col, lw=0.9)
a1.set_yscale("log")
a1.set_ylim(0.8, 1600)
ticks = [0, 1, 5, 10, 100, 1000]
a1.set_yticks([t + 1 for t in ticks])
a1.set_yticklabels([str(t) for t in ticks])
a1.minorticks_off()
a1.set_ylabel("Comparable patients\n(median, IQR)")
a2.set_ylim(0, 100)
a2.set_yticks([0, 25, 50, 75, 100])
a2.set_ylabel("No comparable\npatient (%)")
for ax in (a1, a2):
    ax.set_xlim(0.6, 8.4)
    ax.set_xticks(x)
a1.set_xticklabels(STEPS, fontsize=6, linespacing=0.95)
a2.set_xticklabels([str(i) for i in x], fontsize=6)
a2.set_xlabel("Matching step (as left)", labelpad=1)
h = [Line2D([], [], color=col, marker=MARK[c], ms=3.4, mec=col, mew=0, lw=1.0, label=f"{name} (n = {n_pat[c]})")  # no white edge: the key line stays solid
     for c, name, col in CANCERS]
a1.legend(handles=h, loc="upper right", frameon=False, fontsize=6, handlelength=2.0, borderaxespad=0.2,
          labelspacing=0.3)
a2.legend(handles=[Line2D([], [], color=INK, lw=1.0, label="Missing matches missing"),
                   Line2D([], [], color=INK, lw=0.9, ls="--", label="Missing matches nothing")],
          loc="upper left", frameon=False, fontsize=6, handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)

tab = d[d.variant.eq("primary") & d.step.isin([4, 6, 8])]
txt = "; ".join(f"{name}: " + ", ".join(
    f"step {int(r.step)} median {r['median']:.0f} ({r.q1:.0f}-{r.q3:.0f}), none {100 * r.share_zero:.0f}%"
    for _, r in tab[tab.cancer.eq(c)].sort_values("step").iterrows()) for c, name, _ in CANCERS)
legend = (
    "Support for a treatment decision among real patients (GENIE-BPC; counts only, no outcome). Decision: start of a "
    "recorded systemic regimen after metastatic disease is documented, one decision per patient drawn at random "
    f"(lung {n_pat['NSCLC']}, colorectal {n_pat['CRC']} and breast {n_pat['BRCA']} patients). At each step the "
    "decision is described by one more criterion, using only information dated on or before the decision: stage at "
    "diagnosis; driver (lung EGFR, ALK or KRAS; colorectal mismatch-repair deficiency or MSI-high, BRAF V600E or KRAS; "
    "breast receptor subtype); order of the regimen after metastatic disease; treatment class; latest disease status; "
    "previous regimen (best status and duration band); age band and sex. Left, number of other patients matching every "
    "criterion so far (points, median; lines, interquartile range; log scale of n + 1, so 0 is shown); a missing value "
    "matches only the same missing value. Right, share of decisions with no comparable patient; dashed, sensitivity "
    "analysis in which a missing value matches nothing. " + txt + ". Under these criteria, comparable records become "
    "scarce in these cohorts; the panel does not establish equivalence to rare-cancer incidence or that prediction is "
    "impossible."
)
save_panel(fig, "n6_support_ladder", legend=legend,
           sources=["outputs/exploratory_n6_support_ladder/n6_support_ladder_v1.csv",
                    "outputs/exploratory_n6_support_ladder/manifest.json"],
           confirm=["Cancer colours reuse the tumour-family colours of Extended Data Fig. 4 (lung, gastrointestinal, breast).",
                    "Shared missing values are common in lung and colorectal ('not yet tested'); the dashed lines show "
                    "the stricter reading."],
           message="As a decision is described more precisely, comparable real patients become scarce in all three cancers.")
