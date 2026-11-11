"""Panel f6c_prime_map: PRIME arm means through the restricted state map, by KRAS exon 2 status.

Message: in KRAS wild-type patients the two arms diverge over time; in KRAS mutant patients they
do not (descriptive: assessment-weighted band means, no intervals; the state reached on treatment
is itself affected by treatment).

Sources: outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz (every lesion
assessment, first to last; positions come from the assessment table, not the transition table,
which lacks each patient's first assessment). KRAS exon 2 =
BMMTR1, failed/missing excluded. Trial data under agreement: aggregates only are drawn.
Run:  uv run python scripts/figures/panels/f6c_prime_map.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BLUE, CHEMO, INK, ROOT, fk, save_panel  # noqa: E402

SRC = "outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz"
d = pd.read_csv(ROOT / SRC)
PAN, FOL = "Panitumumab + FOLFOX", "FOLFOX alone"
COL = {PAN: BLUE, FOL: CHEMO}
d["mo"] = d["day"] / 30.4375
edges = [0, 2, 4, 6, 9, 12, 18]
# band 0 = baseline (each patient's first lesion assessment, on or before day 14 after randomisation,
# the rule of ed09b_progression_field); later assessments fall into time bands 1-6
is_base = d["assessment_index"].eq(0) & d["day"].le(14)
d["band"] = pd.cut(d["mo"], edges, right=False, labels=False) + 1
d.loc[is_base, "band"] = 0
d.loc[d["assessment_index"].eq(0) & ~is_base, "band"] = float("nan")  # late first assessments: no baseline, dropped
FAC = [("Wild-type", "KRAS exon 2 wild-type"), ("Mutant", "KRAS exon 2 mutant")]
# counts describe what is drawn: rows without a time band (late first assessments, assessments
# after 18 months) are dropped before counting
drawn = d.dropna(subset=["band"])
npat = drawn.groupby(["kras", "arm"]).SUBJID.nunique()
nass = drawn.groupby(["kras", "arm"]).size()

W, H = 180.0, 44.0
fig = fk.figure(width=W, height_mm=H)
PB, PH, PW = 15.0, 25.0, 72.0
axes = [fig.add_axes([l / W, PB / H, PW / W, PH / H]) for l in (17.0, 105.0)]  # noqa: E741
summ = {}
for ax, (k, name) in zip(axes, FAC):
    for arm in (FOL, PAN):
        g = d[(d.arm == arm) & (d.kras == k)].dropna(subset=["band"]).groupby("band")
        m1, m2 = g.pc1.mean(), g.pc2.mean()
        summ[(k, arm)] = (m1, m2)
        for i in range(len(m1) - 1):
            ax.annotate("", xy=(m1.iloc[i + 1], m2.iloc[i + 1]), xytext=(m1.iloc[i], m2.iloc[i]),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.32", color=COL[arm], lw=1.0,
                                        shrinkA=0, shrinkB=0), zorder=4)
        ax.plot(m1, m2, "o", color=COL[arm], ms=2.8, mec="white", mew=0.3, zorder=5)
        ax.plot(m1.iloc[0], m2.iloc[0], "o", ms=5.5, mfc="none", mec=INK, mew=0.6, zorder=6)
        ax.plot(m1.iloc[-1], m2.iloc[-1], "s", ms=5.5, mfc="none", mec=INK, mew=0.6, zorder=6)
allx = np.concatenate([v[0].to_numpy() for v in summ.values()])
ally = np.concatenate([v[1].to_numpy() for v in summ.values()])
xl = (allx.min() - 0.02, allx.max() + 0.05)
yl = (ally.min() - 0.02, ally.max() + 0.02)
for j, (ax, (k, name)) in enumerate(zip(axes, FAC)):
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xlabel("Restricted PRIME projection PC1 (a.u.)")
    if j == 0:
        ax.set_ylabel("Restricted PRIME\nprojection PC2 (a.u.)")
    else:
        ax.set_yticklabels([])
    ax.text(0.02, 1.02, f"{name} ({npat[(k, PAN)]} panitumumab + FOLFOX, {npat[(k, FOL)]} FOLFOX alone)",
            transform=ax.transAxes, fontsize=6, ha="left", va="bottom")
from matplotlib.lines import Line2D  # noqa: E402
fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, marker="o", ms=3, label="Panitumumab + FOLFOX"),
                    Line2D([], [], color=CHEMO, lw=1.2, marker="o", ms=3, label="FOLFOX alone"),
                    Line2D([], [], marker="o", mfc="none", mec=INK, ls="", ms=5, label="Baseline"),
                    Line2D([], [], marker="s", mfc="none", mec=INK, ls="", ms=5, label="Last band (12–18 months)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=6, handlelength=1.6,
           columnspacing=1.6)

legend = ("Average position of patients in the PRIME trial (NCT00364013; first-line metastatic colorectal cancer; Project "
          "Data Sphere) in the clinical state map over time, in patients with KRAS exon 2 wild-type (left) and mutant (right) "
          "tumours; both panels share the same scales. Blue, panitumumab + FOLFOX; purple, FOLFOX alone. Each dot is the "
          "average position of the assessments in one time band: baseline (ring; each patient's first lesion assessment, on "
          "or before day 14 after randomisation), then later assessments at 0–2, 2–4, 4–6, 6–9, 9–12 and 12–18 months after "
          "randomisation (square, last band); arrows join successive bands. Patients assessed more "
          "often weigh more and the patients contributing change between bands, so arrows are not individual paths and "
          "no interval is shown. Patients (assessments), wild-type: panitumumab + FOLFOX "
          f"{npat[('Wild-type', PAN)]} ({nass[('Wild-type', PAN)]}), FOLFOX alone {npat[('Wild-type', FOL)]} ({nass[('Wild-type', FOL)]}); "
          f"mutant: panitumumab + FOLFOX {npat[('Mutant', PAN)]} ({nass[('Mutant', PAN)]}), FOLFOX alone "
          f"{npat[('Mutant', FOL)]} ({nass[('Mutant', FOL)]}); patients with unknown or failed KRAS testing are not shown. "
          "Axes are the first two components of the state map fixed before the trial was examined, restricted to the 11 "
          "variables the trial supplies (lesion size and site, laboratory values, weight), centred on the trial mean, in "
          "arbitrary units (restricted PRIME projection). They combine tumour and host measurements and are not validated "
          "clinical scales: PC1 rises mainly with liver tumour burden, disease extent and hepatic laboratory impairment; PC2 "
          "rises with marrow impairment grade (lower haemoglobin, platelets or white cells) and falls with liver and other "
          "tumour burden. The state reached on treatment is itself affected by "
          "treatment: the display is descriptive.")
save_panel(fig, "f6c_prime_map", legend=legend,
           sources=[SRC, "outputs/exploratory_prime_trial_geometry/restricted_axis_correlations.json"],
           confirm=["Band means use every assessment (assessment table), including the first, with an explicit baseline "
                    "point (first assessment on or before day 14; 7 patients with a later first assessment lose it). "
                    "Patients per facet count everyone with a lesion assessment and known "
                    "KRAS exon 2 (BMMTR1; failed/missing excluded).",
                    "No axis direction words: marrow reserve is an impairment grade. Loadings: PC1 r = 0.84 liver tumour, 0.51 extent, 0.50 hepatic labs; PC2 r = 0.67 marrow "
                    "grade, -0.59 liver tumour, -0.39 extent (restricted_axis_correlations.json).",
                    "Descriptive display only.",
                    "Time bands are declared display choices; arrows join band means; no error bars.",
                    "Patient and assessment counts are taken after dropping rows without a time band, so they match "
                    "what is drawn."],
           message="Descriptive band means: in both KRAS exon 2 groups both arms move from baseline towards lower PC1; "
                   "arm differences are small and untested.")
