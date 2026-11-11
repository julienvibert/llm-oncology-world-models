"""PRIME, Kaplan-Meier progression-free survival by arm and KRAS exon 2 status (Extended Data Fig. 8c).

Message: the classical view of the PRIME comparison, one progression-free survival curve per arm and KRAS group.
Data: PRIME ADSL and biomarker tables (Project Data Sphere; restricted, read from PDS_ROOT, default ~/PDS); patients of
the ITT population with a known KRAS exon 2 result. Only aggregate curves are drawn.
Same drawing as panel a of n12_prime_three_views.py, without its other panels and their inputs.
Run: uv run python scripts/figures/panels/n12a_prime_km.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHEMO, PAL, save_panel  # noqa: E402
from _f3util import axes_mm, new_fig  # noqa: E402

PDS = Path(os.environ.get("PDS_ROOT", Path.home() / "PDS"))
TRIAL = PDS / "NCT00364013" / "AllProvidedFiles_309 (1)" / "PDS_DSA_20050203"
MONTH_DAYS = 30.4375
ARMS = ("Panitumumab + FOLFOX", "FOLFOX alone")
KRAS = ("Wild-type", "Mutant")
PANI, FOLF = PAL["osimertinib"], CHEMO          # targeted + chemotherapy vs chemotherapy, as in Fig. 3d
ARM_COL = {"Panitumumab + FOLFOX": PANI, "FOLFOX alone": FOLF}
SOURCES = ["PRIME adsl_pds2019 and biomark_pds2019 (Project Data Sphere; aggregate curves only)"]

# ---------------------------------------------------------------- a: Kaplan-Meier, ITT with known KRAS
sl = pd.read_sas(TRIAL / "adsl_pds2019.sas7bdat", format="sas7bdat", encoding="latin-1")
bm = pd.read_sas(TRIAL / "biomark_pds2019.sas7bdat", format="sas7bdat", encoding="latin-1")
sl["kras"] = sl["SUBJID"].map(bm.set_index("SUBJID")["BMMTR1"])
sl = sl[sl.kras.isin(KRAS)]
sl["m"] = pd.to_numeric(sl["PFSDYCR"], errors="coerce") / MONTH_DAYS
sl["e"] = pd.to_numeric(sl["PFSCR"], errors="coerce").fillna(0).astype(int)


def km_curve(t: np.ndarray, e: np.ndarray):
    xs, ys, s = [0.0], [1.0], 1.0
    for u in np.unique(t[e == 1]):
        at = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        xs += [u, u]
        ys += [s, s * (1 - d / at)]
        s *= 1 - d / at
    return np.array(xs), np.array(ys)


W, H = 84.0, 46.0
fig = new_fig(W, H)
itt_n = {}
for j, kras in enumerate(KRAS):
    ax = axes_mm(fig, 10.0 + j * 37.0, 9.0, 33.0, 28.0)
    for arm in ARMS:
        g = sl[(sl.kras == kras) & (sl.TRT == arm)]
        itt_n[(kras, arm)] = len(g)
        x, y = km_curve(g.m.to_numpy(float), g.e.to_numpy(int))
        keep = x <= 24
        x, y = np.r_[x[keep], 24.0], np.r_[y[keep], y[keep][-1]]      # survival at 24 = last value on or before 24
        keep = np.ones(len(x), bool)
        ax.plot(x[keep], 100 * y[keep], color=ARM_COL[arm], lw=1.0, drawstyle="steps-post")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_yticks([0, 50, 100])
    if j:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Progression-free (%)")
    ax.set_title(f"KRAS {kras.lower()}", fontsize=6.5, pad=2)
    ax.set_xlabel("Months after randomization", fontsize=6.5)
fig.legend(handles=[Line2D([], [], color=ARM_COL[a], lw=1.0, label=("Panitumumab + FOLFOX" if a != "FOLFOX alone"
                                                                      else "FOLFOX alone")) for a in ARMS],
           loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, fontsize=6, frameon=False, handlelength=1.5)
save_panel(fig, "n12a_prime_km", sources=SOURCES,
           legend=("Kaplan-Meier progression-free survival (central review; progression or death) in PRIME, by randomized "
                   "arm, in patients with a known KRAS exon 2 result: wild-type, "
                   f"{itt_n[('Wild-type', ARMS[0])]} panitumumab + FOLFOX and {itt_n[('Wild-type', ARMS[1])]} FOLFOX alone; "
                   f"mutant, {itt_n[('Mutant', ARMS[0])]} and {itt_n[('Mutant', ARMS[1])]}. First 24 months shown. "
                   "Observed trial data."),
           message="The classical view: one progression-free survival curve per arm and KRAS group.")
