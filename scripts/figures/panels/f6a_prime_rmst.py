"""Panel f6a_prime_rmst: PRIME, months gained with panitumumab + FOLFOX over 24 months, by subgroup.

Message: KRAS exon 2 wild-type gains progression-free time (+1.43 months, 0.09 to 2.74); the mutant
estimate is negative (-1.14, -2.60 to 0.24). Which other subgroups exclude zero is computed from the
table (with the shared 2,000-resample bootstrap, ECOG 0 does narrowly: +1.27, 0.01 to 2.53).

Source: outputs/exploratory_pds_prime_trial_geometry/subgroup_rmst24.csv.
Interval: bootstrap percentile 95% interval over patients within arm (2,000 resamples, seed 20261028,
months = days / 30.4375; shared with n8_prime_time_in_state).
Run:  uv run python scripts/figures/panels/f6a_prime_rmst.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BLUE, CHEMO, GREY, INK, ROOT, fk, save_panel  # noqa: E402

df = pd.read_csv(ROOT / "outputs" / "exploratory_pds_prime_trial_geometry" / "subgroup_rmst24.csv")
assert len(df) == 14
df = df.sort_values("rmst24_diff_months", ascending=False).reset_index(drop=True)
kras = df.subgroup.eq("KRAS exon 2")
# subgroups other than KRAS whose interval excludes zero (reported, not assumed)
EXCL = [f"{r.subgroup}: {r.level}" for r in df[~kras].itertuples() if r.ci_low > 0 or r.ci_high < 0]


def sg(v: float) -> str:
    return f"{v:.2f}".replace("-", "−")


NAME = {"KRAS exon 2": "KRAS exon 2", "ECOG": "ECOG performance status", "Baseline burden": "Baseline tumour burden",
        "Liver metastases": "Liver metastases", "Metastatic sites": "Metastatic sites",
        "Primary site": "Primary site", "Age": "Age (years)"}
LEVEL = {"Wild-type": "wild-type", "Mutant": "mutant", "no": "absent", "yes": "present",
         "low": "low", "high": "high", "1-2": "1-2", "3+": "3 or more", "0": "0", "<65": "under 65",
         "65+": "65 or over", "Colon": "colon", "Rectal": "rectum"}


def label(r) -> str:
    lev = LEVEL.get(r.level, r.level)
    if r.subgroup == "ECOG":
        lev = r.level
    return f"{NAME[r.subgroup]}: {lev}"


W, H = 180.0, 42.0
fig = fk.figure(width=W, height_mm=H)
PL, PW, PB, PH = 35.0, 78.0, 7.6, 27.6
ax = fig.add_axes([PL / W, PB / H, PW / W, PH / H])
n = len(df)
for i, r in df.iterrows():
    hi = bool(kras[i])
    c = BLUE if (hi and r.rmst24_diff_months > 0) else (CHEMO if hi else INK)
    ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=0.9)
    ax.plot(r.rmst24_diff_months, i, "o", color=c, ms=3.8 if hi else 3.0, mfc=c if hi else "white", mew=0.7, zorder=3)
ax.axvline(0, color=GREY, lw=0.5)
ax.set_xlim(-3, 3.6)
ax.set_ylim(n - 0.5, -0.5)
ax.set_yticks(range(n))
ax.set_yticklabels([label(r) for r in df.itertuples()], fontsize=6)
for t, k in zip(ax.get_yticklabels(), kras):
    if k:
        t.set_fontweight("bold")
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xticklabels(["−3", "−2", "−1", "0", "1", "2", "3"])
ax.set_xlabel("Months gained with panitumumab + FOLFOX (restricted mean PFS, 0-24 months)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
# direction keys above the axis, with arrows
fig.text((PL + PW * (3.0 - 0.15) / 6.6) / W, (PB + PH + 0.7) / H, "\u2190 FOLFOX alone better", fontsize=6, ha="right", va="bottom")
fig.text((PL + PW * (3.0 + 0.15) / 6.6) / W, (PB + PH + 0.7) / H, "panitumumab + FOLFOX better \u2192", fontsize=6, ha="left", va="bottom")

# number columns, aligned to the rows
yfrac = lambda i: (PB + PH * (1 - (i + 0.6) / (n + 0.0))) / H  # noqa: E731
y0, y1 = ax.get_ylim()
def ypos(i):  # row i centre in figure fraction
    return (PB + PH * (i - y0) / (y1 - y0)) / H
X1, X2 = 119.0, 153.0
fig.text(X1 / W, (PB + PH + 0.7) / H, "Patients, panitumumab + FOLFOX\nvs FOLFOX alone", fontsize=6, ha="left", va="bottom")
fig.text(X2 / W, (PB + PH + 0.7) / H, "Months gained\n(95% interval)", fontsize=6, ha="left", va="bottom")
for i, r in df.iterrows():
    b = "bold" if kras[i] else "normal"
    fig.text(X1 / W, ypos(i), f"{int(r.n_experimental)} vs {int(r.n_control)}", fontsize=6, ha="left", va="center", fontweight=b)
    fig.text(X2 / W, ypos(i), f"{sg(r.rmst24_diff_months)} ({sg(r.ci_low)} to {sg(r.ci_high)})", fontsize=6, ha="left",
             va="center", fontweight=b)

rows = "; ".join(f"{label(r)} {sg(r.rmst24_diff_months)} ({sg(r.ci_low)} to {sg(r.ci_high)})" for r in df.itertuples())
legend = ("Difference in the mean number of months alive without disease progression over the first 24 months (restricted "
          "mean progression-free survival), panitumumab + FOLFOX minus FOLFOX alone, in the PRIME trial (NCT00364013; "
          "randomised, first-line metastatic colorectal cancer; Project Data Sphere), in each predefined baseline subgroup, "
          "sorted by the estimate. Points to the right of the vertical line favour panitumumab + FOLFOX. Filled points and bold "
          "labels, KRAS exon 2 status (blue, benefit; purple, point estimate below zero); open points, other subgroups. Horizontal "
          "lines, 95% bootstrap intervals (2,000 resamples of patients within arm, seed 20261028, shared with Fig. 6c; months = "
          "days / 30.4375). Right columns, number of patients with "
          "panitumumab + FOLFOX versus FOLFOX alone, and the estimate with its interval. ECOG, performance status. "
          "Observed trial data, not generated. No test of difference between subgroups is shown. Months (95% interval): " + rows + ".")
save_panel(fig, "f6a_prime_rmst", legend=legend,
           sources=["outputs/exploratory_pds_prime_trial_geometry/subgroup_rmst24.csv"],
           confirm=["All 14 subgroups, sorted by estimate; patients shown as 'n vs n' (experimental vs control) from the CSV; "
                    "KRAS rows are all randomised patients with a KRAS exon 2 result (BMMTR1: 514 wild-type, 352 mutant; 69 "
                    "failed or missing excluded, never counted as mutant), the same intention-to-treat basis as ed09a_forest_hr.",
                    "No P value: the CSV has none and a subgroup-by-treatment interaction test was not computed.",
                    "Colour: KRAS wild-type blue (panitumumab colour), mutant purple (FOLFOX colour) for the sign of the estimate only."],
           message="KRAS exon 2 wild-type gains progression-free time with panitumumab + FOLFOX; the mutant estimate is "
                   "negative. Other subgroups whose interval excludes zero: " + (", ".join(EXCL) or "none") + ".")
