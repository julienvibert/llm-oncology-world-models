"""Panel ed09b_progression_field: local difference in 12-month progression-free survival across the restricted PRIME
state map, KRAS exon 2 wild-type and mutant (censoring-aware).

Source: outputs/exploratory_prime_corrections_2026_10_12/prime_local_km12_surface.csv (kernel-
weighted Kaplan-Meier at 12 months per arm, patient-level bootstrap) and the baseline positions
in outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz (drawn as dots only; no
identifiers). Producer: scripts/explore/build_prime_itt_forest_local_km.py.
Run:  uv run python scripts/figures/panels/ed09b_progression_field.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BLUE, CHEMO, INK, ROOT, fk, save_panel  # noqa: E402

D = ROOT / "outputs/exploratory_prime_corrections_2026_10_12"
surf = pd.read_csv(D / "prime_local_km12_surface.csv")
man = json.loads((D / "manifest.json").read_text())
P = man["parameters"]
visits = pd.read_csv(ROOT / "outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz")
base = visits[visits.assessment_index.eq(0) & visits.day.le(P["baseline_max_day"]) & visits.kras.isin(["Wild-type", "Mutant"])]
CMAP = LinearSegmentedColormap.from_list("pfs", [CHEMO, "#FFFFFF", BLUE])  # positive = more PFS with panitumumab
V = 0.2
PAN, FOL = "Panitumumab + FOLFOX", "FOLFOX alone"
pop = man["cohort"]["map_population"]["by_kras_arm"]

W, H = 180.0, 62.0
fig = fk.figure(width=W, height_mm=H)
PB, PH, PW = 12.0, 44.0, 64.0
axs = [fig.add_axes([l / W, PB / H, PW / W, PH / H]) for l in (14.0, 84.0)]  # noqa: E741
xl = (base.pc1.min() - 0.03, base.pc1.max() + 0.03)
yl = (base.pc2.min() - 0.03, base.pc2.max() + 0.03)
art = None
for j, (ax, k) in enumerate(zip(axs, ["Wild-type", "Mutant"])):
    b = base[base.kras.eq(k)]
    ax.scatter(b.pc1, b.pc2, s=1.5, color=INK, alpha=0.35, lw=0, zorder=1)
    s = surf[surf.kras.eq(k)].reset_index(drop=True)
    sh = s[s.shown]
    ess = sh[["ess_pan", "ess_fol"]].min(axis=1)
    art = ax.scatter(sh.x, sh.y, s=10 + 1.0 * ess, c=sh.diff12, cmap=CMAP, vmin=-V, vmax=V, edgecolor=INK,
                     linewidth=[1.2 if e else 0.4 for e in sh.excludes_zero], zorder=3)
    for num, r in enumerate(sh.itertuples(), 1):  # numbered in panel-c order (PC1, then PC2)
        ax.annotate(str(num), (r.x, r.y), xytext=(4.5, 3.0), textcoords="offset points", fontsize=6,
                    ha="left", va="bottom", color=INK, zorder=4)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.text(0.0, 1.03, f"KRAS exon 2 {k.lower()} ({pop[f'{k} | {PAN}']} vs {pop[f'{k} | {FOL}']} patients)",
            transform=ax.transAxes, fontsize=6, fontweight="bold", va="bottom")
    ax.set_xlabel("Restricted PRIME projection PC1 (a.u.)")
    if j == 0:
        ax.set_ylabel("Restricted PRIME projection PC2 (a.u.)")
    else:
        ax.set_yticklabels([])
cax = fig.add_axes([158 / W, (PB + 8) / H, 3 / W, (PH - 16) / H])
cb = fig.colorbar(art, cax=cax, extend="both")
cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
cb.set_ticklabels(["−0.2", "−0.1", "0", "0.1", "0.2"])
cb.set_label("Difference in 12-month\nprogression-free survival", fontsize=6)
cb.outline.set_visible(False)
fig.text(156 / W, (PB + PH - 6) / H, "higher with\npanitumumab", fontsize=6, va="bottom", ha="left", linespacing=1.0)
fig.text(156 / W, (PB + 6) / H, "higher with\nFOLFOX alone", fontsize=6, va="top", ha="left", linespacing=1.0)

shown = surf[surf.shown]
rng = {k: g.diff12 for k, g in shown.groupby("kras")}
nb = man["cohort"]["map_population"]["with_known_kras"]
legend = (
    "The forest question asked across the state map, in observed PRIME patients (NCT00364013; not generated data). "
    f"Patients with a baseline lesion assessment (on or before day {P['baseline_max_day']} after randomisation) and a known "
    f"KRAS exon 2 result ({nb} of 935 randomised); small black dots, their baseline positions. Axes: the restricted PRIME "
    "projection, that is the first two components of the frozen state map restricted to the 11 coordinates the trial "
    "supplies and centred on the trial (arbitrary units; not validated clinical scales); many patients share identical encoded "
    "states, so dots overlap (darker dots, several patients). PC1 rises mainly with liver "
    "tumour burden, disease extent and hepatic laboratory impairment; PC2 rises with marrow impairment grade (lower "
    "haemoglobin, platelets or white cells) and falls with liver and other tumour burden. Large dots, numbered grid points "
    "(numbers match panel c; patients in brackets, panitumumab + FOLFOX vs FOLFOX alone): colour, local difference in the Kaplan-Meier probability of being alive without "
    "progression at 12 months, panitumumab + FOLFOX minus FOLFOX alone, each patient weighted by closeness of baseline "
    f"position (Gaussian kernel, bandwidth {P['bandwidth_sd']} standard deviations of each axis). Patients censored before 12 "
    "months leave the estimate at censoring; they are not counted as free of progression. Dot size, effective number of "
    f"patients in the smaller arm; grid points with fewer than {int(P['min_ess'])} are not drawn. A thick outline would mark "
    "a 95% bootstrap interval excluding zero: none does (intervals in panel c). Local values range from "
    f"{rng['Wild-type'].min():+.2f} to {rng['Wild-type'].max():+.2f} (wild-type) and {rng['Mutant'].min():+.2f} to "
    f"{rng['Mutant'].max():+.2f} (mutant). Neighbouring grid points share patients and are not independent; local "
    "differences between randomised arms are not individual treatment effects."
).replace("-0.", "−0.")
save_panel(fig, "ed09b_progression_field", legend=legend,
           sources=["outputs/exploratory_prime_corrections_2026_10_12/prime_local_km12_surface.csv",
                    "outputs/exploratory_prime_corrections_2026_10_12/manifest.json",
                    "outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz"],
           confirm=["KRAS exon 2 (BMMTR1; failed/missing excluded); endpoint PFS (progression or death) at 12 months "
                    "with a censoring-aware kernel-weighted Kaplan-Meier, patient-"
                    "level bootstrap (1,000). Same estimand and horizon as panel c.",
                    "Restricted PRIME projection, so that all PRIME displays share one map. No axis direction words; loadings "
                    "described in the legend (from restricted_axis_correlations.json: PC1 r = 0.84 liver tumour, 0.51 "
                    "extent, 0.50 hepatic labs; PC2 r = 0.67 marrow grade, -0.59 liver tumour, -0.39 extent).",
                    "Grid 4 x 4 evenly spaced between the 15th and 85th percentiles of each axis (15 of 16 points drawn per group; the "
                    "low-PC1, low-PC2 corner has too few patients); bandwidth 0.6 SD "
                    "and ESS >= 20 are declared choices.",
                    "Number labels at fontsize 6 above dots."],
           message="Across the state map, local differences in 12-month progression-free survival vary in sign, but no "
                   "local interval excludes zero.")
