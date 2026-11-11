#!/usr/bin/env python3
"""A RANDOMISED TRIAL AS MOVEMENT THROUGH CLINICAL STATE SPACE

PRIME (NCT00364013), panitumumab plus FOLFOX against FOLFOX alone in first-line metastatic
colorectal cancer, from Project Data Sphere. Both arms supplied, 935 patients, with
central-review lesion measurements at every scheduled assessment.

WHY THIS TRIAL. Three properties, all necessary. Both arms are present, so a treatment
contrast exists at all. Lesions are measured centrally with diameters and anatomical sites,
so movement can be computed rather than imputed. And the effect is known to reverse sign by
KRAS status -- panitumumab helps wild-type disease and harms mutant -- which gives a
biological target the geometry can be checked against rather than merely displayed on.

WHAT IS AND IS NOT MODELLED. Every quantity here is observed. The state coordinates come
from measured lesion diameters and sites; the movement is the difference between consecutive
measured visits; the arms are as randomised. No model completes anything. The synthetic
laboratory supplies only the coordinate system -- the frozen basis in
outputs/paper_state_map/ -- and that basis was fixed before this trial was opened.

RANDOMISATION IS WHAT MAKES THIS DIFFERENT. Every other differential field in this work is
model-implied or observational. Here the arm contrast is licensed by design, so a difference
between arm-specific fields is causal at the population level. It is NOT an individual
counterfactual: each patient sits in one arm, and no patient's untreated future is observed.

TWO ENCODING DECISIONS, BOTH DECLARED.

  READER   Every scan is read by two radiologists. One reader is used throughout rather than
           averaged, so that a difference between visits is a change in the patient and not
           a change in who read the film.

  BURDEN   The frozen map encodes anatomical involvement as an ordinal 0-5. PRIME gives
           summed diameters in millimetres per site. The mapping below is a fixed clinical
           binning declared in advance, not fitted.

  ORGAN    The map encodes organ function as an ordinal 0-4. PRIME gives laboratory values
           in SI units, one unit per test throughout. The thresholds below are CTCAE-style
           cut-points against standard upper or lower limits of normal -- fixed clinical
           ranges, as the contract requires, never percentiles of this cohort. Laboratory
           and imaging follow different schedules, so the most recent value is carried
           forward to each assessment; the contract specifies LOCF and forbids interpolation.

THE HOST AXIS IS NOT OPTIONAL. A first version built the tumour coordinates alone and the
result was degenerate: PC2 in the frozen basis is largely host content, so with nothing to
put in it every patient collapsed onto a single diagonal and the two-dimensional state space
was one-dimensional. Laboratory and weight measurements are what give the second axis
independent content.

Run: uv run python scripts/explore/prime_trial_state_geometry.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PDS = Path(os.environ.get("PDS_ROOT", Path.home() / "PDS"))
TRIAL = PDS / "NCT00364013" / "AllProvidedFiles_309 (1)" / "PDS_DSA_20050203"
MAP = ROOT / "outputs" / "paper_state_map"
OUT = ROOT / "outputs" / "exploratory_prime_trial_geometry"

READER = "Radiologist 1"
# Biomarker column used for every PRIME display: BMMTR1, which the trial's own label column
# BMMTNM1 names "KRAS exon 2 (c12/13)" (checked at run time). "Failure" and missing values
# are not a result: they stay as unknown and are excluded from KRAS-split displays.
KRAS_EXON2_COLUMN = "BMMTR1"
KRAS_EXON2_LABEL_COLUMN = "BMMTNM1"
KRAS_EXON2_LABEL = "KRAS exon 2 (c12/13)"
# Millimetres of summed diameter at one site -> the frozen 0-5 ordinal. Declared, not fitted.
BURDEN_MM = [0, 20, 50, 100, 200]
# PRIME lesion sites -> frozen coordinates. Sites with no frozen counterpart fold into other.
SITE_MAP = {
    "Liver": "tumour_hepatic",
    "Pulmonary": "tumour_thoracic",
    "Lymph Nodes": "tumour_nodal",
    "Bone": "tumour_osseous",
    "Abdomen": "tumour_peritoneal",
}
OTHER = "tumour_other_anatomical"
EXTENT = "tumour_extent"
TUMOUR = [*dict.fromkeys(SITE_MAP.values()), OTHER, EXTENT]

# CTCAE-style grades against standard limits of normal, in the SI units PRIME reports.
# Ascending cut-points; a value at or below the first is grade 0.
GRADE = {
    "Creatinine": ([110, 165, 330, 660], "high"),          # umol/L, ULN 110
    "Alkaline Phosphatase": ([130, 325, 650, 2600], "high"),   # U/L, ULN 130
    "Albumin": ([35, 30, 20, 10], "low"),                  # g/L, LLN 35
    "Hemoglobin": ([120, 100, 80, 65], "low"),             # g/L, LLN 120
    "Platelets": ([150, 75, 50, 25], "low"),               # 10^9/L, LLN 150
    "White Blood Cells": ([4, 3, 2, 1], "low"),            # 10^9/L, LLN 4
}
ORGAN = {"host_renal": ["Creatinine"],
         "host_hepatic": ["Alkaline Phosphatase", "Albumin"],
         "host_marrow_reserve": ["Hemoglobin", "Platelets", "White Blood Cells"]}
WEIGHT = "host_relative_weight_loss"
COORDS = TUMOUR + list(ORGAN) + [WEIGHT]


def rd(name: str) -> pd.DataFrame:
    return pd.read_sas(TRIAL / f"{name}_pds2019.sas7bdat", format="sas7bdat",
                       encoding="latin-1")


def burden(mm: float) -> float:
    """Summed diameter at one site as the frozen ordinal, scaled to 0-1."""
    return float(np.searchsorted(BURDEN_MM, mm, side="right") - 1) / 5.0


def grade(value: float, test: str) -> float:
    """A laboratory value as a CTCAE-style 0-4 grade, scaled to 0-1."""
    cuts, direction = GRADE[test]
    if not np.isfinite(value):
        return np.nan
    g = sum(value > c for c in cuts) if direction == "high" else sum(value < c for c in cuts)
    return float(g) / 4.0


def host_state(days: pd.DataFrame) -> pd.DataFrame:
    """Organ-function and weight coordinates, carried forward to each assessment day.

    Laboratory draws and imaging follow different schedules, so a value is carried forward
    from the most recent draw at or before the assessment. The contract specifies LOCF and
    forbids interpolation, so a draw taken after a scan never informs that scan. merge_asof
    with direction="backward" is exactly that rule, and it tolerates repeat measurements on
    a single day where a reindex would not.
    """
    want = days.drop_duplicates().sort_values(["day", "SUBJID"]).reset_index(drop=True)

    lb = rd("adlb")
    lb["day"] = pd.to_numeric(lb["VISITDY"], errors="coerce")
    lb["val"] = pd.to_numeric(lb["LBSTRESN"], errors="coerce")
    lb = lb.dropna(subset=["day", "val"])
    lb = lb[lb["LBTEST"].isin(GRADE)].copy()
    lb["g"] = [grade(v, t) for v, t in zip(lb["val"], lb["LBTEST"])]

    out = want.copy()
    for coord, tests in ORGAN.items():
        # worst grade among the tests defining this organ, one value per patient-day
        per = (lb[lb["LBTEST"].isin(tests)]
               .groupby(["SUBJID", "day"])["g"].max().rename(coord).reset_index()
               .sort_values(["day", "SUBJID"]))
        out = pd.merge_asof(out, per, on="day", by="SUBJID", direction="backward")

    pm = rd("adpm")
    pm["day"] = pd.to_numeric(pm["VISITDY"], errors="coerce")
    pm["kg"] = pd.to_numeric(pm["WEIGHT"], errors="coerce")
    pm = (pm.dropna(subset=["day", "kg"])
          .groupby(["SUBJID", "day"])["kg"].mean().reset_index()
          .sort_values(["day", "SUBJID"]))
    base = pm.sort_values("day").groupby("SUBJID")["kg"].first()
    out = pd.merge_asof(out, pm, on="day", by="SUBJID", direction="backward")
    b = out["SUBJID"].map(base)
    pct = np.clip(100 * (b - out["kg"]) / b, -20, 30)
    out[WEIGHT] = (pct + 20) / 50
    return out.drop(columns=["kg"])


def visits() -> pd.DataFrame:
    """One row per patient per assessment: the observed state at that visit."""
    lesions = rd("adls")
    lesions = lesions[lesions["LSREADER"].eq(READER)].copy()
    lesions["day"] = pd.to_numeric(lesions["VISITDY"], errors="coerce")
    lesions["ld"] = pd.to_numeric(lesions["LSLD"], errors="coerce")
    lesions["sld"] = pd.to_numeric(lesions["LSSLD"], errors="coerce")
    lesions = lesions.dropna(subset=["day"])
    lesions["coord"] = lesions["LSSITE"].map(SITE_MAP).fillna(OTHER)

    # Summed diameter per site per visit, then the frozen ordinal.
    per = (lesions.dropna(subset=["ld"])
           .groupby(["SUBJID", "day", "coord"])["ld"].sum().unstack("coord"))
    per = per.reindex(columns=[c for c in TUMOUR if c != EXTENT]).fillna(0.0)
    state = per.map(burden)

    # Disease extent by the frozen rule: distinct involved classes, capped at five.
    state[EXTENT] = np.minimum((per > 0).sum(axis=1), 5) / 5.0
    state["sld_mm"] = (lesions.dropna(subset=["sld"])
                       .groupby(["SUBJID", "day"])["sld"].first())
    state = state.reset_index()
    host = host_state(state[["SUBJID", "day"]])
    return state.merge(host, on=["SUBJID", "day"], how="left")


def frozen_basis() -> tuple[np.ndarray, np.ndarray, dict]:
    """The frozen PC1/PC2 loadings restricted to the coordinates a trial supplies.

    Restriction, not refitting: the loadings are the ones committed in the state-map
    contract, and only the terms for coordinates PRIME measures are retained. Both this
    trial and any other population must pass through the identical restricted functional
    for their positions to be comparable.
    """
    model = json.loads((MAP / "state_map_contract.json").read_text())["primary_model"]
    names = model["feature_names"]
    comp = np.array(model["components_after_feature_multipliers"])
    idx = [names.index(c) for c in COORDS]
    obs = np.zeros(len(names), bool)
    obs[idx] = True
    reach = {f"PC{i+1}": float(np.abs(comp[i][obs]).sum() / np.abs(comp[i]).sum())
             for i in range(2)}
    return comp[0][idx], comp[1][idx], reach


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v = visits()
    c1, c2, reach = frozen_basis()

    have = v[COORDS].notna().mean()
    v[COORDS] = v[COORDS].fillna(v[COORDS].median())
    x = v[COORDS].to_numpy(float)
    centre = x.mean(axis=0)
    v["pc1"], v["pc2"] = (x - centre) @ c1, (x - centre) @ c2

    sl = rd("adsl").set_index("SUBJID")
    bm = rd("biomark").set_index("SUBJID")
    names = set(bm[KRAS_EXON2_LABEL_COLUMN].dropna().unique())
    assert names == {KRAS_EXON2_LABEL}, names
    v["arm"] = v["SUBJID"].map(sl["TRT"])
    kras = v["SUBJID"].map(bm[KRAS_EXON2_COLUMN])
    v["kras"] = kras.where(kras.isin(["Wild-type", "Mutant"]), "Unknown")
    v = v.sort_values(["SUBJID", "day"])
    v["assessment_index"] = v.groupby("SUBJID").cumcount()

    # Movement is the step between consecutive OBSERVED assessments.
    g = v.groupby("SUBJID")
    for a in ("pc1", "pc2", "day"):
        v[f"d_{a}"] = g[a].diff()
    steps = v.dropna(subset=["d_pc1", "d_day"])
    steps = steps[steps["d_day"] > 0]

    cov = {"patients_with_lesions": int(v["SUBJID"].nunique()),
           "assessments": int(len(v)),
           "transitions": int(len(steps)),
           "median_assessments_per_patient": float(g.size().median()),
           "median_days_between": float(steps["d_day"].median()),
           "observable_loading_mass": reach,
           "coordinate_completeness": {k: round(float(val), 3) for k, val in have.items()},
           "by_arm": {str(k): int(n) for k, n in steps["arm"].value_counts().items()},
           "by_kras": {str(k): int(n) for k, n in steps["kras"].value_counts().items()}}

    # Per-cell counts decide whether a field can be drawn at all.
    for nb in (3, 4, 5):
        qx = pd.qcut(steps["pc1"], nb, labels=False, duplicates="drop")
        qy = pd.qcut(steps["pc2"], nb, labels=False, duplicates="drop")
        n = steps.assign(cx=qx, cy=qy).groupby(["cx", "cy", "arm"]).size()
        cov[f"grid_{nb}x{nb}"] = {"cells_with_both_arms": int(
            (n.unstack("arm").notna().sum(axis=1) == 2).sum()),
            "median_steps_per_cell_arm": float(n.median()),
            "min": int(n.min()), "cells": int(nb * nb)}

    md = ["# A randomised trial as movement through clinical state space", "",
          "PRIME (NCT00364013): panitumumab + FOLFOX versus FOLFOX alone, first-line "
          "metastatic colorectal cancer. Every quantity below is observed — measured lesion "
          "diameters and sites, differences between consecutive measured visits, arms as "
          "randomised. The synthetic laboratory supplies only the coordinate system, frozen "
          "before this trial was opened.", "",
          "## Coverage", "",
          f"- {cov['patients_with_lesions']:,} patients with central-review lesion "
          f"measurements, {cov['assessments']:,} assessments, "
          f"{cov['transitions']:,} transitions between consecutive visits",
          f"- median {cov['median_assessments_per_patient']:.0f} assessments per patient, "
          f"{cov['median_days_between']:.0f} days apart",
          "- transitions by arm: " + ", ".join(f"{k} {n:,}" for k, n in cov["by_arm"].items()),
          "- transitions by KRAS: " + ", ".join(f"{k} {n:,}"
                                                for k, n in cov["by_kras"].items()), "",
          "## How much of the frozen map a trial reaches", "",
          "| axis | observable loading mass |", "|---|---:|"]
    md += [f"| {k} | {100 * val:.0f}% |" for k, val in reach.items()]
    md += ["", "A trial supplies the tumour coordinates and none of the host ones in this "
           "table, because host state lives in laboratory and performance measures handled "
           "separately. Positions here are therefore comparable to other populations passed "
           "through the same restricted functional, and not to the full twenty-feature map.",
           "", "## Can a field be drawn?", "",
           "| grid | cells | cells with both arms | median steps per cell-arm | min |",
           "|---|---:|---:|---:|---:|"]
    for nb in (3, 4, 5):
        k = cov[f"grid_{nb}x{nb}"]
        md.append(f"| {nb}x{nb} | {k['cells']} | {k['cells_with_both_arms']} | "
                  f"{k['median_steps_per_cell_arm']:.0f} | {k['min']} |")
    md += ["", "## Scope", "",
           "Randomisation licenses a causal reading of the difference between arm-specific "
           "fields at the population level. It does not license an individual counterfactual: "
           "each patient is observed in one arm only, and no patient's untreated future "
           "exists in this data. Two radiologists read every scan; one is used throughout so "
           "that a change between visits is a change in the patient rather than in the "
           "reader. The millimetre-to-ordinal binning is declared in the source and is not "
           "fitted to these data."]
    (OUT / "readout.md").write_text("\n".join(md) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"trial": "NCT00364013", "reader": READER, "burden_mm_bins": BURDEN_MM,
         "coordinates": COORDS, "coverage": cov}, indent=2) + "\n")
    steps.to_csv(OUT / "prime_state_steps.csv.gz", index=False)
    # Every assessment, first to last, in the same restricted projection. Patient paths are
    # drawn from this table; the steps table (which has no row for each patient's first
    # assessment, because a step needs a preceding one) is for movement only.
    v.to_csv(OUT / "prime_state_visits.csv.gz", index=False)
    loadings = pd.DataFrame({"coordinate": COORDS, "pc1_loading": c1, "pc2_loading": c2})
    corr = {f"pc{i}": {c: round(float(np.corrcoef(v[c], v[f"pc{i}"])[0, 1]), 3)
                       for c in COORDS if v[c].std() > 0} for i in (1, 2)}
    loadings.to_csv(OUT / "restricted_loadings.csv", index=False)
    (OUT / "restricted_axis_correlations.json").write_text(json.dumps(corr, indent=2) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
