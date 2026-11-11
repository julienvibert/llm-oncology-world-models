#!/usr/bin/env python3
"""Clinical phenotype of the six-model atlas by OncoTree tissue and model (no generation).

Rebuilds, on the frozen six-model atlas (`build_paper_state_map.REFERENCE_RUNS`, 4,434 accepted
trajectories), the descriptive tissue-by-model table (Extended Data Fig. 1a). Baseline
characteristics and the generated course are kept separate:

Baseline (as generated at diagnosis)
- median age at diagnosis, years;
- metastatic or systemic stage, % (`stage` in {metastatic, systemic}; systemic is the
  haematological category);
- median number of disease sites at the first snapshot (`sites.is_baseline`);
- any organ impairment, % (any of renal, hepatic, cardiac, marrow reserve or pulmonary function
  recorded other than `normal`).

Course (generated after diagnosis)
- median survival from diagnosis, months (`os_days` / 30.4375; every accepted trajectory ends
  in death, none is censored);
- median number of systemic treatment lines (`n_treatment_lines`);
- any grade 3 or higher adverse event, % (`n_grade3plus_aes` > 0);
- any new disease site during the course, % (`sites.is_new_at_snapshot` after the first
  snapshot; the flag is also true for the sites present at the first snapshot).

Also verifies the stage-restricted survival comparison of `build_atlas_model_dissection.py`
from the endpoint table: spread of median survival across models in the eight tumour families
of Fig. 1d (CODE2FAM), all stages and metastatic only, with cell counts, a sensitivity analysis
excluding cells with fewer than 5 patients, and bootstrap intervals of the spread.

Run: uv run python scripts/explore/build_atlas_phenotype_six_model.py
Out: outputs/exploratory_atlas_phenotype_six_model/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper_state_map as smap  # noqa: E402
from build_atlas_model_dissection import MODEL_LABEL  # noqa: E402
from build_field_trellis import CODE2FAM  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_atlas_phenotype_six_model"
DAYS_PER_MONTH = 30.4375
ORGANS = ["organ_renal", "organ_hepatic", "organ_cardiac", "organ_marrow_reserve", "organ_pulmonary"]
SEED = 20261014
N_BOOT = 2000
MIN_CELL = 5


def code_to_tissue() -> dict[str, str]:
    roster = yaml.safe_load((ROOT / "data" / "oncotree" / "roster_v1.yaml").read_text())
    return {n["code"]: n["tissue"] for n in roster["nodes"] if n.get("tissue")}


def trajectory_table() -> pd.DataFrame:
    """One row per accepted trajectory with the eight readouts."""
    t = smap.load_reference_tables(smap.REFERENCE_RUNS)
    b = t["baselines"].groupby("_gid").first()
    e = t["endpoints"].set_index("_gid")
    s = t["sites"]
    # the baseline snapshot is the trajectory's first snapshot (snapshots table), not the first
    # snapshot that happens to contain a site row
    base_idx = t["snapshots"].groupby("_gid")["snapshot_index"].min()
    at_base = s["snapshot_index"].eq(s["_gid"].map(base_idx))
    # `site_id` indexes clone-site entries; an anatomical site is a `site_signature`
    base_sites = (s[s["is_baseline"].astype(bool) & at_base]
                  .groupby("_gid")["site_signature"].nunique())
    # `is_new_at_snapshot` is also true for the sites present at baseline, so a new site is one
    # flagged new after the baseline snapshot
    new_site = (s[s["is_new_at_snapshot"].astype(bool) & s["snapshot_index"].gt(s["_gid"].map(base_idx))]
                .groupby("_gid").size().gt(0))
    d = pd.DataFrame(index=e.index)
    d["model"] = e["model_short"].map(MODEL_LABEL)
    d["code"] = e["tumor_oncotree_code"].astype(str)
    d["tissue"] = d["code"].map(code_to_tissue())
    d["family"] = d["code"].map(CODE2FAM)
    d["stage"] = b["stage"].astype(str)
    d["age"] = pd.to_numeric(b["age_at_diagnosis"], errors="coerce")
    d["metastatic_or_systemic"] = d["stage"].isin(["metastatic", "systemic"])
    d["baseline_sites"] = base_sites.reindex(d.index).fillna(0).astype(int)
    d["organ_impairment"] = b[ORGANS].astype(str).ne("normal").any(axis=1)
    d["survival_months"] = e["os_days"].astype(float) / DAYS_PER_MONTH
    d["lines"] = e["n_treatment_lines"].astype(float)
    d["grade3plus_ae"] = e["n_grade3plus_aes"].astype(float).gt(0)
    d["new_site"] = new_site.reindex(d.index).fillna(False).astype(bool)
    d["died"] = e["outcome_type"].astype(str).eq("death")
    assert d["tissue"].notna().all(), "code without tissue"
    return d


def by_tissue(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["tissue", "model"])
    out = pd.DataFrame({
        "n": g.size(),
        "n_codes": g["code"].nunique(),
        "median_age": g["age"].median(),
        "pct_metastatic_or_systemic": 100 * g["metastatic_or_systemic"].mean(),
        "median_baseline_sites": g["baseline_sites"].median(),
        "pct_organ_impairment": 100 * g["organ_impairment"].mean(),
        "median_survival_months": g["survival_months"].median(),
        "median_lines": g["lines"].median(),
        "pct_grade3plus_ae": 100 * g["grade3plus_ae"].mean(),
        "pct_new_site": 100 * g["new_site"].mean(),
    }).reset_index()
    return out


def stage_check(d: pd.DataFrame) -> dict:
    """Spread of median survival across models per family, all stages vs metastatic only."""
    f = d.dropna(subset=["family"])
    rng = np.random.default_rng(SEED)
    res: dict = {}
    for label, x in [("all_stages", f), ("metastatic_only", f[f["stage"].eq("metastatic")])]:
        med = x.groupby(["family", "model"])["survival_months"].median().unstack()
        n = x.groupby(["family", "model"]).size().unstack().fillna(0).astype(int)
        kept = med.where(n >= MIN_CELL)
        spread = med.max(axis=1) - med.min(axis=1)
        spread_kept = kept.max(axis=1) - kept.min(axis=1)
        boot = {}
        for fam, gx in x.groupby("family"):
            vals = []
            groups = [gm["survival_months"].to_numpy() for _m, gm in gx.groupby("model")
                      if len(gm) >= MIN_CELL]
            for _ in range(N_BOOT):
                meds = [np.median(rng.choice(v, len(v), replace=True)) for v in groups]
                vals.append(max(meds) - min(meds))
            boot[fam] = [round(float(q), 1) for q in np.quantile(vals, [0.025, 0.975])]
        res[label] = {
            "n_patients": int(len(x)),
            "n_by_family_model": n.to_dict(),
            "min_cell_n": int(n.to_numpy().min()),
            "cells_below_5": int((n < MIN_CELL).to_numpy().sum()),
            "median_survival_by_family_model": med.round(2).to_dict(),
            "spread_months": spread.round(2).to_dict(),
            "median_spread_months": round(float(spread.median()), 2),
            "spread_months_cells_n_ge_5": spread_kept.round(2).to_dict(),
            "median_spread_months_cells_n_ge_5": round(float(spread_kept.median()), 2),
            "spread_95ci_bootstrap_cells_n_ge_5": boot,
            "longest_model_by_family": med.idxmax(axis=1).to_dict(),
            "longest_model_by_family_cells_n_ge_5": kept.idxmax(axis=1).to_dict(),
        }
    res["pct_metastatic_by_model_8_families"] = (
        100 * f.assign(m=f["stage"].eq("metastatic")).groupby("model")["m"].mean()).round(1).to_dict()
    res["censored_trajectories"] = int((~d["died"]).sum())
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = trajectory_table()
    tab = by_tissue(d)
    tab.to_csv(OUT / "phenotype_by_tissue_model.csv", index=False)
    chk = stage_check(d)
    readout = {
        "n_trajectories": int(len(d)), "n_tissues": int(d["tissue"].nunique()),
        "n_codes": int(d["code"].nunique()), "models": sorted(d["model"].unique()),
        "runs": list(smap.REFERENCE_RUNS), "stage_restricted_survival": chk,
    }
    (OUT / "readout.json").write_text(json.dumps(readout, indent=1, sort_keys=True) + "\n")
    a, m = chk["all_stages"], chk["metastatic_only"]
    lines = [
        "# Six-model atlas phenotype and stage-restricted survival",
        "",
        f"{len(d):,} accepted trajectories, {d['code'].nunique()} codes, "
        f"{d['tissue'].nunique()} OncoTree tissues, six models; "
        f"censored trajectories: {chk['censored_trajectories']}.",
        "",
        "## Stage-restricted survival (eight families of Fig. 1d)",
        "",
        f"- All stages: {a['n_patients']} patients; median spread across models "
        f"{a['median_spread_months']} months; minimum cell n {a['min_cell_n']}.",
        f"- Metastatic only: {m['n_patients']} patients; median spread "
        f"{m['median_spread_months']} months; minimum cell n {m['min_cell_n']}; "
        f"{m['cells_below_5']} cells below 5; excluding them, median spread "
        f"{m['median_spread_months_cells_n_ge_5']} months.",
        f"- Breast spread: all stages {a['spread_months']['Breast']}, metastatic "
        f"{m['spread_months']['Breast']}; sensitivity excluding cells with fewer than 5 patients "
        f"(Gemma 4 has 2 metastatic breast patients): 95% bootstrap interval of the spread "
        f"{m['spread_95ci_bootstrap_cells_n_ge_5']['Breast']}.",
        "- Longest median survival, metastatic only: "
        + ", ".join(f"{k} {v}" for k, v in m["longest_model_by_family"].items()) + ".",
        f"- Metastatic share by model (eight families): {chk['pct_metastatic_by_model_8_families']}.",
        "",
        "Descriptive: restricting to metastatic disease removes one difference in case mix; "
        "metastatic patients can still differ in subtype, biomarkers, burden, condition and "
        "treatment. Generated data, no real-patient reference.",
    ]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
