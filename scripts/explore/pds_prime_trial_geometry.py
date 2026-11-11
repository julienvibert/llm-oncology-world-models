#!/usr/bin/env python3
"""Observed trial geometry for PDS PRIME (NCT00364013).

This is an exploratory, read-only analysis of the Project Data Sphere release
for panitumumab + FOLFOX versus FOLFOX in first-line metastatic colorectal
cancer.  It asks whether the paper's phase-space grammar can be applied to a
real randomized trial:

* conventional PFS/OS readout by assigned arm;
* observed patient movement through a trial-derived state space;
* finite-horizon displacement fields by assigned arm;
* a differential field for panitumumab + FOLFOX minus FOLFOX alone;
* subgroup and local-neighborhood treatment-effect localization.

The outputs are not individual counterfactuals.  Each patient contributes one
observed randomized arm; local effects are arm contrasts among nearby baseline
states.

Run:
  uv run python scripts/explore/pds_prime_trial_geometry.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
import os
from pathlib import Path
import subprocess
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CANDIDATES = (
    Path.home()
    / "PDS"
    / "NCT00364013"
    / "AllProvidedFiles_309 (1)"
    / "PDS_DSA_20050203",
    Path.home() / "Downloads" / "AllProvidedFiles_309" / "PDS_DSA_20050203",
)
OUT = ROOT / "outputs" / "exploratory_pds_prime_trial_geometry"
MONTH_DAYS = 30.44
HORIZON_DAYS = 112.0
KRAS_EXON2_COLUMN = "BMMTR1"
KRAS_EXON2_LABEL_COLUMN = "BMMTNM1"
# KRAS and NRAS exons 2-4 (extended RAS); BMMTR7 (BRAF exon 15) is deliberately excluded.
RAS_COLUMNS = {
    "BMMTR1": "KRAS exon 2 (c12/13)", "BMMTR2": "KRAS exon 3 (c61)", "BMMTR3": "KRAS exon 4 (c117/146)",
    "BMMTR4": "NRAS exon 2 (c12/13)", "BMMTR5": "NRAS exon 3 (c61)", "BMMTR6": "NRAS exon 4 (c117/146)",
    "BMMTR15": "KRAS exon 3 (c59/c61)", "BMMTR16": "NRAS exon 3 (c59/c61)",
}
SEED = 20260804
BOOT = 600
LOCAL_BOOT = 200
# Restricted-mean subgroup differences (Fig. 6a) and the N8 time-in-state parts (Fig. 6c) share ONE
# bootstrap: 2,000 resamples of patients within arm, seed 20261028, months = days / 30.4375 (the
# paper's convention). Other analyses in this script keep SEED, BOOT and MONTH_DAYS.
RMST_SEED = 20261028
RMST_BOOT = 2000
RMST_MONTH_DAYS = 30.4375

ARM_CONTROL = "FOLFOX alone"
ARM_EXPERIMENTAL = "Panitumumab + FOLFOX"
ARM_LABEL = {ARM_CONTROL: "FOLFOX", ARM_EXPERIMENTAL: "Panitumumab + FOLFOX"}
ARM_COLOR = {ARM_CONTROL: "#d17827", ARM_EXPERIMENTAL: "#087f73"}

INK = "#172033"
MUTED = "#667085"
GRID = "#d9dee7"
PAPER = "#fbfaf7"
EXIT = "#a51c30"
BENEFIT = "#087f73"
HARM = "#c45a20"
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "benefit_harm", (HARM, "#f5cfaa", "#fffdf8", "#9bd6cc", BENEFIT)
)
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ("#fffdf8", "#e5eff2", "#90bdca", "#366d8d", "#163d5c")
)

SITE_GROUP = {
    "Liver": "liver",
    "Pulmonary": "lung",
    "Lymph Nodes": "lymph_node",
}

FROZEN_STATE_DIR = ROOT / "outputs" / "paper_state_map"
FROZEN_TUMOUR_FEATURES = [
    "tumour_hepatic",
    "tumour_thoracic",
    "tumour_nodal",
    "tumour_osseous",
    "tumour_marrow",
    "tumour_cns",
    "tumour_peritoneal",
    "tumour_soft_tissue_cutaneous",
    "tumour_other_anatomical",
]
FROZEN_STATE_FEATURES = [
    *FROZEN_TUMOUR_FEATURES,
    "tumour_extent",
    "host_ecog",
    "host_symptoms",
    "host_renal",
    "host_hepatic",
    "host_cardiac",
    "host_marrow_reserve",
    "host_pulmonary",
    "host_relative_weight_loss",
    "host_active_ae_max_grade",
    "host_active_ae_count",
]
FROZEN_SITE_MAP = {
    "Liver": "tumour_hepatic",
    "Pulmonary": "tumour_thoracic",
    "Chest": "tumour_thoracic",
    "Lymph Nodes": "tumour_nodal",
    "Bone": "tumour_osseous",
    "Musculoskeletal": "tumour_osseous",
    "Central Nervous System": "tumour_cns",
    "Abdomen": "tumour_peritoneal",
    "Gastrointestinal": "tumour_peritoneal",
    "Pelvic Site": "tumour_peritoneal",
    "Retroperitoneum": "tumour_peritoneal",
    "Skin": "tumour_soft_tissue_cutaneous",
    "HEENT/Neck": "tumour_soft_tissue_cutaneous",
}

ECOG_MAP = {
    "Fully active": 0.0,
    "Symptoms but ambulatory": 1.0,
    "In bed less than 50% of the time": 2.0,
}

RESPONSE_RANK = {
    "Complete response": -2.0,
    "Partial response": -1.0,
    "Stable disease": 0.0,
    "Progressive disease": 2.0,
    "Unable to evaluate": np.nan,
    "Not applicable": np.nan,
}

STATE_FEATURES = [
    "log_sld",
    "sld_ratio",
    "liver_log",
    "lung_log",
    "ln_log",
    "other_log",
    "new_lesion_any",
    "albumin_low",
    "hgb_low",
    "ldh_log",
    "alp_log",
    "creatinine_log",
    "weight_loss_frac",
    "active_ae_max_grade",
]

SUBGROUPS = [
    ("KRAS exon 2", "kras_exon2", ("Wild-type", "Mutant")),
    ("ECOG", "ecog_group", ("0", "1-2")),
    ("Baseline burden", "burden_group", ("low", "high")),
    ("Liver metastases", "liver_group", ("no", "yes")),
    ("Metastatic sites", "met_sites_group", ("1-2", "3+")),
    ("Primary site", "primary_group", ("Colon", "Rectal")),
    ("Age", "age_group", ("<65", "65+")),
]


@dataclass(frozen=True)
class TrialTables:
    adsl: pd.DataFrame
    adls: pd.DataFrame
    adrsp: pd.DataFrame
    adlb: pd.DataFrame
    adpm: pd.DataFrame
    adae: pd.DataFrame
    biomark: pd.DataFrame


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_data_dir() -> Path:
    override = os.environ.get("PDS_PRIME_ROOT")
    candidates = (Path(override),) if override else DEFAULT_DATA_CANDIDATES
    for candidate in candidates:
        if candidate and (candidate / "adsl_pds2019.sas7bdat").exists():
            return candidate
    joined = "\n  ".join(str(x) for x in candidates if x)
    raise FileNotFoundError(f"Could not find PDS PRIME SAS files. Tried:\n  {joined}")


def _read_sas(base: Path, name: str) -> pd.DataFrame:
    return pd.read_sas(base / f"{name}_pds2019.sas7bdat", format="sas7bdat", encoding="utf-8")


def load_tables(base: Path) -> TrialTables:
    return TrialTables(
        adsl=_read_sas(base, "adsl"),
        adls=_read_sas(base, "adls"),
        adrsp=_read_sas(base, "adrsp"),
        adlb=_read_sas(base, "adlb"),
        adpm=_read_sas(base, "adpm"),
        adae=_read_sas(base, "adae"),
        biomark=_read_sas(base, "biomark"),
    )


def derive_subjects(t: TrialTables) -> pd.DataFrame:
    adsl = t.adsl.copy()
    adsl["arm"] = adsl["TRT"].replace(ARM_LABEL)
    adsl["arm_raw"] = adsl["TRT"]
    adsl["pfs_months"] = pd.to_numeric(adsl["PFSDYCR"], errors="coerce") / MONTH_DAYS
    adsl["pfs_event"] = pd.to_numeric(adsl["PFSCR"], errors="coerce").fillna(0).astype(int)
    adsl["os_months"] = pd.to_numeric(adsl["DTHDY"], errors="coerce") / MONTH_DAYS
    adsl["os_event"] = pd.to_numeric(adsl["DTH"], errors="coerce").fillna(0).astype(int)
    adsl["ecog"] = adsl["B_ECOG"].map(ECOG_MAP)
    adsl["age"] = pd.to_numeric(adsl["AGE"], errors="coerce")
    adsl["baseline_weight"] = pd.to_numeric(adsl["B_WEIGHT"], errors="coerce")
    adsl["baseline_met_sites"] = pd.to_numeric(adsl["B_METANM"], errors="coerce")
    adsl["liver_met"] = adsl["LIVERMET"].eq("Y")
    adsl["primary"] = adsl["DIAGTYPE"].fillna("Unknown")

    biom = t.biomark.set_index("SUBJID")
    adsl = adsl.set_index("SUBJID")
    # KRAS exon 2 (codons 12/13) is BMMTR1 (label column BMMTNM1); "Failure" and missing are
    # unknown, never mutant. The extended-RAS grouping uses the RAS columns only (BMMTR7 is BRAF
    # exon 15, c600).
    assert set(biom[KRAS_EXON2_LABEL_COLUMN].dropna().unique()) == {"KRAS exon 2 (c12/13)"}
    adsl["kras_exon2"] = biom[KRAS_EXON2_COLUMN].reindex(adsl.index).fillna("Unknown")
    for col, name in RAS_COLUMNS.items():
        assert set(biom[col.replace("BMMTR", "BMMTNM")].dropna().unique()) == {name}, col
    marker = biom[list(RAS_COLUMNS)].reindex(adsl.index)
    adsl["any_ras_mut"] = marker.eq("Mutant").any(axis=1)
    tested = marker.notna() & ~marker.eq("Failure")
    adsl["all_tested_wt"] = tested.any(axis=1) & marker.where(tested).eq("Wild-type").all(axis=1)
    adsl["ras_extended"] = np.select(
        [adsl["any_ras_mut"], adsl["all_tested_wt"]],
        ["Mutant", "Wild-type"],
        default="Unknown",
    )
    adsl["ecog_group"] = np.where(adsl["ecog"].le(0), "0", "1-2")
    adsl["liver_group"] = np.where(adsl["liver_met"], "yes", "no")
    adsl["met_sites_group"] = np.where(adsl["baseline_met_sites"].le(2), "1-2", "3+")
    adsl["primary_group"] = adsl["primary"].where(adsl["primary"].isin(["Colon", "Rectal"]), "Other")
    adsl["age_group"] = np.where(adsl["age"].lt(65), "<65", "65+")
    return adsl.reset_index()


def _site_group(site: Any) -> str:
    return SITE_GROUP.get(str(site), "other")


def _frozen_site_group(site: Any) -> str:
    return FROZEN_SITE_MAP.get(str(site), "tumour_other_anatomical")


def derive_lesion_snapshots(t: TrialTables, subjects: pd.DataFrame) -> pd.DataFrame:
    adls = t.adls.copy()
    adls["VISITDY"] = pd.to_numeric(adls["VISITDY"], errors="coerce")
    adls["LSLD"] = pd.to_numeric(adls["LSLD"], errors="coerce")
    adls["LSSLD"] = pd.to_numeric(adls["LSSLD"], errors="coerce")
    central = adls["LSREADER"].isin(["Radiologist 1", "Radiologist 2"])
    target = adls[central & adls["LSCAT"].eq("Target lesion")].dropna(subset=["VISITDY"])
    target["site_group"] = target["LSSITE"].map(_site_group)
    target["frozen_site_group"] = target["LSSITE"].map(_frozen_site_group)

    totals = (
        target.groupby(["SUBJID", "VISITDY", "LSREADER"], as_index=False)
        .agg(target_sld=("LSSLD", "max"), target_count=("LSLD", "count"))
    )
    site = (
        target.groupby(["SUBJID", "VISITDY", "LSREADER", "site_group"])["LSLD"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for col in ("liver", "lung", "lymph_node", "other"):
        if col not in site:
            site[col] = 0.0
    frozen_site = (
        target.groupby(["SUBJID", "VISITDY", "LSREADER", "frozen_site_group"])["LSLD"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for col in FROZEN_TUMOUR_FEATURES:
        if col not in frozen_site:
            frozen_site[col] = 0.0
    reader = totals.merge(site, on=["SUBJID", "VISITDY", "LSREADER"], how="left")
    reader = reader.merge(
        frozen_site[["SUBJID", "VISITDY", "LSREADER", *FROZEN_TUMOUR_FEATURES]],
        on=["SUBJID", "VISITDY", "LSREADER"],
        how="left",
    )
    snap = (
        reader.groupby(["SUBJID", "VISITDY"], as_index=False)
        [
            [
                "target_sld",
                "target_count",
                "liver",
                "lung",
                "lymph_node",
                "other",
                *FROZEN_TUMOUR_FEATURES,
            ]
        ]
        .median()
    )
    new_any = (
        adls.dropna(subset=["VISITDY"])
        .assign(new_lesion_any=lambda x: x["LSNEW"].eq("Y") | x["LSNEWANY"].eq("Y"))
        .groupby(["SUBJID", "VISITDY"])["new_lesion_any"]
        .max()
        .reset_index()
    )
    snap = snap.merge(new_any, on=["SUBJID", "VISITDY"], how="left")
    snap["new_lesion_any"] = snap["new_lesion_any"].fillna(False).astype(float)

    resp = t.adrsp.copy()
    resp["VISITDY"] = pd.to_numeric(resp["VISITDY"], errors="coerce")
    resp = resp.dropna(subset=["VISITDY"])
    resp["reader_rank"] = resp["RSREADER"].map({"Oncologist": 0, "Radiologist 1": 1, "Radiologist 2": 2}).fillna(9)
    resp = (
        resp.sort_values(["SUBJID", "VISITDY", "reader_rank"])
        .drop_duplicates(["SUBJID", "VISITDY"], keep="first")
        [["SUBJID", "VISITDY", "RSRESP", "RSTRESP", "RSNTRESP", "RSNEW"]]
    )
    snap = snap.merge(resp, on=["SUBJID", "VISITDY"], how="left")
    snap["response_rank"] = snap["RSRESP"].map(RESPONSE_RANK)

    baseline_sld = (
        snap.sort_values(["SUBJID", "VISITDY"])
        .groupby("SUBJID", as_index=False)
        .first()[["SUBJID", "target_sld"]]
        .rename(columns={"target_sld": "baseline_sld"})
    )
    snap = snap.merge(baseline_sld, on="SUBJID", how="left")
    snap["sld_ratio_raw"] = snap["target_sld"] / snap["baseline_sld"].replace(0, np.nan)

    subjects_small = subjects[
        [
            "SUBJID",
            "arm",
            "arm_raw",
            "kras_exon2",
            "ras_extended",
            "pfs_months",
            "pfs_event",
            "os_months",
            "os_event",
            "ecog",
            "baseline_weight",
            "age",
            "baseline_met_sites",
            "liver_met",
            "primary",
            "ecog_group",
            "liver_group",
            "met_sites_group",
            "primary_group",
            "age_group",
        ]
    ]
    snap = snap.merge(subjects_small, on="SUBJID", how="left")
    return snap[snap["target_sld"].notna()].copy()


def _merge_previous(
    snapshots: pd.DataFrame,
    obs: pd.DataFrame,
    cols: list[str],
    *,
    tolerance_days: float,
) -> pd.DataFrame:
    pieces = []
    obs = obs[["SUBJID", "VISITDY", *cols]].dropna(subset=["VISITDY"]).copy()
    obs = obs.sort_values(["SUBJID", "VISITDY"])
    for sid, left in snapshots.groupby("SUBJID", sort=False):
        right = obs[obs["SUBJID"].eq(sid)].sort_values("VISITDY")
        if right.empty:
            tmp = left.copy()
            for col in cols:
                tmp[col] = np.nan
            pieces.append(tmp)
            continue
        tmp = pd.merge_asof(
            left.sort_values("VISITDY"),
            right.drop(columns=["SUBJID"]),
            on="VISITDY",
            direction="backward",
            tolerance=tolerance_days,
        )
        pieces.append(tmp)
    return pd.concat(pieces, ignore_index=True)


def add_host_state(t: TrialTables, snapshots: pd.DataFrame) -> pd.DataFrame:
    lab = t.adlb.copy()
    lab["VISITDY"] = pd.to_numeric(lab["VISITDY"], errors="coerce")
    lab["LBSTRESN"] = pd.to_numeric(lab["LBSTRESN"], errors="coerce")
    lab = (
        lab.pivot_table(
            index=["SUBJID", "VISITDY"],
            columns="LBTEST",
            values="LBSTRESN",
            aggfunc="median",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    keep_labs = [
        "Albumin",
        "Hemoglobin",
        "Platelets",
        "White Blood Cells",
        "Lactate Dehydrogenase",
        "Alkaline Phosphatase",
        "Creatinine",
    ]
    snapshots = _merge_previous(snapshots, lab, keep_labs, tolerance_days=35)

    weight = t.adpm.copy()
    weight["VISITDY"] = pd.to_numeric(weight["VISITDY"], errors="coerce")
    weight["WEIGHT"] = pd.to_numeric(weight["WEIGHT"], errors="coerce")
    snapshots = _merge_previous(snapshots, weight[["SUBJID", "VISITDY", "WEIGHT"]], ["WEIGHT"], tolerance_days=35)

    ae = t.adae.copy()
    ae["AESTDYI"] = pd.to_numeric(ae["AESTDYI"], errors="coerce")
    ae["AEENDYI"] = pd.to_numeric(ae["AEENDYI"], errors="coerce")
    ae["AESEVCD"] = pd.to_numeric(ae["AESEVCD"], errors="coerce")
    rows = []
    for sid, g in snapshots.groupby("SUBJID", sort=False):
        subj_ae = ae[ae["SUBJID"].eq(sid)]
        for _, row in g.iterrows():
            day = float(row["VISITDY"])
            active = subj_ae[
                subj_ae["AESTDYI"].le(day)
                & (subj_ae["AEENDYI"].fillna(day).ge(day))
            ]
            rows.append(
                {
                    "SUBJID": sid,
                    "VISITDY": row["VISITDY"],
                    "active_ae_max_grade": float(active["AESEVCD"].max()) if not active.empty else 0.0,
                    "active_ae_count": int(active["AESEVCD"].notna().sum()) if not active.empty else 0,
                    "active_grade3_count": int(active["AESEVCD"].ge(3).sum()) if not active.empty else 0,
                }
            )
    ae_state = pd.DataFrame(rows)
    return snapshots.merge(ae_state, on=["SUBJID", "VISITDY"], how="left")


def fit_state_space(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = snapshots.copy()
    df["log_sld"] = np.log1p(df["target_sld"])
    df["sld_ratio"] = df["sld_ratio_raw"].clip(0.0, 4.0)
    df["liver_log"] = np.log1p(df["liver"])
    df["lung_log"] = np.log1p(df["lung"])
    df["ln_log"] = np.log1p(df["lymph_node"])
    df["other_log"] = np.log1p(df["other"])
    df["albumin_low"] = -pd.to_numeric(df["Albumin"], errors="coerce")
    df["hgb_low"] = -pd.to_numeric(df["Hemoglobin"], errors="coerce")
    df["ldh_log"] = np.log1p(pd.to_numeric(df["Lactate Dehydrogenase"], errors="coerce"))
    df["alp_log"] = np.log1p(pd.to_numeric(df["Alkaline Phosphatase"], errors="coerce"))
    df["creatinine_log"] = np.log1p(pd.to_numeric(df["Creatinine"], errors="coerce"))
    df["weight_loss_frac"] = (
        (pd.to_numeric(df["baseline_weight"], errors="coerce") - pd.to_numeric(df["WEIGHT"], errors="coerce"))
        / pd.to_numeric(df["baseline_weight"], errors="coerce").replace(0, np.nan)
    )
    df["weight_loss_frac"] = df["weight_loss_frac"].clip(-0.15, 0.5)

    features = df[STATE_FEATURES].copy()
    med = features.median(numeric_only=True)
    features = features.fillna(med)
    scaler = StandardScaler()
    z = scaler.fit_transform(features)
    pca = PCA(n_components=4, random_state=SEED)
    pcs = pca.fit_transform(z)
    if np.corrcoef(pcs[:, 0], df["target_sld"].fillna(df["target_sld"].median()))[0, 1] < 0:
        pcs[:, 0] *= -1
        pca.components_[0, :] *= -1
    host_index = (
        df[["albumin_low", "hgb_low", "ldh_log", "alp_log", "weight_loss_frac", "active_ae_max_grade"]]
        .fillna(med.reindex(["albumin_low", "hgb_low", "ldh_log", "alp_log", "weight_loss_frac", "active_ae_max_grade"]))
        .mean(axis=1)
    )
    if np.corrcoef(pcs[:, 1], host_index)[0, 1] < 0:
        pcs[:, 1] *= -1
        pca.components_[1, :] *= -1
    for i in range(4):
        df[f"pc{i+1}"] = pcs[:, i]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=STATE_FEATURES,
        columns=[f"PC{i+1}" for i in range(4)],
    )
    meta = {
        "state_features": STATE_FEATURES,
        "explained_variance": {f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)},
        "imputation_medians": {k: float(v) for k, v in med.items()},
        "pc1_top_loadings": loadings["PC1"].abs().sort_values(ascending=False).head(6).index.tolist(),
        "pc2_top_loadings": loadings["PC2"].abs().sort_values(ascending=False).head(6).index.tolist(),
    }
    return df, loadings.reset_index(names="feature"), meta


def _encode_mm_burden(values: pd.Series) -> pd.Series:
    """Approximate target-lesion millimetres as the frozen 0-5 burden ordinal."""
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    bins = [x.le(0), x.le(10), x.le(30), x.le(60), x.le(100)]
    encoded = np.select(bins, [0, 1, 2, 3, 4], default=5).astype(float) / 5.0
    return pd.Series(encoded, index=values.index)


def _encode_high(values: pd.Series, normal: float, severe: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return ((x - normal) / (severe - normal)).clip(lower=0, upper=1).fillna(0.0)


def _encode_low(values: pd.Series, normal: float, severe: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return ((normal - x) / (normal - severe)).clip(lower=0, upper=1).fillna(0.0)


def add_frozen_projection(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Project PDS states through the frozen paper-v1 clinical state map.

    PDS has exact serial target-lesion measurements, labs, weight and AEs, but
    not all channels in the synthetic state contract.  Unobserved symptoms,
    cardiac and pulmonary function are neutral-coded and declared in the
    metadata.  The original trial-specific PCA is retained as trial_pc*.
    """
    contract_path = FROZEN_STATE_DIR / "state_map_contract.json"
    contract = json.loads(contract_path.read_text())
    model = contract["primary_model"]
    features = list(model["feature_names"])
    if features != FROZEN_STATE_FEATURES:
        raise ValueError("Frozen state-map feature order changed unexpectedly")

    out = states.copy()
    for idx in range(1, 5):
        col = f"pc{idx}"
        if col in out:
            out[f"trial_pc{idx}"] = out[col]

    encoded = pd.DataFrame(index=out.index)
    for feature in FROZEN_TUMOUR_FEATURES:
        encoded[feature] = _encode_mm_burden(out[feature])
    encoded["tumour_extent"] = (
        out[FROZEN_TUMOUR_FEATURES].gt(0).sum(axis=1).clip(upper=5).astype(float) / 5.0
    )
    encoded["host_ecog"] = (pd.to_numeric(out["ecog"], errors="coerce") / 4.0).clip(0, 1).fillna(0.0)
    encoded["host_symptoms"] = 0.0
    encoded["host_renal"] = _encode_high(out["Creatinine"], normal=110, severe=300)
    hepatic_components = pd.concat(
        [
            _encode_low(out["Albumin"], normal=35, severe=20),
            _encode_high(out["Alkaline Phosphatase"], normal=120, severe=600),
            _encode_high(out["Lactate Dehydrogenase"], normal=250, severe=1000),
        ],
        axis=1,
    )
    encoded["host_hepatic"] = hepatic_components.max(axis=1)
    encoded["host_cardiac"] = 0.0
    marrow_components = pd.concat(
        [
            _encode_low(out["Hemoglobin"], normal=120, severe=80),
            _encode_low(out["Platelets"], normal=150, severe=50),
            _encode_low(out["White Blood Cells"], normal=4.0, severe=1.0),
        ],
        axis=1,
    )
    encoded["host_marrow_reserve"] = marrow_components.max(axis=1)
    encoded["host_pulmonary"] = 0.0
    loss_pct = 100.0 * pd.to_numeric(out["weight_loss_frac"], errors="coerce").fillna(0.0)
    encoded["host_relative_weight_loss"] = ((loss_pct + 20.0) / 50.0).clip(0, 1)
    encoded["host_active_ae_max_grade"] = (
        pd.to_numeric(out["active_ae_max_grade"], errors="coerce").fillna(0).clip(0, 5) / 5.0
    )
    encoded["host_active_ae_count"] = (
        pd.to_numeric(out["active_ae_count"], errors="coerce").fillna(0).clip(0, 5) / 5.0
    )

    if encoded[features].isna().any().any():
        missing = encoded.columns[encoded.isna().any()].tolist()
        raise ValueError(f"Frozen projection has missing encoded values: {missing}")
    multipliers = np.asarray(model["feature_multipliers"], dtype=float)
    mean = np.asarray(model["mean_after_feature_multipliers"], dtype=float)
    components = np.asarray(model["components_after_feature_multipliers"], dtype=float)
    matrix = encoded[features].to_numpy(float) * multipliers
    coords = (matrix - mean) @ components.T
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    for feature in features:
        out[f"frozen_{feature}"] = encoded[feature].to_numpy(float)

    loadings = pd.DataFrame(
        np.asarray(model["coefficients_in_encoded_units"], dtype=float).T,
        index=features,
        columns=["PC1", "PC2"],
    )
    meta = {
        "coordinate_system": "frozen_paper_state_map",
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": _sha256(contract_path),
        "explained_variance": {
            "PC1": float(model["explained_variance_ratio"][0]),
            "PC2": float(model["explained_variance_ratio"][1]),
        },
        "feature_mapping": {
            "tumour": "PDS target-lesion millimetres by site mapped to the frozen 0-5 burden ordinal.",
            "host": "ECOG, labs, weight and active AE grade/count mapped to frozen host channels.",
            "neutral_coded_unobserved": ["host_symptoms", "host_cardiac", "host_pulmonary"],
        },
        "trial_specific_pc_retained_as": ["trial_pc1", "trial_pc2", "trial_pc3", "trial_pc4"],
        "pc1_top_loadings": loadings["PC1"].abs().sort_values(ascending=False).head(6).index.tolist(),
        "pc2_top_loadings": loadings["PC2"].abs().sort_values(ascending=False).head(6).index.tolist(),
    }
    return out, loadings.reset_index(names="feature"), meta


def baseline_rows(states: pd.DataFrame) -> pd.DataFrame:
    base = states.sort_values(["SUBJID", "VISITDY"]).groupby("SUBJID", as_index=False).first()
    med = base["target_sld"].median()
    base["burden_group"] = np.where(base["target_sld"].le(med), "low", "high")
    return base


def km_rmst(time: np.ndarray, event: np.ndarray, tau: float) -> float:
    order = np.argsort(time)
    time = np.asarray(time, dtype=float)[order]
    event = np.asarray(event, dtype=int)[order]
    at_risk = len(time)
    surv = 1.0
    last = 0.0
    area = 0.0
    for tval in np.unique(time[time <= tau]):
        area += surv * max(0.0, min(tval, tau) - last)
        if tval >= tau:
            return area
        d = int(((time == tval) & (event == 1)).sum())
        c = int(((time == tval) & (event == 0)).sum())
        if at_risk > 0 and d > 0:
            surv *= 1.0 - d / at_risk
        at_risk -= d + c
        last = tval
    area += surv * max(0.0, tau - last)
    return area


def cox_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    from lifelines import CoxPHFitter

    rows = []
    for label, mask in [
        ("all", pd.Series(True, index=subjects.index)),
        ("KRAS wild-type", subjects["kras_exon2"].eq("Wild-type")),
        ("KRAS mutant", subjects["kras_exon2"].eq("Mutant")),
    ]:
        sub = subjects[mask].dropna(subset=["pfs_months", "pfs_event", "arm_raw"])
        if sub["arm_raw"].nunique() < 2 or sub["pfs_event"].sum() < 10:
            continue
        fit = sub[["pfs_months", "pfs_event", "arm_raw"]].copy()
        fit["pani"] = fit["arm_raw"].eq(ARM_EXPERIMENTAL).astype(int)
        cph = CoxPHFitter(penalizer=0.02)
        cph.fit(fit[["pfs_months", "pfs_event", "pani"]], "pfs_months", "pfs_event")
        ci = cph.confidence_intervals_.loc["pani"]
        rows.append(
            {
                "group": label,
                "n": int(len(sub)),
                "events": int(sub["pfs_event"].sum()),
                "hr_panitumumab_vs_control": float(np.exp(cph.params_["pani"])),
                "ci_low": float(np.exp(ci.iloc[0])),
                "ci_high": float(np.exp(ci.iloc[1])),
                "p": float(cph.summary.loc["pani", "p"]),
            }
        )
    return pd.DataFrame(rows)


def subgroup_rmst(subjects: pd.DataFrame, baseline: pd.DataFrame, *, tau_months: float = 24.0,
                  seed: int = RMST_SEED, n_boot: int = RMST_BOOT, month_days: float = RMST_MONTH_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = subjects.assign(pfs_months=subjects["pfs_months"] * MONTH_DAYS / month_days)  # days / month_days
    merged = subjects.merge(
        baseline[["SUBJID", "burden_group", "pc1", "pc2", "target_sld"]],
        on="SUBJID",
        how="left",
        suffixes=("", "_base"),
    )
    rows = []
    for display, col, levels in SUBGROUPS:
        for level in levels:
            sub = merged[merged[col].eq(level)].dropna(subset=["pfs_months", "pfs_event"])
            n_control = int(sub["arm_raw"].eq(ARM_CONTROL).sum())
            n_exp = int(sub["arm_raw"].eq(ARM_EXPERIMENTAL).sum())
            if n_control < 25 or n_exp < 25:
                continue
            control = sub[sub["arm_raw"].eq(ARM_CONTROL)]
            exp = sub[sub["arm_raw"].eq(ARM_EXPERIMENTAL)]
            diff = km_rmst(exp["pfs_months"].to_numpy(), exp["pfs_event"].to_numpy(), tau_months) - km_rmst(
                control["pfs_months"].to_numpy(), control["pfs_event"].to_numpy(), tau_months
            )
            boot = []
            for _ in range(n_boot):
                cb = control.iloc[rng.integers(0, len(control), len(control))]
                eb = exp.iloc[rng.integers(0, len(exp), len(exp))]
                boot.append(
                    km_rmst(eb["pfs_months"].to_numpy(), eb["pfs_event"].to_numpy(), tau_months)
                    - km_rmst(cb["pfs_months"].to_numpy(), cb["pfs_event"].to_numpy(), tau_months)
                )
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "subgroup": display,
                    "level": level,
                    "n_control": n_control,
                    "n_experimental": n_exp,
                    "rmst24_diff_months": float(diff),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    return pd.DataFrame(rows)


def horizon_displacements(states: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sid, course in states.sort_values(["SUBJID", "VISITDY"]).groupby("SUBJID", sort=False):
        base = baseline[baseline["SUBJID"].eq(sid)]
        if base.empty:
            continue
        b = base.iloc[0]
        target_day = float(b["VISITDY"]) + HORIZON_DAYS
        subj = course[course["VISITDY"].le(target_day)].sort_values("VISITDY")
        if subj.empty:
            continue
        end = subj.iloc[-1]
        pfs_day = float(b["pfs_months"] * MONTH_DAYS) if pd.notna(b["pfs_months"]) else np.inf
        exit_by_tau = bool(pd.notna(b["pfs_months"]) and pfs_day <= HORIZON_DAYS and b["pfs_event"] == 1)
        has_tau_state = float(end["VISITDY"]) >= float(b["VISITDY"]) + 0.5 * HORIZON_DAYS
        rows.append(
            {
                "SUBJID": sid,
                "arm_raw": b["arm_raw"],
                "arm": b["arm"],
                "kras_exon2": b["kras_exon2"],
                "base_pc1": float(b["pc1"]),
                "base_pc2": float(b["pc2"]),
                "end_pc1": float(end["pc1"]),
                "end_pc2": float(end["pc2"]),
                "dx": float(end["pc1"] - b["pc1"]),
                "dy": float(end["pc2"] - b["pc2"]),
                "exit_by_tau": exit_by_tau,
                "has_tau_state": has_tau_state,
                "pfs_months": float(b["pfs_months"]) if pd.notna(b["pfs_months"]) else np.nan,
                "pfs_event": int(b["pfs_event"]) if pd.notna(b["pfs_event"]) else 0,
                "burden_group": b["burden_group"],
                "ecog_group": b["ecog_group"],
                "target_sld": float(b["target_sld"]),
            }
        )
    disp = pd.DataFrame(rows)
    qs1 = np.quantile(baseline["pc1"].dropna(), [0.0, 1 / 3, 2 / 3, 1.0])
    qs2 = np.quantile(baseline["pc2"].dropna(), [0.0, 1 / 3, 2 / 3, 1.0])
    # Avoid exact-boundary duplicate bins.
    qs1[0] -= 1e-6
    qs1[-1] += 1e-6
    qs2[0] -= 1e-6
    qs2[-1] += 1e-6
    disp["bin_x"] = pd.cut(disp["base_pc1"], qs1, labels=False, include_lowest=True)
    disp["bin_y"] = pd.cut(disp["base_pc2"], qs2, labels=False, include_lowest=True)
    cell_rows = []
    for (bx, by, arm), g in disp.groupby(["bin_x", "bin_y", "arm_raw"], dropna=True):
        if len(g) < 12:
            continue
        survivors = g[~g["exit_by_tau"] & g["has_tau_state"]]
        cell_rows.append(
            {
                "bin_x": int(bx),
                "bin_y": int(by),
                "arm_raw": arm,
                "arm": ARM_LABEL.get(arm, arm),
                "n": int(len(g)),
                "n_displacement": int(len(survivors)),
                "x": float(g["base_pc1"].mean()),
                "y": float(g["base_pc2"].mean()),
                "dx": float(survivors["dx"].mean()) if len(survivors) else np.nan,
                "dy": float(survivors["dy"].mean()) if len(survivors) else np.nan,
                "exit_rate_tau": float(g["exit_by_tau"].mean()),
            }
        )
    cells = pd.DataFrame(cell_rows)
    if not cells.empty:
        piv = cells.pivot_table(index=["bin_x", "bin_y"], columns="arm_raw", values=["dx", "dy", "exit_rate_tau", "n"])
        diffs = []
        rng = np.random.default_rng(SEED + 17)
        for idx in piv.index:
            if (ARM_CONTROL in piv.loc[idx, "dx"].index) and (ARM_EXPERIMENTAL in piv.loc[idx, "dx"].index):
                dx = piv.loc[idx, ("dx", ARM_EXPERIMENTAL)] - piv.loc[idx, ("dx", ARM_CONTROL)]
                dy = piv.loc[idx, ("dy", ARM_EXPERIMENTAL)] - piv.loc[idx, ("dy", ARM_CONTROL)]
                control = disp[
                    disp["bin_x"].eq(idx[0])
                    & disp["bin_y"].eq(idx[1])
                    & disp["arm_raw"].eq(ARM_CONTROL)
                ]
                exp = disp[
                    disp["bin_x"].eq(idx[0])
                    & disp["bin_y"].eq(idx[1])
                    & disp["arm_raw"].eq(ARM_EXPERIMENTAL)
                ]
                er = float(control["exit_by_tau"].mean() - exp["exit_by_tau"].mean())
                boot = []
                for _ in range(BOOT):
                    cb = control.iloc[rng.integers(0, len(control), len(control))]
                    eb = exp.iloc[rng.integers(0, len(exp), len(exp))]
                    boot.append(float(cb["exit_by_tau"].mean() - eb["exit_by_tau"].mean()))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                n = min(piv.loc[idx, ("n", ARM_CONTROL)], piv.loc[idx, ("n", ARM_EXPERIMENTAL)])
                cg = cells[(cells["bin_x"].eq(idx[0])) & (cells["bin_y"].eq(idx[1]))]
                diffs.append(
                    {
                        "bin_x": int(idx[0]),
                        "bin_y": int(idx[1]),
                        "x": float(cg["x"].mean()),
                        "y": float(cg["y"].mean()),
                        "dx": float(dx),
                        "dy": float(dy),
                        "exit_rate_reduction": float(er),
                        "exit_ci_low": float(lo),
                        "exit_ci_high": float(hi),
                        "exit_excludes_zero": bool(lo > 0 or hi < 0),
                        "min_arm_n": int(n),
                    }
                )
        diff = pd.DataFrame(diffs)
    else:
        diff = pd.DataFrame()
    return disp, cells.merge(diff.add_prefix("diff_"), left_on=["bin_x", "bin_y"], right_on=["diff_bin_x", "diff_bin_y"], how="left") if not cells.empty and not diff.empty else cells


def local_effect_surface(
    subjects: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    kras_level: str,
    k: int = 45,
    horizon_months: float = 12.0,
) -> pd.DataFrame:
    data = subjects.merge(baseline[["SUBJID", "pc1", "pc2"]], on="SUBJID", how="inner")
    data = data[data["kras_exon2"].eq(kras_level)].dropna(subset=["pc1", "pc2", "pfs_months", "pfs_event"])
    if data["arm_raw"].nunique() < 2 or len(data) < 80:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in kras_level))
    control = data[data["arm_raw"].eq(ARM_CONTROL)].reset_index(drop=True)
    exp = data[data["arm_raw"].eq(ARM_EXPERIMENTAL)].reset_index(drop=True)
    coords_control = control[["pc1", "pc2"]].to_numpy(float)
    coords_exp = exp[["pc1", "pc2"]].to_numpy(float)
    grid_x = np.linspace(data["pc1"].quantile(0.05), data["pc1"].quantile(0.95), 30)
    grid_y = np.linspace(data["pc2"].quantile(0.05), data["pc2"].quantile(0.95), 30)
    rows = []
    for x in grid_x:
        for y in grid_y:
            point = np.array([[x, y]])
            dc = cdist(point, coords_control).ravel()
            de = cdist(point, coords_exp).ravel()
            kc = min(k, len(control))
            ke = min(k, len(exp))
            ci = np.argpartition(dc, kc - 1)[:kc]
            ei = np.argpartition(de, ke - 1)[:ke]
            csub = control.iloc[ci]
            esub = exp.iloc[ei]
            c_risk = ((csub["pfs_months"] <= horizon_months) & csub["pfs_event"].eq(1)).mean()
            e_risk = ((esub["pfs_months"] <= horizon_months) & esub["pfs_event"].eq(1)).mean()
            boot = []
            c_event = ((csub["pfs_months"] <= horizon_months) & csub["pfs_event"].eq(1)).to_numpy(float)
            e_event = ((esub["pfs_months"] <= horizon_months) & esub["pfs_event"].eq(1)).to_numpy(float)
            for _ in range(LOCAL_BOOT):
                cb = c_event[rng.integers(0, len(c_event), len(c_event))]
                eb = e_event[rng.integers(0, len(e_event), len(e_event))]
                boot.append(float(cb.mean() - eb.mean()))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "kras_exon2": kras_level,
                    "x": float(x),
                    "y": float(y),
                    "control_risk12": float(c_risk),
                    "experimental_risk12": float(e_risk),
                    "risk_reduction12": float(c_risk - e_risk),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "n_control": int(kc),
                    "n_experimental": int(ke),
                }
            )
    return pd.DataFrame(rows)


def _plot_km(ax: plt.Axes, subjects: pd.DataFrame, *, group_col: str = "arm_raw", title: str = "PFS by randomized arm") -> None:
    from lifelines import KaplanMeierFitter

    for key, g in subjects.dropna(subset=["pfs_months", "pfs_event"]).groupby(group_col):
        label = ARM_LABEL.get(key, str(key))
        color = ARM_COLOR.get(key, MUTED)
        km = KaplanMeierFitter(label=label)
        km.fit(g["pfs_months"], g["pfs_event"])
        km.plot(ax=ax, ci_show=False, color=color, linewidth=2.2)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Months from randomization")
    ax.set_ylabel("Progression-free survival")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)


def _setup_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#c8ced8")


def plot_main(
    subjects: pd.DataFrame,
    states: pd.DataFrame,
    baseline: pd.DataFrame,
    disp: pd.DataFrame,
    field: pd.DataFrame,
    cox: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2), constrained_layout=True)
    for ax in axes.flat:
        _setup_ax(ax)
    _plot_km(axes[0, 0], subjects)
    if not cox.empty:
        txt = "\n".join(
            f"{r.group}: HR {r.hr_panitumumab_vs_control:.2f} ({r.ci_low:.2f}-{r.ci_high:.2f})"
            for r in cox.itertuples()
        )
        axes[0, 0].text(0.03, 0.05, txt, transform=axes[0, 0].transAxes, fontsize=8, color=INK)

    ax = axes[0, 1]
    for arm_raw, g in baseline.groupby("arm_raw"):
        ax.scatter(
            g["pc1"],
            g["pc2"],
            s=16,
            alpha=0.55,
            color=ARM_COLOR[arm_raw],
            label=ARM_LABEL[arm_raw],
            linewidth=0,
        )
    ax.set_title("Baseline patients projected into frozen state map", loc="left", fontweight="bold")
    ax.set_xlabel(f"PC1 ({meta['explained_variance']['PC1']*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({meta['explained_variance']['PC2']*100:.1f}% var.)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    rng = np.random.default_rng(SEED)
    sampled = []
    for arm_raw, ids in states.groupby("arm_raw")["SUBJID"].unique().items():
        selected = rng.choice(ids, min(26, len(ids)), replace=False)
        sampled.extend(selected)
    for sid, g in states[states["SUBJID"].isin(sampled)].groupby("SUBJID"):
        arm_raw = g["arm_raw"].iloc[0]
        gg = g.sort_values("VISITDY")
        ax.plot(gg["pc1"], gg["pc2"], color=ARM_COLOR[arm_raw], alpha=0.20, linewidth=1.0)
        ax.scatter(gg["pc1"].iloc[0], gg["pc2"].iloc[0], s=10, color=ARM_COLOR[arm_raw], alpha=0.7)
    ax.set_title("Observed patient paths in frozen state map, sampled", loc="left", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    for j, arm_raw in enumerate((ARM_CONTROL, ARM_EXPERIMENTAL)):
        ax = axes[1, j]
        sub = field[field["arm_raw"].eq(arm_raw)].dropna(subset=["dx", "dy"])
        if not sub.empty:
            scat = ax.scatter(
                sub["x"],
                sub["y"],
                c=sub["exit_rate_tau"],
                cmap=RISK_CMAP,
                vmin=0,
                vmax=max(0.35, field["exit_rate_tau"].max()),
                s=52 + sub["n"] * 1.6,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.quiver(
                sub["x"],
                sub["y"],
                sub["dx"],
                sub["dy"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=ARM_COLOR[arm_raw],
                width=0.006,
                alpha=0.95,
            )
            fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.02, label="PFS event by 16 weeks")
        ax.set_title(f"16-week field: {ARM_LABEL[arm_raw]}", loc="left", fontweight="bold")
        ax.set_xlabel("Baseline PC1")
        ax.set_ylabel("Baseline PC2")

    ax = axes[1, 2]
    diff = field.dropna(subset=["diff_dx", "diff_dy", "diff_exit_rate_reduction"]).drop_duplicates(["bin_x", "bin_y"])
    if not diff.empty:
        vmax = max(0.15, float(diff["diff_exit_rate_reduction"].abs().max()))
        n_col = "diff_min_arm_n" if "diff_min_arm_n" in diff.columns else "min_arm_n"
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        resolved = diff["diff_exit_excludes_zero"].fillna(False) if "diff_exit_excludes_zero" in diff else pd.Series(False, index=diff.index)
        scat = ax.scatter(
            diff["x"],
            diff["y"],
            c=diff["diff_exit_rate_reduction"],
            cmap=DIFF_CMAP,
            norm=norm,
            s=76 + diff[n_col] * 2.0,
            edgecolor="none",
            linewidth=0,
            alpha=0.25,
            zorder=2,
        )
        if resolved.any():
            ax.scatter(
                diff.loc[resolved, "x"],
                diff.loc[resolved, "y"],
                c=diff.loc[resolved, "diff_exit_rate_reduction"],
                cmap=DIFF_CMAP,
                norm=norm,
                s=86 + diff.loc[resolved, n_col] * 2.0,
                edgecolor=INK,
                linewidth=0.9,
                alpha=0.92,
                zorder=4,
            )
        ax.quiver(
            diff["x"],
            diff["y"],
            diff["diff_dx"],
            diff["diff_dy"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=INK,
            width=0.006,
            alpha=0.95,
        )
        fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.02, label="PFS-risk reduction")
    ax.set_title("Differential 16-week field: panitumumab added", loc="left", fontweight="bold")
    ax.set_xlabel("Baseline PC1")
    ax.set_ylabel("Baseline PC2")

    fig.suptitle(
        "PDS PRIME: a randomized metastatic CRC trial as observed movement in the frozen state map",
        x=0.02,
        y=1.055,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.savefig(OUT / "fig_pds_prime_trial_geometry.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_pds_prime_trial_geometry.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_subgroups(subgroups: pd.DataFrame, local: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax in axes:
        _setup_ax(ax)
    ax = axes[0]
    if not subgroups.empty:
        sg = subgroups.copy()
        sg["label"] = sg["subgroup"] + ": " + sg["level"]
        sg = sg.sort_values("rmst24_diff_months")
        y = np.arange(len(sg))
        ax.hlines(y, sg["ci_low"], sg["ci_high"], color="#7b8794", linewidth=1.5)
        ax.scatter(sg["rmst24_diff_months"], y, color=np.where(sg["rmst24_diff_months"].ge(0), BENEFIT, HARM), s=42)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(y, sg["label"], fontsize=8)
        ax.set_xlabel("RMST24 gain, months\nPanitumumab + FOLFOX minus FOLFOX")
    ax.set_title("Published-like subgroup localization", loc="left", fontweight="bold")

    for i, kras in enumerate(["Wild-type", "Mutant"]):
        ax = axes[i + 1]
        sub = local[local["kras_exon2"].eq(kras)]
        if not sub.empty:
            vmax = max(0.18, float(sub["risk_reduction12"].abs().max()))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            resolved = sub["ci_excludes_zero"].fillna(False)
            scat = ax.scatter(
                sub["x"],
                sub["y"],
                c=sub["risk_reduction12"],
                cmap=DIFF_CMAP,
                norm=norm,
                s=22,
                marker="s",
                linewidth=0,
                alpha=0.22,
            )
            if resolved.any():
                ax.scatter(
                    sub.loc[resolved, "x"],
                    sub.loc[resolved, "y"],
                    c=sub.loc[resolved, "risk_reduction12"],
                    cmap=DIFF_CMAP,
                    norm=norm,
                    s=30,
                    marker="s",
                    edgecolor=INK,
                    linewidth=0.45,
                    alpha=0.92,
                )
            fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.02, label="12-mo PFS-risk reduction")
        ax.set_title(f"Local benefit surface, KRAS {kras}", loc="left", fontweight="bold")
        ax.set_xlabel("Baseline PC1")
        ax.set_ylabel("Baseline PC2")
    n_resolved = int(local["ci_excludes_zero"].sum()) if not local.empty else 0
    fig.text(
        0.535,
        -0.015,
        f"Local surfaces: k=45 nearest patients per arm. Pale cells have bootstrap CIs crossing zero; "
        f"outlined cells exclude zero ({n_resolved}/{len(local)}).",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(OUT / "fig_pds_prime_subgroup_localization.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_pds_prime_subgroup_localization.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_state_annotations(baseline: pd.DataFrame, meta: dict[str, Any]) -> None:
    """Show where clinically familiar subgroup variables sit on the PCA map."""
    fig, axes = plt.subplots(3, 4, figsize=(15.6, 10.5), constrained_layout=True)
    for ax in axes.flat:
        _setup_ax(ax)

    categorical = [
        ("KRAS exon 2", "kras_exon2", {"Wild-type": BENEFIT, "Mutant": HARM, "Unknown": "#9aa4b2", "Failure": "#c7ccd4"}),
        ("Randomized arm", "arm_raw", {ARM_CONTROL: ARM_COLOR[ARM_CONTROL], ARM_EXPERIMENTAL: ARM_COLOR[ARM_EXPERIMENTAL]}),
        ("ECOG", "ecog_group", {"0": "#2c7fb8", "1-2": "#de7f2d"}),
        ("Liver metastases", "liver_group", {"no": "#7b8da3", "yes": "#6a8f2a"}),
        ("Baseline burden", "burden_group", {"low": "#8cc6cf", "high": "#274c77"}),
        ("Metastatic sites", "met_sites_group", {"1-2": "#8e7cc3", "3+": "#4d2d7a"}),
        ("Primary", "primary_group", {"Colon": "#3c8dbc", "Rectal": "#b66b3d", "Other": "#9aa4b2"}),
        ("Age", "age_group", {"<65": "#6baed6", "65+": "#8856a7"}),
    ]
    for ax, (title, col, palette) in zip(axes.flat[:8], categorical):
        for value, color in palette.items():
            g = baseline[baseline[col].eq(value)]
            if g.empty:
                continue
            ax.scatter(g["pc1"], g["pc2"], s=15, alpha=0.68, color=color, label=str(value), linewidth=0)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=7, loc="best", markerscale=1.2)

    continuous = [
        ("Target-lesion burden", "target_sld", "mm"),
        ("Baseline metastatic sites", "baseline_met_sites", "sites"),
        ("PFS time", "pfs_months", "months"),
        ("OS time", "os_months", "months"),
    ]
    for ax, (title, col, units) in zip(axes.flat[8:], continuous):
        sub = baseline.dropna(subset=[col])
        vmax = float(sub[col].quantile(0.95)) if not sub.empty else 1.0
        scat = ax.scatter(
            sub["pc1"],
            sub["pc2"],
            c=sub[col].clip(upper=vmax),
            s=16,
            alpha=0.72,
            cmap=RISK_CMAP,
            linewidth=0,
        )
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.02, label=units)

    fig.suptitle(
        "Clinical annotations of PRIME patients in the frozen state map",
        x=0.02,
        y=1.025,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    axes[2, 0].text(
        0.02,
        -0.34,
        f"Frozen PC1 {meta['explained_variance']['PC1']*100:.1f}% · "
        f"PC2 {meta['explained_variance']['PC2']*100:.1f}% of atlas-state variance",
        transform=axes[2, 0].transAxes,
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(OUT / "fig_pds_prime_state_annotations.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_pds_prime_state_annotations.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_examples(states: pd.DataFrame, subjects: pd.DataFrame) -> None:
    eligible = states.groupby("SUBJID").filter(lambda g: len(g) >= 5)
    examples = []
    rng = np.random.default_rng(SEED)
    for arm_raw, g in eligible.groupby("arm_raw"):
        ids = g["SUBJID"].unique()
        examples.extend(rng.choice(ids, min(3, len(ids)), replace=False))
    fig, axes = plt.subplots(len(examples), 2, figsize=(11.5, max(6, 2.2 * len(examples))), constrained_layout=True)
    if len(examples) == 1:
        axes = np.array([axes])
    for row, sid in enumerate(examples):
        g = states[states["SUBJID"].eq(sid)].sort_values("VISITDY")
        arm_raw = g["arm_raw"].iloc[0]
        color = ARM_COLOR[arm_raw]
        ax = axes[row, 0]
        _setup_ax(ax)
        sc = ax.scatter(g["pc1"], g["pc2"], c=g["VISITDY"] / MONTH_DAYS, cmap="viridis", s=26, zorder=3)
        ax.plot(g["pc1"], g["pc2"], color=color, linewidth=1.5, alpha=0.75)
        ax.scatter(g["pc1"].iloc[0], g["pc2"].iloc[0], marker="o", s=48, color="white", edgecolor=INK, zorder=4)
        ax.scatter(g["pc1"].iloc[-1], g["pc2"].iloc[-1], marker="X", s=50, color=EXIT, zorder=4)
        ax.set_title(f"{sid} · {ARM_LABEL[arm_raw]}", loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if row == 0:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Month")

        ax = axes[row, 1]
        _setup_ax(ax)
        ax.plot(g["VISITDY"] / MONTH_DAYS, g["target_sld"], color=INK, linewidth=1.8, label="Target SLD")
        ax.plot(g["VISITDY"] / MONTH_DAYS, g["liver"], color="#6a8f2a", linewidth=1.2, label="Liver")
        ax.plot(g["VISITDY"] / MONTH_DAYS, g["lung"], color="#2a78a6", linewidth=1.2, label="Lung")
        ax.plot(g["VISITDY"] / MONTH_DAYS, g["lymph_node"], color="#7c4d9e", linewidth=1.2, label="LN")
        p = subjects[subjects["SUBJID"].eq(sid)].iloc[0]
        if p["pfs_event"] == 1:
            ax.axvline(p["pfs_months"], color=EXIT, linestyle="--", linewidth=1.1)
        ax.set_xlabel("Months")
        ax.set_ylabel("mm")
        if row == 0:
            ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig_pds_prime_example_paths.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_pds_prime_example_paths.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(readout: dict[str, Any]) -> None:
    lines = [
        "# PDS PRIME trial geometry",
        "",
        "Exploratory analysis of Project Data Sphere NCT00364013 / Colorec_Amgen_2006_264.",
        "",
        "## Suitability",
        "",
        f"- Subjects in released analysis set: {readout['cohort']['n_subjects']:,}",
        f"- Arms: {readout['cohort']['arm_counts']}",
        f"- Subjects with serial lesion snapshots: {readout['cohort']['n_subjects_with_lesions']:,}",
        f"- Median lesion snapshots per subject: {readout['cohort']['median_snapshots_per_subject']:.1f}",
        f"- Serial lab rows: {readout['cohort']['n_lab_rows']:,}",
        "- Primary display coordinates: frozen paper state map; trial-specific PCA retained only as a diagnostic.",
        "",
        "## Conventional PFS",
        "",
    ]
    for row in readout["cox_pfs"]:
        lines.append(
            f"- {row['group']}: panitumumab+FOLFOX vs FOLFOX HR "
            f"{row['hr_panitumumab_vs_control']:.2f} "
            f"({row['ci_low']:.2f}-{row['ci_high']:.2f}), "
            f"n={row['n']:,}, events={row['events']:,}"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `fig_pds_prime_trial_geometry.png`: KM, frozen-map projection, observed paths, arm fields, differential field.",
            "- `fig_pds_prime_subgroup_localization.png`: RMST subgroup forest and uncertainty-aware local benefit surfaces.",
            "- `fig_pds_prime_state_annotations.png`: forest variables and endpoint summaries overlaid on the frozen PCA map.",
            "- `fig_pds_prime_example_paths.png`: individual observed phase-space and lesion-burden paths.",
            "",
            "## Interpretation guardrail",
            "",
            "These are observed randomized-arm trajectories, not individual counterfactuals. "
            "The differential field contrasts randomized groups within baseline-state bins or "
            "local neighborhoods.",
            f"- Differential field cells with bootstrap CI excluding zero: "
            f"{readout['field']['differential_exit_ci_excluding_zero']:,}/"
            f"{readout['field']['n_differential_cells']:,}.",
            f"- Local surface cells with bootstrap CI excluding zero: "
            f"{readout['local_surface_ci_excluding_zero']:,}/{readout['local_surface_rows']:,}.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = _find_data_dir()
    tables = load_tables(base)
    subjects = derive_subjects(tables)
    snapshots = derive_lesion_snapshots(tables, subjects)
    snapshots = add_host_state(tables, snapshots)
    states, trial_loadings, trial_meta = fit_state_space(snapshots)
    states, frozen_loadings, frozen_meta = add_frozen_projection(states)
    baseline = baseline_rows(states)
    # Carry baseline burden grouping back to subject frame.
    subjects = subjects.merge(baseline[["SUBJID", "burden_group"]], on="SUBJID", how="left")

    cox = cox_summary(subjects)
    subgroups = subgroup_rmst(subjects, baseline)
    disp, field = horizon_displacements(states, baseline)
    local = pd.concat(
        [
            local_effect_surface(subjects, baseline, kras_level="Wild-type"),
            local_effect_surface(subjects, baseline, kras_level="Mutant"),
        ],
        ignore_index=True,
    )

    states.to_csv(OUT / "state_snapshots.csv.gz", index=False)
    baseline.to_csv(OUT / "baseline_state.csv", index=False)
    subjects.to_csv(OUT / "subject_table.csv", index=False)
    trial_loadings.to_csv(OUT / "trial_specific_state_pca_loadings.csv", index=False)
    frozen_loadings.to_csv(OUT / "frozen_state_map_loadings.csv", index=False)
    subgroups.to_csv(OUT / "subgroup_rmst24.csv", index=False)
    disp.to_csv(OUT / "horizon_displacements.csv", index=False)
    field.to_csv(OUT / "field_cells.csv", index=False)
    local.to_csv(OUT / "local_effect_surface.csv", index=False)
    cox.to_csv(OUT / "cox_pfs.csv", index=False)

    plot_main(subjects, states, baseline, disp, field, cox, frozen_meta)
    plot_subgroups(subgroups, local)
    plot_state_annotations(baseline, frozen_meta)
    plot_examples(states, subjects)

    readout = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "git_commit": _git_commit(),
        "data_dir": str(base),
        "data_hashes": {p.name: _sha256(p) for p in sorted(base.glob("*.sas7bdat"))},
        "cohort": {
            "n_subjects": int(len(subjects)),
            "arm_counts": {str(k): int(v) for k, v in subjects["arm_raw"].value_counts().items()},
            "kras_exon2_counts": {str(k): int(v) for k, v in subjects["kras_exon2"].value_counts(dropna=False).items()},
            "n_subjects_with_lesions": int(states["SUBJID"].nunique()),
            "n_state_snapshots": int(len(states)),
            "median_snapshots_per_subject": float(states.groupby("SUBJID").size().median()),
            "n_lab_rows": int(len(tables.adlb)),
            "n_ae_rows": int(len(tables.adae)),
            "horizon_days": HORIZON_DAYS,
        },
        "state_space": {
            "primary_display": frozen_meta,
            "trial_specific_diagnostic": trial_meta,
        },
        "cox_pfs": cox.to_dict(orient="records"),
        "field": {
            "n_displacement_subjects": int(len(disp)),
            "exit_by_tau_rate": float(disp["exit_by_tau"].mean()) if len(disp) else np.nan,
            "n_field_cells": int(field[["bin_x", "bin_y"]].drop_duplicates().shape[0]) if not field.empty else 0,
            "n_differential_cells": int(field.dropna(subset=["diff_dx", "diff_dy"]).drop_duplicates(["bin_x", "bin_y"]).shape[0])
            if {"diff_dx", "diff_dy"}.issubset(field.columns)
            else 0,
            "differential_exit_ci_excluding_zero": int(
                field.dropna(subset=["diff_dx", "diff_dy"])
                .drop_duplicates(["bin_x", "bin_y"])["diff_exit_excludes_zero"]
                .fillna(False)
                .sum()
            )
            if {"diff_dx", "diff_dy", "diff_exit_excludes_zero"}.issubset(field.columns)
            else 0,
        },
        "subgroup_rows": subgroups.to_dict(orient="records"),
        "local_surface_rows": int(len(local)),
        "local_surface_ci_excluding_zero": int(local["ci_excludes_zero"].sum()) if len(local) else 0,
        "annotation_variables": [
            "kras_exon2",
            "arm_raw",
            "ecog_group",
            "liver_group",
            "burden_group",
            "met_sites_group",
            "primary_group",
            "age_group",
            "target_sld",
            "baseline_met_sites",
            "pfs_months",
            "os_months",
        ],
    }
    (OUT / "readout.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")
    write_readme(readout)
    print(json.dumps({k: readout[k] for k in ("cohort", "field")}, indent=2))
    return 0


def rmst_only(**kw) -> pd.DataFrame:
    """Recompute only the subgroup restricted-mean table (Fig. 6a) from the source tables and the saved
    baseline states, without rewriting the other outputs of this script."""
    subjects = derive_subjects(load_tables(_find_data_dir()))
    baseline = pd.read_csv(OUT / "baseline_state.csv", dtype={"SUBJID": str})
    subjects = subjects.merge(baseline[["SUBJID", "burden_group"]], on="SUBJID", how="left")
    return subgroup_rmst(subjects, baseline, **kw)


if __name__ == "__main__":
    if "--rmst-only" in sys.argv:
        rmst_only().to_csv(OUT / "subgroup_rmst24.csv", index=False)
        raise SystemExit(0)
    raise SystemExit(main())
