"""The state of a GENIE-BPC patient at time t: the tumour block of the atlas CSS, read from imaging.

Each scan lists involved sites by ICD-O code, mapped to the atlas tumour classes, and the regions it covered. For each class, the state at t is the
reading of the last scan in the four months before t that covered that class: 1 if reported, 0 if not, unknown if no such scan. A site can therefore
clear and return, and an unscanned region stays unknown and never reads as recovery. Scans without evidence of cancer read 0 on the regions covered;
uncertain scans and scans reporting cancer without a site are not used. A patient is located when liver and thoracic classes are known; other
unknown classes then take the observed mean on the map only. No identifier is printed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("GENIE_ROOT", Path.home() / "GenieBPC"))
FOLDER = {"NSCLC": "NSCLC_Genie", "CRC": "CRC_GENIE"}
SITES = ["tumour_hepatic", "tumour_thoracic", "tumour_nodal", "tumour_osseous", "tumour_cns", "tumour_peritoneal", "tumour_soft_tissue_cutaneous",
         "tumour_other_anatomical"]
LAB = {"tumour_hepatic": "Liver", "tumour_thoracic": "Thoracic", "tumour_nodal": "Nodal", "tumour_osseous": "Bone", "tumour_cns": "CNS", "tumour_peritoneal": "Peritoneal",
       "tumour_soft_tissue_cutaneous": "Soft tissue", "tumour_other_anatomical": "Other"}
WINDOW = 4.0
COVER = {"tumour_hepatic": ("Abdomen", "Full body"), "tumour_thoracic": ("Chest", "Full body"), "tumour_nodal": ("Neck", "Chest", "Abdomen", "Pelvis", "Full body"),
         "tumour_osseous": ("Spine", "Chest", "Abdomen", "Pelvis", "Full body"), "tumour_cns": ("Brain/Head", "Full body"), "tumour_peritoneal": ("Abdomen", "Pelvis", "Full body"),
         "tumour_soft_tissue_cutaneous": ("Extremity", "Full body"), "tumour_other_anatomical": ("Neck", "Abdomen", "Pelvis", "Full body")}
CORE = ["tumour_hepatic", "tumour_thoracic"]


def site_class(code: str) -> str | None:
    c = code.strip()[:3]
    if c == "C22":
        return "tumour_hepatic"
    if c in ("C33", "C34", "C37", "C38", "C39"):
        return "tumour_thoracic"
    if c == "C77":
        return "tumour_nodal"
    if c in ("C40", "C41"):
        return "tumour_osseous"
    if c in ("C70", "C71", "C72"):
        return "tumour_cns"
    if c == "C48":
        return "tumour_peritoneal"
    if c in ("C44", "C49"):
        return "tumour_soft_tissue_cutaneous"
    if c == "C42":
        return None
    return "tumour_other_anatomical"


def load_scans(cancer: str) -> pd.DataFrame:
    """One row per informative scan: record, month from diagnosis, site flags."""
    im = pd.read_csv(BASE / FOLDER[cancer] / "imaging_level_dataset.csv", low_memory=False)
    cols = [c for c in im.columns if c.startswith("image_casite")]
    out = pd.DataFrame({"r": im.record_id, "t": pd.to_numeric(im.dx_scan_mos, errors="coerce")})
    for f in SITES:
        out[f] = 0.0
    arr = im[cols].astype(object).to_numpy()
    for i, row in enumerate(arr):
        for v in row:
            if isinstance(v, str):
                f = site_class(v)
                if f:
                    out.iat[i, out.columns.get_loc(f)] = 1.0
    ca = im.image_ca.astype(str)
    yes, no = ca.str.startswith("Yes"), ca.str.startswith("No,")
    regions = [c for c in im.columns if c.startswith("image_scansite")]
    seen = im[regions].astype(object)
    for f, names in COVER.items():
        out["cov_" + f] = seen.isin(names).any(axis=1).to_numpy()
    keep = (yes & (out[SITES].sum(axis=1) > 0)) | no
    return out[keep & out.t.notna()].sort_values("t").reset_index(drop=True)


def state_at(scans: pd.DataFrame, t: float | pd.Series) -> pd.DataFrame:
    """Last covering reading of each class in the window before t; t a number, or a Series of times indexed by record."""
    if isinstance(t, pd.Series):
        s = scans[scans.r.isin(t.index)]
        tt = s.r.map(t)
        s = s[(s.t <= tt) & (s.t > tt - WINDOW)]
    else:
        s = scans[(scans.t <= t) & (scans.t > t - WINDOW)]
    x = pd.DataFrame({f: s[s["cov_" + f]].groupby("r")[f].last() for f in SITES})
    x["extent"] = np.minimum(x[SITES].sum(axis=1), 5) / 5
    x.loc[x[SITES].isna().all(axis=1), "extent"] = np.nan
    return x


def located(x: pd.DataFrame) -> pd.Series:
    return x[CORE].notna().all(axis=1)


def weighted_pca(x: pd.DataFrame, cols: list[str], patient: pd.Series) -> tuple:
    """PCA in which each patient weighs the same, however many frames it has; unknowns take the mean. Returns mean, sd, PC1, PC2, variance shares."""
    w = 1.0 / patient.map(patient.value_counts()).to_numpy()
    w = w / w.sum()
    mu = pd.Series({c: np.nansum(x[c].to_numpy() * w) / w[x[c].notna().to_numpy()].sum() for c in cols})
    z = x[cols].fillna(mu)
    sd = np.sqrt(((z - mu) ** 2).mul(w, axis=0).sum()).replace(0, 1)
    u, sv, vt = np.linalg.svd(((z - mu) / sd).to_numpy() * np.sqrt(w)[:, None], full_matrices=False)
    return mu, sd, vt[0] * np.sign(vt[0].sum()), vt[1], sv ** 2 / (sv ** 2).sum()
