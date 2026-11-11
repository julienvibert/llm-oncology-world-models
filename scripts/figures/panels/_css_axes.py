"""Disease and host axes of the Clinical State Space: the first principal component of each block of 10 coordinates.

Fitted once on all atlas states; other data sets are projected with the same loadings. Both axes are oriented so that
higher means more disease or a more impaired host. A display, separate from the frozen map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402

TUMOUR = ["tumour_hepatic", "tumour_thoracic", "tumour_nodal", "tumour_osseous", "tumour_marrow", "tumour_cns",
          "tumour_peritoneal", "tumour_soft_tissue_cutaneous", "tumour_other_anatomical", "tumour_extent"]
HOST = ["host_ecog", "host_symptoms", "host_renal", "host_hepatic", "host_cardiac", "host_marrow_reserve", "host_pulmonary",
        "host_relative_weight_loss", "host_active_ae_max_grade", "host_active_ae_count"]


def _fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0)
    u, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    v = vt[0] * np.sign(vt[0].sum())
    return mean, v, float(s[0] ** 2 / (s ** 2).sum())


_atlas = pd.read_csv(ROOT / "outputs/paper_state_map/state_projection.csv.gz", usecols=TUMOUR + HOST)
_MT, _VT, VAR_DISEASE = _fit(_atlas[TUMOUR].fillna(0).to_numpy(float))
_MH, _VH, VAR_HOST = _fit(_atlas[HOST].fillna(0).to_numpy(float))
LOADINGS = {"disease": dict(zip(TUMOUR, _VT.round(2))), "host": dict(zip(HOST, _VH.round(2)))}
CAPTION = "Disease axis: first component of the 10 tumour coordinates.  Host axis: first component of the 10 host coordinates."


def project(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Disease and host coordinates of each row."""
    return ((d[TUMOUR].fillna(0).to_numpy(float) - _MT) @ _VT, (d[HOST].fillna(0).to_numpy(float) - _MH) @ _VH)
