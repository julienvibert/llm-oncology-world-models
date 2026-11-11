"""Data for the SI5 panels: one generated course decomposed on the frozen state map.

Selection (declared rule, no choice by eye): among lung adenocarcinoma (LUAD) courses of the six-model
atlas with at most 10 recorded events that contain a response, a toxicity onset and a progression, each
of which moves the patient on the map, the lowest SHA-256 of the trajectory id. Snapshots come from
outputs/paper_state_map/state_projection.csv.gz (snapshot k is the state after event k). Because the
map is a fixed linear map of 20 features, centred but not rescaled, each arrow equals the sum over
features of (loading x change in feature); this is checked to 1e-9 on load.
Generated data only.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "explore" / "figpanels"))
sys.path.insert(0, str(HERE.parents[1] / "explore"))
sys.path.insert(0, str(HERE.parents[1]))
from common import ROOT  # noqa: E402
from fig1_data import _read  # noqa: E402

MAP = ROOT / "outputs" / "paper_state_map"
DPM = 30.4375
BANDS = ["absent", "microscopic", "small", "moderate", "bulky", "diffuse"]
SYMP = ["none", "mild", "moderate", "severe"]
SITE = {"tumour_hepatic": "Liver", "tumour_thoracic": "Lung", "tumour_nodal": "Lymph nodes",
        "tumour_osseous": "Bone", "tumour_marrow": "Bone marrow", "tumour_cns": "Brain or CNS",
        "tumour_peritoneal": "Peritoneum", "tumour_soft_tissue_cutaneous": "Soft tissue or skin",
        "tumour_other_anatomical": "Other site"}
EVENT_NAME = {"response": "Response", "progression": "Progression", "toxicity": "Toxicity starts",
              "patient_state_update": "Condition recorded as worse", "systemic_treatment_start": "Treatment starts"}


def describe(feature: str, v: float) -> str:
    """Plain words for an encoded feature value."""
    if feature in SITE:
        return BANDS[int(round(v * 5))]
    return {
        "tumour_extent": f"{int(round(v * 5))} site classes",
        "host_ecog": f"{int(round(v * 4))}",
        "host_symptoms": SYMP[int(round(v * 3))],
        "host_active_ae_max_grade": f"{int(round(v * 5))}",
        "host_active_ae_count": f"{int(round(v * 5))}",
        "host_relative_weight_loss": f"{v * 50 - 20:.0f}%",
    }.get(feature, f"grade {int(round(v * 4))}")


def name(feature: str, labels: dict[str, str]) -> str:
    if feature in SITE:
        return SITE[feature] + " tumour burden"
    return {"tumour_extent": "Disease extent", "host_ecog": "ECOG status",
            "host_symptoms": "Symptoms", "host_active_ae_max_grade": "Worst side-effect grade",
            "host_active_ae_count": "Active side effects (n)"}.get(feature, labels[feature].capitalize())


def load() -> dict:
    load_L = pd.read_csv(MAP / "feature_loadings.csv")
    proj = pd.read_csv(MAP / "state_projection.csv.gz")
    ev = _read("event_rows.csv")
    ev = ev[ev.tumor_oncotree_code == "LUAD"]
    F = load_L.feature.tolist()
    cands = []
    for gid, x in ev.groupby("_gid"):
        x = x.sort_values("event_index")
        if len(x) > 10:
            continue
        s = proj[proj._gid == gid].sort_values("snapshot_index")
        mv = np.hypot(s.pc1.diff(), s.pc2.diff()).fillna(0).to_numpy()
        moved = {}
        for r in x.itertuples():
            if r.event_type == "toxicity" and r.toxicity_transition != "onset":
                continue
            if int(r.event_index) < len(mv):
                moved.setdefault(r.event_type, []).append(mv[int(r.event_index)] > 1e-9)
        if all(any(moved.get(k, [False])) for k in ("response", "progression", "toxicity")):
            cands.append((hashlib.sha256(gid.encode()).hexdigest(), gid))
    cands.sort()
    gid = cands[0][1]
    x = ev[ev._gid == gid].sort_values("event_index")
    s = proj[proj._gid == gid].sort_values("snapshot_index").reset_index(drop=True)
    arrows = []
    for k in range(1, len(s)):
        d = s.loc[k, F].to_numpy(float) - s.loc[k - 1, F].to_numpy(float)
        dx, dy = s.pc1[k] - s.pc1[k - 1], s.pc2[k] - s.pc2[k - 1]
        if np.hypot(dx, dy) < 1e-9:
            continue
        c1, c2 = d * load_L.pc1_coefficient.to_numpy(), d * load_L.pc2_coefficient.to_numpy()
        assert abs(c1.sum() - dx) < 1e-9 and abs(c2.sum() - dy) < 1e-9
        r = x[x.event_index == k].iloc[0]
        ch = [(f, describe(f, s.loc[k - 1, f]), describe(f, s.loc[k, f]), c1[i], c2[i])
              for i, f in enumerate(F) if abs(d[i]) > 1e-12]
        detail = ""
        if r.event_type == "toxicity":
            detail = f"{str(r.ae).replace('_', ' ')}, grade {int(r.ctcae_v5_grade)}"
        arrows.append({"k": k, "event": EVENT_NAME.get(r.event_type, r.event_type), "detail": detail,
                       "month": r.days_from_diagnosis / DPM, "dx": dx, "dy": dy, "changes": ch,
                       "x0": s.pc1[k - 1], "y0": s.pc2[k - 1], "x1": s.pc1[k], "y1": s.pc2[k]})
    death = x[x.event_type == "death"]
    return {"gid": gid, "model": x.model_short.iloc[0], "arrows": arrows, "snaps": s, "labels":
            dict(zip(load_L.feature, load_L.label)), "n_candidates": len(cands),
            "death_month": float(death.days_from_diagnosis.iloc[0] / DPM) if len(death) else None,
            "start": x[x.event_type == "systemic_treatment_start"].iloc[0]}
