"""The Figure 1 example patient: events and state-map snapshots of one generated trajectory.

Selection (fixed rule, scripts/explore/figpanels/fig1_data.py): Claude Sonnet 5, lung
adenocarcinoma, at most 30 ledger events, a progression, a second line and death; lowest
SHA-256 of the trajectory id. Events come from that script's saved exemplar.json; the state
snapshots (tumour burden by site, host features, PC1/PC2) from the frozen state map
(outputs/paper_state_map/state_projection.csv.gz), matched by the same id hash.
Generated data only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402

EXEMPLAR = ROOT / "outputs/exploratory_figure_panels/fig1/data/exemplar.json"
PROJECTION = ROOT / "outputs/paper_state_map/state_projection.csv.gz"
DAYS_PER_MONTH = 30.4375

BURDEN_BANDS = ["absent", "microscopic", "small", "moderate", "bulky", "diffuse"]
SITES = [("tumour_thoracic", "Lung"), ("tumour_nodal", "Lymph nodes"), ("tumour_osseous", "Bone"),
         ("tumour_hepatic", "Liver"), ("tumour_cns", "Brain"), ("tumour_other_anatomical", "Other site")]
HOST = [("host_ecog", "ECOG", 4), ("host_symptoms", "Symptoms", 3),
        ("host_active_ae_max_grade", "Side effects", 5)]
REGIMEN = {"osimertinib": ("Osimertinib", "targeted"),
           "carboplatin_pemetrexed": ("Carboplatin + pemetrexed", "chemotherapy"),
           "carboplatin_etoposide": ("Carboplatin + etoposide", "chemotherapy")}


def load() -> tuple[dict, pd.DataFrame]:
    ex = json.loads(EXEMPLAR.read_text())
    proj = pd.read_csv(PROJECTION)
    gids = [g for g in proj["_gid"].unique()
            if hashlib.sha256(g.encode()).hexdigest()[:12] == ex["gid_hash"]]
    if len(gids) != 1:
        raise SystemExit(f"exemplar not found uniquely in the state projection: {gids}")
    snaps = proj[proj["_gid"] == gids[0]].sort_values("snapshot_index").copy()
    snaps["month"] = snaps["days_from_diagnosis"] / DAYS_PER_MONTH
    return ex, snaps


def disease_segments(ex: dict) -> list[tuple[float, float, str]]:
    """Disease state between assessments (not yet assessed before the first one)."""
    events = ex["events"]
    end = next(e["month"] for e in events if e["type"] == "death")
    seg, state, t0 = [], "unresolved", 0.0
    for e in events:
        if e["type"] in ("response", "progression") and e["type"] != state:
            seg.append((t0, e["month"], state))
            state, t0 = e["type"], e["month"]
    seg.append((t0, end, state))
    return [s for s in seg if s[1] > s[0]]


def treatment_lines(ex: dict) -> list[tuple[float, float, str, str]]:
    events = ex["events"]
    starts = [(e["month"], e["label"].split(": ", 1)[1]) for e in events
              if e["type"] in ("systemic_treatment_start", "systemic_treatment_switch")]
    stop = next(e["month"] for e in events if e["type"] == "systemic_treatment_stop")
    out = []
    for i, (m, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else stop
        name, cls = REGIMEN.get(key, (key.replace("_", " ").capitalize(), "other"))
        out.append((m, end, name, cls))
    return out


def death_month(ex: dict) -> float:
    return next(e["month"] for e in ex["events"] if e["type"] == "death")
