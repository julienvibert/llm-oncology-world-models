"""Display-only reprojection of the saved EGFR attractor table onto the frozen 20-coordinate state map.

Same trajectories, same horizons, same exit and clinical flags; only map_x, map_y are replaced by the frozen-map
PC1, PC2 of the same state (anchor: last state at or before the decision, as in the original builder; later
horizons: linear interpolation of position between the saved states, as in the original builder).
Also used by the same-state and same-patient displays (ed03a, ed03c). Writes an aggregate-free synthetic-data table with hashes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper_precision_ladder_paired_views as paired  # noqa: E402

OUT = ROOT / "outputs" / "paper_egfr_frozen_map_reprojection"
SRC = ROOT / "outputs" / "paper_egfr_remission_attractor" / "trajectory_horizon_attractor.csv"
MONTH_DAYS = paired.MONTH_DAYS


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def reprojected() -> pd.DataFrame:
    states, _e, _l = paired.load_phase1()
    d = pd.read_csv(SRC)
    xs, ys = [], []
    courses = {t: g.sort_values(["days_from_diagnosis", "snapshot_index"]) for t, g in states.groupby("trajectory_id")}
    for t, g in d.groupby("trajectory_id", sort=False):
        c = courses[t]
        dd = c.drop_duplicates("days_from_diagnosis", keep="last")
        days, px, py = dd.days_from_diagnosis.to_numpy(float), dd.pc1.to_numpy(float), dd.pc2.to_numpy(float)
        cut = float(g.cutoff_days.iloc[0])
        for h, em in zip(g.horizon_months, g.endpoint_mo):
            if np.isclose(h, 0.0):
                obs = c[c.days_from_diagnosis <= cut]
                r = obs.iloc[-1] if len(obs) else c.iloc[0]
                x, y = float(r.pc1), float(r.pc2)
            else:
                day = cut + min(float(em), float(h)) * MONTH_DAYS
                x, y = float(np.interp(day, days, px, left=px[0], right=px[-1])), float(np.interp(day, days, py, left=py[0], right=py[-1]))
            xs.append(x), ys.append(y)  # noqa: E702
    out = d.copy()
    # groupby(sort=False) preserves first-appearance order of trajectories; rows are contiguous per trajectory
    order = pd.concat([g for _, g in d.groupby("trajectory_id", sort=False)]).index
    out.loc[order, "map_x_old"] = d.loc[order, "map_x"].to_numpy()
    out.loc[order, "map_y_old"] = d.loc[order, "map_y"].to_numpy()
    out.loc[order, "map_x"], out.loc[order, "map_y"] = xs, ys
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "trajectory_horizon_frozen_map.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sc = ROOT / "outputs" / "paper_state_map" / "state_map_contract.json"
    manifest = {
        "purpose": "display-only reprojection of the EGFR attractor table onto the frozen 20-coordinate state map",
        "unchanged": "trajectories, horizons, exit flags, clinical flags, selections",
        "replaced": "map_x, map_y (old values kept as map_x_old, map_y_old)",
        "source_sha256": _sha(SRC), "state_map_contract_sha256": _sha(sc), "git_commit": commit,
        "interpolation": "anchor = last state at or before decision; later horizons = linear interpolation between saved states",
        "rows": int(len(out)), "trajectories": int(out.trajectory_id.nunique())}
    # rewrite only when something other than the commit changed, so a rebuild leaves tracked
    # files untouched (release rehearsal B7)
    path = OUT / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "git_commit"} == \
                {k: v for k, v in manifest.items() if k != "git_commit"}:
            return out
    path.write_text(json.dumps(manifest, indent=1))
    return out
