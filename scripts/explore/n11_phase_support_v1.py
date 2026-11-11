#!/usr/bin/env python3
"""N11 feasibility: SUPPORT COUNTS ONLY for the six measured moments (N11 specification). The fields themselves are in n11_event_conditioned_fields_v1.py, which reuses `visits()`.

No field, mean displacement, density contour or outcome comparison is computed here.

Data: generated atlas on the frozen state map (outputs/paper_state_map/state_projection.csv.gz) and the atlas event
ledgers, through the loaders of scripts/build_paper_group_state_vector_field_views.py, as Extended Data Fig. 4a.
Start visits: every month (day m x 30.44, the ED4a clock) at which a trajectory has a living state. Follow-up 3 months:
"followed" if alive with a state 3 months later, "died" if death within 3 months, otherwise left out (record ends).

Phases (fixed in the specification; event times from each trajectory's own ledger; literal event-based labels):
  1 before the first systemic treatment start (all living months if the trajectory never starts treatment)
  2 first 3 months after the first systemic treatment start, before any progression (starts after a progression are
    not in this phase)
  3 after the first response recorded after the first treatment start, before the next progression
  4 3 months or more after that first response, no progression recorded since (a subset of 3; continued response is
    not verified)
  5 first 6 months after the first progression
  6 first 3 months after the next treatment start or switch following the first treatment start (not a verified
    clinical line)
The six phases are overlapping event-conditioned views, not successive stages of one course.
Cells: the ED4a grid (side 0.12 map units from the ED4a window origin). Support for an arrow (amended): at least 30
distinct courses with a start visit in the cell AND at least 30 distinct courses contributing a followed endpoint.
Deaths are counted as distinct trajectories with a start visit in the phase followed by death within 3 months (the
monthly-window count is kept only as a column, labelled windows). Person-time coverage: share of a phase's start visits
(one month of person-time each) in supported cells.
Output: outputs/exploratory_n11_phase_support/ (aggregate counts + manifest with code and input hashes).
Run: uv run python scripts/explore/n11_phase_support_v1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
with matplotlib.rc_context():
    import build_paper_group_state_vector_field_views as o  # noqa: E402

FIELD = ROOT / "outputs" / "paper_atlas_finite_displacement_field" / "manifest.json"
PROJ = ROOT / "outputs" / "paper_state_map" / "state_projection.csv.gz"
OUT = ROOT / "outputs" / "exploratory_n11_phase_support"
MONTH = 30.44
FOLLOW = 3.0
MIN_COURSES, MIN_ENDPOINTS = 30, 30
PHASES = {1: "before the first systemic treatment",
          2: "first 3 months after the first treatment start, no progression yet",
          3: "after the first response, before the next progression",
          4: "3 months or more after the first response, no progression since",
          5: "first 6 months after the first progression",
          6: "first 3 months after the next treatment start or switch"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first(days: list[float], after: float = -np.inf) -> float:
    later = [d for d in days if d >= after]
    return min(later) if later else np.inf


def phases_at(t: float, ev: dict[str, list[float]]) -> list[int]:
    t1 = first(ev.get("systemic_treatment_start", []))
    p1 = first(ev.get("progression", []))
    r1 = first(ev.get("response", []), t1) if np.isfinite(t1) else np.inf
    p_after_r = first(ev.get("progression", []), r1) if np.isfinite(r1) else np.inf
    t2 = min(first(ev.get("systemic_treatment_switch", []), t1 + 1e-9) if np.isfinite(t1) else np.inf,
             first(ev.get("systemic_treatment_start", []), t1 + 1e-9) if np.isfinite(t1) else np.inf)
    out = []
    if t < t1:
        out.append(1)
    if t1 <= t < t1 + 3 * MONTH and t < p1:
        out.append(2)
    if r1 <= t < p_after_r:
        out.append(3)
        if t >= r1 + 3 * MONTH:
            out.append(4)
    if p1 <= t < p1 + 6 * MONTH:
        out.append(5)
    if t2 <= t < t2 + 3 * MONTH:
        out.append(6)
    return out


def visits() -> tuple[pd.DataFrame, dict]:
    """Monthly start visits with phase, cell, start and end positions and fate (one row per visit and phase)."""
    man = json.loads(FIELD.read_text())
    x0, y0, cell = man["window_x"][0], man["window_y"][0], man["cell_side_map_units"]
    proj = o.load_projection()
    ids = set(proj["trajectory_id"].astype(str))
    events = o.load_events(sorted(proj["_run"].astype(str).unique()), ids)
    deaths = o.death_days(events)
    paths = o.build_paths(proj)
    ev_by: dict[str, dict[str, list[float]]] = {}
    for tid, g in events.groupby("trajectory_id"):
        d = pd.to_numeric(g["days_from_diagnosis"], errors="coerce")
        ev_by[str(tid)] = {t: sorted(d[g["event_type"].eq(t)].dropna().tolist()) for t in g["event_type"].unique()}
    rows = []
    for tid, path in paths.items():
        dd = deaths.get(tid)
        ev = ev_by.get(str(tid), {})
        for m in range(0, 241):
            day = m * MONTH
            start = o.state_at_day(path, day, dd)
            if start is None:
                if dd is not None and day >= dd:
                    break
                continue
            ph = phases_at(day, ev)
            if not ph:
                continue
            end_day = day + FOLLOW * MONTH
            if dd is not None and dd <= end_day:
                fate = "died"
            elif o.state_at_day(path, end_day, dd) is not None:
                fate = "followed"
            else:
                fate = "left_out"
            ix = int(np.floor((float(start["pc1"]) - x0) / cell))
            iy = int(np.floor((float(start["pc2"]) - y0) / cell))
            end = o.state_at_day(path, end_day, dd) if fate == "followed" else None
            ex, ey = (float(end["pc1"]), float(end["pc2"])) if end is not None else (np.nan, np.nan)
            for p in ph:
                rows.append((p, tid, m, ix, iy, float(start["pc1"]), float(start["pc2"]), ex, ey, fate))
    v = pd.DataFrame(rows, columns=["phase", "trajectory_id", "month", "ix", "iy", "x", "y", "ex", "ey", "fate"])
    return v, {"x0": x0, "y0": y0, "cell": cell, "n_paths": len(paths)}


def supported_cells(g: pd.DataFrame) -> pd.DataFrame:
    """Per cell: distinct courses starting, distinct courses contributing a followed endpoint, supported flag."""
    cells = g.groupby(["ix", "iy"]).agg(courses=("trajectory_id", "nunique"))
    contrib = g[g.fate.eq("followed")].groupby(["ix", "iy"])["trajectory_id"].nunique().rename("contributing")
    cells = cells.join(contrib).fillna({"contributing": 0})
    cells["supported"] = (cells.courses >= MIN_COURSES) & (cells.contributing >= MIN_ENDPOINTS)
    return cells


def main() -> int:
    v, meta = visits()
    summary = []
    for p, g in v.groupby("phase"):
        cells = supported_cells(g)
        ok = cells[cells.supported].index
        inside = g.set_index(["ix", "iy"]).index.isin(ok)
        summary.append({"phase": p, "label": PHASES[p], "trajectories": int(g.trajectory_id.nunique()),
                        "start_visits": int(len(g)),
                        "trajectories_followed_at_3_months": int(g.loc[g.fate.eq("followed"), "trajectory_id"].nunique()),
                        "distinct_trajectories_died_within_3_months": int(g.loc[g.fate.eq("died"), "trajectory_id"].nunique()),
                        "windows_followed_by_death": int((g.fate == "died").sum()),
                        "left_out_record_ends": int((g.fate == "left_out").sum()),
                        "cells_with_any_visit": int(len(cells)), "cells_supported": int(len(ok)),
                        "person_time_share_in_supported_cells": round(float(inside.mean()), 4)})
    for p in PHASES:
        if p not in set(v.phase):
            summary.append({"phase": p, "label": PHASES[p], "trajectories": 0, "start_visits": 0})
    s = pd.DataFrame(summary).sort_values("phase")
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "n11_phase_support_v1.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "script": "scripts/explore/n11_phase_support_v1.py", "counts_only": True, "month_days": MONTH,
        "follow_months": FOLLOW, "grid": {"cell": meta["cell"], "origin": [meta["x0"], meta["y0"]]},
        "min_distinct_courses_starting": MIN_COURSES, "min_distinct_courses_contributing": MIN_ENDPOINTS,
        "trajectories_in_atlas": meta["n_paths"], "amended": "Codex 28 Oct: distinct deaths, distinct contributors, literal labels",
        "provenance": {"code_sha256": sha256(Path(__file__)),
                       "inputs_sha256": {str(PROJ.relative_to(ROOT)): sha256(PROJ),
                                         str(FIELD.relative_to(ROOT)): sha256(FIELD)}}}, indent=1) + "\n")
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
