#!/usr/bin/env python3
"""N11 measured row: event-conditioned fields and accumulation regions of the generated atlas (N11
specification).

Reuses `n11_phase_support_v1.visits()` (monthly start visits, six literal event-based phases, ED4a grid, 3-month
follow-up). For each phase:
  field      per supported cell (>= 30 distinct courses starting and >= 30 distinct courses contributing a followed
             endpoint), the mean 3-month displacement over the followed start visits, with a whole-trajectory
             bootstrap (2,000 resamples of the courses in the cell, seed 20261028) for the 95% interval of each
             component and of the length;
  density    time-weighted density of start positions (one month of person-time per start visit) on a 120 x 120 grid
             over the ED4a window, Gaussian smoothing (1.5 bins), and the level enclosing the 50% highest-density
             region ("where generated patients accumulate"; not an attractor or a potential);
  deaths     distinct trajectories with a start visit in the phase followed by death within 3 months;
  patient    the Fig. 1c generated patient's first monthly start visit in each phase, and its path.
The phases overlap and are event-conditioned views, not successive stages of one course; arrows summarize survivors.
Output: outputs/exploratory_n11_event_fields/ (aggregate CSV/NPZ + manifest with code and input hashes).
Run: uv run python scripts/explore/n11_event_conditioned_fields_v1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "scripts" / "figures" / "panels"))
import _exemplar as X  # noqa: E402
import n11_phase_support_v1 as S  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_n11_event_fields"
B, SEED, NGRID, SMOOTH, HDR = 2000, 20261028, 120, 1.5, 0.50


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_field(g: pd.DataFrame, rng: np.random.Generator) -> dict:
    f = g[g.fate.eq("followed")]
    per = f.assign(dx=f.ex - f.x, dy=f.ey - f.y).groupby("trajectory_id").agg(sx=("dx", "sum"), sy=("dy", "sum"),
                                                                               n=("dx", "size"))
    sx, sy, n = per.sx.to_numpy(), per.sy.to_numpy(), per.n.to_numpy()
    mx, my = sx.sum() / n.sum(), sy.sum() / n.sum()
    idx = rng.integers(0, len(n), size=(B, len(n)))
    bx = sx[idx].sum(1) / n[idx].sum(1)
    by = sy[idx].sum(1) / n[idx].sum(1)
    bl = np.hypot(bx, by)
    return {"dx": mx, "dy": my, "length": float(np.hypot(mx, my)),
            "dx_lo": np.percentile(bx, 2.5), "dx_hi": np.percentile(bx, 97.5),
            "dy_lo": np.percentile(by, 2.5), "dy_hi": np.percentile(by, 97.5),
            "length_lo": np.percentile(bl, 2.5), "length_hi": np.percentile(bl, 97.5),
            "start_x": float(g.x.mean()), "start_y": float(g.y.mean()),
            "followed_visits": int(n.sum()), "contributing_courses": int(len(n))}


def main() -> int:
    v, meta = S.visits()
    man = json.loads(S.FIELD.read_text())
    wx, wy = man["window_x"], man["window_y"]
    xe = np.linspace(wx[0], wx[1], NGRID + 1)
    ye = np.linspace(wy[0], wy[1], NGRID + 1)
    rng = np.random.default_rng(SEED)
    fields, phases, dens = [], [], {}
    for p in sorted(S.PHASES):
        g = v[v.phase.eq(p)]
        cells = S.supported_cells(g)
        for (ix, iy), row in cells[cells.supported].iterrows():
            r = cell_field(g[(g.ix == ix) & (g.iy == iy)], rng)
            fields.append({"phase": p, "ix": ix, "iy": iy, "courses": int(row.courses), **r,
                           "cx": meta["x0"] + (ix + 0.5) * meta["cell"], "cy": meta["y0"] + (iy + 0.5) * meta["cell"]})
        h, _, _ = np.histogram2d(g.x, g.y, bins=[xe, ye])
        sm = gaussian_filter(h, SMOOTH)
        sm = sm / sm.sum()
        flat = np.sort(sm.ravel())[::-1]
        level = float(flat[np.searchsorted(np.cumsum(flat), HDR)])
        dens[f"phase{p}"] = sm
        phases.append({"phase": p, "label": S.PHASES[p], "trajectories": int(g.trajectory_id.nunique()),
                       "start_visits": int(len(g)), "supported_cells": int(cells.supported.sum()),
                       "distinct_deaths_within_3_months": int(g.loc[g.fate.eq("died"), "trajectory_id"].nunique()),
                       "hdr50_level": level})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fields).to_csv(OUT / "n11_fields_v1.csv", index=False)
    pd.DataFrame(phases).to_csv(OUT / "n11_phases_v1.csv", index=False)
    # the Fig. 1c generated patient: its first monthly start visit in each phase (generated data, no identifier saved)
    _, snaps = X.load()
    tid = str(snaps["trajectory_id"].iloc[0])
    ex = v[v.trajectory_id.astype(str).eq(tid)].sort_values("month")
    assert len(ex), "Fig. 1c patient not found among the start visits"
    first = ex.groupby("phase").first().reset_index()[["phase", "month", "x", "y", "ex", "ey", "fate"]]
    first.to_csv(OUT / "n11_exemplar_v1.csv", index=False)
    snaps[["month", "pc1", "pc2"]].to_csv(OUT / "n11_exemplar_path_v1.csv", index=False)
    np.savez_compressed(OUT / "n11_density_v1.npz", x_edges=xe, y_edges=ye, **dens)
    (OUT / "manifest.json").write_text(json.dumps({
        "script": "scripts/explore/n11_event_conditioned_fields_v1.py", "bootstrap": B, "seed": SEED,
        "density_grid": NGRID, "smoothing_bins": SMOOTH, "hdr_mass": HDR, "follow_months": S.FOLLOW,
        "support": {"min_distinct_courses_starting": S.MIN_COURSES,
                    "min_distinct_courses_contributing": S.MIN_ENDPOINTS},
        "provenance": {"code_sha256": sha256(Path(__file__)), "support_code_sha256": sha256(Path(S.__file__)),
                       "inputs_sha256": {str(S.PROJ.relative_to(ROOT)): sha256(S.PROJ),
                                         str(S.FIELD.relative_to(ROOT)): sha256(S.FIELD)}},
        "aggregate_only": True}, indent=1) + "\n")
    print(pd.DataFrame(phases).drop(columns="hdr50_level").to_string(index=False))
    f = pd.DataFrame(fields)
    print(f.groupby("phase").agg(cells=("dx", "size"), mean_dx=("dx", "mean"), mean_dy=("dy", "mean"),
                                 median_len=("length", "median")).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
