#!/usr/bin/env python3
"""Perturbation-response matrix measured from the treatment decision (corrected time origin).

The exploratory producer (``scripts/explore/build_perturbation_response_matrix.py``, kept
unchanged for provenance) measured the "three-month" state change from the FIRST state row
of each future, i.e. from diagnosis. In the factorial design the treatment decision is not
at diagnosis for two of the three history levels:

  H0  no prior record; decision at diagnosis (day 0)
  H1  progression recorded 1 month after diagnosis, then the decision (day 32)
  H2  progression recorded 4 months after diagnosis, then the decision (day 123)

So the old H2 window (day 0 to day 91) ended before its decision, and the old H1 window
contained only two months after it. This producer measures every readout from the exact
persisted decision cutoff of each future (``observed_prefix.cutoff_date`` minus the
diagnosis date in its conditioning specification), with the controlled assay's rule:

  * anchor state   = last state recorded at or before the cutoff (LOCF);
  * horizon state  = last state recorded at or before cutoff + 3 months (LOCF);
  * exit           = death at or before cutoff + 3 months. An exited future has no
                     horizon state, so its three state-change readouts are missing (not
                     carried forward from before death). Toxicity is still read.

Endpoint origins (all deaths are observed; no future is censored):

  d_severity, d_burden, d_host  change from the decision to decision + 3 months in
                                state-map PC1, disease extent and ECOG (frozen
                                20-coordinate map, ``outputs/paper_state_map``).
  tox                           worst active toxicity grade (encoded) in the anchor state
                                or any state recorded after the decision up to decision +
                                3 months.
  pfs1                          first-line progression-free time: from the start of the
                                assigned first line to the first later progression or
                                death. The line starts on the decision day in every active
                                future (checked); supportive-care futures have no first
                                line and are missing, as in the original matrix.
  exit12                        death within 365 days after the decision (diagnosis-origin
                                ``os_days`` minus the cutoff).
  os                            survival from the decision (``os_days`` minus the cutoff).

The effect estimator, scale, permutation scheme, number of permutations, seed and the
order of random draws are those of the original producer.

Outputs (``outputs/perturbation_matrix_decision_aligned/``): ``readout.json`` (matrix, raw
spreads, history permutation null, per-history summaries, source hashes, parameters),
``readout.md`` (old vs new), ``future_readouts.csv`` (one row per generated future; model
output, no patient data).

Run: uv run python scripts/build_perturbation_matrix_decision_aligned.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PHASE1_RUN = "interventional_phase1_sonnet_full_v1c"
MAP = ROOT / "outputs" / "paper_state_map"
OLD_READOUT = ROOT / "outputs" / "exploratory_perturbation_matrix" / "readout.json"
OUT = ROOT / "outputs" / "perturbation_matrix_decision_aligned"
MONTH_DAYS = 30.4
TAU_MONTHS = 3.0
DEATH_WINDOW_DAYS = 365.0
NPERM = 400
SEED = 20260803

INPUTS = ("C", "X", "H", "A")
READOUTS = ("d_severity", "d_burden", "d_host", "tox", "pfs1", "exit12", "os")
STATE_READOUTS = ("d_severity", "d_burden", "d_host")
HISTORY_WORDING = {
    "H0": "no prior record; decision at diagnosis",
    "H1": "progression 1 month after diagnosis, then the treatment decision",
    "H2": "progression 4 months after diagnosis, then the treatment decision",
}


# ---------------------------------------------------------------- pure functions


def locf_state(course: pd.DataFrame, day: float) -> pd.Series:
    """Last state recorded at or before ``day``; same-day ties resolved by snapshot order.

    Raises if nothing is recorded by ``day``: the anchor must be observed, never borrowed
    from the future.
    """
    ordered = course.sort_values(["days_from_diagnosis", "snapshot_index"])
    eligible = ordered[ordered["days_from_diagnosis"] <= day + 1e-9]
    if eligible.empty:
        raise ValueError(f"No state recorded at or before day {day}")
    return eligible.iloc[-1]


def decision_window_readouts(
    course: pd.DataFrame,
    cutoff_days: float,
    death_days: float,
    *,
    horizon_days: float = TAU_MONTHS * MONTH_DAYS,
) -> dict[str, Any]:
    """State readouts from the exact decision cutoff to cutoff + horizon.

    ``course`` needs days_from_diagnosis, snapshot_index, pc1, tumour_extent, host_ecog and
    host_active_ae_max_grade. A future that dies at or before the horizon has exited: its
    state-change readouts are NaN.
    """
    target = float(cutoff_days) + float(horizon_days)
    anchor = locf_state(course, float(cutoff_days))
    exited = bool(float(death_days) <= target)
    days = course["days_from_diagnosis"].astype(float)
    window = course[(days > float(cutoff_days) + 1e-9) & (days <= target + 1e-9)]
    tox = float(np.nanmax(np.r_[float(anchor["host_active_ae_max_grade"]),
                                window["host_active_ae_max_grade"].to_numpy(float)]))
    out: dict[str, Any] = {
        "anchor_day": float(anchor["days_from_diagnosis"]),
        "horizon_day": target,
        "exited": exited,
        "tox": tox,
        "anchor_pc1": float(anchor["pc1"]),
    }
    if exited:
        out.update({k: np.nan for k in STATE_READOUTS})
        out["horizon_state_day"] = np.nan
        return out
    future = locf_state(course, target)
    out.update({
        "d_severity": float(future["pc1"] - anchor["pc1"]),
        "d_burden": float(future["tumour_extent"] - anchor["tumour_extent"]),
        "d_host": float(future["host_ecog"] - anchor["host_ecog"]),
        "horizon_state_day": float(future["days_from_diagnosis"]),
    })
    return out


def post_decision_endpoints(
    os_days_from_diagnosis: float,
    cutoff_days: float,
    pfs_line_1_days: float,
    line_1_start_days: float,
) -> dict[str, float]:
    """Convert diagnosis-origin survival to decision origin; check the first-line origin."""
    os_dec = float(os_days_from_diagnosis) - float(cutoff_days)
    if os_dec <= 0:
        raise ValueError("Death at or before the decision cutoff")
    pfs = float(pfs_line_1_days)
    if np.isfinite(pfs) and abs(float(line_1_start_days) - float(cutoff_days)) > 1e-9:
        raise ValueError("First line does not start on the decision day")
    return {"os": os_dec / MONTH_DAYS,
            "exit12": float(os_dec <= DEATH_WINDOW_DAYS),
            "pfs1": pfs / MONTH_DAYS if np.isfinite(pfs) else np.nan}


def effect(d: pd.DataFrame, inp: str, col: str,
           rng: np.random.Generator) -> tuple[float, float, int, np.ndarray]:
    """Null-corrected standardised spread of ``col`` across levels of ``inp``.

    Verbatim method of ``scripts/explore/build_perturbation_response_matrix.effect``:
    stratify on the other three inputs, cell means first, spread = max - min of level
    means averaged over strata, scale = median within-cell SD, null = 400 permutations of
    the level label within each stratum, reported = (observed - null mean) / scale.
    """
    others = [k for k in INPUTS if k != inp]
    sub = d.dropna(subset=[col])
    if sub[inp].nunique() < 2:
        return (np.nan, np.nan, 0, np.array([]))

    within = sub.groupby(list(INPUTS))[col].std(ddof=1)
    scale = float(np.nanmedian(within)) or float(sub[col].std(ddof=1)) or 1.0

    def spread(frame: pd.DataFrame, label: pd.Series) -> float:
        vals = []
        for _, g in frame.groupby(others, observed=True):
            m = g.assign(_L=label.loc[g.index]).groupby("_L")[col].mean()
            if m.notna().sum() >= 2:
                vals.append(m.max() - m.min())
        return float(np.mean(vals)) if vals else np.nan

    obs = spread(sub, sub[inp])
    null = []
    for _ in range(NPERM):
        lab = sub.groupby(others, observed=True)[inp].transform(
            lambda s: rng.permutation(s.to_numpy()))
        null.append(spread(sub, lab))
    nm = float(np.nanmean(null))
    return ((obs - nm) / scale, obs / scale, int(sub.groupby(others).ngroups),
            np.asarray(null, dtype=float) / scale)


def effect_matrix(d: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Run ``effect`` over inputs x readouts in the original order with one generator."""
    rng = np.random.default_rng(seed)
    M = np.full((len(INPUTS), len(READOUTS)), np.nan)
    raw = np.full_like(M, np.nan)
    nulls: dict[str, list[float]] = {}
    for i, inp in enumerate(INPUTS):
        for j, col in enumerate(READOUTS):
            M[i, j], raw[i, j], _n, null = effect(d, inp, col, rng)
            if inp == "H":
                nulls[col] = null.tolist()
    return {"M": M, "raw": raw, "history_nulls": nulls}


# ---------------------------------------------------------------- data assembly


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_futures() -> pd.DataFrame:
    """One row per future: design inputs and decision-aligned readouts."""
    import build_paper_precision_ladder_paired_views as paired
    import build_paper_state_map as smap

    states, endpoints, labels = paired.load_phase1()
    # Cutoffs re-read from the persisted conditioning specifications, then checked
    # against the saved controlled-assay records.
    cutoffs = smap.decision_cutoffs(PHASE1_RUN, labels["trajectory_id"].unique())
    saved = labels.set_index("trajectory_id")["cutoff_days"]
    fresh = pd.Series(cutoffs).reindex(saved.index)
    if not np.allclose(fresh.to_numpy(float), saved.to_numpy(float)):
        raise ValueError("Persisted cutoffs differ from phase1_p0_records.csv")

    events = pd.read_csv(ROOT / "runs" / PHASE1_RUN / "derivation" / "event_rows.csv",
                         low_memory=False)
    starts = events[events["event_type"].isin(["systemic_treatment_start",
                                                "systemic_treatment_switch"])
                    & pd.to_numeric(events["line"], errors="coerce").eq(1)]
    line1 = starts.groupby("trajectory_id")["days_from_diagnosis"].min()

    end = endpoints.set_index("_gid")
    if not end["outcome_type"].astype(str).eq("death").all():
        raise ValueError("Censored futures present; the endpoint conversion assumes none")
    rows = []
    for gid, course in states.groupby("_gid", sort=False):
        first = course.iloc[0]
        tid = str(first["trajectory_id"])
        cutoff = float(cutoffs[tid])
        os_days = float(end.loc[gid, "os_days"])
        w = decision_window_readouts(course, cutoff, os_days)
        e = post_decision_endpoints(os_days, cutoff,
                                    float(pd.to_numeric(end.loc[gid, "pfs_line_1_days"],
                                                        errors="coerce")),
                                    float(line1.get(tid, np.nan)))
        rows.append({"_gid": gid, "trajectory_id": tid, "C": str(first["context"]),
                     "X": str(first["state"]), "H": str(first["history"]),
                     "A": str(first["arm"]), "cutoff_days": cutoff, **w, **e})
    d = pd.DataFrame(rows)

    p0 = pd.read_csv(MAP / "phase1_p0_records.csv").set_index("_gid")
    if not (d.set_index("_gid")["exited"] == p0["exited"].reindex(d["_gid"]).to_numpy()).all():
        raise ValueError("Exit flags differ from the controlled-assay records")
    return d


def history_summary(d: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for h, g in d.groupby("H"):
        v = g["d_severity"].dropna()
        out[str(h)] = {
            "wording": HISTORY_WORDING[str(h)],
            "decision_cutoff_days": sorted(set(g["cutoff_days"].astype(float))),
            "n_futures": int(len(g)),
            "n_exited_by_3_months": int(g["exited"].sum()),
            "n_with_state_change_readout": int(len(v)),
            "n_pc1_changed": int((v.abs() > 1e-12).sum()),
            "mean_pc1_change": float(v.mean()),
            "median_os_from_decision_months": float(g["os"].median()),
            "death_by_12_months_after_decision": int(g["exit12"].sum()),
        }
    return out


def _fmt(m: np.ndarray) -> list[list[float | None]]:
    return [[None if not np.isfinite(x) else float(x) for x in row] for row in m]


def write_readout(d: pd.DataFrame, res: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "future_readouts.csv", index=False)
    old = json.loads(OLD_READOUT.read_text())
    hnull = np.asarray(res["history_nulls"]["d_severity"], float)
    obs_h = float(res["raw"][2, 0])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    derivation = ROOT / "runs" / PHASE1_RUN / "derivation"
    sources = {
        "endpoint_rows.csv": derivation / "endpoint_rows.csv",
        "event_rows.csv": derivation / "event_rows.csv",
        "snapshot_rows.csv": derivation / "snapshot_rows.csv",
        "disease_site_rows.csv": derivation / "disease_site_rows.csv",
        "baseline_rows.csv": derivation / "baseline_rows.csv",
        "state_map_contract.json": MAP / "state_map_contract.json",
        "phase1_p0_records.csv": MAP / "phase1_p0_records.csv",
        "old_readout.json": OLD_READOUT,
        "producer": Path(__file__).resolve(),
    }
    readout = {
        "description": ("Controlled perturbation-response matrix, every readout measured from "
                        "the exact persisted treatment-decision cutoff (Codex R1 correction)."),
        "n_futures": int(len(d)),
        "design": {k: int(d[k].nunique()) for k in INPUTS},
        "inputs": list(INPUTS), "readouts": list(READOUTS),
        "readout_definitions": {
            "d_severity": "change in state-map PC1 from decision to decision + 3 months (LOCF); "
                          "missing if death by then",
            "d_burden": "change in disease extent, same window and exit rule",
            "d_host": "change in ECOG, same window and exit rule",
            "tox": "worst active toxicity grade (encoded) from decision to decision + 3 months",
            "pfs1": "first-line progression-free time from first-line start (= decision day) "
                    "to first progression or death, months; supportive-care futures missing",
            "exit12": "death within 365 days after the decision",
            "os": "survival from the decision, months (os_days - cutoff)",
        },
        "history_levels": HISTORY_WORDING,
        "effect_null_corrected": _fmt(res["M"]),
        "effect_raw": _fmt(res["raw"]),
        "history_permutation_null_sd": hnull.tolist(),
        "history_permutation_nulls_by_readout": res["history_nulls"],
        "history_pc1_observed_raw": obs_h,
        "history_pc1_n_null_ge_observed": int((hnull >= obs_h).sum()),
        "history_open_cell": history_summary(d),
        "n_exited_by_3_months": int(d["exited"].sum()),
        "n_with_state_change_readout": int(d["d_severity"].notna().sum()),
        "old_window_matrix": {
            "source": "outputs/exploratory_perturbation_matrix/readout.json",
            "window": "first state row (diagnosis) to + 3 months; outcomes from diagnosis",
            "effect_null_corrected": old["effect_null_corrected"],
            "effect_raw": old["effect_raw"],
        },
        "parameters": {"horizon_months": TAU_MONTHS, "month_days": MONTH_DAYS,
                       "death_window_days": DEATH_WINDOW_DAYS, "n_permutations": NPERM,
                       "seed": SEED, "run_id": PHASE1_RUN,
                       "state_map": "outputs/paper_state_map/state_map_contract.json "
                                    "(frozen 20-coordinate transform)"},
        "n_permutations": NPERM, "seed": SEED,
        "source_sha256": {k: _sha256(p) for k, p in sources.items()},
        "git_commit": commit,
    }
    (OUT / "readout.json").write_text(json.dumps(readout, indent=2) + "\n")

    names = ["state-map PC1 change", "extent change", "ECOG change", "toxicity",
             "first-line PFS", "death by 12 mo", "OS"]
    labels = ["EGFR status", "Current condition", "History", "Treatment"]
    oldM = np.array(old["effect_null_corrected"], float)
    lines = ["# Perturbation-response matrix measured from the treatment decision", "",
             f"{len(d)} futures; 2 x 5 x 3 x 4 design; {int(d['exited'].sum())} died within "
             "3 months of the decision (state-change readouts missing for them).", "",
             "Cells: null-corrected effect (within-cell SD units). old = diagnosis-origin "
             "window of the exploratory producer; new = decision-aligned.", "",
             "| input | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for i, lab in enumerate(labels):
        cells = [f"{oldM[i, j]:.2f} -> {res['M'][i, j]:.2f}" for j in range(len(names))]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += ["", "## History level, state-map PC1 change", "",
              "| level | wording | cutoff (days) | n | exited | changed | mean change |",
              "|---|---|---|---|---|---|---|"]
    for h, s in history_summary(d).items():
        lines.append(f"| {h} | {s['wording']} | {s['decision_cutoff_days'][0]:.0f} | "
                     f"{s['n_futures']} | {s['n_exited_by_3_months']} | {s['n_pc1_changed']} | "
                     f"{s['mean_pc1_change']:.3f} |")
    lines += ["", f"History x PC1: observed raw spread {obs_h:.2f}; "
              f"{int((hnull >= obs_h).sum())} of {NPERM} shuffles at or above it."]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    return readout


def main() -> int:
    d = load_futures()
    res = effect_matrix(d)
    write_readout(d, res)
    print((OUT / "readout.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
