#!/usr/bin/env python3
"""Score the v1 Phase-1 EGFR interventional gate.

This is deliberately narrow and mechanical. It scores the pre-registered gate in
docs/interventional_run_spec_v1.md from already-generated derivation CSVs; it
does not generate trajectories.

Endpoint definitions:
  * active forced arms: endpoint_rows.csv:pfs_line_1_days
  * supportive/BSC: first post-cutoff generated progression minus cutoff;
    fallback to death time when no generated progression precedes death

Examples:
  uv run python scripts/explore/score_interventional_phase1.py --self-test

  uv run python scripts/build_rev5_derivation.py --run-root runs/<phase1_run> --force
  uv run python scripts/explore/score_interventional_phase1.py \\
    --run-root runs/<phase1_run> \\
    --manifest /private/tmp/interventional_phase1_egfr_v1c/arm_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = ROOT / "outputs" / "exploratory_interventional_phase1"
DEFAULT_MANIFEST = Path("/private/tmp/interventional_phase1_egfr_v1c/arm_manifest.json")
MONTH = 30.4
ACTIVE_ARMS = ("osimertinib", "chemo", "pembrolizumab")
SUPPORTIVE = "supportive"
CONTEXT_MUT = "EGFR_1L"
CONTEXT_WT = "EGFR_WT_1L"


def _accepted_mask(df: pd.DataFrame) -> pd.Series:
    if "accepted" not in df:
        return pd.Series(True, index=df.index)
    return df["accepted"].astype(str).str.lower().isin({"true", "1", "yes"})


def _load_manifest(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    manifest = json.loads(path.read_text())
    policy_map: dict[str, dict] = {}
    condition_map: dict[str, dict] = {}
    for row in manifest.values():
        policy_map[row["policy_id"]] = {
            "context": row["context"],
            "arm_key": row["arm_key"],
            "role": row.get("role", ""),
        }
        condition_map[row["condition_id"]] = {
            "context": row["context"],
            "state_id": row.get("state_id"),
            "state_name": row.get("state_name"),
            "history_id": row.get("history_id"),
            "history_name": row.get("history_name"),
            "tumour_index_current": row.get("tumour_index_current"),
            "host_index": row.get("host_index"),
            "decision_cutoff_date": row.get("decision_cutoff_date"),
            "decision_cutoff_days": row.get("decision_cutoff_days"),
        }
    return policy_map, condition_map


def _attach_manifest(df: pd.DataFrame, policy_map: dict[str, dict], condition_map: dict[str, dict]) -> pd.DataFrame:
    d = df[df["policy_id"].isin(policy_map) & df["condition_id"].isin(condition_map)].copy()
    d["context"] = d["policy_id"].map(lambda p: policy_map[p]["context"])
    d["arm_key"] = d["policy_id"].map(lambda p: policy_map[p]["arm_key"])
    d["role"] = d["policy_id"].map(lambda p: policy_map[p].get("role", ""))
    d["state_id"] = d["condition_id"].map(lambda c: condition_map[c].get("state_id"))
    d["state_name"] = d["condition_id"].map(lambda c: condition_map[c].get("state_name"))
    d["history_id"] = d["condition_id"].map(lambda c: condition_map[c].get("history_id"))
    d["history_name"] = d["condition_id"].map(lambda c: condition_map[c].get("history_name"))
    d["tumour_index_current"] = d["condition_id"].map(lambda c: condition_map[c].get("tumour_index_current"))
    d["host_index"] = d["condition_id"].map(lambda c: condition_map[c].get("host_index"))
    d["decision_cutoff_date"] = d["condition_id"].map(lambda c: condition_map[c].get("decision_cutoff_date"))
    d["decision_cutoff_days"] = d["condition_id"].map(lambda c: condition_map[c].get("decision_cutoff_days"))
    d["model"] = d["model_short"]
    return d


def load_phase1_tables(run_root: Path, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    derivation = run_root / "derivation"
    endpoint_path = derivation / "endpoint_rows.csv"
    event_path = derivation / "event_rows.csv"
    if not endpoint_path.exists() or not event_path.exists():
        raise FileNotFoundError(
            f"missing derivation CSVs under {derivation}; run build_rev5_derivation.py first"
        )
    policy_map, condition_map = _load_manifest(manifest_path)

    endpoint = pd.read_csv(endpoint_path)
    endpoint = endpoint[_accepted_mask(endpoint)].copy()
    endpoint = _attach_manifest(endpoint, policy_map, condition_map)
    event = pd.read_csv(event_path)
    event = event[_accepted_mask(event)].copy()
    event = _attach_manifest(event, policy_map, condition_map)
    return endpoint, event


def _prefix_cutoff_days(event: pd.DataFrame) -> pd.DataFrame:
    if event.empty:
        return pd.DataFrame(columns=["trajectory_id", "cutoff_days"])
    e = event.copy()
    e["days_from_diagnosis"] = pd.to_numeric(e["days_from_diagnosis"], errors="coerce")
    fixed = e[e["event_source"].astype(str).eq("observed_prefix")]
    if fixed.empty:
        return pd.DataFrame(columns=["trajectory_id", "cutoff_days"])
    return (
        fixed.groupby("trajectory_id", as_index=False)["days_from_diagnosis"]
        .max()
        .rename(columns={"days_from_diagnosis": "cutoff_days"})
    )


def supportive_ttp_table(endpoint: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    """Per supportive trajectory: post-cutoff time-to-progression in days.

    If no generated progression is present, use death/OS time as a conservative
    disease-worsening fallback and flag it. Times are measured from the current
    decision point, not from diagnosis.
    """
    supp = endpoint[endpoint["arm_key"].eq(SUPPORTIVE)].copy()
    if supp.empty:
        return pd.DataFrame()
    supp["os_days"] = pd.to_numeric(supp["os_days"], errors="coerce")
    if "decision_cutoff_days" in supp:
        supp["cutoff_days"] = pd.to_numeric(supp["decision_cutoff_days"], errors="coerce")
    else:
        supp["cutoff_days"] = np.nan
    if supp["cutoff_days"].isna().any():
        cutoff = _prefix_cutoff_days(event)
        supp = supp.merge(cutoff, on="trajectory_id", how="left", suffixes=("", "_from_prefix"))
        missing_cutoff = supp["cutoff_days"].isna()
        fallback = supp["cutoff_days_from_prefix"].fillna(0.0)
        supp["cutoff_days"] = supp["cutoff_days"].where(~missing_cutoff, fallback)
        supp = supp.drop(columns=["cutoff_days_from_prefix"])

    ev = event[event["arm_key"].eq(SUPPORTIVE)].copy()
    ev["days_from_diagnosis"] = pd.to_numeric(ev["days_from_diagnosis"], errors="coerce")
    ev = ev[~ev["event_source"].astype(str).eq("observed_prefix")]
    prog = ev[ev["event_type"].eq("progression")]
    if not prog.empty:
        prog = prog.merge(supp[["trajectory_id", "cutoff_days"]], on="trajectory_id", how="inner")
        prog = prog[prog["days_from_diagnosis"] > prog["cutoff_days"]]
        first_prog = (
            prog.groupby("trajectory_id", as_index=False)["days_from_diagnosis"]
            .min()
            .rename(columns={"days_from_diagnosis": "first_progression_days_from_dx"})
        )
    else:
        first_prog = pd.DataFrame(columns=["trajectory_id", "first_progression_days_from_dx"])
    out = supp.merge(first_prog, on="trajectory_id", how="left")
    out["used_death_fallback"] = out["first_progression_days_from_dx"].isna()
    out["supportive_ttp_days"] = np.where(
        out["used_death_fallback"],
        out["os_days"] - out["cutoff_days"],
        out["first_progression_days_from_dx"] - out["cutoff_days"],
    )
    out["supportive_ttp_mo"] = out["supportive_ttp_days"] / MONTH
    keep = [
        "trajectory_id",
        "context",
        "condition_id",
        "state_id",
        "history_id",
        "arm_key",
        "model",
        "cutoff_days",
        "supportive_ttp_days",
        "supportive_ttp_mo",
        "used_death_fallback",
    ]
    return out[keep]


def phase1_endpoint_table(endpoint: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    active = endpoint[endpoint["arm_key"].isin(ACTIVE_ARMS)].copy()
    active["pfs_line_1_days"] = pd.to_numeric(active["pfs_line_1_days"], errors="coerce")
    active["pfs_mo"] = active["pfs_line_1_days"] / MONTH
    active_rows = active[
        [
            "trajectory_id",
            "context",
            "condition_id",
            "state_id",
            "state_name",
            "history_id",
            "history_name",
            "arm_key",
            "role",
            "model",
            "tumour_index_current",
            "host_index",
            "pfs_line_1_days",
            "pfs_mo",
        ]
    ].copy()
    active_rows["endpoint_kind"] = "forced_line_pfs"
    active_rows["endpoint_days"] = active_rows["pfs_line_1_days"]
    active_rows["endpoint_mo"] = active_rows["pfs_mo"]

    supp = supportive_ttp_table(endpoint, event)
    if not supp.empty:
        meta_cols = [
            "trajectory_id",
            "state_name",
            "history_name",
            "role",
            "tumour_index_current",
            "host_index",
        ]
        supp_meta = endpoint[endpoint["arm_key"].eq(SUPPORTIVE)][meta_cols].drop_duplicates("trajectory_id")
        supp = supp.merge(supp_meta, on="trajectory_id", how="left")
        supp["endpoint_kind"] = np.where(
            supp["used_death_fallback"],
            "supportive_death_fallback",
            "supportive_first_progression",
        )
        supp["endpoint_days"] = supp["supportive_ttp_days"]
        supp["endpoint_mo"] = supp["supportive_ttp_mo"]
        supp["pfs_line_1_days"] = np.nan
        supp["pfs_mo"] = np.nan
        supp["role"] = supp["role"].fillna("no_systemic_control")
        supp = supp[
            [
                "trajectory_id",
                "context",
                "condition_id",
                "state_id",
                "state_name",
                "history_id",
                "history_name",
                "arm_key",
                "role",
                "model",
                "tumour_index_current",
                "host_index",
                "pfs_line_1_days",
                "pfs_mo",
                "endpoint_kind",
                "endpoint_days",
                "endpoint_mo",
                "used_death_fallback",
            ]
        ]
    else:
        supp = pd.DataFrame(columns=[*active_rows.columns, "used_death_fallback"])

    active_rows["used_death_fallback"] = False
    return pd.concat([active_rows, supp], ignore_index=True)


def median_tables(endpoint_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "context",
        "condition_id",
        "state_id",
        "state_name",
        "history_id",
        "history_name",
        "arm_key",
        "model",
    ]
    cell = (
        endpoint_long.groupby(group_cols, dropna=False)
        .agg(
            median_endpoint_mo=("endpoint_mo", "median"),
            median_endpoint_days=("endpoint_days", "median"),
            n=("endpoint_mo", "count"),
            death_fallback_n=("used_death_fallback", "sum"),
        )
        .reset_index()
    )
    context = (
        endpoint_long.groupby(["context", "arm_key", "model"], dropna=False)
        .agg(
            median_endpoint_mo=("endpoint_mo", "median"),
            median_endpoint_days=("endpoint_days", "median"),
            n=("endpoint_mo", "count"),
            death_fallback_n=("used_death_fallback", "sum"),
        )
        .reset_index()
    )
    return cell, context


def _median_value(context_table: pd.DataFrame, *, context: str, arm: str, model: str) -> float:
    d = context_table[
        context_table["context"].eq(context)
        & context_table["arm_key"].eq(arm)
        & context_table["model"].eq(model)
    ]
    if d.empty:
        return float("nan")
    return float(d["median_endpoint_mo"].iloc[0])


def _paired_delta(cell_table: pd.DataFrame, *, context: str, model: str) -> float:
    d = cell_table[cell_table["context"].eq(context) & cell_table["model"].eq(model)]
    wide = d.pivot_table(
        index=["state_id", "history_id"],
        columns="arm_key",
        values="median_endpoint_mo",
        aggfunc="first",
    )
    if "osimertinib" not in wide or "chemo" not in wide:
        return float("nan")
    return float((wide["osimertinib"] - wide["chemo"]).median())


def score_gate(cell_table: pd.DataFrame, context_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(context_table["model"].dropna().unique()):
        mut_osi = _median_value(context_table, context=CONTEXT_MUT, arm="osimertinib", model=model)
        mut_chemo = _median_value(context_table, context=CONTEXT_MUT, arm="chemo", model=model)
        mut_bsc = _median_value(context_table, context=CONTEXT_MUT, arm=SUPPORTIVE, model=model)
        wt_osi = _median_value(context_table, context=CONTEXT_WT, arm="osimertinib", model=model)
        wt_chemo = _median_value(context_table, context=CONTEXT_WT, arm="chemo", model=model)
        wt_pembro = _median_value(context_table, context=CONTEXT_WT, arm="pembrolizumab", model=model)
        wt_bsc = _median_value(context_table, context=CONTEXT_WT, arm=SUPPORTIVE, model=model)
        mut_best_active = np.nanmax([mut_osi, mut_chemo, _median_value(context_table, context=CONTEXT_MUT, arm="pembrolizumab", model=model)])
        wt_best_active = np.nanmax([wt_osi, wt_chemo, wt_pembro])
        mut_paired_delta = _paired_delta(cell_table, context=CONTEXT_MUT, model=model)
        wt_paired_delta = _paired_delta(cell_table, context=CONTEXT_WT, model=model)

        egfr_mut_osi_gt_chemo = bool(mut_osi > mut_chemo)
        egfr_mut_osi_gt_bsc = bool(mut_osi > mut_bsc)
        egfr_wt_osi_not_top = bool(wt_osi < np.nanmax([wt_chemo, wt_pembro]))
        bsc_control = bool(mut_bsc < mut_best_active and wt_bsc < wt_best_active)
        no_impossible_reversal = bool(mut_paired_delta > wt_paired_delta)
        checks = [
            egfr_mut_osi_gt_chemo,
            egfr_mut_osi_gt_bsc,
            egfr_wt_osi_not_top,
            bsc_control,
            no_impossible_reversal,
        ]
        rows.append(
            {
                "model": model,
                "phase1_pass": bool(all(checks)),
                "checks_passed": int(sum(checks)),
                "checks_total": len(checks),
                "egfr_mut_osi_gt_chemo": egfr_mut_osi_gt_chemo,
                "egfr_mut_osi_gt_bsc_ttp": egfr_mut_osi_gt_bsc,
                "egfr_wt_osi_not_top": egfr_wt_osi_not_top,
                "bsc_control": bsc_control,
                "no_impossible_reversal": no_impossible_reversal,
                "egfr_mut_osi_mo": mut_osi,
                "egfr_mut_chemo_mo": mut_chemo,
                "egfr_mut_supportive_ttp_mo": mut_bsc,
                "egfr_wt_osi_mo": wt_osi,
                "egfr_wt_chemo_mo": wt_chemo,
                "egfr_wt_pembro_mo": wt_pembro,
                "egfr_wt_supportive_ttp_mo": wt_bsc,
                "egfr_mut_paired_osi_minus_chemo_mo": mut_paired_delta,
                "egfr_wt_paired_osi_minus_chemo_mo": wt_paired_delta,
            }
        )
    return pd.DataFrame(rows)


def _write_readout(out_dir: Path, gate: pd.DataFrame, supportive: pd.DataFrame) -> None:
    lines = ["# Interventional Phase 1 gate readout", ""]
    for row in gate.to_dict("records"):
        verdict = "PASS" if row["phase1_pass"] else "FAIL"
        lines.append(f"## {row['model']} — {verdict}")
        lines.append("")
        for key in [
            "egfr_mut_osi_gt_chemo",
            "egfr_mut_osi_gt_bsc_ttp",
            "egfr_wt_osi_not_top",
            "bsc_control",
            "no_impossible_reversal",
        ]:
            lines.append(f"- `{key}`: {bool(row[key])}")
        lines.append("")
        lines.append(
            "Median endpoints (months): "
            f"EGFR-mut osi {row['egfr_mut_osi_mo']:.1f}, chemo {row['egfr_mut_chemo_mo']:.1f}, "
            f"BSC-TTP {row['egfr_mut_supportive_ttp_mo']:.1f}; "
            f"EGFR-WT osi {row['egfr_wt_osi_mo']:.1f}, chemo {row['egfr_wt_chemo_mo']:.1f}, "
            f"pembro {row['egfr_wt_pembro_mo']:.1f}, BSC-TTP {row['egfr_wt_supportive_ttp_mo']:.1f}."
        )
        lines.append("")
    if not supportive.empty:
        fallback = int(supportive["used_death_fallback"].sum())
        lines.append(f"Supportive death-fallback trajectories: {fallback}/{len(supportive)}.")
        lines.append("")
    (out_dir / "phase1_gate_readout.md").write_text("\n".join(lines) + "\n")


def score_phase1(run_root: Path, manifest_path: Path, out_dir: Path) -> pd.DataFrame:
    endpoint, event = load_phase1_tables(run_root, manifest_path)
    endpoint_long = phase1_endpoint_table(endpoint, event)
    supportive = supportive_ttp_table(endpoint, event)
    cell, context = median_tables(endpoint_long)
    gate = score_gate(cell, context)

    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_long.to_csv(out_dir / "phase1_endpoint_long.csv", index=False)
    supportive.to_csv(out_dir / "phase1_supportive_ttp_trajectory.csv", index=False)
    cell.to_csv(out_dir / "phase1_cell_arm_medians.csv", index=False)
    context.to_csv(out_dir / "phase1_context_arm_medians.csv", index=False)
    gate.to_csv(out_dir / "phase1_gate_scores.csv", index=False)
    _write_readout(out_dir, gate, supportive)
    return gate


def _mock_run(root: Path, *, wrong_target_failure: bool = False) -> tuple[Path, Path]:
    run = root / ("bad" if wrong_target_failure else "good")
    deriv = run / "derivation"
    deriv.mkdir(parents=True)
    manifest = {}
    endpoint_rows = []
    event_rows = []

    def cell(ctx: str, state: str, history: str, arm: str, role: str) -> str:
        cid = f"{ctx}__{state}__{history}"
        cell_id = f"{cid}__{arm}"
        decision_cutoff_days = 0 if history == "H0" else 32
        manifest[cell_id] = {
            "arm_key": arm,
            "policy_id": f"{ctx}__{arm}",
            "context": ctx,
            "condition_id": cid,
            "state_id": state,
            "state_name": state,
            "history_id": history,
            "history_name": history,
            "role": role,
            "tumour_index_current": 5,
            "host_index": 1,
            "decision_cutoff_date": "2018-03-01" if history == "H0" else "2018-04-02",
            "decision_cutoff_days": decision_cutoff_days,
        }
        return cid

    roles = {
        CONTEXT_MUT: {"osimertinib": "matched_standard", "chemo": "active_alternative", "pembrolizumab": "weak_io", SUPPORTIVE: "control"},
        CONTEXT_WT: {"osimertinib": "wrong_target", "chemo": "active_standard", "pembrolizumab": "weak_io", SUPPORTIVE: "control"},
    }
    pfs_good = {
        CONTEXT_MUT: {"osimertinib": 300, "chemo": 150, "pembrolizumab": 75},
        CONTEXT_WT: {"osimertinib": 60 if not wrong_target_failure else 240, "chemo": 160, "pembrolizumab": 110},
    }
    for ctx in [CONTEXT_MUT, CONTEXT_WT]:
        for state in ["S1", "S2"]:
            for history, cutoff in [("H0", 0), ("H1", 31)]:
                for arm, role in roles[ctx].items():
                    cid = cell(ctx, state, history, arm, role)
                    for future in [1, 2, 3]:
                        tid = f"{cid}__{arm}__claude_sonnet5__f{future:02d}"
                        pfs = np.nan if arm == SUPPORTIVE else pfs_good[ctx][arm] + future
                        os_days = (cutoff + 85 + future) if arm == SUPPORTIVE and future == 3 else 700
                        endpoint_rows.append(
                            {
                                "trajectory_id": tid,
                                "condition_id": cid,
                                "policy_id": f"{ctx}__{arm}",
                                "model_short": "claude_sonnet5",
                                "future_index": future,
                                "accepted": True,
                                "os_days": os_days,
                                "pfs_line_1_days": pfs,
                            }
                        )
                        if cutoff:
                            event_rows.append(
                                {
                                    "trajectory_id": tid,
                                    "condition_id": cid,
                                    "policy_id": f"{ctx}__{arm}",
                                    "model_short": "claude_sonnet5",
                                    "future_index": future,
                                    "accepted": True,
                                    "event_type": "progression",
                                    "days_from_diagnosis": cutoff,
                                    "event_source": "observed_prefix",
                                }
                            )
                        if arm == SUPPORTIVE and future != 3:
                            event_rows.append(
                                {
                                    "trajectory_id": tid,
                                    "condition_id": cid,
                                    "policy_id": f"{ctx}__{arm}",
                                    "model_short": "claude_sonnet5",
                                    "future_index": future,
                                    "accepted": True,
                                    "event_type": "progression",
                                    "days_from_diagnosis": cutoff + 60 + future,
                                    "event_source": "generated",
                                }
                            )
    manifest_path = root / ("manifest_bad.json" if wrong_target_failure else "manifest_good.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(endpoint_rows).to_csv(deriv / "endpoint_rows.csv", index=False)
    pd.DataFrame(event_rows).to_csv(deriv / "event_rows.csv", index=False)
    return run, manifest_path


def _self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good_run, good_manifest = _mock_run(root)
        bad_run, bad_manifest = _mock_run(root, wrong_target_failure=True)
        good_gate = score_phase1(good_run, good_manifest, root / "out_good")
        bad_gate = score_phase1(bad_run, bad_manifest, root / "out_bad")
        good = good_gate.iloc[0]
        bad = bad_gate.iloc[0]
        assert bool(good["phase1_pass"]), good
        assert bool(good["egfr_mut_osi_gt_chemo"]), good
        assert bool(good["egfr_mut_osi_gt_bsc_ttp"]), good
        assert bool(good["egfr_wt_osi_not_top"]), good
        assert bool(good["bsc_control"]), good
        assert bool(good["no_impossible_reversal"]), good
        assert not bool(bad["phase1_pass"]), bad
        assert not bool(bad["egfr_wt_osi_not_top"]), bad
        supp = pd.read_csv(root / "out_good" / "phase1_supportive_ttp_trajectory.csv")
        assert int(supp["used_death_fallback"].sum()) == 8, supp["used_death_fallback"].sum()
        h1 = supp[supp["history_id"].eq("H1")]
        assert set(h1["cutoff_days"]) == {32}, h1[["history_id", "cutoff_days"]].drop_duplicates()
    print("[self-test] PASS — Phase-1 gate is mechanical, BSC TTP is computable, WT failure fails")
    print(good_gate.round(2).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.run_root is None:
        raise SystemExit("--run-root is required unless --self-test is used")
    gate = score_phase1(args.run_root, args.manifest, args.out_dir)
    verdicts = ", ".join(f"{r.model}={'PASS' if r.phase1_pass else 'FAIL'}" for r in gate.itertuples())
    print(f"[phase1] {verdicts}")
    print(f"[done] wrote {args.out_dir / 'phase1_gate_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
