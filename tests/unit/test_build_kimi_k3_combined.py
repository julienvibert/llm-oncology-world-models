"""Tests for scripts/explore/build_kimi_k3_combined.py on synthetic fixtures (no run folders)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "explore" / "build_kimi_k3_combined.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_kimi_k3_combined", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _split_mock(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    """A synthetic full run split into a main run (f01, f02) and a top-up run (f03 renamed f01)."""
    full_run, manifest_path = mod.scorer._mock_run(tmp_path)
    main_run = tmp_path / "main"
    retry_run = tmp_path / "retry"
    for name in mod.DERIVATION_FILES:
        df = pd.read_csv(full_run / "derivation" / name)
        is_f03 = df["trajectory_id"].str.endswith("__f03")
        (main_run / "derivation").mkdir(parents=True, exist_ok=True)
        (retry_run / "derivation").mkdir(parents=True, exist_ok=True)
        df[~is_f03].to_csv(main_run / "derivation" / name, index=False)
        retry = df[is_f03].copy()
        retry["trajectory_id"] = retry["trajectory_id"].str.replace("__f03", "__f01", regex=False)
        retry.to_csv(retry_run / "derivation" / name, index=False)
    return full_run, main_run, retry_run, json.loads(manifest_path.read_text())


def test_merge_suffixes_retry_ids_and_keeps_main_first(tmp_path: Path) -> None:
    _, main_run, retry_run, _ = _split_mock(tmp_path)
    counts = mod.merge_derivations(main_run, retry_run, tmp_path / "merged")
    merged = pd.read_csv(tmp_path / "merged" / "derivation" / "endpoint_rows.csv")
    n_main = len(pd.read_csv(main_run / "derivation" / "endpoint_rows.csv"))
    assert counts["endpoint_rows.csv"] == len(merged)
    assert not merged["trajectory_id"].iloc[:n_main].str.endswith(mod.RETRY_SUFFIX).any()
    assert merged["trajectory_id"].iloc[n_main:].str.endswith(mod.RETRY_SUFFIX).all()
    assert merged["trajectory_id"].is_unique


def test_merge_refuses_id_collision(tmp_path: Path) -> None:
    _, main_run, _, _ = _split_mock(tmp_path)
    with pytest.raises(SystemExit, match="collision"):
        mod.merge_derivations(main_run, main_run, tmp_path / "merged", suffix="")


def test_build_combined_matches_scoring_the_unsplit_run_values(tmp_path: Path) -> None:
    full_run, main_run, retry_run, manifest = _split_mock(tmp_path)
    out = tmp_path / "out"
    mod.build_combined(main_run, retry_run, out, manifest)
    assert sorted(p.name for p in out.iterdir()) == sorted(mod.OUTPUT_FILES)
    ref = tmp_path / "ref"
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(manifest))
    mod.scorer.score_phase1(full_run, mp, ref)
    # Same futures, different IDs: the gate scores and medians must agree.
    for name in ("phase1_gate_scores.csv", "phase1_context_arm_medians.csv", "phase1_cell_arm_medians.csv"):
        assert (out / name).read_bytes() == (ref / name).read_bytes(), name


def test_compare_dirs_detects_identical_and_different(tmp_path: Path) -> None:
    _, main_run, retry_run, manifest = _split_mock(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    mod.build_combined(main_run, retry_run, a, manifest)
    mod.build_combined(main_run, retry_run, b, manifest)
    assert all(mod.compare_dirs(a, b).values())
    (b / "phase1_gate_readout.md").write_text("changed\n")
    result = mod.compare_dirs(a, b)
    assert not result["phase1_gate_readout.md"]
    shutil.rmtree(b)
    assert not any(mod.compare_dirs(a, b).values())


def test_check_cells_match_config(tmp_path: Path) -> None:
    cells = [{"id": "c1", "condition_id": "x", "policy_id": "p", "baseline_seed": {"a": 1}, "operator_constraints": "t"}]
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"cells": cells}))
    assert mod.check_cells_match_config(cells, cfg) == 1
    bad = [dict(cells[0], baseline_seed={"a": 2})]
    cfg.write_text(yaml.safe_dump({"cells": bad}))
    with pytest.raises(SystemExit, match="differ"):
        mod.check_cells_match_config(cells, cfg)


def test_compare_dirs_reports_missing_reference_as_not_compared(tmp_path: Path) -> None:
    _, main_run, retry_run, manifest = _split_mock(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    mod.build_combined(main_run, retry_run, a, manifest)
    mod.build_combined(main_run, retry_run, b, manifest)
    (b / "phase1_gate_readout.md").unlink()
    result = mod.compare_dirs(a, b)
    assert result["phase1_gate_readout.md"] is None
    assert all(v for k, v in result.items() if k != "phase1_gate_readout.md")
