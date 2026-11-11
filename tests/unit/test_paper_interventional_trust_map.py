"""Focused tests for the paper-facing interventional trust map."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_paper_interventional_trust_map as trust  # noqa: E402


def _sentinel_frame() -> pd.DataFrame:
    rows = []
    for axis, value in (("static", 100.0), ("dynamic", 25.0)):
        row: dict[str, object] = {
            "context_id": axis,
            "label": axis,
            "axis": axis,
            "core_signal": "direction",
        }
        for model in trust.SENTINEL_MODELS:
            row[f"{model}_primary_pct"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_axis_summary_keeps_static_and_dynamic_separate() -> None:
    summary = trust.axis_summary(_sentinel_frame())
    first = summary[summary["model"].eq(trust.SENTINEL_MODELS[0])].set_index("axis")

    assert first.loc["static", "mean_primary_pct"] == pytest.approx(100.0)
    assert first.loc["dynamic", "mean_primary_pct"] == pytest.approx(25.0)


def test_load_sentinel_rejects_out_of_range_scores(tmp_path: Path) -> None:
    frame = _sentinel_frame()
    frame.loc[0, f"{trust.SENTINEL_MODELS[0]}_primary_pct"] = 101.0
    path = tmp_path / "sentinel.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="complete percentages"):
        trust.load_sentinel(path)


def test_load_phase1_preserves_locked_model_order(tmp_path: Path) -> None:
    paths = []
    for index, model in enumerate(reversed(trust.PHASE1_ORDER)):
        row: dict[str, object] = {
            "model": model,
            "phase1_pass": index != 0,
            "checks_passed": 5 if index != 0 else 4,
            "checks_total": 5,
            "egfr_mut_osi_mo": 20.0,
            "egfr_wt_osi_mo": 5.0,
        }
        row.update({key: True for key, _ in trust.PHASE1_CHECKS})
        path = tmp_path / f"{model}.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        paths.append(path)

    loaded = trust.load_phase1(tuple(paths))

    assert loaded["model"].tolist() == list(trust.PHASE1_ORDER)
