"""Focused tests for the finite-horizon remission-attractor figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_paper_egfr_remission_attractor as attractor  # noqa: E402


def test_clinical_remission_is_raw_feature_and_exit_gated() -> None:
    assert attractor.clinical_remission(
        first_line_active=True,
        tumour_burden=1.0,
        ecog=1.0,
        symptoms=1.0,
    )
    assert not attractor.clinical_remission(
        first_line_active=False,
        tumour_burden=1.0,
        ecog=1.0,
        symptoms=1.0,
    )
    assert not attractor.clinical_remission(
        first_line_active=True,
        tumour_burden=2.0,
        ecog=1.0,
        symptoms=1.0,
    )


def test_state_at_day_does_not_read_a_future_snapshot() -> None:
    course = pd.DataFrame(
        {
            "days_from_diagnosis": [10.0, 20.0],
            "snapshot_index": [1, 2],
            "state": ["observed", "future"],
        }
    )

    assert attractor.state_at_day(course, 15.0)["state"] == "observed"


def test_dwell_requires_every_sampled_state_in_interval() -> None:
    rows = []
    patterns = {
        "persistent": [True, True, True],
        "temporary_escape": [True, False, True],
        "not_captured": [False, True, True],
    }
    for trajectory_id, values in patterns.items():
        for horizon, remission in zip((3.0, 4.5, 6.0), values, strict=True):
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "context": "EGFR_1L",
                    "arm": "chemo",
                    "horizon_months": horizon,
                    "clinical_remission": remission,
                }
            )

    result = attractor.dwell_summary(
        pd.DataFrame(rows),
        start_months=3.0,
        end_months=6.0,
    )
    row = result[
        result["context"].eq("EGFR_1L") & result["arm"].eq("chemo")
    ].iloc[0]

    assert row["n_in_remission_at_start"] == 2
    assert row["n_continuously_in_remission"] == 1
    assert row["continuous_dwell_fraction"] == pytest.approx(0.5)


def test_capture_excludes_anchors_already_in_remission() -> None:
    table = pd.DataFrame(
        {
            "trajectory_id": ["already", "captured", "not_captured"] * 2,
            "context": ["EGFR_1L"] * 6,
            "arm": ["chemo"] * 6,
            "horizon_months": [0.0] * 3 + [3.0] * 3,
            "clinical_remission": [True, False, False, True, True, False],
            "first_line_active": [True] * 6,
        }
    )

    row = attractor.capture_summary(table).iloc[0]

    assert row["n_nonremission_anchors"] == 2
    assert row["n_captured"] == 1
    assert row["capture_fraction"] == pytest.approx(0.5)
