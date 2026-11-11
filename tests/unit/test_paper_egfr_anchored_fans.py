"""Focused tests for the EGFR anchored-fan paper figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_paper_egfr_anchored_fans as fans  # noqa: E402


def test_anchor_position_carries_last_observed_state() -> None:
    """The supportive anchor must not interpolate into its generated future."""
    course = pd.DataFrame(
        {
            "days_from_diagnosis": [122.0, 140.0],
            "snapshot_index": [1, 2],
            "map_x": [1.0, 10.0],
            "map_y": [2.0, 20.0],
        }
    )

    assert fans.anchor_position(course, 123.0) == (1.0, 2.0)
    assert fans.position_at_day(course, 123.0) != (1.0, 2.0)


def test_empirical_survival_uses_every_uncensored_endpoint() -> None:
    x, y = fans.empirical_survival(pd.Series([1.0, 2.0, 3.0]))

    assert x[0] == 0.0
    assert y[0] == 1.0
    assert y[-1] == 0.0
    assert len(x) == len(y) == 7


def test_residence_summary_reports_fixed_horizon_retention() -> None:
    endpoints = pd.DataFrame(
        {
            "context": ["EGFR_1L"] * 4,
            "arm_key": ["osimertinib"] * 4,
            "endpoint_mo": [2.0, 8.0, 13.0, 20.0],
        }
    )

    row = fans.residence_summary(
        endpoints,
        arms=("osimertinib",),
    ).iloc[0]

    assert row["median_endpoint_mo"] == pytest.approx(10.5)
    assert row["remain_at_6mo"] == pytest.approx(0.75)
    assert row["remain_at_12mo"] == pytest.approx(0.5)
