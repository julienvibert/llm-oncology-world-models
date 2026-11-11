"""Focused tests for the survival-to-occupancy identity."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_survival_in_state_space as survival_map  # noqa: E402


def _trajectories() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_gid": ["c1", "c2", "o1", "o2"],
            "pair_id": ["p1", "p2", "p1", "p2"],
            "arm": ["chemo", "chemo", "osimertinib", "osimertinib"],
            "anchor_pc1": [0.0, 1.0, 0.0, 1.0],
            "anchor_pc2": [0.0, 1.0, 0.0, 1.0],
            "pfs_months": [2.0, 8.0, 6.0, 10.0],
            "restricted_months": [2.0, 6.0, 6.0, 6.0],
        }
    )


def _intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_gid": ["c1", "c2", "c2", "o1", "o2"],
            "arm": ["chemo", "chemo", "chemo", "osimertinib", "osimertinib"],
            "start_month": [0.0, 0.0, 3.0, 0.0, 0.0],
            "end_month": [2.0, 3.0, 6.0, 6.0, 6.0],
            "dwell_months": [2.0, 3.0, 3.0, 6.0, 6.0],
            "pc1": [0.0, 0.0, 1.0, 0.0, 1.0],
            "pc2": [0.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


def test_time_conservation_is_trajectory_specific() -> None:
    audit = survival_map.time_conservation_audit(_intervals(), _trajectories())

    assert audit["absolute_error"].max() == pytest.approx(0.0)
    assert audit.set_index("_gid").loc["c2", "integrated_occupancy_months"] == 6.0


def test_occupancy_grid_sum_equals_arm_rmst() -> None:
    exact, _smooth, _x, _y, _table = survival_map.occupancy_grids(
        _intervals(),
        _trajectories(),
        limits=(-0.5, 1.5, -0.5, 1.5),
        bins=4,
    )

    expected = _trajectories().groupby("arm")["restricted_months"].mean()
    assert exact["chemo"].sum() == pytest.approx(expected["chemo"])
    assert exact["osimertinib"].sum() == pytest.approx(expected["osimertinib"])
    assert (exact["osimertinib"] - exact["chemo"]).sum() == pytest.approx(2.0)


def test_paired_rmst_contrast_resamples_pairs_deterministically() -> None:
    first = survival_map.paired_rmst_contrast(
        _trajectories(),
        n_bootstrap=200,
        seed=11,
    )
    second = survival_map.paired_rmst_contrast(
        _trajectories(),
        n_bootstrap=200,
        seed=11,
    )

    assert first == second
    assert first[0] == pytest.approx(2.0)


def test_active_positions_moves_exited_futures_off_map() -> None:
    at_four = survival_map.active_positions(
        _intervals(),
        _trajectories(),
        arm="chemo",
        horizon_months=4.0,
    )

    assert at_four["_gid"].tolist() == ["c2"]


def test_coincident_futures_are_counted_without_jittering_state() -> None:
    active = survival_map.active_positions(
        _intervals(),
        _trajectories(),
        arm="chemo",
        horizon_months=0.0,
    )
    grouped = survival_map.grouped_active_positions(active)

    assert len(active) == 2
    assert grouped[["pc1", "pc2", "n_futures"]].to_dict(orient="records") == [
        {"pc1": 0.0, "pc2": 0.0, "n_futures": 2}
    ]


def test_frame_transitions_distinguish_movement_stability_and_exit() -> None:
    transitions = survival_map.frame_transitions(
        _intervals(),
        _trajectories(),
        arm="chemo",
        start_month=1.0,
        step_months=2.0,
    ).set_index("_gid")

    assert bool(transitions.loc["c1", "exit_next"])
    assert transitions.loc["c2", "dx"] == pytest.approx(1.0)
    assert transitions.loc["c2", "dy"] == pytest.approx(1.0)
    assert not bool(transitions.loc["c2", "no_recorded_change"])


def test_validate_paired_anchors_rejects_post_anchor_mismatch() -> None:
    broken = _trajectories().copy()
    broken.loc[broken["_gid"].eq("o1"), "anchor_pc1"] = 0.2

    with pytest.raises(ValueError, match="identical anchors"):
        survival_map.validate_paired_anchors(broken)


def test_km_coordinates_uses_complete_exit_times() -> None:
    x, survival = survival_map.km_coordinates([2.0, 8.0], 6.0)

    assert x.tolist() == [0.0, 2.0, 6.0]
    assert survival.tolist() == [1.0, 0.5, 0.5]
