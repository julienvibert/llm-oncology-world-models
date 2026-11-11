from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_precision_ladder_paired_views as ladder  # noqa: E402


def test_locf_row_never_uses_a_future_state() -> None:
    course = pd.DataFrame(
        {
            "days_from_diagnosis": [0.0, 10.0, 20.0],
            "snapshot_index": [0, 1, 2],
            "pc1": [0.0, 1.0, 2.0],
        }
    )

    assert ladder.locf_row(course, 19.9)["pc1"] == 1.0


def test_peer_mask_is_state_first_and_excludes_heldout_pair() -> None:
    outcomes = pd.DataFrame(
        {
            "arm": ["osimertinib", "osimertinib", "chemo", "chemo"],
            "context": ["EGFR_1L"] * 4,
            "state": ["S2"] * 4,
            "history": ["H1", "H0", "H1", "H0"],
            "future_index": [1, 2, 1, 2],
        }
    )
    exemplar = pd.Series(
        {
            "context": "EGFR_1L",
            "state": "S2",
            "history": "H1",
            "future_index": 1,
        }
    )

    broad = ladder.peer_mask(outcomes, exemplar, ["state"], arm="chemo")
    exact = ladder.peer_mask(
        outcomes,
        exemplar,
        ["state", "context", "history"],
        arm="chemo",
    )

    assert broad.tolist() == [False, False, False, True]
    assert exact.sum() == 0


def test_rmst_is_restricted_to_the_declared_horizon() -> None:
    assert ladder.rmst(np.asarray([6.0, 30.0]), horizon_months=24.0) == 15.0


def test_bootstrap_rmst_contrast_is_deterministic() -> None:
    treatment = np.asarray([12.0, 24.0, 30.0])
    control = np.asarray([6.0, 12.0, 18.0])

    first = ladder.bootstrap_rmst_contrast(
        treatment,
        control,
        n_bootstrap=100,
        seed=7,
    )
    second = ladder.bootstrap_rmst_contrast(
        treatment,
        control,
        n_bootstrap=100,
        seed=7,
    )

    assert first == second
    assert first[0] == 8.0
