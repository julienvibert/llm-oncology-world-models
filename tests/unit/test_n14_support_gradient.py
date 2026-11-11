"""N14 support-gradient helpers: within-cancer tie-aware ranks and thirds, paired gain sign (constructed data only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
_PATH = list(sys.path)
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import n14_support_gradient_v1 as N  # noqa: E402

sys.path[:] = _PATH


def test_rank_is_within_group_and_scaled_by_group_size() -> None:
    v = np.array([0.1, 0.9, 0.5, 5.0, 1.0])
    g = np.array(["A", "A", "A", "B", "B"])
    r = N.within_group_rank(v, g)
    assert r == pytest.approx([1 / 3, 1.0, 2 / 3, 1.0, 0.5])


def test_rank_ties_share_the_average_rank() -> None:
    r = N.within_group_rank(np.array([0.2, 0.2, 0.7, 0.1]), np.array(["A"] * 4))
    assert r[0] == r[1] == pytest.approx(2.5 / 4)


def test_thirds_without_ties_are_equal() -> None:
    v = np.arange(9.0)
    t = N.within_group_thirds(v, np.array(["A"] * 9))
    assert t.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_thirds_keep_ties_together_and_are_formed_within_group() -> None:
    v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 20.0, 30.0])
    g = np.array(["A"] * 6 + ["B"] * 3)
    t = N.within_group_thirds(v, g)
    # four tied lowest values in A share one third, even though that makes it larger than 2
    assert len(set(t[:4])) == 1
    assert t[:6].tolist() == [2, 2, 2, 2, 3, 3]
    # B is ranked on its own, whatever its scale
    assert t[6:].tolist() == [1, 2, 3]


def test_paired_gain_positive_when_matched_scores_lower() -> None:
    scores = pd.DataFrame({N.GENERIC: [0.30, 0.20], N.MATCHED: [0.25, 0.26]})
    gain = N.paired_gain(scores, N.GENERIC)
    assert gain.tolist() == pytest.approx([0.05, -0.06])


def test_pooled_rho_uses_within_cancer_ranks() -> None:
    # gain rises with support inside each cancer although the cancers sit on different scales
    support = np.array([0.1, 0.2, 0.3, 10.0, 20.0, 30.0])
    gain = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    cancer = np.array(["A"] * 3 + ["B"] * 3)
    assert N.pooled_rho(gain, support, cancer) == pytest.approx(1.0)
