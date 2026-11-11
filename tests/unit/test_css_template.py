"""Estimators of the shared Clinical State Space page: Kaplan-Meier with ties, and map support."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/figures"), str(ROOT / "scripts/figures/panels")]
with matplotlib.rc_context():  # the figure kit sets global style on import; keep it out of other tests
    import _css_template as T  # noqa: E402


def test_tied_event_and_censoring_do_not_depend_on_row_order() -> None:
    w = np.ones(3)
    a = T.km_curve(np.array([1.0, 1, 2]), np.array([0, 1, 1]), w, [1.0])[0]
    b = T.km_curve(np.array([1.0, 1, 2]), np.array([1, 0, 1]), w, [1.0])[0]
    assert a == b and abs(a - 2 / 3) < 1e-12


def test_all_censored_stays_at_one() -> None:
    assert T.km_curve(np.array([1.0, 2, 3]), np.zeros(3, int), np.ones(3), [5.0])[0] == 1.0


def test_matches_lifelines_with_ties() -> None:
    rng = np.random.default_rng(1)
    t = rng.integers(1, 12, 200).astype(float)
    e = rng.integers(0, 2, 200)
    grid = np.arange(0.0, 13.0)
    ref = KaplanMeierFitter().fit(t, e).survival_function_at_times(grid).to_numpy()
    assert np.allclose(T.km_curve(t, e, np.ones(200), grid), ref)
    assert np.allclose(T.wkm(t, e, np.ones((1, 200)), 6.0), ref[6])


def test_weights_equal_replication() -> None:
    t, e = np.array([1.0, 2, 3, 4]), np.array([1, 0, 1, 1])
    w = np.array([2.0, 1, 3, 1])
    assert np.allclose(T.km_curve(t, e, w, [1.0, 3.0, 4.0]), T.km_curve(np.repeat(t, 2 * w.astype(int)), np.repeat(e, 2 * w.astype(int)), np.ones(14), [1.0, 3.0, 4.0]))


def test_band_contains_estimate_and_uses_clusters() -> None:
    b = pd.DataFrame({"t_os": np.tile(np.arange(1.0, 11), 3), "e_os": 1, "cluster": np.repeat(np.arange(10), 3)})
    lo, hi = T.km_band(b, [5.0], n=100)
    assert lo[0] <= T.km_curve(b.t_os.to_numpy(), b.e_os.to_numpy(), np.ones(30), [5.0])[0] <= hi[0]


def _map(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    g = {"base": pd.DataFrame({"x": 0.0, "y": 0.0, "t_os": t, "e_os": e})}
    cs = T.outcome_map(ax, g, 12.0, (-1, 1), (-1, 1), 0.2, 5.0)
    z = np.ma.filled(cs.get_array(), np.nan) if hasattr(cs, "get_array") else None
    plt.close(fig)
    return z


def test_map_blank_when_everyone_is_censored_before_the_horizon() -> None:
    import matplotlib.pyplot as plt
    seen = {}
    orig = plt.Axes.contourf

    def spy(self, x, y, v, **k):  # the values handed to contourf
        seen["v"] = np.asarray(v)
        return orig(self, x, y, np.where(np.isfinite(v), v, 50.0), **k)
    plt.Axes.contourf = spy
    try:
        _map(np.ones(30), np.zeros(30, int))
        assert np.isnan(seen["v"]).all()
        _map(np.full(30, 20.0), np.zeros(30, int))
        assert np.nanmax(seen["v"]) == 100.0
    finally:
        plt.Axes.contourf = orig


def test_progression_kept_on_os_censoring_date_and_dropped_on_death_date() -> None:
    b = pd.DataFrame({"t_pf": [5.0, 7.0, 3.0, 4.0], "e_pf": [1, 1, 1, 0], "t_os": [5.0, 7.0, 9.0, 9.0], "e_os": [1, 0, 1, 1]})
    assert list(T.endpoint_progression(b).index) == [1, 2]
