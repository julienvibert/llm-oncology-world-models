"""N6 regimen rules: the regimen known at the decision, mixed stop dates, same-day merging.

Synthetic treatment records only (no registry data): (start, stop, agent, regimen number) in
registry months. Covers two cases: agents of a regimen can start on different days,
and a regimen with one missing stop has no established end.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
_PATH, _RC = list(sys.path), matplotlib.rcParams.copy()
for p in ("scripts", "scripts/explore", "src"):
    sys.path.insert(0, str(ROOT / p))

import genie_operator_registry_v2 as GV2  # noqa: E402
import n6_support_ladder_v1 as N  # noqa: E402

sys.path[:] = _PATH
matplotlib.rcParams.update(_RC)

DAY = 1.0 / GV2.DAYS
# carboplatin day 0, pemetrexed 10 days later, same regimen 1; osimertinib regimen 2 at month 6
TX = [(0.0, 3.0, "Carboplatin", 1), (10 * DAY, 3.0, "Pemetrexed", 1), (6.0, 9.0, "Osimertinib", 2)]


def agents(records: list) -> set:
    return {ag for _, _, ag, _ in records}


def test_partner_started_later_is_excluded_at_the_decision() -> None:
    members = N.regimen_members(TX, 0.0)
    assert agents(members) == {"Carboplatin", "Pemetrexed"}          # membership by regimen number
    assert agents(N.known_at(members, 0.0)) == {"Carboplatin"}        # not added retrospectively


def test_partner_counts_once_started_by_the_decision() -> None:
    members = N.regimen_members(TX, 0.0)
    assert agents(N.known_at(members, 10 * DAY)) == {"Carboplatin", "Pemetrexed"}


def test_other_regimens_are_not_members() -> None:
    assert agents(N.regimen_members(TX, 6.0)) == {"Osimertinib"}
    assert N.regimen_members(TX, 3.0) == []


def test_same_day_regimens_merge() -> None:
    tx = [(2.0, 4.0, "Letrozole", 3), (2.0 + 0.2 * DAY, 5.0, "Goserelin", 4)]
    assert agents(N.regimen_members(tx, 2.0)) == {"Letrozole", "Goserelin"}


def test_regimens_a_week_apart_do_not_merge() -> None:
    tx = [(2.0, 4.0, "Letrozole", 3), (2.0 + 7 * DAY, 5.0, "Goserelin", 4)]
    assert agents(N.regimen_members(tx, 2.0)) == {"Letrozole"}


def test_all_known_stops_give_the_latest_stop() -> None:
    stop, kind = N.regimen_stop([(0.0, 3.0, "a", 1), (0.1, 4.0, "b", 1)])
    assert (stop, kind) == (4.0, "all_known")


def test_one_missing_stop_makes_the_regimen_stop_unknown() -> None:
    stop, kind = N.regimen_stop([(0.0, 3.0, "a", 1), (0.1, math.nan, "b", 1)])
    assert math.isnan(stop) and kind == "mixed"


def test_all_missing_stops() -> None:
    stop, kind = N.regimen_stop([(0.0, math.nan, "a", 1), (0.1, math.nan, "b", 1)])
    assert math.isnan(stop) and kind == "all_missing"


def test_unknown_stop_runs_to_the_decision() -> None:
    stop, _ = N.regimen_stop([(0.0, 3.0, "a", 1), (0.1, math.nan, "b", 1)])
    tau = 8.0
    end = min(stop if math.isfinite(stop) else math.inf, tau)
    assert end == tau                                                  # not 3.0, the latest known stop


def test_class_of_regimen_known_at_decision_ignores_later_partner() -> None:
    assert N.treatment_class(TX, 0.0) == N.classify_agents(["Carboplatin"])
    assert N.treatment_class(TX, 0.0) == N.treatment_class_same_day(TX, 0.0)


def test_no_regimen_at_the_time_is_unknown() -> None:
    assert N.treatment_class(TX, 3.0) == N.UNKNOWN


def test_driver_without_result_is_missing_and_renamed() -> None:
    assert N.missing(2, N.NO_RESULT)
    assert not N.missing(2, "tested, none of these")
