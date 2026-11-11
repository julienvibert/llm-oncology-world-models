"""Gate-3 eligibility precheck: section-10 fixtures that concern eligibility and the
observation adapter (`manuscript/SCORING_SPEC_2026-09-28.md`).

Hand-written courses only; no registry data. These check the pipeline, not a hypothesis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))

import disease_extent_v1 as DE  # noqa: E402
import four_arm_eligibility_v1 as E  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

TAU = 10.0          # landmark on the diagnosis clock
CUTOFF = "2015-01-01"


def mets(site_detail: bool = True, sites: tuple = ()) -> DE.MetRecord:
    return DE.MetRecord(stage="IV", ever=1.0, first_month=1.0, sites=sites,
                        site_detail_available=site_detail, undated_sites=0)


def patient(*, status=(), presence=(), death=math.inf, last_contact=TAU + 40.0,
            lines=None, met=None) -> GV2.Patient:
    """A registry-shaped patient; times are months from diagnosis."""
    lines = lines if lines is not None else [
        {"start": TAU, "stop": TAU + 30.0, "label": "Carboplatin, Pemetrexed"}]
    return GV2.Patient(pid="toy", context={"mets": met if met is not None else mets()},
                       status=list(status), presence=list(presence), lines=lines,
                       death=death, last_contact=last_contact)


def cal(p: GV2.Patient) -> list[float]:
    saved = sorted({round(w - TAU, 4) for w, _ in p.status + p.presence
                    if TAU < w <= TAU + 24})
    return E.calendar_for(p, TAU, saved)


def resolve_real(p: GV2.Patient, horizon: float, conv: str = "primary") -> E.Resolution:
    with E.convention(conv):
        course = E.real_course(p, TAU, horizon)
    return E.resolve(course, cal(p), horizon)


def event(month: float, kind: str, **extra) -> dict:
    from datetime import date, timedelta
    when = date.fromisoformat(CUTOFF) + timedelta(days=round(month * GF_DAYS))
    return {"date": when.isoformat(), "type": kind, **extra}


GF_DAYS = 30.4375
GEN_KW = {"cutoff": CUTOFF, "landmark_presence": "evidence", "landmark_sites": [],
          "landmark_extent": "unknown", "index_regimen": "carboplatin + pemetrexed"}


def resolve_generated(ledger: list[dict], calendar: list[float], horizon: float,
                      landmark_state: str = "stable") -> E.Resolution:
    with E.convention("primary"):
        course = E.generated_course(ledger, horizon, landmark_state=landmark_state,
                                    **GEN_KW)
    return E.resolve(course, calendar, horizon)


# --- survival and censoring -------------------------------------------------------------

def test_censoring_at_month_eight_of_twelve_makes_full_target_unavailable() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 3, "stable"), (TAU + 6, "stable")],
                last_contact=TAU + 8.0)
    r12 = resolve_real(p, 12.0)
    assert not r12.full_core
    assert r12.blocks == (False, False, False, False)      # no alive-at-12, no tx total
    assert r12.reasons == ("survival_unresolved_registry",)
    assert resolve_real(p, 6.0).full_core                   # the documented prefix is fine


def test_death_with_documented_course_resolves_only_post_death_time() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 2, "progressing")], death=TAU + 5.0,
                last_contact=TAU + 5.0)
    assert resolve_real(p, 12.0).full_core


def test_death_with_undocumented_pre_death_state_is_not_eligible() -> None:
    # nothing on record before the landmark or before death: state unknown from time zero
    p = patient(status=[], presence=[], death=TAU + 5.0, last_contact=TAU + 5.0)
    r = resolve_real(p, 12.0)
    assert r.blocks[0] is True                              # survival is resolved ...
    assert not r.full_core                                  # ... but death alone is not enough
    assert E.DISEASE_AT_LANDMARK in r.reasons


# --- chronology ------------------------------------------------------------------------

def test_later_evidence_wins_in_both_directions() -> None:
    worse_then_ned = patient(status=[(TAU - 1, "stable"), (TAU + 2, "progressing")],
                             presence=[(TAU + 5, "no evidence")])
    with E.convention("primary"):
        path = E.real_course(worse_then_ned, TAU, 12.0).path
    assert E.state_at(path, 3.0) == "progressing"
    assert E.state_at(path, 6.0) == "no_active_disease"

    ned_then_worse = patient(status=[(TAU - 1, "stable"), (TAU + 5, "progressing")],
                             presence=[(TAU + 2, "no evidence")])
    with E.convention("primary"):
        path = E.real_course(ned_then_worse, TAU, 12.0).path
    assert E.state_at(path, 3.0) == "no_active_disease"
    assert E.state_at(path, 6.0) == "progressing"


def test_same_time_contradiction_stays_unresolved() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 3, "progressing")],
                presence=[(TAU + 3, "no evidence")])
    r = resolve_real(p, 12.0)
    assert not r.full_core
    assert E.DISEASE_AFTER_LANDMARK in r.reasons


# --- the observation adapter -------------------------------------------------------------

def test_donor_first_assessed_at_month_five_gets_no_invented_earlier_state() -> None:
    donor = patient(status=[(TAU + 5, "stable")])           # nothing before month five
    with E.convention("primary"):
        source = E.real_course(donor, TAU, 12.0).path
    seen = E.observe(source, [0.0, 2.0, 4.0, 6.0, 9.0], 12.0)
    assert E.state_at(seen, 0.0) == "unknown"
    assert E.state_at(seen, 2.0) == "unknown"
    assert E.state_at(seen, 4.5) == "unknown"               # not back-filled from month 5
    assert E.state_at(seen, 6.0) == "stable"                # first read after the evidence
    assert E.first_unresolved(seen, 12.0) == 0.0


def test_sampling_an_expiring_donor_does_not_renew_its_evidence_clock() -> None:
    donor = patient(status=[(TAU - 1, "stable"), (TAU + 1, "stable"), (TAU + 9, "stable")])
    with E.convention("sensitivity"):
        source = E.real_course(donor, TAU, 12.0).path
    assert E.state_at(source, 6.0) == "unknown"             # expired at month 5 on its own
    seen = E.observe(source, [0.0, 4.0, 7.0, 10.0], 12.0)
    assert E.state_at(seen, 4.5) == "stable"                # read at 4, still supported
    assert E.state_at(seen, 5.5) == "unknown"               # the read at 4 did not renew it
    assert E.state_at(seen, 8.0) == "unknown"               # read at 7: no support there
    assert E.state_at(seen, 10.5) == "stable"


def test_adapter_reproduces_a_real_course_on_its_own_calendar() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 2.3, "responding"),
                        (TAU + 7.7, "progressing")], presence=[(TAU + 5.1, "evidence")])
    for conv in ("primary", "sensitivity"):
        with E.convention(conv):
            course = E.real_course(p, TAU, 12.0)
        seen = E.observe(course.path, cal(p), 12.0)
        assert [(round(a, 6), round(b, 6), s) for a, b, s in seen] == \
            [(round(a, 6), round(b, 6), s) for a, b, s in course.path]


def test_horizon_prefix_invariance_of_resolution() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 2, "stable"), (TAU + 14, "progressing")],
                last_contact=TAU + 18.0)
    assert resolve_real(p, 12.0).full_core
    assert not resolve_real(p, 24.0).full_core
    with E.convention("primary"):
        short = E.real_course(p, TAU, 12.0).path
        long = E.real_course(p, TAU, 24.0).path
    assert [s for s in long if s[0] < 12.0][:-1] == list(short)[:-1]


def test_saved_calendar_must_match_source_records() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 2, "stable")])
    with pytest.raises(ValueError):
        E.calendar_for(p, TAU, [2.0, 3.0])


# --- generated courses ---------------------------------------------------------------

def test_generated_death_before_real_progression_keeps_h_fixed_and_resolves() -> None:
    ledger = [event(3.0, "death", cause="x")]
    r = resolve_generated(ledger, [0.0, 6.0], 12.0)
    assert r.full_core                                      # death resolves months 3-12


def test_generated_silence_is_read_not_filtered() -> None:
    # silent until death at month 30: the asserted landmark state persists as a fact
    silent = [event(30.0, "death", cause="x")]
    r = resolve_generated(silent, [0.0, 3.0, 6.0, 9.0], 12.0)
    assert r.full_core
    # a silent course with an UNKNOWN starting state is unresolved, and makes the whole
    # cell unavailable rather than being dropped from it
    blind = resolve_generated(silent, [0.0, 3.0], 12.0, landmark_state="unknown")
    good = [r] * 8
    assert E.llm_cell(9, good + [blind]) == (False, "accepted_repeat_unresolved")
    assert E.llm_cell(8, good) == (True, "")


def test_unmapped_generated_regimen_leaves_treatment_unresolved() -> None:
    ledger = [event(2.0, "systemic_treatment_switch",
                    regimen={"label": "investigational agent"}),
              event(30.0, "death", cause="x")]
    r = resolve_generated(ledger, [0.0, 3.0], 12.0)
    assert r.blocks[:3] == (True, True, True)
    assert not r.blocks[3] and E.UNMAPPED in r.reasons


# --- donor cards and cells ---------------------------------------------------------------

def test_one_unresolved_donor_member_makes_the_card_unavailable_before_subsampling() -> None:
    ok = E.Resolution((True, True, True, True), ())
    bad = E.unresolved("survival_unresolved_registry")
    assert E.donor_card([ok] * 20) == (True, "")
    assert E.donor_card([ok] * 19 + [bad]) == (False, "card_member_unresolved")
    # no rescue by keeping the nineteen complete members
    assert E.donor_card([ok] * 19) == (False, "card_size_not_twenty")


def test_all_rejected_cell_still_emits_a_row() -> None:
    ok = E.Resolution((True, True, True, True), ())
    expected = [("q1", arm) for arm in E.ARMS]
    rows = E.emit_cells(expected, {("q1", "setting"): [ok] * 10})
    assert len(rows) == 4
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["patient_only"]["n_accepted"] == 0
    assert by_arm["patient_only"]["eligible"] is False
    assert by_arm["patient_only"]["reason"] == "fewer_than_seven_accepted"
    assert by_arm["setting"]["eligible"] is True


# --- treatment channel -------------------------------------------------------------------

def test_overlapping_and_already_active_regimens_do_not_block_resolution() -> None:
    lines = [{"start": TAU - 3.0, "stop": TAU + 4.0, "label": "Letrozole"},
             {"start": TAU, "stop": TAU + 8.0, "label": "Palbociclib"},
             {"start": TAU + 2.0, "stop": TAU + 6.0, "label": "Fulvestrant"}]
    p = patient(status=[(TAU - 1, "stable"), (TAU + 3, "stable")], lines=lines)
    assert resolve_real(p, 12.0).full_core


def test_missing_stop_dates_are_unresolved_and_coded_by_position() -> None:
    p = patient(status=[(TAU - 1, "stable"), (TAU + 3, "stable")])
    with E.convention("primary"):
        in_window = E.real_course(p, TAU, 12.0, stop_unknown=[(TAU + 1.0, "ongoing")])
        before = E.real_course(p, TAU, 12.0, stop_unknown=[(TAU - 20.0, "undated")])
        after_h = E.real_course(p, TAU, 12.0, stop_unknown=[(TAU + 13.0, "ongoing")])
    r = E.resolve(in_window, cal(p), 12.0)
    assert not r.full_core and f"{E.STOP_IN_WINDOW}_ongoing" in r.reasons
    r = E.resolve(before, cal(p), 12.0)
    assert not r.full_core and f"{E.STOP_BEFORE_LANDMARK}_undated" in r.reasons
    assert r.full_core_ignoring(frozenset({f"{E.STOP_BEFORE_LANDMARK}_undated"}))
    assert E.resolve(after_h, cal(p), 12.0).full_core


def test_unmapped_real_regimen_in_window_is_unresolved() -> None:
    lines = [{"start": TAU, "stop": TAU + 8.0, "label": "Investigational Drug"}]
    p = patient(status=[(TAU - 1, "stable"), (TAU + 3, "stable")], lines=lines)
    r = resolve_real(p, 12.0)
    assert not r.full_core and E.UNMAPPED in r.reasons


# --- BRCA ------------------------------------------------------------------------------

def test_brca_site_detail_absent_leaves_common_core_unchanged() -> None:
    status = [(TAU - 1, "stable"), (TAU + 3, "responding"), (TAU + 7, "progressing")]
    with_sites = patient(status=status, met=mets(site_detail=True))
    no_sites = patient(status=status, met=mets(site_detail=False))
    for h in E.HORIZONS[:3]:
        a, b = resolve_real(with_sites, h), resolve_real(no_sites, h)
        assert a == b and a.full_core
