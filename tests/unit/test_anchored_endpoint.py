"""PFS extraction must be unambiguous, or must refuse."""

from __future__ import annotations

import pytest

from synthonccases.anchored import extract_many, extract_pfs

LM = "2016-01-01"


def rec(*, timeline=None, death=None, last_contact=None) -> dict:
    out = {"trajectory": {"timeline": timeline or [], "outcome": {}}}
    if death:
        out["trajectory"]["outcome"]["date_of_death"] = death
    if last_contact:
        out["trajectory"]["outcome"]["last_contact_date"] = last_contact
    return out


def test_progression_event_sets_the_endpoint():
    e = extract_pfs(rec(timeline=[{"event_type": "progression", "date": "2016-07-01"}]), LM)
    assert e.ok and e.event == 1 and e.source == "progression_event"
    assert e.months == pytest.approx(6.0, abs=0.1)


def test_the_earliest_qualifying_event_wins():
    e = extract_pfs(rec(timeline=[{"event_type": "progression", "date": "2016-10-01"},
                                  {"event_type": "recurrence", "date": "2016-04-01"}]), LM)
    assert e.months == pytest.approx(3.0, abs=0.1)


def test_death_ends_progression_free_survival():
    e = extract_pfs(rec(death="2016-04-01"), LM)
    assert e.ok and e.event == 1 and e.source == "death"


def test_a_line_progression_date_counts_even_without_an_event():
    e = extract_pfs(rec(timeline=[{"event_type": "treatment_line",
                                   "date": "2016-01-01",
                                   "progression_date": "2016-05-01"}]), LM)
    assert e.ok and e.source == "line_progression_date"


def test_no_progression_is_censored_not_progression_free_forever():
    e = extract_pfs(rec(last_contact="2017-01-01"), LM)
    assert e.ok and e.event == 0 and e.source == "last_contact"
    assert e.months == pytest.approx(12.0, abs=0.2)


def test_events_before_the_landmark_are_ignored():
    """A patient's history is not their future. Prior progressions must not be counted."""
    e = extract_pfs(rec(timeline=[{"event_type": "progression", "date": "2015-01-01"}],
                        last_contact="2016-06-01"), LM)
    assert e.event == 0 and e.source == "last_contact"


def test_incoherent_records_refuse_rather_than_returning_a_number():
    assert not extract_pfs(rec(death="2015-01-01"), LM).ok        # death before landmark
    assert not extract_pfs({}, LM).ok                              # no trajectory
    assert not extract_pfs(rec(), LM).ok                           # nothing to go on
    assert not extract_pfs(rec(timeline=[]), "not-a-date").ok      # bad landmark
    assert not extract_pfs("a string", LM).ok
    for bad in (extract_pfs({}, LM), extract_pfs(rec(), LM)):
        assert bad.reason                                          # always says why


def test_extract_many_reports_what_it_dropped():
    records = [rec(timeline=[{"event_type": "progression", "date": "2016-04-01"}]),
               rec(timeline=[{"event_type": "progression", "date": "2016-07-01"}]),
               rec(last_contact="2017-01-01"),                     # censored
               {}]                                                 # unparseable
    samples, audit = extract_many(records, LM)
    assert len(samples) == 2
    assert audit == {"n_records": 4, "n_samples": 2, "n_censored": 1, "n_failed": 1,
                     "failure_reasons": ["record has no trajectory block"]}
