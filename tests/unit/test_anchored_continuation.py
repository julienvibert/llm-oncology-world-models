"""The continuation contract: a real patient in, a legal future out, nothing rewritten."""

from __future__ import annotations


from pathlib import Path

import pytest

from synthonccases.anchored import (
    compose_prompt, pfs_from_future, render_template, validate_future_ledger,
    verify_continuation_template,
)
from synthonccases.anchored import contract as C

CUTOFF = "2018-03-01"


def future(*events, **kw):
    return validate_future_ledger({"ledger": list(events)}, cutoff=CUTOFF, **kw)


def ev(day: str, type_: str, **extra):
    return {"date": day, "type": type_, **extra}


DEATH = ev("2020-01-01", "death", cause_category="cancer_related", cause="disease",
           eol_care_pattern="hospice")


# ------------------------------------------------------------------ the template ----------
def test_the_continuation_template_is_the_declared_edits_and_nothing_else():
    """The diff is code. A hand-edit to either file, or an edit not declared here, fails."""
    assert verify_continuation_template()
    expected = C._apply_continuation_edits(C.ANCHORED_TEMPLATE.read_text())
    assert C.CONTINUATION_TEMPLATE.read_text() == expected


def test_the_continuation_prompt_forbids_a_baseline_and_says_missing_means_unobserved():
    text = render_template("", "continuation")
    assert '{ "ledger": [ ... ] }' in text
    assert '{ "baseline_patch": { ... }, "ledger": [ ... ] }' not in text
    assert "There is no `baseline_patch`" in text
    assert "NOT OBSERVED" in text
    assert "## Required Patch Leaves" not in text
    # and it must not still say the trajectory starts at diagnosis
    assert "The trajectory starts at diagnosis" not in text


def test_the_two_contracts_are_different_prompts_and_only_one_is_the_frozen_parent():
    latent = render_template("", "latent_completion")
    cont = render_template("", "continuation")
    assert latent == C.FROZEN_PARENT.read_text()      # the inherited arm IS the frozen prompt
    assert cont != latent
    with pytest.raises(ValueError, match="unknown contract"):
        render_template("", "nonsense")


def test_the_continuation_prompt_never_asks_for_leaves_it_forbids():
    body = compose_prompt("", contract="continuation", baseline_seed={},
                          pinned_paths=(), required_patch_leaves=("a.b.c",),
                          policy_text="")
    assert "a.b.c" not in body and "{{" not in body


# ------------------------------------------------------------------ the validator ---------
def test_a_legal_continuation_is_accepted_and_yields_its_endpoint():
    r = future(ev("2018-09-01", "progression"), DEATH)
    assert r.accepted and r.n_events == 2
    months, why = pfs_from_future(r, cutoff=CUTOFF)
    assert why == "progression" and 5.9 < months < 6.2


@pytest.mark.parametrize("payload,kind", [
    ({"baseline_patch": {}, "ledger": [DEATH]}, "wrote_baseline"),
    ({"ledger": [], }, "empty_ledger"),
    ({"ledger": "nope"}, "ledger_not_a_list"),
    ({"trajectory": [DEATH]}, "unexpected_keys"),
    ({}, "no_ledger"),
])
def test_a_response_that_answers_a_different_question_is_refused_by_kind(payload, kind):
    assert validate_future_ledger(payload, cutoff=CUTOFF).failure_kind == kind


def test_an_event_at_or_before_the_cutoff_rewrites_observed_history():
    """The single most important check: the endpoint is measured from that day."""
    assert future(ev(CUTOFF, "progression"), DEATH).failure_kind == \
        "event_at_or_before_cutoff"
    assert future(ev("2017-12-01", "progression"), DEATH).failure_kind == \
        "event_at_or_before_cutoff"


def test_restating_the_fixed_prefix_is_refused():
    prefix = [{"date": "2018-03-01", "type": "systemic_treatment_start",
               "regimen": {"label": "FOLFOX", "components": ["oxaliplatin"]}}]
    # a later, legally-dated copy of a prefix event is still the model rewriting history
    r = validate_future_ledger(
        {"ledger": [{"date": "2018-03-01", "type": "systemic_treatment_start",
                     "regimen": {"label": "FOLFOX", "components": ["oxaliplatin"]}}, DEATH]},
        cutoff="2018-02-01", prefix=prefix)
    assert r.failure_kind == "restated_prefix"


def test_chronology_and_terminal_rules():
    assert future(ev("2019-01-01", "progression"), ev("2018-06-01", "progression"),
                  DEATH).failure_kind == "dates_out_of_order"
    assert future(DEATH, ev("2021-01-01", "progression")).failure_kind == \
        "events_after_death"
    assert future(DEATH, {**DEATH, "date": "2020-06-01"}).failure_kind == "multiple_deaths"
    # a course that stops after progression is STRUCTURALLY fine: it resolved PFS and not
    # OS, which is a fact about the trajectory, not a reason to throw it away
    assert future(ev("2018-09-01", "progression")).accepted
    # ... unless the caller specifically wants complete courses
    assert future(ev("2018-09-01", "progression"),
                  require_terminal=True).failure_kind == "no_terminal_event"


def test_an_invented_event_type_is_refused_against_the_frozen_schema():
    assert future(ev("2018-09-01", "tumour_shrank"), DEATH).failure_kind == \
        "unknown_event_type"


@pytest.mark.parametrize("event_type", [
    "systemic_treatment_start",
    "systemic_treatment_switch",
])
def test_a_new_treatment_course_needs_its_regimen_but_classes_are_only_a_warning(
    event_type,
):
    assert future(ev("2018-09-01", event_type), DEATH).failure_kind == \
        "treatment_without_regimen"
    r = future(ev("2018-09-01", "systemic_treatment_switch",
                  regimen={"label": "FOLFIRI", "components": ["irinotecan"]}), DEATH)
    assert r.accepted and any("component_classes" in w for w in r.warnings)


@pytest.mark.parametrize("event_type", [
    "systemic_treatment_stop",
    "systemic_treatment_modification",
])
def test_treatment_stop_and_modification_can_refer_to_the_active_regimen(event_type):
    r = future(ev("2018-09-01", event_type), DEATH)
    assert r.accepted


def test_a_truncated_course_still_resolves_the_endpoint_it_reached():
    """The bar belongs to the endpoint. Rejecting this would discard a valid PFS datum for
    a reason -- length -- correlated with the datum's own value."""
    from synthonccases.anchored.future import endpoints
    r = future(ev("2018-06-01", "progression"),
               ev("2018-07-01", "systemic_treatment_switch",
                  regimen={"label": "FOLFIRI", "components": ["irinotecan"]}))
    e = endpoints(r, cutoff=CUTOFF)
    assert e.pfs_valid and 2.9 < e.pfs_months < 3.2 and e.pfs_event == "progression"
    assert not e.full_course_valid and not e.death_reached and e.os_months is None
    assert e.last_event_month is not None and e.n_events == 2

    whole = future(ev("2018-06-01", "progression"), DEATH)
    e2 = endpoints(whole, cutoff=CUTOFF)
    assert e2.pfs_valid and e2.full_course_valid and e2.os_months > e2.pfs_months
    assert set(e2.to_dict()) >= {"pfs_valid", "full_course_valid", "death_reached",
                                 "os_months", "last_event_month"}


def test_a_course_with_no_progression_is_censored_not_scored_as_infinite():
    r = validate_future_ledger(
        {"ledger": [ev("2018-09-01", "response"),
                    ev("2019-01-01", "toxicity", ae="fatigue")]}, cutoff=CUTOFF)
    assert r.accepted
    months, why = pfs_from_future(r, cutoff=CUTOFF)
    assert months is None and why == "no_progression_or_death"


def test_a_refused_ledger_never_produces_a_number():
    """A parse failure that becomes a data point is indistinguishable from a prediction."""
    r = future(ev("2017-01-01", "progression"), DEATH)
    months, why = pfs_from_future(r, cutoff=CUTOFF)
    assert months is None and why == "event_at_or_before_cutoff"


def test_a_manifest_path_outside_the_repo_is_recorded_not_fatal():
    """`relative_to` raises for a path outside the root, which killed a build after all the
    work was done and before anything was written."""
    import sys
    sys.path.insert(0, str(C.ROOT / "scripts" / "explore"))
    from build_anchored_pfs_arms_v1 import ROOT, recorded_path

    assert recorded_path(ROOT / "outputs" / "x" / "queries.jsonl") == "outputs/x/queries.jsonl"
    outside = Path("/tmp/somewhere/queries.jsonl")
    assert recorded_path(outside) == str(outside.resolve())


# ------------------------------------------------- sequencing ------------------------------
def test_sequencing_enters_the_seed_only_when_it_was_resulted_by_the_landmark():
    """A panel that came back after this treatment started was not available when the
    decision was taken; handing it over leaks the future of the decision being predicted."""
    from synthonccases.anchored import baseline_seed
    base = {"cancer": "NSCLC", "histology": "Adenocarcinoma", "age_at_start": 64.0,
            "months_since_dx": 29.0, "female": 1.0, "stage_iv": 1.0,
            "alterations": "EGFR L858R", "driver_bucket": "EGFR"}

    seed, pinned = baseline_seed({**base, "panel_reported": 1.0})
    mp = seed["identity_card"]["molecular_profile"]
    assert mp["primary_driver_bucket"] == "EGFR"
    assert {"name": "EGFR", "value": "L858R"} in mp["biomarkers"]
    assert "identity_card.molecular_profile.biomarkers" in pinned

    later, _ = baseline_seed({**base, "panel_reported": 0.0})
    assert "molecular_profile" not in later["identity_card"]


def test_sequenced_with_nothing_found_is_not_the_same_as_never_sequenced():
    from synthonccases.anchored import baseline_seed
    seed, _ = baseline_seed({"cancer": "CRC", "panel_reported": 1.0,
                             "alterations": "", "driver_bucket": "none found"})
    # THE SEED CANNOT HOLD THIS FACT, and must not pretend to. "targeted sequencing panel"
    # is an assay, not an analyte, so a biomarker row carrying it fabricates a measurement
    # that was never taken -- the same category error as the prose row it replaced. The
    # schema has no assay-provenance field, so the distinction is carried in prose.
    assert "molecular_profile" not in seed["identity_card"]
    never, _ = baseline_seed({"cancer": "CRC", "panel_reported": 0.0,
                              "alterations": "", "driver_bucket": ""})
    assert "molecular_profile" not in never["identity_card"]
    # WHERE THE DISTINCTION ACTUALLY LIVES. Both are silent in the seed, so the summary is
    # the thing that has to keep them apart -- and it does.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "explore"))
    import render_reference_card_v1 as RC
    ran = RC.DRIVER.get("none recorded on the available panel")
    assert ran and ran != RC.DRIVER.get("unknown")


def test_a_positive_panel_carries_analytes_not_sentences():
    from synthonccases.anchored import baseline_seed
    seed, pinned = baseline_seed({"cancer": "CRC", "panel_reported": 1.0,
                                  "alterations": "BRAF G466V", "driver_bucket": "braf"})
    mp = seed["identity_card"]["molecular_profile"]
    assert {"name": "BRAF", "value": "G466V"} in mp["biomarkers"]
    # `name` is an analyte and `value` is its result -- the contract the schema documents
    # and the model is asked to follow. A sentence here models a convention we would
    # reject in generated output.
    for b in mp["biomarkers"]:
        assert ";" not in str(b["value"]), f"prose in a biomarker value: {b}"
        assert len(str(b["name"]).split()) <= 2, f"an assay, not an analyte: {b}"
    # A PIN MUST NAME A PATH THE SEED CONTAINS.
    for path in pinned:
        node = seed
        for part in path.split("."):
            assert part in node, f"pinned path not present in seed: {path}"
            node = node[part]


# ------------------------------------------------- evidence summaries ---------------------
def test_the_summary_is_kaplan_meier_not_the_median_of_the_progressed():
    """The median of the uncensored subset is biased short by exactly the patients who were
    doing well when we lost sight of them."""
    import numpy as np

    from synthonccases.anchored.summary import km_summary
    # four progress early, six are censored late and would have progressed later still
    t = np.array([1.0, 1.5, 2.0, 2.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
    e = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = km_summary(t, e)
    assert s.n == 10 and s.n_events == 4
    assert not s.median_reached          # only 40% progressed: no median exists
    assert s.at_horizon[6.0] > 0.5 and s.n_at_risk[6.0] == 6
    naive = float(np.median(t[e == 1]))
    assert naive < 3.0                   # what the old descriptive line would have said


def test_a_median_that_was_not_reached_is_never_extrapolated():
    import numpy as np

    from synthonccases.anchored.summary import format_km, km_summary
    t = np.array([10.0] * 8)
    e = np.zeros(8)
    text = format_km(km_summary(t, e), label="Cohort")
    assert "not reached" in text
    assert "still at risk" in text


def test_strata_are_chosen_by_scaled_residual_times_prognostic_weight():
    """Raw, a difference of eight years of age outranks 0.6 disease sites for no reason but
    the unit it is measured in."""
    from synthonccases.anchored.summary import choose_strata
    residual = {"age": 2.0, "sites": 1.5, "noise": 3.0}
    axes = choose_strata(residual, allowed=["age", "sites", "noise"],
                         weights={"age": 0.1, "sites": 0.9, "noise": 0.0}, k=2)
    assert axes[0] == "sites"            # 1.5 x 0.9 beats 2.0 x 0.1
    assert "noise" not in axes           # zero prognostic weight is never chosen


def test_a_split_that_leaves_one_side_empty_is_not_offered():
    import numpy as np
    import pandas as pd

    from synthonccases.anchored.summary import MIN_STRATUM, stratify
    rows = pd.DataFrame({"flag": [1.0] * 9 + [0.0]})
    groups = stratify(rows, {"flag": 1.0}, "flag")
    sizes = sorted(int(m.sum()) for _, m, _ in groups)
    assert sizes[0] < MIN_STRATUM        # the renderer drops this split
    assert any(is_q for _, _, is_q in groups)
    assert np.array([m for _, m, _ in groups]).sum() == len(rows)


# ------------------------------------------------- prefix status episodes -----------------
def test_repeated_assessments_of_one_episode_are_one_ledger_event():
    """Ten quarterly scans all reading "progressing" are ONE episode, not ten events.

    The prefix ledger used to emit a row per assessment, which rendered a patient scanned
    every few months as a run of `progression` events with no treatment between them --
    reading as years of continuous untreated progression. It also disagreed with both
    neighbours: the prose summary counted episodes and the scorer counted `entries_into`,
    so the model was shown one accounting and graded on another.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "explore"))
    import canonical_future_v1 as CF
    import genie_operator_registry_v2 as GV2

    class P:
        pid = "T"
        presence: list = []
        context: dict = {}

    p = P()
    # One episode, re-confirmed every two months -- well inside STATUS_VALIDITY.
    p.status = [(12.0 + 2 * i, "progressing") for i in range(10)]
    path = CF.state_path(p, 10.0, 40.0)
    entries = sum(1 for i, (_a, _b, st) in enumerate(path)
                  if st == "progressing" and (i == 0 or path[i - 1][2] != "progressing"))
    assert entries == 1, f"ten assessments of one episode became {entries} events"

    # WHAT BREAKS AN EPISODE differs between the two conventions, so each is asserted
    # against its own rule rather than one being assumed.
    p.status = [(12.0, "progressing"), (12.0 + 3 * GV2.STATUS_VALIDITY, "progressing")]

    def entries_under(mode):
        path = CF.state_path(p, 10.0, 60.0, persistence=mode)
        return sum(1 for i, (_a, _b, st) in enumerate(path)
                   if st == "progressing" and (i == 0 or path[i - 1][2] != "progressing"))

    # the conservative convention breaks it whenever an assessment passes four months
    assert entries_under("assessment_validity") == 2
    # the primary one carries the documented state until a recorded event CHANGES it, so a
    # gap of any length is still one continuing episode. What ends it is follow-up ending,
    # and `defined_window` closes the window there -- which is why there is no gap
    # constant here to tune.
    assert entries_under("under_followup") == 1
    p.status = [(12.0, "progressing"), (12.0 + 30.0, "progressing")]
    assert entries_under("under_followup") == 1
