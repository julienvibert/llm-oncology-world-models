"""The properties that have silently broken measurements in this project before."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthonccases.anchored import (
    AnchorPolicy, Corpus, available, render, retrieve, retrieve_pair,
)


def make_corpus(n: int = 60) -> Corpus:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "patient": [f"p{i // 2}" for i in range(n)],      # two lines per patient
        "arm": ["A" if i % 2 == 0 else "B" for i in range(n)],
        "cancer": ["BRCA" if i < n // 2 else "CRC" for i in range(n)],
        "burden": rng.normal(size=n),
        "ecog": rng.normal(size=n),
        "pfs_months": rng.exponential(6.0, size=n),
    })
    return Corpus(frame, channels=["burden", "ecog"], subject_col="patient",
                  action_col="arm", outcome_cols=["pfs_months"])


def query_from(corpus: Corpus, i: int = 0) -> pd.Series:
    q = corpus.frame.iloc[i].copy()
    q["query_id"] = f"q{i}"
    return q


# ------------------------------------------------------------------ policy ----------------
def test_policy_hash_is_stable_and_sensitive():
    a = AnchorPolicy(k=40, label="x")
    assert a.hash == AnchorPolicy(label="x", k=40).hash        # field order is irrelevant
    assert a.hash != a.variant(k=41).hash                      # every value matters
    assert a.hash != a.variant(label="y").hash


def test_policy_rejects_incoherent_settings():
    with pytest.raises(ValueError):
        AnchorPolicy(k=5, render_target=10)
    with pytest.raises(ValueError):
        AnchorPolicy(k=5, min_support=10)
    with pytest.raises(ValueError):
        AnchorPolicy(target_action="B", same_action=True)


def test_corpus_refuses_outcome_as_channel():
    frame = pd.DataFrame({"patient": ["a"], "arm": ["A"], "y": [1.0]})
    with pytest.raises(ValueError, match="cannot also be channels"):
        Corpus(frame, channels=["y"], subject_col="patient", action_col="arm",
               outcome_cols=["y"])


# ------------------------------------------------------------------ retrieval -------------
def test_a_patient_is_never_their_own_anchor():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    res = retrieve(corpus, q, AnchorPolicy(k=10, min_support=1, render_target=5))
    subjects = corpus.frame.iloc[list(res.anchor_index)]["patient"]
    assert q["patient"] not in set(subjects)


def test_anchors_respect_the_action_filter():
    corpus = make_corpus()
    q = query_from(corpus, 0)                                   # arm A
    res = retrieve(corpus, q, AnchorPolicy(k=10, min_support=1, render_target=5))
    arms = set(corpus.frame.iloc[list(res.anchor_index)]["arm"])
    assert arms == {"A"}


def test_exact_match_fields_are_applied_before_distance():
    corpus = make_corpus()
    q = query_from(corpus, 0)                                   # BRCA
    res = retrieve(corpus, q, AnchorPolicy(k=10, min_support=1, render_target=5,
                                           match_exact=("cancer",)))
    assert set(corpus.frame.iloc[list(res.anchor_index)]["cancer"]) == {"BRCA"}


def test_too_little_support_refuses_rather_than_degrading():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    res = retrieve(corpus, q, AnchorPolicy(k=40, min_support=40,
                                           match_exact=("cancer",), render_target=5))
    assert not res.supported
    assert res.support_label == "unsupported"
    assert res.n_anchors == 0
    assert res.note                                            # says why


def test_retrieval_is_deterministic():
    corpus = make_corpus()
    q = query_from(corpus, 3)
    p = AnchorPolicy(k=8, min_support=1, render_target=4)
    a, b = retrieve(corpus, q, p), retrieve(corpus, q, p)
    # field-by-field rather than ==: the audit carries NaN where a quantity is not
    # computable, and NaN != NaN would fail this for a reason that has nothing to do with
    # determinism
    assert a.anchor_index == b.anchor_index
    assert a.distances == b.distances
    assert a.tier_counts == b.tier_counts and a.n_strict == b.n_strict


def test_nearest_anchors_are_actually_nearest():
    corpus = make_corpus()
    q = query_from(corpus, 5)
    res = retrieve(corpus, q, AnchorPolicy(k=6, min_support=1, render_target=3))
    assert list(res.distances) == sorted(res.distances)


def test_a_short_strict_set_is_topped_up_and_the_split_is_recorded():
    """A strict set of three is a fact about the decision, not a defect to pad over."""
    corpus = make_corpus()
    q = query_from(corpus, 0)
    # k caps the strict set at 4; the tier rule supplies the rest, labelled
    p = AnchorPolicy(k=4, min_support=1, render_target=10, match_exact=("cancer",),
                     tiers=(("any cancer, same treatment", ()),))
    res = retrieve(corpus, q, p)
    assert res.n_strict == 4                        # the strict set, as selected
    assert res.n_anchors == 10                      # topped up to the display target
    assert dict(res.tier_counts) == {"strict": 4, "any cancer, same treatment": 6}
    assert len(res.tier_labels) == res.n_anchors    # every anchor says how it got in


def test_refusing_to_relax_leaves_the_strict_set_alone():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=4, min_support=1, render_target=10, match_exact=("cancer",),
                     tiers=(("any cancer", ()),), relax_to_fill=False)
    res = retrieve(corpus, q, p)
    assert res.n_anchors == res.n_strict == 4
    assert dict(res.tier_counts) == {"strict": 4}


def test_a_sham_control_is_never_counted_as_support():
    """A random draw is not evidence. Letting it inherit direct/transport would put the
    control inside the support and quadrant analyses as though it were data."""
    corpus = make_corpus()
    q = query_from(corpus, 0)
    real = retrieve(corpus, q, AnchorPolicy(k=8, min_support=1, render_target=4))
    sham = retrieve(corpus, q, AnchorPolicy(k=8, min_support=1, render_target=4,
                                            shuffle_anchors=True))
    assert real.support_label == "direct" and not real.is_sham and real.supported
    assert sham.support_label == "sham" and sham.is_sham
    assert not sham.supported          # has anchors, is not support
    assert sham.has_anchors


def test_the_prompt_never_claims_sham_anchors_were_retrieved_by_similarity():
    """The same neutral wording in both arms; the facts are read off the rows."""
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=8, min_support=1, render_target=4)
    real = render("compact", corpus, retrieve(corpus, q, p), p)
    sham = render("compact", corpus, retrieve(corpus, q, p.variant(shuffle_anchors=True)), p)
    for block in (real, sham):
        assert "retrieved by similarity" not in block
        assert "Patient examples from the source corpus" in block
    # the real arm can honestly say they share a treatment; the sham arm cannot, and says so
    assert "all received" in real
    assert "different treatments" in sham or "several cancers" in sham


def test_shuffle_control_changes_the_anchors_but_not_the_count():
    corpus = make_corpus()
    q = query_from(corpus, 1)
    real = AnchorPolicy(k=8, min_support=1, render_target=8)
    sham = real.variant(shuffle_anchors=True)
    a, b = retrieve(corpus, q, real), retrieve(corpus, q, sham)
    assert a.n_anchors == b.n_anchors                # the prompt looks the same size
    assert a.policy_hash != b.policy_hash            # and the run is distinguishable
    # DISJOINT, not merely different: any overlap lets the real evidence leak into the
    # control, and a control that shares anchors with the treatment proves nothing
    assert set(a.anchor_index).isdisjoint(b.anchor_index)


def test_shuffle_is_constructible_even_when_the_eligible_pool_is_exhausted():
    """The control must exist exactly where the method is aimed: sparse neighbourhoods.

    Restricting the sham pool to the same eligibility filters made it unbuildable whenever
    the eligible pool was no larger than k. The corpus has thousands of patients; the
    control should use them.
    """
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=40, min_support=1, render_target=4, match_exact=("cancer",),
                     shuffle_anchors=True)
    res = retrieve(corpus, q, p)
    # anchors exist to render, but they are NOT support
    assert res.has_anchors and res.n_anchors > 0
    assert not res.supported


def test_shuffle_ignores_the_eligibility_filters_by_default():
    """Sham anchors may be any patient at all — that is what makes it a control."""
    corpus = make_corpus()
    q = query_from(corpus, 0)                       # arm A, BRCA
    p = AnchorPolicy(k=12, min_support=1, render_target=4, match_exact=("cancer",),
                     shuffle_anchors=True)
    rows = corpus.frame.iloc[list(retrieve(corpus, q, p).anchor_index)]
    assert set(rows["arm"]) != {"A"} or set(rows["cancer"]) != {"BRCA"}


def test_the_stricter_scope_still_refuses_when_it_cannot_be_built():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=40, min_support=1, render_target=4, match_exact=("cancer",),
                     shuffle_anchors=True, shuffle_scope="eligible")
    res = retrieve(corpus, q, p)
    assert not res.supported and "not constructible" in res.note


def test_residual_is_query_minus_anchor_centre():
    corpus = make_corpus()
    q = query_from(corpus, 2)
    res = retrieve(corpus, q, AnchorPolicy(k=6, min_support=1, render_target=3))
    rows = corpus.frame.iloc[list(res.anchor_index)]
    assert res.residual["burden"] == pytest.approx(q["burden"] - rows["burden"].mean())


def test_supervised_weights_change_which_anchors_win():
    corpus = make_corpus()
    q = query_from(corpus, 7)
    p = AnchorPolicy(k=4, min_support=1, render_target=2, metric="supervised")
    flat = retrieve(corpus, q, p)
    # collapse the second channel; a different neighbourhood must win
    skewed = retrieve(corpus.with_weights(np.array([1.0, 0.02])), q, p)
    assert set(flat.anchor_index) != set(skewed.anchor_index)


def test_paired_retrieval_differs_only_in_action():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=8, min_support=1, render_target=4)
    ra, rb = retrieve_pair(corpus, q, p, "A", "B")
    assert set(corpus.frame.iloc[list(ra.anchor_index)]["arm"]) == {"A"}
    assert set(corpus.frame.iloc[list(rb.anchor_index)]["arm"]) == {"B"}
    assert ra.support_label == rb.support_label == "transport"


# ------------------------------------------------------------------ rendering -------------
def test_every_renderer_states_support_and_count():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=10, min_support=1, render_target=3)
    res = retrieve(corpus, q, p)
    for name in available():
        block = render(name, corpus, res, p)
        if name == "none":
            assert block == ""
            continue
        assert str(res.n_anchors) in block


def test_an_unsupported_query_renders_an_explicit_refusal():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=40, min_support=40, match_exact=("cancer",), render_target=4)
    res = retrieve(corpus, q, p)
    for name in available():
        if name == "none":
            continue
        block = render(name, corpus, res, p)
        assert "None available" in block


def test_outcomes_can_be_withheld_from_the_prompt():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=10, min_support=1, render_target=3)
    res = retrieve(corpus, q, p)
    assert "pfs_months" in render("table", corpus, res, p)
    assert "pfs_months" not in render("table", corpus, res, p.variant(
        include_outcomes=False, include_aggregate=False))


def test_render_target_bounds_what_is_shown_not_what_is_retrieved():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=12, min_support=1, render_target=3)
    res = retrieve(corpus, q, p)
    block = render("table", corpus, res, p)
    assert res.n_anchors == 12
    # count DATA rows only: every pipe line except the column header and the ---- rule
    pipe_lines = [ln for ln in block.splitlines() if ln.startswith("|")]
    data_rows = [ln for ln in pipe_lines if not set(ln) <= set("|- ")]
    assert len(data_rows) == 3 + 1                # three anchors plus the column header
    assert "Showing 3 of 12" in block


# ------------------------------------------------------------------ support audit ---------
def test_audit_records_spread_dispersion_and_source():
    corpus = make_corpus()
    q = query_from(corpus, 4)
    res = retrieve(corpus, q, AnchorPolicy(k=10, min_support=1, render_target=4))
    # spread is reported in STANDARDISED units, the scale the distance ranking used and
    # the scale the prompt names
    rows = corpus.frame.iloc[list(res.anchor_index)]
    z = corpus.matrix[list(res.anchor_index)]
    expected = np.linalg.norm(z - z.mean(axis=0), axis=1).mean()
    assert res.anchor_spread == pytest.approx(expected)
    v = rows["pfs_months"]
    assert res.outcome_dispersion == pytest.approx(v.quantile(.75) - v.quantile(.25))
    assert res.max_distance_used == pytest.approx(max(res.distances))
    assert res.corpus_id and len(res.corpus_id) == 16


def test_relaxed_axes_distinguish_rung_one_from_rung_two():
    corpus = make_corpus()
    q = query_from(corpus, 0)
    direct = retrieve(corpus, q, AnchorPolicy(k=8, min_support=1, render_target=4))
    assert direct.support_label == "direct" and direct.relaxed_axes == ()
    relaxed = retrieve(corpus, q, AnchorPolicy(k=8, min_support=1, render_target=4,
                                               same_action=False))
    assert relaxed.support_label == "transport" and relaxed.relaxed_axes == ("action",)


def test_corpus_id_separates_corpora_that_must_not_be_pooled():
    a = make_corpus()
    b = make_corpus(40)
    assert a.corpus_id != b.corpus_id
    assert a.corpus_id == make_corpus().corpus_id      # and is stable for the same data


# ------------------------------------------------------------------ censoring -------------
def test_censoring_is_stated_in_words_not_as_a_bare_indicator():
    """A model reading "1.38 / 0.00" will read progression where there was none."""
    frame = pd.DataFrame({
        "patient": [f"p{i}" for i in range(30)],
        "arm": ["A"] * 30,
        "burden": np.linspace(0, 1, 30),
        "pfs_mos": np.linspace(1.0, 20.0, 30),
        "pfs_event": [1.0] * 15 + [0.0] * 15,
    })
    corpus = Corpus(frame, channels=["burden"], subject_col="patient", action_col="arm",
                    outcome_cols=["pfs_mos", "pfs_event"],
                    outcome_time_col="pfs_mos", outcome_event_col="pfs_event")
    q = frame.iloc[0].copy()
    q["query_id"] = "q"
    # render ALL retrieved anchors: the nearest few are all events here, so a small
    # a small render_target would never show a censored row and the test would pass
    # vacuously
    p = AnchorPolicy(k=20, min_support=1, render_target=20)
    block = render("compact", corpus, retrieve(corpus, q, p), p)
    assert "progressed" in block
    # not "follow-up ended": the endpoint stopped being observable, which is not the same
    # as the patient leaving the record
    assert "no progression seen through" in block
    # the raw indicator must not appear as its own column
    assert "| pfs_event |" not in block
    # and the aggregate must not average a censored sample into one median
    assert "no progression was seen before observation stopped" in block


def test_no_future_is_shown_for_a_censored_patient():
    """"no progression through 0.7 mo" beside "started another line at 0.7 mo" tells the
    model they progressed — handing back the future the censoring says we do not have."""
    frame = pd.DataFrame({
        "patient": [f"p{i}" for i in range(20)],
        "arm": ["A"] * 20,
        "burden": np.linspace(0, 1, 20),
        "pfs_mos": [0.7] * 20,
        "pfs_event": [1.0] * 10 + [0.0] * 10,
        "os_mos": [88.0] * 20,
        "os_event": [1.0] * 20,
    })
    corpus = Corpus(frame, channels=["burden"], subject_col="patient", action_col="arm",
                    outcome_cols=["pfs_mos", "pfs_event", "os_mos", "os_event"],
                    outcome_time_col="pfs_mos", outcome_event_col="pfs_event",
                    extra_outcomes=[("overall survival", "os_mos", "os_event")])
    q = frame.iloc[0].copy()
    q["query_id"] = "q"
    p = AnchorPolicy(k=20, min_support=1, render_target=20)
    block = render("compact", corpus, retrieve(corpus, q, p), p)
    rows = [ln for ln in block.splitlines() if ln.startswith("| 0")]
    censored = [ln for ln in rows if "no progression seen through" in ln]
    assert censored, "expected censored rows in the rendered table"
    for ln in censored:
        assert "88.0 mo" not in ln          # no survival time leaked
        assert "not known" in ln
    # and the wording never claims follow-up ended when it did not
    assert "follow-up ended" not in block


# ------------------------------------------------- the three launch blockers ---------------
def _sparse_corpus() -> Corpus:
    """Six patients share the query's driver; the rest share only the cancer.

    The shape the relaxation rule exists for. Earlier tests only made the strict set short
    by capping `k`, which is a different code path: the eligible pool was always large.
    """
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame({
        "patient": [f"p{i}" for i in range(n)],
        "arm": ["A"] * n,
        "cancer": ["LUAD"] * n,
        "driver": ["EGFR" if i < 7 else "KRAS" for i in range(n)],
        "burden": rng.normal(size=n),
        "ecog": rng.normal(size=n),
        "pfs_months": rng.exponential(6.0, size=n),
    })
    return Corpus(frame, channels=["burden", "ecog"], subject_col="patient",
                  action_col="arm", outcome_cols=["pfs_months"])


def test_relaxation_runs_when_the_STRICT_POOL_itself_is_too_small():
    corpus = _sparse_corpus()
    q = query_from(corpus, 0)                        # EGFR: six others match exactly
    p = AnchorPolicy(k=40, min_support=10, render_target=10,
                     match_exact=("cancer", "driver"),
                     tiers=(("same cancer, different driver", ("cancer",)),))
    res = retrieve(corpus, q, p)
    assert res.n_eligible == 6                       # genuinely below min_support
    assert res.n_strict == 6 and res.n_anchors == 10
    assert dict(res.tier_counts) == {"strict": 6, "same cancer, different driver": 4}
    # a topped-up set is rung two, never rung one
    assert res.support_label == "transport" and res.supported


def test_a_topped_up_set_still_refuses_when_even_relaxation_cannot_fill_it():
    corpus = _sparse_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=40, min_support=10, render_target=10,
                     match_exact=("cancer", "driver"))       # no tiers to fall back on
    res = retrieve(corpus, q, p)
    assert not res.supported and res.n_anchors == 0
    assert "6 strict" in res.note


def test_every_anchor_has_a_distance_even_after_relaxation():
    """Relaxed anchors have no entry in the strict pool's distance vector."""
    corpus = _sparse_corpus()
    q = query_from(corpus, 0)
    res = retrieve(corpus, q, AnchorPolicy(
        k=40, min_support=10, render_target=10, match_exact=("cancer", "driver"),
        tiers=(("same cancer", ("cancer",)),)))
    assert len(res.distances) == res.n_anchors


def test_the_unanchored_arm_is_the_frozen_prompt_even_with_no_anchors():
    """R3 must not learn that retrieval was attempted, let alone that it failed."""
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=40, min_support=40, match_exact=("cancer",), render_target=4)
    unsupported = retrieve(corpus, q, p)
    assert not unsupported.has_anchors
    assert render("none", corpus, unsupported, p) == ""
    supported = retrieve(corpus, q, AnchorPolicy(k=10, min_support=1, render_target=3))
    assert render("none", corpus, supported, p) == ""


def test_fields_kept_out_of_the_table_are_kept_out_of_the_residual_too():
    """Excluding a channel from the columns and handing it back through the residual would
    put the number in front of the model after deciding it should not see it."""
    corpus = make_corpus()
    q = query_from(corpus, 0)
    p = AnchorPolicy(k=10, min_support=1, render_target=3, fields=("burden",),
                     context_fields=("burden",), n_varying=0)
    res = retrieve(corpus, q, p)
    assert "ecog" in res.residual                      # retrieval still uses it
    for name in ("compact", "table", "prose", "aggregate_only"):
        assert "ecog" not in render(name, corpus, res, p)


# ------------------------------------------------- the query patient ----------------------
def test_the_seed_carries_only_what_the_source_recorded():
    from synthonccases.anchored import baseline_seed
    seed, pinned = baseline_seed({"cancer": "NSCLC", "histology": "Adenocarcinoma",
                                  "age_at_start": 70.0, "months_since_dx": 24.0,
                                  "female": 0.0, "stage_iv": 1.0})
    assert seed["identity_card"]["diagnosis"]["oncotree_code"] == "LUAD"
    assert seed["identity_card"]["demographics"]["age_at_diagnosis"] == 68   # at DIAGNOSIS
    assert seed["identity_card"]["demographics"]["sex"] == "male"
    assert seed["identity_card"]["staging"]["stage_class"] == "metastatic"
    assert set(pinned) == {"identity_card.diagnosis.oncotree_code",
                           "identity_card.demographics.age_at_diagnosis",
                           "identity_card.demographics.sex",
                           "identity_card.staging.stage_class"}
    # what GENIE-BPC does not record must not be asserted as normal
    assert "baseline_health" not in seed["identity_card"]


def test_the_seed_omits_a_leaf_it_cannot_derive_rather_than_guessing():
    from synthonccases.anchored import baseline_seed
    seed, pinned = baseline_seed({"cancer": "BRCA"})       # no age, no sex, no stage
    assert seed == {"identity_card": {"diagnosis": {"oncotree_code": "BRCA"}}}
    assert pinned == ("identity_card.diagnosis.oncotree_code",)


def test_histology_decides_the_code_and_absent_histology_stays_at_the_parent():
    """Mapping every NSCLC row to LUAD asserted adenocarcinoma for the squamous 7%."""
    from synthonccases.anchored.query import oncotree_code
    assert oncotree_code("NSCLC", "Adenocarcinoma") == "LUAD"
    assert oncotree_code("NSCLC", "Squamous cell") == "LUSC"
    assert oncotree_code("NSCLC", "Carcinoma") == "NSCLC"          # NOS: not specifiable
    assert oncotree_code("NSCLC", None) == "NSCLC"
    assert oncotree_code("CRC", "Adenocarcinoma", "C18.0") == "COAD"
    assert oncotree_code("CRC", "Adenocarcinoma", "C20.9") == "READ"
    assert oncotree_code("CRC", "Adenocarcinoma", None) == "COADREAD"
    assert oncotree_code("BRCA", "Invasive lobular carcinoma") == "ILC"
    assert oncotree_code("BRCA", "Other histology") == "BRCA"


def test_the_prefix_is_dated_from_real_intervals_and_stops_at_the_landmark():
    from datetime import date
    from synthonccases.anchored.query import prefix_events
    facts = [{"months": 24.0, "type": "progression"},
             {"months": 0.0, "type": "systemic_treatment_start",
              "regimen": {"label": "FOLFOX", "components": ["oxaliplatin"]}},
             {"months": 30.0, "type": "death"},                    # after the landmark
             {"months": -0.5, "type": "progression"}]              # before diagnosis
    out = prefix_events(facts, origin=date(2015, 7, 1), landmark_months=25.0)
    assert [e["type"] for e in out["events"]] == ["systemic_treatment_start", "progression"]
    assert out["events"][0]["date"] == "2015-07-01"                # interval 0 = diagnosis
    assert out["events"][0]["regimen"]["label"] == "FOLFOX"
    assert out["cutoff_date"] == "2017-07-31"        # 25 months of 30.4375 days, exactly
    assert out["n_dropped_pre_diagnosis"] == 1
    # chronological, which the frozen validator requires
    assert [e["date"] for e in out["events"]] == sorted(e["date"] for e in out["events"])


def test_the_payload_the_frozen_runner_lowers_carries_patient_and_history():
    from datetime import date
    from synthonccases.anchored.query import (
        baseline_seed, conditioning_payload, prefix_events,
    )
    from synthonccases.oncollm.condition_lower import lower_conditioning_cell
    from synthonccases.oncollm.conditioning_audit import validate_observed_prefix

    origin = date(2016, 7, 1)
    seed, pinned = baseline_seed({"cancer": "NSCLC", "histology": "Squamous cell",
                                  "age_at_start": 61.0, "months_since_dx": 12.0,
                                  "female": 1.0, "stage_iv": 1.0})
    fixed = prefix_events([{"months": 12.0, "type": "systemic_treatment_start",
                            "regimen": {"label": "Carboplatin", "components": ["carbo"]}}],
                          origin=origin, landmark_months=12.0)
    fixed.pop("n_dropped_pre_diagnosis")
    spec = lower_conditioning_cell(
        conditioning_payload=conditioning_payload(seed, pinned, fixed))
    validate_observed_prefix(spec, baseline_diagnosis_date=origin)
    assert spec.baseline_seed["identity_card"]["diagnosis"]["oncotree_code"] == "LUSC"
    assert len(spec.observed_prefix.events) == 1
    # the landmark treatment sits AT the cutoff: the model continues from it, never restates it
    assert spec.observed_prefix.cutoff_date == spec.observed_prefix.events[0].date


def test_one_scan_recorded_on_several_rows_becomes_one_event():
    """Repetition in a prompt is emphasis: three identical progression events would have
    made progression look like the patient's dominant tempo."""
    from datetime import date
    from synthonccases.anchored.query import prefix_events
    facts = [{"months": 6.0, "type": "progression", "rationale": "imaging"},
             {"months": 6.0, "type": "progression", "rationale": "imaging"},
             {"months": 6.4, "type": "progression", "rationale": "imaging"},   # 12 days
             {"months": 9.0, "type": "progression", "rationale": "imaging"}]   # 90 days
    out = prefix_events(facts, origin=date(2015, 7, 1), landmark_months=24.0)
    assert len(out["events"]) == 2
    assert out["n_duplicates_collapsed"] == 1 and out["n_merged_within_window"] == 1


def test_a_treatment_change_between_two_reads_keeps_them_apart():
    from datetime import date
    from synthonccases.anchored.query import prefix_events
    facts = [{"months": 6.0, "type": "progression"},
             {"months": 6.2, "type": "systemic_treatment_switch",
              "regimen": {"label": "FOLFIRI", "components": ["irinotecan"]}},
             {"months": 6.4, "type": "progression"}]
    out = prefix_events(facts, origin=date(2015, 7, 1), landmark_months=24.0)
    assert [e["type"] for e in out["events"]] == [
        "progression", "systemic_treatment_switch", "progression"]
    assert out["n_merged_within_window"] == 0


def test_the_landmark_names_its_own_treatment_and_labels_the_concurrent_ones():
    from datetime import date
    from synthonccases.anchored.query import prefix_events
    facts = [{"months": 12.0, "type": "systemic_treatment_switch",
              "regimen": {"label": "Trastuzumab", "components": ["trastuzumab"]}},
             {"months": 12.0, "type": "systemic_treatment_switch",
              "regimen": {"label": "Letrozole", "components": ["letrozole"]}},
             # an imaging read dated ON the landmark: the restaging that drove the switch,
             # and downstream of the decision. The anchors' features exclude it too.
             {"months": 12.0, "type": "progression"}]
    out = prefix_events(facts, origin=date(2016, 7, 1), landmark_months=12.0,
                        landmark_action="Letrozole")
    assert out["n_observations_at_landmark_dropped"] == 1
    assert out["n_treatment_events_at_cutoff"] == 2
    assert out["index_treatment_matched"] is True
    got = {(e["regimen"]["label"], e["rationale"]) for e in out["events"]}
    assert ("Letrozole", "the treatment this trajectory starts from; continue from here") \
        in got
    assert any(lab == "Trastuzumab" and "concurrent" in why for lab, why in got)


def test_a_landmark_whose_treatment_is_missing_is_reported_not_guessed():
    from datetime import date
    from synthonccases.anchored.query import prefix_events
    out = prefix_events([{"months": 3.0, "type": "systemic_treatment_start",
                          "regimen": {"label": "FOLFOX", "components": ["oxaliplatin"]}}],
                        origin=date(2016, 7, 1), landmark_months=12.0,
                        landmark_action="Pembrolizumab")
    assert out["index_treatment_matched"] is False        # the caller refuses on this


# ------------------------------------------------- missingness ----------------------------
def _corpus_with_gaps() -> Corpus:
    """Two kinds of zero: nobody looked, and somebody looked and found nothing."""
    frame = pd.DataFrame({
        "patient": [f"p{i}" for i in range(12)],
        "arm": ["A"] * 12,
        "cancer": ["LUAD"] * 12,
        "sites": [0.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0, 1.0, 0.0, 4.0, 2.0, 1.0],
        "imaging_observed": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "age": [60.0 + i for i in range(12)],
        "pfs_months": [3.0 + i for i in range(12)],
    })
    return Corpus(frame, channels=["sites", "imaging_observed", "age"],
                  subject_col="patient", action_col="arm", outcome_cols=["pfs_months"],
                  observation_flags={"imaging_observed": ("sites",)})


def test_an_unassessed_channel_contributes_nothing_to_the_distance():
    """Otherwise 'never scanned' is numerically identical to 'scanned, nothing found'."""
    corpus = _corpus_with_gaps()
    i = corpus.channels.index("sites")
    absent = corpus.frame["imaging_observed"].to_numpy() == 0
    assert np.allclose(corpus.matrix[absent, i], 0.0)      # the standardised mean
    assert not np.allclose(corpus.matrix[~absent, i], 0.0)


def test_an_unassessed_value_is_named_in_the_table_never_printed_as_zero():
    corpus = _corpus_with_gaps()
    q = corpus.frame.iloc[2].copy()
    q["query_id"] = "q"
    p = AnchorPolicy(k=8, min_support=1, render_target=8, metric="euclidean",
                     fields=("sites", "age"), context_fields=("age",))
    block = render("table", corpus, retrieve(corpus, q, p), p)
    assert "not assessed" in block
    # and the summary must not average the fill value in
    assert "not assessed)" in block or "of 8," in block


def test_the_query_own_unassessed_channels_are_stated_and_kept_out_of_the_residual():
    corpus = _corpus_with_gaps()
    q = corpus.frame.iloc[0].copy()          # this patient was never scanned
    q["query_id"] = "q"
    p = AnchorPolicy(k=8, min_support=1, render_target=5, metric="euclidean",
                     fields=("sites", "age"), context_fields=("age",))
    res = retrieve(corpus, q, p)
    assert "sites" in res.query_unobserved
    block = render("compact", corpus, res, p)
    assert "Not assessed for this patient" in block
    # a residual on a channel nobody measured is arithmetic on a fill value
    assert "- sites:" not in block.split("versus those comparable patients")[-1]


def test_support_is_labelled_with_the_space_it_was_measured_in():
    """A hundred neighbours matched on cancer and stage say nothing about a channel the
    source never records."""
    corpus = make_corpus()
    q = query_from(corpus, 0)
    res = retrieve(corpus, q, AnchorPolicy(k=10, min_support=1, render_target=3,
                                           match_exact=("cancer",)))
    assert "cancer" in res.support_space and "burden" in res.support_space
    assert res.unobserved_axes == ()          # this corpus declares none
