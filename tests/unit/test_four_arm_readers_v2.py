"""Regression fixtures for the versioned gate-3/4 readers (corrections R1-R5, 12 Oct 2026)
and the section-6 fallback windows. Hand-written courses only; no registry data.

These check the pipeline, not a hypothesis. Specification:
`manuscript/SCORING_SPEC_amendment_2026-10-12.md`.
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))

import disease_extent_v1 as DE  # noqa: E402
import four_arm_eligibility_v2 as E  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

TAU = 10.0
CUTOFF = "2015-01-01"


def ev(month: float, kind: str, **extra) -> dict:
    when = date.fromisoformat(CUTOFF) + timedelta(days=round(month * R.GENERATED_DAYS))
    return {"date": when.isoformat(), "type": kind, **extra}


def months(m: float) -> float:
    """The month a fixture event lands on after day rounding."""
    return round(m * R.GENERATED_DAYS) / R.GENERATED_DAYS


def lesion(site: str, burden: str, kinetic: str, clone: str = "C0") -> dict:
    return {"clone": clone, "site": site, "site_class": site, "burden": burden,
            "kinetic": kinetic}


def gen(ledger: list[dict], horizon: float = 12.0, *, state: str = "progressing",
        presence: str | None = "evidence", sites: list[str] | None = None,
        extent: str = "yes", index: str = "carboplatin + pemetrexed") -> E.Course:
    return E.generated_course(ledger, horizon, cutoff=CUTOFF, landmark_state=state,
                              landmark_presence=presence, landmark_sites=sites,
                              landmark_extent=extent, index_regimen=index)


def rounded(path) -> list[tuple[float, float, str]]:
    return [(round(a, 3), round(b, 3), s) for a, b, s in path]


# ======================================================================================
# R1: generated truth updates cumulatively, in event order
# ======================================================================================

LIVER_CLEARS = [ev(3.0, "patient_state_update", patient={"ecog_ps": 1},
                   disease=[lesion("liver", "absent", "cleared")]),
                ev(30.0, "death", cause="x")]


@pytest.mark.parametrize("conv", ["primary", "sensitivity"])
def test_liver_only_clearance_is_progressing_then_ned_until_death(conv: str) -> None:
    with E.convention(conv):
        c = gen(LIVER_CLEARS, 12.0, sites=["liver"])
    t3 = months(3.0)
    assert rounded(c.path) == [(0.0, round(t3, 3), "progressing"),
                               (round(t3, 3), 12.0, "no_active_disease")]
    # no artificial alternation at any month
    assert all(E.state_at(c.path, m + 0.5) == "no_active_disease" for m in range(3, 12))
    # the query-calendar projection: the change is seen at the next read, then holds
    seen = E.observe(c.path, [0.0, 2.0, 5.0, 8.0, 11.0], 12.0)
    assert rounded(seen) == [(0.0, 5.0, "progressing"), (5.0, 12.0, "no_active_disease")]
    with E.convention(conv):
        r = E.resolve(c, [0.0, 2.0, 5.0, 8.0, 11.0], 12.0)
    assert r.full_core


def test_liver_clearance_course_is_identical_under_both_conventions() -> None:
    with E.convention("primary"):
        a = gen(LIVER_CLEARS, 24.0, sites=["liver"])
    with E.convention("sensitivity"):
        b = gen(LIVER_CLEARS, 24.0, sites=["liver"])
    assert a.path == b.path


def test_reciprocal_ned_then_new_site_is_progressing_without_reverting() -> None:
    ledger = [ev(5.0, "progression", disease=[lesion("liver", "small", "appearing")]),
              ev(8.0, "patient_state_update", patient={"ecog_ps": 1}),
              ev(30.0, "death", cause="x")]
    c = gen(ledger, 24.0, state="no_active_disease", presence="no evidence", sites=[])
    t5 = round(months(5.0), 3)
    assert rounded(c.path) == [(0.0, t5, "no_active_disease"), (t5, 24.0, "progressing")]
    seen = E.observe(c.path, [0.0, 3.0, 6.0, 9.0, 15.0], 24.0)
    assert rounded(seen) == [(0.0, 6.0, "no_active_disease"), (6.0, 24.0, "progressing")]


def test_ned_superseded_by_a_later_status_only_event() -> None:
    # a progression event naming no lesion change still makes an older NED reading active
    ledger = [ev(2.0, "response", disease=[lesion("liver", "absent", "cleared")]),
              ev(6.0, "progression", disease=[lesion("lung", "small", "growing")]),
              ev(30.0, "death", cause="x")]
    c = gen(ledger, 12.0, sites=["liver"])
    assert [s for _a, _b, s in c.path] == ["progressing", "no_active_disease", "progressing"]


def test_genuine_same_time_contradiction_stays_unknown_until_the_next_update() -> None:
    # 'progression' whose own disease block clears every known lesion: NED and active
    # disease asserted at the same instant. Not created by the adapter; kept distinct.
    ledger = [ev(4.0, "progression", disease=[lesion("liver", "absent", "cleared")]),
              ev(7.0, "response", disease=[lesion("lung", "small", "shrinking")]),
              ev(30.0, "death", cause="x")]
    g = R.generated_course(ledger, cutoff=CUTOFF, landmark_state="stable", horizon=12.0,
                           landmark_presence="evidence", landmark_sites=["liver"],
                           landmark_extent="yes", index_regimen="carboplatin")
    assert g.conflicts == 1
    assert [s for _a, _b, s in g.path] == ["stable", "unknown", "responding"]
    c = gen(ledger, 12.0, state="stable", sites=["liver"])
    r = E.resolve(c, [0.0, 3.0, 6.0, 9.0], 12.0)
    assert not r.full_core and E.DISEASE_AFTER_LANDMARK in r.reasons


def test_restated_values_never_beat_newer_evidence_in_a_long_silence() -> None:
    ledger = [ev(1.0, "response", disease=[lesion("liver", "small", "shrinking")]),
              ev(2.0, "patient_state_update", patient={"ecog_ps": 0},
                 disease=[lesion("liver", "absent", "cleared")]),
              ev(40.0, "death", cause="x")]
    c = gen(ledger, 24.0, sites=["liver"])
    assert [s for _a, _b, s in c.path] == ["progressing", "responding",
                                            "no_active_disease"]


def test_unknown_baseline_clearance_is_not_remission() -> None:
    ledger = [ev(3.0, "response", disease=[lesion("liver", "absent", "cleared")]),
              ev(30.0, "death", cause="x")]
    # metastatic with no named site: completeness cannot be checked
    c = gen(ledger, 12.0, state="stable", sites=[], extent="yes")
    assert "no_active_disease" not in {s for _a, _b, s in c.path}


# ======================================================================================
# R3: drug classes
# ======================================================================================

def test_previously_missing_drugs_are_classified() -> None:
    chemo = ("floxuridine", "mitomycin", "vinblastine sulfate", "ixabepilone")
    targeted = ("everolimus", "alpelisib", "capivasertib", "savolitinib", "vemurafenib",
                "olaratumab", "margetuximab")
    for d in chemo:
        assert R.classify(d).modalities == frozenset({"chemotherapy"}), d
    for d in targeted:
        assert R.classify(d).modalities == frozenset({"targeted"}), d
    assert R.classify("Goserlin Acetate(Zoladex)").modalities == frozenset({"endocrine"})
    assert R.classify("Ziv Aflibercept").modalities == frozenset({"targeted"})
    assert R.classify("T-DM1").drug == "trastuzumab emtansine"
    # GENIE truncates some synonym lists without a closing parenthesis
    assert R.classify("Dacarbazine(DTIC,Deticene").modalities == frozenset({"chemotherapy"})


def test_adc_convention_and_adjuncts() -> None:
    both = frozenset({"chemotherapy", "targeted"})
    for name in ("Trastuzumab Emtansine", "ado-trastuzumab_emtansine",
                 "Trastuzumab Deruxtecan", "sacituzumab_govitecan"):
        assert R.classify(name).modalities == both, name
    for name in ("Leucovorin Calcium(Wellcovorin)", "levoleucovorin", "tipiracil"):
        assert R.classify(name).kind == R.ADJUNCT
    assert R.classify("Trifluridine and Tipiracil Hydrochloride").modalities == \
        frozenset({"chemotherapy"})


def test_identity_is_never_guessed() -> None:
    for name in ("Investigational Drug", "investigational pi3k inhibitor", "Other NOS",
                 "chemotherapy", "regn5067", "novelagent", ""):
        assert R.classify(name).kind == R.UNRECOGNISED, name


def test_drug_table_has_no_conflicting_alias_and_flags_conventions() -> None:
    table = R.load_drug_table()
    assert table["goserlin"].drug == "goserelin"
    import csv
    rows = list(csv.DictReader(R.DRUG_TABLE.open()))
    adc = [r for r in rows if "ADC" in r["note"]]
    assert adc and all(r["confirm"] == "confirmed by Julien 2026-10-12" for r in adc)
    assert not [r for r in rows if r["confirm"] == "for Julien to confirm"]
    assert R.classify("tretinoin").modalities == frozenset({"targeted"})


def test_partially_unrecognised_combination_has_no_modality_set() -> None:
    start = R.RegimenStart(0.0, (R.classify("letrozole"), R.classify("novelagent")))
    assert start.modality_set is None and start.cancer_directed is True
    adj = R.RegimenStart(0.0, (R.classify("leucovorin"),))
    assert adj.cancer_directed is False and adj.modality_set == frozenset()
    blank = R.RegimenStart(0.0, (R.classify("tx"),))
    assert blank.cancer_directed is None


def test_partially_unrecognised_generated_regimen_is_unresolved() -> None:
    ledger = [ev(30.0, "death", cause="x")]
    c = gen(ledger, 12.0, state="stable", index="letrozole + novelagent")
    r = E.resolve(c, [0.0, 3.0], 12.0)
    assert r.blocks[:3] == (True, True, True) and not r.blocks[3]
    assert "treatment_modality_unrecognised_agent_exposed" in r.reasons


def test_adjunct_only_regimen_is_not_a_regimen_and_not_exposure() -> None:
    ledger = [ev(4.0, "systemic_treatment_switch",
                 regimen={"label": "bsc", "components": ["best_supportive_care"]}),
              ev(30.0, "death", cause="x")]
    c = gen(ledger, 12.0, state="stable", index="fluorouracil + leucovorin")
    vals, _ = R.treatment_coordinates(c.treatment, c.death, 12.0)
    assert vals["n_subsequent_regimens"] == 0.0
    assert vals["months_any"] == pytest.approx(months(4.0))
    assert vals["months_chemotherapy"] == pytest.approx(months(4.0))
    assert vals["n_modality_changes"] == 0.0


# ======================================================================================
# R2: real agent fields
# ======================================================================================

def agent(reg: str, name: str, start: float, *, stop: float = math.nan,
          last: float = math.nan, dc: str = "Yes") -> R.RealAgent:
    return R.RealAgent(regimen=reg, name=name, start=TAU + start,
                       stop=TAU + stop if math.isfinite(stop) else math.nan,
                       last_admin=TAU + last if math.isfinite(last) else math.nan,
                       discontinued=dc, investigational="investigational" in name.lower(),
                       source="matched")


def test_stop_versus_last_administration_are_kept_distinct() -> None:
    stop = R.real_agent_exposure(agent("1:1", "Letrozole", 0.0, stop=5.0), TAU)
    assert stop[:2] == (((0.0, 5.0),), ())
    ongoing = R.real_agent_exposure(agent("1:1", "Letrozole", 0.0, last=5.0, dc="No"),
                                    TAU)
    assert ongoing[0] == ((0.0, 5.0),) and ongoing[1] == ((5.0, math.inf),)
    assert ongoing[2] == "not_discontinued_after_last_administration"
    unknown = R.real_agent_exposure(agent("1:1", "Letrozole", 0.0, dc="Unknown"), TAU)
    assert unknown[0] == () and unknown[1] == ((0.0, math.inf),)
    yes_no_stop = R.real_agent_exposure(agent("1:1", "Letrozole", -2.0), TAU)
    assert yes_no_stop[1] == ((-2.0, math.inf),)
    redacted = R.real_agent_exposure(agent("1:1", "Investigational Drug", 0.0, stop=4.0),
                                     TAU)
    assert redacted[0] == () and redacted[2] == "investigational_redacted"


def test_no_discontinuation_is_not_carried_to_last_follow_up() -> None:
    tx = R.real_treatment([agent("1:1", "Letrozole", 0.0, last=5.0, dc="No")], TAU)
    short, _ = R.treatment_coordinates(tx, math.inf, 3.0)
    assert short["months_endocrine"] == pytest.approx(3.0)          # documented to 5
    long, why = R.treatment_coordinates(tx, math.inf, 12.0)
    assert long["months_endocrine"] is None and long["months_any"] is None
    assert "treatment_exposure_unresolved_not_discontinued_after_last_administration" in why
    assert long["months_chemotherapy"] == 0.0                       # other channels known


def test_pre_landmark_treatment_overlapping_the_window_is_included() -> None:
    agents = [agent("1:1", "Goserlin Acetate", -6.0, stop=4.0),
              agent("1:2", "Palbociclib", 0.0, stop=8.0),
              agent("1:2", "Letrozole", 0.0, stop=8.0)]
    vals, _ = R.treatment_coordinates(R.real_treatment(agents, TAU), math.inf, 12.0)
    assert vals["months_endocrine"] == pytest.approx(8.0)
    assert vals["months_targeted"] == pytest.approx(8.0)
    assert vals["n_subsequent_regimens"] == 0.0


def test_pre_landmark_unknown_end_blocks_treatment() -> None:
    agents = [agent("1:1", "Letrozole", -12.0, dc="Yes"), agent("1:2", "Palbociclib", 0.0,
                                                                  stop=8.0)]
    vals, why = R.treatment_coordinates(R.real_treatment(agents, TAU), math.inf, 12.0)
    assert vals["months_endocrine"] is None
    assert "treatment_exposure_unresolved_discontinued_without_stop_pre_landmark" in why


def _write_genie(root: Path, *, extra_timeline: str = "") -> Path:
    root.mkdir()
    (root / "cancer_level_dataset_index.csv").write_text("record_id,ca_seq\nP1,0\n")
    head = ",".join(["record_id", "ca_seq", "regimen_number", "drugs_dc_ynu"]
                    + [f"{f}{k}" for k in range(1, 6)
                       for f in ("drugs_drug_", "dx_drug_start_int_", "dx_drug_end_int_",
                                 "dx_drug_end_or_lastadm_int_")])
    blank = ",,,"
    rows = [
        # index cancer, regimen 1: carboplatin (stop 91 d) + pemetrexed (stop 91 d)
        "P1,0,1,Yes,Carboplatin(Paraplatin),304,395,395,Pemetrexed Disodium(Alimta),304,"
        "395,395," + ",".join([blank] * 3),
        # index cancer, regimen 2: two masked investigational drugs (timeline holds one)
        "P1,0,2,Yes,Investigational Drug,500,,,Investigational Drug,500,,,"
        + ",".join([blank] * 3),
        # filed under another ca_seq, but the timeline attributes it to the index cancer
        "P1,1,3,No,Letrozole,600,,700," + ",".join([blank] * 4),
    ]
    (root / "regimen_cancer_level_dataset.csv").write_text(
        head + "\n" + "\n".join(rows) + "\n")
    tl = ["PATIENT_ID\tSTART_DATE\tSTOP_DATE\tAGENT\tREGIMEN_NUMBER\tCA_SEQ\tDRUGS_DC_YN",
          "P1\t304\t395\tCarboplatin\t1\t0\tYes", "P1\t304\t395\tPemetrexed Disodium\t1\t0\tYes",
          "P1\t500\t\tInvestigational Drug\t2\t0\tYes", "P1\t600\t\tLetrozole\t3\t0\tNo"]
    if extra_timeline:
        tl.append(extra_timeline)
    (root / "data_timeline_treatment.txt").write_text("\n".join(tl) + "\n")
    return root


def test_agent_reader_reconciles_the_two_exports(tmp_path: Path) -> None:
    agents, counts = R.load_real_agents(_write_genie(tmp_path / "g"))
    a = agents["P1"]
    assert counts["matched"] == 3
    assert counts["regimen_table_only"] == 1                      # second masked drug
    assert counts["timeline_row_filed_under_other_ca_seq"] == 1
    letro = next(x for x in a if x.name == "Letrozole")
    assert letro.discontinued == "No" and math.isnan(letro.stop)
    assert letro.last_admin == pytest.approx(700 / R.REGISTRY_DAYS)
    carbo = next(x for x in a if x.name.startswith("Carboplatin"))
    assert carbo.stop == pytest.approx(395 / R.REGISTRY_DAYS)


def test_agent_reader_refuses_an_unexplained_timeline_row(tmp_path: Path) -> None:
    root = _write_genie(tmp_path / "g", extra_timeline="P1\t900\t950\tCisplatin\t9\t0\tYes")
    with pytest.raises(ValueError):
        R.load_real_agents(root)


# ======================================================================================
# R4: per-agent intervals and generated modifications
# ======================================================================================

def test_components_keep_their_own_stops() -> None:
    agents = [agent("1:1", "Carboplatin", 0.0, stop=3.0),
              agent("1:1", "Pemetrexed Disodium", 0.0, stop=3.0),
              agent("1:1", "Bevacizumab", 0.0, stop=9.0)]
    vals, _ = R.treatment_coordinates(R.real_treatment(agents, TAU), math.inf, 12.0)
    assert vals["months_chemotherapy"] == pytest.approx(3.0)        # not nine
    assert vals["months_targeted"] == pytest.approx(9.0)
    assert vals["months_any"] == pytest.approx(9.0)


def test_component_discontinuation_reaches_the_reader() -> None:
    # Fixture: letrozole + unmapped agent, then letrozole discontinued.
    ledger = [ev(4.0, "systemic_treatment_modification",
                 modification_type="discontinuation_of_component",
                 affected_components=["letrozole"]),
              ev(30.0, "death", cause="x")]
    c = gen(ledger, 12.0, state="stable", index="letrozole + novelagent")
    vals, _ = R.treatment_coordinates(c.treatment, c.death, 12.0)
    assert vals["months_endocrine"] is None                       # novelagent still running
    letro = next(a for a in c.treatment.agents if a.name == "letrozole")
    assert letro.segments == ((0.0, pytest.approx(months(4.0))),)


def test_component_discontinuation_with_known_drugs_gives_traced_durations() -> None:
    ledger = [ev(4.0, "systemic_treatment_modification",
                 modification_type="discontinuation_of_component",
                 affected_components=["palbociclib"]),
              ev(9.0, "systemic_treatment_stop"), ev(30.0, "death", cause="x")]
    c = gen(ledger, 12.0, state="stable", index="letrozole + palbociclib")
    vals, _ = R.treatment_coordinates(c.treatment, c.death, 12.0)
    assert vals["months_targeted"] == pytest.approx(months(4.0))
    assert vals["months_endocrine"] == pytest.approx(months(9.0))
    assert vals["months_any"] == pytest.approx(months(9.0))
    assert vals["n_subsequent_regimens"] == 0.0                    # not a new regimen


def test_short_hold_is_exposure_long_hold_is_a_gap() -> None:
    short = [ev(2.0, "systemic_treatment_modification", modification_type="dose_hold",
                affected_components=["pemetrexed"]),
             ev(3.0, "systemic_treatment_modification", modification_type="dose_resume",
                affected_components=["pemetrexed"]),
             ev(30.0, "death", cause="x")]
    long = [dict(short[0]),
            ev(5.0, "systemic_treatment_modification", modification_type="dose_resume",
               affected_components=["pemetrexed"]),
            ev(30.0, "death", cause="x")]
    vs, _ = R.treatment_coordinates(gen(short, 12.0).treatment, 30.0, 12.0)
    vl, _ = R.treatment_coordinates(gen(long, 12.0).treatment, 30.0, 12.0)
    assert vs["months_chemotherapy"] == pytest.approx(12.0)
    # carboplatin continues, so chemotherapy time is unchanged; pemetrexed has a gap
    assert vl["months_chemotherapy"] == pytest.approx(12.0)
    pem = next(a for a in gen(long, 12.0).treatment.agents if a.name == "pemetrexed")
    assert len(pem.segments) == 2
    assert pem.segments[0][1] == pytest.approx(months(2.0))
    assert pem.segments[1][0] == pytest.approx(months(5.0))


def test_ambiguous_discontinuation_is_unresolved_not_ignored() -> None:
    ledger = [ev(4.0, "systemic_treatment_modification",
                 modification_type="discontinuation_of_component",
                 affected_components=["cisplatin"]),
              ev(30.0, "death", cause="x")]
    vals, why = R.treatment_coordinates(gen(ledger, 12.0).treatment, 30.0, 12.0)
    assert vals["months_chemotherapy"] is None
    assert "treatment_exposure_unresolved_generated_ambiguous_modification" in why


def test_switch_closes_every_component_and_counts_one_regimen() -> None:
    ledger = [ev(6.0, "systemic_treatment_switch",
                 regimen={"label": "docetaxel", "components": ["docetaxel"]}),
              ev(30.0, "death", cause="x")]
    vals, _ = R.treatment_coordinates(gen(ledger, 12.0).treatment, 30.0, 12.0)
    assert vals["n_subsequent_regimens"] == 1.0
    assert vals["n_modality_changes"] == 0.0                        # chemo -> chemo
    assert vals["first_change"] == 0.0 and vals["time_to_first_change"] == 12.0


def test_start_during_active_treatment_is_read_as_the_explicit_switch() -> None:
    """Gate-3 v2 R1: frozen replay promotes such a start to a switch; never additive."""
    reg = {"label": "docetaxel", "components": ["docetaxel"]}
    tail = [ev(30.0, "death", cause="x")]
    as_start = gen([ev(6.0, "systemic_treatment_start", regimen=reg)] + tail, 12.0).treatment
    as_switch = gen([ev(6.0, "systemic_treatment_switch", regimen=reg)] + tail, 12.0).treatment
    assert as_start.agents == as_switch.agents and as_start.starts == as_switch.starts
    assert as_start.active_start_conversions == 1 and as_switch.active_start_conversions == 0
    assert R.treatment_coordinates(as_start, 30.0, 12.0) == \
        R.treatment_coordinates(as_switch, 30.0, 12.0)
    index_agents = [a for a in as_start.agents if a.regimen == "g1"]
    assert all(a.segments[-1][1] == pytest.approx(months(6.0)) for a in index_agents)


def test_start_after_a_stop_is_not_a_conversion() -> None:
    ledger = [ev(3.0, "systemic_treatment_stop"),
              ev(6.0, "systemic_treatment_start",
                 regimen={"label": "docetaxel", "components": ["docetaxel"]}),
              ev(30.0, "death", cause="x")]
    assert gen(ledger, 12.0).treatment.active_start_conversions == 0


def test_exposure_stops_at_death() -> None:
    vals, _ = R.treatment_coordinates(gen([ev(5.0, "death", cause="x")], 12.0).treatment,
                                      months(5.0), 12.0)
    assert vals["months_any"] == pytest.approx(months(5.0))


# ======================================================================================
# R5: exposure versus regimen events at exactly H
# ======================================================================================

def test_unmapped_regimen_exactly_at_h_blocks_change_but_not_exposure() -> None:
    agents = [agent("1:1", "Carboplatin", 0.0, stop=3.0),
              agent("1:2", "Investigational Drug", 12.0, stop=14.0)]
    vals, why = R.treatment_coordinates(R.real_treatment(agents, TAU), math.inf, 12.0)
    # zero-length overlap with [0, 12): exposure coordinates resolved ...
    assert vals["months_any"] == pytest.approx(3.0)
    assert vals["months_chemotherapy"] == pytest.approx(3.0)
    # ... but the start at H is a regimen event whose modality set is unknown
    assert vals["n_subsequent_regimens"] == 1.0
    assert vals["n_modality_changes"] is None and vals["first_change"] is None
    assert "treatment_modality_set_unresolved" in why


def test_change_exactly_at_h_has_indicator_one_and_just_after_h_none() -> None:
    at_h = [agent("1:1", "Carboplatin", 0.0, stop=3.0), agent("1:2", "Osimertinib", 12.0,
                                                               stop=20.0)]
    vals, _ = R.treatment_coordinates(R.real_treatment(at_h, TAU), math.inf, 12.0)
    assert vals["first_change"] == 1.0 and vals["time_to_first_change"] == 12.0
    after = [agent("1:1", "Carboplatin", 0.0, stop=3.0), agent("1:2", "Osimertinib", 12.01,
                                                                stop=20.0)]
    vals, _ = R.treatment_coordinates(R.real_treatment(after, TAU), math.inf, 12.0)
    assert vals["first_change"] == 0.0 and vals["n_subsequent_regimens"] == 0.0


def test_co_dated_starts_are_one_group_whatever_their_order() -> None:
    a = [agent("1:1", "Carboplatin", 0.0, stop=3.0),
         agent("1:2", "Letrozole", 5.0, stop=9.0), agent("1:3", "Goserlin Acetate", 5.0,
                                                          stop=9.0)]
    b = [a[0], a[2], a[1]]
    va, _ = R.treatment_coordinates(R.real_treatment(a, TAU), math.inf, 12.0)
    vb, _ = R.treatment_coordinates(R.real_treatment(b, TAU), math.inf, 12.0)
    assert va == vb and va["n_subsequent_regimens"] == 1.0 and va["n_modality_changes"] == 1


# ======================================================================================
# Eligibility and coordinates use the same object; a traced example
# ======================================================================================

def met() -> DE.MetRecord:
    return DE.MetRecord(stage="IV", ever=1.0, first_month=1.0, sites=(),
                        site_detail_available=True, undated_sites=0)


def real(status, *, death=math.inf, last_contact=TAU + 20.0, agents=()) -> E.Course:
    p = GV2.Patient(pid="toy", context={"mets": met()}, status=list(status),
                    presence=[], lines=[], death=death, last_contact=last_contact)
    with E.convention("primary"):
        return E.real_course(p, TAU, 12.0, list(agents))


TRACED_AGENTS = [agent("1:1", "Carboplatin", 0.0, stop=3.0),
                 agent("1:1", "Pemetrexed Disodium", 0.0, stop=3.0),
                 agent("1:1", "Pembrolizumab", 0.0, stop=9.0),
                 agent("1:2", "Docetaxel", 10.0, stop=11.5)]
TRACED_STATUS = [(TAU - 1, "stable"), (TAU + 2, "responding"), (TAU + 6, "stable"),
                 (TAU + 9, "progressing")]


def test_traced_coordinate_values() -> None:
    c = real(TRACED_STATUS, agents=TRACED_AGENTS)
    cal = [0.0, 2.0, 6.0, 9.0]
    out = E.coordinates(c, cal, 12.0)
    raw = out["raw"]
    assert raw["alive_at_h"] == 1.0 and raw["months_alive"] == 12.0
    assert raw["months_stable"] == pytest.approx(5.0)
    assert raw["months_responding"] == pytest.approx(4.0)
    assert raw["months_progressing"] == pytest.approx(3.0)
    assert raw["months_no_active_disease"] == 0.0
    assert raw["entries_progressing"] == 1.0 and raw["entries_responding"] == 1.0
    assert raw["months_any"] == pytest.approx(10.5)
    assert raw["months_chemotherapy"] == pytest.approx(4.5)
    assert raw["months_immunotherapy"] == pytest.approx(9.0)
    assert raw["months_targeted"] == 0.0 and raw["months_endocrine"] == 0.0
    assert raw["n_subsequent_regimens"] == 1.0
    assert raw["n_modality_changes"] == 1.0
    assert raw["first_change"] == 1.0 and raw["time_to_first_change"] == pytest.approx(10.0)
    v = out["weighted_vector"]
    assert len(v) == 17
    assert v[0] == pytest.approx(math.sqrt(0.125))                       # alive, survival
    assert v[3] == pytest.approx(0.25 * 4.0 / 12.0)                      # responding
    assert v[6] == pytest.approx(math.sqrt(0.125) * 0.5)                 # 1 entry -> 1/2
    assert v[8] == pytest.approx((1 / 6) * 10.5 / 12.0)                  # any therapy
    # squared block weights sum to 1/4 per block for unit coordinates
    assert E.resolve(c, cal, 12.0).full_core


def test_resolution_and_values_come_from_the_same_treatment_object() -> None:
    agents = TRACED_AGENTS + [agent("1:3", "Letrozole", 11.0, last=11.2, dc="No")]
    c = real(TRACED_STATUS, agents=agents)
    out = E.coordinates(c, [0.0, 2.0, 6.0, 9.0], 12.0)
    r = E.resolve(c, [0.0, 2.0, 6.0, 9.0], 12.0)
    assert out["raw"]["months_endocrine"] is None and out["weighted_vector"] is None
    assert not r.blocks[3]
    assert r.blocks[3] == all(out["raw"][k] is not None for k in R.TREATMENT_COORDS)


# ======================================================================================
# Section 6 fallback windows
# ======================================================================================

def test_common_window_uses_donor_support_and_post_death_time() -> None:
    q = real([(TAU - 1, "stable"), (TAU + 3, "stable"), (TAU + 6, "progressing")],
             agents=TRACED_AGENTS[:1])
    cal = [0.0, 3.0, 6.0]
    qs = E.supported(q, cal, 12.0)
    assert E.measure(qs) == pytest.approx(12.0)
    # donor followed only to month 5: comparable time 0-5
    censored = real([(TAU - 1, "stable")], last_contact=TAU + 5.0, agents=TRACED_AGENTS[:1])
    assert E.measure(E.donor_window(qs, censored, cal, 12.0)) == pytest.approx(5.0)
    # donor dead at month 2: its post-death time is known, so the whole window counts
    dead = real([(TAU - 1, "stable")], death=TAU + 2.0, last_contact=TAU + 2.0,
                agents=TRACED_AGENTS[:1])
    assert E.measure(E.donor_window(qs, dead, cal, 12.0)) == pytest.approx(12.0)
    # donor with no state at its landmark and none after: no comparable time
    blind = real([], agents=TRACED_AGENTS[:1])
    assert E.measure(E.donor_window(qs, blind, cal, 12.0)) == 0.0


def test_donor_first_documented_late_is_read_only_from_the_next_query_visit() -> None:
    q = real([(TAU - 1, "stable"), (TAU + 3, "stable"), (TAU + 6, "stable")],
             agents=TRACED_AGENTS[:1])
    late = real([(TAU + 4, "stable")], agents=TRACED_AGENTS[:1])
    w = E.donor_window(E.supported(q, [0.0, 3.0, 6.0], 12.0), late, [0.0, 3.0, 6.0], 12.0)
    assert w == [(6.0, 12.0)]


def test_fallback_patient_needs_seven_usable_donors_of_one_month() -> None:
    six = [[(0.0, 2.0)]] * 6 + [[(0.0, 0.5)]] * 14
    seven = [[(0.0, 1.0)]] * 7 + [[]] * 13
    assert not E.fallback_patient(six)["eligible"]
    got = E.fallback_patient(seven)
    assert got["eligible"] and got["n_usable"] == 7
    assert got["scored_months"] == pytest.approx(7.0) and not got["all_twenty_usable"]
    assert E.fallback_patient([[(0.0, 3.0)]] * 20)["all_twenty_usable"]
