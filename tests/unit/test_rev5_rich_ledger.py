"""Tests for the paper-v1 Rev5-rich ledger ontology."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthonccases.oncollm.replay import (
    BaselineObject,
    LedgerEvent,
    ReplayError,
    replay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "replay_spike" / "her2_brca_baseline_001.json"


def _baseline() -> BaselineObject:
    return BaselineObject.model_validate(json.loads(BASELINE_PATH.read_text()))


def _evt(date_: str, type_: str, **kwargs) -> LedgerEvent:
    return LedgerEvent.model_validate({"date": date_, "type": type_, **kwargs})


def _regimen(label: str, *components: str) -> dict:
    return {"label": label, "components": list(components or (label,))}


def _death(date_: str = "2026-01-30") -> LedgerEvent:
    return _evt(
        date_,
        "death",
        cause_category="cancer_progression",
        cause="progressive metastatic cancer",
        eol_care_pattern="hospice",
    )


def _minimal_ledger_to_death() -> list[LedgerEvent]:
    return [
        _evt(
            "2024-02-15",
            "systemic_treatment_start",
            regimen=_regimen("THP", "trastuzumab", "pertuzumab", "docetaxel"),
        ),
        _evt(
            "2024-05-01",
            "response",
            disease=[{"clone": "C0", "site": "liver", "burden": "small", "kinetic": "shrinking"}],
        ),
        _evt(
            "2025-01-10",
            "progression",
            disease=[{"clone": "C0", "site": "liver", "burden": "moderate", "kinetic": "growing"}],
        ),
        _evt(
            "2025-01-20",
            "systemic_treatment_switch",
            regimen=_regimen("T-DM1", "trastuzumab_emtansine"),
            reason="progression",
        ),
        _death(),
    ]


class TestSystemicTreatment:
    def test_no_emitted_line_required_and_course_order_is_derived(self) -> None:
        traj = replay(_baseline(), _minimal_ledger_to_death())

        assert traj.ledger[0].treatment_id == "tx001"
        assert traj.ledger[0].line == 1
        assert traj.ledger[3].stopped_treatment_id == "tx001"
        assert traj.ledger[3].treatment_id == "tx002"
        assert traj.ledger[3].line == 2
        assert traj.treatment_history[0].line == 1
        assert traj.treatment_history[0].regimen.label == "THP"

    def test_systemic_treatment_modification_requires_active_treatment(self) -> None:
        ledger = [
            _evt(
                "2024-03-01",
                "systemic_treatment_modification",
                modification_type="dose_hold",
                reason="toxicity",
            ),
            _death("2024-04-01"),
        ]

        with pytest.raises(ReplayError, match="requires an active treatment"):
            replay(_baseline(), ledger)

    def test_systemic_treatment_modification_is_materialized(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-15",
                "systemic_treatment_modification",
                modification_type="dose_hold",
                reason="toxicity",
                affected_components=["docetaxel"],
            ),
        )

        traj = replay(_baseline(), ledger)
        mod = next(e for e in traj.ledger if e.type == "systemic_treatment_modification")
        assert mod.modification_type == "dose_hold"
        assert mod.affected_components == ["docetaxel"]


class TestToxicity:
    def test_toxicity_onset_and_resolution_updates_active_state(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="resolved",
                ae="peripheral_neuropathy",
            ),
        )

        traj = replay(_baseline(), ledger)

        onset_snapshot = next(s for s in traj.snapshots if s.event_id == "e002")
        assert [tox.ae for tox in onset_snapshot.active_aes] == ["peripheral_neuropathy"]
        assert traj.resolved_aes[0].ae == "peripheral_neuropathy"
        assert traj.resolved_aes[0].resolution_date.isoformat() == "2024-04-01"
        assert traj.snapshots[-1].active_aes == []

    def test_toxicity_resolution_without_prior_active_toxicity_warns(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt("2024-03-01", "toxicity", toxicity_transition="resolved", ae="rash"),
        )

        traj = replay(_baseline(), ledger)

        assert "SOFT_TOXICITY_RESOLUTION_WITHOUT_ONSET" in {
            warning.rule_id for warning in traj.validation_log
        }

    def test_toxicity_onset_requires_category_and_ctcae_v5_grade(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt("2024-03-01", "toxicity", toxicity_transition="onset", ae="rash"),
        )

        with pytest.raises(ReplayError, match="ae_category"):
            replay(_baseline(), ledger)

    def test_toxicity_category_uses_metabolic_endocrine_not_endocrine(self) -> None:
        event = _evt(
            "2024-03-01",
            "toxicity",
            toxicity_transition="onset",
            ae="hyperglycemia",
            ae_category="metabolic_endocrine",
            ctcae_v5_grade=2,
        )
        assert event.ae_category == "metabolic_endocrine"

        with pytest.raises(ValidationError, match="ae_category"):
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="thyroiditis",
                ae_category="endocrine",
                ctcae_v5_grade=2,
            )

    def test_resolution_must_omit_ctcae_v5_grade(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="resolved",
                ae="rash",
                ctcae_v5_grade=1,
            ),
        )

        with pytest.raises(ReplayError, match="omit ctcae_v5_grade"):
            replay(_baseline(), ledger)

    def test_grade5_toxicity_warns_without_death_terminal(self) -> None:
        ledger = [
            _evt(
                "2024-02-15",
                "systemic_treatment_start",
                regimen=_regimen("THP", "trastuzumab", "pertuzumab", "docetaxel"),
            ),
            _evt(
                "2024-04-10",
                "toxicity",
                toxicity_transition="onset",
                ae="sepsis",
                ae_category="infectious",
                ctcae_v5_grade=5,
                causes=["docetaxel"],
            ),
            _evt("2026-03-15", "last_followup"),
        ]

        traj = replay(_baseline(), ledger)

        assert "SOFT_TOXICITY_GRADE5_NOT_FATAL" in {v.rule_id for v in traj.validation_log}

    def test_grade5_toxicity_with_non_treatment_related_death_warns(self) -> None:
        # G5 onset 5 days before death, but death cause is cancer_progression.
        # Isolates CAUSE_MISMATCH from DELAYED.
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            len(ledger) - 1,
            _evt(
                "2026-01-25",
                "toxicity",
                toxicity_transition="onset",
                ae="sepsis",
                ae_category="infectious",
                ctcae_v5_grade=5,
                causes=["trastuzumab_emtansine"],
            ),
        )

        traj = replay(_baseline(), ledger)
        rule_ids = {v.rule_id for v in traj.validation_log}
        assert "SOFT_TOXICITY_GRADE5_CAUSE_MISMATCH" in rule_ids
        assert "SOFT_TOXICITY_GRADE5_NOT_FATAL" not in rule_ids
        assert "SOFT_TOXICITY_GRADE5_DELAYED" not in rule_ids

    def test_grade5_toxicity_with_treatment_related_death_does_not_warn(self) -> None:
        # G5 onset 5 days before death, cause_category treatment_related.
        # None of the G5 audits should fire.
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            len(ledger) - 1,
            _evt(
                "2026-01-25",
                "toxicity",
                toxicity_transition="onset",
                ae="sepsis",
                ae_category="infectious",
                ctcae_v5_grade=5,
                causes=["trastuzumab_emtansine"],
            ),
        )
        ledger[-1] = _evt(
            "2026-01-30",
            "death",
            cause_category="treatment_related",
            cause="septic shock under cytotoxic chemotherapy",
            eol_care_pattern="inpatient_palliative",
        )

        traj = replay(_baseline(), ledger)
        rule_ids = {v.rule_id for v in traj.validation_log}
        assert "SOFT_TOXICITY_GRADE5_CAUSE_MISMATCH" not in rule_ids
        assert "SOFT_TOXICITY_GRADE5_NOT_FATAL" not in rule_ids
        assert "SOFT_TOXICITY_GRADE5_DELAYED" not in rule_ids

    def test_grade5_toxicity_with_delayed_death_warns(self) -> None:
        # G5 onset early in the trajectory, treatment_related death much later.
        # CAUSE_MISMATCH should NOT fire (cause is correct); DELAYED should
        # fire because the death is >14 days after the G5 event.
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-04-10",
                "toxicity",
                toxicity_transition="onset",
                ae="sepsis",
                ae_category="infectious",
                ctcae_v5_grade=5,
                causes=["docetaxel"],
            ),
        )
        ledger[-1] = _evt(
            "2026-01-30",
            "death",
            cause_category="treatment_related",
            cause="late sequelae of treatment toxicity",
            eol_care_pattern="inpatient_palliative",
        )

        traj = replay(_baseline(), ledger)
        rule_ids = {v.rule_id for v in traj.validation_log}
        assert "SOFT_TOXICITY_GRADE5_DELAYED" in rule_ids
        assert "SOFT_TOXICITY_GRADE5_CAUSE_MISMATCH" not in rule_ids
        assert "SOFT_TOXICITY_GRADE5_NOT_FATAL" not in rule_ids

    def test_grade_change_updates_active_grade_in_place(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="grade_change",
                ae="peripheral_neuropathy",
                ctcae_v5_grade=3,
            ),
        )

        traj = replay(_baseline(), ledger)

        post_grade_change = next(s for s in traj.snapshots if s.event_id == "e003")
        active = [tox for tox in post_grade_change.active_aes if tox.ae == "peripheral_neuropathy"]
        assert len(active) == 1
        assert active[0].ctcae_v5_grade == 3
        # Onset date is preserved on the in-place update.
        assert active[0].onset_date.isoformat() == "2024-03-01"

    def test_grade_change_without_prior_onset_warns(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="grade_change",
                ae="rash",
                ctcae_v5_grade=3,
            ),
        )

        traj = replay(_baseline(), ledger)

        assert "SOFT_TOXICITY_GRADE_CHANGE_WITHOUT_ONSET" in {
            v.rule_id for v in traj.validation_log
        }

    def test_grade_change_to_same_grade_warns_as_noop(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="grade_change",
                ae="peripheral_neuropathy",
                ctcae_v5_grade=2,
            ),
        )

        traj = replay(_baseline(), ledger)

        assert "SOFT_TOXICITY_GRADE_CHANGE_NOOP" in {
            v.rule_id for v in traj.validation_log
        }

    def test_grade_change_requires_ctcae_v5_grade(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="grade_change",
                ae="peripheral_neuropathy",
            ),
        )

        with pytest.raises(ReplayError, match="grade_change requires ctcae_v5_grade"):
            replay(_baseline(), ledger)

    def test_grade_change_materialized_event_inherits_ae_category_and_causes(self) -> None:
        # The prompt promises that grade_change / resolved can omit ae_category
        # and causes (replay carries them from the onset). This must also hold
        # at the materialized event level so event-row analyses (derivation.py)
        # see ae_category / causes on every toxicity event, not only onsets.
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="grade_change",
                ae="peripheral_neuropathy",
                ctcae_v5_grade=3,
            ),
        )

        traj = replay(_baseline(), ledger)

        grade_change_event = next(
            e for e in traj.ledger
            if e.type == "toxicity" and e.toxicity_transition == "grade_change"
        )
        assert grade_change_event.ae_category == "neurologic"
        assert grade_change_event.causes == ["docetaxel"]

    def test_resolved_materialized_event_inherits_ae_category_and_causes(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="resolved",
                ae="peripheral_neuropathy",
            ),
        )

        traj = replay(_baseline(), ledger)

        resolved_event = next(
            e for e in traj.ledger
            if e.type == "toxicity" and e.toxicity_transition == "resolved"
        )
        assert resolved_event.ae_category == "neurologic"
        assert resolved_event.causes == ["docetaxel"]

    def test_grade_change_explicit_ae_category_overrides_inherited(self) -> None:
        # If the model explicitly re-emits ae_category on grade_change (e.g.,
        # for disambiguation), the explicit value wins over inheritance.
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "toxicity",
                toxicity_transition="onset",
                ae="peripheral_neuropathy",
                ae_category="neurologic",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-01",
                "toxicity",
                toxicity_transition="grade_change",
                ae="peripheral_neuropathy",
                ae_category="general_systemic",
                ctcae_v5_grade=3,
                causes=["trastuzumab"],
            ),
        )

        traj = replay(_baseline(), ledger)

        grade_change_event = next(
            e for e in traj.ledger
            if e.type == "toxicity" and e.toxicity_transition == "grade_change"
        )
        assert grade_change_event.ae_category == "general_systemic"
        assert grade_change_event.causes == ["trastuzumab"]

    def test_grade_change_to_grade5_with_cause_mismatch_warns(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-10",
                "toxicity",
                toxicity_transition="onset",
                ae="pneumonitis",
                ae_category="pulmonary",
                ctcae_v5_grade=2,
                causes=["docetaxel"],
            ),
        )
        ledger.insert(
            2,
            _evt(
                "2024-04-20",
                "toxicity",
                toxicity_transition="grade_change",
                ae="pneumonitis",
                ctcae_v5_grade=5,
            ),
        )

        traj = replay(_baseline(), ledger)

        assert "SOFT_TOXICITY_GRADE5_CAUSE_MISMATCH" in {
            v.rule_id for v in traj.validation_log
        }


class TestDeathAudits:
    def test_age_related_frailty_with_active_disease_warns(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger[-1] = _evt(
            "2026-01-30",
            "death",
            cause_category="age_related_frailty",
            cause="progressive frailty and decline in advanced age",
            eol_care_pattern="hospice",
        )

        traj = replay(_baseline(), ledger)

        assert "SOFT_FRAILTY_DEATH_WITH_ACTIVE_DISEASE" in {
            v.rule_id for v in traj.validation_log
        }

    def test_age_related_frailty_without_active_disease_does_not_warn(self) -> None:
        ledger = _minimal_ledger_to_death()
        # Clear all baseline burden before the frailty death.
        ledger.insert(
            -1,
            _evt(
                "2025-12-01",
                "response",
                disease=[
                    {"clone": "C0", "site": "left_breast", "burden": "absent", "kinetic": "cleared"},
                    {"clone": "C0", "site": "liver", "burden": "absent", "kinetic": "cleared"},
                ],
            ),
        )
        ledger[-1] = _evt(
            "2026-01-30",
            "death",
            cause_category="age_related_frailty",
            cause="progressive frailty and decline in advanced age",
            eol_care_pattern="hospice",
        )

        traj = replay(_baseline(), ledger)

        assert "SOFT_FRAILTY_DEATH_WITH_ACTIVE_DISEASE" not in {
            v.rule_id for v in traj.validation_log
        }


class TestClonalEvolution:
    def test_clonal_evolution_event_uses_explicit_new_clone_id(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            2,
            _evt(
                "2024-09-01",
                "clonal_evolution_event",
                new_clone_id="C1",
                alteration="gene_x_variant_y",
                alteration_type="mutation",
                variant="variant_y",
                sample_source="liquid_biopsy_ctDNA",
                parent_clone="C0",
                recognized_date=None,
            ),
        )
        ledger.insert(
            3,
            _evt(
                "2024-09-15",
                "progression",
                disease=[{"clone": "C1", "site": "liver", "burden": "small", "kinetic": "appearing"}],
            ),
        )

        traj = replay(_baseline(), ledger)

        event = next(e for e in traj.ledger if e.type == "clonal_evolution_event")
        assert event.assigned_clone_id == "C1"
        assert event.alteration_id == "amol_001"
        c1 = next(c for c in traj.final_clone_registry if c.clone_id == "C1")
        assert c1.first_presence_date.isoformat() == "2024-09-01"
        assert c1.parent_clone_id_or_null == "C0"
        assert traj.acquired_alterations[0].first_presence_date.isoformat() == "2024-09-01"
        assert traj.acquired_alterations[0].recognized_date is None

    def test_wrong_new_clone_id_fails(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            2,
            _evt(
                "2024-09-01",
                "clonal_evolution_event",
                new_clone_id="C2",
                alteration="gene_x_variant_y",
                sample_source="tissue_biopsy",
                parent_clone="C0",
            ),
        )

        with pytest.raises(ReplayError, match="new_clone_id"):
            replay(_baseline(), ledger)

    def test_histologic_transformation_alteration_type_accepted(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            2,
            _evt(
                "2024-09-01",
                "clonal_evolution_event",
                new_clone_id="C1",
                alteration="small_cell_transformation",
                alteration_type="histologic_transformation",
                sample_source="tissue_biopsy",
                parent_clone="C0",
                mechanism="lineage_plasticity_under_targeted_therapy",
            ),
        )

        traj = replay(_baseline(), ledger)

        event = next(e for e in traj.ledger if e.type == "clonal_evolution_event")
        assert event.alteration_type == "histologic_transformation"
        assert event.alteration == "small_cell_transformation"
        c1 = next(c for c in traj.final_clone_registry if c.clone_id == "C1")
        assert c1.first_presence_date.isoformat() == "2024-09-01"

    def test_translocation_alteration_type_accepted(self) -> None:
        """`translocation` is the clinically standard term for MYC/BCL2
        events in DLBCL ('double-hit lymphoma'); the schema must accept it
        alongside `rearrangement` and `fusion`."""
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            2,
            _evt(
                "2024-09-01",
                "clonal_evolution_event",
                new_clone_id="C1",
                alteration="MYC-IGH",
                alteration_type="translocation",
                sample_source="tissue_biopsy",
                parent_clone="C0",
            ),
        )

        traj = replay(_baseline(), ledger)

        event = next(e for e in traj.ledger if e.type == "clonal_evolution_event")
        assert event.alteration_type == "translocation"
        assert event.alteration == "MYC-IGH"

    def test_parent_clone_unknown_fails(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            2,
            _evt(
                "2024-09-01",
                "clonal_evolution_event",
                new_clone_id="C1",
                alteration="gene_x_variant_y",
                sample_source="tissue_biopsy",
                parent_clone="C99",
            ),
        )

        with pytest.raises(ReplayError, match="CLONAL_EVOLUTION_PARENT_REFERENCE"):
            replay(_baseline(), ledger)


class TestPatientStateAndLocalIntervention:
    def test_systemic_treatment_allows_disease_control_intent(self) -> None:
        event = _evt(
            "2024-02-15",
            "systemic_treatment_start",
            regimen=_regimen("THP", "trastuzumab", "pertuzumab", "docetaxel"),
            intent="disease_control",
        )

        traj = replay(_baseline(), [event, _death()])

        assert traj.ledger[0].intent == "disease_control"

    def test_local_intervention_rejects_systemic_only_intent(self) -> None:
        with pytest.raises(ValidationError, match="not valid for local_intervention"):
            _evt(
                "2024-06-15",
                "local_intervention",
                procedure="radiation_to_bone",
                target_sites=["bone"],
                intent="disease_control",
                modality="radiation",
            )

    @pytest.mark.parametrize("intent", ["adjuvant", "neoadjuvant"])
    def test_local_intervention_allows_adjuvant_neoadjuvant_intent(
        self, intent: str,
    ) -> None:
        event = _evt(
            "2024-06-15",
            "local_intervention",
            procedure="breast_radiotherapy",
            target_sites=["breast"],
            intent=intent,
            modality="radiation",
        )

        assert event.intent == intent

    def test_patient_state_update_carries_intercurrent_physical_event(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            1,
            _evt(
                "2024-03-01",
                "patient_state_update",
                state_change_cause="acute_myocardial_infarction",
                patient={
                    "ecog_ps": 3,
                    "organ_function": {"cardiac": "severe_impairment"},
                },
            ),
        )

        traj = replay(_baseline(), ledger)

        snapshot = next(s for s in traj.snapshots if s.event_id == "e002")
        assert snapshot.patient_state.ecog_ps == 3
        assert snapshot.patient_state.organ_function.cardiac == "severe_impairment"

    def test_local_intervention_surgery_modality_with_immediate_burden_delta(self) -> None:
        ledger = _minimal_ledger_to_death()
        ledger.insert(
            2,
            _evt(
                "2024-06-15",
                "local_intervention",
                procedure="hepatic_resection",
                target_sites=["liver"],
                intent="local_control",
                modality="surgery",
                disease=[{"clone": "C0", "site": "liver", "burden": "absent", "kinetic": "cleared"}],
            ),
        )

        traj = replay(_baseline(), ledger)

        intervention = next(e for e in traj.ledger if e.type == "local_intervention")
        assert intervention.target_sites == ["liver"]
        snapshot = next(s for s in traj.snapshots if s.event_id == "e003")
        liver = next(c for c in snapshot.clonal_state if c.site == "liver")
        assert liver.burden == "absent"


class TestObsoleteFields:
    def test_ledger_schema_rejects_emitted_line(self) -> None:
        event = {
            "date": "2024-02-15",
            "type": "systemic_treatment_start",
            "line": 1,
            "regimen": _regimen("THP", "trastuzumab"),
        }

        parsed = LedgerEvent.model_validate(event)

        assert not hasattr(parsed, "line")

    @pytest.mark.parametrize("field,value", [
        ("weight_change_kg", -4),
        ("treatment_tolerance", "poor"),
        ("pro", {"qol": 50}),
    ])
    def test_patient_delta_rejects_removed_fields(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            LedgerEvent.model_validate({
                "date": "2024-05-01",
                "type": "patient_state_update",
                "patient": {field: value},
            })


class TestBackwardCompat:
    def test_legacy_event_names_still_coerce_for_old_artifacts(self) -> None:
        event = LedgerEvent.model_validate({
            "date": "2024-02-15",
            "type": "treatment_start",
            "line": 1,
            "regimen": "THP",
        })

        assert event.type == "systemic_treatment_start"
        assert event.regimen is not None
        assert event.regimen.label == "THP"

    def test_legacy_organ_trend_names_still_coerce_for_old_artifacts(self) -> None:
        event = LedgerEvent.model_validate({
            "date": "2024-05-01",
            "type": "response",
            "disease": [{"clone": "C0", "organ": "liver", "burden": "small", "trend": "shrinking"}],
        })

        assert event.disease is not None
        assert event.disease[0].site == "liver"
        assert event.disease[0].kinetic == "shrinking"
