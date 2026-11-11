"""Tests for the Rev5 lean-ledger replay spike."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from synthonccases.oncollm.replay import ReplayError, replay

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "replay_spike"


def _load_fixture() -> tuple[dict, list[dict]]:
    baseline = json.loads((FIXTURE_DIR / "her2_brca_baseline_001.json").read_text())
    ledger = json.loads((FIXTURE_DIR / "her2_brca_ledger_001.json").read_text())
    return baseline, ledger


class TestHappyPath:
    def test_hand_written_fixture_replays_cleanly(self) -> None:
        baseline, ledger = _load_fixture()

        trajectory = replay(baseline, ledger)

        assert trajectory.validation_log == []
        assert len(trajectory.snapshots) == 9
        assert len(trajectory.ledger) == 8
        assert trajectory.snapshots[0].state_id == "s0"
        assert trajectory.snapshots[0].event_type == "baseline"
        assert trajectory.snapshots[0].total_burden_band == "bulky"
        assert trajectory.ledger[0].event_id == "e001"
        assert trajectory.ledger[0].treatment_id == "tx001"
        assert trajectory.ledger[3].stopped_treatment_id == "tx001"
        assert trajectory.ledger[3].treatment_id == "tx002"
        assert trajectory.ledger[5].stopped_treatment_id == "tx002"
        assert trajectory.ledger[5].treatment_id == "tx003"

        final = trajectory.snapshots[-1]
        assert final.event_type == "death"
        assert final.vital_status == "deceased"
        assert final.outcome is not None
        assert final.outcome.type == "death"
        assert final.outcome.cause_category == "cancer_progression"
        assert final.active_treatments == []

    def test_materialized_snapshot_carries_forward_unchanged_state(self) -> None:
        baseline, ledger = _load_fixture()

        trajectory = replay(baseline, ledger)

        response_snapshot = trajectory.snapshots[2]
        burden_by_organ = {entry.organ: entry.burden for entry in response_snapshot.clonal_state}
        assert burden_by_organ["left_breast"] == "bulky"
        assert burden_by_organ["liver"] == "small"
        assert response_snapshot.patient_state.ecog == 0
        assert response_snapshot.patient_state.weight_kg == 68
        assert response_snapshot.patient_state.organ_function.renal == "normal"

    def test_materialized_clonal_event_preserves_self_annotation_buckets(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "clonal_evolution_event",
                "new_clone_id": "C1",
                "gene": "MET",
                "alteration": "MET amplification",
                "alteration_type": "amplification",
                "variant": None,
                "driver_bucket": "met",
                "actionability": "resistance_marker",
                "mechanism_bucket": "bypass_signaling",
                "sample_source": "liquid_biopsy_ctDNA",
                "parent_clone": "C0",
                "mechanism": "bypass signaling resistance",
            },
            {"date": "2024-03-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        event = trajectory.ledger[0]
        assert event.gene == "MET"
        assert event.driver_bucket == "met"
        assert event.actionability == "resistance_marker"
        assert event.mechanism_bucket == "bypass_signaling"

    def test_new_driver_and_treatment_buckets_validate(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "systemic_treatment_start",
                "regimen": {
                    "label": "cetuximab irinotecan",
                    "components": ["cetuximab", "irinotecan"],
                    "component_classes": [
                        "egfr_monoclonal_antibody",
                        "cytotoxic_chemotherapy",
                    ],
                },
            },
            {
                "date": "2024-03-01",
                "type": "clonal_evolution_event",
                "new_clone_id": "C1",
                "gene": "SDH",
                "alteration": "SDH-deficient signature",
                "alteration_type": "expression_low",
                "variant": None,
                "driver_bucket": "sdh_deficient",
                "actionability": "guideline_relevant",
                "mechanism_bucket": "phenotypic_plasticity",
                "sample_source": "other_specified",
                "sample_source_description": "external methylation panel",
                "parent_clone": "C0",
                "mechanism": "SDH-deficient biology",
            },
            {"date": "2024-04-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        assert trajectory.ledger[0].regimen is not None
        assert trajectory.ledger[0].regimen.component_classes == [
            "egfr_monoclonal_antibody", "cytotoxic_chemotherapy"
        ]
        assert trajectory.ledger[1].driver_bucket == "sdh_deficient"
        assert trajectory.ledger[1].sample_source_description == "external methylation panel"

    def test_description_fields_survive_materialization(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "toxicity",
                "ae": "unusual rash",
                "ae_category": "other_specified",
                "ae_category_description": "cetuximab-like acneiform rash",
                "toxicity_transition": "onset",
                "ctcae_v5_grade": 2,
            },
            {
                "date": "2024-03-01",
                "type": "toxicity",
                "ae": "unusual rash",
                "toxicity_transition": "grade_change",
                "ctcae_v5_grade": 3,
            },
            {
                "date": "2024-03-15",
                "type": "local_intervention",
                "procedure": "nonstandard local control",
                "target_sites": ["liver"],
                "intent": "local_control",
                "modality": "other_specified",
                "modality_description": "electrochemotherapy",
            },
            {
                "date": "2024-04-01",
                "type": "death",
                "cause_category": "cancer_progression",
                "cause": "progressive cancer",
                "eol_care_pattern": "other_specified",
                "eol_care_description": "home hospital pathway",
            },
        ]

        trajectory = replay(baseline, ledger)

        assert trajectory.ledger[0].ae_category_description == (
            "cetuximab-like acneiform rash"
        )
        assert trajectory.ledger[1].ae_category == "other_specified"
        assert trajectory.ledger[1].ae_category_description == (
            "cetuximab-like acneiform rash"
        )
        assert trajectory.ledger[2].modality_description == "electrochemotherapy"
        assert trajectory.ledger[3].eol_care_description == "home hospital pathway"
        assert trajectory.snapshots[-1].outcome is not None
        assert trajectory.snapshots[-1].outcome.eol_care_description == (
            "home hospital pathway"
        )

    def test_long_clinical_rationale_is_preserved(self) -> None:
        baseline, _ = _load_fixture()
        rationale = (
            "The patient developed progressive fatigue, anorexia, and declining "
            "functional reserve after multiple systemic treatments, with worsening "
            "symptom burden despite supportive care; the event is written as a rich "
            "clinical explanation rather than a terse code because this free text is "
            "part of the measurement surface."
        )
        assert len(rationale) > 200
        ledger = [
            {
                "date": "2024-02-15",
                "type": "patient_state_update",
                "patient": {
                    "ecog_ps": 2,
                    "symptom_burden": "moderate",
                    "dominant_symptoms": ["fatigue", "anorexia"],
                },
                "state_change_cause": "progressive systemic symptoms",
                "rationale": rationale,
            },
            {"date": "2024-03-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        assert trajectory.ledger[0].rationale == rationale


class TestCloneIntegrity:
    def test_unknown_clone_reference_hard_fails(self) -> None:
        baseline, ledger = _load_fixture()
        ledger[2]["disease"][0]["clone"] = "C99"

        with pytest.raises(ReplayError, match="CLONE_REFERENCE"):
            replay(baseline, ledger)


class TestTreatmentLifecycle:
    def test_treatment_stop_before_start_hard_fails(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {"date": "2024-02-15", "type": "systemic_treatment_stop"},
            {"date": "2024-03-01", "type": "last_followup"},
        ]

        with pytest.raises(ReplayError, match="TREATMENT_LIFECYCLE"):
            replay(baseline, ledger)

    def test_treatment_switch_without_active_with_regimen_auto_promotes_to_start(self) -> None:
        """A systemic_treatment_switch emitted while no treatment is active
        must auto-promote to systemic_treatment_start IFF a valid regimen is
        present. The input ledger is preserved unchanged; the materialized
        ledger and treatment_history reflect the corrected start semantics.
        Soft warning SOFT_LIFECYCLE_AUTO_PROMOTED_SWITCH_TO_START records the
        event_index, date, and new regimen label."""
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "systemic_treatment_switch",
                "regimen": {"label": "T-DM1", "components": ["trastuzumab_emtansine"]},
            },
            {"date": "2024-03-01", "type": "last_followup"},
        ]

        materialized = replay(baseline, ledger)

        # Input ledger preserved unchanged.
        assert ledger[0]["type"] == "systemic_treatment_switch"

        # Materialized ledger reflects the inverse auto-promotion.
        types = [ev.type for ev in materialized.ledger]
        assert types[0] == "systemic_treatment_start"

        promotions = [
            v for v in materialized.validation_log
            if v.rule_id == "SOFT_LIFECYCLE_AUTO_PROMOTED_SWITCH_TO_START"
        ]
        assert len(promotions) == 1
        msg = promotions[0].message
        assert "2024-02-15" in msg
        assert "systemic_treatment_switch" in msg
        assert "systemic_treatment_start" in msg
        assert "'T-DM1'" in msg
        assert promotions[0].event_id == "e001"

        # Final snapshot has T-DM1 active as line 1.
        final_active = [
            (c.regimen.label, c.line)
            for c in materialized.snapshots[-1].active_treatments
        ]
        assert final_active == [("T-DM1", 1)]

    def test_treatment_switch_without_active_without_regimen_still_hard_fails(self) -> None:
        """The inverse auto-promote is scoped: it must NOT fire when the
        switch event is missing its required `regimen`. That case falls
        through to the EVENT_CONTRACT hard error so the contract violation
        is visible, not silently auto-fixed."""
        baseline, _ = _load_fixture()
        ledger = [
            {"date": "2024-02-15", "type": "systemic_treatment_switch"},  # no regimen
            {"date": "2024-03-01", "type": "last_followup"},
        ]

        with pytest.raises(ReplayError, match="EVENT_CONTRACT"):
            replay(baseline, ledger)

    def test_treatment_start_while_active_auto_promotes_to_switch(self) -> None:
        """A second systemic_treatment_start while a treatment is active must
        not hard-fail — it is auto-promoted to systemic_treatment_switch with
        a SOFT_LIFECYCLE_AUTO_PROMOTED_START_TO_SWITCH warning. The input
        ledger is preserved unchanged; only the materialized ledger and
        treatment_history reflect the corrected switch semantics."""
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "systemic_treatment_start",
                "regimen": {"label": "THP", "components": ["trastuzumab", "pertuzumab", "docetaxel"]},
            },
            {
                "date": "2024-03-01",
                "type": "systemic_treatment_start",
                "regimen": {"label": "T-DM1", "components": ["trastuzumab_emtansine"]},
            },
            {"date": "2024-04-01", "type": "last_followup"},
        ]

        materialized = replay(baseline, ledger)

        # Input ledger preserved unchanged (the raw dicts are not Pydantic
        # models, so the assertion below proves the caller's data is intact).
        assert ledger[1]["type"] == "systemic_treatment_start"

        # Materialized ledger reflects the auto-promotion: second event is
        # now a switch.
        types = [ev.type for ev in materialized.ledger]
        assert types[0] == "systemic_treatment_start"
        assert types[1] == "systemic_treatment_switch"

        # validation_log carries the soft warning with the expected context:
        # event_id, date, active regimen label, new regimen label.
        promotions = [
            v for v in materialized.validation_log
            if v.rule_id == "SOFT_LIFECYCLE_AUTO_PROMOTED_START_TO_SWITCH"
        ]
        assert len(promotions) == 1
        msg = promotions[0].message
        assert "2024-03-01" in msg
        assert "systemic_treatment_start" in msg
        assert "systemic_treatment_switch" in msg
        assert "'THP'" in msg
        assert "'T-DM1'" in msg
        assert promotions[0].event_id == "e002"

        # treatment_history captures CLOSED courses only — THP closed by the
        # auto-promoted switch. T-DM1 stays open through last_followup, so it
        # lives in the final snapshot's active_treatments, not in history.
        closed = [(c.regimen.label, c.line) for c in materialized.treatment_history]
        assert closed == [("THP", 1)]
        final_active = [
            (c.regimen.label, c.line)
            for c in materialized.snapshots[-1].active_treatments
        ]
        assert final_active == [("T-DM1", 2)]

    def test_treatment_start_after_explicit_stop_is_not_auto_promoted(self) -> None:
        """When a prior treatment is explicitly stopped, a subsequent
        systemic_treatment_start is the correct event type (treatment-holiday
        restart) — no auto-promotion, no soft warning."""
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "systemic_treatment_start",
                "regimen": {"label": "THP", "components": ["trastuzumab", "pertuzumab", "docetaxel"]},
            },
            {
                "date": "2024-04-01",
                "type": "systemic_treatment_stop",
                "reason": "toxicity",
            },
            {
                "date": "2024-06-01",
                "type": "systemic_treatment_start",
                "regimen": {"label": "THP", "components": ["trastuzumab", "pertuzumab", "docetaxel"]},
            },
            {"date": "2024-07-01", "type": "last_followup"},
        ]

        materialized = replay(baseline, ledger)
        types = [ev.type for ev in materialized.ledger]
        assert types == [
            "systemic_treatment_start",
            "systemic_treatment_stop",
            "systemic_treatment_start",
            "last_followup",
        ]
        promotions = [
            v for v in materialized.validation_log
            if v.rule_id == "SOFT_LIFECYCLE_AUTO_PROMOTED_START_TO_SWITCH"
        ]
        assert promotions == []

    def test_treatment_stop_without_restart_is_materialized(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "systemic_treatment_start",
                "regimen": {"label": "THP", "components": ["trastuzumab", "pertuzumab", "docetaxel"]},
            },
            {"date": "2024-08-01", "type": "systemic_treatment_stop"},
            {"date": "2024-10-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        assert trajectory.ledger[1].stopped_treatment_id == "tx001"
        assert trajectory.snapshots[2].active_treatments == []

    def test_treatment_stop_then_later_restart_is_materialized(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-02-15",
                "type": "systemic_treatment_start",
                "regimen": {"label": "THP", "components": ["trastuzumab", "pertuzumab", "docetaxel"]},
            },
            {"date": "2024-08-01", "type": "systemic_treatment_stop"},
            {
                "date": "2024-10-01",
                "type": "systemic_treatment_start",
                "regimen": {"label": "T-DM1", "components": ["trastuzumab_emtansine"]},
            },
            {"date": "2025-01-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        assert trajectory.ledger[0].treatment_id == "tx001"
        assert trajectory.ledger[1].stopped_treatment_id == "tx001"
        assert trajectory.ledger[2].treatment_id == "tx002"
        assert trajectory.snapshots[-1].active_treatments[0].line == 2


class TestChronology:
    def test_non_monotonic_ledger_hard_fails(self) -> None:
        baseline, ledger = _load_fixture()
        ledger[1], ledger[2] = ledger[2], ledger[1]

        with pytest.raises(ReplayError, match="DATE_MONOTONICITY"):
            replay(baseline, ledger)


class TestTerminality:
    def test_death_without_cause_hard_fails(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [{"date": "2026-01-30", "type": "death"}]

        with pytest.raises(ReplayError, match="death requires cause"):
            replay(baseline, ledger)

    def test_event_after_death_hard_fails(self) -> None:
        baseline, ledger = _load_fixture()
        ledger.append({"date": "2026-02-15", "type": "last_followup"})

        with pytest.raises(ReplayError, match="TERMINALITY"):
            replay(baseline, ledger)

    def test_two_death_events_hard_fail(self) -> None:
        baseline, ledger = _load_fixture()
        ledger[-1] = {
            "date": "2026-01-30",
            "type": "death",
            "cause_category": "cancer_progression",
            "cause": "progressive cancer",
            "eol_care_pattern": "hospice",
        }
        ledger.append({
            "date": "2026-02-15",
            "type": "death",
            "cause_category": "cancer_progression",
            "cause": "progressive cancer",
            "eol_care_pattern": "hospice",
        })

        with pytest.raises(ReplayError, match="TERMINALITY"):
            replay(baseline, ledger)


class TestEventContracts:
    @pytest.mark.parametrize("event_type", ["response", "progression"])
    def test_response_and_progression_require_disease_delta(self, event_type: str) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {"date": "2024-05-01", "type": event_type},
            {"date": "2024-06-01", "type": "last_followup"},
        ]

        with pytest.raises(ReplayError, match="requires a non-empty disease delta"):
            replay(baseline, ledger)

    def test_recognized_date_before_event_date_hard_fails(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-05-01",
                "type": "response",
                "disease": [{"clone": "C0", "site": "liver", "burden": "small", "kinetic": "shrinking"}],
                "recognized_date": "2024-04-30",
            },
            {"date": "2024-06-01", "type": "last_followup"},
        ]

        with pytest.raises(ReplayError, match="RECOGNITION_TIMING"):
            replay(baseline, ledger)


class TestPatientStateRange:
    def test_invalid_ecog_hard_fails(self) -> None:
        baseline, ledger = _load_fixture()
        ledger[2]["patient"]["ecog_ps"] = 9

        with pytest.raises(ValueError, match="ecog_ps"):
            replay(baseline, ledger)

    def test_organ_function_delta_merges_with_carry_forward(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-05-01",
                "type": "progression",
                "disease": [{"clone": "C0", "site": "liver", "burden": "bulky", "kinetic": "growing"}],
                "patient": {"organ_function": {"hepatic": "mild_impairment"}},
            },
            {"date": "2024-06-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        organ_function = trajectory.snapshots[1].patient_state.organ_function
        assert organ_function.hepatic == "mild_impairment"
        assert organ_function.renal == "normal"
        assert organ_function.cardiac == "normal"
        assert organ_function.marrow_reserve == "normal"


class TestWeightState:
    def test_absolute_weights_carry_forward_from_baseline(self) -> None:
        baseline, ledger = _load_fixture()

        trajectory = replay(baseline, ledger)

        weights = [snapshot.patient_state.weight_kg for snapshot in trajectory.snapshots]
        assert weights == [68, 68, 68, 64, 64, 58, 58, 55, 55]


class TestLastFollowupAlt:
    def test_last_followup_terminal_snapshot_stays_alive(self) -> None:
        baseline, ledger = _load_fixture()
        ledger = deepcopy(ledger)
        ledger[-1] = {"date": "2026-01-30", "type": "last_followup"}

        trajectory = replay(baseline, ledger)

        final = trajectory.snapshots[-1]
        assert final.event_type == "last_followup"
        assert final.vital_status == "alive"
        assert final.outcome is not None
        assert final.outcome.type == "last_followup"
        assert final.outcome.cause is None
        assert final.active_treatments[0].treatment_id == "tx003"


class TestSoftValidationLog:
    def test_soft_warnings_are_recorded_without_halting_replay(self) -> None:
        baseline, _ = _load_fixture()
        ledger = [
            {
                "date": "2024-04-01",
                "type": "progression",
                "disease": [{"clone": "C0", "site": "liver", "burden": "bulky", "kinetic": "growing"}],
                "patient": {"ecog_ps": 4},
            },
            {
                "date": "2024-05-01",
                "type": "response",
                "disease": [{"clone": "C0", "site": "liver", "burden": "absent", "kinetic": "appearing"}],
                "patient": {"ecog_ps": 0},
            },
            {"date": "2024-06-01", "type": "last_followup"},
        ]

        trajectory = replay(baseline, ledger)

        rule_ids = {violation.rule_id for violation in trajectory.validation_log}
        assert "SOFT_IMPLAUSIBLE_ECOG_JUMP" in rule_ids
        assert "SOFT_KINETIC_BURDEN_MISMATCH" in rule_ids
