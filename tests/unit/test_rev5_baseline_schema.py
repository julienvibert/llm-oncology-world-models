"""Tests for the paper-v1 BaselineObject shape (post-migration)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthonccases.oncollm.replay import BaselineObject, LedgerEvent, Regimen, Staging


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
BASELINE_PATHS = (
    FIXTURE_ROOT / "replay_spike" / "her2_brca_baseline_001.json",
    FIXTURE_ROOT / "rev5_poc" / "her2_brca_baseline_002.json",
    FIXTURE_ROOT / "rev5_poc" / "her2_brca_baseline_003.json",
)
LUAD_BASELINE_PATH = FIXTURE_ROOT / "rev5_poc" / "egfr_luad_baseline_001.json"


def _load(path: Path = BASELINE_PATHS[0]) -> dict:
    return json.loads(path.read_text())


def test_brca_baseline_fixtures_parse_with_paper_v1_shape() -> None:
    for path in BASELINE_PATHS:
        baseline = BaselineObject.model_validate(_load(path))

        mp = baseline.identity_card.molecular_profile
        assert mp.alterations[0].gene == "ERBB2"
        assert mp.alterations[0].alteration_type == "amplification"
        assert mp.alterations[0].variant is None
        assert mp.alterations[0].driver_bucket == "erbb2_her2"
        assert mp.alterations[0].actionability == "standard_of_care"
        assert mp.alterations[0].is_primary_driver is True
        assert any(b.name == "HER2 IHC" for b in mp.biomarkers)
        assert not hasattr(mp, "receptor_status")
        assert not hasattr(mp, "testing_performed")

        staging = baseline.identity_card.staging
        diagnosis = baseline.identity_card.diagnosis
        assert diagnosis.primary_site_class == "breast"
        assert staging.stage_class == "metastatic"
        assert staging.formal_stage is not None
        assert staging.formal_stage.system == "ajcc_tnm_8"
        assert staging.formal_stage.group_label == "IV"
        assert staging.formal_stage.tnm is not None
        assert staging.formal_stage.tnm.M == "M1"

        assert not hasattr(baseline.baseline_patient_state, "pro")

        mh = baseline.medical_history
        assert not hasattr(mh, "de_novo_vs_recurrent")
        assert not hasattr(mh, "prior_treatments")


def test_egfr_luad_baseline_fixture_parses_with_paper_v1_shape() -> None:
    baseline = BaselineObject.model_validate(_load(LUAD_BASELINE_PATH))

    ic = baseline.identity_card
    assert ic.diagnosis.oncotree_code == "LUAD"
    assert ic.diagnosis.histology_class == "lung_adenocarcinoma"
    assert ic.diagnosis.primary_site_class == "lung"
    assert ic.staging.stage_class == "metastatic"
    assert ic.staging.formal_stage is not None
    assert ic.staging.formal_stage.group_label == "IVB"
    assert ic.staging.formal_stage.tnm is not None
    assert ic.staging.formal_stage.tnm.M == "M1c"

    mp = ic.molecular_profile
    assert any(
        alt.gene == "EGFR" and alt.variant == "L858R"
        for alt in mp.alterations
    )
    assert any(b.name == "PD-L1 TPS" and b.value == "15%" for b in mp.biomarkers)
    assert ic.baseline_health.pulmonary == "normal"

    assert baseline.medical_history.smoking_status == "never"
    assert baseline.medical_history.tobacco_exposure_band == "none"
    assert baseline.medical_history.pack_years == 0.0
    assert baseline.clone_registry[0].drivers == ["EGFR_L858R", "TP53_R175H"]


def test_comorbidities_require_category() -> None:
    data = _load(BASELINE_PATHS[1])
    comorbidities = data["medical_history"]["comorbidities"]
    assert all("category" in c for c in comorbidities)
    assert comorbidities[0]["category"] == "cardiovascular"
    baseline = BaselineObject.model_validate(data)
    assert baseline.medical_history.comorbidities[0].category == "cardiovascular"


def test_comorbidity_status_unknown_rejected() -> None:
    data = _load()
    data["medical_history"]["comorbidities"] = [
        {
            "category": "cardiovascular",
            "term": "hypertension",
            "status": "unknown",
        }
    ]
    with pytest.raises(ValidationError, match="status"):
        BaselineObject.model_validate(data)


def test_comorbidity_onset_rejected() -> None:
    data = _load()
    data["medical_history"]["comorbidities"][0]["onset"] = "before_diagnosis"
    with pytest.raises(ValidationError, match="onset"):
        BaselineObject.model_validate(data)


def test_smoking_status_unknown_rejected() -> None:
    data = _load()
    data["medical_history"]["smoking_status"] = "unknown"
    with pytest.raises(ValidationError, match="smoking_status"):
        BaselineObject.model_validate(data)


def test_medical_history_required() -> None:
    data = _load()
    data.pop("medical_history")
    with pytest.raises(ValidationError, match="medical_history"):
        BaselineObject.model_validate(data)


def test_prior_treatments_rejected() -> None:
    data = _load()
    data["medical_history"]["prior_treatments"] = []
    with pytest.raises(ValidationError, match="prior_treatments"):
        BaselineObject.model_validate(data)


def test_tobacco_consistency_never_smoker() -> None:
    data = _load()
    mh = data["medical_history"]
    mh["smoking_status"] = "never"
    mh["tobacco_exposure_band"] = "none"
    mh["pack_years"] = 0.0
    BaselineObject.model_validate(data)

    mh["pack_years"] = 10.0
    with pytest.raises(ValidationError, match="pack_years"):
        BaselineObject.model_validate(data)


def test_tobacco_consistency_former_smoker() -> None:
    data = _load(BASELINE_PATHS[1])
    mh = data["medical_history"]
    mh["smoking_status"] = "former"
    mh["tobacco_exposure_band"] = "light"
    mh["pack_years"] = 7.5
    BaselineObject.model_validate(data)

    mh["pack_years"] = 0.0
    with pytest.raises(ValidationError, match="pack_years"):
        BaselineObject.model_validate(data)


def test_tobacco_consistency_current_smoker_requires_non_none_band() -> None:
    data = _load()
    mh = data["medical_history"]
    mh["smoking_status"] = "current"
    mh["pack_years"] = 5.0
    mh["tobacco_exposure_band"] = "none"
    with pytest.raises(ValidationError, match="tobacco_exposure_band"):
        BaselineObject.model_validate(data)


def test_contextual_factors_shape() -> None:
    data = _load()
    data["medical_history"]["contextual_factors"] = [
        {
            "category": "prior_non_index_therapy_exposure",
            "term": "mediastinal radiation for prior Hodgkin lymphoma",
            "status": "historical",
            "trajectory_relevance": "relevant to cardiopulmonary reserve",
        }
    ]
    baseline = BaselineObject.model_validate(data)
    factors = baseline.medical_history.contextual_factors
    assert factors[0].category == "prior_non_index_therapy_exposure"
    assert factors[0].term == "mediastinal radiation for prior Hodgkin lymphoma"


def test_baseline_health_requires_pulmonary() -> None:
    data = _load()
    data["identity_card"]["baseline_health"].pop("pulmonary")
    with pytest.raises(ValidationError, match="pulmonary"):
        BaselineObject.model_validate(data)


@pytest.mark.parametrize(
    "field_name",
    [
        "contraindications",
        "frailty_band",
        "autoimmune_disease_history",
        "neuropathy_at_baseline",
    ],
)
def test_baseline_health_context_fields_required(field_name: str) -> None:
    data = _load()
    data["identity_card"]["baseline_health"].pop(field_name)
    with pytest.raises(ValidationError, match=field_name):
        BaselineObject.model_validate(data)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("frailty_band", "unknown"),
        ("neuropathy_at_baseline", "unknown"),
    ],
)
def test_baseline_health_unknown_values_rejected(
    field_name: str,
    bad_value: str,
) -> None:
    data = _load()
    data["identity_card"]["baseline_health"][field_name] = bad_value
    with pytest.raises(ValidationError, match=field_name):
        BaselineObject.model_validate(data)


def test_self_annotation_buckets_optional_at_pydantic_level() -> None:
    data = _load()
    data["identity_card"]["diagnosis"].pop("histology_bucket")
    data["identity_card"]["diagnosis"].pop("primary_site_class")
    data["identity_card"]["molecular_profile"].pop("primary_driver_bucket")
    for burden in data["baseline_disease_burden"]:
        burden.pop("site_class")

    baseline = BaselineObject.model_validate(data)
    assert baseline.identity_card.diagnosis.histology_bucket is None
    assert baseline.identity_card.diagnosis.primary_site_class is None
    assert baseline.identity_card.molecular_profile.primary_driver_bucket is None
    assert all(site.site_class is None for site in baseline.baseline_disease_burden)


def test_baseline_alteration_requires_homogeneous_molecular_fields() -> None:
    for field_name in ("variant", "driver_bucket", "actionability", "is_primary_driver"):
        data = _load()
        data["identity_card"]["molecular_profile"]["alterations"][0].pop(field_name)
        with pytest.raises(ValidationError, match=field_name):
            BaselineObject.model_validate(data)


def test_biomarkers_required_but_empty_list_valid() -> None:
    data = _load()
    data["identity_card"]["molecular_profile"].pop("biomarkers")
    with pytest.raises(ValidationError, match="biomarkers"):
        BaselineObject.model_validate(data)

    data = _load()
    data["identity_card"]["molecular_profile"]["biomarkers"] = []
    baseline = BaselineObject.model_validate(data)
    assert baseline.identity_card.molecular_profile.biomarkers == []


def test_no_actionable_driver_can_parse_empty_alterations() -> None:
    data = _load()
    mp = data["identity_card"]["molecular_profile"]
    mp["primary_driver_bucket"] = "no_actionable_driver"
    mp["alterations"] = []
    baseline = BaselineObject.model_validate(data)
    assert baseline.identity_card.molecular_profile.alterations == []
    assert (
        baseline.identity_card.molecular_profile.primary_driver_bucket
        == "no_actionable_driver"
    )


def test_expanded_histology_buckets_parse() -> None:
    for bucket in ("liposarcoma", "mesothelioma", "chordoma"):
        data = _load()
        data["identity_card"]["diagnosis"]["histology_bucket"] = bucket
        baseline = BaselineObject.model_validate(data)
        assert baseline.identity_card.diagnosis.histology_bucket == bucket


def test_fusion_driver_bucket_parses() -> None:
    data = _load()
    mp = data["identity_card"]["molecular_profile"]
    mp["primary_driver_bucket"] = "gene_fusion_or_rearrangement"
    mp["alterations"][0]["gene"] = "EWSR1"
    mp["alterations"][0]["alteration_type"] = "fusion"
    mp["alterations"][0]["alteration"] = "EWSR1-FLI1 fusion"
    mp["alterations"][0]["variant"] = "EWSR1-FLI1"
    mp["alterations"][0]["driver_bucket"] = "gene_fusion_or_rearrangement"

    baseline = BaselineObject.model_validate(data)
    assert (
        baseline.identity_card.molecular_profile.primary_driver_bucket
        == "gene_fusion_or_rearrangement"
    )


def test_invalid_histology_bucket_rejected() -> None:
    data = _load()
    data["identity_card"]["diagnosis"]["histology_bucket"] = "not_a_histology"
    with pytest.raises(ValidationError, match="histology_bucket"):
        BaselineObject.model_validate(data)


def test_primary_site_class_parse_and_invalid_rejected() -> None:
    for site_class in (
        "retroperitoneum",
        "thyroid",
        "pleura",
        "testis",
        "thymus",
        "adrenal",
        "eye",
        "peripheral_nervous_system",
        "bone_marrow",
        "lymph_node",
        "brain",
        "primary_site",
    ):
        data = _load()
        data["identity_card"]["diagnosis"]["primary_site_class"] = site_class
        baseline = BaselineObject.model_validate(data)
        assert baseline.identity_card.diagnosis.primary_site_class == site_class

    data["identity_card"]["diagnosis"]["primary_site_class"] = "left_breast"
    with pytest.raises(ValidationError, match="primary_site_class"):
        BaselineObject.model_validate(data)


def test_expanded_site_class_aliases_parse() -> None:
    for site_class in (
        "central_nervous_system",
        "bone_marrow_blood",
        "lymphoid_tissue",
        "retroperitoneum",
        "skin",
        "thyroid",
        "testis",
        "thymus",
        "peripheral_nervous_system",
        "eye",
    ):
        event = LedgerEvent(
            date="2024-02-01",
            type="progression",
            disease=[
                {
                    "clone": "C0",
                    "site": "measurable disease",
                    "site_class": site_class,
                    "burden": "small",
                    "kinetic": "appearing",
                }
            ],
        )
        assert event.disease is not None
        assert event.disease[0].site_class == site_class


def test_expanded_regimen_classes_parse() -> None:
    regimen = Regimen(
        label="trastuzumab deruxtecan",
        components=["trastuzumab deruxtecan"],
        component_classes=["antibody_drug_conjugate"],
    )
    assert regimen.components == ["trastuzumab deruxtecan"]
    assert regimen.component_classes == ["antibody_drug_conjugate"]

    cellular = Regimen(
        label="CAR T-cell therapy",
        components=["axicabtagene ciloleucel"],
        component_classes=["cellular_therapy"],
    )
    assert cellular.component_classes == ["cellular_therapy"]

    her2_tki = Regimen(
        label="HER2 TKI backbone",
        components=["HER2 TKI"],
        component_classes=["her2_tki"],
    )
    assert her2_tki.component_classes == ["her2_tki"]

    kit_tki = Regimen(
        label="imatinib",
        components=["imatinib"],
        component_classes=["kit_pdgfra_tki"],
    )
    assert kit_tki.component_classes == ["kit_pdgfra_tki"]

    vegfr_tki = Regimen(
        label="pembrolizumab axitinib",
        components=["pembrolizumab", "axitinib"],
        component_classes=["immune_checkpoint_inhibitor", "vegfr_multikinase_tki"],
    )
    assert vegfr_tki.component_classes == [
        "immune_checkpoint_inhibitor",
        "vegfr_multikinase_tki",
    ]


def test_invalid_component_class_rejected() -> None:
    with pytest.raises(ValidationError, match="component_classes"):
        Regimen(
            label="trastuzumab deruxtecan",
            components=["trastuzumab deruxtecan"],
            component_classes=["trastuzumab_deruxtecan"],
        )


def test_component_classes_must_align_with_components() -> None:
    with pytest.raises(ValidationError, match="same length"):
        Regimen(
            label="agent_a_agent_b",
            components=["agent_a", "agent_b"],
            component_classes=["other_targeted"],
        )


def test_combo_specific_regimen_classes_rejected() -> None:
    for invalid_class in ("amivantamab_based", "chemo_immunotherapy"):
        with pytest.raises(ValidationError, match="component_classes"):
            Regimen(
                label="agent_a_agent_b",
                components=["agent_a", "agent_b"],
                component_classes=["bispecific_antibody", invalid_class],
            )


def test_regimen_class_is_not_part_of_clean_v1_contract() -> None:
    with pytest.raises(ValidationError, match="regimen_class"):
        Regimen(
            label="THP",
            components=["trastuzumab", "pertuzumab", "docetaxel"],
            regimen_class="anti_her2",
        )

    regimen = Regimen(
        label="THP",
        components=["trastuzumab", "pertuzumab", "docetaxel"],
        component_classes=["anti_her2", "anti_her2", "cytotoxic_chemotherapy"],
    )
    assert regimen.component_classes == [
        "anti_her2",
        "anti_her2",
        "cytotoxic_chemotherapy",
    ]


def test_clonal_evolution_driver_bucket_optional_at_pydantic_level() -> None:
    event = LedgerEvent.model_validate(
        {
            "date": "2024-09-01",
            "type": "clonal_evolution_event",
            "new_clone_id": "C1",
            "gene": "MET",
            "alteration": "MET amplification",
            "alteration_type": "amplification",
            "variant": None,
            "sample_source": "liquid_biopsy_ctDNA",
            "parent_clone": "C0",
            "mechanism": "bypass signaling resistance",
        }
    )
    assert event.driver_bucket is None
    assert event.mechanism_bucket is None
    assert event.gene == "MET"


def test_clonal_evolution_buckets_literal_enforced() -> None:
    event = LedgerEvent.model_validate(
        {
            "date": "2024-09-01",
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
        }
    )
    assert event.driver_bucket == "met"
    assert event.actionability == "resistance_marker"
    assert event.mechanism_bucket == "bypass_signaling"
    assert event.gene == "MET"

    with pytest.raises(ValidationError, match="driver_bucket"):
        LedgerEvent.model_validate(
            {
                "date": "2024-09-01",
                "type": "clonal_evolution_event",
                "new_clone_id": "C1",
                "gene": "MET",
                "alteration": "MET amplification",
                "alteration_type": "amplification",
                "variant": None,
                "driver_bucket": "MET_amplification",
                "sample_source": "liquid_biopsy_ctDNA",
            }
        )

    with pytest.raises(ValidationError, match="mechanism_bucket"):
        LedgerEvent.model_validate(
            {
                "date": "2024-09-01",
                "type": "clonal_evolution_event",
                "new_clone_id": "C1",
                "gene": "MET",
                "alteration": "MET amplification",
                "alteration_type": "amplification",
                "variant": None,
                "driver_bucket": "met",
                "actionability": "resistance_marker",
                "mechanism_bucket": "met_amplification",
                "sample_source": "liquid_biopsy_ctDNA",
            }
        )


def test_refined_primary_driver_buckets_parse_and_old_combo_rejected() -> None:
    data = _load()
    mp = data["identity_card"]["molecular_profile"]
    for bucket in (
        "pik3ca_pi3k_akt",
        "esr1",
        "pten",
        "vhl",
        "ar",
        "nras",
        "rb1",
        "tp53",
        "myc",
    ):
        mp["primary_driver_bucket"] = bucket
        baseline = BaselineObject.model_validate(data)
        assert baseline.identity_card.molecular_profile.primary_driver_bucket == bucket

    mp["primary_driver_bucket"] = "pi3k_akt_esr1"
    with pytest.raises(ValidationError, match="primary_driver_bucket"):
        BaselineObject.model_validate(data)


def test_family_history_is_structured() -> None:
    data = _load(BASELINE_PATHS[0])
    baseline = BaselineObject.model_validate(data)
    fh = baseline.medical_history.family_history
    assert fh[0].relation == "mother"
    assert fh[0].cancer_type == "breast_cancer"


def test_family_history_string_entries_rejected() -> None:
    data = _load()
    data["medical_history"]["family_history"] = ["breast_cancer_mother"]
    with pytest.raises(ValidationError):
        BaselineObject.model_validate(data)


def test_staging_stage_class_recurrent_rejected() -> None:
    with pytest.raises(ValidationError, match="stage_class"):
        Staging(stage_class="recurrent")


def test_baseline_has_no_de_novo_vs_recurrent_field() -> None:
    baseline = BaselineObject.model_validate(_load())
    with pytest.raises(AttributeError):
        baseline.medical_history.de_novo_vs_recurrent  # noqa: B018


def test_sex_legacy_F_rejected() -> None:
    data = _load()
    data["identity_card"]["demographics"]["sex"] = "F"
    with pytest.raises(ValidationError, match="sex"):
        BaselineObject.model_validate(data)


def test_rich_baseline_json_dump_round_trips() -> None:
    baseline = BaselineObject.model_validate(_load(BASELINE_PATHS[2]))

    dumped = baseline.model_dump(mode="json")
    reparsed = BaselineObject.model_validate(dumped)

    assert reparsed.model_dump(mode="json") == dumped
