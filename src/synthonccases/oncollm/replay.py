"""Deterministic replay for SynthOncCases latent trajectory ledgers.

The LLM emits a baseline patch plus a chronological event ledger. This module
validates that object and deterministically materializes full patient-state
snapshots without additional LLM calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BurdenBand = Literal["absent", "microscopic", "small", "moderate", "bulky", "diffuse_or_leukemic"]
Kinetic = Literal["appearing", "growing", "stable", "shrinking", "cleared"]
SymptomBurden = Literal["none", "mild", "moderate", "severe"]
TobaccoExposureBand = Literal["none", "light", "moderate", "heavy"]
OrganFunctionValue = Literal[
    "normal",
    "mild_impairment",
    "moderate_impairment",
    "severe_impairment",
    "failure",
]
EventType = Literal[
    "systemic_treatment_start",
    "systemic_treatment_switch",
    "systemic_treatment_stop",
    "systemic_treatment_modification",
    "response",
    "progression",
    "clonal_evolution_event",
    "toxicity",
    "local_intervention",
    "patient_state_update",
    "death",
    # Engine support for future censoring / continuation studies. Paper-v1
    # prompts use death-only.
    "last_followup",
]
HistologyBucket = Literal[
    "adenocarcinoma",
    "squamous_cell_carcinoma",
    "small_cell_carcinoma",
    "large_cell_neuroendocrine_carcinoma",
    "well_differentiated_neuroendocrine_tumor",
    "ductal_carcinoma",
    "lobular_carcinoma",
    "serous_carcinoma",
    "clear_cell_carcinoma",
    "endometrioid_carcinoma",
    "mucinous_carcinoma",
    "signet_ring_cell_carcinoma",
    "urothelial_carcinoma",
    "hepatocellular_carcinoma",
    "cholangiocarcinoma",
    "pancreatic_ductal_adenocarcinoma",
    "renal_cell_carcinoma",
    "melanoma",
    "gastrointestinal_stromal_tumor",
    "leiomyosarcoma",
    "liposarcoma",
    "undifferentiated_pleomorphic_sarcoma",
    "synovial_sarcoma",
    "angiosarcoma",
    "osteosarcoma",
    "chondrosarcoma",
    "ewing_sarcoma",
    "rhabdomyosarcoma",
    "malignant_peripheral_nerve_sheath_tumor",
    "other_sarcoma",
    "chordoma",
    "glioblastoma",
    "astrocytoma",
    "oligodendroglioma",
    "ependymoma",
    "other_glioma",
    "mesothelioma",
    "diffuse_large_b_cell_lymphoma",
    "follicular_lymphoma",
    "hodgkin_lymphoma",
    "plasma_cell_myeloma",
    "acute_myeloid_leukemia",
    "acute_lymphoblastic_leukemia",
    "chronic_lymphocytic_leukemia_sll",
    "myelodysplastic_or_mpn",
    "other_hematologic_malignancy",
    "other",
]
PrimarySiteClass = Literal[
    "breast",
    "lung",
    "colorectal",
    "upper_gastrointestinal",
    "pancreatobiliary",
    "liver",
    "kidney",
    "urinary_tract",
    "prostate",
    "gynecologic_ovary_fallopian_peritoneal",
    "gynecologic_uterus_cervix",
    "thyroid",
    "skin",
    "head_neck",
    "central_nervous_system",
    "brain",
    "peripheral_nervous_system",
    "eye",
    "bone",
    "soft_tissue",
    "soft_tissue_extremity",
    "soft_tissue_trunk_chest_wall",
    "retroperitoneum",
    "visceral_soft_tissue",
    "bone_marrow_blood",
    "bone_marrow",
    "lymphoid_tissue",
    "lymph_node",
    "pleura",
    "mesothelioma",
    "peritoneum",
    "adrenal",
    "testis",
    "thymus",
    "penis",
    "vulva_vagina",
    "primary_site",
    "other",
]
SiteClass = Literal[
    "primary_site",
    "breast",
    "bone_marrow",
    "bone_marrow_blood",
    "bone",
    "central_nervous_system",
    "liver",
    "lung",
    "pleura",
    "leptomeningeal_cns",
    "brain",
    "lymph_node",
    "lymphoid_tissue",
    "colorectal",
    "upper_gastrointestinal",
    "pancreatobiliary",
    "kidney",
    "urinary_tract",
    "prostate",
    "gynecologic_ovary_fallopian_peritoneal",
    "gynecologic_uterus_cervix",
    "thyroid",
    "skin",
    "head_neck",
    "soft_tissue",
    "soft_tissue_extremity",
    "soft_tissue_trunk_chest_wall",
    "retroperitoneum",
    "visceral_soft_tissue",
    "adrenal",
    "peritoneum",
    "testis",
    "thymus",
    "penis",
    "vulva_vagina",
    "peripheral_nervous_system",
    "eye",
    "other",
]
PrimaryDriverBucket = Literal[
    "egfr",
    "alk",
    "ros1",
    "ret",
    "met",
    "braf",
    "kras",
    "ntrk",
    "erbb2_her2",
    "brca_hrd",
    "pik3ca_pi3k_akt",
    "esr1",
    "pten",
    "vhl",
    "ar",
    "nras",
    "rb1",
    "tp53",
    "myc",
    "fgfr",
    "idh",
    "kit_pdgfra",
    "gene_fusion_or_rearrangement",
    "sdh_deficient",
    "mdm2_cdk4_amplification",
    "mmr_msi",
    "other_actionable",
    "other_driver",
    "no_actionable_driver",
]
ClonalMechanismBucket = Literal[
    "on_target_secondary_alteration",
    "target_amplification",
    "target_loss_or_downregulation",
    "bypass_signaling",
    "downstream_pathway_reactivation",
    "dna_repair_restoration",
    "lineage_or_histologic_transformation",
    "phenotypic_plasticity",
    "immune_escape",
    "pharmacologic_sanctuary_or_cns",
    "polyclonal_or_heterogeneous_resistance",
    "other",
]
RegimenClass = Literal[
    "cytotoxic_chemotherapy",
    "immune_checkpoint_inhibitor",
    "other_immunotherapy",
    "egfr_tki",
    "egfr_monoclonal_antibody",
    "alk_tki",
    "ros1_ret_met_tki",
    "kras_g12c_inhibitor",
    "braf_mek_targeted",
    "anti_her2",
    "her2_tki",
    "kit_pdgfra_tki",
    "vegfr_multikinase_tki",
    "antibody_drug_conjugate",
    "parp_inhibitor",
    "endocrine",
    "cdk4_6_inhibitor",
    "androgen_axis_therapy",
    "anti_vegf",
    "monoclonal_antibody_non_checkpoint",
    "bispecific_antibody",
    "cellular_therapy",
    "radioligand_therapy",
    "anti_cd20_antibody",
    "btk_inhibitor",
    "bcl2_inhibitor",
    "proteasome_inhibitor",
    "immunomodulatory_imid",
    "anti_cd38_antibody",
    "hypomethylating_agent",
    "other_targeted",
    "supportive_or_eol_only",
    "other",
]
ContextualFactorCategory = Literal[
    "substance_use",
    "medication_exposure",
    "prior_non_index_therapy_exposure",
    "occupation",
    "environmental_exposure",
    "social_support",
    "care_access",
    "reproductive_hormonal",
    "other",
]
LocalInterventionModality = Literal[
    "surgery",
    "radiation",
    "ablation",
    "embolization",
    "brachytherapy",
    "endoscopic_or_percutaneous_palliation",
    "other_specified",
]
TreatmentReason = Literal[
    "progression",
    "toxicity",
    "planned_completion",
    "patient_preference",
    "clinical_judgment",
    "death_or_transition_to_eol",
]
SystemicTreatmentModificationType = Literal[
    "dose_reduction",
    "dose_hold",
    "dose_resume",
    "discontinuation_of_component",
]
ToxicityTransition = Literal["onset", "grade_change", "resolved"]
AECategory = Literal[
    "hematologic",
    "gastrointestinal",
    "dermatologic",
    "neurologic",
    "cardiac",
    "pulmonary",
    "hepatic",
    "renal",
    "metabolic_endocrine",
    "immune_related",
    "infectious",
    "general_systemic",
    "other_specified",
]
SampleSource = Literal[
    "tissue_biopsy",
    "liquid_biopsy_ctDNA",
    "bone_marrow_aspirate",
    "bone_marrow_biopsy",
    "cytology",
    "other_specified",
]
AlterationType = Literal[
    "mutation",
    "amplification",
    "deletion",
    "fusion",
    "rearrangement",
    "translocation",
    "copy_number_gain",
    "copy_number_loss",
    "expression_high",
    "expression_low",
    "methylation",
    "wild_type",
    "histologic_transformation",
    "other_specified",
]
SystemicTreatmentIntent = Literal[
    "curative",
    "adjuvant",
    "neoadjuvant",
    "disease_control",
    "symptom_palliation",
]
LocalInterventionIntent = Literal[
    "curative",
    "local_control",
    "symptom_palliation",
    "diagnostic",
    "adjuvant",
    "neoadjuvant",
]
EventIntent = Literal[
    "curative",
    "adjuvant",
    "neoadjuvant",
    "disease_control",
    "symptom_palliation",
    "local_control",
    "diagnostic",
]
SYSTEMIC_TREATMENT_INTENTS = frozenset(
    {"curative", "adjuvant", "neoadjuvant", "disease_control", "symptom_palliation"}
)
LOCAL_INTERVENTION_INTENTS = frozenset(
    {
        "curative",
        "local_control",
        "symptom_palliation",
        "diagnostic",
        "adjuvant",
        "neoadjuvant",
    }
)
DeathCauseCategory = Literal[
    "cancer_progression",
    "treatment_related",
    "non_cancer_medical",
    "age_related_frailty",
    "external_cause",
]
EOLCarePattern = Literal[
    "hospice",
    "home_palliative",
    "inpatient_palliative",
    "icu_or_life_sustaining_care_near_death",
    "no_specialist_eol_care",
    "other_specified",
]
VitalStatus = Literal["alive", "deceased"]

_BURDEN_RANK: dict[BurdenBand, int] = {
    "absent": 0,
    "microscopic": 1,
    "small": 2,
    "moderate": 3,
    "bulky": 4,
    "diffuse_or_leukemic": 5,
}


class ReplayError(ValueError):
    """Hard replay violation that prevents deterministic materialization."""

    def __init__(self, rule_id: str, message: str) -> None:
        super().__init__(f"{rule_id}: {message}")
        self.rule_id = rule_id
        self.message = message


class _ReplayModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Demographics(_ReplayModel):
    age_at_diagnosis: int = Field(ge=0)
    sex: Literal["female", "male", "intersex_or_other"]
    ecog_ps_at_diagnosis: int = Field(ge=0, le=4)


class Diagnosis(_ReplayModel):
    diagnosis_date: date
    oncotree_code: str
    diagnosis_label: str | None = None
    histology_class: str
    histology_bucket: HistologyBucket | None = None
    primary_site: str
    primary_site_class: PrimarySiteClass | None = None
    histology_description: str | None = None


StageClass = Literal[
    "in_situ",
    "localized",
    "locoregional",
    "metastatic",
    "systemic",
]
FormalStageSystem = Literal[
    "ajcc_tnm_8",
    "figo",
    "ann_arbor_lugano",
    "eln_2022",
    "r_iss",
    "ipss_r",
    "ipss_m",
    "bclc",
    "who_cns",
    "rai_binet",
    "other_specified",
]


class TNM(_ReplayModel):
    T: str
    N: str
    M: str


class FormalStage(_ReplayModel):
    system: FormalStageSystem
    group_label: str | None = None
    tnm: TNM | None = None
    components: dict[str, Any] = Field(default_factory=dict)
    system_description: str | None = None


class Staging(_ReplayModel):
    stage_class: StageClass
    formal_stage: FormalStage | None = None


ComorbidityCategory = Literal[
    "cardiovascular",
    "pulmonary",
    "renal",
    "hepatic",
    "metabolic_endocrine",
    "autoimmune_inflammatory",
    "neurologic",
    "psychiatric",
    "infectious",
    "prior_malignancy",
    "other",
]


class Comorbidity(_ReplayModel):
    category: ComorbidityCategory
    term: str
    status: Literal["controlled", "uncontrolled", "resolved"]


class ContextualFactor(_ReplayModel):
    category: ContextualFactorCategory
    term: str
    status: Literal["active", "historical", "resolved"]
    trajectory_relevance: str


Actionability = Literal[
    "standard_of_care",
    "guideline_relevant",
    "investigational",
    "prognostic_only",
    "resistance_marker",
    "no_known_actionability",
]


class Alteration(_ReplayModel):
    gene: str
    alteration_type: AlterationType
    alteration: str
    variant: str | None
    driver_bucket: PrimaryDriverBucket
    actionability: Actionability
    is_primary_driver: bool


class Biomarker(_ReplayModel):
    name: str
    value: str | int | float


class MolecularProfile(_ReplayModel):
    alterations: list[Alteration]
    biomarkers: list[Biomarker]
    primary_driver_bucket: PrimaryDriverBucket | None = None


class OrganFunction(_ReplayModel):
    renal: OrganFunctionValue
    hepatic: OrganFunctionValue
    cardiac: OrganFunctionValue
    marrow_reserve: OrganFunctionValue
    pulmonary: OrganFunctionValue


class OrganFunctionDelta(_ReplayModel):
    renal: OrganFunctionValue | None = None
    hepatic: OrganFunctionValue | None = None
    cardiac: OrganFunctionValue | None = None
    marrow_reserve: OrganFunctionValue | None = None
    pulmonary: OrganFunctionValue | None = None


class BaselineHealth(_ReplayModel):
    """Baseline organ function and treatment-relevant health constraints."""

    renal: OrganFunctionValue
    hepatic: OrganFunctionValue
    cardiac: OrganFunctionValue
    marrow_reserve: OrganFunctionValue
    pulmonary: OrganFunctionValue
    contraindications: list[str]
    frailty_band: Literal["fit", "pre_frail", "frail"]
    autoimmune_disease_history: bool
    neuropathy_at_baseline: Literal[
        "none", "grade_1", "grade_2", "grade_3_plus"
    ]


FamilyHistoryRelation = Literal[
    "mother",
    "father",
    "sibling",
    "child",
    "maternal_grandparent",
    "paternal_grandparent",
    "maternal_aunt_uncle",
    "paternal_aunt_uncle",
    "other_relative",
]


class FamilyHistoryEntry(_ReplayModel):
    relation: FamilyHistoryRelation
    cancer_type: str


class MedicalHistory(_ReplayModel):
    """Patient history that is fixed at baseline (medical context, prior
    cancers, smoking) but conceptually distinct from the in-trajectory
    clinical state captured by identity_card and baseline_patient_state."""
    comorbidities: list[Comorbidity]
    smoking_status: Literal["never", "former", "current"]
    tobacco_exposure_band: TobaccoExposureBand
    pack_years: float = Field(ge=0.0)
    family_history: list[FamilyHistoryEntry]
    contextual_factors: list[ContextualFactor]

    @field_validator("pack_years")
    @classmethod
    def _finite_pack_years(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pack_years must be finite")
        return value

    @model_validator(mode="after")
    def _check_tobacco_consistency(self) -> MedicalHistory:
        if self.smoking_status == "never":
            if self.pack_years != 0.0:
                raise ValueError("never-smoker requires pack_years == 0.0")
            if self.tobacco_exposure_band != "none":
                raise ValueError("never-smoker requires tobacco_exposure_band='none'")
            return self

        if self.pack_years <= 0.0:
            raise ValueError("former/current smoker requires pack_years > 0.0")
        if self.tobacco_exposure_band == "none":
            raise ValueError(
                "former/current smoker requires tobacco_exposure_band other than 'none'"
            )
        return self


class IdentityCard(_ReplayModel):
    demographics: Demographics
    diagnosis: Diagnosis
    staging: Staging
    molecular_profile: MolecularProfile
    baseline_health: BaselineHealth


class CloneRegistryEntry(_ReplayModel):
    clone_id: str
    clone_role: Literal["founder", "subclone", "resistance_subclone", "transformed_subclone"]
    drivers: list[str]
    first_presence_date: date
    parent_clone_id_or_null: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_first_evidence_date(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "first_presence_date" not in data and "first_evidence_date" in data:
            data["first_presence_date"] = data.pop("first_evidence_date")
        return data


class DiseaseBurdenChange(_ReplayModel):
    clone: str
    site: str
    site_class: SiteClass | None = None
    burden: BurdenBand
    kinetic: Kinetic

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "site" not in data and "organ" in data:
            data["site"] = data.pop("organ")
        if "kinetic" not in data and "trend" in data:
            kinetic = data.pop("trend")
            # Legacy `clearing` was dropped because it overlaps with shrinking.
            data["kinetic"] = "shrinking" if kinetic == "clearing" else kinetic
        return data

    @property
    def organ(self) -> str:
        """Archived-artifact alias for the current ``site`` field."""
        return self.site

    @property
    def trend(self) -> Kinetic:
        """Archived-artifact alias for the current ``kinetic`` field."""
        return self.kinetic


class BaselinePatientState(_ReplayModel):
    weight_kg: float
    symptom_burden: SymptomBurden

    @field_validator("weight_kg")
    @classmethod
    def _finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight_kg must be finite")
        return value


class BaselineObject(_ReplayModel):
    baseline_id: str
    identity_card: IdentityCard
    clone_registry: list[CloneRegistryEntry]
    baseline_disease_burden: list[DiseaseBurdenChange]
    baseline_patient_state: BaselinePatientState
    medical_history: MedicalHistory


class PatientDelta(_ReplayModel):
    ecog_ps: int | None = Field(default=None, ge=0, le=4)
    weight_kg: float | None = None
    symptom_burden: SymptomBurden | None = None
    dominant_symptoms: list[str] | None = None
    organ_function: OrganFunctionDelta | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_patient_delta(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "ecog_ps" not in data and "ecog" in data:
            data["ecog_ps"] = data.pop("ecog")
        # `weight_change_kg`, `treatment_tolerance`, and `pro` were removed
        # from the paper-v1 emitted schema. They cannot be faithfully converted
        # here without replay context, so leave them to extra=forbid.
        return data

    @field_validator("weight_kg")
    @classmethod
    def _finite_weight(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("weight_kg must be finite")
        return value


class Regimen(_ReplayModel):
    label: str
    components: list[str]
    component_classes: list[RegimenClass] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return {"label": text, "components": [text] if text else []}
        return value

    @field_validator("components")
    @classmethod
    def _components_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("regimen.components must be non-empty")
        return value

    @model_validator(mode="after")
    def _component_classes_align_with_components(self) -> "Regimen":
        if self.component_classes is None:
            return self
        if len(self.component_classes) != len(self.components):
            raise ValueError(
                "regimen.component_classes must have the same length as regimen.components"
            )
        return self


def canonicalize_legacy_event_dict(event: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite archived event variants into the current ledger shape.

    Returns a new dict (does not mutate ``event``). Mirrors the
    ``LedgerEvent._coerce_legacy_event_shape`` validator so validation and
    CSV-emission paths stay coherent. Only covers rewrites ``LedgerEvent``
    already accepts; unmappable historical event types are left untouched and
    handled locally by the caller.
    """
    data = dict(event)
    type_map = {
        "treatment_start": "systemic_treatment_start",
        "treatment_switch": "systemic_treatment_switch",
        "treatment_stop": "systemic_treatment_stop",
        "molecular_lead_event": "clonal_evolution_event",
        "pro_update": "patient_state_update",
        "surgery": "local_intervention",
    }
    original_type = event.get("type")
    if data.get("type") in type_map:
        data["type"] = type_map[data["type"]]
    if data.get("type") == "local_intervention" and original_type == "surgery":
        data.setdefault("modality", "surgery")
    data.pop("line", None)
    if data.get("type") == "toxicity":
        if "ctcae_v5_grade" not in data and "ctcae_grade" in data:
            data["ctcae_v5_grade"] = data.pop("ctcae_grade")
        data.setdefault("toxicity_transition", "onset")
        data.pop("resolution_date", None)
        data.pop("current_status", None)
        data.pop("attributed_to", None)
    reason_map = {
        "patient_choice": "patient_preference",
        "completion": "planned_completion",
    }
    if data.get("reason") in reason_map:
        data["reason"] = reason_map[data["reason"]]
    if data.get("sample_source") == "other":
        data["sample_source"] = "other_specified"
    if data.get("modality") in {"cryotherapy"}:
        data["modality"] = "ablation"
    if data.get("modality") == "other":
        data["modality"] = "other_specified"
    intent_map = {
        "palliative": "symptom_palliation",
        "cytoreductive": "local_control",
    }
    if data.get("intent") in intent_map:
        data["intent"] = intent_map[data["intent"]]
    return data


class LedgerEvent(_ReplayModel):
    date: date
    type: EventType
    recognized_date: date | None = None
    disease: list[DiseaseBurdenChange] | None = None
    patient: PatientDelta | None = None
    regimen: Regimen | None = None
    cause: str | None = None
    cause_category: DeathCauseCategory | None = None
    eol_care_pattern: EOLCarePattern | None = None
    eol_care_description: str | None = None
    reason: TreatmentReason | None = None
    modification_type: SystemicTreatmentModificationType | None = None
    affected_components: list[str] | None = None
    ae: str | None = None
    ae_category: AECategory | None = None
    ae_category_description: str | None = None
    toxicity_transition: ToxicityTransition | None = None
    ctcae_v5_grade: int | None = Field(default=None, ge=1, le=5)
    causes: list[str] | None = None
    new_clone_id: str | None = None
    gene: str | None = None
    alteration: str | None = None
    alteration_type: AlterationType | None = None
    variant: str | None = None
    driver_bucket: PrimaryDriverBucket | None = None
    actionability: Actionability | None = None
    mechanism_bucket: ClonalMechanismBucket | None = None
    detection_method: str | None = None
    sample_source: SampleSource | None = None
    sample_source_description: str | None = None
    parent_clone: str | None = None
    mechanism: str | None = None
    state_change_cause: str | None = None
    rationale: str | None = None
    procedure: str | None = None
    target_sites: list[str] | None = None
    intent: EventIntent | None = None
    modality: LocalInterventionModality | None = None
    modality_description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_event_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return canonicalize_legacy_event_dict(value)

    @field_validator("ae")
    @classmethod
    def _ae_must_be_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("ae must be non-empty when present")
        return value

    @model_validator(mode="after")
    def _intent_matches_event_type(self) -> LedgerEvent:
        if self.intent is None:
            return self
        if self.type in {"systemic_treatment_start", "systemic_treatment_switch"}:
            if self.intent not in SYSTEMIC_TREATMENT_INTENTS:
                raise ValueError(
                    f"intent {self.intent!r} is not valid for {self.type}; "
                    f"use one of {sorted(SYSTEMIC_TREATMENT_INTENTS)!r}"
                )
        elif self.type == "local_intervention":
            if self.intent not in LOCAL_INTERVENTION_INTENTS:
                raise ValueError(
                    f"intent {self.intent!r} is not valid for local_intervention; "
                    f"use one of {sorted(LOCAL_INTERVENTION_INTENTS)!r}"
                )
        return self


class TreatmentCourse(_ReplayModel):
    treatment_id: str
    # Derived course order/line-like index. The LLM never emits this.
    line: int
    regimen: Regimen
    start_date: date
    end_date: date | None = None
    end_reason: TreatmentReason | None = None


class OutcomeState(_ReplayModel):
    type: Literal["death", "last_followup"]
    date: date
    cause_category: DeathCauseCategory | None = None
    cause: str | None = None
    eol_care_pattern: EOLCarePattern | None = None
    eol_care_description: str | None = None


class PatientState(_ReplayModel):
    ecog_ps: int = Field(ge=0, le=4)
    weight_kg: float
    symptom_burden: SymptomBurden
    organ_function: OrganFunction
    dominant_symptoms: list[str] | None = None

    @field_validator("weight_kg")
    @classmethod
    def _finite_materialized_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight_kg must be finite")
        return value

    @property
    def ecog(self) -> int:
        """Archived-artifact alias for the current ``ecog_ps`` field."""
        return self.ecog_ps


class MaterializedLedgerEvent(_ReplayModel):
    event_id: str
    date: date
    type: EventType
    recognized_date: date | None = None
    disease: list[DiseaseBurdenChange] | None = None
    patient: PatientDelta | None = None
    # Derived course order/line-like index. The LLM never emits this.
    line: int | None = None
    regimen: Regimen | None = None
    cause: str | None = None
    cause_category: DeathCauseCategory | None = None
    eol_care_pattern: EOLCarePattern | None = None
    eol_care_description: str | None = None
    treatment_id: str | None = None
    stopped_treatment_id: str | None = None
    reason: TreatmentReason | None = None
    modification_type: SystemicTreatmentModificationType | None = None
    affected_components: list[str] | None = None
    ae: str | None = None
    ae_category: AECategory | None = None
    ae_category_description: str | None = None
    toxicity_transition: ToxicityTransition | None = None
    ctcae_v5_grade: int | None = None
    causes: list[str] | None = None
    new_clone_id: str | None = None
    gene: str | None = None
    alteration: str | None = None
    alteration_type: AlterationType | None = None
    variant: str | None = None
    driver_bucket: PrimaryDriverBucket | None = None
    actionability: Actionability | None = None
    mechanism_bucket: ClonalMechanismBucket | None = None
    detection_method: str | None = None
    sample_source: SampleSource | None = None
    sample_source_description: str | None = None
    parent_clone: str | None = None
    mechanism: str | None = None
    state_change_cause: str | None = None
    rationale: str | None = None
    procedure: str | None = None
    target_sites: list[str] | None = None
    intent: EventIntent | None = None
    modality: LocalInterventionModality | None = None
    modality_description: str | None = None
    resolved_treatment_id: str | None = None
    assigned_clone_id: str | None = None
    alteration_id: str | None = None


class ToxicityRecord(_ReplayModel):
    """Materialized AE descriptor — tracked in replay state and snapshots."""
    event_id: str
    ae: str
    ae_category: AECategory
    ae_category_description: str | None = None
    ctcae_v5_grade: int = Field(ge=1, le=5)
    causes: list[str] | None = None
    resolved_treatment_id: str | None = None
    onset_date: date
    recognized_date: date | None = None
    resolution_date: date | None = None


class AcquiredAlteration(_ReplayModel):
    """Materialized acquired alteration tracked in replay state."""
    alteration_id: str
    gene: str | None = None
    alteration: str
    alteration_type: AlterationType | None = None
    variant: str | None = None
    driver_bucket: PrimaryDriverBucket | None = None
    actionability: Actionability | None = None
    mechanism_bucket: ClonalMechanismBucket | None = None
    first_presence_date: date
    recognized_date: date | None = None
    detection_method: str | None = None
    sample_source: SampleSource | None = None
    sample_source_description: str | None = None
    supporting_clone: str  # pipeline-assigned clone_id created by the lead event
    mechanism: str | None = None


class Snapshot(_ReplayModel):
    state_id: str
    event_id: str | None
    date: date
    event_type: str
    clonal_state: list[DiseaseBurdenChange]
    patient_state: PatientState
    total_burden_band: BurdenBand
    active_treatments: list[TreatmentCourse]
    vital_status: VitalStatus
    outcome: OutcomeState | None = None
    # Replay-owned state extensions derived from ledger events.
    active_aes: list[ToxicityRecord] = Field(default_factory=list)
    clone_registry: list[CloneRegistryEntry] = Field(default_factory=list)
    acquired_alterations: list[AcquiredAlteration] = Field(default_factory=list)


class Violation(_ReplayModel):
    severity: Literal["warning"] = "warning"
    rule_id: str
    message: str
    event_id: str | None = None


class MaterializedTrajectory(_ReplayModel):
    baseline: BaselineObject
    ledger: list[MaterializedLedgerEvent]
    snapshots: list[Snapshot]
    active_treatments_at_event: list[list[TreatmentCourse]]
    validation_log: list[Violation]
    # Replay-derived longitudinal summaries.
    acquired_alterations: list[AcquiredAlteration] = Field(default_factory=list)
    final_clone_registry: list[CloneRegistryEntry] = Field(default_factory=list)
    resolved_aes: list[ToxicityRecord] = Field(default_factory=list)
    # Closed treatment courses (in chronological close-order) with end_date +
    # end_reason populated. The summary/viewer reads this directly rather than
    # re-deriving from the ledger. Open courses still live in the final
    # snapshot's active_treatments.
    treatment_history: list[TreatmentCourse] = Field(default_factory=list)


@dataclass(frozen=True)
class _ReplayState:
    date: date
    event_id: str | None
    event_type: str
    disease_by_key: dict[tuple[str, str], DiseaseBurdenChange]
    patient_state: PatientState
    active_treatments: list[TreatmentCourse]
    treatment_history: list[TreatmentCourse]
    outcome: OutcomeState | None
    next_treatment_index: int
    # Replay state copied with dataclasses.replace at each materialization step.
    clone_registry: list[CloneRegistryEntry]
    active_aes: list[ToxicityRecord]
    resolved_aes: list[ToxicityRecord]
    acquired_alterations: list[AcquiredAlteration]
    next_acquired_clone_index: int       # next acquired clone gets id C{this}
    next_alteration_index: int           # next acquired alteration gets id amol_{this:03d}
    clonal_evolution_count: int


def replay(
    baseline: BaselineObject | Mapping[str, Any],
    ledger: Sequence[LedgerEvent | Mapping[str, Any]],
) -> MaterializedTrajectory:
    """Apply ledger deltas to baseline and return full materialized snapshots."""

    baseline_model = _coerce_baseline(baseline)
    ledger_models = [_coerce_event(event) for event in ledger]

    _validate_baseline(baseline_model)
    validate_chronology(ledger_models)
    _validate_terminality(ledger_models)

    state = _initial_state(baseline_model)
    snapshots = [snapshot_from_event(state, None, state_id="s0")]
    materialized_events: list[MaterializedLedgerEvent] = []
    validation_log: list[Violation] = []
    seen_date_type: set[tuple[date, EventType]] = set()
    recent_progression_date: date | None = None  # for SOFT_PROGRESSION_WITHOUT_REASON

    for index, event in enumerate(ledger_models, start=1):
        event_id = f"e{index:03d}"
        if (event.date, event.type) in seen_date_type:
            validation_log.append(
                Violation(
                    rule_id="SOFT_SAME_DAY_REPEATED_EVENT_TYPE",
                    message=f"Repeated {event.type} event on {event.date.isoformat()}",
                    event_id=event_id,
                )
            )
        seen_date_type.add((event.date, event.type))

        validation_log.extend(
            validate_event_applicability(
                event, state, event_id=event_id,
                recent_progression_date=recent_progression_date,
            )
        )

        # Auto-promote (forward): a model may emit systemic_treatment_start
        # while another treatment is still active when it means a line
        # transition. Rather than losing the trajectory's clinical content,
        # treat the event as systemic_treatment_switch and record the
        # auto-promotion as a soft warning. The original event in
        # `ledger_models` (and on disk in parsed_ledger.json) is untouched;
        # only the materialized form and the treatment history reflect the
        # corrected semantics.
        #
        # Auto-promote (inverse): symmetric case — a model may emit
        # systemic_treatment_switch while no systemic treatment is active when
        # it means starting the first regimen. Promote to
        # systemic_treatment_start IFF the event has a valid regimen. If the
        # regimen is missing, fall through to the hard contract validator so
        # the EVENT_CONTRACT error fires correctly.
        effective_event = event
        if event.type == "systemic_treatment_start" and state.active_treatments:
            active = state.active_treatments[-1]
            active_label = active.regimen.label if active.regimen else "(unknown)"
            new_label = event.regimen.label if event.regimen else "(unknown)"
            validation_log.append(
                Violation(
                    rule_id="SOFT_LIFECYCLE_AUTO_PROMOTED_START_TO_SWITCH",
                    message=(
                        f"event[{index}] {event.date.isoformat()} "
                        f"type=systemic_treatment_start emitted while "
                        f"regimen={active_label!r} is active; replay "
                        f"auto-promoted to systemic_treatment_switch "
                        f"(new regimen={new_label!r})"
                    ),
                    event_id=event_id,
                )
            )
            effective_event = event.model_copy(
                update={"type": "systemic_treatment_switch"}
            )
        elif (
            event.type == "systemic_treatment_switch"
            and not state.active_treatments
            and event.regimen is not None
        ):
            new_label = event.regimen.label
            validation_log.append(
                Violation(
                    rule_id="SOFT_LIFECYCLE_AUTO_PROMOTED_SWITCH_TO_START",
                    message=(
                        f"event[{index}] {event.date.isoformat()} "
                        f"type=systemic_treatment_switch emitted while no "
                        f"systemic treatment is active; replay auto-promoted "
                        f"to systemic_treatment_start "
                        f"(new regimen={new_label!r})"
                    ),
                    event_id=event_id,
                )
            )
            effective_event = event.model_copy(
                update={"type": "systemic_treatment_start"}
            )

        _validate_hard_event_contract(effective_event, state)

        state, materialized_event = apply_delta(
            state, effective_event, event_id=event_id
        )
        materialized_events.append(materialized_event)
        snapshot = snapshot_from_event(state, materialized_event, state_id=f"s{index}")
        snapshots.append(snapshot)

        if event.type == "progression":
            recent_progression_date = event.date

    validate_terminal(snapshots[-1])

    # Grade 5 toxicity should align with a death-terminated trajectory whose
    # cause_category is treatment_related AND whose date is close to the G5
    # event. CTCAE grade 5 means death due to AE; in practice a few-day gap is
    # plausible (terminal hospitalization, withdrawal of care) but a months- or
    # years-long gap is incoherent. The 14-day window is the paper-v1 bound for
    # "G5 event is the proximate cause of the death event."
    #
    # Both onset-at-5 and grade_change-to-5 are evaluated identically. The
    # cause-mismatch and date-delay checks are independent — both can fire on
    # the same G5 event.
    terminal_event = ledger_models[-1] if ledger_models else None
    terminal_is_death = terminal_event is not None and terminal_event.type == "death"
    terminal_cause_category = (
        terminal_event.cause_category if terminal_is_death else None
    )
    for mat_event in materialized_events:
        if not (mat_event.type == "toxicity" and mat_event.ctcae_v5_grade == 5):
            continue
        if not terminal_is_death:
            validation_log.append(
                Violation(
                    rule_id="SOFT_TOXICITY_GRADE5_NOT_FATAL",
                    message=(
                        f"Grade 5 toxicity {mat_event.ae!r} has "
                        f"trajectory terminal_is_death={terminal_is_death}"
                    ),
                    event_id=mat_event.event_id,
                )
            )
            continue
        if terminal_cause_category != "treatment_related":
            validation_log.append(
                Violation(
                    rule_id="SOFT_TOXICITY_GRADE5_CAUSE_MISMATCH",
                    message=(
                        f"Grade 5 toxicity {mat_event.ae!r} but death "
                        f"cause_category={terminal_cause_category!r} "
                        f"(expected 'treatment_related')"
                    ),
                    event_id=mat_event.event_id,
                )
            )
        days_g5_to_death = (terminal_event.date - mat_event.date).days
        if days_g5_to_death > 14:
            validation_log.append(
                Violation(
                    rule_id="SOFT_TOXICITY_GRADE5_DELAYED",
                    message=(
                        f"Grade 5 toxicity {mat_event.ae!r} on "
                        f"{mat_event.date.isoformat()} but death is "
                        f"{days_g5_to_death} days later "
                        f"({terminal_event.date.isoformat()}); CTCAE G5 "
                        f"means death due to AE, expected proximate "
                        f"timing (≤14 days)"
                    ),
                    event_id=mat_event.event_id,
                )
            )

    # age_related_frailty death with non-absent active disease in the terminal
    # snapshot. Old patients can die of frailty with cancer still present, so
    # this is a finding surfaced for expert review rather than a contradiction.
    if (
        terminal_is_death
        and terminal_event is not None
        and terminal_event.cause_category == "age_related_frailty"
    ):
        terminal_snapshot = snapshots[-1]
        has_active_disease = any(
            entry.burden != "absent" for entry in terminal_snapshot.clonal_state
        )
        if has_active_disease:
            validation_log.append(
                Violation(
                    rule_id="SOFT_FRAILTY_DEATH_WITH_ACTIVE_DISEASE",
                    message=(
                        "death.cause_category=='age_related_frailty' but terminal "
                        "snapshot has non-absent active disease burden"
                    ),
                    event_id=terminal_snapshot.event_id,
                )
            )

    return MaterializedTrajectory(
        baseline=baseline_model,
        ledger=materialized_events,
        snapshots=snapshots,
        active_treatments_at_event=[snapshot.active_treatments for snapshot in snapshots[1:]],
        validation_log=validation_log,
        acquired_alterations=list(state.acquired_alterations),
        final_clone_registry=list(state.clone_registry),
        resolved_aes=list(state.resolved_aes),
        treatment_history=list(state.treatment_history),
    )


def validate_chronology(ledger: Sequence[LedgerEvent]) -> None:
    """Dates must be monotonically non-decreasing in emitted order."""

    for previous, current in zip(ledger, ledger[1:]):
        if current.date < previous.date:
            raise ReplayError(
                "DATE_MONOTONICITY",
                f"event on {current.date.isoformat()} follows later event on {previous.date.isoformat()}",
            )


def validate_event_applicability(
    event: LedgerEvent,
    state: _ReplayState,
    *,
    event_id: str | None = None,
    recent_progression_date: date | None = None,
) -> list[Violation]:
    """Return non-blocking warnings for clinically unusual transitions."""

    violations: list[Violation] = []
    if event.patient is not None and event.patient.ecog_ps is not None:
        if state.patient_state.ecog_ps >= 4 and event.patient.ecog_ps <= 1:
            violations.append(
                Violation(
                    rule_id="SOFT_IMPLAUSIBLE_ECOG_JUMP",
                    message=f"ECOG changed from {state.patient_state.ecog_ps} to {event.patient.ecog_ps} in one event",
                    event_id=event_id,
                )
            )

    for change in event.disease or []:
        if change.kinetic == "appearing" and change.burden == "absent":
            violations.append(
                Violation(
                    rule_id="SOFT_KINETIC_BURDEN_MISMATCH",
                    message="kinetic 'appearing' paired with absent burden",
                    event_id=event_id,
                )
            )
        if change.kinetic == "cleared" and change.burden != "absent":
            violations.append(
                Violation(
                    rule_id="SOFT_KINETIC_BURDEN_MISMATCH",
                    message=f"kinetic '{change.kinetic}' paired with burden '{change.burden}'",
                    event_id=event_id,
                )
            )

    # Progression preceded a systemic treatment switch within
    # 30 days but the switch's reason is EXPLICITLY non-progression. A missing
    # reason field does not fire this rule, since the model has not asserted a
    # contradicting reason.
    if (
        event.type == "systemic_treatment_switch"
        and recent_progression_date is not None
        and event.reason is not None
        and event.reason != "progression"
    ):
        days_since = (event.date - recent_progression_date).days
        if 0 <= days_since <= 30:
            violations.append(
                Violation(
                    rule_id="SOFT_PROGRESSION_WITHOUT_REASON",
                    message=(
                        f"treatment_switch on {event.date.isoformat()} follows a "
                        f"progression on {recent_progression_date.isoformat()} "
                        f"({days_since}d ago) but reason={event.reason!r}"
                    ),
                    event_id=event_id,
                )
            )

    if event.type == "toxicity" and event.toxicity_transition == "resolved":
        if _find_active_toxicity(state.active_aes, event) is None:
            violations.append(
                Violation(
                    rule_id="SOFT_TOXICITY_RESOLUTION_WITHOUT_ONSET",
                    message=f"toxicity resolution for {event.ae!r} has no matching active toxicity",
                    event_id=event_id,
                )
            )

    if event.type == "toxicity" and event.toxicity_transition == "grade_change":
        match_index = _find_active_toxicity(state.active_aes, event)
        if match_index is None:
            violations.append(
                Violation(
                    rule_id="SOFT_TOXICITY_GRADE_CHANGE_WITHOUT_ONSET",
                    message=f"toxicity grade_change for {event.ae!r} has no matching active toxicity",
                    event_id=event_id,
                )
            )
        else:
            prior_grade = state.active_aes[match_index].ctcae_v5_grade
            if event.ctcae_v5_grade == prior_grade:
                violations.append(
                    Violation(
                        rule_id="SOFT_TOXICITY_GRADE_CHANGE_NOOP",
                        message=(
                            f"toxicity grade_change for {event.ae!r} keeps grade "
                            f"{prior_grade} (no-op transition)"
                        ),
                        event_id=event_id,
                    )
                )

    return violations


def apply_delta(
    state: _ReplayState,
    event: LedgerEvent,
    *,
    event_id: str = "e000",
) -> tuple[_ReplayState, MaterializedLedgerEvent]:
    """Purely apply one event's deltas to the current replay state."""

    disease_by_key = dict(state.disease_by_key)
    for change in event.disease or []:
        disease_by_key[(change.clone, change.site)] = change

    active_treatments = list(state.active_treatments)
    treatment_history = list(state.treatment_history)
    next_treatment_index = state.next_treatment_index
    clone_registry = list(state.clone_registry)
    active_aes = list(state.active_aes)
    resolved_aes = list(state.resolved_aes)
    acquired_alterations = list(state.acquired_alterations)
    next_acquired_clone_index = state.next_acquired_clone_index
    next_alteration_index = state.next_alteration_index
    clonal_evolution_count = state.clonal_evolution_count

    treatment_id: str | None = None
    stopped_treatment_id: str | None = None
    resolved_treatment_id: str | None = None
    assigned_clone_id: str | None = None
    alteration_id: str | None = None
    # For grade_change / resolved, inherit ae_category and causes from the
    # matched onset record so event-row analyses (derivation.py) see the
    # category/attribution on every toxicity event, not only on onsets. The
    # prompt promises this carry-forward; without it, derivation pivots on
    # toxicity rows lose ae_category for grade_change/resolved events.
    inherited_ae_category: AECategory | None = None
    inherited_ae_category_description: str | None = None
    inherited_causes: list[str] | None = None

    patient_state = state.patient_state

    if event.type == "systemic_treatment_start":
        treatment_id = f"tx{next_treatment_index:03d}"
        active_treatments.append(
            TreatmentCourse(
                treatment_id=treatment_id,
                line=next_treatment_index,
                regimen=event.regimen or Regimen(label="", components=[""]),
                start_date=event.date,
            )
        )
        next_treatment_index += 1

    elif event.type == "systemic_treatment_switch":
        stopped = active_treatments.pop()
        stopped_treatment_id = stopped.treatment_id
        treatment_history.append(
            stopped.model_copy(update={"end_date": event.date, "end_reason": event.reason})
        )
        treatment_id = f"tx{next_treatment_index:03d}"
        active_treatments.append(
            TreatmentCourse(
                treatment_id=treatment_id,
                line=next_treatment_index,
                regimen=event.regimen or Regimen(label="", components=[""]),
                start_date=event.date,
            )
        )
        next_treatment_index += 1

    elif event.type == "systemic_treatment_stop":
        stopped = active_treatments.pop()
        stopped_treatment_id = stopped.treatment_id
        treatment_history.append(
            stopped.model_copy(update={"end_date": event.date, "end_reason": event.reason})
        )

    elif event.type == "toxicity":
        if event.toxicity_transition == "onset":
            if len(active_treatments) == 1 and event.causes:
                active_components = set(active_treatments[-1].regimen.components)
                if any(cause in active_components for cause in event.causes):
                    resolved_treatment_id = active_treatments[-1].treatment_id
            active_aes.append(
                ToxicityRecord(
                    event_id=event_id,
                    ae=event.ae or "",
                    ae_category=event.ae_category or "other_specified",
                    ae_category_description=event.ae_category_description,
                    ctcae_v5_grade=event.ctcae_v5_grade or 1,
                    causes=event.causes,
                    resolved_treatment_id=resolved_treatment_id,
                    onset_date=event.date,
                    recognized_date=event.recognized_date,
                )
            )
        elif event.toxicity_transition == "grade_change":
            # Update existing active record in place. Direction (worsening vs
            # improving) is derivable from prior grade vs new grade; we don't
            # create a new active record. If no match (soft-warned upstream),
            # the event is a no-op for state.
            match_index = _find_active_toxicity(active_aes, event)
            if match_index is not None and event.ctcae_v5_grade is not None:
                matched = active_aes[match_index]
                inherited_ae_category = matched.ae_category
                inherited_ae_category_description = matched.ae_category_description
                inherited_causes = matched.causes
                active_aes[match_index] = matched.model_copy(
                    update={
                        "ctcae_v5_grade": event.ctcae_v5_grade,
                        "recognized_date": (
                            event.recognized_date
                            if event.recognized_date is not None
                            else matched.recognized_date
                        ),
                    }
                )
        elif event.toxicity_transition == "resolved":
            match_index = _find_active_toxicity(active_aes, event)
            if match_index is not None:
                matched = active_aes[match_index]
                inherited_ae_category = matched.ae_category
                inherited_ae_category_description = matched.ae_category_description
                inherited_causes = matched.causes
                resolved = active_aes.pop(match_index).model_copy(update={"resolution_date": event.date})
                resolved_aes.append(resolved)

    elif event.type == "clonal_evolution_event":
        assigned_clone_id = event.new_clone_id or f"C{next_acquired_clone_index}"
        next_acquired_clone_index += 1
        clonal_evolution_count += 1
        alteration_id = f"amol_{next_alteration_index:03d}"
        next_alteration_index += 1
        clone_registry.append(
            CloneRegistryEntry(
                clone_id=assigned_clone_id,
                clone_role="resistance_subclone",
                drivers=[event.alteration] if event.alteration else [],
                first_presence_date=event.date,
                parent_clone_id_or_null=event.parent_clone,
            )
        )
        acquired_alterations.append(
            AcquiredAlteration(
                alteration_id=alteration_id,
                gene=event.gene,
                alteration=event.alteration or "",
                alteration_type=event.alteration_type,
                variant=event.variant,
                driver_bucket=event.driver_bucket,
                actionability=event.actionability,
                mechanism_bucket=event.mechanism_bucket,
                first_presence_date=event.date,
                recognized_date=event.recognized_date,
                detection_method=event.detection_method,
                sample_source=event.sample_source,
                sample_source_description=event.sample_source_description,
                supporting_clone=assigned_clone_id,
                mechanism=event.mechanism,
            )
        )

    elif event.type == "local_intervention":
        # Local interventions may carry immediate disease or patient deltas,
        # already merged into state here.
        pass

    elif event.type in {"systemic_treatment_modification", "patient_state_update"}:
        pass

    outcome = state.outcome
    if event.type == "death":
        for treatment in active_treatments:
            treatment_history.append(
                treatment.model_copy(update={"end_date": event.date, "end_reason": event.reason})
            )
        active_treatments = []
        outcome = OutcomeState(
            type="death",
            date=event.date,
            cause_category=event.cause_category,
            cause=event.cause,
            eol_care_pattern=event.eol_care_pattern,
            eol_care_description=event.eol_care_description,
        )
    elif event.type == "last_followup":
        outcome = OutcomeState(type="last_followup", date=event.date, cause=None)

    if event.patient is not None:
        patient_state = _apply_patient_delta(patient_state, event.patient)

    next_state = _ReplayState(
        date=event.date,
        event_id=event_id,
        event_type=event.type,
        disease_by_key=disease_by_key,
        patient_state=patient_state,
        active_treatments=active_treatments,
        treatment_history=treatment_history,
        outcome=outcome,
        next_treatment_index=next_treatment_index,
        clone_registry=clone_registry,
        active_aes=active_aes,
        resolved_aes=resolved_aes,
        acquired_alterations=acquired_alterations,
        next_acquired_clone_index=next_acquired_clone_index,
        next_alteration_index=next_alteration_index,
        clonal_evolution_count=clonal_evolution_count,
    )
    materialized_event = MaterializedLedgerEvent(
        event_id=event_id,
        date=event.date,
        type=event.type,
        recognized_date=event.recognized_date,
        disease=event.disease,
        patient=event.patient,
        line=active_treatments[-1].line if treatment_id and active_treatments else None,
        regimen=event.regimen,
        cause=event.cause,
        cause_category=event.cause_category,
        eol_care_pattern=event.eol_care_pattern,
        eol_care_description=event.eol_care_description,
        treatment_id=treatment_id,
        stopped_treatment_id=stopped_treatment_id,
        reason=event.reason,
        modification_type=event.modification_type,
        affected_components=event.affected_components,
        ae=event.ae,
        ae_category=(
            event.ae_category if event.ae_category is not None else inherited_ae_category
        ),
        ae_category_description=(
            event.ae_category_description
            if event.ae_category_description is not None
            else inherited_ae_category_description
        ),
        toxicity_transition=event.toxicity_transition,
        ctcae_v5_grade=event.ctcae_v5_grade,
        causes=(event.causes if event.causes is not None else inherited_causes),
        new_clone_id=event.new_clone_id,
        gene=event.gene,
        alteration=event.alteration,
        alteration_type=event.alteration_type,
        variant=event.variant,
        driver_bucket=event.driver_bucket,
        actionability=event.actionability,
        mechanism_bucket=event.mechanism_bucket,
        detection_method=event.detection_method,
        sample_source=event.sample_source,
        sample_source_description=event.sample_source_description,
        parent_clone=event.parent_clone,
        mechanism=event.mechanism,
        state_change_cause=event.state_change_cause,
        rationale=event.rationale,
        procedure=event.procedure,
        target_sites=event.target_sites,
        intent=event.intent,
        modality=event.modality,
        modality_description=event.modality_description,
        resolved_treatment_id=resolved_treatment_id,
        assigned_clone_id=assigned_clone_id,
        alteration_id=alteration_id,
    )
    return next_state, materialized_event


def _find_active_toxicity(active: Sequence[ToxicityRecord], event: LedgerEvent) -> int | None:
    """Find the active toxicity resolved by a toxicity-resolution event."""

    for index in range(len(active) - 1, -1, -1):
        record = active[index]
        if record.ae != event.ae:
            continue
        if event.causes and record.causes and set(event.causes) != set(record.causes):
            continue
        return index
    return None


def snapshot_from_event(
    state: _ReplayState,
    event: MaterializedLedgerEvent | None,
    *,
    state_id: str = "s0",
) -> Snapshot:
    """Emit a full materialized snapshot from the current replay state."""

    return Snapshot(
        state_id=state_id,
        event_id=event.event_id if event is not None else None,
        date=state.date,
        event_type=event.type if event is not None else "baseline",
        clonal_state=_sorted_disease_state(state.disease_by_key),
        patient_state=state.patient_state,
        total_burden_band=_derive_total_burden_band(state.disease_by_key.values()),
        active_treatments=state.active_treatments,
        vital_status="deceased" if state.outcome is not None and state.outcome.type == "death" else "alive",
        outcome=state.outcome,
        active_aes=list(state.active_aes),
        clone_registry=list(state.clone_registry),
        acquired_alterations=list(state.acquired_alterations),
    )


def validate_terminal(snapshot: Snapshot) -> None:
    """Final snapshot must satisfy the terminal event contract."""

    if snapshot.event_type == "death":
        if snapshot.vital_status != "deceased" or snapshot.outcome is None or snapshot.outcome.type != "death":
            raise ReplayError("TERMINAL_CONTRACT", "death snapshot must have deceased vital status and death outcome")
        if not snapshot.outcome.cause:
            raise ReplayError("TERMINAL_CONTRACT", "death outcome requires cause")
        if snapshot.outcome.cause_category is None:
            raise ReplayError("TERMINAL_CONTRACT", "death outcome requires cause_category")
        if snapshot.outcome.eol_care_pattern is None:
            raise ReplayError("TERMINAL_CONTRACT", "death outcome requires eol_care_pattern")
    elif snapshot.event_type == "last_followup":
        if snapshot.vital_status != "alive" or snapshot.outcome is None or snapshot.outcome.type != "last_followup":
            raise ReplayError("TERMINAL_CONTRACT", "last_followup snapshot must have alive vital status")
    else:
        raise ReplayError("TERMINAL_CONTRACT", "final snapshot must be death or last_followup")


def _coerce_baseline(value: BaselineObject | Mapping[str, Any]) -> BaselineObject:
    if isinstance(value, BaselineObject):
        return value
    return BaselineObject.model_validate(value)


def _coerce_event(value: LedgerEvent | Mapping[str, Any]) -> LedgerEvent:
    if isinstance(value, LedgerEvent):
        return value
    return LedgerEvent.model_validate(value)


def _initial_state(baseline: BaselineObject) -> _ReplayState:
    """Construct the baseline state before applying ledger events.

    Baseline clone IDs are validated separately as contiguous C0..C{N-1}.
    Acquired clones therefore start at C{N}, where N is the number of founder
    clones, so deterministic replay never collides with a founder clone ID.
    """
    patient_state = PatientState(
        ecog_ps=baseline.identity_card.demographics.ecog_ps_at_diagnosis,
        weight_kg=baseline.baseline_patient_state.weight_kg,
        symptom_burden=baseline.baseline_patient_state.symptom_burden,
        organ_function=OrganFunction(
            renal=baseline.identity_card.baseline_health.renal,
            hepatic=baseline.identity_card.baseline_health.hepatic,
            cardiac=baseline.identity_card.baseline_health.cardiac,
            marrow_reserve=baseline.identity_card.baseline_health.marrow_reserve,
            pulmonary=baseline.identity_card.baseline_health.pulmonary,
        ),
    )
    disease_by_key = {
        (entry.clone, entry.site): entry
        for entry in baseline.baseline_disease_burden
    }
    n_founders = len(baseline.clone_registry)
    return _ReplayState(
        date=baseline.identity_card.diagnosis.diagnosis_date,
        event_id=None,
        event_type="baseline",
        disease_by_key=disease_by_key,
        patient_state=patient_state,
        active_treatments=[],
        treatment_history=[],
        outcome=None,
        next_treatment_index=1,
        clone_registry=list(baseline.clone_registry),
        active_aes=[],
        resolved_aes=[],
        acquired_alterations=[],
        next_acquired_clone_index=n_founders,
        next_alteration_index=1,
        clonal_evolution_count=0,
    )


def _validate_baseline(baseline: BaselineObject) -> None:
    clone_ids = [entry.clone_id for entry in baseline.clone_registry]
    if len(set(clone_ids)) != len(clone_ids):
        raise ReplayError("CLONE_REGISTRY_UNIQUE", "clone_registry contains duplicate clone_id values")

    # Baseline clone IDs must be contiguous C0..C{N-1} so deterministic
    # acquired-clone naming never collides with an existing baseline ID.
    expected = [f"C{i}" for i in range(len(clone_ids))]
    if clone_ids != expected:
        raise ReplayError(
            "CLONE_REGISTRY_NAMING",
            f"baseline.clone_registry must use contiguous C0..C{{N-1}} ids in "
            f"order so acquired-clone naming under §4.2.2.1 cannot collide; "
            f"got {clone_ids!r}, expected {expected!r}",
        )

    clone_set = set(clone_ids)
    for burden in baseline.baseline_disease_burden:
        if burden.clone not in clone_set:
            raise ReplayError("CLONE_REFERENCE", f"baseline disease burden references unknown clone {burden.clone!r}")


def _validate_terminality(ledger: Sequence[LedgerEvent]) -> None:
    terminal_indices = [
        index for index, event in enumerate(ledger)
        if event.type in {"death", "last_followup"}
    ]
    if len(terminal_indices) != 1:
        raise ReplayError("TERMINALITY", f"ledger must contain exactly one terminal event; got {len(terminal_indices)}")
    if terminal_indices[0] != len(ledger) - 1:
        raise ReplayError("TERMINALITY", "terminal event must be the final ledger event")


def _validate_hard_event_contract(event: LedgerEvent, state: _ReplayState) -> None:
    if event.recognized_date is not None and event.recognized_date < event.date:
        raise ReplayError("RECOGNITION_TIMING", "recognized_date cannot precede event date")

    # Clone references must resolve against the current clone_registry, which
    # includes founders + any clones introduced by prior clonal_evolution_events.
    registry_ids = {entry.clone_id for entry in state.clone_registry}
    for change in event.disease or []:
        if change.clone not in registry_ids:
            raise ReplayError(
                "CLONE_REFERENCE",
                f"disease delta references unknown clone {change.clone!r} "
                f"(not in clone_registry at event date — either a typo, a missing "
                f"founder, or a clone whose creating clonal_evolution_event has not "
                f"yet fired in the ledger)",
            )

    if event.type in {"response", "progression"} and not event.disease:
        raise ReplayError("EVENT_CONTRACT", f"{event.type} requires a non-empty disease delta")

    if event.type == "systemic_treatment_start":
        if event.regimen is None:
            raise ReplayError("EVENT_CONTRACT", "systemic_treatment_start requires regimen")
        if state.active_treatments:
            raise ReplayError("TREATMENT_LIFECYCLE", "systemic_treatment_start cannot occur while treatment is active")

    if event.type == "systemic_treatment_switch":
        if event.regimen is None:
            raise ReplayError("EVENT_CONTRACT", "systemic_treatment_switch requires regimen")
        if not state.active_treatments:
            raise ReplayError("TREATMENT_LIFECYCLE", "systemic_treatment_switch requires an active treatment")

    if event.type == "systemic_treatment_stop":
        if not state.active_treatments:
            raise ReplayError("TREATMENT_LIFECYCLE", "systemic_treatment_stop requires an active treatment")

    if event.type == "systemic_treatment_modification":
        if not state.active_treatments:
            raise ReplayError("TREATMENT_LIFECYCLE", "systemic_treatment_modification requires an active treatment")
        if event.modification_type is None:
            raise ReplayError("EVENT_CONTRACT", "systemic_treatment_modification requires modification_type")

    if event.type == "death":
        if not event.cause:
            raise ReplayError("EVENT_CONTRACT", "death requires cause")
        if event.cause_category is None:
            raise ReplayError("EVENT_CONTRACT", "death requires cause_category")
        if event.eol_care_pattern is None:
            raise ReplayError("EVENT_CONTRACT", "death requires eol_care_pattern")
        if event.eol_care_pattern == "other_specified" and not event.eol_care_description:
            raise ReplayError("EVENT_CONTRACT", "death.eol_care_pattern other_specified requires eol_care_description")

    if event.type == "toxicity":
        if event.toxicity_transition is None:
            raise ReplayError("EVENT_CONTRACT", "toxicity requires toxicity_transition")
        if not event.ae or not event.ae.strip():
            raise ReplayError("EVENT_CONTRACT", "toxicity requires ae")
        if event.toxicity_transition == "onset":
            if event.ae_category is None:
                raise ReplayError("EVENT_CONTRACT", "toxicity onset requires ae_category")
            if event.ctcae_v5_grade is None:
                raise ReplayError("EVENT_CONTRACT", "toxicity onset requires ctcae_v5_grade")
            if not (1 <= event.ctcae_v5_grade <= 5):
                raise ReplayError("TOXICITY_GRADE_RANGE", f"ctcae_v5_grade={event.ctcae_v5_grade} out of [1,5]")
        if event.toxicity_transition == "grade_change":
            if event.ctcae_v5_grade is None:
                raise ReplayError("EVENT_CONTRACT", "toxicity grade_change requires ctcae_v5_grade")
            if not (1 <= event.ctcae_v5_grade <= 5):
                raise ReplayError("TOXICITY_GRADE_RANGE", f"ctcae_v5_grade={event.ctcae_v5_grade} out of [1,5]")
        if event.toxicity_transition == "resolved":
            if event.ctcae_v5_grade is not None:
                raise ReplayError("EVENT_CONTRACT", "toxicity resolution must omit ctcae_v5_grade")

    if event.type == "clonal_evolution_event":
        if not event.alteration or not event.alteration.strip():
            raise ReplayError("EVENT_CONTRACT", "clonal_evolution_event requires alteration")
        expected_clone_id = f"C{state.next_acquired_clone_index}"
        if event.new_clone_id is None:
            raise ReplayError("EVENT_CONTRACT", "clonal_evolution_event requires new_clone_id")
        if event.new_clone_id != expected_clone_id:
            raise ReplayError(
                "CLONAL_EVOLUTION_CLONE_ID",
                f"clonal_evolution_event.new_clone_id must be {expected_clone_id!r}; got {event.new_clone_id!r}",
            )
        if event.sample_source is None:
            raise ReplayError("EVENT_CONTRACT", "clonal_evolution_event requires sample_source")
        if event.sample_source == "other_specified" and not event.sample_source_description:
            raise ReplayError("EVENT_CONTRACT", "sample_source other_specified requires sample_source_description")
        if event.parent_clone is not None and event.parent_clone not in registry_ids:
            raise ReplayError(
                "CLONAL_EVOLUTION_PARENT_REFERENCE",
                f"clonal_evolution_event.parent_clone={event.parent_clone!r} is not in "
                f"clone_registry at event date — must reference a founder or a previously "
                f"introduced acquired clone.",
            )

    if event.type == "patient_state_update":
        if event.patient is None:
            raise ReplayError("EVENT_CONTRACT", "patient_state_update requires patient delta")

    if event.type == "local_intervention":
        if not event.procedure or not event.procedure.strip():
            raise ReplayError("EVENT_CONTRACT", "local_intervention requires procedure")
        if event.intent is None:
            raise ReplayError("EVENT_CONTRACT", "local_intervention requires intent")
        if event.modality is None:
            raise ReplayError("EVENT_CONTRACT", "local_intervention requires modality")
        if event.modality == "other_specified" and not event.modality_description:
            raise ReplayError("EVENT_CONTRACT", "local_intervention modality other_specified requires modality_description")


def _apply_patient_delta(patient_state: PatientState, delta: PatientDelta) -> PatientState:
    update: dict[str, Any] = {}
    if delta.ecog_ps is not None:
        update["ecog_ps"] = delta.ecog_ps
    if delta.weight_kg is not None:
        update["weight_kg"] = delta.weight_kg
    if delta.symptom_burden is not None:
        update["symptom_burden"] = delta.symptom_burden
    if delta.dominant_symptoms is not None:
        update["dominant_symptoms"] = delta.dominant_symptoms
    if delta.organ_function is not None:
        organ_update = {
            key: value
            for key, value in delta.organ_function.model_dump().items()
            if value is not None
        }
        if organ_update:
            update["organ_function"] = patient_state.organ_function.model_copy(update=organ_update)
    if not update:
        return patient_state
    return patient_state.model_copy(update=update)


def _derive_total_burden_band(entries: Sequence[DiseaseBurdenChange]) -> BurdenBand:
    if not entries:
        return "absent"
    return max((entry.burden for entry in entries), key=lambda burden: _BURDEN_RANK[burden])


def _sorted_disease_state(
    disease_by_key: Mapping[tuple[str, str], DiseaseBurdenChange],
) -> list[DiseaseBurdenChange]:
    return [
        disease_by_key[key]
        for key in sorted(disease_by_key)
    ]
