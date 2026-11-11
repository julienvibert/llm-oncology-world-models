#!/usr/bin/env python3
"""Build the pre-PCA clinical-state feature ledger for paper v1.

This is a semantic and measurement audit, not a state-map fit. It deliberately
does not import PCA, UMAP, treatment effects, outcomes, or survival labels.

The builder:

1. classifies available information as current state X, piecewise context C,
   full prior history H, interval-valued action A, endpoint/EXIT E, or
   generator provenance theta;
2. derives candidate *current physical state* measurements from the complete
   six-model atlas;
3. reports types, ranges, missingness, zero inflation, within-trajectory
   change, reversibility, between-model ranges, and empirical redundancy;
4. leaves ambiguous engineering choices visibly unresolved for clinician
   review before any PCA is fitted.

Run:
  uv run python scripts/build_paper_state_feature_ledger.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "paper_state_feature_ledger"

REFERENCE_RUNS = (
    "pillar1_anchored_atlas_tranche1_v4_n5_live01",
    "pillar1_anchored_atlas_gemma4_n5_live01",
    "sonnet5_mini_atlas_n100_2026_07_08",
    "sonnet5_full_atlas_topup_remaining128_n5_2026_07_09",
    "kimi_k3_off_full_atlas_n5_complete",
    "glm52_full_atlas_n5",
)

BURDEN_RANK = {
    "absent": 0.0,
    "microscopic": 1.0,
    "small": 2.0,
    "moderate": 3.0,
    "bulky": 4.0,
    "diffuse_or_leukemic": 5.0,
}
SYMPTOM_RANK = {
    "none": 0.0,
    "mild": 1.0,
    "moderate": 2.0,
    "severe": 3.0,
}
ORGAN_RANK = {
    "normal": 0.0,
    "mild_impairment": 1.0,
    "moderate_impairment": 2.0,
    "severe_impairment": 3.0,
    "failure": 4.0,
}
ORGAN_FIELDS = (
    "renal",
    "hepatic",
    "cardiac",
    "marrow_reserve",
    "pulmonary",
)
AE_CATEGORIES = (
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
)
SYMPTOM_SYSTEM_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "neurologic",
        r"\b(headache|confusion|cogniti|seiz|ataxi|somnolen|obtund|diplopi|"
        r"hemipar|dysarth|gait|neuropath|numb|paresthe|weakness of|memory|"
        r"visual|vision|vertigo|dizz|syncope|encephal|mental status)\b",
    ),
    (
        "respiratory",
        r"\b(dyspn|shortness of breath|breathless|cough|hypoxi|hemopty|"
        r"orthopn|pleuritic|respiratory|wheez|stridor)\b",
    ),
    (
        "gastrointestinal_hepatobiliary",
        r"\b(abdom|anorexi|appetite|early satiety|nause|vomit|emesis|diarr|"
        r"constipat|dysphag|odynophag|ascites|jaundice|bloat|bowel|melena|"
        r"hematoche|hepat|satiety|stool|rectal|tenesmus|reflux)\b",
    ),
    (
        "hematologic_bleeding",
        r"\b(bleed|bruis|petechi|ecchym|anemi|cytopen|thromb|neutrop|"
        r"lymphopen|pancytopen|coagul|transfusion)\b",
    ),
    (
        "dermatologic",
        r"\b(rash|prurit|skin|cutaneous|erythroder|ulcer|lesion|alopecia|"
        r"desquamat|mucositis|stomatitis)\b",
    ),
    (
        "genitourinary_renal",
        r"\b(urin|hematur|dysuri|flank|renal|kidney|oliguri|nocturi|"
        r"frequency|incontinence)\b",
    ),
    (
        "musculoskeletal",
        r"\b(bone pain|back pain|joint|arthral|myalg|limb pain|neck pain|"
        r"musculoskeletal|fracture|mobility)\b",
    ),
    (
        "cardiovascular",
        r"\b(chest pain|palpitation|cardiac|heart|edema|oedema|tachy|"
        r"brady|hypotens|hypertens)\b",
    ),
    (
        "constitutional",
        r"\b(fatigue|weakness|asthen|cachexi|weight loss|fever|night sweat|"
        r"malaise|letharg|decondition|chills|failure to thrive)\b",
    ),
    (
        "local_mass_or_compression",
        r"\b(mass|lymphaden|swelling|fullness|pressure|hoarse|obstruct|"
        r"compression|distension|enlargement)\b",
    ),
    (
        "pain_other",
        r"\b(pain|discomfort|ache|tender)\b",
    ),
)
SYMPTOM_SYSTEMS = tuple(system for system, _pattern in SYMPTOM_SYSTEM_PATTERNS) + (
    "unmapped",
)

INK = "#172033"
MUTED = "#64748b"
GRID = "#dce4ec"
DOMAIN_COLOR = {
    "tumour phenotype": "#7a5195",
    "host phenotype": "#168a8a",
    "adverse condition": "#d98120",
    "molecular mode": "#a83f45",
}


@dataclass(frozen=True)
class LedgerEntry:
    """One explicit semantic decision or unresolved candidate."""

    object_id: str
    label: str
    role: str
    domain: str
    time_structure: str
    source: str
    raw_representation: str
    proposed_representation: str
    map_status: str
    rationale: str
    known_issue: str
    real_data_bridge: str
    audit_feature_ids: str = ""


@dataclass(frozen=True)
class AuditFeature:
    """Metadata for one numerical quantity audited before feature selection."""

    feature_id: str
    label: str
    domain: str
    source: str
    raw_type: str
    interpretation: str
    candidate_status: str


def _entries() -> tuple[LedgerEntry, ...]:
    """Return the semantic ledger with the final paper-v1 map decisions."""
    return (
        LedgerEntry(
            "anatomic_burden_profile",
            "Current anatomical tumour-burden profile",
            "X",
            "tumour phenotype",
            "mutable at event snapshots",
            "disease_site_rows: canonical site class + burden band",
            "one active row per clone-site entry",
            "nine grouped maximum-burden ranks, each divided by 5",
            "included in final map as nine grouped coordinates",
            "Describes where tumour is physically present now, including CNS.",
            "There are 32 site classes; rare classes and primary-site labels require "
            "clinically approved grouping. Maximum burden discards multiplicity.",
            "Partly portable when longitudinal metastatic-site status is collected.",
            "site__*",
        ),
        LedgerEntry(
            "maximum_site_burden",
            "Maximum current site burden",
            "X",
            "tumour phenotype",
            "mutable at event snapshots",
            "snapshot_rows.total_burden_band",
            "ordinal band: absent to diffuse/leukaemic",
            "rank 0-5, explicitly named maximum rather than total burden",
            "excluded from final map",
            "Captures intensity at the single most involved active site.",
            "Replay defines total_burden_band as the maximum site band; it is not "
            "tumour volume and may duplicate the anatomical profile.",
            "Portable only as an ordinal clinical abstraction.",
            "tumour_burden_max_rank",
        ),
        LedgerEntry(
            "disease_extent_counts",
            "Current disease extent counts",
            "X",
            "tumour phenotype",
            "mutable at event snapshots",
            "snapshot_rows + disease_site_rows",
            "clone-site entries, unique anatomical sites, or site classes",
            "unique resolved original site classes, capped at 5 and divided by 5",
            "included in final map as disease extent",
            "A count may complement maximum burden with spatial extent.",
            "The historical num_active_disease_sites counts clone-site entries, not "
            "unique anatomical sites. All count definitions partially duplicate the "
            "anatomical profile.",
            "Portable if site ascertainment is comparable.",
            "n_clonal_site_entries;n_unique_anatomic_sites;n_site_compartments",
        ),
        LedgerEntry(
            "ecog",
            "Current performance status",
            "X",
            "host phenotype",
            "mutable at event snapshots",
            "snapshot_rows.ecog_ps",
            "ECOG integer 0-4",
            "retain ordinal levels; do not imply equal biological intervals",
            "core candidate",
            "Direct current functional phenotype.",
            "Correlated with symptoms but not clinically identical.",
            "Widely portable, although real observation is intermittent.",
            "ecog_rank",
        ),
        LedgerEntry(
            "symptom_burden",
            "Current overall symptom burden",
            "X",
            "host phenotype",
            "mutable at event snapshots",
            "snapshot_rows.symptom_burden",
            "none, mild, moderate, severe",
            "ordinal rank 0-3 divided by 3; named symptoms remain excluded",
            "included in final map",
            "Symptoms are physical/experienced patient state regardless of cause.",
            "One global band is coarse and overlaps ECOG, organ dysfunction and "
            "adverse conditions.",
            "Potentially portable through structured PRO collection.",
            "symptom_rank;n_dominant_symptoms",
        ),
        LedgerEntry(
            "organ_function",
            "Current organ-function profile",
            "X",
            "host phenotype",
            "mutable at event snapshots",
            "snapshot_rows.organ_{renal,hepatic,cardiac,marrow_reserve,pulmonary}",
            "five ordinal values: normal to failure",
            "five separate ranks 0-4; no duplicated max/sum/count composites",
            "core candidate",
            "Describes current physical host function independent of cause.",
            "A treatment-related cardiac failure and cancer-related cardiac failure "
            "share the same physical coordinate; attribution belongs outside X.",
            "Portable if organ function is prospectively standardized.",
            "organ_renal_rank;organ_hepatic_rank;organ_cardiac_rank;"
            "organ_marrow_reserve_rank;organ_pulmonary_rank",
        ),
        LedgerEntry(
            "relative_weight",
            "Current weight relative to personal baseline",
            "X",
            "host phenotype",
            "mutable at event snapshots",
            "snapshot_rows.weight_kg + each trajectory's baseline snapshot",
            "continuous kilograms",
            "signed weight loss clipped to [-20%, +30%] and mapped to [0, 1]",
            "included in final map",
            "Captures cachexia, fluid change and recovery without encoding body size.",
            "Direction can have different meanings; baseline may already reflect illness.",
            "Portable when baseline and serial weight are available.",
            "weight_change_pct",
        ),
        LedgerEntry(
            "active_adverse_conditions",
            "Current active adverse-condition profile",
            "X",
            "adverse condition",
            "mutable; onset, grade change and resolution",
            "materialized snapshots.active_aes",
            "condition name, system category and CTCAE grade",
            "maximum grade / 5 plus active count capped at 5 and divided by 5",
            "included in final map as two global coordinates",
            "An active rash, diarrhoea or cytopenia is current physical state even "
            "when its treatment attribution is ignored.",
            "May double-count symptoms or organ impairment. A causal label such as "
            "'treatment toxicity' is not itself a physical coordinate.",
            "Potentially portable through longitudinal AE capture.",
            "active_condition_count;active_condition_grade_max;"
            "active_condition_grade_sum;active_condition__*",
        ),
        LedgerEntry(
            "active_clone_count",
            "Current number of active tumour clones",
            "X_or_C",
            "molecular mode",
            "mutable at clonal events",
            "snapshot_rows.num_active_clones",
            "count of clone identifiers with active disease",
            "retain clone identity in context; count is sensitivity-only",
            "excluded from primary map",
            "Could describe current tumour-system complexity.",
            "Not a routine clinical measurement; clone identity and resistance "
            "mechanism may be more meaningful as changing context.",
            "Low portability outside densely profiled cohorts.",
            "n_active_clones",
        ),
        LedgerEntry(
            "acquired_molecular_state",
            "Acquired resistance and transformation state",
            "C",
            "molecular mode",
            "piecewise-changing at detected molecular events",
            "snapshots.acquired_alterations + event_rows molecular fields",
            "specific alteration, mechanism or histologic transformation",
            "time-varying context/mode; retain identity, not only a count",
            "never a PCA coordinate; use to condition path law",
            "It changes the transition law and treatment response.",
            "The count alone is a lossy, mostly irreversible history proxy.",
            "Portable when repeat biopsy or ctDNA is available.",
            "n_acquired_alterations",
        ),
        LedgerEntry(
            "cancer_identity",
            "Cancer type, histology and primary site",
            "C",
            "disease context",
            "piecewise-constant; histology may transform",
            "baseline_rows + transformation events",
            "categorical descriptors",
            "context trajectory C(t), with explicit mode-switch times",
            "never a PCA coordinate; condition/facet dynamics",
            "Defines the disease regime rather than current physical phenotype.",
            "Histology is not necessarily constant.",
            "Routinely portable.",
        ),
        LedgerEntry(
            "host_context",
            "Age, sex, smoking, comorbidity and baseline frailty",
            "C",
            "host context",
            "static or piecewise-changing",
            "baseline_rows + future medical events when available",
            "mixed categorical and continuous",
            "context trajectory C(t); distinguish baseline disease from current "
            "organ manifestation in X(t)",
            "never a PCA coordinate; condition/facet dynamics",
            "Modifies transition probabilities without being the current phenotype map.",
            "The present ledger does not generate every possible incident comorbidity.",
            "Partly portable; definitions vary.",
        ),
        LedgerEntry(
            "baseline_biomarker_context",
            "Baseline driver and biomarkers",
            "C",
            "molecular mode",
            "piecewise-constant until a new molecular event",
            "baseline alteration/biomarker rows",
            "categorical molecular profile",
            "context trajectory C(t)",
            "never a PCA coordinate; condition/facet dynamics",
            "Changes the treatment-response law.",
            "Assay timing and detectability matter.",
            "Portable in molecularly characterized cohorts.",
        ),
        LedgerEntry(
            "complete_history",
            "Complete prior clinical path",
            "H",
            "history",
            "grows with every event",
            "all X(s), C(s), A(s) for s<t",
            "variable-length marked path",
            "H_t = {(X_s,C_s,A_s): s<t}; summaries are observation operators",
            "never a PCA coordinate",
            "This is the ideal conditioning object used by generation.",
            "Sparse real data require prespecified summaries rather than pretending "
            "the full path was observed.",
            "Only partially observed in routine care.",
        ),
        LedgerEntry(
            "history_summaries",
            "Sparse-history summaries",
            "H_tilde",
            "history",
            "updated at clinically significant events",
            "derived from treatment, response, progression and duration records",
            "line, sequence, durations, prior response, occupancy and toxicity",
            "explicit observation operator K_M(H), selected per real-data task",
            "never a PCA coordinate",
            "Approximates complete history when the full path is unavailable.",
            "No single summary is universally sufficient.",
            "Designed around available real-world landmarks.",
        ),
        LedgerEntry(
            "systemic_action",
            "Systemic treatment exposure",
            "A",
            "action",
            "on/off interval with changes at start, switch, hold or stop",
            "treatment_rows start_date/end_date + component rows",
            "regimen/components and exposure interval",
            "piecewise-constant A(t); dosing cycles may be abstracted as continuous "
            "exposure intervals",
            "never a PCA coordinate",
            "Represents the perturbation, not patient state.",
            "Dose intensity is simplified unless explicitly queried.",
            "Portable from medication administration records.",
        ),
        LedgerEntry(
            "local_action",
            "Local intervention",
            "A",
            "action",
            "point event or bounded treatment interval",
            "event_rows local interventions",
            "modality, target site and intent",
            "impulse or interval in A(t), depending on modality",
            "never a PCA coordinate",
            "Represents surgery, radiotherapy, ablation or palliation.",
            "A single on/off convention may be inappropriate for all modalities.",
            "Portable when procedures and radiotherapy are captured.",
        ),
        LedgerEntry(
            "progression_endpoint",
            "Progression or treatment-line exit",
            "E",
            "endpoint",
            "event at a defined horizon or line boundary",
            "event_rows + endpoint_rows",
            "dated event and line-specific time",
            "co-equal path-law readout; may be absorbing only for a line-specific map",
            "never a PCA coordinate",
            "Progression is an endpoint/event; the resulting physical phenotype remains X.",
            "Definitions differ across generated and observational cohorts.",
            "Portable with explicit endpoint definitions.",
        ),
        LedgerEntry(
            "death_exit",
            "Death / absorbing EXIT",
            "E",
            "endpoint",
            "absorbing event",
            "endpoint_rows + terminal snapshot",
            "dated cause-specific or all-cause event",
            "explicit EXIT outside the living-state map",
            "never a PCA coordinate",
            "Death is not a very severe point inside the living patient map.",
            "Cause attribution is separate from all-cause exit.",
            "Routinely portable.",
        ),
        LedgerEntry(
            "censoring",
            "Censoring / end of observation",
            "E",
            "observation process",
            "possible observation boundary",
            "endpoint_rows.censored",
            "binary plus last-follow-up time",
            "retain in general formalism",
            "never a PCA coordinate",
            "Needed for real cohorts.",
            "Generated paper-v1 atlas has no censoring by contract.",
            "Essential in real data.",
        ),
        LedgerEntry(
            "generator_provenance",
            "Generator and run provenance",
            "theta",
            "provenance",
            "constant within a generated future",
            "model/run/future/hash/acceptance columns",
            "categorical identifiers and hashes",
            "audit and trust-map axis",
            "never a PCA coordinate; never a patient coordinate",
            "Identifies which model-implied distribution produced a branch.",
            "Mixing it into X would confuse model disagreement with patient state.",
            "Not applicable to biological patients.",
        ),
    )


BASE_AUDIT_FEATURES = (
    AuditFeature(
        "tumour_burden_max_rank",
        "maximum site burden",
        "tumour phenotype",
        "snapshot total_burden_band",
        "ordinal 0-5",
        "maximum current burden band across active sites",
        "derived summary; unresolved",
    ),
    AuditFeature(
        "n_clonal_site_entries",
        "clone–site entries",
        "tumour phenotype",
        "snapshot num_active_disease_sites",
        "count",
        "active clone–site pairs; historically mislabeled disease sites",
        "derived summary; unresolved",
    ),
    AuditFeature(
        "n_unique_anatomic_sites",
        "unique anatomical sites",
        "tumour phenotype",
        "disease-site signature",
        "count",
        "distinct current canonical-class + anatomical-label sites",
        "derived summary; unresolved",
    ),
    AuditFeature(
        "n_site_compartments",
        "site compartments",
        "tumour phenotype",
        "disease-site canonical class",
        "count",
        "distinct current canonical anatomical classes",
        "derived summary; unresolved",
    ),
    AuditFeature(
        "n_active_clones",
        "active clones",
        "molecular mode",
        "snapshot num_active_clones",
        "count",
        "current clone identifiers with non-absent disease",
        "X versus C unresolved",
    ),
    AuditFeature(
        "n_acquired_alterations",
        "acquired alterations",
        "molecular mode",
        "snapshot n_acquired_alterations",
        "count",
        "current materialized acquired-alteration records",
        "prefer identity in C; count audited only",
    ),
    AuditFeature(
        "ecog_rank",
        "ECOG performance status",
        "host phenotype",
        "snapshot ecog_ps",
        "ordinal 0-4",
        "current performance status",
        "core candidate",
    ),
    AuditFeature(
        "symptom_rank",
        "overall symptom burden",
        "host phenotype",
        "snapshot symptom_burden",
        "ordinal 0-3",
        "current overall symptoms irrespective of cause",
        "core candidate",
    ),
    AuditFeature(
        "n_dominant_symptoms",
        "named dominant symptoms",
        "host phenotype",
        "snapshot dominant_symptoms",
        "multilabel count",
        "number of named current dominant symptoms",
        "representation unresolved",
    ),
    AuditFeature(
        "weight_kg_absolute",
        "absolute weight",
        "host phenotype",
        "snapshot weight_kg",
        "continuous kg",
        "current body weight",
        "exclude from PCA; engineering input",
    ),
    AuditFeature(
        "weight_change_pct",
        "weight change from baseline",
        "host phenotype",
        "current and baseline weight",
        "continuous percent",
        "within-person change from first materialized snapshot",
        "core candidate; scaling unresolved",
    ),
    *tuple(
        AuditFeature(
            f"organ_{organ}_rank",
            f"{organ.replace('_', ' ')} function",
            "host phenotype",
            f"snapshot organ_{organ}",
            "ordinal 0-4",
            f"current {organ.replace('_', ' ')} impairment irrespective of cause",
            "core candidate",
        )
        for organ in ORGAN_FIELDS
    ),
    AuditFeature(
        "active_condition_count",
        "active adverse-condition count",
        "adverse condition",
        "materialized snapshot active_aes",
        "count",
        "number of currently active adverse conditions",
        "physical overlap unresolved",
    ),
    AuditFeature(
        "active_condition_grade_max",
        "maximum active-condition grade",
        "adverse condition",
        "materialized snapshot active_aes",
        "ordinal 0-5",
        "highest CTCAE grade among current adverse conditions",
        "physical overlap unresolved",
    ),
    AuditFeature(
        "active_condition_grade_sum",
        "summed active-condition grades",
        "adverse condition",
        "materialized snapshot active_aes",
        "derived count-weighted severity",
        "sum of CTCAE grades across current adverse conditions",
        "derived and overlap unresolved",
    ),
)


def _accepted(frame: pd.DataFrame) -> pd.DataFrame:
    if "accepted" not in frame:
        return frame.copy()
    keep = frame["accepted"].astype(str).str.lower().isin({"true", "1", "1.0"})
    return frame.loc[keep].copy()


def _with_keys(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = frame.copy()
    out["_run"] = run_id
    out["_gid"] = run_id + "::" + out["trajectory_id"].astype(str)
    return out


def load_reference_tables(
    run_ids: Sequence[str] = REFERENCE_RUNS,
) -> dict[str, pd.DataFrame]:
    """Load accepted six-model derivation tables with collision-safe keys."""
    tables: dict[str, list[pd.DataFrame]] = {
        "snapshots": [],
        "sites": [],
        "endpoints": [],
    }
    for run_id in run_ids:
        derivation = ROOT / "runs" / run_id / "derivation"
        if not derivation.exists():
            raise FileNotFoundError(derivation)
        for key, filename in (
            ("snapshots", "snapshot_rows.csv"),
            ("sites", "disease_site_rows.csv"),
            ("endpoints", "endpoint_rows.csv"),
        ):
            path = derivation / filename
            if not path.exists():
                raise FileNotFoundError(path)
            frame = _with_keys(_accepted(pd.read_csv(path, low_memory=False)), run_id)
            tables[key].append(frame)
    return {
        key: pd.concat(frames, ignore_index=True)
        for key, frames in tables.items()
    }


def _ordinal(series: pd.Series, mapping: Mapping[str, float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = series.astype(str).str.strip().str.lower().map(mapping)
    return numeric.where(numeric.notna(), labels).astype(float)


def _dominant_symptom_count(series: pd.Series) -> pd.Series:
    def count(value: object) -> float:
        if pd.isna(value) or not str(value).strip():
            return 0.0
        return float(
            len(
                [
                    item
                    for item in str(value).replace("|", ";").split(";")
                    if item.strip()
                ]
            )
        )

    return series.map(count).astype(float)


def symptom_vocabulary_audit(series: pd.Series) -> pd.DataFrame:
    """Count raw and minimally normalized named-symptom labels.

    This normalization only resolves punctuation, underscores, case and repeated
    whitespace. It deliberately does not assert that clinical synonyms such as
    "dyspnoea" and "shortness of breath" are the same construct.
    """
    rows: list[dict[str, str]] = []
    for value in series.dropna().astype(str):
        for item in value.replace("|", ";").split(";"):
            raw = item.strip()
            if not raw:
                continue
            normalized = " ".join(
                raw.lower().replace("_", " ").replace("-", " ").split()
            )
            rows.append({"raw_label": raw, "minimally_normalized_label": normalized})
    if not rows:
        return pd.DataFrame(
            columns=["raw_label", "minimally_normalized_label", "n_mentions"]
        )
    return (
        pd.DataFrame(rows)
        .value_counts(["raw_label", "minimally_normalized_label"])
        .rename("n_mentions")
        .reset_index()
        .sort_values(["n_mentions", "raw_label"], ascending=[False, True])
        .reset_index(drop=True)
    )


def classify_symptom_system(label: object) -> str:
    """Map one named symptom to a provisional, auditable physical system."""
    normalized = " ".join(
        str(label).lower().replace("_", " ").replace("-", " ").split()
    )
    for system, pattern in SYMPTOM_SYSTEM_PATTERNS:
        # Most alternatives are clinical stems (for example ``abdom`` and
        # ``dyspn``), so retain the leading word boundary but deliberately do
        # not require the stem to end the word.
        if re.search(pattern.removesuffix(r"\b"), normalized):
            return system
    return "unmapped"


def add_symptom_system_candidates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add provisional per-system symptom presence and global-rank intensity.

    The intensity candidate multiplies binary system presence by the one global
    symptom-burden rank. It is reported only as a sensitivity representation:
    the source does not contain a separate structured severity for each named
    symptom.
    """
    out = frame.copy()
    presence_columns = tuple(f"symptom_system__{system}" for system in SYMPTOM_SYSTEMS)
    for column in presence_columns:
        out[column] = 0.0

    for index, value in out["dominant_symptoms"].items():
        if pd.isna(value) or not str(value).strip():
            continue
        systems = {
            classify_symptom_system(item)
            for item in str(value).replace("|", ";").split(";")
            if item.strip()
        }
        for system in systems:
            out.at[index, f"symptom_system__{system}"] = 1.0

    intensity_columns: list[str] = []
    for system in SYMPTOM_SYSTEMS:
        presence = f"symptom_system__{system}"
        intensity = f"symptom_intensity__{system}"
        out[intensity] = out[presence] * out["symptom_rank"]
        intensity_columns.append(intensity)
    return out, (*presence_columns, *intensity_columns)


def symptom_mapping_audit(
    vocabulary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach provisional systems to raw labels and report mapping coverage."""
    out = vocabulary.copy()
    out["provisional_system"] = out["minimally_normalized_label"].map(
        classify_symptom_system
    )
    mentions = out.groupby("provisional_system", as_index=False).agg(
        n_mentions=("n_mentions", "sum"),
        n_raw_labels=("raw_label", "nunique"),
        n_normalized_labels=("minimally_normalized_label", "nunique"),
    )
    total_mentions = int(out["n_mentions"].sum())
    unmapped_mentions = int(
        out.loc[out["provisional_system"].eq("unmapped"), "n_mentions"].sum()
    )
    meta = {
        "n_symptom_mentions": total_mentions,
        "n_unmapped_symptom_mentions": unmapped_mentions,
        "mapped_symptom_mention_pct": (
            100.0 * (total_mentions - unmapped_mentions) / total_mentions
            if total_mentions
            else np.nan
        ),
    }
    return out, {**meta, "by_system": mentions.to_dict(orient="records")}


def derive_site_features(sites: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Derive bounded per-class burden plus three competing extent definitions."""
    required = {
        "_gid",
        "snapshot_index",
        "site_class_canonical",
        "site_signature",
        "burden_band",
    }
    missing = required.difference(sites.columns)
    if missing:
        raise ValueError(f"disease-site table missing columns: {sorted(missing)}")

    active = sites.copy()
    active["site_class_canonical"] = (
        active["site_class_canonical"].fillna("other").astype(str)
    )
    active["site_burden_rank"] = _ordinal(active["burden_band"], BURDEN_RANK)
    keys = ["_gid", "snapshot_index"]

    counts = (
        active.groupby(keys, as_index=False)
        .agg(
            n_unique_anatomic_sites=("site_signature", "nunique"),
            n_site_compartments=("site_class_canonical", "nunique"),
        )
        .copy()
    )
    profile = active.pivot_table(
        index=keys,
        columns="site_class_canonical",
        values="site_burden_rank",
        aggfunc="max",
        fill_value=0.0,
    )
    profile.columns = [f"site__{column}" for column in profile.columns]
    site_columns = tuple(sorted(profile.columns))
    profile = profile.reset_index()
    return counts.merge(profile, on=keys, how="outer"), site_columns


def summarize_active_conditions(snapshot: Mapping[str, Any]) -> dict[str, float]:
    """Summarize active treatment-attributed adverse events in one snapshot."""
    aes = snapshot.get("active_aes") or []
    if not isinstance(aes, list):
        aes = []
    grades: list[float] = []
    by_category = {category: 0.0 for category in AE_CATEGORIES}
    for record in aes:
        if not isinstance(record, Mapping):
            continue
        grade = pd.to_numeric(record.get("ctcae_v5_grade"), errors="coerce")
        if pd.isna(grade):
            continue
        grade_float = float(grade)
        grades.append(grade_float)
        category = str(record.get("ae_category") or "other_specified")
        if category not in by_category:
            category = "other_specified"
        by_category[category] = max(by_category[category], grade_float)
    out = {
        "active_condition_count": float(len(grades)),
        "active_condition_grade_max": max(grades, default=0.0),
        "active_condition_grade_sum": float(sum(grades)),
    }
    out.update(
        {
            f"active_condition__{category}": grade
            for category, grade in by_category.items()
        }
    )
    return out


def _accepted_attempt_index(result: Mapping[str, Any]) -> int | None:
    for attempt in result.get("attempts") or []:
        if isinstance(attempt, Mapping) and bool(attempt.get("accepted")):
            value = attempt.get("attempt_index")
            return int(value) if value is not None else None
    return None


def load_active_condition_features(
    run_ids: Sequence[str] = REFERENCE_RUNS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read grade-resolved active conditions from accepted materialized snapshots."""
    rows: list[dict[str, Any]] = []
    accepted_trajectories = 0
    missing_materialized = 0
    for run_id in run_ids:
        trajectories = ROOT / "runs" / run_id / "trajectories"
        for result_path in sorted(trajectories.glob("*/trajectory_result.json")):
            result = json.loads(result_path.read_text())
            if not bool(result.get("accepted")):
                continue
            accepted_trajectories += 1
            attempt_index = _accepted_attempt_index(result)
            if attempt_index is None:
                missing_materialized += 1
                continue
            materialized_path = (
                result_path.parent
                / f"attempt_{attempt_index:02d}"
                / "materialized_trajectory.json"
            )
            if not materialized_path.exists():
                missing_materialized += 1
                continue
            materialized = json.loads(materialized_path.read_text())
            trajectory_id = str(result.get("trajectory_dir") or result_path.parent.name)
            for snapshot_index, snapshot in enumerate(
                materialized.get("snapshots") or []
            ):
                if not isinstance(snapshot, Mapping):
                    continue
                row: dict[str, Any] = {
                    "_gid": f"{run_id}::{trajectory_id}",
                    "snapshot_index": snapshot_index,
                }
                row.update(summarize_active_conditions(snapshot))
                rows.append(row)
    return pd.DataFrame(rows), {
        "accepted_trajectories": accepted_trajectories,
        "missing_materialized_trajectories": missing_materialized,
    }


def derive_state_candidates(
    snapshots: pd.DataFrame,
    sites: pd.DataFrame,
    active_conditions: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build the auditable current-state candidate matrix without scaling."""
    out = snapshots.copy()
    terminal = out["vital_status"].astype(str).str.lower().eq("deceased")
    terminal |= out["outcome_type"].astype(str).str.lower().eq("death")
    out = out.loc[~terminal].copy()
    out["snapshot_index"] = pd.to_numeric(
        out["snapshot_index"], errors="raise"
    ).astype(int)

    out["tumour_burden_max_rank"] = _ordinal(
        out["total_burden_band"], BURDEN_RANK
    )
    out["n_clonal_site_entries"] = pd.to_numeric(
        out["num_active_disease_sites"], errors="coerce"
    )
    out["n_active_clones"] = pd.to_numeric(
        out["num_active_clones"], errors="coerce"
    )
    out["n_acquired_alterations"] = pd.to_numeric(
        out["n_acquired_alterations"], errors="coerce"
    )
    out["ecog_rank"] = pd.to_numeric(out["ecog_ps"], errors="coerce")
    out["symptom_rank"] = _ordinal(out["symptom_burden"], SYMPTOM_RANK)
    out["n_dominant_symptoms"] = _dominant_symptom_count(
        out["dominant_symptoms"]
    )
    out, _symptom_system_columns = add_symptom_system_candidates(out)
    out["weight_kg_absolute"] = pd.to_numeric(out["weight_kg"], errors="coerce")
    for organ in ORGAN_FIELDS:
        out[f"organ_{organ}_rank"] = _ordinal(
            out[f"organ_{organ}"], ORGAN_RANK
        )

    ordered = out.sort_values(["_gid", "snapshot_index"])
    baseline_weight = (
        ordered.groupby("_gid", sort=False)["weight_kg_absolute"]
        .first()
        .rename("baseline_weight_kg")
    )
    out = out.join(baseline_weight, on="_gid")
    valid_baseline = out["baseline_weight_kg"].gt(0)
    out["weight_change_pct"] = np.where(
        valid_baseline,
        100.0
        * (out["weight_kg_absolute"] - out["baseline_weight_kg"])
        / out["baseline_weight_kg"],
        np.nan,
    )

    site_frame, site_columns = derive_site_features(sites)
    out = out.merge(
        site_frame,
        on=["_gid", "snapshot_index"],
        how="left",
        validate="one_to_one",
    )
    site_and_count_columns = [
        "n_unique_anatomic_sites",
        "n_site_compartments",
        *site_columns,
    ]
    out[site_and_count_columns] = out[site_and_count_columns].fillna(0.0)

    if not active_conditions.empty:
        out = out.merge(
            active_conditions,
            on=["_gid", "snapshot_index"],
            how="left",
            validate="one_to_one",
        )
    condition_columns = [
        "active_condition_count",
        "active_condition_grade_max",
        "active_condition_grade_sum",
        *(f"active_condition__{category}" for category in AE_CATEGORIES),
    ]
    for column in condition_columns:
        if column not in out:
            out[column] = np.nan
    return out, site_columns


def audit_features(
    frame: pd.DataFrame,
    features: Sequence[AuditFeature],
) -> pd.DataFrame:
    """Compute pooled and equal-model measurement properties."""
    ordered = frame.sort_values(["_gid", "snapshot_index"]).copy()
    rows: list[dict[str, Any]] = []
    for spec in features:
        values = pd.to_numeric(ordered[spec.feature_id], errors="coerce")
        valid = values.notna()
        current = values
        previous = ordered.groupby("_gid", sort=False)[spec.feature_id].shift()
        paired = current.notna() & previous.notna()
        changed = paired & ~np.isclose(current, previous)
        onset = paired & np.isclose(previous, 0.0) & current.gt(0.0)
        resolution = paired & previous.gt(0.0) & np.isclose(current, 0.0)

        model_stats: list[dict[str, float]] = []
        for _model, group in ordered.assign(_value=values).groupby(
            "model_short", sort=False
        ):
            series = group["_value"].dropna()
            if series.empty:
                continue
            group_previous = group.groupby("_gid", sort=False)["_value"].shift()
            group_paired = group["_value"].notna() & group_previous.notna()
            group_changed = group_paired & ~np.isclose(
                group["_value"], group_previous
            )
            model_stats.append(
                {
                    "zero_pct": 100.0 * float(np.isclose(series, 0.0).mean()),
                    "median": float(series.median()),
                    "change_pct": (
                        100.0 * float(group_changed[group_paired].mean())
                        if group_paired.any()
                        else np.nan
                    ),
                }
            )
        by_model = pd.DataFrame(model_stats)
        observed = values[valid]
        rows.append(
            {
                **asdict(spec),
                "n_snapshots": int(valid.sum()),
                "missing_pct": 100.0 * float((~valid).mean()),
                "zero_pct_pooled": (
                    100.0 * float(np.isclose(observed, 0.0).mean())
                    if len(observed)
                    else np.nan
                ),
                "zero_pct_equal_model": (
                    float(by_model["zero_pct"].mean())
                    if not by_model.empty
                    else np.nan
                ),
                "n_distinct": int(observed.nunique()),
                "minimum": float(observed.min()) if len(observed) else np.nan,
                "p10": float(observed.quantile(0.10)) if len(observed) else np.nan,
                "median": float(observed.median()) if len(observed) else np.nan,
                "p90": float(observed.quantile(0.90)) if len(observed) else np.nan,
                "maximum": float(observed.max()) if len(observed) else np.nan,
                "change_pct_pooled": (
                    100.0 * float(changed[paired].mean())
                    if paired.any()
                    else np.nan
                ),
                "change_pct_equal_model": (
                    float(by_model["change_pct"].mean())
                    if not by_model.empty
                    else np.nan
                ),
                "onset_count": int(onset.sum()),
                "resolution_count": int(resolution.sum()),
                "resolution_per_onset": (
                    float(resolution.sum() / onset.sum()) if onset.sum() else np.nan
                ),
                "model_median_min": (
                    float(by_model["median"].min()) if not by_model.empty else np.nan
                ),
                "model_median_max": (
                    float(by_model["median"].max()) if not by_model.empty else np.nan
                ),
                "model_zero_pct_min": (
                    float(by_model["zero_pct"].min()) if not by_model.empty else np.nan
                ),
                "model_zero_pct_max": (
                    float(by_model["zero_pct"].max()) if not by_model.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def redundancy_audit(
    frame: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    threshold: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the full Spearman matrix and high-redundancy feature pairs."""
    numeric = frame[list(feature_ids)].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(method="spearman", min_periods=50)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(feature_ids):
        for right in feature_ids[:i]:
            rho = correlation.loc[left, right]
            if pd.isna(rho) or abs(float(rho)) < threshold:
                continue
            paired = numeric[[left, right]].dropna()
            equal_pct = (
                100.0 * float(np.isclose(paired[left], paired[right]).mean())
                if len(paired)
                else np.nan
            )
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "spearman_rho": float(rho),
                    "n_paired": int(len(paired)),
                    "exactly_equal_pct": equal_pct,
                }
            )
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["spearman_rho", "exactly_equal_pct"],
            key=lambda series: series.abs(),
            ascending=False,
        ).reset_index(drop=True)
    return correlation, pairs


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> float:
    valid = numerator.notna() & denominator.notna()
    if not valid.any():
        return float("nan")
    return 100.0 * float(np.isclose(numerator[valid], denominator[valid]).mean())


def consistency_audit(
    state: pd.DataFrame,
    site_columns: Sequence[str],
    *,
    ae_coverage: Mapping[str, int],
    endpoints: pd.DataFrame,
) -> dict[str, Any]:
    """Quantify known semantic aliases and source coverage."""
    max_site = (
        state[list(site_columns)].max(axis=1)
        if site_columns
        else pd.Series(np.nan, index=state.index)
    )
    occupied_compartments = (
        state[list(site_columns)].gt(0).sum(axis=1)
        if site_columns
        else pd.Series(np.nan, index=state.index)
    )
    condition_columns = [
        f"active_condition__{category}" for category in AE_CATEGORIES
    ]
    max_condition_grade = state[condition_columns].max(axis=1)
    occupied_condition_categories = state[condition_columns].gt(0).sum(axis=1)
    endpoint_censored = (
        endpoints["censored"].astype(str).str.lower().isin({"true", "1", "1.0"})
    )
    snapshot_ae_count = pd.to_numeric(state["n_active_aes"], errors="coerce")
    detailed_ae_count = pd.to_numeric(
        state["active_condition_count"], errors="coerce"
    )
    return {
        "n_living_snapshots": int(len(state)),
        "n_trajectories": int(state["_gid"].nunique()),
        "n_models": int(state["model_short"].nunique()),
        "n_tumour_codes": int(state["tumor_oncotree_code"].nunique()),
        "n_canonical_site_classes": int(len(site_columns)),
        "max_site_burden_matches_replay_max_pct": _safe_pct(
            state["tumour_burden_max_rank"], max_site
        ),
        "site_compartment_count_matches_profile_pct": _safe_pct(
            state["n_site_compartments"], occupied_compartments
        ),
        "snapshot_ae_count_matches_materialized_pct": _safe_pct(
            snapshot_ae_count, detailed_ae_count
        ),
        "max_condition_grade_matches_category_profile_pct": _safe_pct(
            state["active_condition_grade_max"], max_condition_grade
        ),
        "active_condition_presence_matches_category_profile_pct": _safe_pct(
            state["active_condition_count"].gt(0).astype(float),
            occupied_condition_categories.gt(0).astype(float),
        ),
        "active_condition_count_matches_category_count_pct": _safe_pct(
            state["active_condition_count"], occupied_condition_categories
        ),
        "materialized_ae_snapshot_coverage_pct": (
            100.0 * float(detailed_ae_count.notna().mean())
        ),
        "accepted_materialized_trajectories": int(
            ae_coverage["accepted_trajectories"]
        ),
        "missing_materialized_trajectories": int(
            ae_coverage["missing_materialized_trajectories"]
        ),
        "n_censored_generated_endpoints": int(endpoint_censored.sum()),
        "n_generated_endpoints": int(len(endpoints)),
        "weight_change_min_pct": float(state["weight_change_pct"].min()),
        "weight_change_p001_pct": float(state["weight_change_pct"].quantile(0.001)),
        "weight_change_p999_pct": float(state["weight_change_pct"].quantile(0.999)),
        "weight_change_max_pct": float(state["weight_change_pct"].max()),
        "weight_change_abs_gt30_pct": 100.0
        * float(state["weight_change_pct"].abs().gt(30).mean()),
    }


def symptom_system_deep_audit(
    state: pd.DataFrame,
    audit: pd.DataFrame,
    mapping_meta: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize recording frequency and overlap of symptom-system candidates."""
    indexed = audit.set_index("feature_id")
    mention_rows = {
        str(row["provisional_system"]): row
        for row in mapping_meta["by_system"]
    }
    has_named_symptom = (
        state["dominant_symptoms"].fillna("").astype(str).str.strip().ne("")
    )
    rows: list[dict[str, Any]] = []
    for system in SYMPTOM_SYSTEMS:
        feature_id = f"symptom_system__{system}"
        row = indexed.loc[feature_id]
        mention = mention_rows.get(system, {})
        conditional = state.loc[has_named_symptom, ["model_short", feature_id]]
        conditional_by_model = conditional.groupby("model_short")[feature_id].mean()
        rows.append(
            {
                "system": system,
                "n_mentions": int(mention.get("n_mentions", 0)),
                "n_normalized_labels": int(
                    mention.get("n_normalized_labels", 0)
                ),
                "snapshot_prevalence_equal_model_pct": (
                    100.0 - float(row["zero_pct_equal_model"])
                ),
                "snapshot_prevalence_model_min_pct": (
                    100.0 - float(row["model_zero_pct_max"])
                ),
                "snapshot_prevalence_model_max_pct": (
                    100.0 - float(row["model_zero_pct_min"])
                ),
                "prevalence_among_named_equal_model_pct": (
                    100.0 * float(conditional_by_model.mean())
                ),
                "prevalence_among_named_model_min_pct": (
                    100.0 * float(conditional_by_model.min())
                ),
                "prevalence_among_named_model_max_pct": (
                    100.0 * float(conditional_by_model.max())
                ),
                "change_pct_equal_model": float(row["change_pct_equal_model"]),
                "rho_overall_symptom_rank": float(
                    state[[feature_id, "symptom_rank"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "rho_ecog": float(
                    state[[feature_id, "ecog_rank"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "structured_system_intensity_available": False,
                "derived_intensity_definition": (
                    "system_presence * global_symptom_rank"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "snapshot_prevalence_equal_model_pct", ascending=False
    )


def symptom_recording_deep_audit(
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Distinguish a named symptom's absence from lack of structured naming."""
    frame = state[["model_short", "symptom_rank", "dominant_symptoms"]].copy()
    frame["globally_symptomatic"] = frame["symptom_rank"].gt(0)
    frame["has_named_symptom"] = (
        frame["dominant_symptoms"].fillna("").astype(str).str.strip().ne("")
    )
    frame["symptomatic_without_names"] = (
        frame["globally_symptomatic"] & ~frame["has_named_symptom"]
    )
    frame["names_despite_zero_global_rank"] = (
        ~frame["globally_symptomatic"] & frame["has_named_symptom"]
    )

    rows: list[dict[str, Any]] = []
    for model, group in frame.groupby("model_short", sort=True):
        symptomatic = group["globally_symptomatic"]
        rows.append(
            {
                "model_short": model,
                "n_snapshots": int(len(group)),
                "globally_symptomatic_pct": 100.0 * float(symptomatic.mean()),
                "has_named_symptom_pct": 100.0
                * float(group["has_named_symptom"].mean()),
                "symptomatic_without_names_pct": 100.0
                * float(group["symptomatic_without_names"].mean()),
                "named_coverage_among_symptomatic_pct": (
                    100.0
                    * float(
                        group.loc[symptomatic, "has_named_symptom"].mean()
                    )
                ),
                "names_despite_zero_global_rank_pct": 100.0
                * float(group["names_despite_zero_global_rank"].mean()),
            }
        )
    by_model = pd.DataFrame(rows)
    symptomatic = frame["globally_symptomatic"]
    meta = {
        "globally_symptomatic_pct_pooled": 100.0 * float(symptomatic.mean()),
        "has_named_symptom_pct_pooled": 100.0
        * float(frame["has_named_symptom"].mean()),
        "symptomatic_without_names_pct_pooled": 100.0
        * float(frame["symptomatic_without_names"].mean()),
        "named_coverage_among_symptomatic_pct_pooled": 100.0
        * float(frame.loc[symptomatic, "has_named_symptom"].mean()),
        "named_coverage_among_symptomatic_pct_equal_model": float(
            by_model["named_coverage_among_symptomatic_pct"].mean()
        ),
        "named_coverage_among_symptomatic_model_min_pct": float(
            by_model["named_coverage_among_symptomatic_pct"].min()
        ),
        "named_coverage_among_symptomatic_model_max_pct": float(
            by_model["named_coverage_among_symptomatic_pct"].max()
        ),
        "interpretation": (
            "an empty dominant_symptoms field is not evidence that each symptom "
            "system is absent"
        ),
    }
    return by_model, meta


def site_multiplicity_deep_audit(
    sites: pd.DataFrame,
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit whether anatomical-label multiplicity is a trustworthy coordinate."""
    living_keys = state[["_gid", "snapshot_index"]].drop_duplicates()
    active = sites.merge(
        living_keys,
        on=["_gid", "snapshot_index"],
        how="inner",
        validate="many_to_one",
    ).copy()
    active["site_class_canonical"] = (
        active["site_class_canonical"].fillna("other").astype(str)
    )
    keys = ["_gid", "snapshot_index", "site_class_canonical"]
    per_snapshot_class = (
        active.groupby(keys, as_index=False)
        .agg(
            n_unique_labels=("site_signature", "nunique"),
            n_clone_site_entries=("site_signature", "size"),
            anatomical_labels=(
                "site_anatomic_label",
                lambda values: "; ".join(sorted({str(value) for value in values})),
            ),
        )
        .copy()
    )
    by_class = (
        per_snapshot_class.groupby("site_class_canonical", as_index=False)
        .agg(
            n_involved_snapshots=("n_unique_labels", "size"),
            mean_unique_labels=("n_unique_labels", "mean"),
            p90_unique_labels=(
                "n_unique_labels",
                lambda values: float(pd.Series(values).quantile(0.90)),
            ),
            max_unique_labels=("n_unique_labels", "max"),
            multiplicity_gt1_pct=(
                "n_unique_labels",
                lambda values: 100.0 * float(pd.Series(values).gt(1).mean()),
            ),
        )
        .copy()
    )
    clone_duplication = (
        per_snapshot_class.assign(
            duplicated_by_clone=lambda frame: frame["n_clone_site_entries"].gt(
                frame["n_unique_labels"]
            )
        )
        .groupby("site_class_canonical")["duplicated_by_clone"]
        .mean()
        .mul(100.0)
    )
    by_class["clone_duplication_pct"] = by_class["site_class_canonical"].map(
        clone_duplication
    )
    examples = per_snapshot_class.sort_values(
        ["n_unique_labels", "n_clone_site_entries"],
        ascending=False,
    ).head(100)
    meta = {
        "involved_snapshot_classes": int(len(per_snapshot_class)),
        "involved_snapshot_classes_with_multiple_labels_pct": 100.0
        * float(per_snapshot_class["n_unique_labels"].gt(1).mean()),
        "involved_snapshot_classes_with_clone_duplication_pct": 100.0
        * float(
            per_snapshot_class["n_clone_site_entries"]
            .gt(per_snapshot_class["n_unique_labels"])
            .mean()
        ),
        "living_snapshots_with_more_sites_than_compartments_pct": 100.0
        * float(
            state["n_unique_anatomic_sites"]
            .gt(state["n_site_compartments"])
            .mean()
        ),
        "site_identity_contract": (
            "canonical site class + free-text anatomical label; not a persistent "
            "radiologic lesion identifier"
        ),
    }
    return (
        by_class.sort_values("multiplicity_gt1_pct", ascending=False),
        examples,
        meta,
    )


def weight_deep_audit(
    state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit longitudinal weight and several transparent bounded candidates."""
    frame = state.copy()
    ratio = frame["weight_kg_absolute"] / frame["baseline_weight_kg"]
    transforms = {
        "raw_percent_change": frame["weight_change_pct"],
        "log_weight_ratio_x100": 100.0 * np.log(ratio.where(ratio.gt(0))),
        "clinically_clipped_percent_m30_p20": frame["weight_change_pct"].clip(
            -30.0, 20.0
        ),
        "smooth_tanh_percent_scale20": np.tanh(
            frame["weight_change_pct"] / 20.0
        ),
    }
    summaries: list[dict[str, Any]] = []
    for transform, values in transforms.items():
        for model, index in frame.groupby("model_short", sort=False).groups.items():
            series = pd.to_numeric(values.loc[index], errors="coerce").dropna()
            summaries.append(
                {
                    "transform": transform,
                    "model_short": model,
                    "n": int(len(series)),
                    "minimum": float(series.min()),
                    "p01": float(series.quantile(0.01)),
                    "p10": float(series.quantile(0.10)),
                    "median": float(series.median()),
                    "p90": float(series.quantile(0.90)),
                    "p99": float(series.quantile(0.99)),
                    "maximum": float(series.max()),
                    "mean": float(series.mean()),
                    "sd": float(series.std()),
                }
            )

    ordered = frame.sort_values(["_gid", "snapshot_index"]).copy()
    ordered["previous_weight_kg"] = ordered.groupby("_gid")[
        "weight_kg_absolute"
    ].shift()
    ordered["previous_day"] = ordered.groupby("_gid")["days_from_diagnosis"].shift()
    ordered["weight_jump_pct"] = (
        100.0
        * (ordered["weight_kg_absolute"] - ordered["previous_weight_kg"])
        / ordered["previous_weight_kg"]
    )
    ordered["jump_interval_days"] = (
        ordered["days_from_diagnosis"] - ordered["previous_day"]
    )
    ordered["weight_jump_pct_per_30d"] = np.where(
        ordered["jump_interval_days"].gt(0),
        ordered["weight_jump_pct"] * 30.4 / ordered["jump_interval_days"],
        np.nan,
    )
    jumps = ordered[ordered["previous_weight_kg"].notna()].copy()
    outlier_columns = [
        "_run",
        "_gid",
        "snapshot_index",
        "model_short",
        "tumor_oncotree_code",
        "days_from_diagnosis",
        "previous_day",
        "baseline_weight_kg",
        "previous_weight_kg",
        "weight_kg_absolute",
        "weight_change_pct",
        "weight_jump_pct",
        "jump_interval_days",
        "weight_jump_pct_per_30d",
    ]
    outliers = (
        jumps.assign(_abs_jump=jumps["weight_jump_pct"].abs())
        .sort_values("_abs_jump", ascending=False)
        .head(100)[outlier_columns]
    )
    meta = {
        "n_weight_intervals": int(len(jumps)),
        "n_weight_changes": int(jumps["weight_jump_pct"].ne(0).sum()),
        "same_day_nonzero_weight_changes": int(
            (
                jumps["jump_interval_days"].eq(0)
                & jumps["weight_jump_pct"].ne(0)
            ).sum()
        ),
        "weight_jump_abs_p99_pct": float(
            jumps["weight_jump_pct"].abs().quantile(0.99)
        ),
        "weight_jump_abs_max_pct": float(jumps["weight_jump_pct"].abs().max()),
        "weight_jump_abs_gt10_pct": 100.0
        * float(jumps["weight_jump_pct"].abs().gt(10).mean()),
        "recommended_for_next_bakeoff": (
            "compare prespecified clinical clipping (-30%, +20%) with a smooth "
            "tanh transform; reject unbounded raw percentage"
        ),
    }
    return pd.DataFrame(summaries), jumps[outlier_columns], outliers, meta


def active_condition_system_deep_audit(
    state: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Report sparsity and overlap for the grade-resolved adverse-event profile."""
    indexed = audit.set_index("feature_id")
    organ_match = {
        "renal": "organ_renal_rank",
        "hepatic": "organ_hepatic_rank",
        "cardiac": "organ_cardiac_rank",
        "pulmonary": "organ_pulmonary_rank",
        "hematologic": "organ_marrow_reserve_rank",
    }
    rows: list[dict[str, Any]] = []
    for category in AE_CATEGORIES:
        feature_id = f"active_condition__{category}"
        row = indexed.loc[feature_id]
        matched_organ = organ_match.get(category)
        rows.append(
            {
                "system": category,
                "source_definition": (
                    "maximum CTCAE grade among active treatment-attributed adverse "
                    "events in this system"
                ),
                "snapshot_prevalence_equal_model_pct": (
                    100.0 - float(row["zero_pct_equal_model"])
                ),
                "snapshot_prevalence_model_min_pct": (
                    100.0 - float(row["model_zero_pct_max"])
                ),
                "snapshot_prevalence_model_max_pct": (
                    100.0 - float(row["model_zero_pct_min"])
                ),
                "change_pct_equal_model": float(row["change_pct_equal_model"]),
                "onset_count": int(row["onset_count"]),
                "resolution_count": int(row["resolution_count"]),
                "rho_ecog": float(
                    state[[feature_id, "ecog_rank"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "rho_overall_symptom_rank": float(
                    state[[feature_id, "symptom_rank"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "matched_organ_feature": matched_organ or "",
                "rho_matched_organ": (
                    float(
                        state[[feature_id, matched_organ]]
                        .corr(method="spearman")
                        .iloc[0, 1]
                    )
                    if matched_organ
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "snapshot_prevalence_equal_model_pct", ascending=False
    )


def reviewer_feature_decision_table(
    audit: pd.DataFrame,
    *,
    n_snapshots: int,
    n_site_classes: int,
    symptom_recording_meta: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the clinician-review table and record the frozen map decision."""
    indexed = audit.set_index("feature_id")

    def characteristics(feature_id: str) -> str:
        row = indexed.loc[feature_id]
        return (
            f"n={int(row['n_snapshots']):,}; missing {row['missing_pct']:.1f}%; "
            f"zero/normal {row['zero_pct_equal_model']:.1f}%; "
            f"median {row['median']:.1f} [p10 {row['p10']:.1f}, "
            f"p90 {row['p90']:.1f}]; changes at "
            f"{row['change_pct_equal_model']:.1f}% of transitions"
        )

    rows: list[dict[str, str]] = []

    def add(
        candidate_id: str,
        block: str,
        role: str,
        exact_measurement: str,
        dataset_characteristics: str,
        proposed_encoding: str,
        paper_v1_default: str,
        rationale: str,
        reviewer_choice: str,
    ) -> None:
        rows.append(
            {
                "candidate_id": candidate_id,
                "block": block,
                "role": role,
                "exact_measurement": exact_measurement,
                "dataset_characteristics": dataset_characteristics,
                "proposed_encoding": proposed_encoding,
                "paper_v1_default": paper_v1_default,
                "rationale_or_risk": rationale,
                "reviewer_choice": reviewer_choice,
            }
        )

    add(
        "anatomical_burden_profile",
        "tumour anatomy",
        "X",
        "Maximum active burden rank within each of the current canonical "
        "anatomical classes.",
        f"{n_site_classes} coordinates across {n_snapshots:,} living snapshots; "
        "site prevalence ranges from <1% to about 40%; at least one active site "
        "is absent in 6.7% of snapshots.",
        "Nine prespecified anatomical groups, each encoded as maximum burden "
        "rank / 5; generic primary-site records are resolved from baseline context.",
        "INCLUDE in nine grouped coordinates",
        "This is the direct current tumour-distribution measurement. Rare "
        "coordinates must not be inflated by empirical robust scaling.",
        "FROZEN for paper-v1",
    )
    add(
        "maximum_site_burden",
        "tumour summary",
        "X-derived",
        "Maximum burden band across every active anatomical site; it is not "
        "total tumour volume.",
        characteristics("tumour_burden_max_rank")
        + "; exactly derivable as the maximum of the anatomical profile.",
        "Ordinal 0-5 only if the anatomical profile is not retained.",
        "EXCLUDE when profile is included",
        "It is a deterministic summary and would duplicate the most severe "
        "anatomical coordinate.",
        "Accept exclusion / opt in as a simpler alternative map",
    )
    add(
        "occupied_site_compartments",
        "tumour summary",
        "X-derived",
        "Number of non-zero canonical anatomical burden coordinates.",
        characteristics("n_site_compartments")
        + "; exactly derivable from the anatomical profile.",
        "Unique resolved original anatomical classes, capped at 5 and divided by 5.",
        "INCLUDE as disease extent",
        "The nine-coordinate grouping otherwise loses how many distinct original "
        "physical classes are involved.",
        "FROZEN for paper-v1",
    )
    add(
        "label_based_site_multiplicity",
        "tumour extent",
        "X candidate",
        "Number of distinct canonical-class plus free-text anatomical labels, "
        "not persistent radiological lesions.",
        characteristics("n_unique_anatomic_sites")
        + "; exceeds occupied classes in 18.8% of living snapshots.",
        "Capped integer count, if retained.",
        "OPTIONAL sensitivity; default EXCLUDE",
        "It may capture within-compartment multiplicity, but identity depends on "
        "free-text labels and is not lesion-stable.",
        "Opt in / opt out / require a lesion-identity contract",
    )
    add(
        "clone_site_entry_count",
        "tumour extent",
        "X-derived",
        "Number of active clone-site pairs; historically labelled number of "
        "active disease sites.",
        characteristics("n_clonal_site_entries"),
        "None proposed for the primary map.",
        "EXCLUDE",
        "Mixes clonal multiplicity with anatomy and substantially duplicates the "
        "site profile.",
        "Accept exclusion / opt in only as molecular-complexity sensitivity",
    )
    add(
        "ecog_performance_status",
        "host function",
        "X",
        "Current ECOG performance status emitted on the patient state.",
        characteristics("ecog_rank"),
        "Ordinal 0-4 divided by 4; retain ordinal interpretation.",
        "INCLUDE",
        "Clinically interpretable global functional state and portable to real "
        "cohorts.",
        "Accept / exclude",
    )
    add(
        "global_symptom_burden",
        "experienced state",
        "X candidate",
        "Independent LLM-emitted category none/mild/moderate/severe. It is not "
        "calculated from named symptoms and has no supplied scoring rubric.",
        characteristics("symptom_rank")
        + "; Spearman rho with ECOG is 0.787.",
        "Ordinal 0-3 divided by 3.",
        "INCLUDE",
        "Symptoms matter clinically, but this generated score is non-standardized, "
        "coarse and partly duplicates ECOG.",
        "FROZEN for paper-v1; disclose non-standardized rubric",
    )
    add(
        "named_symptom_profile",
        "experienced state",
        "X candidate",
        "Free-text dominant symptom labels, named only when considered "
        "trajectory-significant.",
        "1,670 raw labels; only "
        f"{symptom_recording_meta['named_coverage_among_symptomatic_pct_equal_model']:.1f}% "
        "equal-model coverage among globally symptomatic snapshots.",
        "No valid zero-filled encoding under the current observation contract.",
        "EXCLUDE",
        "An empty field is not true absence; one-hot or system encoding would "
        "mostly measure model-specific recording behaviour.",
        "Accept exclusion / require a standardized symptom instrument",
    )
    for organ, clinical_label in (
        ("renal", "Renal"),
        ("hepatic", "Hepatic"),
        ("cardiac", "Cardiac"),
        ("marrow_reserve", "Marrow reserve"),
        ("pulmonary", "Pulmonary"),
    ):
        add(
            f"organ_{organ}",
            "host organ function",
            "X",
            f"{clinical_label} function: normal, mild, moderate or severe "
            "impairment, or failure.",
            characteristics(f"organ_{organ}_rank"),
            "Ordinal 0-4 divided by 4; zero means normal.",
            "INCLUDE",
            "Cause-blind current physiology. Rare impairment is clinically "
            "meaningful and must not be variance-inflated.",
            "Accept / exclude / merge with another host coordinate",
        )
    add(
        "relative_weight",
        "host constitution",
        "X candidate",
        "Current weight relative to the first materialized patient-state weight.",
        characteristics("weight_change_pct")
        + "; observed range -44.4% to +79.2%; zero-day jumps exist.",
        "Weight loss = 100*(baseline-current)/baseline; clip to [-20%, +30%], "
        "then map to [0, 1] as (loss+20)/50.",
        "INCLUDE",
        "Within-person change captures cachexia/recovery, but generated extremes "
        "and event-time artefacts require a bounded, versioned transform.",
        "FROZEN for paper-v1",
    )
    add(
        "maximum_active_ae_grade",
        "global adverse burden",
        "X candidate",
        "Worst CTCAE grade among all currently active treatment-attributed "
        "adverse events; zero when none is active.",
        characteristics("active_condition_grade_max"),
        "Ordinal 0-5 divided by 5.",
        "INCLUDE as AE intensity",
        "CTCAE grade is more standardized than the symptom score and keeps "
        "toxicity visible without adding 13 system dimensions; it remains "
        "cause-attributed rather than purely physiological.",
        "FROZEN for paper-v1",
    )
    add(
        "active_ae_count",
        "global adverse burden",
        "X candidate",
        "Number of currently active treatment-attributed adverse events.",
        characteristics("active_condition_count"),
        "Count capped at 5 and divided by 5.",
        "INCLUDE as AE multiplicity",
        "Captures multiplicity missed by worst grade; the grade sum remains "
        "excluded because it mixes intensity and multiplicity.",
        "FROZEN for paper-v1",
    )
    add(
        "summed_active_ae_grades",
        "global adverse burden",
        "X-derived",
        "Sum of CTCAE grades across all currently active adverse events.",
        characteristics("active_condition_grade_sum")
        + "; rho with active AE count is 0.974.",
        "None proposed.",
        "EXCLUDE",
        "A derived mixture of severity and multiplicity with almost no separation "
        "from event count.",
        "Accept exclusion / opt in instead of the other global AE measures",
    )
    add(
        "system_specific_active_ae_profile",
        "adverse-event systems",
        "X candidate",
        "Maximum active CTCAE grade separately for 13 treatment-attributed "
        "adverse-event categories.",
        "Complete by replay, but equal-model prevalence ranges from about 1% to "
        "21% and model-specific ranges are broad.",
        "Thirteen ordinal 0-5 coordinates with block balancing, if used.",
        "OPTIONAL Extended Data; default EXCLUDE",
        "Physically richer but high-dimensional, cause-attributed and partly "
        "overlapping with organ function.",
        "Opt in / opt out / retain selected systems",
    )
    add(
        "active_clone_count",
        "molecular complexity",
        "X or C candidate",
        "Number of active generated tumour clone identifiers.",
        characteristics("n_active_clones"),
        "Capped count only in a sensitivity analysis.",
        "OPTIONAL sensitivity; default treat clone identity as C",
        "Not routinely observed and less meaningful than the identity of a "
        "resistance mechanism or transformation.",
        "Place in X / place in C / exclude",
    )
    add(
        "acquired_alteration_count",
        "molecular mode",
        "C",
        "Number of currently materialized acquired alterations.",
        characteristics("n_acquired_alterations"),
        "Do not encode the count in PCA; retain specific alteration identities "
        "and their onset times in context C(t).",
        "EXCLUDE from X",
        "Resistance identity changes the dynamics; a count is a lossy, mostly "
        "irreversible history proxy.",
        "Accept as context / opt into X only with explicit justification",
    )
    add(
        "absolute_weight",
        "host constitution",
        "engineering input",
        "Current absolute weight in kilograms.",
        characteristics("weight_kg_absolute"),
        "Use only to derive within-person relative weight.",
        "EXCLUDE from PCA",
        "Would encode body size and sex-related morphology rather than change in "
        "the current disease course.",
        "Accept exclusion / opt in",
    )
    return pd.DataFrame(rows)


def reviewer_anatomical_coordinate_table(
    audit: pd.DataFrame,
    site_columns: Sequence[str],
) -> pd.DataFrame:
    """Expose every anatomical coordinate so a reviewer can regroup or exclude it."""
    site = audit.set_index("feature_id").loc[list(site_columns)].reset_index()
    return pd.DataFrame(
        {
            "feature_id": site["feature_id"],
            "anatomical_class": site["feature_id"]
            .str.removeprefix("site__")
            .str.replace("_", " "),
            "encoding": "maximum current burden rank 0-5; zero absent",
            "prevalence_equal_model_pct": 100.0 - site["zero_pct_equal_model"],
            "prevalence_model_min_pct": 100.0 - site["model_zero_pct_max"],
            "prevalence_model_max_pct": 100.0 - site["model_zero_pct_min"],
            "change_pct_equal_model": site["change_pct_equal_model"],
            "maximum_observed_rank": site["maximum"],
            "paper_v1_default": (
                "map into one of nine frozen groups; fixed-range scale"
            ),
            "reviewer_choice": "group assignment recorded in final map contract",
        }
    ).sort_values("prevalence_equal_model_pct", ascending=False)


def write_reviewer_feature_table(
    table: pd.DataFrame,
    anatomy: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write a portable Markdown companion for external clinical review."""

    def safe(value: object) -> str:
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "# Paper-v1 state features — clinician/reviewer decision table",
        "",
        "> This audit preceded the frozen paper-v1 decision. The final 10+10 "
        "contract is implemented by `build_paper_state_map.py`.",
        "",
        "## Frozen compact default",
        "",
        "- **Include:** nine grouped anatomical tumour-burden coordinates plus "
        "five-level capped disease extent; ECOG; global symptom burden; five "
        "organ-function coordinates; bounded relative weight; maximum active-AE "
        "grade and five-level capped active-AE count.",
        "- **Default exclude but retain as explicit sensitivity choices:** "
        "label-based site multiplicity, system-specific AEs and active clone count.",
        "- **Exclude:** named symptoms under the current missingness contract, "
        "maximum site burden, AE grade sum, absolute weight and "
        "acquired-alteration count.",
        "",
        "The global symptom score is emitted directly by the generator. It is not "
        "calculated from named symptoms, and the operative prompt supplies no "
        "validated scoring rubric beyond `none/mild/moderate/severe`.",
        "",
        "## Decision table",
        "",
        "| candidate | block | role | exact measurement | current dataset | "
        "proposed encoding | paper-v1 default | rationale/risk | reviewer choice |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {safe(row.candidate_id)} | {safe(row.block)} | {safe(row.role)} | "
            f"{safe(row.exact_measurement)} | {safe(row.dataset_characteristics)} | "
            f"{safe(row.proposed_encoding)} | {safe(row.paper_v1_default)} | "
            f"{safe(row.rationale_or_risk)} | {safe(row.reviewer_choice)} |"
        )

    lines += [
        "",
        "## Audited anatomical coordinates and final grouping record",
        "",
        "| anatomical class | equal-model prevalence % | model range % | "
        "changes % | maximum rank | reviewer choice |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in anatomy.itertuples(index=False):
        lines.append(
            f"| {safe(row.anatomical_class)} | "
            f"{row.prevalence_equal_model_pct:.1f} | "
            f"{row.prevalence_model_min_pct:.1f}–"
            f"{row.prevalence_model_max_pct:.1f} | "
            f"{row.change_pct_equal_model:.1f} | "
            f"{row.maximum_observed_rank:.0f} | {safe(row.reviewer_choice)} |"
        )

    lines += [
        "",
        "## Frozen scaling rule",
        "",
        "Use fixed clinical ranges rather than empirical robust scaling. The "
        "primary 10+10 map is mean-centred without empirical z-scoring or block "
        "balancing. An equal-block map is retained only as sensitivity analysis. "
        "The display map is fitted without outcomes, treatments or future information.",
    ]
    (out_dir / "reviewer_feature_decision_table.md").write_text(
        "\n".join(lines) + "\n"
    )


def _short_feature_label(feature_id: str) -> str:
    if feature_id.startswith("site__"):
        return feature_id.removeprefix("site__").replace("_", " ")
    labels = {feature.feature_id: feature.label for feature in BASE_AUDIT_FEATURES}
    return labels.get(feature_id, feature_id.replace("_", " "))


def draw_audit_figure(
    audit: pd.DataFrame,
    correlation: pd.DataFrame,
    site_columns: Sequence[str],
    out_dir: Path,
) -> None:
    """Draw a compact clinician-review dashboard; this is not a paper result."""
    main_ids = [
        feature.feature_id
        for feature in BASE_AUDIT_FEATURES
        if feature.feature_id
        not in {"weight_kg_absolute", "n_acquired_alterations"}
    ]
    main = audit.set_index("feature_id").loc[main_ids].copy()
    site = (
        audit[audit["feature_id"].isin(site_columns)]
        .sort_values("zero_pct_equal_model")
        .head(14)
        .copy()
    )

    fig = plt.figure(figsize=(17.0, 14.0), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.05),
        hspace=0.43,
        wspace=0.28,
        left=0.08,
        right=0.96,
        bottom=0.15,
        top=0.91,
    )

    ax = fig.add_subplot(grid[0, 0])
    ordered = main.sort_values("zero_pct_equal_model", ascending=False)
    y = np.arange(len(ordered))
    prevalence = 100.0 - ordered["zero_pct_equal_model"].to_numpy(float)
    colors = [DOMAIN_COLOR.get(domain, MUTED) for domain in ordered["domain"]]
    ax.barh(y, prevalence, color=colors, alpha=0.9)
    ax.set_yticks(y, [_short_feature_label(value) for value in ordered.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("snapshots with a non-zero value, equal-model mean (%)")
    ax.set_title("A · Is the proposed measurement usually present?")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[0, 1])
    ordered = main.sort_values("change_pct_equal_model", ascending=True)
    y = np.arange(len(ordered))
    colors = [DOMAIN_COLOR.get(domain, MUTED) for domain in ordered["domain"]]
    ax.barh(
        y,
        ordered["change_pct_equal_model"].to_numpy(float),
        color=colors,
        alpha=0.9,
    )
    ax.set_yticks(y, [_short_feature_label(value) for value in ordered.index], fontsize=8)
    ax.set_xlabel("consecutive snapshot pairs in which value changes (%)")
    ax.set_title("B · Does it actually move within trajectories?")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 0])
    site = site.sort_values("zero_pct_equal_model", ascending=False)
    y = np.arange(len(site))
    ax.barh(
        y,
        100.0 - site["zero_pct_equal_model"].to_numpy(float),
        color=DOMAIN_COLOR["tumour phenotype"],
        alpha=0.88,
    )
    ax.set_yticks(
        y,
        [_short_feature_label(value) for value in site["feature_id"]],
        fontsize=8,
    )
    upper = max(
        50.0,
        5.0 * np.ceil((100.0 - site["zero_pct_equal_model"]).max() / 5.0),
    )
    ax.set_xlim(0, upper)
    ax.set_xlabel("snapshots with current involvement, equal-model mean (%)")
    ax.set_title(
        "C · Anatomical distribution means a burden vector\n"
        "(maximum active burden within each canonical site class)"
    )
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 1])
    corr_ids = [
        feature_id
        for feature_id in main_ids
        if feature_id in correlation.columns
    ]
    corr = correlation.loc[corr_ids, corr_ids].to_numpy(float)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    labels = [_short_feature_label(value) for value in corr_ids]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_title("D · Redundancy audit (Spearman correlation)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("ρ")

    for axis in fig.axes[:-1]:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Pre-PCA clinical-state audit — measurements, dynamics and redundancy",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.025,
        "Diagnostic only · no PCA fitted · no outcome, treatment or future information used",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(out_dir / "fig_state_feature_audit.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "fig_state_feature_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_deep_audit_figure(
    symptom_systems: pd.DataFrame,
    site_multiplicity: pd.DataFrame,
    active_conditions: pd.DataFrame,
    state: pd.DataFrame,
    mapping_meta: Mapping[str, Any],
    symptom_recording_meta: Mapping[str, Any],
    out_dir: Path,
) -> None:
    """Draw the detailed candidate-engineering audit."""
    fig = plt.figure(figsize=(17.0, 13.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        hspace=0.38,
        wspace=0.32,
        left=0.09,
        right=0.96,
        bottom=0.10,
        top=0.91,
    )

    ax = fig.add_subplot(grid[0, 0])
    symptoms = symptom_systems.sort_values(
        "snapshot_prevalence_equal_model_pct", ascending=True
    )
    y = np.arange(len(symptoms))
    center = symptoms["snapshot_prevalence_equal_model_pct"].to_numpy(float)
    lower = symptoms["snapshot_prevalence_model_min_pct"].to_numpy(float)
    upper = symptoms["snapshot_prevalence_model_max_pct"].to_numpy(float)
    ax.barh(y, center, color="#2b8c87", alpha=0.9)
    ax.errorbar(
        center,
        y,
        xerr=np.vstack([center - lower, upper - center]),
        fmt="none",
        ecolor=INK,
        elinewidth=1.0,
        capsize=2,
    )
    ax.set_yticks(
        y,
        [value.replace("_", " ") for value in symptoms["system"]],
        fontsize=8,
    )
    ax.set_xlabel("snapshots naming this symptom system (%)")
    ax.set_title(
        "A · Provisional symptom systems\n"
        f"{mapping_meta['mapped_symptom_mention_pct']:.1f}% of mentions mapped; "
        f"names in only "
        f"{symptom_recording_meta['named_coverage_among_symptomatic_pct_equal_model']:.1f}% "
        "of symptomatic snapshots"
    )
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[0, 1])
    sites = (
        site_multiplicity[site_multiplicity["n_involved_snapshots"].ge(100)]
        .sort_values("multiplicity_gt1_pct", ascending=True)
        .tail(16)
    )
    y = np.arange(len(sites))
    ax.barh(
        y,
        sites["multiplicity_gt1_pct"],
        color="#7a5195",
        alpha=0.9,
    )
    ax.set_yticks(
        y,
        [value.replace("_", " ") for value in sites["site_class_canonical"]],
        fontsize=8,
    )
    ax.set_xlabel("involved snapshots with >1 distinct anatomical label (%)")
    ax.set_title(
        "B · Multiplicity is label-based, not lesion-based\n"
        "canonical class + free-text anatomical label"
    )
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 0])
    conditions = active_conditions.sort_values(
        "snapshot_prevalence_equal_model_pct", ascending=True
    )
    y = np.arange(len(conditions))
    center = conditions["snapshot_prevalence_equal_model_pct"].to_numpy(float)
    lower = conditions["snapshot_prevalence_model_min_pct"].to_numpy(float)
    upper = conditions["snapshot_prevalence_model_max_pct"].to_numpy(float)
    ax.barh(y, center, color="#df8829", alpha=0.9)
    ax.errorbar(
        center,
        y,
        xerr=np.vstack([center - lower, upper - center]),
        fmt="none",
        ecolor=INK,
        elinewidth=1.0,
        capsize=2,
    )
    ax.set_yticks(
        y,
        [value.replace("_", " ") for value in conditions["system"]],
        fontsize=8,
    )
    ax.set_xlabel("snapshots with an active adverse condition (%)")
    ax.set_title(
        "C · Active adverse-condition profile\n"
        "maximum current CTCAE grade per recorded toxicity system"
    )
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 1])
    models = sorted(state["model_short"].unique())
    values = [
        state.loc[state["model_short"].eq(model), "weight_change_pct"]
        .dropna()
        .to_numpy(float)
        for model in models
    ]
    ax.boxplot(
        values,
        vert=False,
        tick_labels=models,
        showfliers=False,
        whis=(1, 99),
        patch_artist=True,
        boxprops={"facecolor": "#6ba6c8", "edgecolor": INK},
        medianprops={"color": "#a63d40", "linewidth": 1.5},
        whiskerprops={"color": INK},
        capprops={"color": INK},
    )
    ax.axvline(0.0, color=MUTED, lw=1.0)
    ax.axvline(-30.0, color="#a63d40", ls="--", lw=1.0)
    ax.axvline(20.0, color="#a63d40", ls="--", lw=1.0)
    ax.set_xlabel("current weight change from personal baseline (%)")
    ax.set_title(
        "D · Weight is strongly model-dependent\n"
        "boxes = IQR, whiskers = 1st–99th percentile; dashed = candidate bounds"
    )
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    for axis in fig.axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Detailed pre-PCA engineering audit — symptoms, sites, adverse conditions, weight",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Audit candidates · final groupings and transforms are frozen in the 10+10 map",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(
        out_dir / "fig_state_feature_deep_audit.png",
        dpi=190,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "fig_state_feature_deep_audit.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _fmt(value: Any, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def write_readout(
    out_dir: Path,
    ledger: pd.DataFrame,
    audit: pd.DataFrame,
    redundancy: pd.DataFrame,
    consistency: Mapping[str, Any],
    site_columns: Sequence[str],
    symptom_vocabulary: pd.DataFrame,
    correlation: pd.DataFrame,
    symptom_systems: pd.DataFrame,
    symptom_mapping_meta: Mapping[str, Any],
    symptom_recording: pd.DataFrame,
    symptom_recording_meta: Mapping[str, Any],
    site_multiplicity: pd.DataFrame,
    site_multiplicity_meta: Mapping[str, Any],
    active_condition_systems: pd.DataFrame,
    weight_meta: Mapping[str, Any],
) -> None:
    """Write the clinician-facing ledger readout."""
    selected = audit.set_index("feature_id")
    lines = [
        "# Clinical-state feature ledger — pre-PCA gate",
        "",
        "> This document classifies and audits measurements and fits no PCA. The "
        "clinician-approved decision is recorded at the end; the executable "
        "10+10 contract and fitted transform are in `build_paper_state_map.py` "
        "and `outputs/paper_state_map/state_map_contract.json`.",
        "",
        "The proposed opt-in/opt-out table for external review is "
        "`reviewer_feature_decision_table.md`; all 32 anatomical coordinates are "
        "listed individually there.",
        "",
        "## Formal roles",
        "",
        "- **X(t)** is the current physical/experienced patient phenotype.",
        "- **C(t)** is static or piecewise-changing context (for example histologic "
        "transformation or acquired resistance).",
        "- **H(t)** is ideally the complete prior marked path. Sparse summaries are "
        "observation operators, not the definition of history.",
        "- **A(t)** is an interval-valued exposure; treatment cycles may be abstracted "
        "as continuous on/off treatment periods.",
        "- **E(t)** contains progression/line exit and absorbing death/EXIT.",
        "- **theta** is generator provenance and is never a patient coordinate.",
        "",
        "## Coverage",
        "",
        f"- {consistency['n_living_snapshots']:,} living snapshots from "
        f"{consistency['n_trajectories']:,} accepted trajectories, "
        f"{consistency['n_models']} models and "
        f"{consistency['n_tumour_codes']} tumour codes.",
        f"- {consistency['n_canonical_site_classes']} canonical anatomical site "
        "classes were observed.",
        "- Grade-resolved active-condition data cover "
        f"{_fmt(consistency['materialized_ae_snapshot_coverage_pct'])}% of living "
        "snapshots; snapshot counts agree with materialized active records in "
        f"{_fmt(consistency['snapshot_ae_count_matches_materialized_pct'])}%.",
        f"- Generated endpoints: {consistency['n_generated_endpoints']:,}; censored: "
        f"{consistency['n_censored_generated_endpoints']:,}.",
        "",
        "## Two corrected labels",
        "",
        "1. `total_burden_band` is the **maximum burden band across active sites**, "
        "not total tumour volume. It matches the maximum reconstructed anatomical "
        f"site burden in {_fmt(consistency['max_site_burden_matches_replay_max_pct'])}% "
        "of living snapshots.",
        "2. `num_active_disease_sites` counts **active clone–site entries**, not unique "
        "anatomical sites. The ledger therefore audits clone–site entries, unique "
        "anatomical sites and canonical compartments separately.",
        "3. The number of occupied site compartments is a deterministic count of the "
        "non-zero anatomical-profile coordinates "
        f"({_fmt(consistency['site_compartment_count_matches_profile_pct'])}% exact).",
        "",
        "## What anatomical distribution means",
        "",
        "For each snapshot, distribution is a vector with one element per canonical "
        "anatomical class (brain, liver, bone, marrow, lung, nodes, etc.). Each element "
        "is zero when absent and otherwise the maximum current ordinal burden in that "
        "class. CNS remains in X. This audit exposed all 32 classes before the "
        "clinician-approved map grouped them into nine anatomical coordinates.",
        "",
        "Consequently, if the full anatomical profile is retained, maximum burden and "
        "occupied-compartment count add no information. Unique anatomical-site count "
        "can still add multiplicity within one compartment.",
        "",
        "## Current-state measurement audit",
        "",
        "| measurement | domain | type | zero % | median [p10,p90] | changes % | "
        "onsets / resolutions | status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    display_ids = [feature.feature_id for feature in BASE_AUDIT_FEATURES]
    for feature_id in display_ids:
        row = selected.loc[feature_id]
        lines.append(
            f"| {row['label']} | {row['domain']} | {row['raw_type']} | "
            f"{_fmt(row['zero_pct_equal_model'])} | "
            f"{_fmt(row['median'])} [{_fmt(row['p10'])}, {_fmt(row['p90'])}] | "
            f"{_fmt(row['change_pct_equal_model'])} | "
            f"{int(row['onset_count'])} / {int(row['resolution_count'])} | "
            f"{row['candidate_status']} |"
        )

    site = (
        audit[audit["feature_id"].isin(site_columns)]
        .sort_values("zero_pct_equal_model")
        .head(16)
    )
    lines += [
        "",
        "## Most prevalent anatomical burden coordinates",
        "",
        "| site class | involved snapshots, equal-model % | changes % | onsets / "
        "resolutions |",
        "|---|---:|---:|---:|",
    ]
    for row in site.itertuples(index=False):
        lines.append(
            f"| {_short_feature_label(row.feature_id)} | "
            f"{_fmt(100.0 - row.zero_pct_equal_model)} | "
            f"{_fmt(row.change_pct_equal_model)} | "
            f"{row.onset_count} / {row.resolution_count} |"
        )

    same_system_pairs = (
        ("renal", "renal"),
        ("hepatic", "hepatic"),
        ("cardiac", "cardiac"),
        ("marrow_reserve", "hematologic"),
        ("pulmonary", "pulmonary"),
    )
    overlap_lines = []
    for organ, condition in same_system_pairs:
        left = f"organ_{organ}_rank"
        right = f"active_condition__{condition}"
        overlap_lines.append(
            f"- {organ.replace('_', ' ')} function versus current "
            f"{condition.replace('_', ' ')} adverse-condition grade: "
            f"rho={correlation.loc[left, right]:+.3f}."
        )
    n_raw_symptoms = int(symptom_vocabulary["raw_label"].nunique())
    n_normalized_symptoms = int(
        symptom_vocabulary["minimally_normalized_label"].nunique()
    )
    lines += [
        "",
        "## Symptoms and active adverse conditions",
        "",
        f"- Named symptoms contain {n_raw_symptoms:,} raw labels and "
        f"{n_normalized_symptoms:,} labels even after only punctuation/case "
        "normalization. They are not ready to become one-hot PCA coordinates without "
        "a clinical vocabulary.",
        "- The category-specific adverse-condition profile exactly reconstructs "
        "whether any active condition is present and its maximum grade "
        f"({_fmt(consistency['active_condition_presence_matches_category_profile_pct'])}% "
        "and "
        f"{_fmt(consistency['max_condition_grade_matches_category_profile_pct'])}%, "
        "respectively).",
        "- Category count equals condition count in only "
        f"{_fmt(consistency['active_condition_count_matches_category_count_pct'])}% "
        "because patients may have multiple simultaneous conditions in one system.",
        "- Same-system overlap with the organ-function fields is modest:",
        *overlap_lines,
        "",
        "This supports treating the active adverse-condition profile as potentially "
        "additional current state, while excluding its global maximum and carefully "
        "deciding whether multiplicity should receive one extra coordinate.",
        "",
        "### What system-specific active adverse condition means",
        "",
        "The source is not a generic disease registry. It is the treatment-attributed "
        "active adverse-event list stored in each materialized snapshot. For each of "
        "13 CTCAE systems, the candidate coordinate is the maximum grade among events "
        "active at that instant. For example, a grade-3 cardiac adverse event sets the "
        "cardiac-adverse-condition coordinate to 3 until grade change or resolution. "
        "It does **not** replace cardiac organ function, and it does not claim that "
        "every cardiac disorder is treatment caused.",
        "",
        "| adverse-condition system | snapshot prevalence % | model range % | "
        "changes % | rho with symptom rank | matched-organ rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in active_condition_systems.itertuples(index=False):
        matched = (
            _fmt(row.rho_matched_organ, 3)
            if np.isfinite(row.rho_matched_organ)
            else "—"
        )
        lines.append(
            f"| {row.system.replace('_', ' ')} | "
            f"{_fmt(row.snapshot_prevalence_equal_model_pct)} | "
            f"{_fmt(row.snapshot_prevalence_model_min_pct)}–"
            f"{_fmt(row.snapshot_prevalence_model_max_pct)} | "
            f"{_fmt(row.change_pct_equal_model)} | "
            f"{_fmt(row.rho_overall_symptom_rank, 3)} | {matched} |"
        )

    lines += [
        "",
        "### Provisional symptom-system audit",
        "",
        "Named symptoms provide location/type but no structured per-symptom severity. "
        "The only structured intensity is the single global symptom-burden rank. The "
        "deep audit therefore compares: (a) global rank alone; (b) provisional system "
        "presence plus global rank; and (c) a sensitivity encoding `system presence × "
        "global rank`. Option (c) is derived and must not be described as observed "
        "system-specific intensity.",
        "",
        f"The transparent keyword audit maps "
        f"{_fmt(symptom_mapping_meta['mapped_symptom_mention_pct'])}% of symptom "
        "mentions; unmapped labels remain explicit.",
        "",
        "**Absence is not reliably observed.** Across models, only "
        f"{_fmt(symptom_recording_meta['named_coverage_among_symptomatic_pct_equal_model'])}% "
        "of snapshots with non-zero overall symptom burden name any dominant symptom "
        "(model range "
        f"{_fmt(symptom_recording_meta['named_coverage_among_symptomatic_model_min_pct'])}–"
        f"{_fmt(symptom_recording_meta['named_coverage_among_symptomatic_model_max_pct'])}%; "
        f"pooled {_fmt(symptom_recording_meta['named_coverage_among_symptomatic_pct_pooled'])}%). "
        "An empty list therefore cannot be encoded as absence from every symptom "
        "system. The system profile is an incompletely observed candidate, not a "
        "ready PCA block.",
        "",
        "| model | symptomatic snapshots % | any named symptom % | named coverage "
        "among symptomatic % |",
        "|---|---:|---:|---:|",
    ]
    for row in symptom_recording.itertuples(index=False):
        lines.append(
            f"| {row.model_short} | {_fmt(row.globally_symptomatic_pct)} | "
            f"{_fmt(row.has_named_symptom_pct)} | "
            f"{_fmt(row.named_coverage_among_symptomatic_pct)} |"
        )

    lines += [
        "",
        "| provisional symptom system | mentions | snapshot prevalence % | "
        "model range % | among named snapshots % | changes % | "
        "rho with global rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in symptom_systems.itertuples(index=False):
        lines.append(
            f"| {row.system.replace('_', ' ')} | {row.n_mentions:,} | "
            f"{_fmt(row.snapshot_prevalence_equal_model_pct)} | "
            f"{_fmt(row.snapshot_prevalence_model_min_pct)}–"
            f"{_fmt(row.snapshot_prevalence_model_max_pct)} | "
            f"{_fmt(row.prevalence_among_named_equal_model_pct)} | "
            f"{_fmt(row.change_pct_equal_model)} | "
            f"{_fmt(row.rho_overall_symptom_rank, 3)} |"
        )

    lines += [
        "",
        "### Site multiplicity audit",
        "",
        "The sum of non-zero site-vector entries is the number of occupied anatomical "
        "classes. It cannot recover multiple labels within one class. However, the "
        "available identity is only `canonical class + free-text anatomical label`, "
        "not a persistent radiologic lesion identifier.",
        "",
        f"- {_fmt(site_multiplicity_meta['living_snapshots_with_more_sites_than_compartments_pct'])}% "
        "of living snapshots have more distinct anatomical labels than occupied classes.",
        f"- {_fmt(site_multiplicity_meta['involved_snapshot_classes_with_multiple_labels_pct'])}% "
        "of involved snapshot-class records contain more than one anatomical label.",
        f"- {_fmt(site_multiplicity_meta['involved_snapshot_classes_with_clone_duplication_pct'])}% "
        "contain the same site signature under multiple clones.",
        "",
        "| site class | involved snapshots | >1 label % | clone duplication % | "
        "maximum labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (
        site_multiplicity[site_multiplicity["n_involved_snapshots"].ge(100)]
        .sort_values("multiplicity_gt1_pct", ascending=False)
        .head(15)
        .itertuples(index=False)
    ):
        lines.append(
            f"| {row.site_class_canonical.replace('_', ' ')} | "
            f"{row.n_involved_snapshots:,} | "
            f"{_fmt(row.multiplicity_gt1_pct)} | "
            f"{_fmt(row.clone_duplication_pct)} | "
            f"{int(row.max_unique_labels)} |"
        )

    lines += [
        "",
        "## Relative-weight engineering warning",
        "",
        "Percentage change from the first materialized snapshot ranges from "
        f"{_fmt(consistency['weight_change_min_pct'])}% to "
        f"{_fmt(consistency['weight_change_max_pct'])}%; the 0.1st–99.9th percentile "
        f"range is {_fmt(consistency['weight_change_p001_pct'])}% to "
        f"{_fmt(consistency['weight_change_p999_pct'])}%. "
        f"{_fmt(consistency['weight_change_abs_gt30_pct'], 2)}% of living snapshots "
        "exceed ±30%. Clinical clipping or a bounded transform must be prespecified; "
        "raw percentage change should not enter PCA unchanged.",
        "",
        f"Across {weight_meta['n_weight_intervals']:,} consecutive intervals, "
        f"{weight_meta['n_weight_changes']:,} change weight. The 99th-percentile "
        f"absolute single-interval jump is "
        f"{_fmt(weight_meta['weight_jump_abs_p99_pct'])}% and the maximum is "
        f"{_fmt(weight_meta['weight_jump_abs_max_pct'])}%; "
        f"{weight_meta['same_day_nonzero_weight_changes']:,} non-zero changes occur "
        "on a zero-day interval. The next bake-off should reject unbounded percentage "
        "change and compare a clinically clipped signed percentage with a smooth "
        "bounded transform.",
    ]

    lines += [
        "",
        "## High empirical redundancy (|Spearman rho| >= 0.75)",
        "",
    ]
    if redundancy.empty:
        lines.append("- None at the prespecified threshold.")
    else:
        for row in redundancy.head(25).itertuples(index=False):
            lines.append(
                f"- `{row.feature_left}` versus `{row.feature_right}`: "
                f"rho={row.spearman_rho:+.3f}, exactly equal in "
                f"{row.exactly_equal_pct:.1f}% of paired snapshots."
            )

    lines += [
        "",
        "## Semantic ledger",
        "",
        "| object | role | time structure | map status | central issue |",
        "|---|---|---|---|---|",
    ]
    for row in ledger.itertuples(index=False):
        lines.append(
            f"| **{row.label}** | `{row.role}` | {row.time_structure} | "
            f"{row.map_status} | {row.known_issue} |"
        )

    lines += [
        "",
        "## Clinician-approved paper-v1 decision",
        "",
        "The final primary display uses nine grouped anatomical burden coordinates, "
        "five-level capped disease extent, ECOG, global symptoms, five separate "
        "organ functions, bounded relative weight loss, maximum active-AE grade and "
        "five-level capped active-AE count. It excludes named symptoms, maximum site "
        "burden, label-based multiplicity, AE grade sum, system-specific AE profiles "
        "and clone/alteration counts. `build_paper_state_map.py` is the executable "
        "contract and records the primary-site resolution rule.",
        "",
        "No outcome, treatment assignment, future event, PCA, UMAP or supervised "
        "criterion was used to produce this audit or choose among its empirical "
        "feature summaries.",
    ]
    (out_dir / "readout.md").write_text("\n".join(lines) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build(
    *,
    out_dir: Path = DEFAULT_OUT,
    run_ids: Sequence[str] = REFERENCE_RUNS,
) -> dict[str, Any]:
    """Build the complete pre-PCA ledger and return its manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_reference_tables(run_ids)
    active_conditions, ae_coverage = load_active_condition_features(run_ids)
    state, site_columns = derive_state_candidates(
        tables["snapshots"],
        tables["sites"],
        active_conditions,
    )

    site_audit_features = tuple(
        AuditFeature(
            column,
            _short_feature_label(column),
            "tumour phenotype",
            "disease-site canonical class + burden",
            "ordinal 0-5, zero when absent",
            "maximum current burden within this anatomical class",
            "anatomical profile; grouping unresolved",
        )
        for column in site_columns
    )
    condition_audit_features = tuple(
        AuditFeature(
            f"active_condition__{category}",
            f"{category.replace('_', ' ')} adverse-condition grade",
            "adverse condition",
            "materialized snapshot active_aes",
            "ordinal 0-5, zero when absent",
            "maximum current CTCAE grade in this system category",
            "physical overlap unresolved",
        )
        for category in AE_CATEGORIES
    )
    symptom_audit_features = tuple(
        AuditFeature(
            f"symptom_system__{system}",
            f"{system.replace('_', ' ')} symptom presence",
            "host phenotype",
            "dominant_symptoms provisional keyword mapping",
            "binary",
            "at least one named current symptom mapped to this system",
            "provisional ontology; not frozen",
        )
        for system in SYMPTOM_SYSTEMS
    ) + tuple(
        AuditFeature(
            f"symptom_intensity__{system}",
            f"{system.replace('_', ' ')} derived symptom intensity",
            "host phenotype",
            "system presence multiplied by global symptom rank",
            "derived ordinal 0-3",
            "sensitivity representation; not observed per-system intensity",
            "derived sensitivity only",
        )
        for system in SYMPTOM_SYSTEMS
    )
    audit_specs = (
        *BASE_AUDIT_FEATURES,
        *site_audit_features,
        *condition_audit_features,
        *symptom_audit_features,
    )
    audit = audit_features(state, audit_specs)

    redundancy_ids = [
        feature.feature_id
        for feature in audit_specs
        if feature.feature_id != "weight_kg_absolute"
        and not feature.feature_id.startswith("symptom_system__")
        and not feature.feature_id.startswith("symptom_intensity__")
    ]
    correlation, redundancy = redundancy_audit(state, redundancy_ids)
    consistency = consistency_audit(
        state,
        site_columns,
        ae_coverage=ae_coverage,
        endpoints=tables["endpoints"],
    )
    ledger = pd.DataFrame(asdict(entry) for entry in _entries())
    symptom_vocabulary = symptom_vocabulary_audit(
        tables["snapshots"]["dominant_symptoms"]
    )
    mapped_symptom_vocabulary, symptom_mapping_meta = symptom_mapping_audit(
        symptom_vocabulary
    )
    symptom_systems = symptom_system_deep_audit(
        state,
        audit,
        symptom_mapping_meta,
    )
    symptom_recording, symptom_recording_meta = symptom_recording_deep_audit(
        state
    )
    site_multiplicity, site_multiplicity_examples, site_multiplicity_meta = (
        site_multiplicity_deep_audit(tables["sites"], state)
    )
    weight_transforms, weight_intervals, weight_outliers, weight_meta = (
        weight_deep_audit(state)
    )
    active_condition_systems = active_condition_system_deep_audit(state, audit)
    reviewer_features = reviewer_feature_decision_table(
        audit,
        n_snapshots=len(state),
        n_site_classes=len(site_columns),
        symptom_recording_meta=symptom_recording_meta,
    )
    reviewer_anatomy = reviewer_anatomical_coordinate_table(
        audit,
        site_columns,
    )

    ledger.to_csv(out_dir / "classification_ledger.csv", index=False)
    audit.to_csv(out_dir / "state_feature_audit.csv", index=False)
    redundancy.to_csv(out_dir / "redundancy_pairs.csv", index=False)
    correlation.to_csv(out_dir / "spearman_correlation.csv")
    symptom_vocabulary.to_csv(out_dir / "dominant_symptom_vocabulary.csv", index=False)
    mapped_symptom_vocabulary.to_csv(
        out_dir / "dominant_symptom_provisional_system_map.csv",
        index=False,
    )
    symptom_systems.to_csv(out_dir / "symptom_system_audit.csv", index=False)
    symptom_recording.to_csv(
        out_dir / "symptom_recording_by_model.csv",
        index=False,
    )
    site_multiplicity.to_csv(
        out_dir / "site_multiplicity_by_class.csv",
        index=False,
    )
    site_multiplicity_examples.to_csv(
        out_dir / "site_multiplicity_examples.csv",
        index=False,
    )
    active_condition_systems.to_csv(
        out_dir / "active_adverse_condition_system_audit.csv",
        index=False,
    )
    weight_transforms.to_csv(
        out_dir / "weight_transform_audit.csv",
        index=False,
    )
    weight_outliers.to_csv(
        out_dir / "weight_outlier_examples.csv",
        index=False,
    )
    reviewer_features.to_csv(
        out_dir / "reviewer_feature_decision_table.csv",
        index=False,
    )
    reviewer_anatomy.to_csv(
        out_dir / "reviewer_anatomical_coordinate_table.csv",
        index=False,
    )
    write_reviewer_feature_table(
        reviewer_features,
        reviewer_anatomy,
        out_dir,
    )
    pd.DataFrame(
        {
            "site_feature_id": site_columns,
            "site_class": [column.removeprefix("site__") for column in site_columns],
        }
    ).to_csv(out_dir / "anatomical_site_classes.csv", index=False)
    draw_audit_figure(audit, correlation, site_columns, out_dir)
    draw_deep_audit_figure(
        symptom_systems,
        site_multiplicity,
        active_condition_systems,
        state,
        symptom_mapping_meta,
        symptom_recording_meta,
        out_dir,
    )
    write_readout(
        out_dir,
        ledger,
        audit,
        redundancy,
        consistency,
        site_columns,
        symptom_vocabulary,
        correlation,
        symptom_systems,
        symptom_mapping_meta,
        symptom_recording,
        symptom_recording_meta,
        site_multiplicity,
        site_multiplicity_meta,
        active_condition_systems,
        weight_meta,
    )

    source_hashes: dict[str, str] = {}
    for run_id in run_ids:
        for filename in (
            "snapshot_rows.csv",
            "disease_site_rows.csv",
            "endpoint_rows.csv",
        ):
            path = ROOT / "runs" / run_id / "derivation" / filename
            source_hashes[f"{run_id}/{filename}"] = _sha256(path)
    manifest = {
        "artifact": "paper_state_feature_ledger",
        "status": "pre_pca_semantic_and_measurement_audit",
        "git_commit": _git_commit(),
        "script_sha256": _sha256(Path(__file__)),
        "reference_runs": list(run_ids),
        "source_hashes": source_hashes,
        "counts": consistency,
        "n_ledger_entries": int(len(ledger)),
        "n_audited_features": int(len(audit)),
        "n_redundancy_pairs": int(len(redundancy)),
        "n_reviewer_feature_decisions": int(len(reviewer_features)),
        "n_raw_dominant_symptom_labels": int(
            symptom_vocabulary["raw_label"].nunique()
        ),
        "n_minimally_normalized_dominant_symptom_labels": int(
            symptom_vocabulary["minimally_normalized_label"].nunique()
        ),
        "symptom_mapping": {
            key: value
            for key, value in symptom_mapping_meta.items()
            if key != "by_system"
        },
        "symptom_recording": symptom_recording_meta,
        "site_multiplicity": site_multiplicity_meta,
        "weight_audit": weight_meta,
        "pca_fitted": False,
        "outcomes_used": False,
        "treatments_used": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(out_dir=args.out_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
