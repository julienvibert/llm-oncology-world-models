# OncoTree Curation Worksheet — Review Guide

**File:** [`curation_worksheet_v1.yaml`](curation_worksheet_v1.yaml) (897 nodes, 120 main_type groups, ~10k lines).

**Source roster:** [`roster_v1.yaml`](roster_v1.yaml), OncoTree release `oncotree_2025_10_03`.

**Scaffold:** the worksheet skeleton was generated from the roster by `scripts/scaffold_curation_worksheet.py` (not part of this release); the curation columns were then filled by hand.

---

## What this file is

A clinician curation worksheet: every OncoTree node in `roster_v1.yaml`, grouped by `main_type` and pre-filled with mechanical context (level, parent, leaf status, main_type). The actual clinical classification lives in three columns per node:

- **`tier`** ∈ {`primary`, `secondary`, `excluded`}. Filled by the curating oncologist.
- **`exclusion_reason`** — required when `tier == excluded`; one of the five categories:
  - `non_clinical_grouping_node`
  - `pediatric_only_unless_intentional`
  - `premalignant_or_in_situ_only_without_invasive_analogue`
  - `no_clinical_management_distinction_from_parent`
  - `outside_adult_oncology_scope`
- **`primary_non_leaf_rationale`** — required iff `tier == primary AND is_leaf == False`; null otherwise. Documents the clinical rationale for treating a non-leaf OncoTree node as a headline primary entity despite the default leaf-only rule. See the "Primary non-leaf exceptions" section below.
- **`secondary_role`** — required when `tier == secondary`; classifies how a secondary-tier label is used downstream. One of:
  - `targeted_stress` — definitely sampled in the stress / sensitivity panel. Distinct biology, actionable rare entity, or specifically informative for the rare-cancer story. Examples: malignant phyllodes (`MPT`), breast angiosarcoma (`BA`), NUT carcinoma of the lung (`NUTCL`).
  - `pool` — valid rare entity, eligible for stratified random sampling from the secondary panel. Not guaranteed to be sampled — drawn against a per-stratum budget. Examples: rare metaplastic breast variants, rare large-cell lung variants.
  - `rollup_only` — useful taxonomy umbrella; pooled-level claims roll up from leaf-tier results. **Not sampled directly.** Examples: `NSCLC`, `BRCA` (invasive breast carcinoma umbrella), `LNET` (lung neuroendocrine umbrella).
  - `catalog_only` — documented but not generated. Ultra-rare or clinically unstable labels where a synthetic trajectory would be redundant or unreliable. Kept for completeness.
  - **Generation-eligibility:** only `targeted_stress` and `pool` are sampled. `rollup_only` and `catalog_only` are kept for taxonomy/audit but never reach the runner.
- **`notes`** — free-text rationale. Optional but encouraged for non-obvious calls.

The mechanical pre-fills live in:

- **`suggested_tier`** — set only when the call is mechanical (level-1 tissue grouping nodes → `excluded`). Otherwise `null`.
- **`suggested_exclusion_reason`** — set in lock-step with `suggested_tier`.

These two are advisory; the curator may override them.

---

## Tier semantics

- **`primary`** — Clinically meaningful entities for headline atlas analysis. Suitable for cross-cancer comparisons, anchor-sensitivity experiments, and counterfactual probes.
  - **By default: leaf-only.** Non-leaf OncoTree nodes are typically aggregators (e.g. `NSCLC` is the parent of `LUAD`, `LUSC`, `LCLC`, ...); generating against an aggregator forces the model to pick a leaf anyway, and analytical roll-ups for pooled-level claims (e.g. "NSCLC overall") are computed from leaf-tier results, not by marking the aggregator as primary.
  - **Documented exception:** clinically canonical non-leaf entities (CCRCC, DLBCLNOS, FL, MCL, CML, HL, etc.) MAY be marked `primary` — but ONLY by filling `primary_non_leaf_rationale` with a non-empty string explaining why this non-leaf entity is the clinical headline despite having OncoTree children. The adapter rejects primary-on-non-leaf without a rationale (and rejects a rationale on any non-primary node). See the "Primary non-leaf exceptions" section below.
- **`secondary`** — Valid but rare, redundant, ultra-specific, or taxonomy-heavy labels. Retained in stress / sensitivity analyses only. Not in headline tables. **Permitted on any level** (leaf or non-leaf). **Stratified by `secondary_role`** into four roles (above).
- **`excluded`** — Grouping nodes, benign/premalignant-only without an invasive analogue, pediatric-only entities (unless intentionally included), labels outside the adult-oncology scope of the study, or labels with no distinct management signal from their parent.

---

## Primary non-leaf exceptions

The default `primary = leaf-only` rule keeps headline analyses methodologically clean: the OncoTree leaf is the diagnosis code the model is asked to generate from, and pooled-level claims roll up leaf-tier results.

A small number of OncoTree non-leaf nodes are clinically canonical headline entities whose OncoTree children are rare variants rather than co-equal subtypes. Examples:

| Code | Entity | Child(ren) | Why it's the headline |
|---|---|---|---|
| `CCRCC` | Renal Clear Cell Carcinoma | `SCCRCC` (sarcomatoid CCRCC) | ~70-80% of all RCC; the child is a rare aggressive variant. |
| `DLBCLNOS` | DLBCL, NOS | 2 rare variants | The clinical headline DLBCL; children are uncommon sub-categorizations. |
| `FL` | Follicular Lymphoma | grade-specific children | The clinical headline indolent NHL entity. |
| `MCL` | Mantle Cell Lymphoma | 1 child | Distinct B-cell lymphoma with single rare variant in OncoTree. |
| `CML` | Chronic Myelogenous Leukemia | 1 child | Core CML entity. |
| `HL` | Hodgkin Lymphoma | 2 children | Core HL entity. |

When such a node is marked `primary`, the curator also fills `primary_non_leaf_rationale` with a sentence explaining the exception. The adapter and tests enforce both directions — primary-on-non-leaf without rationale is rejected, rationale without primary-on-non-leaf is rejected.

Recommended rationale shape:

> "<Entity name> is the clinically canonical <main_type> headline entity (~X% of all <umbrella>). The OncoTree child(ren) (<list>) are rare variants, not co-equal subtypes. Marking <code> primary preserves the headline entity while children stay secondary/<role>."

---

## Workflow

1. Each node in `curation_worksheet_v1.yaml` was reviewed by `main_type` group (nodes sorted by `level`, then `code`).
2. For each node, `tier` was filled, with `exclusion_reason` when excluded, `secondary_role` when secondary, `primary_non_leaf_rationale` when primary on a non-leaf node, and `notes` where helpful.
3. `scripts/port_worksheet_to_overlay.py` ports the curated decisions into `roster_curation_v1.yaml` (the machine-read overlay), and `scripts/build_oncotree_roster.py` builds `roster_v1.yaml` from it.

---

## Downstream use

- The anchored atlas generates from `tier=primary` codes.
- Only secondary codes with `secondary_role` in {`targeted_stress`, `pool`} are eligible for generation; `rollup_only` and `catalog_only` are never sampled.
