# SynthOncCases Latent Trajectory Prompt v1 - Neutralized Framing Control (continuation contract)

{{TASK_OPENING}} The JSON is a model of the patient's physical system over
time, not a clinical chart. Emit the disease changes, clonal evolution,
host-state changes, systemic/local interventions, toxicities, and terminal
events needed to reconstruct the evolving patient state deterministically.

Use only oncology facts, disease biology, interventions, and care patterns that
could have occurred by the study knowledge cutoff: {{KNOWLEDGE_CUTOFF}}.
Patient events may occur after this date; the cutoff limits available clinical
knowledge and interventions, not the simulated calendar.

## Output Contract

Emit exactly one JSON object:

```json
{ "ledger": [ ... ] }
```

There is no `baseline_patch` in this contract. The patient facts for this
run are given under `## Baseline Seed` and the observed history under
`## Fixed Prefix`. Both are fixed. Emitting a baseline, or any object other
than `ledger`, is hard-rejected.

The seed is deliberately partial. Fields it does not mention were NOT OBSERVED
for this patient by the source; they are not normal, not absent, and not yours
to decide. Reason about the future under that uncertainty rather than resolving
it: where an unobserved factor would change the course, that shows as a wider
range of possible courses, which is the correct answer.

`ledger` is the chronological event list, and it contains ONLY events strictly
after the cutoff date given under `## Fixed Prefix`. Do not restate, revise or
re-date anything already in the prefix. End with exactly one `death` event.
The terminal death cause may be cancer-related or non-cancer-related according
to the emitted physical trajectory.

Do not emit markdown, comments, prose, full snapshots, event IDs, treatment IDs,
clone-registry rows for acquired clones, active treatment lists, active toxicity
lists, aggregate disease summaries, treatment line numbers, treatment
tolerance, PRO scores, reports, notes, assessment schedules, or routine stable
visits.

## Completeness Rule

Include every clinically meaningful state transition needed to reconstruct the
trajectory: meaningful responses, progressions/recurrences/new disease sites,
clonal evolution events, host-state changes, toxicity onsets and resolutions,
systemic treatment starts/switches/stops/modifications, local interventions, and
death.

## Generation Contract

The trajectory continues from the cutoff date in `## Fixed Prefix`, which is
the day the index treatment started. Everything before it already happened and
is given; your first event must be strictly after it.

For open clinical concepts, emit the raw/specific value first, then choose the
closest broad bucket from the schema enum. The injected Pydantic schemas below
are the authoritative source of the current enum values. Prose examples below
are not exhaustive enum lists; if an example and the schema ever diverge, the
schema wins.

Enum order is arbitrary. Do not infer prevalence, clinical preference, default
status, or likelihood from an enum member appearing earlier or later in the
schema. Choose the raw clinical state first from the seed, fixed prefix, and
clinical plausibility; classify it afterward.

Field modality:

- Raw/specific text fields preserve the model's native clinical description.
  Examples include `histology_class`, `primary_site`, `site`, `regimen.label`,
  `regimen.components`, `gene`, `alteration`, `variant`, `mechanism`,
  `comorbidities[].term`, `contextual_factors[].term`, `ae`,
  `dominant_symptoms[]`, `procedure`, `cause`, and `rationale`.
- Structured self-annotation fields make the same emission analyzable:
  `histology_bucket`, `primary_site_class`, `site_class`, `primary_driver_bucket`,
  `driver_bucket`, `mechanism_bucket`, `regimen.component_classes`,
  `actionability`, and
  `contextual_factors.category`.
- The bucket/actionability enums exist to make your raw clinical choices
  analyzable downstream. They are NOT menus directing those choices. Decide the
  clinically appropriate raw value first, then classify it into the closest
  structured value.
- Use `other` only when no schema bucket fits, not as a way to avoid
  commitment.

Required self-annotations are defined by the schemas and enforced at validation:
histology, diagnosis primary site, disease site, molecular
driver/actionability, clonal mechanism, and regimen component class must all be
filled when their parent object is emitted.

The requirement to self-annotate a molecular driver bucket is an analyzability
requirement, not evidence that the patient has an actionable alteration, a
targeted-therapy option, or a resistance-biased trajectory.

At baseline, `identity_card.molecular_profile.alterations[]` is the raw
gene-level molecular profile. When a canonical/actionable driver is present,
mark exactly one row as `is_primary_driver: true`; its `driver_bucket` must
match `identity_card.molecular_profile.primary_driver_bucket`.

No actionable/canonical driver is a valid molecular interpretation. In that
case, use `primary_driver_bucket: "no_actionable_driver"` and emit either
`alterations: []` or non-primary alterations such as TP53/passenger findings
with no `is_primary_driver: true` row. The founder clone's `drivers[]` may also
be empty. Do not invent molecular drivers.

`biomarkers` is also a required list. Biomarkers are clinically meaningful
baseline disease-state descriptors, not longitudinal laboratory measurements.
Include any that characterize the baseline disease biology, such as diagnosis
or subtype, receptor or molecular phenotype, immune context, lineage or tumor
phenotype, or any other relevant aspect; otherwise emit `[]`. Do not invent biomarkers
to fill the field.

For baseline and acquired molecular alterations, use explicit `variant: null`
only when no specific variant applies. For acquired clonal events, reuse the
same molecular core and add `mechanism` plus `mechanism_bucket`; `gene` may be
`null` only for true non-gene events such as histologic or lineage
transformation.

`contextual_factors` is a required baseline list, but it is not a biography.
Include only trajectory-significant contextual factors: factors that materially
affect the cancer trajectory, management, toxicity risk, care access, or
survival. If none are significant, emit `[]`.

Tobacco fields must be internally consistent: `smoking_status: "never"`
requires `pack_years: 0.0` and `tobacco_exposure_band: "none"`;
`smoking_status: "former"` or `smoking_status: "current"` requires
`pack_years > 0.0` and `tobacco_exposure_band` of `"light"`, `"moderate"`,
or `"heavy"`.

Baseline comorbidities belong in `medical_history.comorbidities`. Treatment
adverse events belong in `toxicity` events. Clinically important non-cancer
intercurrent illnesses or injuries that change patient state belong in
`patient_state_update` with free-text `state_change_cause`.

All JSON snippets below are shape examples only. Do not copy their dates,
regimen labels, components, clone IDs, sites, intents, or event ordering as
clinical advice. Choose enum values from the injected Pydantic schemas according
to the baseline seed, fixed prefix, and run-specific instructions for this run.

## Event Types

Allowed `type` values are defined in the Pydantic Event Schema below. There is
no generic `other` event type.

## Systemic Treatment Events

`systemic_treatment_start` and `systemic_treatment_switch` require `regimen`.

Use `systemic_treatment_start` only for the first systemic regimen, or after a prior systemic regimen has already been stopped.

If a patient is already on systemic therapy and the regimen changes to another regimen, emit `systemic_treatment_switch`, not a second `systemic_treatment_start`.

Use `systemic_treatment_stop` only when treatment stops and no replacement regimen begins on that same clinical transition.
Do not emit a `systemic_treatment_stop` immediately followed by a
`systemic_treatment_start` on the same or adjacent date. Use
`systemic_treatment_switch` instead. Reserve `systemic_treatment_stop` for
genuine treatment-free intervals, such as toxicity recovery, a planned break,
eligibility re-evaluation, or patient choice.

```json
{
  "date": "2024-02-01",
  "type": "systemic_treatment_start",
  "regimen": {
    "label": "agent_a_agent_b",
    "components": ["agent_a", "agent_b"],
    "component_classes": ["other_targeted", "cytotoxic_chemotherapy"]
  },
  "intent": "disease_control"
}
```

For combination regimens, `label` and `components[]` are the source of truth.
Include every clinically active systemic component in `components[]`, in the
same order as `component_classes[]`. `component_classes[]` is a coarse
per-component annotation; do not remove, rename, or simplify components to fit
a bucket.

Use schema enum values for treatment `intent`, switch/stop/modification
`reason`, and `modification_type`.

`systemic_treatment_modification` requires `modification_type`.

On `systemic_treatment_modification`, optional `affected_components[]` names
the regimen components changed by the modification. Omit it when the whole
regimen is modified or the affected component is not clinically specified.

Do not emit line number, duration, best response, or PFS.

## Disease Events

`response` and `progression` require non-empty `disease[]` deltas. They are
latent disease-state changes, not assessment labels.

Each `disease` item:

```json
{
  "clone": "C0",
  "site": "liver",
  "site_class": "liver",
  "burden": "small",
  "kinetic": "shrinking"
}
```

Only include changed clone-site states.

Use schema enum values for `burden`, `kinetic`, and `site_class`. Every
emitted `disease[]` entry must include `site_class`.

For disease site updates, `burden` and `kinetic` are separate axes. If disease has cleared at a site, use `burden: "absent"` with `kinetic: "cleared"`; do not put `"cleared"` or `"stable"` in `burden`.

Do not emit lesion measurements or aggregate burden/extent summaries.

## Clonal Evolution Events

Use `clonal_evolution_event` for acquired molecular resistance, new subclones,
histologic/lineage transformation, or another clone-creating biological event.
The event creates the new clone; do not emit a full acquired clone-registry row.

Required: `new_clone_id`, `gene`, `alteration`, `alteration_type`, `variant`,
`driver_bucket`, `actionability`, `mechanism`, `mechanism_bucket`, and
`sample_source`. Use the next clone ID: if baseline clones are `C0...C{N-1}`,
the first acquired clone is `C{N}`. Use `variant: null` when no specific
variant exists; do not omit the `variant` key.

```json
{
  "date": "2025-04-02",
  "recognized_date": null,
  "type": "clonal_evolution_event",
  "new_clone_id": "C1",
  "gene": "MET",
  "alteration": "MET amplification",
  "alteration_type": "amplification",
  "variant": null,
  "driver_bucket": "met",
  "actionability": "resistance_marker",
  "mechanism": "bypass signaling resistance to prior targeted therapy",
  "mechanism_bucket": "bypass_signaling",
  "sample_source": "liquid_biopsy_ctDNA",
  "parent_clone": "C0"
}
```

Use schema enum values for `alteration_type`, `driver_bucket`, `actionability`,
`mechanism_bucket`, and `sample_source`.

Use `histologic_transformation` for acquired histologic or lineage shifts
under therapy (small-cell transformation, neuroendocrine transformation,
Richter transformation, etc.). The `alteration` text carries the specific
transformation phenotype.

When a latent event is never clinically recognized, set `recognized_date` to
null. If recognized, use the date it became clinically known.

## Toxicity Events

Toxicity is stateful: emit onset, grade changes, and resolution as separate
events. The deterministic state engine reconstructs the active toxicity list
from this sequence.

Use schema enum values for `toxicity_transition` and `ae_category`.
`toxicity_transition` distinguishes new toxicity onset, CTCAE grade changes of
an already-active toxicity, and resolution of an active toxicity.

Onset requires `toxicity_transition`, `ae`, `ae_category`, and
`ctcae_v5_grade`.

```json
{
  "date": "2024-04-12",
  "type": "toxicity",
  "toxicity_transition": "onset",
  "ae": "peripheral_neuropathy",
  "ae_category": "neurologic",
  "ctcae_v5_grade": 2,
  "causes": ["agent_a"]
}
```

Grade change requires `toxicity_transition`, `ae`, and the new
`ctcae_v5_grade`. `ae_category` and `causes` are not required because the
deterministic state engine carries them from the matching onset.

```json
{
  "date": "2024-06-01",
  "type": "toxicity",
  "toxicity_transition": "grade_change",
  "ae": "peripheral_neuropathy",
  "ctcae_v5_grade": 3
}
```

Resolution:

```json
{
  "date": "2024-09-20",
  "type": "toxicity",
  "toxicity_transition": "resolved",
  "ae": "peripheral_neuropathy"
}
```

A toxicity that changes phenotype mid-course (e.g. peripheral neuropathy
evolving from pure sensory to mixed sensorimotor) is a new `onset` with a new
`ae`, not a `grade_change`. `grade_change` is reserved for the same `ae`
shifting CTCAE grade.

Always include grade 3-5 toxicities. Include grade 2 toxicities when they change
treatment, patient state, monitoring, or persist meaningfully. Include grade 1
only when clinically important to the trajectory. Emit `grade_change` for
clinically meaningful CTCAE-grade transitions of an already-active toxicity -
any change into or out of grade 3+, or any change that drives a dose
modification, organ-function shift, ECOG change, or contributes to death. Do
not emit `grade_change` for minor fluctuations that do not change management.

## Patient State

Patient deltas can appear on any event. Use current values, not deltas.

```json
{
  "date": "2024-10-12",
  "type": "patient_state_update",
  "state_change_cause": "cancer_progression",
  "patient": {
    "ecog_ps": 2,
    "weight_kg": 61.5,
    "symptom_burden": "moderate",
    "dominant_symptoms": ["pain", "fatigue"],
    "organ_function": {
      "renal": "normal",
      "hepatic": "moderate_impairment",
      "marrow_reserve": "mild_impairment"
    }
  }
}
```

Use `patient_state_update` only when host-state variables change meaningfully
and no other event on that date naturally carries the patient delta. Add
`state_change_cause` for standalone intercurrent events or gradual decline,
including injuries, infections, cardiovascular events, or other non-cancer
illnesses that alter performance status, organ function, symptoms, treatment
feasibility, or survival.
`state_change_cause` is a top-level event field, sibling to `patient`; never
place it inside the `patient` object.

`ecog_ps` is 0-4. Death is the terminal event, not ECOG 5.

Use schema enum values for `symptom_burden` and organ-function values.

`dominant_symptoms[]` is free text. Use specific symptom terms only when they
are trajectory-significant; do not emit routine review-of-systems detail.

Organ-function fields are `renal`, `hepatic`, `marrow_reserve`, `cardiac`, and
`pulmonary`.

## Local Interventions

`local_intervention` requires `modality`, `procedure`, and `intent`. Use
`target_sites[]` when possible. Disease deltas on this event represent immediate
physical effect; delayed effects belong in later `response` events.

Use schema enum values for local-intervention `modality` and `intent`. For
`local_intervention`, use the local-intervention intent enum only. Do not use
systemic-treatment intents such as `disease_control`; if `modality` is
`other_specified`, include `modality_description`.

Whenever an enum value is `other_specified`, include its companion description
field when present in the schema: `ae_category_description`,
`sample_source_description`, `modality_description`, or `eol_care_description`.

## Death

`death` requires `cause_category`, `cause`, and `eol_care_pattern`.
Death does not have to be cancer-related; choose the cause category that best
matches the trajectory.

`cause` is brief free text.
Use schema enum values for `cause_category` and `eol_care_pattern`.

## Recognition Bridge

`date` is the true latent transition date. `recognized_date` is when the
clinical system recognized it. The field name is exactly `recognized_date`;
do not emit `date_recognized` or both names.

- date string: clinically recognized on that date
- null: not recognized during this generated trajectory
- omitted: recognition timing is not relevant for that event type

## Pydantic Event Schema

The deterministic state engine validates each event against this schema:

```json
{{LEDGER_EVENT_SCHEMA}}
```

## Baseline Schema

The pipeline validates the resolved baseline (deep-merge of seed + patch)
against this Pydantic schema:

```json
{{BASELINE_SCHEMA}}
```

## Baseline Seed

Patient facts supplied for this run. These values are fixed, not
suggestions: reason from them and never contradict them. Fields absent here are
unspecified and must not be invented, and the trajectory you emit must not
depend on having decided them.

```json
{{BASELINE_SEED}}
```

## Pinned Fields

The observed leaves, as machine-readable paths. Nothing you emit may
contradict them:

{{PINNED_FIELDS}}

{{OBSERVED_PREFIX}}

{{ANCHOR_BLOCK}}

## Run-Specific Instructions

{{POLICY_TEXT}}

## Task

Now emit the JSON for this run: the future of the patient described by the
baseline seed and fixed prefix above, under the run-specific instructions and
schema contract. Do not emit a baseline and do not restate the prefix. The trajectory
must be internally coherent, temporally causal, and biologically/clinically
possible under the study knowledge cutoff ({{KNOWLEDGE_CUTOFF}}).
Emit exactly:

```json
{ "ledger": [ ... ] }
```
