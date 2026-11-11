# Specifications N6, N8 and N11: verbatim excerpts (28 October 2026)

These specifications were written in a planning document of the study, which is not released. Each section below is copied verbatim, with no edit, from that file as committed in the development repository at the commits and times given (commit identifiers refer to the development repository, which is not public). Where the first and last cited versions differ, both are given. The excerpts were extracted on 10 November 2026; they document what was written at those times and do not show that the analyses were preregistered. Mentions of Codex refer to the AI assistant used for code and statistical review, and Julien to the first author (Methods, Use of large language model assistants).

## N6: commit 382f6d6 (2026-10-28 12:38:19 +0100)

### N6 specification (for review; counts only; definitions completed after Codex 28 Oct)

- **Question.** How many comparable real patients remain as a treatment decision is described
  more precisely? The rare-cancer point (a common cancer described precisely has the support of
  a rare one) stays qualitative, in the text. **No incidence calculation.**
- **Data.** GENIE-BPC NSCLC, CRC and breast (gitignored folders; aggregate output only).
  Decision = start of a recorded systemic regimen after metastatic disease is documented. Only
  information dated on or before the decision is used.
- **Primary unit.** One decision per patient, drawn at random (seed 20261028); comparators are
  the other patients' drawn decisions. **Sensitivity:** all decisions on both sides; a
  comparator count is the number of **distinct other patients** with at least one matching
  decision; every record of the index patient is excluded.
- **Steps (cumulative, fixed order).**
  1. Cancer (NSCLC, CRC, breast).
  2. Stage group at diagnosis (`stage_dx`): I, II, III, IV, unknown.
  3. Driver, counted only if reported on or before the decision, otherwise "not yet tested";
     a result with no report date counts as "not yet tested" (sensitivity: counted as known).
     One category per patient, by fixed precedence when several are present:
     NSCLC EGFR > ALK > KRAS > tested, none of these; breast HR+HER2−, HER2+, triple-negative
     (`bca_subtype`); CRC MMR-deficient or MSI-high > BRAF > KRAS > tested, none of these.
     The number of patients with more than one driver is reported.
  4. **Regimen depth**: order of this regimen among systemic regimens started after metastatic
     disease (1, 2, 3, 4 or more). Not "line of therapy".
  5. Treatment class started (existing reader classes).
  6. Latest disease status before the decision: responding, stable, progressing (mixed merged,
     as in the reader), **no active disease** (documented), **not assessed** (no documented
     assessment). The last two are never merged.
  7. Previous regimen: **best** documented status during that regimen (responding > stable >
     progressing; no active disease and not assessed kept separate; "last" status is already
     step 6) and duration band (<3, 3-6, 6-12, ≥12 months; "first regimen" when there is none).
     Duration = regimen start to its documented stop or to the start of the next regimen,
     whichever is earlier (overlapping regimens end at the next start); with neither, "duration
     unknown".
  8. Age band (<50, 50-64, 65-74, ≥75) and sex.
- **Missing values.** Primary: "unknown" is its own category and matches only "unknown".
  Sensitivity: "unknown" matches nothing. At each step, the share of matches that rely on a
  shared unknown is reported separately from matches on documented values.
- **Readouts per step and cancer.** Median (IQR) number of comparators; share of decisions with
  no comparator; share with fewer than 5. **No survival intervals.**
- **Output.** Aggregate table and one panel for Fig. 6; Methods paragraph. No identifiers.
- **Not claimed.** That any level of matching is clinically sufficient, or that a model can
  replace missing support.
- **Code.** `scripts/explore/n6_support_ladder_v1.py` (to write), reusing the decision, regimen,
  class, status and previous-regimen readers of `test_a_feasibility_v1.py`, which is a starting
  point, not a validated reader: each reused reader gets a count check against its source table
  before use. Months = days / 30.4375.

## N6: commit 3297e79 (2026-10-28 17:26:01 +0100)

### N6 specification (for review; counts only; definitions completed after Codex 28 Oct)

- **Question.** How many comparable real patients remain as a treatment decision is described
  more precisely? The rare-cancer point (a common cancer described precisely has the support of
  a rare one) stays qualitative, in the text. **No incidence calculation.**
- **Data.** GENIE-BPC NSCLC, CRC and breast (gitignored folders; aggregate output only).
  Decision = start of a recorded systemic regimen after metastatic disease is documented. Only
  information dated on or before the decision is used.
- **Primary unit.** One decision per patient, drawn at random (seed 20261028); comparators are
  the other patients' drawn decisions. **Sensitivity:** all decisions on both sides; a
  comparator count is the number of **distinct other patients** with at least one matching
  decision; every record of the index patient is excluded.
- **Steps (cumulative, fixed order).**
  1. Cancer (NSCLC, CRC, breast).
  2. Stage group at diagnosis (`stage_dx`): I, II, III, IV, unknown.
  3. Driver, counted only if reported on or before the decision, otherwise "not yet tested";
     a result with no report date counts as "not documented as known at the decision" (a
     category of its own, not "not yet tested"; sensitivity: counted as known).
     One category per patient, by fixed precedence when several are present:
     NSCLC EGFR > ALK > KRAS > tested, none of these; breast HR+HER2−, HER2+, triple-negative
     (`bca_subtype`); CRC MMR-deficient or MSI-high > BRAF > KRAS > tested, none of these.
     The number of patients with more than one driver is reported.
  4. **Regimen depth**: order of this regimen among systemic regimens started after metastatic
     disease (1, 2, 3, 4 or more). Not "line of therapy".
  5. Treatment class started (existing reader classes).
  6. Latest disease status before the decision: responding, stable, progressing (mixed merged,
     as in the reader), **no active disease** (documented), **not assessed** (no documented
     assessment). The last two are never merged.
  7. Previous regimen: **best** documented status during that regimen (no active disease >
     responding > stable > progressing; "not assessed" only when no status was documented
     during the regimen; "last" status is already step 6) and duration band (<3, 3-6, 6-12, ≥12 months; "first regimen" when there is none).
     Duration = regimen start to its documented stop or to the start of the next regimen,
     whichever is earlier (overlapping regimens end at the next start); with neither, "duration
     unknown".
  8. Age band (<50, 50-64, 65-74, ≥75) and sex.
- **Missing values.** Primary: "unknown" is its own category and matches only "unknown".
  Sensitivity: "unknown" matches nothing. At each step, the share of matches that rely on a
  shared unknown is reported separately from matches on documented values.
- **Readouts per step and cancer.** Median (IQR) number of comparators; share of decisions with
  no comparator; share with fewer than 5. **No survival intervals.**
- **Output.** Aggregate table and one panel for Fig. 6; Methods paragraph. No identifiers.
- **Not claimed.** That any level of matching is clinically sufficient, or that a model can
  replace missing support.
- **Code.** `scripts/explore/n6_support_ladder_v1.py` (to write), reusing the decision, regimen,
  class, status and previous-regimen readers of `test_a_feasibility_v1.py`, which is a starting
  point, not a validated reader: each reused reader gets a count check against its source table
  before use. Months = days / 30.4375.

## N8: commit 382f6d6 (2026-10-28 12:38:19 +0100)

### N8 specification (for review, not frozen; PRIME, observed trial, no LLM)

- **Question.** Over a fixed horizon, how is the time of each randomized arm divided between
  alive without progression, alive after progression, and dead, by KRAS exon 2 status?
- **Population.** PRIME (NCT00364013) patients with known KRAS exon 2 status (`BMMTR1`;
  "failure" and missing excluded), analysed by randomized arm (`TRT`), as in Fig. 6a.
- **Events.** Progression-free survival by central review (`PFSDYCR`, `PFSCR`; event =
  progression or death), the trial's endpoint; overall survival from `DTHDY`, `DTH`.
- **Estimator.** Restricted means from Kaplan-Meier curves, censoring handled by the estimator:
  alive without progression = RMST(PFS, τ); alive after progression = RMST(OS, τ) − RMST(PFS, τ);
  dead = τ − RMST(OS, τ). Panitumumab minus FOLFOX for each part, within each KRAS group.
- **Horizon.** τ = 24 months, fixed now. Reported with the number at risk at τ per arm and KRAS
  group; if fewer than 20 patients remain at risk for OS in any group, τ = 18 months, decided on
  the at-risk counts before the parts are computed.
- **Uncertainty.** 95% bootstrap intervals, patients resampled within arm and KRAS group (2,000
  resamples, seed 20261028). No P values beyond those already reported for Fig. 6a.
- **Readiness checks (aggregate, 28 Oct, revised after Codex).**
  - `DTHDY` is filled for all 245 living patients (median 1,062 days) and 690 deaths. The data
    dictionary (`DDT_203_v2.xlsx`) labels it only "Death Day"; that it is the last-known-alive
    day for living patients is **an assumption still to be confirmed** from the protocol or CSR
    definition of overall survival. Until then N8 is not computed.
  - Every death is also a PFS event (0 deaths with PFS event = 0), so the endpoints are
    compatible on events.
  - The three records with OS time shorter than PFS time are **all censored on both endpoints**
    (no death, no PFS event; OS censoring 4-64 days earlier). No correction is applied: the
    original trial endpoints are kept as they are, each estimated separately by Kaplan-Meier.
    (The earlier "truncate PFS at OS" rule is withdrawn.)
  - The 0-day censored PFS records are handled as in Fig. 6a; the PFS numbers must reconcile
    with Fig. 6a before N8 is shown.
  - Each component is checked to be nonnegative with its interval (RMST(OS) ≥ RMST(PFS) in each
    arm and KRAS group, and in every bootstrap resample; the share of resamples violating it is
    reported). Summing to τ by construction is not taken as a check.
- **Display.** A stacked bar per arm and KRAS group (three parts summing to τ), next to the trial
  paths (Fig. 6). Legend: this shows the value of describing whole courses in a randomized
  comparison; **it does not validate the CSS coordinates or individual treatment benefit.**
- **Output.** Aggregate only (PRIME data-use agreement). Code
  `scripts/explore/n8_prime_time_in_state_v1.py` (to write).

## N8: commit 3297e79 (2026-10-28 17:26:01 +0100)

### N8 specification (for review, not frozen; PRIME, observed trial, no LLM)

- **Question.** Over a fixed horizon, how is the time of each randomized arm divided between
  alive without progression, alive after progression, and dead, by KRAS exon 2 status?
- **Population.** PRIME (NCT00364013) patients with known KRAS exon 2 status (`BMMTR1`;
  "failure" and missing excluded), analysed by randomized arm (`TRT`), as in Fig. 6a.
- **Events.** Progression-free survival by central review (`PFSDYCR`, `PFSCR`; event =
  progression or death), the trial's endpoint; overall survival from `DTHDY`, `DTH`.
- **Estimator.** Restricted means from Kaplan-Meier curves, censoring handled by the estimator:
  alive without progression = RMST(PFS, τ); alive after progression = RMST(OS, τ) − RMST(PFS, τ);
  dead = τ − RMST(OS, τ). Panitumumab + FOLFOX minus FOLFOX for each part, within each KRAS group.
- **Horizon.** τ = 24 months, fixed now. Reported with the number at risk at τ per arm and KRAS
  group; if fewer than 20 patients remain at risk for OS in any group, τ = 18 months, decided on
  the at-risk counts before the parts are computed.
- **Uncertainty.** 95% bootstrap intervals, patients resampled within arm and KRAS group (2,000
  resamples, seed 20261028). No P values beyond those already reported for Fig. 6a.
- **Readiness checks (aggregate, 28 Oct, revised after Codex).**
  - `DTHDY` is filled for all 245 living patients (median 1,062 days) and 690 deaths. The data
    dictionary (`DDT_203_v2.xlsx`) labels it only "Death Day"; that it is the last-known-alive
    day for living patients is **an assumption still to be confirmed** from the protocol or CSR
    definition of overall survival. Until then N8 is not computed.
  - Every death is also a PFS event (0 deaths with PFS event = 0), so the endpoints are
    compatible on events.
  - The three records with OS time shorter than PFS time are **all censored on both endpoints**
    (no death, no PFS event; OS censoring 4-64 days earlier). No correction is applied: the
    original trial endpoints are kept as they are, each estimated separately by Kaplan-Meier.
    (The earlier "truncate PFS at OS" rule is withdrawn.)
  - The 0-day censored PFS records are handled as in Fig. 6a; the PFS numbers must reconcile
    with Fig. 6a before N8 is shown.
  - Each component is checked to be nonnegative with its interval (RMST(OS) ≥ RMST(PFS) in each
    arm and KRAS group, and in every bootstrap resample; the share of resamples violating it is
    reported). Summing to τ by construction is not taken as a check.
- **Display.** A stacked bar per arm and KRAS group (three parts summing to τ), next to the trial
  paths (Fig. 6). Legend: this shows the value of describing whole courses in a randomized
  comparison; **it does not validate the CSS coordinates or individual treatment benefit.**
- **Output.** Aggregate only (PRIME data-use agreement). Code
  `scripts/explore/n8_prime_time_in_state_v1.py`; output `outputs/exploratory_n8_prime_time_in_state/`.

## N11: commit e3c0606 (2026-10-28 16:42:30 +0100)

### N11 specification: event-conditioned landscapes, concept and measurement (Julien approved 28 Oct; for Codex)

- **Why.** Jakob's favourite figure was the 29 July conceptual storyboard
  (`outputs/exploratory_event_conditioned_attractor_storyboard/fig01_concept_event_conditioned_landscape.pdf`):
  six moments of one course, each with a field, remission and cancer basins, and the net pull at
  the patient. Codex (12 Oct, R3) removed it because its parameters and basins are invented. The
  compromise keeps the concept as a labelled hypothesis and adds its measured counterpart.
- **Top row: concept.** The storyboard redrawn in the paper's style, labelled "conceptual model:
  field strengths and basin positions are illustrative, not estimated". No numbers presented as
  data (the c and u values are removed or marked illustrative).
- **Bottom row: the same six moments measured on the generated atlas** (4,434 trajectories,
  six models, frozen state map). Phases from each trajectory's own events, fixed now: (1) before
  the first systemic treatment; (2) first 3 months of first-line treatment; (3) after a first
  response, before progression; (4) response maintained for at least 3 months; (5) first 6 months
  after the first progression; (6) first 3 months after a second-line start. Start visits on a
  monthly grid are assigned to the phase they fall in.
  - **Field:** per cell of the ED4a grid, mean 3-month displacement among start visits of that
    phase that are alive 3 months later; drawn only in cells with at least 30 unique courses (the
    ED4a support rule). Deaths within 3 months are shown separately (count per phase), never as an
    arrow destination.
  - **Accumulation regions ("basins", measured):** contours of where generated patients spend time
    in each phase (time-weighted density of positions, highest-density region holding 50% of the
    phase's person-time). Labelled as regions where generated patients accumulate, not attractors
    or potentials.
  - **Path:** the generated patient of Fig. 1c, with its position at each moment.
- **Not claimed.** Forces, potentials or stability; causal effects of treatment (treatments in
  the atlas are generated, not assigned); that real patients follow these fields. The figure shows
  whether LLM-generated courses behave as the concept says; disagreements are reported as findings.
- **Feasibility first.** Support counts per phase and cell before any field is drawn; a phase
  without supported cells is shown blank with its n.
- **Placement.** Candidate for Fig. 2 (replacing the constructed panels c and d, which move to
  Extended Data); current Fig. 2 stays as fallback.

## N11: commit 3297e79 (2026-10-28 17:26:01 +0100)

### N11 specification: event-conditioned landscapes, concept and measurement (Julien approved 28 Oct; for Codex)

- **Why.** Jakob's favourite figure was the 29 July conceptual storyboard
  (`outputs/exploratory_event_conditioned_attractor_storyboard/fig01_concept_event_conditioned_landscape.pdf`):
  six moments of one course, each with a field, remission and cancer basins, and the net pull at
  the patient. Codex (12 Oct, R3) removed it because its parameters and basins are invented. The
  compromise keeps the concept as a labelled hypothesis and adds its measured counterpart.
- **Top row: concept.** The storyboard redrawn in the paper's style, labelled "conceptual model:
  field strengths and basin positions are illustrative, not estimated". No numbers presented as
  data (the c and u values are removed or marked illustrative).
- **Bottom row: the same six moments measured on the generated atlas** (4,434 trajectories,
  six models, frozen state map). Phases from each trajectory's own events, fixed now: (1) before
  the first systemic treatment; (2) first 3 months of first-line treatment; (3) after a first
  response, before progression; (4) response maintained for at least 3 months; (5) first 6 months
  after the first progression; (6) first 3 months after a second-line start. Start visits on a
  monthly grid are assigned to the phase they fall in.
  - **Field:** per cell of the ED4a grid, mean 3-month displacement among start visits of that
    phase that are alive 3 months later; drawn only in cells with at least 30 unique courses (the
    ED4a support rule). Deaths within 3 months are shown separately (count per phase), never as an
    arrow destination.
  - **Accumulation regions ("basins", measured):** contours of where generated patients spend time
    in each phase (time-weighted density of positions, highest-density region holding 50% of the
    phase's person-time). Labelled as regions where generated patients accumulate, not attractors
    or potentials.
  - **Path:** the generated patient of Fig. 1c, with its position at each moment.
- **Not claimed.** Forces, potentials or stability; causal effects of treatment (treatments in
  the atlas are generated, not assigned); that real patients follow these fields. The figure shows
  whether LLM-generated courses behave as the concept says; disagreements are reported as findings.
- **Feasibility first.** Support counts per phase and cell before any field is drawn; a phase
  without supported cells is shown blank with its n.
- **Placement.** Candidate for Fig. 2 (replacing the constructed panels c and d, which move to
  Extended Data); current Fig. 2 stays as fallback.
- **Feasibility counts (28 Oct; `scripts/explore/n11_phase_support_v1.py`, counts only).** All six
  phases definable from the event ledger (no proxy). Per phase: trajectories / start visits /
  followed at 3 months / died within 3 months / ED4a cells with >= 30 courses and >= 30 followed
  endpoints / share of the phase's person-time in those cells:
  (1) 4,379 / 16,690 / 16,611 / 79 / 39 / 94.9%; (2) 3,720 / 11,138 / 11,131 / 7 / 54 / 86.3%;
  (3) 4,035 / 35,246 / 35,068 / 178 / 49 / 91.9%; (4) 3,618 / 23,318 / 23,164 / 154 / 41 / 90.5%;
  (5) 4,399 / 25,147 / 21,807 / 3,340 / 83 / 89.5%; (6) 4,217 / 12,603 / 11,423 / 1,180 / 61 / 85.1%.
  No start visit left out for a record ending before 3 months. Phase 4 is a subset of phase 3.

- **Amendments after Codex (28 Oct, on the feasibility counts).** Deaths reported as distinct
  trajectories per phase, not monthly windows (the earlier 3,340 counts windows; the "escape"
  reading is withdrawn). Support for an arrow: at least 30 distinct courses starting in the cell
  and at least 30 distinct courses contributing a followed endpoint; bootstrap by whole
  trajectory. Literal event-based phase labels: (2) "first 3 months after the first systemic
  start, before any progression" (starts after a progression excluded, stated); (4) "3 months or
  more after a first response, no progression recorded since" (not verified continued
  response); (6) "first 3 months after the next treatment start following progression" (not a
  verified clinical line). The six columns are overlapping event-conditioned views, not six
  successive stages of one course; density concentrations and survivor-conditioned moves do not
  validate attractors, resistance or causal treatment effects.

