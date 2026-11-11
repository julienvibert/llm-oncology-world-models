# Scoring specification: amendment of 12 October 2026

Amends `manuscript/SCORING_SPEC_2026-09-28.md` (and its 30 September amendment).
Author: Claude Code, implementing Codex's required hand-off in
`review/2026-10-12_codex_review_gate3_eligibility_precheck.md` (R1-R6).
Clinical decisions: Julien. Specification owner: Codex.

**Status: a dated post-hoc freeze, not a preregistration.** It was written after the first
eligibility precheck (commit `d0a407a`) and its counts had been seen, and after exploratory
results of the wider programme. It changes how the record and the forecasts are READ. It
does not change the targets, coordinates, scaling, floors, arms, contrasts or fallback
rule. No score, distance, arm effect or outcome agreement has been computed under it.
The clinical conventions (drug classes, section 3) and the donor fallback decision
(section 9, J1) were confirmed by Julien on 12 October 2026, before any score.

Code bound to this amendment: `scripts/four_arm_readers_v2.py`,
`scripts/four_arm_eligibility_v2.py`, `scripts/four_arm_drug_classes_v1.csv` and the entry
point `scripts/score_four_arm_final_v1.py --precheck-only`. Tests:
`tests/unit/test_four_arm_readers_v2.py`. The rerun's manifest
(`outputs/four_arm_preprint_20260928/manifest_20261012.json`) records the hash of this file
and of every file executed. The first precheck's outputs and manifest are kept unchanged.

## 1. Generated disease is cumulative truth (R1)

The earlier reader restated the status and presence channels separately so that a
scan-expiry rule would not fire. A superseded value then received new timestamps and
beat the newer one. A liver-only lesion cleared at month 3 read as NED, progressing, NED
and so on. That alternation was produced by the adapter, not by the model.

From now on:

1. **One cumulative state, updated in event order.** Two variables are carried: presence
   (evidence / no evidence / equivocal) and a curated-style status (responding, stable,
   progressing). They start from the landmark the prompt gave. Each event changes only
   what it supplies. Nothing is restated, so no old value gets a new evidence time.
2. **Same-date events are one update.** Presence is the cumulative clone/site picture
   after all of them. The status is taken from their disease event types; if they
   disagree, the worst wins (progressing, then stable, then responding), as for real
   same-time records. A newly appearing distant site asserts progression.
3. **Supersession is strictly by time.** A later NED reading clears an older status. A
   later active status, or a new site, turns an older NED reading back into disease
   present.
4. **Complete response.** A `response` whose cumulative burden has cleared at every known
   site is no active disease. If the landmark extent was not known, clearing what was
   named is not remission (unchanged).
5. **Genuine contradictions stay distinct.** NED and active disease asserted at the same
   time are a contradiction made by the model. Example: a `progression` event whose own
   disease block clears every known lesion. This was previously read as NED. The state
   is now unknown until a later event updates either channel. None of the 17,679
   accepted repeats contains one. Three repeats do have unknown time after the landmark
   at 24 months: an evidence-only update follows NED with no status, so cancer is
   present but its trajectory is not stated.
6. **No expiry, and one reader under both conventions.** Generated assertions are truth
   updates, not scans. They do not expire, and the course is the same under the primary
   and the sensitivity conventions. The conservative reading changes only how real and
   donor records are read and how their conflicts are handled. In the first precheck the
   generated course was re-projected under the conservative record rule; that is withdrawn.
7. **Reading on the query calendar** is unchanged: the one observation adapter reads the
   state at each calendar time and holds it to the next read. Generated new sites go
   through the same reads as every other state change, rather than bypassing them.

Regression fixtures: the liver-only clearance under both conventions, including its
calendar projection; the reciprocal NED-to-new-site case; NED superseded by a later
status-only event; a genuine contradiction and its recovery; long silence without revival.

## 2. Real treatment from the curated agent fields (R2)

Source: `regimen_cancer_level_dataset.csv`, agent slots 1-5: `drugs_drug_k`,
`dx_drug_start_int_k`, `dx_drug_end_int_k`, `dx_drug_end_or_lastadm_int_k`, and the
regimen's `drugs_dc_ynu`. These are restricted to the index cancer (`DE.index_cancer`) in
all three cohorts. Definitions follow the analytic data guides: NSCLC v2.0 pp. 70-72,
BRCA v1.0 pp. 93-96, CRC v2.0 pp. 84-87.

1. **Reconciliation with the timeline export**, by patient, regimen number, agent and
   start day. Two kinds of unmatched record occur, and both are kept:
   - Regimen-table slots absent from the timeline. All are duplicate "Investigational
     Drug" slots within one regimen (142 / 458 / 79 slots in NSCLC / BRCA / CRC).
   - Timeline rows of the index cancer whose curated record is filed under another
     `ca_seq` with identical dates (10 / 41 / 35). These take that record's fields.

   Any other unmatched timeline row is an error.
2. **Stop and last administration stay distinct.**
   - Discontinued (Yes) with a stop: exposure is documented from start to stop.
   - Discontinued without a stop: unresolved from the start.
   - No or Unknown, with a last-administration date: documented from start to last
     administration, unresolved afterwards.
   - No or Unknown, without one: unresolved from the start.

   Nothing is carried to the patient's last follow-up. Doing so would be a named
   imputation, not the documented GENIE rule.
3. **Investigational drugs.** The name is masked and the duration redacted. Exposure is
   unresolved from the start.
4. **A documented stop equal to the start** (a single administration) is zero documented
   exposure, as recorded. It is not lengthened.
5. **Pre-landmark treatment** overlapping the window is included (section 7 already
   required this; the earlier numeric projector did not do it).
6. **Agents dated before diagnosis** (negative start; 11 / 4 / 7 slots) contribute
   exposure. The regimen-start event is the regimen's earliest agent start on or after
   diagnosis. This is the rule that built the saved landmarks, so the index regimen still
   starts at time zero.
7. **Registry clock**: 30.4 days per month, the clock of the saved landmarks.
8. **One treatment object** per course. It is used both for eligibility and for the
   coordinate values.

## 3. Drug classes (R3)

`scripts/four_arm_drug_classes_v1.csv` (version 1, tracked, aggregate). Each drug has a
class, its aliases (GENIE spellings and generated spellings, including explicit
misspellings), a note and a confirmation flag. Names are normalised in one way:
lower case, parenthetical synonyms dropped, underscores read as spaces, spaces collapsed.
Matching is then exact. There is no substring matching and no fuzzy matching. The same
table is used for real queries, donors and generated components (`regimen.components`,
not the label).

- **Modality classes**: chemotherapy, immunotherapy, targeted, endocrine. A drug may
  carry two.
- **Adjunct**: recognised, but contributes no modality and no exposure.
  - Modulators: leucovorin and its forms; tipiracil.
  - Supportive or non-oncology medicines: corticosteroids, analgesics, antiemetics,
    bone-modifying agents, antibiotics, anticoagulants.
  - Local therapy: radiation.
  - Explicit "no therapy": observation, none, no cancer-directed therapy.

  A regimen made only of adjuncts is not a cancer-directed regimen and is not counted.
- **Unrecognised**: anything not in the table, including masked investigational agents,
  "Other NOS", "Other hormone" and bare class words ("chemotherapy", "PI3K inhibitor").
  No modality is ever inferred for an unidentified agent, and no identity is guessed.
- **Partially unrecognised combinations** have an unresolved modality set: one
  recognised companion does not resolve it. A GENIE agent is cancer-directed by curation,
  so its dated exposure counts towards any-therapy time even when its class is unknown.
  A generated regimen made only of unrecognised components has unknown
  cancer-directedness, so the regimen count is unresolved.
- **Added in this version**:
  - Drugs that were missing: floxuridine, mitomycin, vinblastine, ixabepilone,
    everolimus, alpelisib, capivasertib, savolitinib, vemurafenib, olaratumab,
    margetuximab, T-DM1.
  - GENIE spellings: "Goserlin Acetate" and "Ziv Aflibercept".
  - Every other GENIE agent name in the three releases, and every generated component
    that names an agent.
- **Study conventions, confirmed by Julien on 12 October 2026** (flagged in the table).
  The four modality coordinates are analysis bins, not pharmacological classes:
  - Antibody-drug conjugates count as chemotherapy plus targeted (antibody plus
    cytotoxic payload): trastuzumab emtansine, trastuzumab deruxtecan and sacituzumab
    govitecan. One ADC is one treatment: the two channels overlap and are never summed
    or counted as two agents. The payload names "emtansine" and "deruxtecan", when given
    as separate components next to trastuzumab, are chemotherapy, so the split form gives
    the same set. These payload entries exist only to read that split representation;
    they do not imply standalone administration.
  - IMiDs (thalidomide, lenalidomide, pomalidomide) count as immunotherapy, distinct
    from checkpoint inhibitors.
  - Somatostatin analogues (octreotide, lanreotide) count as endocrine. Endocrine means
    hormonal mechanisms broadly, not only sex-hormone therapy.
  - Radiopharmaceuticals and radioimmunoconjugates (Lu-177 dotatate, radium-223,
    Y-90 ibritumomab) count as targeted. Targeted means agents directed at a molecular
    or tissue target, including radiopharmaceuticals.
  - Tretinoin counts as targeted (a differentiating agent directed at PML-RARA). This
    replaces the earlier proposal of chemotherapy. It remains cancer-directed exposure.
  - Drug aliases stay explicit; no fuzzy matching (Codex review of the v2 readers, R2).

  Outside the IMiDs, these are rare in the index cancers.

## 4. Per-agent exposure (R4)

1. Each real agent and each generated component has its own exposure intervals. There
   is no shared last stop across a regimen's components.
2. Exposure is a union within each modality and within any-therapy time. Overlaps
   between modalities are permitted and are never summed. Exposure stops at death, or
   at last contact for a censored course.
3. **Generated modifications.**
   - `discontinuation_of_component` closes the named components on that date.
   - `dose_hold` interrupts exposure only when it lasts eight weeks or more: until
     `dose_resume`, or until the component ends if it is never resumed. This is GENIE's
     own regimen-break rule; shorter holds remain exposure.
   - `dose_reduction` does not change exposure.
   - A modification naming a component that is not active is ambiguous. A discontinuation
     then makes every active component unresolved from that date to the end of the
     regimen. A hold does so only for a hold of eight weeks or more.
   - An interrupted exposure is not a new regimen.

   Asymmetry kept on purpose: GENIE would curate a resumption after eight weeks as a new
   regimen; generated resumptions are not counted as regimens. The two regimen counts are
   therefore not identical clinical measures. This qualification stays beside any
   regimen-count comparison, which is not called treatment-sequence concordance (Codex
   review of the v2 readers, R4).
4. A switch or stop closes every active component, and a start opens new ones. A start
   recorded while treatment is still active is read as a switch: it closes the active
   components before opening its own. The frozen prompt allows a start only for the first
   regimen or after a stop, and frozen replay promotes such a start to a switch
   (`SOFT_LIFECYCLE_AUTO_PROMOTED_START_TO_SWITCH`), keeping the raw event. The reader
   follows that contract and never reads the start as an added concurrent regimen. The
   number of conversions per arm is reported as a diagnostic
   (`generated_active_starts_read_as_switch_at_24` in the precheck manifest), and a test
   checks that such a start gives the same treatment object as the explicit switch
   (Codex review of the v2 readers, R1). Rerun of 12 October, 24-month reader: 195 starts
   in 174 ledgers (generic cohort), 165 in 163 (patients like me), 135 in 128 (setting)
   and 134 in 123 (patient only), of 17,679 projected ledgers in all (about 4,400 per arm). Eligibility
   counts are unchanged, because the reading itself did not change.

## 5. Exposure versus regimen events (R5)

- **Exposure coordinates** use positive-duration overlap with [0, min(death, H)).
- **Regimen-event coordinates** use recorded starts in (0, H] before death. A start
  exactly at H counts.
  - Starts within half a day form one modality-set group, so their row order cannot
    create a switch.
  - The time-zero group (index regimen and co-dated starts) is the reference and is
    never counted as subsequent.
  - Adjunct-only starts are skipped.
- **Changes.** A change is a group whose modality set differs from the previous group's.
  - If either set is unresolved, that comparison is unresolved. The change count is then
    unresolved.
  - The first-change indicator and time stay resolved only when a definite change occurs
    before any unresolved comparison.
  - The subsequent-regimen count stays resolved whenever every group's cancer-directedness
    is known.

Example: an unrecognised regimen starting exactly at H leaves exposure resolved and the
change coordinates unresolved.

The treatment block resolves only when all nine coordinates resolve. Reason codes now
name the cause: an unresolved exposure kind, with a pre-landmark flag; an unrecognised
agent exposed; an unresolved modality set; or unknown cancer-directedness.

## 6. Donor conflicts and observation (clarifications, unchanged in effect)

- A genuine conflict in a donor's record is loss of documentation from that time until
  the donor's next supported read. It is never resolved by row order, and it is never
  read as remission or progression.
- Sampling a donor on the query calendar never renews its evidence clock.
- Windows are fixed before any LLM future is read, and are identical across arms and
  repeats.

## 7. Section 6 fallback coverage, as implemented

- **Query support.** For a query and horizon, the supported set is the time in [0, H]
  that the query's observed path documents. Known post-death time is included as the
  absorbing exit.
- **Common window W.** Each donor is read on the query's calendar through the same
  adapter. W is the intersection of query support and donor support.
- **Usable donor**: |W| of at least one month.
- **Eligible patient**: at least seven usable members of that card, and the corresponding
  LLM arm cell with at least seven originally accepted repeats. The section-7 cell floor
  is used here; full resolution of every repeat is not required, because unresolved
  generated time inside W is charged rather than excluded.
- **Scored time**: the sum of |W| over usable donors.
- **"Both"** means eligible under both conventions.
- The four-arm analogue uses the query's own supported time (at least one month) and all
  four cells with at least seven accepted repeats.
- **Reporting** (Codex review of the v2 readers): coverage (queries with an eligible
  donor comparison) and supported duration (scored time) are reported separately;
  eligibility at a 24-month horizon does not mean 24 fully documented months. The strict
  joint donor comparison (at most 34 queries, primary, 3 months) is descriptive or
  omitted, never used to rescue a preferred contrast. Queries excluded because a redacted
  investigational drug leaves treatment unresolved are counted by cancer, horizon and
  comparison, and the remaining subset is not called representative of all 450 patients.

## 8. Coordinate builder

`four_arm_eligibility_v2.coordinates` returns the 17 raw values (clinical units), their
transformations (months divided by H; counts n/(1+n); indicators unchanged) and the
block-weighted vector (sqrt((1/4)/d_b)). It returns no value for an unresolved
coordinate. Disease and event coordinates come from the path observed on the query
calendar. It computes no distance. A traced synthetic example is in the tests, and a
private three-query trace is written by `--traced-example 3`.

## 9. Rerun of the eligibility precheck, and proposed frozen decisions

Primary convention, all 450 queries (first precheck in parentheses):

| Population | 3 mo | 6 mo | 12 mo | 24 mo |
|---|---|---|---|---|
| Real target | 288 (260) | 278 (248) | 260 (229) | 216 (193) |
| Four-arm intersection | 286 (246) | 269 (209) | 233 (139) | 181 (55) |
| Four-arm intersection, both conventions | 248 (201) | 194 (140) | 130 (70) | 90 (18) |
| Matched LLM vs matched donors (joint) | 34 (21) | 20 (7) | 7 (2) | 2 (0) |
| Generic LLM vs generic donors (joint) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

Section 6 coverage, primary: queries with at least seven usable donors and an eligible cell.

| Card | 3 mo | 6 mo | 12 mo | 24 mo |
|---|---|---|---|---|
| Generic (vs generic-cohort arm) | 386 | 412 | 426 | 437 |
| Matched (vs patients-like-me arm) | 381 | 411 | 426 | 437 |

Decisions, frozen before any score (confirmed by Julien on 12 October 2026):

1. **Generic donors.** The joint comparison is unavailable at every horizon. Section 6
   and the scalar comparisons are the donor result.
2. **Matched donors.** Section 6 is the donor result at every horizon, applied
   consistently (Codex R6). The joint matched comparison meets the twenty-patient floor
   by the letter at 3 months (34; 22 of them CRC) and exactly at 6 months (20). It falls
   to 8 and 1 under both conventions. It would be reported as descriptive only, with no
   inferential claim. **Confirmed by Julien on 12 October 2026 (J1)**, as endorsed by
   Codex: section 6 is the donor result at every horizon.
3. **Four-arm joint score.** The floor is met at every horizon pooled, and in every
   cancer under the primary convention. Under both conventions, BRCA meets the floor
   exactly at 12 months (20 patients) and falls below it at 24 months (12 patients); the
   BRCA joint comparison under both conventions is descriptive only at 24 months. Deaths by 24 months make up 51.9% of the
   primary 24-month set (181), 80.0% of the both-convention set (90) and 42.2% of the full
   cohort (450). The 80% figure describes only the both-convention set. Both sets are
   selected populations, shown with their composition; passing a size floor does not
   remove selection. Every score table states its population and death composition. The
   earlier descriptive-only restriction, based on 18 patients at 24 months, no longer
   applies to the pooled set (Codex review of the v2 readers, R3).

Aggregate tables: `eligibility_precheck_20261012.csv`, `population_composition_20261012.csv`,
`coverage_20261012.csv`, `fallback_coverage_20261012.csv`,
`diagnostic_treatment_reasons_20261012.csv`, all in the gitignored
`outputs/four_arm_preprint_20260928/`.

## 10. Remaining ambiguities, not resolved here

- **Zero-duration GENIE exposure** (a stop equal to the start, for a non-investigational
  drug) is scored as recorded. Clinically, one cycle may cover about three weeks.
- **Investigational drugs** started before the landmark leave treatment unresolved
  indefinitely. The redacted end has no documented bound. Among real targets excluded
  by treatment alone, 41 carry this reason at 3 to 12 months (37 at 24 months).
- **Generated event dates** are asserted onset dates (`date`), not `recognized_date`.
  This is unchanged, and it is not reviewed here.
- **Calendar reads after the real patient's last assessment** hold the generated state to
  H. This is the adapter semantics already reported in the first precheck (item 7.3).

## 11. Gate 4, step 1: scorer definitions where the specification is silent

Added on 12 October 2026, before any score is computed. Code: `scripts/four_arm_scores_v1.py`
(pure functions, no data access); fixtures: `tests/unit/test_four_arm_scores_v1.py`.

- **Primary and sensitivity scores.** The equal-fan-size energy score is the primary joint
  comparison. The as-delivered score and the LLM-only fair estimator are sensitivities;
  they answer different questions and are never presented as interchangeable results.
- **Section 6 losses.** Occupancy total variation and time-aligned disagreement are
  expressed as fractions of the window length |W| (0 identical, 1 completely different),
  so each usable donor window carries equal weight in the donor average. |W| in months is
  reported beside each loss. A forecast with no documented label at a time inside W counts
  as unresolved there and is charged. Living intervals are clipped at death before the
  post-death exit is added, so no living time accrues after death.
- **Dependence versus chronology.** The block shuffles test relations between survival,
  disease and treatment, not event order. Order is checked separately by time-aligned
  disagreement and by the progression-to-regimen timing diagnostic. The alive-at-H stratum
  of each repeat is read from its survival block.
- **Progression entry.** A documented entry into progression is a transition into
  progressing after month 0; a patient already progressing at the landmark has no entry
  until a later documented return to progression.
- **Censoring weights.** The reverse Kaplan-Meier risk set at time s contains every
  patient with time at or after s, including a death at s. G is estimated within each
  cancer on the fixed query cohort, with left limits. Weights are never capped; a required
  G below 0.05 flags the estimate as unsupported. Both Kaplan-Meier helpers use one tie
  rule: times within 1e-9 months of the previous time form one group, and event counts
  and risk sets then use exact equality on the grouped times. In the cohort bootstrap,
  censoring weights are re-estimated inside every resampled cohort; Cox models and
  retrieval are not refitted.
- **Donor survival.** A donor card's Kaplan-Meier estimate at H is unavailable when no
  member is followed to H and the curve has not reached zero.
- **ECE bins.** Distinct predicted values are taken in order, and a bin closes once its
  cumulative count reaches the next k·n/5; identical predictions are never split. With no
  evaluable patient (no known outcome, or zero total weight) ECE is unavailable, not zero.
- **Inference.** 2,000 resampled patient cohorts, stratified by cancer; each sampled
  patient brings all its arms, repeats and donor cards. Seeds are SHA-256 of 20260928 and
  the comparison keys. Wilcoxon P is not computed when every paired difference is exactly
  zero; no numerical threshold is applied.
  Holm adjustment counts unavailable slots in the family size.
- **First attempt.** Read from `attempts_detail[0]` (acceptance flag and raw response).
  Unavailable means not saved (no detail, no flag, or a raw response that is absent or
  null). A saved empty response is an observed output: unaccepted, it is invalid.
  In the frozen run, 13,922 of 18,000 requested slots have a valid first attempt; every
  slot keeps its first raw response.
- **Cohort entry point (step 3), acceptance gates** (Codex review of step 1): the seven-
  repeat floor and complete projection are enforced before any score; subsets and seed
  keys are fixed across block, joint and bootstrap use; all arms and donor cards of a
  patient move together; unsupported or empty results stay unavailable with a reason row
  and keep their place in the Holm family; both shuffle versions, the fixed donor windows
  and the timing diagnostic are kept beside the joint score.

## 12. Gate 4, step 3: implementation choices of the cohort extraction

Added on 12 October 2026, written before the extraction's results were read. Code:
`scripts/four_arm_extract_v1.py`, run as `score_four_arm_final_v1.py --score`. Items marked
**for review** are choices the specification leaves open.

- **Eligibility.** The extraction re-derives every precheck four-arm row and stops if one
  does not resolve. Equal-size subsets use the keys `(family, query, arm, H)`, without the
  convention, so the primary and sensitivity readings score the same subsets.
- **Primary joint question.** Equal-size energy score, primary convention, pooled; four
  contrasts at four horizons; one Holm family of 16 slots. Per-cancer, as-delivered, fair,
  per-block, equal-cancer-weighted, first-attempt and common-convention results are
  sensitivities without adjusted P values.
- **Donor result (section 6 fallback). For review:** time-aligned disagreement is the
  primary loss of the eight-slot donor family, because it keeps the order of states;
  occupancy total variation is reported beside it, unadjusted. Population: the fallback
  rule (at least seven donors with at least one month of common window) and the receiving
  LLM arm with at least seven accepted repeats projectable at H. Unresolved generated time
  is charged. The strict all-twenty joint donor comparison is omitted.
- **Four-arm course diagnostic.** Query supported time of at least one month; every arm
  with at least seven accepted repeats projectable at H; unresolved time charged.
- **Survival. For review:** an LLM probability is the proportion alive at H among accepted
  repeats, available only when at least seven are accepted and every accepted repeat
  resolves survival at H (RMST needs resolution at 24 months). Donor probabilities are
  Kaplan-Meier estimates on the card's residual follow-up, unavailable without support to
  H. The comparison population at each H holds the queries where all seven forecasters
  (four arms, two donor cards, Cox) are available. Censoring weights are estimated on the
  450-query cohort within cancer; the Brier denominator is the comparison population. In
  each of the 2,000 resampled cohorts the weights are re-estimated and only comparison
  patients are scored. Cox models and retrieval are not refitted.
- **Cox.** The existing cross-validated model is re-fitted from its saved landmark frame
  with the same covariates, ridge (0.1), missingness rule and fold seed (20260908); fold IDs
  are now saved, the 3-month readout is added and the 6-, 12- and 24-month forecasts are
  checked against the saved ones (manifest, `cox_rebuild_check`).
- **Dependence.** Per fan, the mean shift over 200 permutations; summarised per arm and
  horizon as the mean over patients with a cohort-bootstrap interval.
- **First attempt.** Patients of the primary four-arm set whose four arms each hold at
  least seven valid first attempts, all projectable; compared with the full pipeline on the
  same patients.
- **Timing.** Per horizon, on the primary four-arm set: fractions of courses with a
  documented progression, with a later regimen, and with a regimen before any progression,
  averaged per patient; median paired time over all paired courses.
