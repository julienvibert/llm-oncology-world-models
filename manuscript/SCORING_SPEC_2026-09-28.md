# Four-arm scoring closure: specification for review

Date: 2026-09-28. Source checkpoint: `47c0f3f`.
Owner: Codex. Independent reviewer: Claude Code. Clinical decisions: Julien.
Status: **amended 30 September after CC review; awaiting verification of amendments and
the eligibility-only precheck. No full-cohort scoring authorised by this document alone**.

**Amendment, 30 September 2026.** Incorporates CC's R1-R7 at `aa8578b` and the
qualifications accepted in the review's 30 September response. Sections 6-10 now specify
both delivered-forecast and equal-size scores, the pre-score eligibility count, population
composition, a survival-stratified shuffle, the secondary missing-time diagnostic, the
primary observation convention and the donor multiplicity family. These remain post-hoc
decisions made after exploratory results, not a preregistration. The precheck is required
work, not an existing result; gate 3 remains open until its review.

This is gate 3 of [the shared preprint plan](PREPRINT_PLAN_2026-09-28.md), not another
project plan. The development data and several results have already been examined.
This document freezes proposed post-hoc scoring decisions before the final recomputation;
it is not a preregistration. The original generation design and four contrasts retain
their original provenance. Any substantive revision after review needs a dated amendment.

## 1. In plain language

We will ask whether the forecasts get survival probabilities right, reproduce documented
disease and treatment courses, and reproduce those outcomes together. We will compare
the four LLM arms with the information in their donor cards used directly. No patient,
donor, prompt, model or generated response will be replaced.

There are three different comparisons, with different information requirements:

1. Survival probabilities can use censored observations through established survival
   methods. Not knowing whether a patient is alive at month 24 is not the same as death.
2. Disease-course agreement can use intervals on which the real records support a
   comparison. The LLM and donor must be judged on the same intervals. This is a
   documented-course diagnostic, not a score of the complete future distribution.
3. A joint distribution score needs complete values for every included coordinate.
   We will report its eligible population, not manufacture missing values to enlarge it.
   It measures the declared course summaries, not every detail of a raw trajectory.

Known measurement errors must be resolved. A small, uncertain or negative effect is a
finished result. Neither weights nor eligibility will be tuned to obtain significance.

CC's main review decisions are the common observation adapter (section 4), the distinction
between interval-supported donor diagnostics and a full-card distribution (sections 6-7),
the fixed 17-coordinate scaling, and the complete-documentation population for the joint
score. These are proposed definitions to approve, not existing results to endorse.

## 2. Frozen inputs and boundaries

| Input | Source |
|---|---|
| Definitive responses | `runs/four_arm_v1_definitive_p40b20_20260826/attempts.jsonl` |
| Generation manifest | Same directory, `manifest.json` |
| Saved patient/donor selection | `outputs/exploratory_query_universe/queries.jsonl` |
| Exact prompt instruments and patient payloads | `outputs/exploratory_four_arm_prompts/quartets.jsonl` and `provenance.json` |
| Real courses | The same local GENIE releases used to build that universe; paths/hashes recorded in the extraction manifest |
| Previous results, for reconciliation only | `outputs/exploratory_four_arm_deep/` and its chronological `FINDINGS.md` |

Generation was bound to `28c12e43c4e133730b7d59c02935ea649881f388`, with
`deepseek/deepseek-v4-flash-0731`, ten requested repeats and at most three attempts per
repeat. The saved prompt provenance names the full query-universe digest
`037ccee7149c54e19a3bfc78c644333641a059758f5f5b115a342d8fb77a2662`.
The short checksum in the run manifest is not a substitute for checking the full digest.

Before extraction, hash the actual inputs and record the analysis commit and this
specification's hash. Assert task-ID uniqueness, the complete requested task grid,
query/arm/repeat linkage, response hashes where recorded, and agreement with the saved
prompt hashes. Do not print clinical records in logs. Unexpected missing input or a
projection exception is an explicit error, not a silently skipped row.

No new generation, donor retrieval, carve, acceptance rule or molecular/clinical filter.
The held-out confirmation cohort stays untouched. The previously reported 18,000 requested
slots and 17,679 accepted responses are reconciliation checks, not newly verified counts
from this session. The first-attempt sensitivity uses the same saved slots.

## 3. Arms, targets and populations

Arms, in fixed order: `patient_only`, `setting`, `generic_cohort`, `patients_like_me`.
Contrasts, later minus earlier: 2-1, 3-2, 4-3, 4-1. Negative differences in loss favour
the later arm. Arm 4 versus arm 3 tests the complete retrieval strategy, including
treatment relevance; it does not isolate history or patient similarity alone.

Time zero is the saved regimen-start landmark. Use the exact saved index action, not a
reselected same-month regimen. Real times remain on their diagnosis-relative clock before
subtracting the saved landmark. Generated dates use 30.4375 days per month. Retain
sub-month precision in computation. Events at time zero are baseline, not new future events.

Report horizons **3, 6, 12 and 24 months separately**. Twelve months is the continuity
readout for the existing main analysis; longer-horizon results are not selected because
their effects are larger. Do not concatenate nested horizons into one vector.

For each target and horizon, save eligibility by query, arm and forecaster with reason
codes. Main four-arm tables use the intersection of eligible queries across all four arms.
Baseline tables use the intersection across the forecasters actually compared; give the
four-arm estimates again on that subset rather than placing unmatched rows beside them.
Primary-versus-sensitivity and horizon-trend comparisons additionally show their common
patient set. Per-horizon maximal eligible sets remain visible separately.

The unit of analysis is the patient, not a generated repeat or donor appearance. Pooled
estimates average patient scores; report cancer-specific results alongside and an equal-
cancer-weighted sensitivity if differential evaluability changes the original balance.

## 4. Read the record without changing the experiment

The primary convention remains chronological forward persistence. Start from the
landmark facts the prompt actually received. Later evidence updates only the information
it supplies. Do not revive discarded stale records, read a first assessment backward, or
infer remission from subsequent follow-up. Same-time contradictions stay unresolved.
Repeated confirmations of an unchanged state are not new clinical events.

The sensitivity remains the already implemented conservative reading with genuine
four-month expiry and conservative conflict handling. Both components must be stated:
it is not a pure experiment on the expiry duration. Re-extract real, donor and generated
courses under each convention; do not reuse the primary extraction on one side.

Generated disease is a cumulative clone/site picture, not the last event's changed sites.
A cleared lesion does not imply complete response while another known lesion is active.
Unknown initial extent is not evidence that all disease has cleared. No assertion about
host, toxicity or molecular realism enters a GENIE score when the real channel is absent.

### Observation alignment

Separate construction of a course from observing it. Keep a source-supported path,
evidence times, gaps, terminal event and follow-up boundary for each real donor. Align
donor time zero to that donor's own saved landmark, not the query's diagnosis or first
treatment. Never replace the donor's starting state with the query's state.

For disease comparisons, the query's scoring calendar contains time zero and its saved
assessment times, checked against the source presence/status records. Timed curated
new-site documentation is also an information update and must be represented consistently
in the real path and the scoring calendar. Record this calendar explicitly; it is
evaluation information, never new prompt input.

At a query observation time, read the forecast's latest supported state. Generated
asserted states can be read there without treating their event frequency as scan frequency.
A donor value must already be supported by that donor's record at that relative time.
Sampling it does not create a donor assessment or reset its source validity clock.
Unknown donor intervals, expiry and loss to follow-up remain gaps. Generated new-site
effects on disease state receive the same observation treatment, rather than bypassing
the calendar while response events wait for an assessment. Retain the original site
documentation channel separately for the site-specific secondary analysis.

The implementation must use one explicit observation adapter for these comparisons,
not three similar-looking derivations. Check it on small hand-written courses before any
cohort result. The adapter is not an assertion that observation schedules are independent
of disease: calendar-conditioned results retain that limitation.

### Death and censoring

The whole-course target remains `[0, H]`; a generated death cannot shorten it. Each
course stops accruing disease-state time, events and treatment at its own death. Survival
records the absorbing exit through H. Zeros after death in accumulated exposure do not
mean remission, and are not unknown follow-up. If a real patient dies early while a
forecast continues treatment, that extra predicted treatment remains an error.

Loss to follow-up before H does not resolve the complete H-month course. Keep partial
documentation for named diagnostics; do not substitute a shortened window into the
complete-horizon score. A future ending without a valid terminal record is unresolved,
not presumed alive. Honour the original acceptance decision while reporting any later
projection limitation separately.

## 5. Survival and existing scalar readouts

Retain the existing Cox covariates, cancer stratification, ridge setting and ten-fold
patient-level cross-validation. Freeze/persist fold IDs; preprocessing is fitted within
each training fold. Predictions are joined by patient label, never stratum-return order.
Add the three-month readout without changing model selection. No new tuning.

Donor survival forecasts are Kaplan-Meier estimates from the exact card members, with
their own residual survival/follow-up. LLM probabilities are the accepted fan's proportion
alive. Use the cross-validated per-cancer Kaplan-Meier floor where already specified.
If a donor curve has no support to H and has not reached zero, do not extrapolate its
last positive survival estimate: report that horizon unavailable for that comparison.

Report raw Brier score, predicted versus observed survival, reliability plots and
tie-preserving ECE. ECE is descriptive, not a proper score or the primary ranking metric.
Use at most five probability bins, never splitting identical predicted probabilities;
show bin counts and the binning rule. Keep a single RMST-to-24-month ranking statistic for
C-index comparisons and paired uncertainty; do not infer equivalence from overlapping CIs.

For censoring-adjusted Brier, use a cancer-stratified reverse Kaplan-Meier censoring
estimate on the fixed query cohort. Deaths by H use `1/G(T-)`, observed survival through H
uses `1/G(H-)`; earlier censoring contributes no observed outcome. Here G is the survival
function of censoring time, with left limits because a recorded last contact exactly at H
counts as observed through H. A death exactly at H counts as a death. Use the cohort count,
not the number of observed endpoints, as the denominator. Handle censoring/death ties
explicitly. Independent censoring within cancer is an assumption, not a demonstrated fact.
Report weight support and effective sample size. If any required G is below 0.05, do not
quietly cap weights and call the answer unchanged: flag that estimate unsupported and
retain the explicitly evaluable-subset result. No additional censoring model is fitted.

Report the existing finite-fan Brier correction for the LLM as a sensitivity,
`(p-y)^2 - p(1-p)/(m-1)`, under the independent-draw assumption. Do not apply it to the
fixed retrieved donor library or a Cox curve. For IPCW, weight the corrected individual
loss by the same censoring weights. Preserve negative corrected values, if any.

Treatment scalar comparisons use forecast means versus observed values on identical
complete-horizon patients, in original units: months on any systemic therapy and number
of subsequent recorded regimens. Do not call regimens lines of therapy. A censored donor
cannot contribute a complete treatment total. More detailed treatment readouts below
must not be represented as exact-drug or guideline concordance.

## 6. Fair disease-course diagnostic with donors

For query q and each saved donor d, form the relative-time set W on which q and the
calendar-aligned donor are supported, inside `[0, H]`. Include known post-death time as
the absorbing exit; exclude genuinely unobserved living time. Require at least one month
of comparable time. W is fixed before looking at any LLM future, and is identical for
all LLM arms compared against that donor.

On W compute occupancy total variation and time-aligned state disagreement for both
the donor and every LLM repeat against q. The four living states and death are distinct;
death is an exit, not a fifth disease state. Unresolved generated time inside W is
charged as disagreement and reported separately; it never reduces W. The occupancy
diagnostic uses a separate unresolved-output mass for this safeguard, not a real clinical
state or a coordinate in the joint score below.

Also report a secondary diagnostic excluding unresolved generated time from both sides
of each paired comparison. For each generated repeat use the intersection of W with its
resolved time, and evaluate that repeat and the donor on exactly that smaller interval.
Apply the same one-month minimum and report retained time, exclusions and results beside
the safeguarded diagnostic. This version conditions on the generated output being complete
and can reward silence; it cannot replace the primary diagnostic or determine eligibility
for it. Its purpose is to distinguish documented clinical disagreement from incomplete
generated output.

First average each LLM's repeat losses on the donor's W. Then average paired LLM-minus-
donor losses over the usable members of that card, then over patients. Require at least
seven usable card members for a patient-level diagnostic. Report usable donors, scored
time and missingness reasons; also report the subset with all twenty members usable.
This compares **documented donor examples on common intervals**, not a complete
twenty-patient predictive law. It cannot establish complete-course superiority. Generic
and matched cards are analysed separately; do not pool their different masks.

For four-arm comparisons without donors, the analogous window is the real query's
supported time, including known post-death time, identical across arms and repeats.
Retain the old intersection/shared-alive-window result only as a clearly labelled
historical sensitivity, not the main complete-course comparison.

Persistence is a disease-only baseline, not an implicit claim that the patient never
dies. If evaluated on the living-state diagnostic, restrict every compared method to
the same real-living documented intervals and name that conditional question. Do not
insert persistence into the full joint score without an actual survival/treatment forecast.

## 7. One joint observable-summary score

### Target and coordinates

The proposed common core has **17 coordinates in four blocks**, constructed at each H.
It excludes site-level anatomy from the common score because BRCA does not collect it.
The site channel remains an NSCLC/CRC secondary result with its own denominator.

| Block | Coordinates | Transformation before distance |
|---|---|---|
| Survival (2) | Alive at H; restricted months alive through H | Binary unchanged; months divided by H |
| Disease (4) | Months in NED, responding, stable, progressing | Each divided by H, not by a forecast-dependent known-time denominator |
| Events (2) | Documented entries into progressing and responding after the landmark | Each count n mapped to `n/(1+n)` |
| Treatment (9) | Months on any systemic therapy; four per-modality durations; subsequent regimen count; started-regimen modality-set change count; whether a first such change occurs; time to that change, capped at H | Durations and capped time divided by H; counts as above; indicator unchanged |

The four modalities remain chemotherapy, immunotherapy, targeted and endocrine. Exposure
is a union within each modality and within any-treatment time; overlaps between modalities
are permitted and must not be summed. Include treatment already active at the landmark,
stop all exposure at death, and do not count the index regimen as subsequent. Unmapped
therapy is not silently off-treatment or absent from every modality: unresolved modality
exposure makes that coordinate unavailable.

The change measure follows recorded regimen starts and their modality sets, not presumed
progression or a clinical line definition. Co-dated starts form one modality-set group;
their row order cannot create switches. Its first-change time equals H when no change
occurs by H, with indicator zero. A change at H has indicator one. Unknown/censored timing
is not encoded as no change. Report uncapped clinical units and raw counts alongside the
bounded transformations; these scales are transparent post-hoc choices, not learned biology.

For block b of dimension d_b, multiply its transformed coordinates by
`sqrt((1/4)/d_b)`. Concatenate blocks and use ordinary Euclidean distance, not squared
distance or an average of block scores. Each block has equal total squared-distance
weight. Do not estimate scales from pooled generated outcomes or tune weights by arm.

### Eligibility, with no favourable missingness rule

Forward persistence defines the primary joint-score population. Four-month expiry and
its conservative conflict rule define a separate sensitivity population. Report each
population's coverage, and compare scoring conventions on the patients and forecasters
eligible under both. A change in patient composition must not be presented as a change
in forecast performance.

The full-core real target requires known survival through H or a recorded death, a
documented disease state throughout its living window under the selected convention,
and interpretable treatment/event channels. Death resolves only the post-death remainder
without a disease assessment; every required pre-death coordinate must still be supported.
This is a complete-documentation subpopulation; its composition and
coverage must be shown. Do not impute intervals, drop coordinates per draw, or label
unknown as healthy to enter the score.

For each LLM cell require at least seven originally accepted repeats. Every accepted
repeat in that cell must project to the full core: if one is unresolved, mark the cell's
full-core score unavailable rather than discard that difficult draw. Pair all arms on
their common eligible queries and report the resulting selection. Missingness is not
solved by this restriction; interpretation is conditional on complete measurable forecasts.
The safeguarded diagnostic and validity/coverage results remain beside it.

For a donor full-core forecast, **all twenty saved card members** must resolve the same
coordinates through H. Do not remove censored or poorly assessed donors and renormalize
the rest as though they were the original card. Compare LLM and donor scores on that
common subset only. If this yields inadequate support, report coverage and omit the
joint donor claim; section 6 and valid scalar comparisons are the prespecified fallback.
No donor missingness model or survival-conditioned imputation is added. Equal-size
subsampling below applies only after all twenty donors meet these requirements; it cannot
rescue an incomplete card.

### Eligibility-only precheck before scoring

Before calculating joint scores, tabulate eligibility by horizon and cancer under forward
persistence, with exclusion reasons and without any outcome-distance or arm-effect values:

- real queries with all core coordinates resolved;
- LLM cells with at least seven accepted repeats, every accepted repeat fully resolved,
  separately by arm and on the four-arm intersection;
- donor cards with all twenty members resolved, separately for generic and matched cards;
- the intersection for the four-arm comparison, and for each card's corresponding LLM
  arm versus its direct-donor forecast.

Eligibility uses follow-up, death and documentation, not agreement between forecasts and
outcomes. Count deaths only as resolving their post-death portion. Do not estimate coverage
from survival alone or assume a valid ledger resolves every coordinate. Repeat the coverage
table under expiry so the common-convention population is explicit.

Freeze these counts before computing scores. If a horizon has fewer than twenty paired
patients for a joint donor comparison, apply the section 6/scalar fallback for that target
and horizon now. Cancer-specific inference uses the same twenty-patient floor. Do not
relax the all-twenty rule after seeing coverage or score differences. An unavailable target
still receives a row and its reason. CC reviews this precheck before full recomputation.

### Population composition beside every joint-score table

For each horizon and comparison set, report its patient count, cancer mix, observed
deaths by H, documented survival through H and unresolved survival status, against the
same counts in all 450 queries. Include six-month observed death counts and fractions for
both populations, labelled as a composition descriptor even in a shorter-horizon table.
Show denominators and censoring rather than treating unresolved patients as survivors.

Complete documentation may select a different mix of deaths and survivors; it does not
necessarily enrich deaths, because pre-death documentation is also required. Interpretation
of joint scores is conditional on the displayed population. These descriptors accompany
primary, sensitivity and donor-comparison tables, not only the overall coverage report.

### Score and its limits

For an empirical forecast with vectors z_i, weights w_i summing to one and real vector y:

```text
ES = sum_i w_i ||z_i-y|| - 0.5 sum_i sum_j w_i w_j ||z_i-z_j||.
```

Use equal weights within each LLM fan or full donor card. This scores the finite forecast
actually available; donor members are not assumed independent. Report per-block energy
scores for interpretation, but do not sum them and call the sum a joint score.

Report two distinct questions:

1. **Forecast as delivered:** the formula above uses every originally accepted complete
   LLM repeat or every member of the complete twenty-donor card. Unequal sizes are part of
   what was delivered. This is a valid finite-forecast score, not an estimate free of
   ensemble-size effects.
2. **Forecast at equal size:** the primary comparison across forecasters gives each the
   same number of trajectories per patient. For the four-arm table, m* is the minimum
   accepted fan size across all four arms. For a donor comparison, m* is the smaller of
   the corresponding LLM fan and the complete donor card. Eligibility is determined first
   and does not change with m*.

For a forecast larger than m*, average the ordinary empirical score over 200 deterministic
random subsets of size m*, sampled without replacement within each subset. Repeated subsets
across draws are permitted. When a forecast already has size m*, use its complete score.
Use seed 20260928 and a stable hash of comparison family, query, forecaster and horizon;
do not use process-dependent Python hash values. Save the seed keys and m* in the private
analysis rows. Draw once and reuse the same subsets for joint and per-block scores and
throughout patient-level bootstrap resampling. This isolates forecast cardinality, not
donor selection, dependence, censoring selection or clinical comparability.

Four-arm m* is shared across all four contrasts for a patient; do not change it per arm
pair. Donor comparisons use the LLM arm that received that card: generic-cohort generation
versus generic donors, and matched generation versus matched donors. Other LLM-versus-donor
contrasts, if displayed, are descriptive. Present the as-delivered results alongside the
equal-size results rather than silently replacing the original forecast question.

Also report the LLM-only fair estimator, replacing the pairwise denominator m^2 by
m(m-1) and excluding self-pairs. It estimates the generator score under independent,
identically distributed accepted draws. Provider heterogeneity and the retry filter
qualify that interpretation. Do not make this correction for the donor library.

Energy score is proper for the fixed vector distribution. That theorem does not make
this vector the entire clinical path, nor remove informative documentation, censoring,
accepted-output selection or complete-case selection. Do not describe these restricted
empirical averages as an unbiased score of the population's full future law. The score
does not guarantee sensitivity to every dependence error and occupancy does not retain
order. Those are the reasons for the next checks.

## 8. Dependence and order checks

First, a deterministic synthetic fixture: two coordinates with identical marginals but
perfectly coupled versus oppositely coupled joint outcomes. Under the fixed distance,
the correctly coupled forecast must have lower average energy score over the known truth
distribution. A reversal of state order with equal occupancy must leave occupancy equal
and change time-aligned disagreement. These test the scorer, not an LLM hypothesis.

Then, within each eligible query-arm fan, permute complete disease/event and treatment
blocks independently across repeats, holding the survival block fixed. Keep disease and
its event counts together. Use 200 permutations, deterministic seed 20260928 keyed by
query/arm/horizon. All marginal block distributions must remain exactly unchanged.
Report shuffled minus original joint score, without retuning weights if it is small.
Shuffled combinations may be clinically incoherent; that is intentional for this
dependence diagnostic, not a new forecast arm or causal experiment.

Add a second version that permutes only within alive-at-H strata. It preserves the
association of each block with terminal survival status, but may still create impossible
combinations because death times within the deceased stratum can differ. Report this as
a survival-stratified dependence check, not a clinically coherent synthetic population.
Singleton strata stay unchanged; report their sizes and the number of fans in which any
permutation is possible. Marginal distributions must remain unchanged in both versions.
Run these diagnostics on the original complete fans, with original and shuffled fans of
identical size, and label them separately from equal-size forecaster comparisons.

One companion clinical timing diagnostic: time from the first post-landmark documented
entry into progression to the first subsequent systemic regimen start in the window.
Report whether both events occur, paired times only when observable, and the number of
courses where a new regimen precedes any documented progression. Do not call a regimen
change proof of progression. Do not omit absent event pairs and report timing alone.
No claim of biological onset precision follows from recorded detection times.

## 9. Retries, fans and inference

Primary forecasts remain the original fixed, at-most-three-attempt pipeline. First-attempt
validity is a separate endpoint on all requested slots. Inspect the first entry of
`attempts_detail`, including its acceptance flag and raw response, rather than treating
`attempts_used == 1` alone as proof of validity. Never substitute the successful retry
for a failed first attempt. Missing raw first responses are reported as unavailable;
do not reconstruct them or silently classify them as clinical failures.

First-attempt clinical sensitivity uses originally valid first responses only, with the
same projection rules and seven-repeat cell floor. Compare pipelines on shared patients
and report their separate maximal populations. A descriptive two-or-more-repeat table
may additionally show the extent of the floor restriction, but is not a replacement
primary result. No refilling, smoothing or duplicated futures to reach a floor.

Use 2,000 patient-level paired bootstrap resamples, stratified by cancer, seed 20260928.
Keep each patient's arms, repeats, masks and fixed donor cards together. Report mean loss
differences and percentile 95% intervals. For survival IPCW, re-estimate censoring weights
inside resamples; do not refit Cox or retrieval. These intervals condition on the fitted
out-of-fold forecasts and shared donor library; they do not measure library/model-training
uncertainty. Report cancer estimates, eligible n and exclusions even when sparse. No
inferential conclusion for a subgroup with fewer than twenty paired patients.

Secondary two-sided paired Wilcoxon tests remain distinct from CIs for a mean: they test
different features of the paired distribution. For the new joint-score table, show raw
p-values and Holm adjustment across its four arm contrasts at four horizons (16 slots;
unavailable slots remain counted). A separate family contains the generic LLM-versus-generic
donor and matched LLM-versus-matched donor contrasts across four horizons (eight slots;
unavailable slots again remain counted). Both families use the equal-size comparative
scores as primary. As-delivered, LLM fair-score, per-cancer, component and observation-rule
sensitivity tables report effect sizes and intervals as exploratory checks, not additional
independent discovery claims. Do not characterize an effect as absent or equivalent from
a nonsignificant test.

No experiment is repeated merely to narrow an interval. Observed 24-month evaluability,
valid-ledger fan size, full-core evaluability and time coverage are separate quantities.
Nested-horizon trends use the same patients and dimensionless scaling, with raw-unit
component results retained. Do not infer a trend from different complete-case populations.

## 10. Required checks before cohort recomputation

| Fixture or invariant | Required result |
|---|---|
| Real progression at six months; generated death at three | H remains fixed; death/exposure differences remain scored |
| Censoring at month eight of H=12 | Full target unavailable; no alive-at-12 or completed treatment total |
| Donor first assessed at month five | No invented donor state at query visits before five |
| Donor evidence expires, then sampled at a query visit | Sampling does not renew the evidence clock |
| Later NED versus earlier progression, and reverse order | Strictly later evidence wins; same-time contradiction unresolved |
| Presence-only note after stale pre-landmark progression | No revival of discarded dynamics |
| Identical courses on identical calendars | Zero course discrepancy; horizon prefix invariance |
| Generated silence over a documented progression interval | Diagnostic charges the silence; joint cell not silently filtered |
| One clone/site clears while another remains active | No complete-response assignment; ordering of same-event entries invariant |
| Overlapping systemic therapies; a regimen active at time zero | Union duration, no missing active exposure, no index-as-subsequent |
| Same-month starts and same-day stop/start | Row-order invariant; no fabricated sequential regimen change |
| BRCA site detail absent | Site score unavailable, not zero; common core unchanged |
| No modality change versus change exactly at H | Distinguished by the occurrence indicator |
| Correct joint dependence versus shuffled dependence | Synthetic fixture discriminates; marginal distributions unchanged |
| All responses rejected, or a horizon absent from projections | Full requested grid and failure/coverage rows still emitted |
| First attempt fails and retry passes | Counted once as retry success, never first-attempt success |
| Censoring/death tie and support ending before H | Defined IPCW limits; no unsupported positive-tail extrapolation |
| A delivered forecast larger than the common m* | Hand-enumerated small-example subset scores agree; same subsets used for core and blocks |
| Four arms with different accepted fan sizes | One common m* for all four contrasts; original eligibility and delivered scores retained |
| One unresolved member of a twenty-donor card | Card unavailable before subsampling; no rescue by selecting complete members |
| Death with undocumented pre-death state time | Death alone does not make the full target eligible |
| Alive-at-H-stratified shuffle | Marginals and survival strata preserved; singleton strata unchanged; differing death times still flagged as a limitation |
| Unresolved generated time excluded in the secondary diagnostic | Donor and generated repeat use the same reduced interval; primary window and eligibility unchanged |

These checks are acceptance criteria for implementation, not claims that the current
code already passes. Correct code failures before extracting results; document any
scientific-definition change as an amendment rather than a silent repair.

## 11. Implementation boundary and hand-off

Before gate 4, record the minimum dependency surface in `docs/pipeline_scope.md`.
The current four-arm stack is still listed there as pending classification. Promote
only what the approved recomputation requires; do not port or extend the older PFS
geometry scorer simply because it already contains an energy-score function.

Read/reuse candidates: `canonical_future_v1.py`, `generated_future_v1.py`,
`genie_operator_registry_v2.py`, their real-data loaders, and the frozen query/prompt
readers. Existing four-arm extraction, baseline and readout scripts provide provenance
and reference definitions; they must not become parallel competing scorers. List the
actual dependency closure before code changes. Generation-facing code stays untouched.

Expected new implementation surface, subject to that classification: one read-only
`scripts/score_four_arm_final_v1.py` entry point, a small testable scoring module if
needed, and focused regression tests. It consumes the frozen run; it cannot call a
provider or rebuild prompts. Scope approval for gate 4 does not authorize a repository-
wide refactor. Changes to shared core require the appropriate broader test tier.

Write new outputs only to the persistent, gitignored
`outputs/four_arm_preprint_20260928/`; preserve all earlier exploratory outputs. Save
input/specification/code hashes, explicit conventions, failure reasons, private paired
patient rows, aggregate tables and a reconciliation note for superseded figures. No
patient identifiers, raw GENIE cards or raw responses enter tracked/public artifacts.

Expected aggregate products:

- `coverage.csv`: requested slots, attempts, acceptance, usable fans, truth support and
  full-core support, by arm/cancer/horizon and analysis population.
- `eligibility_precheck.csv`: the frozen eligibility-only counts, exclusions, complete-card
  intersections and horizon-specific fallback decisions, written before any score table.
- `population_composition.csv`: patient count, cancer mix, observed death, survival and
  unresolved status at H, and six-month deaths for each analysis set and the full cohort.
- `survival.csv`: probabilities, Brier/IPCW, ECE, ranking and paired comparisons.
- `course_diagnostics.csv`: symmetric donor/LLM and four-arm documented-time comparisons.
- `joint_scores.csv`: core/block scores, full-card donor subset, contrasts and uncertainty.
- `sensitivity.csv`: conservative reading, first-attempt results and shared-population
  comparisons; dependence/order checks included with their denominators.
- `manifest.json` and `RECONCILIATION.md`: what was run, what was not estimable, and why
  an old number differs. Empty targets still get an explicit non-estimable row.

CC reviews this specification before implementation, then reviews fixtures and a small
source-to-score inspection before the complete extraction. After independent result
review, CC alone promotes numbers to `CLAIMS.md`, the manuscript and figures. Codex
does not update those files with provisional scores.

At the day-three scope checkpoint, inadequate joint-donor support triggers the already
approved fallback, not relaxed missingness rules. Report valid component comparisons
and the restricted joint LLM result where available, without claiming superiority over
donors for a complete course. A failure to validate the joint target itself means no
joint-score claim in the first preprint. Known errors in retained analyses still block it.

## Method references for review

- Gneiting and Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).
- Ferro (2014), [Fair scores for ensemble forecasts](https://doi.org/10.1002/qj.2270).
- Gerds and Schumacher (2006), [Consistent estimation of the expected Brier score in general survival models with right-censored event times](https://doi.org/10.1002/bimj.200610301).

The score formulas and their assumptions are supported by these methodological sources;
the coordinate selection, eligibility, masks, diagnostic windows and submission fallback
above are this project's post-hoc choices. Bibliography integration belongs to CC's
release work, not an unverified citation insertion into the manuscript here.
