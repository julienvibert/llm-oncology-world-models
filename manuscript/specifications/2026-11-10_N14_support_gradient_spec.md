<!-- Provenance header added on 11 November 2026; everything below the line is the specification as committed in f565de0 (development repository), unedited. -->

Specification of the analysis in Supplementary Fig. 16, kept verbatim as written on 10 November 2026. Mentions of Codex refer to the AI assistant used for code and statistical review, and Julien to the first author (Methods, Use of large language model assistants). The SHA-256 recorded in the analysis manifest is that of the file without this header.

---

# N14 specification: does the gain from matched examples depend on how closely they match? (draft for review, not run)

Written 10 November 2026, before any result. Status: **specification only**. No outcome has been read for this question. To be reviewed by Codex and Julien before computation. Exploratory and post hoc with respect to the registry experiment; it will be reported as such, whatever the direction of the result, or not reported.

## Question

In the GENIE-BPC experiment, does **matching itself** (matched rather than generic examples from the same cohort) help more when the 20 matched example patients agree more closely with the patient on the matcher's clinical criteria? This tests an uncertainty in the link between the conditioning experiment (Fig. 4, Fig. 5) and the ladder of comparable patients (Fig. 6d).

Three questions are distinguished: (1) are forecasts worse for patients with weaker matches (could only reflect harder patients; not tested); (2) does adding matched examples help less when matches are weaker (secondary); (3) does matching itself, beyond providing cohort examples, help more when matches are stronger (**primary**). The quantity is always a **paired gain** for the same patient. Wording: "weaker clinical agreement", never "farther away" or "lower density".

Amended 10 November after Codex's review, before any computation: primary contrast changed to matched versus generic; secondary analyses reduced to one; tie handling, missingness and reporting commitment added.

## Matching measure (generation-time, verified 10 November)

Source: `outputs/exploratory_query_universe/queries.jsonl`, written when the queries and donor cards were selected, before generation (the four-arm prompts are built from it: `scripts/explore/emit_four_arm_prompts_v1.py`, `four_arm_baselines_v1.py`, `four_arm_deep_extract_v1.py`). Aggregate checks without identifiers: 450 queries (150 per cancer), all `contrast_eligible`, `support_boundary` present for all, candidates comparable on cancer, treatment modality set and era from 43 to several hundred (median 282).

Fields, as defined in `scripts/explore/build_query_universe_v1.py` (lines 575-640):

- `support_boundary`: the clinical agreement score (`card_match.similarity`, the matcher used to rank donors) of the **20th, weakest admitted matched donor**. Higher means closer agreement. It is a signed agreement score, not a distance or radius (an earlier reading as a "support radius" was withdrawn, BLUEPRINT section 1). This is the **primary measure**: how far the matcher had to reach to fill the card.
- `mean_agreement_relevant`: mean agreement of the 20 matched donors (card quality). Secondary.
- `n_candidates`: number of donors comparable on cancer, treatment modality and era before ranking. Descriptive only.

To verify before computation: (1) that the hash of `queries.jsonl` matches the one recorded in the generation manifest of the four-arm run, so the measure is the one used to build the prompts; (2) how unknown values and any molecular term enter the **generation-time** `similarity` code (the query-builder comment on an additive molecular adjustment may be stale; the active matcher is reported to average signed agreement over a fixed set of criteria, unknowns contributing zero). Verify from the code at the generation commit, not from comments. The analysis is done within cancer (ranks within cancer). Within-cancer ranks do not remove differences in missingness: for each tertile, the share of matching criteria unknown for the patient is reported, and the interpretation states that weaker agreement can partly reflect sparser documentation.

## Population, outcome and contrast (one primary)

- **Primary outcome:** the per-patient whole-course joint score at **12 months** (energy score on 17 summaries, lower is better), primary reading, equal-size fans, exactly as in Fig. 5d (233 patients). Source: `outputs/four_arm_preprint_20260928/private/joint_patient_rows_scores_20261012.csv` (patient level, never exported).
- **Primary contrast:** gain = score(generic examples) minus score(matched examples), per patient; positive means matching helped beyond cohort examples. The population is the 233 patients evaluable at 12 months, and conclusions refer to this selected population.
- **Primary statistic:** Spearman correlation between the gain and `support_boundary` ranked within cancer (rank divided by cancer size), pooled over the three cancers. 95% interval from 2,000 patient bootstrap resamples stratified by cancer (seed 20261110). Two-sided P from 2,000 permutations of the support ranks within cancer.
- **Display (descriptive, fixed now):** mean gain with bootstrap interval in within-cancer thirds of `support_boundary`. The score is discrete: thirds are formed with average ranks so that patients with the same score are always in the same group (group sizes may then be unequal and are reported). Spearman correlation uses average ranks for ties.

## Secondary (one only)

Matched examples minus the record alone, same outcome, horizon and statistic (question 2).

The survival Brier result (generic and matched similar at 12 months) concerns a different outcome; N14 is not presented as explaining it.

## Not claimed

- The measure describes agreement on the matcher's criteria among donors that already share cancer, treatment modality and era. Every patient had at least 43 such candidates, so the experiment covers **weaker matching, not absence of comparable patients**. It cannot test the empty end of the ladder of Fig. 6d.
- A correlation does not show that match quality causes the gain; patients with weaker matches may differ in other ways (rarer regimens, longer histories). Cancer is handled by within-cancer ranks; nothing else is adjusted.
- No threshold, measure or horizon will be chosen after seeing results. A null or reversed result is reported the same way.

## Outputs (aggregate only)

`outputs/exploratory_n14_support_gradient/`: correlation, interval and P for primary and secondary analyses; tertile means with intervals and counts (each tertile about 25 patients per cancer, no cell below 5); manifest with input hashes, including the hash of `queries.jsonl`. Script `scripts/explore/n14_support_gradient_v1.py`, with a unit test on constructed data for the within-cancer ranking and the paired gain.

## Effort and reporting

About two hours of code and checks from saved files, no generation. Not required for submission. **Reporting is fixed now, whatever the direction:** one Supplementary panel and one sentence (ladder section or Methods), labelled exploratory and post hoc, with the population (233 patients) and the limits above.
