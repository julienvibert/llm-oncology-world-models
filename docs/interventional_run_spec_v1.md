# Interventional run — v1 launch spec (surgical POC)

**Status:** costed launch plan, NO generation yet. Companion to `interventional_field_design_v1.md`
(the full design) and `paper_figure_plan_v1.md` (F3/F4/F5). Branch `feat/interventional-f345`.

**Discipline:** one controlled perturbation POC, NOT an exhaustive treatment simulator. Every line
below is a scope limit. If a temptation isn't in this doc, it's out for v1.

## Phased, gated — do the "simple" (pre-cutoff) trial FIRST

The post-cutoff trial (F4) is a *stress test*; there is no point paying for it if the machinery
fails on a KNOWN effect. So v1 runs in two gated phases, and **Phase 2 only happens if Phase 1
clears its gate.** The post-cutoff trial pick is therefore deferred — not needed yet.

### Phase 1 — pre-cutoff calibration + effect field (F3 + F5)
- **Contexts (2):**
  1. **EGFR-mut LUAD, 1L** — positive case (FLAURA anchor: osimertinib ≫ chemo). Feeds F5.
  2. **EGFR-WT LUAD, 1L** — **negative control**: targeted therapy must NOT redirect.
- **Arms (4):** osimertinib (matched) · carboplatin+pemetrexed (alternative) · pembrolizumab
  (wrong-target specificity control) · BSC (control for Δv).
- **Grid:** 5 current states × 3 histories per context (clinically sampled, then phase-space
  coverage-checked). History fixed to canonical in the main atlas; 3-variant only in the R4 panel.
- **Model:** Sonnet only. **Futures:** 3. **Endpoint:** active-arm forced-line PFS primary;
  BSC/supportive scored by treatment-agnostic disease-worsening time (first post-cutoff
  progression; death fallback if no progression is emitted); OS secondary.
- **Volume/cost:** ~2 × (5×3×4×3) = 360 trajectories, **~$87** accepted-future estimate
  (~$105 with 20% retry cushion).
- **Figures:** F3 trust-map scorecard (direction/ranking) + F5 phase-space effect field & the
  "same dot, different history" R4 panel.

**Supportive/BSC arm contract.** The BSC arm is not a recommendation task. It is
a forced supportive-care-only counterfactual. Every BSC cell carries a fixed
decision cutoff, including immediate-decision cells with an empty fixed prefix,
and cells with a history prefix place the decision cutoff strictly after the
last fixed history event. Active treatment arms start the assigned treatment at
the same cell-wide decision cutoff. Hard acceptance-audited constraints forbid
`systemic_treatment_start`, `systemic_treatment_switch`, and
`systemic_treatment_stop` after cutoff. The model should still emit disease
evolution, patient-state changes, local palliation, supportive medications or
procedures, hospice/palliative transitions, and terminal events when clinically
appropriate.

### PHASE-1 PASS/FAIL RUBRIC — pre-registered BEFORE generation (no post-hoc edits)

Written in plain terms and locked before any paid call, so the verdict cannot be rationalised after
seeing the output. The gate must be mechanically computable, not an eyeballed judgement. All five
checks must hold to proceed to Phase 2:

**Endpoint definitions.**
- Active forced arms: median `endpoint_rows.csv:pfs_line_1_days`.
- Supportive/BSC arm: treatment-agnostic disease-worsening time, defined as the first
  post-cutoff `progression` event minus the manifest-defined decision cutoff. If the model emits
  death without a prior progression, use death time as a conservative fallback and flag the
  trajectory.
- OS is secondary/descriptive and does not replace PFS/TTP in the hard gate.

1. **EGFR-mut, positive biology vs chemo:** median osimertinib forced-line PFS must exceed median
   chemo forced-line PFS.
2. **EGFR-mut, positive biology vs BSC:** median osimertinib forced-line PFS must exceed median
   supportive disease-worsening time.
3. **EGFR-WT specificity (the decisive control):** osimertinib must **not rank first** among active
   arms by median forced-line PFS. If wrong-target osimertinib ranks first in EGFR-WT → **STOP.**
   Do not spend on F4/F5 — this is a specificity failure to diagnose first.
4. **BSC control:** supportive disease-worsening time must be lower than the best active-arm PFS in
   each context.
5. **No impossible reversal:** median `(osimertinib PFS − chemo PFS)` must be larger in EGFR-mut
   than in EGFR-WT.

The phase-space/vector-field shift is still reported in F5, but it is descriptive unless converted
to a separate pre-specified scalar later. It is not part of the hard Phase-1 pass/fail gate.

Fail any → fix seeds/prompt/scoring before re-running; a Phase-1 failure ends v1 interventional
spend until diagnosed. This rubric is the gate — the figures (F3/F5) are only read if it passes.

### Phase 2 — post-cutoff virtual trial (F4) — ONLY IF Phase 1 clears
- **Context (1):** one post-cutoff metastatic trial. **[DECISION DEFERRED]** — pick after Phase 1
  passes. Requirements: after Sonnet's training cutoff · known individual drugs · clean metastatic
  PFS readout · ideally subgroups (for the F5 forest overlay). Candidates drawn from the verified
  anchor set (FLAURA/ALEX/KEYNOTE-024/177, RCC, melanoma…) — choose one whose *trial* is post-cutoff.
- **Volume/cost:** ~1 × (5×3×4×3) = 180 trajectories, **~$44** accepted-future estimate
  (~$53 with 20% retry cushion), unless Phase-1 ambiguity justifies a higher `n`.
- **Figure:** F4 observed-vs-predicted scatter + best-arm rank; pre-cutoff calibration dots vs
  post-cutoff stress-test dots in one panel.

## Totals & stop rules
- **Phase 1 first:** ~$87 accepted-future estimate (~$105 with 20% retry cushion).
  **Phase 2 (conditional):** +$45–90. **Hard budget cap: $300.**
- Stop if: Phase-1 direction accuracy poor (fix seeds/prompt first) · negative control redirects
  (specificity failure) · cost projection exceeds cap.

## Explicitly OUT of v1 (scope creep guard)
4th/5th contexts (pancreas fit/frail, HGSOC) · multi-model phenotype atlas · exact-magnitude
calibration · held-out anything · precision-medicine atlas expansion · any hybrid generation.

## Reuse (no new engine)
`build_ovat_egfr.py` (seed battery + forced arms) · `composite_space.py` (phase space) ·
`fig5_cf_perturbation.py` (field figure) · sentinel-battery gates · frozen `poc_runner` generation
path (unmodified). Only new work: trial→context/seed mapping, the F3 scorecard, the F4 scatter.
