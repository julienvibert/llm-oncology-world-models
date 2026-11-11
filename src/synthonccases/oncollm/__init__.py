"""OncoLLM-Atlas — successor pipeline to atlas-v1.

Scope: all-stage, OncoTree-native, complete-trajectory, counterfactual-ready
synthetic patient generation. Operationally distinct from `synthonccases.atlas`
which encodes the metastatic-only, 15-cancer, follow-up-windowed atlas-v1
assumptions and is treated as legacy reusable machinery.

The design documents of this package are internal working notes of the development
repository and are not part of the release.

Public surface:
  - schema: load + validate against the OncoLLMRecord_v1 JSON Schema
            (Draft-07 + FormatChecker; 'format: date' actually fires).
  - prompt: load the unified prompt template (minimal / guided) + compute
            prompt_hash; build the input_payload dict; infer mode from
            payload.
  - roster: load the materialized OncoTree roster + select included codes
            + indexed lookups by code (tissue, main_type).
  - constraints: build + validate `conditioning_constraints` payloads
                 (strict identity allow-list; sub-schema-checked).
  - validator (Phase 1.4): cross-block audit + XREF carry-forward over
            OncoLLMRecord_v1. ValidationIssue + ValidationReport (frozen);
            collect-all-failures; ERROR / WARNING severity.
  - payload (Phase 1.5a): pure composer of provider-facing system+user
            prompt pair plus the canonical conditioning_constraints block
            the runner injects back before validation.
  - cf_seed (Phase 1.5b): deterministic identity_card field reset for
            CF re-feed. Pure transformation; does NOT validate. Spec at
            docs/cf_seed_reset_v1.md.
  - provider (Phase 1.5c): OncoLLM-native provider layer. Thin wrapper
            over synthonccases.generate.providers.OpenRouterProvider; owns
            schema lowering (strips pipeline-owned blocks), strict
            response_format construction, JSON parse + metadata mapping.
            ProviderParseError carries raw + usage so the runner's
            failed-attempts log keeps full fidelity.
  - runner_types (Phase 1.5d-1): frozen RunContext, SampleSpec, CellSpec,
            CellSummary dataclasses. Pure types; behaviour in 1.5d-2/3.
  - lockfile (Phase 1.5d-1): write-once + drift-detection lockfile
            lifecycle. Re-skins atlas semantics with OncoLLM-native fields
            (schema/lowered-schema/prompt/roster hashes, structured-output
            mode, validator version). atlas.lockfile is NOT imported —
            its validator encodes atlas-v1 clinical fields with no
            analogue here. atlas.io + atlas.hashing are reused directly.
  - attempt (Phase 1.5d-2): per-attempt generation. derive_sample_id,
            derive_record_id, AcceptedAttempt, FailedAttempt,
            generate_one_attempt. Pure function (no disk I/O); composes
            payload, calls provider, injects pipeline-owned blocks, runs
            validator, runs CF identity-equality check (byte-equal vs
            reset seed; no masking).
  - runner (Phase 1.5d-3): resume-safe run_cell orchestration. RunPaths,
            ResumeState, run_cell. Writes accepted CPT + sidecar +
            raw_response + optional projection row in commit order
            (sidecar last as commit marker); appends failed-attempt rows
            under fcntl flock; lockfile lifecycle via oncollm.lockfile;
            thread-local provider pool so a fully-resumed run never
            instantiates a provider client. Reuses atlas.io for atomic
            JSON writes and JSONL append-under-flock — these are generic
            primitives, not atlas-v1 clinical machinery.

Stubs awaiting code-change pass:
  - projection: derived analysis views on OncoLLMRecord_v1 instances.
  - smoke: Phase 2 cost-calibrating smoke design.
"""
