#!/usr/bin/env python3
"""Run a small anchored-PFS provider smoke.

This is the missing live-call bridge for the anchored PFS harness:

1. read ``queries.jsonl`` and the already-built ``prompts_<arm>.jsonl`` files;
2. compose the real frozen-runner prompt with one anchor block;
3. call one OpenRouter model once per trajectory sample;
4. validate through the existing ``classify_response`` gates;
5. extract PFS from the replayed ledger and write the scorer's JSONL contract.

It deliberately does not replace ``scripts/run_paper_v1.py``. That runner remains the
frozen generation surface; this script is a thin exploratory caller for a new experiment
whose prompt is composed by ``synthonccases.anchored.contract``.

Run:
  uv run python scripts/explore/run_anchored_pfs_smoke_v1.py --dry-run
  uv run python scripts/explore/run_anchored_pfs_smoke_v1.py --model deepseekflash
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthonccases.anchored import contract as C  # noqa: E402
from synthonccases.generate.providers import (  # noqa: E402
    DEFAULT_REQUEST_SEED,
    DEFAULT_REQUEST_TEMPERATURE,
    GenerationResult,
    get_provider,
)
from synthonccases.oncollm.conditioning_audit import compute_required_patch_leaves  # noqa: E402
from synthonccases.oncollm.condition_lower import lower_conditioning_cell  # noqa: E402
from synthonccases.anchored import endpoints, validate_future_ledger  # noqa: E402
from synthonccases.oncollm.poc_runner import (  # noqa: E402
    _SYSTEM_PROMPT, classify_response, strip_code_fence,
)

from score_anchored_pfs_v1 import quadrant  # noqa: E402

IN_DIR = ROOT / "outputs" / "exploratory_anchored_pfs_v1"
DEFAULT_RUN_ROOT = IN_DIR / "live_smokes"
MODEL_ALIASES = {
    "deepseekflash": ("deepseek/deepseek-v4-flash-0731", "deepseek_v4_flash_0731"),
    "deepseek_v4_flash": ("deepseek/deepseek-v4-flash", "deepseek_v4_flash"),
    "deepseek_v4_flash_0731": ("deepseek/deepseek-v4-flash-0731",
                               "deepseek_v4_flash_0731"),
    "gemma4": ("google/gemma-4-31b-it", "gemma4"),
}
DAYS_PER_MONTH = 30.4375


def _load_queries(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run build_anchored_pfs_arms_v1.py first.")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row["query_id"])
        if query_id in rows:
            raise ValueError(f"duplicate query_id in {path}: {query_id}")
        spec = lower_conditioning_cell(
            conditioning_payload=row["conditioning_payload"])
        rows[query_id] = {"raw": row, "spec": spec}
    return rows


def _load_prompt_payloads(in_dir: Path, arms: list[str]) -> pd.DataFrame:
    frames = []
    for arm in arms:
        path = in_dir / f"prompts_{arm}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        frame = pd.read_json(path, lines=True)
        if "arm" in frame and not frame["arm"].eq(arm).all():
            got = sorted(frame.loc[~frame["arm"].eq(arm), "arm"].astype(str).unique())
            raise ValueError(f"{path} contains rows for the wrong arm: {got}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    needed = {"query_id", "arm", "anchor_block"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"prompt payloads are missing columns: {sorted(missing)}")
    return out


def _select_queries(payloads: pd.DataFrame, *, query_count: int) -> list[str]:
    """Prefer one R4 query from each support/roughness quadrant, then fill."""
    r4 = payloads[payloads["arm"].eq("R4_same_action")].copy()
    if len(r4) and {"n_anchors", "outcome_dispersion"} <= set(r4.columns):
        r4["quadrant"] = quadrant(
            r4["n_anchors"].to_numpy(float),
            r4["outcome_dispersion"].to_numpy(float),
        )
        picks = []
        priority = [
            "high support, rough — TARGET",
            "high support, flat — kNN suffices",
            "low support, rough — risky",
            "low support, flat — extrapolate",
        ]
        for name in priority:
            sub = r4[r4["quadrant"].eq(name)]
            if len(sub):
                picks.append(str(sub.iloc[0]["query_id"]))
            if len(picks) >= query_count:
                return picks
        for q in r4["query_id"]:
            q = str(q)
            if q not in picks:
                picks.append(q)
            if len(picks) >= query_count:
                return picks
        return picks
    return [str(q) for q in payloads["query_id"].drop_duplicates().head(query_count)]


def _compose(anchor_block: str, spec: Any, contract: str) -> str:
    # Under `continuation` there is no baseline_patch, so there are no required leaves to
    # ask for; compose_prompt enforces that too, and passing them here would compute a list
    # the template no longer has a section for.
    required = (() if contract == "continuation"
                else compute_required_patch_leaves(spec.baseline_seed))
    text = C.compose_prompt(
        anchor_block,
        contract=contract,
        baseline_seed=dict(spec.baseline_seed),
        pinned_paths=spec.pinned_paths,
        required_patch_leaves=required,
        policy_text="",
        conditioning_spec=spec,
        terminal_contract="death_only",
    )
    if "{{" in text or "}}" in text:
        leftover = [ln for ln in text.splitlines() if "{{" in ln or "}}" in ln][:3]
        raise ValueError(f"unsubstituted placeholder reached provider prompt: {leftover}")
    return text


def _extract_pfs_months(parsed: Any, cutoff_date: date) -> float | None:
    """Earliest progression/death/last-follow-up strictly after the landmark."""
    ledger = [
        ev.model_dump(mode="json", exclude_none=True)
        for ev in parsed.materialized.ledger
    ]
    for ev in ledger:
        raw = ev.get("date")
        if not raw:
            continue
        when = date.fromisoformat(str(raw))
        if when <= cutoff_date:
            continue
        if ev.get("type") in {"progression", "death", "last_followup"}:
            return (when - cutoff_date).days / DAYS_PER_MONTH
    return None


def _accept_continuation(raw_text: str, spec: Any) -> tuple[str | None, str | None,
                                                            dict[str, Any], list[str]]:
    """Acceptance for the continuation contract: a legal future, and its endpoint.

    Full replay is not applied. It would demand disease and host deltas on every event --
    the bar that forced the query patient's observed imaging progressions out of the fixed
    prefix while the anchors kept theirs. What the primary endpoint needs is exactly what is
    checked here: events strictly after the cutoff, in order, ending in one death, not
    restating what was already observed.
    """
    prefix = spec.observed_prefix
    if prefix is None:
        return "endpoint_extraction", "query has no observed-prefix cutoff date", {}, []
    try:
        payload = json.loads(strip_code_fence(raw_text))
    except (json.JSONDecodeError, ValueError) as exc:
        return "unparseable_json", f"{type(exc).__name__}: {exc}", {}, []
    fixed = [ev.model_dump(mode="json", exclude_none=True) for ev in prefix.events]
    result = validate_future_ledger(payload, cutoff=prefix.cutoff_date, prefix=fixed)
    if not result.accepted:
        return result.failure_kind, result.detail, {}, []
    # ENDPOINT-SPECIFIC VALIDITY. The generator is asked for the whole future to death and
    # every trajectory it gives is kept as one. What it is USABLE for is judged per
    # endpoint: a course that reaches progression and is then cut off has resolved PFS and
    # not OS. Scoring each trajectory only for what it resolved is what keeps the PFS
    # benchmark unbiased -- one bar for all endpoints discards long courses more often than
    # short ones, and length is the quantity being measured.
    e = endpoints(result, cutoff=prefix.cutoff_date).to_dict()
    if not e["pfs_valid"]:
        return "endpoint_extraction", e["reason"], e, list(result.warnings)
    return None, None, e, list(result.warnings)


def knowledge_probe(model_id: str, questions: dict[str, dict[str, str]]) -> list[dict]:
    """Ask the model, in words, what it believes about each sentinel rule.

    Two failures look identical in the sentinel table: a model that does not know the rule,
    and a model that knows it and does not apply it while generating. The first is a
    knowledge limit; the second is a dissociation between stating a fact and simulating
    under it, and it is the more interesting finding. This separates them for the price of a
    few short calls.

    The answers are recorded VERBATIM and never graded automatically. An automatic grader
    would be a language model marking a language model on the question at issue.
    """
    rows: list[dict] = []
    for pair, forms in sorted(questions.items()):
        for form, question in sorted(forms.items()):
            started = time.perf_counter()
            try:
                got = _call_provider(model_id, str(question).strip())
                rows.append({
                    "pair": pair, "form": form, "question": str(question).strip(),
                    "answer": got.text, "model_id": model_id,
                    "response_model": got.response_model, "response_id": got.response_id,
                    "reported_cost": got.reported_cost,
                    "duration_s": time.perf_counter() - started,
                    "graded": False,
                    "note": "verbatim; read by a human, never scored automatically",
                })
            except Exception as exc:                                    # noqa: BLE001
                rows.append({"pair": pair, "form": form,
                             "question": str(question).strip(), "answer": None,
                             "error": f"{type(exc).__name__}: {exc}", "graded": False})
    return rows


#: The provider contract the C/X/H/A battery already runs under. Matched here deliberately:
#: the preregistration claims the two experiments share a contract so their comparison is about
#: constructed-versus-real patients, and a different provider contract would silently make it
#: about the contract instead. `require_parameters` additionally makes OpenRouter refuse a
#: provider that will not honour the decoding settings -- which matters, because reasoning
#: tokens leaked past `forced_off` on the battery run.
class _UseClassifyResponse(Exception):
    """Control-flow marker: fall through to the non-continuation branch."""


_RESPONSE_FORMAT = {"type": "json_object"}
_REQUIRED_REASONING_MODE = "forced_off"
_PROVIDER_LOCAL = threading.local()

#: EXCLUSION, not pinning. On the first GENIE partial one backend (atlas-cloud) returned
#: reasoning tokens past `forced_off` on 161 of 201 attempts. That is demonstrated and specific,
#: so it is excluded by name. The stronger remedy -- pinning a single backend -- was considered
#: and rejected: the arm-differential backend mixture that would have justified it is NOT
#: established (chi-square p = 0.29 over 23 queries), prompt length barely differs by arm
#: (1.1x, so arm is near-invisible to a length-sensitive router), and `only` with
#: allow_fallbacks=False would make one backend's uptime a single point of failure across a
#: ~12 h run while diverging from how every prior run in this project, including the C/X/H/A
#: battery this one is meant to be comparable with, was served.
#: `data_collection`/`zdr` declare the retention terms the amended governance statement commits
#: to, so the claim is enforced at the request rather than asserted in prose.
_DEFAULT_PROVIDER_POLICY: dict[str, Any] = {
    "only": ["novita", "siliconflow", "deepinfra"],
    "allow_fallbacks": True,
    "data_collection": "deny",
    "zdr": True,
}

#: Raised from the class default of 16,384. Truncation is CORRELATED WITH THE OUTCOME here:
#: the contract asks for the whole course to death, so a long survivor costs more tokens than
#: a short one and is rejected more often. Rejecting long courses preferentially is
#: informative censoring on the endpoint being scored.
_OUTPUT_TOKEN_CAP = 24576


def _registered_provider_policy(prereg_doc: dict[str, Any]) -> dict[str, Any] | None:
    """The provider policy currently in force, or None if none was registered.

    THE LAST policy amendment wins, so a later dated amendment supersedes an earlier one.
    That is what makes the A3 red branch executable: switching to a pinned rerun needs a new
    amendment and nothing else -- no constant to edit, no flag to remember, and no way for
    the code and the document to disagree about which policy is in force.
    """
    policy = None
    for amendment in prereg_doc.get("amendments") or []:
        found = (amendment or {}).get("provider_policy")
        if found:
            policy = dict(found)
    return policy


def resolved_provider_policy(args) -> dict[str, Any] | None:
    """The routing policy this invocation will actually send.

    Read from the preregistration, which is the source of truth. `_DEFAULT_PROVIDER_POLICY`
    is only the fallback for an invocation with no prereg on hand, and enforce_definitive
    checks the two against each other so a stale constant is caught rather than ignored.
    """
    if getattr(args, "no_provider_policy", False):
        return None
    prereg = getattr(args, "prereg", None)
    if prereg is not None and Path(prereg).exists():
        registered = _registered_provider_policy(yaml.safe_load(Path(prereg).read_text()))
        if registered is not None:
            return registered
    return dict(_DEFAULT_PROVIDER_POLICY)


def interleave_arm_blocks(tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Shuffle the ORDER of each query's arm blocks, keeping every arm's repeats contiguous.

    Submitted in fixed R3 -> R4 -> R6 order, the arms occupy different wall-clock windows, and
    with free routing a backend whose capacity varies through the night would be associated
    with arm for a reason that has nothing to do with the experiment. That is the confound the
    exclusion policy does NOT remove, so the design removes it instead.

    Repeats stay contiguous within an arm block so a query's prompt prefix is still reused by
    the provider cache -- 85% of input tokens were cache reads on the partial, and scattering
    the repeats would spend that for nothing. Seeded, so a resume reproduces the same plan.
    """
    import random

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for task in tasks:
        grouped[task["query_id"]][task["arm"]].append(task)

    out: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        blocks = [grouped[query_id][arm] for arm in sorted(grouped[query_id])]
        # A per-query stream so adding a query cannot renumber another query's draw.
        rng = random.Random(f"{seed}:{query_id}")
        rng.shuffle(blocks)
        for block in blocks:
            out.extend(block)
    if len(out) != len(tasks):
        raise AssertionError("interleaving changed the task count")
    return out


def normalize_provider(name: str) -> str:
    """Fold an OpenRouter DISPLAY name onto its slug.

    Routing policies are written in slugs (`atlas-cloud`, `sail-research`) and responses
    carry display names (`Atlas Cloud`, `Sail Research`). Comparing the two raw would let an
    excluded backend serve a call and pass the gate, which is the one thing the gate exists
    to prevent.
    """
    folded = str(name or "").strip().lower()
    for ch in (" ", "_", "."):
        folded = folded.replace(ch, "-")
    while "--" in folded:
        folded = folded.replace("--", "-")
    return folded.strip("-")


def _provider_verdict(policy: dict[str, Any] | None, served: str) -> tuple[str, str] | None:
    """Judge one call's serving backend against the routing policy.

    Two rejections, and the second is the one that looks optional and is not. An UNNAMED
    backend cannot be checked against the exclusion list, cannot support the arm-balance
    diagnostic, and contradicts the governance statement's promise that provider identity is
    recorded per call. An exclusion that cannot be verified has the same defect as an
    unverified pin.
    """
    if not policy:
        return None
    if not served:
        return ("provider_unnamed",
                "the response named no serving backend; provider identity is recorded per "
                "call and an unverifiable exclusion is not an exclusion")
    lowered = normalize_provider(served)
    banned = {normalize_provider(name) for name in policy.get("ignore") or []}
    if lowered in banned:
        return ("provider_excluded_violation",
                f"policy excludes {sorted(banned)} but the call was served by {served!r}")
    allowed = {normalize_provider(name) for name in policy.get("only") or []}
    if allowed and lowered not in allowed:
        return ("provider_pin_violation",
                f"policy pins {sorted(allowed)} but the call was served by {served!r}")
    return None


def _usage(generation: GenerationResult) -> dict[str, Any]:
    """Provider-returned request and usage facts for one API response."""
    return {
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "reasoning_tokens": generation.reasoning_tokens,
        "prompt_cache_usage_reported": generation.prompt_cache_usage_reported,
        "cached_input_tokens": generation.cached_input_tokens,
        "cache_write_tokens": generation.cache_write_tokens,
        "reported_cost": generation.reported_cost,
        "response_model": generation.response_model,
        "response_id": generation.response_id,
        # The backend that served the call. The first GENIE partial did not record it, though
        # it did record response_id, so its 909 attempts were attributable by resolving each id
        # through OpenRouter's generation endpoint -- a second API pass over every call.
        # Recording the name inline makes the run self-describing and, more importantly, lets
        # the pin be verified while it runs rather than afterwards.
        "served_provider": getattr(generation, "served_provider", "") or "",
        "retries": generation.retries,
        "max_tokens_sent": generation.max_tokens_sent,
        "max_tokens_param": generation.max_tokens_param,
        "reasoning_mode": generation.reasoning_mode,
        "temperature_sent": generation.temperature_sent,
        "top_p_sent": generation.top_p_sent,
        "finish_reason": generation.finish_reason,
    }


def _prompt_cache_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate provider prompt-cache telemetry over every billed attempt."""
    usages = [
        detail.get("usage") or {}
        for attempt in attempts
        for detail in attempt.get("attempts_detail", [])
    ]
    total_input = sum(int(usage.get("input_tokens") or 0) for usage in usages)
    cached_input = sum(int(usage.get("cached_input_tokens") or 0) for usage in usages)
    cache_write = sum(int(usage.get("cache_write_tokens") or 0) for usage in usages)
    reported = sum(bool(usage.get("prompt_cache_usage_reported")) for usage in usages)
    reads = sum(int(usage.get("cached_input_tokens") or 0) > 0 for usage in usages)
    writes = sum(int(usage.get("cache_write_tokens") or 0) > 0 for usage in usages)
    return {
        "attempts": len(usages),
        "attempts_with_cache_accounting": reported,
        "attempts_with_cache_reads": reads,
        "attempts_with_cache_writes": writes,
        "input_tokens": total_input,
        "cached_input_tokens": cached_input,
        "cache_write_tokens": cache_write,
        "cached_input_fraction": cached_input / total_input if total_input else 0.0,
    }


def _call_provider(model_id: str, prompt: str,
                   provider_policy: dict[str, Any] | None = None) -> GenerationResult:
    # One client per worker thread keeps at most `max_parallel` HTTP connection pools alive.
    # Constructing a fresh OpenAI client for each of 15,000 calls leaks the benefit of keep-alive
    # and leaves resource cleanup to garbage collection during an overnight run.
    provider = getattr(_PROVIDER_LOCAL, "provider", None)
    cached_policy = getattr(_PROVIDER_LOCAL, "policy", None)
    if provider is None or provider.model_id != model_id or cached_policy != provider_policy:
        provider = get_provider(
            model_id, use_openrouter=True,
            response_format=_RESPONSE_FORMAT, require_parameters=True,
            disable_response_cache=True,
            provider_policy=dict(provider_policy) if provider_policy else None,
            max_tokens=_OUTPUT_TOKEN_CAP,
        )
        _PROVIDER_LOCAL.provider = provider
        _PROVIDER_LOCAL.policy = dict(provider_policy) if provider_policy else None
    return provider.generate(_SYSTEM_PROMPT, prompt)


def _one_call(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    spec = task["spec"]
    prompt = task["prompt"]
    generation: GenerationResult | None = None
    details: list[dict[str, Any]] = []
    failure_kind = None
    failure_message = None
    pfs = None
    try:
        for attempt_index in range(1, int(task["max_attempts"]) + 1):
            generation = _call_provider(task["model_id"], prompt,
                                        task.get("provider_policy"))
            warnings: list[str] = []
            endpoint_flags: dict[str, Any] = {}
            truncated = str(generation.finish_reason or "").lower() == "length"
            # A PIN THAT IS NOT VERIFIED IS A PREFERENCE. Two distinct failures, and the
            # second is the one that is easy to get wrong: a call that does not NAME its
            # backend verifies nothing, so accepting it would let an unpinned call through
            # under a manifest that claims pinning. Both are rejected.
            # Computed per attempt into a LOCAL: `failure_kind` survives across retries, so
            # reading it here would let attempt 1's rejection condemn attempt 2's success.
            served = str(getattr(generation, "served_provider", "") or "")
            pin_failure = _provider_verdict(task.get("provider_policy"), served)
            if pin_failure is not None:
                failure_kind, failure_message = pin_failure
                details.append({
                    "attempt_index": attempt_index,
                    "warnings": [],
                    "truncated": truncated,
                    "accepted": False,
                    "failure_kind": failure_kind,
                    "failure_message": failure_message,
                    "raw_response_sha256": hashlib.sha256(
                        generation.text.encode()).hexdigest(),
                    "usage": _usage(generation),
                })
                continue
            # REASONING MUST BE OFF, AND THAT IS A GATE. The battery recorded
            # `reasoning_mode=forced_off` on 562 of 565 calls and still saw up to 16,384
            # reasoning tokens on some of them, which is the whole budget spent thinking. A
            # trajectory generated with reasoning on was produced under different conditions
            # from the rest of the run, so it is rejected rather than quietly averaged in.
            reasoning_tokens = int(getattr(generation, "reasoning_tokens", 0) or 0)
            reasoning_mode = str(getattr(generation, "reasoning_mode", "") or "")
            if reasoning_mode != _REQUIRED_REASONING_MODE or reasoning_tokens > 0:
                failure_kind = "provider_parameter_violation"
                failure_message = (
                    "registered decoding requires reasoning_mode=forced_off and zero "
                    f"reasoning tokens; provider returned mode={reasoning_mode!r}, "
                    f"tokens={reasoning_tokens}")
                details.append({
                    "attempt_index": attempt_index,
                    "warnings": [],
                    "truncated": truncated,
                    "accepted": False,
                    "failure_kind": failure_kind,
                    "failure_message": failure_message,
                    "raw_response_sha256": hashlib.sha256(
                        generation.text.encode()).hexdigest(),
                    "usage": _usage(generation),
                })
                continue
            # A MALFORMED RESPONSE IS A TYPED RETRY, NOT A LOST CALL. On the first
            # qualification a provider returned a body the parser could not walk
            # ("'str' object has no attribute 'get'"), the exception escaped to the outer
            # handler, and that handler discarded the usage block -- so a call whose backend
            # was perfectly well identified was counted as an UNNAMED provider and dragged
            # the metadata fill rate below its threshold. Handled here so the provider
            # identity survives and the attempt is retried like any other bad response.
            try:
                if task["contract"] == "continuation":
                    failure_kind, failure_message, endpoint_flags, warnings = (
                        _accept_continuation(generation.text, spec))
                    pfs = endpoint_flags.get("pfs_months")
                else:
                    raise _UseClassifyResponse
            except _UseClassifyResponse:
                pass
            except Exception as exc:                                    # noqa: BLE001
                details.append({
                    "attempt_index": attempt_index,
                    "warnings": [],
                    "truncated": truncated,
                    "accepted": False,
                    "failure_kind": "malformed_response",
                    "failure_message": f"{type(exc).__name__}: {exc}",
                    "raw_response_sha256": hashlib.sha256(
                        generation.text.encode()).hexdigest(),
                    "usage": _usage(generation),
                })
                failure_kind = "malformed_response"
                failure_message = f"{type(exc).__name__}: {exc}"
                continue
            if task["contract"] != "continuation":
                failure_kind, failure_message, parsed = classify_response(
                    generation.text,
                    baseline_seed=dict(spec.baseline_seed),
                    pinned_paths=spec.pinned_paths,
                    terminal_contract="death_only",
                    conditioning_spec=spec,
                )
                pfs = None
                if failure_kind is None and parsed is not None:
                    cutoff = (spec.observed_prefix.cutoff_date
                              if spec.observed_prefix else None)
                    if cutoff is None:
                        failure_kind = "endpoint_extraction"
                        failure_message = "query has no observed-prefix cutoff date"
                    else:
                        pfs = _extract_pfs_months(parsed, cutoff)
                        if pfs is None:
                            failure_kind = "endpoint_extraction"
                            failure_message = ("no progression/death/last_followup after "
                                               "cutoff")
            # A COURSE CUT OFF IS NOT A BAD COURSE. The contract asks for the whole
            # trajectory to death, so a long one costs more tokens than a short one, and a
            # rejection caused by the token limit is correlated with the outcome being
            # scored. Naming it separately keeps that visible instead of burying it in
            # `no_terminal_event`, where it would look like a model that cannot finish.
            if failure_kind and truncated:
                failure_kind = "output_truncated"
                failure_message = (f"finish_reason=length after "
                                   f"{generation.output_tokens} output tokens; "
                                   f"the course did not reach its terminal event")
            details.append({
                "attempt_index": attempt_index,
                "warnings": warnings,
                "truncated": truncated,
                **{k: endpoint_flags.get(k) for k in
                   ("pfs_valid", "full_course_valid", "death_reached", "os_months",
                    "last_event_month", "n_events")},
                "accepted": failure_kind is None,
                "failure_kind": failure_kind,
                "failure_message": failure_message,
                "raw_response_sha256": hashlib.sha256(
                    generation.text.encode()).hexdigest(),
                "usage": _usage(generation),
            })
            if failure_kind is None:
                break
        return {
            **{k: task[k] for k in ("query_id", "arm", "repeat_index",
                                   "model_id", "model_short")},
            "accepted": failure_kind is None,
            "failure_kind": failure_kind,
            "failure_message": failure_message,
            "pfs_months": pfs,
            "contract": task["contract"],
            # what this trajectory is valid FOR, recorded per sample. `full_course_valid`
            # false with `pfs_valid` true is a usable PFS datum from an incomplete course,
            # and the count of those is what tells us whether the token limit is biting.
            **{k: endpoint_flags.get(k) for k in
               ("pfs_valid", "full_course_valid", "death_reached", "os_months",
                "last_event_month", "n_events")},
            "finish_reason": (generation.finish_reason if generation is not None else None),
            # WHICH ATTEMPT PRODUCED THIS SAMPLE. Retrying until acceptance is rejection
            # sampling: if hard trajectories fail more often, the accepted samples are a
            # biased draw from the very distribution being scored. Carried per sample so
            # first-attempt and later-attempt draws can be compared before anything is
            # concluded from them.
            "accepted_on_attempt": len(details) if failure_kind is None else None,
            "attempts_used": len(details),
            "attempts_detail": details,
            "duration_s": time.perf_counter() - started,
            "prompt_sha256": task["prompt_sha256"],
            "raw_response_sha256": (
                hashlib.sha256(generation.text.encode()).hexdigest()
                if generation is not None else None
            ),
            "raw_response": generation.text if generation is not None else None,
            "usage": (_usage(generation) if generation is not None else {}),
        }
    except Exception as exc:  # noqa: BLE001 - transport smoke must report everything
        return {
            **{k: task[k] for k in ("query_id", "arm", "repeat_index",
                                   "model_id", "model_short")},
            "accepted": False,
            "failure_kind": "transport_or_runtime",
            "failure_message": f"{type(exc).__name__}: {exc}",
            "pfs_months": None,
            "contract": task["contract"],
            "attempts_used": len(details),
            "attempts_detail": details,
            "duration_s": time.perf_counter() - started,
            "prompt_sha256": task["prompt_sha256"],
            "raw_response_sha256": None,
            "raw_response": None,
            "usage": {},
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


#: A4 pass thresholds, fixed before the smoke runs. A rule chosen after seeing the numbers
#: is not a rule, and "truncation decides whether to raise the cap" without a number leaves
#: the decision post-hoc.
QUALIFICATION_FILL_RATE_MIN = 1.0
QUALIFICATION_TRUNCATION_MAX = 0.05
QUALIFICATION_CONTAMINATION_MAX = 0.0
QUALIFICATION_MASD_MAX = 0.15


def qualification_summary(attempts: list[dict[str, Any]],
                          policy: dict[str, Any] | None) -> dict[str, Any]:
    """A4: the three numbers that decide whether the definitive run may start.

    Reported SEPARATELY because they have different remedies. Of the discarded partial's 89
    length-terminated attempts, 55 were reasoning violations from one backend and only 34
    were real output truncations -- excluding that backend removes the first group and none
    of the second, so a single "truncation rate" would credit the exclusion with a fix it
    does not deliver and hide whether the output cap needs raising.
    """
    records = [(row, detail)
               for row in attempts
               for detail in (row.get("attempts_detail") or [{}])]
    total = len(records) or 1

    named, providers = 0, Counter()
    contaminated = truncated_genuine = truncated_contaminated = 0
    for _, detail in records:
        usage = detail.get("usage") or {}
        served = normalize_provider(usage.get("served_provider") or "")
        is_contaminated = int(usage.get("reasoning_tokens") or 0) > 0
        is_length = str(usage.get("finish_reason") or "").lower() == "length" \
            or bool(detail.get("truncated"))
        if served:
            named += 1
            providers[served] += 1
        contaminated += is_contaminated
        truncated_contaminated += is_length and is_contaminated
        truncated_genuine += is_length and not is_contaminated

    # ONE BACKEND PER ACCEPTED REQUEST. An earlier version counted every attempt of an
    # accepted request, so a request that failed twice on backend A and succeeded on B
    # contributed two A's and one B to the arm's composition -- the opposite of what the
    # balance diagnostic asks, which is which backend produced the SCORED generation.
    by_arm: dict[str, Counter] = defaultdict(Counter)
    for row in attempts:
        if not row.get("accepted"):
            continue
        details = row.get("attempts_detail") or []
        winner = next((d for d in details if d.get("accepted")), details[-1] if details else {})
        served = normalize_provider((winner.get("usage") or {}).get("served_provider") or "")
        if served:
            by_arm[str(row.get("arm"))][served] += 1

    shares = {arm: {p: c / max(sum(counts.values()), 1) for p, c in counts.items()}
              for arm, counts in by_arm.items()}
    masd = 0.0
    for provider in {p for arm_shares in shares.values() for p in arm_shares}:
        seen = [arm_shares.get(provider, 0.0) for arm_shares in shares.values()]
        if len(seen) > 1:
            masd = max(masd, max(seen) - min(seen))

    banned = {normalize_provider(n) for n in (policy or {}).get("ignore") or []}
    allowed = {normalize_provider(n) for n in (policy or {}).get("only") or []}
    served_by_excluded = sum(c for p, c in providers.items()
                             if p in banned or (allowed and p not in allowed))

    fill_rate = named / total
    contamination_rate = contaminated / total
    genuine_truncation_rate = truncated_genuine / total
    failures = []
    if fill_rate < QUALIFICATION_FILL_RATE_MIN:
        failures.append(
            f"served_provider fill rate {fill_rate:.1%} < {QUALIFICATION_FILL_RATE_MIN:.0%}; "
            "the routing policy is unverifiable and the definitive run must not start")
    if contamination_rate > QUALIFICATION_CONTAMINATION_MAX:
        failures.append(f"reasoning contamination {contamination_rate:.1%} > "
                        f"{QUALIFICATION_CONTAMINATION_MAX:.0%}")
    if served_by_excluded:
        failures.append(f"{served_by_excluded} attempts served by a backend the policy "
                        "does not permit")
    if genuine_truncation_rate > QUALIFICATION_TRUNCATION_MAX:
        failures.append(
            f"genuine truncation {genuine_truncation_rate:.1%} > "
            f"{QUALIFICATION_TRUNCATION_MAX:.0%}; raise the output cap and re-smoke ONCE. "
            "If the second smoke still exceeds it, that is a blocking finding, "
            "not a threshold to relax")
    if masd > QUALIFICATION_MASD_MAX:
        failures.append(
            f"provider composition MASD {masd:.3f} > {QUALIFICATION_MASD_MAX:.2f}; arms were "
            "served by materially different backend mixtures. At smoke size this is "
            "indicative, not the A3 verdict, but it is a reason to qualify the pool rather "
            "than launch")
    return {
        "n_attempts": len(records),
        "n_requests": len(attempts),
        "served_provider_fill_rate": round(fill_rate, 4),
        "provider_share_overall": {p: round(c / max(named, 1), 4)
                                   for p, c in providers.most_common()},
        "attempts_by_provider": dict(providers.most_common()),
        "provider_share_by_arm_accepted": {
            arm: {p: round(v, 4) for p, v in sorted(arm_shares.items())}
            for arm, arm_shares in sorted(shares.items())},
        "masd": round(masd, 4),
        "reasoning_contamination_rate": round(contamination_rate, 4),
        "genuine_truncation_rate": round(genuine_truncation_rate, 4),
        "contaminated_truncation_count": truncated_contaminated,
        "served_by_excluded_backend": served_by_excluded,
        "thresholds": {"fill_rate_min": QUALIFICATION_FILL_RATE_MIN,
                       "contamination_max": QUALIFICATION_CONTAMINATION_MAX,
                       "genuine_truncation_max": QUALIFICATION_TRUNCATION_MAX,
                       "masd_max": QUALIFICATION_MASD_MAX},
        "passed": not failures,
        "failures": failures,
    }


def _aggregate_attempts(attempts: list[dict[str, Any]], out_path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[(str(row["query_id"]), str(row["arm"]))].append(row)
    rows = []
    for (query_id, arm), vals in sorted(grouped.items()):
        samples = [
            float(v["pfs_months"]) for v in vals
            if v["accepted"] and v["pfs_months"] is not None
        ]
        rows.append({
            "query_id": query_id,
            "arm": arm,
            "model": vals[0]["model_id"],
            "model_short": vals[0]["model_short"],
            "refused": len(samples) == 0,
            "pfs_samples": samples,
            "n_requested": len(vals),
            "n_accepted": len(samples),
            "failure_counts": dict(Counter(
                str(v["failure_kind"]) for v in vals if not v["accepted"])),
        })
    _write_jsonl(out_path, rows)


def _task_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["arm"]), str(row["query_id"]), int(row["repeat_index"])


def _slim_attempt(row: dict[str, Any]) -> dict[str, Any]:
    """Keep raw text on disk without retaining all 15,000 responses in memory."""
    return {key: value for key, value in row.items() if key != "raw_response"}


def _validate_task_matrix(
    payloads: pd.DataFrame,
    *,
    query_ids: list[str],
    arms: list[str],
) -> None:
    """Require exactly one payload for every registered query-by-arm cell."""
    expected = {(str(query_id), str(arm)) for query_id in query_ids for arm in arms}
    counts = Counter(
        (str(row.query_id), str(row.arm))
        for row in payloads[["query_id", "arm"]].itertuples(index=False)
    )
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if missing or extra or duplicates:
        raise SystemExit(
            "[fatal] prompt payload matrix is not one row per query and arm: "
            f"missing={missing[:5]}, extra={extra[:5]}, duplicates={duplicates[:5]}"
        )


def _manifest_differences(existing: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Top-level differences are enough to identify a scientifically different run."""
    keys = sorted(set(existing) | set(expected))
    return [key for key in keys if existing.get(key) != expected.get(key)]


def _prepare_run(
    run_root: Path,
    manifest: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a fresh run or validate an exact, append-only resume.

    The manifest is immutable once written. A resume is accepted only when its full run
    identity matches and every existing row belongs to the current task plan with the same
    prompt hash. This prevents a valid attempts file from becoming permission to mix runs.
    """
    manifest_path = run_root / "manifest.json"
    attempts_path = run_root / "attempts.jsonl"
    entries = list(run_root.iterdir())

    if dry_run:
        if entries:
            raise SystemExit(
                f"[fatal] dry-run directory {run_root} is not empty. Use a fresh --run-root "
                "so prompt review cannot inherit stale files."
            )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return [], tasks

    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"[fatal] cannot read existing run manifest: {exc}") from exc
        differences = _manifest_differences(existing_manifest, manifest)
        if differences:
            raise SystemExit(
                f"[fatal] {run_root} belongs to a different run; manifest fields differ: "
                f"{differences}. Use a fresh --run-root."
            )
    elif entries:
        raise SystemExit(
            f"[fatal] {run_root} contains files but no manifest.json. Use a fresh --run-root "
            "rather than mixing two runs."
        )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if not attempts_path.exists():
        leftovers = sorted(path.name for path in run_root.iterdir()
                           if path.name != "manifest.json")
        if leftovers:
            raise SystemExit(
                f"[fatal] {run_root} has no attempts.jsonl but contains {leftovers}. "
                "Use a fresh --run-root."
            )
        return [], tasks

    planned = {_task_key(task): task for task in tasks}
    if len(planned) != len(tasks):
        raise SystemExit("[fatal] current task plan contains duplicate call keys")

    done_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    with attempts_path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = _task_key(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"[fatal] invalid resume row at {attempts_path}:{line_number}: {exc}"
                ) from exc
            if key in seen:
                raise SystemExit(f"[fatal] duplicate completed call in attempts.jsonl: {key}")
            if key not in planned:
                raise SystemExit(f"[fatal] attempts.jsonl contains an unplanned call: {key}")
            task = planned[key]
            checks = {
                "model_id": task["model_id"],
                "model_short": task["model_short"],
                "contract": task["contract"],
                "prompt_sha256": task["prompt_sha256"],
            }
            mismatches = [name for name, value in checks.items() if row.get(name) != value]
            if mismatches:
                raise SystemExit(
                    f"[fatal] resume row {key} differs from the current task in {mismatches}"
                )
            seen.add(key)
            done_rows.append(_slim_attempt(row))

    pending = [task for task in tasks if _task_key(task) not in seen]
    return done_rows, pending


def enforce_definitive(prereg: Path, args, model_id: str, model_short: str) -> None:
    """Refuse to spend unless the run IS the registered design. Reads it, never restates it.

    Hashing the preregistration proves WHICH design was on disk; it proves nothing about what
    was actually run. A locked file happily coexists with the wrong model, the wrong landmark
    count or a two-arm subset, and the readout would carry a LOCKED stamp for a run that was
    not the registered one. The required values are read out of the file so this check cannot
    drift away from the design it is checking.
    """
    doc = yaml.safe_load(prereg.read_text())
    alloc = doc.get("allocation", {})
    want_arms = sorted(alloc.get("arms", []))
    problems = []
    if int(alloc.get("landmarks", -1)) != int(args.query_count):
        problems.append(f"landmarks: registered {alloc.get('landmarks')}, "
                        f"about to run {args.query_count}")
    if int(alloc.get("futures_per_landmark", -1)) != int(args.repeats):
        problems.append(f"futures: registered {alloc.get('futures_per_landmark')}, "
                        f"about to run {args.repeats}")
    if sorted(args.arms) != want_arms:
        problems.append(f"arms: registered {want_arms}, about to run {sorted(args.arms)}")
    # The prereg names the model by its SHORT name; the runner resolves a provider id. Compare
    # on a normalised form so `deepseek_v4_flash_0731` matches `deepseek/deepseek-v4-flash-0731`
    # rather than failing on punctuation.
    def _norm(x: str) -> str:
        return "".join(ch for ch in str(x).lower() if ch.isalnum())

    want_model = str(alloc.get("model", ""))
    if want_model and _norm(want_model) not in _norm(model_id) \
            and _norm(want_model) != _norm(model_short):
        problems.append(f"model: registered {want_model}, about to run {model_id}")
    # A definitive run may not route freely, and "some provider" is not the registered design
    # either. The policy is READ OUT of the amendment and compared exactly, for the same reason
    # the landmark count is: a check that restates the design instead of reading it drifts away
    # from the design it is checking.
    want_policy = _registered_provider_policy(doc)
    have_policy = resolved_provider_policy(args)
    if want_policy is None:
        problems.append(
            "provider: the preregistration declares no routing policy; amendment "
            "A2_provider_routing must define one before a definitive run")
    elif have_policy != want_policy:
        problems.append(
            f"provider policy: registered {want_policy}, about to send {have_policy}")
    elif _DEFAULT_PROVIDER_POLICY != want_policy:
        # The runner reads the policy from the document, so this can only mean the fallback
        # constant has drifted away from the amendment. Harmless today, wrong the first time
        # someone runs without a prereg, and a silent divergence between code and design.
        problems.append(
            f"provider policy: the registered policy is {want_policy} but the runner's "
            f"fallback constant is {_DEFAULT_PROVIDER_POLICY}; update the constant")
    if args.contract != "continuation":
        problems.append(f"contract: registered continuation, about to run {args.contract}")
    try:
        import subprocess
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, cwd=ROOT, check=True).stdout.strip()
        if dirty:
            problems.append("the working tree is dirty; the recorded commit would not "
                            "reproduce this run")
    except Exception:                                       # noqa: BLE001
        problems.append("git state could not be read")
    if problems:
        raise SystemExit("[fatal] --definitive was requested but this is not the registered "
                         "design:\n  - " + "\n  - ".join(problems))


def _validate_cache_preflight(args) -> None:
    """Require a serial repeated-prompt design that can actually reveal a cache hit."""
    if not args.cache_preflight:
        return
    problems = []
    if args.query_count != 1:
        problems.append("--query-count must be 1")
    if len(args.arms) != 1:
        problems.append("exactly one --arms value is required")
    if args.repeats < 3:
        problems.append("--repeats must be at least 3")
    if args.max_parallel != 1:
        problems.append("--max-parallel must be 1 so the first request warms the cache")
    if args.definitive:
        problems.append("--cache-preflight and --definitive are mutually exclusive")
    if problems:
        raise SystemExit("[fatal] invalid cache preflight:\n  - " + "\n  - ".join(problems))


def _enforce_cache_preflight_result(
    *, enabled: bool, accepted: int, total: int, cache: dict[str, Any],
) -> None:
    """Require a cache hit and a usable trajectory from every qualification call."""
    if not enabled:
        return
    problems = []
    if accepted != total:
        problems.append(f"accepted {accepted}/{total}; provider qualification requires 100%")
    if int(cache.get("cached_input_tokens") or 0) <= 0:
        problems.append("observed no provider prompt-cache read tokens")
    if problems:
        raise SystemExit(
            "[fatal] cache/provider preflight failed:\n  - " + "\n  - ".join(problems)
            + "\nDo not launch the definitive run until both routing and cache eligibility pass."
        )


def _run_provenance(in_dir: Path, query_ids: list[str], prereg: Path | None) -> dict:
    """Hash the design, the patients, the inputs and the code, before anything is generated."""
    def _sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    manifest_path = in_dir / "manifest.json"
    out: dict[str, Any] = {
        # the exact landmarks, order-independent, so a re-run can prove it used the same ones
        "n_landmarks": len(query_ids),
        "landmark_ids_sha256": _sha("\n".join(sorted(query_ids)).encode()),
        "input_manifest_sha256": _sha(manifest_path.read_bytes())
        if manifest_path.exists() else "",
    }
    if prereg is not None and prereg.exists():
        doc = yaml.safe_load(prereg.read_text())
        out["preregistration"] = {
            "path": str(prereg), "status": str(doc.get("status", "MISSING")),
            "sha256": _sha(prereg.read_bytes())}
        if str(doc.get("status")) != "LOCKED":
            raise SystemExit(
                f"[fatal] {prereg.name} is {doc.get('status')}, not LOCKED. Generation against a "
                "design still being edited is not preregistered generation.")
    else:
        out["preregistration"] = {"status": "NOT SUPPLIED"}
    try:
        import subprocess
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=ROOT, check=True).stdout.strip()
        out["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=ROOT, check=True).stdout.strip())
    except Exception:                                   # noqa: BLE001 — provenance is best-effort
        out["git_commit"] = "unavailable"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseekflash",
                    help="Alias or OpenRouter model ID. deepseekflash maps to 0731.")
    ap.add_argument("--model-short", default=None)
    ap.add_argument("--in-dir", type=Path, default=IN_DIR)
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--arms", nargs="+",
                    default=["R3_no_anchors", "R4_same_action", "R6_shuffled"])
    ap.add_argument("--query-count", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--contract", choices=("continuation", "latent_completion"),
                    default="continuation",
                    help="continuation (primary): real patient fixed, future events only. "
                         "latent_completion (ablation): the inherited synthetic contract, "
                         "where the model also writes the unobserved baseline -- a course "
                         "conditioned on an invented ECOG is a course for another patient, "
                         "so this arm answers whether that helps, it is not the default.")
    ap.add_argument("--max-parallel", type=int, default=20)
    ap.add_argument("--no-provider-policy", action="store_true",
                    help="send no routing policy at all. Only defensible for throwaway "
                         "plumbing checks: the registered policy excludes the backend shown "
                         "to violate the reasoning contract and declares the retention terms "
                         "the governance statement commits to.")
    ap.add_argument("--arm-order-seed", type=int, default=20260818,
                    help="seed for shuffling each query's arm-block order. Fixed submission "
                         "order would put arms in different wall-clock windows, associating "
                         "them with time-varying backend capacity.")
    ap.add_argument("--timeout-seconds", type=float, default=600.0,
                    help="per-provider-call timeout; 600 s matches the C/X/H/A run, whose "
                         "observed maximum was 491 s")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--cache-preflight",
        action="store_true",
        help="run one prompt serially at least three times and fail unless OpenRouter reports "
             "a provider prompt-cache read; response caching remains forcibly disabled",
    )
    ap.add_argument("--definitive", action="store_true",
                    help="refuse to run unless this IS the registered design (use for the "
                         "paper run; a locked prereg alone does not prove what was run)")
    ap.add_argument("--prereg", type=Path,
                    default=ROOT / "manuscript" / "genie_anchored_pfs_prereg_v1.yaml",
                    help="the design this run implements; must be LOCKED, and is hashed in")
    ap.add_argument("--probe-prereg", type=Path, default=None,
                    help="path to the amendment YAML; when given, its pre-registered "
                         "knowledge-probe questions are asked once per pair and the verbatim "
                         "answers are written to knowledge_probe.jsonl")
    args = ap.parse_args()
    positive = {
        "query-count": args.query_count,
        "repeats": args.repeats,
        "max-attempts": args.max_attempts,
        "max-parallel": args.max_parallel,
        "timeout-seconds": args.timeout_seconds,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"[fatal] these arguments must be positive: {invalid}")
    _validate_cache_preflight(args)

    model_id, default_short = MODEL_ALIASES.get(args.model, (args.model, args.model))
    model_short = args.model_short or default_short
    if args.definitive:
        if not (args.prereg and args.prereg.exists()):
            raise SystemExit("[fatal] --definitive requires --prereg")
        enforce_definitive(args.prereg, args, model_id, model_short)
        print(f"definitive mode: this run matches {args.prereg.name} on landmarks, futures, "
              "arms, model and contract, on a clean tree")
    os.environ["OPENROUTER_TIMEOUT_SECONDS"] = str(args.timeout_seconds)
    C.verify_anchored_template()

    queries = _load_queries(args.in_dir / "queries.jsonl")
    # THE PREFIX AND THE CONTRACT MUST AGREE. Imaging progressions belong in the prefix
    # under `continuation`, where a prefix event is a conditioning fact. Under the
    # `latent_completion` ablation the prefix is replayed, and replay needs deltas this
    # release does not record for an imaging read -- so the run would fail one call at a
    # time and read like a model problem. Rebuild the queries with --no-prefix-imaging.
    build_manifest = json.loads((args.in_dir / "manifest.json").read_text())
    usable = (build_manifest.get("queries") or {}).get("usable_contracts", ["continuation"])
    if args.contract not in usable:
        raise SystemExit(
            f"queries.jsonl was built for {usable} and this run asks for "
            f"{args.contract!r}. Rebuild with:\n"
            "  uv run python scripts/explore/build_anchored_pfs_arms_v1.py "
            "--no-prefix-imaging")
    payloads = _load_prompt_payloads(args.in_dir, args.arms)
    chosen = _select_queries(payloads, query_count=args.query_count)
    payloads = payloads[payloads["query_id"].astype(str).isin(chosen)].copy()
    if payloads.empty:
        raise SystemExit("no prompt payloads selected")
    if len(chosen) != args.query_count:
        raise SystemExit(
            f"[fatal] requested {args.query_count} landmarks but only selected {len(chosen)}"
        )
    _validate_task_matrix(payloads, query_ids=chosen, arms=args.arms)

    run_root = args.run_root or (
        DEFAULT_RUN_ROOT / (
            f"{model_short}_q{len(chosen)}x{len(args.arms)}x{args.repeats}"
            + ("_dryrun" if args.dry_run else "")
        )
    )
    run_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    policy = resolved_provider_policy(args)
    for _, row in payloads.sort_values(["query_id", "arm"]).iterrows():
        qid = str(row["query_id"])
        if qid not in queries:
            raise ValueError(f"{qid} missing from queries.jsonl")
        spec = queries[qid]["spec"]
        prompt = _compose(str(row["anchor_block"]), spec, args.contract)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        for repeat in range(1, args.repeats + 1):
            tasks.append({
                "query_id": qid,
                "arm": str(row["arm"]),
                "repeat_index": repeat,
                "model_id": model_id,
                "model_short": model_short,
                "spec": spec,
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "max_attempts": args.max_attempts,
                "contract": args.contract,
                "provider_policy": policy,
            })

    # Arms are submitted in a shuffled per-query order so they do not occupy systematically
    # different wall-clock windows. Deterministic, so a resume rebuilds the identical plan.
    tasks = interleave_arm_blocks(tasks, args.arm_order_seed)

    manifest = {
        # RUN-START PROVENANCE, all of it, before the first call. A readout that cannot say
        # which design, which patients and which code produced it is not checkable later, and
        # none of this can be reconstructed after the fact.
        "provenance": _run_provenance(args.in_dir, chosen, args.prereg),
        "model_id": model_id,
        "model_short": model_short,
        "arms": args.arms,
        "query_ids": chosen,
        "repeats": args.repeats,
        "max_attempts": args.max_attempts,
        "contract": args.contract,
        "n_calls": len(tasks),
        "max_parallel": args.max_parallel,
        "timeout_seconds": args.timeout_seconds,
        "prompt_provenance": C.provenance(),
        "provider_contract": {
            "response_format": _RESPONSE_FORMAT,
            "require_parameters": True,
            "provider_policy": resolved_provider_policy(args),
            "arm_block_order_seed": args.arm_order_seed,
            "max_output_tokens": _OUTPUT_TOKEN_CAP,
            "prompt_cache": {
                "mode": "provider_implicit_opportunistic",
                "accounting": "usage.prompt_tokens_details",
                "openrouter_session_id": None,
                "sticky_routing": False,
                "required_for_launch": False,
            },
            "response_cache": {
                "enabled": False,
                "enforcement": "X-OpenRouter-Cache: false",
                "reason": "identical requests must remain independent trajectory draws",
            },
            "reasoning_mode_required": _REQUIRED_REASONING_MODE,
            "reasoning_tokens_required": 0,
            "temperature_sent": DEFAULT_REQUEST_TEMPERATURE,
            "top_p_sent": None,
            "seed_sent": DEFAULT_REQUEST_SEED,
        },
        "input_dir": str(args.in_dir),
        "dry_run": args.dry_run,
        "cache_preflight": args.cache_preflight,
    }
    print(f"model: {model_id} ({model_short})")
    print(f"contract: {args.contract}"
          + ("" if args.contract == "continuation"
             else "   [ABLATION: the model also writes the unobserved baseline]"))
    print(f"selected {len(chosen)} queries x {len(args.arms)} arms x "
          f"{args.repeats} repeats = {len(tasks)} calls")
    # RESUME. The manifest is checked before it can be overwritten, and every completed row is
    # matched to its exact prompt. A valid attempts file is not by itself permission to append.
    before = len(tasks)
    done_rows, tasks = _prepare_run(
        run_root, manifest, tasks, dry_run=args.dry_run)
    if done_rows:
        print(f"RESUMING: {len(done_rows)} calls already on disk, {before - len(tasks)} "
              f"skipped, {len(tasks)} remaining")
        if not tasks:
            print("nothing left to generate; re-aggregating and exiting")

    print(f"run_root: {run_root}")

    if args.dry_run:
        preview = [
            {k: t[k] for k in ("query_id", "arm", "repeat_index",
                              "model_id", "model_short", "prompt_sha256")}
            for t in tasks
        ]
        _write_jsonl(run_root / "call_plan.jsonl", preview)
        review_dir = run_root / "prompt_review"
        review_dir.mkdir()
        seen_arms: set[str] = set()
        for task in tasks:
            if task["arm"] in seen_arms:
                continue
            seen_arms.add(task["arm"])
            safe_query = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in str(task["query_id"])
            )
            (review_dir / f"{task['arm']}__{safe_query}.txt").write_text(task["prompt"])
        print("dry-run only; wrote call_plan.jsonl and one exact provider prompt per arm")
        return 0

    if args.probe_prereg:
        import yaml
        prereg = yaml.safe_load(args.probe_prereg.read_text())
        questions = (prereg.get("knowledge_probes") or {}).get("questions") or {}
        if not questions:
            raise SystemExit(f"{args.probe_prereg} carries no knowledge_probes.questions")
        print(f"\nknowledge probe: {sum(len(v) for v in questions.values())} questions "
              f"from {args.probe_prereg.name}")
        probes = knowledge_probe(model_id, questions)
        _write_jsonl(run_root / "knowledge_probe.jsonl", probes)
        for r in probes:
            head = (r.get("answer") or r.get("error") or "")[:96].replace("\n", " ")
            print(f"   {r['pair']:28s} {r['form']:15s} {head}")
        print("   written verbatim to knowledge_probe.jsonl — read it, do not grade it")

    # APPEND, do not rewrite. Keep at most `max_parallel` calls in flight: submitting all
    # 15,000 up front makes ThreadPoolExecutor wait through the whole queue after Ctrl-C while
    # the main thread is no longer persisting results.
    attempts: list[dict[str, Any]] = list(done_rows)
    attempts_path = run_root / "attempts.jsonl"
    pool = cf.ThreadPoolExecutor(max_workers=args.max_parallel)
    pending: dict[cf.Future, dict[str, Any]] = {}
    task_iter = iter(tasks)
    for task in task_iter:
        pending[pool.submit(_one_call, task)] = task
        if len(pending) >= args.max_parallel:
            break
    try:
        with attempts_path.open("a") as sink:
            done_count = 0
            while pending:
                ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    pending.pop(future)
                    row = future.result()
                    sink.write(json.dumps(row, default=str) + "\n")
                    sink.flush()
                    attempts.append(_slim_attempt(row))
                    done_count += 1
                    status = "ok" if row["accepted"] else row["failure_kind"]
                    print(f"[{done_count:05d}/{len(tasks):05d}] {row['arm']} "
                          f"{row['query_id']} r{row['repeat_index']} -> {status} "
                          f"after {row['attempts_used']} attempt(s) "
                          f"({row['duration_s']:.1f}s)")
                    try:
                        next_task = next(task_iter)
                    except StopIteration:
                        continue
                    pending[pool.submit(_one_call, next_task)] = next_task
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    # Keep raw responses out of the scorer input. They remain in attempts.jsonl, in a
    # gitignored directory, for debugging parse failures.
    _write_jsonl(run_root / "attempts_slim.jsonl", attempts)
    _aggregate_attempts(attempts, run_root / "generations.jsonl")

    # A4: written on EVERY run, not only when --qualification is passed. The three numbers are
    # cheap, and a definitive run that silently stops reporting them is how a regression in the
    # metadata path would go unnoticed until the analysis needed the provider column.
    summary = qualification_summary(attempts, resolved_provider_policy(args))
    (run_root / "qualification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("\nQUALIFICATION SUMMARY")
    print(f"  served_provider fill rate    {summary['served_provider_fill_rate']:.1%} "
          f"(threshold {QUALIFICATION_FILL_RATE_MIN:.0%})")
    print(f"  reasoning contamination      {summary['reasoning_contamination_rate']:.1%}")
    print(f"  genuine truncation           {summary['genuine_truncation_rate']:.1%} "
          f"(threshold {QUALIFICATION_TRUNCATION_MAX:.0%})")
    print(f"  contaminated truncation      {summary['contaminated_truncation_count']} "
          "(removed by exclusion, not by a larger cap)")
    print(f"  served by excluded backend   {summary['served_by_excluded_backend']}")
    print(f"  provider composition MASD    {summary['masd']:.3f} "
          f"(threshold {QUALIFICATION_MASD_MAX:.2f})")
    if summary["provider_share_by_arm_accepted"]:
        print("  accepted provider share by arm:")
        for arm, shares in summary["provider_share_by_arm_accepted"].items():
            rendered = ", ".join(f"{p} {s:.0%}" for p, s in shares.items())
            print(f"    {arm:<20} {rendered}")
    for failure in summary["failures"]:
        print(f"  [FAIL] {failure}")
    print(f"  verdict: {'PASS' if summary['passed'] else 'FAIL'}")

    accepted = sum(1 for a in attempts if a["accepted"])
    costs = [
        float((detail.get("usage") or {}).get("reported_cost"))
        for a in attempts
        for detail in a.get("attempts_detail", [])
        if (detail.get("usage") or {}).get("reported_cost") is not None
    ]
    reasoning = sum(int((detail.get("usage") or {}).get("reasoning_tokens") or 0)
                    for a in attempts for detail in a.get("attempts_detail", []))
    cache = _prompt_cache_summary(attempts)
    print(f"\naccepted {accepted}/{len(attempts)} calls")
    print(f"reported cost: ${sum(costs):.4f}" if costs else "reported cost: not returned")
    print(f"reasoning tokens: {reasoning}")
    print(
        "prompt cache: "
        f"{cache['cached_input_tokens']:,}/{cache['input_tokens']:,} input tokens "
        f"({cache['cached_input_fraction']:.1%}); "
        f"{cache['attempts_with_cache_reads']} read attempt(s), "
        f"{cache['attempts_with_cache_writes']} write attempt(s), "
        f"accounting on {cache['attempts_with_cache_accounting']}/{cache['attempts']} attempt(s)"
    )
    (run_root / "prompt_cache_summary.json").write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {run_root / 'generations.jsonl'}")
    _enforce_cache_preflight_result(
        enabled=args.cache_preflight,
        accepted=accepted,
        total=len(attempts),
        cache=cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
