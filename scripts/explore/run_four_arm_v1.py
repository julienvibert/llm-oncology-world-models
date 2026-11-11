#!/usr/bin/env python3
"""The four-arm runner: consumes the saved prompts, qualifies every call, persists everything.

An earlier version of this file sent prompts and saved the text. That is not enough, and
the gap is invisible until the money is gone: "36 calls" meant 36 billed attempts, not 36
comparable futures. Six things had to hold before spending, and each one is here rather
than in a check that ran beforehand.

BINDING IS VERIFIED AGAINST WHAT IS SENT. Artifacts must be built at clean HEAD, and every
prompt is re-hashed HERE and compared with the hash recorded beside it. Trusting the stored
hash checks that the file is self-consistent, not that the text going to the provider is
the text that was reviewed.

QUALIFICATION IS NOT OPTIONAL. The provider routing policy, the backend pin, the
reasoning-off gate and truncation detection all come from the PFS runner, which learned
them the expensive way. Calling the bare provider helper dropped all four.

ACCEPTANCE IS THE CONTINUATION GATE, NOT FULL REPLAY. Full replay demands disease and host
deltas on every event -- ECOG, organ function, burden -- which GENIE does not supply, so
meeting that bar would mean inventing patient state. What is checked is what the experiment
needs: exact `{"ledger": [...]}` shape, every event a legal `LedgerEvent`, dates strictly
after the landmark and in order, no restatement of the prefix, exactly one terminal death,
and -- the part that has no home anywhere else -- successful projection onto the observable
ruler the real patients are measured on.

NOT PFS. The PFS runner rejects a course that never progresses. Here that is a perfectly
good observable future, so its endpoint gate is deliberately not reused.

SMOKE AND FULL RUNS DO NOT COLLIDE. Separate roots by default, a manifest bound to the
artifacts, and completed-task detection, so a resumed run continues instead of duplicating
and a mismatched manifest refuses instead of mixing.

COST IS READ FROM THE KEY THE PROVIDER ACTUALLY RETURNS. `_usage` reports `reported_cost`;
summing `cost_usd` produced a confident $0.00 and a projected full-run cost of $0.00.

THE OUTPUT ROOT IS REFUSED INSIDE THE RUNNER. A preflight that checks one hardcoded
destination does not constrain a `--run-root` given on the command line an hour later.

DRY RUN BY DEFAULT. `--live` is required to contact a provider, and this file never reads
an API key: the environment supplies it to the provider layer.

GOVERNANCE. GENIE-BPC under DUA; transmission cleared. Raw responses are written to the
persistent run root and never to the repository.

Run:  uv run python scripts/explore/run_four_arm_v1.py --smoke
      uv run python scripts/explore/run_four_arm_v1.py --smoke --live
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import emit_four_arm_prompts_v1 as EM  # noqa: E402
import generated_future_v1 as GF  # noqa: E402
from synthonccases.anchored import validate_future_ledger  # noqa: E402
from synthonccases.oncollm.conditioning_audit import (  # noqa: E402
    BaselineSeedError,
    validate_baseline_seed,
)
from synthonccases.oncollm.parse_utils import strip_code_fence  # noqa: E402

PROMPTS = ROOT / "outputs" / "exploratory_four_arm_prompts"
UNIVERSE = ROOT / "outputs" / "exploratory_query_universe"
DEFAULT_ROOTS = {"smoke": ROOT / "runs" / "four_arm_v1_smoke",
                 "full": ROOT / "runs" / "four_arm_v1"}
EPHEMERAL = ("/tmp", "/var/tmp", "/private/tmp", "/dev/shm")
OUTPUT_TOKEN_CAP = 24_576
FUTURES_PER_ARM = 10
FULL_RUN_CALLS = 18_000
#: PRESPECIFIED, before any data exists. Rejected slots are not refilled, so the realised
#: fan is smaller than the requested one; below this, an arm's estimate is too thin to
#: carry a contrast and the run should be fixed rather than reported.
MIN_ACCEPTABLE_FAN = 7
#: THE REGISTERED MODEL. The smoke that qualified this pipeline ran DeepSeek; the default
#: still said Claude, so the definitive run could have gone to a different model than the
#: one whose behaviour was characterised -- silently, and only visibly afterwards in the
#: manifest. Changing it now takes an explicit flag and is stamped on the run.
REGISTERED_MODEL = "deepseek/deepseek-v4-flash-0731"
#: Attempts per call. Retries are KEPT -- at the observed patient-only first-attempt rate,
#: ten calls would yield about three valid futures, below the fan floor -- but the run must
#: describe them: the main analysis uses this fixed pipeline, and first-attempt validity by
#: arm is a required endpoint in its own right, not a footnote.
MAX_ATTEMPTS = 3
#: GLOBAL provider calls in flight. The previous 15,000-call run used a bounded global
#: queue rather than serial patient blocks. Forty is deliberately more aggressive than
#: that run's usual twenty because these continuations are longer; the cache barrier below
#: still prevents a block's repeated calls from racing its first call.
DEFAULT_MAX_PARALLEL = 40


def require_persistent(path: Path) -> Path:
    """Refuse a destination whose contents do not survive.

    In the RUNNER, not only the preflight: a check against one hardcoded path says nothing
    about a --run-root supplied on the command line afterwards.
    """
    resolved = path.expanduser().resolve()
    for bad in EPHEMERAL:
        if str(resolved) == bad or str(resolved).startswith(bad + "/"):
            raise SystemExit(
                f"[refused] {resolved} is ephemeral. Raw generations are the one artefact "
                f"that cannot be rebuilt -- scoring can be corrected and rerun against "
                f"saved trajectories, a trajectory written here and lost is gone with the "
                f"money that produced it. Pass --run-root somewhere persistent.")
    resolved.mkdir(parents=True, exist_ok=True)
    probe = resolved / ".writable"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(f"[refused] {resolved} is not writable: {exc}") from exc
    return resolved


def artifact_binding() -> dict:
    """HEAD, the tree state, and the hashes the artifacts were built under.

    Returned rather than printed so the manifest records exactly what was checked.
    """
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    prov = json.loads((PROMPTS / "provenance.json").read_text())
    manifest = json.loads((UNIVERSE / "manifest.json").read_text())
    return {"git_commit": head, "tree_dirty": dirty,
            "prompts_commit": prov["git_commit"],
            "universe_commit": manifest["git_commit"],
            "universe_checksum": manifest["queries_sha256"],
            "universe_checksum_as_recorded": prov["query_universe_checksum_as_recorded"]}


def enforce_binding(b: dict) -> None:
    """Refuse to spend against artifacts that are not the ones at HEAD."""
    problems = []
    if b["tree_dirty"]:
        problems.append("the working tree is dirty, so a run would point at code that is "
                        "in no commit")
    for label, got in (("prompts", b["prompts_commit"]),
                       ("query universe", b["universe_commit"])):
        if got != b["git_commit"]:
            problems.append(f"the {label} were built at {got[:12]}, not HEAD "
                            f"{b['git_commit'][:12]}; rebuild them before spending")
    if b["universe_checksum"] != b["universe_checksum_as_recorded"]:
        problems.append("the prompts record a different query-universe hash than the "
                        "universe manifest does")
    if problems:
        raise SystemExit("[refused] " + "\n[refused] ".join(problems))


def load_tasks(smoke: bool, repeats: int) -> list[dict]:
    """One task per (query, arm, repeat), reading the SAVED prompts and RE-HASHING them.

    The prompt text is taken verbatim from quartets.jsonl rather than recomposed, and its
    hash is recomputed here: the recorded hash proves the file is self-consistent, not that
    what reaches the provider is what was reviewed.
    """
    quartets = [json.loads(line) for line in (PROMPTS / "quartets.jsonl").open()]
    if smoke:
        picked, seen = [], set()
        for qt in quartets:
            if qt["cancer"] not in seen:
                picked.append(qt)
                seen.add(qt["cancer"])
        quartets = picked[:3]
    tasks, mismatches = [], []
    for qt in quartets:
        for arm in EM.ARMS:
            d = qt["arms"][arm]
            got = hashlib.sha256(d["rendered_prompt"].encode()).hexdigest()
            if got != d["rendered_prompt_sha256"]:
                mismatches.append(f"{qt['query_id']}/{arm}")
                continue
            for r in range(1, repeats + 1):
                tasks.append({
                    "task_id": f"{qt['query_id']}|{arm}|{r}",
                    "query_id": qt["query_id"], "cancer": qt["cancer"],
                    "depth": qt["depth"], "arm": arm, "repeat_index": r,
                    "prompt": d["rendered_prompt"],
                    "prompt_sha256": got,
                    "landmark_state": qt.get("landmark_state", "unknown"),
                    "landmark_presence": qt.get("landmark_presence"),
                    "landmark_sites": qt.get("landmark_sites"),
                    "landmark_extent": qt.get("landmark_extent"),
                    "real_evaluable_horizons": qt.get("real_evaluable_horizons"),
                    "index_regimen": qt.get("index_regimen"),
                    "scoring_horizons": tuple(qt.get("scoring_horizons", (3, 6, 12, 24))),
                    "cutoff_date": d["conditioning_payload"]["fixed_events"]["cutoff_date"],
                    "prefix_events": d["conditioning_payload"]["fixed_events"]["events"],
                    "conditioning_payload": d["conditioning_payload"],
                })
    if mismatches:
        raise SystemExit(f"[refused] {len(mismatches)} prompt(s) do not match their "
                         f"recorded hash, so the reviewed text is not the text that would "
                         f"be sent: {mismatches[:5]}")
    return tasks


def blocks(tasks: list[dict], seed: int) -> list[list[dict]]:
    """Tasks grouped into (query, arm) blocks, block ORDER varied per query.

    Arm order is shuffled so no arm occupies a systematically different wall-clock window;
    repeats stay together because the block is the unit the cache serves.
    """
    import random

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for t in tasks:
        grouped[(t["query_id"], t["arm"])].append(t)
    by_query: dict[str, list[list[dict]]] = defaultdict(list)
    for (qid, _arm), block in grouped.items():
        by_query[qid].append(sorted(block, key=lambda t: t["repeat_index"]))
    out = []
    for qid in sorted(by_query):
        bs = sorted(by_query[qid], key=lambda b: b[0]["arm"])
        random.Random(f"{seed}:{qid}").shuffle(bs)
        out.extend(bs)
    return out


def accept(raw_text: str, task: dict) -> tuple[str | None, str | None, dict]:
    """The continuation gate, and then the ruler. Returns (failure_kind, message, flags).

    Deliberately NOT the PFS runner's acceptance: that one rejects a course which never
    progresses, and here such a course is a perfectly good observable future. Everything
    before the endpoint question is shared, so `validate_future_ledger` does the structural
    work and only the endpoint differs.
    """
    try:
        payload = json.loads(strip_code_fence(raw_text))
    except (json.JSONDecodeError, ValueError) as exc:
        return "unparseable_json", f"{type(exc).__name__}: {exc}", {}
    # THIS EXPERIMENT SCORES THE WHOLE COURSE. The shared validator defaults to allowing
    # an incomplete ledger because the earlier PFS experiment could still use a course
    # that reached progression before truncation. Here the prompt and terminal contract
    # require generation through death, so accepting a ledger that merely stops would make
    # "complete futures" mean something different at generation and at scoring.
    result = validate_future_ledger(payload, cutoff=task["cutoff_date"],
                                    prefix=task["prefix_events"], require_terminal=True)
    if not result.accepted:
        return result.failure_kind, result.detail, {}
    # THE WHOLE SCHEMA, NOT JUST THE STRUCTURE. `validate_future_ledger` checks types,
    # dates, ordering and the regimen essentials; it does not put each event through the
    # Pydantic model the prompt actually injected. We do not need full replay -- that would
    # demand host state GENIE never supplied -- but an event should satisfy the schema the
    # model was handed, or the enums in that schema are decoration.
    from synthonccases.oncollm.replay import LedgerEvent
    for i, ev in enumerate(payload.get("ledger") or []):
        try:
            LedgerEvent.model_validate(ev)
        except Exception as exc:                                        # noqa: BLE001
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return "event_schema_violation", f"event {i}: {first}", {}
    # SCOREABILITY. A ledger can be perfectly legal and still not land on the ruler the
    # real patients are measured on, and that failure is otherwise invisible until
    # analysis -- by which point the calls are paid for.
    events = payload.get("ledger") or []
    flags = GF.scoreability(events, cutoff=task["cutoff_date"],
                            landmark_state=task["landmark_state"],
                            landmark_presence=task.get("landmark_presence"),
                            landmark_sites=task.get("landmark_sites"),
                            landmark_extent=task.get("landmark_extent"),
                            index_regimen=task.get("index_regimen"),
                            horizons=task["scoring_horizons"])
    if flags["failure"]:
        return "not_scoreable", flags["failure"], flags
    flags["n_events"] = len(events)
    flags["warnings"] = list(result.warnings)
    return None, None, flags


def real_evaluable(results: list[dict]) -> dict:
    """How many REAL patients are still evaluable at each horizon.

    The generated fan says whether a ledger parsed. This says whether there is anything to
    compare it against: a real patient censored at month 9 cannot adjudicate a 24-month
    claim however well-formed the generated one is. Reported beside the ledger fan so the
    two are never read as the same number.
    """
    per: dict[float, set] = defaultdict(set)
    total: set = set()
    for r in results:
        qid = r.get("query_id")
        total.add(qid)
        for h in (r.get("real_evaluable_horizons") or ()):
            per[float(h)].add(qid)
    return {h: {"evaluable_queries": len(per.get(h, ())), "n_queries": len(total)}
            for h in sorted(set(per) | {3.0, 6.0, 12.0, 24.0})}


def valid_ledger_fan(results: list[dict]) -> dict:
    """VALID GENERATED LEDGERS per (query, arm), at each horizon.

    NOT real-patient evaluability, and the earlier name implied it was. The contract forces
    generation through death, so a compliant trajectory always resolves every horizon: in
    the qualifying smoke this statistic equalled the repeat count at 3, 6, 12 AND 24 months
    for all twelve cells. What it actually detects is a ledger that failed to parse or
    stopped early -- useful, but it says nothing about whether the REAL patient was still
    under observation at that horizon. That is a property of GENIE follow-up, is reported
    separately as `real_evaluable_at`, and is what limits the comparison.
    """
    # EVERY CONFIGURED HORIZON IS INITIALISED TO ZERO, for every query-arm, BEFORE
    # counting. Building the table only from resolved futures made a completely unusable
    # horizon VANISH from the report: if no 24-month future resolved anywhere, 24 months
    # simply had no row, the floor check found nothing to compare, and no warning was
    # emitted. The worst possible result rendered as silence.
    horizons: set[float] = set()
    for r in results:
        horizons |= {float(h) for h in
                     ((r.get("scoreability") or {}).get("scoreable_at") or {})}
        horizons |= {float(h) for h in (r.get("scoring_horizons") or ())}
    seen_cells = {(r.get("query_id"), r.get("arm")) for r in results}
    per: dict[float, dict[tuple, int]] = {
        h: dict.fromkeys(seen_cells, 0) for h in horizons}
    for r in results:
        cell = (r.get("query_id"), r.get("arm"))
        for h, at in ((r.get("scoreability") or {}).get("scoreable_at") or {}).items():
            if (at or {}).get("alive_at_horizon") is not None:
                per[float(h)][cell] += 1
    out = {}
    for h, cells in per.items():
        counts = sorted(cells.get(c, 0) for c in seen_cells)
        out[h] = {"min_fan": counts[0] if counts else 0,
                  "median_fan": counts[len(counts) // 2] if counts else 0,
                  "cells_below_floor": sum(1 for c in counts if c < MIN_ACCEPTABLE_FAN),
                  "n_cells": len(counts)}
    return dict(sorted(out.items()))


def call(task: dict, model_id: str, live: bool, policy: dict | None,
         max_attempts: int) -> dict:
    """One qualified attempt. Dry by default; `--live` is the only path that contacts a
    provider.

    Reuses the PFS runner's provider layer and its gates -- routing policy, backend pin,
    reasoning-off, truncation -- because those were learned from a run that had already
    been paid for once.
    """
    started = time.perf_counter()
    base = {k: v for k, v in task.items()
            if k not in ("prompt", "conditioning_payload", "prefix_events")}
    if not live:
        return {**base, "dry_run": True, "prompt_chars": len(task["prompt"]),
                "seconds": round(time.perf_counter() - started, 3)}
    from run_anchored_pfs_smoke_v1 import (
        _REQUIRED_REASONING_MODE, _call_provider, _provider_verdict, _usage,
    )

    details: list[dict] = []
    failure_kind = failure_message = None
    flags: dict[str, Any] = {}
    generation = None
    for attempt_index in range(1, max_attempts + 1):
        generation = _call_provider(model_id, task["prompt"], policy)
        truncated = str(generation.finish_reason or "").lower() == "length"
        served = str(getattr(generation, "served_provider", "") or "")
        usage = _usage(generation)

        # A PIN THAT IS NOT VERIFIED IS A PREFERENCE, and a call that does not NAME its
        # backend verifies nothing.
        verdict = _provider_verdict(policy, served)
        if verdict is not None:
            failure_kind, failure_message = verdict
        else:
            # REASONING MUST BE OFF. A trajectory generated with reasoning on was produced
            # under different conditions from the rest of the run.
            rtok = int(getattr(generation, "reasoning_tokens", 0) or 0)
            rmode = str(getattr(generation, "reasoning_mode", "") or "")
            if rmode != _REQUIRED_REASONING_MODE or rtok > 0:
                failure_kind = "provider_parameter_violation"
                failure_message = (f"requires reasoning_mode={_REQUIRED_REASONING_MODE} "
                                   f"and zero reasoning tokens; got mode={rmode!r}, "
                                   f"tokens={rtok}")
            else:
                try:
                    failure_kind, failure_message, flags = accept(generation.text, task)
                except Exception as exc:                                # noqa: BLE001
                    failure_kind = "malformed_response"
                    failure_message = f"{type(exc).__name__}: {exc}"
        # A COURSE CUT OFF IS NOT A BAD COURSE. Truncation is correlated with the outcome
        # -- a long survivor costs more tokens -- so naming it separately keeps informative
        # censoring visible instead of hidden inside `no_terminal_event`.
        if failure_kind and truncated:
            failure_kind = "output_truncated"
            failure_message = (f"finish_reason=length after {generation.output_tokens} "
                               f"output tokens")
        # EVERY ATTEMPT'S RAW TEXT, not only the one that survived. A rejected first
        # response is the evidence for the selection the retry loop performs -- without it
        # we can tell THAT the first attempt failed but cannot inspect or replay WHY, and
        # the rejected sample is exactly what has to be characterised before its exclusion
        # can be defended.
        details.append({"attempt_index": attempt_index, "accepted": failure_kind is None,
                        "failure_kind": failure_kind, "failure_message": failure_message,
                        "truncated": truncated, "usage": usage,
                        "raw_response": generation.text,
                        "raw_response_sha256": hashlib.sha256(
                            generation.text.encode()).hexdigest(),
                        "completion_chars": len(generation.text or "")})
        if failure_kind is None:
            break

    return {**base, "dry_run": False,
            "accepted": failure_kind is None,
            "failure_kind": failure_kind, "failure_message": failure_message,
            "scoreability": flags,
            "attempts_used": len(details), "attempts_detail": details,
            "raw_response": generation.text if generation is not None else None,
            "raw_response_sha256": (hashlib.sha256(generation.text.encode()).hexdigest()
                                    if generation is not None else None),
            "usage": _usage(generation) if generation is not None else {},
            "completion_chars": len(getattr(generation, "text", "") or ""),
            "seconds": round(time.perf_counter() - started, 3)}


def resume_state(run_root: Path, binding: dict, model_id: str,
                 fresh: bool, execution_policy: dict | None = None) -> tuple[set[str], Path]:
    """Completed task ids, and the manifest this run must agree with.

    A run root carries a manifest naming the artifacts and model it was started against.
    Reopening it with different ones is refused rather than appended to, because
    `attempts.jsonl` is append-only and a mixed file cannot be separated afterwards.
    """
    manifest_path = run_root / "manifest.json"
    want = {"prompts_commit": binding["prompts_commit"],
            "universe_checksum": binding["universe_checksum"], "model_id": model_id}
    if execution_policy is not None:
        want["execution_policy"] = execution_policy
    if manifest_path.exists() and not fresh:
        have = json.loads(manifest_path.read_text())
        drift = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        if drift:
            raise SystemExit(
                f"[refused] {run_root} was started against different artifacts and "
                f"appending would mix two runs in one append-only file: {drift}. Use "
                f"--run-root for a new directory, or --fresh to start this one over.")
    else:
        manifest_path.write_text(json.dumps({**want, **binding}, indent=2) + "\n")

    done: set[str] = set()
    attempts = run_root / "attempts.jsonl"
    if attempts.exists() and not fresh:
        for line in attempts.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue        # a torn last line from an interrupted run is not fatal
            if row.get("task_id") and not row.get("dry_run"):
                done.add(row["task_id"])
    elif fresh and attempts.exists():
        attempts.rename(attempts.with_suffix(".jsonl.superseded"))
    return done, attempts


def automatic_parallel_blocks(max_parallel: int, repeats: int) -> int:
    """How many cache-warmed blocks can keep the global call pool occupied.

    A full block has one warm-up call and ``repeats - 1`` tail calls. With forty global
    slots and ten repeats, five active blocks expose 45 tail calls -- enough to keep the
    pool full without interleaving dozens of prompt prefixes. Smoke blocks have only two
    tails, so more blocks may be active at once.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    tails = max(1, repeats - 1)
    return min(max_parallel, max(1, (max_parallel + tails - 1) // tails))


def execute_blocks(
    plan: list[list[dict]],
    invoke: Callable[[dict], dict],
    *,
    max_parallel: int,
    max_parallel_blocks: int,
    on_result: Callable[[dict], None] | None = None,
    on_block_done: Callable[[int, int, list[dict], list[dict], bool], None] | None = None,
) -> tuple[list[dict], int]:
    """Execute several blocks concurrently while preserving each cache barrier.

    The first request of every ``(query, arm)`` block must FINISH before that block's
    repeated requests are eligible. Several independent first requests may run together;
    once warm, their tails share one bounded global pool. This recovers the former
    15,000-call runner's throughput without making the cache claim false again.

    Results are handed to ``on_result`` in the caller thread as soon as they complete, so
    append-only persistence remains serialized and an interruption strands as little paid
    work as possible.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if max_parallel_blocks < 1:
        raise ValueError("max_parallel_blocks must be >= 1")
    if max_parallel_blocks > max_parallel:
        raise ValueError("max_parallel_blocks cannot exceed max_parallel")
    if not plan:
        return [], 0

    results: list[dict] = []
    warm_blocks = 0
    pending: dict[cf.Future, tuple[int, bool]] = {}
    states: dict[int, dict[str, Any]] = {}
    next_block = 0
    completed_blocks = 0

    def submit(pool: cf.ThreadPoolExecutor, block_number: int, task: dict,
               *, warm: bool) -> None:
        future = pool.submit(invoke, task)
        pending[future] = (block_number, warm)
        states[block_number]["inflight"] += 1

    def activate(pool: cf.ThreadPoolExecutor) -> None:
        nonlocal next_block
        while (next_block < len(plan) and len(states) < max_parallel_blocks
               and len(pending) < max_parallel):
            number = next_block + 1
            block = plan[next_block]
            next_block += 1
            states[number] = {
                "block": block,
                "tail": deque(block[1:]),
                "inflight": 0,
                "warm_complete": False,
                "results": [],
            }
            submit(pool, number, block[0], warm=True)

    def release_tails(pool: cf.ThreadPoolExecutor) -> None:
        # Round-robin: one very long block cannot occupy the entire queue while another
        # already-warmed prefix waits and cools.
        while len(pending) < max_parallel:
            progressed = False
            for number in list(states):
                state = states[number]
                if not state["warm_complete"] or not state["tail"]:
                    continue
                submit(pool, number, state["tail"].popleft(), warm=False)
                progressed = True
                if len(pending) >= max_parallel:
                    break
            if not progressed:
                break

    pool = cf.ThreadPoolExecutor(max_workers=max_parallel)
    try:
        activate(pool)
        while pending:
            ready, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in ready:
                number, was_warm = pending.pop(future)
                row = future.result()
                state = states[number]
                state["inflight"] -= 1
                state["results"].append(row)
                if was_warm:
                    state["warm_complete"] = True
                results.append(row)
                if on_result is not None:
                    on_result(row)

            release_tails(pool)

            completed = [
                number for number, state in states.items()
                if state["warm_complete"] and not state["tail"] and not state["inflight"]
            ]
            for number in completed:
                completed_blocks += 1
                state = states.pop(number)
                block_results = state["results"]
                tail_results = block_results[1:]
                cache_hit = any(
                    int((detail.get("usage") or {}).get("cached_input_tokens") or 0) > 0
                    for row in tail_results
                    for detail in row.get("attempts_detail", [])
                )
                warm_blocks += int(cache_hit)
                if on_block_done is not None:
                    on_block_done(completed_blocks, len(plan), state["block"], block_results,
                                  cache_hit)

            activate(pool)
            release_tails(pool)
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results, warm_blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None,
                    help="defaults to runs/four_arm_v1_smoke or runs/four_arm_v1")
    ap.add_argument("--model-id", default=REGISTERED_MODEL,
                    help=f"registered model (default {REGISTERED_MODEL}); anything else "
                         f"requires --unregistered-model")
    ap.add_argument("--unregistered-model", action="store_true",
                    help="permit a model other than the registered one. The run is then "
                         "marked as off-register.")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                    help="maximum provider calls in flight globally (default 40)")
    ap.add_argument("--parallel-blocks", type=int, default=None,
                    help="simultaneously active cache-warmed query-arm blocks; defaults "
                         "to enough blocks to fill --max-parallel from their tails")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    ap.add_argument("--smoke", action="store_true",
                    help="three patients, four arms, three repeats")
    ap.add_argument("--fresh", action="store_true",
                    help="set aside an existing attempts file and start this root over")
    ap.add_argument("--no-provider-policy", action="store_true",
                    help="send no routing policy. Recorded, and it disables the backend "
                         "pin check, so the run cannot claim pinning.")
    ap.add_argument("--live", action="store_true",
                    help="contact the provider. Without it nothing is sent and nothing "
                         "is billed.")
    args = ap.parse_args()
    if args.smoke:
        args.repeats = min(args.repeats, 3)
    if args.max_parallel < 1:
        raise SystemExit("[refused] --max-parallel must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("[refused] --max-attempts must be >= 1")
    parallel_blocks = (args.parallel_blocks if args.parallel_blocks is not None
                       else automatic_parallel_blocks(args.max_parallel, args.repeats))
    if parallel_blocks < 1 or parallel_blocks > args.max_parallel:
        raise SystemExit(
            "[refused] --parallel-blocks must be >= 1 and cannot exceed "
            "--max-parallel")

    if args.model_id != REGISTERED_MODEL and not args.unregistered_model:
        raise SystemExit(
            f"[refused] --model-id {args.model_id} is not the registered model "
            f"{REGISTERED_MODEL}. The qualifying smoke characterised the registered one; a "
            f"definitive run on another model measures something else. Pass "
            f"--unregistered-model to do it deliberately.")
    run_root = require_persistent(
        args.run_root or DEFAULT_ROOTS["smoke" if args.smoke else "full"])
    binding = artifact_binding()
    enforce_binding(binding)

    tasks = load_tasks(args.smoke, args.repeats)
    for t in tasks:
        try:
            validate_baseline_seed(t["conditioning_payload"]["baseline_seed"])
        except BaselineSeedError as exc:
            raise SystemExit(f"[refused] {t['query_id']}: {exc}") from exc

    # RESOLVED WHETHER OR NOT WE SEND, so the dry run reports the routing a live run would
    # actually use. Resolving it only under --live made the dry run print "none", which
    # reads as "the live run will not pin either".
    policy = None
    if not args.no_provider_policy:
        from run_anchored_pfs_smoke_v1 import _DEFAULT_PROVIDER_POLICY
        policy = dict(_DEFAULT_PROVIDER_POLICY)

    execution_policy = {
        "max_parallel_calls": args.max_parallel,
        "max_parallel_blocks": parallel_blocks,
        "max_attempts_per_future": args.max_attempts,
        "repeats_per_query_arm": args.repeats,
        "cache_barrier": "first request of each block completes before its tails",
        "provider_policy": policy,
    }
    done, attempts_path = resume_state(
        run_root, binding, args.model_id, args.fresh, execution_policy=execution_policy)
    todo = [t for t in tasks if t["task_id"] not in done]
    plan = blocks(todo, 20260824)

    print(f"{'LIVE' if args.live else 'DRY RUN -- nothing is sent, nothing is billed'}")
    print(f"  run root   {run_root}")
    print(f"  commit     {binding['git_commit'][:12]}  clean, artifacts bound")
    print(f"  prompts    {len(tasks)} tasks, {len(tasks) - len(todo)} already complete")
    print(f"  plan       {len(plan)} blocks, {len(todo)} calls, "
          f"{args.repeats} repeats per block")
    print(f"  dispatch   {args.max_parallel} global calls, "
          f"{parallel_blocks} cache-warmed blocks active")
    print(f"  routing    {policy or 'none (pin not verifiable)'}")

    with attempts_path.open("a") as sink:
        def persist(row: dict) -> None:
            sink.write(json.dumps(row, default=str) + "\n")
            sink.flush()

        def report(i: int, total: int, block: list[dict], block_results: list[dict],
                   cache_hit: bool) -> None:
            ok = sum(1 for r in block_results if r.get("accepted"))
            cache = "cache hit" if cache_hit else "no cache hit"
            print(f"  block {i}/{total}  {block[0]['query_id']} / "
                  f"{block[0]['arm']}  {len(block)} calls, {ok} accepted, {cache}",
                  flush=True)

        invoke = lambda task: call(  # noqa: E731
            task, args.model_id, args.live, policy, args.max_attempts)
        results, warm_blocks = execute_blocks(
            plan, invoke, max_parallel=args.max_parallel,
            max_parallel_blocks=parallel_blocks, on_result=persist,
            on_block_done=report)

    summary: dict[str, Any] = {
        **binding, "live": args.live, "run_root": str(run_root),
        "model_id": args.model_id, "provider_policy": policy,
        "execution_policy": execution_policy,
        "n_blocks": len(plan), "n_calls": len(results),
        "n_previously_complete": len(tasks) - len(todo),
        "blocks_with_a_cache_hit": warm_blocks, "output_token_cap": OUTPUT_TOKEN_CAP}
    if args.live:
        us = [r["usage"] for r in results if r.get("usage")]
        # EVERY attempt is billed, including the rejected ones, so cost sums over
        # attempts_detail rather than over accepted results.
        every = [d["usage"] for r in results for d in r.get("attempts_detail", [])
                 if d.get("usage")]
        accepted = [r for r in results if r.get("accepted")]
        kinds: dict[str, int] = defaultdict(int)
        for r in results:
            if not r.get("accepted"):
                kinds[r.get("failure_kind") or "unknown"] += 1
        # THE RETRY LOOP IS A SELECTION STEP, so it is accounted rather than absorbed. A
        # single "acceptance_rate: 1.0" hid that 36 requested futures cost 51 billed
        # attempts, that 13 calls were retried, and that every one of the 15 intermediate
        # failures had a nameable cause. Retries are KEPT -- the patient-only first-attempt
        # rate would leave about three valid futures per ten calls, below the floor -- but
        # first-attempt validity is a REQUIRED ENDPOINT in its own right, because it is the
        # only sample no filter has touched.
        first_pass = [r for r in results if r.get("attempts_used") == 1]
        by_arm: dict[str, dict] = {}
        for r in results:
            cell = by_arm.setdefault(r["arm"], {"requested": 0, "first_attempt_valid": 0,
                                                "post_retry_valid": 0, "attempts": 0})
            cell["requested"] += 1
            cell["first_attempt_valid"] += int(r.get("attempts_used") == 1)
            cell["post_retry_valid"] += int(bool(r.get("accepted")))
            cell["attempts"] += int(r.get("attempts_used") or 1)
        for cell in by_arm.values():
            cell["first_attempt_rate"] = round(
                cell["first_attempt_valid"] / max(1, cell["requested"]), 3)
        rejected: dict[str, int] = defaultdict(int)
        for r in results:
            for d in r.get("attempts_detail", []):
                if not d.get("accepted"):
                    rejected[d.get("failure_kind") or "unknown"] += 1
        summary |= {
            "requested_futures": len(results),
            "billed_provider_attempts": len(every),
            "n_calls_retried": sum(1 for r in results if (r.get("attempts_used") or 1) > 1),
            "first_attempt_valid": len(first_pass),
            "first_attempt_rate": round(len(first_pass) / max(1, len(results)), 3),
            "post_retry_valid": len(accepted),
            "attempt_accounting_by_arm": by_arm,
            # EVERY intermediate failure, not only the final ones. `failure_kinds` counts
            # calls that never succeeded, which was empty while fifteen real failures had
            # happened inside the retry loop.
            "rejected_attempt_reasons": dict(sorted(rejected.items(),
                                                    key=lambda kv: -kv[1])),
            "max_attempts": args.max_attempts,
            "retry_policy_note": (
                "Main analysis: the fixed maximum-three-attempt pipeline. First-attempt "
                "validity by arm is a separate required endpoint. First-attempt-only "
                "clinical scoring is a SENSITIVITY analysis, not primary: at the observed "
                "patient-only rate ten calls yield about three valid futures, below the "
                "fan floor."),
            # THE REAL PATIENT'S follow-up, which is what actually limits the comparison.
            "real_evaluable_at": real_evaluable(results),
        }
        summary |= {
            "input_tokens": sum(u.get("input_tokens", 0) or 0 for u in every),
            "cached_input_tokens": sum(u.get("cached_input_tokens", 0) or 0
                                       for u in every),
            "output_tokens": sum(u.get("output_tokens", 0) or 0 for u in every),
            # THE KEY THE PROVIDER ACTUALLY RETURNS. `cost_usd` does not exist in `_usage`,
            # so summing it produced a confident $0.00 and a $0.00 full-run projection.
            "cost_usd_reported": sum(u.get("reported_cost", 0) or 0 for u in every),
            "n_billed_attempts": len(every),
            "n_accepted": len(accepted),
            "acceptance_rate": round(len(accepted) / max(1, len(results)), 3),
            "failure_kinds": dict(kinds),
            "median_completion_chars": sorted(
                r.get("completion_chars", 0) for r in results)[len(results) // 2] if results
            else 0,
        }
        # WHAT THE RUN ACTUALLY DOES, said plainly. A rejected task is NOT retried in a
        # later pass -- its slot is recorded and the fan for that (query, arm) is simply
        # one smaller. So the run issues 18,000 calls and RETAINS fewer. That may well be
        # preferable to unbounded rejection sampling, which biases the kept sample toward
        # whatever the model finds easy to emit; but it has to be described as what it is,
        # and the smallest fan we will accept has to be fixed before the data exists.
        per_call = summary["cost_usd_reported"] / max(1, len(results))
        summary["projected_full_run_usd"] = round(per_call * FULL_RUN_CALLS, 2)
        rate = max(1e-9, summary["acceptance_rate"])
        summary["expected_futures_retained"] = int(round(FULL_RUN_CALLS * rate))
        summary["min_acceptable_fan"] = MIN_ACCEPTABLE_FAN
        # THE FLOOR IS PER QUERY x ARM x HORIZON, not per accepted response. Seven accepted
        # trajectories are not seven usable 24-month futures: a course that truncates or is
        # censored at month 5 resolves the 3- and 6-month horizons and nothing beyond. A
        # single acceptance rate hides that, and it hides it WORST at the longest horizon,
        # which is the one most likely to carry the headline contrast.
        summary["valid_ledger_fan"] = valid_ledger_fan(results)
        thin = {h: v for h, v in summary["valid_ledger_fan"].items()
                if v["min_fan"] < MIN_ACCEPTABLE_FAN}
        summary["fan_projection_note"] = (
            f"{FULL_RUN_CALLS:,} calls are issued and rejected slots are not refilled, so "
            f"~{summary['expected_futures_retained']:,} responses are retained. What each "
            f"contrast can actually use is counted per query x arm x horizon below; "
            f"refilling instead would be rejection sampling on the distribution being "
            f"measured.")
        if thin:
            summary["fan_warning"] = (
                f"at horizon(s) {sorted(thin)} the smallest query-arm fan is below the "
                f"prespecified floor of {MIN_ACCEPTABLE_FAN}: "
                + "; ".join(f"{h}mo min={v['min_fan']} median={v['median_fan']}"
                            for h, v in sorted(thin.items()))
                + ". Raise FUTURES_PER_ARM or fix the dominant failure kind before "
                  "launching.")
        summary["cache_hit_rate"] = round(
            summary["cached_input_tokens"] / max(1, summary["input_tokens"]), 3)
        if us and not summary["cached_input_tokens"]:
            summary["cache_warning"] = (
                "no cached input reported on any call. The barrier permits caching but "
                "cannot guarantee it: tail calls may route to a different backend.")
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n  blocks with a cache hit  {warm_blocks}/{max(1, len(plan))}")
    if args.live:
        print(f"  {summary['requested_futures']} futures requested, "
              f"{summary['billed_provider_attempts']} provider attempts billed "
              f"({summary['n_calls_retried']} calls retried)")
        print(f"  first-attempt valid  {summary['first_attempt_valid']}/"
              f"{summary['requested_futures']} ({summary['first_attempt_rate']:.0%})   "
              f"post-retry valid  {summary['post_retry_valid']}/"
              f"{summary['requested_futures']}")
        for arm, c in sorted(summary["attempt_accounting_by_arm"].items()):
            print(f"      {arm:18s} first-attempt {c['first_attempt_valid']}/"
                  f"{c['requested']} ({c['first_attempt_rate']:.0%})   post-retry "
                  f"{c['post_retry_valid']}/{c['requested']}   attempts {c['attempts']}")
        if summary["rejected_attempt_reasons"]:
            print("  rejected attempts, by reason:")
            for k, n in summary["rejected_attempt_reasons"].items():
                print(f"      {k:32s} {n}")
        if summary["failure_kinds"]:
            print(f"  never succeeded: {summary['failure_kinds']}")
        print(f"  input {summary['input_tokens']:,} "
              f"(cached {summary['cached_input_tokens']:,}, "
              f"{summary['cache_hit_rate']:.0%})")
        print(f"  output {summary['output_tokens']:,} over "
              f"{summary['n_billed_attempts']} billed attempts")
        print(f"  cost ${summary['cost_usd_reported']:.2f}  ->  full run "
              f"${summary['projected_full_run_usd']:,.2f} for {FULL_RUN_CALLS:,} calls")
        print(f"  {summary['fan_projection_note']}")
        print("  VALID GENERATED LEDGERS per query-arm (not follow-up availability):")
        for h, v in summary["valid_ledger_fan"].items():
            real = summary["real_evaluable_at"].get(h, {})
            print(f"      {h:>5.0f} mo   ledgers min {v['min_fan']}, median "
                  f"{v['median_fan']}, {v['cells_below_floor']}/{v['n_cells']} below "
                  f"{MIN_ACCEPTABLE_FAN}   |   real patients evaluable "
                  f"{real.get('evaluable_queries', 0)}/{real.get('n_queries', 0)}")
        if summary.get("fan_warning"):
            print(f"  WARNING  {summary['fan_warning']}")
        if summary.get("cache_warning"):
            print(f"  WARNING  {summary['cache_warning']}")
    else:
        print("  no usage, cost or acceptance figures: nothing was sent. Run with --live "
              "to measure them before approving 18,000 calls.")
    print(f"\nwritten to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
