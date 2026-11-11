#!/usr/bin/env python3
"""Everything that must hold before the first billed call. Spends nothing.

A live run that fails on call 900 has already spent the money for 900. This checks, without
contacting a provider, the things that are cheap now and expensive later:

    seeds          every payload through the CANONICAL validator, not the exploratory
                   literal check. That one sees enum violations; this one also catches
                   wrong types, leaves that do not exist in the schema, supplied-but-
                   unpinned values and malformed nesting.
    output root    refuses an ephemeral destination. Raw generations are the one artefact
                   that cannot be rebuilt -- scoring can be corrected afterwards, a
                   trajectory written to /tmp and lost cannot.
    schedule       arm blocks varied in order so no arm occupies a systematically
                   different wall-clock window, repeats kept contiguous so the provider
                   cache still sees a shared prefix.
    provenance     the universe, the prompts and HEAD agree, and the tree is clean.
    prompts        every one composes, carries no unsubstituted placeholder, and fits.
    budget         what the run will cost, before it is spent.

GOVERNANCE. GENIE-BPC under DUA; transmission cleared. No provider is contacted here.

Run:  uv run python scripts/explore/preflight_four_arm_v1.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import emit_four_arm_prompts_v1 as EM  # noqa: E402
from synthonccases.oncollm.conditioning_audit import (  # noqa: E402
    BaselineSeedError,
    validate_baseline_seed,
)

PROMPTS = ROOT / "outputs" / "exploratory_four_arm_prompts"
UNIVERSE = ROOT / "outputs" / "exploratory_query_universe"

FUTURES_PER_ARM = 10
#: Approximate. Stated as an estimate rather than presented as a count.
CHARS_PER_TOKEN = 4.0
MAX_CONTEXT_TOKENS = 200_000

#: Destinations whose contents do not survive. Raw generations are irreplaceable.
EPHEMERAL = ("/tmp", "/var/tmp", "/private/tmp", "/dev/shm")


def output_root_ok(path: Path) -> tuple[bool, str]:
    resolved = str(path.resolve())
    for bad in EPHEMERAL:
        if resolved == bad or resolved.startswith(bad + "/"):
            return False, (f"{resolved} is ephemeral. Raw generations are the one artefact "
                           f"that cannot be rebuilt: scoring can be corrected afterwards, "
                           f"a trajectory written here and lost cannot.")
    if not path.parent.exists():
        return False, f"{resolved} has no existing parent directory"
    return True, resolved


def schedule_ok(tasks: list[dict]) -> list[str]:
    """Arm order varied per query, repeats contiguous."""
    bad = []
    by_query: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        by_query[t["query_id"]].append(t["arm"])
    first_arms = defaultdict(int)
    for arms in by_query.values():
        first_arms[arms[0]] += 1
        # contiguity: an arm must not reappear after a different arm intervened
        seen, last = set(), None
        for a in arms:
            if a != last and a in seen:
                bad.append("an arm's repeats are not contiguous, so the provider cache "
                           "cannot reuse the shared prefix")
                break
            seen.add(a)
            last = a
    n = max(1, len(by_query))
    worst = max(first_arms.values()) / n if first_arms else 1.0
    if worst > 0.45:
        bad.append(f"one arm goes first in {worst:.0%} of queries, so it would occupy a "
                   f"systematically different wall-clock window")
    return bad


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    quartets = [json.loads(line) for line in (PROMPTS / "quartets.jsonl").open()]
    prov = json.loads((PROMPTS / "provenance.json").read_text())
    manifest = json.loads((UNIVERSE / "manifest.json").read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())

    # ---- provenance -------------------------------------------------------------------
    if dirty:
        problems.append("the working tree is dirty; a run would point at code that is not "
                        "in any commit")
    for label, got in (("prompts", prov["git_commit"]),
                       ("query universe", manifest["git_commit"])):
        if got != head:
            problems.append(f"the {label} were built at {got[:12]}, not HEAD {head[:12]}")
    if prov["query_universe_checksum_as_recorded"] != manifest["queries_sha256"]:
        problems.append("the prompts record a different query-universe hash than the "
                        "universe manifest does")

    # ---- seeds, through the CANONICAL validator ---------------------------------------
    seed_failures = 0
    for qt in quartets:
        payload = qt["arms"]["patient_only"]["conditioning_payload"]
        try:
            validate_baseline_seed(payload["baseline_seed"])
        except BaselineSeedError as exc:
            seed_failures += 1
            if seed_failures <= 3:
                problems.append(f"{qt['query_id']}: {exc}")
        # supplied but NOT pinned: a value the model is given and never told is fixed
        supplied = set(_leaves(payload["baseline_seed"]))
        unpinned = supplied - set(payload.get("pinned_paths", ()))
        if unpinned and seed_failures <= 3:
            problems.append(f"{qt['query_id']}: supplied but unpinned: {sorted(unpinned)}")
    if seed_failures > 3:
        problems.append(f"... and {seed_failures - 3} further seed failures")

    # ---- schedule ---------------------------------------------------------------------
    from run_anchored_pfs_smoke_v1 import interleave_arm_blocks
    tasks = [{"query_id": qt["query_id"], "arm": arm, "repeat_index": r}
             for qt in quartets for arm in EM.ARMS
             for r in range(1, FUTURES_PER_ARM + 1)]
    tasks = interleave_arm_blocks(tasks, 20260824)
    problems += sorted(set(schedule_ok(tasks)))

    # ---- prompts ----------------------------------------------------------------------
    lengths = []
    for qt in quartets:
        for arm in EM.ARMS:
            text = qt["arms"][arm]["rendered_prompt"]
            lengths.append(len(text))
            if "{{" in text or "}}" in text:
                problems.append(f"{qt['query_id']}/{arm}: unsubstituted placeholder")
    tokens = np.array(lengths) / CHARS_PER_TOKEN
    over = int((tokens > MAX_CONTEXT_TOKENS).sum())
    if over:
        problems.append(f"{over} prompts exceed {MAX_CONTEXT_TOKENS:,} tokens")

    # ---- budget -----------------------------------------------------------------------
    calls = len(quartets) * len(EM.ARMS) * FUTURES_PER_ARM
    in_tokens = tokens.sum() * FUTURES_PER_ARM

    print("PREFLIGHT -- nothing is sent to a provider by this script\n")
    print(f"  commit                {head[:12]}  {'clean' if not dirty else 'DIRTY'}")
    print(f"  quartets / prompts    {len(quartets)} / {len(quartets) * len(EM.ARMS)}")
    print(f"  seeds validated       {len(quartets) - seed_failures}/{len(quartets)} "
          f"through the canonical validator")
    print(f"  prompt size           median {np.median(tokens):,.0f} tokens, "
          f"max {tokens.max():,.0f}, none over {MAX_CONTEXT_TOKENS:,}")
    print(f"  schedule              {len(tasks):,} tasks, arm blocks varied, repeats "
          f"contiguous")
    print("\n  A LIVE RUN WOULD COST:")
    print(f"      calls             {calls:,}  ({len(quartets)} patients x "
          f"{len(EM.ARMS)} arms x {FUTURES_PER_ARM} futures)")
    print(f"      input tokens      ~{in_tokens / 1e6:,.1f}M before any cache reuse")
    print(f"      cached prefix     repeats are contiguous, so {FUTURES_PER_ARM - 1} of "
          f"every {FUTURES_PER_ARM} calls should read a warm prefix")
    for note in notes:
        print(f"      {note}")

    ok, detail = output_root_ok(ROOT / "runs" / "four_arm_v1")
    print(f"\n  output root           {'ok  ' if ok else 'REFUSED'} {detail}")
    if not ok:
        problems.append(detail)

    print(f"\n{'PREFLIGHT PASSES -- ready to launch' if not problems else 'BLOCKED'}")
    for p in problems[:12]:
        print(f"  FAIL  {p}")
    return 0 if not problems else 1


def _leaves(obj, prefix: str = "") -> list[str]:
    out = []
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out += _leaves(v, path)
        else:
            out.append(path)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
