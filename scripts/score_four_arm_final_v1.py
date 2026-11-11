#!/usr/bin/env python3
"""Four-arm final scoring: eligibility precheck and scores.

    uv run python scripts/score_four_arm_final_v1.py --precheck-only

Specification: `manuscript/SCORING_SPEC_2026-09-28.md` (sections 2-4, 6, 7, 10, 11), its
30 September amendment, and `manuscript/SCORING_SPEC_amendment_2026-10-12.md` (reader
corrections R1-R5 after review; post-hoc freeze, not preregistration). This run
counts who is eligible for the joint 17-coordinate score, by horizon and cancer, under the
primary (forward persistence) and sensitivity (four-month expiry with conservative
conflicts) conventions, with exclusion reason codes, and the section-6 fallback coverage
(common windows, usable donors, scored time). It computes NO score, distance, arm effect
or outcome agreement. Eligibility uses follow-up, death and documentation only.

Version 2 (12 October 2026) reads through `four_arm_eligibility_v2` and
`four_arm_readers_v2` and writes date-suffixed outputs (`*_20261012.*`). The first
precheck's unsuffixed outputs and manifest are kept as history; that run is reproducible
from commit d0a407a.

Read-only: it consumes the frozen run, the saved query universe and the saved prompts, and
the local GENIE-BPC releases. It never calls a provider and never rebuilds a prompt.

GOVERNANCE. GENIE-BPC under DUA. Patient-level rows go only to the gitignored
`outputs/four_arm_preprint_20260928/private/`; console output is aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "src"))
import four_arm_eligibility_v2 as E  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402
import four_arm_scores_v1 as S  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

from synthonccases.oncollm.parse_utils import strip_code_fence  # noqa: E402

RUN = ROOT / "runs" / "four_arm_v1_definitive_p40b20_20260826"
UNIVERSE = ROOT / "outputs" / "exploratory_query_universe"
PROMPTS = ROOT / "outputs" / "exploratory_four_arm_prompts"
OUT = ROOT / "outputs" / "four_arm_preprint_20260928"
SPEC = ROOT / "manuscript" / "SCORING_SPEC_2026-09-28.md"
AMENDMENT = ROOT / "manuscript" / "review" / "2026-09-30_codex_gate3_amendment.md"
AMENDMENT_V2 = ROOT / "manuscript" / "SCORING_SPEC_amendment_2026-10-12.md"

#: Frozen facts the inputs must reproduce (spec section 2).
GENERATION_COMMIT = "28c12e43c4e133730b7d59c02935ea649881f388"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
UNIVERSE_DIGEST = "037ccee7149c54e19a3bfc78c644333641a059758f5f5b115a342d8fb77a2662"
REPEATS = 10
MAX_ATTEMPTS = 3
EXPECTED_SLOTS = 18_000
EXPECTED_ACCEPTED = 17_679
CANCERS = ("NSCLC", "BRCA", "CRC")
GENIE_FILES = ("data_timeline_cancer_diagnosis.txt", "data_clinical_patient.txt",
               "data_clinical_supp_survival.txt", "data_timeline_imaging.txt",
               "data_timeline_medonc.txt", "cancer_level_dataset_index.csv",
               "regimen_cancer_level_dataset.csv", "data_timeline_treatment.txt",
               "med_onc_note_level_dataset.csv")


class InputError(RuntimeError):
    """A frozen input failed a section-2 check. Never silently skipped."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


# --------------------------------------------------------------------------------------
# Section 2: frozen inputs and their checks
# --------------------------------------------------------------------------------------

def first_attempt_bound(r: dict) -> str:
    """First-attempt status, with the frozen binding asserted (reader correction R3).

    A valid first attempt must have no retry and BE the saved response, so projecting the
    saved response projects the first attempt. Anything else is an InputError.
    """
    status = S.first_attempt_status(r)
    if status == "valid":
        first = (r.get("attempts_detail") or [{}])[0]
        if int(r["attempts_used"]) != 1 or not r["accepted"] \
                or first.get("raw_response") != r.get("raw_response"):
            raise InputError("a valid first attempt is not the saved response")
    return status


def load_and_check() -> tuple[dict, dict, list[dict], dict]:
    """Queries, quartets, attempt summaries; raises InputError on any failed check."""
    checks: dict[str, object] = {}
    text = (UNIVERSE / "queries.jsonl").read_text()
    digest = sha256_text(text)
    prov = json.loads((PROMPTS / "provenance.json").read_text())
    umanifest = json.loads((UNIVERSE / "manifest.json").read_text())
    rmanifest = json.loads((RUN / "manifest.json").read_text())
    if not (digest == UNIVERSE_DIGEST == prov["query_universe_sha256"]):
        raise InputError("full query-universe digest does not match the saved provenance")
    if not (digest.startswith(rmanifest["universe_checksum"])
            and digest.startswith(umanifest["queries_sha256"])):
        raise InputError("short universe checksums disagree with the full digest")
    if not (rmanifest["git_commit"] == rmanifest["prompts_commit"] == prov["git_commit"]
            == GENERATION_COMMIT) or rmanifest.get("tree_dirty"):
        raise InputError("generation commit binding broken")
    pol = rmanifest["execution_policy"]
    if rmanifest["model_id"] != MODEL_ID or pol["repeats_per_query_arm"] != REPEATS \
            or pol["max_attempts_per_future"] != MAX_ATTEMPTS:
        raise InputError("run manifest disagrees with the frozen generation parameters")
    checks["query_universe_digest"] = digest

    queries = {}
    for line in text.splitlines():
        q = json.loads(line)
        if q["query_id"] in queries:
            raise InputError("duplicate query id in the universe")
        queries[q["query_id"]] = q
    quartets = {}
    for line in (PROMPTS / "quartets.jsonl").open():
        qt = json.loads(line)
        if qt["query_id"] in quartets:
            raise InputError("duplicate query id in the quartets")
        quartets[qt["query_id"]] = qt
    if set(queries) != set(quartets) or len(queries) != 450:
        raise InputError("query universe and prompt quartets do not cover the same 450")
    bad_prompt = 0
    for qid, qt in quartets.items():
        q = queries[qid]
        if (qt["cancer"], qt["depth"]) != (q["cancer"], q["depth"]):
            raise InputError("quartet cancer/depth disagrees with its query")
        if qt["index_regimen"] != q["action"]["regimen"]:
            raise InputError("quartet index regimen disagrees with the saved action")
        if abs(q["action"]["start_month"] - q["tau_months"]) > 1e-9:
            raise InputError("saved index action does not start at the saved landmark")
        for key in ("generic_donors", "relevant_donors"):
            if len({d["patient"] for d in q[key]}) != E.CARD_SIZE:
                raise InputError("a donor card does not hold twenty distinct members")
        for arm in E.ARMS:
            d = qt["arms"][arm]
            if sha256_text(d["rendered_prompt"]) != d["rendered_prompt_sha256"]:
                bad_prompt += 1
    if bad_prompt:
        raise InputError(f"{bad_prompt} saved prompts do not match their recorded hash")
    checks["prompt_text_hashes_verified"] = 450 * len(E.ARMS)

    # attempts: stream once, keep only what the precheck needs (no raw text retained
    # except the accepted ledger, parsed later)
    seen: set[str] = set()
    rows: list[dict] = []
    n_resp_hash = n_detail_hash = n_first_bound = 0
    with (RUN / "attempts.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            tid = r["task_id"]
            if tid in seen:
                raise InputError("duplicate task id in the run")
            seen.add(tid)
            qid, arm, rep = r["query_id"], r["arm"], int(r["repeat_index"])
            if tid != f"{qid}|{arm}|{rep}" or qid not in quartets or arm not in E.ARMS \
                    or not 1 <= rep <= REPEATS:
                raise InputError("task id does not link query, arm and repeat")
            qt = quartets[qid]
            if r["prompt_sha256"] != qt["arms"][arm]["rendered_prompt_sha256"]:
                raise InputError("attempt prompt hash disagrees with the saved prompt")
            fixed = qt["arms"][arm]["conditioning_payload"]["fixed_events"]
            if r["cutoff_date"] != fixed["cutoff_date"]:
                raise InputError("attempt cutoff date disagrees with the saved payload")
            for key in ("cancer", "depth", "landmark_state", "landmark_presence",
                        "landmark_sites", "landmark_extent", "index_regimen"):
                if r[key] != qt[key]:
                    raise InputError(f"attempt field {key} disagrees with its quartet")
            if r.get("dry_run"):
                raise InputError("dry-run row in the definitive run")
            if r.get("raw_response_sha256") is not None:
                if sha256_text(r.get("raw_response") or "") != r["raw_response_sha256"]:
                    raise InputError("response hash mismatch")
                n_resp_hash += 1
            detail = r.get("attempts_detail") or []
            for d in detail:
                if d.get("raw_response_sha256") is not None and d.get("raw_response") \
                        is not None:
                    if sha256_text(d["raw_response"]) != d["raw_response_sha256"]:
                        raise InputError("attempt-detail response hash mismatch")
                    n_detail_hash += 1
            if not 1 <= int(r["attempts_used"]) <= MAX_ATTEMPTS \
                    or len(detail) != int(r["attempts_used"]):
                raise InputError("attempt accounting inconsistent")
            if bool(r["accepted"]) != bool(detail[-1].get("accepted")):
                raise InputError("final acceptance disagrees with the last attempt")
            status = first_attempt_bound(r)
            n_first_bound += status == "valid"
            rows.append({
                "query_id": qid, "arm": arm, "repeat": rep, "cancer": r["cancer"],
                "accepted": bool(r["accepted"]), "attempts_used": int(r["attempts_used"]),
                "first_valid": status == "valid",
                "first_raw_unavailable": status == "unavailable",
                "first_status": status,
                "ledger_text": r.get("raw_response") if r["accepted"] else None,
            })
    grid = {f"{q}|{a}|{k}" for q in quartets for a in E.ARMS for k in range(1, REPEATS + 1)}
    if seen != grid:
        raise InputError("the run does not hold exactly the requested 450 x 4 x 10 grid")
    n_acc = sum(r["accepted"] for r in rows)
    if len(rows) != EXPECTED_SLOTS or n_acc != EXPECTED_ACCEPTED:
        raise InputError(f"reconciliation failed: {len(rows)} slots, {n_acc} accepted")
    checks |= {"task_ids_unique": True, "complete_requested_grid": True,
               "query_arm_repeat_linkage": True, "requested_slots": len(rows),
               "accepted_responses": n_acc, "response_hashes_verified": n_resp_hash,
               "attempt_detail_hashes_verified": n_detail_hash,
               "attempt_prompt_hashes_match_saved": len(rows),
               "valid_first_attempts_bound_to_saved_response": n_first_bound}
    return queries, quartets, rows, checks


# --------------------------------------------------------------------------------------
# The precheck (version 2: corrected readers, amendment of 12 October 2026)
# --------------------------------------------------------------------------------------

SUFFIX = "_20261012"


def run(out: Path) -> int:
    t0 = time.time()
    queries, quartets, attempts, checks = load_and_check()
    print(f"[inputs] section-2 checks passed: {checks['requested_slots']} slots, "
          f"{checks['accepted_responses']} accepted, universe digest verified")

    by_cancer = {c: [q for q in queries.values() if q["cancer"] == c] for c in CANCERS}
    calendars: dict[str, list[float]] = {}
    real_res: dict[tuple, E.Resolution] = {}
    survival: dict[tuple[str, float], str] = {}
    died6: dict[str, bool] = {}
    member_res: dict[tuple, list[E.Resolution]] = {}
    fallback: dict[tuple, dict] = {}                 # (qid, card, h, conv) -> section 6
    query_support: dict[tuple, float] = {}           # (qid, h, conv) -> supported months
    genie_hashes: dict[str, dict[str, str]] = {}
    reconciliation: dict[str, dict[str, int]] = {}
    adapter_mismatch: Counter = Counter()
    diag_unrecognised_real: Counter = Counter()

    for cancer, qs in by_cancer.items():
        genie = GV2.GENIE_ROOT / GV2.COHORTS[cancer]
        if not genie.exists():
            raise InputError(f"GENIE release missing for {cancer}")
        genie_hashes[cancer] = {f: sha256_file(genie / f) for f in GENIE_FILES}
        patients = {p.pid: p for p in GV2.load(genie)}
        agents, reconciliation[cancer] = R.load_real_agents(genie)
        course_cache: dict[tuple, E.Course] = {}

        def course(pid: str, tau: float, h: float, conv: str,
                   patients=patients, agents=agents, cache=course_cache) -> E.Course:
            key = (pid, tau, h, conv)
            if key not in cache:
                p = patients.get(pid)
                if p is None:
                    raise InputError("a query or donor patient is absent from GENIE")
                if not any(abs(ln["start"] - tau) < 1e-9 for ln in p.lines):
                    raise InputError("no regimen starts at a saved landmark")
                with E.convention(conv):
                    try:
                        cache[key] = E.real_course(p, tau, h, agents.get(pid, []))
                    except ValueError as exc:
                        raise InputError(str(exc)) from exc
            return cache[key]

        for q in qs:
            qid, tau = q["query_id"], float(q["tau_months"])
            p = patients.get(q["patient"])
            if p is None:
                raise InputError("a query patient is absent from GENIE")
            calendars[qid] = E.calendar_for(p, tau, quartets[qid]["assessment_times"])
            cal = calendars[qid]
            death = p.death - tau if math.isfinite(p.death) else math.inf
            died6[qid] = death <= 6.0
            for h in E.HORIZONS:
                survival[(qid, h)] = ("died" if death <= h else "alive"
                                      if p.last_contact - tau >= h else "unresolved")
                for conv in E.CONVENTIONS:
                    c = course(q["patient"], tau, h, conv)
                    r = E.resolve(c, cal, h)
                    if r.blocks[0]:
                        end = min(c.death, h)
                        direct = E.first_unresolved(list(c.path), end) is None
                        if direct != r.blocks[1]:
                            adapter_mismatch[conv] += 1
                    real_res[(qid, h, conv)] = r
                    qsup = E.supported(c, cal, h)
                    query_support[(qid, h, conv)] = E.measure(qsup)
                    for card, (field, _arm) in E.CARDS.items():
                        members = [course(d["patient"], float(d["tau_months"]), h, conv)
                                   for d in q[field]]
                        member_res[(qid, card, h, conv)] = [E.resolve(m, cal, h)
                                                            for m in members]
                        fallback[(qid, card, h, conv)] = E.fallback_patient(
                            [E.donor_window(qsup, m, cal, h) for m in members])
                if h == 12.0:
                    c = course(q["patient"], tau, h, "primary")
                    end = min(c.death, h)
                    diag_unrecognised_real.update({
                        R.normalise(ag.name) for ag in c.treatment.agents
                        if ag.cls.kind == R.UNRECOGNISED
                        and R._positive(ag.segments + ag.unresolved, 0.0, end)})
        print(f"[real] {cancer}: {len(qs)} queries and their donors resolved "
              f"({time.time() - t0:.0f}s)")

    # ---- generated repeats (convention-independent reader: one course per horizon) -------
    gen_res: dict[tuple, E.Resolution] = {}
    accepted_keys: dict[tuple[str, str], list[int]] = defaultdict(list)
    failures: Counter = Counter()
    terminal_kinds: Counter = Counter()
    diag_unrecognised_gen: Counter = Counter()
    active_start_conv: Counter = Counter()     # reader correction R1 (version 2): starts read as switches
    gen_conflicts = 0
    for i, a in enumerate(attempts):
        if not a["accepted"]:
            continue
        qid, arm, rep = a["query_id"], a["arm"], a["repeat"]
        accepted_keys[(qid, arm)].append(rep)
        qt = quartets[qid]
        kw = {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]
              ["cutoff_date"], "landmark_state": qt["landmark_state"],
              "landmark_presence": qt["landmark_presence"],
              "landmark_sites": qt["landmark_sites"],
              "landmark_extent": qt["landmark_extent"],
              "index_regimen": qt["index_regimen"]}
        try:
            ledger = json.loads(strip_code_fence(a["ledger_text"]))["ledger"]
        except Exception:                                               # noqa: BLE001
            failures["accepted_response_unparseable"] += 1
            for h in E.HORIZONS:
                for conv in E.CONVENTIONS:
                    gen_res[(qid, arm, rep, h, conv)] = E.unresolved(
                        "accepted_response_unparseable")
            continue
        a["ledger_text"] = None
        for h in E.HORIZONS:
            try:
                c = E.generated_course(ledger, h, **kw)
                r = E.resolve(c, calendars[qid], h)
            except R.NotScoreable:
                failures["projection_not_scoreable"] += 1
                r = E.unresolved("projection_not_scoreable")
                c = None
            except Exception as exc:                                    # noqa: BLE001
                failures[f"projection_exception_{type(exc).__name__}"] += 1
                r = E.unresolved(f"projection_exception_{type(exc).__name__}")
                c = None
            for conv in E.CONVENTIONS:
                gen_res[(qid, arm, rep, h, conv)] = r
            if c is not None and h == 24.0:
                terminal_kinds[c.terminal] += 1
                active_start_conv[f"{arm}_conversions"] += c.treatment.active_start_conversions
                active_start_conv[f"{arm}_ledgers_with_any"] += int(
                    c.treatment.active_start_conversions > 0)
                for ag in c.treatment.agents:
                    if ag.cls.kind == R.UNRECOGNISED:
                        diag_unrecognised_gen[R.normalise(ag.name)] += 1
        if (i + 1) % 3000 == 0:
            print(f"[generated] {i + 1} rows read, {time.time() - t0:.0f}s")
    # genuine generated contradictions (count once per accepted ledger, 24-month reader)
    gen_conflicts = sum(1 for k, r in gen_res.items() if k[3] == 24.0 and k[4] == "primary"
                        and E.DISEASE_AFTER_LANDMARK in r.reasons)

    # ---- cells, cards, query sets -----------------------------------------------------
    expected = [(qid, arm) for qid in sorted(queries) for arm in E.ARMS]
    cell: dict[tuple, tuple[bool, str, Counter]] = {}
    for h in E.HORIZONS:
        for conv in E.CONVENTIONS:
            reps = {k: [gen_res[(k[0], k[1], rep, h, conv)] for rep in accepted_keys[k]]
                    for k in expected}
            for row in E.emit_cells(expected, reps):
                sub = Counter({r for x in reps[(row["query_id"], row["arm"])]
                               if not x.full_core for r in x.reasons})
                cell[(row["query_id"], row["arm"], h, conv)] = (row["eligible"],
                                                                row["reason"], sub)
    card: dict[tuple, tuple[bool, str, Counter]] = {}
    for key, members in member_res.items():
        ok, why = E.donor_card(members)
        sub = Counter({r for m in members if not m.full_core for r in m.reasons})
        card[key] = (ok, why, sub)

    def sets_for(qid: str, h: float, conv: str) -> dict[str, tuple[bool, list[str]]]:
        real_ok = real_res[(qid, h, conv)].full_core
        rr = [] if real_ok else ["real_target_unavailable"]
        out = {"real_target": (real_ok, rr)}
        cells_ok = {}
        for arm in E.ARMS:
            ok = cell[(qid, arm, h, conv)][0]
            cells_ok[arm] = ok
            why = [] if ok else [f"llm_cell_{arm}_unavailable"]
            out[f"llm_cell:{arm}"] = (ok, why)
            out[f"real_and_arm:{arm}"] = (real_ok and ok, rr + why)
        cw = [f"llm_cell_{a}_unavailable" for a in E.ARMS if not cells_ok[a]]
        out["llm_cells_all_four_arms"] = (not cw, cw)
        out["four_arm_intersection"] = (real_ok and not cw, rr + cw)
        for name, (_field, arm) in E.CARDS.items():
            cok = card[(qid, name, h, conv)][0]
            cwhy = [] if cok else [f"card_{name}_unavailable"]
            out[f"card:{name}"] = (cok, cwhy)
            awhy = [] if cells_ok[arm] else [f"llm_cell_{arm}_unavailable"]
            out[f"donor_comparison:{name}"] = (real_ok and cells_ok[arm] and cok,
                                               rr + awhy + cwhy)
            out[f"four_arm_and_card:{name}"] = (real_ok and not cw and cok,
                                                rr + cw + cwhy)
        return out

    membership: dict[tuple, dict] = {}
    for h in E.HORIZONS:
        for qid in queries:
            p_sets = sets_for(qid, h, "primary")
            s_sets = sets_for(qid, h, "sensitivity")
            membership[(qid, h, "primary")] = p_sets
            membership[(qid, h, "sensitivity")] = s_sets
            membership[(qid, h, "both")] = {
                k: (p_sets[k][0] and s_sets[k][0],
                    ([] if p_sets[k][0] else ["unavailable_under_primary"])
                    + ([] if s_sets[k][0] else ["unavailable_under_sensitivity"]))
                for k in p_sets}
    populations = list(membership[(next(iter(queries)), 3.0, "primary")])

    # ---- aggregate outputs --------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    private = out / "private"
    private.mkdir(exist_ok=True)
    cancer_of = {qid: q["cancer"] for qid, q in queries.items()}
    strata = [("all", list(queries))] + [(c, [q["query_id"] for q in by_cancer[c]])
                                         for c in CANCERS]

    def detail_reasons(pop: str, qids: list[str], h: float, conv: str) -> Counter:
        sub: Counter = Counter()
        convs = ["primary", "sensitivity"] if conv == "both" else [conv]
        for qid in qids:
            if membership[(qid, h, conv)][pop][0]:
                continue
            for cv in convs:
                tag = f"{cv}:" if conv == "both" else ""
                if pop in ("real_target",) or pop.startswith(("real_and_arm", "four_arm",
                                                              "donor_comparison")):
                    rres = real_res[(qid, h, cv)]
                    if not rres.full_core:
                        sub.update(f"{tag}real:{r}" for r in rres.reasons)
                arms = ([pop.split(":")[1]] if pop.startswith(("llm_cell:", "real_and_arm"))
                        else list(E.ARMS) if pop.startswith(("four_arm", "llm_cells_all"))
                        else [E.CARDS[pop.split(":")[1]][1]]
                        if pop.startswith("donor_comparison") else [])
                for arm in arms:
                    ok, why, s = cell[(qid, arm, h, cv)]
                    if not ok:
                        sub[f"{tag}cell_{arm}:{why}"] += 1
                        sub.update(f"{tag}cell_{arm}:repeat:{r}" for r in s)
                cards = ([pop.split(":")[1]] if pop.startswith(("card:", "donor_comparison",
                                                                "four_arm_and_card"))
                         else [])
                for name in cards:
                    ok, why, s = card[(qid, name, h, cv)]
                    if not ok:
                        sub[f"{tag}card_{name}:{why}"] += 1
                        sub.update(f"{tag}card_{name}:member:{r}" for r in s)
        return sub

    def decision(pop: str, cancer: str, n: int, conv: str) -> str:
        floor = n >= E.PAIRED_FLOOR
        if cancer != "all":
            return ("cancer-specific inference permitted (floor met)" if floor
                    else "below twenty-patient floor: no cancer-specific inference")
        if pop.startswith("donor_comparison"):
            gov = "" if conv == "primary" else " [non-governing convention]"
            return ("joint donor comparison retained (floor met)" + gov if floor else
                    "fallback now: section-6 interval diagnostic and scalar comparisons; "
                    "joint donor claim omitted at this horizon" + gov)
        if pop in ("four_arm_intersection",) or pop.startswith(("real_and_arm",
                                                                 "four_arm_and_card")):
            return ("floor met" if floor else
                    "below twenty-patient floor: descriptive only, no inferential joint claim")
        return ""

    prec_rows = []
    for h in E.HORIZONS:
        for conv in ("primary", "sensitivity", "both"):
            for cancer, qids in strata:
                for pop in populations:
                    elig = [q for q in qids if membership[(q, h, conv)][pop][0]]
                    comp = Counter(r for q in qids if not membership[(q, h, conv)][pop][0]
                                   for r in membership[(q, h, conv)][pop][1])
                    sub = detail_reasons(pop, qids, h, conv)
                    prec_rows.append({
                        "horizon_months": h, "convention": conv, "cancer": cancer,
                        "population": pop,
                        "unit": ("llm_cells" if pop.startswith("llm_cell")
                                 else "donor_cards" if pop.startswith("card:")
                                 else "queries"),
                        "n_total": len(qids), "n_eligible": len(elig),
                        "n_excluded": len(qids) - len(elig),
                        "exclusion_components": "; ".join(
                            f"{k}={v}" for k, v in comp.most_common()),
                        "exclusion_detail": "; ".join(
                            f"{k}={v}" for k, v in sub.most_common()),
                        "floor_met": len(elig) >= E.PAIRED_FLOOR,
                        "fallback_decision": decision(pop, cancer, len(elig), conv)})
    write_csv(out / f"eligibility_precheck{SUFFIX}.csv", prec_rows)

    comp_sets = ["full_cohort_450", "real_target", "four_arm_intersection",
                 "donor_comparison:generic", "donor_comparison:matched",
                 "four_arm_and_card:generic", "four_arm_and_card:matched"]

    def describe(qids: list[str], h: float) -> dict:
        n = len(qids)
        st = Counter(survival[(q, h)] for q in qids)
        d6 = sum(died6[q] for q in qids)
        mix = Counter(cancer_of[q] for q in qids)
        return {"n": n, **{f"n_{c}": mix.get(c, 0) for c in CANCERS},
                **{f"share_{c}": (round(mix.get(c, 0) / n, 4) if n else "")
                   for c in CANCERS},
                "observed_deaths_by_H": st["died"],
                "documented_alive_through_H": st["alive"],
                "survival_unresolved_at_H": st["unresolved"],
                "frac_observed_deaths_by_H": round(st["died"] / n, 4) if n else "",
                "frac_survival_unresolved_at_H": round(st["unresolved"] / n, 4) if n else "",
                "observed_deaths_by_6mo": d6,
                "frac_observed_deaths_by_6mo": round(d6 / n, 4) if n else ""}

    comp_rows = []
    for h in E.HORIZONS:
        for conv in ("primary", "sensitivity", "both"):
            for cancer, qids in strata:
                full = describe(qids, h)
                for s in comp_sets:
                    members = (qids if s == "full_cohort_450" else
                               [q for q in qids if membership[(q, h, conv)][s][0]])
                    row = {"horizon_months": h, "convention": conv, "cancer": cancer,
                           "set": s, **describe(members, h)}
                    row |= {f"full_{k}": v for k, v in full.items()}
                    comp_rows.append(row)
    write_csv(out / f"population_composition{SUFFIX}.csv", comp_rows)

    att_by = defaultdict(list)
    for a in attempts:
        att_by[(a["arm"], a["cancer"])].append(a)
    cov_rows = []
    for h in E.HORIZONS:
        for conv in E.CONVENTIONS:
            for arm in E.ARMS:
                for cancer, qids in strata:
                    rows_ = [a for c in (CANCERS if cancer == "all" else (cancer,))
                             for a in att_by[(arm, c)]]
                    acc_res = [gen_res[(q, arm, rep, h, conv)] for q in qids
                               for rep in accepted_keys[(q, arm)]]
                    cov_rows.append({
                        "horizon_months": h, "convention": conv, "arm": arm,
                        "cancer": cancer, "requested_slots": len(rows_),
                        "billed_attempts": sum(a["attempts_used"] for a in rows_),
                        "accepted": sum(a["accepted"] for a in rows_),
                        "first_attempt_valid": sum(a["first_valid"] for a in rows_),
                        "first_attempt_raw_unavailable": sum(a["first_raw_unavailable"]
                                                             for a in rows_),
                        "accepted_repeats_full_core": sum(r.full_core for r in acc_res),
                        "accepted_repeats_unresolved": sum(not r.full_core
                                                           for r in acc_res),
                        "cells_requested": len(qids),
                        "cells_with_ge7_accepted": sum(len(accepted_keys[(q, arm)])
                                                       >= E.MIN_ACCEPTED for q in qids),
                        "cells_full_core_eligible": sum(cell[(q, arm, h, conv)][0]
                                                        for q in qids),
                        "real_survival_resolved": sum(survival[(q, h)] != "unresolved"
                                                      for q in qids),
                        "real_full_core": sum(real_res[(q, h, conv)].full_core
                                              for q in qids),
                        "cells_eligible_with_real_target": sum(
                            cell[(q, arm, h, conv)][0] and real_res[(q, h, conv)].full_core
                            for q in qids)})
    write_csv(out / f"coverage{SUFFIX}.csv", cov_rows)

    # ---- section 6 fallback coverage (windows only; no LLM future is read) -------------
    def ge7(qid: str, arm: str) -> bool:
        return len(accepted_keys[(qid, arm)]) >= E.MIN_ACCEPTED

    fb_rows = []
    for h in E.HORIZONS:
        for conv in ("primary", "sensitivity", "both"):
            convs = ["primary", "sensitivity"] if conv == "both" else [conv]
            for cancer, qids in strata:
                for name, (_field, arm) in E.CARDS.items():
                    elig = [q for q in qids
                            if all(fallback[(q, name, h, c)]["eligible"] for c in convs)]
                    with_cell = [q for q in elig if ge7(q, arm)]
                    fbs = [fallback[(q, name, h, convs[0])] for q in with_cell]
                    usable = [f["n_usable"] for f in fbs]
                    fb_rows.append({
                        "horizon_months": h, "convention": conv, "cancer": cancer,
                        "comparison": f"{name}_donors_vs_{arm}", "n_queries": len(qids),
                        "n_query_supported_ge1mo": sum(
                            all(query_support[(q, h, c)] >= E.MIN_WINDOW_MONTHS - E.EPS
                                for c in convs) for q in qids),
                        "n_ge7_usable_donors": len(elig),
                        "n_ge7_usable_and_llm_cell_ge7_accepted": len(with_cell),
                        "n_all_twenty_usable_and_llm_cell": sum(
                            all(fallback[(q, name, h, c)]["all_twenty_usable"]
                                for c in convs) for q in with_cell),
                        "usable_donors_median": (float(np.median(usable)) if usable else ""),
                        "usable_donors_min": (min(usable) if usable else ""),
                        "scored_donor_months_total": round(sum(f["scored_months"]
                                                               for f in fbs), 2),
                        "scored_months_per_usable_donor_mean": (
                            round(sum(f["scored_months"] for f in fbs) / sum(usable), 3)
                            if usable and sum(usable) else ""),
                        "floor_met_20": len(with_cell) >= E.PAIRED_FLOOR,
                        "reported_convention_for_sizes": convs[0]})
                four = [q for q in qids
                        if all(query_support[(q, h, c)] >= E.MIN_WINDOW_MONTHS - E.EPS
                               for c in convs) and all(ge7(q, a) for a in E.ARMS)]
                fb_rows.append({
                    "horizon_months": h, "convention": conv, "cancer": cancer,
                    "comparison": "four_arm_query_supported_window", "n_queries": len(qids),
                    "n_query_supported_ge1mo": sum(
                        all(query_support[(q, h, c)] >= E.MIN_WINDOW_MONTHS - E.EPS
                            for c in convs) for q in qids),
                    "n_ge7_usable_donors": "",
                    "n_ge7_usable_and_llm_cell_ge7_accepted": len(four),
                    "n_all_twenty_usable_and_llm_cell": "",
                    "usable_donors_median": "", "usable_donors_min": "",
                    "scored_donor_months_total": round(sum(
                        query_support[(q, h, convs[0])] for q in four), 2),
                    "scored_months_per_usable_donor_mean": "",
                    "floor_met_20": len(four) >= E.PAIRED_FLOOR,
                    "reported_convention_for_sizes": convs[0]})
    write_csv(out / f"fallback_coverage{SUFFIX}.csv", fb_rows)

    # ---- diagnostics: what blocks the treatment block now --------------------------------
    tx_rows = []
    for h in E.HORIZONS:
        for cancer, qids in strata:
            blocked = [real_res[(q, h, "primary")] for q in qids]
            only_tx = [r for r in blocked if r.blocks[:3] == (True, True, True)
                       and not r.blocks[3]]
            reasons = Counter(x for r in only_tx for x in r.reasons
                              if x.startswith("treatment_"))
            tx_rows.append({"horizon_months": h, "convention": "primary",
                            "cancer": cancer, "real_targets": len(qids),
                            "full_core": sum(r.full_core for r in blocked),
                            "blocked_only_by_treatment": len(only_tx),
                            "treatment_reasons_non_exclusive": "; ".join(
                                f"{k}={v}" for k, v in reasons.most_common())})
    write_csv(out / f"diagnostic_treatment_reasons{SUFFIX}.csv", tx_rows)

    # ---- private patient-level rows ------------------------------------------------------
    write_csv(private / f"real_resolution{SUFFIX}.csv", [
        {"query_id": q, "cancer": cancer_of[q], "horizon_months": h, "convention": c,
         **dict(zip(E.BLOCKS, r.blocks, strict=True)), "full_core": r.full_core,
         "reasons": ";".join(r.reasons), "survival_status": survival[(q, h)]}
        for (q, h, c), r in sorted(real_res.items())])
    write_csv(private / f"llm_cells{SUFFIX}.csv", [
        {"query_id": q, "arm": a, "horizon_months": h, "convention": c, "eligible": ok,
         "reason": why, "n_accepted": len(accepted_keys[(q, a)]),
         "repeat_reasons": ";".join(f"{k}" for k in sorted(s))}
        for (q, a, h, c), (ok, why, s) in sorted(cell.items())])
    write_csv(private / f"donor_cards{SUFFIX}.csv", [
        {"query_id": q, "card": n, "horizon_months": h, "convention": c, "eligible": ok,
         "reason": why, "n_members": len(member_res[(q, n, h, c)]),
         "n_members_resolved": sum(m.full_core for m in member_res[(q, n, h, c)]),
         "member_reasons": ";".join(sorted(s)),
         **{f"fallback_{k}": v for k, v in fallback[(q, n, h, c)].items()}}
        for (q, n, h, c), (ok, why, s) in sorted(card.items())])
    write_csv(private / f"query_sets{SUFFIX}.csv", [
        {"query_id": q, "cancer": cancer_of[q], "horizon_months": h, "convention": c,
         **{pop: v[0] for pop, v in sets.items()}}
        for (q, h, c), sets in sorted(membership.items())])

    # ---- manifest -----------------------------------------------------------------------
    code = [Path(__file__), ROOT / "scripts" / "four_arm_eligibility_v1.py",
            ROOT / "scripts" / "four_arm_eligibility_v2.py",
            ROOT / "scripts" / "four_arm_readers_v2.py", R.DRUG_TABLE,
            *[ROOT / "scripts" / "explore" / f for f in (
                "canonical_future_v1.py", "generated_future_v1.py",
                "genie_operator_registry_v2.py", "genie_operator_registry_v1.py",
                "disease_extent_v1.py", "trajectory_feature_poc_v1.py")],
            ROOT / "src" / "synthonccases" / "disease_state.py",
            ROOT / "src" / "synthonccases" / "oncollm" / "parse_utils.py"]
    manifest = {
        "what": "gate-3 eligibility-only precheck, version 2 (corrected readers R1-R5 and "
                "section-6 fallback coverage; no scores, distances, arm effects or "
                "outcome agreement)",
        "supersedes_as_current": "manifest.json (first precheck, kept as history)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_commit": git("rev-parse", "HEAD"),
        "analysis_tree_dirty": bool(git("status", "--porcelain")),
        "analysis_tree_note": "version-2 code is uncommitted at run time; code hashes below "
                              "identify the exact files executed",
        "specification": {str(p.relative_to(ROOT)): sha256_file(p)
                          for p in (SPEC, AMENDMENT, AMENDMENT_V2)},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            RUN / "attempts.jsonl", RUN / "manifest.json", RUN / "summary.json",
            UNIVERSE / "queries.jsonl", UNIVERSE / "manifest.json",
            PROMPTS / "quartets.jsonl", PROMPTS / "provenance.json")},
        "genie_release_sha256": {c: {"path": f"~/GenieBPC/{GV2.COHORTS[c]}", **v}
                                 for c, v in genie_hashes.items()},
        "input_checks": checks,
        "conventions": {k: {"record_rule": r, "persistence": p}
                        for k, (r, p) in E.CONVENTIONS.items()},
        "generated_reader": "four_arm_readers_v2.generated_course (cumulative truth, "
                            "identical under both conventions)",
        "parameters": {"horizons_months": list(E.HORIZONS), "min_accepted": E.MIN_ACCEPTED,
                       "card_size": E.CARD_SIZE, "paired_floor": E.PAIRED_FLOOR,
                       "section6_min_window_months": E.MIN_WINDOW_MONTHS,
                       "section6_min_usable_donors": E.MIN_USABLE_DONORS,
                       "status_validity_months": GV2.STATUS_VALIDITY,
                       "registry_days_per_month": R.REGISTRY_DAYS,
                       "generated_days_per_month": R.GENERATED_DAYS,
                       "generated_hold_break_months": round(R.HOLD_BREAK_MONTHS, 4)},
        "treatment_reconciliation_agent_counts": reconciliation,
        "real_courses_changed_by_adapter_on_own_calendar": dict(adapter_mismatch),
        "generated_projection_failures": dict(failures),
        "generated_terminal_kinds_at_24": dict(terminal_kinds),
        "generated_repeats_with_disease_unknown_after_landmark_at_24": gen_conflicts,
        "diagnostics": {
            "real_unrecognised_agents_exposed_by_12mo_query_count": dict(
                diag_unrecognised_real.most_common(30)),
            "generated_repeats_with_unrecognised_component_at_24": dict(
                diag_unrecognised_gen.most_common(30)),
            "generated_active_starts_read_as_switch_at_24": dict(sorted(
                active_start_conv.items()))},
        "outputs": sorted(str(p.relative_to(ROOT)) for p in out.rglob(f"*{SUFFIX}.csv")),
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / f"manifest{SUFFIX}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] aggregate tables and manifest in {out.relative_to(ROOT)} "
          f"({manifest['runtime_seconds']}s)")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path.name}")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def traced(out: Path, n: int, horizon: float = 12.0) -> int:
    """Private source-to-coordinate trace for `n` queries of the primary four-arm set.

    For each: the real target, the first accepted repeat of every arm and the first
    matched donor, through `E.coordinates` (raw, transformed, block-weighted). Values only:
    no distance, score, arm effect or agreement. Written under `private/` (gitignored);
    the console shows counts only.
    """
    sets = pd.read_csv(out / "private" / f"query_sets{SUFFIX}.csv")
    pick = sets[(sets["horizon_months"] == horizon) & (sets["convention"] == "primary")
                & sets["four_arm_intersection"]].sort_values("query_id")
    pick = pick.groupby("cancer").head(max(1, n // len(CANCERS)))["query_id"].tolist()[:n]
    queries = {json.loads(x)["query_id"]: json.loads(x)
               for x in (UNIVERSE / "queries.jsonl").read_text().splitlines()}
    quartets = {json.loads(x)["query_id"]: json.loads(x)
                for x in (PROMPTS / "quartets.jsonl").open()}
    first: dict[tuple[str, str], str] = {}
    for line in (RUN / "attempts.jsonl").open():
        r = json.loads(line)
        if r["query_id"] in pick and r["accepted"] and (r["query_id"], r["arm"]) not in first:
            first[(r["query_id"], r["arm"])] = r["raw_response"]
    trace = []
    for cancer in CANCERS:
        mine = [q for q in pick if queries[q]["cancer"] == cancer]
        if not mine:
            continue
        genie = GV2.GENIE_ROOT / GV2.COHORTS[cancer]
        patients = {p.pid: p for p in GV2.load(genie)}
        agents, _ = R.load_real_agents(genie)
        for qid in mine:
            q, qt = queries[qid], quartets[qid]
            p = patients[q["patient"]]
            tau = float(q["tau_months"])
            cal = E.calendar_for(p, tau, qt["assessment_times"])
            with E.convention("primary"):
                real = E.real_course(p, tau, horizon, agents.get(q["patient"], []))
                d0 = q["relevant_donors"][0]
                donor = E.real_course(patients[d0["patient"]], float(d0["tau_months"]),
                                      horizon, agents.get(d0["patient"], []))
            item = {"query_id": qid, "cancer": cancer, "horizon_months": horizon,
                    "calendar": cal,
                    "real": E.coordinates(real, cal, horizon),
                    "matched_donor_1": E.coordinates(donor, cal, horizon),
                    "matched_donor_1_window": E.donor_window(
                        E.supported(real, cal, horizon), donor, cal, horizon)}
            for arm in E.ARMS:
                kw = {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]
                      ["cutoff_date"], "landmark_state": qt["landmark_state"],
                      "landmark_presence": qt["landmark_presence"],
                      "landmark_sites": qt["landmark_sites"],
                      "landmark_extent": qt["landmark_extent"],
                      "index_regimen": qt["index_regimen"]}
                ledger = json.loads(strip_code_fence(first[(qid, arm)]))["ledger"]
                item[f"llm_{arm}_first_accepted"] = E.coordinates(
                    E.generated_course(ledger, horizon, **kw), cal, horizon)
            trace.append(item)
    path = out / "private" / f"traced_coordinates{SUFFIX}.json"
    path.write_text(json.dumps(trace, indent=1, default=str) + "\n")
    print(f"[traced] {len(trace)} queries traced to {path.relative_to(ROOT)} "
          "(private; values only, no distances)")
    return 0


def _es_terms(z: np.ndarray, y: np.ndarray) -> dict:
    """Both terms of the equal-weight energy score, for the trace."""
    first = float(np.linalg.norm(z - y, axis=1).mean())
    spread = float(np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)).mean())
    return {"mean_distance_to_truth": first, "half_mean_pairwise_distance": 0.5 * spread,
            "energy_score": first - 0.5 * spread}


def traced_scores(out: Path, horizon: float = 12.0, need_subsets: bool = False) -> int:
    """One patient traced from source record to every score (private output).

    Selection rule: among queries in the primary four-arm intersection at H = 12 months,
    ordered by SHA-256 of the query id, the first whose matched card passes the section-6
    fallback (>= 7 donors with >= 1 month of common window). With `need_subsets`, the first
    such query whose four-arm m* is below ten, so that the equal-size subsets are exercised
    (a wiring criterion added after the first trace showed m* = 10; it uses fan sizes only,
    no score). Written under `private/` (gitignored); the console shows no identifier and
    no score.
    """
    sets = pd.read_csv(out / "private" / f"query_sets{SUFFIX}.csv")
    pool = sets[(sets["horizon_months"] == horizon) & (sets["convention"] == "primary")
                & sets["four_arm_intersection"]]["query_id"].tolist()
    pool.sort(key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    queries = {json.loads(x)["query_id"]: json.loads(x)
               for x in (UNIVERSE / "queries.jsonl").read_text().splitlines()}
    quartets = {json.loads(x)["query_id"]: json.loads(x)
                for x in (PROMPTS / "quartets.jsonl").open()}
    cohorts: dict[str, tuple] = {}
    n_acc: Counter = Counter()
    if need_subsets:
        for line in (RUN / "attempts.jsonl").open():
            r = json.loads(line)
            n_acc[(r["query_id"], r["arm"])] += int(r["accepted"])
        pool = [x for x in pool if min(n_acc[(x, a)] for a in E.ARMS) < REPEATS]

    def cohort(cancer: str) -> tuple:
        if cancer not in cohorts:
            genie = GV2.GENIE_ROOT / GV2.COHORTS[cancer]
            cohorts[cancer] = ({p.pid: p for p in GV2.load(genie)},
                               R.load_real_agents(genie)[0])
        return cohorts[cancer]

    def labelled(course: E.Course, cal: list[float]) -> list:
        end = min(course.death, horizon)
        return S.labelled(E.observe(course.path, cal, end), course.death, horizon)

    chosen = None
    for qid in pool:
        q, qt = queries[qid], quartets[qid]
        patients, agents = cohort(q["cancer"])
        p = patients[q["patient"]]
        tau = float(q["tau_months"])
        cal = E.calendar_for(p, tau, qt["assessment_times"])
        with E.convention("primary"):
            real = E.real_course(p, tau, horizon, agents.get(q["patient"], []))
            donors = [E.real_course(patients[d["patient"]], float(d["tau_months"]), horizon,
                                    agents.get(d["patient"], []))
                      for d in q["relevant_donors"]]
        qsup = E.supported(real, cal, horizon)
        windows = [E.donor_window(qsup, d, cal, horizon) for d in donors]
        if E.fallback_patient(windows)["eligible"]:
            chosen = (qid, q, qt, cal, real, donors, windows, qsup)
            break
    if chosen is None:
        print("[trace-scores] no query meets the selection rule")
        return 1
    qid, q, qt, cal, real, donors, windows, qsup = chosen

    ledgers: dict[str, list[tuple[int, list]]] = defaultdict(list)
    for line in (RUN / "attempts.jsonl").open():
        r = json.loads(line)
        if r["query_id"] == qid and r["accepted"]:
            ledgers[r["arm"]].append((int(r["repeat_index"]),
                                      json.loads(strip_code_fence(r["raw_response"]))["ledger"]))
    rc = E.coordinates(real, cal, horizon)
    y = np.asarray(rc["weighted_vector"], float)
    trace: dict = {
        "query_id": qid, "cancer": q["cancer"], "horizon_months": horizon,
        "selection_rule": traced_scores.__doc__.split("Selection rule:")[1].split("Written")[0]
        .strip(),
        "calendar": cal,
        "real": {"source_path": list(real.path), "death": real.death,
                 "follow_end": real.follow_end,
                 "treatment_agents": [{"name": a.name, "classes": sorted(a.cls.modalities),
                                       "segments": a.segments, "unresolved": a.unresolved}
                                      for a in real.treatment.agents],
                 "observed_labelled": labelled(real, cal), "coordinates": rc,
                 "query_supported_window": qsup},
        "block_layout": {b: [sl.start, sl.stop] for b, sl in S.BLOCKS.items()},
        "block_weights": {b: math.sqrt(0.25 / len(n)) for b, n in E.COORDINATES},
        "arms": {}}
    fans: dict[str, np.ndarray] = {}
    paths: dict[str, list] = {}
    for arm in E.ARMS:
        kw = {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]["cutoff_date"],
              "landmark_state": qt["landmark_state"],
              "landmark_presence": qt["landmark_presence"],
              "landmark_sites": qt["landmark_sites"],
              "landmark_extent": qt["landmark_extent"], "index_regimen": qt["index_regimen"]}
        reps = []
        for rep, ledger in sorted(ledgers[arm]):
            c = E.generated_course(ledger, horizon, **kw)
            co = E.coordinates(c, cal, horizon)
            reps.append({"repeat": rep, "source_path": list(c.path), "death": c.death,
                         "regimen_starts": [st.time for st in c.treatment.starts],
                         "observed_labelled": labelled(c, cal), "coordinates": co,
                         "timing": S.progression_to_regimen(
                             co["observed_path"],
                             [st.time for st in c.treatment.starts if st.time > 0],
                             min(c.death, horizon))})
        if len(reps) < E.MIN_ACCEPTED or any(r["coordinates"]["weighted_vector"] is None
                                              for r in reps):
            raise RuntimeError("selected query violates the cell floor; selection is wrong")
        fans[arm] = np.asarray([r["coordinates"]["weighted_vector"] for r in reps], float)
        paths[arm] = [r["observed_labelled"] for r in reps]
        trace["arms"][arm] = {"repeats": reps}
    m_star = S.four_arm_m_star([len(fans[a]) for a in E.ARMS])
    trace["m_star"] = m_star
    for arm in E.ARMS:
        z = fans[arm]
        keys = ("four_arm", qid, arm, horizon)
        subsets = S.equal_size_subsets(len(z), m_star, *keys)
        js = S.joint_scores(z, y, m_star, *keys)
        a = trace["arms"][arm]
        a["seed_keys"] = list(keys)
        a["seed"] = S.stable_seed(*keys)
        a["subsets_first_5"] = None if subsets is None else subsets[:5].tolist()
        a["as_delivered_terms"] = _es_terms(z, y)
        if subsets is not None:
            a["first_subset_terms"] = _es_terms(z[subsets[0]], y)
        a["scores"] = {"equal_size_primary": js.equal_size, "as_delivered": js.as_delivered,
                       "fair_llm_only": js.fair, "blocks_equal_size": js.blocks_equal_size,
                       "m": js.m}
        a["dependence"] = S.dependence_check(z, y, qid, arm, horizon)
        a["arm_diagnostic_query_window"] = S.arm_diagnostic(labelled(real, cal),
                                                            paths[arm], qsup)
    # section 6: matched card versus the patients-like-me arm (the arm that received it)
    dlab = [labelled(d, cal) for d in donors]
    trace["donor_matched"] = {
        "members": [{"rank": i + 1, "window": w, "window_months": E.measure(w),
                     "observed_labelled": dl, "death": d.death}
                    for i, (d, w, dl) in enumerate(zip(donors, windows, dlab, strict=True))],
        "fallback": E.fallback_patient(windows),
        "per_donor_losses": [
            None if E.measure(w) < E.MIN_WINDOW_MONTHS - E.EPS else
            {"donor": S.course_losses(labelled(real, cal), dl, w),
             "llm_patients_like_me_mean": {
                 k: float(np.mean([S.course_losses(labelled(real, cal), r, w)[k]
                                   for r in paths["patients_like_me"]]))
                 for k in (*S.LOSSES, "unresolved_fraction")}}
            for dl, w in zip(dlab, windows, strict=True)],
        "diagnostic": S.donor_diagnostic(labelled(real, cal), list(zip(dlab, windows,
                                                                        strict=True)),
                                         paths["patients_like_me"])}
    trace["real"]["timing"] = S.progression_to_regimen(
        rc["observed_path"], [st.time for st in real.treatment.starts if st.time > 0],
        min(real.death, horizon))
    trace["selection"] = "first with m* below ten" if need_subsets else "first eligible"
    name = "traced_scores_subsets" if need_subsets else "traced_scores"
    path = out / "private" / f"{name}{SUFFIX}.json"
    path.write_text(json.dumps(trace, indent=1, default=str) + "\n")
    print(f"[trace-scores] one query traced at H={horizon:g} to {path.relative_to(ROOT)} "
          "(private; no identifier or score printed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precheck-only", action="store_true",
                    help="eligibility-only precheck (the only implemented mode)")
    ap.add_argument("--traced-example", type=int, default=0, metavar="N",
                    help="after the precheck, trace N queries through the coordinate "
                         "builder (private output, values only)")
    ap.add_argument("--trace-scores", action="store_true",
                    help="gate 4 step 2: trace ONE query from record to every score "
                         "(private output; reads the saved precheck tables)")
    ap.add_argument("--score", action="store_true",
                    help="gate 4 step 3: the cohort extraction (four_arm_extract_v1), after "
                         "the precheck tables exist; aggregate outputs plus private rows")
    args = ap.parse_args()
    if args.score and not args.precheck_only:
        import four_arm_extract_v1 as X
        return X.extract(OUT)
    if args.trace_scores and not args.precheck_only:
        return traced_scores(OUT) or traced_scores(OUT, need_subsets=True)
    if not args.precheck_only:
        ap.error("choose --precheck-only, --trace-scores or --score")
    code = run(OUT)
    if args.traced_example:
        code = code or traced(OUT, args.traced_example)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
