"""N2 state occupancy over time (completion plan 28 Oct 2026, N2): traced example only.

Question. Month by month over 24 months after the landmark, what share of patients is
responding, stable, progressing, without active disease, unresolved or dead, in the registry
course and in each arm's generated futures?

Two modes. --trace: one patient traced (every month, the state
assigned to the observed course and to each accepted future, and why). --extract: the cohort,
aggregate only.

Inclusion (--extract), fixed before extraction and identical to the Fig. 5c disagreement
population at 24 months (`four_arm_extract_v1`, section-6 four-arm diagnostic): primary
convention; at least E.MIN_WINDOW_MONTHS of documented time to 24 months; in every arm at
least E.MIN_ACCEPTED accepted futures and every accepted future readable at 24 months.
Common patient set: at month m every source (observed and four arms) uses the same patients,
those not censored at m. Generated shares are averaged within patient first, then over
patients.

Month clocks: courses come from the scoring readers, registry months = days / 30.4 and
generated months = days / 30.4375 (`four_arm_readers_v2`), as in Fig. 5; the largest offset
between the clocks within 24 months is 24 x (1 - 30.4 / 30.4375) = 0.03 months (about one day).
Kept, not converted, so that N2 and the scored comparison read courses identically.

Patient. The Figure 5a case (declared rule `_fans.select_eventful_query`, which reads the
observed course only), so the trace can be checked against a figure the reader already sees.

Readings (as in the scored comparison, `score_four_arm_final_v1.traced_scores`):
  observed   registry course under the primary convention, read on the patient's own
             assessment calendar: S.labelled(E.observe(path, calendar, end), death, H)
  generated  each accepted future, read on the SAME calendar (primary reading), and its raw
             asserted path (sensitivity reading)
Month m (0..24): one rule for both sides, `month_state` (death first, then censoring, then the
reading at m). Courses are built and read to READ_HORIZON = 25 months so that month 24 is read
like every other month (an assessment exactly at 24 is seen); nothing after month 24 is used. Month 0 is unresolved when the record
has no documented state at the landmark (genuine missing information; traced 28 Oct). A month is in the denominator if the patient is
dead by m or still followed at m (follow_end > m); otherwise it is censored and excluded,
never counted as unresolved. Shares are therefore distributions among the patients retained at
each month, not censoring-adjusted population estimates. Death is kept separate from unresolved time.

Output: trace, outputs/four_arm_preprint_20260928/private/n2_trace_v1.json (gitignored,
GENIE-BPC DUA); extraction, outputs/exploratory_n2_state_occupancy/ (aggregate shares and
counts only). The console prints counts only, no identifier.

Run: uv run python scripts/explore/n2_state_occupancy_v1.py --trace | --extract
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "scripts" / "figures" / "panels"))
sys.path.insert(0, str(ROOT / "src"))
import _fans as F  # noqa: E402
import four_arm_eligibility_v2 as E  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402
import four_arm_scores_v1 as S  # noqa: E402

from synthonccases.oncollm.parse_utils import strip_code_fence  # noqa: E402

HORIZON = 24.0
READ_HORIZON = HORIZON + 1.0  # read month 24 as an ordinary month; nothing later is used
MONTHS = list(range(0, 25))
OUT = ROOT / "outputs" / "four_arm_preprint_20260928" / "private" / "n2_trace_v1.json"
AGG = ROOT / "outputs" / "exploratory_n2_state_occupancy"
SCORED = ROOT / "outputs" / "four_arm_preprint_20260928" / "private" / "course_patient_rows_scores_20261012.csv"
#: registry months -> generated-clock months (days / 30.4 -> days / 30.4375), for the clock check only
CLOCK = R.REGISTRY_DAYS / R.GENERATED_DAYS
DENOMINATOR = ("distribution among the patients retained at that month (dead by then, or alive and still "
               "followed); censored patients leave the denominator; not a censoring-adjusted population estimate")


def sha256(path: Path) -> str:
    """SHA-256 of a file (code and input provenance for the manifest)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


CENSORED = "censored"


def state_at(segs: list, t: float) -> str | None:
    """Label of the segment covering t (half-open [a, b)); None if no segment covers t."""
    for a, b, s in segs:
        if a - 1e-9 <= t < b - 1e-9:
            return s
    return None


def month_state(segs: list, death: float, m: float, follow_end: float = float("inf")) -> str:
    """THE month rule, the same for observed and generated courses, in this order:
    (1) dead if death <= m (a death exactly at month m counts as dead at m);
    (2) censored if follow-up ends at or before m (observed courses only; generated futures
        have no own censoring and are excluded through the patient's denominator);
    (3) otherwise the reading's state at m, month 24 included (readings extend to
        READ_HORIZON, so a change exactly at month 24 is seen).
    A month with no covering segment is "unresolved"."""
    if death <= m + 1e-9:
        return S.DEAD
    if follow_end <= m + 1e-9:
        return CENSORED
    return state_at(segs, m) or S.UNRESOLVED


def read(course: E.Course, cal: list[float]) -> list:
    """The scored reading: the course on the patient's calendar, death absorbing."""
    end = min(course.death, READ_HORIZON)
    return S.labelled(E.observe(course.path, cal, end), course.death, READ_HORIZON)


def raw(course: E.Course) -> list:
    """Sensitivity reading: the asserted path itself, death absorbing, no calendar."""
    end = min(course.death, READ_HORIZON)
    segs = [(a, min(b, end), s) for a, b, s in course.path if min(b, end) > a]
    return S.labelled(segs, course.death, READ_HORIZON)


def observed_month(real: E.Course, segs: list, cal: list[float], m: int) -> dict:
    """Observed state at month m (month rule above), with the reason."""
    st = month_state(segs, real.death, m, real.follow_end)
    if st == S.DEAD:
        why = f"death at month {real.death:.2f}"
    elif st == CENSORED:
        why = f"censored: follow-up ends at month {real.follow_end:.2f}"
    else:
        why = f"last calendar read at month {max(c for c in [0.0, *cal] if c <= m + 1e-9):.2f}"
    return {"month": m, "state": st, "why": why}


def trace() -> int:
    qid, info = F.select_eventful_query()
    q = next(x for x in F._jsonl(F.QUERIES) if x["query_id"] == qid)
    qt = next(x for x in F._jsonl(F.PROMPTS) if x["query_id"] == qid)
    patients, agents = F._cohort(q["cancer"])
    p = patients[q["patient"]]
    tau = float(q["tau_months"])
    cal = E.calendar_for(p, tau, qt["assessment_times"])
    with E.convention("primary"):
        real = E.real_course(p, tau, READ_HORIZON, agents.get(q["patient"], []))
    real_segs = read(real, cal)
    obs = [observed_month(real, real_segs, cal, m) for m in MONTHS]
    in_denominator = [o["state"] != CENSORED for o in obs]

    arms: dict = {}
    for row in F._jsonl(F.RUN):
        if row["query_id"] != qid or not row.get("accepted"):
            continue
        arm = row["arm"]
        kw = {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]["cutoff_date"],
              "landmark_state": qt["landmark_state"], "landmark_presence": qt["landmark_presence"],
              "landmark_sites": qt["landmark_sites"], "landmark_extent": qt["landmark_extent"],
              "index_regimen": qt["index_regimen"]}
        ledger = json.loads(strip_code_fence(row["raw_response"]))["ledger"]
        try:
            c = E.generated_course(ledger, READ_HORIZON, **kw)
        except R.NotScoreable:
            arms.setdefault(arm, {"futures": [], "not_scoreable": 0})["not_scoreable"] += 1
            continue
        on_cal, as_said = read(c, cal), raw(c)
        arms.setdefault(arm, {"futures": [], "not_scoreable": 0})["futures"].append({
            "repeat": int(row["repeat_index"]), "death": c.death,
            "calendar_reading": [month_state(on_cal, c.death, m) if keep else None
                                 for m, keep in zip(MONTHS, in_denominator)],
            "raw_reading": [month_state(as_said, c.death, m) if keep else None
                            for m, keep in zip(MONTHS, in_denominator)]})
    for a in arms.values():
        a["futures"].sort(key=lambda f: f["repeat"])
        n = len(a["futures"])
        for key in ("calendar_reading", "raw_reading"):
            a[f"share_{key}"] = [{s: k / n for s, k in Counter(f[key][i] for f in a["futures"]).items()}
                                 if in_denominator[i] else None for i in range(len(MONTHS))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "what": "N2 traced example (private, GENIE-BPC DUA): Figure 5a case, every month 0-24",
        "case_rule": "_fans.select_eventful_query (reads the observed course only)",
        "case_tag": info["tag"], "query_id": qid, "cancer": q["cancer"],
        "calendar": cal, "real_follow_end": real.follow_end, "real_death": real.death,
        "observed_labelled_segments": real_segs, "observed_by_month": obs,
        "arms": arms}, indent=1, default=str) + "\n")
    n_cens = sum(not k for k in in_denominator)
    print(f"[n2 trace] one case; {len(cal)} calendar reads; {sum(in_denominator)} months in the "
          f"denominator, {n_cens} censored; futures per arm "
          + ", ".join(f"{a} {len(v['futures'])} (+{v['not_scoreable']} not scoreable)" for a, v in arms.items())
          + f"; written to {OUT.relative_to(ROOT)} (private)")
    return 0


def generated_kw(qt: dict, arm: str) -> dict:
    return {"cutoff": qt["arms"][arm]["conditioning_payload"]["fixed_events"]["cutoff_date"],
            "landmark_state": qt["landmark_state"], "landmark_presence": qt["landmark_presence"],
            "landmark_sites": qt["landmark_sites"], "landmark_extent": qt["landmark_extent"],
            "index_regimen": qt["index_regimen"]}


def extract() -> int:
    queries = {q["query_id"]: q for q in F._jsonl(F.QUERIES)}
    prompts = {q["query_id"]: q for q in F._jsonl(F.PROMPTS)}
    ledgers: dict[tuple, list] = defaultdict(list)
    for row in F._jsonl(F.RUN):
        if row.get("accepted"):
            ledgers[(row["query_id"], row["arm"])].append(row["raw_response"])
    excluded: Counter = Counter()
    clock: Counter = Counter()
    # per patient: observed month states and, per arm and reading, within-patient shares
    patients_out: list[dict] = []
    for qid in sorted(queries):
        q, qt = queries[qid], prompts[qid]
        pats, agents = F._cohort(q["cancer"])
        p = pats[q["patient"]]
        tau = float(q["tau_months"])
        cal = E.calendar_for(p, tau, qt["assessment_times"])
        with E.convention("primary"):
            real24 = E.real_course(p, tau, HORIZON, agents.get(q["patient"], []))
            real = E.real_course(p, tau, READ_HORIZON, agents.get(q["patient"], []))
        if E.measure(E.supported(real24, cal, HORIZON)) < E.MIN_WINDOW_MONTHS - S.EPS:
            excluded["documented time < minimum window"] += 1
            continue
        fans: dict[str, list] = {}
        for arm in F.ARMS:
            courses = []
            for text in ledgers[(qid, arm)]:
                try:
                    ledger = json.loads(strip_code_fence(text))["ledger"]
                    E.generated_course(ledger, HORIZON, **generated_kw(qt, arm))     # readable at 24
                    courses.append(E.generated_course(ledger, READ_HORIZON, **generated_kw(qt, arm)))
                except (R.NotScoreable, ValueError, KeyError):
                    courses.append(None)
            fans[arm] = courses
        if any(len(f) < E.MIN_ACCEPTED or any(c is None for c in f) for f in fans.values()):
            excluded["an arm below the minimum fan or with an unreadable future"] += 1
            continue
        real_segs = read(real, cal)
        obs = [month_state(real_segs, real.death, m, real.follow_end) for m in MONTHS]
        rec = {"query_id": qid, "cancer": q["cancer"], "observed": obs, "arms": {}}
        # clock check: registry-derived times (calendar, observed course) moved to the generated clock
        cal_h = [c * CLOCK for c in cal]
        segs_h = [(a * CLOCK, b * CLOCK, st) for a, b, st in real_segs]
        obs_h = [month_state(segs_h, real.death * CLOCK, m, real.follow_end * CLOCK) for m in MONTHS]
        clock["observed_patient_months"] += sum(o != CENSORED for o in obs)
        clock["observed_changed"] += sum(a != b for a, b in zip(obs, obs_h, strict=True))
        for arm, f in fans.items():
            for reading, fn in (("calendar", lambda c: read(c, cal)), ("raw", raw)):
                states = [[month_state(fn(c), c.death, m) for m in MONTHS] for c in f]
                rec["arms"][(arm, reading)] = [Counter(st[i] for st in states) for i in range(len(MONTHS))]
                rec.setdefault("fan", {})[arm] = len(f)
                if reading == "calendar":
                    for c, st in zip(f, states, strict=True):
                        st_h = [month_state(read(c, cal_h), c.death, m) for m in MONTHS]
                        clock["generated_future_months"] += len(MONTHS)
                        clock["generated_changed"] += sum(a != b for a, b in zip(st, st_h, strict=True))
        patients_out.append(rec)

    # the included patients must be exactly the saved Fig. 5c scoring population at 24 months
    scored = pd.read_csv(SCORED)
    scored = set(scored.loc[scored.horizon_months.eq(HORIZON) & scored.convention.eq("primary"), "query_id"])
    mine = {r["query_id"] for r in patients_out}
    assert mine == scored, (len(mine - scored), len(scored - mine))

    rows = []
    for cancer in ("all", *sorted({r["cancer"] for r in patients_out})):
        group = [r for r in patients_out if cancer in ("all", r["cancer"])]
        for i, m in enumerate(MONTHS):
            live = [r for r in group if r["observed"][i] != CENSORED]
            n, n_cens = len(live), len(group) - len(live)
            if not n:
                continue
            obs_count = Counter(r["observed"][i] for r in live)
            for state, k in sorted(obs_count.items()):
                rows.append({"cancer": cancer, "month": m, "source": "observed", "reading": "registry",
                             "state": state, "share": round(k / n, 6), "n_patients": n,
                             "n_censored": n_cens, "n_group": len(group)})
            for arm in F.ARMS:
                for reading in ("calendar", "raw"):
                    acc: Counter = Counter()
                    for r in live:
                        cnt = r["arms"][(arm, reading)][i]
                        tot = sum(cnt.values())
                        for state, k in cnt.items():
                            acc[state] += k / tot / n
                    assert abs(sum(acc.values()) - 1) < 1e-9
                    for state, v in sorted(acc.items()):
                        rows.append({"cancer": cancer, "month": m, "source": arm, "reading": reading,
                                     "state": state, "share": round(v, 6), "n_patients": n,
                                     "n_censored": n_cens, "n_group": len(group)})
    AGG.mkdir(parents=True, exist_ok=True)
    with (AGG / "n2_state_occupancy_v1.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fan_sizes = Counter(n for r in patients_out for n in r["fan"].values())
    (AGG / "manifest.json").write_text(json.dumps({
        "script": "scripts/explore/n2_state_occupancy_v1.py --extract",
        "horizon_months": HORIZON, "read_horizon_months": READ_HORIZON,
        "min_accepted": E.MIN_ACCEPTED, "min_window_months": E.MIN_WINDOW_MONTHS,
        "convention": "primary", "queries": len(queries), "included": len(patients_out),
        "included_by_cancer": dict(Counter(r["cancer"] for r in patients_out)),
        "excluded": dict(excluded), "fan_sizes": dict(sorted(fan_sizes.items())),
        "patient_set": "identical to the saved Fig. 5c scoring population at 24 months (asserted by identity)",
        "denominator": DENOMINATOR,
        "month_clocks": {"registry_days": R.REGISTRY_DAYS, "generated_days": R.GENERATED_DAYS,
                         "kept": "as in the scoring readers (amendment to the agreed harmonization, Codex 28 Oct)",
                         "harmonization_check": {
                             **dict(clock),
                             "observed_share_changed": round(clock["observed_changed"] / clock["observed_patient_months"], 5),
                             "generated_share_changed": round(clock["generated_changed"] / clock["generated_future_months"], 5)}},
        "provenance": {"code_sha256": sha256(Path(__file__)),
                       "inputs_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in (F.RUN, F.QUERIES, F.PROMPTS)}},
        "aggregate_only": True}, indent=1) + "\n")
    print(f"[n2 extract] {len(patients_out)} of {len(queries)} patients included "
          f"({dict(excluded)}); {len(rows)} aggregate rows to {AGG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace", action="store_true", help="write the one-patient traced example")
    mode.add_argument("--extract", action="store_true", help="the cohort, aggregate shares only")
    args = ap.parse_args()
    sys.exit(trace() if args.trace else extract())
