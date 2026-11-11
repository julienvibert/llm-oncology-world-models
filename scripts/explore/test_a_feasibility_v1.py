#!/usr/bin/env python3
"""Test A, stage 1: feasibility counts only (no forecasts, no outcome comparisons).

Specification of 14 Oct 2026 (revision 4): counts only, existing data, no API calls.

Per cancer, reports:
  1. which history descriptors can be computed before the decision, and their missingness;
  2. documented history in the 12 months before the decision;
  3. comparators meeting the shared exact criteria (cancer, treatment class started, regimen
     depth, latest status), among eligible patients;
  4. follow-up sufficient for a 12-month forecast (death or at least 12 months of follow-up).

Decision eligibility uses only information available at the decision: the start of a recorded
regimen after metastatic disease is documented, patient alive at the start, at least one
assessment after the metastatic diagnosis and on or before the decision. Follow-up is NOT
required for eligibility (unlike the registry experiment's landmark rule); it is only counted.
One decision per patient, drawn uniformly among eligible ones (seed 20261014).

Reuses the existing readers (genie_operator_registry_v2: statuses valid for 4 months, regimens
and their modalities, metastasis dates). Note: those readers use 30.4 days per month; counts are
unaffected, forecasting would align the convention first.

GENIE-BPC under DUA: aggregate counts only, written to gitignored outputs.
Run: uv run python scripts/explore/test_a_feasibility_v1.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import disease_extent_v1 as DE  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_test_a_feasibility"
SEED = 20261014
WINDOW = 12.0          # months of history before the decision
MIN_DOC = 3.0          # documented months required in that window
K = 20
HORIZON = 12.0
STATES = ("responding", "stable", "progressing", "mixed")


def metastatic_onset(p: GV2.Patient) -> float | None:
    """Earliest time at which metastatic disease is documented (None if never)."""
    mets = p.context.get("mets")
    for t in sorted({w for w, _ in p.status} | {ln["start"] for ln in p.lines}):
        if DE.extent_at_landmark(mets, t).documented == "yes":
            # refine: the extent reader answers "documented by t"; take the first such t
            return t
    return None


def eligible_decisions(p: GV2.Patient) -> list[dict]:
    onset = metastatic_onset(p)
    if onset is None:
        return []
    out = []
    for ln in p.lines:
        tau = float(ln["start"])
        if tau < onset or p.death <= tau:
            continue
        if not any(onset <= w <= tau for w, _ in p.status):
            continue
        out.append({"tau": tau, "line": ln})  # onset is the first event time with metastases documented
    return out


def describe(p: GV2.Patient, tau: float, ln: dict) -> dict:
    lo = tau - WINDOW
    docs = GV2.documented(p, tau)
    occ = {s: sum(max(0.0, min(b, tau) - max(a, lo)) for a, b in docs.get(s, [])) for s in STATES}
    seen = [(w, s) for w, s in p.status if w <= tau]
    entries = Counter()
    prev = None
    for w, s in seen:
        if s != prev and s in ("progressing", "responding"):
            entries[s] += 1
        prev = s
    earlier = [x for x in p.lines if x["start"] < tau - GV2.SAME_DAY]
    prev_line = max(earlier, key=lambda x: x["start"]) if earlier else None
    prev_status = None
    if prev_line:
        inside = [s for w, s in seen if prev_line["start"] <= w <= min(prev_line["stop"], tau)]
        prev_status = inside[-1] if inside else None
    exposure = sum(max(0.0, min(x["stop"], tau) - max(x["start"], lo)) for x in earlier)
    number = float(ln.get("number", np.nan))
    return {
        "doc_months": GV2.observed_span(p, lo, tau),
        "occupancy_any": sum(occ.values()) > 0,
        "entries_progression": entries["progressing"],
        "entries_response": entries["responding"],
        "has_previous_regimen": prev_line is not None,
        "previous_regimen_status_known": prev_status is not None,
        "previous_regimen_duration": (min(prev_line["stop"], tau) - prev_line["start"])
        if prev_line and np.isfinite(prev_line["stop"]) else None,
        "exposure_months": exposure,
        "latest_status": seen[-1][1] if seen else "none",
        "depth": "1" if number == 1 else "2" if number == 2 else "3+" if number >= 3 else "unknown",
        "treatment_class": "+".join(sorted(ln.get("modalities", ()))) or "unknown",
        "followup_ok": p.death <= tau + HORIZON or p.last_contact >= tau + HORIZON,
        "died_12": p.death <= tau + HORIZON,
    }


def run(cancer: str) -> dict:
    patients = GV2.load(GV2.GENIE_ROOT / GV2.COHORTS[cancer])
    rng = np.random.default_rng(SEED)
    rows = []
    n_patients_any = 0
    for p in sorted(patients, key=lambda q: q.pid):
        dec = eligible_decisions(p)
        if not dec:
            continue
        n_patients_any += 1
        d = dec[int(rng.integers(len(dec)))]
        rows.append(describe(p, d["tau"], d["line"]))
    n = len(rows)
    doc = np.array([r["doc_months"] for r in rows])
    ok = [r for r in rows if r["doc_months"] >= MIN_DOC]
    key = lambda r: (r["treatment_class"], r["depth"], r["latest_status"])  # noqa: E731
    strata = Counter(key(r) for r in ok)
    comps = np.array([strata[key(r)] - 1 for r in ok])
    fu = np.array([r["followup_ok"] for r in ok])
    pct = lambda x: round(100 * float(np.mean(x)), 1) if len(x) else None  # noqa: E731
    return {
        "patients_with_eligible_decision": n_patients_any,
        "decisions_drawn": n,
        "descriptors": {
            "occupancy_computable_pct": pct([r["occupancy_any"] for r in rows]),
            "has_previous_regimen_pct": pct([r["has_previous_regimen"] for r in rows]),
            "previous_regimen_status_known_pct_of_those_with_one": pct(
                [r["previous_regimen_status_known"] for r in rows if r["has_previous_regimen"]]),
            "previous_regimen_duration_known_pct_of_those_with_one": pct(
                [r["previous_regimen_duration"] is not None for r in rows if r["has_previous_regimen"]]),
            "treatment_class_unknown_pct": pct([r["treatment_class"] == "unknown" for r in rows]),
            "depth_unknown_pct": pct([r["depth"] == "unknown" for r in rows]),
            "latest_status_counts": dict(Counter(r["latest_status"] for r in rows)),
        },
        "documented_history": {
            "median_documented_months_prior_12": round(float(np.median(doc)), 2) if n else None,
            "quartiles": [round(float(q), 2) for q in np.percentile(doc, [25, 75])] if n else None,
            "below_3_months_pct": pct(doc < MIN_DOC),
            "eligible_with_3_months": len(ok),
        },
        "comparators_exact_criteria": {
            "strata": len(strata),
            "median_available": float(np.median(comps)) if len(comps) else None,
            "pct_with_at_least_20": pct(comps >= K),
            "pct_with_at_least_10": pct(comps >= 10),
            "pct_with_none": pct(comps == 0),
        },
        "followup_12_months": {
            "pct_death_or_12m_followup": pct(fu),
            "n_short_followup_alive": int((~fu).sum()) if len(fu) else 0,
            "n_deaths_within_12m": int(sum(r["died_12"] for r in ok)),
            "n_alive_with_12m_followup": int(sum(r["followup_ok"] and not r["died_12"] for r in ok)),
        },
        "confirmation_group": "identities not stored (counts only in the query-universe manifest); "
                              "earlier history analyses and this test use all eligible patients, "
                              "so the group cannot be assumed untouched",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {c: run(c) for c in ("NSCLC", "CRC", "BRCA")}
    (OUT / "feasibility_counts.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
