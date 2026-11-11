#!/usr/bin/env python3
"""The reference-evidence card: one renderer, two donor sets, no way to tell them apart.

The card is the ONLY thing that differs between the patients-like-me arm and the generic
cohort arm, so it must be impossible for the model to infer which it received. That means:

  * one neutral header for both, naming ONLY what is true of both sets -- same cancer,
    comparable era. Treatment class is deliberately absent: the generic cohort is not
    treatment-matched, so claiming it would be false for one arm and would identify the
    card from its first line;
  * no arm label, no similarity score, no mention of relevance or selection, no hint of
    the criteria that chose one set and not the other;
  * DONOR ORDER CARRIES NO SIGNAL. Sorting by distance would encode the selection rule in
    the ordering itself, and sorting by outcome would leak the answer, so donors are
    ordered by a deterministic hash of their identifier;
  * identical fields, identical missing-value conventions, identical donor count.

Terminal states are never ambiguous. A donor either died at a stated month or was last
observed alive at a stated month; censoring is never rendered as though the course ended
or as though the patient survived the whole horizon.

Everything is relative months from that donor's own treatment start, out to the maximum
scored horizon of 24 months. One year per donor, no dates, no identifiers.

NO ENGINEERED SUMMARY. The card carries the donors' raw courses and nothing else. A
summary in our own outcome coordinates would show the model the ruler we intend to score
it with, and would invite it to reproduce our means and quartiles -- the pilot's failure
mode, where location was corrected and dispersion collapsed. It would also privilege one
horizon, when we intend to score at 3, 6, 12 and 24 months. The courses support every
horizon because they are the events themselves.

This rendering is not claimed to be optimal. It is one simple, clinically motivated
choice: twenty retrieved real courses supplied as in-context examples.

GOVERNANCE. GENIE-BPC under DUA: cards are written only to the gitignored outputs tree.

Run:  uv run python scripts/explore/render_reference_card_v1.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_future_v1 as CF  # noqa: E402
import disease_extent_v1 as DE  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
UNIVERSE = ROOT / "outputs" / "exploratory_query_universe"
OUT = ROOT / "outputs" / "exploratory_reference_cards"
H = 12.0

CARD_HORIZON = 24.0
#: Donor age is rendered as a BAND, not a year. Matching already uses a decade window, so
#: banding costs the model nothing it was using -- and the card now combines age, calendar
#: year, histology, molecular status, metastatic sites, exact regimens and outcomes for a
#: real patient, which is materially more identifying than the earlier format even without
#: names. The query patient's own age is not banded: they are the subject of the question,
#: not evidence about someone else.
AGE_BAND = 5
#: GENIE's curated categories, rendered clinically WITHOUT inventing RECIST precision.
#: The source does not establish partial or complete response, so we do not claim it.
CLINICAL = {"responding": "response or improvement", "stable": "stable disease",
            "progressing": "progressive disease", "mixed": "mixed response"}
SALTS = ("disodium", "hydrochloride", "hcl", "calcium", "sodium", "sulfate", "tartrate",
         "mesylate", "acetate", "citrate", "dihydrate", "succinate", "maleate",
         "phosphate", "besylate", "fumarate")

#: The header must be TRUE OF BOTH SETS or it becomes the tell. The generic cohort is
#: matched on cancer and era only, so treatment class cannot be claimed here -- saying
#: "same treatment class" would be false for one arm and would let the model identify
#: which card it received from the header alone.
HEADER = ("Reference cohort: {n} patients with the same cancer, from a comparable "
          "calendar era. Each began the regimen shown at month 0 of their own course. "
          "All times are months from that start. Stage, metastatic extent and molecular "
          "status are as recorded at that patient's own start.")

#: How a donor's molecular status is stated. Never 'wild-type': a panel that reported no
#: relevant alteration has not established a normal gene.
DRIVER = {"unknown": "molecular status unknown",
          "none recorded on the available panel":
              "no relevant alteration recorded on the available panel"}


def canonical(label: str) -> str:
    """A canonical regimen name, not an uncontrolled raw string.

    Salt and formulation words carry no clinical information and vary between records, so
    'Pemetrexed Disodium' and 'Pemetrexed' must render identically or the same regimen
    would look like two. Repeated 'Investigational Drug' entries collapse to one.
    """
    name, _unnamed = regimen_name(label)
    return name


def regimen_name(label: str) -> tuple[str, bool]:
    """Returns the rendered name and whether the regimen is ENTIRELY unnamed.

    A trial regimen with named agents keeps them and marks the extra component as
    investigational, because the named agents are what the model can reason about. A
    regimen that is only 'Investigational Drug' identifies no treatment at all, and
    treating it as a regimen identity would let two unrelated trials look like the same
    therapy.
    """
    named, investigational = [], False
    for piece in str(label).replace(";", ",").split(","):
        words = [w for w in piece.strip().lower().split() if w and w not in SALTS]
        if not words:
            continue
        text = " ".join(words)
        if "investigational" in text:
            investigational = True
        else:
            named.append(text)
    unique = list(dict.fromkeys(named))
    if not unique:
        return ("unnamed investigational therapy" if investigational
                else "unspecified regimen"), True
    body = " + ".join(sorted(unique))
    return (f"{body} + investigational agent" if investigational else body), False


def stable_order(donor_ids: list[str]) -> list[str]:
    """Deterministic and signal-free. Not distance, which would encode the selection rule;
    not outcome, which would leak the answer."""
    return sorted(donor_ids,
                  key=lambda d: hashlib.sha256(d.encode()).hexdigest())


def plural(n: float) -> str:
    """'1 month', not '1 months'. Small, but the card is read by a clinician."""
    return f"{n:.0f} month{'' if abs(round(n)) == 1 else 's'}"


def prior_treatment(patient, tau: float) -> str:
    """The last two named regimens with durations; older therapy by count and total.

    A donor without history is not an example, it is an outcome with a partial label -- the
    model can see what happened but not to whom, and cannot use the history we matched on.
    Deep courses are summarised rather than listed so every donor costs the same number of
    lines: uneven length would itself encode regimen depth, which differs systematically
    between the two arms.
    """
    prior = [ln for ln in patient.lines if ln["start"] < tau - 1e-9]
    if not prior:
        return "none"
    recent = "; then ".join(
        f"{canonical(ln.get('label', ''))} for {plural(ln['stop'] - ln['start'])}"
        for ln in prior[-2:])
    if len(prior) > 2:
        earlier = sum(max(0.0, ln["stop"] - ln["start"]) for ln in prior[:-2])
        return (f"{len(prior)} prior regimens over {plural(earlier)} earlier, most "
                f"recently {recent}")
    return (f"{len(prior)} prior regimen{'' if len(prior) == 1 else 's'}: {recent}")


def prior_regimen_list(patient, tau: float) -> str:
    """EVERY prior regimen, named and numbered, for the QUERY patient only.

    The donor summary above names the last two and counts the rest, for a reason that
    holds for donors: uneven card length would itself encode regimen depth, which differs
    systematically between the two evidence arms. That reason does not apply to the
    patient being asked about, and the cost there is real -- a model continuing a
    fifth-line course needs to know which agents are already spent, or it will re-prescribe
    one. The full sequence IS in the ledger as start/switch events, but reconstructing it
    from dated events is work we can do for the model instead of hoping it does it.
    """
    prior = [ln for ln in patient.lines if ln["start"] < tau - 1e-9]
    if not prior:
        return "none"
    return "; ".join(
        f"{i}) {canonical(ln.get('label', ''))} "
        f"({plural(ln['stop'] - ln['start'])}, starting "
        f"{plural(tau - ln['start'])} before this treatment)"
        for i, ln in enumerate(prior, 1))


def prior_course(patient, tau: float) -> str:
    """The documented disease-state path from diagnosis to the landmark, on ONE operator.

    Not a single best response: that collapses order and duration, and a patient who
    responded then progressed is not the same as one who progressed then responded.

    HISTORY IS NOT NARROWED TO MATCH THE LEDGER. H runs diagnosis-to-landmark and includes
    pre-treatment disease behaviour, which is clinically important and which the structured
    ledger cannot encode -- the ledger's events begin at the first systemic treatment. The
    two therefore describe different windows ON PURPOSE, and each says which window it
    describes rather than quietly reporting different numbers for the same-sounding thing.

    THE COUNT IS NAMED FOR WHAT IT MEASURES. `state_path` marks an assessment as expired
    after STATUS_VALIDITY and the state falls to unknown, so progressing -> six unassessed
    months -> progressing is two DOCUMENTED ENTRIES into a progressing state. That is not
    proof of two biological episodes; it may be one progression nobody looked at for half a
    year. Calling it "progression episodes" asserted the biology. It is now called what it
    is, and the earlier hand-rolled run-collapse -- which disagreed with this operator on
    38% of queries -- is gone, so the prose and the ledger no longer answer to two
    different definitions.
    """
    if not [w for w, _s in patient.status if w < tau - 1e-9]:
        return "no disease assessment recorded before this treatment"
    path = CF.state_path(patient, 0.0, tau)
    if not path:
        return "no disease assessment recorded before this treatment"
    # DURATION IS NOT MANUFACTURED. A single assessment documents an instant, not a
    # month, and `max(1.0, b - a)` was inventing time that the rule about undocumented
    # time being unknown explicitly forbids.
    spell = " then ".join(
        f"{CLINICAL.get(st, st.replace('_', ' '))}"
        + (f" for {plural(b - a)}" if b - a >= 0.5 else " (single assessment)")
        for a, b, st in path[-3:])
    # THE STATE AT THE START IS NOT AN ENTRY. Counting path[0] would score a patient
    # already progressing at their first assessment as having entered progression, which
    # is the same error `project()` fixes on the future side.
    entries = [a for i, (a, _b, st) in enumerate(path)
               if st == "progressing" and (i > 0 and path[i - 1][2] != "progressing")]
    tail = (f"; since diagnosis, {len(entries)} documented entr"
            f"{'y' if len(entries) == 1 else 'ies'} into a progressing state, most recent "
            f"{plural(tau - entries[-1])} ago" if entries
            else "; no documented entry into a progressing state since diagnosis")
    return spell + tail


def anatomical_history(mets, tau: float) -> str:
    """When each distant site appeared, which is what "known sites" alone cannot say."""
    if mets is None or not mets.site_detail_available:
        return "site-level detail is not collected in this cohort"
    seen = [(m, s) for m, s in mets.sites if m <= tau + 1e-9]
    if not seen:
        return "no distant site documented before this treatment"
    return "; ".join(
        f"{s} " + ("documented at diagnosis" if m <= 1.0 else f"appeared at month {m:.0f}")
        for m, s in seen)


def prefix_record(patient, tau: float, mets, driver: str, cancer: str = "",
                  landmark_year=None) -> dict:
    """Everything knowable at the landmark, and NOTHING that computes the future.

    The query renderer used to call `course()`, which projects the whole future and then
    discards it. The output was leak-free and the truncation test confirmed it, but
    "prefix-only by construction" was not true -- it was prefix-only by discipline, which
    is the weaker guarantee that has failed repeatedly in this project. This function
    cannot leak because it never reaches for a future value.
    """
    f = GV2.prefix_features(patient, tau)
    extent_rec = DE.extent_at_landmark(mets, tau)
    landmark = next((ln for ln in patient.lines if abs(ln["start"] - tau) < 1e-9), None)
    pres = GV2.recent(patient.presence, tau)
    pres_age = next((tau - w for w, _v in reversed(patient.presence)
                     if tau - GV2.STATUS_VALIDITY <= w <= tau + 1e-9), None)
    dyn = GV2.recent(patient.status, tau)
    dyn_age = next((tau - w for w, _v in reversed(patient.status)
                    if tau - GV2.STATUS_VALIDITY <= w <= tau + 1e-9), None)
    return {
        "cancer": cancer, "landmark_year": landmark_year,
        "regimen": canonical(landmark.get("label", "")) if landmark
        else "unspecified regimen",
        "line": (landmark["number"] if landmark
                 and np.isfinite(landmark.get("number", np.nan)) else None),
        "bucket": landmark["bucket"] if landmark else "unspecified",
        "extent": DE.render_extent(extent_rec), "driver": driver,
        "stage_iv": extent_rec.stage_at_diagnosis,
        "mets_documented": extent_rec.documented,
        "evidence_label": GV2.genie_state(patient, tau, mets).evidence,
        "age": f.get("age_at_diagnosis", float("nan")) + tau / 12.0,
        "sex": "woman" if f.get("sex_female") else "man",
        "histology": ("adenocarcinoma" if f.get("histology_adeno")
                      else "squamous cell" if f.get("histology_squamous") else None),
        "months_since_diagnosis": tau,
        "presence": pres, "presence_months_ago": pres_age,
        "dynamics": dyn, "dynamics_months_ago": dyn_age,
        "sites": sorted(s for _m, s in extent_rec.sites),
        "prior_treatment": prior_treatment(patient, tau),
        "prior_regimen_list": prior_regimen_list(patient, tau),
        "prior_course": prior_course(patient, tau),
        "progression_episodes": sum(
            1 for i, (w, st) in enumerate(patient.status)
            if st == "progressing" and w < tau
            and (i == 0 or patient.status[i - 1][1] != "progressing")),
    }


def course(patient, tau: float, mets, driver: str, cancer: str = "",
           landmark_year=None) -> dict:
    """A donor: the prefix record PLUS the canonical future.

    Built on `prefix_record` rather than duplicating it, so the query prefix and the donor
    prefix cannot drift apart -- they are the same function.
    """
    rec = prefix_record(patient, tau, mets, driver, cancer, landmark_year)
    fut = CF.project(patient, tau, CARD_HORIZON)
    fut_lines = CF.render(fut, canonical)
    same_day = sum(1 for ln in patient.lines
                   if abs(ln["start"] - tau) < GV2.SAME_DAY
                   and abs(ln["start"] - tau) > 1e-9)
    return {**rec,
            "future": fut_lines,
            "future_treatment_course": fut_lines[0],
            "future_disease_course": fut_lines[1],
            "future_new_distant_sites": fut_lines[2],
            "future_terminal": fut_lines[3],
            "same_month_starts": same_day}


def age_band(age: float) -> str:
    """Donor age as a band. Matching already uses a decade window, so banding costs the
    model nothing it was using, and the card is otherwise a longitudinal record combining
    year, histology, molecular status, sites, regimens and outcomes for a real patient."""
    lo = int(age // AGE_BAND) * AGE_BAND
    return f"{lo}-{lo + AGE_BAND - 1}"


def depth_label(n) -> str:
    """REGIMEN starts, not inferred clinical lines. Rendering "line 3" would assert a
    line-of-therapy determination the data cannot support."""
    if not n:
        return "regimen sequence unknown"
    return {1: "regimen 1", 2: "regimen 2"}.get(int(n), "regimen 3+")


def sketch(index: int, year: int | None, c: dict) -> str:
    out = [f"  {index:2d}. {year if year else 'year unknown'}"]
    hist = f" {c['histology']}" if c["histology"] else ""
    out.append(f"      Context: {age_band(c['age'])}-year-old {c['sex']};{hist} "
               f"diagnosed {c['months_since_diagnosis']:.0f} months earlier; "
               f"{c['extent']}")
    now = []
    if c["presence"]:
        now.append(f"{'cancer present' if c['presence'] == 'evidence' else c['presence']}"
                   f" by oncologist assessment {c['presence_months_ago']:.1f} months "
                   f"earlier")
    else:
        now.append("no recent oncologist assessment")
    if c["dynamics"]:
        now.append(f"{CLINICAL.get(c['dynamics'], c['dynamics'])} "
                   f"{c['dynamics_months_ago']:.1f} months earlier")
    else:
        now.append("no recent disease-status assessment")
    if c["sites"]:
        now.append(f"documented sites: {', '.join(c['sites'])}")
    out.append("      At treatment start: " + "; ".join(now) + ".")
    driver = c["driver"]
    out.append(f"      Molecular: {DRIVER.get(driver, driver)}")
    out.append(f"      History: {depth_label(c['line'])}; "
               f"prior treatment: {c['prior_treatment']}")
    out.append(f"      Prior disease course: {c['prior_course']}")
    out.append(f"      Action: {c['regimen']}")
    if c["same_month_starts"]:
        out.append(f"      Also started in the landmark month, order as recorded: "
                   f"{c['same_month_starts']} further regimen(s)")
    # THE FUTURE, from the one canonical projection the scorer also reads.
    for row in c["future"]:
        out.append("      " + row)
    return "\n".join(out)


def query_prefix(patient, tau: float, mets, driver: str, diagnosis_year, landmark_year):
    """The QUERY patient's prefix. Deliberately NOT the donor renderer.

    Reusing the donor renderer put two things in the query prefix that must never be
    there. It bands the age -- correct for a donor, wrong for the subject of the question.
    And it carries the "also started in the landmark month" line, which counts a regimen
    beginning two weeks LATER: exact same-date records are merged upstream, so anything
    that line can still report is future information.

    There is no future section at all here. The query's future is the answer.
    """
    c = prefix_record(patient, tau, mets, driver)
    hist = f" {c['histology']};" if c["histology"] else ""
    now = []
    if c["presence"]:
        now.append(f"{'cancer present' if c['presence'] == 'evidence' else c['presence']} "
                   f"by oncologist assessment {c['presence_months_ago']:.1f} months "
                   f"earlier")
    else:
        now.append("no recent oncologist assessment")
    if c["dynamics"]:
        now.append(f"{CLINICAL.get(c['dynamics'], c['dynamics'])} "
                   f"{c['dynamics_months_ago']:.1f} months earlier")
    else:
        now.append("no recent disease-status assessment")
    if c["sites"]:
        now.append(f"documented sites: {', '.join(c['sites'])}")
    return "\n".join([
        # EXACT age, not a band: this is the patient being asked about, not evidence
        # about someone else.
        f"Context: {c['age']:.0f}-year-old {c['sex']};{hist} diagnosed "
        f"{c['months_since_diagnosis']:.0f} months earlier, in {diagnosis_year}; "
        f"{c['extent']}",
        f"At treatment start ({landmark_year}): " + "; ".join(now) + ".",
        # THE CANONICAL DERIVED STATE, stated plainly. It is what retrieval matched on,
        # and it is not always readable from the assessments above: a state of
        # "progressing" can come from a newly documented distant site with no response
        # assessment at all, and the summary would then have said only "no recent
        # disease-status assessment".
        f"Overall disease state at this treatment: "
        f"{GV2.genie_state(patient, tau, mets).state.replace('_', ' ')}.",
        # ASSAY TIMING, stated ONCE for the whole panel. It used to be fused into a
        # biomarker value string ("reported before this treatment started; BRAF G466V");
        # the seed now carries analytes only, so the provenance says itself here, in
        # prose, where a sentence belongs.
        f"Molecular: {DRIVER.get(driver, driver)}"
        + ("" if driver == "unknown"
           else " (panel resulted before this treatment started)"),
        # EVERY prior regimen for the query patient, not the two-line donor summary: the
        # model is about to prescribe, and needs to know what is already spent.
        f"History: {depth_label(c['line'])}; prior treatment: "
        f"{c['prior_regimen_list']}",
        f"Prior disease course: {c['prior_course']}",
        f"Action: starting {c['regimen']}",
    ])


def render_card(donors: list[tuple[str, int | None, dict]]) -> str:
    body = [HEADER.format(n=len(donors)), ""]
    body += [sketch(i + 1, yr, c) for i, (_pid, yr, c) in enumerate(donors)]
    return "\n".join(body)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(line) for line in (UNIVERSE / "queries.jsonl").open()]
    by_cancer: dict[str, list] = {}
    for q in queries:
        by_cancer.setdefault(q["cancer"], []).append(q)

    report = {}
    for cancer, rows in by_cancer.items():
        folder = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE",
                  "CRC": "CRC_GENIE"}[cancer]
        patients = {p.pid: p for p in GV2.load(GENIE_ROOT / folder)}
        mets = DE.load_metastasis(GENIE_ROOT / folder)
        q = rows[0]
        cards = {}
        for arm in ("relevant_donors", "generic_donors"):
            records = {d["patient"]: d for d in q[arm]}
            donors = []
            for pid in stable_order(list(records)):
                d = records[pid]
                if d.get("year") is None or d.get("tau_months") is None:
                    raise ValueError(
                        f"donor {pid} has no persisted landmark. The card must render the "
                        "exact treatment, era and future that retrieval matched on; "
                        "reconstructing a landmark here would show different evidence.")
                p = patients.get(pid)
                if p is None:
                    raise ValueError(f"donor {pid} is not in the loaded cohort")
                donors.append((pid, d["year"],
                               course(p, float(d["tau_months"]), mets.get(pid),
                                      d["molecular_driver"], cancer, d["year"])))
            cards[arm] = render_card(donors)
        leak = (q["patient"] in cards["relevant_donors"]
                or q["patient"] in cards["generic_donors"])
        same_shape = (cards["relevant_donors"].split("\n")[0]
                      == cards["generic_donors"].split("\n")[0])
        report[cancer] = {
            "example_query": q["query_id"],
            "correct_chars": len(cards["relevant_donors"]),
            "wrong_chars": len(cards["generic_donors"]),
            "correct_sha": hashlib.sha256(cards["relevant_donors"].encode()).hexdigest()[:12],
            "wrong_sha": hashlib.sha256(cards["generic_donors"].encode()).hexdigest()[:12],
            "identical_header": same_shape,
            "query_id_in_card": leak,
            # Phrases that would betray the SELECTION RULE. Bare "distant" is no longer
            # among them: it now carries its clinical sense in "no distant metastatic
            # disease recorded", and banning a word the cards must legitimately use would
            # trade a real leak check for a permanent false alarm.
            "banned_words_present": [w for w in
                                     ("distance", "similar", "nearest", "most distant",
                                      "distant in", "patients-like-me", "correct card",
                                      "wrong card", "control")
                                     if w in cards["relevant_donors"].lower()
                                     or w in cards["generic_donors"].lower()],
        }
        (OUT / f"example_{cancer}_patients_like_me.txt").write_text(
            cards["relevant_donors"])
        (OUT / f"example_{cancer}_generic_cohort.txt").write_text(
            cards["generic_donors"])
        r = report[cancer]
        print(f"{cancer}: correct {r['correct_chars']:5d} chars "
              f"({r['correct_sha']}), wrong {r['wrong_chars']:5d} chars "
              f"({r['wrong_sha']}); identical header {r['identical_header']}; "
              f"query id in card {r['query_id_in_card']}; "
              f"banned words {r['banned_words_present'] or 'none'}")

    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nexample cards written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
