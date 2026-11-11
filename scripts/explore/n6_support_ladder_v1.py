#!/usr/bin/env python3
"""N6 GENIE support ladder: how many comparable real patients remain as a decision is described
more precisely (completion plan 28 Oct 2026, N6; counts only).

Specification: manuscript/review/2026-10-28_completion_plan.md, "N6 specification". No survival, no incidence, no outcome of any kind.

Decision: start of a recorded systemic regimen after metastatic disease is documented, patient
alive, at least one assessment between metastatic onset and the decision (the test-A rule,
`test_a_feasibility_v1.eligible_decisions`). Only information dated on or before the decision.
Primary unit: one decision per patient drawn at random (seed 20261028); comparators are the other
patients' drawn decisions. Sensitivity: all decisions, counting DISTINCT other patients with at
least one matching decision, every record of the index patient excluded.

Steps (cumulative, fixed order):
  1 cancer  2 stage at diagnosis  3 driver  4 regimen depth (order among regimens started
    after metastatic onset)  5 treatment class
  6 latest status  7 previous regimen (best status, duration band)  8 age band and sex
Driver definitions (fixed here, before counting; one category by precedence):
  NSCLC  EGFR (L858R, exon 19 in-frame deletion, exon 20 insertion, G719X, L861Q, S768I) >
         ALK (any ALK fusion) > KRAS (G12, G13, Q61) > tested, none of these
  CRC    MMR-deficient or MSI-high (pathology) > BRAF V600E > KRAS (G12, G13, A59, Q61, K117,
         A146) > tested, none of these
  BRCA   index BCA_SUBTYPE: HR+HER2-, HER2+ (HR+ or HR-), triple-negative
A result counts only if its report (sequencing or pathology date) is on or before the decision;
a sequenced sample with no date is "not documented as known at the decision"; no dated result
before the decision is "no documented test result by the decision". Sequencing is dated by the panel REPORT date
(`cancer_panel_test_level_dataset.csv`, dx_cpt_rep_days); checked 28 Oct: the sequencing
timeline START_DATE equals it within one day for 99.5% (NSCLC), 97.2% (breast) and 99.4% (CRC)
of samples, whereas the specimen date precedes it by a median of 43-154 days. The timeline date
is used only when no report date exists (count in the manifest).
Regimens are read from the raw treatment file by REGIMEN MEMBERSHIP (index cancer, regimen
number; regimens starting within GV2.SAME_DAY are one regimen, as in the registry reader). The
regimen known at the decision is the members of the regimen starting at the decision whose own
start is on or before the decision; agents added later are never added retrospectively. Stops are kept missing when absent (the registry reader replaces a missing stop by the
start); a regimen's stop is unknown if ANY member's stop is missing (the latest known stop does
not establish the end). Previous-regimen end = its stop, or the next regimen start (the decision)
when the stop is unknown, whichever is earlier. Treatment class at the decision: the known
members classified with the reviewed drug table (`four_arm_readers_v2.classify`): union of
modalities, adjuncts ignored, any unrecognised agent adds the explicit token "unclassified".
Status: curated statuses of the registry reader (mixed merged into progressing), plus "no active
disease" from the raw CANCER_STATUS field ("no evidence of cancer") on dates with no curated
status; "not assessed" when nothing is documented.
Missing values, defined per feature (`missing`): stage "unknown"; driver "no documented test
result by the decision", "not documented as known at the decision" or "unknown" (positives and "tested, none of these" are
known); depth never missing; treatment class containing "unclassified"; latest status "not
assessed"; previous regimen whose best status is "not assessed" ("first regimen" is known); age
or sex "unknown". Primary: a missing value matches only the same missing value; sensitivity: a
key with any missing value matches nothing. The share of matched pairs relying on a shared
missing value is reported per step.

GENIE-BPC under DUA: aggregate counts only (outputs/exploratory_n6_support_ladder/); the console
prints counts only, no identifier. Months: registry clock, days / 30.4 (the reader's), used only
for duration and age bands.
Run: uv run python scripts/explore/n6_support_ladder_v1.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import disease_extent_v1 as DE  # noqa: E402
import four_arm_readers_v2 as R  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402
import test_a_feasibility_v1 as TA  # noqa: E402

SEED = 20261028
DATING: dict[str, dict] = {}
FALLBACK_UNKNOWN = "--fallback-unknown" in sys.argv
OUT = ROOT / "outputs" / "exploratory_n6_support_ladder" / ("sensitivity_fallback_unknown" if FALLBACK_UNKNOWN else "")
UNKNOWN = "unknown"
NOT_KNOWN = "not documented as known at the decision"
NO_RESULT = "no documented test result by the decision"
STEPS = ("cancer", "stage", "driver", "depth", "treatment_class", "latest_status",
         "previous_regimen", "age_sex")
BEST = ("no active disease", "responding", "stable", "progressing")      # best first
EGFR_POINT = {"p.L858R", "p.L861Q", "p.S768I"}
KRAS_NSCLC = ("p.G12", "p.G13", "p.Q61")
KRAS_CRC = ("p.G12", "p.G13", "p.A59", "p.Q61", "p.K117", "p.A146")


def sha256(path: Path) -> str:
    """SHA-256 of a file (code and input provenance for the manifest)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(genie: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(genie / name, sep="\t", comment="#", low_memory=False)


def months(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / GV2.DAYS


def index_diagnosis(genie: Path) -> pd.DataFrame:
    dx = read(genie, "data_timeline_cancer_diagnosis.txt")
    if "INDEX_CANCER" in dx:
        dx = dx[dx["INDEX_CANCER"].astype(str).str.lower().isin({"yes", "true", "1"})]
    return dx.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")


def stage_of(raw) -> str:
    s = str(raw)
    return {"Stage 0": "0", "Stage I": "I", "Stage II": "II", "Stage III": "III", "Stage IV": "IV",
            "Stage I-III NOS": "I-III NOS"}.get(s, UNKNOWN)


def egfr_hit(r) -> bool:
    hg, vc, ex = str(r["HGVSp_Short"]), str(r["Variant_Classification"]), str(r["Exon_Number"])
    exon = ex.split("/")[0]
    return (hg in EGFR_POINT or hg.startswith("p.G719") or (exon == "19" and vc == "In_Frame_Del")
            or (exon == "20" and vc == "In_Frame_Ins"))


def sample_findings(cancer: str, genie: Path) -> dict[str, dict[str, set]]:
    """Per sample: the driver labels it carries (empty set = sequenced, none of these)."""
    mut = read(genie, "data_mutations_extended.txt")
    out: dict[str, set] = defaultdict(set)
    for _, r in mut.iterrows():
        g, hg, s = r["Hugo_Symbol"], str(r["HGVSp_Short"]), str(r["Tumor_Sample_Barcode"])
        if cancer == "NSCLC":
            if g == "EGFR" and egfr_hit(r):
                out[s].add("EGFR")
            elif g == "KRAS" and hg.startswith(KRAS_NSCLC):
                out[s].add("KRAS")
        elif cancer == "CRC":
            if g == "BRAF" and hg == "p.V600E":
                out[s].add("BRAF")
            elif g == "KRAS" and hg.startswith(KRAS_CRC):
                out[s].add("KRAS")
    if cancer == "NSCLC" and (genie / "data_fusions.txt").exists():
        fus = read(genie, "data_fusions.txt")
        alk = fus["Hugo_Symbol"].astype(str).eq("ALK") | fus["Fusion"].astype(str).str.contains(r"\bALK\b")
        for s in fus.loc[alk, "Tumor_Sample_Barcode"].astype(str):
            out[s].add("ALK")
    return out


def driver_tables(cancer: str, genie: Path):
    """Per patient: dated sequencing reports [(month, labels)], undated sequenced samples' labels,
    and (CRC) dated MMR/MSI results [(month, deficient)]."""
    seq = read(genie, "data_timeline_sequencing.txt")
    cpt = pd.read_csv(genie / "cancer_panel_test_level_dataset.csv", low_memory=False)
    report = dict(zip(cpt["cpt_genie_sample_id"].astype(str),
                      pd.to_numeric(cpt["dx_cpt_rep_days"], errors="coerce") / GV2.DAYS, strict=True))
    rep_m = seq["SAMPLE_ID"].astype(str).map(report)
    DATING[cancer] = {"samples": int(len(seq)), "report_date": int(rep_m.notna().sum()),
                      "timeline_fallback": int((rep_m.isna() & months(seq["START_DATE"]).notna()).sum())}
    # sensitivity (--fallback-unknown): a sample without a panel report date is not documented as known
    seq["_m"] = rep_m if FALLBACK_UNKNOWN else rep_m.where(rep_m.notna(), months(seq["START_DATE"]))
    found = sample_findings(cancer, genie)
    dated, undated = defaultdict(list), defaultdict(set)
    for _, r in seq.iterrows():
        pid, s = str(r["PATIENT_ID"]), str(r["SAMPLE_ID"])
        if np.isfinite(r["_m"]):
            dated[pid].append((float(r["_m"]), found.get(s, set())))
        else:
            undated[pid] |= found.get(s, set()) or {"none"}
    mmr = defaultdict(list)
    if cancer == "CRC":
        path = read(genie, "data_timeline_pathology.txt")
        path["_m"] = months(path["START_DATE"])
        for _, r in path.iterrows():
            res = [str(r.get(c)) for c in ("MMR_RESULT", "MMR_RESULT_2", "MMR_RESULT_3")]
            msi = str(r.get("MSI_HIGH_ANY"))
            deficient = msi == "Yes" or any(x.startswith("Loss") for x in res)
            proficient = msi == "No" or any(x.startswith("No loss") for x in res)
            if (deficient or proficient) and np.isfinite(r["_m"]):
                mmr[str(r["PATIENT_ID"])].append((float(r["_m"]), deficient))
    return dated, undated, mmr


def driver_at(cancer: str, pid: str, tau: float, dx: pd.DataFrame, tabs) -> str:
    if cancer == "BRCA":
        s = str(dx["BCA_SUBTYPE"].get(pid)) if pid in dx.index else "nan"
        return {"HR+, HER2-": "HR+HER2-", "HR+, HER2+": "HER2+", "HR-, HER2+": "HER2+",
                "Triple Negative": "triple-negative"}.get(s, UNKNOWN)
    dated, undated, mmr = tabs
    known = [lab for t, lab in dated.get(pid, []) if t <= tau + GV2.SAME_DAY]
    labels = set().union(*known) if known else set()
    if cancer == "CRC" and any(d for t, d in mmr.get(pid, []) if t <= tau + GV2.SAME_DAY):
        return "MMR-deficient or MSI-high"
    order = ("EGFR", "ALK", "KRAS") if cancer == "NSCLC" else ("BRAF", "KRAS")
    for g in order:
        if g in labels:
            return g
    if known:
        return "tested, none of these"
    if undated.get(pid):
        return NOT_KNOWN
    return NO_RESULT


def ned_records(genie: Path) -> dict[str, list[float]]:
    out = defaultdict(list)
    for name in ("data_timeline_imaging.txt", "data_timeline_medonc.txt"):
        f = read(genie, name)
        f["_m"] = months(f["START_DATE"])
        no = f["CANCER_STATUS"].astype(str).str.startswith("No,") & f["_m"].notna() & (f["_m"] >= 0)
        for pid, t in zip(f.loc[no, "PATIENT_ID"].astype(str), f.loc[no, "_m"], strict=True):
            out[pid].append(float(t))
    return out


def statuses(p: GV2.Patient, ned: list[float]) -> list[tuple[float, str]]:
    cur = [(float(w), "progressing" if s == "mixed" else s) for w, s in p.status]
    have = {round(w, 3) for w, _ in cur}
    extra = [(t, "no active disease") for t in ned if round(t, 3) not in have]
    return sorted(cur + extra)


def band(x: float, edges: tuple, labels: tuple) -> str:
    if not np.isfinite(x):
        return UNKNOWN
    for e, lab in zip(edges, labels, strict=False):
        if x < e:
            return lab
    return labels[-1]


def raw_treatment(genie: Path) -> dict[str, list[tuple[float, float, str, int]]]:
    """Per patient, index cancer only: (start, stop or NaN, agent, regimen number) in registry months."""
    keep = DE.index_cancer(genie)
    tx = read(genie, "data_timeline_treatment.txt")
    tx["_start"], tx["_stop"] = months(tx["START_DATE"]), months(tx["STOP_DATE"])
    tx = tx[tx["_start"].notna() & (tx["_start"] >= 0)]
    tx = tx[[keep.get(str(a)) == b for a, b in zip(tx["PATIENT_ID"], tx["CA_SEQ"], strict=True)]]
    out = defaultdict(list)
    for pid, a, b, ag, rn in zip(tx["PATIENT_ID"].astype(str), tx["_start"], tx["_stop"], tx["AGENT"],
                                 tx["REGIMEN_NUMBER"], strict=True):
        out[pid].append((float(a), float(b), ag, int(rn)))
    return out


def regimen_members(tx: list, t: float) -> list:
    """All records of the regimen(s) whose regimen start (earliest member start) is within SAME_DAY
    of t: regimen membership by number, same-day regimens merged as in the registry reader."""
    first: dict[int, float] = {}
    for a, _, _, rn in tx:
        first[rn] = min(first.get(rn, np.inf), a)
    numbers = {rn for rn, a in first.items() if abs(a - t) < GV2.SAME_DAY}
    return [r for r in tx if r[3] in numbers]


def known_at(members: list, tau: float) -> list:
    """Members whose own start is on or before the decision (later agents are never added)."""
    return [r for r in members if r[0] <= tau + GV2.SAME_DAY]


def regimen_stop(members: list) -> tuple[float, str]:
    """(stop, kind): stop unknown (NaN) if any member lacks a stop; kind all_known / mixed / all_missing."""
    stops = [b for _, b, _, _ in members]
    n_missing = sum(not np.isfinite(b) for b in stops)
    if n_missing == 0:
        return max(stops), "all_known"
    return np.nan, "all_missing" if n_missing == len(stops) else "mixed"


def classify_agents(agents: list) -> str:
    classes = [R.classify(ag) for ag in agents]
    if not classes:
        return UNKNOWN
    mods = sorted({m for c in classes if c.kind == "modality" for m in c.modalities})
    if any(c.kind == R.UNRECOGNISED for c in classes):
        mods.append("unclassified")
    return "+".join(mods) or "adjunct only"


def treatment_class(tx: list, tau: float) -> str:
    """Class of the regimen known at the decision (membership, agents started by the decision)."""
    return classify_agents([ag for _, _, ag, _ in known_at(regimen_members(tx, tau), tau)])


def treatment_class_same_day(tx: list, tau: float) -> str:
    """The previous rule (agents starting the same day as the decision), kept for the change count."""
    return classify_agents([ag for a, _, ag, _ in tx if abs(a - tau) < GV2.SAME_DAY])


def missing(step: int, value: str) -> bool:
    """Is the value of feature `step` (0-based, order of STEPS) missing?"""
    v = str(value)
    return {1: v == UNKNOWN,
            2: v in (NO_RESULT, NOT_KNOWN, UNKNOWN),
            4: "unclassified" in v.split("+") or v == UNKNOWN,
            5: v == "not assessed",
            6: v.startswith("not assessed"),
            7: UNKNOWN in v.split()}.get(step, False)


def key_missing(key: tuple) -> bool:
    return any(missing(i, v) for i, v in enumerate(key))


def features(cancer: str, p: GV2.Patient, d: dict, dx: pd.DataFrame, tabs, ned, tx: list,
             meta: Counter) -> tuple:
    tau = d["tau"]
    st = statuses(p, ned.get(p.pid, []))
    seen = [s for w, s in st if w <= tau + GV2.SAME_DAY]
    # depth = order among regimens started after metastatic onset (the reader's own number counts
    # every regimen since diagnosis, adjuvant included); same-day starts are one regimen
    onset = TA.metastatic_onset(p)
    starts = sorted({round(x["start"], 4) for x in p.lines if x["start"] >= onset - GV2.SAME_DAY
                     and x["start"] <= tau + GV2.SAME_DAY})
    number = len(starts)
    depth = "4+" if number >= 4 else str(number)
    earlier = [x for x in p.lines if x["start"] < tau - GV2.SAME_DAY]
    if earlier:
        prev = max(earlier, key=lambda x: x["start"])
        raw_stop, kind = regimen_stop(known_at(regimen_members(tx, prev["start"]), tau))
        meta["previous_regimens"] += 1
        meta[f"previous_regimen_stop_{kind}"] += 1
        end = min(raw_stop if np.isfinite(raw_stop) else np.inf, tau)
        inside = [s for w, s in st if prev["start"] <= w <= end]
        best = next((b for b in BEST if b in inside), "not assessed")
        dur = band(end - prev["start"], (3, 6, 12), ("<3", "3-6", "6-12", ">=12"))
        previous = f"{best} | {dur}"
    else:
        previous = "first regimen"
    age = float(p.context.get("age", np.nan)) + tau / 12.0
    sex = {"Male": "M", "Female": "F"}.get(str(p.context.get("sex")), UNKNOWN)
    return (cancer,
            stage_of(dx["STAGE_DX"].get(p.pid)) if p.pid in dx.index else UNKNOWN,
            driver_at(cancer, p.pid, tau, dx, tabs),
            depth,
            treatment_class(tx, tau),
            seen[-1] if seen else "not assessed",
            previous,
            band(age, (50, 65, 75), ("<50", "50-64", "65-74", ">=75")) + " " + sex)


def ladder(rows: list[tuple[str, tuple]], all_rows: list[tuple[str, tuple]]) -> list[dict]:
    """rows: (patient, features) of the index decisions; all_rows: the comparator pool."""
    out = []
    for k in range(1, len(STEPS) + 1):
        pool: dict[tuple, set] = defaultdict(set)
        for pid, f in all_rows:
            pool[f[:k]].add(pid)
        for variant in ("primary", "missing_matches_nothing"):
            counts, shared, pairs = [], 0, 0
            for pid, f in rows:
                key = f[:k]
                if variant == "missing_matches_nothing" and key_missing(key):
                    n = 0
                else:
                    n = len(pool[key] - {pid})
                counts.append(n)
                pairs += n
                shared += n if key_missing(key) else 0
            c = np.asarray(counts)
            out.append({"step": k, "step_name": STEPS[k - 1], "variant": variant, "n_index": len(c),
                        "median": float(np.median(c)), "q1": float(np.percentile(c, 25)),
                        "q3": float(np.percentile(c, 75)), "share_zero": round(float((c == 0).mean()), 4),
                        "share_lt5": round(float((c < 5).mean()), 4),
                        "share_pairs_shared_missing": round(shared / pairs, 4) if pairs else None})
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    results, counts, all_changed = [], {}, {}
    for cancer in ("NSCLC", "BRCA", "CRC"):
        genie = GV2.GENIE_ROOT / GV2.COHORTS[cancer]
        pts = GV2.load(genie)
        dx = index_diagnosis(genie)
        tabs = driver_tables(cancer, genie) if cancer != "BRCA" else None
        ned = ned_records(genie)
        tx = raw_treatment(genie)
        meta: Counter = Counter()
        every, drawn = [], []
        for p in sorted(pts, key=lambda q: q.pid):
            ds = TA.eligible_decisions(p)
            if not ds:
                continue
            fs = [features(cancer, p, d, dx, tabs, ned, tx.get(p.pid, []), Counter()) for d in ds]
            every += [(p.pid, f) for f in fs]
            pick = int(rng.integers(len(fs)))
            changed = [treatment_class_same_day(tx.get(p.pid, []), d["tau"]) != f[4] for d, f in zip(ds, fs, strict=True)]
            meta["class_changed_vs_same_day_rule_drawn"] += changed[pick]
            all_changed[cancer] = all_changed.get(cancer, 0) + sum(changed)
            features(cancer, p, ds[pick], dx, tabs, ned, tx.get(p.pid, []), meta)   # stop counts, drawn only
            drawn.append((p.pid, fs[pick]))
        multi = 0
        if tabs is not None:
            dated = tabs[0]
            for pid, _ in drawn:
                labs = set().union(*[lab for _, lab in dated.get(pid, [])]) if dated.get(pid) else set()
                multi += len(labs) > 1
        counts[cancer] = {"patients": len(pts), "patients_with_decision": len(drawn), **dict(meta),
                          "missing_by_step": {s: int(sum(missing(i, f[i]) for _, f in drawn))
                                              for i, s in enumerate(STEPS)},
                          "decisions": len(every), "class_changed_vs_same_day_rule_all_decisions": all_changed.get(cancer, 0),
                          "patients_with_more_than_one_driver": multi,
                          **{f"category_counts_step_{s}": dict(Counter(f[i] for _, f in drawn).most_common())
                             for i, s in enumerate(STEPS) if s not in ("cancer", "age_sex")}}
        for r in ladder(drawn, drawn):
            results.append({"cancer": cancer, "unit": "one_decision_per_patient", **r})
        for r in ladder(every, every):
            results.append({"cancer": cancer, "unit": "all_decisions_distinct_patients", **r})
        print(f"[n6] {cancer}: {len(drawn)} patients with an eligible decision, {len(every)} decisions")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "n6_support_ladder_v1.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    (OUT / "manifest.json").write_text(json.dumps({
        "script": "scripts/explore/n6_support_ladder_v1.py", "seed": SEED, "steps": STEPS,
        "month_days": GV2.DAYS, "counts": counts, "driver_dating": DATING,
        "sequencing_date_check": "timeline START_DATE = panel report date within 1 day for 99.5% / 97.2% / 99.4% of "
                                 "NSCLC / breast / CRC samples; specimen date earlier by a median 43-154 days; "
                                 "drivers dated by the report date",
        "provenance": {"code_sha256": sha256(Path(__file__)), "inputs_sha256": {
            f"{c}/{n}": sha256(GV2.GENIE_ROOT / GV2.COHORTS[c] / n) for c in GV2.COHORTS
            for n in ("data_timeline_cancer_diagnosis.txt", "data_timeline_sequencing.txt",
                      "data_mutations_extended.txt", "data_timeline_imaging.txt", "data_timeline_medonc.txt",
                      "data_timeline_treatment.txt", "cancer_panel_test_level_dataset.csv",
                      "data_fusions.txt", "data_timeline_pathology.txt") if (GV2.GENIE_ROOT / GV2.COHORTS[c] / n).exists()}},
        "aggregate_only": True}, indent=1) + "\n")
    print(f"[n6] {len(results)} aggregate rows to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
