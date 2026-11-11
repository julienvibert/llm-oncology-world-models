"""Characteristics of the real patients analyzed: GENIE-BPC forecast patients and PRIME (aggregates only).

Descriptive table, no test. GENIE-BPC: the 450 query patients of the forecasting experiment, by cancer, from the
saved query universe and the public BPC release files. PRIME: patients with known KRAS exon 2 status and the map
subset (baseline lesion assessment by day 14), by arm. GENIE-BPC counts of 1 to 4 are shown as <5. No identifier
is written or printed.

Output: manuscript/supplementary_tables/supplementary_table_3_patients.md and
outputs/exploratory_patient_characteristics/patient_characteristics.csv (gitignored folder).
Run: uv run python scripts/explore/patient_characteristics_table.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GENIE = Path(os.environ.get("GENIE_ROOT", Path.home() / "GenieBPC"))
FOLDER = {"NSCLC": "NSCLC_Genie", "CRC": "CRC_GENIE", "BrCa": "BRCA_GENIE"}
OUT = ROOT / "outputs" / "exploratory_patient_characteristics"


def cnt(n: int, total: int, small: bool) -> str:
    """Count with percentage of the known total; for registry data, counts of 1 to 4 and complements of 1 to 4
    (which would reveal a small count by subtraction) are masked."""
    if small and 0 < n < 5:
        return "<5"
    if small and 0 < total - n < 5:
        return f">{total - 5}"
    return f"{n} ({100 * n / total:.0f}%)" if total else "0"


def miss(s: pd.Series) -> int:
    return int(s.isna().sum() + s.astype(str).str.strip().isin(["", "nan", "Unknown", "unknown", "None"]).sum() - s.isna().sum())


def med(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.median():.0f} ({x.quantile(0.25):.0f}-{x.quantile(0.75):.0f})" if len(x) else "-"


def genie() -> list[dict]:
    q = pd.DataFrame([json.loads(line) for line in open(ROOT / "outputs/exploratory_query_universe/queries.jsonl")])
    rows = []
    for cancer, g in q.groupby("cancer", sort=False):
        key = next(k for k in FOLDER if k.lower() in str(cancer).lower() or str(cancer).lower() in k.lower())
        pt = pd.read_csv(GENIE / FOLDER[key] / "patient_level_dataset.csv", low_memory=False).set_index("record_id")
        ca = pd.read_csv(GENIE / FOLDER[key] / "cancer_level_dataset_index.csv", low_memory=False)
        ca = ca.sort_values("ca_seq").groupby("record_id").first()
        ids = g["patient"]
        n = len(g)
        assert n == 150 and ids.is_unique, "forecast population: 150 distinct patients per cancer"
        sex = pt.naaccr_sex_code.reindex(ids).astype(str)
        stage = ca.stage_dx_iv.reindex(ids).astype(str)
        age = ca.age_dx.reindex(ids)
        r = {"Cohort": f"GENIE-BPC {key}", "n": str(n),
             "Missing (age / sex / stage)": " / ".join("<5" if 0 < k < 5 else str(k) for k in (int(age.isna().sum()), miss(pt.naaccr_sex_code.reindex(ids)), miss(ca.stage_dx_iv.reindex(ids)))),
             "Age at diagnosis, median (IQR), years": med(ca.age_dx.reindex(ids)),
             "Female": cnt(int(sex.str.contains("Female").sum()), n, True),
             "Stage IV at diagnosis": cnt(int(stage.eq("Stage IV").sum()), n, True),
             "Year of diagnosis, range": f"{int(g.diagnosis_year.min())}-{int(g.diagnosis_year.max())}",
             "Year of forecast treatment start, range": f"{int(g.landmark_year.min())}-{int(g.landmark_year.max())}",
             "Regimen 1 / 2 / 3 or later": " / ".join(cnt(int((g.depth.astype(str) == d).sum()), n, True) for d in ("regimen 1", "regimen 2", "regimen 3+"))}
        rows.append(r)
    return rows


def prime() -> list[dict]:
    s = pd.read_csv(ROOT / "outputs/exploratory_pds_prime_trial_geometry/subject_table.csv", low_memory=False)
    v = pd.read_csv(ROOT / "outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz", usecols=["SUBJID", "day"])
    s = s[s.kras_exon2.isin(["Wild-type", "Mutant"])]
    mapped = set(v.SUBJID[v.day.le(14)])
    known = s.groupby("arm").size().to_dict()
    n8 = pd.read_csv(ROOT / "outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv")
    assert len(s) == 866 and s.SUBJID.is_unique, "known KRAS exon 2 status: 866 patients"
    m = s[s.SUBJID.isin(mapped)]
    assert len(m) == 439 and m.groupby(["kras_exon2"]).size().to_dict() == {"Mutant": 179, "Wild-type": 260}, "map patients as in Fig. 6"
    _ = (known, n8)
    rows = []
    for label, sub in (("PRIME, known KRAS exon 2 status", s), ("PRIME, map patients", s[s.SUBJID.isin(mapped)])):
        for arm, g in sub.groupby("arm", sort=False):
            n = len(g)
            rows.append({"Cohort": f"{label}: {arm}", "n": str(n),
                         "Missing (age / sex / ECOG / liver)": f"{miss(g.age)} / {miss(g.SEX)} / {miss(pd.to_numeric(g.ecog, errors='coerce'))} / {miss(g.liver_met)}",
                         "Age, median (IQR), years": med(g.age),
                         "Female": cnt(int(g.SEX.astype(str).str.upper().str.startswith("F").sum()), n, False),
                         "KRAS exon 2 wild-type": cnt(int(g.kras_exon2.eq("Wild-type").sum()), n, False),
                         "ECOG 0, of known": cnt(int(pd.to_numeric(g.ecog, errors="coerce").eq(0).sum()), int(pd.to_numeric(g.ecog, errors="coerce").notna().sum()), False),
                         "Liver metastases": cnt(int(g.liver_met.astype(bool).sum()), n, False)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g, p = genie(), prime()
    pd.DataFrame(g + p).to_csv(OUT / "patient_characteristics.csv", index=False)

    def table(rows: list[dict]) -> str:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(r.get(c, "-") for c in cols) + " |" for r in rows]
        return "\n".join(out)
    md = ("### Supplementary Table 3 | Characteristics of the real patients analyzed\n\n"
          "GENIE-BPC: the 450 patients of the forecasting experiment (Figs 4 and 5), by cancer; counts of 1 to 4, and counts within 4 of the total, are masked. "
          "PRIME: patients with known KRAS exon 2 status and the map subset (baseline lesion assessment by day 14; Fig. 6), by arm. "
          "Descriptive; no test.\n\n" + table(g) + "\n\n" + table(p) + "\n")
    (ROOT / "manuscript/supplementary_tables/supplementary_table_3_patients.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
