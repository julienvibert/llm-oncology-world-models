"""Assemble the Source Data candidate for the main and Extended Data figures from existing tables (local only).

No recomputation: for every panel of the main and Extended Data figures in scripts/figures/layouts.json, reads the
sources recorded in the panel's _meta.json and sorts each into one of five classes:

  included        synthetic aggregate table, copied into the package
  deposit         synthetic run folder or patient-level synthetic table, provided in the data deposit
  pending         aggregate of GENIE-BPC or PRIME data; listed, not copied, until checked for release
                  (cells counting 1 to 4 patients are counted automatically)
  not included    patient-level registry or trial data, or a file holding registry query identifiers; the panel
                  needs an aggregate table derived for release
  no data         schematic, manuscript text or code

Output: <out>/source_data_<commit>/Figure_<n>/... plus SOURCE_DATA_INDEX.md, and the same index at
manuscript/release/source_data_index.md. Nothing is uploaded.
Run: uv run python scripts/release/build_source_data.py [--out outputs/release_candidates]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "release"))
import mask_small_cells as MS  # noqa: E402

PANELS = ROOT / "outputs" / "figures" / "panels"
S = MS.Spec
# Explicit description of every registry or trial table released as Source Data (units and denominators declared,
# never inferred). A real-data table missing here makes the build fail.
SPECS = {
    "f4c_survival_12m_aggregates.csv": S(counts=("n_patients",), not_shares=("mean_pred_12m_pct", "observed_ipcw_12m_pct")),
    "f4b_early_deaths_aggregates.csv": S(counts=("n_patients", "events", "denominator"), denoms=("denominator",),
                                         shares={"deaths_within_6m_pct": ("pct", "denominator")},
                                         not_shares=("lo_pct", "hi_pct")),
    "f4h_joint_score_aggregates.csv": S(counts=("n_patients", "deaths_by_horizon"), denoms=("n_patients",)),
    "ed07b_gap_by_horizon_aggregates.csv": S(counts=("n",)),
    "f4g_calibration_aggregates.csv": S(counts=("n_in_bin",), shares={"observed_pct": ("pct", "n_in_bin")},
                                        not_shares=("predicted_pct",)),
    "f4e_fan_coverage_aggregates.csv": S(counts=("n_patients_died",),
                                         shares={"share_within_p10_p90_pct": ("pct", "n_patients_died")},
                                         not_shares=("lo_pct", "hi_pct")),
    "ed07e_donor_agreement_aggregates.csv": S(counts=("n_patients",)),
    "f4f_c_index_aggregates.csv": S(counts=("n_patients",)),
    "f4d_bias_deaths_aggregates.csv": S(counts=("n_patients_died",)),
    "f5c_course_disagreement_aggregates.csv": S(counts=("n_patients",), not_shares=("unresolved_fraction_mean",)),
    "f5d_course_contrasts_aggregates.csv": S(counts=("n_patients",)),
    "f4i_treatment_aggregates.csv": S(counts=("n_patients",)),
    "ed08c_sensitivity_aggregates.csv": S(counts=("n_patients",)),
    # real rows: one course per patient (n_courses = n_patients); generated rows: shares of model futures
    "ed08d_regimen_timing_aggregates.csv": S(counts=("n_patients", "n_courses"), shares={
        "fraction_regimen_precedes_any_progression": ("frac", "n_courses"), "fraction_progression": ("frac", "n_courses"),
        "fraction_both": ("frac", "n_courses")}),
    "population_composition_scores_20261012.csv": S(
        counts=("n", "n_NSCLC", "n_BRCA", "n_CRC", "deaths_by_h", "alive_through_h", "unresolved_at_h", "deaths_by_6",
                "cohort_n", "cohort_n_NSCLC", "cohort_n_BRCA", "cohort_n_CRC", "cohort_deaths_by_h",
                "cohort_alive_through_h", "cohort_unresolved_at_h", "cohort_deaths_by_6"),
        denoms=("n", "cohort_n"),
        shares={"death_fraction_6": ("frac", "n"), "cohort_death_fraction_6": ("frac", "cohort_n")}),
    # share = patients in the state / n_patients (patients retained in the monthly calculation: dead by then or
    # alive and still followed; n_group is the starting cohort and n_censored = n_group - n_patients), per
    # scripts/explore/n2_state_occupancy_v1.py; the state shares of one group and month partition 1. Only the
    # registry rows are patients; generated rows are averages over model futures and are not masked.
    # Masked with the exact shares and monthly denominators; n_patients and n_censored are then removed (per-cancer
    # counts could be recovered from the all-cancer counts or from the precision of the shares) and the shares left
    # visible are rounded to two decimals (see release_copy).
    "n2_state_occupancy_v1.csv": S(counts=("n_patients", "n_censored", "n_group"), denoms=("n_group",),
                                   shares={"share": ("frac", "n_patients")},
                                   partition=(("cancer", "month", "source", "reading"), ("share",)),
                                   real_rows=("source", "observed")),
    # share_pairs_shared_missing is a share of matched pairs; checked against n_index as a conservative bound
    "n6_support_ladder_v1.csv": S(counts=("n_index",), shares={"share_zero": ("frac", "n_index"),
                                  "share_lt5": ("frac", "n_index"), "share_pairs_shared_missing": ("frac", "n_index")}),
    "n8_time_in_state_v1.csv": S(counts=("n",)),
    "subgroup_rmst24.csv": S(counts=("n_control", "n_experimental")),
    "prime_forest_hr_itt935.csv": S(counts=("n", "n_pan", "n_fol", "events", "n_zero_day", "n_without_zero_day"),
                                    denoms=("n",)),
}
# Listed in the metadata of the Extended Data Fig. 8 panel but not read by its script (it reads the PRIME tables
# directly); they belong to an analysis that is not reported and are not shipped.
NOT_USED = {"n12_components_v1.csv", "n12_fine_cells_v1.csv"}
RIBBON_PARTITION = ("CR or PR", "Stable", "Progressive")  # RECIST categories among evaluable assessments (PRIME)


def panel_tables(pid: str) -> list[Path]:
    """Plotted-value tables a panel script saved itself (aggregates; observed ribbon shares with denominators)."""
    d = PANELS / pid
    out = sorted(d.glob("*_aggregates.csv"))
    for f in sorted(d.glob("*_ribbon_shown.csv")):
        den = d / f.name.replace("_ribbon_shown", "_ribbon_denominators")
        if den.exists() and pd.read_csv(den).drop(columns=["row", "group"], errors="ignore").notna().to_numpy().any():
            out.append(f)
    return out


def ribbon_copy(f: Path, dest: Path) -> int:
    """Copy a clinical-course ribbon (already masked in the figure, scripts/figures/panels/_css_template.py) after an
    independent check. Real patients: every group of a PRIME panel; the observed group of a GENIE-BPC panel."""
    s = pd.read_csv(f, index_col=0)
    d = pd.read_csv(f.parent / f.name.replace("_ribbon_shown", "_ribbon_denominators"), index_col=0)
    real = list(s.group.unique()) if "prime" in f.name else list(s.group[s.observed.astype(bool)].unique())
    problems = MS.verify_ribbon(s, d, real, RIBBON_PARTITION if "prime" in f.name else ())
    if problems:
        raise SystemExit(f"{f}: " + "; ".join(problems))
    months = [c for c in s.columns if c not in ("group", "observed")]
    n = int(sum((s[s.group == g][months].isna() & d[d.group == g][months].reindex(index=s[s.group == g].index).notna())
                .to_numpy().sum() for g in real))
    dest.parent.mkdir(parents=True, exist_ok=True)
    s.drop(columns=["observed"]).to_csv(dest)
    return n


def drop_counts_round_shares(out: pd.DataFrame, spec, drop: tuple[str, ...]) -> pd.DataFrame:
    """After masking: remove count columns and round the visible shares of real patients to two decimals.
    Suppressed shares stay blank; the suppression record keeps only the columns released."""
    out = out.drop(columns=list(drop))
    real = out[spec.real_rows[0]].astype(str) == spec.real_rows[1]
    vis = real & out["share"].notna()
    out.loc[vis, "share"] = pd.to_numeric(out.loc[vis, "share"]).round(2)
    if "suppressed_columns" in out.columns:
        kept = out["suppressed_columns"].fillna("").map(lambda s: [c for c in s.split(";") if c and c not in drop])
        out["masked_cells"] = kept.map(len)
        out["suppressed_columns"] = kept.map(";".join)
    return out


def release_copy(f: Path, dest: Path) -> int:
    """Masked copy of a registry or trial aggregate under its explicit spec; fail unless the independent check passes."""
    if f.name not in SPECS:
        raise SystemExit(f"{f}: no Source Data spec (units and denominators must be declared)")
    d = pd.read_csv(f, low_memory=False)
    spec = SPECS[f.name]
    if spec.real_rows is not None:
        # rows of generated data carry the patient counts of the real cohort they were read against; they repeat
        # the counts of the real rows, so they are left blank there and only the masked real rows give them
        col, val = spec.real_rows
        other = d[col].astype(str) != val
        d.loc[other, [c for c in spec.counts if c in d.columns]] = float("nan")
    out, n = MS.mask(d, spec)
    problems = MS.verify(d, out, SPECS[f.name])
    if problems:
        raise SystemExit(f"{f}: " + "; ".join(problems))
    if f.name == "n2_state_occupancy_v1.csv":
        out = drop_counts_round_shares(out, spec, ("n_patients", "n_censored"))
        n = int(out["masked_cells"].sum()) if "masked_cells" in out.columns else 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return n


RESTRICTED = re.compile(r"cand_css_four_arm|pds_prime|prime_trial|n8_prime|n12_prime|prime_corrections|four_arm|genie|"
                        r"query_universe|n2_state_occupancy|n6_support|fig1_same_dot", re.I)
# per-patient rows or registry query identifiers
PATIENT_LEVEL = {"real_side.csv", "generated_side.csv", "series_cal.csv.gz", "state_cal.csv.gz", "state_snapshots.csv.gz",
                 "queries.jsonl"}
SYNTHETIC_PATIENT_LEVEL = {"state_projection.csv.gz"}
COUNT_COLUMNS = ("n", "n_patients", "n_group", "n_index", "n_control", "n_experimental", "n_evaluable", "n_censored")


def path_of(src: str) -> str:
    return re.split(r"\s|\(", src.strip(), maxsplit=1)[0].rstrip(";")


def small_cells(f: Path) -> int | None:
    if f.suffix not in {".csv", ".gz"}:
        return None
    d = pd.read_csv(f, low_memory=False)
    cols = [c for c in COUNT_COLUMNS if c in d.columns]
    if not cols:
        return None
    v = d[cols].apply(pd.to_numeric, errors="coerce")
    return int(((v >= 1) & (v <= 4)).sum().sum())


def classify(src: str) -> tuple[str, Path | None]:
    p = path_of(src)
    f = ROOT / p
    if not p.startswith(("outputs/", "runs/")):
        return ("no data" if not re.match(r"GENIE|PRIME", p) else "not included"), None
    if f.name in PATIENT_LEVEL or (RESTRICTED.search(p) and f.suffix == ".json" and "manifest" not in f.name
                                   and "readout" not in f.name and "deep_results" not in f.name):
        return "not included", f
    if RESTRICTED.search(p):
        return "pending", f
    if p.startswith("runs/") or f.is_dir() or f.name in SYNTHETIC_PATIENT_LEVEL or not f.exists():
        return "deposit", f
    return "included", f


# synthetic parts extracted unchanged from files that also hold registry aggregates: source -> (key, file name)
EXTRACT = {"outputs/exploratory_fig1_same_dot/readout_v2.json": ("per_model", "same_dot_per_model.csv")}


def panel_sources():
    """(figure, panel letters, panel id, source) for every panel of the main and Extended Data figures."""
    layouts = json.loads((ROOT / "scripts/figures/layouts.json").read_text())
    for fig, spec in layouts.items():
        if fig.startswith("_") or fig.startswith("Supplementary"):
            continue
        for p in spec["panels"]:
            meta = ROOT / "outputs/figures/panels" / p["id"] / f"{p['id']}_meta.json"
            srcs = json.loads(meta.read_text()).get("sources", []) if meta.exists() else ["(no metadata)"]
            label = ",".join(p.get("letters", [])) or p.get("label", "") or "all"
            for s in srcs:
                yield fig, label, p["id"], s


def pending_sources() -> list[str]:
    return sorted({path_of(s) for *_, s in panel_sources() if classify(s)[0] == "pending"})


README = """# Source Data

Large language models as oncology world models (Vibert and Kather). One folder per main or Extended Data figure;
SOURCE_DATA_INDEX.md lists, for every panel, each source table and how it is provided.

Panels built from generated data (model outputs, no patient information): the plotted values, or the generated
records in the Zenodo data deposit.

Panels built from GENIE-BPC registry or PRIME trial data: aggregate tables without direct identifiers (cohort-level
counts, percentages, means, intervals and P values). Cells that show, or would let a reader recover, a count of 1 to 4
patients are left blank: directly, as a complement of a total, as a share of a stated denominator, or by subtraction
within a group that sums to a total. In these tables a blank cell means suppressed, missing or not applicable, never
zero; the column masked_cells records the number of values suppressed in that row and suppressed_columns names them;
pre-existing missing or not-applicable values are not included in either (both columns are absent when nothing in the
table was suppressed). In the clinical
course tables (*_ribbon_shown.csv), values are percentages of the patients with a known value in the interval, except
the Alive and Progression-free rows, which are Kaplan-Meier estimates; a blank cell is a suppressed value or fewer
than five patients with a known value. In n2_state_occupancy_v1.csv, the shares of registry patients are given to two
decimals, and the numbers of patients followed and censored each month are not given.

Not provided: Kaplan-Meier curves of registry or trial patients and panels that show individual patients; these can
be rebuilt only with access to the GENIE-BPC or PRIME data (see Data availability).
"""

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "release_candidates")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    base = a.out / f"source_data_{commit}"
    if base.exists():
        shutil.rmtree(base)
    rows = []
    real_panels = {pid for _f, _l, pid, s in panel_sources() if classify(s)[0] in ("pending", "not included")}
    shipped = set()
    for fig, label, pid in sorted({(fg, lb, p) for fg, lb, p, _s in panel_sources() if p in real_panels}):
        folder = base / fig.replace("Extended Data Figure ", "Extended_Data_Figure_").replace("Figure ", "Figure_")
        for t in panel_tables(pid):
            if (fig, t) in shipped:
                continue
            shipped.add((fig, t))
            n = (ribbon_copy if t.name.endswith("_ribbon_shown.csv") else release_copy)(t, folder / pid / t.name)
            rows.append({"Figure": fig, "Panel": label, "Panel id": pid, "Source": str(t.relative_to(ROOT)),
                         "Class": "included (registry/trial aggregate)",
                         "Note": f"plotted values; {n} cells masked" if n else "plotted values; no small cell"})
    for fig, label, pid, s in panel_sources():
        folder = base / fig.replace("Extended Data Figure ", "Extended_Data_Figure_").replace("Figure ", "Figure_")
        cls, f = classify(s)
        note = ""
        if path_of(s) in EXTRACT:
            key, name = EXTRACT[path_of(s)]
            dest = folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(json.loads(f.read_text())[key]).to_csv(dest, index=False)
            cls, note = "included", f"synthetic part only ({key}), as {name}"
        elif cls == "included":
            dest = folder / f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        elif cls == "pending" and f is not None and f.name in NOT_USED:
            cls, note = "not used", "listed in the panel metadata, not read by the panel script"
        elif cls == "pending" and f is not None and f.is_file():
            if panel_tables(pid):
                cls, note = "plotted values in the panel table", "the panel's plotted-value table is included"
            elif f.suffix in {".csv", ".gz"} and (fig, f) not in shipped:
                shipped.add((fig, f))
                n = release_copy(f, folder / f.relative_to(ROOT).with_suffix(".csv") if f.suffix == ".gz"
                                 else folder / f.relative_to(ROOT))
                cls, note = "included (registry/trial aggregate)", f"{n} cells masked" if n else "no small cell"
            elif f.suffix in {".csv", ".gz"}:
                cls, note = "included (registry/trial aggregate)", "see above"
            else:
                cls, note = "not needed", "metadata or readout; not needed to plot"
        rows.append({"Figure": fig, "Panel": label, "Panel id": pid, "Source": path_of(s), "Class": cls,
                     "Note": note})
    df = pd.DataFrame(rows).drop_duplicates()
    summary = df.drop_duplicates(["Source", "Class"]).Class.value_counts().to_dict()
    md = ["# Source Data index (generated; do not edit)", "",
          f"From `scripts/release/build_source_data.py` at commit {commit}. Existing tables only, no recomputation. "
          "Classes: included (synthetic aggregate, in the package); included (registry/trial aggregate): plotted values "
          "or cohort-level tables of GENIE-BPC or PRIME patients, cells showing or revealing counts of 1 to 4 masked "
          "(scripts/release/mask_small_cells.py); deposit (synthetic runs or patient-level synthetic "
          "tables, in the data deposit); plotted values in the panel table (the panel's own plotted-value table is "
          "included instead of its GENIE-BPC or PRIME source table); not needed (metadata or readout); not used "
          "(listed in panel metadata, not read by the panel script); "
          "not included (patient-level registry or trial data; an aggregate table must be derived for release); "
          "no data (schematic, text or code).", "",
          "Distinct sources by class: " + ", ".join(f"{k} {v}" for k, v in sorted(summary.items())), "",
          "| Figure | Panel | Panel id | Source | Class | Note |", "|---|---|---|---|---|---|"]
    md += [f"| {r.Figure} | {r.Panel} | {r._3} | `{r.Source}` | {r.Class} | {r.Note} |" for r in df.itertuples()]
    text = "\n".join(md) + "\n"
    base.mkdir(parents=True, exist_ok=True)
    (base / "SOURCE_DATA_INDEX.md").write_text(text)
    (base / "README.md").write_text(README)
    (ROOT / "manuscript/release/source_data_index.md").write_text(text)
    print(f"{base}: {summary}")


if __name__ == "__main__":
    main()
