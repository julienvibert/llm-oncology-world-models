#!/usr/bin/env python3
"""Molecular status as it was knowable AT the landmark, not as we know it now.

A sequencing report dated after the landmark is not evidence the treating clinician had.
Handing it to the model is leakage dressed as baseline data: the model would be told the
EGFR result that arrived three months into the course it is being asked to predict. So the
report date gates everything -- `dx_cpt_rep_mos` is months from diagnosis to report, the
same clock as tau, and only reports at or before tau are admitted.

FOUR RULES, each of which exists because its violation is silent:

  1. NEVER INFER GENOTYPE FROM TREATMENT. A patient on osimertinib almost certainly has an
     EGFR mutation. Encoding that would manufacture the very association the experiment
     measures -- the model would receive the answer through the treatment field.

  2. NEVER SAY WILD-TYPE. A panel that reports no EGFR alteration has not established a
     normal gene: coverage, sensitivity and variant class all limit it. The strongest
     honest statement is "no relevant alteration recorded on the available panel", and
     that is only available when the panel actually covered the gene.

  3. UNKNOWN IS A VALUE. Absent, late, and undated reports are unknown -- distinct from
     tested-and-negative. In retrieval, unknown must not be treated as agreement with
     anything; concordant drivers pull donors closer, discordant push them apart, and
     unknown does neither.

  4. DRIVERS ONLY. The panels report hundreds of genes and most alterations here are
     passengers or of uncertain significance. GENIE public data carries NO oncogenicity
     annotation -- no OncoKB call, no curated driver flag -- so a driver cannot be read
     off the file. It comes from a per-cancer allowlist of clinically relevant genes,
     restricted to coding consequences, and the exact protein change is named so the
     reader judges rather than trusts our bucket.

GOVERNANCE. GENIE-BPC under DUA. Aggregate output only, gitignored.

Run:  uv run python scripts/explore/molecular_prefix_v1.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genie_operator_registry_v2 as GV2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENIE_ROOT = Path.home() / "GenieBPC"
COHORTS = {"NSCLC": "NSCLC_Genie", "BRCA": "BRCA_GENIE", "CRC": "CRC_GENIE"}
OUT = ROOT / "outputs" / "exploratory_molecular_prefix"

#: Clinically relevant genes per cancer: the ones that change management or that a
#: clinician would state in a case presentation. Deliberately short. A longer list would
#: readmit the passengers this filter exists to exclude.
RELEVANT: dict[str, tuple[str, ...]] = {
    "NSCLC": ("EGFR", "ALK", "ROS1", "BRAF", "KRAS", "MET", "RET", "ERBB2", "NTRK1",
              "NTRK2", "NTRK3"),
    "BRCA": ("ERBB2", "PIK3CA", "BRCA1", "BRCA2", "ESR1", "AKT1", "PTEN"),
    "CRC": ("KRAS", "NRAS", "BRAF", "ERBB2", "PIK3CA", "MLH1", "MSH2", "MSH6", "PMS2",
            "POLE"),
}
#: The gene whose status defines the headline bucket, where the cancer has one.
HEADLINE = {"NSCLC": ("EGFR", "ALK", "ROS1", "KRAS", "BRAF"),
            "BRCA": ("ERBB2", "PIK3CA", "BRCA1", "BRCA2"),
            "CRC": ("KRAS", "NRAS", "BRAF")}
#: Non-coding and synonymous classes carry no clinical call.
NON_CODING = {"Silent", "Intron", "3'UTR", "5'UTR", "5'Flank", "3'Flank", "IGR", "RNA",
              "Splice_Region"}


@dataclass(frozen=True)
class Molecular:
    """What the panel established at or before the landmark."""

    panel_reported: bool          # a report existed at or before the landmark
    report_month: float | None    # months from diagnosis, on tau's clock
    genes_covered: frozenset[str]
    alterations: tuple[str, ...]  # named, e.g. "EGFR L858R"
    driver_bucket: str            # a headline gene, "none recorded", or "unknown"


def panel_genes(genie: Path) -> dict[str, frozenset[str]]:
    """Which genes each assay actually covers. Absence of a gene from the panel is why
    'not tested' and 'tested, nothing found' must stay distinguishable."""
    out: dict[str, frozenset[str]] = {}
    for path in genie.glob("data_gene_panel_*.txt"):
        stable, genes = None, frozenset()
        for raw in path.read_text().splitlines():
            if raw.startswith("stable_id:"):
                stable = raw.split(":", 1)[1].strip()
            elif raw.startswith("gene_list:"):
                genes = frozenset(raw.split(":", 1)[1].split())
        if stable:
            out[stable] = genes
    return out


def load_reports(genie: Path) -> pd.DataFrame:
    """One row per sequencing report, with its month on the diagnosis clock."""
    cpt = pd.read_csv(genie / "cancer_panel_test_level_dataset.csv", low_memory=False)
    keep = cpt[["record_id", "cpt_genie_sample_id", "dx_cpt_rep_mos"]].copy()
    keep.columns = ["pid", "sample", "report_month"]
    keep["pid"] = keep["pid"].astype(str)
    return keep


def molecular_at_landmark(pid: str, tau: float, reports: pd.DataFrame,
                          variants: dict[str, list[tuple[str, str, str]]],
                          matrix: dict[str, str], panels: dict[str, frozenset[str]],
                          cancer: str) -> Molecular:
    """The molecular picture available at the landmark, and nothing later.

    Undated reports are excluded rather than assumed early: `dx_cpt_rep_mos` missing means
    we cannot place the report relative to the landmark, and an unplaceable report cannot
    be shown to have preceded it.
    """
    rows = reports[reports["pid"] == pid]
    available = [r for r in rows.itertuples()
                 if pd.notna(r.report_month) and float(r.report_month) <= tau + 1e-9]
    if not available:
        return Molecular(False, None, frozenset(), (), "unknown")

    first = min(available, key=lambda r: float(r.report_month))
    covered: set[str] = set()
    named: list[str] = []
    for r in available:
        covered |= panels.get(matrix.get(r.sample, ""), frozenset())
        for gene, change, klass in variants.get(r.sample, []):
            if gene in RELEVANT[cancer] and klass not in NON_CODING:
                named.append(f"{gene} {change}" if change else gene)

    unique = tuple(dict.fromkeys(named))
    headline = next((g for g in HEADLINE[cancer]
                     if any(a.startswith(g + " ") or a == g for a in unique)), None)
    if headline:
        bucket = headline
    elif covered & set(HEADLINE[cancer]):
        # tested and nothing found. NOT wild-type.
        bucket = "none recorded on the available panel"
    else:
        bucket = "unknown"
    return Molecular(True, float(first.report_month), frozenset(covered), unique, bucket)


def render_molecular(m: Molecular, cancer: str) -> str:
    """The sentence the model reads. Never 'wild-type', never a genotype inferred from
    treatment, and unknown stated as unknown."""
    if not m.panel_reported:
        return ("No tumour sequencing report available at or before the current treatment; "
                "molecular status unknown.")
    if m.alterations:
        return ("Tumour sequencing reported before the current treatment; clinically "
                f"relevant findings: {', '.join(m.alterations)}.")
    if m.driver_bucket == "unknown":
        return ("Tumour sequencing reported before the current treatment, but it did not "
                "cover the clinically relevant genes for this cancer; molecular status "
                "unknown.")
    return ("Tumour sequencing reported before the current treatment; no relevant "
            "alteration recorded on the available panel.")


def build(cancer: str, genie: Path) -> tuple[list[dict], dict]:
    patients = GV2.load(genie)
    reports = load_reports(genie)
    panels = panel_genes(genie)
    matrix = dict(pd.read_csv(genie / "data_gene_matrix.txt", sep="\t",
                              low_memory=False)[["SAMPLE_ID", "mutations"]].dropna().values)

    variants: dict[str, list[tuple[str, str, str]]] = {}
    muts = pd.read_csv(genie / "data_mutations_extended.txt", sep="\t", low_memory=False,
                       usecols=["Hugo_Symbol", "HGVSp_Short", "Variant_Classification",
                                "Tumor_Sample_Barcode"])
    for r in muts.itertuples():
        variants.setdefault(r.Tumor_Sample_Barcode, []).append(
            (r.Hugo_Symbol, str(r.HGVSp_Short or "").lstrip("p."),
             r.Variant_Classification))
    fusions = genie / "data_fusions.txt"
    if fusions.exists():
        fus = pd.read_csv(fusions, sep="\t", low_memory=False)
        for r in fus.itertuples():
            variants.setdefault(r.Tumor_Sample_Barcode, []).append(
                (r.Hugo_Symbol, "fusion", "Fusion"))

    rows = []
    for p in patients:
        starts = [ln["start"] for ln in p.lines if ln["start"] > 0]
        if not starts:
            continue
        tau = float(starts[len(starts) // 2])
        m = molecular_at_landmark(p.pid, tau, reports, variants, matrix, panels, cancer)
        # the leakage check: what WOULD have been available with no date gate at all
        all_rows = reports[reports["pid"] == p.pid]
        rows.append({
            "panel_reported": m.panel_reported,
            "driver_bucket": m.driver_bucket,
            "n_alterations": len(m.alterations),
            "has_any_report_ever": len(all_rows) > 0,
            "sentence": render_molecular(m, cancer),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "n_patients": len(frame),
        "report_available_at_landmark": float(frame["panel_reported"].mean()),
        "report_exists_at_any_time": float(frame["has_any_report_ever"].mean()),
        "reports_excluded_as_post_landmark_or_undated": float(
            (frame["has_any_report_ever"] & ~frame["panel_reported"]).mean()),
        "driver_bucket": frame["driver_bucket"].value_counts(normalize=True).round(3).to_dict(),
        "median_alterations_when_reported": float(
            frame[frame["panel_reported"]]["n_alterations"].median()),
    }
    return rows, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for cancer, folder in COHORTS.items():
        genie = GENIE_ROOT / folder
        if not genie.exists():
            continue
        rows, summary = build(cancer, genie)
        report[cancer] = summary
        print(f"\n=== {cancer}  ({summary['n_patients']} patients)")
        print(f"  sequencing report exists at any time      "
              f"{summary['report_exists_at_any_time']:.0%}")
        print(f"  available AT or before the landmark       "
              f"{summary['report_available_at_landmark']:.0%}")
        print(f"  withheld as post-landmark or undated      "
              f"{summary['reports_excluded_as_post_landmark_or_undated']:.0%}  "
              f"<- would have leaked without the date gate")
        for bucket, share in sorted(summary["driver_bucket"].items(),
                                    key=lambda kv: -kv[1])[:5]:
            print(f"      {bucket:42s} {share:.0%}")
        print(f"  example: {rows[0]['sentence']}")

    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
