#!/usr/bin/env python3
"""Assemble a candidate Zenodo deposit of the SYNTHETIC data behind the paper (local only).

Nothing is uploaded. The deposit tree uses repository-relative paths (``runs/<run>/...``,
``outputs/<dir>/...``), so that unpacking it at the root of a fresh checkout restores the files
where the figure scripts expect them (``scripts/release/fetch_and_verify.py`` does that after
checking the checksums). ``--package`` then packs the tree into a few archives for Zenodo.

Content (explicit allow-list, nothing else is copied):

  * the six-model native atlas runs (6 run folders): run manifest and config, per-attempt
    rendered prompts, raw model responses, parsed ledgers, accepted trajectories, rejected
    attempts (``failed_attempt.json``), run inventories and the derivation tables;
  * the controlled-input runs: the factorial / EGFR run (Sonnet 5), the EGFR checks of GLM-5.2
    and Kimi K3 (main run, top-up and retry), and the 11 runs of the 12-scenario benchmark;
  * small synthetic tables and readouts read by the figure scripts or cited in CLAIMS.md;
  * synthetic-only EXPORTS of files whose original also holds a registry aggregate
    (``EXPORTS``): the registry block is removed, every other value is checked equal to the
    original, and the original stays private.

Strictness (the build exits non-zero rather than produce a doubtful deposit):

  1. every source in the allow-list is REQUIRED: a missing folder or export input fails;
  2. a file refused inside an allowed folder (restricted path pattern) fails unless it matches
     an EXPECTED exclusion (``EXPECTED_EXCLUSIONS``, each with its reason); a selected file
     matching an expected exclusion also fails;
  3. a copied text file with a GENIE-BPC identifier pattern (``scripts/check_identifiers.py``),
     a registry data marker (``RESTRICTED_MARKERS``), an API-key pattern or, after the path
     scrub, the local user name, fails;
  4. paths shipped in the code snapshot (git-tracked and kept by ``build_review_snapshot.keep``) are never
     copied, so overlaying the deposit never changes a file of the code release; tracked outputs that the
     snapshot leaves out (outputs/ is not part of the code release) are deposited;
  5. only data formats are copied (``DATA_SUFFIXES``), plus an explicit list of synthetic PNGs.

Release-copy transformations (originals are never modified; every transformed file is listed
in ``TRANSFORMATIONS.tsv`` with the SHA-256 of the original and of the release copy):

  * ``local-path-v1``: in text files, absolute path prefixes that reveal the local machine or
    user (``/Users/<name>/``, ``/home/<name>/``, ``/private/tmp/``,
    ``/private/var/folders/<a>/<b>/T/``) are replaced with ``<LOCAL_PATH>/``;
  * ``provider-key-id-v1``: in API-key management URLs inside provider error messages, the
    account key identifier (a hash, not the key) is replaced with ``<KEY_ID>``;
  * ``synthetic-only-export-v1``: see ``EXPORTS``.

Deposit metadata at the deposit root (never overlaid onto a checkout): ``DEPOSIT_README.md``,
``DEPOSIT_MANIFEST.tsv`` (path, bytes, sha256, group, origin, panels), ``TRANSFORMATIONS.tsv``
and ``SHA256SUMS`` (``shasum -a 256 -c`` format). A package adds ``ARCHIVES_SHA256SUMS``.

    uv run python scripts/release/build_deposit.py --out <tree>                  # build tree
    uv run python scripts/release/build_deposit.py --out <tree> --reuse          # complete/refresh
    uv run python scripts/release/build_deposit.py --out <tree> --package <dir>  # + archives
    uv run python scripts/release/build_deposit.py --out <tree> --list           # dry list
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check_identifiers import count_identifiers  # noqa: E402

# ---------------------------------------------------------------------------------------------
# The plan: REQUIRED sources, EXPECTED exclusions, exports
# ---------------------------------------------------------------------------------------------

ATLAS_RUNS = [
    "pillar1_anchored_atlas_tranche1_v4_n5_live01",  # Qwen 3.6 Flash + Gemini 3.1 Flash Lite
    "pillar1_anchored_atlas_gemma4_n5_live01",  # Gemma 4
    "glm52_full_atlas_n5",  # GLM-5.2
    "kimi_k3_off_full_atlas_n5_complete",  # Kimi K3
    "sonnet5_mini_atlas_n100_2026_07_08",  # Sonnet 5 (first 100 cells)
    "sonnet5_full_atlas_topup_remaining128_n5_2026_07_09",  # Sonnet 5 (remaining cells)
]
EGFR_SONNET_RUN = "interventional_phase1_sonnet_full_v1c"
EGFR_CHECK_RUNS = [
    "interventional_phase1_glm52_full_v1c",
    "interventional_phase1_kimi_k3_off_full_v1c",
    "interventional_phase1_kimi_k3_off_topup75_v1c_retry2",
]
SCENARIO_RUNS = [  # the 12-scenario benchmark (10 run folders; explicit, never globbed)
    "sentinel_usecase_braf_melanoma_v5_2026_07_13",
    "sentinel_usecase_ccrcc_v5_2026_07_13",
    "sentinel_usecase_crc_flip_v5_2026_07_13",
    "sentinel_usecase_egfr_flip_v5_2026_07_13",
    "sentinel_usecase_egfr_post_osi_explicit_resistance_v5_v2_2026_07_13",
    "sentinel_usecase_egfr_post_osi_v5_2026_07_13",
    "sentinel_usecase_her2_v5_2026_07_13",
    "sentinel_usecase_hgsoc_v5_2026_07_13",
    "sentinel_usecase_pancreas_v5_2026_07_13",
    "sentinel_usecase_pdl1_v5_2026_07_13",
]
NO_MANIFEST_RUNS = {  # settings in the run's config.yaml
    "interventional_phase1_kimi_k3_off_full_v1c",
}

# Panel labels in the current numbering (scripts/figures/layouts.json): the panels whose scripts opened files
# of these runs when every panel was rebuilt from the code release and the deposit in an isolated folder
# (10 November 2026). Runs read by no panel feed numbers through their scored tables (Fig. 3e,f; ED Fig. 4a,b).
ATLAS_PANELS = ("Fig. 2c; ED Fig. 2b, 3; Supp. Figs 1, 3, 7, 8 (derivation tables); Supp. Fig. 2a-c (Kimi K3); "
                "via outputs/paper_state_map: Figs 1c, 2a, 3")
EGFR_PANELS = "Fig. 3; ED Fig. 4d,e (trajectory files and derivation tables read)"
SCENARIO_PANELS = "Supp. Fig. 9 (derivation tables and trajectories); producer of the ED Fig. 4a,b scenario table"



@dataclass(frozen=True)
class Source:
    group: str
    root: str  # repository-relative folder (REQUIRED to exist)
    panels: str  # panels that read it (or provenance role)
    files: tuple[str, ...] = ("**/*",)  # globs under root
    derivation_panels: str | None = None  # panels for derivation/*.csv, if different


@dataclass(frozen=True)
class Exclusion:
    pattern: str  # fnmatch pattern on repository-relative paths ('*' crosses '/')
    reason: str


@dataclass(frozen=True)
class Export:
    """A synthetic-only JSON export of a private original (top-level keys dropped)."""

    group: str
    source: str  # repository-relative original (REQUIRED; stays private)
    dest: str  # deposit-relative path of the export
    drop_keys: tuple[str, ...]  # top-level keys removed (must be present in the original)
    drop_label: str  # how the dropped content is described in deposit metadata
    panels: str


@dataclass(frozen=True)
class Plan:
    root: Path
    sources: tuple[Source, ...]
    exclusions: tuple[Exclusion, ...]
    exports: tuple[Export, ...] = ()
    synthetic_pngs: frozenset[str] = frozenset()
    restricted_path: re.Pattern[str] = field(default_factory=lambda: RESTRICTED_PATH)


SYNTHETIC_PNGS = frozenset({  # binary inputs read by a current panel (synthetic): numeric arrays, no pickled objects
    "outputs/exploratory_n11_event_fields/n11_density_v1.npz",  # Supp. Fig. 6
})

# Tables: every file a panel script of the current layout opened in the isolated rebuild (10 November 2026), and
# every synthetic output that manuscript/TRACE_ALL.md cites for a reported number. ``trace_problems`` checks at
# build time that every output the trace cites is deposited, rebuilt by a panel, or an expected exclusion.
TABLE_FILES = {  # folder -> ((file, what reads it), ...)
    "outputs/exploratory_egfr_pre_progression_audit": (
        ("arm_counts_medians.csv", "numbers in Supplementary Note 1"),
        ("osi_minus_chemo.csv", "numbers in Supplementary Note 1"),
        ("per_future_audit.csv", "per-future flags behind Supplementary Note 1"),
    ),
    "outputs/exploratory_atlas_archetype": (
        ("readout.json", "numbers in Figure 1 legend, Main, Methods"),
    ),
    "outputs/exploratory_atlas_dissection": (
        ("readout.json", "numbers in Figure 1 legend, Main, Methods, Supplementary Table 4"),
    ),
    "outputs/exploratory_atlas_phenotype_six_model": (
        ("phenotype_by_tissue_model.csv", "ED Fig. 1a; numbers in Extended Data Figure 1 legend"),
        ("readout.json", "numbers in Figure 1 legend, Methods"),
    ),
    "outputs/exploratory_field_aggregation_audit": (
        ("readout.json", "Supp. Fig. 7b; Supp. Fig. 7c; numbers in Figure 2 legend, Main, Supplementary Figure 7"),
    ),
    "outputs/exploratory_figure_panels/fig1/data": (
        ("driver_entropy.csv", "numbers in Figure 1 legend, Main, Methods"),
        ("exemplar.json", "Fig. 1c; Fig. 2a"),
        ("modal_share.csv", "Fig. 1b; numbers in Figure 1 legend, Main, Methods"),
        ("survival_by_family.csv", "Fig. 1d; numbers in Figure 1 legend, Main, Methods"),
        ("survival_by_family_test.csv", "Fig. 1d"),
    ),
    "outputs/exploratory_interventional_phase1": (
        ("phase1_context_arm_medians.csv", "Fig. 3e; Fig. 3f"),
        ("phase1_endpoint_long.csv", "Fig. 3f; numbers in Extended Data Figure 4 legend, Main, Methods"),
        ("phase1_gate_scores.csv", "ED Fig. 4a; ED Fig. 4b"),
    ),
    "outputs/exploratory_interventional_phase1_glm52_full": (
        ("phase1_cell_arm_medians.csv", "numbers in Figure 3 legend, Main, Methods"),
        ("phase1_context_arm_medians.csv", "Fig. 3e; Fig. 3f; numbers in Extended Data Figure 4 legend, Figure 3 legend, Main, Methods"),
        ("phase1_endpoint_long.csv", "Fig. 3f; numbers in Figure 3 legend, Main, Methods"),
        ("phase1_gate_readout.md", "numbers in Figure 3 legend, Main, Methods"),
        ("phase1_gate_scores.csv", "ED Fig. 4a; ED Fig. 4b; numbers in Figure 3 legend, Main, Methods"),
        ("phase1_supportive_ttp_trajectory.csv", "numbers in Figure 3 legend, Main, Methods"),
    ),
    "outputs/exploratory_interventional_phase1_kimi_k3_combined": (
        ("phase1_context_arm_medians.csv", "Fig. 3e; Fig. 3f"),
        ("phase1_endpoint_long.csv", "Fig. 3f"),
        ("phase1_gate_scores.csv", "ED Fig. 4a; ED Fig. 4b"),
    ),
    "outputs/exploratory_n11_event_fields": (
        ("manifest.json", "numbers in Supplementary Figure 6"),
        ("n11_exemplar_path_v1.csv", "Supp. Fig. 6"),
        ("n11_exemplar_v1.csv", "Supp. Fig. 6"),
        ("n11_fields_v1.csv", "Supp. Fig. 6"),
        ("n11_phases_v1.csv", "Supp. Fig. 6; numbers in Supplementary Figure 6"),
    ),
    "outputs/exploratory_native_atlas_geometry": (
        ("readout.json", "ED Fig. 1b; ED Fig. 1c; Supp. Fig. 5c; numbers in Extended Data Figure 1 legend, Figure 1 legend, Main, Methods, Supplementary Figure 5, Supplementary Table 4"),
    ),
    "outputs/exploratory_perturbation_matrix": (
        ("readout.json", "numbers in Main"),
    ),
    "outputs/exploratory_precision_ladder": (
        ("readout.json", "Supp. Fig. 15a; Supp. Fig. 15b; Supp. Fig. 15c; numbers in Figure 6 legend, Main, Supplementary Figure 15, Supplementary Table 4"),
    ),
    "outputs/exploratory_sentinel_benchmark": (
        ("sentinel_trust_map_table.csv", "ED Fig. 4a; ED Fig. 4b; numbers in Extended Data Figure 4 legend, Figure 3 legend, Main, Methods"),
    ),
    "outputs/paper_atlas_finite_displacement_field": (
        ("manifest.json", "numbers in Supplementary Figure 7"),
    ),
    "outputs/paper_conditioning_ladder_frozen_map": (
        ("readout.json", "numbers in Main"),
    ),
    "outputs/paper_egfr_dynamics_summary": (
        ("readout.json", "numbers in Main"),
    ),
    "outputs/paper_egfr_remission_attractor": (
        ("trajectory_horizon_attractor.csv", "numbers in Main"),
    ),
    "outputs/paper_interventional_trust_map": (
        ("egfr_gate_scores_used.csv", "Fig. 3e"),
        ("readout.json", "Fig. 3e; numbers in Extended Data Figure 4 legend, Figure 3 legend, Main, Methods, Supplementary Table 3"),
    ),
    "outputs/paper_state_embedding_bakeoff": (
        ("embedding_coordinates.csv.gz", "Supp. Fig. 4a"),
        ("landmark_states_and_pcs.csv.gz", "Supp. Fig. 4a"),
        ("manifest.json", "numbers in Supplementary Figure 4"),
        ("representative_trajectories.csv", "Supp. Fig. 4a"),
        ("target_tumour_trajectory_states.csv.gz", "Supp. Fig. 4a"),
    ),
    "outputs/paper_state_map": (
        ("event_validation.csv", "Supp. Fig. 4d"),
        ("feature_loadings.csv", "Supp. Fig. 4c; Supp. Fig. 8a; Supp. Fig. 8b"),
        ("manifest.json", "Supp. Fig. 4b; numbers in Figure 2 legend, Main"),
        ("phase1_p0_records.csv", "ED Fig. 4d; ED Fig. 4e; Fig. 3"),
        ("state_map_contract.json", "ED Fig. 4d; ED Fig. 4e; Fig. 3; Supp. Fig. 7a; numbers in Supplementary Figure 4"),
        ("state_projection.csv.gz", "ED Fig. 2b; ED Fig. 3; ED Fig. 4d; ED Fig. 4e; ED Fig. 6; ED Fig. 9; Fig. 1c; Fig. 2a; Fig. 2c; Fig. 3; Fig. 5; Fig. 6; Supp. Fig. 10; Supp. Fig. 14; Supp. Fig. 2c; Supp. Fig. 4b; Supp. Fig. 5a; Supp. Fig. 5b; Supp. Fig. 7a; Supp. Fig. 7d; Supp. Fig. 8a; Supp. Fig. 8b; Supp. Fig. 9; numbers in Figure 1 legend, Methods, Supplementary Figure 4, Supplementary Table 4"),
    ),
    "outputs/paper_survival_in_state_space": (
        ("readout.json", "numbers in Figure 3 legend, Main, Methods"),
    ),
    "outputs/perturbation_matrix_decision_aligned": (
        ("readout.json", "ED Fig. 4c; numbers in Data availability, Figure 3 legend, Main, Methods"),
    ),
}


RESTRICTED_PATH = re.compile(
    r"genie|prime|pds|four_arm|query_universe|hybrid|real_backbone|memory_kernel|"
    r"state_space_gate|fig1_same_dot", re.I)

# Restricted material that is deliberately NOT deposited. A refusal that matches none of these
# fails the build (it means a restricted file sits inside an allowed folder, or the plan drifted).
EXPECTED_EXCLUSIONS = (
    Exclusion("runs/four_arm_*",
              "registry experiment: generated futures conditioned on GENIE-BPC patients; "
              "query ids carry GENIE identifiers and prompts carry patient histories"),
    Exclusion("outputs/four_arm_*",
              "scores and per-patient extracts from the registry experiment (GENIE-BPC)"),
    Exclusion("outputs/exploratory_four_arm*",
              "registry-experiment analyses and patient prompts (GENIE-BPC)"),
    Exclusion("outputs/exploratory_query_universe*",
              "GENIE-BPC patient and donor selections"),
    Exclusion("outputs/*prime*", "PRIME trial (Project Data Sphere) tables and aggregates"),
    Exclusion("outputs/*pds*", "Project Data Sphere downloads and derived tables"),
    Exclusion("outputs/exploratory_genie*", "GENIE-BPC-derived readouts"),
    Exclusion("outputs/exploratory_memory_kernel_genie*", "GENIE-BPC-derived readouts"),
    Exclusion("outputs/exploratory_state_space_gate*", "GENIE-BPC-derived readouts"),
    Exclusion("*hybrid*", "hybrid real-backbone work (generation on real GENIE-BPC backbones)"),
    Exclusion("*real_backbone*", "hybrid real-backbone work (real GENIE-BPC backbones)"),
    Exclusion("outputs/exploratory_fig1_same_dot/*",
              "original readouts also hold a GENIE-BPC aggregate block; kept private. "
              "A synthetic-only export of readout*.json is deposited instead"),
)

# Runs that sent REAL patient data to a model API (the authors' inventory of external processing of
# registry and trial records, 14 October 2026). Each is an expected exclusion
# with the reason "real patient data", and ``real_patient_hits`` makes the build fail if any
# deposit path lies in one of them, whatever the allow-list says.
REAL_GENIE_RUNS = (
    # runner-based runs (GENIE patient ID in the prompt)
    "runs/genie_baseline_smoke_2026_06_07_live01",
    "runs/genie_baseline_frontier_same10_2026_06_08_live01",
    "runs/genie_cf_smoke_2026_06_08_live01",
    "runs/synthonccases_2026_06_08_090234",
    "runs/synthonccases_2026_07_08_100115",
    "runs/sonnet5_treatment_screener_2026_07_08",
    # four-arm live runs
    "runs/four_arm_v1_smoke_live_20260826_unpinned",
    "runs/four_arm_v1_smoke_live_20260826",
    "runs/four_arm_v1_smoke_live_20260826_deepseek0731",
    "runs/four_arm_v1_smoke_live_parallel40_20260826",
    "runs/four_arm_v1_smoke_live_terminal_gate_20260826",
    "runs/four_arm_v1_definitive_20260826",
    "runs/four_arm_v1_definitive_p40b20_20260826",
    # hybrid real-backbone pilots
    "outputs/hybrid_pilot",
    "outputs/hybrid_pilot_baseline_impute_gemini",
    "outputs/hybrid_pilot_baseline_impute_sonnet",
    # anchored-PFS live smokes (whole tree: live smokes, dry run, donor tables)
    "outputs/exploratory_anchored_pfs_v1",
)
REAL_GENIE_PREFIXES = ("runs/genie_baseline_", "runs/genie_cf_")
REAL_PRIME_RUNS = (
    "runs/prime_virtual_trial_gemma4_v1",
    "runs/synthonccases_2026_08_05_154822",
    "runs/prime_vt_deepseek0731_forcedoff_retest",
    "runs/prime_virtual_trial_deepseek0731_smoke10",
    "runs/prime_virtual_trial_deepseek0731_smoke20",
    "runs/prime_virtual_lattice_gemma4_blockA_p20",
    "runs/prime_virtual_lattice_gemma4_blockB_p20",
    "runs/synthonccases_2026_08_06_112915",
    "runs/prime_virtual_lattice_deepseek0731_blockB_p20",
)
BORDERLINE_RUNS = (  # not shown to be purely constructed: excluded
    "runs/prime_factorial_grid_gemma4_p20",
)
REAL_PATIENT_REASON_GENIE = ("real patient data: GENIE-BPC records sent to a model API "
                             "(external processing inventory, table A)")
REAL_PATIENT_REASON_PRIME = ("real patient data: PRIME (Project Data Sphere) records sent to a "
                             "model API (external processing inventory, table B)")
BORDERLINE_REASON = ("borderline, excluded: prompts built from PRIME trial aggregates "
                     "(constructed 'modal patient'); goes with the registry-aggregate decision")
REAL_PATIENT_EXCLUSIONS = (
    tuple(Exclusion(f"{r}/*", REAL_PATIENT_REASON_GENIE) for r in REAL_GENIE_RUNS)
    + tuple(Exclusion(f"{p}*", REAL_PATIENT_REASON_GENIE) for p in REAL_GENIE_PREFIXES)
    + tuple(Exclusion(f"{r}/*", REAL_PATIENT_REASON_PRIME) for r in REAL_PRIME_RUNS)
    + tuple(Exclusion(f"{r}/*", BORDERLINE_REASON) for r in BORDERLINE_RUNS)
)
EXPECTED_EXCLUSIONS = REAL_PATIENT_EXCLUSIONS + EXPECTED_EXCLUSIONS
FORBIDDEN_FOLDERS = REAL_GENIE_RUNS + REAL_PRIME_RUNS + BORDERLINE_RUNS


def real_patient_hits(paths, folders: tuple[str, ...] = FORBIDDEN_FOLDERS,
                      prefixes: tuple[str, ...] = REAL_GENIE_PREFIXES) -> list[str]:
    """Deposit paths that lie in a run known to hold real patient data (must be empty)."""
    return [p for p in paths if p.startswith(prefixes)
            or any(p == f or p.startswith(f + "/") for f in folders)]


EXPORTS = tuple(
    Export(
        group="synthetic_export",
        source=f"outputs/exploratory_fig1_same_dot/{name}.json",
        dest=f"outputs/exploratory_fig1_same_dot_synthetic_only/{name}_synthetic_only.json",
        drop_keys=("real_genie",),
        drop_label="top-level block holding the GENIE-BPC NSCLC aggregate (1 key)",
        panels="numbers in ED Fig. 2 legend and Methods (synthetic values only; TRACE_ALL.md cites readout_v2.json)",
    )
    for name in ("readout_v2",)
)
# outputs the trace cites that are derived from GENIE-BPC or PRIME records although their path does not say so
REGISTRY_DERIVED = (
    Exclusion("outputs/exploratory_n2_state_occupancy*", "state occupancy of the registry experiment (GENIE-BPC)"),
    Exclusion("outputs/exploratory_n6_support_ladder*", "support ladder computed on GENIE-BPC and PRIME patients"),
    Exclusion("outputs/exploratory_n8_prime_time_in_state*", "PRIME time in clinical states"),
    Exclusion("outputs/exploratory_n12_prime_decomposition*", "PRIME decomposition"),
    Exclusion("outputs/exploratory_n14_support_gradient*", "donor support gradient of the registry experiment"),
    Exclusion("outputs/exploratory_patient_characteristics*", "characteristics of GENIE-BPC and PRIME patients"),
    Exclusion("outputs/exploratory_cand_css_four_arm*", "registry-experiment numbers and series (GENIE-BPC)"),
)
# outputs of analyses the paper does not report (cited only through a grouped ledger entry)
NOT_REPORTED = (
    Exclusion("outputs/exploratory_n10_precision_ladder*", "N10 arrow ladder: not reported in the paper"),
)
TRACE_FILE = "manuscript/TRACE_ALL.md"
# TRACE output entries that are not files (panel outputs are rebuilt by the panel scripts; placeholders)
TRACE_REBUILT = ("outputs/figures/",)
EXPORT_NOTE_KEY = "_export_note"


def sources(root: Path = ROOT) -> tuple[Source, ...]:
    out = [Source("atlas_run", f"runs/{r}", "provenance; re-derivation",
                  derivation_panels=ATLAS_PANELS) for r in ATLAS_RUNS]
    out.append(Source("egfr_factorial_run", f"runs/{EGFR_SONNET_RUN}",
                      EGFR_PANELS,
                      derivation_panels=EGFR_PANELS))
    out += [Source("egfr_check_run", f"runs/{r}",
                   "producer of the Fig. 3e,f and ED Fig. 4a,b EGFR check scores") for r in EGFR_CHECK_RUNS]
    out += [Source("scenario_benchmark_run", f"runs/{r}", SCENARIO_PANELS) for r in SCENARIO_RUNS]
    out += [Source("table", folder, "; ".join(sorted({w for _, w in items})), files=tuple(f for f, _ in items))
            for folder, items in TABLE_FILES.items()]
    out += [Source("table", str(Path(p).parent), "Supp. Fig. 6", files=(Path(p).name,)) for p in sorted(SYNTHETIC_PNGS)]
    return tuple(out)


def default_plan() -> Plan:
    return Plan(root=ROOT, sources=sources(), exclusions=EXPECTED_EXCLUSIONS + REGISTRY_DERIVED + NOT_REPORTED, exports=EXPORTS,
                synthetic_pngs=SYNTHETIC_PNGS)


def trace_outputs(root: Path = ROOT) -> list[str]:
    """Outputs cited in the output column of manuscript/TRACE_ALL.md (numbers of the current manuscript)."""
    out = []
    for ln in (root / TRACE_FILE).read_text().splitlines():
        c = ln.split(" | ")
        if ln.startswith("| ") and len(c) == 6:
            out += re.findall(r"`([^`]+)`", c[4])
    return sorted(set(out))


def trace_problems(plan: Plan, deposited: set[str]) -> list[str]:
    """Trace outputs that are neither deposited, rebuilt by a panel, nor an expected exclusion."""
    if not (plan.root / TRACE_FILE).is_file():
        return []
    exports = {e.source: e.dest for e in plan.exports}
    bad = []
    for o in trace_outputs(plan.root):
        if o.startswith(TRACE_REBUILT) or o in exports or matches(o.rstrip("/*"), plan.exclusions) \
                or plan.restricted_path.search(o):
            continue
        if o.endswith("/"):
            if not any(d.startswith(o) for d in deposited):
                bad.append(o)
        elif o not in deposited:
            bad.append(o)
    return bad


# registry data markers: column names, release folders, dataset names, aggregate keys
RESTRICTED_MARKERS = re.compile(
    r"real_genie|GenieBPC|NSCLC_Genie|CRC_GENIE|BRCA_GENIE|genie_release|GENIE_ROOT|"
    r"PDS_ROOT|Colorec_Amgen|adsl_pds|\bcohort_ca_dx\b|\bca_dx_index\b|\bprissmm\b", re.I)
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{20,}|AIza[0-9A-Za-z_-]{30,}")
DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
TEXT_SUFFIXES = DATA_SUFFIXES
GZ_TEXT = (".csv.gz", ".tsv.gz", ".jsonl.gz")  # compressed tables: scanned and scrubbed decompressed


def is_text(rel: str) -> bool:
    return Path(rel).suffix.lower() in TEXT_SUFFIXES or rel.lower().endswith(GZ_TEXT)


def as_text(rel: str, data: bytes) -> bytes:
    return gzip.decompress(data) if rel.lower().endswith(GZ_TEXT) else data
SKIP_NAMES = {".DS_Store"}
META_FILES = ("DEPOSIT_README.md", "DEPOSIT_MANIFEST.tsv", "TRANSFORMATIONS.tsv",
              "LICENSE.txt")
LICENCE_SCOPE = (
    "This licence covers the synthetic data in this deposit only. No AACR Project GENIE "
    "BPC data and no PRIME trial (Project Data Sphere) data are included; those data remain "
    "under their own terms and data-use agreements. The model outputs were generated under the "
    "terms of service of the model providers and of the API services used to reach them "
    "(see each run's manifest or config). The code is released separately under "
    "the MIT licence in the code repository.")
LICENCE_TEXT = (
    "SynthOncCases synthetic data deposit\n"
    "Copyright (c) 2026 The SynthOncCases authors\n\n"
    "Licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).\n"
    "You are free to share and adapt the material for any purpose, provided you give "
    "appropriate credit, provide a link to the licence and indicate if changes were made.\n"
    "Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode\n\n"
    + LICENCE_SCOPE + "\n")

# local-path-v1: absolute prefixes that reveal the machine or the user name
SCRUB_RULE = "local-path-v1"
SCRUB_TOKEN = "<LOCAL_PATH>/"
_SEG = r"[^/\s\"'<>`]+"
LOCAL_PATH = re.compile(
    rf"/(?:Users|home)/{_SEG}/|/private/tmp/|/private/var/folders/{_SEG}/{_SEG}/T/")
# provider-key-id-v1: API-key management URLs in provider error messages carry an account key
# identifier (a hash, not the key itself); the identifier is redacted, the URL kept
KEY_RULE = "provider-key-id-v1"
KEY_ID = re.compile(r"(https?://[^\s\"'<>]*?/keys/)[0-9a-f]{16,}")
KEY_RULE_TEXT = ("in API-key management URLs inside provider error messages "
                 "('https://<host>/.../keys/<hex id>'), the key identifier is replaced with "
                 "'<KEY_ID>'")
SCRUB_RULES = ((SCRUB_RULE, LOCAL_PATH, SCRUB_TOKEN), (KEY_RULE, KEY_ID, r"\1<KEY_ID>"))
SCRUB_RULE_TEXT = ("'/Users/<name>/', '/home/<name>/', '/private/tmp/' and "
                   "'/private/var/folders/<a>/<b>/T/' are replaced with '<LOCAL_PATH>/' "
                   "in text files of the release copy")

ARCHIVE_OF_GROUP = {
    "atlas_run": "atlas_runs",
    "egfr_factorial_run": "egfr_factorial_run",
    "egfr_check_run": "egfr_check_runs",
    "scenario_benchmark_run": "scenario_benchmark_runs",
    "table": "tables",
    "synthetic_export": "tables",
}
ARCHIVE_PREFIX = "synthonccases_deposit_"
ARCHIVE_SUMS = "ARCHIVES_SHA256SUMS"


class BuildError(Exception):
    """The deposit cannot be built as planned (the CLI exits non-zero)."""


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def tracked_paths(root: Path = ROOT) -> set[str]:
    r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(f"{root} is not a git checkout (needed to exclude tracked files)")
    return set(r.stdout.splitlines())


def released_code(tracked: set[str]) -> set[str]:
    """Tracked paths that the code snapshot ships (outputs/ and runs/ are left out of it)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_review_snapshot as S
    return {p for p in tracked if S.keep(p, set())}


def git_head(root: Path = ROOT) -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_names() -> set[str]:
    """User names that must not survive in a release copy (short names are too ambiguous)."""
    names = {Path.home().name}
    try:
        names.add(getpass.getuser())
    except Exception:  # noqa: BLE001 - no login name available
        pass
    return {n for n in names if len(n) >= 4 and n not in {"root", "runner", "user"}}


def scrub_text(data: bytes) -> tuple[bytes, dict[str, int]]:
    """Apply the release-copy rules. Returns the new bytes and replacements per rule."""
    text = data.decode("utf-8", errors="surrogateescape")
    counts = {}
    for rule, pattern, repl in SCRUB_RULES:
        text, n = pattern.subn(repl, text)
        if n:
            counts[rule] = n
    return (text.encode("utf-8", errors="surrogateescape"), counts) if counts else (data, {})


def matches(rel: str, exclusions: tuple[Exclusion, ...]) -> Exclusion | None:
    for e in exclusions:
        if fnmatch.fnmatchcase(rel, e.pattern) or fnmatch.fnmatchcase(rel + "/", e.pattern):
            return e
    return None


@dataclass
class Selection:
    chosen: list[tuple[str, Source]]
    refused: list[str]  # repository-relative paths refused by the restricted path pattern
    skipped_suffix: Counter  # rendering files not deposited, by suffix
    skipped_tracked: int


def select(plan: Plan, tracked: set[str]) -> Selection:
    """Files to copy. A missing REQUIRED source raises BuildError."""
    missing = [s.root for s in plan.sources if not (plan.root / s.root).is_dir()]
    # a file named without a wildcard is REQUIRED too
    missing += [f"{s.root}/{f}" for s in plan.sources for f in s.files
                if not any(ch in f for ch in "*?[") and not (plan.root / s.root / f).is_file()]
    missing += [e.source for e in plan.exports if not (plan.root / e.source).is_file()]
    if missing:
        raise BuildError("required source(s) missing: " + ", ".join(missing))
    bad_src = [s.root for s in plan.sources if plan.restricted_path.search(s.root)]
    if bad_src:
        raise BuildError("allow-list names a restricted path: " + ", ".join(bad_src))
    sel = Selection([], [], Counter(), 0)
    seen: set[str] = set()
    for src in plan.sources:
        base = plan.root / src.root
        for pattern in src.files:
            for p in sorted(base.glob(pattern)):
                if not p.is_file() or p.is_symlink() or p.name in SKIP_NAMES:
                    continue
                rel = p.relative_to(plan.root).as_posix()
                if rel in seen:
                    continue
                if rel in tracked:
                    sel.skipped_tracked += 1
                    continue  # rule 4: tracked files stay with the code
                if plan.restricted_path.search(rel.removeprefix(src.root)):
                    sel.refused.append(rel)
                    continue
                if not is_text(rel) and rel not in plan.synthetic_pngs:
                    sel.skipped_suffix[p.suffix.lower() or p.name] += 1
                    continue
                seen.add(rel)
                sel.chosen.append((rel, src))
    return sel


def check_selection(plan: Plan, sel: Selection) -> list[str]:
    """Unexpected refusals and selected files that an expected exclusion forbids."""
    errors = [f"unexpected refusal (no expected exclusion matches): {r}"
              for r in sel.refused if matches(r, plan.exclusions) is None]
    errors += [f"selected file matches expected exclusion '{e.pattern}': {rel}"
               for rel, _ in sel.chosen if (e := matches(rel, plan.exclusions))]
    return errors


def export_bytes(plan: Plan, exp: Export) -> bytes:
    """Build a synthetic-only export and verify it against the original."""
    original = json.loads((plan.root / exp.source).read_text())
    if not isinstance(original, dict):
        raise BuildError(f"export source is not a JSON object: {exp.source}")
    absent = [k for k in exp.drop_keys if k not in original]
    if absent:
        raise BuildError(f"export {exp.source}: key(s) to drop are absent (plan is stale)")
    kept = {k: v for k, v in original.items() if k not in exp.drop_keys}
    note = (f"Synthetic-only export of {exp.source} for the public deposit. Removed: "
            f"{exp.drop_label}. Every other key and value is identical to the original, "
            "which is kept private.")
    out = {EXPORT_NOTE_KEY: note, **kept}
    data = (json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode()
    # verification: parse back, compare with the original minus the dropped keys
    back = json.loads(data)
    back.pop(EXPORT_NOTE_KEY)
    if back != kept or any(k in back for k in exp.drop_keys):
        raise BuildError(f"export {exp.dest}: values differ from the original")
    if RESTRICTED_MARKERS.search(json.dumps(back)) or re.search(r"genie", " ".join(
            _all_keys(back)), re.I):
        raise BuildError(f"export {exp.dest}: registry content remains after the drop")
    return data


def _all_keys(obj: object) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [k for v in obj.values() for k in _all_keys(v)]
    if isinstance(obj, list):
        return [k for v in obj for k in _all_keys(v)]
    return []


def content_problems(rel: str, data: bytes, names: set[str]) -> list[str]:
    if not is_text(rel):
        return []
    text = as_text(rel, data).decode("utf-8", errors="ignore")
    out = []
    if count_identifiers(text):
        out.append("identifier pattern")
    if RESTRICTED_MARKERS.search(text):
        out.append("registry data marker")
    if SECRET.search(text):
        out.append("secret pattern")
    if LOCAL_PATH.search(text) or any(n in text for n in names):
        out.append("local path or user name after scrub")
    if KEY_ID.search(text):
        out.append("API key identifier after scrub")
    return out


def mentions_registry(rel: str, data: bytes) -> bool:
    """File merely names GENIE/PRIME (reviewed by hand, listed in README)."""
    if not is_text(rel):
        return False
    pat = re.compile(r"\bGENIE\b|\bPRIME\b|Project Data Sphere", re.I)
    return bool(pat.search(as_text(rel, data).decode("utf-8", errors="ignore")))


def run_provenance(root: Path, run_dirs: list[str]) -> list[dict]:
    rows = []
    for rd in run_dirs:
        run = rd.removeprefix("runs/")
        mf = root / rd / "manifest.json"
        if not mf.is_file():
            rows.append({"run": run, "models": "?", "git_commit": "?", "started": "?",
                         "accepted": "?", "failed": "?",
                         "template_sha_ok": "no manifest; settings in the run's config.yaml"})
            continue
        m = json.loads(mf.read_text())
        models = ", ".join(x.get("id", "?") for x in m.get("models", []) if isinstance(x, dict))
        commit, tpath, tsha = (m.get("git_commit"), m.get("prompt_template_path"),
                               m.get("prompt_template_sha256"))
        verified = "n/a"
        if commit and tpath and tsha:
            r = subprocess.run(["git", "show", f"{commit}:{tpath}"], cwd=root,
                               capture_output=True)
            verified = ("yes" if r.returncode == 0 and sha256_bytes(r.stdout) == tsha
                        else ("commit or file not in history" if r.returncode else "MISMATCH"))
        rows.append({"run": run, "models": models, "git_commit": (commit or "")[:10],
                     "started": (m.get("started_at") or "")[:10],
                     "accepted": m.get("trajectories_accepted"),
                     "failed": m.get("trajectories_failed"), "template_sha_ok": verified})
    return rows


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------

@dataclass
class BuildResult:
    rows: list[dict]
    transformations: list[dict]
    selection: Selection
    head: str


def _write_if_changed(dst: Path, data: bytes, digest: str, reuse: bool) -> bool:
    """Write unless an identical file (same SHA-256) is already there. True if written."""
    if reuse and dst.is_file() and sha256(dst) == digest:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return True


def build(plan: Plan, out: Path, reuse: bool = False, names: set[str] | None = None,
          log=print) -> BuildResult:
    """Build the deposit tree in ``out``. Raises BuildError on any failed rule."""
    names = local_names() if names is None else names
    out = out.resolve()
    if out.is_relative_to(plan.root.resolve()):
        raise BuildError("--out must be outside the repository")
    if out.exists() and any(out.iterdir()):
        if not reuse:
            raise BuildError(f"{out} is not empty (use --reuse to refresh an earlier build)")
        if not ((out / "SHA256SUMS").is_file() or (out / "DEPOSIT_MANIFEST.tsv").is_file()):
            raise BuildError(f"--reuse refused: {out} does not look like an earlier deposit")
    tracked = released_code(tracked_paths(plan.root))
    sel = select(plan, tracked)
    errors = check_selection(plan, sel)
    if errors:
        raise BuildError("; ".join(errors[:20]))
    export_dests = {e.dest for e in plan.exports}
    hits = real_patient_hits([rel for rel, _ in sel.chosen] + sorted(export_dests)
                             + [s.root for s in plan.sources])
    if hits:
        raise BuildError(f"{len(hits)} path(s) in a real-patient run, e.g. {hits[0]}")
    clash = export_dests & ({rel for rel, _ in sel.chosen} | tracked)
    if clash:
        raise BuildError(f"export path collides with a source or tracked file: {sorted(clash)}")
    missing_trace = trace_problems(plan, {rel for rel, _ in sel.chosen})
    if missing_trace:
        raise BuildError("output(s) cited in the trace are not deposited: " + ", ".join(missing_trace[:20]))
    wanted = {rel for rel, _ in sel.chosen} | export_dests
    out.mkdir(parents=True, exist_ok=True)
    for stale in (*META_FILES, "SHA256SUMS"):
        (out / stale).unlink(missing_ok=True)
    for f in [f for f in out.rglob("*") if f.is_file()]:
        if f.relative_to(out).as_posix() not in wanted:
            f.unlink()  # a file dropped from the plan must not linger
    rows, transforms, problems, written = [], [], [], 0
    for rel, src in sel.chosen:
        original = (plan.root / rel).read_bytes()
        if rel.lower().endswith(GZ_TEXT):
            inner, counts = scrub_text(gzip.decompress(original))
            data = gzip.compress(inner, mtime=0) if counts else original
        else:
            data, counts = (scrub_text(original) if is_text(rel) else (original, {}))
        digest = sha256_bytes(data)
        problems += [f"{p}: {rel}" for p in content_problems(rel, data, names)]
        written += _write_if_changed(out / rel, data, digest, reuse)
        origin = "copy"
        if counts:
            origin = "copy+" + "+".join(counts)
            transforms.append({"path": rel, "transformation": "+".join(counts), "source": rel,
                               "detail": "; ".join(f"{r}: {n} replacement(s)"
                                                   for r, n in counts.items()),
                               "sha256_original": sha256_bytes(original),
                               "sha256_release": digest})
        rows.append({"path": rel, "bytes": len(data), "sha256": digest, "group": src.group,
                     "origin": origin, "panels": _panels_for(src, rel),
                     "mentions": mentions_registry(rel, data)})
    for exp in plan.exports:
        data = export_bytes(plan, exp)
        digest = sha256_bytes(data)
        problems += [f"{p}: {exp.dest}" for p in content_problems(exp.dest, data, names)]
        written += _write_if_changed(out / exp.dest, data, digest, reuse)
        transforms.append({"path": exp.dest, "transformation": "synthetic-only-export-v1",
                           "source": exp.source, "detail": f"removed {exp.drop_label}; "
                           f"added key {EXPORT_NOTE_KEY}; all other values verified equal",
                           "sha256_original": sha256(plan.root / exp.source),
                           "sha256_release": digest})
        rows.append({"path": exp.dest, "bytes": len(data), "sha256": digest, "group": exp.group,
                     "origin": f"export of {exp.source}", "panels": exp.panels,
                     "mentions": mentions_registry(exp.dest, data)})
    if problems:
        raise BuildError(f"{len(problems)} content problem(s): " + "; ".join(problems[:20]))
    head = git_head(plan.root)
    _write_metadata(plan, out, rows, transforms, sel, head)
    log(f"deposit: {len(rows)} data files ({written} written, {len(rows) - written} already "
        f"identical), {sum(r['bytes'] for r in rows) / 1e6:.1f} MB, "
        f"{len(transforms)} transformed -> {out}")
    log(f"refused (all expected): {len(sel.refused)}; tracked skipped: {sel.skipped_tracked}; "
        f"rendering files not deposited: {dict(sel.skipped_suffix)}")
    return BuildResult(rows, transforms, sel, head)


def _panels_for(src: Source, rel: str) -> str:
    if src.derivation_panels and "/derivation/" in rel and rel.endswith(".csv"):
        return src.derivation_panels
    if src.group == "egfr_factorial_run" and "/trajectories/" in rel:
        return EGFR_PANELS
    return src.panels


def _write_metadata(plan: Plan, out: Path, rows: list[dict], transforms: list[dict],
                    sel: Selection, head: str) -> None:
    (out / "LICENSE.txt").write_text(LICENCE_TEXT)
    with (out / "DEPOSIT_MANIFEST.tsv").open("w") as f:
        f.write("path\tbytes\tsha256\tgroup\torigin\tpanels\n")
        for r in rows:
            f.write(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\t{r['group']}\t{r['origin']}\t"
                    f"{r['panels']}\n")
    with (out / "TRANSFORMATIONS.tsv").open("w") as f:
        f.write("path\ttransformation\tsource\tdetail\tsha256_original\tsha256_release\n")
        for t in transforms:
            f.write(f"{t['path']}\t{t['transformation']}\t{t['source']}\t{t['detail']}\t"
                    f"{t['sha256_original']}\t{t['sha256_release']}\n")
    run_dirs = sorted({"/".join(r["path"].split("/")[:2]) for r in rows
                       if r["path"].startswith("runs/")})
    write_readme(plan, out, rows, transforms, run_provenance(plan.root, run_dirs), sel, head)
    with (out / "SHA256SUMS").open("w") as f:
        for r in rows:
            f.write(f"{r['sha256']}  {r['path']}\n")
        for m in META_FILES:
            f.write(f"{sha256(out / m)}  {m}\n")


def write_readme(plan: Plan, out: Path, rows: list[dict], transforms: list[dict],
                 prov: list[dict], sel: Selection, head: str) -> None:
    by_group: Counter = Counter()
    bytes_group: Counter = Counter()
    for r in rows:
        by_group[r["group"]] += 1
        bytes_group[r["group"]] += r["bytes"]
    total = sum(bytes_group.values())
    n_scrub = sum(SCRUB_RULE in t["transformation"].split("+") for t in transforms)
    n_key = sum(KEY_RULE in t["transformation"].split("+") for t in transforms)
    lines = [
        "# SynthOncCases: synthetic data deposit",
        "",
        f"Built {datetime.now(timezone.utc):%Y-%m-%d} by `scripts/release/build_deposit.py`. "
        "DOI: https://doi.org/10.5281/zenodo.24137513. "
        "Code: https://github.com/julienvibert/llm-oncology-world-models, release v1.0.",
        "",
        "Licence: CC BY 4.0 (`LICENSE.txt`). " + LICENCE_SCOPE,
        "",
        "## What this is",
        "",
        "The deposit holds synthetic patient trajectories generated by large language models, with "
        "the inputs and results needed for the analyses reported in the paper: the six-model "
        "native atlas, the controlled-input experiments (factorial lung-cancer and EGFR "
        "experiment, EGFR checks of GLM-5.2 and Kimi K3, the 12-scenario clinical-reasoning "
        "benchmark) and the result tables that the figures and "
        "the reported numbers read. No file contains patient data. Each run folder holds the run "
        "config, the rendered prompt sent for every attempt (`prompt.txt`), the raw response "
        "(`raw_response.txt`), the parsed and accepted trajectory, rejected attempts with their "
        "failure reasons (`failed_attempt.json`), and, where they exist, the run inventory, the derivation "
        "tables (`derivation/*.csv`; absent for the runs with no accepted future) and the run manifest "
        "(absent for three runs, below). The result "
        "tables were computed by the authors' analysis code from these runs.",
        "",
        "## How to use",
        "",
        "The record also holds the code (`paper_code_release_*.zip`). Download the data files into one "
        "folder that does not contain the code archive: the five `synthonccases_deposit_*.tar.gz` "
        f"archives, `{ARCHIVE_SUMS}`, `SHA256SUMS`, `DEPOSIT_MANIFEST.tsv`, `TRANSFORMATIONS.tsv`, "
        f"`DEPOSIT_README.md` and `LICENSE.txt`. `{ARCHIVE_SUMS}` gives the SHA-256 of each archive and "
        "of the loose metadata files. From the root of the unzipped code:",
        "",
        "```bash",
        "python3 scripts/release/fetch_and_verify.py <folder-with-the-downloaded-files> "
        "--checkout .",
        "```",
        "",
        "This checks the archives, unpacks them into a new folder, checks every file against "
        "`SHA256SUMS`, refuses any path that is tracked in git, and copies the files into "
        "place. Then rebuild the figures (`README.md` of the code release, section 4; "
        "`REPRODUCTION_INDEX.md` lists the inputs of each figure panel). Paths inside the archives are "
        "repository-relative.",
        "",
        "## Contents",
        "",
        f"{len(rows)} data files, {total / 1e6:.1f} MB. `DEPOSIT_MANIFEST.tsv` lists every "
        "file with bytes, SHA-256, group, origin and the figure panels that read it. `SHA256SUMS` lists "
        f"these {len(rows)} files and the {len(META_FILES)} metadata files ("
        + ", ".join(f"`{m}`" for m in META_FILES) + "), which `fetch_and_verify.py` checks but does not "
        f"copy into the checkout: it reports {len(rows) + len(META_FILES)} files verified and {len(rows)} copied.",
        "",
        "| group | archive | files | MB |",
        "|---|---|---|---|",
    ]
    lines += [f"| {g} | `{ARCHIVE_PREFIX}{ARCHIVE_OF_GROUP.get(g, g)}.tar.gz` | {by_group[g]} | "
              f"{bytes_group[g] / 1e6:.1f} |" for g in sorted(by_group)]
    lines += [
        "", "## Changes made to the release copy", "",
        "The original run folders are unedited. The release copy differs from them in three "
        "documented ways; `TRANSFORMATIONS.tsv` lists each changed file with the SHA-256 of the "
        "original and of the release copy.", "",
        f"- `{SCRUB_RULE}` ({n_scrub} file(s)): {SCRUB_RULE_TEXT}. These are run metadata "
        "fields (paths of local inventories, configuration files); no model output changed.",
        f"- `{KEY_RULE}` ({n_key} file(s)): {KEY_RULE_TEXT}. These are transport errors "
        "('key limit exceeded') recorded for failed requests; no model output changed.",
    ]
    for e in plan.exports:
        lines.append(f"- `synthetic-only-export-v1`: `{e.dest}` is `{e.source}` without the "
                     f"{e.drop_label}. Every other value was checked equal to the original. "
                     f"The key `{EXPORT_NOTE_KEY}` says so in the file.")
    lines += ["", "## Provenance of the runs", "",
              "Code commit, start date and counts from each run's `manifest.json` (commits refer to the "
              "development repository). `template ok` = the prompt template at that commit has the "
              "SHA-256 recorded in the manifest.", "",
              "| run | models | commit | started | accepted | failed | template ok |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| `{r['run']}` | {r['models']} | `{r['git_commit']}` | {r['started']} | "
              f"{r['accepted']} | {r['failed']} | {r['template_sha_ok']} |" for r in prov]
    lines += ["", "## Not included, and why", "",
              "Rule: no registry (AACR Project GENIE BPC) or trial (PRIME, Project Data Sphere) "
              "data, and nothing generated from or computed on a real patient record, even as "
              "an aggregate. Those follow the data-use terms and are rebuilt by authorised users "
              "(`README.md` of the code release, section 6). The code release in the same record holds "
              "a few aggregate counts of the real patients (Supplementary Table 3 and the scoring "
              "specifications). Not included in this deposit:", ""]
    lines += ["- Registry data and everything derived from it: AACR Project GENIE BPC records, the "
              "generated futures conditioned on them (the registry experiment), their prompts, query "
              "selections and scores, including aggregate tables.",
              "- PRIME trial data (Project Data Sphere) and the tables derived from them.",
              "- Model runs that the paper does not report. `scripts/release/build_deposit.py` keeps the full list of excluded "
              "folders and refuses to build a deposit that includes any of them."]
    rend = ", ".join(f"{n} {s}" for s, n in sorted(sel.skipped_suffix.items()))
    lines += ["- Prompt templates, schemas and code: in the code release of the same record. The run "
              "configurations and each rendered prompt actually sent "
              "(`prompt.txt`) are deposited.",
              "- Renderings" + (f" ({rend})" if rend else "") + " of the copied output folders: rebuilt "
              "by the figure scripts.", ""]
    mentions = [r["path"] for r in rows if r["mentions"]]
    if mentions:
        lines += ["Files that mention GENIE or PRIME by name only (reviewed: methodological "
                  "text, e.g. 'GENIE-observable mask'; no registry values):", ""]
        lines += [f"- `{m}`" for m in mentions] + [""]
    (out / "DEPOSIT_README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------------------------

def read_manifest(tree: Path) -> list[dict]:
    lines = (tree / "DEPOSIT_MANIFEST.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def package(tree: Path, pkg: Path, mtime: int = 0, compresslevel: int = 6,
            log=print) -> list[dict]:
    """Pack a built deposit tree into one tar.gz per component plus loose metadata.

    Archives are deterministic (sorted members, fixed mtime/owner, gzip mtime 0). Every member
    is checked against ``SHA256SUMS`` while it is packed. Writes ``ARCHIVES_SHA256SUMS``."""
    pkg = pkg.resolve()
    if pkg.exists() and any(pkg.iterdir()):
        raise BuildError(f"package folder {pkg} is not empty")
    sums = {}
    for ln in (tree / "SHA256SUMS").read_text().splitlines():
        d, _, rel = ln.partition("  ")
        sums[rel] = d
    rows = read_manifest(tree)
    if {r["path"] for r in rows} | set(META_FILES) != set(sums):
        raise BuildError("DEPOSIT_MANIFEST.tsv and SHA256SUMS disagree")
    by_archive: dict[str, list[str]] = {}
    for r in rows:
        name = ARCHIVE_OF_GROUP.get(r["group"])
        if name is None:
            raise BuildError(f"no archive for group {r['group']}")
        by_archive.setdefault(name, []).append(r["path"])
    pkg.mkdir(parents=True, exist_ok=True)
    out = []
    for name in sorted(by_archive):
        path = pkg / f"{ARCHIVE_PREFIX}{name}.tar.gz"
        with path.open("wb") as raw, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                              compresslevel=compresslevel) as gz, \
                tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel in sorted(by_archive[name]):
                data = (tree / rel).read_bytes()
                if sha256_bytes(data) != sums[rel]:
                    raise BuildError(f"tree file changed since the build: {rel}")
                info = tarfile.TarInfo(rel)
                info.size, info.mtime, info.mode = len(data), mtime, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
        out.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                    "members": len(by_archive[name])})
        log(f"  {path.name}: {len(by_archive[name])} files, {path.stat().st_size / 1e6:.1f} MB")
    for m in (*META_FILES, "SHA256SUMS"):
        shutil.copy2(tree / m, pkg / m)
        out.append({"file": m, "bytes": (pkg / m).stat().st_size, "sha256": sha256(pkg / m),
                    "members": 0})
    with (pkg / ARCHIVE_SUMS).open("w") as f:
        for o in out:
            f.write(f"{o['sha256']}  {o['file']}\n")
    log(f"package: {len(out) + 1} files -> {pkg}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="deposit tree (outside the repo)")
    ap.add_argument("--list", action="store_true", help="list the selection; copy nothing")
    ap.add_argument("--reuse", action="store_true",
                    help="refresh an earlier tree; files with the right SHA-256 are kept")
    ap.add_argument("--package", type=Path, help="also pack the tree into archives here")
    a = ap.parse_args(argv)
    plan = default_plan()
    try:
        if a.list:
            sel = select(plan, released_code(tracked_paths(plan.root)))
            errors = check_selection(plan, sel)
            for rel, src in sel.chosen:
                print(f"{src.group}\t{rel}")
            for e in plan.exports:
                print(f"{e.group}\t{e.dest}\t(export of {e.source})")
            print(f"{len(sel.chosen) + len(plan.exports)} files; refused {len(sel.refused)}; "
                  f"problems {len(errors)}", file=sys.stderr)
            for e in errors:
                print(f"FAIL {e}", file=sys.stderr)
            return 1 if errors else 0
        build(plan, a.out, reuse=a.reuse)
        if a.package:
            if a.package.resolve().is_relative_to(plan.root.resolve()):
                raise BuildError("--package must be outside the repository")
            package(a.out, a.package)
    except BuildError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
