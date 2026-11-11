#!/usr/bin/env python3
"""Build the code snapshot for editors and reviewers from one commit (local only; nothing is uploaded).

The snapshot holds only the code and inputs needed to reproduce what the paper shows, from the tracked
files of one commit, without git history (so that the historical identifiers of the private repository
are not shared). What is kept is DERIVED, not hand-picked (``closure``):

  1. roots: the script of every panel of ``scripts/figures/layouts.json``; the figure assembly and the
     manuscript build chain; the producing scripts of the numbers currently traced in
     ``manuscript/TRACE_ALL.md`` (script column); the generation, derivation and scoring code of the
     deposited runs and of the registry (four-arm) experiment (``RUN_CODE``); the prompt templates and
     configs named by the deposited runs' manifests and configs;
  2. the static import closure of those roots (ast; repo-local modules found on the script's own folder,
     ``src/`` and every ``sys.path.insert`` the code makes, evaluated statically);
  3. files named by path in the kept code (``ROOT / "a" / "b"`` chains and string literals) and in the
     kept configs and shell scripts;
  4. tests whose repo-local code dependencies are all kept, with the fixtures they read;
  5. ``ROOT_FILES`` (environment, licence) and the release README (``scripts/release/RELEASE_README.md``,
     shipped as ``README.md``).

Files that are needed and hold identifiers are copied with the identifiers masked (``MASK``); any other
needed file that holds identifiers fails the build. The result is scanned with
``scripts/check_identifiers.py --release`` and for credential-like strings; the build fails if anything
is found.

    uv run python scripts/release/build_review_snapshot.py [--commit HEAD] [--out outputs/release_candidates]
    uv run python scripts/release/build_review_snapshot.py --list     # print the kept files and why

Output: <out>/review_snapshot_<commit>.zip and <out>/review_snapshot_<commit>_report.json (the report
lists every kept file with the reason it is kept).
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import functools
import getpass
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_identifiers as C  # noqa: E402

# Top-level groups that may hold released code or inputs. ``keep`` is the coarse filter (also used by
# build_deposit.released_code: a deposit path is never one of these); ``closure`` decides what is shipped.
KEEP_TOP = {"src", "scripts", "tests", "configs", "prompts", "schemas", "disease_cards", "anchor_cells", "data", "docs"}
# manuscript/ is not code: only its files read by the shipped build chain and the cited specifications are shipped
CANDIDATE_TOP = KEEP_TOP | {"manuscript"}
ROOT_FILES = {"pyproject.toml", "uv.lock", "LICENSE", ".python-version"}
RELEASE_README = "scripts/release/RELEASE_README.md"  # shipped as README.md at the snapshot root
MASK = {"scripts/explore/sentinel_benchmark_spec_v1.yaml"}  # needed by the scenario benchmark
# never shipped even when reachable: the git-history guard (the snapshot has no history)
EXTRA_DROP = {"tests/unit/test_no_new_identifiers.py",
              "docs/generation_design_v1.md",  # development notes; no shipped code needs them
              "docs/model_cutoff_policy_v1.md",
              # run notes of the development repository (documentation strings refer to them; no code reads them)
              "docs/run_provenance/interventional_phase1_kimi_k3_off_full_v1c.md",
              "docs/run_provenance/interventional_phase1_kimi_k3_off_topup75_v1c.md",
              "docs/run_provenance/sentinel_benchmark_scoring.md",
              "docs/run_provenance/sentinel_usecase_egfr_post_osi_explicit_resistance_v5_2026_07_13.md"}  # design note of the development repository; no shipped code needs it

# Fixed roots: figure assembly, manuscript build chain, release tooling.
BUILD_CHAIN = (  # figure rebuild and assembly; the manuscript DOCX build is not part of the release (it needs
    # internal planning files: ARTICLE_PLAN, FIGURE_SELECTION, FIGURE_RECOVERY, CLAIMS.md)
    "scripts/figures/assemble.py",
    "scripts/figures/rebuild_panels.py",
    "scripts/check_identifiers.py",
    "scripts/explore/figpanels/fig1d_survival_test.py",  # recomputes the Fig. 1d P values the panel reads
)
RELEASE_TOOLS = ("scripts/release/build_review_snapshot.py", "scripts/release/build_deposit.py",
                 "scripts/release/fetch_and_verify.py", "scripts/release/build_source_data.py",
                 "scripts/release/check_small_cells.py", "scripts/release/mask_small_cells.py")
# Scripts in the trace's script column that are displays, not producers: each is the only reader of internal
# files and the entry also names the scripts that produced the numbers.
TRACE_NOT_PRODUCERS = {
    "scripts/explore/supervisor_meeting_figures_20260921.py":
        "display surface for the 21 September 2026 supervisor meeting; reads the saved four-arm estimates whose "
        "producers (four_arm_*, first_pass_four_arm_v1.py) are listed in the same ledger entry; it also reads the "
        "internal article plan and an old draft",
}
# Analyses the paper does not report: cited only through grouped ledger entries, or replaced by a smaller route.
# They are not roots; a kept file that still needs one fails the build.
REMOVED_FROM_PAPER = {
    "scripts/explore/n10_precision_ladder_arrows_v1.py": "N10 arrow ladder, not reported in the paper",
    "scripts/explore/n12_prime_decomposition_v1.py": "N12 decomposition of PRIME, not reported in the paper",
    "scripts/explore/n12_prime_decomposition_support_v1.py": "support analysis of the N12 decomposition",
    "scripts/figures/panels/n12_prime_three_views.py":
        "drew the N12 panels; the reported Kaplan-Meier panel (ED Fig. 8c) is drawn by n12a_prime_km.py",
}
REMOVED_OUTPUTS = ("outputs/exploratory_n10_precision_ladder", "outputs/exploratory_n12_prime_decomposition")
# Internal manuscript files never shipped; a kept file that needs one fails the build
INTERNAL = ("manuscript/CLAIMS.md", "manuscript/DRAFT_v2_working_*", "manuscript/ARTICLE_PLAN_*",
            "manuscript/FIGURE_SELECTION_*", "manuscript/FIGURE_RECOVERY_*", "manuscript/review/2026-10-28_completion_plan.md",
            "manuscript/FIGURES.md", "manuscript/FIGURES.json", "manuscript/DRAFT.md", "manuscript/DRAFT.docx",
            "manuscript/TRACE.md", "manuscript/TRACE_ALL.md")
TRACE = "manuscript/TRACE_ALL.md"
# Generation, derivation and scoring code of the deposited runs and of the registry experiment (whose data
# are restricted, so the code is released alone). Each entry is a root of the closure.
RUN_CODE = {
    "scripts/run_paper_v1.py": "generation CLI of every deposited run (poc_runner engine)",
    "scripts/build_rev5_derivation.py": "derivation tables (derivation/*.csv) of the deposited runs",
    "scripts/explore/build_sonnet_mini_atlas.py": "config builder of the Sonnet 5 atlas runs",
    "scripts/explore/build_interventional_phase1.py": "config and arm-manifest builder of the EGFR runs",
    "scripts/explore/score_interventional_phase1.py": "scorer of the EGFR runs",
    "scripts/explore/build_kimi_k3_combined.py": "merge and scoring of the two Kimi K3 EGFR-check runs",
    "scripts/explore/build_sentinel_battery.py": "config builder of the 12-scenario benchmark runs",
    "scripts/explore/build_post_osi_resistance_control.py": "config builder of the post-osimertinib scenario runs",
    "scripts/explore/ed6_provenance/rescore.py": "re-scoring of the scenario benchmark (reconstructed spec)",
    "scripts/explore/ed6_provenance/hgsoc_os_sensitivity.py": "overall-survival sensitivity of the platinum contrast",
    "scripts/explore/ed6_provenance/reconstruct_manifest.py": "reconstruction of the scenario-benchmark spec",
    "scripts/explore/build_query_universe_v1.py": "registry experiment: query universe (GENIE-BPC, restricted)",
    "scripts/explore/emit_four_arm_prompts_v1.py": "registry experiment: prompts (GENIE-BPC, restricted)",
    "scripts/explore/preflight_four_arm_v1.py": "registry experiment: preflight",
    "scripts/explore/run_four_arm_v1.py": "registry experiment: generation runner",
    "scripts/score_four_arm_final_v1.py": "registry experiment: final scorer",
}
# panel scripts and helpers that put these folders on sys.path: an import is looked up there too
TEXT = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".cfg", ".ini", ".sh", ".csv", ".tsv", ".html", ".r", ".R"}
# data files named by path are added; a named folder is expanded only below these groups (never code folders)
DATA_TOP = {"configs", "prompts", "schemas", "disease_cards", "anchor_cells", "data", "docs", "manuscript"}
# kept non-Python files whose path references are followed (configs name prompt templates; shell scripts name
# scripts); Markdown and JSON are not followed (notes and ledgers name historical files)
FOLLOW_SUFFIX = {".sh", ".yaml", ".yml"}
FOLLOW_TOP = {"configs", "data", "scripts"}  # specifications and notes elsewhere are kept but not followed
PATHLIKE = re.compile(r"(?<![\w./-])((?:src|scripts|tests|configs|prompts|schemas|disease_cards|anchor_cells|data|docs|"
                      r"manuscript)/[\w./*-]+)")
HOME = str(Path.home())
USER = getpass.getuser()
CREDENTIALS = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|sk-or-v1-[0-9a-f]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,})")


def baseline_paths(commit: str) -> set[str]:
    """Identifier-bearing paths as recorded in the exported commit (not the working tree)."""
    raw = subprocess.check_output(["git", "show", f"{commit}:scripts/identifier_baseline.json"], cwd=ROOT, text=True)
    return set(json.loads(raw)["counts"])


def shape(o):
    """Structure of a parsed YAML document: keys and value types, without values."""
    if isinstance(o, dict):
        return {k: shape(v) for k, v in o.items()}
    if isinstance(o, list):
        return [shape(v) for v in o]
    return type(o).__name__


def keep(path: str, drop: set[str]) -> bool:
    """Coarse filter: could this tracked path belong to the code release? (outputs/ and runs/ never do)."""
    if path in drop:
        return False
    top = path.split("/", 1)[0]
    return top in KEEP_TOP if "/" in path else path in ROOT_FILES


# ---------------------------------------------------------------------------------------------
# static closure
# ---------------------------------------------------------------------------------------------

def _eval(node: ast.AST, env: dict[str, str], file: str) -> str | None:
    """Static value of a path expression (Path(__file__).resolve().parents[2] / "x", str(ROOT), ...)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return file if node.id == "__file__" else env.get(node.id)
    if isinstance(node, ast.NamedExpr):
        v = _eval(node.value, env, file)
        if v is not None:
            env[node.target.id] = v
        return v
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        a, b = _eval(node.left, env, file), _eval(node.right, env, file)
        return os.path.join(a, b) if a is not None and b is not None else None
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            v = _eval(node.value, env, file)
            return os.path.dirname(v) if v is not None else None
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        v = _eval(node.value.value, env, file)
        if v is None or not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return None
        for _ in range(node.slice.value + 1):
            v = os.path.dirname(v)
        return v
    if isinstance(node, ast.Call):
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if name in {"Path", "str", "PurePath", "abspath", "realpath", "fspath"} and len(node.args) == 1:
            return _eval(node.args[0], env, file)
        if name in {"resolve", "absolute", "expanduser"} and isinstance(f, ast.Attribute):
            return _eval(f.value, env, file)
        if name == "dirname" and len(node.args) == 1:
            v = _eval(node.args[0], env, file)
            return os.path.dirname(v) if v is not None else None
        if name in {"join", "joinpath"} and isinstance(f, ast.Attribute):
            parts = ([] if name == "join" else [_eval(f.value, env, file)]) + [_eval(a, env, file) for a in node.args]
            return os.path.join(*parts) if parts and None not in parts else None
    return None


def _module_env(tree: ast.AST, file: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for _ in range(2):  # second pass resolves names assigned from later definitions
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                v = _eval(n.value, env, file)
                if v is not None and n.targets[0].id not in env:
                    env[n.targets[0].id] = v
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                v = _eval(n.value, env, file)
                if v is not None and n.target.id not in env:
                    env[n.target.id] = v
    return env


@functools.lru_cache(maxsize=None)
def _scan_py(path: Path) -> tuple[list[tuple[str, int]], list[str], list[str]]:
    """Imports (module, level), sys.path additions and path-like values of one Python file."""
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return [], [], []
    file = str(path)
    env = _module_env(tree, file)
    imports: list[tuple[str, int]] = []
    syspath: list[str] = []
    values: list[str] = []
    inner = {id(c) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
             for c in (n.left, n.right)}  # parts of a longer path chain are not paths themselves
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
              and re.fullmatch(r"[\w-]+", n.value)]
    for n in ast.walk(tree):
        # DIR / f"{name}.py" (scripts loaded by name): any string constant of the file may be that name
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.right, ast.JoinedStr)
                and n.right.values and isinstance(n.right.values[-1], ast.Constant)):
            d = _eval(n.left, env, file)
            suffix = n.right.values[-1].value
            if d is not None and isinstance(suffix, str) and suffix.startswith("."):
                values += [os.path.join(d, c + suffix) for c in consts]
        if isinstance(n, ast.Import):
            imports += [(a.name, 0) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            imports.append((mod, n.level))
            imports += [(f"{mod}.{a.name}" if mod else a.name, n.level) for a in n.names]  # submodules
        elif isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr in {"insert", "append"} and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "path" and n.args):
                v = _eval(n.args[-1], env, file)
                if v is not None:
                    syspath.append(v)
            if isinstance(f, ast.Attribute) and f.attr == "import_module" and n.args:
                v = _eval(n.args[0], env, file)
                if v:
                    imports.append((v, 0))
        if isinstance(n, (ast.BinOp, ast.Call, ast.Constant)) and id(n) not in inner:
            v = _eval(n, env, file)
            if v and ("/" in v or v.endswith((".py", ".json", ".yaml", ".yml", ".md", ".csv", ".txt"))):
                values.append(v)
    return imports, syspath, values


_DIRS: set[str] = set()  # tracked folders (namespace packages), set by closure()


def _resolve_module(name: str, level: int, file: Path, paths: list[Path], files: set[str], base: Path) -> list[str]:
    """Repo-local files that ``import name`` can load (package __init__ files included)."""
    if level:
        anchor = file.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        search = [anchor]
    else:
        search = paths
    parts = [p for p in name.split(".") if p]
    for d in search:
        hits = []
        cur = d
        for part in parts:
            cur = cur / part
            ri, rm = _rel(cur / "__init__.py", base), _rel(cur.with_suffix(".py"), base)
            if rm in files:  # module (or "from mod import name": the rest are attributes)
                hits.append(rm)
                break
            if ri in files:
                hits.append(ri)
            elif _rel(cur, base) not in _DIRS:  # neither a package nor a namespace folder
                break
        if hits:
            return hits
    return []


def _rel(p: Path, base: Path) -> str | None:
    try:
        return Path(os.path.normpath(p)).relative_to(base).as_posix()
    except ValueError:
        return None


NOT_SHIPPED_HITS: dict[str, str] = {}  # file not shipped -> a kept file that names it


def _path_targets(value: str, file: Path, base: Path, files: set[str], dirs: set[str]) -> list[str]:
    """Tracked files a path-like value names (relative to the root, to the file, or absolute)."""
    cands = [Path(value)] if os.path.isabs(value) else [base / value, file.parent / value]
    out: list[str] = []
    for c in cands:
        r = _rel(c, base)
        if r is None or r.startswith(".."):
            continue
        if "*" in r:
            # only an explicit data-folder glob ("configs/x_*.yaml"); a bare "*.py" or "**/*" names nothing
            if "*" not in r.split("/")[0] and r.split("/", 1)[0] in DATA_TOP and "*" not in r.rsplit("/", 1)[0]:
                out += [f for f in files if fnmatch.fnmatchcase(f, r)]
        elif r in files:
            out.append(r)
        elif r in dirs and r.count("/") >= 1 and r.split("/", 1)[0] in DATA_TOP:
            out += [f for f in files if f.startswith(r + "/")]
        if out:
            break
    hit = [o for o in out if o in NOT_SHIPPED]
    for o in hit:
        NOT_SHIPPED_HITS.setdefault(o, _rel(file, base) or str(file))
    return [o for o in out if o not in NOT_SHIPPED]


def _data_ref(value: str, base: Path) -> str | None:
    """outputs/... or runs/... path a value names (these are deposited data, not code)."""
    v = value
    if os.path.isabs(v):
        r = _rel(Path(v), base)
        if r is None:
            return None
        v = r
    v = v.lstrip("./")
    if v.startswith(("outputs/", "runs/")) and len(v.split("/")) >= 2 and v.split("/")[1]:
        return v
    return None


def trace_scripts(base: Path) -> list[str]:
    """Producing scripts in the script column of the current trace (manuscript/TRACE_ALL.md)."""
    out: list[str] = []
    for ln in (base / TRACE).read_text().splitlines():
        cells = ln.split("|")
        if ln.startswith("|") and len(cells) >= 8:
            out += re.findall(r"`([^`]+\.(?:py|sh))`", cells[-2])
    return sorted(set(out))


def panel_scripts(base: Path) -> dict[str, str]:
    """Script of every panel of layouts.json, resolved as scripts/figures/rebuild_panels.py does."""
    sys.path.insert(0, str(base / "scripts" / "figures"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_rebuild_panels_snapshot", base / "scripts/figures/rebuild_panels.py")
        rb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rb)
    finally:
        sys.path.pop(0)
    layout = json.loads((base / "scripts/figures/layouts.json").read_text())
    out = {}
    for k, v in layout.items():
        if k.startswith("_"):
            continue
        for p in v["panels"]:
            out.setdefault(f"scripts/figures/panels/{rb.script_for(p['id'])[0]}", []).append(f"{k}: {p['id']}")
    return {s: "panel " + "; ".join(ids) for s, ids in out.items()}


def run_inputs(repo: Path, files: set[str]) -> dict[str, str]:
    """Prompt templates and configs named by the deposited runs' manifests and configs (read from the
    local runs/ folders, which are not tracked)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_deposit as B
    out: dict[str, str] = {}
    runs = B.ATLAS_RUNS + [B.EGFR_SONNET_RUN] + B.EGFR_CHECK_RUNS + B.SCENARIO_RUNS
    config_ids = {}
    for f in files:
        if f.startswith("configs/") and f.endswith((".yaml", ".yml")):
            m = re.search(r"^config_id:\s*(\S+)", (repo / f).read_text(errors="ignore"), re.M)
            if m:
                config_ids.setdefault(m.group(1), []).append(f)
    for r in runs:
        d = repo / "runs" / r
        if not d.is_dir():
            continue
        names = []
        if (d / "manifest.json").is_file():
            m = json.loads((d / "manifest.json").read_text())
            names.append(m.get("prompt_template_path"))
        if (d / "config.yaml").is_file():
            c = yaml.safe_load((d / "config.yaml").read_text()) or {}
            names.append(((c.get("prompt") or {}).get("template_path")))
            names += config_ids.get(c.get("config_id"), [])
        for n in names:
            if n and n in files:
                out.setdefault(n, f"named by deposited run {r}")
    return out


# The list of specification files the code cites (the former Supplementary Table 4, removed from the paper).
SPEC_TABLE = "manuscript/review/2026-11-11_methods_rewrite/removed_supplementary_table_4_specifications.md"
SPEC_GLOBS: dict[str, str] = {}  # run notes of the development repository are not shipped
# Files that kept code names but that are not shipped, with the reason. None is needed to rebuild a public result.
NOT_SHIPPED = {
    "manuscript/review/2026-09-30_codex_gate3_amendment.md":
        "development note hashed into the registry scorer's manifest (score_four_arm_final_v1.py); not cited by the "
        "paper; the cited scoring specification and its amendment of 12 October 2026 are shipped",
    "manuscript/genie_anchored_pfs_prereg_v1.yaml":
        "default argument of run_anchored_pfs_smoke_v1.py, whose helpers the registry runner imports; not cited",
    "scripts/identifier_baseline.json": "identifier counts of the development repository (the release scan does not use it)",
    "scripts/scratch/build_anchored_atlas_config.py":
        "historical generator of the atlas configs; the configs and the deposited run configs are shipped",
}


def provenance_files(base: Path, files: set[str]) -> dict[str, str]:
    """Specification files cited by the code (listed in ``SPEC_TABLE``) and the run-provenance notes."""
    out = {}
    if (base / SPEC_TABLE).is_file():
        for ref in re.findall(r"`([^`]+)`", (base / SPEC_TABLE).read_text()):
            if ref in files:
                out[ref] = "specification cited by the code"
    for g, why in SPEC_GLOBS.items():
        for f in sorted(files):
            if fnmatch.fnmatchcase(f, g) and "/" not in f[len(g.split("*")[0]):]:
                out.setdefault(f, why)
    return out


def closure(base: Path, files: set[str], repo: Path | None = None) -> dict:
    """Files of ``files`` (tree-relative, under ``base``) needed to reproduce the paper, with reasons."""
    repo = repo or base
    dirs = {"/".join(f.split("/")[:i]) for f in files for i in range(1, f.count("/") + 1)}
    _DIRS.clear()
    _DIRS.update(dirs)
    reasons: dict[str, str] = {}
    roots: dict[str, str] = {}
    roots.update(panel_scripts(base))
    roots.update({p: "figure assembly and manuscript build chain" for p in BUILD_CHAIN})
    roots.update({p: "release tooling" for p in RELEASE_TOOLS})
    roots.update({p: "producer of numbers traced in TRACE_ALL.md" for p in trace_scripts(base)
                  if p not in roots and p not in TRACE_NOT_PRODUCERS and p not in REMOVED_FROM_PAPER})
    roots.update({p: r for p, r in RUN_CODE.items() if p not in roots})
    roots.update({p: r for p, r in run_inputs(repo, files).items() if p not in roots})
    trace_missing = sorted(p for p in trace_scripts(base) if p not in files)
    missing_roots = sorted(p for p in roots if p not in files and p not in trace_missing)
    src = base / "src"
    work: list[tuple[str, tuple]] = []
    for p, why in roots.items():
        if p in files:
            reasons[p] = why
            work.append((p, ()))
    seen: dict[str, frozenset[str]] = {}
    local_tops = {f.split("/")[-1].removesuffix(".py") for f in files if f.endswith(".py")} | {
        d.split("/")[-1] for d in dirs}
    # sys.path is process-wide: a folder one kept module inserts is searched for every kept module (fixpoint)
    gpath: set[str] = set()
    edges: dict[str, set[str]] = {}
    datarefs: dict[str, set[str]] = {}
    unresolved: dict[str, list[str]] = {}
    while True:
        while work:
            rel, _ = work.pop()
            g = frozenset(gpath)
            if seen.get(rel) == g:
                continue
            seen[rel] = g
            path = base / rel
            if rel.endswith(".py"):
                imports, syspath, values = _scan_py(path)
                if rel == "scripts/release/build_review_snapshot.py":
                    values = []  # its own path constants are the rules above, not inputs
                gpath.update(syspath)
                paths = [path.parent, *map(Path, syspath), *map(Path, sorted(gpath)), src]
                unresolved.pop(rel, None)
                for name, level in imports:
                    hits = _resolve_module(name, level, path, paths, files, base)
                    for h in hits:
                        reasons.setdefault(h, f"imported by {rel}")
                        edges.setdefault(rel, set()).add(h)
                        work.append((h, ()))
                    if (not hits and not level and name in local_tops and "." not in name
                            and name not in sys.stdlib_module_names):
                        unresolved.setdefault(rel, []).append(name)
                for v in values:
                    d = _data_ref(v, base)
                    if d:
                        datarefs.setdefault(rel, set()).add(d)
                    for t in _path_targets(v, path, base, files, dirs):
                        if t != rel:
                            reasons.setdefault(t, f"named by path in {rel}")
                            if t.endswith(".py"):
                                edges.setdefault(rel, set()).add(t)
                            if t.endswith((".py", *FOLLOW_SUFFIX)):
                                work.append((t, ()))
            elif path.suffix in FOLLOW_SUFFIX and rel.split("/", 1)[0] in FOLLOW_TOP:
                for v in PATHLIKE.findall(path.read_text(errors="ignore")):
                    for t in _path_targets(v.rstrip("."), path, base, files, dirs):
                        if t != rel:
                            reasons.setdefault(t, f"named by path in {rel}")
                            if t.endswith((".py", *FOLLOW_SUFFIX)):
                                work.append((t, ()))
        # a manuscript JSON read by kept code may name further manuscript files it loads (one level, not followed)
        for rel in [r for r in reasons if r.startswith("manuscript/") and r.endswith(".json")]:
            for v in PATHLIKE.findall((base / rel).read_text(errors="ignore")):
                if v.startswith("manuscript/") and v.endswith(".json") and v in files:
                    reasons.setdefault(v, f"named by {rel}")
        stale = [r for r in unresolved if seen.get(r) != frozenset(gpath)]
        if not stale:
            break
        work += [(r, ()) for r in stale]
    not_shipped = {k: f"named by {v}; not shipped: {NOT_SHIPPED[k]}" for k, v in NOT_SHIPPED_HITS.items()}
    # dated specifications and provenance notes: kept at the paths the manuscript cites, not followed
    for p, why in provenance_files(base, files).items():
        reasons.setdefault(p, why)
    # package __init__ files of every kept module under src/
    for k in list(reasons):
        if k.startswith("src/") and k.endswith(".py"):
            parts = k.split("/")
            for i in range(2, len(parts)):
                init = "/".join(parts[:i]) + "/__init__.py"
                if init in files:
                    reasons.setdefault(init, f"package of {k}")
    code = {k for k in reasons if k.endswith(".py")}
    # tests whose repo-local code dependencies are all kept
    tests_kept: dict[str, str] = {}
    tests_dropped: dict[str, list[str]] = {}
    for t in sorted(f for f in files if f.startswith("tests/") and f.endswith(".py") and "/test_" in "/" + f):
        deps, extra, stack, done = set(), set(), [t], set()
        while stack:
            r = stack.pop()
            if r in done:
                continue
            done.add(r)
            imports, syspath, values = _scan_py(base / r)
            paths = [(base / r).parent, *map(Path, syspath), src, base]
            for name, level in imports:
                for h in _resolve_module(name, level, base / r, paths, files, base):
                    (stack.append(h) if h.startswith("tests/") else deps.add(h))
            for v in values:
                for x in _path_targets(v, base / r, base, files, dirs):
                    if x.startswith("tests/"):
                        extra.add(x)
                        if x.endswith(".py"):
                            stack.append(x)
                    elif x.endswith(".py"):
                        deps.add(x)
                    else:
                        extra.add(("data", x))
        bad = sorted(d for d in deps if d not in code)
        # a test whose inputs (configs, prompts, data) only tests would ship is left out with them
        bad += sorted(x[1] for x in extra if isinstance(x, tuple) and x[1] not in reasons)
        # or that reads a file that is not shipped
        bad += sorted(n for n in NOT_SHIPPED for r in done if Path(n).name in _py_text(base, r))
        if bad:
            tests_dropped[t] = bad
            continue
        tests_kept[t] = "test of kept code"
        for r in done:
            if r != t:
                tests_kept.setdefault(r, f"used by {t}")
        for x in extra:
            if isinstance(x, tuple):
                tests_kept.setdefault(x[1], f"read by {t}")
            else:
                tests_kept.setdefault(x, f"read by {t}")
    for t, why in tests_kept.items():
        reasons.setdefault(t, why)
    for c in ("tests/__init__.py", "tests/unit/__init__.py", "tests/conftest.py", "tests/unit/conftest.py",
              "tests/fixtures/__init__.py"):
        if c in files and tests_kept:
            reasons.setdefault(c, "test package")
    for f in ROOT_FILES:
        if f in files:
            reasons.setdefault(f, "environment and licence")
    if RELEASE_README in files:
        reasons[RELEASE_README] = "release README (shipped as README.md)"
    for f in EXTRA_DROP:
        reasons.pop(f, None)
    internal_needed = {k: reasons.pop(k) for k in list(reasons) if any(fnmatch.fnmatchcase(k, g) for g in INTERNAL)}
    internal_needed.update({k: reasons.pop(k) for k in list(reasons) if k in REMOVED_FROM_PAPER})
    # a test naming an internal file in a fixture string, or a release tool that reads it only when present in the
    # development repository, does not need it in the release
    internal_needed = {k: v for k, v in internal_needed.items()
                       if not v.startswith(("read by tests/", "named by path in scripts/release/"))}
    return {"kept": reasons, "internal_needed": internal_needed, "not_shipped": not_shipped, "missing_roots": missing_roots, "trace_scripts_missing": trace_missing, "unresolved_imports": unresolved,
            "tests_dropped": tests_dropped, "edges": edges, "datarefs": datarefs, "roots": roots}


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# generated release documents: reproduction index and generation provenance
# ---------------------------------------------------------------------------------------------

BOUNDARY = ("manuscript/DRAFT.md", "manuscript/SUPPLEMENTARY_FIGURES.md", "manuscript/supplementary_tables",
            "scripts/figures/layouts.json")
# panels whose inputs are registry-derived but that scripts/figures/rebuild_panels.py does not list as restricted
RESTRICTED_EXTRA = {"n14_support_gradient": "donor support gradient of the registry experiment (GENIE-BPC)"}
FIGURE_NAMES = ([f"Figure {i}" for i in range(1, 7)] + [f"Extended Data Figure {i}" for i in range(1, 10)]
                + [f"Supplementary Figure {i}" for i in range(1, 17)])


def _deposit_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_deposit as B
    return B


def trace_rows(base: Path) -> list[dict]:
    rows = []
    for ln in (base / TRACE).read_text().splitlines():
        c = [x.strip() for x in ln.split(" | ")]
        if not ln.startswith("| ") or len(c) != 6 or c[0] in {"| section", "|---"} or c[0].startswith("|---"):
            continue
        rows.append({"section": c[0].removeprefix("| "), "claim": c[1].strip("*"), "ledger": c[3],
                     "outputs": re.findall(r"`([^`]+)`", c[4]), "scripts": re.findall(r"`([^`]+)`", c[5].rstrip(" |"))})
    return rows


def reachable(root: str, edges: dict[str, set[str]]) -> set[str]:
    seen, stack = set(), [root]
    while stack:
        r = stack.pop()
        if r not in seen:
            seen.add(r)
            stack += sorted(edges.get(r, ()))
    return seen


def _is_restricted(paths: list[str], B) -> bool:
    return any(B.RESTRICTED_PATH.search(p) or B.matches(p.rstrip("/*"), B.EXPECTED_EXCLUSIONS + B.REGISTRY_DERIVED) for p in paths)


def boundary_versions(commit: str) -> dict[str, str]:
    """Last commit (at the build commit) of each file that sets the inclusion boundary."""
    out = {}
    for p in BOUNDARY:
        out[p] = subprocess.check_output(["git", "log", "-1", "--format=%h %cs", commit, "--", p], cwd=ROOT,
                                         text=True).strip()
    return out


def _observed_inputs(reads: list[str]) -> list[str]:
    """Deposited inputs among the files a script opened in the isolated rebuild (outputs/ and runs/, not its
    own outputs/figures/ products); files of one run are shown as the run folder."""
    out = set()
    for r in reads:
        if r.startswith("runs/"):
            out.add("/".join(r.split("/")[:2]) + "/")
        elif r.startswith("outputs/") and not r.startswith("outputs/figures/"):
            out.add(r)
    return sorted(out)


def reproduction_index(base: Path, res: dict, commit: str, reads: dict | None = None,
                       resolved: list[dict] | None = None) -> str:
    """Display items, Supplementary Tables and text-only numbers -> scripts -> inputs -> public/restricted."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rb_index", base / "scripts/figures/rebuild_panels.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    layout = json.loads((base / "scripts/figures/layouts.json").read_text())
    if [k for k in layout if not k.startswith("_")] != FIGURE_NAMES:
        raise SystemExit("layouts.json figure names differ from the manuscript boundary (FIGURE_NAMES)")
    edges, refs = res["edges"], res["datarefs"]
    L = ["# Reproduction index", "",
         f"Generated by `scripts/release/build_review_snapshot.py` from commit `{commit}`. Inclusion boundary "
         "(the manuscript version this index follows; last commit of each file at the build commit):", ""]
    L += [f"- `{p}`: {v}" for p, v in boundary_versions(commit).items()]
    L += ["", "Status: **public** = rebuilds from this code and the synthetic Zenodo deposit; **restricted** = needs "
          "AACR Project GENIE BPC or PRIME (Project Data Sphere) data, or tables derived from them, which are not "
          "redistributed. Without them, a restricted script stops when it opens its first missing input (usually "
          "`FileNotFoundError` naming a `~/GenieBPC`, `~/PDS` or `outputs/...` path): this means *restricted input "
          "unavailable*, not a defect. `scripts/figures/rebuild_panels.py --public` skips these panels and says so.",
          "", ("Inputs are the deposited files each script opened when the panels were rebuilt from this release "
               "and the deposit in an isolated folder (" + reads["_meta"]["date"] + "); restricted scripts show what "
               "they opened before stopping." if reads else
               "Inputs are the `outputs/` and `runs/` paths named in the script and the modules it imports "
               "(static reading of the code)."), "",
          "## Figures, Extended Data and Supplementary Figures", "",
          "Route: **(a) analysis rerun** = the result is recomputed from the deposited raw runs (their responses or "
          "the derivation tables built from them) by a shipped script; **(b) rendered from deposited results** = the "
          "panel or number is drawn or read from deposited result tables (or is a constructed value); **(c) restricted "
          "data required**. For panels, (a) and (b) follow the files the panel script actually opened in the isolated "
          "rebuild, which ran every public panel. For numbers, (a) means the producing script reads deposited runs; "
          "those scripts were not rerun in the isolated check unless stated.", "",
          "ROUTE_SUMMARY", "",
          "Panel identifiers (such as `cand_*` or `prop_*`) are file names of the panel scripts and outputs, kept for "
          "traceability; they carry no status. The figure and panel letter in the first columns are the published "
          "ones.", "",
          "| display item | panel | script | inputs (outputs/, runs/) | status | route |", "|---|---|---|---|---|---|"]
    counts: dict[str, dict[str, int]] = {}

    def tally(kind: str, route: str) -> str:
        counts.setdefault(kind, {})
        counts[kind][route.split(":")[0]] = counts[kind].get(route.split(":")[0], 0) + 1
        return route
    for name in FIGURE_NAMES:
        for p in layout[name]["panels"]:
            script, args = rb.script_for(p["id"])
            rel = f"scripts/figures/panels/{script}"
            key = f"{script} {' '.join(args)}".strip()
            if reads and key in reads:
                ins = _observed_inputs(reads[key]["reads"])
                if not reads[key]["ok"]:
                    ins = ins + ["(stopped: restricted input unavailable)"
                                 if p["id"] in rb.RESTRICTED or p["id"] in RESTRICTED_EXTRA
                                 else "(FAILED in the isolated rebuild)"]
            else:
                ins = sorted({d for f in reachable(rel, edges) for d in refs.get(f, ())
                              if not d.startswith("outputs/figures/")})
            status = ("restricted" if p["id"] in rb.RESTRICTED else
                      "restricted (not yet in rebuild_panels.RESTRICTED)" if p["id"] in RESTRICTED_EXTRA else "public")
            if status != "public":
                route = "(c) restricted data required"
            elif any(i.startswith("runs/") for i in ins):
                route = ("(a) analysis rerun: recomputed from deposited runs by the panel script; run in the isolated "
                         "check" if reads else "(a) analysis rerun: route documented, not rerun in the isolated check")
            else:
                route = ("(b) rendered from deposited results" if ins else
                         "(b) rendered from deposited results: constructed values, no data input")
            L.append(f"| {name} | {p.get('label', '-')} `{p['id']}` | `{rel}`{' ' + ' '.join(args) if args else ''} | "
                     f"{'; '.join(f'`{i}`' for i in ins) or 'none (constructed or hard-coded values)'} | {status} | "
                     f"{tally('panels', route)} |")
    def block(title: str, pred) -> list[str]:
        groups: dict[tuple, dict] = {}
        for r in resolved:
            if not pred(r["section"]):
                continue
            sec = r["section"].split(":")[0] if title.startswith("Supp") else r["section"]
            key = (sec, tuple(r["sources"]), tuple(r["producers"]), route_status(r), r["restricted"])
            groups.setdefault(key, {"claims": []})["claims"].append(r["claim"])
        out = ["", f"## {title}", "", "| location | numbers | source(s) | producing script(s) | source status | data | "
               "route |", "|---|---|---|---|---|---|---|"]
        for (sec, srcs, prods, rs, restricted), g in sorted(groups.items(), key=lambda kv: kv[0][:3]):
            reads_runs = any(d.startswith("runs/") for sc in prods for f in reachable(sc, edges) for d in refs.get(f, ()))
            if restricted:
                route = "(c) restricted data required"
            elif reads_runs:
                route = "(a) analysis rerun: route documented, not rerun in the isolated check"
            elif srcs:
                route = "(b) rendered from deposited results: read from the deposited output(s)"
            else:
                route = "(b) rendered from deposited results: constructed or definitional value"
            nums = ", ".join(g["claims"][:12]) + (f" (+{len(g['claims']) - 12})" if len(g["claims"]) > 12 else "")
            out.append(f"| {sec} | {nums} | {'; '.join(f'`{x}`' for x in srcs) or '-'} | "
                       f"{'; '.join(f'`{x}`' for x in prods) or '-'} | {rs.split(':')[0]} | "
                       f"{'restricted' if restricted else 'public'} | {route} |")
            k = route.split(":")[0]
            counts.setdefault("numbers", {})
            counts["numbers"][k] = counts["numbers"].get(k, 0) + len(g["claims"])
            counts.setdefault("number sources", {})
            counts["number sources"][rs.split(":")[0]] = counts["number sources"].get(rs.split(":")[0], 0) + len(g["claims"])
        return out
    L += ["", "## Numbers of the text, legends and tables", "",
          "This index maps figure panels only. The scripts that produce the numbers of the text, the legends and the "
          "Supplementary Tables are included in the release; the Methods describe each analysis. No automated "
          "number-to-source table is provided, because file-name matching could not attribute each number to its "
          "producing script reliably."]
    if res.get("trace_scripts_missing"):
        L += ["", "Scripts named in the trace that do not exist at this commit (stale ledger entries): "
              + ", ".join(f"`{x}`" for x in res["trace_scripts_missing"]) + "."]
    summary = ["Counts by route:", ""] + [f"- {kind}: " + "; ".join(f"{k} {v}" for k, v in sorted(c.items()))
                                          for kind, c in counts.items()]
    ROUTE_COUNTS.clear()
    ROUTE_COUNTS.update(counts)
    return "\n".join(("\n".join(summary) if x == "ROUTE_SUMMARY" else x) for x in L) + "\n"


ROUTE_COUNTS: dict[str, dict[str, int]] = {}


NA = "not applicable"
# Definitions checked by reading the sentence, the figure legend and the producing output (10 November 2026).
# Key: (location prefix, substring of the tracer's context). Value: (denominator, horizon, uncertainty, where the
# definition is stated). A design count, label or constructed value gets NA in all three columns.
CURATED = [
    ("Main", "By only specifying tumor type",
     "generated patients within each of 148 cancer types, per model (4,434 trajectories, six models); five baseline "
     "features", NA + " (baseline at diagnosis)",
     "range of the six model means; Fig. 1b: 95% bootstrap intervals over cancer types", "Fig. 1b legend"),
    ("Main", "Models tended to emit",
     "generated patients within each of 148 cancer types, per model; five baseline features", NA + " (baseline)",
     "range of the six model means; Fig. 1b: 95% bootstrap intervals over cancer types", "Fig. 1b legend"),
    ("Main", "systematically varying patient features in 12",
     "48 model-scenario results (12 scenarios x 4 models): 29 of 36 where a biomarker or treatment class decides; "
     "1 of 12 where previous treatment or frailty decides", "the endpoint of each scenario (Supplementary Table 2)",
     "none (counts)", "Results; Extended Data Fig. 4a,b; outputs/exploratory_sentinel_benchmark/sentinel_trust_map_table.csv"),
    ("Main", "Across the 12 scenarios",
     "36 and 12 model-scenario results (12 scenarios x 4 models)", "the endpoint of each scenario (Supplementary Table 2)",
     "none (counts)", "Extended Data Fig. 4a,b; sentinel_trust_map_table.csv"),
    ("Main", "Continuing real patient",
     "449 GENIE-BPC patients with all seven forecasters", "12 months from the start of the forecast treatment",
     "point estimates; observed survival censoring-weighted; Fig. 4d: 95% bootstrap intervals over patients",
     "Fig. 4d legend"),
    ("Main", "context and examples modestly reduced",
     "449 GENIE-BPC patients with all seven forecasters", "12 months from the start of the forecast treatment",
     "point estimates; observed survival censoring-weighted; Fig. 4d: 95% bootstrap intervals over patients",
     "Fig. 4d legend; Extended Data Fig. 5a,b"),
    ("Main", "We obtained 4,434", NA, NA, NA, "design count"),
    ("Main", "Yet heterogeneity between models", "pairs of the six models (similarity of cancer types between models)", NA,
     "range over model pairs; no interval", "Results; Supplementary Fig. 5"),
    ("Main", "Survival also depended on the model",
     "generated lung cancer patients of each model (n per model in Fig. 1d)", "median survival from diagnosis",
     "difference between model medians; Fig. 1d: 95% bootstrap intervals, 2,000 resamples", "Fig. 1d legend"),
    ("Main", "In this study we chose 20 variables",
     "20 state variables; variance shares of the principal components of the fitted state map", NA, "none",
     "Results; outputs/paper_state_map/manifest.json"),
    ("Main", "We built 12 scenarios", NA, NA, NA, "design count"),
    ("Main", "In a factorial experiment", NA, NA, NA, "design count"),
    ("Main", "spent on average 6.8 more months",
     "EGFR-mutant, 45 Claude Sonnet 5 futures per arm from 15 starting records", "first 24 months",
     "95% bootstrap interval over starting records", "Fig. 3a legend"),
    ("Main", "EGFR-mutant futures started on chemotherapy",
     "45 EGFR-mutant chemotherapy futures (Claude Sonnet 5); medians among futures without an early switch",
     "switch by month 12; median first-line progression-free survival", "none (counts and medians)",
     "Results; Methods; Fig. 3a"),
    ("Main", "In a three-model EGFR experiment",
     "EGFR-negative futures, 45 per arm and model (GLM-5.2 chemotherapy 42)",
     "median time before first-line progression",
     "point differences in the text; Fig. 3f: 95% bootstrap intervals over starting records (4,000 resamples)",
     "Fig. 3e,f legend"),
    ("Main", "We applied the four settings to 450", NA, NA, NA, "design count"),
    ("Main", "With the record alone, the LLM was overoptimistic",
     "observed: the GENIE-BPC patients of Fig. 4c; generated: accepted futures per setting (Methods: 4,407 to 4,426)",
     "deaths within 6 months of the start of the forecast treatment",
     "Fig. 4c: Wilson 95% interval (observed), 95% bootstrap intervals over patients (generated)", "Fig. 4c legend"),
    ("Main", "predictions with generic or matched examples fell below",
     "GENIE-BPC patients with all seven forecasters (449; 447 at 24 months)", "24 months from the start of the "
     "forecast treatment", "point estimates; observed survival censoring-weighted", "Extended Data Fig. 5a legend"),
    ("Main", "Among the 260 patients who died",
     "260 GENIE-BPC patients who died", "observed survival time of each patient",
     "none (share of patients whose survival lies between the 10th and 90th percentiles of their futures)",
     "Results; Extended Data Fig. 5c"),
    ("Main", "Matched examples lowered the Brier score",
     "449 GENIE-BPC patients with all seven forecasters", "survival at 12 months",
     "mean paired difference with 95% bootstrap interval over patients; unadjusted two-sided Wilcoxon signed-rank P",
     "Fig. 4d legend; Methods"),
    ("Main", "A Cox model cross-validated",
     "449 GENIE-BPC patients with all seven forecasters", "Brier score at 12 months; C-index (Extended Data Fig. 5e)",
     "point estimates; paired differences reported as inconclusive", "Fig. 4d legend; Extended Data Fig. 5e"),
    ("Main", "Month by month, generated futures",
     "437 GENIE-BPC patients; shares among patients dead or under follow-up each month", "months 6 to 12",
     "none (shares)", "Fig. 5a legend"),
    ("Main", "To visualize these courses", NA, NA, NA, "selection count (109 of 150 lung cancer patients)"),
    ("Main", "Generated and recorded courses disagreed",
     "evaluable GENIE-BPC patients at the horizon (Extended Data Fig. 7a,b)", "12 months", "none in the text",
     "Results; Extended Data Fig. 7a,b"),
    ("Main", "The LLM also over-predicted documented systemic",
     "317 evaluable patients (months on therapy, Fig. 5b); 233 patients (new regimen before progression)",
     "first 12 months", "none (means and shares averaged over each patient's futures)",
     "Fig. 5b legend; Extended Data Fig. 7c,e"),
    ("Main", "A score combining 17 clinical summaries",
     "233 patients at 12 months, 181 at 24 months (Fig. 5d: 286, 269, 233, 181 at four horizons), all three cancers",
     "12 and 24 months",
     "mean paired difference, 95% bootstrap intervals (2,000 paired resamples stratified by cancer); two-sided "
     "Wilcoxon signed-rank P, Holm-adjusted across 16 contrasts", "Fig. 5d legend"),
    ("Main", "Randomized trials usually collect", NA, NA, NA, "design count (935 randomized patients, 14 features)"),
    ("Main", "consistent in direction with the published findings",
     "PRIME patients with known KRAS exon 2 status, by arm (wild-type 260 and 254; mutant 176 and 176)",
     "first 24 months after randomization", "95% bootstrap intervals, 2,000 resamples within arm",
     "Fig. 6a legend; Extended Data Fig. 8c"),
    ("Main", "Beside Kaplan-Meier curves", NA, NA, NA, "selection count (439 patients assessed by day 14)"),
    ("Main", "In the generated atlas, the median survival of comparators",
     "decisions of the generated atlas (Supplementary Fig. 15)", "survival (median absolute error, months)",
     "range over 10 random draws of comparators", "Results; Supplementary Fig. 15"),
    ("Main", "The median number of comparable patients",
     "one decision per GENIE-BPC patient: lung 940, colorectal 1,021, breast 779", NA,
     "medians (interquartile ranges in Fig. 6d)", "Fig. 6d legend"),
    ("Main", "Missing documentation also reduced support",
     "one decision per GENIE-BPC patient: lung 940, colorectal 1,021, breast 779", NA, "none (shares)", "Fig. 6d legend"),
    # principal figure comparisons: the legend numbers are design counts or the definitions themselves
    ("Figure 3 legend", "", NA, NA, NA, "Fig. 3 legend: design counts and definitions (futures per arm, resamples)"),
    ("Figure 4 legend", "", NA, NA, NA, "Fig. 4 legend: design counts and definitions"),
    ("Figure 5 legend", "", NA, NA, NA, "Fig. 5 legend: design counts, denominators and definitions"),
    ("Figure 6 legend", "", NA, NA, NA, "Fig. 6 legend: design counts, denominators and definitions"),
]


def _full_sentence(prefix: str, corpus: str) -> tuple[str, bool]:
    i = corpus.find(prefix)
    if i < 0:
        return prefix, True
    m = re.match(r".*?[.!?](?=\s|$)", corpus[i:i + 400])
    return (m.group(0) if m else prefix).strip(), not m


# ---------------------------------------------------------------------------------------------
# claim resolution: from the ledger's candidate sources to the output that holds each number
# ---------------------------------------------------------------------------------------------

# Files whose folder name suggests registry data but whose content is synthetic (checked by reading the producer)
SYNTHETIC_OVERRIDES = {
    "outputs/exploratory_cand_css_four_arm/crossover_numbers.csv":
        "EGFR experiment (Claude Sonnet 5 futures), written by scripts/explore/cand_css_crossover_numbers.py",
}
# Files that hold synthetic and registry rows: the status of a number follows the row that holds it
MIXED_FILES = {"outputs/exploratory_cand_css_four_arm/text_numbers.csv"}
REGISTRY_ROW = re.compile(r"NSCLC|CRC|BRCA|BrCa|observed|patient_only|patients_like_me|GENIE|PRIME|registry", re.I)
WRITE_CALL = re.compile(r"\.to_csv\(|write_text\(|json\.dump\(|np\.savez|savez_compressed|\.to_json\(|open\([^)]*['\"]w")
NUM_IN_FILE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
MAX_SCAN = 40_000_000


@functools.lru_cache(maxsize=None)
def _file_rows(path: str) -> tuple[str, ...]:
    p = Path(path)
    try:
        if p.stat().st_size > MAX_SCAN:
            return ()
        data = p.read_bytes()
        if p.name.endswith(".gz"):
            import gzip
            data = gzip.decompress(data)
        return tuple(data.decode("utf-8", errors="ignore").splitlines())
    except OSError:
        return ()


def _token_value(tok: str) -> tuple[float, int, bool] | None:
    t = tok.replace(",", "")
    pct = t.endswith("%")
    t = t.rstrip("%")
    try:
        v = float(t)
    except ValueError:
        return None
    dp = len(t.split(".")[1]) if "." in t else 0
    return v, dp, pct


def _checkable(tok: str) -> bool:
    """Small integers (counts below 100 without decimals or %) occur everywhere; a value search proves nothing."""
    tv = _token_value(tok)
    return bool(tv) and (tv[1] > 0 or tv[2] or tv[0] >= 100)


def _rows_with_value(path: str, tok: str) -> list[str]:
    tv = _token_value(tok)
    if not tv:
        return []
    v, dp, pct = tv
    hits = []
    for row in _file_rows(path):
        for m in NUM_IN_FILE.finditer(row):
            try:
                x = abs(float(m.group(0)))
            except ValueError:
                continue
            cands = [x] + ([100 * x] if pct or dp else [])
            if any(round(c, dp) == v for c in cands):
                hits.append(row)
                break
    return hits


def _expand(repo: Path, out: str) -> list[str]:
    if "<" in out:
        return []
    if "*" in out:
        return sorted(p.relative_to(repo).as_posix() for p in repo.glob(out) if p.is_file())
    p = repo / out
    if p.is_dir():
        return sorted(q.relative_to(repo).as_posix() for q in p.rglob("*") if q.is_file()
                      and q.suffix in {".csv", ".json", ".tsv", ".md", ".txt", ".gz"})
    return [out] if p.is_file() else []


@functools.lru_cache(maxsize=None)
def _writers(repo: Path, rel: str) -> tuple[str, ...]:
    """Tracked scripts that name the file and write files: the producers of ``rel``."""
    name = Path(rel).name
    folder = Path(rel).parent.name
    out = []
    for f in _tracked_py(repo):
        t = _py_text(repo, f)
        if name in t and folder in t and WRITE_CALL.search(t) and not f.startswith(
                ("tests/", "scripts/figures/panels/", "scripts/release/", "scripts/figures_codex/", "scripts/scratch/")):
            out.append(f)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _tracked_py(repo: Path) -> tuple[str, ...]:
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=repo, capture_output=True, text=True)
    return tuple(r.stdout.split())


@functools.lru_cache(maxsize=None)
def _py_text(repo: Path, f: str) -> str:
    try:
        return (repo / f).read_text(errors="ignore")
    except OSError:
        return ""


def _restricted_file(rel: str, rows: list[str], B) -> bool:
    if rel in SYNTHETIC_OVERRIDES:
        return False
    if rel in MIXED_FILES:
        return any(REGISTRY_ROW.search(r) for r in rows)
    return _is_restricted([rel], B)


def resolve_claims(base: Path, repo: Path = ROOT, kept: set[str] | None = None) -> list[dict]:
    """Every traced number with the output that holds it and its producer.

    The ledger gives candidate outputs and scripts per entry; an entry can group several results. For each number the
    candidates are narrowed to the cited outputs whose content holds the value (exact, or rounded to the printed
    decimals; percentages also as fractions), and the producer to the script that writes that output. A value search
    locates a source; it does not prove identity (a value can recur), so the method of each route is recorded."""
    import importlib.util
    B = _deposit_module()
    spec = importlib.util.spec_from_file_location("_trace_claims_release", base / "scripts/trace_claims.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    rows, unsourced = tc.trace_all(tc.DRAFT.read_text(), tc.CLAIMS.read_text(), sorted(tc.SUPP_TABLES.glob("*.md")),
                                   tc.SUPP_FIGS.read_text() if tc.SUPP_FIGS.exists() else "")
    if unsourced:
        raise SystemExit(f"{len(unsourced)} unsourced numbers in the manuscript; claim sources not built")
    corpus = " ".join([tc.DRAFT.read_text(), *(t.read_text() for t in sorted(tc.SUPP_TABLES.glob("*.md"))),
                       tc.SUPP_FIGS.read_text() if tc.SUPP_FIGS.exists() else ""]).replace("\n", " ").replace("|", "/")
    out = []
    for r in rows:
        # notes and manuscript files are not result sources
        outs = [o for o in re.findall(r"`([^`]+)`", r["outputs"])
                if not o.startswith(("manuscript/", *REMOVED_OUTPUTS)) and not o.endswith(("FINDINGS.md", "_handover.md"))]
        scr = [x for x in re.findall(r"`([^`]+)`", r["scripts"])
               if x not in TRACE_NOT_PRODUCERS and x not in REMOVED_FROM_PAPER]
        context, cut = _full_sentence(r["context"], corpus)
        rec = {"section": r["section"], "scope": r["scope"], "claim": r["claim"], "context": context, "cut": cut,
               "ledger": r["ledger"],  # internal heading: used for the worklist, never printed in shipped files
               "ledger_outputs": outs, "ledger_scripts": scr}
        files = [f for o in outs for f in _expand(repo, o)]
        if not outs and not scr:
            rec.update(method="constructed or definitional value (no data source)", sources=[], producers=[],
                       restricted=False)
        elif not _checkable(r["claim"]):
            rec.update(method="count below 100: not located by value; candidate sources of the ledger entry",
                       sources=outs, producers=scr, restricted=_is_restricted(outs + scr, B))
        else:
            located = {f: _rows_with_value(str(repo / f), r["claim"]) for f in files}
            located = {f: v for f, v in located.items() if v}
            if located:
                prods = sorted({w for f in located for w in _writers(repo, f)} - set(TRACE_NOT_PRODUCERS)
                               - set(REMOVED_FROM_PAPER))
                in_ledger = [p for p in prods if p in scr]
                prods = in_ledger or [p for p in prods if kept is None or p in kept] or (scr if len(scr) == 1 else [])
                rec.update(sources=sorted(located), producers=prods,
                           method=("value located in the source; producer writes it" if prods else
                                   "value located in the source; producer not identified"),
                           restricted=any(_restricted_file(f, v, B) for f, v in located.items())
                           or (not prods and _is_restricted(scr, B)))
            else:
                rec.update(sources=outs, producers=scr,
                           method="value not located in the cited outputs; candidate sources of the ledger entry",
                           restricted=_is_restricted(outs + scr, B))
        out.append(rec)
    return out


def _sig_digits(tok: str) -> int:
    return len(tok.replace(",", "").replace("%", "").replace(".", "").lstrip("0"))


def route_status(r: dict) -> str:
    """'located' needs one cited output holding the value, one writer, and a value of at least three significant
    digits (shorter values recur too often for a value search to single out a source)."""
    if not r["sources"] and not r["producers"]:
        return "not applicable (constructed or definitional value)"
    if (r["method"].endswith("producer writes it") and len(r["sources"]) == 1 and len(r["producers"]) == 1
            and _sig_digits(r["claim"]) >= 3):
        return "located: one source holds the value and one script writes it"
    if len(r["sources"]) <= 1 and len(r["producers"]) <= 1:
        return "single source: the ledger names one source and one producer for this result"
    return "unresolved: the ledger entry groups several results; sources listed are candidates"


def claim_sources(resolved: list[dict]) -> str:
    """CLAIM_SOURCES.tsv: every traced number of the current manuscript, its context, source, producer and status."""
    lines = ["location\tscope\tnumber\tcontext\tpopulation_denominator\thorizon\tinterval_or_test\t"
             "definition_source\tdefinition_status\tsource_files\tproducing_scripts\tsource_status\tsource_method\t"
             "data_status\tnotes"]
    for r in resolved:
        hit = next((c for c in CURATED if r["section"].startswith(c[0]) and c[1] in r["context"]), None)
        constructed = not r["sources"] and not r["producers"]
        if hit:
            den, hor, unc, src = hit[2:]
            dstat = NA if (den, hor, unc) == (NA, NA, NA) else "verified"
        elif constructed:
            den = hor = unc = NA
            src, dstat = "constructed or definitional value", NA
        else:
            den = hor = unc = "not yet extracted"
            src, dstat = "-", "not yet extracted"
        rs = route_status(r)
        status = ("public" if not r["restricted"] else
                  "restricted" if rs.startswith("located") else "restricted (conservative: a candidate source is restricted)")
        notes = []
        if r["cut"]:
            notes.append("context is the tracer's first 150 characters")
        if re.search(r"\w\s{2,}\w", r["context"]):
            notes.append("the tracer blanks tokens it does not count as claims (horizons, dates, versions)")
        cells = [r["section"], r["scope"], r["claim"], r["context"], den, hor, unc, src, dstat,
                 "; ".join(r["sources"]) or "-", "; ".join(r["producers"]) or "-", rs, r["method"], status,
                 "; ".join(notes) or "-"]
        lines.append("\t".join(c.replace("\t", " ") for c in cells))
    return "\n".join(lines) + "\n"


def provenance_md(base: Path, commit: str, kept: dict[str, str], shipped: Path) -> str:
    """Runs, models, dates, generation commits and hashes of the deposited data, from the run manifests."""
    B = _deposit_module()
    L = ["# Provenance", "",
         "Models, dates, code commits and checksums of the generated data. Commits refer to the development "
         f"repository; this release is a snapshot of commit `{commit}`.", "",
         "## Generated runs (deposited on Zenodo)", "",
         "| run | role | models | generation commit | date | prompt template | template SHA-256 | accepted | rejected |",
         "|---|---|---|---|---|---|---|---|---|"]
    roles = ({r: "atlas" for r in B.ATLAS_RUNS} | {B.EGFR_SONNET_RUN: "EGFR experiment"}
             | {r: "EGFR check" for r in B.EGFR_CHECK_RUNS} | {r: "scenario benchmark" for r in B.SCENARIO_RUNS})
    for r, role in roles.items():
        mf = ROOT / "runs" / r / "manifest.json"
        if not mf.is_file():
            L.append(f"| `{r}` | {role} | see `config.yaml` of the run | not recorded | | | | | |")
            continue
        m = json.loads(mf.read_text())
        models = ", ".join(x.get("id", "?") for x in m.get("models", []) if isinstance(x, dict))
        L.append(f"| `{r}` | {role} | {models} | `{(m.get('git_commit') or '?')[:12]}` | "
                 f"{(m.get('started_at') or '')[:10]} | `{m.get('prompt_template_path', '')}` | "
                 f"`{(m.get('prompt_template_sha256') or '')[:16]}` | {m.get('trajectories_accepted')} | "
                 f"{m.get('trajectories_failed')} |")
    L += ["", "Available run manifests contain the full hashes. Each run folder also holds its `config.yaml`, the "
          "rendered prompt of every attempt and the rejected attempts.", "",
          "## Registry experiment (GENIE-BPC data; not deposited)", "",
          "| run | model | generation commit |", "|---|---|---|"]
    fa = ROOT / "runs" / "four_arm_v1_definitive_p40b20_20260826" / "manifest.json"
    if fa.is_file():
        m = json.loads(fa.read_text())
        L.append(f"| `four_arm_v1_definitive_p40b20_20260826` | {m.get('model_id')} | "
                 f"`{(m.get('git_commit') or '')[:12]}` |")
    L += ["", "Code: `scripts/explore/build_query_universe_v1.py`, `emit_four_arm_prompts_v1.py`, "
          "`run_four_arm_v1.py` and `scripts/score_four_arm_final_v1.py`.", "",
          "## Prompts and specifications", "",
          "| file | SHA-256 |", "|---|---|"]
    arts = {p: why for p, why in kept.items()
            if why.startswith(("specification", "dated specification", "scoring specification"))
            or p in MASK or p.endswith("_reconstructed_2026-10-14.json") or p.startswith("prompts/")}
    for p in sorted(arts):
        f = shipped / p
        if f.is_file():
            L.append(f"| `{p}` | `{hashlib.sha256(f.read_bytes()).hexdigest()}` |")
    rows, bad = recorded_code_hashes(commit, kept, shipped)
    if bad:
        raise SystemExit(f"released code differs from the version whose hash an output records: {bad}")
    if rows:
        L += ["", "## Code hashes recorded in the outputs", "",
              "Outputs record the SHA-256 of the code that wrote them. Where the released file differs, it differs "
              "in comments only.", "",
              "| file | recorded SHA-256 | released SHA-256 |", "|---|---|---|"]
        L += [f"| `{r['file']}` | `{r['recorded'][:16]}` | `{r['released'][:16]}` |" for r in rows]
    return "\n".join(L) + "\n"


def _strip_docstrings(src: str) -> str:
    t = ast.parse(src)
    for n in ast.walk(t):
        if (isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                and isinstance(n.body[0].value.value, str)):
            n.body[0].value.value = ""
    return ast.dump(t)


def recorded_code_hashes(commit: str, kept: dict[str, str], shipped: Path) -> tuple[list[dict], list[str]]:
    """Shipped code files whose SHA-256 is recorded in a deposited output or run manifest.

    The recorded hash is matched against the file's versions in the history; the released copy must be byte-identical
    to that version or differ from it only in comments and docstrings (same AST apart from docstrings)."""
    B = _deposit_module()
    plan = B.default_plan()
    sel = B.select(plan, B.released_code(B.tracked_paths(plan.root)))
    where: dict[str, set[str]] = {}
    for rel, _ in sel.chosen:
        if not B.is_text(rel) or (rel.startswith("runs/") and not rel.endswith("manifest.json")):
            continue
        text = B.as_text(rel, (plan.root / rel).read_bytes()).decode("utf-8", errors="ignore")
        for h in set(re.findall(r"\b[0-9a-f]{64}\b", text)):
            where.setdefault(h, set()).add(rel)
    rows, bad = [], []
    for p in sorted(k for k in kept if k.endswith(".py")):
        revs = subprocess.run(["git", "log", "--format=%H", "-n", "60", commit, "--", p], cwd=ROOT,
                              capture_output=True, text=True).stdout.split()
        for rev in revs:
            blob = subprocess.run(["git", "show", f"{rev}:{p}"], cwd=ROOT, capture_output=True).stdout
            h = hashlib.sha256(blob).hexdigest()
            if h not in where:
                continue
            released = (shipped / p).read_bytes()
            rh = hashlib.sha256(released).hexdigest()
            if rh == h:
                rel_note = "identical"
            elif _strip_docstrings(blob.decode()) == _strip_docstrings(released.decode()):
                rel_note = "differs only in comments and docstrings (abstract syntax tree identical apart from docstrings)"
            else:
                rel_note = "CODE DIFFERS"
                bad.append(p)
            rows.append({"file": p, "recorded": h, "released": rh, "note": rel_note,
                         "where": ", ".join(sorted(where[h])[:3]) + (" ..." if len(where[h]) > 3 else "")})
            break
    return rows, bad


RECONSTRUCTED = {
    "interventional_phase1_kimi_k3_off_full_v1c":
        "the arm manifest of the 120-cell design (original lost) is rebuilt by "
        "`scripts/explore/build_interventional_phase1.py`; the combined Kimi K3 table is rebuilt byte for byte by "
        "`scripts/explore/build_kimi_k3_combined.py`. Equivalence is shown, not identity with the lost file.",
    "interventional_phase1_kimi_k3_off_topup75_v1c": "none (0 accepted futures; used by no result).",
    "sentinel_usecase_egfr_post_osi_explicit_resistance_v5_2026_07_13":
        "none for this run (0 accepted; replaced by `..._v5_v2_2026_07_13`). The benchmark arm manifest is "
        "reconstructed in `scripts/explore/sentinel_battery_manifest_reconstructed_2026-10-14.json` "
        "(`docs/run_provenance/sentinel_benchmark_scoring.md`).",
}


def mask(text: str) -> tuple[str, int]:
    """Each distinct identifier gets its own neutral label, so values stay distinct if used as keys."""
    labels: dict[str, str] = {}

    def sub(m: re.Match) -> str:
        return labels.setdefault(m.group(0), f"GENIE-REDACTED-{len(labels) + 1:02d}")
    n = 0
    for p in C.PATTERNS:
        text, k = p.subn(sub, text)
        n += k
    return text, n


PURPOSE_GROUPS = (  # purpose of each group of files, for FILE_MANIFEST.tsv (the per-file reason is also given)
    ("scripts/figures/", "figure panels, layout and assembly"),
    ("scripts/release/", "release tooling (snapshot, deposit, Source Data, checks)"),
    ("scripts/explore/", "analysis producing paper numbers or tables, or generation/scoring code of the runs"),
    ("scripts/", "analysis, generation, scoring or manuscript build"),
    ("src/", "synthonccases package: generation engine, validation, derivation"),
    ("tests/", "unit tests of the shipped code"),
    ("configs/", "generation configs of the runs, or test inputs"),
    ("prompts/", "prompt templates of the runs"),
    ("data/", "reference data (OncoTree roster)"),
    ("docs/", "specification or run-provenance note"),
    ("manuscript/supplementary_tables/", "Supplementary Tables 2-8 (sources of the supplement build)"),
    ("manuscript/", "manuscript source, claim ledger and trace, or dated specification"),
    ("", "environment, licence or release document"),
)
GENERATED = {"README.md": "release README (how to reproduce; from scripts/release/RELEASE_README.md)",
             "REPRODUCTION_INDEX.md": "generated: display item / table / text number -> script -> inputs -> status",
             "PROVENANCE.md": "generated: generation commit and hashes per run; specifications",
             "FILE_MANIFEST.tsv": "generated: every file with its purpose and SHA-256",
             "SNAPSHOT_MANIFEST.json": "generated: commit, builder hash, SHA-256 of every file"}


def purpose(path: str) -> str:
    return next(why for pre, why in PURPOSE_GROUPS if path.startswith(pre))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "release_candidates")
    ap.add_argument("--list", action="store_true", help="print the kept files with their reason; build nothing")
    ap.add_argument("--overlay", nargs="*", default=[], metavar="PATH",
                    help="take these files from the working tree instead of the commit (rehearsal of uncommitted "
                         "release files; recorded in the report and the status becomes REHEARSAL)")
    ap.add_argument("--reads", type=Path, help="files opened per panel script in an isolated rebuild (JSON); "
                                               "used for the inputs column of REPRODUCTION_INDEX.md")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "--short", a.commit], cwd=ROOT, text=True).strip()
    ident = baseline_paths(commit)
    drop = (ident - MASK) | EXTRA_DROP
    raw = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=ROOT)
    builder = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    name_out = f"paper_code_release_{commit}"
    report = {"commit": commit, "builder_sha256": builder, "boundary": boundary_versions(commit), "overlay": {},
              "kept": 0, "left_out": 0, "kept_by_group": {},
              "left_out_by_group": {}, "needed_but_identifier_bearing": [], "missing_roots": [],
              "unresolved_imports": {}, "masked": {}, "masked_yaml_structure_ok": {}, "identifier_scan": None,
              "credentials": [], "local_paths_scrubbed": {}, "local_user_name": [], "files": {}}
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full"
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            for m in members:
                dest = full / m.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(m).read())
        tracked = {m.name for m in members}
        for o in a.overlay:
            src_p = ROOT / o
            if not src_p.is_file():
                raise SystemExit(f"overlay file missing: {o}")
            (full / o).parent.mkdir(parents=True, exist_ok=True)
            (full / o).write_bytes(src_p.read_bytes())
            tracked.add(o)
            report["overlay"][o] = hashlib.sha256(src_p.read_bytes()).hexdigest()
        res = closure(full, {p for p in tracked if keep(p, set()) or p.split("/", 1)[0] in CANDIDATE_TOP}, repo=ROOT)
        kept = res["kept"]
        report["missing_roots"] = res["missing_roots"]
        report["trace_scripts_missing"] = res["trace_scripts_missing"]
        report["unresolved_imports"] = res["unresolved_imports"]
        report["tests_left_out"] = res["tests_dropped"]
        report["needed_but_identifier_bearing"] = sorted(p for p in kept if p in drop)
        report["internal_needed"] = res["internal_needed"]
        if a.list:
            for p in sorted(kept):
                print(f"{p}\t{kept[p]}")
            print(f"{len(kept)} kept of {len(tracked)} tracked; missing roots {res['missing_roots']}; "
                  f"unresolved {len(res['unresolved_imports'])}", file=sys.stderr)
            return 0
        base = Path(tmp) / name_out
        for name in sorted(tracked):
            group = name.split("/")[0] if "/" in name else "(root)"
            if name not in kept or name in drop:
                report["left_out"] += 1
                report["left_out_by_group"][group] = report["left_out_by_group"].get(group, 0) + 1
                continue
            report["kept_by_group"][group] = report["kept_by_group"].get(group, 0) + 1
            data = (full / name).read_bytes()
            if name in MASK:
                text, n = mask(data.decode())
                a_, b_ = yaml.safe_load(data.decode()), yaml.safe_load(text)
                report["masked_yaml_structure_ok"][name] = shape(a_) == shape(b_)
                data = text.encode()
                report["masked"][name] = n
            if Path(name).suffix in TEXT and HOME.encode() in data:
                # local home path becomes "~", as in the deposit
                k = data.count(HOME.encode())
                data = data.replace(HOME.encode(), b"~")
                report["local_paths_scrubbed"][name] = k
            if Path(name).suffix in TEXT and USER.encode() in data:
                report["local_paths_scrubbed"][name] = report["local_paths_scrubbed"].get(name, 0) + data.count(USER.encode())
                data = data.replace(USER.encode(), b"<user>")
            out_name = "README.md" if name == RELEASE_README else name
            dest = base / out_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            report["files"][out_name] = kept[name]
            report["kept"] += 1
        reads = json.loads(a.reads.read_text()) if a.reads else None
        resolved = resolve_claims(full, kept=set(kept))
        (base / "REPRODUCTION_INDEX.md").write_text(reproduction_index(full, res, commit, reads, resolved))
        (base / "PROVENANCE.md").write_text(provenance_md(full, commit, kept, base))
        with (base / "FILE_MANIFEST.tsv").open("w") as f:
            f.write("path\tpurpose\twhy kept\tsha256\n")
            # The manifest does not list itself (its hash is in SNAPSHOT_MANIFEST.json).
            for p in sorted(x for x in base.rglob("*") if x.is_file() and x != base / "FILE_MANIFEST.tsv"):
                r = p.relative_to(base).as_posix()
                why = report["files"].get(r, "")
                f.write(f"{r}\t{GENERATED.get(r) or purpose(r)}\t{why}\t"
                        f"{hashlib.sha256(p.read_bytes()).hexdigest()}\n")
        files = sorted(p for p in base.rglob("*") if p.is_file())
        (base / "SNAPSHOT_MANIFEST.json").write_text(json.dumps({
            "commit": commit, "builder_sha256": builder, "boundary": report["boundary"], "overlay": report["overlay"],
            "masked": report["masked"],
            "files": {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=1) + "\n")
        found, rasters, unchecked = C.scan_paths(sorted(p for p in base.rglob("*") if p.is_file()))
        report["identifier_scan"] = {"files_with_identifiers": {str(Path(k).relative_to(base)): v for k, v in found.items()},
                                     "not_checked": [str(Path(u).relative_to(base)) for u in unchecked]}
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".env", ".cfg", ".ini",
                                            ".sh", ".tsv"}:
                if CREDENTIALS.search(p.read_text(errors="ignore")):
                    report["credentials"].append(str(p.relative_to(base)))
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in TEXT and USER in p.read_text(errors="ignore"):
                report["local_user_name"].append(str(p.relative_to(base)))
        # one-commit repository without history: run manifests stamp a git commit
        env = {"GIT_AUTHOR_NAME": "snapshot", "GIT_AUTHOR_EMAIL": "snapshot@localhost", "GIT_COMMITTER_NAME": "snapshot",
               "GIT_COMMITTER_EMAIL": "snapshot@localhost", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "commit", "-q", "-m", f"Paper code release, snapshot of {commit} (no history)"]):
            subprocess.run(cmd, cwd=base, check=True, env=env)
        report["snapshot_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=base, text=True).strip()
        a.out.mkdir(parents=True, exist_ok=True)
        z = a.out / f"{name_out}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(base.parent))
    scan = report["identifier_scan"]
    ok = (not scan["files_with_identifiers"] and not report["credentials"] and not scan["not_checked"]
          and not report["needed_but_identifier_bearing"] and not report["missing_roots"]
          and not report["internal_needed"]
          and not report["local_user_name"] and all(report["masked_yaml_structure_ok"].values()))
    scan["rasters"] = [str(Path(r).relative_to(base)) for r in rasters]
    report["status"] = "FAILED" if not ok else ("REHEARSAL (uncommitted overlay)" if report["overlay"] else
                                                ("NEEDS VISUAL REVIEW" if rasters else "CLEAN"))
    (a.out / f"{name_out}_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['status']}: {report['kept']} files kept, {report['left_out']} left out, masked {sum(report['masked'].values())} "
          f"identifiers; {len(scan['not_checked'])} files not checked; {len(report['credentials'])} credential hits; "
          f"needed but identifier-bearing: {report['needed_but_identifier_bearing']}; "
          f"missing roots: {report['missing_roots']} -> {z}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
