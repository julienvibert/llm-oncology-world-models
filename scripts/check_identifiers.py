#!/usr/bin/env python3
"""Guard against patient identifiers entering tracked files or commit messages.

Patterns (no clinical data needed): GENIE-BPC patient and sample identifiers
(GENIE-<SITE>-<digits>, GENIE-MSK-P-<7 digits>, with optional -T<nn> sample suffixes) and bare
MSK-style patient identifiers (P-<7 digits>). Fabricated fixtures are allowed when the digits are
all nines or the site is TEST.

Known historical occurrences in the working repository are frozen in `scripts/identifier_baseline.json` as COUNTS per path
(never values). The check fails on any new path with an occurrence or on an increased count.

    uv run python scripts/check_identifiers.py --tracked          # all tracked text files
    uv run python scripts/check_identifiers.py --staged           # staged additions (pre-commit)
    uv run python scripts/check_identifiers.py --message FILE     # commit message (commit-msg)
    uv run python scripts/check_identifiers.py --write-baseline   # only after a reviewed cleanup

Output never prints an identifier: only paths and counts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "identifier_baseline.json"
PATTERNS = [
    re.compile(r"GENIE-(?!TEST\b)[A-Z]{2,6}-(?:P-)?(\d{3,})(?:-T\d{2})?"),
    re.compile(r"(?<![A-Za-z0-9-])P-(\d{7})(?:-T\d{2})?(?![0-9])"),
]
# exports whose text is extracted and scanned (labels can escape through figures)
DOC_SUFFIXES = {".svg", ".pdf", ".docx", ".xlsx", ".pptx"}
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff"}
# fonts and icons carry no data; other binary containers cannot be read as text and are
# reported as NOT CHECKED; .gz is decompressed and scanned
SKIP_SUFFIXES = RASTER_SUFFIXES | {".ico", ".woff", ".ttf", ".otf"}


def count_identifiers(text: str) -> int:
    """Occurrences that look like real identifiers (fabricated all-nines values excluded)."""
    n = 0
    for pat in PATTERNS:
        for m in pat.finditer(text):
            if set(m.group(1)) != {"9"}:
                n += 1
    return n


def _office_text(data: bytes) -> str:
    """Text of one Office XML part, rebuilt across formatting runs: text nodes inside a
    paragraph (Word w:p, PowerPoint a:p, Excel si/row) are joined without separators, so an
    identifier split over several runs is still seen whole."""
    xml = data.decode("utf-8", "ignore")
    xml = re.sub(r"</(?:w:p|a:p|si|row)>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def extract_text(path: Path) -> str | None:
    """Text of a file, including the text layer of SVG, PDF and Office exports.

    Returns None when the text could not be extracted (tool failure, corrupt file, or a PDF
    without a text layer): such a file is NOT CHECKED, never treated as clean."""
    suf = path.suffix.lower()
    if suf == ".pdf":
        try:
            r = subprocess.run(["pdftotext", "-q", str(path), "-"], capture_output=True,
                               text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return r.stdout
    if suf in {".docx", ".xlsx", ".pptx"}:
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                return "\n".join(_office_text(z.read(n)) for n in z.namelist()
                                  if n.endswith(".xml"))
        except (OSError, zipfile.BadZipFile, KeyError):
            return None
    if suf == ".gz":
        import gzip
        try:
            with gzip.open(path, "rt", errors="ignore") as fh:
                return fh.read()
        except (OSError, EOFError):
            return None
    if suf in {".zip", ".bundle", ".pkl", ".parquet", ".npy", ".npz"}:
        return None
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return None


def scan_paths(paths) -> tuple[dict[str, int], list[str], list[str]]:
    """Counts per path; raster images that need a visual check; files NOT CHECKED because
    their text could not be extracted."""
    found, rasters, unchecked = {}, [], []
    for p in paths:
        if not p.is_file() or p == BASELINE:
            continue
        suf = p.suffix.lower()
        if suf in RASTER_SUFFIXES:
            rasters.append(str(p))
            continue
        if suf in SKIP_SUFFIXES:
            continue
        text = extract_text(p)
        if text is None:
            unchecked.append(str(p))
            continue
        n = count_identifiers(text)
        if n:
            found[str(p)] = n
    return found, rasters, unchecked


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def scan_tracked() -> dict[str, int]:
    found, _, _ = scan_paths(ROOT / rel for rel in _git("ls-files").splitlines())
    return {str(Path(k).relative_to(ROOT)): v for k, v in found.items()}


def scan_staged() -> dict[str, int]:
    """Count identifiers in added lines of the staged diff, per path; staged exports (SVG,
    PDF, Office) are extracted from the staged blob and scanned whole. An export whose text
    cannot be extracted is reported as a violation (not checked)."""
    import tempfile
    out: dict[str, int] = {}
    for rel in _git("diff", "--cached", "--name-only", "--diff-filter=AM").splitlines():
        suf = Path(rel).suffix.lower()
        if suf not in DOC_SUFFIXES:
            continue
        blob = subprocess.run(["git", "show", f":{rel}"], cwd=ROOT, capture_output=True,
                              check=True).stdout
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / Path(rel).name
            tmp.write_bytes(blob)
            text = extract_text(tmp)
        n = -1 if text is None else count_identifiers(text)
        if n:
            out[rel + (" (not checked: no extractable text)" if n < 0 else "")] = max(n, 1)
    path = None
    for line in _git("diff", "--cached", "--unified=0", "--no-color").splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and path and Path(path).suffix.lower() not in SKIP_SUFFIXES \
                and Path(path).suffix.lower() not in DOC_SUFFIXES \
                and path != str(BASELINE.relative_to(ROOT)):
            n = count_identifiers(line[1:])
            if n:
                out[path] = out.get(path, 0) + n
    return out


def load_baseline() -> dict[str, int]:
    return json.loads(BASELINE.read_text())["counts"] if BASELINE.exists() else {}


def violations(found: dict[str, int], baseline: dict[str, int]) -> list[str]:
    return [f"{p}: {n} (baseline {baseline.get(p, 0)})" for p, n in sorted(found.items())
            if n > baseline.get(p, 0)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tracked", action="store_true")
    g.add_argument("--staged", action="store_true")
    g.add_argument("--message", type=Path)
    g.add_argument("--write-baseline", action="store_true")
    g.add_argument("--release", nargs="+", type=Path, metavar="DIR",
                   help="release clearance: scan every file under these folders (text, SVG, PDF, "
                        "Office); list raster images for visual review; no baseline applies")
    a = ap.parse_args()
    if a.write_baseline:
        BASELINE.write_text(json.dumps({
            "note": "Counts of known historical patient-identifier occurrences per tracked path, "
                    "frozen until the phase-2 release cleanup. Never store values here.",
            "counts": scan_tracked()}, indent=1, sort_keys=True) + "\n")
        return 0
    if a.release:
        files = [f for d in a.release for f in (d.rglob("*") if d.is_dir() else [d])]
        found, rasters, unchecked = scan_paths(files)
        for f, n in sorted(found.items()):
            print(f"  {f}: {n}")
        for f in sorted(unchecked):
            print(f"  NOT CHECKED (text not extractable): {f}")
        print(f"release scan: {len(found)} file(s) with identifiers; {len(unchecked)} file(s) "
              f"not checked; {len(rasters)} raster image(s) need a visual check")
        return 1 if (found or unchecked) else 0
    if a.message:
        n = count_identifiers(a.message.read_text(errors="ignore"))
        if n:
            print(f"identifier check: the commit message contains {n} patient identifier(s)")
            return 1
        return 0
    if a.staged:
        # staged additions are checked strictly: no new identifier may be added anywhere
        bad = [f"{p}: {n} added" for p, n in sorted(scan_staged().items())]
    else:
        bad = violations(scan_tracked(), load_baseline())
    if bad:
        print("identifier check failed (counts only, values never printed):")
        print("\n".join(f"  {b}" for b in bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
