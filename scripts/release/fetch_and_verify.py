#!/usr/bin/env python3
"""Verify a downloaded synthetic-data deposit and overlay it onto a checkout.

Standard library only (runs with any Python 3.11+, no `uv sync` needed). Nothing is downloaded
by this script: download every file of the Zenodo record yourself (https://doi.org/10.5281/zenodo.24137513) into one
folder, then pass that folder.

    python3 scripts/release/fetch_and_verify.py <download-folder | deposit-folder | archive> \
        [--checkout .] [--verify-only] [--extract-to DIR] [--force]

Accepted inputs:
  * a download folder holding ``ARCHIVES_SHA256SUMS`` (the packaged deposit): every listed
    archive and loose metadata file is checked against it first (missing file, extra file or
    mismatch = failure); the archives are then unpacked into one new empty folder
    (``--extract-to``, default ``<folder>.extracted``) and the loose metadata copied next to them;
  * an already unpacked deposit folder holding ``SHA256SUMS``;
  * a single archive (tar or zip) of a deposit folder.

Then:
  1. archive members with absolute paths, ``..``, links or duplicates are refused;
  2. every line of ``SHA256SUMS`` is checked (missing file or hash mismatch = failure); files
     present but not listed are failures;
  3. overlay (unless ``--verify-only``): every path listed in ``DEPOSIT_MANIFEST.tsv`` is copied
     to the same relative path in the checkout. The overlay refuses paths tracked in git (a
     deposit never changes tracked code or tables), paths escaping the checkout, and existing
     files with different content (``--force`` replaces them). Identical files are left alone.
Exit code 0 only when everything verified (and, if requested, was overlaid).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

META = {"SHA256SUMS", "DEPOSIT_MANIFEST.tsv", "DEPOSIT_README.md", "TRANSFORMATIONS.tsv",
        "LICENSE.txt"}
ARCHIVE_SUMS = "ARCHIVES_SHA256SUMS"
IGNORED = {".DS_Store"}


class VerifyError(Exception):
    """A check failed; the message says which."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(rel: str) -> bool:
    p = PurePosixPath(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts and "\\" not in rel


def parse_sums(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """(digest, relative path) pairs and malformed-line errors."""
    pairs, errors = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        rel = rel.strip()
        if len(digest) != 64 or not safe_rel(rel):
            errors.append(f"{path.name} line {i}: malformed or unsafe path")
            continue
        pairs.append((digest, rel))
    return pairs, errors


def _empty_dest(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise VerifyError(f"extract folder {dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)


def _extract_tar(archive: Path, dest: Path, seen: set[str]) -> int:
    with tarfile.open(archive) as t:
        members = t.getmembers()
        for m in members:
            if not safe_rel(m.name) or not (m.isfile() or m.isdir()):
                raise VerifyError(f"unsafe member in {archive.name}: {m.name!r}")
            if m.isfile():
                if m.name in seen:
                    raise VerifyError(f"member in more than one archive: {m.name!r}")
                seen.add(m.name)
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
        else:  # pragma: no cover - Python without extraction filters
            t.extractall(dest)
    return sum(m.isfile() for m in members)


def extract(archive: Path, dest: Path) -> Path:
    """Unpack one archive (tar or zip) of a deposit folder."""
    _empty_dest(dest)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise VerifyError("duplicate member in archive")
            for info in z.infolist():
                n = info.filename
                if not safe_rel(n.rstrip("/")):
                    raise VerifyError(f"unsafe member in archive: {n!r}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:  # a symbolic link
                    raise VerifyError(f"link member in archive: {n!r}")
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        _extract_tar(archive, dest, set())
    else:
        raise VerifyError(f"not a folder, zip or tar archive: {archive}")
    kids = [k for k in dest.iterdir() if k.name != "__MACOSX"]
    if not (dest / "SHA256SUMS").exists() and len(kids) == 1 and kids[0].is_dir():
        return kids[0]  # an archive may wrap everything in one top folder
    return dest


def verify_package(pkg: Path) -> tuple[list[str], list[str]]:
    """Check ARCHIVES_SHA256SUMS. Returns errors and the listed file names."""
    pairs, errors = parse_sums(pkg / ARCHIVE_SUMS)
    listed = [rel for _, rel in pairs]
    for digest, rel in pairs:
        f = pkg / rel
        if "/" in rel:
            errors.append(f"{ARCHIVE_SUMS}: nested path not allowed: {rel}")
        elif not f.is_file():
            errors.append(f"missing: {rel}")
        elif sha256(f) != digest:
            errors.append(f"archive checksum mismatch: {rel}")
    for f in pkg.iterdir():
        if f.name not in IGNORED and f.name != ARCHIVE_SUMS and f.name not in listed:
            errors.append(f"not in {ARCHIVE_SUMS}: {f.name}")
    if "SHA256SUMS" not in listed:
        errors.append(f"{ARCHIVE_SUMS} does not list SHA256SUMS")
    return errors, listed


def unpack_package(pkg: Path, dest: Path, listed: list[str]) -> Path:
    """Extract every listed archive into ``dest`` and copy the loose metadata files."""
    _empty_dest(dest)
    seen: set[str] = set()
    n = 0
    for name in listed:
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            n += _extract_tar(pkg / name, dest, seen)
    for name in listed:
        if not name.endswith((".tar.gz", ".tgz", ".tar")):
            if name in seen:
                raise VerifyError(f"loose file also inside an archive: {name}")
            shutil.copy2(pkg / name, dest / name)
    print(f"unpacked: {n} files from {sum(x.endswith(('.tar.gz', '.tgz', '.tar')) for x in listed)}"
          f" archive(s) -> {dest}")
    return dest


def verify(dep: Path) -> tuple[list[str], list[str]]:
    """Check every file of an unpacked deposit against SHA256SUMS. Errors, listed paths."""
    sums = dep / "SHA256SUMS"
    if not sums.is_file():
        return [f"no SHA256SUMS in {dep}"], []
    pairs, errors = parse_sums(sums)
    listed = [rel for _, rel in pairs]
    for digest, rel in pairs:
        f = dep / rel
        if not f.is_file():
            errors.append(f"missing: {rel}")
        elif sha256(f) != digest:
            errors.append(f"checksum mismatch: {rel}")
    listed_set = set(listed)
    for f in dep.rglob("*"):
        if f.is_file() and f.name not in IGNORED:
            rel = f.relative_to(dep).as_posix()
            if rel not in listed_set and rel != "SHA256SUMS":
                errors.append(f"not in SHA256SUMS: {rel}")
    return errors, listed


def manifest_paths(dep: Path) -> list[str]:
    rows = (dep / "DEPOSIT_MANIFEST.tsv").read_text().splitlines()
    return [r.split("\t", 1)[0] for r in rows[1:] if r.strip()]


def tracked(checkout: Path) -> set[str]:
    r = subprocess.run(["git", "ls-files"], cwd=checkout, capture_output=True, text=True)
    if r.returncode != 0:
        raise VerifyError(f"{checkout} is not a git checkout")
    return set(r.stdout.splitlines())


def overlay(dep: Path, checkout: Path, paths: list[str], force: bool) -> list[str]:
    errors = []
    tr = tracked(checkout)
    root = checkout.resolve()
    copied = same = 0
    for rel in paths:
        if rel in META:
            continue
        if not safe_rel(rel):
            errors.append(f"unsafe path: {rel}")
            continue
        if rel in tr:
            errors.append(f"tracked in git, refused: {rel}")
            continue
        dst = root / rel
        if not dst.resolve().is_relative_to(root):
            errors.append(f"escapes checkout: {rel}")
            continue
        src = dep / rel
        if dst.exists():
            if sha256(dst) == sha256(src):
                same += 1
                continue
            if not force:
                errors.append(f"exists with different content (use --force): {rel}")
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"overlay: {copied} copied, {same} already identical, {len(errors)} refused")
    return errors


def run(deposit: Path, checkout: Path, verify_only: bool = False,
        extract_to: Path | None = None, force: bool = False) -> int:
    dep = deposit
    if dep.is_dir() and (dep / ARCHIVE_SUMS).is_file():
        errors, listed = verify_package(dep)
        print(f"archives: {len(listed)} listed files, {len(errors)} problem(s)")
        for e in errors[:50]:
            print(f"  {e}")
        if errors:
            return 1
        dest = extract_to or dep.with_name(dep.name + ".extracted")
        dep = unpack_package(dep, dest, listed)
    elif dep.is_file():
        dep = extract(dep, extract_to or dep.with_name(dep.name + ".extracted"))
    errors, listed = verify(dep)
    print(f"verify: {len(listed)} listed files, {len(errors)} problem(s)")
    for e in errors[:50]:
        print(f"  {e}")
    if errors:
        return 1
    paths = manifest_paths(dep)
    missing = sorted(set(paths) - set(listed))
    if missing:
        print(f"  {len(missing)} manifest path(s) without a checksum, e.g. {missing[0]}")
        return 1
    if verify_only:
        return 0
    errs = overlay(dep, checkout, paths, force)
    for e in errs[:50]:
        print(f"  {e}")
    return 1 if errs else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("deposit", type=Path)
    ap.add_argument("--checkout", type=Path, default=Path("."))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--extract-to", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        return run(a.deposit, a.checkout, a.verify_only, a.extract_to, a.force)
    except VerifyError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
