"""Mask small cells in aggregate tables of registry or trial patients before release as Source Data, and verify the
released copy independently.

Every table is described by an explicit ``Spec``: its count columns, its denominator columns, its share columns with
their unit ("pct" for 0-100, "frac" for 0-1) and their denominator, and, when rows partition a total, the group
columns and the columns that sum to it. Units are never inferred from values; a share-like column that the spec does
not declare makes the build fail.

Masking (``mask``) blanks a cell when it shows, or lets a reader recover, a count of 1 to 4 patients:
  * a count of 1 to 4, or a count whose complement against a denominator of the same row is 1 to 4;
  * a share whose implied count (share x denominator), or its complement, is 1 to 4;
  * secondary suppression: in a partition group, when exactly one cell of a column is blanked, the smallest other
    value is blanked too, so the first cannot be recovered by subtraction.

``verify`` re-checks the released copy against the original without reusing the masking code path: visible counts,
complements and implied counts must not be 1 to 4, and no partition group may have exactly one suppressed cell.

A blank cell in a released table means suppressed, missing or not applicable, never zero; ``masked_cells`` and
``suppressed_columns`` record only the values suppressed here. Masking is a disclosure control;
it does not establish permission to release.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SHARE_LIKE = re.compile(r"share|fraction|pct|percent|proportion", re.I)


@dataclass(frozen=True)
class Spec:
    counts: tuple[str, ...] = ()
    denoms: tuple[str, ...] = ()  # columns used as denominators for count complements
    shares: dict[str, tuple[str, str]] = field(default_factory=dict)  # column -> (unit "pct"|"frac", denominator)
    not_shares: tuple[str, ...] = ()  # share-like names that are not shares of a patient count (bounds, estimates)
    partition: tuple[tuple[str, ...], tuple[str, ...]] | None = None  # (group columns, columns summing to a total)
    real_rows: tuple[str, str] | None = None  # (column, value) selecting rows of real patients; None = every row


def small(x) -> np.ndarray:
    r = np.round(np.asarray(x, float), 6)
    return (r >= 0.5) & (r < 4.5)


def _num(d: pd.DataFrame) -> pd.DataFrame:
    return d.apply(pd.to_numeric, errors="coerce")


def check_spec(d: pd.DataFrame, spec: Spec) -> list[str]:
    """Declared columns exist, units are respected, and no share-like column is left undeclared."""
    out, num = [], _num(d)
    for c in (*spec.counts, *spec.denoms, *spec.shares, *spec.not_shares):
        if c not in d.columns:
            out.append(f"declared column {c!r} missing")
    for c, (unit, den) in spec.shares.items():
        if c in d.columns:
            hi = num[c].max()
            if unit not in ("pct", "frac") or (unit == "frac" and hi > 1 + 1e-9) or (unit == "pct" and hi > 100 + 1e-9):
                out.append(f"{c!r}: values incompatible with unit {unit!r} (max {hi})")
            if den not in d.columns:
                out.append(f"{c!r}: denominator {den!r} missing")
    declared = {*spec.shares, *spec.not_shares}
    for c in d.columns:
        if SHARE_LIKE.search(c) and c not in declared:
            out.append(f"share-like column {c!r} not declared in the spec")
    return out


def _implied(num: pd.DataFrame, c: str, unit: str, den: str) -> np.ndarray:
    return (num[c] * num[den] / (100.0 if unit == "pct" else 1.0)).to_numpy(float)


def _real(d: pd.DataFrame, spec: Spec) -> pd.Series:
    if spec.real_rows is None:
        return pd.Series(True, index=d.index)
    col, val = spec.real_rows
    return d[col].astype(str).eq(val)


def flags(d: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Cells to mask among rows of real patients (model futures are not patient counts)."""
    num = _num(d).where(_real(d, spec), np.nan)
    m = pd.DataFrame(False, index=d.index, columns=d.columns)
    for c in spec.counts:
        m[c] |= small(num[c])
        for dn in spec.denoms:
            if dn != c:
                m[c] |= small(num[dn] - num[c])
    for c, (unit, den) in spec.shares.items():
        k = _implied(num, c, unit, den)
        m[c] |= small(k) | small(num[den].to_numpy(float) - k)
    return m


def _secondary(d: pd.DataFrame, m: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    if spec.partition is None:
        return m
    groups, cols = spec.partition
    num, m = _num(d), m.copy()
    for _, idx in d.groupby(list(groups), dropna=False).groups.items():
        for c in cols:
            hit = m.loc[idx, c]
            if int(hit.sum()) == 1:
                rest = num.loc[idx, c][~hit].dropna()
                if len(rest):
                    m.loc[rest.idxmin(), c] = True
    return m


def mask(d: pd.DataFrame, spec: Spec) -> tuple[pd.DataFrame, int]:
    """Masked copy and number of cells blanked; raises if the spec does not describe the table."""
    problems = check_spec(d, spec)
    if problems:
        raise ValueError("; ".join(problems))
    m = _secondary(d, flags(d, spec), spec)
    n = int(m.to_numpy().sum())
    if not n:
        return d, 0
    out = d.astype(object).mask(m, np.nan)
    out["masked_cells"] = m.sum(axis=1).astype(int)
    out["suppressed_columns"] = m.apply(lambda r: ";".join(c for c in m.columns if r[c]), axis=1)
    return out, n


def verify(original: pd.DataFrame, released: pd.DataFrame, spec: Spec) -> list[str]:
    """Independent check of a released copy. Returns the problems found (empty when the copy is safe)."""
    rel = released.drop(columns=["masked_cells", "suppressed_columns"], errors="ignore")
    out = check_spec(rel, spec)
    real = _real(rel, spec)
    num, orig = _num(rel).where(real, np.nan), _num(original).where(real, np.nan)
    for c in spec.counts:
        if small(num[c]).any():
            out.append(f"{c!r}: visible count of 1 to 4")
        for dn in spec.denoms:
            if dn != c and small(num[dn] - num[c]).any():
                out.append(f"{c!r}: visible complement of 1 to 4 against {dn!r}")
    for c, (unit, den) in spec.shares.items():
        k = _implied(num, c, unit, den)
        if (small(k) | small(num[den].to_numpy(float) - k)).any():
            out.append(f"{c!r}: visible share implying 1 to 4 patients")
    if spec.partition is not None:
        groups, cols = spec.partition
        for key, idx in rel.groupby(list(groups), dropna=False).groups.items():
            for c in cols:
                suppressed = int((orig.loc[idx, c].notna() & num.loc[idx, c].isna()).sum())
                if suppressed == 1:
                    out.append(f"{c!r}: one suppressed cell in group {key}, recoverable by subtraction")
    return out


def verify_ribbon(shown: pd.DataFrame, den: pd.DataFrame, real_groups: list[str],
                  partition_rows: tuple[str, ...] = ()) -> list[str]:
    """Independent check of a clinical-course ribbon (percentages by row and month; denominators in a parallel
    table). Rows of real patients only; ``partition_rows`` are row labels sharing one denominator and summing to
    100%: no column may have exactly one of them blank while its denominator is known."""
    out = []
    months = [c for c in shown.columns if c not in ("group", "observed")]
    for g in real_groups:
        s = shown[shown.group == g][months].apply(pd.to_numeric, errors="coerce")
        dd = den[den.group == g][months].apply(pd.to_numeric, errors="coerce").reindex(index=s.index)
        k = s.to_numpy(float) * dd.to_numpy(float) / 100
        bad = np.isfinite(k) & (small(k) | small(dd.to_numpy(float) - k))
        if bad.any():
            out.append(f"group {g!r}: {int(bad.sum())} visible shares implying 1 to 4 patients")
        part = [r for r in partition_rows if r in s.index]
        if part and len(part) == len(partition_rows):
            blank = s.loc[part].isna() & dd.loc[part].notna()
            for mth in months:
                if int(blank[mth].sum()) == 1:
                    out.append(f"group {g!r}, month {mth}: one blank partition cell, recoverable by subtraction")
    return out
