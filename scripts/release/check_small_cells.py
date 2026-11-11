"""Check registry and trial outputs for small cells: counts of 1 to 4, shown directly or recoverable (read-only).

Two checks, both on existing files (nothing is recomputed or masked here; masking does not establish permission to
release):

1. Figure ribbons. For every panel that saved <panel>_ribbon_shown.csv (shares shown in a clinical-course ribbon)
   and <panel>_ribbon_denominators.csv, the count behind each shown observed share is share x denominator / 100;
   a shown value is flagged when that count, or its complement (denominator minus count), is 1 to 4.
2. Source Data tables classed "pending" by scripts/release/build_source_data.py. In each row: count columns of 1 to
   4; complements of 1 to 4 against a denominator column of the same row; and counts recovered from share or
   fraction columns times a denominator column.

Output: manuscript/release/small_cell_check.md. Run: uv run python scripts/release/check_small_cells.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "release"))
import build_source_data as SD  # noqa: E402

PANELS = ROOT / "outputs" / "figures" / "panels"
DENOMS = ("n_group", "n_index", "cohort_n", "n", "n_evaluable")
COUNT = re.compile(r"^(n|n_.+|.+_n|cohort_n.*|deaths_by_.+|alive_through_.+|unresolved_at_.+|cohort_(deaths|alive|unresolved).+)$")
SHARE = re.compile(r"^(share.*|.*_fraction.*)$")


def small(x: np.ndarray) -> np.ndarray:
    r = np.round(x, 6)
    return (r >= 0.5) & (r < 4.5)


def ribbons() -> list[dict]:
    out = []
    for f in sorted(PANELS.glob("*/*_ribbon_shown.csv")):
        panel = f.parent.name
        s = pd.read_csv(f, index_col=0)
        d = pd.read_csv(PANELS / panel / f"{panel}_ribbon_denominators.csv", index_col=0)
        obs = s[s.observed.astype(bool)] if "observed" in s else s.iloc[0:0]
        n_shown = n_flag = 0
        for grp in obs.group.unique():
            m = obs[obs.group == grp].drop(columns=["group", "observed"])
            den = d[d.group == grp].drop(columns=["group"]).reindex(index=m.index, columns=m.columns)
            mv, dv = m.to_numpy(float), den.to_numpy(float)
            ok = np.isfinite(mv) & np.isfinite(dv)  # survival rows (Kaplan-Meier) have no denominator
            num = mv * dv / 100
            flag = ok & (small(num) | small(dv - num))
            n_shown += int(ok.sum())
            n_flag += int(flag.sum())
        if len(obs):
            out.append({"Panel": panel, "Observed shares shown": n_shown, "Flagged": n_flag})
    return out


def tables() -> list[dict]:
    seen, out = set(), []
    for src in SD.pending_sources():
        f = ROOT / src
        if src in seen or f.suffix not in {".csv", ".gz"} or not f.is_file():
            continue
        seen.add(src)
        d = pd.read_csv(f, low_memory=False)
        num = d.apply(pd.to_numeric, errors="coerce")
        counts = [c for c in d.columns if COUNT.match(c) and num[c].notna().any() and (num[c].dropna() % 1 == 0).all()]
        dens = [c for c in DENOMS if c in counts]
        shares = [c for c in d.columns if SHARE.match(c) and num[c].notna().any() and num[c].max() <= 1]
        direct = sum(int(small(num[c].to_numpy(float)).sum()) for c in counts)
        comp = sum(int(small((num[dn] - num[c]).to_numpy(float)).sum()) for dn in dens for c in counts if c != dn)
        derived = sum(int((small(v) | small(num[dn].to_numpy(float) - v)).sum())
                      for dn in dens for c in shares for v in [(num[c] * num[dn]).to_numpy(float)])
        out.append({"Table": src, "Rows": len(d), "Count columns": ", ".join(counts) or "-",
                    "Direct 1-4": direct, "Complement 1-4": comp, "From shares 1-4": derived})
    return out


def main() -> None:
    r, t = ribbons(), tables()
    md = ["# Small-cell check (generated; do not edit)", "",
          "From `scripts/release/check_small_cells.py`. A flag means a count of 1 to 4 is shown or recoverable "
          "(directly, as a complement, or as share x denominator). Survival estimates (Kaplan-Meier) are not checked. "
          "Masking does not establish permission to release.", "",
          "## Figure ribbons (observed registry or trial patients)", "",
          "| Panel | Observed shares shown | Flagged |", "|---|---|---|"]
    md += [f"| {x['Panel']} | {x['Observed shares shown']} | {x['Flagged']} |" for x in r]
    md += ["", "## Registry and trial Source Data tables", "",
           "| Table | Rows | Count columns | Direct 1-4 | Complement 1-4 | From shares 1-4 |", "|---|---|---|---|---|---|"]
    md += [f"| `{x['Table']}` | {x['Rows']} | {x['Count columns']} | {x['Direct 1-4']} | {x['Complement 1-4']} | "
           f"{x['From shares 1-4']} |" for x in t]
    (ROOT / "manuscript/release/small_cell_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
