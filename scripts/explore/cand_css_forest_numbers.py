"""Estimates drawn in the classical-summary forests of the CSS pages, written to a table for the text.

Runs the page builders with page drawing replaced by a recorder, so the numbers are those of the panels
(same data, seeds and resamples). Output: outputs/exploratory_cand_css_four_arm/forest_numbers.csv
Run: uv run python -W ignore scripts/explore/cand_css_forest_numbers.py
"""
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
sys.path.insert(0, str(ROOT / "scripts" / "figures" / "panels"))
import _css_template as T  # noqa: E402

rows: list[dict] = []


def record(panel: str, groups: list, start: str, axes: str, frames: list, h: float, hd: float, window: float, xl: tuple,
           yl: tuple, table: list, rib_times: list, km_max: float, forest: list, *_: object, **__: object) -> None:
    for f in forest:
        rows.append({"panel": panel, "item": f[0], "value": f[1], "lo": f[2], "hi": f[3],
                     "value2": f[4] if len(f) > 4 else None, "lo2": f[5] if len(f) > 5 else None,
                     "hi2": f[6] if len(f) > 6 else None})


T.page = record
# pages to record; default all three. `egfr` alone needs only the deposited synthetic runs.
sys.argv = ["x", *(sys.argv[1:] or ["prime", "egfr", "genie"])]
runpy.run_path(str(ROOT / "scripts" / "figures" / "panels" / "cand_css_pages.py"), run_name="pages")
(ROOT / "outputs" / "exploratory_cand_css_four_arm").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).round(3).to_csv(ROOT / "outputs" / "exploratory_cand_css_four_arm" / "forest_numbers.csv", index=False)
