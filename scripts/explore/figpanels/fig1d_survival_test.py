"""Test of model differences in generated survival within each cancer family (Fig. 1d; Julien C9, 1 Nov 2026).

Per generated trajectory, survival = last generated month (diagnosis to death; the atlas contract
generates until death, checked below: every course included has a death event). Within each of the
eight cancer families, Kruskal-Wallis test across the six models (all times are observed deaths, no
censoring, so a rank test is valid); P values Holm-adjusted across the 8 families.
Output: outputs/exploratory_figure_panels/fig1/data/survival_by_family_test.csv (aggregate).
Run: uv run python scripts/explore/figpanels/fig1d_survival_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "scripts"))
from build_atlas_model_dissection import MODEL_LABEL  # noqa: E402
from build_field_trellis import CODE2FAM  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_figure_panels" / "fig1" / "data" / "survival_by_family_test.csv"
MONTH_FIX = 30.4 / 30.4375  # producer months (30.4 d) -> manuscript months (30.4375 d)


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main() -> int:
    s = enrich_snapshots(load_tables())
    s["family"] = s["tumor_oncotree_code"].astype(str).map(CODE2FAM)
    s["model"] = s["model_short"].astype(str).map(lambda m: MODEL_LABEL.get(m, m))
    os_tab = (s.groupby(["model", "_gid"])["month"].max().reset_index()
              .merge(s.groupby("_gid")["family"].first().rename("family"), on="_gid")
              .dropna(subset=["family"]))
    os_tab["month"] = os_tab["month"] * MONTH_FIX
    g = s.dropna(subset=["family"]).groupby("_gid")
    assert (g["death_month"].max() - g["month"].max()).abs().max() < 1e-6, "a course does not end in death"
    rows = []
    for fam, g in os_tab.groupby("family"):
        groups = [v["month"].to_numpy(float) for _, v in g.groupby("model")]
        h, p = kruskal(*groups)
        rows.append({"family": fam, "n_models": len(groups), "n_total": len(g), "H": h, "p": p})
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r["p"].tolist())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)
    print(r.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
