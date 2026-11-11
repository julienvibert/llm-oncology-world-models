"""Numbers quoted in the manuscript text, from saved outputs only: no generation, no fitting.

(1) Early switch to osimertinib in the chemotherapy arm of the EGFR-mutant Claude Sonnet 5 futures.
(2) Percent dead at 12 and 24 months, GENIE-BPC NSCLC and CRC: observed, record alone and matched examples.
Writes aggregates to outputs/exploratory_cand_css_four_arm/text_numbers.csv (gitignored folder).
Run: uv run python scripts/explore/cand_css_text_numbers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/figures"), str(ROOT / "scripts/figures/panels")]
import _css_template as T  # noqa: E402
import build_paper_precision_ladder_paired_views as paired  # noqa: E402

M = 30.4375
rows: list[dict] = []


def rmst(t: np.ndarray, tau: float = 24.0) -> float:
    return float(np.minimum(t, tau).mean())  # no censoring: every future was generated until death


st, en, _ = paired.load_phase1()
st = st[st.context.eq("EGFR_1L") & st.arm.isin(["osimertinib", "chemo"])].copy()
en = en[en._gid.isin(st._gid.unique())].set_index("_gid")
st["t"] = (st.days_from_diagnosis - st.cutoff_days) / M
pf = en.pfs_line_1_days / M
arm = st.groupby("_gid").arm.first()
osi = st[st.active_treatment_regimens_joined.fillna("").str.lower().str.contains("osimertinib") & st.t.ge(0)].groupby("_gid").t.min()
ch = arm.index[arm.eq("chemo")]
sw = [g for g in ch if g in osi.index and osi[g] < pf[g]]
rest = [g for g in ch if g not in sw]
rows += [dict(item="chemo futures", value=len(ch)), dict(item="early switch futures", value=len(sw)),
         dict(item="early switch earliest month", value=osi[sw].min()), dict(item="early switch latest month", value=osi[sw].max()),
         dict(item="PF months of 24, osimertinib", value=rmst(pf[arm.index[arm.eq("osimertinib")]].to_numpy())),
         dict(item="PF months of 24, chemo all", value=rmst(pf[ch].to_numpy())),
         dict(item="PF months of 24, early switch", value=rmst(pf[sw].to_numpy())),
         dict(item="PF months of 24, no early switch", value=rmst(pf[rest].to_numpy()))]

SITES = ROOT / "outputs/exploratory_cand_css_four_arm/sites.csv.gz"  # built from GENIE-BPC records (restricted)
if SITES.is_file():
    f = pd.read_csv(SITES, usecols=["source", "cancer", "query", "repeat", "death", "follow_end"])
    f = f.drop_duplicates(["source", "cancer", "query", "repeat"])
else:
    print(f"{SITES} not found (GENIE-BPC data): writing the synthetic EGFR rows only")
    f = None
rng = np.random.default_rng(20261106)
for cancer in (("NSCLC", "CRC") if f is not None else ()):
    q = f[f.cancer.eq(cancer)]
    u = q["query"].unique()
    boots = rng.integers(0, len(u), (1000, len(u)))
    for src in ("observed", "patient_only", "patients_like_me"):
        d = q[q.source.eq(src)]
        end = d.follow_end.astype(float).fillna(np.inf).to_numpy()
        dd = d.death.astype(float).to_numpy()
        t = np.minimum(dd, end)
        e = (dd <= end).astype(int)
        k = np.isfinite(t)
        w0 = 1 / d.groupby("query")["query"].transform("size").to_numpy()
        est = 100 * (1 - T.km_curve(t[k], e[k], w0[k], [12.0, 24.0]))
        idx = pd.Series(np.arange(len(u)), index=u).reindex(d["query"]).to_numpy()
        bs = []
        for b in boots:
            c = np.bincount(b, minlength=len(u))[idx]
            bs.append(100 * (1 - T.km_curve(t[k], e[k], (w0 * c)[k], [12.0, 24.0])))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for j, m in enumerate((12, 24)):
            rows.append(dict(item=f"% dead at {m} mo, {cancer}, {src}", value=est[j], lo=lo[j], hi=hi[j], n=len(u)))
out = pd.DataFrame(rows).round(2)
(ROOT / "outputs/exploratory_cand_css_four_arm").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "outputs/exploratory_cand_css_four_arm/text_numbers.csv", index=False)
print(out.to_string())
