#!/usr/bin/env python3
"""N8 PRIME: time alive without progression, alive after progression and dead, by randomized arm
and KRAS exon 2 status (completion plan 28 Oct 2026, N8; observed trial, no LLM).

Specification: manuscript/review/2026-10-28_completion_plan.md, "N8 specification".
Population  PRIME (NCT00364013) patients with a known KRAS exon 2 result (BMMTR1 "Wild-type" or
            "Mutant"; failure and missing excluded), by randomized arm (TRT), as in Fig. 6a.
Endpoints   PFS by central review (PFSDYCR days, PFSCR: progression or death) and overall survival
            (DTHDY days, DTH: death), both kept exactly as in the trial file (no truncation, no
            reconciliation of one endpoint to the other). OS censoring: the protocol (Amendment 2,
            21 Jan 2009, section 10.2.1.2) censors patients alive at the cutoff "at their last contact
            date"; DTHDY/DTH come from the endpoint analysis dataset (A_ENDPT, data dictionary
            DDT_203_v2) and DTHDY is filled for every living patient, so DTHDY is read as the OS
            time (death or last contact). Strong indication, not an explicit field definition.
Estimator   Kaplan-Meier restricted means at tau: alive progression-free = RMST(PFS, tau); alive
            after progression = RMST(OS, tau) - RMST(PFS, tau); dead = tau - RMST(OS, tau).
            Panitumumab + FOLFOX minus FOLFOX alone for each part, within each KRAS group.
Horizon     tau = 24 months (730.5 days); tau = 18 months only if fewer than 20 patients remain at
            risk for OS at 24 months in any arm x KRAS group (decided on the at-risk counts before
            any part is computed).
Uncertainty ONE bootstrap shared with Fig. 6a: 2,000 resamples of
            patients within arm, seed 20261028, months = days / 30.4375
            (`pds_prime_trial_geometry.RMST_*`). `subgroup_rmst` draws from one generator, KRAS
            wild-type then mutant first, each resample drawing FOLFOX patients then panitumumab
            patients; the same patient indices are used here for PFS and OS, so the three parts stay
            consistent within a resample, and the progression-free row equals Fig. 6a (asserted).
            95% percentile intervals. Resamples where RMST(OS) < RMST(PFS) in a group are counted
            and reported, never discarded or truncated. (Superseded: 600 resamples, seed 20260804,
            30.44 days; and an earlier separate 2,000-resample run.)
Reconciliation with Fig. 6a: arm and KRAS counts, PFS events and the 16 zero-day censored PFS
records are checked against outputs/exploratory_prime_corrections_2026_10_12/prime_forest_hr_itt935.csv.
Months and restricted means as in Fig. 6a: days / 30.4375, its `km_rmst`. Aggregate output only (PRIME data-use agreement) to
outputs/exploratory_n8_prime_time_in_state/.
Run: uv run python scripts/explore/n8_prime_time_in_state_v1.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PDS = Path(os.environ.get("PDS_ROOT", Path.home() / "PDS"))
TRIAL = PDS / "NCT00364013" / "AllProvidedFiles_309 (1)" / "PDS_DSA_20050203"
FOREST = ROOT / "outputs" / "exploratory_prime_corrections_2026_10_12" / "prime_forest_hr_itt935.csv"
OUT = ROOT / "outputs" / "exploratory_n8_prime_time_in_state"
PAN, FOL = "Panitumumab + FOLFOX", "FOLFOX alone"
KRAS = ("Wild-type", "Mutant")
TAUS = (24.0, 18.0)
MIN_AT_RISK = 20
SUBGROUPS6A = ROOT / "outputs" / "exploratory_pds_prime_trial_geometry" / "subgroup_rmst24.csv"
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
import pds_prime_trial_geometry as G  # noqa: E402  (the Fig. 6a producer: loader, km_rmst, seed, BOOT)

MONTH_DAYS = G.RMST_MONTH_DAYS            # shared with Fig. 6a
BOOT, SEED = G.RMST_BOOT, G.RMST_SEED
TO_MONTHS = G.MONTH_DAYS / G.RMST_MONTH_DAYS   # the loader gives days / 30.44; convert to days / 30.4375
PARTS = ("alive_progression_free", "alive_after_progression", "dead")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parts(pfs_m, pfs_e, os_m, os_e, tau: float) -> dict:
    p = G.km_rmst(pfs_m, pfs_e, tau)
    o = G.km_rmst(os_m, os_e, tau)
    return {"alive_progression_free": p, "alive_after_progression": o - p, "dead": tau - o}


def main() -> int:
    # the Fig. 6a subject frame, in its row order (the bootstrap reuses its draws)
    subj = G.derive_subjects(G.load_tables(G._find_data_dir()))
    assert len(subj) == 935
    d = pd.DataFrame({"arm": subj["arm_raw"].to_numpy(), "kras": subj["kras_exon2"].to_numpy(),
                      "pfs_m": subj["pfs_months"].to_numpy() * TO_MONTHS, "pfs_e": subj["pfs_event"].to_numpy(),
                      "os_m": subj["os_months"].to_numpy() * TO_MONTHS, "os_e": subj["os_event"].to_numpy()})
    assert d[["arm", "pfs_m", "pfs_e", "os_m", "os_e"]].notna().all().all()
    d = d[d.kras.isin(KRAS)].reset_index(drop=True)

    # reconciliation with Fig. 6a (counts, PFS events, zero-day censored records)
    f = pd.read_csv(FOREST).set_index("label")
    recon = {}
    for k, label in (("Wild-type", "KRAS exon 2 wild-type"), ("Mutant", "KRAS exon 2 mutant")):
        g = d[d.kras.eq(k)]
        mine = {"n": len(g), "n_pan": int(g.arm.eq(PAN).sum()), "n_fol": int(g.arm.eq(FOL).sum()),
                "events": int(g.pfs_e.sum()), "n_zero_day": int(g.pfs_m.le(0).sum())}
        saved = {c: int(f.loc[label, c]) for c in mine}
        assert mine == saved, (k, mine, saved)
        recon[k] = mine
    assert d.loc[d.pfs_m.le(0), "pfs_e"].eq(0).all()          # zero-day records are censored

    # horizon from OS at-risk counts only (before any part is computed)
    at_risk = {tau: {f"{k} | {a}": int(d[(d.kras == k) & (d.arm == a)].os_m.ge(tau).sum())
                     for k in KRAS for a in (PAN, FOL)} for tau in TAUS}
    tau = TAUS[0] if min(at_risk[TAUS[0]].values()) >= MIN_AT_RISK else TAUS[1]
    assert tau == 24.0, "the Fig. 6a bootstrap is at 24 months; an 18-month horizon would need its own scheme"

    groups = {(k, a): d[(d.kras == k) & (d.arm == a)].reset_index(drop=True) for k in KRAS for a in (PAN, FOL)}
    arr = {key: [g[c].to_numpy() for c in ("pfs_m", "pfs_e", "os_m", "os_e")] for key, g in groups.items()}
    est = {key: parts(*v, tau) for key, v in arr.items()}
    # Fig. 6a draw order: one generator; KRAS wild-type then mutant; per resample FOLFOX then panitumumab
    rng = np.random.default_rng(SEED)
    boot = {key: [] for key in groups}
    negative = {key: 0 for key in groups}
    for k in KRAS:
        for _ in range(BOOT):
            for a in (FOL, PAN):
                v = arr[(k, a)]
                idx = rng.integers(0, len(v[0]), len(v[0]))
                r = parts(*(x[idx] for x in v), tau)
                boot[(k, a)].append(r)
                negative[(k, a)] += int(r["alive_after_progression"] < 0)

    rows = []
    for (k, a), e in est.items():
        for part in PARTS:
            b = np.array([r[part] for r in boot[(k, a)]])
            rows.append({"kras": k, "arm": a, "part": part, "months": round(e[part], 3),
                         "lo": round(float(np.percentile(b, 2.5)), 3), "hi": round(float(np.percentile(b, 97.5)), 3),
                         "n": len(groups[(k, a)]), "tau_months": tau})
    for k in KRAS:
        for part in PARTS:
            b = np.array([p[part] - q[part] for p, q in zip(boot[(k, PAN)], boot[(k, FOL)], strict=True)])
            rows.append({"kras": k, "arm": "difference (panitumumab + FOLFOX minus FOLFOX)", "part": part,
                         "months": round(est[(k, PAN)][part] - est[(k, FOL)][part], 3),
                         "lo": round(float(np.percentile(b, 2.5)), 3), "hi": round(float(np.percentile(b, 97.5)), 3),
                         "n": len(groups[(k, PAN)]) + len(groups[(k, FOL)]), "tau_months": tau})
    # the progression-free difference must equal Fig. 6a's saved estimate and interval
    f6a = pd.read_csv(SUBGROUPS6A).query("subgroup == 'KRAS exon 2'").set_index("level")
    agree = {}
    for k in KRAS:
        pf = [r for r in rows if r["kras"] == k and r["part"] == "alive_progression_free" and r["arm"].startswith("difference")][0]
        ref = f6a.loc[k]
        agree[k] = {"n8": [pf["months"], pf["lo"], pf["hi"]],
                    "fig6a": [round(float(ref.rmst24_diff_months), 3), round(float(ref.ci_low), 3), round(float(ref.ci_high), 3)]}
        assert agree[k]["n8"] == agree[k]["fig6a"], (k, agree[k])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "n8_time_in_state_v1.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "script": "scripts/explore/n8_prime_time_in_state_v1.py", "tau_months": tau,
        "os_at_risk_by_tau": {str(t): v for t, v in at_risk.items()}, "min_at_risk_rule": MIN_AT_RISK,
        "bootstrap": BOOT, "seed": SEED, "month_days": MONTH_DAYS,
        "reconciliation_with_fig6a": recon, "pfs_difference_equals_fig6a": agree,
        "resamples_with_os_rmst_below_pfs_rmst": {f"{k} | {a}": v for (k, a), v in negative.items()},
        "provenance": {"code_sha256": sha256(Path(__file__)), "inputs_sha256": {
            "adsl_pds2019.sas7bdat": sha256(TRIAL / "adsl_pds2019.sas7bdat"),
            "biomark_pds2019.sas7bdat": sha256(TRIAL / "biomark_pds2019.sas7bdat"),
            "prime_forest_hr_itt935.csv": sha256(FOREST), "subgroup_rmst24.csv": sha256(SUBGROUPS6A)},
            "fig6a_producer_sha256": sha256(ROOT / "scripts" / "explore" / "pds_prime_trial_geometry.py")},
        "aggregate_only": True}, indent=1, default=int) + "\n")
    print(f"[n8] tau {tau} months; OS at risk at 24 months {at_risk[24.0]}; reconciliation with Fig. 6a OK; "
          f"resamples with RMST(OS) < RMST(PFS): {negative}")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
