#!/usr/bin/env python3
"""PRIME, analyses of 12 October 2026: intention-to-treat hazard ratios and a
censoring-aware local progression-free survival surface, both on KRAS exon 2.

Read-only analysis of saved trial files and the saved restricted PRIME projection. No model
calls, no generation. Writes aggregate tables only (no patient rows) to
outputs/exploratory_prime_corrections_2026_10_12/ (gitignored with the other PRIME outputs).

Declared choices
----------------
Biomarker   BMMTR1, named "KRAS exon 2 (c12/13)" by the trial's label column BMMTNM1 (checked).
            "Failure" and missing are unknown: excluded from KRAS rows, never counted as mutant.
Endpoint    Progression-free survival by central review (PFSDYCR days, PFSCR event = progression
            or death), the trial file's endpoint. Months = days / 30.4375.
Population  Survival estimates: all 935 randomised patients (intention to treat). Sixteen
            records have a PFS time of 0 days, all censored (no event). They are kept at time 0:
            a censored record at time 0 is in no risk set at any event time, so it contributes no
            information to the Cox likelihood or the Kaplan-Meier estimate; it only enters n. The
            script checks that the hazard ratios equal those with the 16 records removed.
            Local surface: patients in the restricted PRIME projection with a baseline assessment
            (first lesion assessment on or before day 14 after randomisation) and a known KRAS
            exon 2 result.
Local       Kernel-weighted Kaplan-Meier at 12 months (365.25 days) in each arm, at grid points
estimator   of the baseline position (restricted PRIME projection PC1/PC2). Gaussian kernel on
            axes scaled by their standard deviation (bandwidth 0.6 SD). Difference =
            panitumumab + FOLFOX minus FOLFOX alone in 12-month progression-free probability.
            Censoring is handled by the weighted product-limit estimator: a patient censored
            before 12 months leaves the risk set at censoring and is never counted as event-free.
Uncertainty Patient-level bootstrap (patients resampled within arm and KRAS group, 1,000
            replicates, seed 20261012), percentile 95% intervals; also for the local estimate
            minus the KRAS-group estimate (does a location differ from its group?).
Grid        4 x 4 evenly spaced points between the 15th and 85th percentiles of each axis of the map
            population; a point is shown only where both arms have an effective sample size of
            at least 20 patients. The grid points are not independent (overlapping kernels).

Run:  uv run python scripts/explore/build_prime_itt_forest_local_km.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[2]
PDS = Path(os.environ.get("PDS_ROOT", Path.home() / "PDS"))
TRIAL = PDS / "NCT00364013" / "AllProvidedFiles_309 (1)" / "PDS_DSA_20050203"
VISITS = ROOT / "outputs" / "exploratory_prime_trial_geometry" / "prime_state_visits.csv.gz"
OUT = ROOT / "outputs" / "exploratory_prime_corrections_2026_10_12"

KRAS_COLUMN, KRAS_LABEL_COLUMN, KRAS_LABEL = "BMMTR1", "BMMTNM1", "KRAS exon 2 (c12/13)"
PAN, FOL = "Panitumumab + FOLFOX", "FOLFOX alone"
MONTH_DAYS = 30.4375
HORIZON_DAYS = 365.25
BASELINE_MAX_DAY = 14
BANDWIDTH_SD = 0.6
MIN_ESS = 20.0
GRID_N = 4
GRID_Q = (0.15, 0.85)
BOOT = 1000
SEED = 20261012


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rd(name: str) -> pd.DataFrame:
    return pd.read_sas(TRIAL / f"{name}_pds2019.sas7bdat", format="sas7bdat", encoding="latin-1")


def subjects() -> pd.DataFrame:
    sl = rd("adsl")
    bm = rd("biomark").set_index("SUBJID")
    assert set(bm[KRAS_LABEL_COLUMN].dropna().unique()) == {KRAS_LABEL}
    kras = sl["SUBJID"].map(bm[KRAS_COLUMN])  # map on the raw key, before conversion
    out = pd.DataFrame({
        "SUBJID": pd.to_numeric(sl["SUBJID"]).astype(int),
        "arm": sl["TRT"],
        "days": pd.to_numeric(sl["PFSDYCR"], errors="coerce"),
        "event": pd.to_numeric(sl["PFSCR"], errors="coerce"),
        "kras": kras.where(kras.isin(["Wild-type", "Mutant"]), "Unknown"),
        "ecog": sl["B_ECOG"],
        "liver": sl["LIVERMET"],
    })
    assert out["days"].notna().all() and out["event"].notna().all()
    assert len(out) == 935
    return out


def cox(sub: pd.DataFrame) -> dict:
    d = sub.assign(t=sub["days"] / MONTH_DAYS, pani=sub["arm"].eq(PAN).astype(float))
    m = CoxPHFitter().fit(d[["t", "event", "pani"]], "t", "event")
    r = m.summary.loc["pani"]
    return {"hr": float(r["exp(coef)"]), "lo": float(r["exp(coef) lower 95%"]),
            "hi": float(r["exp(coef) upper 95%"]), "p": float(r["p"]),
            "n": int(len(d)), "n_pan": int(d["pani"].sum()), "n_fol": int((1 - d["pani"]).sum()),
            "events": int(d["event"].sum()), "n_zero_day": int(d["days"].le(0).sum())}


def forest(s: pd.DataFrame) -> pd.DataFrame:
    rows = [("All patients", s.index == s.index),
            ("KRAS exon 2 wild-type", s["kras"].eq("Wild-type")),
            ("KRAS exon 2 mutant", s["kras"].eq("Mutant")),
            ("ECOG performance status 0", s["ecog"].eq("Fully active")),
            ("ECOG performance status 1 or 2",
             s["ecog"].isin(["Symptoms but ambulatory", "In bed less than 50% of the time"])),
            ("Liver metastases present", s["liver"].eq("Y")),
            ("Liver metastases absent", s["liver"].eq("N"))]
    out = []
    for label, mask in rows:
        full = cox(s[mask])
        nz = cox(s[mask & s["days"].gt(0)])
        assert abs(full["hr"] - nz["hr"]) < 1e-6 and abs(full["lo"] - nz["lo"]) < 1e-6, label
        out.append({"label": label, **full, "n_without_zero_day": nz["n"]})
    return pd.DataFrame(out)


def weighted_km(days: np.ndarray, event: np.ndarray, w: np.ndarray, tau: float) -> float:
    """Kaplan-Meier with case weights, evaluated at tau. Censoring removes weight from the risk set."""
    keep = w > 0
    days, event, w = days[keep], event[keep], w[keep]
    s = 1.0
    for t in np.unique(days[(event == 1) & (days <= tau)]):
        at_risk = w[days >= t].sum()
        d = w[(days == t) & (event == 1)].sum()
        if at_risk > 0:
            s *= 1.0 - d / at_risk
    return s


def kernel(pos: np.ndarray, point: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (pos - point) / (scale * BANDWIDTH_SD)
    return np.exp(-0.5 * (z ** 2).sum(axis=1))


def local_surface(s: pd.DataFrame, visits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = visits[visits["assessment_index"].eq(0) & visits["day"].le(BASELINE_MAX_DAY)]
    base = base[["SUBJID", "pc1", "pc2"]].merge(s, on="SUBJID", how="inner")
    pop = {"patients_in_projection": int(visits["SUBJID"].nunique()),
           "with_baseline_assessment": int(len(base))}
    base = base[base["kras"].isin(["Wild-type", "Mutant"])].reset_index(drop=True)
    pop["with_known_kras"] = int(len(base))
    pop["by_kras_arm"] = {f"{k} | {a}": int(n) for (k, a), n in base.groupby(["kras", "arm"]).size().items()}
    pos_all = base[["pc1", "pc2"]].to_numpy(float)
    scale = pos_all.std(axis=0)
    gx = np.linspace(*np.quantile(pos_all[:, 0], GRID_Q), GRID_N)
    gy = np.linspace(*np.quantile(pos_all[:, 1], GRID_Q), GRID_N)
    grid = np.array([(x, y) for x in gx for y in gy])
    rng = np.random.default_rng(SEED)
    rows, refs = [], []
    for k in ("Wild-type", "Mutant"):
        arms = {}
        for a in (PAN, FOL):
            g = base[base["kras"].eq(k) & base["arm"].eq(a)]
            arms[a] = (g[["pc1", "pc2"]].to_numpy(float), g["days"].to_numpy(float), g["event"].to_numpy(int))

        def estimate(idx: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
            loc = np.zeros(len(grid))
            glob = 0.0
            for sign, a in ((1, PAN), (-1, FOL)):
                p, d, e = (x[idx[a]] for x in arms[a])
                glob += sign * weighted_km(d, e, np.ones(len(d)), HORIZON_DAYS)
                for j, pt in enumerate(grid):
                    loc[j] += sign * weighted_km(d, e, kernel(p, pt, scale), HORIZON_DAYS)
            return loc, glob

        ident = {a: np.arange(len(arms[a][1])) for a in arms}
        est, glob = estimate(ident)
        ess = {}
        for a in arms:
            w = np.stack([kernel(arms[a][0], pt, scale) for pt in grid])
            ess[a] = w.sum(axis=1) ** 2 / (w ** 2).sum(axis=1)
        bl, bg = [], []
        for _ in range(BOOT):
            idx = {a: rng.integers(0, len(arms[a][1]), len(arms[a][1])) for a in arms}
            loc_b, glob_b = estimate(idx)
            bl.append(loc_b)
            bg.append(glob_b)
        bl, bg = np.array(bl), np.array(bg)
        lo, hi = np.percentile(bl, [2.5, 97.5], axis=0)
        dlo, dhi = np.percentile(bl - bg[:, None], [2.5, 97.5], axis=0)
        glo, ghi = np.percentile(bg, [2.5, 97.5])
        refs.append({"kras": k, "population": "map population", "diff12": glob, "lo": glo, "hi": ghi,
                     "n_pan": int(len(arms[PAN][1])), "n_fol": int(len(arms[FOL][1]))})
        for j, (x, y) in enumerate(grid):
            rows.append({"kras": k, "grid_i": j // GRID_N, "grid_j": j % GRID_N, "x": x, "y": y,
                         "diff12": est[j], "lo": lo[j], "hi": hi[j],
                         "excludes_zero": bool(lo[j] > 0 or hi[j] < 0),
                         "minus_group": est[j] - glob, "minus_group_lo": dlo[j], "minus_group_hi": dhi[j],
                         "differs_from_group": bool(dlo[j] > 0 or dhi[j] < 0),
                         "ess_pan": ess[PAN][j], "ess_fol": ess[FOL][j],
                         "shown": bool(min(ess[PAN][j], ess[FOL][j]) >= MIN_ESS)})
    pop["axis_sd"] = {"pc1": float(scale[0]), "pc2": float(scale[1])}
    return pd.DataFrame(rows), pd.DataFrame(refs), pop


def itt_km_reference(s: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    out = []
    for k in ("Wild-type", "Mutant"):
        g = {a: s[s["kras"].eq(k) & s["arm"].eq(a)] for a in (PAN, FOL)}

        def km(df: pd.DataFrame) -> float:
            return weighted_km(df["days"].to_numpy(float), df["event"].to_numpy(int), np.ones(len(df)), HORIZON_DAYS)
        est = km(g[PAN]) - km(g[FOL])
        boot = [km(g[PAN].iloc[rng.integers(0, len(g[PAN]), len(g[PAN]))])
                - km(g[FOL].iloc[rng.integers(0, len(g[FOL]), len(g[FOL]))]) for _ in range(BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"kras": k, "population": "intention to treat", "diff12": est, "lo": lo, "hi": hi,
                    "n_pan": int(len(g[PAN])), "n_fol": int(len(g[FOL])),
                    "pfs12_pan": km(g[PAN]), "pfs12_fol": km(g[FOL])})
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = subjects()
    visits = pd.read_csv(VISITS)
    fr = forest(s)
    surf, ref_map, pop = local_surface(s, visits)
    ref = pd.concat([itt_km_reference(s), ref_map], ignore_index=True)
    fr.to_csv(OUT / "prime_forest_hr_itt935.csv", index=False)
    surf.to_csv(OUT / "prime_local_km12_surface.csv", index=False)
    ref.to_csv(OUT / "prime_km12_reference.csv", index=False)
    shown = surf[surf["shown"]]
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "parameters": {"kras_column": KRAS_COLUMN, "kras_label": KRAS_LABEL, "endpoint":
                       "PFS central review (PFSDYCR, PFSCR: progression or death)", "month_days": MONTH_DAYS,
                       "horizon_days": HORIZON_DAYS, "baseline_max_day": BASELINE_MAX_DAY,
                       "bandwidth_sd": BANDWIDTH_SD, "min_ess": MIN_ESS, "grid": f"{GRID_N}x{GRID_N}",
                       "grid_quantiles": GRID_Q, "bootstrap": BOOT, "seed": SEED,
                       "zero_day_records": "kept at time 0 (censored); HR identical to exclusion (asserted)"},
        "cohort": {"randomised": int(len(s)), "arms": s["arm"].value_counts().to_dict(),
                   "kras": s["kras"].value_counts().to_dict(), "zero_day_pfs": int(s["days"].le(0).sum()),
                   "zero_day_pfs_events": int((s["days"].le(0) & s["event"].eq(1)).sum()),
                   "map_population": pop},
        "summary": {"grid_points_shown": {k: int(v) for k, v in shown.groupby("kras").size().items()},
                    "shown_excluding_zero": int(shown["excludes_zero"].sum()),
                    "shown_differing_from_group": int(shown["differs_from_group"].sum()),
                    "diff12_range_shown": {k: [float(g.diff12.min()), float(g.diff12.max())]
                                           for k, g in shown.groupby("kras")}},
        "inputs": {str(VISITS.relative_to(ROOT)): sha256(VISITS),
                   **{p.name: sha256(p) for p in sorted(TRIAL.glob("*.sas7bdat"))
                      if p.name.split("_")[0] in {"adsl", "biomark"}}},
        "outputs": {p.name: sha256(OUT / p.name) for p in (OUT / "prime_forest_hr_itt935.csv",
                    OUT / "prime_local_km12_surface.csv", OUT / "prime_km12_reference.csv")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("cohort", "summary")}, indent=2, default=str))
    print(fr.round(3).to_string())
    print(ref.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
