#!/usr/bin/env python3
"""THE TRELLIS — one state space, many conditional fields, laid out as small multiples.

The conditioning explosion is a layout problem, not a modelling problem. Rows are tumour
families, columns are treatment epochs (before systemic therapy, 1st line, 2nd line,
3rd line and beyond). Every panel is the SAME map with the SAME axes and the SAME arrow
scale, so panels can be read against each other.

Rules, all of them load-bearing:
  * arrows are FIXED-HORIZON displacements x(t+tau) - x(t), tau = 3 months, interpolated;
  * facets are STRUCTURAL (tumour, line) and never treatment — an observational atlas
    faceted by treatment gives confounded contrasts, and we have already measured that
    going the wrong way (sicker patients get treated, so the on-treatment field looks
    worse). Treatment effects come only from the counterfactual arm;
  * panels with too few transitions are drawn BLANK rather than dropped, so coverage is
    part of the result.

ATTRACTOR TEST per panel. Fit v = A(x - c) and read the eigenvalues of A. Both negative
= converging. Two controls, because the naive version is not trustworthy:
  * a permutation null (shuffle which displacement belongs to which starting state) —
    this only rules out pure noise;
  * a LAG-CORRECTED refit, where the position is measured at t but the displacement is
    measured from t+delta to t+delta+tau. Regression to the mean arises because the same
    noisy x appears in both the position and the displacement; using a later origin
    removes it. Convergence that survives the lag correction is the only kind worth
    reporting.

Run: uv run python scripts/explore/build_field_trellis.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_phase_plane_grammar_draft import fit_combined_axis  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_field_trellis"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#cbd5e1"
DOT, ARROW = "#93b8d6", "#1f2937"
TAU = 3.0
DELTA = 1.0               # lag used by the regression-to-the-mean control
MIN_PANEL = 150           # transitions needed before a panel is drawn
MIN_CELL = 12
NX, NY = 5, 4
N_PERM = 300
SEED = 20260728

FAMILY = {
    "Lung": ["LUAD", "LUSC", "SCLC", "NSCLC", "LCLC", "NSCLCPD", "LUNE", "LUAS"],
    "Breast": ["BRCA", "IDC", "ILC", "IBC", "BRCNOS", "MDLC", "BRCANOS"],
    "Gastrointestinal": ["COAD", "READ", "COADREAD", "STAD", "ESCA", "PAAD", "CHOL",
                          "HCC", "GIST", "ESCC", "GEJ", "SBC", "PANET", "PANEC", "ACC"],
    "Genitourinary": ["PRAD", "BLCA", "CCRCC", "RCC", "PRCC", "CHRCC", "UCU", "URCA",
                       "SEM", "NSGCT", "TGCT"],
    "Gynaecological": ["HGSOC", "OV", "OVT", "UCEC", "CESC", "USC", "UCS", "OCS", "VMM"],
    "Sarcoma": ["LMS", "DDLS", "WDLS", "SYNS", "ES", "OS", "MFS", "UPS", "MPNST",
                 "DES", "ASPS", "EHAE", "CHS", "CHDM", "RMS", "ERMS", "ARMS"],
    "Skin / melanoma": ["SKCM", "MEL", "ACRM", "MUP", "UM", "BCC", "CSCC", "MCC"],
    "Head & neck": ["HNSC", "OPHSC", "NPC", "HNSCUP", "OCSC", "LXSC", "HPHSC", "THPA",
                     "THFO", "THHC", "THME"],
}
CODE2FAM = {c: f for f, codes in FAMILY.items() for c in codes}

EPOCHS = [(None, "before systemic therapy"), (1, "1st line"), (2, "2nd line"), (3, "3rd line +")]


def epoch_of(v: float) -> str | None:
    if not np.isfinite(v):
        return "before systemic therapy"
    if v <= 1:
        return "1st line"
    if v == 2:
        return "2nd line"
    return "3rd line +"


def horizon_rows(s: pd.DataFrame) -> pd.DataFrame:
    """Fixed-horizon displacements, tagged with tumour family and treatment epoch.

    Also carries a lag-corrected displacement measured from t+DELTA, used to strip the
    regression-to-the-mean component out of the convergence test.
    """
    rows = []
    for _gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values(["month", "snapshot_index"])
        t = g["month"].to_numpy(float)
        x, y = g["sev"].to_numpy(float), g["pc2"].to_numpy(float)
        line = pd.to_numeric(g["active_treatment_lines_joined"], errors="coerce").to_numpy(float)
        fam = CODE2FAM.get(str(g["tumor_oncotree_code"].iloc[0]))
        if fam is None or len(t) < 2:
            continue
        for i in range(len(t)):
            if t[i] + TAU > t[-1]:
                continue
            dx = float(np.interp(t[i] + TAU, t, x)) - x[i]
            dy = float(np.interp(t[i] + TAU, t, y)) - y[i]
            lx = ly = np.nan
            if t[i] + DELTA + TAU <= t[-1]:                  # lag-corrected displacement
                lx = float(np.interp(t[i] + DELTA + TAU, t, x)) - float(np.interp(t[i] + DELTA, t, x))
                ly = float(np.interp(t[i] + DELTA + TAU, t, y)) - float(np.interp(t[i] + DELTA, t, y))
            rows.append((fam, epoch_of(line[i]), x[i], y[i], dx, dy, lx, ly))
    return pd.DataFrame(rows, columns=["family", "epoch", "x", "y", "dx", "dy", "lx", "ly"])


def stability(d: pd.DataFrame, cols=("dx", "dy")) -> dict:
    live = d.dropna(subset=list(cols))
    if len(live) < 40:
        return {"ok": False}
    design = np.column_stack([live["x"], live["y"], np.ones(len(live))])
    bx, *_ = np.linalg.lstsq(design, live[cols[0]].to_numpy(float), rcond=None)
    by, *_ = np.linalg.lstsq(design, live[cols[1]].to_numpy(float), rcond=None)
    eig = np.real(np.linalg.eigvals(np.array([[bx[0], bx[1]], [by[0], by[1]]])))
    return {"ok": True, "lam_max": float(np.max(eig)), "lam_min": float(np.min(eig)),
            "n": int(len(live))}


def permutation_p(d: pd.DataFrame, observed: float, rng: np.random.Generator) -> float:
    """Does the convergence beat pure noise?  (It does not control regression to the mean.)"""
    live = d.dropna(subset=["dx", "dy"])
    if len(live) < 40:
        return np.nan
    hits = 0
    for _ in range(N_PERM):
        perm = live.assign(dx=rng.permutation(live["dx"].to_numpy()),
                           dy=rng.permutation(live["dy"].to_numpy()))
        st = stability(perm)
        if st.get("ok") and st["lam_max"] <= observed:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    s = enrich_snapshots(load_tables())
    s, _p, _sc, _sg = fit_combined_axis(s)
    d = horizon_rows(s)

    fams = [f for f in FAMILY if (d["family"].eq(f)).sum() >= MIN_PANEL]
    epochs = [lab for _v, lab in EPOCHS]
    xlim = (float(d["x"].quantile(0.01)), float(d["x"].quantile(0.80)))
    ylim = (float(d["y"].quantile(0.01)), float(d["y"].quantile(0.95)))
    xe = np.linspace(xlim[0], xlim[1], NX + 1)
    ye = np.linspace(ylim[0], ylim[1], NY + 1)
    XC, YC = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))

    fig, axes = plt.subplots(len(fams), len(epochs),
                             figsize=(3.35 * len(epochs), 2.95 * len(fams)),
                             squeeze=False)
    report: dict = {}
    for r, fam in enumerate(fams):
        for c, ep in enumerate(epochs):
            ax = axes[r][c]
            w = d[d["family"].eq(fam) & d["epoch"].eq(ep)
                  & d["x"].between(*xlim) & d["y"].between(*ylim)]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRIDC)
            if r == 0:
                ax.set_title(ep, fontsize=10.5, color=INK)
            if c == 0:
                ax.set_ylabel(fam, fontsize=10.5, color=INK)

            if len(w) < MIN_PANEL:
                ax.text(0.5, 0.5, f"too few\n({len(w)})", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#b6bfcc")
                continue

            u = np.full((NY, NX), np.nan)
            v = np.full((NY, NX), np.nan)
            for a in range(NX):
                for b in range(NY):
                    m = w["x"].between(xe[a], xe[a + 1]) & w["y"].between(ye[b], ye[b + 1])
                    if m.sum() >= MIN_CELL:
                        u[b, a], v[b, a] = w.loc[m, "dx"].mean(), w.loc[m, "dy"].mean()
            ax.scatter(w["x"], w["y"], s=3, color=DOT, alpha=0.18, edgecolors="none")
            ax.quiver(XC, YC, u, v, color=ARROW, angles="xy", scale_units="xy", scale=1,
                      width=0.011)

            naive = stability(w)
            lagged = stability(w, cols=("lx", "ly"))
            pval = permutation_p(w, naive["lam_max"], rng) if naive.get("ok") else np.nan
            conv_naive = naive.get("ok") and naive["lam_max"] < 0
            conv_lag = lagged.get("ok") and lagged["lam_max"] < 0
            report[f"{fam} · {ep}"] = {
                "n": int(len(w)), "mean_dx": float(w["dx"].mean()),
                "lam_max": naive.get("lam_max"), "lam_max_lag": lagged.get("lam_max"),
                "perm_p": float(pval) if np.isfinite(pval) else None,
                "converging_naive": bool(conv_naive), "converging_lag_corrected": bool(conv_lag),
            }
            tag = "★ converging" if conv_lag else ("(converging, uncorrected)" if conv_naive else "")
            ax.text(0.03, 0.96, f"n={len(w):,}  Δx={w['dx'].mean():+.2f}\n{tag}",
                    transform=ax.transAxes, fontsize=7.6, va="top",
                    color="#0f8b8d" if conv_lag else MUTED)

    axes[-1][0].set_xlabel("PC1 (severity) →", fontsize=9.5)
    fig.suptitle(f"One state space, many conditional fields — {TAU:.0f}-month arrows, "
                 "shared axes and shared arrow scale\n"
                 "facets are structural (tumour family × treatment epoch), never treatment",
                 fontsize=13, color=INK, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "field_trellis.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "field_trellis.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Conditional field trellis", "",
             f"tau = {TAU:.0f} months; {len(d):,} fixed-horizon displacements; "
             f"{d['family'].nunique()} tumour families mapped.", "",
             "★ = converging after the lag correction (position at t, displacement from "
             f"t+{DELTA:.0f} to t+{DELTA + TAU:.0f}), which removes regression to the mean.", "",
             "| panel | n | mean Δ severity | λmax naive | λmax lag-corrected | perm p | converging |",
             "|---|---|---|---|---|---|---|"]
    for k, val in report.items():
        lam = f"{val['lam_max']:+.3f}" if val["lam_max"] is not None else "n/a"
        lag = f"{val['lam_max_lag']:+.3f}" if val["lam_max_lag"] is not None else "n/a"
        pv = f"{val['perm_p']:.3f}" if val["perm_p"] is not None else "n/a"
        conv = "yes" if val["converging_lag_corrected"] else (
            "uncorrected only" if val["converging_naive"] else "no")
        lines.append(f"| {k} | {val['n']:,} | {val['mean_dx']:+.3f} | {lam} | {lag} | {pv} | {conv} |")
    n_naive = sum(v["converging_naive"] for v in report.values())
    n_lag = sum(v["converging_lag_corrected"] for v in report.values())
    lines += ["", f"Panels drawn: {len(report)}. Converging before correction: {n_naive}. "
              f"After the lag correction: {n_lag}.",
              "", "The permutation null only rules out pure noise; it does not control",
              "regression to the mean, which is why the lag-corrected column is the one",
              "that decides whether a panel gets a star."]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines[:9]))
    print(f"\nPanels drawn: {len(report)}; converging naive {n_naive}; lag-corrected {n_lag}")
    print(f"wrote {OUT}/field_trellis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
