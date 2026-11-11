#!/usr/bin/env python3
"""Exploratory: the (severity, momentum) phase plane as a single coordinate system.

Design exercise; other scripts import its combined-PCA helpers. Three questions, answered
on generated data:

  FIG C  Is the a-priori tumour/host split justified?  Fit ONE combined PCA over all
         tumour+host features and look at what PC1 actually is. If tumour and host
         features co-load on PC1, the split is not imposed.

  FIG A  Does the physics phase plane work clinically?  x = combined severity PC1
         (the DOT), y = d(severity)/dt over a trailing window (the ARROW). One point
         per patient-visit. The same-snapshot problem becomes a VERTICAL SLICE:
         chart-identical patients, stratified only by momentum.

  FIG B  Treatment as a FORCE.  In this plane a treatment does not just move position,
         it changes momentum. Finite-horizon (tau=6mo), survival-conditioned
         displacement per arm, osimertinib vs chemo, from the locked Sonnet EGFR
         Phase-1 run. Differential = the effect field.

Atlas = FIG A/C. Sonnet EGFR interventional run = FIG B.
Outputs gitignored. Run: uv run python scripts/explore/build_phase_plane_grammar_draft.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite_tumor_host_phase_space import (  # noqa: E402
    HOST_FEATURES,
    TUMOUR_FEATURES,
    AtlasTables,
    accepted,
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_phase_plane_grammar"
EGFR_RUN = ROOT / "runs" / "interventional_phase1_sonnet_full_v1c" / "derivation"

INK, MUTED, GRID = "#1f2937", "#64748b", "#e5e7eb"
PURPLE, BLUE, RED, TEAL = "#5e3c99", "#3b7dd8", "#c0392b", "#0f8b8d"
SEED = 20260728

MIN_GAP = 1.0          # months: minimum lookback to define momentum
LOOKBACK = 3.0         # months: target trailing window for momentum
MOM_CLIP = 1.5         # severity units / month, plotting clip
TAU = 6.0              # months: finite horizon for the treatment operator
TOL = 1.75             # months: tolerance when matching a snapshot to t+TAU
ARM_A, ARM_B = "osimertinib", "chemo"
CONTEXT = "EGFR_1L"

ALL_FEATURES = TUMOUR_FEATURES + HOST_FEATURES


# --------------------------------------------------------------------------- axes
def fit_combined_axis(s: pd.DataFrame) -> tuple[pd.DataFrame, PCA, RobustScaler, float]:
    """ONE PCA over tumour AND host features together — no a-priori block split."""
    fit = s[~s["terminal"]]
    scaler = RobustScaler(quantile_range=(10, 90))
    x_fit = scaler.fit_transform(fit[ALL_FEATURES].fillna(0.0).to_numpy(float))
    pca = PCA(n_components=4, random_state=SEED)
    pca.fit(x_fit)

    # orient PC1 toward severity using a pre-specified clinical index (no circularity:
    # this only fixes the SIGN, never the direction the data chose)
    ref = (
        fit["tumour_burden_score"] + 0.7 * fit["num_active_sites_num"]
        + fit["ecog"] + 0.6 * fit["organ_max"]
    ).to_numpy(float)
    sign = 1.0 if np.corrcoef(pca.transform(x_fit)[:, 0], ref)[0, 1] >= 0 else -1.0

    scores = pca.transform(scaler.transform(s[ALL_FEATURES].fillna(0.0).to_numpy(float)))
    out = s.copy()
    out["sev"] = sign * scores[:, 0]
    out["pc2"] = scores[:, 1]
    med = out.loc[~out["terminal"], "sev"].median()
    iqr = out.loc[~out["terminal"], "sev"].quantile(0.75) - out.loc[~out["terminal"], "sev"].quantile(0.25)
    out["sev"] = (out["sev"] - med) / (iqr if np.isfinite(iqr) and iqr else 1.0)
    return out, pca, scaler, sign


def add_momentum(s: pd.DataFrame) -> pd.DataFrame:
    """Momentum = d(severity)/dt over a trailing window ending at this visit.

    Backward-looking only: uses the observed past, never the future. Visits with no
    prior observation at least MIN_GAP months back have no defined momentum.
    """
    parts = []
    for _gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values(["month", "snapshot_index"]).copy()
        t = g["month"].to_numpy(float)
        v = g["sev"].to_numpy(float)
        mom = np.full(len(g), np.nan)
        for i in range(len(g)):
            cand = np.where(t <= t[i] - MIN_GAP)[0]
            if cand.size == 0:
                continue
            j = cand[np.argmin(np.abs(t[cand] - (t[i] - LOOKBACK)))]
            mom[i] = (v[i] - v[j]) / (t[i] - t[j])
        g["mom"] = mom
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- fig C
def fig_combined_vs_split(s: pd.DataFrame, pca: PCA, scaler: RobustScaler, sign: float) -> dict:
    load = pd.DataFrame({
        "feature": ALL_FEATURES,
        "block": ["tumour"] * len(TUMOUR_FEATURES) + ["host"] * len(HOST_FEATURES),
        "pc1": sign * pca.components_[0],
        "pc2": pca.components_[1],
    }).sort_values("pc1")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    ax = axes[0]
    colors = [PURPLE if b == "tumour" else TEAL for b in load["block"]]
    ax.barh(load["feature"], load["pc1"], color=colors)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title("PC1 loadings — ONE combined PCA\n(purple = tumour features, teal = host features)",
                 fontsize=10, color=INK)
    ax.set_xlabel("loading on PC1")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ev = pca.explained_variance_ratio_
    ax.bar(range(1, len(ev) + 1), ev * 100, color=MUTED)
    ax.set_xticks(range(1, len(ev) + 1))
    ax.set_xlabel("component")
    ax.set_ylabel("% variance")
    ax.set_title(f"PC1 alone carries {ev[0] * 100:.0f}% of state variance", fontsize=10, color=INK)

    # do tumour and host actually separate? correlation of the two block-PC1s
    ax = axes[2]
    sub = s[~s["terminal"]].sample(min(4000, (~s["terminal"]).sum()), random_state=SEED)
    tload = load.set_index("feature").loc[TUMOUR_FEATURES, "pc1"]
    hload = load.set_index("feature").loc[HOST_FEATURES, "pc1"]
    ax.scatter(sub["sev"], sub["pc2"], s=3, alpha=0.15, color=BLUE, edgecolors="none")
    ax.set_xlabel("PC1  (severity)")
    ax.set_ylabel("PC2")
    ax.set_title("state cloud in the data-driven plane\n(PC2 = the residual contrast, not 'host')",
                 fontsize=10, color=INK)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FIG C (diagnostic) — is the a-priori tumour/host split justified?",
                 fontsize=12.5, color=INK, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "figC_combined_vs_split.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    return {
        "pc1_var": float(ev[0]),
        "tumour_mean_abs_pc1": float(tload.abs().mean()),
        "host_mean_abs_pc1": float(hload.abs().mean()),
        "n_host_in_top_half": int((load["block"].eq("host") & load["pc1"].abs().ge(load["pc1"].abs().median())).sum()),
    }


# --------------------------------------------------------------------------- fig A
def fig_phase_plane(s: pd.DataFrame) -> dict:
    d = s[(~s["terminal"]) & s["mom"].notna() & s["future_death_12m"].notna()].copy()
    d["mom_p"] = d["mom"].clip(-MOM_CLIP, MOM_CLIP)

    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.8], wspace=0.30)

    # --- A1: the phase plane with a few trajectories drawn as orbits
    ax = fig.add_subplot(gs[0, 0])
    xe = np.linspace(d["sev"].quantile(0.01), d["sev"].quantile(0.99), 26)
    ye = np.linspace(-MOM_CLIP, MOM_CLIP, 22)
    risk, _, _ = np.histogram2d(d["sev"], d["mom_p"], bins=[xe, ye], weights=d["future_death_12m"])
    cnt, _, _ = np.histogram2d(d["sev"], d["mom_p"], bins=[xe, ye])
    with np.errstate(invalid="ignore"):
        rate = np.where(cnt >= 12, risk / cnt, np.nan)
    im = ax.pcolormesh(xe, ye, rate.T, cmap="RdYlBu_r", vmin=0, vmax=1, shading="flat", alpha=0.9)
    fig.colorbar(im, ax=ax, label="P(death within 12 months)", fraction=0.045)

    rng = np.random.default_rng(SEED)
    gids = rng.choice(d["_gid"].unique(), size=min(6, d["_gid"].nunique()), replace=False)
    for gid in gids:
        g = d[d["_gid"].eq(gid)].sort_values("month")
        if len(g) < 4:
            continue
        ax.plot(g["sev"], g["mom_p"], color=INK, lw=1.0, alpha=0.55, zorder=3)
        ax.scatter(g["sev"].iloc[-1], g["mom_p"].iloc[-1], s=18, color=INK, zorder=4)

    ax.axhline(0, color=INK, lw=0.9, ls="--", alpha=0.7)
    s0 = float(d["sev"].median())
    half = 0.35
    ax.axvspan(s0 - half, s0 + half, color="#111827", alpha=0.10, zorder=2)
    ax.text(s0, MOM_CLIP * 0.93, "same snapshot", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("DOT — clinical severity  (combined PC1)")
    ax.set_ylabel("ARROW — momentum  d(severity)/dt  [per month]")
    ax.set_title("A1 · the clinical phase plane\ntrajectories are orbits; a chart records only the x-coordinate",
                 fontsize=10.5, color=INK)
    ax.text(d["sev"].quantile(0.02), -MOM_CLIP * 0.88, "improving", fontsize=9, color=TEAL, style="italic")
    ax.text(d["sev"].quantile(0.02), MOM_CLIP * 0.78, "deteriorating", fontsize=9, color=RED, style="italic")

    # --- A2: the vertical slice = the same-snapshot problem
    ax = fig.add_subplot(gs[0, 1])
    band = d[d["sev"].between(s0 - half, s0 + half)].copy()
    q = band["mom"].quantile([1 / 3, 2 / 3]).to_numpy()
    band["tercile"] = np.digitize(band["mom"], q)
    labels = ["improving\n(lowest 3rd)", "stable\n(middle 3rd)", "deteriorating\n(highest 3rd)"]
    rates = [band.loc[band["tercile"].eq(k), "future_death_12m"].mean() for k in range(3)]
    ns = [int(band["tercile"].eq(k).sum()) for k in range(3)]
    ax.bar(labels, [r * 100 for r in rates], color=[TEAL, MUTED, RED])
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(i, r * 100 + 1.5, f"{r * 100:.0f}%\nn={n}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("12-month death (%)")
    ax.set_ylim(0, max(rates) * 100 * 1.28)
    ax.set_title("A2 · inside ONE vertical slice\nidentical chart, different arrow", fontsize=10.5, color=INK)

    # --- A3: does momentum add over the snapshot? grouped holdout
    aucs = _auc_snapshot_vs_momentum(d)
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(["dot\nonly", "dot +\narrow"], [aucs["snapshot"], aucs["with_momentum"]],
           color=[MUTED, PURPLE])
    for i, v in enumerate([aucs["snapshot"], aucs["with_momentum"]]):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_ylim(0.5, max(aucs["with_momentum"], aucs["snapshot"]) + 0.06)
    ax.set_ylabel("AUC · 12-month death")
    ax.set_title(f"A3 · the number\n(+{aucs['delta']:.3f} AUC, grouped holdout)", fontsize=10.5, color=INK)

    for a in fig.get_axes():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FIG A (draft) — the same-snapshot problem as a phase plane", fontsize=13, color=INK, y=1.02)
    fig.savefig(OUT / "figA_phase_plane.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_visits": int(len(d)), "n_traj": int(d["_gid"].nunique()),
        "slice_center": s0, "slice_rates": rates, "slice_n": ns,
        "auc_snapshot": aucs["snapshot"], "auc_with_momentum": aucs["with_momentum"],
    }


def _auc_snapshot_vs_momentum(d: pd.DataFrame) -> dict:
    sub = d.sample(min(12000, len(d)), random_state=SEED)
    y = sub["future_death_12m"].to_numpy(float)
    groups = sub["_gid"].to_numpy()
    x_snap = sub[["sev", "pc2"]].to_numpy(float)
    x_full = sub[["sev", "pc2", "mom"]].to_numpy(float)
    out = {}
    for name, x in (("snapshot", x_snap), ("with_momentum", x_full)):
        preds = np.zeros(len(sub))
        for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
            m = LogisticRegression(max_iter=2000).fit(x[tr], y[tr])
            preds[te] = m.predict_proba(x[te])[:, 1]
        out[name] = float(roc_auc_score(y, preds))
    out["delta"] = out["with_momentum"] - out["snapshot"]
    return out


# --------------------------------------------------------------------------- fig B
def load_egfr(pca: PCA, scaler: RobustScaler, sign: float, med_iqr: tuple[float, float]) -> pd.DataFrame:
    def _read(name: str) -> pd.DataFrame:
        df = accepted(pd.read_csv(EGFR_RUN / name, low_memory=False))
        df["_run"] = "sonnet_egfr"
        df["_gid"] = df["_run"] + "::" + df["trajectory_id"].astype(str)
        df["arm"] = df["trajectory_id"].astype(str).str.split("__").str[3]
        df["context"] = df["trajectory_id"].astype(str).str.split("__").str[0]
        return df

    s = enrich_snapshots(AtlasTables(
        snapshots=_read("snapshot_rows.csv"), sites=_read("disease_site_rows.csv"),
        events=_read("event_rows.csv"), endpoints=_read("endpoint_rows.csv")))
    # project onto the ATLAS axis — same coordinate system, one matrix multiply
    scores = pca.transform(scaler.transform(s[ALL_FEATURES].fillna(0.0).to_numpy(float)))
    med, iqr = med_iqr
    s["sev"] = (sign * scores[:, 0] - med) / iqr
    s["pc2"] = scores[:, 1]
    s = add_momentum(s)
    return s[s["context"].eq(CONTEXT)].copy()


def horizon_displacement(s: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Survival-conditioned displacement in (severity, momentum) over TAU months."""
    rows = []
    for _gid, g in s[s["arm"].eq(arm)].groupby("_gid"):
        g = g.sort_values(["month", "snapshot_index"])
        t = g["month"].to_numpy(float)
        x = g["sev"].to_numpy(float)
        m = g["mom"].to_numpy(float)
        death = t.max()
        for i in range(len(g)):
            if not np.isfinite(m[i]) or t[i] + TAU > death:
                continue
            j = int(np.argmin(np.abs(t - (t[i] + TAU))))
            if abs(t[j] - (t[i] + TAU)) > TOL or not np.isfinite(m[j]):
                continue
            rows.append((x[i], np.clip(m[i], -MOM_CLIP, MOM_CLIP), x[j] - x[i], np.clip(m[j], -MOM_CLIP, MOM_CLIP) - np.clip(m[i], -MOM_CLIP, MOM_CLIP)))
    return pd.DataFrame(rows, columns=["x", "y", "dx", "dy"])


def fig_force_field(s: pd.DataFrame) -> dict:
    pa, pb = horizon_displacement(s, ARM_A), horizon_displacement(s, ARM_B)
    if len(pa) < 10 or len(pb) < 10:
        return {"status": "insufficient", "n_osi": len(pa), "n_chemo": len(pb)}

    allx = np.r_[pa.x, pb.x]
    ally = np.r_[pa.y, pb.y]
    xe = np.linspace(np.percentile(allx, 3), np.percentile(allx, 97), 4)
    ye = np.linspace(np.percentile(ally, 3), np.percentile(ally, 97), 4)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])

    def binned(p: pd.DataFrame):
        u = np.full((len(yc), len(xc)), np.nan)
        v = np.full((len(yc), len(xc)), np.nan)
        n = np.zeros((len(yc), len(xc)))
        for a in range(len(xc)):
            for b in range(len(yc)):
                m = p.x.between(xe[a], xe[a + 1]) & p.y.between(ye[b], ye[b + 1])
                n[b, a] = m.sum()
                if m.sum() >= 5:
                    u[b, a], v[b, a] = p.dx[m].mean(), p.dy[m].mean()
        return u, v, n

    ua, va, na = binned(pa)
    ub, vb, nb = binned(pb)
    du, dv = ua - ub, va - vb
    X, Y = np.meshgrid(xc, yc)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (u, v, n, title, col) in zip(axes, [
        (ua, va, na, f"B1 · osimertinib  ({len(pa)} transitions)", PURPLE),
        (ub, vb, nb, f"B2 · chemotherapy  ({len(pb)} transitions)", BLUE),
        (du, dv, np.minimum(na, nb), "B3 · differential  osi − chemo", RED),
    ]):
        ax.quiver(X, Y, u, v, color=col, angles="xy", scale_units="xy", scale=1.0,
                  width=0.010, headwidth=4.0)
        ax.axhline(0, color=INK, lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("severity (combined PC1)")
        ax.set_ylabel("momentum  d(severity)/dt")
        ax.set_title(title, fontsize=10.5, color=col)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xe[0] - 0.4, xe[-1] + 0.4)
        ax.set_ylim(ye[0] - 0.25, ye[-1] + 0.25)

    fig.suptitle(f"FIG B (draft) — treatment as a FORCE: {TAU:.0f}-month survival-conditioned "
                 "displacement in the phase plane (Sonnet EGFR-mut, n=3/cell)",
                 fontsize=12.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figB_force_field.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "ok", "n_osi": int(len(pa)), "n_chemo": int(len(pb)),
        "mean_dmom_osi": float(np.nanmean(va)), "mean_dmom_chemo": float(np.nanmean(vb)),
        "mean_dsev_osi": float(np.nanmean(ua)), "mean_dsev_chemo": float(np.nanmean(ub)),
    }


# ------------------------------------------------------------------- fig B' (hazard)
def _progression_times() -> dict:
    ev = accepted(pd.read_csv(EGFR_RUN / "event_rows.csv", low_memory=False))
    ev["_gid"] = "sonnet_egfr::" + ev["trajectory_id"].astype(str)
    ev = ev[ev["event_type"].astype(str).str.contains("progress", case=False, na=False)]
    ev["m"] = pd.to_numeric(ev["days_from_diagnosis"], errors="coerce") / 30.4
    return ev.groupby("_gid")["m"].apply(lambda x: sorted(x.dropna())).to_dict()


def hazard_points(s: pd.DataFrame, arm: str, prog: dict) -> pd.DataFrame:
    """Per visit: does progression or death occur within TAU months?  This is the
    ABSORBING-EXIT risk — the thing the displacement field is blind to."""
    rows = []
    for gid, g in s[s["arm"].eq(arm)].groupby("_gid"):
        g = g.sort_values(["month", "snapshot_index"])
        t = g["month"].to_numpy(float)
        x = g["sev"].to_numpy(float)
        m = g["mom"].to_numpy(float)
        death, pts = t.max(), prog.get(gid, [])
        for i in range(len(g)):
            if not np.isfinite(m[i]):
                continue
            exit_by_tau = (
                any(t[i] < p <= t[i] + TAU for p in pts) or death <= t[i] + TAU
            )
            rows.append((x[i], np.clip(m[i], -MOM_CLIP, MOM_CLIP), float(exit_by_tau)))
    return pd.DataFrame(rows, columns=["x", "y", "e"])


def fig_hazard_field(s: pd.DataFrame) -> dict:
    prog = _progression_times()
    ha, hb = hazard_points(s, ARM_A, prog), hazard_points(s, ARM_B, prog)
    xe = np.linspace(np.percentile(np.r_[ha.x, hb.x], 3), np.percentile(np.r_[ha.x, hb.x], 97), 6)
    ye = np.linspace(np.percentile(np.r_[ha.y, hb.y], 3), np.percentile(np.r_[ha.y, hb.y], 97), 5)

    def grid(p: pd.DataFrame) -> np.ndarray:
        g = np.full((len(ye) - 1, len(xe) - 1), np.nan)
        for a in range(len(xe) - 1):
            for b in range(len(ye) - 1):
                m = p.x.between(xe[a], xe[a + 1]) & p.y.between(ye[b], ye[b + 1])
                if m.sum() >= 8:
                    g[b, a] = p.e[m].mean()
        return g

    ga, gb = grid(ha), grid(hb)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ax, (g, title, cmap, lim) in zip(axes, [
        (ga, f"B1' · osimertinib — P(exit within {TAU:.0f} mo)", "RdYlBu_r", (0, 1)),
        (gb, f"B2' · chemotherapy — P(exit within {TAU:.0f} mo)", "RdYlBu_r", (0, 1)),
        (ga - gb, "B3' · differential  osi − chemo\n(blue = osimertinib protects)", "RdBu_r", (-0.5, 0.5)),
    ]):
        im = ax.pcolormesh(xe, ye, g, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="flat")
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.axhline(0, color=INK, lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("severity (combined PC1)")
        ax.set_ylabel("momentum  d(severity)/dt")
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FIG B' (draft) — the arms differ in the HAZARD OF EXIT, not in the flow. "
                 "Same geometry, different transit time.", fontsize=12.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figB2_hazard_field.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"hazard_osi": float(ha.e.mean()), "hazard_chemo": float(hb.e.mean()),
            "n_osi": int(len(ha)), "n_chemo": int(len(hb))}


# --------------------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = enrich_snapshots(load_tables())
    s, pca, scaler, sign = fit_combined_axis(s)
    raw = pca.transform(scaler.transform(s[ALL_FEATURES].fillna(0.0).to_numpy(float)))[:, 0] * sign
    med = float(np.median(raw[~s["terminal"].to_numpy()]))
    iqr = float(np.subtract(*np.percentile(raw[~s["terminal"].to_numpy()], [75, 25])))
    s = add_momentum(s)

    c = fig_combined_vs_split(s, pca, scaler, sign)
    a = fig_phase_plane(s)
    egfr = load_egfr(pca, scaler, sign, (med, iqr))
    b = fig_force_field(egfr)
    h = fig_hazard_field(egfr)

    lines = [
        "# Phase-plane grammar — draft readout", "",
        f"Atlas visits: {a['n_visits']:,} from {a['n_traj']:,} trajectories.", "",
        "## FIG C — combined vs split axes",
        f"- PC1 explains {c['pc1_var'] * 100:.1f}% of state variance.",
        f"- mean |PC1 loading|: tumour {c['tumour_mean_abs_pc1']:.3f}, host {c['host_mean_abs_pc1']:.3f}.",
        f"- host features in the top half of |PC1 loading|: {c['n_host_in_top_half']}/{len(HOST_FEATURES)}.", "",
        "## FIG A — phase plane",
        f"- same-snapshot slice centred at severity {a['slice_center']:.2f}.",
        "- 12-month death by momentum tercile: "
        + ", ".join(f"{r * 100:.0f}% (n={n})" for r, n in zip(a["slice_rates"], a["slice_n"])),
        f"- AUC dot only {a['auc_snapshot']:.3f} -> dot+arrow {a['auc_with_momentum']:.3f}.", "",
        "## FIG B — treatment force field (displacement)", f"- {b}",
        "", "## FIG B' — treatment hazard field (absorbing exit)", f"- {h}",
    ]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
