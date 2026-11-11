#!/usr/bin/env python3
"""A CLINICAL TRIAL DRAWN IN THE STATE MAP — osimertinib, EGFR-mutant and wild-type.

The first concrete application of the frozen framework
(docs/conditional_dynamics_framework_v1.md), on the locked Sonnet Phase-1 run.

Design, per the framework's critical path:
  * every arm starts from an IDENTICAL state and fixed history — 15 designed anchors per
    context (5 states x 3 histories). Trajectories are plotted as DISPLACEMENT FROM THEIR
    OWN ANCHOR, so all arms begin at the origin by construction and the fans diverge.
    That is the shared-decision-anchor comparison, drawn.
  * raw futures are visually dominant; the arm mean is a thin summary, because the field
    explains ~1.7% of individual displacement and the spread is the signal.
  * movement AND exit/residence are both reported — displacement is nearly arm-invariant
    in this system while exit hazard is not.
  * the wrong-target context is shown beside the matched one.

The map excludes cns_burden and the two duplicate organ measures. In the standard
16-feature block PC1 loads 0.996 on CNS burden and is not a severity axis
(scripts/explore/build_feature_geometry_audit.py). On the 12 features kept here PC1
loadings run 0.18-0.41 across burden, sites, clones, mutations, ECOG, symptoms and organ
function, and PC1 carries 54% of variance. Axes are fitted ONCE on the atlas and the
interventional run is projected into them.

n=3 accepted futures per designed cell, 45 per arm per context. Structure and ranking are
the readout; absolute magnitudes are not calibrated.

Run: uv run python scripts/explore/build_trial_in_the_map.py
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
from sklearn.decomposition import PCA  # noqa: E402
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

RUN = ROOT / "runs" / "interventional_phase1_sonnet_full_v1c" / "derivation"
OUT = ROOT / "outputs" / "exploratory_trial_in_the_map"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#d8dee6"
MONTH = 30.4
SEED = 20260729

DROP = {"cns_burden", "marrow_burden", "organ_sum", "organ_impaired_count"}
FEAT = [f for f in TUMOUR_FEATURES + HOST_FEATURES if f not in DROP]

ARMS = ["osimertinib", "chemo", "pembrolizumab", "supportive"]
ARM_LABEL = {"osimertinib": "osimertinib", "chemo": "carboplatin/pemetrexed",
             "pembrolizumab": "pembrolizumab", "supportive": "best supportive care"}
ARM_COLOR = {"osimertinib": "#5e3c99", "chemo": "#2a7ab0",
             "pembrolizumab": "#1b9e77", "supportive": "#9ca3af"}
CTX = [("EGFR_1L", "EGFR-mutant — osimertinib is standard of care"),
       ("EGFR_WT_1L", "EGFR wild-type — osimertinib is the wrong target")]
GRID_T = np.arange(0.0, 24.5, 0.5)


def fit_map() -> tuple[PCA, RobustScaler, float, pd.DataFrame]:
    s = enrich_snapshots(load_tables())
    fit = s[~s["terminal"]]
    sc = RobustScaler(quantile_range=(10, 90))
    x = sc.fit_transform(fit[FEAT].fillna(0.0).to_numpy(float))
    pca = PCA(n_components=3, random_state=SEED).fit(x)
    ref = (fit["tumour_burden_score"] + 0.7 * fit["num_active_sites_num"] + fit["ecog"]).to_numpy(float)
    sign = 1.0 if np.corrcoef(pca.transform(x)[:, 0], ref)[0, 1] >= 0 else -1.0
    load = pd.DataFrame({"feature": FEAT, "pc1": sign * pca.components_[0],
                         "pc2": pca.components_[1]}).sort_values("pc1", ascending=False)
    return pca, sc, sign, load


def _read(name: str) -> pd.DataFrame:
    df = accepted(pd.read_csv(RUN / name, low_memory=False))
    df["_run"] = "sonnet_egfr"
    df["_gid"] = df["_run"] + "::" + df["trajectory_id"].astype(str)
    p = df["trajectory_id"].astype(str).str.split("__", expand=True)
    df["context"], df["state_id"], df["history_id"], df["arm"] = p[0], p[1], p[2], p[3]
    return df


def load_run(pca, sc, sign) -> tuple[pd.DataFrame, dict, dict]:
    s = enrich_snapshots(AtlasTables(
        snapshots=_read("snapshot_rows.csv"), sites=_read("disease_site_rows.csv"),
        events=_read("event_rows.csv"), endpoints=_read("endpoint_rows.csv")))
    z = pca.transform(sc.transform(s[FEAT].fillna(0.0).to_numpy(float)))
    s["pc1"], s["pc2"] = sign * z[:, 0], z[:, 1]

    ev = _read("event_rows.csv")
    ev["m"] = pd.to_numeric(ev["days_from_diagnosis"], errors="coerce") / MONTH
    # Active arms carry a forced systemic_treatment_start; the supportive arm carries none
    # by design, so it inherits the SHARED cell-wide decision cutoff from the active arms
    # of the same (context, state, history) cell — the same t0 the locked scorer uses.
    st = ev[ev["event_type"].astype(str).eq("systemic_treatment_start")]
    t0 = st.groupby("_gid")["m"].min().to_dict()
    cell = s.groupby("_gid")[["context", "state_id", "history_id"]].first()
    cell["t0"] = cell.index.map(t0)
    by_cell = cell.dropna(subset=["t0"]).groupby(
        ["context", "state_id", "history_id"])["t0"].median().to_dict()
    for gid, row in cell.iterrows():
        if gid not in t0:
            key = (row["context"], row["state_id"], row["history_id"])
            if key in by_cell:
                t0[gid] = float(by_cell[key])
    prog = ev[ev["event_type"].astype(str).str.contains("progress", case=False, na=False)] \
        .dropna(subset=["m"]).groupby("_gid")["m"].apply(lambda x: sorted(x)).to_dict()
    death = s.groupby("_gid")["month"].max().to_dict()
    return s, t0, prog, death


def anchored(s: pd.DataFrame, t0: dict, prog: dict, death: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per trajectory: displacement from its own anchor, and its exit time."""
    paths, exits = [], []
    for gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values(["month", "snapshot_index"])
        a = t0.get(gid)
        if a is None:
            continue
        t = g["month"].to_numpy(float)
        x, y = g["pc1"].to_numpy(float), g["pc2"].to_numpy(float)
        if t[-1] <= a:
            continue
        x0, y0 = float(np.interp(a, t, x)), float(np.interp(a, t, y))
        after = [p for p in prog.get(gid, []) if p > a]
        tex = (min(after) if after else death.get(gid, t[-1])) - a
        rel = t - a
        m = rel >= 0
        paths.append(pd.DataFrame({
            "gid": gid, "context": g["context"].iloc[0], "arm": g["arm"].iloc[0],
            "t": rel[m], "dx": x[m] - x0, "dy": y[m] - y0}))
        exits.append({"gid": gid, "context": g["context"].iloc[0], "arm": g["arm"].iloc[0],
                      "exit_months": float(tex), "progressed": bool(after)})
    return pd.concat(paths, ignore_index=True), pd.DataFrame(exits)


def mean_path(p: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    """Survival-conditioned ensemble mean: only trajectories still on the map contribute."""
    ex = exits.set_index("gid")["exit_months"].to_dict()
    rows = []
    for tt in GRID_T:
        vals = []
        for gid, g in p.groupby("gid", sort=False):
            if ex.get(gid, 0) < tt or g["t"].max() < tt:
                continue
            vals.append((np.interp(tt, g["t"], g["dx"]), np.interp(tt, g["t"], g["dy"])))
        if len(vals) >= 5:
            v = np.array(vals)
            rows.append({"t": tt, "dx": v[:, 0].mean(), "dy": v[:, 1].mean(), "n": len(v)})
    return pd.DataFrame(rows)


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pca, sc, sign, load = fit_map()
    s, t0, prog, death = load_run(pca, sc, sign)
    paths, exits = anchored(s, t0, prog, death)

    fig = plt.figure(figsize=(17.5, 10.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 0.9], hspace=0.36, wspace=0.32)
    report: dict = {}

    lim = float(np.nanpercentile(np.abs(np.r_[paths["dx"], paths["dy"]]), 97))
    for c, (ctx, title) in enumerate(CTX):
        ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
        for arm in ARMS:
            p = paths[paths["context"].eq(ctx) & paths["arm"].eq(arm)]
            e = exits[exits["context"].eq(ctx) & exits["arm"].eq(arm)]
            for _gid, g in p.groupby("gid", sort=False):
                ax.plot(g["dx"], g["dy"], color=ARM_COLOR[arm], lw=0.7, alpha=0.20, zorder=2)
            mp = mean_path(p, e)
            if len(mp):
                ax.plot(mp["dx"], mp["dy"], color=ARM_COLOR[arm], lw=2.6, zorder=4,
                        label=f"{ARM_LABEL[arm]}  (n={e.shape[0]})")
                ax.scatter(mp["dx"].iloc[-1], mp["dy"].iloc[-1], s=55,
                           color=ARM_COLOR[arm], zorder=5, edgecolor="white", lw=1.2)
        ax.scatter([0], [0], s=220, facecolor="white", edgecolor=INK, lw=2.2, zorder=6)
        ax.text(0, -lim * 0.10, "shared decision anchor", ha="center", fontsize=9.5,
                color=INK, weight="bold")
        ax.axhline(0, color=GRIDC, lw=0.9)
        ax.axvline(0, color=GRIDC, lw=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.75, lim * 0.75)
        ax.set_xlabel("← better    change in severity (PC1)    worse →", fontsize=10)
        if c == 0:
            ax.set_ylabel("change in PC2", fontsize=10)
        ax.set_title(title, fontsize=11.5, color=INK)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    # ---- exit curves
    for c, (ctx, _t) in enumerate(CTX):
        ax = fig.add_subplot(gs[1, c])
        for arm in ARMS:
            e = exits[exits["context"].eq(ctx) & exits["arm"].eq(arm)]["exit_months"].sort_values()
            if e.empty:
                continue
            y = 1.0 - np.arange(1, len(e) + 1) / len(e)
            ax.step(np.r_[0, e], np.r_[1, y], where="post", color=ARM_COLOR[arm], lw=2.2)
            report[f"{ctx}·{arm}"] = {"n": int(len(e)), "median_exit_months": float(e.median())}
        ax.set_xlim(0, 36)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("months since the decision", fontsize=9.5)
        if c == 0:
            ax.set_ylabel("still on first line", fontsize=9.5)
        ax.set_title(f"{'EGFR-mutant' if c == 0 else 'EGFR wild-type'} · time to progression",
                     fontsize=10.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- incremental months vs BSC
    ax = fig.add_subplot(gs[1, 2])
    width, xs = 0.36, np.arange(2)
    for k, arm in enumerate(["osimertinib", "chemo"]):
        vals = []
        for ctx, _t in CTX:
            a = exits[exits["context"].eq(ctx) & exits["arm"].eq(arm)]["exit_months"].median()
            b = exits[exits["context"].eq(ctx) & exits["arm"].eq("supportive")]["exit_months"].median()
            vals.append(a - b)
        ax.bar(xs + (k - 0.5) * width, vals, width=width, color=ARM_COLOR[arm],
               label=ARM_LABEL[arm])
        for i, v in enumerate(vals):
            ax.text(xs[i] + (k - 0.5) * width, v + 0.4, f"{v:+.1f}", ha="center",
                    fontsize=9, color=INK)
        report[f"incremental_{arm}"] = [float(v) for v in vals]
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["EGFR-mutant", "EGFR wild-type"], fontsize=9.5)
    ax.set_ylabel("months gained vs\nbest supportive care", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("the trial result, in months", fontsize=10.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- what the axis means
    ax = fig.add_subplot(gs[1, 3])
    top = load.head(8)[::-1]
    ax.barh(range(len(top)), top["pc1"], color="#7f8c9b")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=7.5)
    ax.set_xlabel("PC1 loading", fontsize=9.5)
    ax.set_title(f"what the severity axis is\n(PC1 = {100 * pca.explained_variance_ratio_[0]:.0f}% "
                 "of variance)", fontsize=10.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("A clinical trial drawn in the state map — Sonnet, EGFR first line, "
                 "every arm from the same anchor\n"
                 "faint lines are individual generated futures; bold is the "
                 "survival-conditioned ensemble mean; n=3 per designed cell",
                 fontsize=13, color=INK, y=0.99)
    fig.savefig(OUT / "trial_in_the_map.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "trial_in_the_map.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# A clinical trial drawn in the state map", "",
             f"Map: {len(FEAT)} features, cns_burden and duplicate organ measures excluded; "
             f"PC1 = {100 * pca.explained_variance_ratio_[0]:.1f}% of variance, loadings "
             f"{load['pc1'].min():.2f}–{load['pc1'].max():.2f}.", "",
             "| context | arm | n | median time to progression (months) |", "|---|---|---|---|"]
    for ctx, _t in CTX:
        for arm in ARMS:
            k = f"{ctx}·{arm}"
            if k in report:
                lines.append(f"| {ctx} | {ARM_LABEL[arm]} | {report[k]['n']} | "
                             f"{report[k]['median_exit_months']:.2f} |")
    lines += ["", "## Months gained versus best supportive care", "",
              "| arm | EGFR-mutant | EGFR wild-type |", "|---|---|---|"]
    for arm in ("osimertinib", "chemo"):
        v = report[f"incremental_{arm}"]
        lines.append(f"| {ARM_LABEL[arm]} | {v[0]:+.2f} | {v[1]:+.2f} |")
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    build()
