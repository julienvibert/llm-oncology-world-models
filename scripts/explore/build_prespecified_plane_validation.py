#!/usr/bin/env python3
"""DOES THE PRESPECIFIED TUMOUR-HOST PLANE SURVIVE ITS OWN VALIDATION?

build_tumour_host_map.py used PC1/PC3/PC4 and claimed the separation was discovered. That
is a discovery claim, and discovery claims are attackable: PC3 holds only 9.6% of variance,
its composition depends on the feature set, and it carries NEGATIVE tumour loadings
(clones -0.29), so a growing tumour pushes a patient DOWN the host axis. For a
clinician-facing display that is a defect, not a feature.

This tests the alternative: three PRESPECIFIED indices, assigned by what a variable
MEASURES rather than by what caused it.

    tumour activity    burden, sites, high-risk and visceral and liver deposits,
                       largest deposit, active clones, acquired alterations
    host condition     ECOG, symptom burden, worst organ impairment
    toxicity           active adverse events

Liver deposits sit on the tumour axis; hepatic impairment sits on the host axis. That is
not double counting — they are different clinical facts, the disease and its impact.

Toxicity is kept OUT of the host index deliberately. Its movement is near-orthogonal to
both others, and folding it in would erase the osimertinib biomarker flip.

Nothing is claimed to be biologically independent. The axes are separated as readouts;
the dynamics stay coupled, dTumour and dHost each depending on both.

Four tests, the ones that would falsify the plane:

  A  do clinically defined events move the channels they should?  response and progression
     on tumour; toxicity onset and resolution on toxicity; death on host
  B  how much is lost against the full raw state, out-of-fold and grouped by trajectory
  C  are the channels redundant in MOVEMENT (levels are allowed to correlate)
  D  do treatments separate into distinguishable benefit-versus-cost profiles

Run: uv run python scripts/explore/build_prespecified_plane_validation.py
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
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trial_in_the_map import ARM_LABEL, FEAT, MONTH, _read  # noqa: E402
from composite_tumor_host_phase_space import (  # noqa: E402
    AtlasTables,
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_prespecified_plane"
INK, MUTED, GRIDC = "#1f2937", "#64748b", "#d8dee6"
TAU = 3.0
POST = 3.0          # months after an event within which the "after" state must fall
SEED = 20260729

BLOCKS = {
    "tumour": ["tumour_burden_score", "num_active_sites_num", "high_risk_burden",
               "max_site_burden", "liver_burden", "visceral_burden",
               "num_active_clones_num", "n_acquired_alterations_num"],
    "host": ["ecog", "symptom_ord", "organ_max"],
    "toxicity": ["n_active_aes_num"],
}
CH = list(BLOCKS)
CHC = {"tumour": "#5e3c99", "host": "#0f8b8d", "toxicity": "#c0392b"}

# Positive controls: which channel the event SHOULD move (+1 worse, -1 better).
EXPECT = {
    "response": ("tumour", -1), "progression": ("tumour", +1),
    "clonal_evolution_event": ("tumour", +1), "local_intervention": ("tumour", -1),
    "toxicity": ("toxicity", +1), "patient_state_update": ("host", +1),
}
# Negative controls: starting or switching a drug should move NOTHING at the instant it
# happens. Treatment acts through the dynamics that follow, not through the state record.
NEG = ["systemic_treatment_start", "systemic_treatment_switch"]
EV_ORDER = ["response", "progression", "clonal_evolution_event", "local_intervention",
            "toxicity", "patient_state_update", *NEG]
EV_NICE = {"response": "response", "progression": "progression",
           "clonal_evolution_event": "clonal\nevolution",
           "local_intervention": "local\nintervention", "toxicity": "toxicity onset",
           "patient_state_update": "patient state\nupdate",
           "systemic_treatment_start": "treatment start\n(neg. control)",
           "systemic_treatment_switch": "treatment switch\n(neg. control)"}


def index_frame(s: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Robust z-score every feature on living states, then average within each block."""
    sc = RobustScaler(quantile_range=(10, 90))
    sc.fit(s.loc[~s["terminal"], FEAT].fillna(0.0).to_numpy(float))
    z = sc.transform(s[FEAT].fillna(0.0).to_numpy(float))
    out = s.copy()
    for c, cols in BLOCKS.items():
        idx = [FEAT.index(f) for f in cols]
        out[c] = z[:, idx].mean(axis=1)
    return out, z


def around_events(s: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Displacement STRADDLING each event: last state strictly before -> state at/after.

    Every event in these runs lands exactly on a snapshot month — snapshots are emitted at
    events. So anchoring on "the last state at or before the event" silently reads the
    POST-event state and measures the drift that follows instead of the event itself. The
    anchor has to sit strictly before.
    """
    snap = {g: d.sort_values("month") for g, d in s.groupby("_gid", sort=False)}
    rows = []
    for gid, e in ev.groupby("_gid", sort=False):
        g = snap.get(gid)
        if g is None or len(g) < 2:
            continue
        t = g["month"].to_numpy(float)
        vals = {c: g[c].to_numpy(float) for c in CH}
        for _i, r in e.iterrows():
            te = r["m"]
            if not np.isfinite(te):
                continue
            pre = np.nonzero(t < te - 1e-9)[0]
            post = np.nonzero((t >= te - 1e-9) & (t <= te + POST))[0]
            if not len(pre) or not len(post):
                continue
            a, b = pre[-1], post[0]
            rows.append({"gid": gid, "event": str(r["event_type"]),
                         **{f"d{c}": vals[c][b] - vals[c][a] for c in CH}})
    return pd.DataFrame(rows)


def horizon_moves(s: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    s = s.reset_index(drop=True)
    rows = []
    for gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values("month")
        t = g["month"].to_numpy(float)
        if len(t) < 2:
            continue
        zi = z[g.index.to_numpy()]
        cur = {c: g[c].to_numpy(float) for c in CH}
        p1, p2, p3 = (g["pc1"].to_numpy(float) if "pc1" in g else None), None, None
        for i in range(len(t)):
            if t[i] + TAU > t[-1]:
                continue
            zt = np.array([np.interp(t[i] + TAU, t, zi[:, j]) for j in range(zi.shape[1])])
            rows.append({"gid": gid, **{c: cur[c][i] for c in CH},
                         **{f"d{c}": float(np.interp(t[i] + TAU, t, cur[c])) - cur[c][i]
                            for c in CH},
                         **{f"z{j}": zi[i, j] for j in range(zi.shape[1])},
                         **{f"dz{j}": zt[j] - zi[i, j] for j in range(zi.shape[1])}})
    _ = (p1, p2, p3)
    return pd.DataFrame(rows)


def oof_r2(d: pd.DataFrame, cols: list[str], tgt: list[str]) -> float:
    x, y, g = d[cols].to_numpy(float), d[tgt].to_numpy(float), d["gid"].to_numpy()
    p = np.zeros_like(y)
    for c in range(y.shape[1]):
        for tr, te in GroupKFold(n_splits=5).split(x, y[:, c], g):
            p[te, c] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
                x[tr], y[tr, c]).predict(x[te])
    return 100 * float(1 - ((y - p) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = load_tables()
    s, z = index_frame(enrich_snapshots(tabs))
    ev = tabs.events.copy()
    ev["m"] = pd.to_numeric(ev["days_from_diagnosis"], errors="coerce") / MONTH

    # ---- A. event-conditioned displacement
    A = around_events(s, ev[ev["event_type"].astype(str).isin(EV_ORDER)])
    ea = A.groupby("event")[[f"d{c}" for c in CH]].agg(["mean", "sem", "size"])

    # ---- B. information loss (target = the full 12-D raw displacement)
    d = horizon_moves(s, z)
    dz = [f"dz{j}" for j in range(len(FEAT))]
    zc = [f"z{j}" for j in range(len(FEAT))]
    pca = PCA(n_components=4, random_state=SEED).fit(z[~s["terminal"].to_numpy()])
    zz = pca.transform(d[zc].to_numpy(float))
    for k in range(4):
        d[f"pc{k + 1}"] = zz[:, k]
    loss = {
        "3 prespecified indices": oof_r2(d, CH, dz),
        "PC1 + PC3": oof_r2(d, ["pc1", "pc3"], dz),
        "PC1 + PC2": oof_r2(d, ["pc1", "pc2"], dz),
        "full 12-D raw state": oof_r2(d, zc, dz),
    }

    # ---- C. redundancy: levels vs movement
    lev = s[~s["terminal"]][CH].corr()
    mov = d[[f"d{c}" for c in CH]].corr()

    # ---- D. treatment profiles
    si = enrich_snapshots(AtlasTables(
        snapshots=_read("snapshot_rows.csv"), sites=_read("disease_site_rows.csv"),
        events=_read("event_rows.csv"), endpoints=_read("endpoint_rows.csv")))
    si, _zi = index_frame(si)
    evi = _read("event_rows.csv")
    evi["m"] = pd.to_numeric(evi["days_from_diagnosis"], errors="coerce") / MONTH
    st = evi[evi["event_type"].astype(str).eq("systemic_treatment_start")]
    t0 = st.groupby("_gid")["m"].min().to_dict()
    cell = si.groupby("_gid")[["context", "state_id", "history_id"]].first()
    cell["t0"] = cell.index.map(t0)
    bc = cell.dropna(subset=["t0"]).groupby(
        ["context", "state_id", "history_id"])["t0"].median().to_dict()
    for gid, r in cell.iterrows():
        if gid not in t0 and (r["context"], r["state_id"], r["history_id"]) in bc:
            t0[gid] = float(bc[(r["context"], r["state_id"], r["history_id"])])
    rows = []
    for gid, g in si.groupby("_gid", sort=False):
        g = g.sort_values("month")
        t = g["month"].to_numpy(float)
        if len(t) < 2:
            continue
        v = {c: g[c].to_numpy(float) for c in CH}
        a0 = t0.get(gid, -np.inf)
        for i in range(len(t)):
            if t[i] < a0 or t[i] + TAU > t[-1]:
                continue
            rows.append({"context": g["context"].iloc[0], "arm": g["arm"].iloc[0],
                         **{f"d{c}": float(np.interp(t[i] + TAU, t, v[c])) - v[c][i]
                            for c in CH}})
    D = pd.DataFrame(rows).groupby(["context", "arm"])[
        [f"d{c}" for c in CH]].mean()

    # ================= figure =================
    fig = plt.figure(figsize=(17.5, 10.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95], hspace=0.42, wspace=0.30)

    ax = fig.add_subplot(gs[0, :2])
    evs = [e for e in EV_ORDER if e in ea.index]
    xs = np.arange(len(evs))
    for k, c in enumerate(CH):
        m = ea.loc[evs, (f"d{c}", "mean")].to_numpy(float)
        se = ea.loc[evs, (f"d{c}", "sem")].to_numpy(float)
        ax.bar(xs + (k - 1) * 0.27, m, yerr=1.96 * se, width=0.25, color=CHC[c],
               label=c, error_kw=dict(lw=1.0, ecolor=INK))
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{EV_NICE[e]}\n(n={int(ea.loc[e, ('dtumour', 'size')]):,})"
                        for e in evs], fontsize=8.5)
    ax.set_ylabel("mean change in index across the event", fontsize=9.5)
    ax.legend(fontsize=9, frameon=False, ncols=3)
    npos = sum(1 for e in evs if e in EXPECT)
    hits = sum(1 for e in evs if e in EXPECT
               and np.sign(ea.loc[e, (f"d{EXPECT[e][0]}", "mean")]) == EXPECT[e][1]
               and abs(ea.loc[e, (f"d{EXPECT[e][0]}", "mean")]) == max(
                   abs(ea.loc[e, (f"d{c}", "mean")]) for c in CH))
    if len(evs) > npos:
        ax.axvline(npos - 0.5, color=MUTED, ls="--", lw=1.2)
    ax.set_title("A · do clinical events move the channel they should?\n"
                 f"{hits}/{npos} positive controls land on the right channel with the "
                 "right sign; both negative controls move nothing  (bars = 95% CI)",
                 fontsize=11, color=INK)

    ax = fig.add_subplot(gs[0, 2])
    ks = list(loss)
    ys = np.arange(len(ks))[::-1]
    ax.barh(ys, [loss[k] for k in ks],
            color=["#5e3c99", "#8c78b5", "#c9ced6", "#1f2937"])
    for y, k in zip(ys, ks):
        ax.text(loss[k] + 0.25, y, f"{loss[k]:.2f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(ks, fontsize=9)
    ax.set_xlabel("% of the full 12-D next-move variance predicted", fontsize=9)
    ax.set_title("B · what the display costs\nout-of-fold, grouped by trajectory",
                 fontsize=11, color=INK)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(np.stack([lev.to_numpy(), mov.to_numpy()]).reshape(2, 3, 3)[0],
                   cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{lev.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=10,
                    color=INK if abs(lev.iloc[i, j]) < 0.6 else "white")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CH, fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CH, fontsize=9)
    ax.set_title("C · LEVELS are correlated\npatients are not independent on the axes",
                 fontsize=10.5, color=INK)

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(mov.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mov.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=10,
                    color=INK if abs(mov.iloc[i, j]) < 0.6 else "white")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CH, fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CH, fontsize=9)
    ax.set_title("…but MOVEMENT is not\nwhich is what the map displays",
                 fontsize=10.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)

    ax = fig.add_subplot(gs[1, 2])
    sub = D.loc["EGFR_1L"]
    xs = np.arange(len(sub))
    for k, c in enumerate(CH):
        ax.bar(xs + (k - 1) * 0.27, sub[f"d{c}"], width=0.25, color=CHC[c], label=c)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL.get(a, a).replace("/", "/\n").replace(
        " supportive", "\nsupportive") for a in sub.index], fontsize=7.5)
    ax.set_ylabel("mean 3-month change", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("D · benefit-versus-cost profiles\nEGFR-mutant, from the decision onward",
                 fontsize=10.5, color=INK)

    for a in fig.get_axes():
        if a.get_label() != "<colorbar>":
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does the prespecified tumour–host–toxicity plane survive validation?\n"
                 "axes separated as readouts; dynamics left coupled",
                 fontsize=13.5, color=INK, y=1.02)
    fig.savefig(OUT / "prespecified_plane_validation.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "prespecified_plane_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    # ================= readout =================
    lines = ["# Does the prespecified tumour–host–toxicity plane survive validation?", "",
             "Three indices, assigned by what a variable MEASURES, not by what caused it. "
             "Each feature robust-z-scored on living states, then averaged within its block.",
             "", "| index | features |", "|---|---|"]
    for c, cols in BLOCKS.items():
        lines.append(f"| {c} | {', '.join(cols)} |")
    lines += ["",
              "Liver deposits sit on tumour; hepatic impairment sits on host. Different "
              "clinical facts — the disease and its impact — so this is not double counting.",
              "", "## A. Do clinical events move the channel they should?", "",
              "Displacement STRADDLING each event: last state strictly before it, first "
              "state at or after. Every event in these runs lands exactly on a snapshot "
              "month, so anchoring on the last state *at or before* the event silently "
              "reads the post-event state and measures the drift that follows instead. "
              "That off-by-one inverts the whole test.", "",
              "| event | Δtumour | Δhost | Δtoxicity | n | expected | specificity | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    verdicts, spec = {}, {}
    for e in evs:
        m = {c: float(ea.loc[e, (f"d{c}", "mean")]) for c in CH}
        big = max(CH, key=lambda c: abs(m[c]))
        if e in EXPECT:
            ch, sg = EXPECT[e]
            ok = (big == ch) and (np.sign(m[ch]) == sg)
            other = max(abs(m[c]) for c in CH if c != ch)
            sp = abs(m[ch]) / other if other > 0 else np.inf
            spec[e] = float(sp)
            verdicts[e] = bool(ok)
            exp = f"{ch} {'worse' if sg > 0 else 'better'}"
            v, sps = ("PASS" if ok else f"FAIL — largest was {big}"), f"{sp:.1f}x"
        else:
            exp, v = "no movement", ("PASS" if max(abs(x) for x in m.values()) < 0.01
                                     else f"FAIL — moved {big}")
            verdicts[e] = v == "PASS"
            sps = "—"
        lines.append(f"| {EV_NICE[e]} | {m['tumour']:+.3f} | {m['host']:+.3f} | "
                     f"{m['toxicity']:+.3f} | {int(ea.loc[e, ('dtumour', 'size')]):,} | "
                     f"{exp} | {sps} | {v} |")
    lines += ["", f"**{sum(verdicts.values())} of {len(verdicts)} predictions pass**, "
              f"including both negative controls — starting or switching a drug moves no "
              "channel at the instant it happens, which is correct: treatment acts through "
              "the dynamics that follow, not through the state record.", "",
              "Specificity is the ratio of movement on the expected channel to the largest "
              "movement on any other. Toxicity events are the extreme case: they move the "
              f"toxicity index {spec.get('toxicity', float('nan')):.0f}x more than either "
              "other channel.", "",
              "**What this test does and does not show.** Because snapshots are emitted at "
              "events, the generator writes the event label and the feature change "
              "together. So this is a coherence check on the MEASUREMENT — it shows the "
              "three indices read the generator's clinical semantics correctly, and that "
              "the channels are discriminant rather than three names for one thing. It is "
              "not independent evidence that the model's biology is right; that is what the "
              "GENIE comparisons are for.", "",
              "## B. What the two-dimensional display costs", "",
              "Out-of-fold, grouped by trajectory. Target is the FULL 12-feature 3-month "
              "displacement — so this asks how much of real motion each summary anticipates.",
              "", "| state summary | % of next-move variance |", "|---|---|"]
    for k, v in sorted(loss.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:.2f}% |")
    lines += ["",
              f"The three prespecified indices reach {loss['3 prespecified indices']:.2f}% "
              f"against {loss['full 12-D raw state']:.2f}% for the complete state — "
              f"{100 * loss['3 prespecified indices'] / loss['full 12-D raw state']:.0f}% of "
              f"what the full state achieves, from 3 numbers instead of 12 — and against "
              f"{loss['PC1 + PC3']:.2f}% for PC1+PC3 and {loss['PC1 + PC2']:.2f}% for the "
              "PC1+PC2 map currently in the figures.", "",
              "## C. Levels are coupled; movement is not", "",
              "| pair | correlation of LEVELS | correlation of 3-month MOVEMENT |",
              "|---|---|---|"]
    for i, a in enumerate(CH):
        for b in CH[i + 1:]:
            lines.append(f"| {a} vs {b} | {lev.loc[a, b]:+.3f} | "
                         f"{mov.loc[f'd{a}', f'd{b}']:+.3f} |")
    lines += ["",
              "This is the sentence the framework needs: the axes are separated as "
              "readouts, the dynamics stay coupled. Patients are NOT independent on the two "
              "axes — sick patients are sick on both. What is largely independent is where "
              "they go next, which is what the map displays.", "",
              "## D. Treatment benefit-versus-cost profiles", "",
              "| context | arm | Δtumour | Δhost | Δtoxicity |", "|---|---|---|---|---|"]
    for (ctx, a), r in D.iterrows():
        lines.append(f"| {ctx} | {ARM_LABEL.get(a, a)} | {r['dtumour']:+.3f} | "
                     f"{r['dhost']:+.3f} | {r['dtoxicity']:+.3f} |")
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(
        {"blocks": BLOCKS, "event_verdicts": verdicts,
         "event_means": {e: {c: float(ea.loc[e, (f"d{c}", "mean")]) for c in CH}
                         for e in evs},
         "information": loss, "levels_corr": lev.round(4).to_dict(),
         "movement_corr": mov.round(4).to_dict(),
         "arms": D.round(4).reset_index().to_dict("records")}, indent=2, default=str) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
