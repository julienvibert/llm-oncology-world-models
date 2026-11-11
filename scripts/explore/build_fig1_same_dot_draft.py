#!/usr/bin/env python3
"""Same state, different history: two patients who look identical today are not the same patient
(first version; the analysis is build_fig1_same_dot_v2.py, which imports helpers from here).

One coordinate system for the whole paper: the joint clinical state space, PC1 x PC2 of a
single combined PCA over tumour AND host features (no a-priori split). A patient is a DOT
moving through it. History is not an axis — it is the TRAIL the dot came in on.

Panel A  metastatic lung adenocarcinoma trajectories. Many patients pass through the same
         small region of state space — the same chart today. Their trails differ; their
         futures diverge. Trails grey, futures coloured by 12-month outcome.
Panel B  the synthetic quantification inside that matched region: 12-month death split by
         whether the trail contains an earlier progression.
Panel C  the real-data anchor: the same contrast measured in GENIE-BPC NSCLC at matched
         current state, matched recent course and matched time since diagnosis
         (read from the committed GENIE readout; not recomputed here).

Panel C is real-patient derived and aggregate-only. Outputs gitignored.
Run: uv run python scripts/explore/build_fig1_same_dot_draft.py
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
from build_phase_plane_grammar_draft import INK, MUTED, fit_combined_axis  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402

OUT = ROOT / "outputs" / "exploratory_fig1_same_dot"
GENIE_READOUT = ROOT / "outputs" / "exploratory_memory_kernel_genie" / "readout.json"

DIED, ALIVE = "#c0392b", "#2a7ab0"
TUMOUR = None             # None = auto-pick the code with the cleanest convergence
TRAIL_MONTHS = 9.0        # how much past to draw behind the dot
FUTURE_MONTHS = 12.0      # how much future to draw ahead of it
MIN_TRAIL = 6.0           # a patient must have this much observed past to be drawn
DEEP_WINDOW = 3.0         # "earlier progression" means older than this
GRID = 5                  # search grid for the busiest region of state space
BAND = (0.35, 0.65)       # per-tumour matched band for the pooled quantification
SEED = 20260728


def progression_times(events: pd.DataFrame) -> dict[str, list[float]]:
    e = events[events["event_type"].astype(str).eq("progression")].copy()
    e["m"] = pd.to_numeric(e["days_from_diagnosis"], errors="coerce") / 30.4
    return {k: sorted(v) for k, v in e.dropna(subset=["m"]).groupby("_gid")["m"].apply(list).items()}


def find_matched_cell(s: pd.DataFrame) -> tuple[float, float, float, float]:
    """The small region of state space visited by the most distinct patients."""
    elig = s[s["month"].ge(MIN_TRAIL) & ~s["terminal"]]
    xe = np.linspace(elig["sev"].quantile(0.05), elig["sev"].quantile(0.95), GRID + 1)
    ye = np.linspace(elig["pc2"].quantile(0.05), elig["pc2"].quantile(0.95), GRID + 1)
    best, box = -1, None
    for i in range(GRID):
        for j in range(GRID):
            m = (elig["sev"].between(xe[i], xe[i + 1]) & elig["pc2"].between(ye[j], ye[j + 1]))
            n = elig.loc[m, "_gid"].nunique()
            if n > best:
                best, box = n, (xe[i], xe[i + 1], ye[j], ye[j + 1])
    return box


def collect(s: pd.DataFrame, box: tuple[float, float, float, float],
            prog: dict[str, list[float]]) -> list[dict]:
    """One entry per patient passing through the box: trail, future, outcome, history."""
    x0, x1, y0, y1 = box
    out = []
    for gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values(["month", "snapshot_index"])
        hit = g[g["sev"].between(x0, x1) & g["pc2"].between(y0, y1)
                & g["month"].ge(MIN_TRAIL) & ~g["terminal"]]
        if hit.empty:
            continue
        t = float(hit["month"].iloc[0])
        trail = g[g["month"].between(t - TRAIL_MONTHS, t)]
        future = g[g["month"].between(t, t + FUTURE_MONTHS)]
        if len(trail) < 2 or len(future) < 2:
            continue
        y = hit["future_death_12m"].iloc[0]
        if not np.isfinite(y):
            continue
        pts = prog.get(gid, [])
        out.append({
            "gid": gid, "t": t, "trail": trail, "future": future, "died": bool(y),
            "deep_prog": bool(any(p <= t - DEEP_WINDOW for p in pts)),
            "trail_len": float(trail["month"].max() - trail["month"].min()),
            "n_prog": int(sum(1 for p in pts if p <= t)),
        })
    return out


def pooled_matched_contrast(s: pd.DataFrame, prog: dict[str, list[float]]) -> dict:
    """Within each tumour type: matched current state (PC1+PC2 band) and matched time since
    diagnosis, one visit per patient, split by progression older than the momentum window."""
    rows = []
    for code, sub in s.groupby("tumor_oncotree_code", sort=False):
        elig = sub[sub["month"].ge(MIN_TRAIL) & ~sub["terminal"]]
        if elig["_gid"].nunique() < 5:
            continue
        xlo, xhi = elig["sev"].quantile(BAND)
        ylo, yhi = elig["pc2"].quantile(BAND)
        cell = elig[elig["sev"].between(xlo, xhi) & elig["pc2"].between(ylo, yhi)]
        if cell.empty:
            continue
        mlo, mhi = cell["month"].quantile([0.10, 0.90])
        cell = cell[cell["month"].between(mlo, mhi)]
        for gid, g in cell.groupby("_gid", sort=False):     # ONE visit per patient
            r = g.sort_values("month").iloc[0]
            y = r["future_death_12m"]
            if not np.isfinite(y):
                continue
            pts = prog.get(gid, [])
            rows.append({"code": code, "died": bool(y),
                         "deep": bool(any(p <= r["month"] - DEEP_WINDOW for p in pts))})
    d = pd.DataFrame(rows)
    out = {}
    for flag, label in ((False, "no earlier\nprogression"), (True, "earlier\nprogression")):
        sel = d[d["deep"].eq(flag)]
        out[label] = {"n": int(len(sel)), "death_12m_pct": float(100 * sel["died"].mean())}
    out["_n_codes"] = int(d["code"].nunique())
    return out


def pick_exemplars(entries: list[dict], k: int = 6) -> list[dict]:
    """Readable exemplars: the ones whose futures end furthest apart, balanced by outcome."""
    def endpoint(e):
        return np.hypot(e["future"]["sev"].iloc[-1] - e["future"]["sev"].iloc[0],
                        e["future"]["pc2"].iloc[-1] - e["future"]["pc2"].iloc[0])
    died = sorted([e for e in entries if e["died"]], key=endpoint, reverse=True)
    alive = sorted([e for e in entries if not e["died"]], key=endpoint, reverse=True)
    return died[: k // 2] + alive[: k // 2]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    s = enrich_snapshots(tables)
    s, _p, _sc, _sg = fit_combined_axis(s)
    prog = progression_times(tables.events)
    pooled = pooled_matched_contrast(s, prog)

    codes = [TUMOUR] if TUMOUR else list(s["tumor_oncotree_code"].astype(str).unique())
    best = (None, None, [])
    for code in codes:
        sub = s[s["tumor_oncotree_code"].astype(str).eq(code)].copy()
        if sub["_gid"].nunique() < 10:
            continue
        b = find_matched_cell(sub)
        ent = collect(sub, b, prog)
        mixed = min(sum(e["died"] for e in ent), sum(not e["died"] for e in ent))
        if mixed >= 2 and len(ent) > len(best[2]):       # need both outcomes to be shown
            best = (code, b, ent)
    tumour, box, entries = best
    if tumour is None:
        raise SystemExit("no tumour type gave a usable convergence cell")
    s = s[s["tumor_oncotree_code"].astype(str).eq(tumour)].copy()

    fig = plt.figure(figsize=(16.0, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 0.75, 0.75], wspace=0.30)

    # ---------------- Panel A: the state space
    ax = fig.add_subplot(gs[0, 0])
    pts = pd.concat([e["trail"] for e in entries] + [e["future"] for e in entries])
    xlim = (pts["sev"].quantile(0.01) - 0.25, pts["sev"].quantile(0.99) + 0.25)
    ylim = (pts["pc2"].quantile(0.01) - 0.25, pts["pc2"].quantile(0.99) + 0.25)
    bg = s[~s["terminal"]]
    bg = bg[bg["sev"].between(*xlim) & bg["pc2"].between(*ylim)]
    ax.scatter(bg["sev"], bg["pc2"], s=4, color="#dde3ea", alpha=0.55, edgecolors="none", zorder=1)

    for e in entries:                                   # the honest fan: everyone, faint
        ax.plot(e["trail"]["sev"], e["trail"]["pc2"], color=MUTED, lw=0.5, alpha=0.20, zorder=2)
        ax.plot(e["future"]["sev"], e["future"]["pc2"],
                color=DIED if e["died"] else ALIVE, lw=0.5, alpha=0.22, zorder=2)

    for e in pick_exemplars(entries):                   # readable exemplars, bold
        col = DIED if e["died"] else ALIVE
        ax.plot(e["trail"]["sev"], e["trail"]["pc2"], color="#6b7480", lw=1.8, alpha=0.9,
                ls=(0, (4, 2)), zorder=4)
        ax.scatter(e["trail"]["sev"].iloc[0], e["trail"]["pc2"].iloc[0], s=30,
                   facecolor="white", edgecolor="#6b7480", lw=1.4, zorder=5)
        ax.plot(e["future"]["sev"], e["future"]["pc2"], color=col, lw=2.4, alpha=0.95, zorder=4)
        ax.annotate("", xy=(e["future"]["sev"].iloc[-1], e["future"]["pc2"].iloc[-1]),
                    xytext=(e["future"]["sev"].iloc[-2], e["future"]["pc2"].iloc[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4), zorder=5)

    cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    ax.scatter([cx], [cy], s=260, facecolor="white", edgecolor=INK, lw=2.2, zorder=6)
    ax.annotate(f"the same chart today\n({len(entries)} patients)", xy=(cx, cy),
                xytext=(cx + 0.10 * (xlim[1] - xlim[0]), cy + 0.34 * (ylim[1] - ylim[0])),
                fontsize=10.5, color=INK, weight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3), zorder=7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("clinical state — PC1  (overall severity)", fontsize=10)
    ax.set_ylabel("clinical state — PC2", fontsize=10)
    ax.set_title(f"A · {tumour}: same dot, different trail, different future",
                 fontsize=11.5, color=INK)
    ax.plot([], [], color="#6b7480", lw=1.8, ls=(0, (4, 2)), label="how they got here (past 9 months)")
    ax.plot([], [], color=DIED, lw=2.4, label="died within 12 months")
    ax.plot([], [], color=ALIVE, lw=2.4, label="alive at 12 months")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    # ---------------- Panel B: synthetic quantification in the matched cell
    ax = fig.add_subplot(gs[0, 1])
    grp = {k: v for k, v in pooled.items() if not k.startswith("_")}
    vals = [grp[k]["death_12m_pct"] for k in grp]
    ns = [grp[k]["n"] for k in grp]
    ax.bar(range(2), vals, color=[ALIVE, DIED], width=0.55)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 1.5, f"{v:.0f}%\n{n} patients", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(2))
    ax.set_xticklabels(list(grp), fontsize=9)
    ax.set_ylabel("died within 12 months (%)", fontsize=10)
    ax.set_ylim(0, max([v for v in vals if np.isfinite(v)]) * 1.32)
    ax.set_title(f"B · matched state, matched time,\nall {pooled['_n_codes']} tumour types (simulated)",
                 fontsize=11, color=INK)

    # ---------------- Panel C: the real anchor
    ax = fig.add_subplot(gs[0, 2])
    real = json.loads(GENIE_READOUT.read_text())["recursive_test"]["+ time-since-dx matched"]
    rv = [real["no prior progression"]["death_12m"] * 100, real["prior progression"]["death_12m"] * 100]
    rn = [real["no prior progression"]["n_patients"], real["prior progression"]["n_patients"]]
    ax.bar(range(2), rv, color=[ALIVE, DIED], width=0.55)
    for i, (v, n) in enumerate(zip(rv, rn)):
        ax.text(i, v + 0.6, f"{v:.1f}%\n{n} patients", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(2))
    ax.set_xticklabels(list(grp), fontsize=9)
    ax.set_ylabel("died within 12 months (%)", fontsize=10)
    ax.set_ylim(0, max(rv) * 1.38)
    ax.set_title("C · the same contrast in real patients\nGENIE-BPC NSCLC", fontsize=11, color=INK)

    for a in fig.get_axes():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same recorded state, different prior path, different future",
                 fontsize=13.5, color=INK, y=1.01)
    fig.savefig(OUT / "fig1_same_dot.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1_same_dot.pdf", bbox_inches="tight")
    plt.close(fig)

    return {"tumour": tumour, "n_patients_in_cell": len(entries), "pooled": pooled,
            "cell_pc1": [box[0], box[1]], "cell_pc2": [box[2], box[3]],
            "synthetic": grp,
            "real_genie": {"no prior progression": rv[0], "prior progression": rv[1],
                           "n_patients": rn}}


def main() -> int:
    rep = build()
    lines = [
        "# Figure 1 draft — same dot, different trail", "",
        f"Tumour: {rep['tumour']}. Matched region of state space: "
        f"PC1 {rep['cell_pc1'][0]:.2f}..{rep['cell_pc1'][1]:.2f}, "
        f"PC2 {rep['cell_pc2'][0]:.2f}..{rep['cell_pc2'][1]:.2f}.",
        f"Patients passing through it: {rep['n_patients_in_cell']}.", "",
        "## Panel B — simulated, inside the matched region",
    ]
    for k, v in rep["synthetic"].items():
        lines.append(f"- {k.replace(chr(10), ' ')}: {v['death_12m_pct']:.1f}% died within "
                     f"12 months (n={v['n']})")
    lines += ["", "## Panel C — real GENIE-BPC NSCLC anchor",
              f"- no prior progression: {rep['real_genie']['no prior progression']:.1f}%",
              f"- prior progression: {rep['real_genie']['prior progression']:.1f}%",
              f"- patients: {rep['real_genie']['n_patients']}"]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")
    (OUT / "readout.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
