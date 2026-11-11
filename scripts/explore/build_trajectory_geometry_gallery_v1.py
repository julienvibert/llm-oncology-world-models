#!/usr/bin/env python3
"""Build an exploratory visual gallery for trajectory geometry.

This is a design bake-off, not a paper analysis. It uses stored synthetic model
outputs where their semantics are sufficient and deterministic mock values where
the goal is to compare visual grammars rather than report a result. No model calls
and no real patient data are used.

Outputs:
  outputs/exploratory_trajectory_geometry_gallery_v1/

Run:
  uv run python scripts/explore/build_trajectory_geometry_gallery_v1.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exploratory_trajectory_geometry_gallery_v1"
SENTINEL = ROOT / "runs" / "sentinel_usecase_egfr_flip_v5_2026_07_13" / "derivation"
FAN_INTERVALS = ROOT / "outputs" / "paper_egfr_occupancy_flow" / "trajectory_intervals.csv"

EXEMPLAR_ID = "egfr_mut_luad__osimertinib__claude_sonnet5__f01"
MONTH_DAYS = 30.44
SEED = 20260819

BG = "#FAFAF7"
INK = "#17212B"
MUTED = "#63717E"
GRID = "#DCE2E7"
LIGHT = "#EFF2F3"
BLUE = "#2774AE"
ORANGE = "#D97706"
TEAL = "#16817A"
RED = "#C33A43"
MAGENTA = "#A64D79"
VIOLET = "#6656A3"
GREEN = "#3A7D44"
GOLD = "#B58B16"

EVENT_COLORS = {
    "systemic_treatment_start": BLUE,
    "systemic_treatment_switch": ORANGE,
    "systemic_treatment_modification": GOLD,
    "systemic_treatment_stop": MUTED,
    "response": TEAL,
    "progression": RED,
    "recurrence": RED,
    "toxicity": MAGENTA,
    "patient_state_update": VIOLET,
    "clonal_evolution_event": "#7A4E2D",
    "local_intervention": "#4C7A89",
    "death": INK,
}

EVENT_MARKERS = {
    "systemic_treatment_start": "s",
    "systemic_treatment_switch": "s",
    "systemic_treatment_modification": "P",
    "systemic_treatment_stop": "X",
    "response": "v",
    "progression": "^",
    "recurrence": "^",
    "toxicity": "o",
    "patient_state_update": "h",
    "clonal_evolution_event": "D",
    "local_intervention": "*",
    "death": "X",
}

BURDEN = {
    "absent": 0.0,
    "microscopic": 0.4,
    "small": 1.0,
    "moderate": 2.0,
    "bulky": 3.0,
    "diffuse_or_leukemic": 4.0,
}
SYMPTOM = {"none": 0.0, "mild": 1.0, "moderate": 2.0, "severe": 3.0}
ORGAN = {
    "normal": 0.0,
    "mild_impairment": 1.0,
    "moderate_impairment": 2.0,
    "severe_impairment": 3.0,
    "failure": 4.0,
}


@dataclass(frozen=True)
class Course:
    snapshots: pd.DataFrame
    events: pd.DataFrame
    all_snapshots: pd.DataFrame
    all_events: pd.DataFrame


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required stored artifact: {path}")
    return path


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def title_block(fig: plt.Figure, title: str, subtitle: str, status: str) -> None:
    fig.text(0.055, 0.955, title, fontsize=18, fontweight="bold", va="top")
    fig.text(0.055, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.text(
        0.945,
        0.95,
        status.upper(),
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="top",
        color=INK,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.15", "fc": LIGHT, "ec": GRID},
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def load_course() -> Course:
    snapshots = pd.read_csv(require(SENTINEL / "snapshot_rows.csv"))
    events = pd.read_csv(require(SENTINEL / "event_rows.csv"))
    snapshots = snapshots[snapshots["accepted"].astype(bool)].copy()
    events = events[events["accepted"].astype(bool)].copy()
    snapshots["month"] = pd.to_numeric(snapshots["days_from_diagnosis"], errors="coerce") / MONTH_DAYS
    events["month"] = pd.to_numeric(events["days_from_diagnosis"], errors="coerce") / MONTH_DAYS
    snapshots = add_state_coordinates(snapshots)
    chosen_s = snapshots[snapshots["trajectory_id"].eq(EXEMPLAR_ID)].sort_values(
        ["days_from_diagnosis", "snapshot_index"]
    )
    chosen_e = events[events["trajectory_id"].eq(EXEMPLAR_ID)].sort_values(
        ["days_from_diagnosis", "event_index"]
    )
    if chosen_s.empty or chosen_e.empty:
        raise ValueError(f"Exemplar {EXEMPLAR_ID} is not available")
    return Course(chosen_s, chosen_e, snapshots, events)


def add_state_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["tumour"] = d["total_burden_band"].map(BURDEN).fillna(0.0)
    d["ecog"] = pd.to_numeric(d["ecog_ps"], errors="coerce").fillna(0.0)
    d["symptom"] = d["symptom_burden"].map(SYMPTOM).fillna(0.0)
    d["aes"] = pd.to_numeric(d["n_active_aes"], errors="coerce").fillna(0.0).clip(0, 4)
    organ_cols = [
        "organ_renal",
        "organ_hepatic",
        "organ_cardiac",
        "organ_marrow_reserve",
        "organ_pulmonary",
    ]
    organ_matrix = np.column_stack([d[c].map(ORGAN).fillna(0.0) for c in organ_cols])
    d["organ"] = organ_matrix.max(axis=1)
    d["host"] = 0.58 * d["ecog"] + 0.20 * d["symptom"] + 0.14 * d["aes"] + 0.08 * d["organ"]
    d["sites"] = pd.to_numeric(d["num_active_disease_sites"], errors="coerce").fillna(0.0)
    return d


def short_event(event_type: str) -> str:
    return {
        "systemic_treatment_start": "start",
        "systemic_treatment_switch": "switch",
        "systemic_treatment_modification": "modify",
        "systemic_treatment_stop": "stop",
        "patient_state_update": "host update",
        "clonal_evolution_event": "clone",
        "local_intervention": "local Tx",
    }.get(event_type, event_type)


def event_label(row: pd.Series) -> str:
    event_type = str(row["event_type"])
    if event_type in {"systemic_treatment_start", "systemic_treatment_switch"}:
        value = row.get("regimen")
        if pd.notna(value) and str(value).strip():
            return str(value).replace("_", " ")
    if event_type == "toxicity":
        value = row.get("ae")
        grade = row.get("ctcae_v5_grade")
        text = str(value).replace("_", " ") if pd.notna(value) else "toxicity"
        if pd.notna(grade):
            text += f" G{int(float(grade))}"
        return text
    if event_type == "clonal_evolution_event":
        gene = row.get("gene")
        return f"{gene} clone" if pd.notna(gene) else "clonal change"
    return short_event(event_type)


def plot_state_path(course: Course) -> None:
    s = course.snapshots.reset_index(drop=True)
    e = course.events
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.13)
    title_block(
        fig,
        "1 | A clinical course as a path through state space",
        "The full state is high-dimensional; this raw two-axis projection exposes movement, residence and return.",
        "stored synthetic course",
    )

    ax.axvspan(-0.15, 1.25, color="#E7F2EC", alpha=0.75, zorder=0)
    ax.axvspan(2.65, 4.15, color="#F9E7E4", alpha=0.75, zorder=0)
    ax.axhspan(2.2, 4.1, color="#F2EAF4", alpha=0.45, zorder=0)
    ax.text(0.12, 0.18, "low burden\nlow host impact", color=GREEN, fontsize=9)
    ax.text(3.94, 3.58, "high burden\nhigh host impact", color=RED, fontsize=9, ha="right")

    cmap = plt.get_cmap("cividis")
    months = s["month"].to_numpy(float)
    norm = plt.Normalize(months.min(), months.max())
    for i in range(len(s) - 1):
        ax.annotate(
            "",
            xy=(s.loc[i + 1, "tumour"], s.loc[i + 1, "host"]),
            xytext=(s.loc[i, "tumour"], s.loc[i, "host"]),
            arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": cmap(norm(months[i + 1])), "alpha": 0.82},
            zorder=2,
        )
    sizes = 48 + 110 * np.sqrt(np.maximum(np.diff(np.r_[months, months[-1] + 1.0]), 0.1))
    ax.scatter(
        s["tumour"],
        s["host"],
        c=months,
        cmap=cmap,
        norm=norm,
        s=sizes,
        edgecolor=BG,
        linewidth=1.1,
        zorder=3,
    )

    key_types = {"systemic_treatment_start", "response", "progression", "systemic_treatment_switch", "death"}
    offsets = [(9, 10), (8, -20), (8, 12), (-52, 11), (-42, -21)]
    labelled = 0
    for _, row in e[e["event_type"].isin(key_types)].iterrows():
        idx = int(np.argmin(np.abs(months - float(row["month"]))))
        et = str(row["event_type"])
        ax.scatter(
            s.iloc[idx]["tumour"],
            s.iloc[idx]["host"],
            s=92,
            marker=EVENT_MARKERS.get(et, "o"),
            color=EVENT_COLORS.get(et, INK),
            edgecolor=BG,
            linewidth=1.1,
            zorder=5,
        )
        if labelled < len(offsets):
            ax.annotate(
                event_label(row),
                (s.iloc[idx]["tumour"], s.iloc[idx]["host"]),
                xytext=offsets[labelled],
                textcoords="offset points",
                fontsize=8.5,
                color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
            )
            labelled += 1

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("months from diagnosis", color=MUTED)
    ax.set_xlabel("tumour burden coordinate")
    ax.set_ylabel("host impact coordinate")
    ax.set_xlim(-0.18, 4.18)
    ax.set_ylim(-0.12, max(4.0, float(s["host"].max()) + 0.35))
    ax.grid(alpha=0.65)
    clean_axis(ax)
    ax.text(
        0.01,
        -0.14,
        "Node area encodes residence until the next recorded change. Arrows are observed jumps, not velocity.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "fig01_state_path")


def treatment_segments(events: pd.DataFrame) -> list[tuple[float, float, str]]:
    changes = events[
        events["event_type"].isin(
            {"systemic_treatment_start", "systemic_treatment_switch", "systemic_treatment_stop", "death"}
        )
    ].sort_values("month")
    active: tuple[float, str] | None = None
    segments: list[tuple[float, float, str]] = []
    for _, row in changes.iterrows():
        month = float(row["month"])
        et = str(row["event_type"])
        if et in {"systemic_treatment_start", "systemic_treatment_switch"}:
            if active is not None:
                segments.append((active[0], month, active[1]))
            label = event_label(row)
            active = (month, label)
        elif et in {"systemic_treatment_stop", "death"} and active is not None:
            segments.append((active[0], month, active[1]))
            active = None
    if active is not None:
        segments.append((active[0], float(events["month"].max()), active[1]))
    return segments


def step_series(snapshots: pd.DataFrame, column: str, max_month: float) -> tuple[np.ndarray, np.ndarray]:
    x = snapshots["month"].to_numpy(float)
    y = snapshots[column].to_numpy(float)
    if x[-1] < max_month:
        x = np.r_[x, max_month]
        y = np.r_[y, y[-1]]
    return x, y


def plot_event_ribbon(course: Course) -> None:
    s = course.snapshots
    e = course.events
    max_month = float(max(s["month"].max(), e["month"].max()))
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(12.2, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 0.85, 1.0, 1.0, 0.9, 0.85], "hspace": 0.12},
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.11)
    title_block(
        fig,
        "2 | Event-state ribbon: the clinician-readable path",
        "One course retains treatment, disease, host and molecular grammar without reducing the patient to one endpoint.",
        "stored synthetic course",
    )
    labels = ["treatment", "disease events", "tumour state", "host state", "toxicity", "molecular / local"]
    for ax, label in zip(axes, labels, strict=True):
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=26, fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(axis="x", alpha=0.5)

    palette = [BLUE, TEAL, ORANGE, VIOLET, GREEN]
    for i, (start, end, label) in enumerate(treatment_segments(e)):
        axes[0].barh(0, end - start, left=start, height=0.45, color=palette[i % len(palette)], alpha=0.88)
        if end - start > 3.5:
            axes[0].text((start + end) / 2, 0, label, ha="center", va="center", fontsize=8, color="white", clip_on=True)
    axes[0].set_ylim(-0.55, 0.55)

    disease = e[e["event_type"].isin({"response", "progression", "recurrence"})]
    for _, row in disease.iterrows():
        et = str(row["event_type"])
        axes[1].scatter(row["month"], 0, marker=EVENT_MARKERS[et], color=EVENT_COLORS[et], s=72, zorder=3)
        axes[1].text(row["month"], 0.19 if et == "response" else -0.22, short_event(et), ha="center", fontsize=7.5, color=EVENT_COLORS[et])
    axes[1].axhline(0, color=GRID, lw=1.2)
    axes[1].set_ylim(-0.45, 0.45)

    x, y = step_series(s, "tumour", max_month)
    axes[2].step(x, y, where="post", color=RED, lw=2.1)
    axes[2].fill_between(x, 0, y, step="post", color=RED, alpha=0.10)
    axes[2].set_ylim(-0.1, 4.15)
    axes[2].set_yticks([0, 2, 4], labels=["absent", "moderate", "diffuse"])
    axes[2].tick_params(axis="y", left=True, labelleft=True, labelsize=7.5)

    x, y = step_series(s, "host", max_month)
    axes[3].step(x, y, where="post", color=VIOLET, lw=2.1)
    axes[3].fill_between(x, 0, y, step="post", color=VIOLET, alpha=0.10)
    axes[3].set_ylim(-0.1, max(3.7, float(s["host"].max()) + 0.3))
    axes[3].set_yticks([0, 1.5, 3], labels=["low", "moderate", "high"])
    axes[3].tick_params(axis="y", left=True, labelleft=True, labelsize=7.5)

    tox = e[e["event_type"].eq("toxicity")]
    for _, row in tox.iterrows():
        grade = float(row["ctcae_v5_grade"]) if pd.notna(row["ctcae_v5_grade"]) else 1.0
        axes[4].vlines(row["month"], 0, grade, color=MAGENTA, alpha=0.75, lw=1.1)
        axes[4].scatter(row["month"], grade, s=28 + 14 * grade, color=MAGENTA, edgecolor=BG, zorder=3)
    axes[4].set_ylim(-0.1, 5.0)
    axes[4].set_yticks([1, 3, 5], labels=["G1", "G3", "G5"])
    axes[4].tick_params(axis="y", left=True, labelleft=True, labelsize=7.5)

    other = e[e["event_type"].isin({"clonal_evolution_event", "local_intervention", "patient_state_update", "death"})]
    ymap = {"patient_state_update": 0.15, "local_intervention": 0.42, "clonal_evolution_event": 0.69, "death": 0.9}
    for _, row in other.iterrows():
        et = str(row["event_type"])
        axes[5].scatter(row["month"], ymap[et], marker=EVENT_MARKERS[et], color=EVENT_COLORS[et], s=58, edgecolor=BG, zorder=3)
    axes[5].set_ylim(0, 1.05)
    axes[5].set_yticks([0.15, 0.42, 0.69, 0.9], labels=["host", "local", "clone", "death"])
    axes[5].tick_params(axis="y", left=True, labelleft=True, labelsize=7.5)
    axes[-1].set_xlabel("months from diagnosis")
    axes[-1].set_xlim(0, max_month + 1)
    save(fig, "fig02_event_state_ribbon")


def normalized_entropy(values: Iterable[str]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def trajectory_features(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for tid, group in events.sort_values(["trajectory_id", "days_from_diagnosis", "event_index"]).groupby("trajectory_id"):
        types = group["event_type"].astype(str).tolist()
        months = group["month"].to_numpy(float)
        duration = max(float(months.max() - months.min()), 0.1)
        transitions = list(zip(types[:-1], types[1:], strict=False))
        disease = [t for t in types if t in {"response", "progression", "recurrence"}]
        alternations = sum(a != b for a, b in zip(disease[:-1], disease[1:], strict=False))
        tox = group[group["event_type"].eq("toxicity")]["ctcae_v5_grade"]
        tox_burden = float(pd.to_numeric(tox, errors="coerce").fillna(1.0).sum()) / duration
        rows.append(
            {
                "trajectory_id": tid,
                "duration": duration,
                "event_count": float(len(types)),
                "event_richness": float(len(set(types))),
                "composition_entropy": normalized_entropy(types),
                "grammar_diversity": float(len(set(transitions)) / max(len(transitions), 1)),
                "recurrence": float(alternations),
                "toxicity_density": tox_burden,
            }
        )
    return pd.DataFrame(rows)


def plot_fingerprint(course: Course) -> None:
    f = trajectory_features(course.all_events)
    chosen = f[f["trajectory_id"].eq(EXEMPLAR_ID)].iloc[0]
    metrics = [
        ("duration", "clock", "months"),
        ("event_count", "event count", "events"),
        ("event_richness", "mark richness", "event types"),
        ("composition_entropy", "composition entropy", "0-1"),
        ("grammar_diversity", "grammar diversity", "0-1"),
        ("recurrence", "response-progression returns", "cycles"),
        ("toxicity_density", "toxicity density", "grade / month"),
    ]
    fig = plt.figure(figsize=(12.0, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.7, 1.3], left=0.18, right=0.96, top=0.83, bottom=0.10, hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    title_block(
        fig,
        "3 | A trajectory fingerprint is a profile, not one score",
        "The selected course is located against 72 stored synthetic courses on clock, richness, grammar and recurrence.",
        "stored synthetic corpus",
    )

    y = np.arange(len(metrics))[::-1]
    for yi, (key, _, _) in zip(y, metrics, strict=True):
        vals = f[key].to_numpy(float)
        q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
        span = max(float(q95 - q05), 1e-9)
        p25 = float(np.clip((q25 - q05) / span, 0.0, 1.0))
        p50 = float(np.clip((q50 - q05) / span, 0.0, 1.0))
        p75 = float(np.clip((q75 - q05) / span, 0.0, 1.0))
        ax.plot([0.0, 1.0], [yi, yi], color=GRID, lw=4, solid_capstyle="round")
        ax.plot([p25, p75], [yi, yi], color=MUTED, lw=8, alpha=0.38, solid_capstyle="butt")
        ax.scatter(p50, yi, marker="|", s=180, color=INK, linewidth=1.6, zorder=3)
        selected_position = float(np.clip((float(chosen[key]) - q05) / span, 0.0, 1.0))
        ax.scatter(selected_position, yi, s=82, color=BLUE, edgecolor=BG, linewidth=1.2, zorder=4)
        ax.text(1.04, yi, f"raw {float(chosen[key]):.2f}", ha="left", va="center", fontsize=7.8, color=BLUE)
    ax.set_yticks(y, labels=[f"{label}\n({unit})" for _, label, unit in metrics])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(-0.03, 1.18)
    ax.set_xticks([0.0, 0.5, 1.0], labels=["low", "within-coordinate position", "high"])
    ax.set_xlabel("position within the synthetic comparison corpus; each coordinate keeps its own clinical unit")
    ax.grid(axis="x", alpha=0.55)
    clean_axis(ax)
    ax.text(0.01, 1.035, "thin: 5-95%     thick: IQR     black: median     blue: selected course", transform=ax.transAxes, fontsize=8.5, color=MUTED)

    types = course.events["event_type"].astype(str)
    counts = types.value_counts()
    top = counts.index[:8].tolist()
    left = 0.0
    for et in top:
        width = float(counts[et] / counts.sum())
        ax2.barh(0, width, left=left, color=EVENT_COLORS.get(et, MUTED), height=0.42)
        if width > 0.07:
            ax2.text(left + width / 2, 0, short_event(et), ha="center", va="center", fontsize=7.2, color="white")
        left += width
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.55, 0.55)
    ax2.set_yticks([])
    ax2.set_xlabel("event-type composition of selected course")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "fig03_trajectory_fingerprint")


def recurrence_similarity(s: pd.DataFrame) -> np.ndarray:
    matrix = s[["tumour", "ecog", "symptom", "sites", "aes", "organ"]].to_numpy(float)
    scales = np.array([4.0, 4.0, 3.0, max(float(s["sites"].quantile(0.9)), 1.0), 4.0, 4.0])
    matrix = matrix / scales
    distance = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    nonzero = distance[np.triu_indices(len(distance), 1)]
    sigma = max(float(np.quantile(nonzero[nonzero > 0], 0.35)) if np.any(nonzero > 0) else 0.2, 0.12)
    return np.exp(-(distance**2) / (2 * sigma**2))


def plot_recurrence(course: Course) -> None:
    s = course.snapshots.reset_index(drop=True)
    similarity = recurrence_similarity(s)
    sep = np.abs(np.arange(len(s))[:, None] - np.arange(len(s))[None, :])
    candidate = similarity.copy()
    candidate[sep < 3] = -np.inf
    i, j = np.unravel_index(np.argmax(candidate), candidate.shape)
    if i > j:
        i, j = j, i

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.9), gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.13, wspace=0.27)
    title_block(
        fig,
        "4 | Recurrence reveals loops and hysteresis",
        "Off-diagonal similarity identifies a return to a previous state; history determines whether that return is clinically the same.",
        "stored synthetic course",
    )
    panel_label(ax1, "a")
    panel_label(ax2, "b")

    image = ax1.imshow(similarity, origin="lower", cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0, vmax=1, aspect="auto")
    ax1.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.0))
    ax1.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.0))
    ticks = np.linspace(0, len(s) - 1, min(6, len(s))).astype(int)
    labels = [f"{s.loc[k, 'month']:.0f}" for k in ticks]
    ax1.set_xticks(ticks, labels=labels)
    ax1.set_yticks(ticks, labels=labels)
    ax1.set_xlabel("months at state j")
    ax1.set_ylabel("months at state i")
    cbar = fig.colorbar(image, ax=ax1, fraction=0.045, pad=0.03)
    cbar.set_label("state similarity")
    ax1.text(0.02, 1.02, "recurrence matrix", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    ax2.plot(s["tumour"], s["host"], color=MUTED, lw=1.8, alpha=0.75, zorder=1)
    ax2.scatter(s["tumour"], s["host"], c=s["month"], cmap="cividis", s=45, edgecolor=BG, zorder=2)
    for k, color, name in [(i, ORANGE, f"state at {s.loc[i, 'month']:.1f} mo"), (j, TEAL, f"return at {s.loc[j, 'month']:.1f} mo")]:
        ax2.scatter(s.loc[k, "tumour"], s.loc[k, "host"], s=155, facecolor="none", edgecolor=color, linewidth=2.2, zorder=4)
        ax2.annotate(name, (s.loc[k, "tumour"], s.loc[k, "host"]), xytext=(8, 8 if k == i else -17), textcoords="offset points", fontsize=8.5, color=color)
    ax2.annotate(
        "same region, later history",
        xy=(s.loc[j, "tumour"], s.loc[j, "host"]),
        xytext=(0.55, 0.08),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.2},
        color=INK,
        fontsize=9,
    )
    ax2.set_xlabel("tumour burden coordinate")
    ax2.set_ylabel("host impact coordinate")
    ax2.grid(alpha=0.5)
    clean_axis(ax2)
    ax2.text(0.02, 1.02, "the return in state space", transform=ax2.transAxes, fontsize=10, fontweight="bold")
    fig.text(0.52, 0.055, "A geometric loop can be a clinical hysteresis loop: X is similar, but ordered history H is not.", ha="center", fontsize=9, color=MUTED)
    save(fig, "fig04_recurrence_hysteresis")


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        cur = [i]
        for j, item_b in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (item_a != item_b)))
        prev = cur
    return prev[-1]


def type_composition(types: list[str], vocabulary: list[str]) -> np.ndarray:
    counts = Counter(types)
    vector = np.array([counts[x] for x in vocabulary], dtype=float)
    return vector / max(vector.sum(), 1.0)


def plot_similarity_decomposition(course: Course) -> None:
    major = course.events[
        course.events["event_type"].isin(
            {"systemic_treatment_start", "response", "progression", "toxicity", "systemic_treatment_switch", "death"}
        )
    ].copy()
    major = major.drop_duplicates("event_type", keep="first").sort_values("month").head(6)
    base_times = major["month"].to_numpy(float)
    base_types = major["event_type"].astype(str).tolist()
    base_times = base_times - base_times.min()
    if base_times[-1] <= 0:
        base_times = np.arange(len(base_times), dtype=float)

    variants: list[tuple[str, np.ndarray, list[str]]] = []
    variants.append(("reference course", base_times, base_types))
    variants.append(("clock dilation", base_times * 1.65, base_types.copy()))
    insert_t = float(base_times[-1] * 0.55)
    insertion_times = np.r_[base_times, insert_t]
    insertion_types = base_types + ["toxicity"]
    order = np.argsort(insertion_times)
    variants.append(("event insertion", insertion_times[order], [insertion_types[k] for k in order]))
    swapped_t = base_times.copy()
    swapped_types = base_types.copy()
    response_idx = next((k for k, value in enumerate(swapped_types) if value == "response"), None)
    progression_idx = next((k for k, value in enumerate(swapped_types) if value == "progression"), None)
    if response_idx is not None and progression_idx is not None:
        swapped_types[response_idx], swapped_types[progression_idx] = swapped_types[progression_idx], swapped_types[response_idx]
    variants.append(("grammar change", swapped_t, swapped_types))

    vocab = sorted(set(sum((types for _, _, types in variants), [])))
    distances = []
    for _, times, types in variants:
        clock = abs(math.log(max(times[-1], 1e-6) / max(base_times[-1], 1e-6)))
        composition = float(np.abs(type_composition(types, vocab) - type_composition(base_types, vocab)).sum() / 2)
        grammar = levenshtein(base_types, types) / max(len(base_types), len(types), 1)
        if len(types) == len(base_types):
            timing = float(np.sqrt(np.mean(((times / max(times[-1], 1e-6)) - (base_times / max(base_times[-1], 1e-6))) ** 2)))
        else:
            timing = 0.25
        distances.append([clock, composition, grammar, timing])
    dist = np.asarray(distances)
    dist = dist / np.maximum(dist.max(axis=0, keepdims=True), 1e-9)

    fig = plt.figure(figsize=(12.2, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.75], left=0.08, right=0.96, top=0.82, bottom=0.12, wspace=0.28)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1])
    title_block(
        fig,
        "5 | Similarity separates clock, composition and grammar",
        "Courses can share the same event grammar while running on a different clock, or share duration while changing event identity and order.",
        "schematic transformations",
    )
    panel_label(ax, "a")
    panel_label(axm, "b")
    y_positions = np.arange(len(variants))[::-1]
    max_t = max(float(times[-1]) for _, times, _ in variants)
    for y, (label, times, types) in zip(y_positions, variants, strict=True):
        ax.hlines(y, 0, float(times[-1]), color=GRID, lw=2)
        for t, et in zip(times, types, strict=True):
            ax.scatter(t, y, marker=EVENT_MARKERS.get(et, "o"), color=EVENT_COLORS.get(et, MUTED), s=75, edgecolor=BG, linewidth=0.8, zorder=3)
        ax.text(-0.02 * max_t, y, label, ha="right", va="center", fontsize=9.5, fontweight="bold" if y == y_positions[0] else "normal")
    ax.set_ylim(-0.7, len(variants) - 0.3)
    ax.set_xlim(-0.05 * max_t, max_t * 1.03)
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.grid(axis="x", alpha=0.55)
    clean_axis(ax)
    handles = []
    labels = []
    for et in base_types:
        if et in labels:
            continue
        handles.append(plt.Line2D([], [], linestyle="", marker=EVENT_MARKERS.get(et, "o"), color=EVENT_COLORS.get(et, MUTED), markersize=7))
        labels.append(short_event(et))
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=7.5)

    cmap = LinearSegmentedColormap.from_list("distance", ["#F4F6F5", "#E9BF73", RED])
    im = axm.imshow(dist, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axm.set_xticks(range(4), labels=["clock", "composition", "grammar", "relative\ntiming"], rotation=25, ha="right")
    axm.set_yticks(range(len(variants)), labels=[name for name, _, _ in variants])
    for row in range(dist.shape[0]):
        for col in range(dist.shape[1]):
            axm.text(col, row, f"{dist[row, col]:.2f}", ha="center", va="center", fontsize=8, color="white" if dist[row, col] > 0.62 else INK)
    fig.colorbar(im, ax=axm, fraction=0.05, pad=0.03, label="component distance")
    axm.text(0.0, 1.02, "distance profile", transform=axm.transAxes, fontsize=10, fontweight="bold")
    fig.text(0.53, 0.04, "The profile remains interpretable; a single scalar can be reported only after weights are declared.", ha="center", fontsize=9, color=MUTED)
    save(fig, "fig05_similarity_decomposition")


def load_fan() -> pd.DataFrame:
    d = pd.read_csv(require(FAN_INTERVALS))
    chosen = d[d["state"].eq("S1") & d["history"].eq("H0") & d["arm"].isin({"chemo", "osimertinib"})].copy()
    if chosen["trajectory_id"].nunique() < 6:
        raise ValueError("Stored future fan is unexpectedly small")
    return chosen


def fan_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tid, arm), group in frame.sort_values(["trajectory_id", "interval_index"]).groupby(["trajectory_id", "arm"]):
        xy = group[["map_x", "map_y"]].to_numpy(float)
        path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        rows.append(
            {
                "trajectory_id": tid,
                "arm": arm,
                "duration": float(group["endpoint_mo"].iloc[0]),
                "path_length": path_length,
                "n_states": float(len(group)),
            }
        )
    return pd.DataFrame(rows)


def plot_future_fan(fan: pd.DataFrame) -> None:
    metrics = fan_metrics(fan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 7.1), gridspec_kw={"width_ratios": [1.2, 0.8]})
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.14, wspace=0.28)
    title_block(
        fig,
        "6 | The future fan is the probabilistic object",
        "Repeated complete futures from the same C/H/X anchor reveal path diversity; arm means are only summaries of the fan.",
        "stored model assay, not human effect",
    )
    panel_label(ax1, "a")
    panel_label(ax2, "b")
    colors = {"chemo": ORANGE, "osimertinib": BLUE}
    for arm, arm_data in fan.groupby("arm"):
        for _, group in arm_data.sort_values(["trajectory_id", "interval_index"]).groupby("trajectory_id"):
            ax1.plot(group["map_x"], group["map_y"], color=colors[arm], lw=1.1, alpha=0.26)
            ax1.scatter(group["map_x"].iloc[-1], group["map_y"].iloc[-1], color=colors[arm], s=22, alpha=0.65, edgecolor=BG, linewidth=0.4)
        first = arm_data.sort_values("interval_index").groupby("trajectory_id").first()
        last = arm_data.sort_values("interval_index").groupby("trajectory_id").last()
        origin = first[["map_x", "map_y"]].mean().to_numpy(float)
        endpoint = last[["map_x", "map_y"]].mean().to_numpy(float)
        ax1.add_patch(FancyArrowPatch(origin, endpoint, arrowstyle="-|>", mutation_scale=15, lw=2.5, color=colors[arm], zorder=5))
    first_all = fan.sort_values("interval_index").groupby("trajectory_id").first()
    ax1.scatter(first_all["map_x"], first_all["map_y"], s=95, facecolor=BG, edgecolor=INK, linewidth=1.2, zorder=6)
    ax1.set_xlabel("state-map coordinate 1")
    ax1.set_ylabel("state-map coordinate 2")
    ax1.grid(alpha=0.5)
    clean_axis(ax1)
    handles = [plt.Line2D([], [], color=colors[x], lw=2.5, label=x) for x in ["chemo", "osimertinib"]]
    ax1.legend(handles=handles, loc="upper right")
    ax1.text(0.02, 1.02, "complete pre-progression paths", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    for arm, group in metrics.groupby("arm"):
        ax2.scatter(group["duration"], group["path_length"], s=68, color=colors[arm], alpha=0.78, edgecolor=BG, linewidth=0.8, label=arm)
        ax2.scatter(group["duration"].mean(), group["path_length"].mean(), s=190, marker="X", color=colors[arm], edgecolor=INK, linewidth=0.8, zorder=4)
    ax2.set_xlabel("progression-free duration (months)")
    ax2.set_ylabel("path length in state map")
    ax2.grid(alpha=0.5)
    clean_axis(ax2)
    ax2.text(0.02, 1.02, "clock and movement are not the same", transform=ax2.transAxes, fontsize=10, fontweight="bold")
    ax2.text(0.04, 0.06, "large X = fan mean", transform=ax2.transAxes, fontsize=8.5, color=MUTED)
    fig.text(0.51, 0.045, "Stored Sonnet assay, one exact S1/H0 anchor, n = 3 futures per arm. This visualizes fan grammar, not calibration or a human effect.", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "fig06_future_fan")


def plot_chax_matrix() -> None:
    rows = [
        "C | EGFR WT -> mutant",
        "H | naive -> pretreated",
        "X | fit -> frail",
        "A | chemo -> targeted",
        "H x A | resistant -> sensitive",
        "sham | appendectomy absent -> present",
    ]
    columns = [
        "PFS",
        "OS",
        "tumour\nmovement",
        "host\nmovement",
        "response\noccupancy",
        "treatment\nswitches",
        "toxicity\nburden",
        "recurrence\nloops",
        "event\nrichness",
    ]
    values = np.array(
        [
            [0.55, 0.40, -0.62, -0.20, 0.68, -0.15, 0.08, -0.24, 0.10],
            [-0.42, -0.31, 0.45, 0.33, -0.38, 0.28, 0.22, 0.35, 0.18],
            [-0.24, -0.46, 0.18, 0.82, -0.20, 0.05, 0.48, 0.12, 0.06],
            [0.60, 0.42, -0.70, -0.18, 0.76, -0.20, -0.14, -0.31, 0.08],
            [0.38, 0.22, -0.44, -0.10, 0.52, -0.12, 0.02, -0.16, 0.04],
            [0.02, -0.01, 0.00, 0.03, -0.01, 0.00, 0.02, 0.00, 0.01],
        ]
    )
    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    fig.subplots_adjust(left=0.23, right=0.94, top=0.78, bottom=0.20)
    title_block(
        fig,
        "7 | C/H/X/A becomes a perturbation-response atlas",
        "Rows vary one clinically meaningful input. Columns report finite differences in clock, movement, occupancy, grammar and richness.",
        "layout prototype, illustrative values",
    )
    cmap = LinearSegmentedColormap.from_list("response", [BLUE, "#F7F7F3", ORANGE])
    norm = TwoSlopeNorm(vmin=-0.85, vcenter=0.0, vmax=0.85)
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(rows)), labels=rows)
    ax.tick_params(axis="x", labelsize=8.7, pad=9)
    ax.tick_params(axis="y", labelsize=9.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.52 else INK)
    for x in [1.5, 3.5, 4.5, 7.5]:
        ax.axvline(x, color=BG, lw=3.0)
    group_centres = [(0.5, "passage"), (2.5, "movement"), (4.0, "occupancy"), (6.0, "grammar"), (8.0, "richness")]
    for x, label in group_centres:
        ax.text(x, -1.07, label.upper(), ha="center", va="center", fontsize=8, fontweight="bold", color=MUTED)
    ax.set_ylim(len(rows) - 0.5, -1.35)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("standardized finite difference")
    ax.text(
        0.0,
        -0.17,
        "Each cell would carry an anchor-clustered interval and horizon; the sham row measures non-specific responsiveness.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "fig07_chax_response_atlas")


def covariance_ellipse(points: np.ndarray, n_std: float = 1.7) -> tuple[np.ndarray, float, float, float]:
    centre = points.mean(axis=0)
    cov = np.cov(points.T)
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(values, 1e-9))
    return centre, float(width), float(height), float(angle)


def draw_cloud(ax: plt.Axes, points: np.ndarray, color: str, label: str, *, alpha: float = 0.55) -> np.ndarray:
    centre, width, height, angle = covariance_ellipse(points)
    ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.12, lw=1.5))
    ax.scatter(points[:, 0], points[:, 1], s=28, color=color, alpha=alpha, edgecolor=BG, linewidth=0.4, label=label)
    ax.scatter(*centre, s=90, marker="X", color=color, edgecolor=INK, linewidth=0.7, zorder=5)
    return centre


def plot_counterfactual_geometry() -> None:
    rng = np.random.default_rng(SEED)
    reference = rng.multivariate_normal([0.0, 0.05], [[0.42, 0.11], [0.11, 0.23]], 42)
    unanchored = rng.multivariate_normal([1.15, 0.85], [[0.24, 0.03], [0.03, 0.18]], 22)
    factual = rng.multivariate_normal([0.15, 0.16], [[0.18, 0.04], [0.04, 0.13]], 22)
    counterfactual = rng.multivariate_normal([-0.55, -0.24], [[0.20, 0.02], [0.02, 0.15]], 22)

    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.82, bottom=0.13)
    title_block(
        fig,
        "8 | Registration and displacement are different operations",
        "Real evidence can register the factual fan; treatment direction remains a separate, independently audited model property.",
        "conceptual geometry",
    )
    ax.scatter(reference[:, 0], reference[:, 1], s=24, color="#AAB3BA", alpha=0.52, label="observed reference class", zorder=1)
    ref_center, ref_w, ref_h, ref_angle = covariance_ellipse(reference, n_std=2.0)
    ax.add_patch(Ellipse(ref_center, ref_w, ref_h, angle=ref_angle, facecolor="none", edgecolor=MUTED, linestyle="--", lw=1.4, alpha=0.85))
    u = draw_cloud(ax, unanchored, ORANGE, "unanchored factual fan")
    a = draw_cloud(ax, factual, BLUE, "registered factual fan")
    b = draw_cloud(ax, counterfactual, TEAL, "candidate counterfactual fan")
    ax.add_patch(FancyArrowPatch(u, a, arrowstyle="-|>", mutation_scale=17, lw=2.2, color=VIOLET, connectionstyle="arc3,rad=-0.10", zorder=7))
    ax.text((u[0] + a[0]) / 2 + 0.05, (u[1] + a[1]) / 2 + 0.10, "registration", color=VIOLET, fontsize=10, fontweight="bold")
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=17, lw=2.4, color=INK, zorder=7))
    midpoint = (a + b) / 2
    ax.text(midpoint[0], midpoint[1] - 0.18, r"$\Delta_{A\rightarrow B}(z)$", color=INK, fontsize=12, ha="center", fontweight="bold")
    ax.plot([a[0], b[0]], [a[1], a[1]], color=GRID, lw=1, linestyle=":")
    ax.text(b[0], a[1] + 0.06, "clock / level", color=MUTED, fontsize=8, ha="center")
    ax.plot([b[0], b[0]], [a[1], b[1]], color=GRID, lw=1, linestyle=":")
    ax.text(b[0] - 0.04, (a[1] + b[1]) / 2, "shape", color=MUTED, fontsize=8, ha="right", va="center", rotation=90)
    ax.text(-1.55, 1.42, "same C, H, X and missingness\nchange A only", fontsize=9.5, color=INK, bbox={"boxstyle": "square,pad=0.45", "fc": LIGHT, "ec": GRID})
    ax.text(0.60, -0.90, "support controls confidence,\nnot the direction itself", fontsize=9, color=MUTED, ha="center")
    ax.set_xlabel("trajectory coordinate: disease control / clock")
    ax.set_ylabel("trajectory coordinate: host and event shape")
    ax.set_xlim(-1.8, 2.05)
    ax.set_ylim(-1.2, 1.75)
    ax.grid(alpha=0.45)
    clean_axis(ax)
    ax.legend(loc="lower left", ncol=2, fontsize=8.5)
    save(fig, "fig08_registration_counterfactual")


def make_contact_sheet(stems: list[str]) -> None:
    images = [plt.imread(OUT / f"{stem}.png") for stem in stems]
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 20.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.02, hspace=0.08, wspace=0.04)
    fig.suptitle("Trajectory geometry | exploratory visual grammar", fontsize=24, fontweight="bold", y=0.992)
    for index, (ax, image) in enumerate(zip(axes.ravel(), images, strict=True), start=1):
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(0.01, 0.99, str(index), transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold", color=INK, bbox={"boxstyle": "square,pad=0.25", "fc": BG, "ec": GRID})
    fig.savefig(OUT / "contact_sheet.png", dpi=180, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT / "contact_sheet.pdf", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def write_gallery(stems: list[str]) -> None:
    descriptions = [
        ("State path", "Best for the ontology: where the patient moves, pauses and returns."),
        ("Event-state ribbon", "Best clinician-facing representation of one complete course."),
        ("Fingerprint", "Best compact QC profile across clock, richness, grammar and recurrence."),
        ("Recurrence and hysteresis", "Best view of response-relapse loops and same-state/different-history returns."),
        ("Similarity decomposition", "Best explanation of same shape/different clock versus changed grammar."),
        ("Future fan", "Best representation of the model-implied distribution of complete futures."),
        ("C/H/X/A response atlas", "Best population summary of which input perturbations move which path properties."),
        ("Registration and counterfactual", "Best single conceptual summary of calibration versus treatment displacement."),
    ]
    cards = []
    for stem, (name, description) in zip(stems, descriptions, strict=True):
        cards.append(
            f"""
            <article>
              <a href="{html.escape(stem)}.png"><img src="{html.escape(stem)}.png" alt="{html.escape(name)}"></a>
              <h2>{html.escape(name)}</h2>
              <p>{html.escape(description)}</p>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory geometry gallery</title>
<style>
body {{ margin: 0; background: {BG}; color: {INK}; font: 16px/1.5 system-ui, sans-serif; }}
header {{ max-width: 1480px; margin: 0 auto; padding: 44px 28px 18px; }}
h1 {{ margin: 0 0 8px; font-size: 36px; letter-spacing: 0; }}
header p {{ max-width: 900px; color: {MUTED}; margin: 0; }}
main {{ max-width: 1480px; margin: 0 auto; padding: 20px 28px 60px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }}
article {{ border-top: 2px solid {INK}; padding-top: 12px; }}
img {{ width: 100%; display: block; background: white; border: 1px solid {GRID}; }}
h2 {{ font-size: 19px; margin: 12px 0 3px; }}
p {{ margin: 0; color: {MUTED}; }}
@media (max-width: 850px) {{ main {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} }}
</style>
</head>
<body>
<header>
  <h1>Trajectory geometry: eight candidate views</h1>
  <p>Exploratory visual bake-off. Stored data are synthetic model outputs; conceptual panels are labelled. No panel reports a new human-data result.</p>
</header>
<main>{''.join(cards)}</main>
</body>
</html>
"""
    (OUT / "index.html").write_text(document, encoding="utf-8")


def write_readme(stems: list[str]) -> None:
    data = {
        "status": "exploratory visual design bake-off",
        "real_patient_data": False,
        "model_calls": False,
        "exemplar_trajectory": EXEMPLAR_ID,
        "stored_sources": [
            str(SENTINEL.relative_to(ROOT)),
            str(FAN_INTERVALS.relative_to(ROOT)),
        ],
        "figures": stems,
        "interpretation": {
            "measured_stored_synthetic": stems[:4] + [stems[5]],
            "schematic": [stems[4], stems[6], stems[7]],
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    text = """# Trajectory geometry gallery

Exploratory visual design only. No model calls and no real patient data are used.

## Data status

- Figures 1-4 use one stored synthetic Sonnet trajectory and its synthetic comparison corpus.
- Figure 6 uses the stored EGFR model-assay fan (three futures per arm for one exact anchor).
- Figures 5, 7 and 8 are explicitly schematic. Their values must not enter a claim ledger.
- None of the figures depends on the currently disputed GENIE whole-trajectory score.

## Proposed jobs

- Main conceptual figure candidate: Figure 8, with Figure 6 as its measured companion.
- Main single-patient representation candidate: Figure 2.
- Methods/QC candidate: Figure 3.
- Geometry extension: Figure 4 or 5, likely Extended Data rather than both.
- Battery summary: Figure 7 after replacing the illustrative cells with audited estimates.

The gallery HTML is `index.html`; the printable overview is `contact_sheet.pdf`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    course = load_course()
    fan = load_fan()
    plot_state_path(course)
    plot_event_ribbon(course)
    plot_fingerprint(course)
    plot_recurrence(course)
    plot_similarity_decomposition(course)
    plot_future_fan(fan)
    plot_chax_matrix()
    plot_counterfactual_geometry()
    stems = [
        "fig01_state_path",
        "fig02_event_state_ribbon",
        "fig03_trajectory_fingerprint",
        "fig04_recurrence_hysteresis",
        "fig05_similarity_decomposition",
        "fig06_future_fan",
        "fig07_chax_response_atlas",
        "fig08_registration_counterfactual",
    ]
    make_contact_sheet(stems)
    write_gallery(stems)
    write_readme(stems)
    print(f"Wrote {len(stems)} figures and gallery to {OUT}")


if __name__ == "__main__":
    main()
