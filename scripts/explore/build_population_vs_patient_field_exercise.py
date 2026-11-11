#!/usr/bin/env python3
"""Population fields versus history-updated forecasts for one patient.

It builds:

1. a disclosed conceptual figure showing why a static population field should
   not be integrated as an individual patient's deterministic law;
2. same-state/different-history examples selected from stored generated atlas
   transitions while holding cancer and model fixed;
3. six sequential exact-prefix branching landmarks for one generated PAAD
   course, with the forecast fan recalculated after every new observation;
4. a concise trajectory-held-out comparison of state-only, hybrid-clock, and
   full-history prediction.

All data-grounded panels reuse one PCA fitted once from the same 16
contemporaneous tumour + host variables. History never changes the geography;
it changes the conditional future distribution over that geography.

Run:
  uv run python scripts/explore/build_population_vs_patient_field_exercise.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_faceted_clinical_field_atlas import (  # noqa: E402
    INK,
    MONTH_DAYS,
    MUTED,
    PAPER,
    add_phase_labels,
    collapse_same_day,
    interpolate_map_position,
)
from build_history_clock_field_exercise import (  # noqa: E402
    benchmark_clocks,
    history_feature_table,
    prepare_benchmark,
)
from build_one_map_many_futures_exercise import (  # noqa: E402
    BRANCH_ROOT,
    BRANCH_SUMMARY,
    DisplayMap,
    fit_display_map,
    load_derivation,
    map_limits,
    transform_to_map,
)
from composite_tumor_host_phase_space import (  # noqa: E402
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_population_vs_patient_field"

BLUE = "#3474b8"
TEAL = "#138f86"
GREEN = "#208a53"
ORANGE = "#df7900"
RED = "#c63e3e"
PURPLE = "#6846a5"
GOLD = "#c59a24"
GRID = "#dce4ec"
SEED = 20260728

CANCERS = ("LUAD", "HGSOC", "PAAD")
CANCER_LABEL = {
    "LUAD": "Lung adenocarcinoma",
    "HGSOC": "High-grade serous ovarian cancer",
    "PAAD": "Pancreatic adenocarcinoma",
}
PHASE_LABEL = {
    "diagnosis": "at diagnosis",
    "first_line": "during first line",
    "post_progression": "after progression",
}
MODEL_LABEL = {
    "claude_sonnet5": "Claude Sonnet 5",
    "gemini31_lite": "Gemini 3.1 Lite",
    "gemma4": "Gemma 4",
    "glm52_fp8": "GLM-5.2",
    "kimi_k3_off": "Kimi K3",
    "qwen36_flash": "Qwen 3.6 Flash",
}
LANDMARK_LABEL = {
    "p01_after_treatment_start_folfirinox": "After FOLFIRINOX start",
    "p02_after_toxicity_diarrhea_grade_2": "After grade-2 diarrhoea",
    "p03_after_response_pancreas_liver": "After tumour response",
    "p04_after_progression_liver": "After liver progression",
    "p05_after_treatment_switch_gemcitabine_nab_paclitaxel": ("After switch to gemcitabine/nab-paclitaxel"),
    "p06_after_patient_state_update_ecog_2": "After deterioration to ECOG 2",
}

POPULATION_COLOR = "#8d99a8"
PATIENT_COLOR = PURPLE


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(pdf, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))}


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linewidth: float = 2.0,
    alpha: float = 0.95,
    mutation_scale: float = 13,
    zorder: int = 8,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            color=color,
            lw=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
        )
    )


def style_map(
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    *,
    labels: bool = True,
) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(color=GRID, lw=0.65, alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    if labels:
        ax.set_xlabel(
            "Clinical state PC1\nless severe  \u2190     \u2192  more severe",
            fontsize=8,
        )
        ax.set_ylabel("Clinical state PC2", fontsize=8)


def background_points(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    *,
    n: int = 5000,
) -> None:
    pool = atlas[~atlas["terminal"]]
    if len(pool) > n:
        pool = pool.sample(n=n, random_state=SEED)
    ax.scatter(
        pool["map_x"],
        pool["map_y"],
        s=3,
        color="#b8c3ce",
        alpha=0.08,
        linewidth=0,
        rasterized=True,
        zorder=0,
    )


def conceptual_figure() -> dict[str, str]:
    """Draw the population-average and patient-updated distinction."""
    limits = (-2.2, 2.2, -1.8, 1.8)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.5))
    fig.patch.set_facecolor(PAPER)

    # A — a mean vector can describe nobody when futures are multimodal.
    ax = axes[0]
    now = (0.0, 0.0)
    futures_a = np.array([[-1.5, -0.7], [-1.25, -0.45], [-1.55, -0.25]])
    futures_b = np.array([[1.45, 0.75], [1.2, 0.45], [1.55, 0.25]])
    ax.scatter(*now, s=115, facecolor="white", edgecolor=INK, lw=1.8, zorder=10)
    for end in futures_a:
        add_arrow(
            ax,
            now,
            (float(end[0]), float(end[1])),
            GREEN,
            linewidth=1.1,
            alpha=0.35,
            mutation_scale=8,
        )
    for end in futures_b:
        add_arrow(
            ax,
            now,
            (float(end[0]), float(end[1])),
            RED,
            linewidth=1.1,
            alpha=0.35,
            mutation_scale=8,
        )
    mean_end = np.vstack([futures_a, futures_b]).mean(axis=0)
    add_arrow(
        ax,
        now,
        (float(mean_end[0]), float(mean_end[1])),
        POPULATION_COLOR,
        linewidth=3.5,
        mutation_scale=16,
    )
    ax.text(
        0,
        -0.33,
        "population mean",
        ha="center",
        fontsize=8,
        color=POPULATION_COLOR,
        weight="bold",
    )
    style_map(ax, limits)
    ax.set_title(
        "A · One population arrow\nmay describe no individual future",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )

    # B — integrating a static field substitutes population histories.
    ax = axes[1]
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1.7, 1.7, 5),
        np.linspace(-1.3, 1.3, 4),
    )
    for x, y in zip(grid_x.ravel(), grid_y.ravel(), strict=True):
        dx = -0.26 * x - 0.08 * y
        dy = -0.12 * y + 0.06 * x
        add_arrow(
            ax,
            (float(x), float(y)),
            (float(x + dx), float(y + dy)),
            POPULATION_COLOR,
            linewidth=0.8,
            alpha=0.55,
            mutation_scale=7,
            zorder=2,
        )
    static_path = np.array([[1.6, 1.05], [1.15, 0.75], [0.75, 0.52], [0.45, 0.35], [0.25, 0.22]])
    actual_path = np.array([[1.6, 1.05], [1.05, 0.50], [0.10, -0.35], [-0.65, -0.55], [0.25, 0.15]])
    ax.plot(
        static_path[:, 0],
        static_path[:, 1],
        "--o",
        color=POPULATION_COLOR,
        lw=2,
        ms=3,
        label="integrated population field",
    )
    ax.plot(
        actual_path[:, 0],
        actual_path[:, 1],
        "-o",
        color=RED,
        lw=2.6,
        ms=4,
        label="one patient's realised path",
    )
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    style_map(ax, limits)
    ax.set_title(
        "B · Wrong individual reading\nfollowing a static population field",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )

    # C — every new observation triggers a new conditional fan.
    ax = axes[2]
    patient = np.array([[1.6, 1.05], [1.05, 0.50], [0.10, -0.35], [-0.65, -0.55]])
    rng = np.random.default_rng(SEED)
    for index, point in enumerate(patient[:-1]):
        next_point = patient[index + 1]
        fan = next_point + rng.normal(0, [0.22, 0.18], size=(16, 2))
        ax.scatter(
            fan[:, 0],
            fan[:, 1],
            s=15,
            color=PATIENT_COLOR,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        mean = fan.mean(axis=0)
        add_arrow(
            ax,
            (float(point[0]), float(point[1])),
            (float(mean[0]), float(mean[1])),
            PATIENT_COLOR,
            linewidth=2.3,
            mutation_scale=13,
        )
        ax.add_patch(
            Circle(
                next_point,
                radius=0.08,
                facecolor="white",
                edgecolor=PATIENT_COLOR,
                lw=1.5,
                zorder=9,
            )
        )
        ax.text(
            point[0] + 0.07,
            point[1] + 0.08,
            f"visit {index + 1}",
            fontsize=7.2,
            color=INK,
        )
    ax.plot(patient[:, 0], patient[:, 1], color=INK, lw=1.4, zorder=5)
    ax.scatter(
        patient[:, 0],
        patient[:, 1],
        s=38,
        facecolor="white",
        edgecolor=INK,
        lw=1.1,
        zorder=8,
    )
    style_map(ax, limits)
    ax.set_title(
        "C · Correct patient reading\nrecalculate the forecast fan at each visit",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=PATIENT_COLOR,
    )
    fig.suptitle(
        "The geography can be shared while the future remains patient-conditional",
        x=0.03,
        y=1.02,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.03,
        0.945,
        (
            "Disclosed conceptual geometry. A population field is a descriptive "
            "average; a patient forecast is a history-conditioned distribution "
            "that must be updated as the path unfolds."
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.03,
        0.012,
        (
            "A mean arrow can fall between response and progression modes. "
            "For one patient, display possible futures and uncertainty—not a "
            "deterministic streamline."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.995, 0.91), w_pad=1.4)
    return save_figure(fig, "fig01_concept_population_vs_patient")


def horizon_pairs(
    atlas: pd.DataFrame,
    *,
    horizon_months: float,
    tolerance_months: float,
) -> pd.DataFrame:
    """Match state transitions at a configurable horizon."""
    horizon_days = horizon_months * MONTH_DAYS
    tolerance_days = tolerance_months * MONTH_DAYS
    rows: list[dict[str, Any]] = []
    for gid, raw_course in atlas.groupby("_gid", sort=False):
        course = collapse_same_day(raw_course)
        course = course[~course["terminal"]].reset_index(drop=True)
        if len(course) < 2:
            continue
        days = pd.to_numeric(course["days_from_diagnosis"], errors="coerce").to_numpy(float)
        for index, start in course.iterrows():
            future_indices = np.flatnonzero(days > days[index])
            if not len(future_indices):
                continue
            target = days[index] + horizon_days
            nearest = int(future_indices[np.argmin(np.abs(days[future_indices] - target))])
            if abs(days[nearest] - target) > tolerance_days:
                continue
            end = course.iloc[nearest]
            rows.append(
                {
                    "_gid": gid,
                    "model": start["model_short"],
                    "cancer": start["tumor_oncotree_code"],
                    "phase": start["clinical_phase"],
                    "start_day": float(days[index]),
                    "end_day": float(days[nearest]),
                    "x0": float(start["map_x"]),
                    "y0": float(start["map_y"]),
                    "x1": float(end["map_x"]),
                    "y1": float(end["map_y"]),
                    "dx": float(end["map_x"] - start["map_x"]),
                    "dy": float(end["map_y"] - start["map_y"]),
                }
            )
    return pd.DataFrame(rows)


def select_divergent_within_trajectory_pairs(
    transitions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Choose one same-patient, same-state, divergent-future example per cancer."""
    chosen: list[dict[str, Any]] = []
    for cancer in CANCERS:
        candidates: list[dict[str, Any]] = []
        cancer_rows = transitions[transitions["cancer"].eq(cancer)]
        for trajectory_id, group in cancer_rows.groupby(
            "_gid",
            sort=False,
        ):
            group = group.reset_index(drop=True)
            for first_index in range(len(group)):
                for second_index in range(first_index + 1, len(group)):
                    row_a = group.iloc[first_index]
                    row_b = group.iloc[second_index]
                    if row_a["phase"] == row_b["phase"]:
                        continue
                    start_a = np.array([row_a["x0"], row_a["y0"]], dtype=float)
                    start_b = np.array([row_b["x0"], row_b["y0"]], dtype=float)
                    distance = float(np.linalg.norm(start_a - start_b))
                    if distance > 0.08:
                        continue
                    vector_a = np.array([row_a["dx"], row_a["dy"]], dtype=float)
                    vector_b = np.array([row_b["dx"], row_b["dy"]], dtype=float)
                    norm_a = float(np.linalg.norm(vector_a))
                    norm_b = float(np.linalg.norm(vector_b))
                    if min(norm_a, norm_b) < 0.08:
                        continue
                    cosine = float(vector_a @ vector_b / (norm_a * norm_b))
                    if cosine > -0.85:
                        continue
                    difference = float(np.linalg.norm(vector_a - vector_b))
                    score = difference * (1 - cosine) / (distance + 0.03)
                    candidates.append(
                        {
                            "score": score,
                            "state_distance": distance,
                            "cosine": cosine,
                            "future_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                            "cancer": cancer,
                            "model": row_a["model"],
                            "trajectory": trajectory_id,
                            "row_a": row_a.to_dict(),
                            "row_b": row_b.to_dict(),
                        }
                    )
        if not candidates:
            raise ValueError(f"No within-trajectory divergent pair meeting criteria for {cancer}")
        chosen.append(max(candidates, key=lambda item: item["score"]))
    return chosen


def same_state_different_history_figure(
    examples: list[dict[str, Any]],
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    report: list[dict[str, Any]] = []
    for index, (ax, example) in enumerate(zip(axes, examples, strict=True)):
        background_points(ax, atlas)
        row_a = example["row_a"]
        row_b = example["row_b"]
        start_a = (float(row_a["x0"]), float(row_a["y0"]))
        start_b = (float(row_b["x0"]), float(row_b["y0"]))
        end_a = (float(row_a["x1"]), float(row_a["y1"]))
        end_b = (float(row_b["x1"]), float(row_b["y1"]))
        ax.scatter(
            *start_a,
            s=115,
            facecolor="white",
            edgecolor=BLUE,
            lw=2,
            zorder=9,
        )
        ax.scatter(
            *start_b,
            s=65,
            facecolor="white",
            edgecolor=RED,
            lw=2,
            zorder=10,
        )
        add_arrow(
            ax,
            start_a,
            end_a,
            BLUE,
            linewidth=3,
            mutation_scale=16,
            zorder=10,
        )
        add_arrow(
            ax,
            start_b,
            end_b,
            RED,
            linewidth=3,
            mutation_scale=16,
            zorder=11,
        )
        ax.scatter(
            *end_a,
            marker="X",
            s=54,
            facecolor=BLUE,
            edgecolor="white",
            lw=0.6,
            zorder=12,
        )
        ax.scatter(
            *end_b,
            marker="X",
            s=54,
            facecolor=RED,
            edgecolor="white",
            lw=0.6,
            zorder=12,
        )
        phase_a = PHASE_LABEL[str(row_a["phase"])]
        phase_b = PHASE_LABEL[str(row_b["phase"])]
        month_a = float(row_a["start_day"]) / MONTH_DAYS
        month_b = float(row_b["start_day"]) / MONTH_DAYS
        ax.text(
            0.03,
            0.97,
            (
                f"Blue: {phase_a}, month {month_a:.0f}\n"
                f"Red: {phase_b}, month {month_b:.0f}\n"
                f"same generated patient · {MODEL_LABEL[example['model']]}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "#aab5c1",
                "alpha": 0.95,
            },
            zorder=15,
        )
        style_map(ax, limits, labels=True)
        ax.set_title(
            (
                f"{CANCER_LABEL[example['cancer']]}\n"
                f"state distance {example['state_distance']:.3f} · "
                f"future angle {example['future_angle_degrees']:.0f}°"
            ),
            loc="left",
            fontsize=9.4,
            weight="bold",
            color=INK,
        )
        if index:
            ax.set_ylabel("")
        report.append(
            {key: value for key, value in example.items() if key not in {"row_a", "row_b"}}
            | {
                "trajectory": row_a["_gid"],
                "phase_a": row_a["phase"],
                "phase_b": row_b["phase"],
                "start_month_a": month_a,
                "start_month_b": month_b,
                "vector_a": [row_a["dx"], row_a["dy"]],
                "vector_b": [row_b["dx"], row_b["dy"]],
            }
        )
    fig.suptitle(
        "The same visible state can have incompatible futures",
        x=0.035,
        y=1.015,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.955,
        (
            "Stored generated atlas transitions · approximately six-month "
            "movement · the same patient revisits nearly the same visible state "
            "after accumulating a different clinical history."
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.012,
        (
            "Within-trajectory illustrative extremes were selected algorithmically "
            "(state distance ≤0.08; future cosine ≤−0.85). They demonstrate "
            "possibility, not prevalence or causal effect."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.995, 0.91), w_pad=1.2)
    return save_figure(fig, "fig02_data_same_state_different_history"), report


def nearest_snapshot(course: pd.DataFrame, day: float) -> pd.Series:
    index = (pd.to_numeric(course["days_from_diagnosis"], errors="coerce") - day).abs().idxmin()
    return course.loc[index]


def policy_courses(
    branch: pd.DataFrame,
    policy_id: str,
) -> list[pd.DataFrame]:
    selected = branch[branch["policy_id"].eq(policy_id)]
    return [
        course.sort_values(["days_from_diagnosis", "snapshot_index"])
        for _, course in selected.groupby("_gid", sort=False)
    ]


def population_endpoints_at(
    current: tuple[float, float],
    population_pairs: pd.DataFrame,
    *,
    cancer: str,
    n: int = 40,
) -> tuple[np.ndarray, float]:
    pool = population_pairs[population_pairs["cancer"].eq(cancer)].reset_index(drop=True)
    tree = NearestNeighbors(n_neighbors=min(n, len(pool))).fit(pool[["x0", "y0"]])
    query = pd.DataFrame({"x0": [current[0]], "y0": [current[1]]})
    distances, indices = tree.kneighbors(query)
    neighbours = pool.iloc[indices[0]]
    endpoints = np.asarray(current) + neighbours[["dx", "dy"]].to_numpy(float)
    return endpoints, float(distances[0].max())


def patient_branch_filmstrip(
    branch: pd.DataFrame,
    branch_summary: pd.DataFrame,
    population_pairs: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    rows = branch_summary.sort_values("prefix_event_count").reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.7, 9.1), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    report: list[dict[str, Any]] = []
    for index, (ax, summary_row) in enumerate(zip(axes.ravel(), rows.itertuples(index=False), strict=True)):
        policy_id = str(summary_row.policy_id)
        courses = policy_courses(branch, policy_id)
        cutoff = float(summary_row.cutoff_days)
        current_row = nearest_snapshot(courses[0], cutoff)
        current = (float(current_row["map_x"]), float(current_row["map_y"]))
        background_points(ax, atlas)

        incoming = courses[0][courses[0]["days_from_diagnosis"].le(cutoff)]
        ax.plot(
            incoming["map_x"],
            incoming["map_y"],
            color=INK,
            lw=2.2,
            alpha=0.85,
            zorder=6,
        )

        population_endpoints, support_radius = population_endpoints_at(
            current,
            population_pairs,
            cancer="PAAD",
        )
        population_mean = population_endpoints.mean(axis=0)
        ax.scatter(
            population_endpoints[:, 0],
            population_endpoints[:, 1],
            s=13,
            color=POPULATION_COLOR,
            alpha=0.17,
            linewidth=0,
            zorder=1,
        )
        add_arrow(
            ax,
            current,
            (float(population_mean[0]), float(population_mean[1])),
            POPULATION_COLOR,
            linewidth=2.0,
            alpha=0.80,
            mutation_scale=11,
            zorder=4,
        )

        branch_endpoints: list[tuple[float, float]] = []
        terminal_before_horizon = 0
        target = cutoff + 3 * MONTH_DAYS
        palette = plt.cm.Purples(np.linspace(0.42, 0.88, len(courses)))
        for color, course in zip(palette, courses, strict=True):
            maximum = float(pd.to_numeric(course["days_from_diagnosis"], errors="coerce").max())
            endpoint_day = min(target, maximum)
            endpoint = interpolate_map_position(course, endpoint_day)
            branch_endpoints.append(endpoint)
            if maximum < target:
                terminal_before_horizon += 1
            future = course[course["days_from_diagnosis"].ge(cutoff) & course["days_from_diagnosis"].le(endpoint_day)]
            path_x = [current[0], *future["map_x"].astype(float).tolist(), endpoint[0]]
            path_y = [current[1], *future["map_y"].astype(float).tolist(), endpoint[1]]
            ax.plot(
                path_x,
                path_y,
                color=color,
                lw=1.2,
                alpha=0.63,
                zorder=5,
            )
            ax.scatter(
                *endpoint,
                marker="x" if maximum < target else "o",
                s=28,
                color=color,
                lw=1.0,
                zorder=8,
            )
        patient_endpoints = np.asarray(branch_endpoints)
        patient_mean = patient_endpoints.mean(axis=0)
        add_arrow(
            ax,
            current,
            (float(patient_mean[0]), float(patient_mean[1])),
            PATIENT_COLOR,
            linewidth=3.0,
            mutation_scale=15,
            zorder=10,
        )
        ax.scatter(
            *current,
            s=105,
            facecolor="white",
            edgecolor=PATIENT_COLOR,
            lw=2.1,
            zorder=11,
        )

        actual_next: tuple[float, float] | None = None
        if index + 1 < len(rows):
            next_row = rows.iloc[index + 1]
            next_courses = policy_courses(branch, str(next_row["policy_id"]))
            actual_next = interpolate_map_position(
                next_courses[0],
                float(next_row["cutoff_days"]),
            )
            ax.scatter(
                *actual_next,
                marker="D",
                s=45,
                facecolor=RED,
                edgecolor="white",
                lw=0.6,
                zorder=12,
            )
        style_map(ax, limits, labels=False)
        row_index, col_index = divmod(index, 3)
        if row_index == 1:
            ax.set_xlabel(
                "Clinical state PC1\nless severe  \u2190     \u2192  more severe",
                fontsize=7.6,
            )
        if col_index == 0:
            ax.set_ylabel("Clinical state PC2", fontsize=8)
        landmark = str(summary_row.landmark)
        ax.set_title(
            (
                f"Visit {index + 1} · month {cutoff / MONTH_DAYS:.1f}\n"
                f"{LANDMARK_LABEL.get(landmark, landmark.replace('_', ' '))}"
            ),
            loc="left",
            fontsize=8.8,
            weight="bold",
            color=INK,
        )
        ax.text(
            0.03,
            0.96,
            (f"{len(courses)} exact-prefix futures\npurple = updated patient arrow"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=PATIENT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": PATIENT_COLOR,
                "alpha": 0.92,
            },
            zorder=15,
        )
        report.append(
            {
                "visit": index + 1,
                "landmark": landmark,
                "policy_id": policy_id,
                "cutoff_days": cutoff,
                "n_exact_prefix_futures": len(courses),
                "n_terminal_before_three_months": terminal_before_horizon,
                "population_neighbour_count": int(len(population_endpoints)),
                "population_support_radius": support_radius,
                "current": list(current),
                "population_mean_endpoint": population_mean.tolist(),
                "patient_conditioned_mean_endpoint": patient_mean.tolist(),
                "actual_next_landmark": (list(actual_next) if actual_next is not None else None),
            }
        )
    axes[0, 0].plot([], [], color=INK, lw=2, label="fixed observed prefix")
    axes[0, 0].scatter(
        [],
        [],
        s=18,
        color=POPULATION_COLOR,
        alpha=0.45,
        label="nearby PAAD population futures",
    )
    axes[0, 0].plot(
        [],
        [],
        color=PATIENT_COLOR,
        lw=2.5,
        label="exact-history forecast fan",
    )
    axes[0, 0].scatter(
        [],
        [],
        marker="D",
        s=34,
        color=RED,
        label="next fixed-prefix landmark",
    )
    axes[0, 0].legend(frameon=False, fontsize=6.7, loc="lower right")
    fig.suptitle(
        "One patient requires a new forecast after every new observation",
        x=0.035,
        y=1.005,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.958,
        (
            "One stored generated PAAD course followed through six fixed-prefix "
            "landmarks. Grey = local three-month population field; purple = five "
            "new futures generated from this exact accumulated history."
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.012,
        (
            "Each panel forgets the previous forecast and conditions on the newly "
            "observed prefix. Branch endpoints are at three months or terminal "
            "exit if earlier; n=5 is illustrative and not a calibrated uncertainty distribution."
        ),
        fontsize=8.1,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.995, 0.92), h_pad=2.1, w_pad=1.0)
    return save_figure(fig, "fig03_data_one_patient_updated_fans"), report


def conditioning_benchmark_figure(
    benchmark_summary: pd.DataFrame,
    benchmark_report: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    labels = ["State only", "+ hybrid clinical clock", "+ full history"]
    display = benchmark_summary.set_index("model").loc[labels].reset_index()
    short_labels = [
        "Visible dot only\npopulation field",
        "Dot + clinical history\nconditional field",
        "Dot + preceding path\nfull-history forecast",
    ]
    colors = [POPULATION_COLOR, GOLD, PATIENT_COLOR]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    fig.patch.set_facecolor(PAPER)
    bars = ax.bar(
        np.arange(3),
        display["mean_error"],
        yerr=display["sd_error"],
        color=colors,
        alpha=0.90,
        width=0.62,
        error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 3},
    )
    ax.set_xticks(np.arange(3), short_labels, fontsize=8.5)
    ax.set_ylabel("Held-out six-month displacement error\n(lower is better)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, display["mean_error"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
            color=INK,
        )
    state_error = float(display.loc[display["model"].eq("State only"), "mean_error"].iloc[0])
    hybrid_error = float(display.loc[display["model"].eq("+ hybrid clinical clock"), "mean_error"].iloc[0])
    full_error = float(display.loc[display["model"].eq("+ full history"), "mean_error"].iloc[0])
    recovery = (state_error - hybrid_error) / (state_error - full_error)
    ax.annotate(
        f"transparent clinical history recovers {recovery:.0%}\nof the full-history improvement",
        xy=(1, hybrid_error),
        xytext=(1.55, state_error + 0.035),
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.5},
        ha="center",
        fontsize=9,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GOLD,
        },
    )
    ax.set_ylim(0, max(display["mean_error"]) + 0.15)
    fig.suptitle(
        "The visible dot is not a sufficient patient-level state",
        x=0.045,
        y=1.01,
        ha="left",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.93,
        (
            f"{benchmark_report['n_pairs']:,} transitions · "
            f"{benchmark_report['n_trajectories']:,} generated trajectories · "
            "entire trajectories held out together"
        ),
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.015,
        (
            "The hybrid clock contains clinical phase, treatment line and time "
            "since current treatment. Full history additionally contains prior "
            "events and path summaries."
        ),
        fontsize=8.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.90))
    report = {
        "models": display.to_dict(orient="records"),
        "fraction_full_history_gain_recovered_by_hybrid_clock": recovery,
        "n_pairs": benchmark_report["n_pairs"],
        "n_trajectories": benchmark_report["n_trajectories"],
    }
    return save_figure(fig, "fig04_data_conditioning_benchmark"), report


def display_map_report(display: DisplayMap) -> dict[str, Any]:
    return {
        "features": list(display.features),
        "explained_variance": display.explained_variance.tolist(),
        "centers": display.centers.tolist(),
        "scales": display.scales.tolist(),
    }


def write_readme(report: dict[str, Any]) -> None:
    benchmark = report["conditioning_benchmark"]
    same_state = report["same_state_examples"]
    lines = [
        "# Population field versus patient forecast — exploratory exercise",
        "",
        "## Recommended interpretation",
        "",
        "The shared PC1–PC2 map is an observation geography. A population arrow "
        "summarises average finite-horizon displacement among trajectories in a "
        "region; it is not an individual patient's deterministic equation of motion.",
        "",
        "For one patient, the relevant object is a history-conditioned future "
        "distribution. It must be recalculated after every new observation.",
        "",
        "## Data-grounded demonstrations",
        "",
        "- Figure 2 selects two time points from the same stored generated "
        "patient: a near-identical visible state, but divergent six-month futures.",
    ]
    for item in same_state:
        lines.append(
            f"  - {item['cancer']}: state distance "
            f"{item['state_distance']:.3f}; future angle "
            f"{item['future_angle_degrees']:.0f}°."
        )
    lines.extend(
        [
            "- Figure 3 follows one generated PAAD source course through six "
            "fixed-prefix landmarks. Five new futures are generated from the "
            "exact accumulated prefix at every landmark.",
            "- Figure 4 holds complete trajectories out. A hybrid clinical clock "
            f"recovers {benchmark['fraction_full_history_gain_recovered_by_hybrid_clock']:.0%} "
            "of the predictive improvement supplied by full history.",
            "",
            "## Sources and limitations",
            "",
            "- Figure 1 is explicitly mock conceptual geometry.",
            "- Figures 2–4 use stored model-generated project trajectories, not human patients.",
            "- The within-trajectory same-state examples are illustrative extremes "
            "selected by a declared algorithm and do not estimate prevalence.",
            "- The branching fan has only five generated futures per landmark "
            "and is not a calibrated probability distribution.",
            "- Population neighbour clouds exclude terminal endpoints and are descriptive rather than causal.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1 = conceptual_figure()

    atlas_tables = load_tables()
    atlas_raw = enrich_snapshots(atlas_tables)
    display, atlas = fit_display_map(atlas_raw)
    atlas = add_phase_labels(atlas)
    history = history_feature_table(atlas)
    six_month_pairs = horizon_pairs(
        atlas,
        horizon_months=6,
        tolerance_months=2.5,
    )
    benchmark = prepare_benchmark(six_month_pairs, history)
    examples = select_divergent_within_trajectory_pairs(six_month_pairs)

    branch_tables = load_derivation(BRANCH_ROOT)
    branch = transform_to_map(enrich_snapshots(branch_tables), display)
    limits = map_limits(atlas, branch)

    figure_2, same_state_report = same_state_different_history_figure(
        examples,
        atlas,
        limits,
    )
    three_month_pairs = horizon_pairs(
        atlas,
        horizon_months=3,
        tolerance_months=1.5,
    )
    branch_summary = pd.read_csv(BRANCH_SUMMARY)
    figure_3, branch_report = patient_branch_filmstrip(
        branch,
        branch_summary,
        three_month_pairs,
        atlas,
        limits,
    )

    benchmark_summary, benchmark_report, _ = benchmark_clocks(benchmark)
    figure_4, conditioning_report = conditioning_benchmark_figure(
        benchmark_summary,
        benchmark_report,
    )

    report: dict[str, Any] = {
        "contract": {
            "state_space": "one frozen 16-feature atlas PCA",
            "population_field": "local finite-horizon mean displacement",
            "patient_object": "history-conditioned future distribution",
            "individual_rollout_rule": "recondition after every new observation",
        },
        "display_map": display_map_report(display),
        "figures": {
            "concept": figure_1,
            "same_state_different_history": figure_2,
            "one_patient_updated_fans": figure_3,
            "conditioning_benchmark": figure_4,
        },
        "same_state_examples": same_state_report,
        "one_patient_updated_fans": branch_report,
        "conditioning_benchmark": conditioning_report,
        "counts": {
            "atlas_snapshots": int(len(atlas)),
            "atlas_trajectories": int(atlas["_gid"].nunique()),
            "six_month_pairs": int(len(six_month_pairs)),
            "three_month_pairs": int(len(three_month_pairs)),
            "branch_trajectories": int(branch["_gid"].nunique()),
        },
    }
    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    write_readme(report)
    for figure in report["figures"].values():
        print(f"[done] {figure['png']}")
    print(f"[done] {OUT.relative_to(ROOT) / 'README.md'}")
    print(f"[done] {OUT.relative_to(ROOT) / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
