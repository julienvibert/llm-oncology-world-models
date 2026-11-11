#!/usr/bin/env python3
"""One clinical map, many conditional futures — conceptual and data-grounded drafts.

It creates:

* a deliberately simple conceptual figure using disclosed mock geometry;
* a data-grounded figure using persisted generated atlas and branching paths;
* a data-grounded intervention field using the locked Sonnet Phase-1 run.

All data-grounded panels use one PCA fitted once across the configured atlas
derivations. The PCA contains contemporaneous tumour and host state variables
together. Treatment, cancer identity, history, demographics, time, and outcome
are not coordinate inputs; they condition or annotate movement over the map.

Run:
  uv run python scripts/explore/build_one_map_many_futures_exercise.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.colorbar import ColorbarBase  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_fig1_same_dot_draft import (  # noqa: E402
    collect,
    find_matched_cell,
    pick_exemplars,
    progression_times,
)
from build_phase_plane_grammar_draft import ALL_FEATURES  # noqa: E402
from composite_tumor_host_phase_space import (  # noqa: E402
    AtlasTables,
    MONTH_DAYS,
    accepted,
    enrich_snapshots,
    load_tables,
)

OUT = ROOT / "outputs" / "exploratory_one_map_many_futures"
PHASE1_ROOT = ROOT / "runs" / "interventional_phase1_sonnet_full_v1c"
BRANCH_ROOT = (
    ROOT
    / "outputs"
    / "exploratory_virtual_patient_branching_pilot"
    / "live_runs"
    / "live01"
)
BRANCH_SUMMARY = (
    ROOT
    / "outputs"
    / "exploratory_virtual_patient_branching_pilot"
    / "readouts"
    / "live01"
    / "branch_summary.csv"
)

INK = "#162033"
MUTED = "#66758c"
GRID = "#dce4ec"
PAPER = "#fbfcfe"
BLUE = "#3474b8"
TEAL = "#138f86"
GREEN = "#208a53"
ORANGE = "#df7900"
RED = "#c63e3e"
PURPLE = "#6846a5"
GOLD = "#c59a24"
LIGHT = "#eef3f7"
SEED = 20260728

ARM_COLOR = {"osimertinib": TEAL, "chemo": ORANGE}
CONTEXT_LABEL = {
    "EGFR_1L": "EGFR-mutant NSCLC",
    "EGFR_WT_1L": "EGFR-wild-type NSCLC",
}

EXAMPLE_COURSES = (
    (
        "sonnet5_mini_atlas_n100_2026_07_08::"
        "HGSOC_tumor_only__claude_sonnet5__f04",
        "ovarian cancer",
        PURPLE,
    ),
    (
        "sonnet5_mini_atlas_n100_2026_07_08::"
        "PAAD_tumor_only__claude_sonnet5__f02",
        "pancreatic cancer",
        ORANGE,
    ),
    (
        "pillar1_anchored_atlas_tranche1_v4_n5_live01::"
        "HCC_tumor_only__qwen36_flash__f01",
        "liver cancer",
        TEAL,
    ),
)


@dataclass(frozen=True)
class DisplayMap:
    """One atlas-fitted PCA basis reused by every data-grounded panel."""

    features: tuple[str, ...]
    clip_low: np.ndarray
    clip_high: np.ndarray
    scaler: StandardScaler
    pca: PCA
    signs: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    explained_variance: np.ndarray


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_derivation(root: Path) -> AtlasTables:
    """Load one persisted derivation with the same keys as the atlas loader."""
    derivation = require(root / "derivation")
    run = root.name

    def read(name: str) -> pd.DataFrame:
        frame = pd.read_csv(require(derivation / name), low_memory=False)
        frame = accepted(frame)
        frame["_run"] = run
        frame["_gid"] = run + "::" + frame["trajectory_id"].astype(str)
        return frame

    return AtlasTables(
        snapshots=read("snapshot_rows.csv"),
        sites=read("disease_site_rows.csv"),
        events=read("event_rows.csv"),
        endpoints=read("endpoint_rows.csv"),
    )


def state_matrix(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    values = frame[list(features)].fillna(0.0).to_numpy(float)
    return np.clip(values, low, high)


def fit_display_map(frame: pd.DataFrame) -> tuple[DisplayMap, pd.DataFrame]:
    """Fit a robust cross-cancer PCA using current clinical-state variables only."""
    features = tuple(ALL_FEATURES)
    fit = frame[~frame["terminal"]].copy()
    fit_values = fit[list(features)].fillna(0.0)
    low = fit_values.quantile(0.005).to_numpy(float)
    high = fit_values.quantile(0.995).to_numpy(float)
    x_fit = state_matrix(fit, features, low, high)
    # PCA requires comparable feature variance. Standardization after explicit
    # tail clipping prevents sparse summed-burden variables from monopolizing a
    # component when their unscaled IQR is zero.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_fit)
    pca = PCA(n_components=4, random_state=SEED)
    fit_scores = pca.fit_transform(x_scaled)

    severity_reference = (
        fit["tumour_burden_score"]
        + 0.7 * fit["num_active_sites_num"]
        + fit["ecog"]
        + 0.6 * fit["organ_max"]
    ).to_numpy(float)
    host_reference = (
        fit["ecog"]
        + 0.6 * fit["organ_max"]
        + 0.4 * fit["symptom_ord"]
        + 0.25 * fit["n_active_aes_num"]
    ).to_numpy(float)
    signs = np.ones(2)
    if np.corrcoef(fit_scores[:, 0], severity_reference)[0, 1] < 0:
        signs[0] = -1.0
    if np.corrcoef(fit_scores[:, 1], host_reference)[0, 1] < 0:
        signs[1] = -1.0
    signed_fit = fit_scores[:, :2] * signs
    centers = np.median(signed_fit, axis=0)
    q75, q25 = np.quantile(signed_fit, [0.75, 0.25], axis=0)
    scales = q75 - q25
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    display = DisplayMap(
        features=features,
        clip_low=low,
        clip_high=high,
        scaler=scaler,
        pca=pca,
        signs=signs,
        centers=centers,
        scales=scales,
        explained_variance=pca.explained_variance_ratio_,
    )
    return display, transform_to_map(frame, display)


def transform_to_map(frame: pd.DataFrame, display: DisplayMap) -> pd.DataFrame:
    values = state_matrix(frame, display.features, display.clip_low, display.clip_high)
    scores = display.pca.transform(display.scaler.transform(values))[:, :2]
    scores = scores * display.signs
    scores = (scores - display.centers) / display.scales
    out = frame.copy()
    out["map_x"] = scores[:, 0]
    out["map_y"] = scores[:, 1]
    # Compatibility aliases for the matched-neighbourhood helper.
    out["sev"] = out["map_x"]
    out["pc2"] = out["map_y"]
    return out


def map_limits(
    atlas: pd.DataFrame,
    *additional: pd.DataFrame,
) -> tuple[float, float, float, float]:
    base = atlas[~atlas["terminal"]][["map_x", "map_y"]]
    xmin, xmax = base["map_x"].quantile([0.005, 0.995])
    ymin, ymax = base["map_y"].quantile([0.005, 0.995])
    for frame in additional:
        if frame.empty:
            continue
        xmin = min(float(xmin), float(frame["map_x"].quantile(0.005)))
        xmax = max(float(xmax), float(frame["map_x"].quantile(0.995)))
        ymin = min(float(ymin), float(frame["map_y"].quantile(0.005)))
        ymax = max(float(ymax), float(frame["map_y"].quantile(0.995)))
    xpad = 0.06 * max(float(xmax - xmin), 1.0)
    ypad = 0.06 * max(float(ymax - ymin), 1.0)
    return (
        float(xmin - xpad),
        float(xmax + xpad),
        float(ymin - ypad),
        float(ymax + ypad),
    )


def style_map(
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    *,
    labels: bool = True,
) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(color=GRID, lw=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if labels:
        ax.set_xlabel(
            "shared clinical state PC1\nlower combined severity  ←        →  higher",
            fontsize=8.2,
        )
        ax.set_ylabel("shared clinical state PC2\nresidual state pattern", fontsize=8.2)
    ax.tick_params(labelsize=7)


def add_context_strip(ax: plt.Axes, text: str, color: str = BLUE) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.2,
            "alpha": 0.95,
        },
        zorder=20,
    )


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
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=linewidth,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
    )


def mock_landscape(ax: plt.Axes) -> None:
    xx = np.linspace(-3.2, 3.2, 160)
    yy = np.linspace(-2.7, 2.7, 140)
    xg, yg = np.meshgrid(xx, yy)
    z = (
        1.15 * np.exp(-((xg + 1.7) ** 2 / 1.1 + (yg + 0.75) ** 2 / 0.7))
        + 0.80 * np.exp(-((xg - 0.1) ** 2 / 1.8 + (yg - 0.25) ** 2 / 1.4))
        + 0.62 * np.exp(-((xg - 2.1) ** 2 / 0.8 + (yg - 1.2) ** 2 / 0.8))
    )
    ax.contourf(xg, yg, z, levels=8, cmap="Blues", alpha=0.34)
    ax.contour(xg, yg, z, levels=5, colors="#b9c9d8", linewidths=0.6)


def conceptual_figure() -> Path:
    """Draw the visual grammar with disclosed mock trajectories."""
    limits = (-3.2, 3.2, -2.7, 2.7)
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.5))
    fig.patch.set_facecolor(PAPER)

    # A — one fixed geography.
    ax = axes[0]
    mock_landscape(ax)
    paths = [
        (np.array([2.4, 1.7, 0.8, -0.2, -1.35]), np.array([1.55, 1.05, 0.35, -0.35, -0.85]), GREEN, "response"),
        (np.array([-1.8, -1.55, -1.72, -1.48, -1.62]), np.array([-0.72, -0.62, -0.83, -0.70, -0.78]), BLUE, "stable control"),
        (np.array([-1.8, -1.0, 0.0, 1.1, 2.4]), np.array([-0.7, -0.3, 0.2, 0.8, 1.6]), RED, "progression"),
    ]
    for x, y, color, label in paths:
        ax.plot(x, y, "-o", color=color, lw=2.5, ms=4, label=label)
        add_arrow(ax, (x[-2], y[-2]), (x[-1], y[-1]), color)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    style_map(ax, limits)
    ax.set_title("A · One fixed clinical geography\npatients move; the axes do not", loc="left", fontsize=10.5, weight="bold", color=INK)

    # B — memory changes the future at the same dot.
    ax = axes[1]
    mock_landscape(ax)
    now = (0.0, 0.0)
    trails = [
        (np.array([-2.7, -1.8, -0.9, 0.0]), np.array([1.6, 0.9, 0.35, 0.0]), BLUE),
        (np.array([2.4, 1.6, 0.7, 0.0]), np.array([-1.6, -1.0, -0.45, 0.0]), RED),
    ]
    futures = [((-1.2, -0.9), BLUE), ((1.4, 1.0), RED)]
    for x, y, color in trails:
        ax.plot(x, y, color=color, lw=2.4, alpha=0.8)
        add_arrow(ax, (x[-2], y[-2]), now, color)
    ax.scatter(*now, s=180, facecolor="white", edgecolor=INK, lw=2.5, zorder=12)
    for end, color in futures:
        add_arrow(ax, now, end, color, linewidth=3.0, mutation_scale=17)
    ax.text(-2.8, 2.25, "different incoming histories", fontsize=8.5, color=INK, weight="bold")
    ax.text(-0.55, -0.25, "same dot now", fontsize=8.2, color=INK, weight="bold")
    style_map(ax, limits)
    ax.set_title("B · History selects the forecast\nsame location ≠ same future", loc="left", fontsize=10.5, weight="bold", color=INK)

    # C — treatments can share an initial response but differ in residence time.
    ax = axes[2]
    mock_landscape(ax)
    ax.add_patch(Ellipse((-1.65, -0.78), 1.35, 0.95, facecolor=GREEN, edgecolor=GREEN, alpha=0.11, lw=1.5))
    target_x = np.array([0.9, 0.2, -0.8, -1.45, -1.62, -1.48, -1.66])
    target_y = np.array([0.8, 0.2, -0.45, -0.78, -0.70, -0.84, -0.76])
    chemo_x = np.array([0.9, 0.2, -0.8, -1.42, -1.05, -0.25, 0.95])
    chemo_y = np.array([0.8, 0.2, -0.45, -0.76, -0.52, -0.02, 0.78])
    months = np.arange(0, 21, 3)
    ax.plot(chemo_x, chemo_y, "-o", color=ORANGE, lw=2.6, ms=4.5, label="chemotherapy")
    ax.plot(target_x, target_y, "-o", color=TEAL, lw=2.6, ms=4.5, label="targeted therapy")
    ax.scatter(target_x[0], target_y[0], s=95, facecolor="white", edgecolor=INK, lw=1.8, zorder=12)
    add_arrow(ax, (target_x[1], target_y[1]), (target_x[2], target_y[2]), TEAL, linewidth=2.8)
    add_arrow(ax, (chemo_x[-2], chemo_y[-2]), (chemo_x[-1], chemo_y[-1]), ORANGE, linewidth=2.8)
    ax.text(0.45, 0.38, "same initial\nresponse", fontsize=7.8, color=INK, weight="bold")
    ax.text(-2.45, -1.55, "targeted therapy remains\nin control at month 18", fontsize=7.6, color=TEAL, weight="bold")
    ax.text(0.62, 1.22, "chemotherapy escapes\nafter initial control", fontsize=7.6, color=ORANGE, weight="bold")
    for index in (0, 2, 4, 6):
        ax.text(
            target_x[index] - 0.08,
            target_y[index] - 0.18,
            f"{months[index]}m",
            fontsize=6.4,
            color=TEAL,
            ha="center",
        )
    add_context_strip(
        ax,
        "CONTEXT LENS\nsame patient and starting state · two interventions",
        PURPLE,
    )
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    style_map(ax, limits)
    ax.set_title("C · Benefit can be longer residence\nsame initial response; different escape time", loc="left", fontsize=10.5, weight="bold", color=INK)

    # D — the treatment-effect layer can be a residence-time field.
    ax = axes[3]
    mock_landscape(ax)
    basin = np.array([-1.7, -0.8])
    ax.add_patch(Ellipse(tuple(basin), 1.45, 1.0, facecolor=GREEN, edgecolor=GREEN, alpha=0.10, lw=1.6))
    starts = np.array([[-2.3, 0.2], [-1.1, 0.65], [-0.1, -0.1], [0.9, 0.75], [2.0, 1.25]])
    extra_months = np.array([3, 6, 9, 13, 17])
    norm = Normalize(0, 18)
    for start, extra in zip(starts, extra_months, strict=True):
        early_end = start + 0.34 * (basin - start)
        add_arrow(
            ax,
            (float(start[0]), float(start[1])),
            (float(early_end[0]), float(early_end[1])),
            MUTED,
            linewidth=1.5,
            alpha=0.75,
            mutation_scale=11,
            zorder=6,
        )
        face = plt.cm.YlGnBu(norm(float(extra)))
        ax.scatter(
            start[0],
            start[1],
            s=70 + 5 * extra,
            facecolor=face,
            edgecolor=INK,
            lw=1.1,
            zorder=9,
        )
        ax.text(
            start[0],
            start[1],
            f"+{extra}m",
            fontsize=6.4,
            color="white" if extra >= 9 else INK,
            weight="bold",
            ha="center",
            va="center",
            zorder=10,
        )
    ax.text(-2.45, -1.5, "favourable-state\nregion", fontsize=7.6, color=GREEN, weight="bold")
    ax.text(
        0.03,
        0.97,
        "grey arrow = early movement\ncoloured circle = extra months in a good state",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "alpha": 0.95},
        zorder=20,
    )
    style_map(ax, limits)
    ax.set_title("D · The treatment-effect field need not be a vector\nmap the extra time spent in favourable states", loc="left", fontsize=10.5, weight="bold", color=INK)

    fig.suptitle(
        "ONE CLINICAL MAP, MANY CONDITIONAL FUTURES",
        x=0.025,
        y=0.99,
        ha="left",
        fontsize=17,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.93,
        "Conceptual mock-up: arrows show where; trajectories and residence-time layers show how long a patient remains in a favourable state.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.99,
        0.01,
        "CONCEPTUAL ONLY · mock geometry · no empirical values",
        ha="right",
        fontsize=7.5,
        color=RED,
        weight="bold",
    )
    fig.tight_layout(rect=(0.015, 0.035, 0.995, 0.90), w_pad=1.5)
    path = OUT / "fig_concept_one_map_many_futures.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def background_points(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    *,
    n: int = 15000,
) -> None:
    fit = atlas[~atlas["terminal"]]
    if len(fit) > n:
        fit = fit.sample(n=n, random_state=SEED)
    ax.scatter(
        fit["map_x"],
        fit["map_y"],
        s=2.0,
        color="#cfd8e2",
        alpha=0.22,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )


def plot_actual_courses(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    background_points(ax, atlas)
    reports = []
    for gid, label, color in EXAMPLE_COURSES:
        course = atlas[atlas["_gid"].eq(gid)].sort_values(["month", "snapshot_index"])
        if course.empty:
            continue
        ax.plot(course["map_x"], course["map_y"], color=color, lw=2.5, alpha=0.9, zorder=5)
        ax.scatter(
            course["map_x"],
            course["map_y"],
            c=course["month"],
            cmap="viridis",
            s=16,
            edgecolor="white",
            lw=0.35,
            zorder=6,
        )
        start = course.iloc[0]
        end = course.iloc[-1]
        ax.scatter(start["map_x"], start["map_y"], marker="*", s=90, color=color, edgecolor="white", lw=0.7, zorder=8)
        ax.scatter(end["map_x"], end["map_y"], marker="X", s=65, color=color, edgecolor="white", lw=0.7, zorder=8)
        ax.text(
            float(end["map_x"]) + 0.08,
            float(end["map_y"]) + 0.05,
            label,
            fontsize=7.5,
            color=color,
            weight="bold",
            zorder=9,
        )
        reports.append(
            {
                "gid": gid,
                "label": label,
                "n_snapshots": int(len(course)),
                "os_months": float(course["death_month"].max()),
            }
        )
    style_map(ax, limits)
    ax.set_title(
        "A · Individual journeys across one atlas\nactual persisted generated paths",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )
    add_context_strip(ax, "STAR = diagnosis · X = terminal snapshot · colour along each path = time")
    return reports


def plot_matched_neighbourhood(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    events: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> dict[str, object]:
    tumour = "LMS"
    sub = atlas[atlas["tumor_oncotree_code"].astype(str).eq(tumour)].copy()
    box = find_matched_cell(sub)
    entries = collect(sub, box, progression_times(events))
    shown = pick_exemplars(entries, k=min(8, len(entries)))
    background = sub[~sub["terminal"]]
    ax.scatter(background["map_x"], background["map_y"], s=3, color="#d6dee7", alpha=0.25, edgecolors="none", zorder=1)
    for entry in shown:
        future_color = RED if entry["died"] else BLUE
        trail = entry["trail"]
        future = entry["future"]
        ax.plot(trail["map_x"], trail["map_y"], color=MUTED, lw=1.5, alpha=0.65, linestyle="--", zorder=4)
        ax.plot(future["map_x"], future["map_y"], color=future_color, lw=2.0, alpha=0.85, zorder=5)
        current = future.iloc[0]
        ax.scatter(
            current["map_x"],
            current["map_y"],
            s=42,
            facecolor="white",
            edgecolor=future_color,
            lw=1.3,
            zorder=7,
        )
    x0, x1, y0, y1 = box
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=GOLD,
            edgecolor=GOLD,
            alpha=0.12,
            lw=2,
            zorder=2,
        )
    )
    ax.plot([], [], color=MUTED, lw=1.5, linestyle="--", label="observed incoming history")
    ax.plot([], [], color=BLUE, lw=2, label="alive beyond 12 months")
    ax.plot([], [], color=RED, lw=2, label="death within 12 months")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    style_map(ax, limits)
    ax.set_title(
        "B · Similar location, different trails and futures\none PCA neighbourhood in leiomyosarcoma",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )
    add_context_strip(ax, f"{len(entries)} generated trajectories pass through the highlighted neighbourhood")
    return {
        "tumour": tumour,
        "n_in_neighbourhood": int(len(entries)),
        "n_shown": int(len(shown)),
        "box": [float(value) for value in box],
    }


def nearest_snapshot(group: pd.DataFrame, day: float, *, after: bool = False) -> pd.Series:
    candidates = group[group["days_from_diagnosis"].ge(day)] if after else group
    if candidates.empty:
        candidates = group
    index = (candidates["days_from_diagnosis"] - day).abs().idxmin()
    return candidates.loc[index]


def plot_branch_fan(
    ax: plt.Axes,
    branch: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> dict[str, object]:
    summary = pd.read_csv(require(BRANCH_SUMMARY))
    policy = "naturalistic_p01_after_treatment_start_folfirinox"
    meta = summary[summary["policy_id"].eq(policy)]
    if meta.empty:
        raise ValueError(f"Missing branch summary for {policy}")
    cutoff_days = float(meta.iloc[0]["cutoff_days"])
    selected = branch[branch["policy_id"].eq(policy)].copy()
    trajectories = list(selected["_gid"].unique())
    if not trajectories:
        raise ValueError(f"Missing branching snapshots for {policy}")

    first = selected[selected["_gid"].eq(trajectories[0])].sort_values("days_from_diagnosis")
    observed = first[first["days_from_diagnosis"].le(cutoff_days)]
    ax.plot(observed["map_x"], observed["map_y"], color=INK, lw=3.0, zorder=7, label="shared observed history")
    now = nearest_snapshot(first, cutoff_days)
    palette = plt.cm.Purples(np.linspace(0.45, 0.9, len(trajectories)))
    three_month_endpoints = []
    remaining = []
    for color, gid in zip(palette, trajectories, strict=True):
        course = selected[selected["_gid"].eq(gid)].sort_values("days_from_diagnosis")
        future = course[course["days_from_diagnosis"].ge(cutoff_days)]
        ax.plot(future["map_x"], future["map_y"], color=color, lw=2.0, alpha=0.85, zorder=5)
        end = future.iloc[-1]
        ax.scatter(end["map_x"], end["map_y"], marker="X", s=42, color=color, edgecolor="white", lw=0.5, zorder=8)
        three = nearest_snapshot(course, cutoff_days + 3 * MONTH_DAYS, after=True)
        three_month_endpoints.append((float(three["map_x"]), float(three["map_y"])))
        remaining.append(float(end["days_from_diagnosis"] - cutoff_days) / MONTH_DAYS)
    median_three = np.median(np.asarray(three_month_endpoints), axis=0)
    ax.scatter(now["map_x"], now["map_y"], s=160, facecolor="white", edgecolor=PURPLE, lw=2.5, zorder=11)
    add_arrow(
        ax,
        (float(now["map_x"]), float(now["map_y"])),
        (float(median_three[0]), float(median_three[1])),
        PURPLE,
        linewidth=3.2,
        mutation_scale=17,
        zorder=12,
    )
    ax.plot([], [], color=PURPLE, lw=2.0, label="five subsequently generated futures")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    style_map(ax, limits)
    ax.set_title(
        "C · A local arrow is only the beginning\nsame-history PAAD branch: 3-month arrow and full futures",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )
    add_context_strip(ax, "PAAD · after FOLFIRINOX start · 5 accepted generated futures", PURPLE)
    return {
        "policy_id": policy,
        "n_futures": int(len(trajectories)),
        "cutoff_days": cutoff_days,
        "future_months_range": [float(min(remaining)), float(max(remaining))],
    }


def data_journeys_figure(
    atlas: pd.DataFrame,
    atlas_events: pd.DataFrame,
    branch: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[Path, dict[str, object]]:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.1))
    fig.patch.set_facecolor(PAPER)
    courses = plot_actual_courses(axes[0], atlas, limits)
    neighbourhood = plot_matched_neighbourhood(axes[1], atlas, atlas_events, limits)
    fan = plot_branch_fan(axes[2], branch, limits)
    fig.suptitle(
        "THE SAME CLINICAL MAP, DRAWN FROM PERSISTED GENERATED DATA",
        x=0.025,
        y=0.99,
        ha="left",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.935,
        "One atlas-fitted PCA basis is reused in every panel. Paths are stored model-generated trajectories—not human patients and not schematic curves.",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.015, 0.03, 0.995, 0.90), w_pad=1.8)
    path = OUT / "fig_data_shared_map_journeys.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path, {
        "course_examples": courses,
        "matched_neighbourhood": neighbourhood,
        "branch_fan": fan,
    }


def phase1_vectors(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_days: float = 182.5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = snapshots[
        snapshots["condition_id"].astype(str).str.contains("__H1")
        & snapshots["policy_id"].astype(str).str.endswith(("__osimertinib", "__chemo"))
    ]
    eligible_gids = set(eligible["_gid"])
    for gid, course in snapshots[snapshots["_gid"].isin(eligible_gids)].groupby("_gid", sort=False):
        course = course.sort_values(["days_from_diagnosis", "snapshot_index"])
        meta = course.iloc[0]
        condition = str(meta["condition_id"])
        policy = str(meta["policy_id"])
        parts = condition.split("__")
        context, state, history = parts[0], parts[-2], parts[-1]
        arm = policy.split("__")[-1]
        event_course = events[events["_gid"].eq(gid)].copy()
        assigned = event_course[event_course["event_type"].astype(str).eq("systemic_treatment_start")]
        if assigned.empty:
            continue
        cutoff = float(pd.to_numeric(assigned["days_from_diagnosis"], errors="coerce").min())
        start = nearest_snapshot(course, cutoff)
        end = nearest_snapshot(course, cutoff + horizon_days, after=True)
        future_exit = event_course[
            event_course["event_type"].astype(str).isin({"progression", "death"})
            & pd.to_numeric(event_course["days_from_diagnosis"], errors="coerce").gt(cutoff)
        ]
        exit_days = pd.to_numeric(future_exit["days_from_diagnosis"], errors="coerce")
        time_to_exit_months = (
            float(exit_days.min() - cutoff) / MONTH_DAYS if exit_days.notna().any() else np.nan
        )
        rows.append(
            {
                "_gid": gid,
                "context": context,
                "state": state,
                "history": history,
                "arm": arm,
                "x0": float(start["map_x"]),
                "y0": float(start["map_y"]),
                "x1": float(end["map_x"]),
                "y1": float(end["map_y"]),
                "dx": float(end["map_x"] - start["map_x"]),
                "dy": float(end["map_y"] - start["map_y"]),
                "time_to_exit_months": time_to_exit_months,
                "actual_horizon_days": float(end["days_from_diagnosis"] - cutoff),
            }
        )
    return pd.DataFrame(rows)


def summarize_field(vectors: pd.DataFrame) -> pd.DataFrame:
    return (
        vectors.groupby(["context", "state", "history", "arm"], as_index=False)
        .agg(
            x0=("x0", "median"),
            y0=("y0", "median"),
            dx=("dx", "median"),
            dy=("dy", "median"),
            time_to_exit_months=("time_to_exit_months", "median"),
            n=("time_to_exit_months", "size"),
        )
        .assign(x1=lambda d: d["x0"] + d["dx"], y1=lambda d: d["y0"] + d["dy"])
    )


def draw_arm_field(
    ax: plt.Axes,
    summary: pd.DataFrame,
    atlas: pd.DataFrame,
    context: str,
    arm: str,
    limits: tuple[float, float, float, float],
) -> None:
    background_points(ax, atlas, n=8000)
    selected = summary[summary["context"].eq(context) & summary["arm"].eq(arm)]
    norm = Normalize(0, 24)
    for row in selected.sort_values("state").itertuples(index=False):
        color = plt.cm.RdYlGn(norm(float(row.time_to_exit_months)))
        ax.scatter(row.x0, row.y0, s=95, facecolor=color, edgecolor=INK, lw=1.0, zorder=9)
        add_arrow(
            ax,
            (float(row.x0), float(row.y0)),
            (float(row.x1), float(row.y1)),
            ARM_COLOR[arm],
            linewidth=2.4,
            mutation_scale=15,
            zorder=10,
        )
        ax.text(row.x0 + 0.06, row.y0 + 0.06, row.state, fontsize=7.3, color=INK, weight="bold", zorder=11)
    style_map(ax, limits)
    ax.set_title(
        f"{'A' if arm == 'osimertinib' else 'B'} · {arm.capitalize()}\n"
        "arrow = median 6-month movement",
        loc="left",
        fontsize=10,
        weight="bold",
        color=ARM_COLOR[arm],
    )


def draw_delta_field(
    ax: plt.Axes,
    summary: pd.DataFrame,
    atlas: pd.DataFrame,
    context: str,
    limits: tuple[float, float, float, float],
) -> list[dict[str, float | str]]:
    background_points(ax, atlas, n=8000)
    selected = summary[summary["context"].eq(context)]
    rows: list[dict[str, float | str]] = []
    norm = TwoSlopeNorm(vmin=-18, vcenter=0, vmax=18)
    for state in sorted(selected["state"].unique()):
        state_rows = selected[selected["state"].eq(state)].set_index("arm")
        if not {"osimertinib", "chemo"}.issubset(state_rows.index):
            continue
        osi = state_rows.loc["osimertinib"]
        chemo = state_rows.loc["chemo"]
        x0 = float(np.mean([osi["x0"], chemo["x0"]]))
        y0 = float(np.mean([osi["y0"], chemo["y0"]]))
        ddx = float(osi["dx"] - chemo["dx"])
        ddy = float(osi["dy"] - chemo["dy"])
        exit_delta = float(osi["time_to_exit_months"] - chemo["time_to_exit_months"])
        ax.scatter(
            x0,
            y0,
            s=105,
            facecolor=plt.cm.RdBu(norm(exit_delta)),
            edgecolor=INK,
            lw=1.0,
            zorder=9,
        )
        add_arrow(
            ax,
            (x0, y0),
            (x0 + ddx, y0 + ddy),
            PURPLE,
            linewidth=2.7,
            mutation_scale=15,
            zorder=10,
        )
        ax.text(x0 + 0.06, y0 + 0.06, state, fontsize=7.3, color=INK, weight="bold", zorder=11)
        rows.append(
            {
                "state": state,
                "delta_dx": ddx,
                "delta_dy": ddy,
                "delta_time_to_exit_months": exit_delta,
            }
        )
    style_map(ax, limits)
    ax.set_title(
        "C · Treatment effect: osimertinib − chemotherapy\n"
        "purple arrow = difference between movements",
        loc="left",
        fontsize=10,
        weight="bold",
        color=PURPLE,
    )
    return rows


def intervention_figure(
    atlas: pd.DataFrame,
    phase1: pd.DataFrame,
    phase1_events: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> tuple[Path, dict[str, object]]:
    vectors = phase1_vectors(phase1, phase1_events)
    summary = summarize_field(vectors)
    fig, axes = plt.subplots(2, 3, figsize=(16.6, 10.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    delta_reports: dict[str, object] = {}
    for row, context in enumerate(["EGFR_1L", "EGFR_WT_1L"]):
        draw_arm_field(axes[row, 0], summary, atlas, context, "osimertinib", limits)
        draw_arm_field(axes[row, 1], summary, atlas, context, "chemo", limits)
        delta_reports[context] = draw_delta_field(axes[row, 2], summary, atlas, context, limits)
        axes[row, 0].text(
            -0.16,
            0.5,
            CONTEXT_LABEL[context],
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color=INK,
        )
        for column in range(3):
            add_context_strip(
                axes[row, column],
                "H1: treatment immediately after progression · n=3 per state/arm",
                PURPLE,
            )

    risk_cax = fig.add_axes([0.18, 0.058, 0.36, 0.017])
    risk_bar = ColorbarBase(
        risk_cax,
        cmap="RdYlGn",
        norm=Normalize(0, 24),
        orientation="horizontal",
    )
    risk_bar.set_label("dot colour: median months to progression/death (green = longer)", fontsize=8)
    risk_cax.tick_params(labelsize=7)
    delta_cax = fig.add_axes([0.70, 0.058, 0.20, 0.017])
    delta_bar = ColorbarBase(
        delta_cax,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-18, vcenter=0, vmax=18),
        orientation="horizontal",
    )
    delta_bar.set_label("dot colour: Δ months to progression/death (blue favours osimertinib)", fontsize=8)
    delta_cax.tick_params(labelsize=7)
    fig.suptitle(
        "INTERVENTION AS A CONDITIONAL CHANGE IN FLOW—ON THE SAME CLINICAL MAP",
        x=0.025,
        y=0.995,
        ha="left",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.957,
        "Locked Sonnet Phase-1 futures projected into the atlas PCA. Rows change biomarker context; columns change treatment or display their difference.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "GENERATED DATA · exploratory n=3/cell field · qualitative geometry, not calibrated human treatment effects",
        ha="right",
        fontsize=7.5,
        color=RED,
        weight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.91, wspace=0.17, hspace=0.28)
    path = OUT / "fig_data_conditional_intervention_field.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path, {
        "n_trajectory_vectors": int(len(vectors)),
        "history": "H1",
        "horizon_days": 182.5,
        "cell_counts": {
            f"{context}::{arm}": int(value)
            for (context, arm), value in summary.groupby(["context", "arm"])["n"].sum().items()
        },
        "delta_fields": delta_reports,
        "median_time_to_exit_months": {
            f"{context}::{arm}": float(
                vectors[vectors["context"].eq(context) & vectors["arm"].eq(arm)][
                    "time_to_exit_months"
                ].median()
            )
            for context in ["EGFR_1L", "EGFR_WT_1L"]
            for arm in ["osimertinib", "chemo"]
        },
    }


def pca_report(display: DisplayMap) -> dict[str, object]:
    loadings = display.pca.components_[:2] * display.signs[:, None]
    return {
        "features": list(display.features),
        "explained_variance_ratio": [float(value) for value in display.explained_variance[:4]],
        "pc1_top_absolute_loadings": [
            {"feature": display.features[index], "loading": float(loadings[0, index])}
            for index in np.argsort(np.abs(loadings[0]))[::-1][:6]
        ],
        "pc2_top_absolute_loadings": [
            {"feature": display.features[index], "loading": float(loadings[1, index])}
            for index in np.argsort(np.abs(loadings[1]))[::-1][:6]
        ],
        "winsorization": "atlas nonterminal feature-wise 0.5th–99.5th percentile",
        "coordinate_scaling": "each displayed PC centered at atlas median and divided by atlas IQR",
        "excluded_from_coordinates": [
            "treatment",
            "cancer identity",
            "history",
            "sex and demographics",
            "time",
            "outcome",
        ],
    }


def write_readout(
    conceptual_path: Path,
    journeys_path: Path,
    intervention_path: Path,
    display: DisplayMap,
    atlas: pd.DataFrame,
    journey_report: dict[str, object],
    intervention_report: dict[str, object],
    limits: tuple[float, float, float, float],
) -> None:
    report = {
        "figures": {
            "conceptual": str(conceptual_path.relative_to(ROOT)),
            "data_journeys": str(journeys_path.relative_to(ROOT)),
            "data_intervention": str(intervention_path.relative_to(ROOT)),
        },
        "conceptual_provenance": "Disclosed mock geometry; no empirical values.",
        "data_provenance": (
            "Persisted model-generated derivation rows. No path geometry was invented "
            "in the data-grounded panels. These are not human patients."
        ),
        "atlas": {
            "n_snapshots": int(len(atlas)),
            "n_trajectories": int(atlas["_gid"].nunique()),
            "runs": sorted(atlas["_run"].astype(str).unique()),
        },
        "display_map": pca_report(display),
        "shared_limits": [float(value) for value in limits],
        "journeys": journey_report,
        "intervention": intervention_report,
    }
    (OUT / "readout.json").write_text(json.dumps(report, indent=2) + "\n")

    ev = report["display_map"]["explained_variance_ratio"]
    lines = [
        "# One clinical map, many conditional futures — exercise",
        "",
        "## Provenance",
        "",
        "- Conceptual figure: disclosed mock geometry.",
        "- Data figures: persisted generated snapshots/events only; not human patients.",
        "",
        "## Shared display map",
        "",
        f"- {report['atlas']['n_snapshots']:,} snapshots from "
        f"{report['atlas']['n_trajectories']:,} trajectories.",
        f"- PC1 variance: {100 * ev[0]:.1f}%; PC2 variance: {100 * ev[1]:.1f}%.",
        "- PCA inputs: contemporaneous tumour and host state variables together.",
        "- Treatment, cancer identity, history, demographics, time and outcome are excluded from the coordinates.",
        "- Feature-wise 0.5th–99.5th percentile winsorization is used only to prevent extreme generated values from dominating the display basis.",
        "",
        "## Data availability conclusions",
        "",
        f"- Atlas course examples: {len(journey_report['course_examples'])}.",
        f"- Matched LMS neighbourhood: "
        f"{journey_report['matched_neighbourhood']['n_in_neighbourhood']} trajectories; "
        f"{journey_report['matched_neighbourhood']['n_shown']} shown.",
        f"- Same-history PAAD branch: {journey_report['branch_fan']['n_futures']} futures; "
        f"follow-up range {journey_report['branch_fan']['future_months_range'][0]:.1f}–"
        f"{journey_report['branch_fan']['future_months_range'][1]:.1f} months.",
        f"- Intervention vectors: {intervention_report['n_trajectory_vectors']}; "
        "n=3 per state/arm in the displayed H1 slice.",
        "- Median generated time to progression/death:",
        f"  - EGFR-mutant: osimertinib "
        f"{intervention_report['median_time_to_exit_months']['EGFR_1L::osimertinib']:.1f} "
        f"months vs chemotherapy "
        f"{intervention_report['median_time_to_exit_months']['EGFR_1L::chemo']:.1f}.",
        f"  - EGFR-wild-type: osimertinib "
        f"{intervention_report['median_time_to_exit_months']['EGFR_WT_1L::osimertinib']:.1f} "
        f"months vs chemotherapy "
        f"{intervention_report['median_time_to_exit_months']['EGFR_WT_1L::chemo']:.1f}.",
        "",
        "## Important limits",
        "",
        "- Occupied or convergent regions are not labelled as formal attractors.",
        "- Phase-1 treatment geometry is qualitative and generated; it is not a calibrated human causal effect.",
        "- The 2D PCA is a shared display map, not the complete predictive state.",
    ]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conceptual_path = conceptual_figure()

    atlas_tables = load_tables()
    atlas_raw = enrich_snapshots(atlas_tables)
    display, atlas = fit_display_map(atlas_raw)

    branch_tables = load_derivation(BRANCH_ROOT)
    branch = transform_to_map(enrich_snapshots(branch_tables), display)
    phase1_tables = load_derivation(PHASE1_ROOT)
    phase1 = transform_to_map(enrich_snapshots(phase1_tables), display)
    limits = map_limits(atlas, branch, phase1)

    journeys_path, journey_report = data_journeys_figure(
        atlas,
        atlas_tables.events,
        branch,
        limits,
    )
    intervention_path, intervention_report = intervention_figure(
        atlas,
        phase1,
        phase1_tables.events,
        limits,
    )
    write_readout(
        conceptual_path,
        journeys_path,
        intervention_path,
        display,
        atlas,
        journey_report,
        intervention_report,
        limits,
    )
    print(f"[done] {conceptual_path}")
    print(f"[done] {journeys_path}")
    print(f"[done] {intervention_path}")
    print(f"[done] {OUT / 'readout.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
