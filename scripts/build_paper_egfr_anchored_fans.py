#!/usr/bin/env python3
"""Build the paper-facing EGFR anchored-fan virtual-trial figure.

This is a read-only analysis of the locked Sonnet Phase-1 run. It makes no
model calls and changes no generation artifact.

The figure separates three empirical objects:

1. raw future branches from one exact state/history/treatment-decision anchor;
2. a local conditional field across five matched anchors at a fixed horizon;
3. residence/exit curves across all matched state-history cells.

The two-dimensional map is a versioned *display map*, not the estimand. It
reproduces the earlier 13-feature combined host+tumour PCA on the fixed four-run
atlas reference. Treatment, context, history, time, and outcome are not inputs.
The treatment result itself is computed from the pre-registered Phase-1 endpoint
table and therefore does not depend on this display map.

Run:
  uv run python scripts/build_paper_egfr_anchored_fans.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "runs" / "interventional_phase1_sonnet_full_v1c"
DEFAULT_ENDPOINTS = (
    ROOT
    / "outputs"
    / "exploratory_interventional_phase1"
    / "phase1_endpoint_long.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "paper_egfr_anchored_fans"

ATLAS_RUNS = (
    "pillar1_anchored_atlas_tranche1_v4_n5_live01",
    "pillar1_anchored_atlas_gemma4_n5_live01",
    "sonnet5_mini_atlas_n100_2026_07_08",
    "sonnet5_full_atlas_topup_remaining128_n5_2026_07_09",
)

MONTH_DAYS = 30.4
MAP_ID = "combined13_robust10-90_atlas4_v0"
MAP_FEATURES = (
    "tumour_burden_band",
    "num_active_disease_sites",
    "num_active_clones",
    "n_acquired_alterations",
    "ecog_ps",
    "symptom_burden_ord",
    "n_active_aes",
    "organ_renal_ord",
    "organ_hepatic_ord",
    "organ_cardiac_ord",
    "organ_marrow_reserve_ord",
    "organ_pulmonary_ord",
    "weight_kg_clipped",
)
FEATURE_LABELS = {
    "tumour_burden_band": "tumour burden",
    "num_active_disease_sites": "metastatic sites",
    "num_active_clones": "active clones",
    "n_acquired_alterations": "acquired alterations",
    "ecog_ps": "ECOG",
    "symptom_burden_ord": "symptoms",
    "n_active_aes": "active adverse events",
    "organ_renal_ord": "renal impairment",
    "organ_hepatic_ord": "hepatic impairment",
    "organ_cardiac_ord": "cardiac impairment",
    "organ_marrow_reserve_ord": "marrow impairment",
    "organ_pulmonary_ord": "pulmonary impairment",
    "weight_kg_clipped": "weight",
}

BURDEN_SCORE = {
    "absent": 0.0,
    "none": 0.0,
    "microscopic": 0.5,
    "small": 1.0,
    "moderate": 2.0,
    "bulky": 3.0,
    "extensive": 4.0,
    "diffuse_or_leukemic": 4.0,
}
SYMPTOM_SCORE = {
    "none": 0.0,
    "minimal": 0.5,
    "low": 1.0,
    "mild": 1.0,
    "moderate": 2.0,
    "high": 3.0,
    "severe": 3.0,
    "very_severe": 4.0,
    "extreme": 4.0,
}
ORGAN_SCORE = {
    "normal": 0.0,
    "mild_impairment": 1.0,
    "moderate_impairment": 2.0,
    "severe_impairment": 3.0,
    "failure": 4.0,
}
ORGAN_COLUMNS = (
    "organ_renal",
    "organ_hepatic",
    "organ_cardiac",
    "organ_marrow_reserve",
    "organ_pulmonary",
)

CONTEXTS = ("EGFR_1L", "EGFR_WT_1L")
CONTEXT_LABEL = {
    "EGFR_1L": "EGFR-mutant LUAD",
    "EGFR_WT_1L": "EGFR-wild-type LUAD",
}
ARMS = ("supportive", "chemo", "osimertinib")
ARM_LABEL = {
    "supportive": "supportive care",
    "chemo": "carboplatin–pemetrexed",
    "osimertinib": "osimertinib",
}
ARM_SHORT = {
    "supportive": "BSC",
    "chemo": "chemotherapy",
    "osimertinib": "osimertinib",
}
ARM_COLOR = {
    "supportive": "#71717a",
    "chemo": "#d97706",
    "osimertinib": "#0f8f83",
}
STATE_LABEL = {
    "S1": "low burden, fit",
    "S2": "standard metastatic",
    "S3": "CNS involvement",
    "S4": "high burden, fit",
    "S5": "high burden, frail",
}

INK = "#162033"
MUTED = "#64748b"
GRID = "#dce4ec"
PAPER = "#fbfcfe"
EXIT = "#a51c30"
SEED = 20260729


@dataclass(frozen=True)
class DisplayMap:
    """Frozen transformation used only to display state trajectories."""

    medians: pd.Series
    scaler: RobustScaler
    pca: PCA
    explained: tuple[float, float]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _ordinal(
    frame: pd.DataFrame,
    column: str,
    mapping: dict[str, float],
) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    labels = frame[column].astype(str).str.strip().str.lower().map(mapping)
    return numeric.where(numeric.notna(), labels).astype(float)


def add_map_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the 13 contemporaneous variables in the display-map contract."""
    out = frame.copy()
    out["tumour_burden_band"] = _ordinal(
        out,
        "total_burden_band",
        BURDEN_SCORE,
    )
    out["symptom_burden_ord"] = _ordinal(
        out,
        "symptom_burden",
        SYMPTOM_SCORE,
    )
    for organ in ORGAN_COLUMNS:
        out[f"{organ}_ord"] = _ordinal(out, organ, ORGAN_SCORE)
    for column in (
        "num_active_disease_sites",
        "num_active_clones",
        "n_acquired_alterations",
        "ecog_ps",
        "n_active_aes",
        "days_from_diagnosis",
    ):
        out[column] = _numeric(out, column)
    out["weight_kg_clipped"] = _numeric(out, "weight_kg").clip(35, 130)
    return out


def load_atlas_snapshots() -> pd.DataFrame:
    """Load the fixed four-run reference cohort used by the display PCA."""
    frames: list[pd.DataFrame] = []
    for run_id in ATLAS_RUNS:
        path = _require(ROOT / "runs" / run_id / "derivation" / "snapshot_rows.csv")
        frame = pd.read_csv(path, low_memory=False)
        if "accepted" in frame:
            accepted = frame["accepted"].astype(str).str.lower().isin({"true", "1", "1.0"})
            frame = frame[accepted].copy()
        frame["_run"] = run_id
        frames.append(frame)
    return add_map_features(pd.concat(frames, ignore_index=True))


def fit_display_map(atlas: pd.DataFrame) -> tuple[DisplayMap, pd.DataFrame]:
    """Fit the versioned combined-state PCA and project the reference atlas."""
    terminal = atlas["vital_status"].astype(str).str.lower().eq("deceased")
    fit = atlas[~terminal].copy()
    matrix = fit[list(MAP_FEATURES)].copy()
    medians = matrix.median(numeric_only=True)
    matrix = matrix.fillna(medians)
    scaler = RobustScaler(quantile_range=(10, 90))
    scaled = scaler.fit_transform(matrix)
    pca = PCA(n_components=2, random_state=SEED)
    scores = pca.fit_transform(scaled)

    severity = (
        fit["tumour_burden_band"].fillna(0.0)
        + 0.25 * fit["num_active_disease_sites"].fillna(0.0)
        + fit["ecog_ps"].fillna(0.0)
        + 0.5 * fit["symptom_burden_ord"].fillna(0.0)
    )
    if np.corrcoef(scores[:, 0], severity.to_numpy(float))[0, 1] < 0:
        pca.components_[0] *= -1
    if np.corrcoef(scores[:, 1], fit["ecog_ps"].fillna(0.0).to_numpy(float))[0, 1] < 0:
        pca.components_[1] *= -1
    model = DisplayMap(
        medians=medians,
        scaler=scaler,
        pca=pca,
        explained=(
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        ),
    )
    return model, project_to_map(atlas, model)


def project_to_map(frame: pd.DataFrame, model: DisplayMap) -> pd.DataFrame:
    """Project snapshots without refitting the map."""
    matrix = frame[list(MAP_FEATURES)].copy().fillna(model.medians)
    scores = model.pca.transform(model.scaler.transform(matrix))
    out = frame.copy()
    out["map_x"] = scores[:, 0]
    out["map_y"] = scores[:, 1]
    return out


def load_phase1(run_root: Path, model: DisplayMap) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and annotate the locked Phase-1 snapshot and event tables."""
    derivation = _require(run_root / "derivation")
    snapshots = pd.read_csv(_require(derivation / "snapshot_rows.csv"), low_memory=False)
    events = pd.read_csv(_require(derivation / "event_rows.csv"), low_memory=False)
    if "accepted" in snapshots:
        snapshots = snapshots[
            snapshots["accepted"].astype(str).str.lower().isin({"true", "1", "1.0"})
        ].copy()
    parts = snapshots["condition_id"].str.split("__", expand=True)
    snapshots["context"] = parts[0]
    snapshots["state"] = parts[1]
    snapshots["history"] = parts[2]
    snapshots["arm"] = snapshots["policy_id"].str.split("__").str[-1]
    snapshots = project_to_map(add_map_features(snapshots), model)
    return snapshots, events


def cutoff_days_from_specs(run_root: Path, trajectory_ids: pd.Series) -> pd.Series:
    """Read the exact shared decision cutoff from each persisted prompt spec."""
    values: dict[str, float] = {}
    for trajectory_id in sorted(trajectory_ids.astype(str).unique()):
        attempt_paths = sorted(
            (run_root / "trajectories" / trajectory_id).glob(
                "attempt_*/conditioning_spec.json"
            )
        )
        if not attempt_paths:
            raise FileNotFoundError(
                f"No conditioning_spec.json found for {trajectory_id}"
            )
        spec = json.loads(attempt_paths[0].read_text())
        diagnosis = date.fromisoformat(
            spec["baseline_seed"]["identity_card"]["diagnosis"]["diagnosis_date"]
        )
        cutoff = date.fromisoformat(spec["observed_prefix"]["cutoff_date"])
        values[trajectory_id] = float((cutoff - diagnosis).days)
    return trajectory_ids.astype(str).map(values)


def position_at_day(course: pd.DataFrame, day: float) -> tuple[float, float]:
    """Interpolate a display position at an exact finite horizon."""
    ordered = (
        course.sort_values(["days_from_diagnosis", "snapshot_index"])
        .drop_duplicates("days_from_diagnosis", keep="last")
        .reset_index(drop=True)
    )
    days = ordered["days_from_diagnosis"].to_numpy(float)
    x = ordered["map_x"].to_numpy(float)
    y = ordered["map_y"].to_numpy(float)
    return (
        float(np.interp(day, days, x, left=x[0], right=x[-1])),
        float(np.interp(day, days, y, left=y[0], right=y[-1])),
    )


def anchor_position(course: pd.DataFrame, cutoff_day: float) -> tuple[float, float]:
    """Return the last observed state at/before the treatment decision.

    Active arms contain a treatment-start snapshot exactly at the cutoff.
    Supportive arms intentionally do not. Interpolating the supportive state
    between the last fixed history event and its first generated future would
    leak post-decision information into the anchor, so we carry the last fixed
    state forward instead.
    """
    ordered = course.sort_values(
        ["days_from_diagnosis", "snapshot_index"]
    ).reset_index(drop=True)
    observed = ordered[ordered["days_from_diagnosis"] <= cutoff_day]
    if observed.empty:
        row = ordered.iloc[0]
    else:
        row = observed.iloc[-1]
    return float(row["map_x"]), float(row["map_y"])


def path_until(
    course: pd.DataFrame,
    start_day: float,
    end_day: float,
) -> pd.DataFrame:
    """Return an event path with exact start/end positions included."""
    ordered = (
        course.sort_values(["days_from_diagnosis", "snapshot_index"])
        .drop_duplicates("days_from_diagnosis", keep="last")
        .reset_index(drop=True)
    )
    start_x, start_y = anchor_position(ordered, start_day)
    end_x, end_y = position_at_day(ordered, end_day)
    interior = ordered[
        ordered["days_from_diagnosis"].between(
            start_day,
            end_day,
            inclusive="neither",
        )
    ][["days_from_diagnosis", "map_x", "map_y"]]
    edges = pd.DataFrame(
        [
            {"days_from_diagnosis": start_day, "map_x": start_x, "map_y": start_y},
            {"days_from_diagnosis": end_day, "map_x": end_x, "map_y": end_y},
        ]
    )
    return (
        pd.concat([edges.iloc[[0]], interior, edges.iloc[[1]]], ignore_index=True)
        .sort_values("days_from_diagnosis")
        .drop_duplicates("days_from_diagnosis", keep="last")
    )


def build_horizon_table(
    snapshots: pd.DataFrame,
    endpoints: pd.DataFrame,
    run_root: Path,
    horizon_months: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one decision-anchor row per trajectory and audit invariance."""
    meta = (
        snapshots[
            ["trajectory_id", "context", "state", "history", "arm"]
        ]
        .drop_duplicates("trajectory_id")
        .copy()
    )
    meta["cutoff_days"] = cutoff_days_from_specs(run_root, meta["trajectory_id"])
    endpoint_columns = [
        "trajectory_id",
        "endpoint_kind",
        "endpoint_mo",
        "used_death_fallback",
    ]
    meta = meta.merge(
        endpoints[endpoint_columns],
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for item in meta.itertuples(index=False):
        course = snapshots[snapshots["trajectory_id"].eq(item.trajectory_id)]
        x0, y0 = anchor_position(course, float(item.cutoff_days))
        exited = float(item.endpoint_mo) <= horizon_months
        elapsed = min(float(item.endpoint_mo), horizon_months)
        target_day = float(item.cutoff_days) + elapsed * MONTH_DAYS
        xh, yh = position_at_day(course, target_day)
        rows.append(
            {
                "trajectory_id": item.trajectory_id,
                "context": item.context,
                "state": item.state,
                "history": item.history,
                "arm": item.arm,
                "cutoff_days": item.cutoff_days,
                "endpoint_kind": item.endpoint_kind,
                "endpoint_mo": item.endpoint_mo,
                "used_death_fallback": bool(item.used_death_fallback),
                "horizon_months": horizon_months,
                "x0": x0,
                "y0": y0,
                "xh": xh,
                "yh": yh,
                "dx": xh - x0,
                "dy": yh - y0,
                "exited_by_horizon": exited,
                "elapsed_months_drawn": elapsed,
            }
        )
    horizon = pd.DataFrame(rows)
    anchor_audit = (
        horizon.groupby(["context", "state", "history"], as_index=False)
        .agg(
            n_trajectories=("trajectory_id", "nunique"),
            n_arms=("arm", "nunique"),
            x_range=("x0", lambda values: float(values.max() - values.min())),
            y_range=("y0", lambda values: float(values.max() - values.min())),
        )
        .sort_values(["context", "state", "history"])
    )
    return horizon, anchor_audit


def residence_summary(
    endpoints: pd.DataFrame,
    arms: tuple[str, ...] = ARMS,
) -> pd.DataFrame:
    """Summarise model-implied time to first-line exit/progression."""
    selected = endpoints[endpoints["arm_key"].isin(arms)].copy()
    rows: list[dict[str, object]] = []
    for (context, arm), group in selected.groupby(["context", "arm_key"]):
        endpoint = group["endpoint_mo"].astype(float)
        rows.append(
            {
                "context": context,
                "arm": arm,
                "n": int(len(group)),
                "median_endpoint_mo": float(endpoint.median()),
                "mean_endpoint_mo": float(endpoint.mean()),
                "remain_at_3mo": float((endpoint > 3).mean()),
                "remain_at_6mo": float((endpoint > 6).mean()),
                "remain_at_12mo": float((endpoint > 12).mean()),
                "remain_at_18mo": float((endpoint > 18).mean()),
                "remain_at_24mo": float((endpoint > 24).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["context", "arm"])


def movement_summary(horizon: pd.DataFrame) -> pd.DataFrame:
    """Summarise the map-dependent retained-future displacement."""
    rows: list[dict[str, object]] = []
    for (context, arm), group in horizon.groupby(["context", "arm"]):
        retained = group[~group["exited_by_horizon"]]
        rows.append(
            {
                "context": context,
                "arm": arm,
                "horizon_months": float(group["horizon_months"].iloc[0]),
                "n": int(len(group)),
                "n_retained": int(len(retained)),
                "exit_fraction": float(group["exited_by_horizon"].mean()),
                "mean_dx_retained": float(retained["dx"].mean()),
                "mean_dy_retained": float(retained["dy"].mean()),
                "mean_displacement_magnitude_retained": float(
                    np.hypot(retained["dx"], retained["dy"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["context", "arm"])


def empirical_survival(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return an uncensored empirical residence curve."""
    times = np.sort(values.astype(float).to_numpy())
    x = np.r_[0.0, np.repeat(times, 2)]
    before = 1.0 - np.arange(len(times)) / len(times)
    after = 1.0 - (np.arange(len(times)) + 1) / len(times)
    y = np.r_[1.0, np.column_stack([before, after]).ravel()]
    return x, y


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linewidth: float = 2.2,
    alpha: float = 0.9,
    zorder: int = 8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def map_limits(
    snapshots: pd.DataFrame,
    endpoints: pd.DataFrame,
    cutoffs: pd.Series,
) -> tuple[float, float, float, float]:
    """Zoom to the supported first-year Phase-1 region."""
    lookup = dict(zip(endpoints["trajectory_id"], endpoints["endpoint_mo"], strict=True))
    pieces: list[pd.DataFrame] = []
    for trajectory_id, cutoff in zip(
        snapshots["trajectory_id"].drop_duplicates(),
        cutoffs,
        strict=True,
    ):
        endpoint = float(lookup.get(trajectory_id, 12.0))
        end = float(cutoff) + min(endpoint, 12.0) * MONTH_DAYS
        course = snapshots[snapshots["trajectory_id"].eq(trajectory_id)]
        pieces.append(
            course[
                course["days_from_diagnosis"].between(float(cutoff), end)
            ][["map_x", "map_y"]]
        )
    points = pd.concat(pieces, ignore_index=True).dropna()
    xmin, xmax = points["map_x"].quantile([0.005, 0.995])
    ymin, ymax = points["map_y"].quantile([0.005, 0.995])
    xpad = 0.12 * max(float(xmax - xmin), 1.0)
    ypad = 0.12 * max(float(ymax - ymin), 1.0)
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
    show_y: bool,
) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(color=GRID, lw=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("combined clinical state PC1 (provisional display)", fontsize=8.2)
    if show_y:
        ax.set_ylabel("combined clinical state PC2", fontsize=8.2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=7)


def background_points(
    ax: plt.Axes,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> None:
    """Draw only atlas points inside the displayed support window."""
    inside = atlas[
        atlas["map_x"].between(limits[0], limits[1])
        & atlas["map_y"].between(limits[2], limits[3])
        & ~atlas["vital_status"].astype(str).str.lower().eq("deceased")
    ]
    if len(inside) > 9000:
        inside = inside.sample(9000, random_state=SEED)
    ax.scatter(
        inside["map_x"],
        inside["map_y"],
        s=2,
        color="#cbd5e1",
        alpha=0.10,
        edgecolor="none",
        rasterized=True,
        zorder=1,
    )


def draw_exemplar_fans(
    ax: plt.Axes,
    *,
    panel_label: str,
    context: str,
    state: str,
    history: str,
    snapshots: pd.DataFrame,
    horizon: pd.DataFrame,
    endpoints: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
    path_horizon_months: float,
) -> None:
    """Plot every stored branch for one exact shared anchor."""
    background_points(ax, atlas, limits)
    selected = horizon[
        horizon["context"].eq(context)
        & horizon["state"].eq(state)
        & horizon["history"].eq(history)
        & horizon["arm"].isin(ARMS)
    ]
    endpoint_lookup = endpoints.set_index("trajectory_id")
    shown_x: list[float] = []
    shown_y: list[float] = []
    remaining: list[str] = []
    for arm in ARMS:
        arm_rows = selected[selected["arm"].eq(arm)]
        remaining.append(
            f"{ARM_SHORT[arm]} {int((arm_rows['endpoint_mo'] > path_horizon_months).sum())}/"
            f"{len(arm_rows)}"
        )
        for index, item in enumerate(arm_rows.itertuples(index=False)):
            course = snapshots[snapshots["trajectory_id"].eq(item.trajectory_id)]
            endpoint = endpoint_lookup.loc[item.trajectory_id]
            end_elapsed = min(float(endpoint["endpoint_mo"]), path_horizon_months)
            end_day = float(item.cutoff_days) + end_elapsed * MONTH_DAYS
            path = path_until(course, float(item.cutoff_days), end_day)
            shown_x.extend(path["map_x"].astype(float).tolist())
            shown_y.extend(path["map_y"].astype(float).tolist())
            ax.plot(
                path["map_x"],
                path["map_y"],
                color=ARM_COLOR[arm],
                lw=1.45,
                alpha=0.52,
                zorder=4,
            )
            for landmark in (3.0, 6.0, 12.0):
                if landmark > end_elapsed + 1e-8:
                    continue
                point = position_at_day(
                    course,
                    float(item.cutoff_days) + landmark * MONTH_DAYS,
                )
                ax.scatter(
                    *point,
                    s=13,
                    color=ARM_COLOR[arm],
                    alpha=0.65,
                    edgecolor="white",
                    lw=0.25,
                    zorder=5,
                )
            if float(endpoint["endpoint_mo"]) <= path_horizon_months:
                ax.scatter(
                    path["map_x"].iloc[-1],
                    path["map_y"].iloc[-1],
                    marker="X",
                    s=48,
                    color=EXIT,
                    edgecolor="white",
                    lw=0.55,
                    alpha=0.88,
                    zorder=7,
                )
            elif index == 0:
                ax.scatter(
                    path["map_x"].iloc[-1],
                    path["map_y"].iloc[-1],
                    s=28,
                    facecolor=ARM_COLOR[arm],
                    edgecolor="white",
                    lw=0.5,
                    zorder=6,
                )
    start = selected[["x0", "y0"]].mean()
    ax.scatter(
        start["x0"],
        start["y0"],
        s=135,
        facecolor="white",
        edgecolor=INK,
        lw=2.2,
        zorder=10,
    )
    ax.text(
        start["x0"] + 0.06,
        start["y0"] + 0.05,
        "shared\ndecision anchor",
        fontsize=7.1,
        color=INK,
        weight="bold",
        zorder=11,
    )
    style_map(ax, limits, show_y=True)
    xspan = max(max(shown_x) - min(shown_x), 0.45)
    yspan = max(max(shown_y) - min(shown_y), 0.45)
    ax.set_xlim(min(shown_x) - 0.20 * xspan, max(shown_x) + 0.20 * xspan)
    ax.set_ylim(min(shown_y) - 0.25 * yspan, max(shown_y) + 0.25 * yspan)
    ax.text(
        0.98,
        0.96,
        f"remain at {path_horizon_months:g} mo\n" + " · ".join(remaining),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "alpha": 0.94,
        },
        zorder=20,
    )
    ax.set_title(
        (
            f"{panel_label}  {CONTEXT_LABEL[context]} · one exact anchor · "
            f"{state}/{history} · local zoom\n"
            f"all 3 model futures per assigned strategy, truncated at exit/{path_horizon_months:g} mo"
        ),
        loc="left",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )


def draw_local_field(
    ax: plt.Axes,
    *,
    panel_label: str,
    context: str,
    history: str,
    horizon: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> None:
    """Draw raw horizon endpoints plus mean retained-future arrows."""
    background_points(ax, atlas, limits)
    selected = horizon[
        horizon["context"].eq(context)
        & horizon["history"].eq(history)
        & horizon["arm"].isin(("chemo", "osimertinib"))
    ]
    offsets = {"chemo": -0.025, "osimertinib": 0.025}
    for state in sorted(selected["state"].unique()):
        cell = selected[selected["state"].eq(state)]
        anchor = cell[["x0", "y0"]].mean()
        ax.scatter(
            anchor["x0"],
            anchor["y0"],
            s=75,
            facecolor="white",
            edgecolor=INK,
            lw=1.35,
            zorder=9,
        )
        ax.text(
            anchor["x0"] + 0.045,
            anchor["y0"] + 0.045,
            state,
            fontsize=7.2,
            weight="bold",
            color=INK,
            zorder=11,
        )
        for arm in ("chemo", "osimertinib"):
            arm_cell = cell[cell["arm"].eq(arm)]
            retained = arm_cell[~arm_cell["exited_by_horizon"]]
            exited = arm_cell[arm_cell["exited_by_horizon"]]
            for item in retained.itertuples(index=False):
                ax.plot(
                    [item.x0, item.xh],
                    [item.y0, item.yh],
                    color=ARM_COLOR[arm],
                    lw=0.75,
                    alpha=0.26,
                    zorder=3,
                )
                ax.scatter(
                    item.xh,
                    item.yh,
                    s=20,
                    facecolor=ARM_COLOR[arm],
                    edgecolor="white",
                    lw=0.4,
                    alpha=0.72,
                    zorder=5,
                )
            for item in exited.itertuples(index=False):
                ax.plot(
                    [item.x0, item.xh],
                    [item.y0, item.yh],
                    color=EXIT,
                    lw=0.75,
                    alpha=0.22,
                    zorder=3,
                )
                ax.scatter(
                    item.xh,
                    item.yh,
                    marker="X",
                    s=33,
                    color=EXIT,
                    edgecolor="white",
                    lw=0.45,
                    alpha=0.8,
                    zorder=6,
                )
            if not retained.empty:
                mean_end = retained[["xh", "yh"]].mean()
                add_arrow(
                    ax,
                    (
                        float(anchor["x0"]),
                        float(anchor["y0"]) + offsets[arm],
                    ),
                    (
                        float(mean_end["xh"]),
                        float(mean_end["yh"]) + offsets[arm],
                    ),
                    ARM_COLOR[arm],
                    linewidth=2.0,
                    alpha=0.86,
                )
    style_map(ax, limits, show_y=False)
    tau = float(selected["horizon_months"].iloc[0])
    exit_counts = (
        selected.groupby("arm")["exited_by_horizon"].agg(["sum", "count"])
    )
    ax.text(
        0.98,
        0.96,
        (
            f"exit by {tau:g} mo across S1–S5\n"
            f"osimertinib {int(exit_counts.loc['osimertinib', 'sum'])}/"
            f"{int(exit_counts.loc['osimertinib', 'count'])} · "
            f"chemo {int(exit_counts.loc['chemo', 'sum'])}/"
            f"{int(exit_counts.loc['chemo', 'count'])}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "alpha": 0.94,
        },
        zorder=20,
    )
    ax.set_title(
        (
            f"{panel_label}  Across five starting states · {history}, τ={tau:g} mo\n"
            "thin paths = every future · thick arrows = retained-future mean"
        ),
        loc="left",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )


def draw_residence(
    ax: plt.Axes,
    *,
    panel_label: str,
    context: str,
    endpoints: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Draw the complete arm-level residence/exit distribution."""
    selected = endpoints[
        endpoints["context"].eq(context)
        & endpoints["arm_key"].isin(ARMS)
    ]
    for arm in ARMS:
        values = selected[selected["arm_key"].eq(arm)]["endpoint_mo"]
        x, y = empirical_survival(values)
        ax.plot(
            x,
            y,
            color=ARM_COLOR[arm],
            lw=2.5,
            label=ARM_SHORT[arm],
        )
        row = summary[
            summary["context"].eq(context) & summary["arm"].eq(arm)
        ].iloc[0]
        ax.scatter(
            row["median_endpoint_mo"],
            0.5,
            s=25,
            color=ARM_COLOR[arm],
            edgecolor="white",
            lw=0.45,
            zorder=5,
        )
    ax.axvline(6, color="#94a3b8", ls="--", lw=0.9)
    ax.axvline(12, color="#94a3b8", ls="--", lw=0.9)
    ax.text(6, 1.015, "6 mo", ha="center", fontsize=6.5, color=MUTED)
    ax.text(12, 1.015, "12 mo", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("months from treatment decision", fontsize=8.2)
    ax.set_ylabel("fraction without first-line exit/progression", fontsize=8.2)
    ax.grid(color=GRID, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_title(
        (
            f"{panel_label}  Residence is the treatment signal\n"
            "all 5 states × 3 histories × 3 futures = 45/arm"
        ),
        loc="left",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )
    annotation = []
    for arm in ARMS:
        row = summary[
            summary["context"].eq(context) & summary["arm"].eq(arm)
        ].iloc[0]
        annotation.append(
            (
                f"{ARM_SHORT[arm]}: median {row['median_endpoint_mo']:.1f} mo · "
                f"remain 6/12 mo {100 * row['remain_at_6mo']:.0f}%/"
                f"{100 * row['remain_at_12mo']:.0f}%"
            )
        )
    ax.text(
        0.98,
        0.96,
        "\n".join(annotation),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "alpha": 0.94,
        },
    )


def build_figure(
    *,
    snapshots: pd.DataFrame,
    endpoints: pd.DataFrame,
    horizon: pd.DataFrame,
    summary: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
    history: str,
    exemplar_state: str,
    path_horizon_months: float,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Render the two-context anchored-fan result."""
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.2, 10.2),
        gridspec_kw={"width_ratios": [1.12, 1.12, 0.92]},
    )
    fig.patch.set_facecolor(PAPER)
    for row, context in enumerate(CONTEXTS):
        panel_labels = ("A", "B", "C") if row == 0 else ("D", "E", "F")
        draw_exemplar_fans(
            axes[row, 0],
            panel_label=panel_labels[0],
            context=context,
            state=exemplar_state,
            history=history,
            snapshots=snapshots,
            horizon=horizon,
            endpoints=endpoints,
            atlas=atlas,
            limits=limits,
            path_horizon_months=path_horizon_months,
        )
        draw_local_field(
            axes[row, 1],
            panel_label=panel_labels[1],
            context=context,
            history=history,
            horizon=horizon,
            atlas=atlas,
            limits=limits,
        )
        draw_residence(
            axes[row, 2],
            panel_label=panel_labels[2],
            context=context,
            endpoints=endpoints,
            summary=summary,
        )

    handles = [
        Line2D([0], [0], color=ARM_COLOR[arm], lw=2.5, label=ARM_LABEL[arm])
        for arm in ARMS
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=EXIT,
            markeredgecolor="white",
            markersize=8,
            label="first-line exit/progression",
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.025, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8.2,
    )
    fig.suptitle(
        "Same starting patient, different treatment futures",
        x=0.025,
        y=0.986,
        ha="left",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.025,
        0.947,
        (
            "Only assigned treatment changes. Lines are raw model futures; "
            "curves show how long disease control persists."
        ),
        fontsize=9.3,
        color=MUTED,
    )
    fig.text(
        0.025,
        0.018,
        (
            f"Locked Claude Sonnet 5 Phase-1 model assay · n=3 futures/cell · "
            f"{MAP_ID} provisional combined-state display. Active arms use forced-line "
            "PFS; supportive care uses post-anchor time to first generated progression. "
            "Generated conditional futures are not measured human treatment effects or "
            "validated individual predictions."
        ),
        fontsize=7.6,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.015, 0.055, 0.995, 0.87), h_pad=2.5, w_pad=1.4)
    png = out_dir / "fig_egfr_anchored_future_fans.png"
    pdf = out_dir / "fig_egfr_anchored_future_fans.pdf"
    fig.savefig(png, dpi=210, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(pdf, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return png, pdf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_readout(
    *,
    out_dir: Path,
    run_root: Path,
    endpoint_path: Path,
    model: DisplayMap,
    horizon: pd.DataFrame,
    anchor_audit: pd.DataFrame,
    summary: pd.DataFrame,
    figure_paths: tuple[Path, Path],
    history: str,
) -> None:
    """Persist the numerical result and its provenance."""
    loadings = pd.DataFrame(
        {
            "feature": MAP_FEATURES,
            "label": [FEATURE_LABELS[feature] for feature in MAP_FEATURES],
            "pc1_loading": model.pca.components_[0],
            "pc2_loading": model.pca.components_[1],
        }
    )
    loadings.to_csv(out_dir / "display_map_loadings.csv", index=False)
    horizon.to_csv(out_dir / "trajectory_horizon_points.csv", index=False)
    anchor_audit.to_csv(out_dir / "anchor_invariance_audit.csv", index=False)
    summary.to_csv(out_dir / "arm_residence_summary.csv", index=False)
    movement = movement_summary(horizon)
    movement.to_csv(out_dir / "arm_movement_summary.csv", index=False)

    key_results: dict[str, dict[str, float]] = {}
    for context in CONTEXTS:
        rows = summary[summary["context"].eq(context)].set_index("arm")
        key_results[context] = {
            "osimertinib_median_months": float(
                rows.loc["osimertinib", "median_endpoint_mo"]
            ),
            "chemo_median_months": float(rows.loc["chemo", "median_endpoint_mo"]),
            "bsc_median_months": float(
                rows.loc["supportive", "median_endpoint_mo"]
            ),
            "osimertinib_exit_by_6": float(
                1 - rows.loc["osimertinib", "remain_at_6mo"]
            ),
            "chemo_exit_by_6": float(1 - rows.loc["chemo", "remain_at_6mo"]),
            "osimertinib_exit_by_12": float(
                1 - rows.loc["osimertinib", "remain_at_12mo"]
            ),
            "chemo_exit_by_12": float(1 - rows.loc["chemo", "remain_at_12mo"]),
        }

    report = {
        "status": "model_assay_not_human_causal_effect",
        "run_root": str(run_root.relative_to(ROOT)),
        "git_commit": git_commit(),
        "map_id": MAP_ID,
        "map_features": list(MAP_FEATURES),
        "map_explained_variance": {
            "pc1": model.explained[0],
            "pc2": model.explained[1],
        },
        "fixed_history_in_map_panels": history,
        "field_horizon_months": float(horizon["horizon_months"].iloc[0]),
        "n_phase1_trajectories": int(horizon["trajectory_id"].nunique()),
        "n_anchors": int(
            horizon[["context", "state", "history"]].drop_duplicates().shape[0]
        ),
        "max_anchor_x_range": float(anchor_audit["x_range"].max()),
        "max_anchor_y_range": float(anchor_audit["y_range"].max()),
        "endpoint_fallback_count": int(horizon["used_death_fallback"].sum()),
        "input_sha256": {
            "snapshot_rows": sha256(
                run_root / "derivation" / "snapshot_rows.csv"
            ),
            "endpoint_long": sha256(endpoint_path),
        },
        "key_results": key_results,
        "movement_summary": movement.to_dict(orient="records"),
        "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
    }
    (out_dir / "readout.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# EGFR anchored future fans",
        "",
        "## Empirical result",
        "",
        "| context | arm | n | median endpoint | remain at 6 mo | remain at 12 mo |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {CONTEXT_LABEL[row.context]} | {ARM_SHORT[row.arm]} | {row.n} | "
            f"{row.median_endpoint_mo:.2f} mo | {100 * row.remain_at_6mo:.1f}% | "
            f"{100 * row.remain_at_12mo:.1f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- In EGFR-mutant disease, both active arms move retained futures through an "
        "overlapping response region at six months, but residence separates by "
        "12 months: all osimertinib futures remain on first line versus approximately "
        "half of chemotherapy futures.",
        "- In EGFR-wild-type disease, the wrong-target osimertinib arm fails early: "
        "most futures exit by six months while chemotherapy futures remain.",
        "- The long-tail EGFR-wild-type osimertinib futures are retained in the raw "
        "fan and residence curve. They are a model-instability warning, not evidence "
        "of late efficacy.",
        "",
        "## Contract",
        "",
        f"- Display map: `{MAP_ID}`; combined current-state PCA only.",
        f"- PC1 variance {model.explained[0]:.1%}; PC2 {model.explained[1]:.1%}.",
        f"- Map panels fix `{history}`; residence panels average equally over all "
        "five states, three histories and three futures per cell.",
        "- Active-arm endpoint is forced first-line PFS. Supportive-arm endpoint is "
        "post-anchor time to first generated progression.",
        "- Raw futures are model-generated conditional branches, not observed human "
        "trajectories or validated individual predictions.",
        "- Per-cell n=3 supports a qualitative assay result, not calibrated "
        "state-specific heterogeneity.",
        f"- Maximum anchor coordinate mismatch across arms/futures: "
        f"PC1 {anchor_audit['x_range'].max():.3g}, "
        f"PC2 {anchor_audit['y_range'].max():.3g}.",
    ]
    (out_dir / "readout.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--endpoint-long", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--history", choices=("H0", "H1", "H2"), default="H1")
    parser.add_argument("--exemplar-state", choices=tuple(STATE_LABEL), default="S2")
    parser.add_argument("--field-horizon-months", type=float, default=6.0)
    parser.add_argument("--path-horizon-months", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = _require(args.run_root)
    endpoint_path = _require(args.endpoint_long)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atlas_raw = load_atlas_snapshots()
    display_map, atlas = fit_display_map(atlas_raw)
    snapshots, _events = load_phase1(run_root, display_map)
    endpoints = pd.read_csv(endpoint_path)
    endpoints = endpoints[
        endpoints["context"].isin(CONTEXTS)
        & endpoints["arm_key"].isin(ARMS)
    ].copy()
    horizon, anchor_audit = build_horizon_table(
        snapshots,
        endpoints,
        run_root,
        args.field_horizon_months,
    )
    summary = residence_summary(endpoints)

    trajectory_meta = snapshots[
        ["trajectory_id"]
    ].drop_duplicates()
    trajectory_meta["cutoff_days"] = cutoff_days_from_specs(
        run_root,
        trajectory_meta["trajectory_id"],
    )
    limits = map_limits(
        snapshots,
        endpoints,
        trajectory_meta["cutoff_days"],
    )
    figure_paths = build_figure(
        snapshots=snapshots,
        endpoints=endpoints,
        horizon=horizon,
        summary=summary,
        atlas=atlas,
        limits=limits,
        history=args.history,
        exemplar_state=args.exemplar_state,
        path_horizon_months=args.path_horizon_months,
        out_dir=args.out_dir,
    )
    write_readout(
        out_dir=args.out_dir,
        run_root=run_root,
        endpoint_path=endpoint_path,
        model=display_map,
        horizon=horizon,
        anchor_audit=anchor_audit,
        summary=summary,
        figure_paths=figure_paths,
        history=args.history,
    )

    print(f"[egfr anchored fans] figure: {figure_paths[0]}")
    print(summary.to_string(index=False))
    print(
        "[egfr anchored fans] max anchor mismatch: "
        f"PC1={anchor_audit['x_range'].max():.3g}, "
        f"PC2={anchor_audit['y_range'].max():.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
