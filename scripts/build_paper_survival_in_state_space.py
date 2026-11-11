#!/usr/bin/env python3
"""Show a virtual-trial survival result as conserved state-space occupancy.

The same locked EGFR-mutant Phase-1 futures are represented in two coordinate
systems:

* a conventional first-line progression-free survival curve;
* the complete pre-progression paths and their elapsed-time occupation measure
  on the frozen twenty-feature PCA display map.

For horizon ``tau`` and first-line progression time ``T``:

    RMST_a(tau) = E_a[min(T, tau)]
                = sum_B E_a[integral_0^min(T,tau) 1{X(t) in B} dt]

where ``B`` partitions the display map.  The identity is audited numerically;
the PCA only allocates conserved time spatially and does not define the
endpoint.  Event-time paths are piecewise constant (LOCF), and progression is
an explicit absorbing EXIT rather than a destination in PCA space.

The assigned arms intervene on the model-implied distribution Q_theta.  They
are not observed human counterfactual outcomes or calibrated trial estimates.

Run:
  uv run python scripts/build_paper_survival_in_state_space.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_precision_ladder_paired_views as paired  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "paper_survival_in_state_space"
STATE_DIR = ROOT / "outputs" / "paper_state_map"
CONTEXT = "EGFR_1L"
ARMS = ("chemo", "osimertinib")
ARM_LABEL = {
    "chemo": "carboplatin–pemetrexed",
    "osimertinib": "osimertinib",
}
ARM_SHORT = {"chemo": "Chemotherapy", "osimertinib": "Osimertinib"}
ARM_COLOR = {"chemo": "#d07a26", "osimertinib": "#07877d"}
HORIZON_MONTHS = 24.0
MONTH_DAYS = paired.MONTH_DAYS
GRID_BINS = 28
SMOOTHING_SIGMA_CELLS = 0.85
BOOTSTRAP_DRAWS = 5_000
SEED = 20260731

INK = "#172033"
MUTED = "#667085"
GRID = "#d9dee7"
PAPER = "#fbfaf7"
EXIT = "#a51c30"
ANCHOR = "#111827"
OCCUPANCY_CMAP = LinearSegmentedColormap.from_list(
    "clinical_time",
    ("#fffdf8", "#dcecf1", "#86bac8", "#356f91", "#163d5c"),
)
DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "time_difference",
    ("#c46a22", "#f3c999", "#fffdf8", "#9ed8d1", "#087f73"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pair_id(row: pd.Series) -> str:
    return "|".join(
        (
            str(row["context"]),
            str(row["state"]),
            str(row["history"]),
            f"f{int(row['future_index']):02d}",
        )
    )


def load_trial() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load frozen-map states and select the paired EGFR-mutant trial."""
    states, endpoints, _labels = paired.load_phase1()
    states = states[
        states["context"].eq(CONTEXT) & states["arm"].isin(ARMS)
    ].copy()
    endpoints = endpoints[endpoints["_gid"].isin(states["_gid"].unique())].copy()
    atlas = pd.read_csv(
        STATE_DIR / "state_projection.csv.gz",
        usecols=["pc1", "pc2"],
    )
    if states.empty or endpoints.empty:
        raise ValueError("The declared EGFR-mutant trial is empty")
    return states, endpoints, atlas


def _locf_state(course: pd.DataFrame, target_day: float) -> pd.Series:
    ordered = course.sort_values(["days_from_diagnosis", "snapshot_index"])
    available = ordered[ordered["days_from_diagnosis"].le(target_day + 1e-9)]
    if available.empty:
        raise ValueError("No state is observed at or before the decision anchor")
    return available.iloc[-1]


def build_pre_exit_intervals(
    states: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    horizon_months: float = HORIZON_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact LOCF state intervals, truncated at progression or horizon."""
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive")
    endpoint_map = endpoints.set_index("_gid")["pfs_line_1_days"].to_dict()
    interval_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for gid, raw_course in states.groupby("_gid", sort=False):
        course = (
            raw_course.sort_values(["days_from_diagnosis", "snapshot_index"])
            .drop_duplicates("days_from_diagnosis", keep="last")
            .reset_index(drop=True)
        )
        first = course.iloc[0]
        pfs_days = pd.to_numeric(
            pd.Series([endpoint_map.get(gid)]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(pfs_days) or float(pfs_days) <= 0:
            raise ValueError(f"Missing or invalid first-line PFS for {gid}")
        pfs_months = float(pfs_days) / MONTH_DAYS
        restricted_months = min(float(horizon_months), pfs_months)
        anchor_day = float(first["cutoff_days"])
        stop_day = anchor_day + restricted_months * MONTH_DAYS
        anchor = _locf_state(course, anchor_day)
        interior = course[
            course["days_from_diagnosis"].gt(anchor_day + 1e-9)
            & course["days_from_diagnosis"].lt(stop_day - 1e-9)
        ]
        path_states = pd.concat([anchor.to_frame().T, interior], ignore_index=True)
        starts = np.r_[anchor_day, interior["days_from_diagnosis"].to_numpy(float)]
        ends = np.r_[starts[1:], stop_day]
        dwell = (ends - starts) / MONTH_DAYS
        if np.any(dwell <= 0):
            raise ValueError(f"Non-positive state interval for {gid}")

        pair_id = _pair_id(first)
        trajectory_rows.append(
            {
                "_gid": gid,
                "trajectory_id": str(first["trajectory_id"]),
                "pair_id": pair_id,
                "arm": str(first["arm"]),
                "context": str(first["context"]),
                "state": str(first["state"]),
                "history": str(first["history"]),
                "future_index": int(first["future_index"]),
                "anchor_day": anchor_day,
                "anchor_pc1": float(anchor["pc1"]),
                "anchor_pc2": float(anchor["pc2"]),
                "pfs_months": pfs_months,
                "restricted_months": restricted_months,
                "exited_by_horizon": bool(pfs_months <= horizon_months),
            }
        )
        for index, (state_row, start_day, end_day, duration) in enumerate(
            zip(
                path_states.to_dict(orient="records"),
                starts,
                ends,
                dwell,
                strict=True,
            )
        ):
            interval_rows.append(
                {
                    "_gid": gid,
                    "trajectory_id": str(first["trajectory_id"]),
                    "pair_id": pair_id,
                    "arm": str(first["arm"]),
                    "state": str(first["state"]),
                    "history": str(first["history"]),
                    "future_index": int(first["future_index"]),
                    "interval_index": index,
                    "start_month": (float(start_day) - anchor_day) / MONTH_DAYS,
                    "end_month": (float(end_day) - anchor_day) / MONTH_DAYS,
                    "dwell_months": float(duration),
                    "pc1": float(state_row["pc1"]),
                    "pc2": float(state_row["pc2"]),
                    "event_type": str(state_row.get("anchor_event_type", "state")),
                    "is_exit_source": bool(
                        index == len(path_states) - 1 and pfs_months <= horizon_months
                    ),
                }
            )

    intervals = pd.DataFrame(interval_rows).sort_values(
        ["arm", "pair_id", "start_month"]
    )
    trajectories = pd.DataFrame(trajectory_rows).sort_values(["arm", "pair_id"])
    validate_paired_anchors(trajectories)
    return intervals.reset_index(drop=True), trajectories.reset_index(drop=True)


def validate_paired_anchors(trajectories: pd.DataFrame) -> None:
    """Require identical anchors and exactly one future per arm per pair."""
    counts = trajectories.groupby(["pair_id", "arm"]).size().unstack(fill_value=0)
    if set(counts.columns) != set(ARMS) or not (counts.loc[:, list(ARMS)] == 1).all().all():
        raise ValueError("The trial does not contain one future per arm and anchor")
    wide = trajectories.pivot(
        index="pair_id", columns="arm", values=["anchor_pc1", "anchor_pc2"]
    )
    for coordinate in ("anchor_pc1", "anchor_pc2"):
        difference = wide[(coordinate, ARMS[0])] - wide[(coordinate, ARMS[1])]
        if float(difference.abs().max()) > 1e-10:
            raise ValueError("Assigned arms do not begin from identical anchors")


def time_conservation_audit(
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """Verify that pre-exit occupancy equals restricted PFS trajectory by trajectory."""
    integrated = (
        intervals.groupby("_gid", as_index=False)["dwell_months"]
        .sum()
        .rename(columns={"dwell_months": "integrated_occupancy_months"})
    )
    audit = trajectories.merge(integrated, on="_gid", how="left", validate="one_to_one")
    audit["absolute_error"] = (
        audit["restricted_months"] - audit["integrated_occupancy_months"]
    ).abs()
    if float(audit["absolute_error"].max()) > 1e-9:
        raise ValueError("State occupancy does not conserve restricted PFS")
    return audit


def paired_rmst_contrast(
    trajectories: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return paired model-future RMST difference and bootstrap interval."""
    wide = trajectories.pivot(index="pair_id", columns="arm", values="restricted_months")
    differences = (
        wide["osimertinib"].to_numpy(float) - wide["chemo"].to_numpy(float)
    )
    estimate = float(differences.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = differences[rng.integers(0, len(differences), len(differences))]
        draws[index] = float(sampled.mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return estimate, float(low), float(high)


def map_limits(
    atlas: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Return common robust map limits that still contain every trial state."""
    xmin = min(float(atlas["pc1"].quantile(0.005)), float(intervals["pc1"].min()))
    xmax = max(float(atlas["pc1"].quantile(0.995)), float(intervals["pc1"].max()))
    ymin = min(float(atlas["pc2"].quantile(0.005)), float(intervals["pc2"].min()))
    ymax = max(float(atlas["pc2"].quantile(0.995)), float(intervals["pc2"].max()))
    xpad = max(0.08, 0.035 * (xmax - xmin))
    ypad = max(0.08, 0.035 * (ymax - ymin))
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def occupancy_grids(
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    limits: tuple[float, float, float, float],
    bins: int = GRID_BINS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """Allocate conserved pre-exit patient-time to a common rectangular grid."""
    xmin, xmax, ymin, ymax = limits
    x_edges = np.linspace(xmin, xmax, bins + 1)
    y_edges = np.linspace(ymin, ymax, bins + 1)
    exact: dict[str, np.ndarray] = {}
    smooth: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_intervals = intervals[intervals["arm"].eq(arm)]
        n_assigned = int(trajectories[trajectories["arm"].eq(arm)]["_gid"].nunique())
        grid, _, _ = np.histogram2d(
            arm_intervals["pc1"].to_numpy(float),
            arm_intervals["pc2"].to_numpy(float),
            bins=(x_edges, y_edges),
            weights=arm_intervals["dwell_months"].to_numpy(float) / n_assigned,
        )
        exact[arm] = grid
        blurred = gaussian_filter(grid, sigma=SMOOTHING_SIGMA_CELLS, mode="constant")
        if float(blurred.sum()) > 0:
            blurred *= float(grid.sum()) / float(blurred.sum())
        smooth[arm] = blurred
        for ix in range(bins):
            for iy in range(bins):
                rows.append(
                    {
                        "arm": arm,
                        "pc1_low": x_edges[ix],
                        "pc1_high": x_edges[ix + 1],
                        "pc2_low": y_edges[iy],
                        "pc2_high": y_edges[iy + 1],
                        "occupancy_months_per_assigned_future": grid[ix, iy],
                        "display_smoothed_occupancy": blurred[ix, iy],
                    }
                )
    return exact, smooth, x_edges, y_edges, pd.DataFrame(rows)


def occupancy_until(
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    arm: str,
    horizon_months: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    """Return cumulative occupancy through one animation landmark."""
    arm_intervals = intervals[intervals["arm"].eq(arm)].copy()
    arm_intervals["visible_months"] = np.maximum(
        0.0,
        np.minimum(arm_intervals["end_month"], horizon_months)
        - arm_intervals["start_month"],
    )
    arm_intervals = arm_intervals[arm_intervals["visible_months"].gt(0)]
    n_assigned = int(trajectories[trajectories["arm"].eq(arm)]["_gid"].nunique())
    grid, _, _ = np.histogram2d(
        arm_intervals["pc1"].to_numpy(float),
        arm_intervals["pc2"].to_numpy(float),
        bins=(x_edges, y_edges),
        weights=arm_intervals["visible_months"].to_numpy(float) / n_assigned,
    )
    blurred = gaussian_filter(grid, sigma=SMOOTHING_SIGMA_CELLS, mode="constant")
    if float(blurred.sum()) > 0:
        blurred *= float(grid.sum()) / float(blurred.sum())
    return blurred


def active_positions(
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    arm: str,
    horizon_months: float,
) -> pd.DataFrame:
    """Return one LOCF position for every future not yet in EXIT."""
    active_ids = trajectories[
        trajectories["arm"].eq(arm)
        & trajectories["pfs_months"].gt(horizon_months + 1e-9)
    ]["_gid"]
    candidates = intervals[
        intervals["_gid"].isin(active_ids)
        & intervals["start_month"].le(horizon_months + 1e-9)
        & intervals["end_month"].ge(horizon_months - 1e-9)
    ]
    return candidates.sort_values(["_gid", "start_month"]).drop_duplicates("_gid", keep="last")


def grouped_active_positions(active: pd.DataFrame) -> pd.DataFrame:
    """Count coincident futures without displacing them from their true state."""
    if active.empty:
        return pd.DataFrame(columns=["pc1", "pc2", "n_futures"])
    grouped = (
        active.assign(
            pc1_key=active["pc1"].round(10),
            pc2_key=active["pc2"].round(10),
        )
        .groupby(["pc1_key", "pc2_key"], as_index=False)
        .agg(
            pc1=("pc1", "first"),
            pc2=("pc2", "first"),
            n_futures=("_gid", "nunique"),
        )
    )
    return grouped


def frame_transitions(
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    arm: str,
    start_month: float,
    step_months: float = 1.0,
) -> pd.DataFrame:
    """Return one individual next-frame arrow or EXIT transition per active future."""
    if step_months <= 0:
        raise ValueError("step_months must be positive")
    source = active_positions(
        intervals,
        trajectories,
        arm=arm,
        horizon_months=start_month,
    )[["_gid", "pc1", "pc2"]].rename(columns={"pc1": "x0", "pc2": "y0"})
    arm_meta = trajectories[trajectories["arm"].eq(arm)][
        ["_gid", "pfs_months"]
    ]
    transitions = source.merge(arm_meta, on="_gid", how="left", validate="one_to_one")
    end_month = start_month + step_months
    transitions["exit_next"] = transitions["pfs_months"].le(end_month + 1e-9)

    destination = active_positions(
        intervals,
        trajectories,
        arm=arm,
        horizon_months=end_month,
    )[["_gid", "pc1", "pc2"]].rename(columns={"pc1": "x1", "pc2": "y1"})
    transitions = transitions.merge(
        destination,
        on="_gid",
        how="left",
        validate="one_to_one",
    )
    transitions["dx"] = transitions["x1"] - transitions["x0"]
    transitions["dy"] = transitions["y1"] - transitions["y0"]
    transitions["no_recorded_change"] = (
        ~transitions["exit_next"]
        & transitions["dx"].abs().le(1e-10)
        & transitions["dy"].abs().le(1e-10)
    )
    if transitions.loc[~transitions["exit_next"], ["x1", "y1"]].isna().any().any():
        raise ValueError("An active non-exiting future is missing its next-frame state")
    return transitions


def km_coordinates(values: Sequence[float], horizon_months: float) -> tuple[np.ndarray, np.ndarray]:
    """Return right-continuous complete-data Kaplan–Meier coordinates."""
    data = np.asarray(values, dtype=float)
    event_times = np.unique(data[data <= horizon_months])
    x = np.r_[0.0, event_times, horizon_months]
    survival = np.r_[1.0, [float(np.mean(data > event)) for event in event_times], float(np.mean(data > horizon_months))]
    return x, survival


def _format_map_axis(
    axis: plt.Axes,
    *,
    limits: tuple[float, float, float, float],
    title: str,
) -> None:
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    axis.grid(color=GRID, linewidth=0.45, alpha=0.55)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7)


def _draw_atlas_background(axis: plt.Axes, atlas: pd.DataFrame) -> None:
    sample = atlas.sample(min(5_000, len(atlas)), random_state=SEED)
    axis.scatter(
        sample["pc1"],
        sample["pc2"],
        s=2.0,
        color="#c9ced6",
        alpha=0.08,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )


def draw_km_panel(
    axis: plt.Axes,
    trajectories: pd.DataFrame,
    *,
    contrast: tuple[float, float, float],
) -> None:
    """Draw the conventional survival view with RMST as area."""
    for arm in ARMS:
        values = trajectories[trajectories["arm"].eq(arm)]["pfs_months"].to_numpy(float)
        x, survival = km_coordinates(values, HORIZON_MONTHS)
        axis.fill_between(
            x,
            0,
            survival,
            step="post",
            color=ARM_COLOR[arm],
            alpha=0.13,
            linewidth=0,
        )
        axis.step(
            x,
            survival,
            where="post",
            color=ARM_COLOR[arm],
            linewidth=2.4,
            label=ARM_LABEL[arm],
        )
    axis.set_xlim(0, HORIZON_MONTHS)
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Months from first-line decision")
    axis.set_ylabel("Progression-free fraction")
    axis.grid(color=GRID, linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_title(
        "A  When: progression-free survival",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    rmst = trajectories.groupby("arm")["restricted_months"].mean()
    estimate, low, high = contrast
    axis.text(
        0.03,
        0.05,
        (
            f"24-mo RMST  {rmst['osimertinib']:.1f} vs {rmst['chemo']:.1f} mo\n"
            f"Δ = +{estimate:.1f} mo  [{low:+.1f}, {high:+.1f}]"
        ),
        transform=axis.transAxes,
        fontsize=8.1,
        color=INK,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID, "alpha": 0.95},
    )
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")


def draw_paths_panel(
    axis: plt.Axes,
    *,
    arm: str,
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    atlas: pd.DataFrame,
    limits: tuple[float, float, float, float],
    panel_letter: str,
) -> None:
    """Draw every stored pre-progression path from the shared anchors."""
    _draw_atlas_background(axis, atlas)
    arm_intervals = intervals[intervals["arm"].eq(arm)]
    for _gid, course in arm_intervals.groupby("_gid", sort=False):
        course = course.sort_values("start_month")
        axis.plot(
            course["pc1"],
            course["pc2"],
            color=ARM_COLOR[arm],
            linewidth=0.75,
            alpha=0.22,
            zorder=2,
        )
        exit_source = course[course["is_exit_source"]]
        if not exit_source.empty:
            last = exit_source.iloc[-1]
            axis.scatter(last["pc1"], last["pc2"], marker="x", s=13, color=EXIT, linewidths=0.7, alpha=0.55, zorder=3)
    arm_trajectories = trajectories[trajectories["arm"].eq(arm)]
    axis.scatter(
        arm_trajectories["anchor_pc1"],
        arm_trajectories["anchor_pc2"],
        s=16,
        facecolor="white",
        edgecolor=ANCHOR,
        linewidth=0.7,
        alpha=0.85,
        zorder=4,
    )
    n_exit = int(arm_trajectories["exited_by_horizon"].sum())
    _format_map_axis(
        axis,
        limits=limits,
        title=f"{panel_letter}  {ARM_SHORT[arm]} paths until progression",
    )
    axis.text(
        0.97,
        0.04,
        f"{n_exit}/{len(arm_trajectories)} EXIT by 24 mo",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=EXIT,
        fontweight="bold",
    )


def draw_occupancy_panel(
    axis: plt.Axes,
    *,
    arm: str,
    surface: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    trajectories: pd.DataFrame,
    limits: tuple[float, float, float, float],
    vmax: float,
    panel_letter: str,
) -> Any:
    """Draw one arm's cumulative pre-progression occupation measure."""
    image = axis.imshow(
        surface.T,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=OCCUPANCY_CMAP,
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )
    arm_trajectories = trajectories[trajectories["arm"].eq(arm)]
    axis.scatter(
        arm_trajectories["anchor_pc1"],
        arm_trajectories["anchor_pc2"],
        s=9,
        facecolor="none",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.75,
        zorder=3,
    )
    rmst = float(arm_trajectories["restricted_months"].mean())
    n_exit = int(arm_trajectories["exited_by_horizon"].sum())
    _format_map_axis(
        axis,
        limits=limits,
        title=f"{panel_letter}  {ARM_SHORT[arm]} occupancy",
    )
    axis.text(
        0.03,
        0.04,
        f"Σ cells = {rmst:.1f} months/future\nEXIT {n_exit}/{len(arm_trajectories)}",
        transform=axis.transAxes,
        fontsize=7.5,
        color="white",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": INK, "edgecolor": "none", "alpha": 0.86},
        zorder=5,
    )
    return image


def draw_difference_panel(
    axis: plt.Axes,
    *,
    difference: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    trajectories: pd.DataFrame,
    limits: tuple[float, float, float, float],
    contrast: tuple[float, float, float],
    exact_difference: np.ndarray | None = None,
) -> Any:
    """Draw the occupancy decomposition of the scalar RMST difference."""
    bound = max(float(np.max(np.abs(difference))), 1e-6)
    image = axis.imshow(
        difference.T,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=DIFFERENCE_CMAP,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )
    anchors = trajectories[trajectories["arm"].eq("chemo")]
    axis.scatter(
        anchors["anchor_pc1"],
        anchors["anchor_pc2"],
        s=9,
        facecolor="none",
        edgecolor=ANCHOR,
        linewidth=0.4,
        alpha=0.5,
        zorder=3,
    )
    _format_map_axis(
        axis,
        limits=limits,
        title="F  Difference occupancy\nteal: osimertinib · orange: chemotherapy",
    )
    estimate, _low, _high = contrast
    accounting = difference if exact_difference is None else exact_difference
    positive = float(accounting[accounting > 0].sum())
    negative = float(accounting[accounting < 0].sum())
    axis.text(
        0.03,
        0.04,
        f"NET BENEFIT WITH OSIMERTINIB\nΔRMST = {estimate:+.1f} months",
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        color=ARM_COLOR["osimertinib"],
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": GRID, "alpha": 0.94},
        zorder=5,
    )
    axis.text(
        0.97,
        0.96,
        f"teal cells  {positive:+.1f} mo\norange cells  {negative:+.1f} mo",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
        zorder=5,
    )
    return image


def build_main_figure(
    *,
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    atlas: pd.DataFrame,
    exact: dict[str, np.ndarray],
    smooth: dict[str, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    limits: tuple[float, float, float, float],
    contrast: tuple[float, float, float],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Compose the familiar and dynamical views in one paper figure."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#9aa4b2"})
    figure = plt.figure(figsize=(15.2, 10.1), facecolor=PAPER)
    grid = figure.add_gridspec(2, 6, left=0.055, right=0.975, bottom=0.14, top=0.86, wspace=0.82, hspace=0.58)
    km_axis = figure.add_subplot(grid[0, 0:2])
    chemo_paths = figure.add_subplot(grid[0, 2:4])
    osi_paths = figure.add_subplot(grid[0, 4:6])
    chemo_occupancy = figure.add_subplot(grid[1, 0:2])
    osi_occupancy = figure.add_subplot(grid[1, 2:4])
    difference_axis = figure.add_subplot(grid[1, 4:6])

    draw_km_panel(km_axis, trajectories, contrast=contrast)
    draw_paths_panel(
        chemo_paths,
        arm="chemo",
        intervals=intervals,
        trajectories=trajectories,
        atlas=atlas,
        limits=limits,
        panel_letter="B",
    )
    draw_paths_panel(
        osi_paths,
        arm="osimertinib",
        intervals=intervals,
        trajectories=trajectories,
        atlas=atlas,
        limits=limits,
        panel_letter="C",
    )
    common_vmax = max(float(smooth[arm].max()) for arm in ARMS)
    occupancy_image = draw_occupancy_panel(
        chemo_occupancy,
        arm="chemo",
        surface=smooth["chemo"],
        x_edges=x_edges,
        y_edges=y_edges,
        trajectories=trajectories,
        limits=limits,
        vmax=common_vmax,
        panel_letter="D",
    )
    draw_occupancy_panel(
        osi_occupancy,
        arm="osimertinib",
        surface=smooth["osimertinib"],
        x_edges=x_edges,
        y_edges=y_edges,
        trajectories=trajectories,
        limits=limits,
        vmax=common_vmax,
        panel_letter="E",
    )
    difference_image = draw_difference_panel(
        difference_axis,
        difference=smooth["osimertinib"] - smooth["chemo"],
        x_edges=x_edges,
        y_edges=y_edges,
        trajectories=trajectories,
        limits=limits,
        contrast=contrast,
        exact_difference=exact["osimertinib"] - exact["chemo"],
    )

    figure.suptitle(
        "A survival curve is accumulated patient-time before progression",
        x=0.055,
        y=0.96,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.055,
        0.91,
        (
            "The Kaplan–Meier curve says when assigned futures remain progression-free; "
            "the same complete paths say where that time is spent.  "
            "RMST₂₄ = ∫₀²⁴ S(t)dt = Σ state-cell occupancy."
        ),
        fontsize=10.4,
        color=MUTED,
    )
    path_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=ANCHOR, markersize=5, label="shared decision anchor"),
        Line2D([0], [0], marker="x", color=EXIT, linestyle="none", markersize=5, label="source state at progression → EXIT"),
    ]
    figure.legend(handles=path_legend, loc="upper center", bbox_to_anchor=(0.69, 0.895), frameon=False, ncol=2, fontsize=7.5)

    occupancy_bar_axis = figure.add_axes((0.09, 0.067, 0.42, 0.014))
    occupancy_bar = figure.colorbar(occupancy_image, cax=occupancy_bar_axis, orientation="horizontal")
    occupancy_bar.set_label("smoothed pre-progression occupancy (months per assigned future per cell)", fontsize=7.2)
    occupancy_bar.ax.tick_params(labelsize=6.5)
    difference_bar_axis = figure.add_axes((0.675, 0.067, 0.265, 0.014))
    difference_bar = figure.colorbar(difference_image, cax=difference_bar_axis, orientation="horizontal")
    difference_bar.set_label("osimertinib − chemotherapy occupancy", fontsize=7.2)
    difference_bar.ax.tick_params(labelsize=6.5)
    figure.text(
        0.055,
        0.008,
        (
            "Locked Sonnet model assay: 45 exactly paired generated futures per arm, identical anchors; "
            "LOCF paths stop at first-line progression or 24 months. PCA is display only. "
            "Bootstrap interval describes model-future variation, not human trial uncertainty."
        ),
        fontsize=7.1,
        color=MUTED,
    )

    png = out_dir / "fig_survival_in_state_space.png"
    pdf = out_dir / "fig_survival_in_state_space.pdf"
    figure.savefig(png, dpi=240, facecolor=figure.get_facecolor())
    figure.savefig(pdf, facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def build_decomposition_figure(
    *,
    exact: dict[str, np.ndarray],
    smooth: dict[str, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    trajectories: pd.DataFrame,
    limits: tuple[float, float, float, float],
    contrast: tuple[float, float, float],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Contrast one conventional forest estimate with its spatial decomposition."""
    figure = plt.figure(figsize=(13.5, 5.7), facecolor=PAPER)
    grid = figure.add_gridspec(1, 12, left=0.065, right=0.97, bottom=0.22, top=0.79, wspace=1.15)
    forest_axis = figure.add_subplot(grid[0, 0:3])
    map_axis = figure.add_subplot(grid[0, 3:8])
    contribution_axis = figure.add_subplot(grid[0, 8:12])
    estimate, low, high = contrast

    forest_axis.axvline(0, color="#8b95a4", linewidth=1)
    forest_axis.errorbar(
        estimate,
        0,
        xerr=np.asarray([[estimate - low], [high - estimate]]),
        fmt="o",
        color=ARM_COLOR["osimertinib"],
        markersize=7,
        capsize=4,
        linewidth=2,
    )
    forest_axis.set_xlim(min(-2.5, low - 1), max(10.5, high + 1))
    forest_axis.set_ylim(-0.65, 0.65)
    forest_axis.set_yticks([0], ["45 paired\nanchors"])
    forest_axis.set_xlabel("Δ 24-month RMST, months\nosimertinib − chemotherapy")
    forest_axis.grid(axis="x", color=GRID, linewidth=0.6)
    forest_axis.set_title("A  Forest: one number", loc="left", fontsize=11, fontweight="bold", color=INK)
    forest_axis.text(estimate, -0.28, f"{estimate:+.1f} [{low:+.1f}, {high:+.1f}]", ha="center", fontsize=8, color=INK)

    difference = smooth["osimertinib"] - smooth["chemo"]
    difference_image = draw_difference_panel(
        map_axis,
        difference=difference,
        x_edges=x_edges,
        y_edges=y_edges,
        trajectories=trajectories,
        limits=limits,
        contrast=contrast,
        exact_difference=exact["osimertinib"] - exact["chemo"],
    )
    map_axis.set_title(
        "B  State map: where benefit accrued",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    map_axis.set_xlabel("")

    exact_difference = (exact["osimertinib"] - exact["chemo"]).ravel()
    positive_total = float(exact_difference[exact_difference > 1e-12].sum())
    negative_total = float(exact_difference[exact_difference < -1e-12].sum())
    balance_values = np.asarray([positive_total, negative_total, exact_difference.sum()])
    balance_labels = [
        "Teal regions",
        "Orange regions",
        "Net benefit",
    ]
    balance_colors = [
        ARM_COLOR["osimertinib"],
        ARM_COLOR["chemo"],
        "#075f59",
    ]
    y_positions = np.arange(len(balance_values))[::-1]
    contribution_axis.barh(
        y_positions,
        balance_values,
        color=balance_colors,
        height=0.56,
        alpha=0.9,
    )
    contribution_axis.axvline(0, color="#8b95a4", linewidth=0.9)
    contribution_axis.set_yticks(y_positions, balance_labels)
    contribution_axis.tick_params(axis="y", labelsize=8.5)
    contribution_axis.set_xlabel("Contribution to ΔRMST (months)")
    contribution_axis.grid(axis="x", color=GRID, linewidth=0.55)
    contribution_axis.set_title(
        "C  The spatial balance is positive",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    limit = max(abs(negative_total), positive_total) * 1.22
    contribution_axis.set_xlim(-limit, limit)
    for y_position, value in zip(y_positions, balance_values, strict=True):
        contribution_axis.text(
            value + (0.35 if value >= 0 else -0.35),
            y_position,
            f"{value:+.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )

    figure.suptitle(
        "A state-space view decomposes—rather than replaces—the trial summary",
        x=0.065,
        y=0.96,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.065,
        0.875,
        "The forest gives the net benefit; the map shows where more time accumulated under each arm, and the balance recovers the same answer.",
        fontsize=9.8,
        color=MUTED,
    )
    color_axis = figure.add_axes((0.33, 0.095, 0.31, 0.018))
    colorbar = figure.colorbar(difference_image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        "more with chemotherapy  ←  occupancy difference  →  more with osimertinib",
        fontsize=7.5,
    )
    colorbar.ax.tick_params(labelsize=6.5)
    figure.text(
        0.065,
        0.012,
        "Exact cell sums use unsmoothed elapsed time; smoothing is display only. EXIT is first-line progression. Model assay, not human causal evidence.",
        fontsize=7.2,
        color=MUTED,
    )
    png = out_dir / "fig_forest_vs_occupancy_decomposition.png"
    pdf = out_dir / "fig_forest_vs_occupancy_decomposition.pdf"
    figure.savefig(png, dpi=240, facecolor=figure.get_facecolor())
    figure.savefig(pdf, facecolor=figure.get_facecolor())
    plt.close(figure)
    return png, pdf


def draw_animation_particles(
    axis: plt.Axes,
    *,
    arm: str,
    active: pd.DataFrame,
    transitions: pd.DataFrame | None,
) -> None:
    """Draw coincident-future counts and individual next-frame transitions."""
    if transitions is not None and not transitions.empty:
        moving = transitions[
            ~transitions["exit_next"] & ~transitions["no_recorded_change"]
        ]
        if not moving.empty:
            axis.quiver(
                moving["x0"],
                moving["y0"],
                moving["dx"],
                moving["dy"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=ARM_COLOR[arm],
                alpha=0.48,
                width=0.0042,
                headwidth=4.4,
                headlength=5.2,
                headaxislength=4.6,
                zorder=4,
            )
        stationary = transitions[transitions["no_recorded_change"]]
        for row in grouped_active_positions(
            stationary.rename(columns={"x0": "pc1", "y0": "pc2"})
        ).itertuples(index=False):
            axis.scatter(
                row.pc1,
                row.pc2,
                s=62 + 12 * row.n_futures,
                facecolor="none",
                edgecolor=INK,
                linewidth=0.75,
                alpha=0.7,
                zorder=5,
            )
        exiting = transitions[transitions["exit_next"]]
        if not exiting.empty:
            grouped_exit = (
                exiting.assign(x_key=exiting["x0"].round(10), y_key=exiting["y0"].round(10))
                .groupby(["x_key", "y_key"], as_index=False)
                .agg(x0=("x0", "first"), y0=("y0", "first"), n_exit=("_gid", "nunique"))
            )
            for row in grouped_exit.itertuples(index=False):
                axis.scatter(
                    row.x0,
                    row.y0,
                    marker="x",
                    s=45 + 8 * row.n_exit,
                    color=EXIT,
                    linewidth=1.25,
                    zorder=7,
                )
                if row.n_exit > 1:
                    axis.annotate(
                        f"→EXIT ×{row.n_exit}",
                        (row.x0, row.y0),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=6.2,
                        color=EXIT,
                        fontweight="bold",
                        zorder=8,
                    )

    for row in grouped_active_positions(active).itertuples(index=False):
        marker_size = 30 + 12 * row.n_futures
        axis.scatter(
            row.pc1,
            row.pc2,
            s=marker_size,
            color=ARM_COLOR[arm],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.9,
            zorder=6,
        )
        if row.n_futures > 1:
            axis.text(
                row.pc1,
                row.pc2,
                str(row.n_futures),
                ha="center",
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color="white",
                zorder=7,
            )


def build_animation(
    *,
    intervals: pd.DataFrame,
    trajectories: pd.DataFrame,
    atlas: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    limits: tuple[float, float, float, float],
    final_vmax: float,
    out_dir: Path,
) -> Path:
    """Animate survival loss, current particles and accumulated occupancy together."""
    output = out_dir / "animation_survival_in_state_space.gif"
    frame_times = np.arange(0.0, HORIZON_MONTHS + 0.01, 1.0)
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="survival_state_space_") as temp:
        temp_dir = Path(temp)
        for frame_index, current_time in enumerate(frame_times):
            figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), facecolor=PAPER)
            km_axis = axes[0]
            for arm in ARMS:
                values = trajectories[trajectories["arm"].eq(arm)]["pfs_months"].to_numpy(float)
                x, survival = km_coordinates(values, HORIZON_MONTHS)
                km_axis.step(x, survival, where="post", color=ARM_COLOR[arm], linewidth=1.0, alpha=0.22)
                mask = x <= current_time + 1e-9
                partial_x = np.r_[x[mask], current_time]
                current_survival = float(np.mean(values > current_time))
                partial_y = np.r_[survival[mask], current_survival]
                km_axis.step(partial_x, partial_y, where="post", color=ARM_COLOR[arm], linewidth=2.4, label=ARM_SHORT[arm])
                km_axis.scatter([current_time], [current_survival], s=28, color=ARM_COLOR[arm], zorder=4)
            km_axis.axvline(current_time, color="#8b95a4", linestyle="--", linewidth=0.9)
            km_axis.set_xlim(0, HORIZON_MONTHS)
            km_axis.set_ylim(0, 1.04)
            km_axis.set_xlabel("Months")
            km_axis.set_ylabel("Progression-free fraction")
            km_axis.grid(color=GRID, linewidth=0.55)
            km_axis.set_title("Survival: mass remaining on the map", loc="left", fontsize=10, fontweight="bold", color=INK)
            km_axis.legend(frameon=False, fontsize=7, loc="lower left")

            for axis, arm in zip(axes[1:], ARMS, strict=True):
                surface = occupancy_until(
                    intervals,
                    trajectories,
                    arm=arm,
                    horizon_months=current_time,
                    x_edges=x_edges,
                    y_edges=y_edges,
                )
                axis.imshow(
                    surface.T,
                    origin="lower",
                    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                    cmap=OCCUPANCY_CMAP,
                    vmin=0,
                    vmax=final_vmax,
                    interpolation="bilinear",
                    aspect="auto",
                )
                _draw_atlas_background(axis, atlas)
                active = active_positions(
                    intervals,
                    trajectories,
                    arm=arm,
                    horizon_months=current_time,
                )
                transitions = (
                    frame_transitions(
                        intervals,
                        trajectories,
                        arm=arm,
                        start_month=current_time,
                        step_months=1.0,
                    )
                    if current_time < HORIZON_MONTHS
                    else None
                )
                draw_animation_particles(
                    axis,
                    arm=arm,
                    active=active,
                    transitions=transitions,
                )
                assigned = int(trajectories[trajectories["arm"].eq(arm)]["_gid"].nunique())
                exited = assigned - len(active)
                _format_map_axis(
                    axis,
                    limits=limits,
                    title=f"{ARM_SHORT[arm]}: particles + accumulated time",
                )
                axis.text(
                    0.97,
                    0.04,
                    f"on map {len(active)}/{assigned}  ·  EXIT {exited}",
                    transform=axis.transAxes,
                    ha="right",
                    fontsize=7.4,
                    fontweight="bold",
                    color=EXIT if exited else INK,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
                )
            figure.suptitle(
                f"The same virtual trial unfolding through time — month {current_time:.0f}",
                x=0.045,
                y=0.98,
                ha="left",
                fontsize=15,
                fontweight="bold",
                color=INK,
            )
            figure.text(
                0.045,
                0.015,
                (
                    "Bubble number = coincident active futures · arrow = each future's state one month later · "
                    "outer ring = no recorded change · red × = progression before the next frame · shading = accumulated pre-progression time."
                ),
                fontsize=7.5,
                color=MUTED,
            )
            figure.subplots_adjust(
                left=0.055,
                right=0.985,
                bottom=0.18,
                top=0.78,
                wspace=0.26,
            )
            frame_path = temp_dir / f"frame_{frame_index:03d}.png"
            figure.savefig(frame_path, dpi=135, facecolor=figure.get_facecolor())
            plt.close(figure)
            frames.append(Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE))
        durations = [260] * len(frames)
        durations[-1] = 1_300
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            disposal=2,
        )
    return output


def landmark_summary(
    trajectories: pd.DataFrame,
    *,
    landmarks: Sequence[float] = (0.0, 3.0, 6.0, 12.0, 18.0, 24.0),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        values = trajectories[trajectories["arm"].eq(arm)]["pfs_months"].to_numpy(float)
        for landmark in landmarks:
            rows.append(
                {
                    "arm": arm,
                    "month": landmark,
                    "n_assigned": len(values),
                    "n_on_map": int(np.sum(values > landmark)),
                    "n_exit": int(np.sum(values <= landmark)),
                    "progression_free_fraction": float(np.mean(values > landmark)),
                    "rmst_through_landmark": float(np.minimum(values, landmark).mean()),
                }
            )
    return pd.DataFrame(rows)


def write_readout(
    *,
    trajectories: pd.DataFrame,
    audit: pd.DataFrame,
    exact: dict[str, np.ndarray],
    contrast: tuple[float, float, float],
    out_dir: Path,
) -> None:
    summary = trajectories.groupby("arm").agg(
        n_assigned=("_gid", "nunique"),
        median_pfs_months=("pfs_months", "median"),
        rmst24_months=("restricted_months", "mean"),
        exits_by_24=("exited_by_horizon", "sum"),
    )
    estimate, low, high = contrast
    exact_difference = exact["osimertinib"] - exact["chemo"]
    positive_spatial_contribution = float(
        exact_difference[exact_difference > 0].sum()
    )
    negative_spatial_contribution = float(
        exact_difference[exact_difference < 0].sum()
    )
    anchor_counts = (
        trajectories[trajectories["arm"].eq("chemo")]
        .groupby(["anchor_pc1", "anchor_pc2"])
        .size()
    )
    readout = {
        "context": CONTEXT,
        "arms": list(ARMS),
        "horizon_months": HORIZON_MONTHS,
        "endpoint": "first-line progression; complete generated PFS, no censoring",
        "n_pairs": int(trajectories["pair_id"].nunique()),
        "arm_summary": {
            arm: {
                "n_assigned": int(summary.loc[arm, "n_assigned"]),
                "median_pfs_months": float(summary.loc[arm, "median_pfs_months"]),
                "rmst24_months": float(summary.loc[arm, "rmst24_months"]),
                "exits_by_24": int(summary.loc[arm, "exits_by_24"]),
                "occupancy_grid_sum_months": float(exact[arm].sum()),
            }
            for arm in ARMS
        },
        "paired_delta_rmst24_months": {
            "estimate": estimate,
            "bootstrap_95_low": low,
            "bootstrap_95_high": high,
            "interpretation": "model-future variability, not human trial uncertainty",
        },
        "identity_audit": {
            "max_trajectory_error_months": float(audit["absolute_error"].max()),
            "sum_cell_difference_months": float(exact_difference.sum()),
            "delta_rmst_minus_cell_sum": float(estimate - exact_difference.sum()),
        },
        "initial_display_support": {
            "n_assigned_futures_per_arm": int(
                summary.loc["chemo", "n_assigned"]
            ),
            "n_unique_pca_anchor_positions": int(len(anchor_counts)),
            "multiplicity_per_anchor": sorted(int(value) for value in anchor_counts),
            "display_rule": (
                "coincident futures remain at their exact coordinate and are shown "
                "as count-labelled bubbles; no jitter is added"
            ),
        },
        "spatial_difference_balance_months": {
            "more_in_osimertinib_teal_cells": positive_spatial_contribution,
            "more_in_chemotherapy_orange_cells": negative_spatial_contribution,
            "net_delta_rmst24": float(exact_difference.sum()),
        },
        "estimand": (
            "RMST24 equals average complete pre-progression occupancy through 24 months; "
            "the grid partitions conserved time and smoothing is display only"
        ),
        "causal_status": (
            "paired assigned arms are interventions on the locked model-implied distribution "
            "Q_theta, not observed human counterfactuals"
        ),
    }
    (out_dir / "readout.json").write_text(json.dumps(readout, indent=2) + "\n")
    lines = [
        "# Survival in state space",
        "",
        "## Exact estimand bridge",
        "",
        "For each assigned future, LOCF state intervals stop at first-line progression or 24 months. Therefore:",
        "",
        "`RMST24 = mean[min(PFS, 24)] = sum of mean pre-progression months across all state-map cells`",
        "",
        "The Kaplan–Meier curve and occupancy map are not parallel metaphors. They are two projections of the same complete endpoint-path object.",
        "",
        "## Results",
        "",
        "| arm | n | median PFS | RMST24 | EXIT by 24 mo | grid sum |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        lines.append(
            f"| {ARM_LABEL[arm]} | {int(summary.loc[arm, 'n_assigned'])} | "
            f"{summary.loc[arm, 'median_pfs_months']:.2f} mo | "
            f"{summary.loc[arm, 'rmst24_months']:.2f} mo | "
            f"{int(summary.loc[arm, 'exits_by_24'])} | {exact[arm].sum():.2f} mo |"
        )
    lines.extend(
        [
            "",
            f"Paired ΔRMST24: **{estimate:+.2f} months** (paired model-future bootstrap interval {low:+.2f} to {high:+.2f}).",
            "",
            f"The exact occupancy-difference grid sums to **{exact_difference.sum():+.2f} months**; numerical mismatch from ΔRMST is {estimate - exact_difference.sum():.3g} months.",
            f"The signed spatial balance is **{positive_spatial_contribution:+.2f} months** in teal cells and **{negative_spatial_contribution:+.2f} months** in orange cells, yielding the same net {exact_difference.sum():+.2f}-month benefit.",
            "",
            f"At month 0, all {int(summary.loc['chemo', 'n_assigned'])} futures per arm are present but occupy only {len(anchor_counts)} distinct PCA anchors ({', '.join(str(int(value)) for value in sorted(anchor_counts))} futures per anchor). The animation therefore uses count-labelled bubbles rather than artificial jitter.",
            "",
            "## Clinical reading",
            "",
            "- The survival curve shows **when** assigned futures remain before progression.",
            "- The path panels show the complete pre-progression courses; a short-horizon arrow would show only their opening step.",
            "- The occupancy panels integrate how long the futures remain in every displayed state.",
            "- The difference map decomposes the scalar RMST gain into regions where time was gained or lost.",
            "- Each animation arrow points from the current LOCF state to the next one-month frame; a ring denotes no recorded change and a red cross denotes progression before that frame.",
            "- EXIT is progression. No post-progression state contributes to the PFS occupancy estimand.",
            "",
            "## Restraints",
            "",
            "The PCA is a reproducible display coordinate system, not the analytic patient state. Binning and smoothing do not create the estimand; the unsmoothed interval table conserves it exactly. The 45 paired branches per arm are locked Sonnet model outputs. They assay the model and do not estimate human individual treatment effects.",
        ]
    )
    (out_dir / "readout.md").write_text("\n".join(lines) + "\n")


def build(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states, endpoints, atlas = load_trial()
    intervals, trajectories = build_pre_exit_intervals(
        states,
        endpoints,
        horizon_months=args.horizon_months,
    )
    if not np.isclose(args.horizon_months, HORIZON_MONTHS):
        raise ValueError("The current paper composition is fixed at 24 months")
    audit = time_conservation_audit(intervals, trajectories)
    contrast = paired_rmst_contrast(trajectories)
    limits = map_limits(atlas, intervals)
    exact, smooth, x_edges, y_edges, grid_table = occupancy_grids(
        intervals,
        trajectories,
        limits=limits,
    )
    for arm in ARMS:
        rmst = float(trajectories[trajectories["arm"].eq(arm)]["restricted_months"].mean())
        if not np.isclose(float(exact[arm].sum()), rmst, atol=1e-9):
            raise ValueError(f"Occupancy grid does not reproduce RMST for {arm}")

    main_png, main_pdf = build_main_figure(
        intervals=intervals,
        trajectories=trajectories,
        atlas=atlas,
        exact=exact,
        smooth=smooth,
        x_edges=x_edges,
        y_edges=y_edges,
        limits=limits,
        contrast=contrast,
        out_dir=args.out_dir,
    )
    decomposition_png, decomposition_pdf = build_decomposition_figure(
        exact=exact,
        smooth=smooth,
        x_edges=x_edges,
        y_edges=y_edges,
        trajectories=trajectories,
        limits=limits,
        contrast=contrast,
        out_dir=args.out_dir,
    )
    animation = build_animation(
        intervals=intervals,
        trajectories=trajectories,
        atlas=atlas,
        x_edges=x_edges,
        y_edges=y_edges,
        limits=limits,
        final_vmax=max(float(surface.max()) for surface in smooth.values()),
        out_dir=args.out_dir,
    )

    intervals.to_csv(
        args.out_dir / "pre_progression_intervals.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    trajectories.to_csv(args.out_dir / "trajectory_summary.csv", index=False)
    audit.to_csv(args.out_dir / "time_conservation_audit.csv", index=False)
    grid_table.to_csv(
        args.out_dir / "occupancy_grid_24mo.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    landmark_summary(trajectories).to_csv(args.out_dir / "time_landmarks.csv", index=False)
    cell_difference = grid_table.pivot_table(
        index=["pc1_low", "pc1_high", "pc2_low", "pc2_high"],
        columns="arm",
        values="occupancy_months_per_assigned_future",
    ).reset_index()
    cell_difference["delta_occupancy_months"] = (
        cell_difference["osimertinib"] - cell_difference["chemo"]
    )
    cell_difference.to_csv(args.out_dir / "occupancy_difference_cells.csv", index=False)
    write_readout(
        trajectories=trajectories,
        audit=audit,
        exact=exact,
        contrast=contrast,
        out_dir=args.out_dir,
    )

    source_files = {
        "state_map_contract": STATE_DIR / "state_map_contract.json",
        "state_projection": STATE_DIR / "state_projection.csv.gz",
        "phase1_p0_records": STATE_DIR / "phase1_p0_records.csv",
    }
    manifest = {
        "git_commit": _git_commit(),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "context": CONTEXT,
        "arms": list(ARMS),
        "horizon_months": HORIZON_MONTHS,
        "endpoint": "pfs_line_1_days",
        "path_interpolation": "LOCF / piecewise constant",
        "n_pairs": int(trajectories["pair_id"].nunique()),
        "grid_bins": GRID_BINS,
        "display_smoothing_sigma_cells": SMOOTHING_SIGMA_CELLS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for name, path in source_files.items()},
        "outputs": [
            main_png.name,
            main_pdf.name,
            decomposition_png.name,
            decomposition_pdf.name,
            animation.name,
            "pre_progression_intervals.csv.gz",
            "trajectory_summary.csv",
            "time_conservation_audit.csv",
            "occupancy_grid_24mo.csv.gz",
            "occupancy_difference_cells.csv",
            "time_landmarks.csv",
            "readout.json",
            "readout.md",
        ],
        "standing_guard": (
            "Assigned arms are interventions on the model-implied distribution Q_theta; "
            "they are not observed human counterfactuals. PCA and smoothing are display only."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[survival in state space] main figure: {main_png}")
    print(f"[survival in state space] decomposition: {decomposition_png}")
    print(f"[survival in state space] animation: {animation}")
    print(f"[survival in state space] paired ΔRMST24: {contrast[0]:+.2f} months")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizon-months", type=float, default=HORIZON_MONTHS)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
