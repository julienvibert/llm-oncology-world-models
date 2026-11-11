#!/usr/bin/env python3
"""Continuous tumour-host phase space from block-wise PCA.

This script addresses a limitation of the discrete tumour-burden x ECOG
grid: vector fields need a single continuous coordinate system. We build two
clinically constrained composite axes:

  * tumour PC1 from tumour severity features;
  * host PC1 from host vulnerability features.

Treatment is deliberately excluded from the coordinates, so treatment can be
treated as an external perturbation of the vector field. Outcome/death is also
excluded and used only for colouring/validation.

No model calls are made. Reads existing 4-model atlas derivation CSVs and
writes exploratory figures under outputs/.

Run:
  uv run python scripts/explore/composite_tumor_host_phase_space.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs" / "exploratory_composite_phase_space"

ATLAS_DERIVS = (
    ROOT / "runs" / "pillar1_anchored_atlas_tranche1_v4_n5_live01" / "derivation",
    ROOT / "runs" / "pillar1_anchored_atlas_gemma4_n5_live01" / "derivation",
    ROOT / "runs" / "sonnet5_mini_atlas_n100_2026_07_08" / "derivation",
    ROOT / "runs" / "sonnet5_full_atlas_topup_remaining128_n5_2026_07_09" / "derivation",
    ROOT / "runs" / "kimi_k3_off_full_atlas_n5_complete" / "derivation",
    ROOT / "runs" / "glm52_full_atlas_n5" / "derivation",
)

MONTH_DAYS = 30.4
MIN_DT_MONTHS = 1.0
SEED = 20260714
MAX_POINTS = 18000
GRID_N = 13
MIN_TRANSITIONS_PER_CELL = 20

INK = "#1f2937"
MUTED = "#64748b"
GRID = "#e5e7eb"
BLUE = "#31688e"
GREEN = "#1b7837"
ORANGE = "#e08214"
RED = "#b2182b"
PURPLE = "#6a51a3"

MODEL_LABEL = {
    "gemini31_lite": "Gemini",
    "qwen36_flash": "Qwen",
    "gemma4": "Gemma",
    "claude_sonnet5": "Sonnet",
}
MODEL_COLOR = {
    "gemini31_lite": "#31688e",
    "qwen36_flash": "#e08214",
    "gemma4": "#4c9a2a",
    "claude_sonnet5": "#6a51a3",
}

BURDEN_SCORE = {
    "absent": 0.0,
    "microscopic": 0.4,
    "small": 1.0,
    "moderate": 2.0,
    "bulky": 3.0,
    "diffuse_or_leukemic": 4.0,
}
ORGAN_ORDER = {
    "normal": 0.0,
    "mild_impairment": 1.0,
    "moderate_impairment": 2.0,
    "severe_impairment": 3.0,
    "failure": 4.0,
}
SYMPTOM_ORDER = {
    "none": 0.0,
    "minimal": 0.5,
    "mild": 1.0,
    "low": 1.0,
    "moderate": 2.0,
    "high": 3.0,
    "severe": 3.0,
    "very_severe": 4.0,
    "extreme": 4.0,
}
ORGAN_COLS = [
    "organ_renal",
    "organ_hepatic",
    "organ_cardiac",
    "organ_marrow_reserve",
    "organ_pulmonary",
]

HIGH_RISK_SITE_CLASSES = {
    "liver",
    "brain",
    "leptomeningeal_cns",
    "bone_marrow",
    "peritoneum",
    "pleura",
    "lung",
    "adrenal",
    "retroperitoneum",
    "upper_gastrointestinal",
    "pancreatobiliary",
}
CNS_SITE_CLASSES = {"brain", "leptomeningeal_cns"}
VISCERAL_SITE_CLASSES = {
    "liver",
    "lung",
    "peritoneum",
    "pleura",
    "adrenal",
    "retroperitoneum",
    "upper_gastrointestinal",
    "pancreatobiliary",
}

TUMOUR_FEATURES = [
    "tumour_burden_score",
    "num_active_sites_num",
    "high_risk_burden",
    "max_site_burden",
    "liver_burden",
    "cns_burden",
    "marrow_burden",
    "visceral_burden",
    "num_active_clones_num",
    "n_acquired_alterations_num",
]

HOST_FEATURES = [
    "ecog",
    "symptom_ord",
    "n_active_aes_num",
    "organ_max",
    "organ_sum",
    "organ_impaired_count",
]


@dataclass(frozen=True)
class AtlasTables:
    snapshots: pd.DataFrame
    sites: pd.DataFrame
    events: pd.DataFrame
    endpoints: pd.DataFrame


@dataclass(frozen=True)
class AxisModel:
    scaler: RobustScaler
    pca: PCA
    sign: float
    features: list[str]
    label: str


def accepted(df: pd.DataFrame) -> pd.DataFrame:
    if "accepted" not in df.columns:
        return df.copy()
    ok = df["accepted"].astype(str).str.lower().isin({"true", "1", "1.0"})
    return df[ok].copy()


def read_table(name: str) -> pd.DataFrame:
    frames = []
    for deriv in ATLAS_DERIVS:
        path = deriv / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["_run"] = deriv.parent.name
        frame["_gid"] = frame["_run"].astype(str) + "::" + frame["trajectory_id"].astype(str)
        frames.append(accepted(frame))
    if not frames:
        raise FileNotFoundError(f"No `{name}` found under configured atlas derivations")
    return pd.concat(frames, ignore_index=True)


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_tables() -> AtlasTables:
    return AtlasTables(
        snapshots=read_table("snapshot_rows.csv"),
        sites=read_table("disease_site_rows.csv"),
        events=read_table("event_rows.csv"),
        endpoints=read_table("endpoint_rows.csv"),
    )


def build_site_features(sites: pd.DataFrame) -> pd.DataFrame:
    d = sites.copy()
    d["site_class_canonical"] = d["site_class_canonical"].astype(str)
    d["burden_score"] = d["burden_band"].map(BURDEN_SCORE).fillna(0.0)
    d = d[d["burden_score"].gt(0)].copy()
    d["high_risk_burden"] = np.where(d["site_class_canonical"].isin(HIGH_RISK_SITE_CLASSES), d["burden_score"], 0.0)
    d["liver_burden"] = np.where(d["site_class_canonical"].eq("liver"), d["burden_score"], 0.0)
    d["cns_burden"] = np.where(d["site_class_canonical"].isin(CNS_SITE_CLASSES), d["burden_score"], 0.0)
    d["marrow_burden"] = np.where(d["site_class_canonical"].eq("bone_marrow"), d["burden_score"], 0.0)
    d["visceral_burden"] = np.where(d["site_class_canonical"].isin(VISCERAL_SITE_CLASSES), d["burden_score"], 0.0)
    grouped = (
        d.groupby(["_gid", "snapshot_index"], as_index=False)
        .agg(
            tumour_burden_score=("burden_score", "sum"),
            max_site_burden=("burden_score", "max"),
            high_risk_burden=("high_risk_burden", "sum"),
            liver_burden=("liver_burden", "sum"),
            cns_burden=("cns_burden", "sum"),
            marrow_burden=("marrow_burden", "sum"),
            visceral_burden=("visceral_burden", "sum"),
            n_site_rows=("site_class_canonical", "nunique"),
        )
        .copy()
    )
    return grouped


def symptom_to_ord(value: object) -> float:
    if pd.isna(value):
        return 0.0
    num = pd.to_numeric(value, errors="coerce")
    if pd.notna(num):
        return float(num)
    return SYMPTOM_ORDER.get(str(value).strip().lower(), 0.0)


def organ_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in ORGAN_COLS:
        if col in frame:
            out[col] = frame[col].map(lambda v: ORGAN_ORDER.get(str(v).strip().lower(), 0.0)).fillna(0.0)
        else:
            out[col] = 0.0
    return out


def enrich_snapshots(tables: AtlasTables) -> pd.DataFrame:
    s = tables.snapshots.copy()
    s["month"] = numeric(s["days_from_diagnosis"]) / MONTH_DAYS
    s["ecog"] = numeric(s["ecog_ps"]).clip(0, 4)
    s["num_active_sites_num"] = numeric(s["num_active_disease_sites"]).clip(0, 12)
    s["num_active_clones_num"] = numeric(s["num_active_clones"]).clip(0, 10)
    s["n_acquired_alterations_num"] = numeric(s["n_acquired_alterations"]).clip(0, 20)
    s["n_active_aes_num"] = numeric(s["n_active_aes"]).clip(0, 10)
    s["symptom_ord"] = s["symptom_burden"].map(symptom_to_ord).clip(0, 4) if "symptom_burden" in s else 0.0
    weight = numeric(s["weight_kg"], default=np.nan) if "weight_kg" in s else pd.Series(np.nan, index=s.index)
    # Absolute weight is noisy in synthetic data; clip hard and only use very low
    # values as a weak host-reserve signal.
    weight = weight.clip(35, 130)
    s["low_weight_signal"] = ((70.0 - weight) / 20.0).clip(0, 2).fillna(0.0)
    organs = organ_values(s)
    s["organ_max"] = organs.max(axis=1)
    s["organ_sum"] = organs.sum(axis=1)
    s["organ_impaired_count"] = organs.gt(0).sum(axis=1)

    site_features = build_site_features(tables.sites)
    s = s.merge(site_features, on=["_gid", "snapshot_index"], how="left")
    for col in [
        "tumour_burden_score",
        "max_site_burden",
        "high_risk_burden",
        "liver_burden",
        "cns_burden",
        "marrow_burden",
        "visceral_burden",
        "n_site_rows",
    ]:
        s[col] = numeric(s[col]) if col in s else 0.0
    # If the site table is absent for a snapshot, keep the snapshot-level count.
    s.loc[s["num_active_sites_num"].eq(0) & s["n_site_rows"].gt(0), "num_active_sites_num"] = s.loc[
        s["num_active_sites_num"].eq(0) & s["n_site_rows"].gt(0), "n_site_rows"
    ]

    endpoints = tables.endpoints[["_gid", "os_days"]].drop_duplicates("_gid").copy()
    endpoints["death_month"] = numeric(endpoints["os_days"]) / MONTH_DAYS
    s = s.merge(endpoints[["_gid", "death_month"]], on="_gid", how="left")
    s["months_to_death"] = s["death_month"] - s["month"]
    s["future_death_12m"] = ((s["months_to_death"] > 0) & (s["months_to_death"] <= 12)).astype(float)
    s["terminal"] = s["vital_status"].astype(str).str.lower().eq("deceased") | s["outcome_type"].astype(str).str.lower().eq("death")
    return s.sort_values(["_gid", "month", "snapshot_index"]).reset_index(drop=True)


def fit_axis(s: pd.DataFrame, features: list[str], severity: pd.Series, label: str) -> AxisModel:
    fit = s[~s["terminal"]].copy()
    x = fit[features].fillna(0.0).to_numpy(float)
    scaler = RobustScaler(quantile_range=(10, 90))
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=min(3, len(features)), random_state=SEED)
    score = pca.fit_transform(x_scaled)[:, 0]
    corr = np.corrcoef(score, severity.loc[fit.index].to_numpy(float))[0, 1]
    sign = 1.0 if np.isnan(corr) or corr >= 0 else -1.0
    return AxisModel(scaler=scaler, pca=pca, sign=sign, features=features, label=label)


def transform_axis(s: pd.DataFrame, axis: AxisModel) -> np.ndarray:
    x = s[axis.features].fillna(0.0).to_numpy(float)
    return axis.sign * axis.pca.transform(axis.scaler.transform(x))[:, 0]


def fit_composite_axes(s: pd.DataFrame) -> tuple[pd.DataFrame, AxisModel, AxisModel, pd.DataFrame]:
    tumour_severity = (
        s["tumour_burden_score"]
        + 0.7 * s["num_active_sites_num"]
        + 0.5 * s["high_risk_burden"]
        + 0.3 * s["num_active_clones_num"]
        + 0.2 * s["n_acquired_alterations_num"]
    )
    host_severity = s["ecog"] + 0.6 * s["organ_max"] + 0.25 * s["organ_sum"] + 0.4 * s["n_active_aes_num"] + 0.4 * s["symptom_ord"]
    tumour_axis = fit_axis(s, TUMOUR_FEATURES, tumour_severity, "tumour")
    host_axis = fit_axis(s, HOST_FEATURES, host_severity, "host")
    out = s.copy()
    out["tumour_pc1"] = transform_axis(out, tumour_axis)
    out["host_pc1"] = transform_axis(out, host_axis)
    # Put both axes into comparable robust z units for plotting and vector fields.
    for col in ["tumour_pc1", "host_pc1"]:
        med = out.loc[~out["terminal"], col].median()
        iqr = out.loc[~out["terminal"], col].quantile(0.75) - out.loc[~out["terminal"], col].quantile(0.25)
        if not np.isfinite(iqr) or iqr == 0:
            iqr = out[col].std() or 1.0
        out[col] = (out[col] - med) / iqr
    loadings = pca_loadings(tumour_axis, "tumour")
    loadings = pd.concat([loadings, pca_loadings(host_axis, "host")], ignore_index=True)
    return out, tumour_axis, host_axis, loadings


def pca_loadings(axis: AxisModel, block: str) -> pd.DataFrame:
    vals = axis.sign * axis.pca.components_[0]
    # RobustScaler changes scale; this is a PC loading on scaled features, which
    # is exactly what clinicians need to audit directionality.
    return pd.DataFrame({"block": block, "feature": axis.features, "loading": vals})


def pca_variance(axis: AxisModel, block: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block": block,
            "component": [f"PC{i + 1}" for i in range(len(axis.pca.explained_variance_ratio_))],
            "explained_variance": axis.pca.explained_variance_ratio_,
        }
    )


def transition_rows(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in s.groupby("_gid", sort=False):
        g = g.sort_values(["month", "snapshot_index"]).reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            dt = float(b["month"] - a["month"])
            if dt < MIN_DT_MONTHS:
                continue
            rows.append(
                {
                    "_gid": gid,
                    "model_short": a["model_short"],
                    "tumor_oncotree_code": a["tumor_oncotree_code"],
                    "x": float(a["tumour_pc1"]),
                    "y": float(a["host_pc1"]),
                    "dx_6m": float((b["tumour_pc1"] - a["tumour_pc1"]) / dt * 6.0),
                    "dy_6m": float((b["host_pc1"] - a["host_pc1"]) / dt * 6.0),
                    "dt": dt,
                }
            )
    return pd.DataFrame(rows)


def axis_limits(s: pd.DataFrame) -> tuple[float, float, float, float]:
    xmin, xmax = np.nanpercentile(s["tumour_pc1"], [1, 99])
    ymin, ymax = np.nanpercentile(s["host_pc1"], [1, 99])
    pad_x = max(0.45, (xmax - xmin) * 0.06)
    pad_y = max(0.45, (ymax - ymin) * 0.06)
    return float(xmin - pad_x), float(xmax + pad_x), float(ymin - pad_y), float(ymax + pad_y)


def add_risk_background(ax: plt.Axes, s: pd.DataFrame, limits: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = limits
    d = s[s["months_to_death"].notna()].copy()
    xbins = np.linspace(xmin, xmax, GRID_N + 1)
    ybins = np.linspace(ymin, ymax, GRID_N + 1)
    d["xb"] = pd.cut(d["tumour_pc1"], xbins, labels=False, include_lowest=True)
    d["yb"] = pd.cut(d["host_pc1"], ybins, labels=False, include_lowest=True)
    grid = np.full((GRID_N, GRID_N), np.nan)
    for (yb, xb), g in d.groupby(["yb", "xb"]):
        if pd.isna(xb) or pd.isna(yb) or len(g) < 30:
            continue
        grid[int(yb), int(xb)] = float(g["future_death_12m"].mean())
    im = ax.imshow(grid, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="Reds", vmin=0, vmax=1, alpha=0.38, aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("12-mo death", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def add_vector_field(ax: plt.Axes, trans: pd.DataFrame, limits: tuple[float, float, float, float], color: str = "#374151") -> int:
    xmin, xmax, ymin, ymax = limits
    xbins = np.linspace(xmin, xmax, GRID_N + 1)
    ybins = np.linspace(ymin, ymax, GRID_N + 1)
    trans = trans.copy()
    trans["xb"] = pd.cut(trans["x"], xbins, labels=False, include_lowest=True)
    trans["yb"] = pd.cut(trans["y"], ybins, labels=False, include_lowest=True)
    xs: list[float] = []
    ys: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    ns: list[int] = []
    for (yb, xb), g in trans.groupby(["yb", "xb"]):
        if pd.isna(xb) or pd.isna(yb) or len(g) < MIN_TRANSITIONS_PER_CELL:
            continue
        xs.append(float((xbins[int(xb)] + xbins[int(xb) + 1]) / 2))
        ys.append(float((ybins[int(yb)] + ybins[int(yb) + 1]) / 2))
        us.append(float(g["dx_6m"].clip(-1.8, 1.8).mean()))
        vs.append(float(g["dy_6m"].clip(-1.8, 1.8).mean()))
        ns.append(len(g))
    if xs:
        del ns
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1.9,
            width=0.0042,
            color=color,
            alpha=0.92,
            zorder=5,
        )
    return len(xs)


def sample_points(s: pd.DataFrame) -> pd.DataFrame:
    d = s[s["months_to_death"].notna()].copy()
    if len(d) <= MAX_POINTS:
        return d
    return d.sample(MAX_POINTS, random_state=SEED)


def plot_main_space(ax: plt.Axes, s: pd.DataFrame, trans: pd.DataFrame, limits: tuple[float, float, float, float]) -> int:
    add_risk_background(ax, s, limits)
    d = sample_points(s)
    ax.scatter(d["tumour_pc1"], d["host_pc1"], s=5, color="#111827", alpha=0.11, linewidths=0)
    n_arrows = add_vector_field(ax, trans, limits)
    style_phase_axis(ax, limits)
    ax.set_title("A  Continuous tumour-host phase space with empirical velocity field", loc="left", fontsize=10, weight="bold", color=INK)
    ax.text(
        0.01,
        0.98,
        "arrows = mean 6-month drift\nbackground = 12-month death risk",
        transform=ax.transAxes,
        va="top",
        fontsize=7.0,
        color=INK,
        bbox={"boxstyle": "round,pad=0.23", "fc": "white", "ec": GRID, "alpha": 0.88},
    )
    return n_arrows


def style_phase_axis(ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axvline(0, color="#94a3b8", lw=0.8)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.grid(color=GRID, lw=0.65)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("tumour composite PC1  -> worse", fontsize=8)
    ax.set_ylabel("host composite PC1  -> worse", fontsize=8)


def plot_loadings(ax: plt.Axes, loadings: pd.DataFrame) -> None:
    display = loadings.copy()
    display["label"] = display["feature"].str.replace("_", " ", regex=False)
    display = display.sort_values(["block", "loading"])
    y_pos = np.arange(len(display))
    colors = np.where(display["block"].eq("tumour"), BLUE, ORANGE)
    ax.barh(y_pos, display["loading"], color=colors, alpha=0.85)
    ax.axvline(0, color="#94a3b8", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{b[0].upper()}: {lab}" for b, lab in zip(display["block"], display["label"], strict=False)], fontsize=6.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("PC1 loading on robust-scaled feature", fontsize=8)
    ax.set_title("B  Axis audit: what makes tumour/host PC1 increase?", loc="left", fontsize=10, weight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.65)


def choose_example_trajectory(s: pd.DataFrame) -> str:
    candidate = s[s["tumor_oncotree_code"].astype(str).isin({"LUAD", "PAAD", "HGSOC"})].copy()
    if candidate.empty:
        candidate = s.copy()
    stats = (
        candidate.groupby("_gid")
        .agg(
            n=("snapshot_index", "size"),
            x_span=("tumour_pc1", lambda v: float(v.max() - v.min())),
            y_span=("host_pc1", lambda v: float(v.max() - v.min())),
            os=("death_month", "max"),
        )
        .reset_index()
    )
    stats["score"] = stats["n"] + 3.0 * stats["x_span"] + 4.0 * stats["y_span"]
    return str(stats.sort_values("score", ascending=False).iloc[0]["_gid"])


def plot_example_trajectory(ax: plt.Axes, s: pd.DataFrame, events: pd.DataFrame, limits: tuple[float, float, float, float], gid: str) -> dict[str, float | str]:
    traj = s[s["_gid"].eq(gid)].sort_values(["month", "snapshot_index"]).copy()
    if len(traj) >= 2:
        points = traj[["tumour_pc1", "host_pc1"]].to_numpy(float).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(traj["month"].min(), traj["month"].max()), linewidth=2.2)
        lc.set_array(((traj["month"].iloc[:-1] + traj["month"].iloc[1:]) / 2).to_numpy())
        ax.add_collection(lc)
        cb = plt.colorbar(lc, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label("months from diagnosis", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    ax.scatter(traj["tumour_pc1"], traj["host_pc1"], s=22, color="white", edgecolor=INK, linewidth=0.6, zorder=4)
    ep = event_positions(events, traj, gid)
    marker_map = {
        "response": ("v", GREEN),
        "progression": ("^", RED),
        "systemic_treatment_start": ("s", BLUE),
        "systemic_treatment_switch": ("s", ORANGE),
        "clonal_evolution_event": ("D", PURPLE),
        "toxicity": ("o", "#8c564b"),
        "local_intervention": ("*", "#e377c2"),
        "death": ("X", "black"),
    }
    used = set()
    for et, group in ep.groupby("event_type"):
        if et not in marker_map:
            continue
        marker, color = marker_map[et]
        label = et.replace("_", " ")
        ax.scatter(group["x"], group["y"], marker=marker, s=54, color=color, edgecolor="white", linewidth=0.55, label=label if label not in used else None, zorder=5)
        used.add(label)
    style_phase_axis(ax, limits)
    ax.legend(loc="upper left", fontsize=5.8, ncols=2, framealpha=0.92)
    ax.set_title("C  One patient trajectory in the continuous composite space", loc="left", fontsize=10, weight="bold", color=INK)
    ax.text(
        0.01,
        0.02,
        gid.split("::")[-1],
        transform=ax.transAxes,
        fontsize=6.5,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": GRID, "alpha": 0.86},
    )
    return {
        "example_gid": gid,
        "example_snapshots": float(len(traj)),
        "example_os_months": float(traj["death_month"].max()),
    }


def event_positions(events: pd.DataFrame, traj: pd.DataFrame, gid: str) -> pd.DataFrame:
    e = events[events["_gid"].eq(gid)].copy()
    rows = []
    for _, row in e.iterrows():
        day = pd.to_numeric(row.get("days_from_diagnosis"), errors="coerce")
        if pd.isna(day):
            continue
        month = float(day) / MONTH_DAYS
        before = traj[traj["month"].le(month)]
        snap = before.iloc[-1] if not before.empty else traj.iloc[0]
        rows.append({"event_type": row["event_type"], "x": snap["tumour_pc1"], "y": snap["host_pc1"], "month": month})
    return pd.DataFrame(rows)


def plot_model_fields(fig: plt.Figure, s: pd.DataFrame, trans: pd.DataFrame, limits: tuple[float, float, float, float]) -> int:
    models = [m for m in ["gemini31_lite", "qwen36_flash", "gemma4", "claude_sonnet5"] if m in set(s["model_short"])]
    n = len(models)
    axes = fig.subplots(1, n, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    total_arrows = 0
    for ax, model in zip(axes, models, strict=False):
        d = sample_points(s[s["model_short"].eq(model)])
        ax.scatter(d["tumour_pc1"], d["host_pc1"], s=4, color=MODEL_COLOR.get(model, "#555"), alpha=0.10, linewidths=0)
        n_arrows = add_vector_field(ax, trans[trans["model_short"].eq(model)], limits, color=MODEL_COLOR.get(model, "#555"))
        total_arrows += n_arrows
        style_phase_axis(ax, limits)
        ax.set_title(f"{MODEL_LABEL.get(model, model)}\n{n_arrows} field cells", fontsize=9, weight="bold", color=INK)
        if ax is not axes[0]:
            ax.set_ylabel("")
    return total_arrows


def write_readout(metrics: dict[str, float | str], tumour_axis: AxisModel, host_axis: AxisModel) -> None:
    lines = [
        "# Composite Tumour-Host Phase Space",
        "",
        "Block-wise PCA creates continuous, interpretable axes for vector fields.",
        "Treatment and outcome are not included in the coordinates.",
        "",
        "Tumour features:",
        *(f"- `{f}`" for f in tumour_axis.features),
        "",
        "Host features:",
        *(f"- `{f}`" for f in host_axis.features),
        "",
        "Metrics:",
    ]
    for key, value in metrics.items():
        if isinstance(value, str):
            lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append(f"- `{key}`: {value:.4g}")
    lines.extend(
        [
        "",
        "Interpretation:",
        "- This is better suited than the hierarchical grid for vector fields because",
        "  the axes are continuous and globally meaningful.",
        "- PC1 is a severity summary, not a complete state descriptor; PC2/PC3",
        "  variance and loadings are saved for audit.",
        "- Weight is excluded from the main host axis because synthetic weight",
        "  had known outlier artifacts in earlier audits.",
        "- It is still not a fully real-validated coordinate system. For GENIE transfer,",
        "  a reduced Omega-compatible tumour axis would be needed, and the host axis",
            "  is limited by available ECOG/toxicity/organ-function fields.",
        ]
    )
    (OUT / "composite_phase_space_readout.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    snapshots = enrich_snapshots(tables)
    snapshots, tumour_axis, host_axis, loadings = fit_composite_axes(snapshots)
    trans = transition_rows(snapshots)
    limits = axis_limits(snapshots)
    example_gid = choose_example_trajectory(snapshots)

    metrics: dict[str, float | str] = {
        "snapshots": float(len(snapshots)),
        "trajectories": float(snapshots["_gid"].nunique()),
        "models": float(snapshots["model_short"].nunique()),
        "tumour_codes": float(snapshots["tumor_oncotree_code"].nunique()),
        "transitions": float(len(trans)),
        "tumour_pc1_explained_variance": float(tumour_axis.pca.explained_variance_ratio_[0]),
        "host_pc1_explained_variance": float(host_axis.pca.explained_variance_ratio_[0]),
    }

    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.965, bottom=0.075, top=0.88, hspace=0.32, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    metrics["global_vector_field_cells"] = float(plot_main_space(ax_a, snapshots, trans, limits))
    plot_loadings(ax_b, loadings)
    metrics.update(plot_example_trajectory(ax_c, snapshots, tables.events, limits, example_gid))
    fig.suptitle("Continuous tumour-host phase space from clinically constrained composite axes", x=0.06, y=0.975, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.06, 0.945, "Tumour PC1 and host PC1 are fitted separately; higher means worse. Adjacent snapshots define the field.", ha="left", fontsize=8.8, color=MUTED)
    out_png = OUT / "fig_composite_tumor_host_phase_space.png"
    out_svg = OUT / "fig_composite_tumor_host_phase_space.svg"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)

    fig2 = plt.figure(figsize=(15.2, 4.2))
    metrics["model_vector_field_cells_total"] = float(plot_model_fields(fig2, snapshots, trans, limits))
    fig2.suptitle("Model-specific empirical fields in the same tumour-host composite space", x=0.04, y=1.04, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig2.savefig(OUT / "fig_composite_tumor_host_model_fields.png", dpi=180, bbox_inches="tight")
    fig2.savefig(OUT / "fig_composite_tumor_host_model_fields.svg", bbox_inches="tight")
    plt.close(fig2)

    snapshots[
        [
            "_gid",
            "trajectory_id",
            "model_short",
            "tumor_oncotree_code",
            "snapshot_index",
            "month",
            "tumour_pc1",
            "host_pc1",
            "months_to_death",
            "future_death_12m",
            *TUMOUR_FEATURES,
            *HOST_FEATURES,
        ]
    ].to_csv(OUT / "composite_phase_space_points.csv", index=False)
    trans.to_csv(OUT / "composite_phase_space_transitions.csv", index=False)
    loadings.to_csv(OUT / "composite_phase_space_loadings.csv", index=False)
    pd.concat([pca_variance(tumour_axis, "tumour"), pca_variance(host_axis, "host")], ignore_index=True).to_csv(
        OUT / "composite_phase_space_variance.csv", index=False
    )
    write_readout(metrics, tumour_axis, host_axis)

    print(f"[composite] snapshots={len(snapshots)} trajectories={snapshots['_gid'].nunique()} transitions={len(trans)}")
    print(f"[composite] explained tumour={tumour_axis.pca.explained_variance_ratio_[0]:.3f} host={host_axis.pca.explained_variance_ratio_[0]:.3f}")
    print(f"[composite] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
