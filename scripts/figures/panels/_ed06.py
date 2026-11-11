"""Shared computation and drawing for Extended Data Fig. 6 (vector fields of generated trajectories).

Data: the frozen state map applied to the generated atlas (outputs/paper_state_map/state_projection.csv.gz: 105,839
living visit states, 4,434 trajectories, six models, 148 cancer types, 30 trajectories per cancer type) and the death
events of the six runs (runs/<run>/derivation/event_rows.csv). Functions of
scripts/build_paper_group_state_vector_field_views.py are reused for loading, paths and state lookup.

Field definition (the same everywhere): start states are taken at 3, 6, 12, 18 and 24 months after diagnosis for every
trajectory that is alive then; each is followed for 6 months. The map is cut into equal cells; in each cell the arrow is
the mean movement among start states still followed (alive) 6 months later; the share not followed is the share
that died within the 6 months. Trajectories whose record ends alive before the 6 months are left out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

# The builder sets DejaVu Sans and other rcParams at import. Import it inside rc_context so the house style
# (Arial, set by common) survives whatever the import order, including when common was imported first.
with matplotlib.rc_context():
    import build_paper_group_state_vector_field_views as o  # noqa: E402
from common import LIGHT  # noqa: E402

SRC_PROJ = "outputs/paper_state_map/state_projection.csv.gz"
SRC_EVENTS = "runs/<six atlas runs>/derivation/event_rows.csv"
FRAMES_MONTHS = (3, 6, 12, 18, 24)
HORIZON_MONTHS = 6.0
MONTH = 30.44
FAMILY = {
    "Lung": ["LUAD", "LUSC", "SCLC", "NSCLC", "LCLC", "NSCLCPD", "LUNE", "LUAS"],
    "Breast": ["BRCA", "IDC", "ILC", "IBC", "BRCNOS", "MDLC", "BRCANOS"],
    "Gastrointestinal": ["COAD", "READ", "COADREAD", "STAD", "ESCA", "PAAD", "CHOL", "HCC", "GIST", "ESCC", "GEJ",
                         "SBC", "PANET", "PANEC", "ACC"],
    "Genitourinary": ["PRAD", "BLCA", "CCRCC", "RCC", "PRCC", "CHRCC", "UCU", "URCA", "SEM", "NSGCT", "TGCT"],
    "Gynaecological": ["HGSOC", "OV", "OVT", "UCEC", "CESC", "USC", "UCS", "OCS", "VMM"],
    "Sarcoma": ["LMS", "DDLS", "WDLS", "SYNS", "ES", "OS", "MFS", "UPS", "MPNST", "DES", "ASPS", "EHAE", "CHS",
                "CHDM", "RMS", "ERMS", "ARMS"],
    "Skin / melanoma": ["SKCM", "MEL", "ACRM", "MUP", "UM", "BCC", "CSCC", "MCC"],
    "Head & neck": ["HNSC", "OPHSC", "NPC", "HNSCUP", "OCSC", "LXSC", "HPHSC", "THPA", "THFO", "THHC", "THME"],
}
CODE2FAM = {c: f for f, codes in FAMILY.items() for c in codes}
# source: scripts/explore/build_field_trellis.py (FAMILY), the mapping used by the field-aggregation audit


def load_members() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (members, projection): one row per (trajectory, start month) with start and end map positions."""
    proj = o.load_projection()
    ids = set(proj["trajectory_id"].astype(str))
    events = o.load_events(sorted(proj["_run"].astype(str).unique()), ids)
    deaths = o.death_days(events)
    paths = o.build_paths(proj)
    fam = proj.drop_duplicates("trajectory_id").set_index("trajectory_id")["tumor_oncotree_code"].astype(str).map(CODE2FAM)
    rows = []
    for tid, path in paths.items():
        dd = deaths.get(tid)
        for m in FRAMES_MONTHS:
            day = m * MONTH
            start = o.state_at_day(path, day, dd)
            if start is None:
                continue
            end_day = day + HORIZON_MONTHS * MONTH
            if dd is not None and dd <= end_day:
                outcome, ex, ey = "died", np.nan, np.nan
            else:
                end = o.state_at_day(path, end_day, dd)
                if end is None:
                    continue  # record ends alive before the horizon: left out
                outcome, ex, ey = "followed", float(end["pc1"]), float(end["pc2"])
            rows.append((tid, fam.get(tid), m, float(start["pc1"]), float(start["pc2"]), outcome, ex, ey))
    mem = pd.DataFrame(rows, columns=["trajectory_id", "family", "start_month", "x0", "y0", "outcome", "x1", "y1"])
    return mem, proj


def grid_field(mem: pd.DataFrame, xe: np.ndarray, ye: np.ndarray, min_support: int) -> pd.DataFrame:
    ix = np.clip(np.digitize(mem["x0"], xe) - 1, 0, len(xe) - 2)
    iy = np.clip(np.digitize(mem["y0"], ye) - 1, 0, len(ye) - 2)
    m = mem.assign(ix=ix, iy=iy)
    out = []
    for (a, b), g in m.groupby(["ix", "iy"]):
        f = g[g.outcome == "followed"]
        out.append(dict(ix=a, iy=b, cx=g.x0.mean(), cy=g.y0.mean(), n=len(g), n_follow=len(f),
                        n_traj=g.trajectory_id.nunique(),
                        dx=(f.x1 - f.x0).mean() if len(f) else np.nan, dy=(f.y1 - f.y0).mean() if len(f) else np.nan,
                        dead=(g.outcome == "died").mean()))
    r = pd.DataFrame(out)
    return r[(r.n >= min_support) & (r.n_follow > 0)].reset_index(drop=True)


from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

DEATH_CMAP = LinearSegmentedColormap.from_list("death_share", ["#4FA3C7", "#7A4FA3", "#B3122E"])
DEATH_NORM = Normalize(0.0, 0.5)
XLAB = "State-map PC1 (a.u.)"
YLAB = "State-map PC2 (a.u.)"


def draw_field(ax, mem, field, xl, yl, lw=1.0, head=4.5, scale=1.0, stream=True, base_dots=True):
    """Background of start states, optional streamlines, and one arrow per cell coloured by the share who died."""
    if base_dots:
        ax.scatter(mem.x0, mem.y0, s=0.6, color=LIGHT, alpha=0.5, linewidths=0, zorder=0, rasterized=True)
    ax.quiver(field.cx, field.cy, field.dx * scale, field.dy * scale, field.dead, cmap=DEATH_CMAP, norm=DEATH_NORM,
              angles="xy", scale_units="xy", scale=1, width=lw / 400, headwidth=head * 0.8, headlength=head,
              headaxislength=head * 0.85, zorder=4, pivot="tail")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([])
    ax.set_yticks([])


def kernel_field(mem: pd.DataFrame, xl, yl, nx=60, ny=46, h=0.08, min_eff=15.0):
    """Kernel-weighted (Gaussian, bandwidth h in map units) mean movement of followed start states on a lattice.
    Returns lattice axes and arrays dx, dy, dead, n_eff; lattice points with n_eff < min_eff are masked (NaN)."""
    gx, gy = np.linspace(*xl, nx), np.linspace(*yl, ny)
    X, Y = np.meshgrid(gx, gy)
    P = np.c_[X.ravel(), Y.ravel()]
    S = mem[["x0", "y0"]].to_numpy()
    fol = (mem.outcome == "followed").to_numpy()
    dxy = (mem[["x1", "y1"]].to_numpy() - S)
    dxy[~fol] = 0.0
    dead = (~fol).astype(float)
    neff = np.zeros(len(P))
    sw = np.zeros(len(P))
    sdx = np.zeros(len(P))
    sdy = np.zeros(len(P))
    sd = np.zeros(len(P))
    for a in range(0, len(P), 400):
        d2 = ((P[a:a + 400, None, :] - S[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * h * h))
        neff[a:a + 400] = w.sum(1)
        wf = w * fol[None, :]
        sw[a:a + 400] = wf.sum(1)
        sdx[a:a + 400] = (wf * dxy[None, :, 0]).sum(1)
        sdy[a:a + 400] = (wf * dxy[None, :, 1]).sum(1)
        sd[a:a + 400] = (w * dead[None, :]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx, dy, dd = sdx / sw, sdy / sw, sd / neff
    ok = (neff >= min_eff) & (sw > 0)
    for arr in (dx, dy, dd):
        arr[~ok] = np.nan
    sh = X.shape
    return gx, gy, dx.reshape(sh), dy.reshape(sh), dd.reshape(sh), neff.reshape(sh)


def finite_field(mem: pd.DataFrame, xl, yl, cell: float, min_courses: int, min_survivors: int) -> pd.DataFrame:
    """Square cells of side `cell` (map units, equal on both axes). Per cell: start visits, unique courses (trajectories),
    survivor endpoints (visits still alive 6 months later) and the unique courses they come from, mean start position,
    mean six-month displacement among survivors, share of start visits that died within 6 months.
    A cell is kept when it has >= min_courses unique courses and >= min_survivors survivor endpoints."""
    ix = np.floor((mem.x0 - xl[0]) / cell).astype(int)
    iy = np.floor((mem.y0 - yl[0]) / cell).astype(int)
    m = mem.assign(ix=ix, iy=iy)
    rows = []
    for (a, b), g in m.groupby(["ix", "iy"]):
        f = g[g.outcome == "followed"]
        rows.append(dict(ix=a, iy=b, cx=g.x0.mean(), cy=g.y0.mean(), n_start_visits=len(g), n_courses=g.trajectory_id.nunique(),
                         n_survivor_endpoints=len(f), n_survivor_courses=f.trajectory_id.nunique(),
                         dx=(f.x1 - f.x0).mean() if len(f) else np.nan, dy=(f.y1 - f.y0).mean() if len(f) else np.nan,
                         dead_share=(g.outcome == "died").mean()))
    r = pd.DataFrame(rows)
    return r[(r.n_courses >= min_courses) & (r.n_survivor_endpoints >= min_survivors)].reset_index(drop=True)


def equal_limits(xl, yl, w_mm: float, h_mm: float):
    s = max((xl[1] - xl[0]) / w_mm, (yl[1] - yl[0]) / h_mm)
    cx, cy = sum(xl) / 2, sum(yl) / 2
    return (cx - s * w_mm / 2, cx + s * w_mm / 2), (cy - s * h_mm / 2, cy + s * h_mm / 2)
