"""figkit: publication figures at final size, with an audit.

Usage (from any script):

    import sys; sys.path.insert(0, "<path to julien-figures/scripts>")
    import figkit as fk

    fk.use("nature")                                  # journal preset + house style
    fig, axs = fk.subplots(1, 3, width="double", height_mm=45)   # width: "single", "1.5", "double" or mm
    fk.bar_dots(axs[0], ["Ctrl", "KD"], [ctrl, kd], colors=[fk.GREY, fk.ACCENT])
    axs[0].set_ylabel("Relative expression (log2)")
    fk.label_panels(axs)
    fk.save(fig, "out/fig1", legend=legend_text, confirm=["n = mice or litters?"])
    # -> PDF + SVG + PNG + audit + out/fig1_handover.md (status: checked draft)

Everything is in real units: the figure is created at its printed size, so
font sizes in points are the sizes the reader sees. Never rescale afterwards.

Journal keys for use(): "nature" (Nature itself, 89/183 mm), "nrj" (Nature
Research journals: Nature Communications, Nature Genetics, Nature Cancer...,
88/180 mm), "cell" (Cell Press: Cell, Molecular Cell, Cell Reports, Med,
Cancer Cell; 85/114/174 mm), "science" (provisional, unverified).

Functions
  Layout:  use, figure, subplots, label_panels, fig_label, log_axis
  Plots (each returns the legend sentence for its statistics):
           bar_dots (style="bar" or "line"), grouped_bar_dots, box_dots,
           superplot (observations nested in replicates), paired_dots (paired
           samples, unpaired values kept and marked)
  Text:    rich_text, gene_title (italic gene names with digits), p_text, bracket
  Stats:   summary, t_quantile, welch_t, paired_t, wilcoxon_signed_rank (exact,
           small n); all without SciPy, each returns (statistic, P, description)
  Colour:  GREY, LIGHT, ACCENT, CATEGORICAL, DIVERGING_CMAP, cvd_report
  Checks:  audit, save (legend=, confirm=, synthetic=, caption_words=, final=),
           synthetic_stamp
"""

from __future__ import annotations

import colorsys
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation

MM = 1 / 25.4  # inches per mm

# --------------------------------------------------------------------------
# Journal presets. Sources and verification dates: references/journal-specs.md
# --------------------------------------------------------------------------


@dataclass
class Journal:
    name: str
    widths: dict  # name -> mm
    max_height: float  # mm
    font_min: float  # pt, publisher minimum
    font_max: float  # pt, publisher maximum (panel labels excluded)
    label_size: float  # pt
    label_case: str  # "lower" or "upper"
    label_weight: str = "bold"
    notes: list = field(default_factory=list)
    unverified: list = field(default_factory=list)  # fields not confirmed on the publisher's own page

    def height_limit(self, width_mm, caption_words=None):
        """Maximum figure height (mm), and a note on the assumption used."""
        if self.name == "Nature Research Journals":
            single = width_mm < 120
            table = [(50, 220, 225), (150, 180, 210), (300, 130, 185)]  # <words: 1-col, 2-col
            if caption_words is None:
                return (130 if single else 185), "legend length not given: assumed up to 300 words (strictest)"
            for limit, h1, h2 in table:
                if caption_words < limit:
                    return (h1 if single else h2), f"legend of {caption_words} words"
            return (130 if single else 185), f"legend of {caption_words} words exceeds the 300-word table"
        return self.max_height, ""


JOURNALS = {
    # nature.com research figure guide (checked 2026-10-10)
    "nature": Journal("Nature", {"single": 89, "double": 183}, 170, 5, 7, 8, "lower"),
    # Nature Research Journals 'Guide to preparing final artwork' (examples/ PDF)
    "nrj": Journal(
        "Nature Research Journals",
        {"single": 88, "double": 180},
        185,
        5,
        7,
        8,
        "lower",
        notes=["Max height depends on legend length: 2-col 185 mm (<300 words), 210 mm (<150), 225 mm (<50)."],
    ),
    # cell.com figure guidelines (widths and 6-8 pt confirmed 2026-10-10; max height not confirmed)
    "cell": Journal("Cell Press", {"single": 85, "1.5": 114, "double": 174}, 225, 6, 8, 8, "upper",
                    unverified=["maximum height 225 mm (secondary sources)", "panel label size 8 pt"]),
    # Science: NOT verified against science.org; confirm before submission
    "science": Journal(
        "Science (unverified)",
        {"single": 55, "1.5": 120, "double": 183},
        230,
        5,
        7,
        8,
        "upper",
        notes=["Science dimensions not verified against science.org; check before submission."],
        unverified=["all widths", "maximum height", "font sizes", "panel label style"],
    ),
}

# House rule: nothing below 6 pt, whatever the journal allows.
HOUSE_FONT_MIN = 6.0

# --------------------------------------------------------------------------
# Colours. Grey for context, one accent for the message. Categorical set is
# checked for colour-vision deficiency (see cvd_report()).
# --------------------------------------------------------------------------
BLACK = "#000000"
INK = "#1a1a1a"
GREY = "#8c8c8c"  # reference / control / other categories
LIGHT = "#d0d0d0"  # background points, non-highlighted
ACCENT = "#d6402b"  # the condition of interest (softened red)
BLUE = "#2a6fbf"
ORANGE = "#e69f00"
TEAL = "#009e73"
PURPLE = "#8e5fa8"
SKY = "#56b4e9"

DKBLUE = "#0b3c7a"

# Searched, not chosen by eye: the only 4-colour sets (with blue and red) that
# stay distinct from each other AND from grey/black under protan-, deutan- and
# tritanopia. Teal and purple fail against grey or blue; use them only when the
# audit says so. Above 4 categories: facets, shapes or direct labels.
CATEGORICAL = [BLUE, ACCENT, ORANGE, SKY]
DIVERGING_CMAP = "RdBu_r"  # centred on 0 (z-scores, log fold changes)
SEQUENTIAL_CMAP = "Greys"  # or 'Blues', 'viridis' when many levels must be read
FORBIDDEN_CMAPS = {"jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral", "turbo", "gist_ncar", "spectral"}

_current = {"journal": JOURNALS["nature"]}


def use(journal: str = "nature"):
    """Apply journal preset and house style to matplotlib."""
    j = JOURNALS[journal]
    _current["journal"] = j
    text = 7.0 if j.font_max >= 7 else j.font_max
    small = max(HOUSE_FONT_MIN, min(6.0, j.font_max))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": text,
            "axes.labelsize": text,
            "axes.titlesize": text,
            "axes.titleweight": "normal",
            "axes.titlepad": 3,
            "axes.labelpad": 2,
            "xtick.labelsize": small,
            "ytick.labelsize": small,
            "legend.fontsize": small,
            "legend.title_fontsize": small,
            "figure.titlesize": text,
            "figure.labelsize": text,
            "figure.labelweight": "normal",
            "mathtext.default": "regular",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": True,
            # lines
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "patch.linewidth": 0.5,
            "hatch.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.minor.size": 1.2,
            "ytick.minor.size": 1.2,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "errorbar.capsize": 1.5,
            # clean axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "image.cmap": SEQUENTIAL_CMAP,
            # legends: no box, compact
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handleheight": 0.7,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "legend.borderpad": 0.2,
            "legend.labelspacing": 0.3,
            "legend.columnspacing": 0.8,
            # output: editable text, embedded TrueType (Type 42), never Type 3
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "boxplot.flierprops.markersize": 2,
        }
    )
    return j


def journal() -> Journal:
    return _current["journal"]


def width_mm(width) -> float:
    if isinstance(width, (int, float)):
        return float(width)
    return float(journal().widths[width])


def figure(width="double", height_mm=60, **kw):
    """Empty figure at final printed size, constrained layout with tight pads."""
    fig = plt.figure(figsize=(width_mm(width) * MM, height_mm * MM), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=1 * MM, h_pad=1 * MM, wspace=0.02, hspace=0.02)
    return fig


def subplots(nrows=1, ncols=1, width="double", height_mm=60, **kw):
    fig = figure(width, height_mm)
    axs = fig.subplots(nrows, ncols, **kw)
    return fig, axs


# --------------------------------------------------------------------------
# Panel labels
# --------------------------------------------------------------------------


def _fmt_label(s: str) -> str:
    return s.lower() if journal().label_case == "lower" else s.upper()


def label_panels(axes, labels=None, dx_mm=None, dy_mm=1.0):
    """Label panels a, b, c (Nature) or A, B, C (Cell) at the top left of each axes.

    Labels in the same column share an x position (left of the widest y-axis
    labels); labels in the same row share a baseline. Labels are axes children,
    so constrained layout makes room for them.
    """
    axes = list(np.ravel(axes))
    labels = labels or [chr(ord("a") + i) for i in range(len(axes))]
    fig = axes[0].figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # distance from spine to the left edge of tick labels + y label, per axes
    offs = []
    for ax in axes:
        tb = ax.get_tightbbox(r)
        pos = ax.get_window_extent(r)
        offs.append(max(0.0, (pos.x0 - tb.x0) / fig.dpi / MM))
    # group axes into columns by left spine position (within 3 mm)
    lefts = [ax.get_position().x0 * fig.get_figwidth() / MM for ax in axes]
    col_off = {}
    for i, L in enumerate(lefts):
        key = next((k for k in col_off if abs(k - L) < 3), L)
        col_off[key] = max(col_off.get(key, 0), offs[i])
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        key = next(k for k in col_off if abs(k - lefts[i]) < 3)
        dx = (dx_mm if dx_mm is not None else col_off[key] + 1.0) * MM
        has_title = bool(ax.get_title())
        dy = (dy_mm + (3.0 if has_title else 0)) * MM
        t = ax.text(
            0,
            1,
            _fmt_label(lab),
            transform=ax.transAxes + ScaledTranslation(-dx, dy, fig.dpi_scale_trans),
            fontsize=journal().label_size,
            fontweight=journal().label_weight,
            va="bottom",
            ha="left",
            color=BLACK,
        )
        t.set_gid("panel-label")
    return axes


def fig_label(fig, x_mm, y_mm_from_top, label):
    """Panel label at an absolute position (for schematic or image panels)."""
    W, H = fig.get_size_inches() / MM
    t = fig.text(
        x_mm / W,
        1 - y_mm_from_top / H,
        _fmt_label(label),
        fontsize=journal().label_size,
        fontweight=journal().label_weight,
        va="top",
        ha="left",
        color=BLACK,
    )
    t.set_gid("panel-label")
    return t


# --------------------------------------------------------------------------
# Plot helpers that encode the house defaults
# --------------------------------------------------------------------------


def _jitter(n, width, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * width


# ---- statistics: explicit, no silent fallbacks ----------------------------


def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def _betai(a, b, x):
    from math import exp, lgamma, log

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def t_quantile(q, df):
    """Student t quantile (exact to ~1e-10), no SciPy needed."""
    def cdf(t):
        x = df / (df + t * t)
        tail = 0.5 * _betai(df / 2, 0.5, x)
        return 1 - tail if t >= 0 else tail

    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Summary:
    mean: float
    err: float  # nan when undefined (n < 2)
    n: int
    n_missing: int


def summary(values, err="sem", missing="error"):
    """Mean and uncertainty of one group.

    err: 'sem', 'sd' or 'ci95' (Student t, df = n - 1).
    missing: 'error' (default) refuses NaN; 'drop' removes them and counts them,
    so the legend can report the exclusion. n = 1 gives err = nan (undefined),
    never 0.
    """
    v = np.asarray(values, float)
    nan = int(np.isnan(v).sum())
    if nan and missing != "drop":
        raise ValueError(f"{nan} missing value(s) in a group; pass missing='drop' to exclude and report them.")
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        raise ValueError("A group has no values.")
    m = float(v.mean())
    if n < 2:
        e = float("nan")
    elif err == "sem":
        e = float(v.std(ddof=1) / np.sqrt(n))
    elif err == "sd":
        e = float(v.std(ddof=1))
    elif err == "ci95":
        e = float(t_quantile(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    else:
        raise ValueError(f"err must be 'sem', 'sd' or 'ci95', not {err!r}")
    return Summary(m, e, n, nan)


ERR_TEXT = {"sem": "mean ± s.e.m.", "sd": "mean ± s.d.", "ci95": "mean with 95% confidence interval (Student t)"}


def _check_len(what, seq, expected, label):
    if len(seq) != expected:
        raise ValueError(f"{what}: {len(seq)} {label} for {expected} groups. Nothing is drawn rather than drop data.")


def _colors_for(colors, n, default):
    if colors is None:
        return [default] * n
    if isinstance(colors, str):
        return [colors] * n
    colors = list(colors)
    _check_len("colors", colors, n, "colours")
    return colors


def _fragment(err, sums, names, n_unit, shape="Bars"):
    ns = [s.n for s in sums]
    nfrag = f"n = {ns[0]}" if len(set(ns)) == 1 else "n = " + ", ".join(f"{k} ({g})" for k, g in zip(ns, names))
    out = f"{shape}, {ERR_TEXT[err]} of {nfrag} {n_unit} (dots)."
    undefined = [g for s, g in zip(sums, names) if np.isnan(s.err)]
    if undefined:
        out += f" No error bar where n = 1 ({', '.join(map(str, undefined))})."
    miss = [(g, s.n_missing) for s, g in zip(sums, names) if s.n_missing]
    if miss:
        out += " Missing values excluded: " + ", ".join(f"{g} ({k})" for g, k in miss) + "."
    return out


def bar_dots(ax, groups, data, colors=None, err="sem", width=0.7, dot_size=6, seed=0, style="bar",
             n_unit="[biological replicates?]", missing="error"):
    """Mean (bar, or a short line with style='line') with error bar and every value as a dot.

    For small n, style='line' (points + mean line) often says more than a bar.
    Returns the legend sentence, e.g. 'Bars, mean ± s.e.m. of n = 3 biological
    replicates (dots).' `n_unit` says what n counts; the default is a visible
    placeholder. Mismatched groups/data/colors raise instead of dropping data.
    """
    _check_len("data", data, len(groups), "data series")
    colors = _colors_for(colors, len(groups), GREY)
    sums = []
    for i, (vals, c) in enumerate(zip(data, colors)):
        sm = summary(vals, err, missing)
        sums.append(sm)
        if style == "bar":
            ax.bar(i, sm.mean, width=width, color=c, edgecolor="none", zorder=1)
        else:
            ax.plot([i - width * 0.35, i + width * 0.35], [sm.mean] * 2, color=c if c != GREY else INK, lw=1.2, zorder=3)
        if not np.isnan(sm.err):
            ax.errorbar(i, sm.mean, yerr=sm.err, color=INK, lw=0.6, capsize=1.5, capthick=0.6, zorder=3)
        v = np.asarray(vals, float)
        v = v[~np.isnan(v)]
        ax.scatter(i + _jitter(len(v), width * 0.45, seed + i), v, s=dot_size, facecolor="white",
                   edgecolor=INK, linewidth=0.4, zorder=4, clip_on=False)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.tick_params(axis="x", length=0)
    return _fragment(err, sums, groups, n_unit, "Bars" if style == "bar" else "Lines")


def grouped_bar_dots(ax, categories, conditions, data, colors=None, err="sem", width=0.8, dot_size=5, seed=0,
                     n_unit="[biological replicates?]", legend=True, missing="error"):
    """Grouped bars: one group per category, one bar per condition (e.g. -DOX/+DOX per gene).

    data[c][k] = values for condition c, category k. Default colours cover up to
    4 conditions; more conditions need explicit colors (or small multiples).
    """
    _check_len("data", data, len(conditions), "condition series")
    for ci, row in enumerate(data):
        if len(row) != len(categories):
            raise ValueError(f"data[{ci}] has {len(row)} categories, expected {len(categories)}.")
    if colors is None:
        default = [GREY, ACCENT, BLUE, ORANGE]
        if len(conditions) > len(default):
            raise ValueError(f"{len(conditions)} conditions but {len(default)} default colours: pass colors= "
                             "or use small multiples.")
        colors = default[: len(conditions)]
    colors = _colors_for(colors, len(conditions), GREY)
    nc = len(conditions)
    bw = width / nc
    sums, names = [], []
    for ci, (cond, col) in enumerate(zip(conditions, colors)):
        for k in range(len(categories)):
            sm = summary(data[ci][k], err, missing)
            sums.append(sm)
            names.append(f"{categories[k]} {cond}")
            x = k - width / 2 + bw * (ci + 0.5)
            ax.bar(x, sm.mean, width=bw * 0.92, color=col, edgecolor="none", label=cond if k == 0 else "_", zorder=1)
            if not np.isnan(sm.err):
                ax.errorbar(x, sm.mean, yerr=sm.err, color=INK, lw=0.5, capsize=1.2, capthick=0.5, zorder=3)
            vals = np.asarray(data[ci][k], float)
            vals = vals[~np.isnan(vals)]
            ax.scatter(x + _jitter(len(vals), bw * 0.4, seed + 10 * ci + k), vals, s=dot_size, facecolor="white",
                       edgecolor=INK, linewidth=0.35, zorder=4, clip_on=False)
    ax.set_xticks(range(len(categories)), categories)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.tick_params(axis="x", length=0)
    if legend:
        ax.legend(loc="best")
    return _fragment(err, sums, names, n_unit)


def box_dots(ax, groups, data, colors=None, dots=True, dot_size=4, width=0.6, seed=0, highlight=None, vert=True,
             max_dots=200, n_in_labels="auto", n_unit="[samples?]", missing="error"):
    """Tukey box plot (whiskers 1.5 x IQR) with individual values; no separate fliers.

    `highlight` = index (or list) of groups drawn in ACCENT, others white: the
    Mol Cell Fig. 1D/E pattern. n goes in the category labels when it differs
    between groups, otherwise once in the returned legend sentence.
    """
    n = len(groups)
    _check_len("data", data, n, "data series")
    clean, counts, miss = [], [], []
    for g, d in zip(groups, data):
        v = np.asarray(d, float)
        k = int(np.isnan(v).sum())
        if k and missing != "drop":
            raise ValueError(f"{k} missing value(s) in group {g}; pass missing='drop' to exclude and report them.")
        v = v[~np.isnan(v)]
        if not len(v):
            raise ValueError(f"Group {g} has no values.")
        clean.append(v)
        counts.append(len(v))
        if k:
            miss.append((g, k))
    # all groups get dots or none do: mixing reads as a difference between groups
    if dots and max(counts) > max_dots:
        dots = False
    hl = set(np.atleast_1d(highlight)) if highlight is not None else set()
    colors = [ACCENT if i in hl else "white" for i in range(n)] if colors is None else _colors_for(colors, n, "white")
    bp = ax.boxplot(clean, positions=range(n), widths=width, vert=vert, patch_artist=True, showfliers=not dots, whis=1.5,
                    medianprops=dict(color=INK, lw=0.8), whiskerprops=dict(color=INK, lw=0.5),
                    capprops=dict(color=INK, lw=0.5), boxprops=dict(edgecolor=INK, lw=0.5),
                    flierprops=dict(marker="o", markersize=1.5, markerfacecolor=INK, markeredgecolor="none"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    if dots:
        for i, v in enumerate(clean):
            j = _jitter(len(v), width * 0.5, seed + i)
            xy = (i + j, v) if vert else (v, i + j)
            ax.scatter(*xy, s=dot_size, color=INK, alpha=0.6, linewidth=0, zorder=3)
    if n_in_labels == "auto":
        n_in_labels = len(set(counts)) > 1
    labels = [f"{g} (n={c})" for g, c in zip(groups, counts)] if n_in_labels else list(groups)
    if vert:
        ax.set_xticks(range(n), labels)
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_yticks(range(n), labels)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
    for i in hl:
        lab = (ax.get_xticklabels() if vert else ax.get_yticklabels())[i]
        lab.set_fontweight("bold")
    nfrag = f"n = {counts[0]} {n_unit} per group" if len(set(counts)) == 1 else f"n {n_unit} per group in the labels"
    tail = "dots, individual values" if dots else "points beyond the whiskers shown individually"
    out = (f"Boxes, median and interquartile range; whiskers, 1.5 × the interquartile range; {tail}; {nfrag}.")
    if miss:
        out += " Missing values excluded: " + ", ".join(f"{g} ({k})" for g, k in miss) + "."
    return out


def superplot(ax, groups, data, colors=None, err="sem", width=0.7, cell_size=2, rep_size=14, seed=0,
              rep_unit="[independent experiments?]", obs_unit="cells"):
    """SuperPlot (Lord et al., J Cell Biol 2020) for observations nested in replicates.

    data[g] = list of replicates, each a list of observations (e.g. cells per
    experiment). Small grey dots: observations. Large symbols, one shape per
    replicate: replicate means. Line + error bar: mean ± err of replicate means.
    The summary and any test use n = number of replicates, not observations.
    """
    _check_len("data", data, len(groups), "data series")
    colors = _colors_for(colors, len(groups), INK)
    markers = ["o", "s", "^", "D", "v", "P"]
    sums, n_obs = [], []
    for i, (reps, c) in enumerate(zip(data, colors)):
        if len(reps) > len(markers):
            raise ValueError(f"{len(reps)} replicates in {groups[i]}; at most {len(markers)} shapes.")
        means = []
        total = 0
        for r, obs in enumerate(reps):
            v = np.asarray(obs, float)
            if np.isnan(v).any():
                raise ValueError(f"Missing values in {groups[i]} replicate {r + 1}.")
            total += len(v)
            ax.scatter(i + _jitter(len(v), width * 0.7, seed + 100 * i + r), v, s=cell_size, color=LIGHT,
                       linewidth=0, zorder=1, rasterized=len(v) > 500)
            means.append(v.mean())
            ax.scatter(i + (r - (len(reps) - 1) / 2) * width * 0.18, v.mean(), s=rep_size, marker=markers[r],
                       facecolor=c if c != INK else "white", edgecolor=INK, linewidth=0.5, zorder=4)
        sm = summary(means, err)
        sums.append(sm)
        n_obs.append(total)
        # summary behind the replicate symbols; error bar beside them so neither hides the other
        ax.plot([i - width * 0.45, i + width * 0.45], [sm.mean] * 2, color=INK, lw=1.0, zorder=2)
        if not np.isnan(sm.err):
            ax.errorbar(i + width * 0.5, sm.mean, yerr=sm.err, color=INK, lw=0.6, capsize=1.5, capthick=0.6, zorder=2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.tick_params(axis="x", length=0)
    ns = sorted({s.n for s in sums})
    return (f"Small grey dots, individual {obs_unit} ({', '.join(f'{g}: {k}' for g, k in zip(groups, n_obs))}); "
            f"symbols, mean of each of n = {'/'.join(map(str, ns))} {rep_unit}; the same shape marks the same "
            f"replicate in every group (circle, square, triangle: replicates 1, 2, 3); "
            f"line and error bar (beside the symbols), {ERR_TEXT[err]} of the replicate means.")


def rich_text(ax, x, y, parts, coords="axes fraction", ha="center", va="bottom", size=None, **kw):
    """Text mixing italic and upright pieces, e.g. gene names with digits.

    Mathtext cannot do this: digits in $\\mathit{NG13}$ stay upright and '::' is spaced
    as an operator. parts = [("FXR1", "italic"), (" (control)", "normal")].
    Returns the AnnotationBbox; its texts are seen by the audit.
    """
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    size = size or mpl.rcParams["axes.titlesize"]
    boxes = [TextArea(t, textprops=dict(fontstyle=st, fontsize=size, color=INK, **kw)) for t, st in parts]
    pack = HPacker(children=boxes, pad=0, sep=0, align="baseline")
    box_align = ({"left": 0, "center": 0.5, "right": 1}[ha], {"bottom": 0, "center": 0.5, "top": 1, "baseline": 0}[va])
    ab = AnnotationBbox(pack, (x, y), xycoords=coords, box_alignment=box_align, frameon=False, pad=0,
                        annotation_clip=False)
    ax.add_artist(ab)
    return ab


def gene_title(ax, name, suffix="", **kw):
    """Panel/small-multiple title with an italic gene name: gene_title(ax, "FXR1", " (control)")."""
    parts = [(name, "italic")] + ([(suffix, "normal")] if suffix else [])
    return rich_text(ax, 0.5, 1.0, parts, coords=("axes fraction", "axes fraction"), ha="center", va="bottom", **kw)


def log_axis(ax, which="x", base=10, numticks=5):
    """Log axis with 10^n ticks at 7 pt, so exponents (drawn at ~70%) stay >= 5 pt."""
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log", base=base)
    axis.set_major_locator(LogLocator(base=base, numticks=numticks))
    axis.set_major_formatter(LogFormatterMathtext(base=base))
    axis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=which, which="major", labelsize=min(journal().font_max, journal().font_min / 0.7))


def _t_two_sided_p(t, df):
    return float(_betai(df / 2, 0.5, df / (df + t * t)))


def welch_t(a, b):
    """Two-sided Welch t-test (unequal variances). Returns (t, P, description)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values: remove them explicitly and report the exclusion.")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return float(t), _t_two_sided_p(t, df), f"two-sided Welch t-test (df = {df:.1f})"


def paired_t(a, b):
    """Two-sided paired t-test on a - b (pairs with a missing member are excluded and counted)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("Paired samples need equal lengths (use NaN for a missing partner).")
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    n = len(d)
    if n < 2:
        raise ValueError("Fewer than 2 complete pairs.")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    drop = int((~keep).sum())
    desc = f"two-sided paired t-test (n = {n} pairs, df = {n - 1})" + (f"; {drop} incomplete pair(s) excluded" if drop else "")
    return float(t), _t_two_sided_p(t, n - 1), desc


def wilcoxon_signed_rank(a, b, max_exact=20):
    """Exact two-sided Wilcoxon signed-rank test for small paired samples (zero differences dropped, ties mid-ranked)."""
    import itertools

    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[keep]
    d = d[d != 0]
    n = len(d)
    if n == 0 or n > max_exact:
        raise ValueError(f"Exact test implemented for 1-{max_exact} non-zero pairs; got {n}.")
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    absd = np.abs(d)[order]
    i = 0
    while i < n:  # mid-ranks for ties
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    w = ranks[d > 0].sum()
    centre = ranks.sum() / 2
    obs = abs(w - centre)
    count = sum(1 for signs in itertools.product((0, 1), repeat=n) if abs(ranks[np.array(signs, bool)].sum() - centre) >= obs - 1e-9)
    p = count / 2**n
    return float(w), float(p), f"exact two-sided Wilcoxon signed-rank test (n = {n} non-zero pairs)"


def paired_dots(ax, labels, before, after, ids=None, colors=(None, None), n_unit="[patients?]", offset=0.25):
    """Paired samples: one grey line per subject joining its two values.

    A value whose partner is missing (NaN) is kept, drawn as an open symbol beside
    its group, and named in the returned legend sentence: never dropped silently.
    """
    a, b = np.asarray(before, float), np.asarray(after, float)
    if len(a) != len(b):
        raise ValueError("before and after must have one entry per subject (NaN when missing).")
    ids = list(ids) if ids is not None else [str(i + 1) for i in range(len(a))]
    ca = colors[0] or GREY
    cb = colors[1] or ACCENT
    full = ~(np.isnan(a) | np.isnan(b))
    for k in np.where(full)[0]:
        ax.plot([0, 1], [a[k], b[k]], color=LIGHT, lw=0.6, zorder=1)
    ax.scatter(np.zeros(full.sum()), a[full], s=12, color=ca, edgecolor=INK, linewidth=0.4, zorder=3)
    ax.scatter(np.ones(full.sum()), b[full], s=12, color=cb, edgecolor=INK, linewidth=0.4, zorder=3)
    lone = []
    for k in np.where(~full)[0]:
        for x, v, c in ((0, a[k], ca), (1, b[k], cb)):
            if not np.isnan(v):
                ax.scatter(x + (offset if x == 1 else -offset), v, s=12, facecolor="white", edgecolor=c, linewidth=0.8, zorder=3)
                lone.append(f"{ids[k]} ({labels[x]} only)")
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.5, 1.5)
    ax.tick_params(axis="x", length=0)
    out = f"Lines join the two samples of each of n = {int(full.sum())} {n_unit}."
    if lone:
        out += " Open symbols: samples without a partner, not included in paired statistics: " + ", ".join(lone) + "."
    return out


def bracket(ax, x1, x2, y, text, h_frac=0.02):
    """Significance bracket with an exact P value or a symbol defined in the legend.

    Text with super/subscripts is set at the size that keeps the script legible.
    """
    yr = ax.get_ylim()
    h = (yr[1] - yr[0]) * h_frac
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=0.5, clip_on=False)
    size = mpl.rcParams["xtick.labelsize"]
    if "$" in text:
        size = min(journal().font_max, max(size, journal().font_min / 0.7))
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=size)


def highlight_colors(categories, highlight, accent=ACCENT, other=LIGHT):
    hl = set(np.atleast_1d(highlight))
    return [accent if c in hl else other for c in categories]


def p_text(p):
    """Exact P value: P = 0.02, P = 0.003, P = 0.0003; below 0.0001, P = 2.1 × 10^-6.

    Plain decimals down to 0.0001 avoid small exponents; bracket() enlarges
    text that still needs one.
    """
    if p >= 0.0001:
        if p >= 0.01:
            return f"P = {p:.2f}"
        e = int(np.floor(np.log10(p)))
        return f"P = {round(p, -e):.{-e}f}"
    e = int(np.floor(np.log10(p)))
    m = p / 10**e
    return f"P = {m:.1f} × 10$^{{{e}}}$"


# --------------------------------------------------------------------------
# Colour-vision check
# --------------------------------------------------------------------------
_CVD = {
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb_lin):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ np.clip(rgb_lin, 0, 1) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _de(c1, c2, mode=None):
    a, b = _lin(mpl.colors.to_rgb(c1)), _lin(mpl.colors.to_rgb(c2))
    if mode:
        a, b = _CVD[mode] @ a, _CVD[mode] @ b
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def is_grey(c, tol=0.12):
    h, l, s = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return s < tol or l > 0.95 or l < 0.05


def cvd_report(colors, threshold=20.0):
    """Pairs of colours that become hard to tell apart under colour-vision deficiency."""
    colors = list(dict.fromkeys(mpl.colors.to_hex(c) for c in colors))
    bad = []
    for i in range(len(colors)):
        for k in range(i + 1, len(colors)):
            for mode in ("protan", "deutan", "tritan"):
                d = _de(colors[i], colors[k], mode)
                if d < threshold and _de(colors[i], colors[k]) >= threshold:
                    bad.append((colors[i], colors[k], mode, round(d, 1)))
    return bad


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def _numeric(s):
    s = s.replace("−", "-").replace("$", "").replace("{", "").replace("}", "").replace("^", "e").strip()
    s = s.replace("10e", "1e").replace("×", "")
    try:
        float(s)
        return True
    except ValueError:
        return bool(re.fullmatch(r"-?[\d.]+%?", s))


def _drawn_texts(fig):
    """Texts that are actually drawn, with their owning axes (or None)."""
    out = []
    skip = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            all_ticks = axis.get_major_ticks() + axis.get_minor_ticks()
            try:
                drawn = set(map(id, axis._update_ticks()))
            except Exception:
                drawn = set(map(id, all_ticks))
            for t in all_ticks:
                if id(t) not in drawn or not ax.axison or not axis.get_visible():
                    skip.add(id(t.label1))
                    skip.add(id(t.label2))
            if not ax.axison or not axis.get_visible():
                skip.add(id(axis.label))
                skip.add(id(axis.offsetText))
        if not ax.get_visible():
            for t in ax.findobj(Text):
                skip.add(id(t))
    owner = {}
    for ax in fig.axes:
        for t in ax.findobj(Text):
            owner[id(t)] = ax
    for t in fig.findobj(Text):
        if id(t) in skip or not t.get_visible() or not t.get_text().strip():
            continue
        p = t
        visible = True
        while p is not None:
            if hasattr(p, "get_visible") and not p.get_visible():
                visible = False
                break
            p = getattr(p, "_parent_figure", None) if p is not fig else None
        ax = owner.get(id(t))
        if visible:
            out.append((t, ax))
    return out


def _is_colorbar(ax):
    return hasattr(ax, "_colorbar") or ax.get_label() == "<colorbar>"


def audit(fig, allow_small=False, verbose=True, caption_words=None):
    """Check a figure against the journal and house rules.

    Returns a list of (level, message); level is ERROR, WARN or INFO.
    ERRORs must be fixed; WARNs need a reason to keep; INFO is for the
    final visual check.
    """
    j = journal()
    issues = []
    E = lambda m: issues.append(("ERROR", m))
    W = lambda m: issues.append(("WARN", m))
    I = lambda m: issues.append(("INFO", m))

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wmm, Hmm = fig.get_size_inches() / MM
    px_per_mm = fig.dpi * MM

    # 1. size
    std = {k: v for k, v in j.widths.items()}
    if not any(abs(Wmm - v) <= 1 for v in std.values()):
        W(f"Width {Wmm:.1f} mm is not a {j.name} column width {std}.")
    hmax, why = j.height_limit(Wmm, caption_words)
    if Hmm > hmax + 0.5:
        E(f"Height {Hmm:.1f} mm exceeds {j.name} maximum {hmax} mm" + (f" ({why})." if why else "."))
    elif why:
        I(f"Height limit {hmax} mm: {why}.")
    for u in j.unverified:
        I(f"Provisional {j.name} setting, not confirmed on the publisher's page: {u}.")

    # 2. text
    texts = _drawn_texts(fig)
    fmin = j.font_min if allow_small else max(j.font_min, HOUSE_FONT_MIN)
    fig_bb = fig.bbox
    boxes = []
    families = set()
    for t, ax in texts:
        s = t.get_text().strip()
        size = t.get_fontsize()
        label = t.get_gid() == "panel-label" or (len(s) == 1 and s.isalpha() and size >= 7.5 and ax is None)
        fam = t.get_fontname()
        families.add(fam)
        if label:
            if abs(size - j.label_size) > 0.1:
                W(f"Panel label '{s}' is {size:g} pt; {j.name} uses {j.label_size:g} pt.")
            want = s.lower() if j.label_case == "lower" else s.upper()
            if s != want:
                E(f"Panel label '{s}' should be '{want}' for {j.name}.")
        else:
            if size < fmin - 0.05:
                E(f"Text '{s[:40]}' is {size:g} pt (< {fmin:g} pt minimum).")
            if size > j.font_max + 0.05:
                W(f"Text '{s[:40]}' is {size:g} pt (> {j.font_max:g} pt {j.name} maximum).")
            if re.search(r"[\^_]", s) and "$" in s and size * 0.7 < j.font_min - 0.15:
                W(f"Text '{s[:40]}' has super/subscripts at ~{size * 0.7:.1f} pt (< {j.font_min:g} pt): enlarge to "
                  f"{j.font_min / 0.7:.1f} pt or avoid the script (e.g. 'log10' axis values).")
            alpha = t.get_alpha()
            rgba_a = mpl.colors.to_rgba(t.get_color())[3]
            if (alpha if alpha is not None else 1) * rgba_a < 0.6:
                W(f"Text '{s[:40]}' is faint (opacity {(alpha if alpha is not None else 1) * rgba_a:.2f}); text must be fully opaque.")
            if ax is not None and ax.get_rasterized():
                E(f"Text '{s[:40]}' belongs to rasterised axes: not editable and not checkable in the PDF. "
                  "Rasterise only dense data artists (rasterized=True on the scatter/image).")
            if t.get_rasterized():
                E(f"Text '{s[:40]}' is rasterised.")
            if not is_grey(t.get_color()):
                W(f"Coloured text '{s[:40]}' ({mpl.colors.to_hex(t.get_color())}): use black text with a colour key/swatch.")
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0 or bb.height == 0:
            continue
        if bb.x0 < fig_bb.x0 - 1 or bb.x1 > fig_bb.x1 + 1 or bb.y0 < fig_bb.y0 - 1 or bb.y1 > fig_bb.y1 + 1:
            E(f"Text '{s[:40]}' is cut off at the figure edge.")
        if t.get_rotation() not in (0, 0.0, 360) and abs(t.get_rotation() - 90) > 1:
            I(f"Text '{s[:30]}' is rotated {t.get_rotation():g} deg: consider horizontal bars or shorter labels.")
        boxes.append((s, bb.shrunk(0.92, 0.85) if bb.width > 2 and bb.height > 2 else bb))
    special = {f for f in families if f in ("Courier", "Courier New", "Symbol")}
    if special:
        I(f"{sorted(special)} in use: allowed for sequences (Courier) and Greek (Symbol) only.")
    bad_fams = {f for f in families if f not in ("Arial", "Helvetica", "Helvetica Neue", "Liberation Sans")} - special
    if bad_fams:
        E(f"Non-standard fonts in use: {sorted(bad_fams)} (use Arial or Helvetica; is Arial installed?).")
    # overlaps
    n_over = 0
    for i in range(len(boxes)):
        for k in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[k][1]):
                n_over += 1
                if n_over <= 15:
                    E(f"Overlapping text: '{boxes[i][0][:30]}' and '{boxes[k][0][:30]}'.")
    if n_over > 15:
        E(f"... {n_over - 15} more overlapping text pairs.")

    # 3. axes
    colors_used = []
    for idx, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        name = f"axes {idx + 1}" + (f" ('{ax.get_title()[:25]}')" if ax.get_title() else "")
        if _is_colorbar(ax):
            if not (ax.get_ylabel() or ax.get_xlabel() or ax.get_title()):
                E(f"Colour bar in {name} has no label (quantity and unit).")
            continue
        has_data = bool(ax.lines or ax.collections or ax.images or len(ax.patches) > 0)
        if not has_data or not ax.axison:
            continue
        for axis, getlab, which in ((ax.xaxis, ax.get_xlabel, "x"), (ax.yaxis, ax.get_ylabel, "y")):
            if not axis.get_visible():
                continue
            try:
                ticks = axis._update_ticks()
            except Exception:
                ticks = axis.get_major_ticks()
            labs = [t.label1.get_text() for t in ticks if t.label1.get_visible() and t.label1.get_text()]
            numeric = labs and sum(_numeric(x) for x in labs) >= max(1, len(labs) // 2)
            if numeric and not getlab().strip():
                shared = ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
                siblings = [a for a in shared.get_siblings(ax) if a is not ax]
                sup = getattr(fig, "_supxlabel" if which == "x" else "_supylabel", None)
                has_sup = sup is not None and sup.get_visible() and sup.get_text().strip()
                if not has_sup and not any((a.get_xlabel() if which == "x" else a.get_ylabel()).strip() for a in siblings):
                    E(f"{name}: numeric {which} axis has no label (quantity and unit).")
            if any(re.search(r"\de[+-]\d", x) for x in labs):
                E(f"{name}: {which} tick labels in computer notation ('1e+02'); use 10^2 ticks (LogFormatterMathtext).")
            off = axis.get_offset_text().get_text()
            if off.strip():
                W(f"{name}: {which} axis shows an offset/scale '{off}'; put the factor in the axis label instead.")
            if numeric and getlab().strip() and "(" not in getlab() and "[" not in getlab():
                I(f"{name}: {which} label '{getlab()}' has no unit in parentheses; fine only if unitless.")
        if any(gl.get_visible() for gl in ax.get_xgridlines() + ax.get_ygridlines()):
            W(f"{name}: background gridlines (Nature: avoid).")
        # legend
        labelled = ax.get_legend_handles_labels()[1]
        direct = [t for t in ax.texts if t.get_visible() and t.get_text().strip() and t.get_gid() != "panel-label"]
        if len(labelled) >= 2 and ax.get_legend() is None and not fig.legends:
            if len(direct) >= len(labelled):
                I(f"{name}: {len(labelled)} series, no legend; check that the {len(direct)} direct labels name each one.")
            else:
                W(f"{name}: {len(labelled)} labelled series but no legend or direct labels.")
        # legend or annotation text lying on top of data
        data_boxes = []
        for p in ax.patches:
            if p.get_visible():
                bb = p.get_window_extent(r)
                if bb.width > 0.5 and bb.height > 0.5:
                    data_boxes.append(bb)
        data_pts = []
        for ln in ax.lines:
            if ln.get_visible() and len(ln.get_xydata()):
                xy = ln.get_xydata()
                if len(xy) > 1:  # densify segments so a line crossing a box is seen
                    t = np.linspace(0, 1, 8)[:, None, None]
                    xy = (xy[:-1] * (1 - t) + xy[1:] * t).reshape(-1, 2)
                data_pts.append(ax.transData.transform(xy))
        for c in ax.collections:
            off = c.get_offsets()
            if c.get_visible() and len(off) and len(off) < 20000:
                data_pts.append(c.get_offset_transform().transform(off))
        pts = np.vstack(data_pts) if data_pts else np.empty((0, 2))
        covers = []
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            covers.append(("Legend", leg.get_window_extent(r)))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip() and t.get_gid() != "panel-label":
                covers.append((f"Text '{t.get_text()[:25]}'", t.get_window_extent(r)))
        for what, cb in covers:
            hit = any(cb.overlaps(db) for db in data_boxes) if what == "Legend" else False
            if len(pts):
                inside = (pts[:, 0] > cb.x0) & (pts[:, 0] < cb.x1) & (pts[:, 1] > cb.y0) & (pts[:, 1] < cb.y1)
                hit = hit or bool(inside.any())
            if hit:
                W(f"{name}: {what} lies on top of data; move it to empty space, outside the axes, or label directly.")
        for axis, which in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            labs = [t.label1.get_text() for t in axis.get_major_ticks() if t.label1.get_text()]
            if len(labs) >= 3 and all(re.fullmatch(r"\u2212?-?\d+\.0", x) for x in labs):
                W(f"{name}: {which} ticks all end in '.0'; use integer ticks.")
            elif len(labs) >= 3 and any(re.fullmatch(r"\u2212?-?\d+\.5", x) for x in labs) and max(
                    abs(float(x.replace("\u2212", "-"))) for x in labs if _numeric(x)) >= 10:
                W(f"{name}: {which} ticks at half-units ({', '.join(labs)}); round ticks (e.g. 0, 5, 10) read faster.")
        # data hidden by axis limits
        n_out, n_all = 0, 0
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        sources = [ln.get_xydata() for ln in ax.lines if ln.get_visible() and ln.get_clip_on()]
        sources += [np.asarray(c.get_offsets()) for c in ax.collections
                    if c.get_visible() and c.get_clip_on() and c.get_offset_transform() == ax.transData]
        for xy in sources:
            xy = np.asarray(xy, float)
            if xy.ndim != 2 or not len(xy):
                continue
            xy = xy[np.isfinite(xy).all(axis=1)]
            tol_x, tol_y = (x1 - x0) * 1e-6, (y1 - y0) * 1e-6
            out = (xy[:, 0] < x0 - tol_x) | (xy[:, 0] > x1 + tol_x) | (xy[:, 1] < y0 - tol_y) | (xy[:, 1] > y1 + tol_y)
            n_out += int(out.sum())
            n_all += len(xy)
        if n_out:
            E(f"{name}: {n_out} of {n_all} data points lie outside the axis limits and are not shown. Widen the "
              "limits, or show an axis break and state the hidden values in the legend.")
        # colours, lines
        for ln in ax.lines:
            if ln.get_linewidth() < 0.25 and ln.get_linestyle() not in ("None", ""):
                W(f"{name}: line width {ln.get_linewidth():g} pt is below 0.25 pt and may vanish in print.")
            colors_used.append(ln.get_color())
        for p in ax.patches:
            if p.get_visible() and p.get_facecolor()[3] > 0:
                colors_used.append(p.get_facecolor())
        for c in ax.collections:
            fc = c.get_facecolors()
            if c.get_array() is None and 0 < len(fc) <= 50:  # colour-mapped data is checked via its cmap
                colors_used.extend(map(tuple, fc))
            cm = getattr(c, "cmap", None)
            if c.get_array() is not None and cm is not None and cm.name.replace("_r", "") in FORBIDDEN_CMAPS:
                E(f"{name}: rainbow colour map '{cm.name}' (perceptually misleading, not colour-blind safe).")
        for im in ax.images:
            if im.cmap.name.replace("_r", "") in FORBIDDEN_CMAPS:
                E(f"{name}: rainbow colour map '{im.cmap.name}'.")
    hues = list(dict.fromkeys(mpl.colors.to_hex(c) for c in colors_used if not is_grey(c)))
    if len(hues) > 6:
        W(f"{len(hues)} distinct colours in use; use colour only where it encodes something (grey for the rest).")
    for c1, c2, mode, d in cvd_report(hues):
        W(f"Colours {c1} and {c2} are hard to distinguish with {mode}opia (dE={d}); change one or add shape/label.")

    # 4. white space
    tb = fig.get_tightbbox(r)  # inches
    left, bottom = tb.x0 / MM, tb.y0 / MM
    right, top = Wmm - tb.x1 / MM, Hmm - tb.y1 / MM
    for side, v in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        if v > 2.5:
            W(f"{v:.1f} mm of empty margin on the {side}: reduce the figure height/width or enlarge the panels.")
    covered = 0
    for ax in fig.axes:
        if ax.get_visible():
            b = ax.get_tightbbox(r)
            if b is not None:
                covered += b.width * b.height
    frac = covered / (fig_bb.width * fig_bb.height)
    I(f"Axes (with labels) cover ~{min(frac, 1) * 100:.0f}% of the figure area; look for empty gaps between panels.")

    if verbose:
        print(format_report(issues, fig))
    return issues


COVERAGE = ("Mechanical checks only: size, text size/font/opacity/overlap/clipping, axis labels and notation, "
            "legends on data, hidden data, colour-vision pairs, rainbow maps, margins. NOT checked: whether the "
            "claim matches the data, statistical design (what n is, pairing, nesting), colour meaning across "
            "figures, image integrity, content inside raster images. 0 errors = a checked draft, not a "
            "submission-ready figure.")


def format_report(issues, fig=None):
    j = journal()
    head = f"Figure audit ({j.name})"
    if fig is not None:
        w, h = fig.get_size_inches() / MM
        head += f": {w:.0f} x {h:.0f} mm"
    lines = [head]
    for level in ("ERROR", "WARN", "INFO"):
        for lv, m in issues:
            if lv == level:
                lines.append(f"  {lv:5s} {m}")
    n_e = sum(1 for lv, _ in issues if lv == "ERROR")
    n_w = sum(1 for lv, _ in issues if lv == "WARN")
    lines.append(f"  -> {n_e} error(s), {n_w} warning(s)")
    lines.append("  " + COVERAGE)
    return "\n".join(lines)


def check_pdf_fonts(pdf_path):
    """Use poppler's pdffonts to confirm embedded TrueType/Type 1 (no Type 3)."""
    if not shutil.which("pdffonts"):
        return [("INFO", "pdffonts not found; font embedding not checked.")]
    out = subprocess.run(["pdffonts", pdf_path], capture_output=True, text=True).stdout.splitlines()[2:]
    issues = []
    for line in out:
        if "Type 3" in line:
            issues.append(("ERROR", f"Type 3 font in PDF: {line.split()[0]} (set pdf.fonttype 42)."))
        parts = line.split()
        if len(parts) >= 5 and "no" in parts[-5:-3]:
            issues.append(("ERROR", f"Font not embedded: {parts[0]}."))
    return issues


def synthetic_stamp(fig, text="SYNTHETIC DATA: layout test, not results"):
    """Visible mark required on any figure made from invented or simulated data."""
    # a suptitle gets its own reserved space in constrained layout, so it never covers data
    t = fig.suptitle(text, x=1.0, ha="right", fontsize=6, color=INK, fontstyle="italic")
    t.set_gid("synthetic-stamp")
    return t


def save(fig, path, formats=("pdf", "svg", "png"), allow_small=False, strict=True, png_dpi=600,
         legend=None, confirm=(), caption_words=None, synthetic=False, final=False):
    """Audit, save the figure, and write <path>_handover.md.

    legend: the legend draft (text). confirm: list of points to confirm.
    synthetic=True stamps the figure visibly. Status is 'checked draft' unless
    final=True, there are no ERRORs, no '[...?]' placeholders remain in the
    legend, nothing is left to confirm and the data are not synthetic.
    With strict=True, ERRORs raise after writing the files for inspection.
    """
    if synthetic and not (fig._suptitle is not None and fig._suptitle.get_gid() == "synthetic-stamp"):
        synthetic_stamp(fig)
    if legend is not None and caption_words is None:
        caption_words = len(legend.split())
    issues = audit(fig, allow_small=allow_small, verbose=True, caption_words=caption_words)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=png_dpi if fmt == "png" else None)
    if "pdf" in formats:
        issues += check_pdf_fonts(f"{path}.pdf")
    with open(f"{path}_audit.txt", "w") as f:
        f.write(format_report(issues, fig) + "\n")
    errs = [m for lv, m in issues if lv == "ERROR"]
    placeholders = re.findall(r"\[[^\]]*\?\]", legend or "")
    blockers = []
    if errs:
        blockers.append(f"{len(errs)} audit error(s)")
    if legend is None:
        blockers.append("no legend written")
    if placeholders:
        blockers.append("legend placeholders " + ", ".join(sorted(set(placeholders))))
    if confirm:
        blockers.append(f"{len(confirm)} point(s) to confirm")
    if synthetic:
        blockers.append("synthetic data")
    if not final:
        blockers.append("not marked final (final=False)")
    status = "READY FOR SUBMISSION" if not blockers else "CHECKED DRAFT, not ready: " + "; ".join(blockers)
    warns = [m for lv, m in issues if lv == "WARN"]
    with open(f"{path}_handover.md", "w") as f:
        f.write(f"# {os.path.basename(path)}\n\nStatus: {status}\n\n")
        f.write("Files: " + ", ".join(f"{os.path.basename(path)}.{x}" for x in formats) + "\n\n")
        f.write("## Legend draft\n\n" + (legend or "(not written)") + "\n\n")
        f.write("## Points to confirm\n\n" + ("\n".join(f"- {c}" for c in confirm) or "(none listed)") + "\n\n")
        f.write("## Audit\n\n" + f"{len(errs)} error(s), {len(warns)} warning(s)\n\n")
        f.write("\n".join(f"- WARN {w}" for w in warns) + ("\n\n" if warns else ""))
        f.write(COVERAGE + "\n")
    print(f"Status: {status}")
    if errs and strict:
        raise RuntimeError(f"{len(errs)} audit error(s); files written for inspection but the figure is not ready:\n- " + "\n- ".join(errs))
    return issues


use("nature")
