"""Shared constructed patient course for the Figure 1 panels (f1a_history, f1b_world_model,
f1c_dynamics, f1d_whole_course). Constructed illustration, no data and no model output.

One constructed patient (metastatic EGFR-mutant lung adenocarcinoma) is followed in an
illustrative two-dimensional state space: x, disease burden (0 none, larger = more disease);
y, physical condition (0 fully active, larger = poorer). Both are toy scales on the same
orientation as the paper's state map (PC1 towards more disease burden, PC2 towards poorer
condition), not measured quantities.

Disease state at each 3-monthly assessment is derived from burden with a RECIST-like rule
(response: >=30% below the reference; progression: >=20% and >=0.3 units above the nadir;
no evidence of disease: burden below 0.15; otherwise stable), so that state labels are read from
the course, not from the position of a region on the plot. Death is an absorbing exit, drawn as a
terminal symbol at the end of a course.
"""

from __future__ import annotations

import numpy as np

from common import INK, PAL, STATE_COLOURS

HIST = "#882255"            # recorded history (wine), as in the paper's earlier schematics
STRAT = {"A": PAL["osimertinib"], "B": PAL["chemotherapy"]}   # strategy A blue, B purple (schematic)
# displayed names: "A" is the treatment input in the C, X, H, A, E notation, so strategies show as 1 and 2
SNAME = {"A": "1", "B": "2"}
EXAMPLES = "#8C6D31"        # optional patient examples
XLIM, YLIM = (0.0, 9.0), (0.3, 3.85)
ANCHOR = np.array([4.1, 1.55])   # current state at the decision (month 16 for patient 1)
NED_B = 0.15

XLABEL = "State dimension 1: more disease  →"
YLABEL = "State dimension 2:\npoorer condition  →"


def assess(burden: np.ndarray, ref: float, nadir0: float | None = None) -> list[str]:
    """RECIST-like disease state at successive assessments (burden values), from a reference."""
    out, nadir = [], (ref if nadir0 is None else nadir0)
    for b in burden:
        if b < NED_B:
            s = "no evidence of disease"
        elif b >= 1.2 * nadir and b - nadir >= 0.3:
            s = "progression"
        elif b <= 0.7 * ref:
            s = "response"
        else:
            s = "stable"
        nadir = min(nadir, b)
        out.append(s)
    return out


# --- recorded histories (knots; smooth curve through them) --------------------------------------
# patient 1: bulky symptomatic disease at diagnosis, osimertinib from month 0; symptoms improve first,
# then deep durable response; slow regrowth from month 11, progression documented at month 15,
# decision at month 16. The return path lies above the outward path (condition worse at equal burden).
P1 = {"months": [0, 1, 2, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, 16],
      "burden": [6.8, 6.3, 5.1, 3.8, 2.6, 1.8, 1.35, 1.15, 1.1, 1.25, 1.9, 3.0, 4.1],
      "cond": [1.75, 1.45, 1.1, 0.8, 0.6, 0.5, 0.47, 0.47, 0.5, 0.62, 0.85, 1.2, 1.55],
      "tx": [(0, 16, "Osimertinib")]}
# patient 2: limited disease at diagnosis, primary resistance with clinical deterioration, platinum
# chemotherapy from month 4, partial response at month 9, at the same displayed state at month 10
P2 = {"months": [0, 1.5, 3, 4, 5, 6, 7.5, 9, 10],
      "burden": [1.9, 2.9, 4.4, 5.9, 6.0, 5.7, 5.0, 4.05, 4.1],
      "cond": [1.5, 2.2, 2.9, 3.3, 3.25, 2.9, 2.25, 1.6, 1.55],
      "tx": [(0, 4, "Osimertinib"), (4, 10, "Platinum chemotherapy")]}


def smooth(knots_t, knots_v, t):
    """Monotone-ish smooth interpolation (cubic Hermite with Catmull-Rom tangents)."""
    kt, kv = np.asarray(knots_t, float), np.asarray(knots_v, float)
    m = np.gradient(kv, kt)
    i = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    h = kt[i + 1] - kt[i]
    s = (t - kt[i]) / h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * kv[i] + h10 * h * m[i] + h01 * kv[i + 1] + h11 * h * m[i + 1]


def history_curve(p: dict, n: int = 240):
    t = np.linspace(p["months"][0], p["months"][-1], n)
    return t, smooth(p["months"], p["burden"], t), smooth(p["months"], p["cond"], t)


def history_assessments(p: dict):
    """3-monthly assessments of a recorded history: (months, burden, cond, states). The reference
    and nadir are reset at the start of each treatment line."""
    end = p["months"][-1]
    m = np.arange(3, end + 1e-9, 3.0)
    b = smooth(p["months"], p["burden"], m)
    c = smooth(p["months"], p["cond"], m)
    states = []
    for start, stop, _ in p["tx"]:
        sel = (m > start) & (m <= stop + 1e-9)
        if sel.any():
            ref = float(smooth(p["months"], p["burden"], np.array([float(start)]))[0])
            states += assess(b[sel], ref=ref)
    return m, b, c, states


# --- possible courses after the decision ---------------------------------------------------------
def _softplus(x, k=1.5):
    return np.log1p(np.exp(k * x)) / k


def simulate_course(strategy: str, rng: np.random.Generator, horizon: float = 24.0, step: float = 0.1,
                    kind: str = "typical"):
    """One constructed future course from ANCHOR under strategy A or B (smooth toy dynamics).

    Burden: shrinks towards a nadir, holds, then regrows after a resistance time. Condition: follows
    burden weakly, plus a treatment-toxicity bump (larger under A), plus accelerating decline when
    disease becomes extensive. Death (absorbing) when condition crosses a course-specific threshold.
    A typical: deep shrinkage with toxicity; A early: early failure. B typical: modest shrinkage,
    little toxicity, earlier regrowth; B good: deeper, durable shrinkage.
    """
    b0, c0 = ANCHOR
    if strategy == "A":
        depth = {"typical": rng.uniform(0.22, 0.42), "early": rng.uniform(0.85, 0.95)}[kind]
        T = {"typical": rng.uniform(8, 19), "early": rng.uniform(1.0, 2.5)}[kind]
        g = rng.uniform(0.13, 0.2) if kind == "typical" else rng.uniform(0.2, 0.26)
        tox = rng.uniform(0.55, 1.0)
    else:
        depth = {"typical": rng.uniform(0.72, 0.92), "good": rng.uniform(0.4, 0.55)}[kind]
        T = {"typical": rng.uniform(5, 10), "good": rng.uniform(10, 16)}[kind]
        g = rng.uniform(0.05, 0.10)
        tox = rng.uniform(0.05, 0.2)
    tau = rng.uniform(1.6, 2.6)
    t = np.arange(0, horizon + 1e-9, step)
    bn = b0 * depth
    b = (bn + (b0 - bn) * np.exp(-t / tau)) * np.exp(g * _softplus(t - T))
    tp = rng.uniform(2.0, 3.0)
    tox_t = tox * (t / tp) * np.exp(1 - t / tp)
    wob = 0.06 * np.sin(t / rng.uniform(2.5, 4.0) + rng.uniform(0, 6))
    bcrit = rng.uniform(5.6, 7.6)
    shift = rng.uniform(-0.35, 0.55) * (1 - np.exp(-t / 6))     # lasting change in function, per course
    c = c0 + shift + 0.16 * (b - b0) + tox_t + wob + 0.45 * np.exp(np.minimum(b - bcrit, 8) / 0.9) - 0.45 * np.exp((b0 - bcrit) / 0.9)
    # every course leaves exactly from the shared decision state: remove the small offsets at t = 0
    b = b * (b0 / b[0])
    c = c - (c[0] - c0)
    c = np.maximum(c, 0.3)
    thr = rng.uniform(3.2, 3.7)
    died = bool(np.any(c >= thr))
    if died:
        j = int(np.argmax(c >= thr))
        t, b, c = t[: j + 1], b[: j + 1], c[: j + 1]
    return {"t": t, "burden": b, "cond": c, "died": died, "death_t": float(t[-1]) if died else None,
            "strategy": strategy, "kind": kind}


def fan(strategy: str, kinds: list[str], seed: int, horizon: float = 24.0) -> list[dict]:
    """A fan of constructed courses with a declared mix of course types."""
    rng = np.random.default_rng(seed)
    return [simulate_course(strategy, rng, horizon=horizon, kind=k) for k in kinds]


FAN_KINDS = {"A": ["early", "typical", "typical", "typical", "typical", "typical", "typical", "early",
                   "typical", "typical"],
             "B": ["typical", "good", "typical", "typical", "typical", "good", "typical", "typical",
                   "typical", "typical"]}


def course_assessments(course: dict):
    """3-monthly assessments of a simulated course (reference = burden at the decision)."""
    t = course["t"]
    m = np.arange(3, t[-1] + 1e-9, 3.0)
    if course["died"]:
        m = m[m < t[-1] - 0.4]
    b = np.interp(m, t, course["burden"])
    c = np.interp(m, t, course["cond"])
    return m, b, c, assess(b, ref=float(ANCHOR[0]))


# --- drawing helpers ------------------------------------------------------------------------------
def state_dots(ax, x, y, states, ms=2.6, z=6, mew=0.35):
    for xi, yi, s in zip(x, y, states):
        ax.plot(xi, yi, "o", ms=ms, mfc=STATE_COLOURS[s], mec="white", mew=mew, zorder=z)


def death_mark(ax, x, y, ms=4.2, z=7, mew=1.1):
    ax.plot(x, y, marker="x", ms=ms, mew=mew + 0.9, color="white", zorder=z)
    ax.plot(x, y, marker="x", ms=ms, mew=mew, color=INK, zorder=z + 0.1)


def style_state_axes(ax, xlabel=True, ylabel=True, fs=6.5):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(XLABEL, fontsize=fs, labelpad=2)
    if ylabel:
        ax.set_ylabel(YLABEL, fontsize=fs, labelpad=2)


STATE_KEY = [("no evidence of disease", "No evidence of disease"), ("response", "Response"),
             ("stable", "Stable disease"), ("progression", "Progression")]


# --- continuous disease-state bands and events (one response, then maintained) ---
NOT_ASSESSED = "unresolved"     # STATE_COLOURS key used for the interval before the first assessment


def state_bands(months, states, start: float, end: float) -> list[tuple[float, float, str]]:
    """Continuous disease-state band: the last documented state is carried forward until the next
    assessment; before the first assessment the state is 'not yet assessed'. Consecutive equal states merge."""
    segs = [(start, float(months[0]) if len(months) else end, NOT_ASSESSED)]
    for i, (m, s) in enumerate(zip(months, states)):
        b = float(months[i + 1]) if i + 1 < len(months) else end
        if b > m:
            segs.append((float(m), b, s))
    out: list[tuple[float, float, str]] = []
    for a, b, s in segs:
        if out and out[-1][2] == s:
            out[-1] = (out[-1][0], b, s)
        elif b > a:
            out.append((a, b, s))
    return out


def first_events(months, states) -> dict[str, float]:
    """Month of the first documented response and of the first progression (each at most once)."""
    ev: dict[str, float] = {}
    for m, s in zip(months, states):
        if s in ("response", "no evidence of disease") and "response" not in ev:
            ev["response"] = float(m)
        if s == "progression" and "progression" not in ev:
            ev["progression"] = float(m)
    return ev
