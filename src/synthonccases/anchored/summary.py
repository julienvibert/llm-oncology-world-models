"""Survival summaries for the evidence block, computed the way a trial reports them.

When individual neighbours are shown, descriptive statistics beside them are a convenience:
the model can read the rows and see the censoring for itself. When the summary IS the
evidence, the same statistics become the claim, and they have to survive the scrutiny a
published estimate would get. Three things change:

* **Kaplan-Meier, not the median of the progressed.** "Progressed in 29 of 30, median 1.6
  months" is the median of the uncensored subset, which is biased short by exactly the
  patients who were doing well when we lost sight of them.
* **An interval, not a point.** Thirty patients and three hundred give the same median and
  very different evidence, and the model cannot tell which it is holding unless the width is
  stated.
* **Something to say when the median is not reached.** A cohort that has not halved by the
  end of follow-up has no median, and inventing one by extrapolation would be the single
  most misleading number in the block. Survival at fixed horizons and the restricted mean
  are reported instead, with the horizon named.

Stratification is the other half. A hard clinical match leaves most of the variation
standing -- measured on this corpus, 88% of the PFS spread survives matching on cancer,
stage, line and regimen -- so a marginal summary of a matched cohort is close to a subgroup
rate. What makes a forecast personal is where the query sits INSIDE that cohort, which is
what :func:`choose_strata` and :func:`stratify` expose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Horizons reported when the median is not reached, and for n-at-risk. Months.
HORIZONS: tuple[float, ...] = (6.0, 12.0)
#: Restricted-mean horizon. Fixed rather than data-dependent: an RMST whose horizon moves
#: with the cohort is not comparable between cohorts, which is the only thing it is for.
RMST_HORIZON = 24.0
#: Below this a stratum is not reported. A KM median from four patients is a number with no
#: information in it, and printing it invites the model to use it anyway.
MIN_STRATUM = 5


@dataclass(frozen=True, slots=True)
class Survival:
    """What a group of patients did, with the uncertainty attached."""

    n: int
    n_events: int
    median: float | None = None
    median_lo: float | None = None
    median_hi: float | None = None
    at_horizon: dict[float, float] = field(default_factory=dict)
    """Survival probability at each horizon."""
    n_at_risk: dict[float, int] = field(default_factory=dict)
    rmst: float | None = None

    @property
    def median_reached(self) -> bool:
        return self.median is not None and np.isfinite(self.median)


def km_summary(times: Sequence[float], events: Sequence[float]) -> Survival:
    """Kaplan-Meier with a confidence interval, n at risk, and a fallback for no median."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times, restricted_mean_survival_time

    t = pd.to_numeric(pd.Series(list(times)), errors="coerce")
    e = pd.to_numeric(pd.Series(list(events)), errors="coerce")
    ok = t.notna() & e.notna() & (t > 0)
    t, e = t[ok].to_numpy(float), e[ok].to_numpy(float)
    if len(t) == 0:
        return Survival(n=0, n_events=0)

    kmf = KaplanMeierFitter().fit(t, e)
    med = float(kmf.median_survival_time_)
    lo = hi = None
    try:
        ci = median_survival_times(kmf.confidence_interval_)
        lo, hi = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
    except (IndexError, ValueError):
        pass
    horizon = {h: float(kmf.predict(h)) for h in HORIZONS}
    at_risk = {h: int((t >= h).sum()) for h in HORIZONS}
    rmst = None
    if len(t) > 1:
        try:
            rmst = float(restricted_mean_survival_time(kmf, t=RMST_HORIZON))
        except (ValueError, ZeroDivisionError):
            rmst = None
    return Survival(n=int(len(t)), n_events=int((e == 1).sum()),
                    median=None if not np.isfinite(med) else med,
                    median_lo=None if lo is None or not np.isfinite(lo) else lo,
                    median_hi=None if hi is None or not np.isfinite(hi) else hi,
                    at_horizon=horizon, n_at_risk=at_risk, rmst=rmst)


def format_km(s: Survival, *, label: str = "") -> str:
    """One clinical sentence. Says what is not known as plainly as what is."""
    if s.n == 0:
        return f"{label}: no patients." if label else "No patients."
    head = f"{label} (n={s.n}, {s.n_events} progressed)" if label else \
        f"n={s.n}, {s.n_events} progressed"
    if s.median_reached:
        ci = ""
        if s.median_lo is not None and s.median_hi is not None:
            ci = f", 95% CI {s.median_lo:.1f}–{s.median_hi:.1f}"
        body = f"median {s.median:.1f} months{ci}"
    else:
        # NEVER extrapolate a median. Say it was not reached and give what was observed.
        body = ("median not reached by the end of follow-up; "
                + ", ".join(f"{p:.0%} progression-free at {h:.0f} months "
                            f"({s.n_at_risk.get(h, 0)} still at risk)"
                            for h, p in s.at_horizon.items()))
    if s.rmst is not None:
        body += (f"; mean progression-free time over the first {RMST_HORIZON:.0f} months "
                 f"{s.rmst:.1f}")
    return f"{head}: {body}."


# ------------------------------------------------------------------ stratification --------
def choose_strata(residual: Mapping[str, float], *, allowed: Sequence[str],
                  weights: Mapping[str, float] | None = None, k: int = 3) -> list[str]:
    """Which axes to split the evidence on, chosen deterministically and in advance.

    Ranked by |residual| x prognostic weight: how far the query sits from the cohort centre
    on that axis, times how much that axis matters for the outcome. The weights come from a
    fit on TRAINING folds only and are handed in; nothing here reads the query's outcome,
    and the ranking is a pure function of the retrieval, so it replays exactly.

    Ties broken by name, not by dictionary order, so the choice does not depend on how the
    residual happened to be built.
    """
    w = dict(weights or {})
    scored = [(abs(float(v)) * abs(float(w.get(c, 1.0))), c)
              for c, v in residual.items() if c in set(allowed)]
    scored.sort(key=lambda p: (-p[0], p[1]))
    return [c for score, c in scored[:k] if score > 0]


def stratify(rows: pd.DataFrame, query: Mapping[str, float], axis: str,
             *, cut: float | None = None) -> list[tuple[str, np.ndarray, bool]]:
    """Split an evidence set on one axis into (label, mask, contains-the-query).

    One axis at a time, never a cross-tabulation: thirty patients cut four ways gives eight
    per cell, and a Kaplan-Meier median from eight patients is a number with no information
    in it. A clinician reads subgroup tables one row at a time for the same reason.
    """
    v = pd.to_numeric(rows[axis], errors="coerce")
    qv = float(query[axis])
    binary = set(v.dropna().unique()) <= {0.0, 1.0}
    threshold = 0.5 if binary else (float(v.median()) if cut is None else cut)
    hi = (v >= threshold).to_numpy()
    if binary:
        labels = (f"{axis} present", f"{axis} absent")
    else:
        labels = (f"{axis} at or above {threshold:.1f}", f"{axis} below {threshold:.1f}")
    q_hi = qv >= threshold
    return [(labels[0], hi, bool(q_hi)), (labels[1], ~hi, bool(not q_hi))]
