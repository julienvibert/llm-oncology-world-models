"""Deterministic anchor retrieval. No LLM, no network, no randomness beyond a stated seed.

Step one of two. This module decides WHICH real patients stand behind a generation; nothing
here writes a prompt, and nothing here calls a model. Keeping the two apart means the
retrieval can be tested exhaustively, replayed exactly, and swapped for a different rule
without touching generation.

Three properties are enforced rather than documented, because each of them has silently
broken a measurement in this project before:

* **A patient is never their own anchor.** Several lines belong to one person and their own
  later line agrees with them for reasons that have nothing to do with the covariates.
* **The query's outcome is never read.** Retrieval sees covariates only. The corpus's
  outcomes are used to fit a supervised metric ONLY when that metric was fitted elsewhere on
  training rows and handed in; this module never fits on rows it is about to retrieve for.
* **Refusal is a first-class result.** A query with too little support returns an empty,
  labelled result rather than a short anchor set. A five-anchor neighbourhood scores worse
  than predicting the base rate, so silently degrading is worse than declining.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .policy import AnchorPolicy, RetrievalResult, SupportLabel


class Corpus:
    """Real patients available as anchors, in one frozen, indexable structure.

    Holds covariates, identifiers, actions and outcomes side by side so retrieval and
    rendering read the same rows, and so the outcome columns can be named explicitly and
    kept out of the distance by construction.
    """

    def __init__(self, frame: pd.DataFrame, *, channels: list[str], subject_col: str,
                 action_col: str, outcome_cols: list[str] | None = None,
                 weights: np.ndarray | None = None,
                 unobserved: list[str] | None = None,
                 outcome_time_col: str | None = None,
                 outcome_event_col: str | None = None,
                 extra_outcomes: list[tuple[str, str, str]] | None = None,
                 standardize: bool = True,
                 observation_flags: dict[str, tuple[str, ...]] | None = None,
                 prognostic_weights: dict[str, float] | None = None) -> None:
        missing = [c for c in (*channels, subject_col, action_col) if c not in frame.columns]
        if missing:
            raise KeyError(f"corpus is missing columns: {missing}")
        overlap = set(channels) & set(outcome_cols or [])
        if overlap:
            raise ValueError(f"outcome columns cannot also be channels: {sorted(overlap)}")
        self.frame = frame.reset_index(drop=True)
        self.channels = list(channels)
        self.subject_col = subject_col
        self.action_col = action_col
        self.outcome_cols = list(outcome_cols or [])
        # Channels the SOURCE never records, as opposed to channels that are genuinely zero
        # for these patients. Rendering must never present the two the same way: "no CNS
        # disease" and "CNS involvement was not assessed" are different clinical statements
        # and only one of them is evidence.
        self.unobserved = list(unobserved or [])
        # A time-to-event outcome is TWO columns that mean nothing apart. Rendered as raw
        # numbers, "1.38 / 0.00" reads as "progressed at 1.4 months" when it means "still
        # progression-free when we lost sight of them" — which would bias every anchor set
        # short. Declaring them lets the renderer state censoring in words.
        self.outcome_time_col = outcome_time_col
        self.outcome_event_col = outcome_event_col
        # Further (label, time_col, event_col) triples rendered beside the primary outcome.
        # A generator asked for a whole realistic course needs to see what happened AFTER
        # progression -- whether the patient started another line, and how long they lived --
        # or it is calibrated only on the first segment of every trajectory.
        self.extra_outcomes = list(extra_outcomes or [])
        # {indicator channel: channels it governs}. A channel whose indicator is 0 was NOT
        # ASSESSED for that patient. Rendering says so instead of printing a number, and
        # the metric below neutralises the cell rather than treating the fill value as an
        # observation -- otherwise every unscanned patient looks like a patient whose scan
        # was clean, which is a different and much better patient.
        self.observation_flags = dict(observation_flags or {})
        # How much each channel matters for the OUTCOME, fitted on training folds elsewhere
        # and handed in. Used only to choose which axes an evidence set is stratified on --
        # never to weight the distance, which is `weights`. Two different jobs, kept apart
        # because conflating them would make the metric depend on the display.
        self.prognostic_weights = dict(prognostic_weights or {})
        raw = self.frame[self.channels].to_numpy(float)
        # STANDARDISE BEFORE MEASURING DISTANCE. On raw columns a Euclidean metric is
        # whichever channel has the widest units: calendar year spans ~15 and age ~60,
        # while every anatomical flag spans 1, so "nearest" silently meant "same year and
        # same age" and no amount of clinical filtering above it would show that. Display
        # and residuals stay in natural units -- the frame is untouched.
        self.standardize = standardize
        self._mu = raw.mean(axis=0) if standardize else np.zeros(raw.shape[1])
        sd = raw.std(axis=0) if standardize else np.ones(raw.shape[1])
        self._sd = np.where(sd > 1e-12, sd, 1.0)
        self._x = (raw - self._mu) / self._sd
        # MISSING-INDICATOR METRIC. An unassessed cell is set to the column mean (zero once
        # standardised), so it contributes nothing to any distance and the indicator channel
        # carries the fact of missingness on its own. Leaving the fill value in would make
        # "never scanned" numerically identical to "scanned, nothing found".
        for flag, governed in self.observation_flags.items():
            if flag not in self.channels:
                raise KeyError(f"observation flag {flag!r} is not a channel")
            absent = self.frame[flag].to_numpy(float) == 0
            if not absent.any():
                continue
            for c in governed:
                if c in self.channels:
                    self._x[absent, self.channels.index(c)] = 0.0
        # the scaling is part of what "nearest" means here, so it belongs in the id: two
        # corpora that differ only in it are not interchangeable
        self.corpus_id = hashlib.sha256(
            (",".join(self.channels) + "|" + str(int(standardize)) + "|"
             + ",".join(sorted(self.observation_flags)) + "|"
             + str(len(self.frame)) + "|"
             + pd.util.hash_pandas_object(self.frame[self.channels],
                                          index=False).sum().astype(str)
             ).encode()).hexdigest()[:16]
        if weights is None:
            self.weights = np.ones(len(self.channels))
        else:
            if len(weights) != len(self.channels):
                raise ValueError("weights must have one entry per channel")
            self.weights = np.asarray(weights, float)

    def __len__(self) -> int:
        return len(self.frame)

    def encode(self, values: np.ndarray, idx: list[int]) -> np.ndarray:
        """Put an outside vector on the same scale as :attr:`matrix`, for those channels."""
        return (np.asarray(values, float) - self._mu[idx]) / self._sd[idx]

    def channel_sd(self, col: str) -> float:
        """Spread of a channel in the corpus, for putting residuals on a common scale."""
        return float(self._sd[self.channels.index(col)]) if col in self.channels else 1.0

    def unobserved_for(self, row: pd.Series | Mapping[str, Any]) -> tuple[str, ...]:
        """Channels this row never had assessed, per its own indicators."""
        out: list[str] = []
        for flag, governed in self.observation_flags.items():
            try:
                seen = float(row[flag])
            except (KeyError, TypeError, ValueError):
                continue
            if seen == 0:
                out.extend(governed)
        return tuple(dict.fromkeys(out))

    @property
    def matrix(self) -> np.ndarray:
        return self._x

    def with_weights(self, weights: np.ndarray) -> Corpus:
        """A copy carrying a supervised metric fitted elsewhere, on training rows only."""
        return Corpus(self.frame, channels=self.channels, subject_col=self.subject_col,
                      action_col=self.action_col, outcome_cols=self.outcome_cols,
                      weights=weights, unobserved=self.unobserved,
                      standardize=self.standardize,
                      observation_flags=self.observation_flags,
                      prognostic_weights=self.prognostic_weights,
                      outcome_time_col=self.outcome_time_col,
                      outcome_event_col=self.outcome_event_col,
                      extra_outcomes=self.extra_outcomes)


def _eligible(corpus: Corpus, query: pd.Series, policy: AnchorPolicy) -> np.ndarray:
    """Rows allowed to be anchors, before any distance is computed."""
    ok = np.ones(len(corpus), bool)
    if policy.exclude_same_subject:
        ok &= (corpus.frame[corpus.subject_col] != query[corpus.subject_col]).to_numpy()
    wanted = policy.target_action if policy.target_action is not None else query[
        corpus.action_col]
    if policy.same_action or policy.target_action is not None:
        ok &= (corpus.frame[corpus.action_col] == wanted).to_numpy()
    for col in policy.match_exact:
        if col not in corpus.frame.columns:
            raise KeyError(f"match_exact names a column the corpus lacks: {col}")
        ok &= (corpus.frame[col] == query[col]).to_numpy()
    return ok


def _distances(corpus: Corpus, query: pd.Series, policy: AnchorPolicy,
               eligible: np.ndarray) -> np.ndarray:
    """Weighted Euclidean over the policy's channels. Archetype metric is a caller concern."""
    cols = list(policy.channels) if policy.channels else corpus.channels
    idx = [corpus.channels.index(c) for c in cols]
    w = corpus.weights[idx] if policy.metric == "supervised" else np.ones(len(idx))
    q = corpus.encode(query[cols].to_numpy(float), idx) * w
    x = corpus.matrix[np.ix_(eligible.nonzero()[0], idx)] * w
    return np.linalg.norm(x - q, axis=1)


def _select(order: np.ndarray, dist: np.ndarray, policy: AnchorPolicy,
            rng: np.random.Generator) -> np.ndarray:
    """Which of the eligible rows become anchors, in the policy's chosen manner."""
    if policy.selection == "nearest":
        return order[: policy.k]
    if policy.selection == "diverse":
        # farthest-point traversal among the nearest 3k: spread across the neighbourhood
        # rather than clustered on one side of it
        pool = order[: min(len(order), 3 * policy.k)]
        chosen = [pool[0]]
        while len(chosen) < min(policy.k, len(pool)):
            rest = [p for p in pool if p not in chosen]
            if not rest:
                break
            chosen.append(max(rest, key=lambda p: min(abs(dist[p] - dist[c])
                                                      for c in chosen)))
        return np.array(chosen)
    if policy.selection == "stratified":
        pool = order[: min(len(order), 3 * policy.k)]
        if len(pool) <= policy.k:
            return pool
        take = np.linspace(0, len(pool) - 1, policy.k).round().astype(int)
        return pool[np.unique(take)]
    raise ValueError(f"unknown selection {policy.selection!r}")


def _label(n_anchors: int, policy: AnchorPolicy, *, n_relaxed: int = 0) -> SupportLabel:
    """Which rung of the evidence ladder this retrieval actually stands on.

    `aggregate` is never returned: no retrieval for published trial evidence exists yet, and
    a label the pipeline cannot back up would be worse than no label.
    """
    if n_anchors < policy.min_support:
        return "unsupported"
    # A set topped up by relaxation is rung 2, not rung 1. "Direct" has to mean every
    # anchor satisfied the clinical criteria we said we would impose; six that did plus
    # four that did not is a transported estimate and is labelled as one.
    if n_relaxed:
        return "transport"
    if policy.same_action and policy.target_action is None:
        return "direct"
    return "transport"


def retrieve(corpus: Corpus, query: pd.Series, policy: AnchorPolicy) -> RetrievalResult:
    """Anchors for one query, or a labelled refusal.

    Deterministic given (corpus, query, policy): the only randomness is the shuffle control,
    which is seeded from the policy so a control arm replays exactly.
    """
    rng = np.random.default_rng(policy.shuffle_seed)
    relaxed = tuple(
        a for a, was in (("action", not policy.same_action and policy.target_action is None),)
        if was)
    # WHAT SUPPORT IS SUPPORT IN. Carried on every result, including the refusals, so a
    # support count can never be read as "we know this patient well".
    cols = list(policy.channels) if policy.channels else corpus.channels
    base = dict(query_id=str(query.get("query_id", query.name)),
                policy_hash=policy.hash, policy_label=policy.label,
                corpus_id=corpus.corpus_id, relaxed_axes=relaxed,
                support_space=tuple([*policy.match_exact, *cols]),
                unobserved_axes=tuple(corpus.unobserved),
                query_unobserved=corpus.unobserved_for(query))

    eligible = _eligible(corpus, query, policy)
    n_eligible = int(eligible.sum())
    # NO early return on a thin strict set. Refusing here would skip the tier relaxation
    # in precisely the case it exists for: "6 strict and 4 relaxed" can only happen if a
    # strict set of 6 is allowed to reach the top-up. The support decision is made AFTER
    # relaxation, against the total.
    if n_eligible == 0 and not (policy.relax_to_fill and policy.tiers):
        return RetrievalResult(anchor_index=(), distances=(), n_eligible=0,
                               support_label="unsupported",
                               note="no eligible patient, and no relaxation rule to fall "
                                    "back on", **base)

    pos = eligible.nonzero()[0]
    dist = (_distances(corpus, query, policy, eligible) if n_eligible
            else np.zeros(0, float))
    order = np.argsort(dist)
    picked_local = _select(order, dist, policy, rng) if n_eligible else np.zeros(0, int)
    n_strict = len(picked_local)
    if policy.max_distance is not None and len(picked_local):
        picked_local = picked_local[dist[picked_local] <= policy.max_distance]
        n_strict = len(picked_local)

    if policy.shuffle_anchors:
        # The control: the same number of anchors, from patients who are NOT the
        # neighbourhood, so the prompt looks identical and only the evidence is wrong.
        # Default scope is the whole corpus -- restricting it to the same eligibility
        # filters made the control unbuildable in exactly the sparse cells the method is
        # aimed at, and weaker everywhere else.
        if policy.shuffle_scope == "eligible":
            pool = pos[np.setdiff1d(order, picked_local, assume_unique=False)]
        else:
            not_self = np.ones(len(corpus), bool)
            if policy.exclude_same_subject:
                not_self &= (corpus.frame[corpus.subject_col]
                             != query[corpus.subject_col]).to_numpy()
            not_self[pos[picked_local]] = False
            pool = not_self.nonzero()[0]
        want = max(len(picked_local), policy.render_target)
        if len(pool) < max(policy.min_support, 1):
            return RetrievalResult(
                anchor_index=(), distances=(), n_eligible=n_eligible,
                support_label="unsupported",
                note=f"shuffle control not constructible under scope "
                     f"{policy.shuffle_scope!r}: {len(pool)} candidates outside the "
                     f"{want}-anchor neighbourhood", **base)
        sham = rng.choice(pool, size=min(want, len(pool)), replace=False)
        return _summarise(corpus, query, policy, sham,
                          labels=["sham"] * len(sham), n_strict=0, n_eligible=n_eligible,
                          distances=None, support_label="sham", is_sham=True, base=base)

    picked = pos[picked_local]
    labels = ["strict"] * len(picked)

    # TIERED RELAXATION. When the strict set cannot fill the display, the shortfall comes
    # from progressively looser rules, IN ORDER, and every anchor carries the rule that
    # admitted it. Relaxing silently would put "same cancer and regimen, different line"
    # beside "same everything" and ask the model to treat them alike; it is a different
    # clinical claim and the model is told which is which.
    if policy.relax_to_fill and len(picked) < policy.render_target:
        for label, fields in policy.tiers:
            if len(picked) >= policy.render_target:
                break
            ok = np.ones(len(corpus), bool)
            if policy.exclude_same_subject:
                ok &= (corpus.frame[corpus.subject_col]
                       != query[corpus.subject_col]).to_numpy()
            for col in fields:
                if col not in corpus.frame.columns:
                    raise KeyError(f"tier {label!r} names a column the corpus lacks: {col}")
                ok &= (corpus.frame[col] == query[col]).to_numpy()
            ok[picked] = False
            cand = ok.nonzero()[0]
            if not len(cand):
                continue
            cols_all = list(policy.channels) if policy.channels else corpus.channels
            idx = [corpus.channels.index(c) for c in cols_all]
            w = corpus.weights[idx] if policy.metric == "supervised" else np.ones(len(idx))
            dd = np.linalg.norm(
                corpus.matrix[np.ix_(cand, idx)] * w
                - corpus.encode(query[cols_all].to_numpy(float), idx) * w, axis=1)
            take = cand[np.argsort(dd)[: policy.render_target - len(picked)]]
            picked = np.concatenate([picked, take])
            labels += [label] * len(take)

    # the support decision, now that relaxation has had its chance
    if len(picked) < policy.min_support:
        return RetrievalResult(
            anchor_index=(), distances=(), n_eligible=n_eligible,
            support_label="unsupported", n_strict=n_strict,
            note=f"{n_strict} strict and {len(picked) - n_strict} relaxed anchors, "
                 f"need {policy.min_support}", **base)

    # distances recomputed inside _summarise over the FULL picked set: relaxed anchors
    # have no entry in `dist`, which is indexed on the strict pool only, so reusing it
    # would silently hand back fewer distances than anchors.
    return _summarise(corpus, query, policy, picked, labels=labels, n_strict=n_strict,
                      n_eligible=n_eligible, distances=None,
                      support_label=_label(len(picked), policy,
                                           n_relaxed=len(picked) - n_strict),
                      is_sham=False, base=base)


def _summarise(corpus: Corpus, query: pd.Series, policy: AnchorPolicy, picked: np.ndarray,
               *, labels: list[str], n_strict: int, n_eligible: int,
               distances: np.ndarray | None, support_label: str, is_sham: bool,
               base: dict) -> RetrievalResult:
    """Build the result and its audit. Shared so the control is measured like the arm."""
    cols = list(policy.channels) if policy.channels else corpus.channels
    rows = corpus.frame.iloc[picked]
    centre = rows[cols].to_numpy(float).mean(axis=0)
    # residual in NATURAL units: "eight years older" is a sentence a clinician can check.
    residual = {c: float(query[c] - m) for c, m in zip(cols, centre, strict=True)}
    # spread in STANDARDISED units, because that is what the prompt calls it and what the
    # distance ranking actually used
    zi = [corpus.channels.index(c) for c in cols]
    z = corpus.matrix[np.ix_(picked, zi)]
    spread = float(np.linalg.norm(z - z.mean(axis=0), axis=1).mean()) if len(picked) > 1 \
        else 0.0
    disp = float("nan")
    if corpus.outcome_cols:
        v = pd.to_numeric(rows[corpus.outcome_cols[0]], errors="coerce").dropna()
        if len(v) > 1:
            disp = float(v.quantile(0.75) - v.quantile(0.25))
    cens = float("nan")
    if corpus.outcome_event_col and corpus.outcome_event_col in rows.columns:
        ev = pd.to_numeric(rows[corpus.outcome_event_col], errors="coerce").dropna()
        if len(ev):
            cens = float((ev == 0).mean())
    if distances is None:
        idx = [corpus.channels.index(c) for c in cols]
        w = corpus.weights[idx] if policy.metric == "supervised" else np.ones(len(idx))
        distances = np.linalg.norm(
            corpus.matrix[np.ix_(picked, idx)] * w
            - corpus.encode(query[cols].to_numpy(float), idx) * w, axis=1)
    counts = [(lab, labels.count(lab)) for lab in dict.fromkeys(labels)]
    return RetrievalResult(
        anchor_index=tuple(int(i) for i in picked),
        distances=tuple(float(d) for d in distances), n_eligible=n_eligible,
        support_label=support_label, n_strict=n_strict, is_sham=is_sham,
        tier_labels=tuple(labels), tier_counts=tuple(counts), censored_fraction=cens,
        residual=residual, anchor_spread=spread, outcome_dispersion=disp,
        max_distance_used=float(np.max(distances)) if len(distances) else 0.0, **base)


def retrieve_pair(corpus: Corpus, query: pd.Series, policy: AnchorPolicy,
                  action_a: str, action_b: str) -> tuple[RetrievalResult, RetrievalResult]:
    """Two anchor sets differing only in action -- the counterfactual construction.

    The contrast between them is an OUTCOME CONTRAST in observational data: treatment was
    chosen for reasons that were not measured. It is an average treatment effect only where
    assignment was randomised.
    """
    pa = policy.variant(same_action=False, target_action=action_a)
    pb = policy.variant(same_action=False, target_action=action_b)
    return retrieve(corpus, query, pa), retrieve(corpus, query, pb)
