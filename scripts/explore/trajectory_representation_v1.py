#!/usr/bin/env python3
"""ONE trajectory representation, used by the gate, the real-data scorer and the anchoring policy.

WHY THIS EXISTS. The state-space gate scored refinements on a feature map whose disease
coordinate was `month < first_progression` and whose terminal coordinate was
`month < death` -- two passage times drawn as step functions. Two courses with the same
first progression and the same death scored at distance EXACTLY 0.0 whether one contained
two responses or none. Every conclusion from that gate is therefore about passage times and
treatment occupancy, not about trajectory shape.

FIVE BLOCKS, EACH PRIMITIVE WITH EXACTLY ONE HOME. The first version of this module
double-counted: passage times appeared in `clock` AND as step-function occupancy, treatment
starts sat in `richness` AND `treatment`, and the off-treatment fraction was a deterministic
function of the on-treatment vector in its own block. A composite over that implicitly
upweighted PFS, OS and scheduling -- the exact quantities the representation exists to see
beyond. Now:

    clock       first progression, death, first response -- the passage times, HERE ONLY
    occupancy   monthly RESPONDING-state indicators (from a response until the next
                progression or death). The one course-state occupancy that is NOT a
                deterministic function of the clock coordinates. Progression-free and alive
                step functions are deliberately ABSENT: they are the clock, redrawn.
    richness    sqrt(#progressions), sqrt(#responses) -- episode counts, deduplicated
    grammar     normalized typed-VP dissimilarities to a frozen panel of PROTOTYPES
    treatment   monthly on-treatment indicators + sqrt(#starts), sqrt(#stops)

STILL NOT FULLY HOLISTIC, and said plainly: post-progression sub-states, toxicity and host
state are absent because the datasets this feeds cannot observe them. Response-progression
loops are visible only through the responding windows, the counts and the grammar.

THE GRAMMAR QUANTITY IS A DISSIMILARITY, NOT A METRIC. Normalizing the VP edit cost by
max(n, m) breaks the triangle inequality -- verified counterexample:
d(RR, PP) = 2.000 > d(RR, RPR) + d(RPR, PP) = 1.667. It is kept because it puts sequences of
different lengths on one scale, and because nothing downstream needs the raw quantity to be
a metric: trajectories enter arithmetic only through their EMBEDDING -- the vector of
dissimilarities to a frozen prototype panel -- whose Euclidean geometry is what the energy
score uses. The raw VP is never added into a norm.

NO INFORMATION CROSSES THE HORIZON. Training prototypes are projected, episode-deduplicated
and truncated at the SAME horizon as every query before selection, so a 12-month analysis
cannot consult month-55 events through the back door of the panel. The first version stored
raw prototypes and did exactly that.

OBSERVATION INTENSITY IS A CONFOUND, NOT A SIGNAL. Episodes deduplicate same-kind events
inside a window -- but NEVER across an intervening treatment event: a progression before a
new line and a progression on it are different clinical facts, a protection the earlier
similarity module had and the first version of this one dropped. Square-root counts bound
the scale; they do not remove observation-process bias, so scan-frequency sensitivity
remains the caller's obligation. This measures OBSERVABLE trajectory shape in a dataset,
never latent biological shape.

Grammar time-normalization is PRESPECIFIED as each course's own duration ("shape-only");
`time_norm="horizon"` is the declared sensitivity, not a tuning knob.

Run:  uv run python scripts/explore/trajectory_representation_v1.py --self-test
Tests: tests/unit/test_trajectory_representation_v1.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BLOCKS = ("clock", "occupancy", "richness", "grammar", "treatment")
EPISODE_WINDOW = 1.0        # months; same-kind events inside it are one clinical episode
VP_Q = 4.0                  # cost per unit of normalised time shift
VP_KAPPA = 2.0              # cost of relabelling one event type as another
N_PROTOTYPES = 12

_ON = {"systemic_treatment_start", "systemic_treatment_switch"}
_OFF = {"systemic_treatment_stop"}
_TREATMENT = _ON | _OFF | {"systemic_treatment_modification"}
_PROG = {"progression", "recurrence"}

#: Clinically adjacent event types cost less to substitute than unrelated ones. Confusing a
#: response with a progression must not cost the same as confusing two treatment events.
_ADJACENT = [{"progression", "recurrence"},
             {"systemic_treatment_start", "systemic_treatment_switch"},
             {"systemic_treatment_stop", "systemic_treatment_modification"}]


def _family(mark: str) -> str:
    return mark.split("::")[0]


def mark_distance(a: str, b: str) -> float:
    """Graded by the event ontology, with treatment CLASS one level finer.

    Identical marks cost 0. Two starts of DIFFERENT classes cost 0.5 -- same family,
    different identity, so FOLFOX-then-FOLFIRI and FOLFOX-then-osimertinib no longer look
    identical. Family-adjacent events cost 0.5; unrelated events 1.0.
    """
    if a == b:
        return 0.0
    fa, fb = _family(a), _family(b)
    if fa == fb:
        return 0.5
    return 0.5 if any(fa in s and fb in s for s in _ADJACENT) else 1.0


def typed_vp(times_a, marks_a, times_b, marks_b, q: float = VP_Q,
             kappa: float = VP_KAPPA, time_norm: str | float = "own") -> float:
    """Normalized typed Victor-Purpura DISSIMILARITY between two marked event sequences.

    Insert or delete an event: cost 1. Align two events: cost
    `min(2, q * |du| + kappa * d_M)` -- the cap is required, or the alignment prefers
    nonsense matches over deletion-plus-reinsertion. Total cost is divided by max(n, m).

    THAT DIVISION MAKES THIS A DISSIMILARITY, NOT A METRIC: the triangle inequality fails
    (d(RR,PP)=2.0 > d(RR,RPR)+d(RPR,PP)=1.667, verified). Kept because it puts sequences of
    different lengths on one scale, and because downstream arithmetic uses only the
    prototype EMBEDDING, never this raw quantity inside a norm.

    `time_norm`: "own" (prespecified primary -- shape-only, each course's own duration) or a
    float horizon for common-scale normalization (the declared sensitivity).
    """
    n, m = len(times_a), len(times_b)
    if n == 0 or m == 0:
        return float(n + m) / max(n, m, 1)
    if time_norm == "own":
        za = max(float(np.max(times_a)), 1e-9)
        zb = max(float(np.max(times_b)), 1e-9)
    else:
        za = zb = max(float(time_norm), 1e-9)
    ua = np.asarray(times_a, float) / za
    ub = np.asarray(times_b, float) / zb
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = min(2.0, q * abs(ua[i - 1] - ub[j - 1])
                      + kappa * mark_distance(marks_a[i - 1], marks_b[j - 1]))
            D[i, j] = min(D[i - 1, j] + 1.0, D[i, j - 1] + 1.0, D[i - 1, j - 1] + sub)
    return float(D[n, m]) / max(n, m)


def episodes(times, marks, horizon: float) -> tuple[list[float], list[str]]:
    """Same-kind events inside the window collapse to their first occurrence -- but NEVER
    across an intervening treatment event.

    Without deduplication, a patient scanned fortnightly through one progression contributes
    three progressions and a quarterly-scanned patient one, and the difference is imaging
    cadence. But two progressions bracketing a treatment change are different clinical
    facts -- progressing before the new line and progressing on it -- and merging them
    erases the failure of a therapy.
    """
    rank = {"systemic_treatment_stop": 0, "systemic_treatment_start": 1,
            "systemic_treatment_switch": 1}
    order = sorted(range(len(times)),
                   key=lambda i: (float(times[i]), rank.get(_family(marks[i]), 2),
                                  marks[i]))
    out_t, out_m, last = [], [], {}
    for i in order:
        t, mark = float(times[i]), marks[i]
        if t > horizon:
            continue
        previous = last.get(mark)
        if previous is not None and t - previous <= EPISODE_WINDOW:
            crossed = any(_family(km) in _TREATMENT and previous < kt <= t
                          for kt, km in zip(out_t, out_m, strict=True))
            if not crossed:
                continue
        last[mark] = t
        out_t.append(t)
        out_m.append(mark)
    return out_t, out_m


def components(times, marks, horizon: float,
               prototypes: list[tuple] | None = None,
               time_norm: str | float = "own") -> dict[str, np.ndarray]:
    """The five blocks for one course, before scaling. Each primitive has one home."""
    t, m = episodes(times, marks, horizon)
    months = np.arange(1.0, horizon + 1.0)

    first_prog = next((x for x, k in zip(t, m, strict=True) if k in _PROG), np.inf)
    first_resp = next((x for x, k in zip(t, m, strict=True) if k == "response"), np.inf)
    death_at = next((x for x, k in zip(t, m, strict=True) if k == "death"), np.inf)

    # RESPONDING-state windows: from each response until the next progression or death.
    responding = np.zeros(len(months))
    for x, k in zip(t, m, strict=True):
        if k != "response":
            continue
        end = min([y for y, kk in zip(t, m, strict=True)
                   if y > x and (kk in _PROG or kk == "death")] + [death_at, horizon])
        responding[(months >= x) & (months < end)] = 1.0

    on, state, idx = np.zeros(len(months)), 1.0, 0
    for j, month in enumerate(months):
        while idx < len(t) and t[idx] <= month:
            if _family(m[idx]) in _ON:
                state = 1.0
            elif _family(m[idx]) in _OFF:
                state = 0.0
            idx += 1
        on[j] = state if month <= death_at else 0.0

    def clip(x):
        return float(min(x, horizon)) if np.isfinite(x) else float(horizon)

    n_prog = sum(1 for k in m if k in _PROG)
    n_resp = sum(1 for k in m if k == "response")
    n_start = sum(1 for k in m if _family(k) in _ON)
    n_stop = sum(1 for k in m if _family(k) in _OFF)
    # TREATMENT IDENTITY: how many DISTINCT classes were started. Sequence identity beyond
    # this lives in the grammar, where the class-typed marks flow through the typed VP.
    n_classes = len({k.split("::", 1)[1] for k in m
                     if _family(k) in _ON and "::" in k})

    out = {
        "clock": np.array([clip(first_prog), clip(death_at), clip(first_resp)]),
        "occupancy": responding,
        # SQUARE ROOT bounds the scale; treatment starts live in `treatment`, not here.
        "richness": np.sqrt(np.array([n_prog, n_resp], float)),
        "treatment": np.concatenate([on, [np.sqrt(n_start), np.sqrt(n_stop),
                                          np.sqrt(float(n_classes))]]),
    }
    if prototypes:
        out["grammar"] = np.array([typed_vp(t, m, pt, pm, time_norm=time_norm)
                                   for pt, pm in prototypes])
    else:
        out["grammar"] = np.zeros(0)
    return out


@dataclass(frozen=True)
class Representation:
    """Frozen scaling and prototype panel. Fitted on TRAINING courses only.

    Prototypes are farthest-point REFERENCE LANDMARKS on the grammar dissimilarity -- no
    within-cluster objective is minimized, so they are not medoids and are not called that.
    They are stored ALREADY projected, deduplicated and horizon-truncated, so no event
    beyond the analysis horizon can reach a query through the panel.
    """
    prototypes: tuple
    coord_scale: dict
    block_scale: dict
    horizon: float
    time_norm: str | float = "own"

    def vector(self, times, marks) -> np.ndarray:
        raw = components(times, marks, self.horizon, list(self.prototypes),
                         self.time_norm)
        return np.concatenate([(raw[b] / self.coord_scale[b]) * self.block_scale[b]
                               for b in BLOCKS])

    def profile(self, a, b) -> dict[str, float]:
        """Per-block distances. A composite alone has repeatedly hidden opposite signs."""
        ra = components(a[0], a[1], self.horizon, list(self.prototypes), self.time_norm)
        rb = components(b[0], b[1], self.horizon, list(self.prototypes), self.time_norm)
        out = {}
        for blk in BLOCKS:
            u = (ra[blk] / self.coord_scale[blk]) * self.block_scale[blk]
            w = (rb[blk] / self.coord_scale[blk]) * self.block_scale[blk]
            out[blk] = float(np.linalg.norm(u - w))
        out["composite"] = float(np.linalg.norm(self.vector(*a) - self.vector(*b)))
        return out


def fit(training: list[tuple], horizon: float, n_prototypes: int = N_PROTOTYPES,
        seed: int = 20260819, time_norm: str | float = "own") -> Representation:
    """Fit coordinate scales, block scales and the prototype panel on TRAINING courses only.

    Training courses are CANONICALIZED FIRST -- projected through episodes() at the analysis
    horizon -- so the panel lives in exactly the space queries live in. Two levels of
    scaling: per-COORDINATE (a wide coordinate must not dominate its block) then per-BLOCK
    (a long block must not dominate the composite). A coordinate with no training variance
    gets INFINITE scale, i.e. zero weight: a feature absent in training must not contribute
    arbitrarily in test data, which a fallback scale of 1.0 allowed.
    """
    rng = np.random.default_rng(seed)
    canonical = [episodes(t, m, horizon) for t, m in training]
    canonical = [(t, m) for t, m in canonical if t]
    if len(canonical) < n_prototypes + 2:
        n_prototypes = max(2, len(canonical) // 2)

    picked = [int(rng.integers(len(canonical)))]
    while len(picked) < n_prototypes:
        best, best_d = None, -1.0
        for i in rng.choice(len(canonical), min(len(canonical), 60), replace=False):
            if i in picked:
                continue
            d = min(typed_vp(*canonical[i], *canonical[j], time_norm=time_norm)
                    for j in picked)
            if d > best_d:
                best, best_d = int(i), d
        if best is None:
            break
        picked.append(best)
    prototypes = tuple(canonical[i] for i in picked)

    raw = [components(t, m, horizon, list(prototypes), time_norm) for t, m in canonical]
    coord_scale, block_scale = {}, {}
    for b in BLOCKS:
        stack = np.array([r[b] for r in raw])
        if stack.size == 0 or stack.shape[1] == 0:
            coord_scale[b], block_scale[b] = 1.0, 0.0
            continue
        sd = stack.std(axis=0)
        coord_scale[b] = np.where(sd > 1e-9, sd, np.inf)
        scaled = stack / coord_scale[b]
        centred = scaled - scaled.mean(axis=0)
        typical = float(np.sqrt((centred ** 2).sum(axis=1).mean()))
        block_scale[b] = 1.0 / typical if typical > 1e-9 else 1.0
    return Representation(prototypes, coord_scale, block_scale, horizon, time_norm)


# ---------------------------------------------------------------------------------------
# ADVERSARIAL FIXTURES, one per block. Each pair holds every OTHER block's primitives as
# equal as the entanglements allow, and the named block must detect the difference.
# ---------------------------------------------------------------------------------------
FIXTURES = {
    # same passage times and counts; only the RESPONDING WINDOW differs (7 vs 1.5 months)
    "occupancy": (
        ([3.0, 5.0, 12.0, 14.0], ["progression", "response", "progression", "death"]),
        ([3.0, 5.0, 6.5, 14.0], ["progression", "response", "progression", "death"]),
    ),
    # same first progression and death; two responses vs none (the old gate's 0.0 pair)
    "richness": (
        ([2.0, 5.0, 9.0, 14.0], ["response", "response", "progression", "death"]),
        ([9.0, 14.0], ["progression", "death"]),
    ),
    # same event multiset at the same times, reordered
    "grammar": (
        ([3.0, 6.0, 9.0, 14.0], ["response", "systemic_treatment_switch",
                                 "progression", "death"]),
        ([3.0, 6.0, 9.0, 14.0], ["systemic_treatment_switch", "response",
                                 "progression", "death"]),
    ),
    "treatment": (
        ([4.0, 5.5, 9.0, 14.0], ["systemic_treatment_stop", "systemic_treatment_start",
                                 "progression", "death"]),
        ([9.0, 14.0], ["progression", "death"]),
    ),
    "clock": (
        ([9.0, 14.0], ["progression", "death"]),
        ([3.0, 14.0], ["progression", "death"]),
    ),
}


def _toy_training(rng: np.random.Generator, n: int = 80) -> list[tuple]:
    training = []
    for _ in range(n):
        p = float(rng.uniform(2, 18))
        d = float(min(p + rng.uniform(1, 10), 23.5))
        seq = [(p, "progression"), (d, "death")]
        if rng.random() < 0.5:
            seq.insert(0, (max(0.5, p - rng.uniform(1, 4)), "response"))
        if rng.random() < 0.5:
            seq.append((min(d - 0.4, p + 0.6), "systemic_treatment_switch"))
        seq.sort()
        training.append(([x for x, _ in seq], [k for _, k in seq]))
    return training


def self_test() -> int:
    horizon = 24.0
    rng = np.random.default_rng(0)
    rep = fit(_toy_training(rng), horizon, n_prototypes=8)

    # NO HORIZON LEAKAGE: a 12-month fit on late-event training stores nothing past 12
    late = [([2.0, 30.0, 55.0], ["progression", "progression", "death"])
            for _ in range(12)]
    rep12 = fit(late, horizon=12.0, n_prototypes=4)
    assert all(max(t) <= 12.0 for t, _ in rep12.prototypes), rep12.prototypes

    # the old gate's blind spot is separated
    assert rep.profile(*FIXTURES["richness"])["composite"] > 0.1

    # EVERY block detects its own fixture -- occupancy now included
    for name, (x, y) in FIXTURES.items():
        prof = rep.profile(x, y)
        assert prof[name] > 0.05, f"{name} cannot see its own fixture: {prof}"

    # occupancy fixture is ISOLATED: clock and richness identical by construction
    op = rep.profile(*FIXTURES["occupancy"])
    assert op["clock"] < 1e-9 and op["richness"] < 1e-9, op

    # grammar: order-sensitivity of the raw dissimilarity at identical times
    assert typed_vp([3.0, 6.0, 9.0], ["response", "systemic_treatment_switch", "death"],
                    [3.0, 6.0, 9.0], ["systemic_treatment_switch", "response", "death"]
                    ) > 0.1

    # VP economics: identity zero, ontology-graded substitution, capped, and NON-METRIC
    t, m = [1.0, 5.0, 9.0], ["response", "progression", "death"]
    assert typed_vp(t, m, t, m) == 0.0
    adjacent = typed_vp(t, m, t, ["response", "recurrence", "death"])
    unrelated = typed_vp(t, m, t, ["response", "systemic_treatment_stop", "death"])
    assert 0 < adjacent < unrelated <= 2.0, (adjacent, unrelated)
    A = ([1.0, 2.0], ["response", "response"])
    B = ([1.0, 2.0], ["progression", "progression"])
    C = ([1.0, 1.5, 2.0], ["response", "progression", "response"])
    assert typed_vp(*A, *B) > typed_vp(*A, *C) + typed_vp(*C, *B), \
        "the documented triangle violation must persist while the /max(n,m) scale is kept"

    # DEDUP RESPECTS TREATMENT BOUNDARIES: two progressions 1.0 apart merge, unless a
    # treatment event sits between them
    _, plain_m = episodes([2.0, 2.8], ["progression", "progression"], horizon)
    assert plain_m.count("progression") == 1, plain_m
    _, split_m = episodes([2.0, 2.5, 2.8],
                          ["progression", "systemic_treatment_start", "progression"],
                          horizon)
    assert split_m.count("progression") == 2, split_m

    # ZERO-VARIANCE COORDINATE IS INERT: training with no responses -> the response count
    # cannot contribute in test
    no_resp = [([float(p), float(p) + 3.0], ["progression", "death"])
               for p in range(2, 20)]
    rep_nr = fit(no_resp, horizon, n_prototypes=4)
    pair = (([5.0, 8.0], ["progression", "death"]),
            ([5.0, 6.0, 7.0, 8.0], ["progression", "response", "response", "death"]))
    assert rep_nr.profile(*pair)["richness"] < 1e-9, rep_nr.profile(*pair)

    # TREATMENT IDENTITY: same schedule, different second drug class must separate --
    # the audit's concrete failure was FOLFOX-then-FOLFIRI vs FOLFOX-then-osimertinib
    # scoring as identical. Grammar carries the identity through class-typed marks; the
    # treatment block carries the distinct-class count.
    assert mark_distance("systemic_treatment_start::platinum",
                         "systemic_treatment_start::targeted TKI") == 0.5
    assert mark_distance("systemic_treatment_start::platinum",
                         "systemic_treatment_start::platinum") == 0.0
    assert mark_distance("systemic_treatment_start::platinum", "progression") == 1.0
    same_class = ([3.0, 8.0], ["systemic_treatment_start::platinum",
                               "systemic_treatment_start::platinum"])
    two_class = ([3.0, 8.0], ["systemic_treatment_start::platinum",
                              "systemic_treatment_start::targeted TKI"])
    ca2 = components(*same_class, horizon)
    cb2 = components(*two_class, horizon)
    assert cb2["treatment"][-1] > ca2["treatment"][-1], "distinct-class count must differ"
    idA = ([6.0, 7.0, 14.0], ["progression", "systemic_treatment_start::antimetabolite",
                              "death"])
    idB = ([6.0, 7.0, 14.0], ["progression", "systemic_treatment_start::targeted TKI",
                              "death"])
    assert typed_vp(*idA, *idB) > 0.05, "class identity must be visible to the grammar"
    # and class-typed starts still drive the on/off state machine
    c_on = components([4.0, 6.0], ["systemic_treatment_stop",
                                   "systemic_treatment_start::platinum"], horizon)
    assert c_on["treatment"][3] == 0.0 and c_on["treatment"][6] == 1.0

    # SINGLE HOME: richness carries no treatment starts; occupancy is the responding state,
    # not a passage-time step function
    c = components([2.0, 5.0, 9.0, 14.0],
                   ["response", "systemic_treatment_switch", "progression", "death"],
                   horizon)
    assert len(c["richness"]) == 2
    assert c["occupancy"].max() == 1.0 and c["occupancy"][0] == 0.0

    print("self-test OK — no horizon leakage through the prototype panel, every block\n"
          "   (occupancy included) detects its own fixture with occupancy isolated from\n"
          "   clock and richness, dedup never merges across a treatment event, a\n"
          "   zero-variance training coordinate is inert in test, each primitive has one\n"
          "   home, and the VP is explicitly a dissimilarity: the triangle violation that\n"
          "   makes it one is asserted, not hidden")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.parse_args()
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
