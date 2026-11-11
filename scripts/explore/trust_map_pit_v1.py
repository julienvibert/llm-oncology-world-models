#!/usr/bin/env python3
"""Score REALITY against the model's imagined distribution, and map where that is trustworthy.

For one real patient at one real decision point the model emits a fan of futures. The recorded
outcome then falls somewhere in that fan -- early, late, or in the middle. Its position is the
probability integral transform, and if the model is calibrated those positions are UNIFORM
across patients. The shape of the departure says how it is wrong:

    piled at 0        reality arrives sooner than the model imagines   -- too optimistic
    piled at 1        reality arrives later                            -- too pessimistic
    piled at 0 AND 1  reality keeps landing outside the fan            -- fan too NARROW
    piled in middle   reality is always near the centre                -- fan too WIDE

This needs no comparator, which is what makes it a map rather than a league table, and it fits
the claim this project can actually defend: not that the model predicts an individual, but that
the individual is a plausible draw from what it imagined.

THE BLIND SPOT, WHICH IS THE REASON THIS FILE IS NOT JUST `pit_values`. A model that IGNORES
THE PATIENT and emits the population-average distribution for everyone scores a PERFECTLY
uniform histogram. Flawless marks, no patient information used -- that is just the probability
integral transform applied to the marginal. Since the C/X/H/A pilot measured a model whose
predictions barely move across a 34-year age range, this failure mode is not hypothetical here.

So the pooled histogram is reported, and then IGNORED in favour of the stratified one. A
patient-ignoring model is uniform overall and skewed WITHIN strata -- too optimistic for the
sick, too pessimistic for the fit, cancelling in the pooled view. The spread of per-stratum
means is the diagnostic, and `self_test` demonstrates it on a generator built to fail exactly
that way.

CENSORING IS NOT DROPPED. `measure_pfs_distribution_bar.pit_values` excludes censored patients,
which is defensible at 80% events and not here: this endpoint is 71% progressed, so dropping
them selects for fast progressors and biases the histogram toward the optimistic end -- an
artefact that looks exactly like the finding. A censored patient contributes a randomised PIT
drawn from the interval its censoring time allows.

Run:  uv run python scripts/explore/trust_map_pit_v1.py --self-test
      uv run python scripts/explore/trust_map_pit_v1.py --generations <gens.jsonl> \
          --scope-dir /tmp/genie
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20260817
#: Uniform variance. The dispersion diagnostic is measured against it, not against zero.
UNIFORM_VAR = 1.0 / 12.0


def randomised_pit(draws: Sequence[float], outcome: float, event: bool,
                   rng: np.random.Generator) -> float:
    """Where the real outcome falls in the fan, as a number in [0, 1].

    With an m-member fan the rank is discrete, so a uniform jitter is added within the rank's
    interval -- without it the histogram is spiky for reasons that have nothing to do with the
    model, and every calibration test fails on the discreteness alone.

    A CENSORED patient is not dropped and not treated as an event. All that is known is that
    the true time lies beyond the censoring time, so the PIT is drawn uniformly from the part
    of the fan still compatible with that: [F(c), 1].
    """
    x = np.sort(np.asarray(draws, dtype=float))
    m = len(x)
    if m == 0:
        return float("nan")
    below = int(np.sum(x < outcome))
    if event:
        return float((below + rng.random()) / (m + 1))
    lower = below / (m + 1)
    return float(lower + rng.random() * (1.0 - lower))


def diagnostics(u: np.ndarray) -> dict:
    """Bias, dispersion and overall departure, from the PIT values of one group."""
    u = np.asarray([v for v in u if np.isfinite(v)], dtype=float)
    if len(u) < 5:
        return {"n": int(len(u)), "mean": float("nan"), "dispersion": float("nan"),
                "ks": float("nan"), "reading": "too few"}
    mean = float(u.mean())
    var = float(u.var(ddof=1))
    ks = float(np.max(np.abs(np.sort(u) - np.linspace(0, 1, len(u), endpoint=False))))
    # dispersion > 1 means the PIT piles at both ends: reality keeps landing outside the fan
    disp = var / UNIFORM_VAR
    if disp > 1.15:
        reading = "fan TOO NARROW — overconfident"
    elif disp < 0.85:
        reading = "fan TOO WIDE — hedging"
    elif mean < 0.42:
        reading = "too OPTIMISTIC — reality arrives sooner"
    elif mean > 0.58:
        reading = "too PESSIMISTIC"
    else:
        reading = "calibrated"
    return {"n": int(len(u)), "mean": mean, "dispersion": disp, "ks": ks, "reading": reading}


def resolution(frame: pd.DataFrame) -> dict:
    """Does the model produce a DIFFERENT fan per patient, or one fan for everybody?

    Calibration cannot answer this and a population-average forecaster passes it outright. The
    ratio here is the honest question: how far apart are different patients' predictions
    compared with how much one patient's prediction wobbles between draws? Below about 1 the
    model is saying roughly the same thing to everyone, and its apparent calibration is a
    property of the population rather than of any patient in it.
    """
    # The per-patient summary is the MEAN, not the median, purely so the noise term is exact:
    # SE(mean) = sd/sqrt(n) holds by construction, whereas the SE of a median depends on the
    # unknown density and using sd/sqrt(n) for it UNDERSTATES the noise -- which then survives
    # the subtraction below and reads as patient signal that is not there.
    med = frame.groupby("query_id")["pfs_samples"].apply(lambda s: float(np.mean(s.iloc[0])))
    within = frame.groupby("query_id")["pfs_samples"].apply(
        lambda s: float(np.std(s.iloc[0], ddof=1)) / np.sqrt(len(s.iloc[0])))
    observed = float(med.std(ddof=1))
    # MEAN of the per-patient VARIANCES, not the median of the standard errors. The decomposition
    # is Var(observed) = Var(true) + E[Var(sampling)], so the expectation has to be taken on the
    # variance scale; taking a median on the SE scale understates it whenever the per-patient
    # spread itself varies -- and it always does -- leaving residual noise that reads as signal.
    noise = float(np.sqrt(np.mean(np.square(within.to_numpy()))))
    # THE OBSERVED SPREAD IS NOT THE SIGNAL. Each patient's median is estimated from ten draws,
    # so the spread between patients already contains the sampling error of that estimate. With
    # a fan this wide the two are comparable: a model emitting ONE distribution for everybody
    # still shows an observed spread of about 3 months purely from resampling. Subtracting the
    # noise variance is what leaves the part attributable to the patient.
    signal2 = observed ** 2 - noise ** 2
    signal = float(np.sqrt(signal2)) if signal2 > 0 else 0.0
    ratio = signal / noise if noise > 0 else float("nan")
    out = {"observed_between_sd": observed, "within_patient_se": noise,
           "patient_signal_sd": signal, "ratio": ratio}
    # THE LABEL MUST NOT CONFLATE TWO THINGS. Predictions varying between queries is NOT
    # evidence of individual reasoning: a model that returns each query's own reference class
    # varies exactly as much as the reference classes do, and scored 4.8 on the ratio above
    # while reading nothing about the patient. So when a reference is available, the spread is
    # split into the part explained by it and the part the model added.
    if "anchor_km_median" in frame.columns:
        one = frame.drop_duplicates("query_id")
        ref = pd.to_numeric(one["anchor_km_median"], errors="coerce")
        pred = one["pfs_samples"].apply(lambda s: float(np.mean(s)))
        ok = ref.notna() & np.isfinite(pred)
        if ok.sum() >= 20:
            r, pr = ref[ok].to_numpy(float), pred[ok].to_numpy(float)
            adj = pr - np.polyval(np.polyfit(r, pr, 1), r)
            out["reference_class_sd"] = float(np.std(r))
            out["model_adjustment_sd"] = float(np.std(adj))
            out["reading"] = ("tracks the reference class only"
                              if np.std(adj) < 0.5 * noise else
                              "adjusts beyond the reference class")
            return out
    out["reading"] = ("one fan for everybody" if ratio < 0.5
                      else "varies between queries — source not separable without a reference")
    return out


def discrimination(frame: pd.DataFrame) -> dict:
    """Do the differences between patients ORDER reality correctly?

    Resolution says the fans differ; this says the differences point the right way. A model can
    vary a lot between patients and still rank them at random, which is worse than useless.
    Concordance over patients with an observed event.
    """
    ev = frame[frame["pfs_event"] == 1]
    if len(ev) < 20:
        return {"n": int(len(ev)), "concordance": float("nan")}
    pred = ev["pfs_samples"].apply(lambda s: float(np.median(s))).to_numpy()
    obs = ev["pfs_mos"].to_numpy()
    conc = disc = 0
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            if obs[i] == obs[j] or pred[i] == pred[j]:
                continue
            same = (obs[i] < obs[j]) == (pred[i] < pred[j])
            conc += same
            disc += not same
    total = conc + disc
    return {"n": int(len(ev)), "concordance": conc / total if total else float("nan")}


def moves_with_reality(frame: pd.DataFrame) -> dict:
    """EXPLORATORY: does the model's adjustment off the reference class track the individual?

    For each landmark, take the anchor cohort's own Kaplan-Meier median as the reference class
    -- what "patients like this one" do on average. Then ask two questions and correlate the
    answers:

        how far did the MODEL move this patient off that reference?
        how far did REALITY move this patient off it?

    NOT THE PRIMARY TEST, and the naive version of it is broken. Correlating (pred - ref) with
    (obs - ref) subtracts the SAME reference from both sides, so a noisy reference manufactures
    correlation out of nothing: a predictor that reads no patient information at all scores
    r = +0.44 when the reference class is uninformative. Measured, not feared.

    The correlation is therefore PARTIAL -- pred against obs with the reference regressed out of
    both -- which returns the same predictor to -0.004 while leaving a genuine reasoner at
    +0.98. Even so this is a residual-correlation diagnostic, reported as exploratory. The
    primary test of whether the model adds anything beyond the evidence it was given is
    `delta_adapt` in score_anchored_pfs_v1: its skill against the anchor curve sampled the same
    number of times, which is a proper scoring rule and handles censoring.
    """
    f = frame.dropna(subset=["anchor_km_median"])
    f = f[f["pfs_event"] == 1]
    if len(f) < 20:
        return {"n": int(len(f)), "r": float("nan"), "reading": "too few"}
    pred = f["pfs_samples"].apply(lambda s: float(np.median(s))).to_numpy()
    ref = f["anchor_km_median"].to_numpy(dtype=float)
    obs = f["pfs_mos"].to_numpy(dtype=float)
    model_dev = pred - ref
    naive = (float(np.corrcoef(model_dev, obs - ref)[0, 1])
             if np.std(model_dev) > 1e-9 else 0.0)

    def _resid(y: np.ndarray) -> np.ndarray:
        return y - np.polyval(np.polyfit(ref, y, 1), ref)

    rp, ro = _resid(pred), _resid(obs)
    if np.std(rp) < 1e-9 or np.std(ro) < 1e-9:
        return {"n": int(len(f)), "r": 0.0, "r_naive": naive,
                "reading": "model adds nothing beyond the reference class"}
    r = float(np.corrcoef(rp, ro)[0, 1])
    reading = ("adjustment tracks the individual" if r > 0.15 else
               "adjustment is unrelated to the individual" if r > -0.15 else
               "adjustment runs against the individual")
    return {"n": int(len(f)), "r": r, "r_naive": naive,
            "model_adjustment_sd": float(np.std(model_dev)),
            "reading": reading + " (EXPLORATORY — delta_adapt is the primary test)"}


def trust_map(frame: pd.DataFrame, strata: Sequence[str], rng: np.random.Generator) -> dict:
    """PIT pooled, then per stratum. The second is the map; the first can flatter a bad model."""
    frame = frame.copy()
    frame["pit"] = [randomised_pit(s, t, bool(e), rng)
                    for s, t, e in zip(frame["pfs_samples"], frame["pfs_mos"],
                                       frame["pfs_event"], strict=True)]
    out: dict = {"pooled": diagnostics(frame["pit"].to_numpy()),
                 "resolution": resolution(frame),
                 "discrimination": discrimination(frame),
                 "moves_with_reality": (moves_with_reality(frame)
                                        if "anchor_km_median" in frame.columns else {}),
                 "strata": {}}
    for axis in strata:
        if axis not in frame.columns:
            continue
        rows = {}
        for value, g in frame.groupby(axis, sort=True):
            if len(g) >= 15:
                rows[str(value)] = diagnostics(g["pit"].to_numpy())
        if len(rows) >= 2:
            means = [r["mean"] for r in rows.values() if np.isfinite(r["mean"])]
            out["strata"][axis] = {
                "groups": rows,
                # THE DIAGNOSTIC THE POOLED VIEW CANNOT GIVE. A patient-ignoring model is
                # uniform overall and skewed within strata; a wide spread here says the model
                # is calibrated on average and wrong everywhere in particular.
                "mean_spread": float(max(means) - min(means)) if len(means) > 1 else 0.0}
    return out


def report(result: dict, label: str) -> str:
    p = result["pooled"]
    lines = [f"=== {label} ===",
             f"  pooled: n={p['n']}  mean {p['mean']:.3f} (0.50 = unbiased)  "
             f"dispersion {p['dispersion']:.2f} (1.00 = calibrated)  KS {p['ks']:.3f}",
             f"          {p['reading']}"]
    r = result.get("resolution", {})
    if r:
        lines.append(f"  resolution: patient signal {r['patient_signal_sd']:.2f} vs sampling "
                     f"noise {r['within_patient_se']:.2f} "
                     f"(observed spread {r['observed_between_sd']:.2f})  "
                     f"ratio {r['ratio']:.2f} -- {r['reading']}")
    d = result.get("discrimination", {})
    if d and np.isfinite(d.get("concordance", float("nan"))):
        lines.append(f"  discrimination: concordance {d['concordance']:.3f} "
                     f"(0.50 = random ordering of patients)")
    mv = result.get("moves_with_reality") or {}
    if mv and np.isfinite(mv.get("r", float("nan"))):
        lines.append(f"  moves with reality: r = {mv['r']:+.3f} on n={mv['n']} "
                     f"-- {mv['reading']}")
    for axis, block in result["strata"].items():
        spread = block["mean_spread"]
        flag = "  <- CONDITIONAL MISCALIBRATION" if spread > 0.15 else ""
        lines.append(f"  by {axis}: spread of stratum means = {spread:.3f}{flag}")
        for value, r in block["groups"].items():
            lines.append(f"      {value:<28} n={r['n']:<5} mean {r['mean']:.3f}  "
                         f"disp {r['dispersion']:.2f}   {r['reading']}")
    return "\n".join(lines)


# =======================================================================================
# SELF-TEST — generators whose failure mode is known in advance
# =======================================================================================


def _synthetic(kind: str, n: int, m: int, rng: np.random.Generator) -> pd.DataFrame:
    """Patients with a real outcome, and a fan built to be right or wrong in a stated way."""
    # each patient has their own true scale; a good model knows it, a lazy one does not
    scale = rng.uniform(3.0, 18.0, n)
    truth = rng.exponential(scale)
    # what "patients like this one" do: the reference class the model is given
    ref = scale * np.log(2.0)
    if kind == "flat_vs_noisy_reference":
        ref = rng.uniform(3.0, 18.0, n)      # a reference unrelated to this patient's outcome
    rows = []
    for i in range(n):
        if kind == "calibrated":
            draws = rng.exponential(scale[i], m)
        elif kind == "optimistic":
            draws = rng.exponential(scale[i] * 2.0, m)          # imagines longer than reality
        elif kind == "overconfident":
            draws = rng.normal(scale[i], scale[i] * 0.05, m)    # fan far too tight
        elif kind == "hedging":
            draws = rng.exponential(scale[i], m) * rng.uniform(0.2, 5.0, m)
        elif kind == "patient_ignoring":
            # THE ONE THAT MATTERS: the population marginal for everyone, patient unread
            draws = rng.exponential(np.mean(scale), m)
        elif kind == "flat_vs_noisy_reference":
            # THE COUNTEREXAMPLE THAT BROKE THE NAIVE STATISTIC. One prediction for everybody,
            # against a reference class that carries no information about the outcome. The
            # naive correlation manufactures r = +0.44 out of subtracting the same noisy
            # reference from both sides; the partial correlation must return ~0.
            draws = np.full(m, float(np.mean(truth)))
        elif kind == "copier":
            # emits the reference class back, with noise but no patient signal of its own
            draws = np.maximum(rng.normal(ref[i], 1.0, m), 0.1)
        elif kind == "reasoner":
            # moves off the reference class in the direction the individual actually went
            draws = np.maximum(rng.normal(ref[i] + 0.6 * (truth[i] - ref[i]), 2.0, m), 0.1)
        else:
            raise ValueError(kind)
        rows.append({"query_id": f"q{i}", "pfs_samples": np.maximum(draws, 0.1),
                     "pfs_mos": truth[i], "pfs_event": 1.0,
                     "anchor_km_median": ref[i],
                     # the stratum a lazy model cannot get right in both halves at once
                     "prognosis": "good" if scale[i] > 10.5 else "poor"})
    return pd.DataFrame(rows)


def self_test() -> int:
    rng = np.random.default_rng(0)
    n, m = 500, 10

    cal = trust_map(_synthetic("calibrated", n, m, rng), ["prognosis"], rng)
    assert 0.44 < cal["pooled"]["mean"] < 0.56, cal["pooled"]
    assert 0.85 < cal["pooled"]["dispersion"] < 1.15, cal["pooled"]

    opt = trust_map(_synthetic("optimistic", n, m, rng), ["prognosis"], rng)
    assert opt["pooled"]["mean"] < 0.42, f"an over-optimistic fan must pile low: {opt['pooled']}"

    over = trust_map(_synthetic("overconfident", n, m, rng), ["prognosis"], rng)
    assert over["pooled"]["dispersion"] > 1.15, f"a too-narrow fan must pile at both ends: {over['pooled']}"

    hedge = trust_map(_synthetic("hedging", n, m, rng), ["prognosis"], rng)
    assert hedge["pooled"]["dispersion"] < 0.85, f"a too-wide fan must pile centrally: {hedge['pooled']}"

    # THE BLIND SPOT, DEMONSTRATED. A model that never reads the patient is uniform POOLED and
    # badly wrong within strata. If the pooled test were the whole method it would pass this
    # generator, and the C/X/H/A pilot showed a model that behaves very much like it.
    lazy = trust_map(_synthetic("patient_ignoring", n, m, rng), ["prognosis"], rng)
    pooled_ok = abs(lazy["pooled"]["mean"] - 0.5) < 0.10
    spread = lazy["strata"]["prognosis"]["mean_spread"]
    assert spread > 0.15, f"the stratified view must catch a patient-ignoring model: {spread:.3f}"

    # RESOLUTION and DISCRIMINATION, which calibration cannot see
    lazy_res = lazy["resolution"]
    assert lazy_res["ratio"] < 0.5, f"a one-fan-for-everybody model must show no resolution: {lazy_res}"
    cal_res = cal["resolution"]
    assert cal_res["ratio"] > 1.0, f"a patient-specific model must show high resolution: {cal_res}"
    # 0.55, not 0.7. The synthetic outcome is a single exponential draw, so even a model that
    # knows each patient's true rate exactly cannot order the observed times much better than
    # this -- the ceiling is set by the outcome's own randomness, not by the forecaster. Real
    # survival C-indices sit near 0.65-0.70 for the same reason.
    assert cal["discrimination"]["concordance"] > 0.55, cal["discrimination"]
    assert lazy["discrimination"]["concordance"] < 0.55, (
        f"a patient-ignoring model must not order patients: {lazy['discrimination']}")

    # EXPLORATORY DIAGNOSTIC. These fixtures prove the partial residual calculation responds to
    # planted signal; `delta_adapt` remains the proper-score primary test on real outcomes.
    copier = trust_map(_synthetic("copier", n, m, rng), ["prognosis"], rng)
    reasoner = trust_map(_synthetic("reasoner", n, m, rng), ["prognosis"], rng)
    assert copier["moves_with_reality"]["r"] < 0.15, (
        f"a copier must not track the individual: {copier['moves_with_reality']}")
    assert reasoner["moves_with_reality"]["r"] > 0.4, (
        f"a reasoner must track the individual: {reasoner['moves_with_reality']}")

    # THE CASE THE FIRST VERSION PASSED AND SHOULD NOT HAVE
    flat = trust_map(_synthetic("flat_vs_noisy_reference", n, m, rng), ["prognosis"], rng)
    mw = flat["moves_with_reality"]
    assert abs(mw["r"]) < 0.15, (
        f"a constant predictor against an uninformative reference must not score: {mw}")
    print(f"   constant predictor, uninformative reference: naive r = {mw['r_naive']:+.3f} "
          f"-> partial r = {mw['r']:+.3f} (the artefact, removed)")
    print(f"   copier r = {copier['moves_with_reality']['r']:+.3f}, "
          f"reasoner r = {reasoner['moves_with_reality']['r']:+.3f} "
          "— the reference-class test separates them")
    print("self-test OK — the transform reads an optimistic fan, a too-narrow fan and a\n"
          f"   too-wide fan correctly, AND catches a model that never reads the patient\n"
          f"   (pooled mean {lazy['pooled']['mean']:.3f}"
          f"{' — passes the pooled test' if pooled_ok else ''}, stratum spread {spread:.3f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=Path)
    ap.add_argument("--scope-dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test or not args.generations:
        return self_test()

    rng = np.random.default_rng(SEED)

    gens = pd.read_json(args.generations, lines=True)
    gens = gens[~gens.get("refused", False).astype(bool)]
    # THE CANONICAL TABLE. query_id is `record_id@<positional index>`, so rebuilding the row
    # sequence here rather than calling the builder produced ids that silently resolved to a
    # different decision for the same patient -- 134 of 500 unmatched and 70 mis-joined on the
    # definitive run. Whatever turns a query_id back into an outcome must call decision_table.
    import build_anchored_pfs_arms_v1 as Bld
    truth = Bld.decision_table()
    keep = ["query_id", "pfs_mos", "pfs_event", "cancer", "line_number"]
    gens = gens.merge(truth[[c for c in keep if c in truth.columns]], on="query_id", how="inner")

    if args.scope_dir:
        files = sorted(args.scope_dir.glob("prompts_*.jsonl"))
        if files:
            audit = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
            # anchor_km_median is NOT optional: `moves_with_reality` needs the reference class.
            # Omitting it here made that diagnostic
            # return nan silently -- which is precisely the failure mode the preflight exists
            # to catch, and it caught this one before any generation was paid for.
            # STRATIFY ON THE QUERY'S OWN EVIDENCE, NOT THE DONOR'S. R6 borrows another
            # query's block by derangement, so its n_anchors and outcome_dispersion describe
            # somebody else's neighbourhood. Conditional calibration asks "how well does the
            # model do where THIS patient's evidence is thin", so the support fields are taken
            # from the R4 row throughout, and only the reference median follows the block the
            # model actually saw.
            own = audit[audit["arm"] == "R4_same_action"]
            own_cols = [c for c in ("query_id", "support_label", "n_anchors",
                                    "outcome_dispersion") if c in own.columns]
            gens = gens.merge(own[own_cols].drop_duplicates("query_id"),
                              on="query_id", how="left")
            if "anchor_km_median" in audit.columns:
                gens = gens.merge(
                    audit[["query_id", "arm", "anchor_km_median"]]
                    .drop_duplicates(["query_id", "arm"]), on=["query_id", "arm"], how="left")
    if "outcome_dispersion" in gens.columns:
        gens["roughness"] = pd.qcut(gens["outcome_dispersion"], 2,
                                    labels=["smooth", "rough"], duplicates="drop")

    axes = [a for a in ("support_label", "roughness", "cancer", "line_number")
            if a in gens.columns]
    out_lines, payload = [], {}
    for arm, g in gens.groupby("arm", sort=True):
        result = trust_map(g, axes, rng)
        payload[str(arm)] = result
        out_lines.append(report(result, str(arm)))
    text = "\n\n".join(out_lines)
    print(text)
    out = args.out or (args.generations.parent / "trust_map.txt")
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
