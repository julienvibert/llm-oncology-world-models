#!/usr/bin/env python3
"""Read-only audit: EGFR alterations and treatment changes before first-line progression.

Reads accepted materialized trajectories of the three EGFR v1c runs; scored set and
endpoint taken from the outputs/*/phase1_endpoint_long.csv used by Fig. 3e,f.
Writes outputs/exploratory_egfr_pre_progression_audit/ (synthetic data only).

Run: uv run python scripts/explore/audit_egfr_pre_progression_v1.py
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / "outputs" / "exploratory_egfr_pre_progression_audit"
OUT.mkdir(parents=True, exist_ok=True)
MONTH = 30.4375  # generated times, as in the scored endpoint tables

MODELS = {
    "claude_sonnet5": (["interventional_phase1_sonnet_full_v1c"], "exploratory_interventional_phase1"),
    "kimi_k3_off": (["interventional_phase1_kimi_k3_off_full_v1c",
                     "interventional_phase1_kimi_k3_off_topup75_v1c_retry2"],
                    "exploratory_interventional_phase1_kimi_k3_combined"),
    "glm52_fp8": (["interventional_phase1_glm52_full_v1c"], "exploratory_interventional_phase1_glm52_full"),
}
RETRY_SUFFIX = "__topup_retry2"

SENSITIZING = re.compile(
    r"L858R|exon\s*19|ex19|E746|del19|19\s*del|exon\s*20\s*ins|ex20ins|insertion|G719|L861Q|S768I"
    r"|activating|sensiti[sz]ing|exon\s*21|exon\s*18", re.I)
RESISTANCE = re.compile(r"T790M|C797|L718|G724|L792|G796", re.I)


def d(s: str | None) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10]) if s else None
    except ValueError:
        return None


def egfr_class(text: str) -> str:
    if RESISTANCE.search(text):
        return "resistance"
    if SENSITIZING.search(text):
        return "sensitizing"
    if re.search(r"amplif", text, re.I):
        return "amplification"
    return "other"


def load_materialized(run: str, traj_dir: str) -> dict:
    base = ROOT / "runs" / run / "trajectories" / traj_dir
    res = json.loads((base / "trajectory_result.json").read_text())
    assert res["accepted"], traj_dir
    k = res["attempts_to_accept"]
    p = base / f"attempt_{k:02d}" / "materialized_trajectory.json"
    return json.loads(p.read_text())


def components(reg: dict | None) -> list[str]:
    if not reg:
        return []
    return [str(c).strip().lower() for c in (reg.get("components") or [])]


def audit(m: dict) -> dict:
    ledger = m["ledger"]
    # line-1 start: earliest start/switch event with line 1
    starts = [(d(e["date"]), e) for e in ledger
              if e["type"] in ("systemic_treatment_start", "systemic_treatment_switch") and e.get("line") == 1]
    start_date, start_ev = min(starts, key=lambda x: x[0])
    tx1 = start_ev.get("treatment_id")
    l1_agents = set(components(start_ev.get("regimen")))
    end_date, end_type = None, None
    for e in ledger:
        ed = d(e["date"])
        if ed and ed > start_date and e["type"] in ("progression", "death", "last_followup"):
            end_date, end_type = ed, e["type"]
            break
    pfs_days = (end_date - start_date).days if end_date else None

    def before(e: dict) -> bool:
        ed = d(e.get("date"))
        return ed is not None and ed > start_date and (end_date is None or ed < end_date)

    def same_day(e: dict) -> bool:
        return end_date is not None and d(e.get("date")) == end_date

    egfr_hits: list[tuple[str, str]] = []  # (class, text) before end
    egfr_sameday: list[tuple[str, str]] = []
    for e in ledger:
        if e["type"] == "clonal_evolution_event":
            text = " ".join(str(e.get(k) or "") for k in ("gene", "alteration", "variant"))
            if "EGFR" in text.upper():
                tgt = egfr_hits if before(e) else egfr_sameday if same_day(e) else None
                if tgt is not None:
                    tgt.append((egfr_class(text), (e.get("alteration") or text).strip()))
    for a in m.get("acquired_alterations") or []:
        text = " ".join(str(a.get(k) or "") for k in ("gene", "alteration", "variant"))
        if "EGFR" in text.upper():
            fp = d(a.get("first_presence_date"))
            if fp and fp > start_date and (end_date is None or fp < end_date):
                egfr_hits.append((egfr_class(text), (a.get("alteration") or text).strip()))
    # clones registered before end
    for c in m.get("final_clone_registry") or []:
        fp = d(c.get("first_presence_date"))
        drivers = " ".join(map(str, c.get("drivers") or []))
        if "EGFR" in drivers.upper() and fp and (end_date is None or fp < end_date):
            egfr_hits.append((egfr_class(drivers), drivers))
    # baseline molecular profile (should be EGFR-free in WT)
    mp = m["baseline"]["identity_card"].get("molecular_profile") or {}
    baseline_egfr = [a.get("alteration") for a in mp.get("alterations") or []
                     if "EGFR" in json.dumps(a).upper()]

    new_agents: set[str] = set()
    tx_events: list[str] = []
    stop_l1 = False
    stop_l1_detail: list[str] = []
    holds = 0
    for e in ledger:
        if not before(e):
            continue
        t = e["type"]
        if t in ("systemic_treatment_start", "systemic_treatment_switch") and e is not start_ev:
            comps = components(e.get("regimen"))
            extra = [c for c in comps if c not in l1_agents]
            tx_events.append(f"{t}:{'+'.join(comps)}")
            new_agents.update(extra)
            if t == "systemic_treatment_switch" and e.get("stopped_treatment_id") == tx1:
                stop_l1 = True
                stop_l1_detail.append(f"switch({e.get('reason')})")
        elif t == "systemic_treatment_stop" and e.get("stopped_treatment_id") == tx1:
            stop_l1 = True
            stop_l1_detail.append(f"stop({e.get('reason')})")
        elif t == "systemic_treatment_modification":
            mt = e.get("modification_type")
            if mt == "dose_hold":
                holds += 1
            if mt == "discontinuation_of_component":
                aff = str(e.get("affected_components") or "").lower()
                tx_events.append(f"discontinue_component:{aff}")
                if "osimertinib" in aff:
                    stop_l1 = True
                    stop_l1_detail.append("discontinue_component(osimertinib)")
    sens = any(c == "sensitizing" for c, _ in egfr_hits)
    return dict(
        pfs_days_recomputed=pfs_days, end_type=end_type,
        egfr_any=bool(egfr_hits), egfr_sensitizing=sens,
        egfr_texts="; ".join(sorted({f"{c}:{t}" for c, t in egfr_hits})),
        egfr_sameday_texts="; ".join(sorted({f"{c}:{t}" for c, t in egfr_sameday})),
        baseline_egfr="; ".join(map(str, baseline_egfr)),
        new_agent=bool(new_agents), new_agents=";".join(sorted(new_agents)),
        tx_events="|".join(tx_events), stop_l1=stop_l1, stop_l1_detail=";".join(stop_l1_detail),
        dose_holds=holds, l1_regimen="+".join(sorted(l1_agents)),
    )


def main() -> None:
    rows = []
    for model, (runs, outdir) in MODELS.items():
        ep = pd.read_csv(ROOT / "outputs" / outdir / "phase1_endpoint_long.csv")
        ep = ep[ep.arm_key.isin(["osimertinib", "chemo"])]
        for _, r in ep.iterrows():
            tid = r.trajectory_id
            if tid.endswith(RETRY_SUFFIX):
                run, tdir = runs[1], tid[: -len(RETRY_SUFFIX)]
            else:
                run, tdir = runs[0], tid
            a = audit(load_materialized(run, tdir))
            rows.append(dict(model=model, context=r.context, arm=r.arm_key, trajectory_id=tid,
                             endpoint_mo=r.endpoint_days / MONTH, pfs_line_1_days=r.pfs_line_1_days, **a))
    df = pd.DataFrame(rows)
    df["pfs_match"] = df.pfs_days_recomputed == df.pfs_line_1_days
    expected = {"osimertinib": "osimertinib", "chemo": "carboplatin+pemetrexed"}
    df["off_protocol_l1"] = df.l1_regimen != df.arm.map(expected)
    df["clean_ab"] = ~(df.egfr_sensitizing | df.new_agent)
    # (c) applied to the osimertinib arm only (chemo induction completion -> maintenance is expected)
    df["clean_abcd"] = df.clean_ab & ~df.off_protocol_l1 & ~(df.stop_l1 & df.arm.eq("osimertinib"))
    df["clean_strict"] = df.clean_abcd & ~df.egfr_any
    df["clean_ab_anyegfr"] = ~(df.egfr_any | df.new_agent)
    df["clean_abc"] = df.clean_ab & ~df.stop_l1
    df.to_csv(OUT / "per_future_audit.csv", index=False)

    print("PFS recompute mismatches:", int((~df.pfs_match).sum()))
    g = df.groupby(["model", "context", "arm"])
    counts = g.agg(n=("trajectory_id", "size"), egfr_any=("egfr_any", "sum"),
                   egfr_sens=("egfr_sensitizing", "sum"), new_agent=("new_agent", "sum"),
                   stop_l1=("stop_l1", "sum"), off_protocol=("off_protocol_l1", "sum"),
                   n_clean_abcd=("clean_abcd", "sum"), n_clean_strict=("clean_strict", "sum"), n_clean_ab=("clean_ab", "sum"),
                   n_clean_abc=("clean_abc", "sum"),
                   med_all=("endpoint_mo", "median"))
    for col in ["clean_ab", "clean_ab_anyegfr", "clean_abc", "clean_abcd", "clean_strict"]:
        counts[f"med_{col}"] = df[df[col]].groupby(["model", "context", "arm"]).endpoint_mo.median()
    counts.to_csv(OUT / "arm_counts_medians.csv")
    pd.set_option("display.width", 250)
    print(counts.round(2).to_string())
    diffs = []
    for (model, ctx), sub in counts.groupby(level=[0, 1]):
        s = sub.droplevel([0, 1])
        diffs.append(dict(model=model, context=ctx,
                          **{f"diff_{k}": s.loc["osimertinib", k] - s.loc["chemo", k]
                             for k in ["med_all", "med_clean_ab", "med_clean_ab_anyegfr", "med_clean_abc", "med_clean_abcd", "med_clean_strict"]}))
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(OUT / "osi_minus_chemo.csv", index=False)
    print(diffs.round(2).to_string())
    for (model, ctx, arm), sub in df.groupby(["model", "context", "arm"]):
        ag = pd.Series(";".join(sub.new_agents).split(";"))
        ag = ag[ag != ""].value_counts().to_dict()
        eg = pd.Series("; ".join(sub.egfr_texts).split("; "))
        eg = eg[eg != ""].value_counts().to_dict()
        sd = pd.Series("; ".join(sub.egfr_sameday_texts).split("; "))
        sd = sd[sd != ""].value_counts().to_dict()
        st = pd.Series(";".join(sub.stop_l1_detail).split(";"))
        st = st[st != ""].value_counts().to_dict()
        print(f"{model} {ctx} {arm}: new agents={ag}\n   egfr pre-prog={eg}\n   egfr same-day={sd}"
              f"\n   l1 stops={st}\n   baseline EGFR={sub.baseline_egfr.replace('', pd.NA).dropna().value_counts().to_dict()}")


if __name__ == "__main__":
    main()
