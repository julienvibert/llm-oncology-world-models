#!/usr/bin/env python3
"""Build the full sentinel CF reasoning battery from constructed seed cards.

Turns docs/sentinel_seed_cards_v1.yaml (clinician-reviewed constructed cards) +
sentinel_benchmark_spec_v1.yaml (contexts / arms / scoring) into a launch-ready CF
generation config. Emits config/seeds/manifest only — it does NOT launch paid work.

Design invariants (enforced here, not just documented):
  * SEEDS ARE SYNTHETIC (constructed) -> no PHI -> git-ok. Each card is lowered into
    a partial BaselineObject seed and validated with `validate_baseline_seed`.
  * NO ANSWER LEAKAGE. The model sees only patient facts + a forced treatment (a real
    regimen name). All scoring labels (soc/wrong_target/sensitive/...) live in
    arm_manifest.json, NEVER in a prompt-facing string. A lint enforces this and the
    build ABORTS on any leak.
  * HISTORY cards (HGSOC relapse, post-osi EGFR) are wired as OBSERVED PREFIXES
    (conditioning.fixed_events, origin=observed_prefix) — prior events the model
    conditions on, not baseline mutations — and validated with `validate_observed_prefix`.
    The model infers platinum-sensitivity / the 2L setting; we never state it.

Run:  uv run python scripts/explore/build_sentinel_battery.py           # dry build + validate + render
      uv run python scripts/explore/build_sentinel_battery.py --out-dir /private/tmp/sentinel_battery
"""

from __future__ import annotations

import argparse
from datetime import date
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from synthonccases.oncollm.condition_lower import lower_conditioning_cell  # noqa: E402
from synthonccases.oncollm.conditioning_audit import (  # noqa: E402
    BaselineObject,
    validate_baseline_seed,
    validate_observed_prefix,
)
from synthonccases.oncollm.poc_runner import load_config  # noqa: E402

CONFIG_ID = "sentinel_reasoning_battery_v1"
SEED_CARDS = ROOT / "docs" / "sentinel_seed_cards_v1.yaml"
SPEC = Path(__file__).resolve().parent / "sentinel_benchmark_spec_v1.yaml"
DEFAULT_OUT = Path(f"/private/tmp/{CONFIG_ID}")
DIAGNOSIS_DATE = date(2018, 3, 1)  # neutral, well inside every model's knowledge window

MODEL_LIBRARY = {
    "gemini31_lite": {"id": "google/gemini-3.1-flash-lite", "short": "gemini31_lite"},
    "qwen36_flash": {"id": "qwen/qwen3.6-flash", "short": "qwen36_flash"},
    "gemma4": {"id": "google/gemma-4-31b-it", "short": "gemma4"},
    "claude_sonnet5": {"id": "anthropic/claude-sonnet-5", "short": "claude_sonnet5"},
    "glm52_fp8": {"id": "GLM-5.2-FP8", "short": "glm52_fp8"},
    # Added 2026-08-05 as the BUILD/DEBUG target. Roughly $0.09-0.18 per 1M tokens on
    # OpenRouter, which at ~10k output tokens per trajectory is ~$0.002 per future — about
    # 500x cheaper than Opus. The whole 416-anchor PRIME design costs a couple of dollars
    # here, so every pipeline defect should be found on this model and never on a paid one.
    #
    # THINKING MODE IS UNRESOLVED. DeepSeek V4 exposes thinking and non-thinking modes and
    # the choice is an experimental variable, not a detail — the roster already distinguishes
    # `kimi_k3_off` for reasoning-off. Pick one, record it in the manifest, and do not mix
    # modes within a run. Non-thinking is the sensible default for a debug target.
    "deepseek_v4_flash": {"id": "deepseek/deepseek-v4-flash", "short": "deepseek_v4_flash"},
    "deepseek_v4_flash_0731": {
        "id": "deepseek/deepseek-v4-flash-0731",
        "short": "deepseek_v4_flash_0731",
    },
}
DEFAULT_MODELS = ["gemini31_lite", "qwen36_flash", "gemma4", "claude_sonnet5"]

# Approx per-accepted-future cost from prior runs (July 2026): cheap ~$0.0085,
# Sonnet ~$0.243. Used only for the launch estimate.
# Measured on the PRIME smoke, 2026-08-05: Gemma $0.00265/trajectory, DeepSeek V4 Flash 0731
# $0.00471. The DeepSeek cost included reasoning tokens (see _OPENROUTER_REASONING_MODELS);
# with reasoning off it is expected to fall toward Gemma's.
COST_PER_FUTURE = {"gemini31_lite": 0.0085, "qwen36_flash": 0.0085, "gemma4": 0.00265,
                   "claude_sonnet5": 0.243, "glm52_fp8": 0.0,
                   "deepseek_v4_flash": 0.0047, "deepseek_v4_flash_0731": 0.0047}

# Roster membership for DeepSeek V4 Flash is decided by RELIABILITY, not by its results.
# Prespecified criterion, set 2026-08-05 before any call: it joins the reported roster only
# if it reaches >= 90% accepted futures within max_attempts on a smoke batch, with valid
# structured output. Deciding after seeing whether its science looks good would be selecting
# a model on its result, which is the thing this project forbids everywhere else.
ROSTER_ADMISSION_ACCEPTANCE_MIN = 0.90

# ---- reviewer-bucket -> schema enum (primary_driver_bucket is a validated leaf) ----
DRIVER_BUCKET = {"her2": "erbb2_her2", "none": "no_actionable_driver",
                 "msi": "mmr_msi", "mmr": "mmr_msi"}
HISTOLOGY_BUCKET = {"LUAD": "adenocarcinoma", "COAD": "adenocarcinoma", "SKCM": "melanoma",
                    "IDC": "ductal_carcinoma", "CCRCC": "renal_cell_carcinoma",
                    "HGSOC": "serous_carcinoma", "PAAD": "pancreatic_ductal_adenocarcinoma"}
SITE_CLASS = {"LUAD": "lung", "COAD": "colorectal", "SKCM": "skin", "IDC": "breast",
              "CCRCC": "kidney", "HGSOC": "gynecologic_ovary_fallopian_peritoneal",
              "PAAD": "pancreatobiliary"}
STAGE_SYSTEM = {"HGSOC": "figo"}  # ovarian is FIGO; everything else ajcc_tnm_8
# plausible dominant metastatic site per tumour (label, site_class enum) for the fixed burden
MET_SITE = {"LUAD": ("liver", "liver"), "COAD": ("liver", "liver"), "SKCM": ("liver", "liver"),
            "IDC": ("bone", "bone"), "CCRCC": ("lung", "lung"),
            "HGSOC": ("peritoneum", "peritoneum"), "PAAD": ("liver", "liver")}

# ---- arm registry: arm_key -> (regimen label the PROMPT sees, components) ----
# `chemo` is tumour-resolved. Labels are real regimen names — never scoring labels.
ARM_REGIMEN: dict[str, tuple[str, tuple[str, ...]]] = {
    "osimertinib": ("osimertinib", ("osimertinib",)),
    "alectinib": ("alectinib", ("alectinib",)),
    "pembrolizumab": ("pembrolizumab", ("pembrolizumab",)),
    "continue_osi": ("osimertinib", ("osimertinib",)),
    "chemo_io": ("carboplatin + pemetrexed + pembrolizumab",
                 ("carboplatin", "pemetrexed", "pembrolizumab")),
    "carbo_pem": ("carboplatin + pemetrexed", ("carboplatin", "pemetrexed")),
    "chemo_bev": ("FOLFOX + bevacizumab", ("fluorouracil", "oxaliplatin", "bevacizumab")),
    "cetuximab": ("FOLFIRI + cetuximab", ("fluorouracil", "irinotecan", "cetuximab")),
    "braf_mek": ("dabrafenib + trametinib", ("dabrafenib", "trametinib")),
    "io_combo": ("nivolumab + ipilimumab", ("nivolumab", "ipilimumab")),
    "her2_chemo": ("trastuzumab + pertuzumab + docetaxel",
                   ("trastuzumab", "pertuzumab", "docetaxel")),
    "endocrine": ("letrozole + palbociclib", ("letrozole", "palbociclib")),
    "io_tki": ("pembrolizumab + axitinib", ("pembrolizumab", "axitinib")),
    "tki": ("sunitinib", ("sunitinib",)),
    "cytotoxic_chemo": ("gemcitabine + doxorubicin", ("gemcitabine", "doxorubicin")),
    "platinum": ("carboplatin + paclitaxel", ("carboplatin", "paclitaxel")),
    "non_platinum": ("pegylated liposomal doxorubicin", ("doxorubicin",)),
    "parp": ("olaparib", ("olaparib",)),
    "folfirinox": ("FOLFIRINOX", ("fluorouracil", "irinotecan", "oxaliplatin", "leucovorin")),
    "gem_nab": ("gemcitabine + nab-paclitaxel", ("gemcitabine", "nab-paclitaxel")),
    "gem": ("gemcitabine", ("gemcitabine",)),
    "supportive": ("best supportive care", ()),
}
CHEMO_BY_CODE = {"LUAD": ("carboplatin + pemetrexed", ("carboplatin", "pemetrexed")),
                 "COAD": ("FOLFOX", ("fluorouracil", "oxaliplatin", "leucovorin")),
                 "IDC": ("paclitaxel", ("paclitaxel",)),
                 "SKCM": ("dacarbazine", ("dacarbazine",))}

# tokens that must NEVER appear in a prompt-facing string (leakage lint)
# scoring/answer labels that must never reach the prompt. NB: `must_not_occur` is a
# constraint MECHANISM (enforcement), not an answer, so it is deliberately not listed.
LEAK_TOKENS = ("soc", "wrong_target", "wrong_here", "wrong_setting", "weak_io", "weak_chemo",
               "weak", "bsc", "not_top", "acceptable_top", "platinum-sensitive",
               "platinum-resistant", "sensitive", "resistant", "negative control", "wrong target")


def _as_plain_dict(value: object) -> dict:
    """Pydantic v1/v2/dict tolerant conversion for preflight-only checks."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)  # type: ignore[arg-type]


def validate_replay_ready_seed(seed: dict, *, label: str) -> None:
    """Mirror replay's baseline assumptions so bad seeds fail before paid calls."""
    baseline = BaselineObject.model_validate(seed)
    b = _as_plain_dict(baseline)
    clone_ids = [entry["clone_id"] for entry in b.get("clone_registry", [])]
    expected = [f"C{i}" for i in range(len(clone_ids))]
    if clone_ids != expected:
        raise SystemExit(
            f"[fatal] {label}: clone_registry must use replay-ready contiguous IDs "
            f"{expected}; got {clone_ids}"
        )
    clone_set = set(clone_ids)
    for i, burden in enumerate(b.get("baseline_disease_burden", []), start=1):
        clone = burden.get("clone")
        if clone not in clone_set:
            raise SystemExit(
                f"[fatal] {label}: baseline_disease_burden[{i}] references unknown "
                f"clone {clone!r}; known clones={sorted(clone_set)}"
            )
    for i, entry in enumerate(b.get("clone_registry", []), start=1):
        parent = entry.get("parent_clone_id_or_null")
        if parent is not None and parent not in clone_set:
            raise SystemExit(
                f"[fatal] {label}: clone_registry[{i}] has unknown parent clone "
                f"{parent!r}; known clones={sorted(clone_set)}"
            )


def _alteration(a: dict) -> dict:
    """Reviewer alteration -> full schema Alteration (infer type; fill required leaves)."""
    txt = str(a.get("alteration", "")).lower()
    db = a.get("driver_bucket", "")
    if txt in ("none", "wild-type", "wild type") or (db == "none" and txt == "none"):
        atype = "wild_type"
    elif "deletion" in txt:
        atype = "deletion"
    elif "fusion" in txt:
        atype = "fusion"
    elif "amplification" in txt or txt.endswith(" amp"):
        atype = "amplification"
    elif "loss" in txt:
        atype = "copy_number_loss"
    else:
        atype = "mutation"
    act = a.get("actionability") or "no_known_actionability"
    if act in ("none", "prognostic"):
        act = "no_known_actionability" if act == "none" else "prognostic_only"
    return {"gene": a["gene"], "alteration_type": atype,
            "alteration": a.get("alteration") or "wild-type", "variant": a.get("variant"),
            "driver_bucket": DRIVER_BUCKET.get(db, db) or "no_actionable_driver",
            "actionability": act, "is_primary_driver": bool(a.get("is_primary_driver", False))}


def _mo(months: int) -> date:
    y, m = divmod(DIAGNOSIS_DATE.month - 1 + months, 12)
    return date(DIAGNOSIS_DATE.year + y, m + 1, DIAGNOSIS_DATE.day)


def _disease_delta(card: dict, *, state: str) -> list[dict]:
    """Replay-ready disease deltas for constructed observed-prefix events."""
    code = card["diagnosis"]["oncotree_code"]
    met_site, met_class = MET_SITE.get(code, ("other", "other"))
    if state == "response":
        return [
            {"clone": "C0", "site": "primary tumour", "site_class": "primary_site",
             "burden": "small", "kinetic": "shrinking"},
            {"clone": "C0", "site": f"{met_site} metastases", "site_class": met_class,
             "burden": "small", "kinetic": "shrinking"},
        ]
    if state == "progression":
        return [
            {"clone": "C0", "site": f"{met_site} metastases", "site_class": met_class,
             "burden": "moderate", "kinetic": "growing"},
        ]
    raise ValueError(f"unknown disease-delta state {state!r}")


def arm_regimen(arm_key: str, oncotree: str) -> tuple[str, tuple[str, ...]]:
    if arm_key == "chemo":
        return CHEMO_BY_CODE.get(oncotree, ("platinum-doublet chemotherapy", ("carboplatin",)))
    if arm_key not in ARM_REGIMEN:
        raise SystemExit(f"[fatal] arm {arm_key!r} has no regimen mapping")
    return ARM_REGIMEN[arm_key]


def lower_seed(card: dict) -> dict:
    """Reviewer card -> partial BaselineObject seed (identity_card + light history)."""
    code = card["diagnosis"]["oncotree_code"]
    demo = card["demographics"]
    mol = card["molecular_profile"]
    bh = card["baseline_health"]
    pdb = mol.get("primary_driver_bucket", "none")
    ident = {
        "demographics": {"age_at_diagnosis": int(demo["age_at_diagnosis"]),
                         "sex": demo["sex"], "ecog_ps_at_diagnosis": int(demo["ecog_ps_at_diagnosis"])},
        "diagnosis": {"diagnosis_date": DIAGNOSIS_DATE.isoformat(), "oncotree_code": code,
                      "histology_class": card["diagnosis"]["histology_class"],
                      "histology_bucket": HISTOLOGY_BUCKET.get(code),
                      "primary_site": card["diagnosis"]["primary_site"],
                      "primary_site_class": SITE_CLASS.get(code)},
        "staging": {"stage_class": "metastatic",
                    "formal_stage": {"system": STAGE_SYSTEM.get(code, "ajcc_tnm_8"), "group_label": "IV"}},
        "molecular_profile": {
            "alterations": [_alteration(a) for a in mol.get("alterations", [])],
            "biomarkers": mol.get("biomarkers", []),
            "primary_driver_bucket": DRIVER_BUCKET.get(pdb, pdb)},
        "baseline_health": {k: bh[k] for k in ("renal", "hepatic", "cardiac", "marrow_reserve",
                                               "pulmonary", "frailty_band")}
        | {"contraindications": [], "autoimmune_disease_history": False,
           "neuropathy_at_baseline": "none"},
    }
    # neutral baseline_id: never encode the answer (sensitive/resistant/...) in a prompt-visible field
    bid = "sentinel_seed_" + hashlib.sha256(card["_id"].encode()).hexdigest()[:10]
    # complete, FIXED baseline blocks — identical across all arms of a card so the LLM never
    # re-imagines burden/weight/clones differently per arm ("same patient, same past").
    alts = ident["molecular_profile"]["alterations"]
    driver_genes = [a["gene"] for a in alts if a["is_primary_driver"]] or [a["gene"] for a in alts] or ["none"]
    clone = "C0"
    met_site, met_class = MET_SITE.get(code, ("other", "other"))
    frail = bh.get("frailty_band") == "frail"
    luad = code == "LUAD"
    return {
        "baseline_id": bid, "identity_card": ident,
        "clone_registry": [{"clone_id": clone, "clone_role": "founder", "drivers": driver_genes,
                            "first_presence_date": DIAGNOSIS_DATE.isoformat(),
                            "parent_clone_id_or_null": None}],
        "baseline_disease_burden": [
            {"clone": clone, "site": "primary tumour", "site_class": "primary_site",
             "burden": "moderate", "kinetic": "growing"},
            {"clone": clone, "site": f"{met_site} metastases", "site_class": met_class,
             "burden": "small", "kinetic": "growing"}],
        "baseline_patient_state": {"weight_kg": 62.0 if frail else 72.0,
                                   "symptom_burden": "moderate" if frail else "mild"},
        "medical_history": {"comorbidities": [], "smoking_status": "former" if luad else "never",
                            "tobacco_exposure_band": "moderate" if luad else "none",
                            "pack_years": 25.0 if luad else 0.0,
                            "family_history": [], "contextual_factors": []}}


def observed_prefix(card: dict) -> dict | None:
    """prior_history prose -> observed-prefix fixed_events. Model INFERS the setting."""
    ph = card.get("prior_history")
    if not ph:
        return None
    if "time_to_progression_on_osi_months" in ph:  # post-osi EGFR
        ttp = int(ph["time_to_progression_on_osi_months"])
        events = [
            {"type": "systemic_treatment_start", "date": DIAGNOSIS_DATE.isoformat(),
             "regimen": {"label": "osimertinib", "components": ["osimertinib"]}},
            {"type": "response", "date": _mo(2).isoformat(), "disease": _disease_delta(card, state="response")},
            {"type": "progression", "date": _mo(ttp).isoformat(),
             "disease": _disease_delta(card, state="progression")},
            {"type": "systemic_treatment_stop", "date": _mo(ttp).isoformat(), "reason": "progression"},
        ]
        cutoff = _mo(ttp)
    else:  # HGSOC relapse; PFI = months from platinum stop to relapse
        pfi = int(ph["platinum_free_interval_months"])
        stop_mo, relapse_mo = 5, 5 + pfi
        events = [
            {"type": "systemic_treatment_start", "date": DIAGNOSIS_DATE.isoformat(),
             "regimen": {"label": "carboplatin + paclitaxel", "components": ["carboplatin", "paclitaxel"]}},
            {"type": "response", "date": _mo(3).isoformat(), "disease": _disease_delta(card, state="response")},
            {"type": "systemic_treatment_stop", "date": _mo(stop_mo).isoformat()},
            {"type": "progression", "date": _mo(relapse_mo).isoformat(),
             "disease": _disease_delta(card, state="progression")},
        ]
        cutoff = _mo(relapse_mo)
    return {"origin": "observed_prefix", "cutoff_date": cutoff.isoformat(), "events": events}


def fixed_intervention_prefix(prefix: dict | None, *, label: str, components: tuple[str, ...]) -> dict:
    """Append the assigned treatment as a fixed counterfactual intervention.

    The model should not choose or emit the assigned arm. It receives the full
    fixed prefix, including the intervention, and generates only events after
    the intervention date.
    """
    if prefix is None:
        intervention_date = DIAGNOSIS_DATE
        events: list[dict] = []
    else:
        intervention_date = date.fromisoformat(prefix["cutoff_date"]) + timedelta(days=1)
        events = list(prefix["events"])
    event = {"type": "systemic_treatment_start", "date": intervention_date.isoformat(),
             "regimen": {"label": label, "components": list(components)}}
    return {"origin": "observed_prefix", "cutoff_date": intervention_date.isoformat(),
            "events": events + [event]}


def validate_replay_ready_prefix(prefix: dict | None, *, label: str) -> None:
    """Observed prefixes are replayed too; enforce generated-event essentials."""
    if prefix is None:
        return
    for i, event in enumerate(prefix.get("events", []), start=1):
        event_type = event.get("type")
        if event_type in {"response", "progression"} and not event.get("disease"):
            raise SystemExit(f"[fatal] {label}: fixed prefix event {i} ({event_type}) "
                             "must include a non-empty disease delta")
        if event_type in {"systemic_treatment_start", "systemic_treatment_switch"}:
            regimen = event.get("regimen", {})
            if not regimen.get("components"):
                raise SystemExit(f"[fatal] {label}: fixed prefix event {i} ({event_type}) "
                                 "must include regimen.components")


def operator_text(label: str, components: tuple[str, ...], *, is_next_line: bool) -> str:
    if not components:
        return ("Management is best supportive / palliative care only - NO systemic anticancer "
                "therapy at any point. Symptoms, disease course, and all later events are unconstrained.")
    line = "next-line" if is_next_line else "first-line"
    return (f"The assigned {line} intervention is fixed in the Fixed Prefix as "
            f"systemic_treatment_start = {label}. Do not choose, omit, or restate that treatment. "
            "Generate only downstream response, toxicity, duration, dose modifications, later "
            "treatment changes, progression, and terminal events after the fixed intervention.")


def _forbid_systemic(cutoff: date | None) -> list[dict]:
    """Hard constraints: no systemic anticancer therapy may occur (post-cutoff if a prior
    course is fixed, else at all). Enforced by the acceptance audit, not just prose."""
    rng = {"date_range": [cutoff.isoformat(), "2035-01-01"]} if cutoff else {}
    return [{"constraint_id": f"forbid_{et}",
             "selector": {"event_type": et, **rng},
             "constraint": {"must_not_occur": True}}
            for et in ("systemic_treatment_start", "systemic_treatment_switch")]


def build_cell(card: dict, arm_key: str, prefix: dict | None) -> tuple[dict, dict]:
    code = card["diagnosis"]["oncotree_code"]
    label, components = arm_regimen(arm_key, code)
    seed = lower_seed(card)
    op = operator_text(label, components, is_next_line=prefix is not None)
    cutoff = date.fromisoformat(prefix["cutoff_date"]) if prefix else None
    conditioning: dict = {"free_text_operator_constraints": op}
    if components:
        conditioning["fixed_events"] = fixed_intervention_prefix(prefix, label=label, components=components)
    else:  # supportive: hard-forbid systemic therapy (validator-enforced, not prose-only)
        conditioning["future_event_constraints"] = _forbid_systemic(cutoff)
        if prefix is not None:
            conditioning["fixed_events"] = prefix
    if components:
        fixed = conditioning["fixed_events"]
        fixed_start = fixed["events"][-1]
        if fixed_start.get("type") != "systemic_treatment_start" or fixed_start.get("regimen", {}).get("components") != list(components):
            raise SystemExit(f"[fatal] {card['_id']}__{arm_key}: active arm intervention is not fixed in prefix")
    elif prefix is not None:
        conditioning["fixed_events"] = prefix
    # policy_id unique per distinct payload: (context, arm) — regimen+line fixed within a context,
    # so `chemo` in LUAD-1L vs LUAD-2L vs breast get distinct ids and no false collision.
    policy_id = f"{card['context']}__{arm_key}"
    cell = {"id": f"{card['_id']}__{arm_key}", "condition_id": card["_id"], "policy_id": policy_id,
            "baseline_seed": seed,  # TOP-LEVEL: what the runner actually pins into the prompt
            "operator_constraints": op, "conditioning": conditioning}
    manifest = {"arm_key": arm_key, "policy_id": policy_id, "regimen_label": label,
                "intervention_fixed_in_prefix": bool(components),
                "context": card["context"], "condition_id": card["_id"]}
    return cell, manifest


def leak_lint(cell: dict) -> list[str]:
    """Return any prompt-facing strings that leak a scoring label (word-boundary match,
    so patient facts like the HGSOC code or a neutral seed id don't trip it)."""
    hay = json.dumps({"op": cell["operator_constraints"], "cond": cell["conditioning"]}).lower()
    return [t for t in LEAK_TOKENS if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", hay)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--accepted-per-cell", type=int, default=3)
    ap.add_argument("--models", nargs="+", choices=sorted(MODEL_LIBRARY), default=DEFAULT_MODELS)
    args = ap.parse_args()

    cards = yaml.safe_load(SEED_CARDS.read_text())["seeds"]
    spec = yaml.safe_load(SPEC.read_text())["contexts"]

    cells, arm_manifest, seed_files, renders, warnings, leaks = [], {}, {}, {}, [], []
    for cid, card in cards.items():
        card = dict(card, _id=cid)
        ctx = card["context"]
        if ctx not in spec:
            warnings.append(f"card {cid}: context {ctx!r} not in spec — skipped")
            continue
        arms = list(spec[ctx].get("arms", {}))
        if not arms:
            warnings.append(f"card {cid}: context {ctx!r} has no arms — skipped")
            continue
        # coherence: a card whose designed SoC isn't an arm in its context is suspicious
        soc = spec[ctx].get("soc")
        if soc and soc not in arms and "seed_variants" not in spec[ctx]:
            warnings.append(f"card {cid}: context {ctx!r} soc={soc!r} not among arms {arms}")
        prefix = observed_prefix(card)
        seed = lower_seed(card)
        BaselineObject.model_validate(seed)          # hard: seed must be a COMPLETE baseline
        validate_baseline_seed(seed)                 # hard: structural path check
        validate_replay_ready_seed(seed, label=cid)  # hard: replay clone/reference contract
        if prefix is not None:
            validate_replay_ready_prefix(prefix, label=cid)
            probe = lower_conditioning_cell(conditioning_payload={
                "baseline_seed": seed, "fixed_events": prefix, "free_text_operator_constraints": ""})
            validate_observed_prefix(probe, baseline_diagnosis_date=DIAGNOSIS_DATE)
        seed_files[cid] = seed
        for arm_key in arms:
            cell, man = build_cell(card, arm_key, prefix)
            fixed = cell["conditioning"].get("fixed_events")
            if fixed is not None:
                validate_replay_ready_prefix(fixed, label=cell["id"])
                probe = lower_conditioning_cell(conditioning_payload={
                    "baseline_seed": seed, "fixed_events": fixed, "free_text_operator_constraints": ""})
                validate_observed_prefix(probe, baseline_diagnosis_date=DIAGNOSIS_DATE)
            found = leak_lint(cell)
            if found:
                leaks.append((cell["id"], found))
            cells.append(cell)
            arm_manifest[cell["id"]] = man | {"role": spec[ctx]["arms"][arm_key]}
        renders[cid] = {"context": ctx, "seed": seed_files[cid],
                        "prefix": prefix, "arms": arms, "sample_operator": cells[-1]["operator_constraints"]}

    if leaks:
        for cid, toks in leaks:
            print(f"[LEAK] {cid}: {toks}")
        raise SystemExit("[fatal] answer-label leakage into prompt-facing text — aborting build")

    models = [MODEL_LIBRARY[m] for m in args.models]
    n_fut = len(cells) * len(models) * args.accepted_per_cell
    cost = sum(len(cells) * args.accepted_per_cell * COST_PER_FUTURE[m["short"]] for m in models)

    out = args.out_dir
    (out / "seeds").mkdir(parents=True, exist_ok=True)
    for cid, seed in seed_files.items():
        (out / "seeds" / f"{cid}.json").write_text(json.dumps(seed, indent=2))
    config = {"schema_version": 1, "config_id": CONFIG_ID,
              "study_reference": {"diagnosis_date": DIAGNOSIS_DATE.isoformat(), "knowledge_cutoff": "2024-12-31"},
              "run_defaults": {"cells_count": len(cells), "accepted_futures_per_cell": args.accepted_per_cell,
                               "max_total_futures_per_cell": args.accepted_per_cell,
                               "max_attempts": 4, "terminal_contract": "death_only"},
              "prompt": {"template_path": "prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md",
                         "enum_order": {"mode": "deterministic_shuffle", "seed": f"{CONFIG_ID}_enum_shuffle_v1"}},
              "provider": {"use_openrouter": True, "response_format": {"type": "json_object"}},
              "models": models, "cells": cells}
    config_path = out / f"{CONFIG_ID}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, width=110))
    # HARD PREFLIGHT: the emitted config must load through the real runner (catches
    # policy_id collisions, condition/seed mismatches, malformed conditioning) BEFORE any spend.
    loaded = load_config(config_path)
    n_loaded = len(getattr(loaded, "cells", []) or [])
    # AND the RESOLVED seed must actually carry the pinned patient (not an empty default) —
    # this is what reaches the prompt. Catches "baseline only under conditioning" silent loss.
    unpinned = []
    replay_bad = []
    for lc in loaded.cells:
        seed = _as_plain_dict(lc.condition.baseline_seed)
        ic = seed.get("identity_card", {})
        if not ic.get("molecular_profile", {}).get("primary_driver_bucket"):
            unpinned.append(lc.id)
        try:
            validate_replay_ready_seed(seed, label=lc.id)
        except SystemExit as exc:
            replay_bad.append(str(exc))
    if unpinned:
        raise SystemExit(f"[fatal] resolved baseline not pinned for {len(unpinned)} cells "
                         f"(e.g. {unpinned[0]}) — the fixed patient is NOT reaching the prompt")
    if replay_bad:
        raise SystemExit(f"[fatal] resolved baseline is not replay-ready for {len(replay_bad)} cells "
                         f"(first: {replay_bad[0]})")
    (out / "arm_manifest.json").write_text(json.dumps(arm_manifest, indent=2))
    seed_hashes = {cid: hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
                   for cid, s in seed_files.items()}
    (out / "seed_hashes.json").write_text(json.dumps(seed_hashes, indent=2))
    (out / "representative_renders.json").write_text(json.dumps(renders, indent=2, default=str))

    print(f"Sentinel reasoning battery: {len(seed_files)} seed cards -> {len(cells)} cells "
          f"({len(spec)} contexts)")
    print(f"  models: {', '.join(m['short'] for m in models)}")
    print(f"  futures: {len(cells)} cells x {len(models)} models x {args.accepted_per_cell} = {n_fut}")
    print(f"  est. cost ~${cost:.0f} for ACCEPTED futures; budget $60-80 (failed/retried attempts also bill)")
    print(f"  config load_config(): PASS ({n_loaded} cells)")
    print("  leakage lint: PASS (no scoring labels in prompt-facing text)")
    print(f"  seeds validated (complete BaselineObject): {len(seed_files)}  |  observed-prefixes: "
          f"{sum(1 for c in cards.values() if c.get('prior_history'))}")
    print("  replay-ready baseline clone/reference contract: PASS")
    if warnings:
        print("  COHERENCE WARNINGS (need review):")
        for w in warnings:
            print(f"    - {w}")
    print(f"  out: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
