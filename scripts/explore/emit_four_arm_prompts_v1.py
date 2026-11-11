#!/usr/bin/env python3
"""Four PROVIDER PROMPTS per patient, differing only where the experiment says they may.

    1  patient only        the CHAX prefix and the index regimen
    2  setting             the same, plus one sentence about the care environment
    3  generic cohort      the same, plus 20 random same-cancer, same-era patients
    4  patients-like-me    the same, plus 20 clinically matched patients

Everything that is supposed to be shared IS shared, and that is asserted rather than
intended. The shared prefix and the index regimen must be byte-identical across all four;
arms 2-4 must carry the identical setting sentence; arms 3 and 4 must come from the same
renderer with no visible arm label. If any of that drifts, the contrast measures the drift.

THE SETTING SENTENCE describes the care environment and says NOTHING about how patients
fare. "A North American academic oncology practice" is a fact about where; "patients here
typically receive five lines" would be a fact about outcomes, and would quietly turn arm 2
into a weak anchoring arm -- making contrast 3-2 a test of retrieval versus summary
statistics rather than of retrieval at all.

THE PATIENT ENTERS THROUGH THE CONDITIONING SPEC, not the anchor block. An earlier version
put a plain-text CHAX summary inside {{ANCHOR_BLOCK}} and hashed the result, which produced
1,800 arm blocks and called them prompts -- with no injected schema, no baseline seed, no
fixed prefix, no terminal contract and no output contract. Under `continuation` the patient
is the baseline_seed plus the observed prefix, and the placeholder carries ONLY what
distinguishes the arms:

    arm 1  nothing
    arm 2  the setting sentence
    arm 3  the setting sentence and the generic cohort card
    arm 4  the setting sentence and the patients-like-me card

PROVENANCE PER PROMPT is the whole rendered text's hash plus the hashes of its parts, so a
later reader can tell WHICH part differed between two prompts without having both. A card
checksum alone cannot distinguish a changed template from a changed card. Hashes are the
FULL 64 characters: a 16-character prefix is fine for a console and is not provenance.

GOVERNANCE. GENIE-BPC under DUA; transmission cleared. Prompts are written only to the
gitignored outputs tree.

Run:  uv run python scripts/explore/emit_four_arm_prompts_v1.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_query_universe_v1 as QU  # noqa: E402
import canonical_future_v1 as CF  # noqa: E402
import clinical_match_v1 as CM  # noqa: E402
import disease_extent_v1 as DE  # noqa: E402
import genie_operator_registry_v2 as GV2  # noqa: E402
import molecular_prefix_v1 as MP  # noqa: E402
import render_reference_card_v1 as RC  # noqa: E402
from synthonccases.anchored import contract as CT  # noqa: E402
from synthonccases.anchored import query as Q  # noqa: E402
from synthonccases.oncollm.condition_lower import (  # noqa: E402
    lower_conditioning_cell,
)

OUT = ROOT / "outputs" / "exploratory_four_arm_prompts"
UNIVERSE = ROOT / "outputs" / "exploratory_query_universe"

ARMS = ("patient_only", "setting", "generic_cohort", "patients_like_me")

#: The renderer writes the extent in several forms; any one of these implies the value.
ANY_OF = {
    "yes": ("stage iv at diagnosis", "metastatic disease subsequently documented"),
    "no": ("no distant metastasis documented",),
}

#: Gene symbol -> the schema's canonical driver bucket. The seed was carrying "BRAF",
#: which is not a value the injected enum admits: the raw biomarker keeps the gene and
#: variant ("BRAF G466V") because that is what the source knows, and only the STRUCTURED
#: bucket is normalised.
DRIVER_BUCKET = {
    "EGFR": "egfr", "ALK": "alk", "ROS1": "ros1", "RET": "ret", "MET": "met",
    "BRAF": "braf", "KRAS": "kras", "NRAS": "nras", "ERBB2": "erbb2_her2",
    "PIK3CA": "pik3ca_pi3k_akt", "AKT1": "pik3ca_pi3k_akt", "PTEN": "pten",
    "ESR1": "esr1", "BRCA1": "brca_hrd", "BRCA2": "brca_hrd",
    "NTRK1": "ntrk", "NTRK2": "ntrk", "NTRK3": "ntrk",
    "MLH1": "mmr_msi", "MSH2": "mmr_msi", "MSH6": "mmr_msi", "PMS2": "mmr_msi",
    "POLE": "other_driver",
}

#: STRUCTURALLY ABSENT FROM THE REGISTRY, stated once and identically in every arm.
#: These are not missing values for this patient -- the source does not collect them for
#: anyone. Saying so is different from silence: silence invites the model to assume a
#: normal performance status and a stable weight, and a normal default would be our
#: assumption pinned as the patient's fact. Donors are never matched on them either.
ABSENT = ("Not recorded by this registry for any patient, and therefore unknown here "
          "rather than normal: performance status, weight, organ function, and treatment "
          "toxicity. Reason about the future under that uncertainty rather than resolving "
          "it.\n"
          # THE TWO WINDOWS, NAMED. The summary above runs from diagnosis; the structured
          # event list begins at the first systemic treatment, because the ledger's event
          # vocabulary cannot encode a pre-treatment disease assessment. They therefore
          # report different counts of the same kind of thing, on purpose. Unlabelled,
          # that reads as a contradiction in the prompt.
          "The structured event list begins at the first systemic treatment; the summary "
          "above covers the whole course since diagnosis, so it may describe disease "
          "behaviour that has no event in the list.\n"
          # LOCAL THERAPY IS INVISIBLE, and a gap must not be read as a decision. GENIE-BPC
          # has no radiation table and no surgery-as-treatment table, so a patient may have
          # been treated in a way this record cannot show.
          "Surgery and radiotherapy are not recorded by this registry, so a gap between "
          "systemic treatments does not mean the patient was untreated.")

#: Setting facts only. A clinician would state these before seeing any outcome.
SETTING = ("Care is provided in a North American academic oncology practice in {year}, "
           "where clinical trials may be available.")


def sha(text: str) -> str:
    """The FULL digest. Truncating to 16 characters is a display convenience and was
    being written into fields labelled sha256."""
    return hashlib.sha256(text.encode()).hexdigest()


def short(digest: str) -> str:
    return digest[:12]


def card_for(q: dict, key: str, patients: dict, mets: dict, cancer: str) -> str:
    recs = {d["patient"]: d for d in q[key]}
    if not recs:
        return ""
    donors = []
    for pid in RC.stable_order(list(recs)):
        d = recs[pid]
        donors.append((pid, d["year"],
                       RC.course(patients[pid], float(d["tau_months"]), mets.get(pid),
                                 d["molecular_driver"], cancer, d["year"])))
    return RC.render_card(donors)


def molecular_index(genie: Path, cancer: str):
    """The panel tables, loaded once per cohort, so the seed can name real alterations."""
    import pandas as pd
    reports = MP.load_reports(genie)
    panels = MP.panel_genes(genie)
    matrix = dict(pd.read_csv(genie / "data_gene_matrix.txt", sep="\t",
                              low_memory=False)[["SAMPLE_ID", "mutations"]].dropna().values)
    variants: dict[str, list] = {}
    muts = pd.read_csv(genie / "data_mutations_extended.txt", sep="\t", low_memory=False,
                       usecols=["Hugo_Symbol", "HGVSp_Short", "Variant_Classification",
                                "Tumor_Sample_Barcode"])
    for m in muts.itertuples():
        variants.setdefault(m.Tumor_Sample_Barcode, []).append(
            (m.Hugo_Symbol, str(m.HGVSp_Short or "").lstrip("p."),
             m.Variant_Classification))
    fus = genie / "data_fusions.txt"
    if fus.exists():
        for m in pd.read_csv(fus, sep="\t", low_memory=False).itertuples():
            variants.setdefault(m.Tumor_Sample_Barcode, []).append(
                (m.Hugo_Symbol, "fusion", "Fusion"))
    return lambda pid, tau: MP.molecular_at_landmark(pid, tau, reports, variants, matrix,
                                                     panels, cancer)


def conditioning(q: dict, p, mets_rec, cancer: str, molecular) -> dict:
    """The patient as a conditioning payload: baseline seed, pinned paths, fixed prefix.

    This is the SHARED channel. Every arm receives exactly this, so the four prompts differ
    only in the auxiliary block, and the assertion that they do is made on the rendered
    prompt rather than on the block.
    """
    tau = q["tau_months"]
    f = GV2.prefix_features(p, tau)
    extent = DE.extent_at_landmark(mets_rec, tau)
    # THE ACTUAL ALTERATIONS, not an empty list read off the action record -- which never
    # carried them. With alterations empty the seed said "no alteration found in the genes
    # this panel covers" AND named EGFR as the primary driver, in the same object.
    mol = molecular(p.pid, tau)
    driver = q["molecular_driver"]
    row = {
        "cancer": cancer,
        "histology": ("adenocarcinoma" if f.get("histology_adeno")
                      else "squamous cell carcinoma" if f.get("histology_squamous")
                      else None),
        # AGE AT THE LANDMARK. `baseline_seed` subtracts months-since-diagnosis to recover
        # the diagnosis age, so passing the diagnosis age subtracted it a second time and
        # made deep-line patients years too young.
        "age_at_start": (f["age_at_diagnosis"] + tau / 12.0
                         if np.isfinite(f.get("age_at_diagnosis", np.nan)) else None),
        "months_since_dx": tau,
        "female": f.get("sex_female"),
        # STAGE AT DIAGNOSIS, which is what the seed leaf means; extent at the landmark is
        # history and reaches the model through the prefix instead.
        "stage_iv": 1.0 if extent.stage_at_diagnosis == "IV" else 0.0,
        # MOLECULAR. `baseline_seed` writes the profile only when panel_reported is set,
        # so with the flag absent 201 queries with a known driver produced zero seeds
        # carrying one -- while the patients-like-me card shows its donors' molecular
        # status, matched on the query's. The query fact was in the evidence arm and not
        # in the patient-only arm, which is an anchoring confound rather than an omission.
        "panel_reported": 0.0 if driver == "unknown" else 1.0,
        "driver_bucket": (None if driver == "unknown" or "none recorded" in driver
                          else DRIVER_BUCKET.get(driver.split()[0], "other_driver")),
        "alterations": "; ".join(mol.alterations) if mol.panel_reported else "",
    }
    seed, pinned = Q.baseline_seed(row)
    origin = Q.diagnosis_date(q["diagnosis_year"])
    # LEDGER EVENT TYPES, which is what the prefix builder and the replay engine speak.
    # "observation" and "treatment" are our words, not the contract's, and reached
    # LedgerEvent validation as invalid literals.
    started = sorted((ln for ln in p.lines if ln["start"] <= tau + 1e-9),
                     key=lambda ln: ln["start"])
    facts = []
    for j, ln in enumerate(started):
        label = RC.canonical(ln.get("label", ""))
        facts.append({
            "months": float(ln["start"]),
            "type": "systemic_treatment_start" if j == 0 else "systemic_treatment_switch",
            "regimen": {"label": label,
                        "components": [c.strip() for c in label.split(" + ") if c.strip()]}})
    # TREATMENT STOPS were missing entirely: a regimen that ended before the landmark was
    # rendered as though still running.
    for j, ln in enumerate(started[:-1]):
        nxt = started[j + 1]["start"]
        if ln["stop"] < nxt - 0.5 and ln["stop"] < tau:
            facts.append({"months": float(ln["stop"]), "type": "systemic_treatment_stop",
                          "reason": "planned_completion"})
    # ONLY WHAT THE LEDGER CAN ENCODE FAITHFULLY. Progression before any systemic therapy
    # was being emitted as `patient_state_update` with no patient or disease payload --
    # 125 events describing a disease assessment through an event type the contract
    # reserves for host-state changes. Such assessments now reach the model through the
    # shared canonical summary instead, where they can be stated plainly.
    first_tx = started[0]["start"] if started else float("inf")
    # ONE EVENT PER EPISODE, NOT PER ASSESSMENT, and from the SAME operator the future is
    # scored on. Emitting a row per status assessment turned a patient scanned quarterly
    # into ten consecutive `progression` events with no treatment between them -- which
    # reads as continuous untreated progression over years, and is really one episode
    # re-confirmed at every visit. It also made the ledger disagree with both of its
    # neighbours: `prior_course` already counted episodes ("counting rows measures scan
    # frequency rather than disease behaviour") and `project()` scores `entries_into`.
    # The model was shown one accounting and graded on another.
    #
    # `state_path` is that operator, run over the prefix window instead of the future. It
    # merges consecutive identical states AND inserts `unknown` when an assessment
    # expires, so progression documented, then six unassessed months, then progression
    # documented again stays TWO episodes rather than collapsing to one.
    site_months = {float(m) for m, _s in (mets_rec.sites if mets_rec else [])
                   if first_tx < m < tau}
    if np.isfinite(first_tx):
        path = CF.state_path(p, first_tx, tau)
        previous = path[0][2] if path else None
        for start, _stop, st in path:
            when = start + first_tx
            if st != previous and st in ("progressing", "responding"):
                # A newly documented distant site already produces its own event below,
                # with the site named. Emitting the path's progression too would double
                # the same episode.
                if not (st == "progressing"
                        and any(abs(when - m) < 1e-6 for m in site_months)):
                    facts.append({
                        "months": float(when),
                        "type": "progression" if st == "progressing" else "response",
                        "rationale": ("recorded as progressive disease"
                                      if st == "progressing"
                                      else "recorded as improving or responding")})
            previous = st
    # METASTATIC SITES, timed. These were absent from the prefix entirely while the cards
    # rendered donor sites and the matcher scored site overlap.
    for m in sorted(site_months):
        for _m, site in (mets_rec.sites if mets_rec else []):
            if abs(float(_m) - m) < 1e-9:
                facts.append({"months": m, "type": "progression",
                              "rationale": f"distant metastasis documented: {site}"})
                break
    fixed = ({} if origin is None else
             Q.prefix_events(facts, origin=origin, landmark_months=tau,
                             landmark_action=q["action"]["regimen"]))
    # ASSERTED BEFORE IT IS DISCARDED. A landmark whose cutoff day does not carry the
    # index treatment leaves the model to infer which regimen it is continuing from, which
    # is the one thing the continuation contract must not leave open.
    if fixed and not fixed.pop("index_treatment_matched", True):
        raise ValueError(f"{q['query_id']}: the index treatment is not on the cutoff date, "
                         f"so the regimen being continued is ambiguous")
    for key in ("n_dropped_pre_diagnosis", "n_duplicates_collapsed",
                "n_merged_within_window", "n_observations_at_landmark_dropped",
                "n_treatment_events_at_cutoff"):
        fixed.pop(key, None)
    return (Q.conditioning_payload(seed, pinned, fixed) if fixed
            else {"baseline_seed": seed, "pinned_paths": list(pinned)})


def build(q: dict, patients: dict, mets: dict, cancer: str, molecular) -> dict[str, dict]:
    p = patients[q["patient"]]
    payload = conditioning(q, p, mets.get(p.pid), cancer, molecular)
    spec = lower_conditioning_cell(conditioning_payload=payload)
    # THE CANONICAL CLINICAL SUMMARY, in the SHARED operator-constraint channel -- every
    # arm, never the arm-specific block. It carries what the ledger cannot encode
    # faithfully: oncologist presence, stable and mixed assessments, prior-course shape.
    # Redundant with the structured prefix by design: the engine needs the checkable form
    # and the model reads the clinical one.
    summary = RC.query_prefix(p, q["tau_months"], mets.get(p.pid), q["molecular_driver"],
                              q["diagnosis_year"], q["landmark_year"]) + "\n" + ABSENT
    setting = SETTING.format(year=q["landmark_year"])
    cards = {"generic_cohort": card_for(q, "generic_donors", patients, mets, cancer),
             "patients_like_me": card_for(q, "relevant_donors", patients, mets, cancer)}

    payload_json = json.dumps(payload, sort_keys=True, default=str)
    out: dict[str, dict] = {}
    for arm in ARMS:
        # THE ANCHOR BLOCK IS ARM-SPECIFIC ONLY. The patient is in the spec.
        parts = [] if arm == "patient_only" else [setting]
        evidence = cards.get(arm, "")
        if evidence:
            parts.append(evidence)
        block = "\n\n".join(parts)
        prompt = CT.compose_prompt(
            block, contract="continuation",
            baseline_seed=dict(spec.baseline_seed), pinned_paths=spec.pinned_paths,
            required_patch_leaves=(), policy_text=summary, conditioning_spec=spec,
            terminal_contract="death_only")
        if "{{" in prompt or "}}" in prompt:
            raise ValueError(f"{q['query_id']}/{arm}: unsubstituted placeholder reached "
                             f"the provider prompt")
        out[arm] = {
            "arm": arm,
            "anchor_block": block,
            "rendered_prompt": prompt,
            "conditioning_payload": payload,
            "setting_sentence": setting if arm != "patient_only" else "",
            "evidence_block": evidence,
            "rendered_prompt_sha256": sha(prompt),
            "conditioning_payload_sha256": sha(payload_json),
            "clinical_summary": summary,
            "clinical_summary_sha256": sha(summary),
            "anchor_block_sha256": sha(block),
            "setting_sha256": sha(setting) if arm != "patient_only" else "",
            "evidence_block_sha256": sha(evidence) if evidence else "",
        }
    return out


def check(built: dict[str, dict], q: dict, card) -> list[str]:
    """What must hold across the four, asserted on the RENDERED PROMPT.

    Checking the arm blocks could not see a difference introduced by composition -- a
    different seed, spec, schema or terminal contract reaching one arm. The whole prompt
    minus its auxiliary block is the shared part, and that is what must be identical.
    """
    bad = []
    # THE SHARED PART OF THE PROMPT, obtained by removing the arm's own block from it.
    # Everything the composer injected -- schema, baseline seed, fixed prefix, terminal
    # contract, knowledge cutoff -- is inside this remainder.
    # Remove the whole ANCHOR LINE, not just the block text. `render_template` DELETES
    # the line for an empty block rather than substituting an empty string, so arm 1's
    # prompt has no anchor scaffolding at all -- subtracting only the block left arms 2-4
    # holding two newlines that arm 1 never had, and every quartet failed.
    shared = {}
    for arm in ARMS:
        text, block = built[arm]["rendered_prompt"], built[arm]["anchor_block"]
        shared[arm] = text.replace("\n" + block + "\n", "", 1) if block else text
    if len({sha(v) for v in shared.values()}) != 1:
        differing = [a for a in ARMS if sha(shared[a]) != sha(shared[ARMS[0]])]
        bad.append(f"the composed prompt differs outside the arm block in {differing} -- "
                   f"seed, spec, schema or contract is not shared")
    if len({built[a]["conditioning_payload_sha256"] for a in ARMS}) != 1:
        bad.append("the conditioning payload is not identical across arms")
    # THE ACTION LINE specifically. A substring test passes when the drug merely appears
    # in the prior-treatment history, which is a different claim.
    action = q["action"]["regimen"]
    for arm in ARMS:
        if action not in built[arm]["rendered_prompt"]:
            bad.append(f"{arm}: the index regimen {action!r} is absent from the prompt")
    settings = {built[a]["setting_sentence"] for a in ARMS[1:]}
    if len(settings) != 1 or not next(iter(settings)):
        bad.append("arms 2-4 do not carry one identical setting sentence")
    if built["patient_only"]["setting_sentence"]:
        bad.append("the patient-only arm carries a setting sentence")
    g, r = built["generic_cohort"]["evidence_block"], built["patients_like_me"][
        "evidence_block"]
    if not g or not r:
        bad.append("an evidence arm has no card")
    elif g.split("\n")[0] != r.split("\n")[0]:
        bad.append("the two cards do not share a header -- the header identifies the arm")
    # PHRASES, not bare words. "relevant" appears in "no relevant alteration recorded on
    # the available panel" -- a clinical statement, not an arm label -- and banning the
    # word would trade a real leak check for a permanent false alarm, exactly as banning
    # "distant" did when it began carrying its clinical sense.
    for label in ("patients like me", "patients-like-me", "generic cohort",
                  "randomly selected", "control group", "matched patients",
                  "similar patients", "relevant patients", "unselected"):
        if label in g.lower() or label in r.lower():
            bad.append(f"a card contains the arm-revealing phrase {label!r}")
    # The evidence blocks must DIFFER: identical cards would mean the contrast is empty.
    if g and r and g == r:
        bad.append("the two cards are identical")

    # EVERY QUERY-SIDE RETRIEVAL CRITERION MUST REACH ALL FOUR PROMPTS.
    #
    # This is the check that would have caught the confound. Donors are selected on the
    # query's molecular driver, presence, sites and stage, and the card DISPLAYS those
    # facts about the donors. If the same fact about the QUERY reaches only the arms that
    # carry a card, then arm 4 quietly tells the model something arm 1 was never told --
    # and the contrast measures the leak rather than the retrieval.
    bad += criterion_audit(built, card)
    bad += leaf_schema_audit(built["patient_only"]["conditioning_payload"])
    return bad


def leaf_schema_audit(payload: dict) -> list[str]:
    """Every supplied leaf, validated against the schema the prompt injects.

    The build reported zero problems while the seed carried a driver bucket the enum does
    not admit, because nothing checked partial seed leaves against their own schema. A
    value that the model is told is fixed, and that its own contract forbids, is a
    contradiction we hand it -- and no cross-arm or criterion check can see it, since it is
    identical in all four arms and semantically present.

    Resolved through the replay models rather than a hand-copied enum list, so the check
    tracks the contract instead of drifting from it.
    """
    from typing import Literal, get_args, get_origin

    from synthonccases.oncollm import replay as RP

    out: list[str] = []
    for path in payload.get("pinned_paths", ()):
        model, value = RP.BaselineObject, payload["baseline_seed"]
        for i, part in enumerate(str(path).split(".")):
            if not isinstance(value, dict) or part not in value:
                out.append(f"pinned path {path!r} is not present in the seed it pins")
                model = None
                break
            value = value[part]
            fields = getattr(model, "model_fields", None)
            if not fields or part not in fields:
                model = None
                break
            ann = fields[part].annotation
            model = ann
            if i == len(str(path).split(".")) - 1:
                # unwrap Optional / list wrappers to reach the literal, if there is one
                args = [a for a in get_args(ann) if a is not type(None)]
                target = args[0] if len(args) == 1 and get_origin(ann) is not Literal \
                    else ann
                allowed = get_args(target) if get_origin(target) is Literal else ()
                if allowed and value not in allowed:
                    out.append(f"pinned leaf {path} = {value!r} is not in the schema "
                               f"vocabulary this prompt injects; allowed: "
                               f"{sorted(str(a) for a in allowed)[:8]}")
    return out


def criterion_audit(built: dict[str, dict], card) -> list[str]:
    """Every matcher criterion, checked against the TRANSMITTED PATIENT.

    The previous version searched the whole shared prompt -- which includes the injected
    schema -- so "EGFR", "female" and "metastatic" were satisfied by the ENUM DEFINITIONS
    whether or not the patient carried them. It could have passed with the molecular
    profile entirely absent, which is the exact failure it was written to catch.
    It now reads only the conditioning payload and the clinical summary: the patient, not
    the instructions about patients.
    """
    pay = built["patient_only"]["conditioning_payload"]
    ident = pay["baseline_seed"].get("identity_card", {})
    summary = built["patient_only"]["clinical_summary"].lower()
    seed_json = json.dumps(pay["baseline_seed"], default=str).lower()
    events = json.dumps(pay.get("fixed_events", {}), default=str).lower()
    patient = seed_json + "\n" + events + "\n" + summary

    #: matcher Card field -> how the transmitted patient must express it.
    wanted: list[tuple[str, object, str]] = [
        ("cancer present at the landmark", card.presence,
         {"evidence": "cancer present", "no evidence": "no evidence",
          "equivocal": "equivocal"}.get(card.presence or "", "")),
        ("stage at diagnosis", card.stage_iv,
         "metastatic" if card.stage_iv else "locoregional"),
        # MATCHED AGAINST THE RENDERER'S BRANCHES, not one hoped-for word. The extent
        # sentence has several forms -- "Stage IV at diagnosis", "metastatic disease
        # subsequently documented", "no named distant-organ site recorded" -- and each
        # attempt to pin one phrase failed on another. What must hold is that the
        # transmitted text implies the same value the matcher scored.
        ("metastatic disease by landmark", card.mets_documented,
         ANY_OF[card.mets_documented] if card.mets_documented in ANY_OF else ""),
        ("molecular driver", None if card.driver in (None, "unknown") else card.driver,
         "" if card.driver in (None, "unknown") or "none recorded" in card.driver
         else card.driver.split()[0].lower()),
        ("disease state at landmark", card.status,
         f"disease state at this treatment: {(card.status or '').replace('_', ' ')}"),
        ("regimen sequence", card.depth, (card.depth or "").lower()),
        ("age within a decade", card.age, f"{card.age:.0f}-year-old" if card.age else ""),
        ("sex", card.female, "female" if card.female else "male"),
        ("histology", card.histology,
         {"adeno": "adenocarcinoma", "squamous": "squamous"}.get(card.histology or "",
                                                                 card.histology or "")),
        ("metastatic sites overlap", sorted(card.sites) if card.sites else None,
         ", ".join(sorted(card.sites)) if card.sites else ""),
        ("time to this treatment", card.tempo,
         f"{card.tempo:.0f} month" if card.tempo else ""),
    ]
    out = []
    for name, value, needle in wanted:
        if value is None or not needle:
            continue                        # genuinely unknown, or nothing to look for
        options = needle if isinstance(needle, tuple) else (needle,)
        if not any(o.lower() in patient for o in options):
            out.append(f"the matcher scored {name!r} but the transmitted patient does not "
                       f"express it (none of {options!r} in seed, prefix or summary)")
    ident_code = ident.get("diagnosis", {}).get("oncotree_code")
    if not ident_code:
        out.append("the seed carries no oncotree_code, yet cancer is a hard constraint")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(line) for line in (UNIVERSE / "queries.jsonl").open()]
    manifest = json.loads((UNIVERSE / "manifest.json").read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:                                                    # noqa: BLE001
        commit, dirty = "unknown", True

    problems: list[str] = []
    quartets: list[dict] = []
    for cancer, folder in QU.COHORTS.items():
        genie = QU.GENIE_ROOT / folder
        if not genie.exists():
            continue
        patients = {p.pid: p for p in GV2.load(genie)}
        mets = DE.load_metastasis(genie)
        molecular = molecular_index(genie, cancer)
        for q in [x for x in queries if x["cancer"] == cancer]:
            built = build(q, patients, mets, cancer, molecular)
            card = CM.describe(patients[q["patient"]], q["tau_months"],
                               q["molecular_driver"], mets.get(q["patient"]))
            bad = check(built, q, card)
            problems += [f"{q['query_id']}: {b}" for b in bad]
            # THE LANDMARK STATE, CARRIED FORWARD FOR SCORING. The generated future is
            # projected onto the real side's ruler, and that projection needs the state the
            # patient is IN at the cutoff -- otherwise a trajectory whose first event is at
            # month 9 scores nine months of `unknown` that the prefix in fact resolved.
            # Emitted here rather than parsed back out of the summary prose, because a
            # scorer that re-derives its inputs by reading English is a second definition.
            # EVERY CHANNEL THE SCORER NEEDS, emitted rather than re-derived. `presence` is
            # what makes `no_active_disease` reachable at all, and the index regimen is
            # already running at the landmark -- that is what the landmark IS -- so a
            # projection without it starts months-on-treatment at the first generated
            # switch and scores a patient who stayed on the index drug as zero.
            _p = patients[q["patient"]]
            quartets.append({"query_id": q["query_id"], "cancer": cancer,
                             "depth": q["depth"],
                             "landmark_state": GV2.genie_state(
                                 _p, q["tau_months"], mets.get(q["patient"])).state,
                             "landmark_presence": GV2.recent(_p.presence, q["tau_months"]),
                             # THE SITES ACTIVE AT THE LANDMARK. The prompt asks the model
                             # to report only CHANGED sites, so without the starting
                             # picture "the liver lesion cleared" reads as a complete
                             # response while the lung is still involved.
                             "landmark_sites": sorted(
                                 {s for m, s in (mets.get(q["patient"]).sites
                                                 if mets.get(q["patient"]) else [])
                                  if m <= q["tau_months"]}),
                             # STAGE IV CAN CARRY NO NAMED SITE -- it is assigned on
                             # criteria the curated site block does not enumerate. The
                             # scorer needs to tell that apart from local disease, because
                             # only the second licenses "the primary cleared" as remission.
                             "landmark_extent": DE.extent_at_landmark(
                                 mets.get(q["patient"]), q["tau_months"]).documented,
                             "index_regimen": q["action"]["regimen"],
                             # THE REAL ASSESSMENT CALENDAR, for the masked comparison. Not
                             # shown to the model -- it is how the generated course is read
                             # at GENIE's resolution, since GENIE records detection rather
                             # than biological onset.
                             "assessment_times": sorted(
                                 {round(w - q["tau_months"], 4)
                                  for w, _v in (_p.status + _p.presence)
                                  if q["tau_months"] < w
                                  <= q["tau_months"] + max(GV2.SCORING_HORIZONS)}),
                             # WHETHER THE REAL PATIENT CAN ADJUDICATE EACH HORIZON. A
                             # patient censored at month 9 cannot settle a 24-month claim
                             # however well-formed the generated one is. The generated fan
                             # cannot show this, because the contract forces generation
                             # through death and every compliant ledger resolves every
                             # horizon.
                             "real_evaluable_horizons": [
                                 h for h in GV2.SCORING_HORIZONS
                                 if np.isfinite(CF.project(_p, q["tau_months"], float(h))
                                                .alive_at_horizon)],
                             "scoring_horizons": list(GV2.SCORING_HORIZONS),
                             "arms": built})

    prov = CT.provenance()
    record = {
        "git_commit": commit,
        "working_tree_clean": not dirty,
        # The query-universe checksum as the builder recorded it. It is truncated at
        # source, which is noted here rather than silently re-hashed into something that
        # would not match the file it describes.
        "query_universe_checksum_as_recorded": manifest["queries_sha256"],
        "query_universe_sha256": sha((UNIVERSE / "queries.jsonl").read_text()),
        "query_universe_commit": manifest["git_commit"],
        "contract": "continuation",
        "terminal_contract": "death_only",
        "knowledge_cutoff": "2024-12-31",
        "continuation_template_sha256": prov["continuation_template_sha256"],
        "frozen_parent_sha256": prov["frozen_parent_sha256"],
        "setting_template": SETTING,
        "setting_template_sha256": sha(SETTING),
        "arms": list(ARMS),
        "n_quartets": len(quartets),
        "problems": problems,
    }
    (OUT / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    # THE INSTRUMENTS THEMSELVES, in full. Storing only hashes meant a run could not be
    # launched or reconstructed from these artefacts without rereading GENIE and rebuilding
    # everything -- and a hash of something you no longer hold proves nothing.
    (OUT / "quartets.jsonl").write_text("".join(json.dumps(qt, default=str) + "\n"
                                                for qt in quartets))

    print(f"{len(quartets)} quartets built ({len(quartets) * 4} prompts)")
    print(f"  git commit            {commit[:12]}  "
          f"{'clean' if not dirty else 'DIRTY -- provenance would not point anywhere'}")
    print(f"  query universe        {manifest['queries_sha256']} "
          f"@ {manifest['git_commit'][:12]}")
    print(f"  continuation template {prov['continuation_template_sha256'][:16]}")
    print(f"  setting sentence      {sha(SETTING)}")
    ex = quartets[0]["arms"]
    print(f"\n  example quartet {quartets[0]['query_id']}:")
    for arm in ARMS:
        d = ex[arm]
        print(f"      {arm:18s} prompt {short(d['rendered_prompt_sha256'])}  "
              f"payload {short(d['conditioning_payload_sha256'])}  "
              f"block {short(d['anchor_block_sha256'])}  "
              f"{len(d['rendered_prompt']):6d} chars")
    print(f"\n  {'ALL CROSS-ARM ASSERTIONS PASS' if not problems else 'PROBLEMS'}")
    for prob in problems[:10]:
        print(f"      FAIL  {prob}")
    # Three readable QUARTETS, one per cancer, so a clinician can lay the four arms of
    # one patient side by side and see that only the permitted things differ.
    # ONE PER CANCER, SPANNING DEPTH. Taking the first query per cohort gave three
    # regimen-1 patients, so a deep-line course -- the case with the most history to render
    # and the most that can go wrong in it -- never appeared in the review set.
    sample = []
    wanted_depth = dict(zip(QU.COHORTS, ("regimen 1", "regimen 2", "regimen 3+")))
    for cancer in QU.COHORTS:
        want = wanted_depth[cancer]
        first = next((qt for qt in quartets if qt["cancer"] == cancer
                      and qt["depth"] == want
                      and "unknown" not in qt["arms"]["patient_only"]["clinical_summary"]
                      .split("Molecular:")[1][:40]),
                     None) or next((qt for qt in quartets if qt["cancer"] == cancer
                                    and qt["depth"] == want), None)
        if not first:
            continue
        sample.append("#" * 78)
        sample.append(f"# {first['query_id']}   ({first['depth']})")
        sample.append("#" * 78)
        for arm in ARMS:
            d = first["arms"][arm]
            sample.append(f"\n{'=' * 78}\nARM {ARMS.index(arm) + 1}: {arm}"
                          f"   [prompt {short(d['rendered_prompt_sha256'])}, "
                          f"{len(d['rendered_prompt'])} chars]\n{'=' * 78}")
            # THE PROVIDER-FACING PROMPT, not the arm block. What a clinician needs to
            # read is what the model receives.
            sample.append(d["rendered_prompt"])
    (OUT / "sample_quartets.txt").write_text("\n".join(sample))

    # A COMPACT REVIEW ARTEFACT. The full quartets are four 65-83k prompts that share all
    # but a few hundred characters, so reading them side by side means reading the same
    # template four times. This prints the shared part ONCE and then only what differs.
    compact = []
    for cancer in QU.COHORTS:
        want = wanted_depth[cancer]
        first = next((qt for qt in quartets if qt["cancer"] == cancer
                      and qt["depth"] == want
                      and "unknown" not in qt["arms"]["patient_only"]["clinical_summary"]
                      .split("Molecular:")[1][:40]),
                     None) or next((qt for qt in quartets if qt["cancer"] == cancer
                                    and qt["depth"] == want), None)
        if not first:
            continue
        a = first["arms"]
        compact.append("#" * 78)
        compact.append(f"# {first['query_id']}   ({first['depth']})")
        compact.append("#" * 78)
        compact.append("\n--- SHARED BY ALL FOUR ARMS " + "-" * 49)
        compact.append("\n[clinical summary, via the shared operator-constraint channel]")
        compact.append(a["patient_only"]["clinical_summary"])
        pay = a["patient_only"]["conditioning_payload"]
        compact.append("\n[baseline seed]")
        compact.append(json.dumps(pay["baseline_seed"], indent=1, default=str))
        compact.append("\n[pinned paths]")
        compact.append("\n".join(f"  {x}" for x in pay["pinned_paths"]))
        ev = pay.get("fixed_events", {})
        compact.append(f"\n[fixed prefix: cutoff {ev.get('cutoff_date', '-')}, "
                       f"{len(ev.get('events', []))} events]")
        for e in ev.get("events", [])[:12]:
            compact.append("  " + json.dumps(e, default=str)[:150])
        if len(ev.get("events", [])) > 12:
            compact.append(f"  ... and {len(ev['events']) - 12} more")
        compact.append("\n--- WHAT DIFFERS BETWEEN ARMS " + "-" * 47)
        for arm in ARMS:
            blk = a[arm]["anchor_block"]
            compact.append(f"\n[{arm}]  {len(blk)} chars, "
                           f"prompt {short(a[arm]['rendered_prompt_sha256'])}")
            compact.append(blk[:1400] + ("\n  ... card continues" if len(blk) > 1400
                                         else "") if blk else "  (nothing)")
    (OUT / "compact_review.txt").write_text("\n".join(compact))
    print(f"\nwritten to {OUT}")
    return 0 if not problems and not dirty else 1


if __name__ == "__main__":
    raise SystemExit(main())
