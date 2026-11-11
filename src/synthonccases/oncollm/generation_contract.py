"""Acceptance-time checks for the SynthOncCases generation contract.

Pydantic validates types and enum membership. This module enforces the
cross-field commitments that make a trajectory analyzable, such as required
self-annotation buckets, coherent molecular-driver flags, non-empty founder
state, and required per-event molecular details. It runs after baseline and
ledger schema validation but before deterministic state reconstruction, so
contract failures surface as their own failure kind instead of being folded
into replay errors.
"""

from __future__ import annotations

from collections.abc import Sequence

from synthonccases.oncollm.replay import BaselineObject, LedgerEvent


def _is_blank(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def check_generation_contract(
    baseline: BaselineObject,
    events: Sequence[LedgerEvent],
) -> str | None:
    """Return a failure message when generation-contract requirements fail."""
    if not baseline.clone_registry:
        return "baseline.clone_registry must contain at least one founder clone"
    if not baseline.baseline_disease_burden:
        return "baseline.baseline_disease_burden must contain at least one site"

    if baseline.identity_card.diagnosis.histology_bucket is None:
        return "identity_card.diagnosis.histology_bucket is required"
    if baseline.identity_card.diagnosis.primary_site_class is None:
        return "identity_card.diagnosis.primary_site_class is required"
    if baseline.identity_card.molecular_profile.primary_driver_bucket is None:
        return "identity_card.molecular_profile.primary_driver_bucket is required"
    alterations = baseline.identity_card.molecular_profile.alterations
    primary_alterations = [
        alteration for alteration in alterations if alteration.is_primary_driver
    ]
    if (
        baseline.identity_card.molecular_profile.primary_driver_bucket
        == "no_actionable_driver"
    ):
        if primary_alterations:
            return (
                "identity_card.molecular_profile.alterations must not mark "
                "is_primary_driver=true when primary_driver_bucket is "
                "no_actionable_driver"
            )
    else:
        if not alterations:
            return (
                "identity_card.molecular_profile.alterations must contain at "
                "least one alteration when primary_driver_bucket is not "
                "no_actionable_driver"
            )
        if len(primary_alterations) != 1:
            return (
                "identity_card.molecular_profile.alterations must contain exactly "
                "one alteration with is_primary_driver=true when "
                "primary_driver_bucket is not no_actionable_driver"
            )
        if (
            primary_alterations[0].driver_bucket
            != baseline.identity_card.molecular_profile.primary_driver_bucket
        ):
            return (
                "identity_card.molecular_profile.primary_driver_bucket must match "
                "the driver_bucket of the alteration marked is_primary_driver=true"
            )

    for index, site in enumerate(baseline.baseline_disease_burden):
        if site.site_class is None:
            return (
                "baseline.baseline_disease_burden"
                f"[{index}].site_class is required"
            )

    for event_index, event in enumerate(events):
        for disease_index, site in enumerate(event.disease or []):
            if site.site_class is None:
                return (
                    f"ledger[{event_index}].disease"
                    f"[{disease_index}].site_class is required"
                )
        if event.regimen is not None and event.regimen.component_classes is None:
            return f"ledger[{event_index}].regimen.component_classes is required"
        if event.type == "clonal_evolution_event":
            if _is_blank(event.new_clone_id):
                return f"ledger[{event_index}].new_clone_id is required"
            # model_fields_set distinguishes explicit null from silent omission;
            # the generation contract requires an explicit commitment.
            if "gene" not in event.model_fields_set:
                return f"ledger[{event_index}].gene is required; use null only for true non-gene events"
            if event.gene is None and event.alteration_type != "histologic_transformation":
                return (
                    f"ledger[{event_index}].gene may be null only for "
                    "histologic_transformation events"
                )
            if _is_blank(event.alteration):
                return f"ledger[{event_index}].alteration is required"
            if event.alteration_type is None:
                return f"ledger[{event_index}].alteration_type is required"
            # model_fields_set distinguishes explicit null from silent omission;
            # the generation contract requires an explicit commitment.
            if "variant" not in event.model_fields_set:
                return f"ledger[{event_index}].variant is required; use null when not applicable"
            if event.driver_bucket is None:
                return f"ledger[{event_index}].driver_bucket is required"
            if event.actionability is None:
                return f"ledger[{event_index}].actionability is required"
            if _is_blank(event.mechanism):
                return f"ledger[{event_index}].mechanism is required"
            if event.mechanism_bucket is None:
                return f"ledger[{event_index}].mechanism_bucket is required"
            if event.sample_source is None:
                return f"ledger[{event_index}].sample_source is required"

    return None
