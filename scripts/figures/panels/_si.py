"""Helpers shared by the SI1 and SI2 panels: treatment classes and drug names in clinical words."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODALITY_COLOURS  # noqa: E402

# antibody-drug conjugates have no colour in the paper plan: a local colour, flagged in confirm
ADC_COLOUR = "#7B5EA7"
CLASS_COLOURS = {**MODALITY_COLOURS, "antibody-drug conjugate": ADC_COLOUR}
CLASS_LABEL = {"chemotherapy": "Chemotherapy", "endocrine": "Endocrine therapy", "targeted": "Targeted therapy",
               "immunotherapy": "Immunotherapy", "antibody-drug conjugate": "Antibody-drug conjugate",
               "other": "Other / none"}

_TARGETED = ("osimertinib", "savolitinib", "ribociclib", "alpelisib")
_ENDOCRINE = ("anastrozole", "letrozole", "fulvestrant", "tamoxifen")
_CHEMO = ("doxorubicin", "cyclophosphamide", "paclitaxel", "carboplatin", "etoposide", "topotecan",
          "capecitabine", "eribulin", "pemetrexed", "gemcitabine", "folfirinox")
_ADC = ("govitecan", "vedotin", "deruxtecan")


def regimen_class(regimen: str) -> str:
    """Class of a stored regimen name; raises on an unknown regimen rather than guessing."""
    r = regimen.lower()
    if r == "no systemic treatment":
        return "other"
    if any(k in r for k in _ADC):
        return "antibody-drug conjugate"
    if any(k in r for k in _TARGETED):
        return "targeted"
    if any(k in r for k in _ENDOCRINE):
        return "endocrine"
    if any(k in r for k in _CHEMO):
        return "chemotherapy"
    raise ValueError(f"unclassified regimen: {regimen}")


def regimen_name(regimen: str) -> str:
    """'doxorubicin_cyclophosphamide' -> 'Doxorubicin + cyclophosphamide'."""
    return _plus(regimen)


def _plus(regimen: str) -> str:
    parts = regimen.lower().split("_")
    # weekly / dose words stay attached to the drug they qualify
    out: list[str] = []
    for p in parts:
        if p in ("weekly", "off") and out:
            out[-1] += f" {p}"
        else:
            out.append(p)
    s = " + ".join(out)
    return s[0].upper() + s[1:]
