"""Colours for the SI5 arrows (event kind) shared by si05a and si05b."""
from common import STATE_COLOURS

EVENT_COLOURS = {
    "Response": STATE_COLOURS["response"],
    "Toxicity starts": "#8E44AD",
    "Progression": STATE_COLOURS["progression"],
    "Condition recorded as worse": "#1B4F72",
}
