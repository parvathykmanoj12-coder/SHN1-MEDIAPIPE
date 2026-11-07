"""Deterministic, conservative decision policy for visually assessed fruit.

The VLM observes; this module alone assigns priority. It never diagnoses safety.
"""
from __future__ import annotations

from typing import Any

URGENCY_WEIGHT = {"EAT_TODAY": 1, "USE_SOON": 2, "STORE_LONGER": 3}
STAGE_TO_URGENCY = {
    "OVERRIPE": "EAT_TODAY",
    "RIPE": "USE_SOON",
    "UNDERRIPE": "STORE_LONGER",
}


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Validate a model observation and calculate a trusted *ordering*, not food safety."""
    fruit = str(observation.get("fruit_name") or "Unknown fruit").strip()[:55]
    appearance = str(observation.get("visible_features") or "Appearance unclear").strip()[:150]
    short_appearance = " ".join(appearance.split()[:7]).rstrip(".;,")
    stage = str(observation.get("ripeness_stage") or "UNKNOWN").upper()
    quality = str(observation.get("view_quality") or "POOR").upper()
    visible = observation.get("fruit_visible") is True
    suspicious = observation.get("possible_mold_or_rot") is True
    try:
        count = int(observation.get("fruit_count", 0))
    except (TypeError, ValueError):
        count = 0

    valid = (
        visible and count == 1 and quality == "CLEAR" and not suspicious
        and stage in STAGE_TO_URGENCY and fruit.lower() not in {"unknown", "unknown fruit", "fruit", "none"}
    )
    urgency = STAGE_TO_URGENCY[stage] if valid else "REVIEW_REQUIRED"
    weight = URGENCY_WEIGHT.get(urgency)

    if suspicious:
        spoken = f"{fruit}. Possible mold or decay. Do not eat based on this scan."
    elif not valid:
        spoken = "Fruit unclear. Please scan one fruit again in better lighting."
    elif urgency == "EAT_TODAY":
        spoken = f"{fruit}. {short_appearance}. Use first today, if sound. Check before eating."
    elif urgency == "USE_SOON":
        spoken = f"{fruit}. {short_appearance}. Appears ripe. Use soon. Check before eating."
    else:
        spoken = f"{fruit}. {short_appearance}. Appears less ripe. Can keep longer. Check before eating."

    return {
        "fruit_name": fruit,
        "surface_description": appearance,
        "ripeness_stage": stage,
        "urgency": urgency,
        "weight": weight,
        "review_required": not valid,
        "possible_mold_or_rot": suspicious,
        "spoken_text": spoken,
    }


def unknown_result(reason: str = "The image could not be assessed.") -> dict[str, Any]:
    result = normalize_observation({
        "fruit_name": "Unknown fruit", "visible_features": reason,
        "ripeness_stage": "UNKNOWN", "view_quality": "POOR",
        "fruit_visible": False, "fruit_count": 0,
        "possible_mold_or_rot": False,
    })
    result["spoken_text"] = f"{reason} Freshness ranking is unavailable. Try again."
    return result
