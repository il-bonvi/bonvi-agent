from __future__ import annotations

from typing import Any

ALLOWED_ACTIVITY_TYPES = {"training", "freeride", "road", "criterium", "ITT", "TTT"}


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Return validation errors. Empty list means valid payload."""
    errors: list[str] = []

    session_info = payload.get("session_info")
    if not isinstance(session_info, dict):
        return ["session_info must be an object"]

    for key in ["session_id", "filename", "cp", "weight", "activity_type"]:
        if key not in session_info:
            errors.append(f"session_info.{key} is required")

    activity_type = session_info.get("activity_type")
    if activity_type not in ALLOWED_ACTIVITY_TYPES:
        errors.append(
            "session_info.activity_type must be one of: " + ", ".join(sorted(ALLOWED_ACTIVITY_TYPES))
        )

    efforts = payload.get("efforts")
    if not isinstance(efforts, list):
        errors.append("efforts must be a list")
    else:
        for i, effort in enumerate(efforts):
            if not isinstance(effort, dict):
                errors.append(f"efforts[{i}] must be an object")
                continue

            for field in ["index", "start_time_sec", "end_time_sec", "duration_sec", "avg_power_w", "start_idx", "end_idx"]:
                if field not in effort:
                    errors.append(f"efforts[{i}].{field} is required")

            start = effort.get("start_time_sec")
            end = effort.get("end_time_sec")
            duration = effort.get("duration_sec")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start > end:
                errors.append(f"efforts[{i}] has start_time_sec > end_time_sec")
            if isinstance(duration, (int, float)) and duration <= 0:
                errors.append(f"efforts[{i}] has non-positive duration_sec")

    if "sprints" not in payload or not isinstance(payload.get("sprints"), list):
        errors.append("sprints must be a list")

    if "ride_statistics" not in payload or not isinstance(payload.get("ride_statistics"), dict):
        errors.append("ride_statistics must be an object")

    if "detection_parameters" not in payload or not isinstance(payload.get("detection_parameters"), dict):
        errors.append("detection_parameters must be an object")

    return errors
