from __future__ import annotations

from src.schema import validate_payload


def test_valid_payload_minimal() -> None:
    payload = {
        "session_info": {
            "session_id": "abc",
            "filename": "ride.fit",
            "cp": 250.0,
            "weight": 70.0,
            "activity_type": "road",
        },
        "ride_statistics": {},
        "efforts": [
            {
                "index": 0,
                "start_time_sec": 10.0,
                "end_time_sec": 20.0,
                "duration_sec": 11.0,
                "avg_power_w": 300.0,
                "start_idx": 10,
                "end_idx": 20,
            }
        ],
        "sprints": [],
        "detection_parameters": {},
    }

    assert validate_payload(payload) == []


def test_invalid_activity_type() -> None:
    payload = {
        "session_info": {
            "session_id": "abc",
            "filename": "ride.fit",
            "cp": 250.0,
            "weight": 70.0,
            "activity_type": "unknown",
        },
        "ride_statistics": {},
        "efforts": [],
        "sprints": [],
        "detection_parameters": {},
    }

    errors = validate_payload(payload)
    assert any("activity_type" in err for err in errors)
