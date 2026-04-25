from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _safe_field(record_data: list[dict[str, Any]], key: str) -> float | None:
    for item in record_data:
        if item.get("name") == key:
            value = item.get("value")
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def load_fit(path: str) -> pd.DataFrame:
    """Load a FIT file into a 1Hz-ish DataFrame indexed by elapsed seconds."""
    fit_path = Path(path)
    if not fit_path.exists():
        raise FileNotFoundError(f"FIT file not found: {fit_path}")

    try:
        from fitparse import FitFile
    except ImportError as exc:
        raise ImportError("fitparse is required to read FIT files. Install requirements.txt") from exc

    fit = FitFile(str(fit_path))
    rows: list[dict[str, float | int]] = []

    for idx, record in enumerate(fit.get_messages("record")):
        values = record.get_values()
        if not values:
            continue

        power = values.get("power")
        if power is None:
            power = _safe_field(record.fields, "power")

        rows.append(
            {
                "second": idx,
                "power": float(power) if power is not None else 0.0,
                "hr": float(values.get("heart_rate", 0.0) or 0.0),
                "altitude": float(values.get("altitude", 0.0) or 0.0),
                "speed": float(values.get("speed", 0.0) or 0.0),
                "cadence": float(values.get("cadence", 0.0) or 0.0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["power", "hr", "altitude", "speed", "cadence"])
        df.index.name = "second"
        return df

    df = df.set_index("second")
    df.index = df.index.astype(int)
    return df
