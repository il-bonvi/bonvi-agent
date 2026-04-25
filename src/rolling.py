from __future__ import annotations

import pandas as pd


def add_rolling(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling power columns.

    Rolling 30s and 60s are centered to match manual gold annotation logic.
    Rolling 1s remains non-centered by definition.
    """
    if windows is None:
        windows = [1, 30, 60]

    if "power" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'power' column")

    out = df.copy()
    for w in windows:
        is_centered = w in {30, 60}
        out[f"rolling_{w}s"] = out["power"].rolling(window=w, center=is_centered, min_periods=1).mean()
    return out
