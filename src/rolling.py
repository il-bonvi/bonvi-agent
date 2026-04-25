from __future__ import annotations

import pandas as pd


def add_rolling(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling power columns with causal windows and min_periods=1."""
    if windows is None:
        windows = [1, 30, 60]

    if "power" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'power' column")

    out = df.copy()
    for w in windows:
        out[f"rolling_{w}s"] = out["power"].rolling(window=w, center=False, min_periods=1).mean()
    return out
