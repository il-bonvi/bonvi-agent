from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.archive import SessionTriple, load_json
from src.loader import load_fit
from src.rolling import add_rolling
from src.schema import validate_payload


@dataclass(frozen=True)
class BuildResult:
    effort_df: pd.DataFrame
    sprint_df: pd.DataFrame
    warnings: list[str]


def _segment_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if inter == 0:
        return 0.0
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union if union > 0 else 0.0


def _best_match(candidate: dict[str, Any], gold_items: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any] | None:
    c_start = int(candidate.get("start_idx", candidate.get("start_time_sec", 0)))
    c_end = int(candidate.get("end_idx", candidate.get("end_time_sec", 0)))

    best = None
    best_iou = 0.0
    for g in gold_items:
        g_start = int(g.get("start_idx", g.get("start_time_sec", 0)))
        g_end = int(g.get("end_idx", g.get("end_time_sec", 0)))
        iou = _segment_iou(c_start, c_end, g_start, g_end)
        if iou > best_iou:
            best_iou = iou
            best = g

    if best is None or best_iou < iou_threshold:
        return None
    return best


def _safe_slice(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    start_idx = max(int(df.index.min()), start_idx)
    end_idx = min(int(df.index.max()), end_idx)
    if end_idx < start_idx:
        return df.iloc[0:0]
    return df.loc[start_idx:end_idx]


def _build_effort_rows(triple: SessionTriple, iou_threshold: float) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    default_payload = load_json(triple.default_json_path)
    gold_payload = load_json(triple.gold_json_path)

    errors_default = validate_payload(default_payload)
    errors_gold = validate_payload(gold_payload)
    if errors_default:
        warnings.append(f"{triple.default_json_path}: invalid payload -> {'; '.join(errors_default)}")
    if errors_gold:
        warnings.append(f"{triple.gold_json_path}: invalid payload -> {'; '.join(errors_gold)}")

    default_info = default_payload.get("session_info", {})
    gold_info = gold_payload.get("session_info", {})
    if default_info.get("session_id") != gold_info.get("session_id"):
        warnings.append(f"{triple.folder}: session_id mismatch between default and gold")

    cp = float(default_info.get("cp", 0.0) or 0.0)
    weight = float(default_info.get("weight", 0.0) or 0.0)
    activity_type = str(default_info.get("activity_type", ""))
    session_id = str(default_info.get("session_id", triple.stem))

    df = add_rolling(load_fit(str(triple.fit_path)), windows=[1, 30, 60])
    ride_duration = float(len(df.index)) if len(df.index) else 1.0

    default_efforts = default_payload.get("efforts", [])
    gold_efforts = gold_payload.get("efforts", [])

    rows: list[dict[str, Any]] = []
    for e in default_efforts:
        start_idx = int(e.get("start_idx", e.get("start_time_sec", 0)))
        end_idx = int(e.get("end_idx", e.get("end_time_sec", 0)))
        segment = _safe_slice(df, start_idx, end_idx)

        matched = _best_match(e, gold_efforts, iou_threshold=iou_threshold)

        avg_power_w = float(e.get("avg_power_w", 0.0) or 0.0)
        duration_sec = float(e.get("duration_sec", max(1, end_idx - start_idx + 1)))
        avg_30 = float(segment["rolling_30s"].mean()) if (not segment.empty and "rolling_30s" in segment.columns) else 0.0
        avg_60 = float(segment["rolling_60s"].mean()) if (not segment.empty and "rolling_60s" in segment.columns) else 0.0
        max_power = float(segment["power"].max()) if (not segment.empty and "power" in segment.columns) else 0.0
        pct_above_cp = (
            float((segment["power"] >= cp).mean())
            if (cp > 0 and not segment.empty and "power" in segment.columns)
            else 0.0
        )

        keep_label = 1 if matched is not None else 0
        start_delta = 0.0
        end_delta = 0.0
        if matched is not None:
            g_start = float(matched.get("start_time_sec", matched.get("start_idx", 0)))
            g_end = float(matched.get("end_time_sec", matched.get("end_idx", 0)))
            d_start = float(e.get("start_time_sec", start_idx))
            d_end = float(e.get("end_time_sec", end_idx))
            start_delta = g_start - d_start
            end_delta = g_end - d_end

        rows.append(
            {
                "session_id": session_id,
                "stem": triple.stem,
                "folder": str(triple.folder),
                "activity_type": activity_type,
                "cp": cp,
                "weight": weight,
                "duration_sec": duration_sec,
                "avg_power_ratio": (avg_power_w / cp) if cp > 0 else 0.0,
                "max_power_ratio": (max_power / cp) if cp > 0 else 0.0,
                "avg_30_ratio": (avg_30 / cp) if cp > 0 else 0.0,
                "avg_60_ratio": (avg_60 / cp) if cp > 0 else 0.0,
                "pct_above_cp": pct_above_cp,
                "start_ratio": float(start_idx / ride_duration),
                "end_ratio": float(end_idx / ride_duration),
                "keep_label": keep_label,
                "start_delta_sec": start_delta,
                "end_delta_sec": end_delta,
            }
        )

    return rows, warnings


def _build_sprint_rows(triple: SessionTriple, iou_threshold: float) -> list[dict[str, Any]]:
    default_payload = load_json(triple.default_json_path)
    gold_payload = load_json(triple.gold_json_path)
    session_info = default_payload.get("session_info", {})

    cp = float(session_info.get("cp", 0.0) or 0.0)
    weight = float(session_info.get("weight", 0.0) or 0.0)
    activity_type = str(session_info.get("activity_type", ""))
    session_id = str(session_info.get("session_id", triple.stem))

    default_sprints = default_payload.get("sprints", [])
    gold_sprints = gold_payload.get("sprints", [])

    rows: list[dict[str, Any]] = []
    for s in default_sprints:
        start_idx = int(s.get("start_idx", 0))
        end_idx = int(s.get("end_idx", 0))
        duration_sec = float(s.get("duration_sec", max(1, end_idx - start_idx + 1)))
        avg_power_w = float(s.get("avg_power_w", 0.0) or 0.0)

        matched = _best_match(s, gold_sprints, iou_threshold=iou_threshold)
        keep_label = 1 if matched is not None else 0
        start_delta = 0.0
        end_delta = 0.0
        if matched is not None:
            start_delta = float(matched.get("start_idx", 0)) - float(start_idx)
            end_delta = float(matched.get("end_idx", 0)) - float(end_idx)

        rows.append(
            {
                "session_id": session_id,
                "stem": triple.stem,
                "folder": str(triple.folder),
                "activity_type": activity_type,
                "cp": cp,
                "weight": weight,
                "duration_sec": duration_sec,
                "avg_power_ratio": (avg_power_w / cp) if cp > 0 else 0.0,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "keep_label": keep_label,
                "start_delta_sec": start_delta,
                "end_delta_sec": end_delta,
            }
        )

    return rows


def build_training_frames(sessions: list[SessionTriple], iou_threshold: float = 0.3) -> BuildResult:
    effort_rows: list[dict[str, Any]] = []
    sprint_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for triple in sessions:
        e_rows, warn = _build_effort_rows(triple, iou_threshold=iou_threshold)
        effort_rows.extend(e_rows)
        warnings.extend(warn)
        sprint_rows.extend(_build_sprint_rows(triple, iou_threshold=iou_threshold))

    effort_df = pd.DataFrame(effort_rows)
    sprint_df = pd.DataFrame(sprint_rows)
    return BuildResult(effort_df=effort_df, sprint_df=sprint_df, warnings=warnings)
