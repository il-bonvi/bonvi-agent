from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_fscore_support
from xgboost import XGBClassifier, XGBRegressor


@dataclass(frozen=True)
class TrainArtifacts:
    metrics: dict[str, Any]
    saved_files: list[str]


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    if "activity_type" in x.columns:
        x = pd.get_dummies(x, columns=["activity_type"], dummy_na=False)
    return x


def _loso_classifier_metrics(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict[str, float]:
    sessions = sorted(df["session_id"].unique())
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for session in sessions:
        train_df = df[df["session_id"] != session]
        test_df = df[df["session_id"] == session]
        if train_df.empty or test_df.empty:
            continue
        if train_df[target_col].nunique() < 2:
            continue

        x_train = _prepare_features(train_df, feature_cols)
        x_test = _prepare_features(test_df, feature_cols)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

        model = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(x_train, train_df[target_col])
        preds = model.predict(x_test)

        y_true_all.extend(test_df[target_col].astype(int).tolist())
        y_pred_all.extend(preds.astype(int).tolist())

    if not y_true_all:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _loso_regressor_mae(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> float:
    sessions = sorted(df["session_id"].unique())
    maes: list[float] = []

    for session in sessions:
        train_df = df[df["session_id"] != session]
        test_df = df[df["session_id"] == session]
        if train_df.empty or test_df.empty:
            continue

        x_train = _prepare_features(train_df, feature_cols)
        x_test = _prepare_features(test_df, feature_cols)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

        model = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(x_train, train_df[target_col])
        preds = model.predict(x_test)
        maes.append(float(mean_absolute_error(test_df[target_col], preds)))

    return float(np.mean(maes)) if maes else 0.0


def _fit_and_save_classifier(df: pd.DataFrame, feature_cols: list[str], target_col: str, out_path: Path) -> None:
    x = _prepare_features(df, feature_cols)
    y = df[target_col].astype(int)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(x.columns)}, out_path)


def _fit_and_save_regressor(df: pd.DataFrame, feature_cols: list[str], target_col: str, out_path: Path) -> None:
    x = _prepare_features(df, feature_cols)
    y = df[target_col].astype(float)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(x, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(x.columns)}, out_path)


def train_from_frames(
    effort_df: pd.DataFrame,
    sprint_df: pd.DataFrame,
    model_root: str | Path,
) -> TrainArtifacts:
    model_root = Path(model_root)
    saved_files: list[str] = []
    metrics: dict[str, Any] = {}

    effort_features = [
        "activity_type",
        "cp",
        "weight",
        "duration_sec",
        "avg_power_ratio",
        "max_power_ratio",
        "avg_30_ratio",
        "avg_60_ratio",
        "pct_above_cp",
        "start_ratio",
        "end_ratio",
    ]

    sprint_features = [
        "activity_type",
        "cp",
        "weight",
        "duration_sec",
        "avg_power_ratio",
        "start_idx",
        "end_idx",
    ]

    if not effort_df.empty:
        cls_metrics = _loso_classifier_metrics(effort_df, effort_features, "keep_label")
        metrics["effort_classifier_loso"] = cls_metrics

        effort_cls_path = model_root / "classifier" / "effort_keep_xgb.joblib"
        _fit_and_save_classifier(effort_df, effort_features, "keep_label", effort_cls_path)
        saved_files.append(str(effort_cls_path))

        effort_pos = effort_df[effort_df["keep_label"] == 1].copy()
        if len(effort_pos) >= 5:
            start_mae = _loso_regressor_mae(effort_pos, effort_features, "start_delta_sec")
            end_mae = _loso_regressor_mae(effort_pos, effort_features, "end_delta_sec")
            metrics["effort_regressor_loso"] = {"start_delta_mae": start_mae, "end_delta_mae": end_mae}

            start_path = model_root / "regressor" / "effort_start_delta_xgb.joblib"
            end_path = model_root / "regressor" / "effort_end_delta_xgb.joblib"
            _fit_and_save_regressor(effort_pos, effort_features, "start_delta_sec", start_path)
            _fit_and_save_regressor(effort_pos, effort_features, "end_delta_sec", end_path)
            saved_files.extend([str(start_path), str(end_path)])
        else:
            metrics["effort_regressor_loso"] = {"start_delta_mae": None, "end_delta_mae": None}

    if not sprint_df.empty:
        sprint_cls_metrics = _loso_classifier_metrics(sprint_df, sprint_features, "keep_label")
        metrics["sprint_classifier_loso"] = sprint_cls_metrics

        sprint_cls_path = model_root / "classifier" / "sprint_keep_xgb.joblib"
        _fit_and_save_classifier(sprint_df, sprint_features, "keep_label", sprint_cls_path)
        saved_files.append(str(sprint_cls_path))

        sprint_pos = sprint_df[sprint_df["keep_label"] == 1].copy()
        if len(sprint_pos) >= 5:
            start_mae = _loso_regressor_mae(sprint_pos, sprint_features, "start_delta_sec")
            end_mae = _loso_regressor_mae(sprint_pos, sprint_features, "end_delta_sec")
            metrics["sprint_regressor_loso"] = {"start_delta_mae": start_mae, "end_delta_mae": end_mae}

            start_path = model_root / "regressor" / "sprint_start_delta_xgb.joblib"
            end_path = model_root / "regressor" / "sprint_end_delta_xgb.joblib"
            _fit_and_save_regressor(sprint_pos, sprint_features, "start_delta_sec", start_path)
            _fit_and_save_regressor(sprint_pos, sprint_features, "end_delta_sec", end_path)
            saved_files.extend([str(start_path), str(end_path)])
        else:
            metrics["sprint_regressor_loso"] = {"start_delta_mae": None, "end_delta_mae": None}

    return TrainArtifacts(metrics=metrics, saved_files=saved_files)
