from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_training import train_from_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train effort/sprint models from processed datasets")
    parser.add_argument("--effort-csv", type=Path, default=Path("data/processed/effort_training.csv"))
    parser.add_argument("--sprint-csv", type=Path, default=Path("data/processed/sprint_training.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--metrics-out", type=Path, default=Path("models/training_metrics.json"))
    parser.add_argument("--history-out", type=Path, default=Path("models/training_history.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("models/training_summary.md"))
    parser.add_argument("--patterns-out", type=Path, default=Path("models/patterns_found.md"))
    return parser.parse_args()


def _extract_metric(metrics: dict, section: str, key: str) -> float | None:
    section_obj = metrics.get(section, {})
    if not isinstance(section_obj, dict):
        return None
    value = section_obj.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_history_row(
    history_out: Path,
    effort_df: pd.DataFrame,
    sprint_df: pd.DataFrame,
    metrics: dict,
) -> None:
    history_out.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sessions_count": int(effort_df["session_id"].nunique()) if "session_id" in effort_df.columns else 0,
        "effort_rows": int(len(effort_df)),
        "sprint_rows": int(len(sprint_df)),
        "effort_accuracy": _extract_metric(metrics, "effort_classifier_loso", "accuracy"),
        "effort_precision": _extract_metric(metrics, "effort_classifier_loso", "precision"),
        "effort_recall": _extract_metric(metrics, "effort_classifier_loso", "recall"),
        "effort_f1": _extract_metric(metrics, "effort_classifier_loso", "f1"),
        "effort_start_delta_mae": _extract_metric(metrics, "effort_regressor_loso", "start_delta_mae"),
        "effort_end_delta_mae": _extract_metric(metrics, "effort_regressor_loso", "end_delta_mae"),
        "sprint_accuracy": _extract_metric(metrics, "sprint_classifier_loso", "accuracy"),
        "sprint_precision": _extract_metric(metrics, "sprint_classifier_loso", "precision"),
        "sprint_recall": _extract_metric(metrics, "sprint_classifier_loso", "recall"),
        "sprint_f1": _extract_metric(metrics, "sprint_classifier_loso", "f1"),
        "sprint_start_delta_mae": _extract_metric(metrics, "sprint_regressor_loso", "start_delta_mae"),
        "sprint_end_delta_mae": _extract_metric(metrics, "sprint_regressor_loso", "end_delta_mae"),
    }

    row_df = pd.DataFrame([row])
    if history_out.exists():
        # Support legacy files that may still use comma separator.
        prev_df = pd.read_csv(history_out, sep=None, engine="python")
        out_df = pd.concat([prev_df, row_df], ignore_index=True)
    else:
        out_df = row_df

    out_df.to_csv(history_out, index=False, sep=";", encoding="utf-8-sig")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _write_summary_md(summary_out: Path, effort_df: pd.DataFrame, sprint_df: pd.DataFrame, metrics: dict) -> None:
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    e_acc = _extract_metric(metrics, "effort_classifier_loso", "accuracy")
    e_f1 = _extract_metric(metrics, "effort_classifier_loso", "f1")
    e_start = _extract_metric(metrics, "effort_regressor_loso", "start_delta_mae")
    e_end = _extract_metric(metrics, "effort_regressor_loso", "end_delta_mae")
    s_acc = _extract_metric(metrics, "sprint_classifier_loso", "accuracy")
    s_f1 = _extract_metric(metrics, "sprint_classifier_loso", "f1")
    s_start = _extract_metric(metrics, "sprint_regressor_loso", "start_delta_mae")
    s_end = _extract_metric(metrics, "sprint_regressor_loso", "end_delta_mae")

    sessions_count = int(effort_df["session_id"].nunique()) if "session_id" in effort_df.columns else 0
    diag = metrics.get("data_diagnostics", {}) if isinstance(metrics.get("data_diagnostics", {}), dict) else {}
    effort_diag = diag.get("effort_class_stats", {}) if isinstance(diag.get("effort_class_stats", {}), dict) else {}
    sprint_diag = diag.get("sprint_class_stats", {}) if isinstance(diag.get("sprint_class_stats", {}), dict) else {}
    notes = metrics.get("training_notes", []) if isinstance(metrics.get("training_notes", []), list) else []

    content = (
        "# Training Summary\n\n"
        f"- Timestamp UTC: {datetime.now(UTC).isoformat()}\n"
        f"- Sessions: {sessions_count}\n"
        f"- Effort rows: {len(effort_df)}\n"
        f"- Sprint rows: {len(sprint_df)}\n\n"
        "## Effort\n"
        f"- Accuracy: {_fmt_pct(e_acc)}\n"
        f"- F1: {_fmt_pct(e_f1)}\n"
        f"- Start delta MAE (s): {_fmt_num(e_start)}\n"
        f"- End delta MAE (s): {_fmt_num(e_end)}\n\n"
        "## Sprint\n"
        f"- Accuracy: {_fmt_pct(s_acc)}\n"
        f"- F1: {_fmt_pct(s_f1)}\n"
        f"- Start delta MAE (s): {_fmt_num(s_start)}\n"
        f"- End delta MAE (s): {_fmt_num(s_end)}\n"
    )
    content += "\n## Diagnostica classi\n"
    content += f"- Effort pos/neg: {effort_diag.get('positives', 0)}/{effort_diag.get('negatives', 0)}\n"
    content += f"- Sprint pos/neg: {sprint_diag.get('positives', 0)}/{sprint_diag.get('negatives', 0)}\n"

    if notes:
        content += "\n## Note training\n"
        for note in notes:
            content += f"- {note}\n"

    summary_out.write_text(content, encoding="utf-8")


def _write_patterns_md(patterns_out: Path, metrics: dict) -> None:
    patterns_out.parent.mkdir(parents=True, exist_ok=True)
    fi = metrics.get("feature_importance", {})
    if not isinstance(fi, dict):
        fi = {}

    lines: list[str] = ["# Patterns Found by Model", ""]
    lines.append("Questo file mostra le feature che il modello usa di piu (importanza).")
    lines.append("Se compaiono pattern utili che non stavi guardando, puoi annotarli e inserirli nel processo.")
    lines.append("")

    sections = [
        ("effort_classifier", "Effort - Keep/Delete"),
        ("effort_start_regressor", "Effort - Correzione Start"),
        ("effort_end_regressor", "Effort - Correzione End"),
        ("sprint_classifier", "Sprint - Keep/Delete"),
        ("sprint_start_regressor", "Sprint - Correzione Start"),
        ("sprint_end_regressor", "Sprint - Correzione End"),
    ]

    for key, title in sections:
        lines.append(f"## {title}")
        values = fi.get(key, [])
        if not values:
            lines.append("- n/a")
            lines.append("")
            continue
        for item in values[:10]:
            feature = item.get("feature", "unknown")
            importance = item.get("importance", 0.0)
            try:
                score = float(importance)
            except (TypeError, ValueError):
                score = 0.0
            lines.append(f"- {feature}: {score:.4f}")
        lines.append("")

    patterns_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not args.effort_csv.exists():
        print(f"Missing effort dataset: {args.effort_csv}")
        return 1

    effort_df = pd.read_csv(args.effort_csv, sep=";")
    sprint_df = pd.read_csv(args.sprint_csv, sep=";") if args.sprint_csv.exists() else pd.DataFrame()

    artifacts = train_from_frames(effort_df=effort_df, sprint_df=sprint_df, model_root=args.model_root)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps({"metrics": artifacts.metrics, "saved_files": artifacts.saved_files}, indent=2),
        encoding="utf-8",
    )

    _append_history_row(
        history_out=args.history_out,
        effort_df=effort_df,
        sprint_df=sprint_df,
        metrics=artifacts.metrics,
    )
    _write_summary_md(
        summary_out=args.summary_out,
        effort_df=effort_df,
        sprint_df=sprint_df,
        metrics=artifacts.metrics,
    )
    _write_patterns_md(patterns_out=args.patterns_out, metrics=artifacts.metrics)

    print("Training completed")
    print(f"Models saved: {len(artifacts.saved_files)}")
    for p in artifacts.saved_files:
        print(f"- {p}")
    print(f"Metrics: {args.metrics_out}")
    print(f"History: {args.history_out}")
    print(f"Summary: {args.summary_out}")
    print(f"Patterns: {args.patterns_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
