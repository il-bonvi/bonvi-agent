from __future__ import annotations

import argparse
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.effort_csv.exists():
        print(f"Missing effort dataset: {args.effort_csv}")
        return 1

    effort_df = pd.read_csv(args.effort_csv)
    sprint_df = pd.read_csv(args.sprint_csv) if args.sprint_csv.exists() else pd.DataFrame()

    artifacts = train_from_frames(effort_df=effort_df, sprint_df=sprint_df, model_root=args.model_root)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps({"metrics": artifacts.metrics, "saved_files": artifacts.saved_files}, indent=2),
        encoding="utf-8",
    )

    print("Training completed")
    print(f"Models saved: {len(artifacts.saved_files)}")
    for p in artifacts.saved_files:
        print(f"- {p}")
    print(f"Metrics: {args.metrics_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
