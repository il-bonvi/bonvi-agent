from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.archive import discover_sessions
from src.dataset_builder import build_training_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build effort/sprint training datasets from archive")
    parser.add_argument("--archive", type=Path, default=Path("data/archive"), help="Archive root path")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output folder")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for default-gold match")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scan = discover_sessions(args.archive)
    if scan.errors:
        print("Archive layout errors found:")
        for err in scan.errors:
            print(f"- {err}")
        return 1

    result = build_training_frames(scan.sessions, iou_threshold=args.iou_threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    effort_csv = args.out_dir / "effort_training.csv"
    sprint_csv = args.out_dir / "sprint_training.csv"
    report_json = args.out_dir / "dataset_build_report.json"

    result.effort_df.to_csv(effort_csv, index=False, sep=";", encoding="utf-8-sig")
    result.sprint_df.to_csv(sprint_csv, index=False, sep=";", encoding="utf-8-sig")

    report = {
        "sessions_used": len(scan.sessions),
        "effort_rows": int(len(result.effort_df)),
        "sprint_rows": int(len(result.sprint_df)),
        "warnings": result.warnings,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Sessions used: {len(scan.sessions)}")
    print(f"Effort rows: {len(result.effort_df)}")
    print(f"Sprint rows: {len(result.sprint_df)}")
    print(f"Saved: {effort_csv}")
    print(f"Saved: {sprint_csv}")
    print(f"Report: {report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
