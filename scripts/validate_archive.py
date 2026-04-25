from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.archive import discover_sessions, load_json
from src.schema import validate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate archive session folders")
    parser.add_argument("--archive", type=Path, default=Path("data/archive"), help="Archive root path")
    parser.add_argument("--report", type=Path, default=Path("data/processed/archive_validation_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan = discover_sessions(args.archive)

    payload_errors: list[str] = []
    for session in scan.sessions:
        default_errors = validate_payload(load_json(session.default_json_path))
        gold_errors = validate_payload(load_json(session.gold_json_path))
        if default_errors:
            payload_errors.append(f"{session.default_json_path}: {'; '.join(default_errors)}")
        if gold_errors:
            payload_errors.append(f"{session.gold_json_path}: {'; '.join(gold_errors)}")

    report = {
        "archive": str(args.archive),
        "valid_session_folders": len(scan.sessions),
        "layout_errors": scan.errors,
        "payload_errors": payload_errors,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validated sessions: {len(scan.sessions)}")
    print(f"Layout errors: {len(scan.errors)}")
    print(f"Payload errors: {len(payload_errors)}")
    print(f"Report: {args.report}")

    return 0 if not scan.errors and not payload_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
