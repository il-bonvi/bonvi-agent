from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class SessionTriple:
    folder: Path
    stem: str
    fit_path: Path
    default_json_path: Path
    gold_json_path: Path


@dataclass(frozen=True)
class ArchiveScanResult:
    sessions: list[SessionTriple]
    errors: list[str]


def discover_sessions(archive_root: str | Path) -> ArchiveScanResult:
    """Discover valid session folders recursively under archive root.

    A valid session folder contains:
    - <stem>.fit
    - <stem>_default.json
    - <stem>_gold.json
    Optional and ignored:
    - effort_modifications.json
    """
    root = Path(archive_root)
    sessions: list[SessionTriple] = []
    errors: list[str] = []

    if not root.exists():
        return ArchiveScanResult(sessions=[], errors=[f"Archive path not found: {root}"])

    candidate_folders = [root]
    candidate_folders.extend(sorted([d for d in root.rglob("*") if d.is_dir()]))

    for folder in candidate_folders:
        files = [f for f in folder.iterdir() if f.is_file()]
        if not files:
            continue

        fit_files = [f for f in files if f.suffix.lower() == ".fit"]
        default_files = [f for f in files if f.name.endswith("_default.json")]
        gold_files = [f for f in files if f.name.endswith("_gold.json")]

        # Skip folders that do not look like a session folder.
        if not (fit_files or default_files or gold_files):
            continue

        if len(fit_files) != 1 or len(default_files) != 1 or len(gold_files) != 1:
            errors.append(
                f"{folder}: expected exactly 1 .fit, 1 _default.json, 1 _gold.json"
            )
            continue

        fit_path = fit_files[0]
        default_path = default_files[0]
        gold_path = gold_files[0]

        fit_stem = fit_path.stem
        default_stem = default_path.name[: -len("_default.json")]
        gold_stem = gold_path.name[: -len("_gold.json")]

        if not (fit_stem == default_stem == gold_stem):
            errors.append(
                f"{folder}: stem mismatch -> fit={fit_stem}, default={default_stem}, gold={gold_stem}"
            )
            continue

        sessions.append(
            SessionTriple(
                folder=folder,
                stem=fit_stem,
                fit_path=fit_path,
                default_json_path=default_path,
                gold_json_path=gold_path,
            )
        )

    return ArchiveScanResult(sessions=sessions, errors=errors)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
