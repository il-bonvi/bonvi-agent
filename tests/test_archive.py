from __future__ import annotations

from pathlib import Path

from src.archive import discover_sessions


def test_discover_sessions_ignores_effort_modifications(tmp_path: Path) -> None:
    session_dir = tmp_path / "gare" / "sessione_1"
    session_dir.mkdir(parents=True)

    (session_dir / "esempio.fit").write_text("", encoding="utf-8")
    (session_dir / "esempio_default.json").write_text("{}", encoding="utf-8")
    (session_dir / "esempio_gold.json").write_text("{}", encoding="utf-8")
    (session_dir / "effort_modifications.json").write_text("{}", encoding="utf-8")

    scan = discover_sessions(tmp_path)

    assert scan.errors == []
    assert len(scan.sessions) == 1
    assert scan.sessions[0].stem == "esempio"


def test_discover_sessions_reports_stem_mismatch(tmp_path: Path) -> None:
    session_dir = tmp_path / "allenamenti" / "sessione_2"
    session_dir.mkdir(parents=True)

    (session_dir / "esempio.fit").write_text("", encoding="utf-8")
    (session_dir / "aaa_default.json").write_text("{}", encoding="utf-8")
    (session_dir / "bbb_gold.json").write_text("{}", encoding="utf-8")

    scan = discover_sessions(tmp_path)

    assert len(scan.sessions) == 0
    assert len(scan.errors) == 1


def test_discover_sessions_supports_root_level_session(tmp_path: Path) -> None:
    (tmp_path / "rootsession.fit").write_text("", encoding="utf-8")
    (tmp_path / "rootsession_default.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rootsession_gold.json").write_text("{}", encoding="utf-8")
    (tmp_path / "effort_modifications.json").write_text("{}", encoding="utf-8")

    scan = discover_sessions(tmp_path)

    assert scan.errors == []
    assert len(scan.sessions) == 1
    assert scan.sessions[0].stem == "rootsession"
