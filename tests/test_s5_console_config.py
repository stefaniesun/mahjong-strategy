from __future__ import annotations

import json
from pathlib import Path

import pytest

from console.config import ALLOWED_TRAINING_OVERRIDES, load_config
from console.storage import append_jsonl, atomic_write_json, read_jsonl, safe_path, sha256_file


def test_strict_config_defaults_and_whitelist(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("project_root: .\ntraining:\n  updates: 7\n", encoding="utf-8")
    config = load_config(path)
    assert config.project_root == tmp_path
    assert config.training.updates == 7
    assert config.agent.host == "127.0.0.1"
    assert set(config.training_values({"device": "cpu"})) >= ALLOWED_TRAINING_OVERRIDES
    with pytest.raises(ValueError, match="not allowed"):
        config.training_values({"script": "evil.py"})


def test_unknown_and_invalid_config_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
    path.write_text("health:\n  evaluator_cpu_pause_c: 80\n  evaluator_cpu_resume_c: 90\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resume"):
        load_config(path)
    path.write_text("training:\n  output_dir: ../outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_config(path)
    path.write_text("evaluation:\n  quick_games: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_config(path)
    path.write_text("agent:\n  port: '8765'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="port"):
        load_config(path)


def test_storage_is_safe_atomic_and_tolerates_only_bad_tail(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert safe_path(root, "a", "b").is_relative_to(root)
    with pytest.raises(ValueError, match="escapes"):
        safe_path(root, "..", "outside")
    state = root / "state.json"
    atomic_write_json(state, {"ok": True})
    assert json.loads(state.read_text()) == {"ok": True}
    rows = root / "rows.jsonl"
    append_jsonl(rows, {"n": 1})
    rows.write_text(rows.read_text() + '{"broken":', encoding="utf-8")
    assert read_jsonl(rows) == [{"n": 1}]
    rows.write_text('{"broken":\n{"n":2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(rows)
    blob = root / "blob"
    blob.write_bytes(b"abc")
    assert sha256_file(blob) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
