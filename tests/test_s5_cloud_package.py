from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s5_cloud_package_is_deterministic_and_contains_only_runtime_assets(tmp_path: Path) -> None:
    from tools.cloud_train_s5 import build_s5_cloud_package

    first = build_s5_cloud_package(tmp_path / "one.zip", project_root=ROOT)
    second = build_s5_cloud_package(tmp_path / "two.zip", project_root=ROOT)

    assert _sha256(first) == _sha256(second)
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("s5_cloud_package_manifest.json"))
        assert names == sorted(names)
        assert "tools/cloud_train_s5.py" in names
        assert "rl/train_rl.py" in names
        assert "engine/game.py" in names
        assert "state/tile_belief.py" in names
        assert "policies/rule_policy.py" in names
        assert "training_artifacts/S4/v1_20260711_repaired_cuda/checkpoints/belief_s4.pt" in names
        assert "training_artifacts/S4/v1_20260711_repaired_cuda/checkpoints/policy_s4.pt" in names
        assert "S5_CLOUD_TRAINING_README.md" in names
        assert "docs/concepts.md" in names
        assert all("s4_decisions.jsonl" not in name for name in names)
        assert all(not name.endswith(".zip") for name in names)
        assert manifest["format_version"] == 2
        assert manifest["inventory"]["policy"] == "closed"
        assert manifest["inventory"]["all_files"] == names
        assert manifest["s4_artifacts"]["policy"]["sha256"] == _sha256(
            ROOT / "training_artifacts/S4/v1_20260711_repaired_cuda/checkpoints/policy_s4.pt"
        )
        for item in manifest["files"]:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]


def test_s5_cloud_package_extracts_verifies_and_runs_cpu_smoke(tmp_path: Path) -> None:
    from tools.cloud_train_s5 import build_s5_cloud_package, parse_args

    package = build_s5_cloud_package(tmp_path / "s5.zip", project_root=ROOT)
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(package) as archive:
        archive.extractall(extract_dir)

    marker = tmp_path / "sitecustomize-ran.marker"
    (extract_dir / "tools" / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unexpected startup import', encoding='utf-8')\n"
        "raise RuntimeError('sitecustomize must not run')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-S", "tools/cloud_train_s5.py", "--mode", "smoke", "--device", "cpu", "--output-dir", str(tmp_path / "out")],
        cwd=extract_dir,
        text=True,
        capture_output=True,
    )
    assert not marker.exists()
    assert completed.returncode != 0
    assert "unexpected package file: tools/sitecustomize.py" in completed.stderr
    assert not (tmp_path / "out" / "s5_cloud_run_manifest.json").exists()
    assert parse_args(["--mode", "smoke", "--device", "cpu"]).mode == "smoke"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "invalid"])


def test_s5_cloud_cli_rejects_startup_without_no_site_flag(tmp_path: Path) -> None:
    """The documented CLI rejects unsafe Python startup before package imports."""
    completed = subprocess.run(
        [sys.executable, "tools/cloud_train_s5.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "python -S tools/cloud_train_s5.py" in completed.stderr


def test_s5_cloud_arena_budget_defaults_by_mode(tmp_path: Path) -> None:
    from tools.cloud_train_s5 import S5CloudRunConfig

    assert S5CloudRunConfig(tmp_path / "smoke", mode="smoke").arena_games == 4
    assert S5CloudRunConfig(tmp_path / "train", mode="train").arena_games == 1000


def test_s5_cloud_formal_train_composes_real_rollout_and_arena(monkeypatch, tmp_path: Path) -> None:
    """Formal training must wire real game adapters, never synthetic steps/metrics."""
    import tools.cloud_train_s5 as cloud

    captured = {}

    def fake_run(config, *, dependencies):
        captured["config"] = config
        captured["dependencies"] = dependencies
        return Mock(
            report_path=tmp_path / "report.json", markdown_path=tmp_path / "report.md",
            checkpoint_path=tmp_path / "latest.pt", publication_manifest_path=tmp_path / "report.manifest.json",
            report={"arena": {"s3_comparison": {"perfect_win_rate": 0.5, "degraded_win_rate": 0.5}}}, global_step=1,
        )

    report_json, report_markdown, checkpoint = "{}", "# report\n", "checkpoint"
    (tmp_path / "report.json").write_text(report_json, encoding="utf-8")
    (tmp_path / "report.md").write_text(report_markdown, encoding="utf-8")
    (tmp_path / "latest.pt").write_text(checkpoint, encoding="utf-8")
    (tmp_path / "report.manifest.json").write_text(json.dumps({
        "format_version": 1,
        "artifacts": {
            "json": {"name": "report.json", "sha256": hashlib.sha256(report_json.encode()).hexdigest()},
            "markdown": {"name": "report.md", "sha256": hashlib.sha256(report_markdown.encode()).hexdigest()},
        },
    }), encoding="utf-8")
    monkeypatch.setattr("rl.train_rl.run_s5_training", fake_run)
    result = cloud.run_s5_cloud_training(
        cloud.S5CloudRunConfig(tmp_path / "out", mode="train", device="cpu", updates=1, episodes_per_update=2, arena_games=2),
        project_root=ROOT,
    )

    assert result.global_step == 1
    dependencies = captured["dependencies"]
    assert dependencies.rollout_factory.__name__ == "rollout"
    assert dependencies.arena_evaluator.__name__ == "arena"
    assert "controlled" not in result.manifest_path.read_text(encoding="utf-8")


def test_s5_cloud_rejects_tampered_package_manifest_before_run(tmp_path: Path) -> None:
    from tools.cloud_train_s5 import S5CloudRunConfig, run_s5_cloud_training

    copied = tmp_path / "project"
    import shutil
    shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "cloud_outputs", "cloud_packages"))
    (copied / "s5_cloud_package_manifest.json").write_text(json.dumps({"format_version": 1, "files": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="package manifest"):
        run_s5_cloud_training(S5CloudRunConfig(copied / "out", mode="smoke", device="cpu"), project_root=copied)


@pytest.mark.parametrize("rogue_name", ("tools/rogue_policy.py", "rl/rogue_policy.py"))
def test_s5_cloud_package_rejects_unknown_extracted_source_file_before_run(tmp_path: Path, rogue_name: str) -> None:
    """The signed inventory is closed: unpacked code cannot add importable files."""
    from tools.cloud_train_s5 import S5CloudRunConfig, build_s5_cloud_package, run_s5_cloud_training

    package = build_s5_cloud_package(tmp_path / "s5.zip", project_root=ROOT)
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(package) as archive:
        archive.extractall(extract_dir)

    rogue = extract_dir / rogue_name
    rogue.parent.mkdir(parents=True, exist_ok=True)
    rogue.write_text("raise RuntimeError('must not run')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected package file"):
        run_s5_cloud_training(
            S5CloudRunConfig(tmp_path / "out", mode="smoke", device="cpu"), project_root=extract_dir
        )


def test_s5_cloud_maps_rollout_hash_seed_to_torch_supported_range() -> None:
    from tools.cloud_train_s5 import _torch_sampling_seed

    assert 0 <= _torch_sampling_seed(2**256 - 1) < 2**63


def test_s5_cloud_runner_rejects_unavailable_required_cuda(monkeypatch, tmp_path: Path) -> None:
    import torch

    from tools.cloud_train_s5 import S5CloudRunConfig, run_s5_cloud_training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was required"):
        run_s5_cloud_training(
            S5CloudRunConfig(output_dir=tmp_path / "out", mode="smoke", device="cuda", updates=1)
        )


def test_s5_cloud_runner_executes_from_tools_directory_context(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-S", "tools/cloud_train_s5.py", "--mode", "smoke", "--device", "cpu", "--output-dir", str(tmp_path / "out")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "out" / "s5_cloud_run_manifest.json").is_file()
