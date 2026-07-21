from __future__ import annotations

from pathlib import Path


def test_deployment_templates_reference_real_entries_and_never_auto_train() -> None:
    root = Path(__file__).parents[1]
    windows = (root / "console/deploy/windows/start-agent.ps1").read_text(encoding="utf-8")
    task = (root / "console/deploy/windows/install-scheduled-task.ps1").read_text(encoding="utf-8")
    service = (root / "console/deploy/linux/s5-console.service").read_text(encoding="utf-8")
    example = (root / "configs/s5_console.example.yaml").read_text(encoding="utf-8")
    page = (root / "console/static/index.html").read_text(encoding="utf-8")
    assert "console.run_agent_a" in windows
    assert "console.run_server_b" in service
    assert "cloud_train_s5" not in windows + task + service
    assert "127.0.0.1" in example
    assert "cdn" not in page.lower()
    assert "docker" not in service.lower()
