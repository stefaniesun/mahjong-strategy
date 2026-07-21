from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console.agent_a import AgentManager, create_app
from console.config import load_config


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command, self.kwargs, self.pid, self.code = command, kwargs, 1234, None
        self.done = threading.Event()
    def poll(self): return self.code
    def wait(self):
        self.done.wait(2)
        return 0 if self.code is None else self.code
    def finish(self, code=0):
        self.code = code
        self.done.set()


def config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(f"project_root: {tmp_path.as_posix()}\ntraining:\n  script: tools/cloud_train_s5.py\n  output_dir: out\nagent:\n  state_dir: state\n", encoding="utf-8")
    (tmp_path / "tools").mkdir(exist_ok=True)
    (tmp_path / "tools/cloud_train_s5.py").touch()
    return load_config(path)


def test_command_whitelist_pause_and_generation_safe_monitor(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []
    def popen(command, **kwargs):
        process = FakeProcess(command, **kwargs); processes.append(process); return process
    manager = AgentManager(config(tmp_path), popen=popen, pid_command=lambda _pid: "cloud_train_s5.py")
    result = manager.start({"updates": 2, "device": "cpu"}, False)
    assert result["status"] == "RUNNING"
    assert processes[0].kwargs["shell"] is False
    assert processes[0].command[processes[0].command.index("--updates") + 1] == "2"
    with pytest.raises(ValueError, match="not allowed"):
        manager.command({"script": "bad.py"}, False)
    assert manager.pause()["status"] == "PAUSING"
    processes[0].finish()
    processes[0].done.wait()


def test_api_snapshot_listing_file_safety_and_bounded_tail(tmp_path: Path) -> None:
    manager = AgentManager(config(tmp_path), pid_command=lambda _pid: None)
    snapshot = manager.output / "checkpoints/snapshots/s5-step-1.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"checkpoint")
    diagnostic = manager.output / "checkpoints/diagnostic.pt"
    diagnostic.write_bytes(b"bad")
    manager.log_file.write_text("\n".join(str(i) for i in range(3000)), encoding="utf-8")
    client = TestClient(create_app(manager.config, manager))
    rows = client.get("/checkpoints").json()
    assert [row["name"] for row in rows] == ["snapshots/s5-step-1.pt"]
    assert client.get("/checkpoints/snapshots/s5-step-1.pt").content == b"checkpoint"
    assert client.get("/checkpoints/..%2Fsecret.pt").status_code in {404, 422}
    assert len(client.get("/log_tail?lines=10").text.splitlines()) == 10


def test_restart_keeps_live_managed_pid_and_prevents_second_start(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    state = cfg.agent_state_dir / "state.json"
    state.parent.mkdir(parents=True)
    script = cfg.project_root / "tools/cloud_train_s5.py"
    state.write_text(f'{{"status":"RUNNING","pid":99,"command":["python","{script.as_posix()}","--output-dir","{cfg.output_dir.as_posix()}"]}}', encoding="utf-8")
    live_command = f"python {script} --output-dir {cfg.output_dir}"
    calls = 0
    def pid_command(_pid):
        nonlocal calls
        calls += 1
        return live_command if calls < 4 else None
    manager = AgentManager(cfg, pid_command=pid_command)
    assert manager.state["status"] == "RUNNING"
    with pytest.raises(RuntimeError, match="already"):
        manager.start({}, True)
    for _ in range(30):
        if manager.state["status"] == "COMPLETED":
            break
        threading.Event().wait(0.1)
    assert manager.state["status"] == "COMPLETED"


def test_dead_pid_self_heals_without_autostart(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    latest = cfg.output_dir / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True)
    latest.touch()
    state = cfg.agent_state_dir / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"status":"RUNNING","pid":99,"command":["python","cloud_train_s5.py"]}', encoding="utf-8")
    manager = AgentManager(cfg, pid_command=lambda _pid: None)
    assert manager.state["status"] == "PAUSED"
    assert manager.process is None


def test_start_failure_is_persisted(tmp_path: Path) -> None:
    def fail(*_args, **_kwargs): raise FileNotFoundError("missing python")
    manager = AgentManager(config(tmp_path), popen=fail)
    with pytest.raises(RuntimeError, match="failed to start"):
        manager.start({}, False)
    assert manager.state["status"] == "ERROR"
