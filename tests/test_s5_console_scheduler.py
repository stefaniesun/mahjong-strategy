from __future__ import annotations

import hashlib
from pathlib import Path

from console.config import load_config
from console.scheduler import EvaluationScheduler


class Client:
    def __init__(self, data: bytes): self.data = data
    def json(self, path):
        if path == "/checkpoints": return [{"name": "latest.pt", "size": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}]
        return {"metrics": [{"global_step": 1, "timestamp": "now"}]}
    def download(self, _name, destination): destination.write_bytes(self.data)
    def post(self, _path, payload=None): return payload or {"ok": True}


def config(tmp_path: Path):
    path = tmp_path / "c.yaml"; path.write_text(f"project_root: {tmp_path.as_posix()}\nevaluation:\n  min_disk_free_gib: 0\nserver:\n  state_dir: state\n", encoding="utf-8")
    return load_config(path)


def test_atomic_sync_metrics_and_stable_tasks(tmp_path: Path) -> None:
    client = Client(b"checkpoint")
    scheduler = EvaluationScheduler(config(tmp_path), client=client, evaluator=lambda *args, **kwargs: {"ok": True})
    scheduler.sync_metrics(); scheduler.sync_metrics()
    files = scheduler.sync_checkpoints()
    assert len(files) == 1 and files[0].read_bytes() == b"checkpoint"
    tasks = scheduler.pending(files)
    assert [task[1] for task in tasks] == ["quick", "formal"]
    assert len({task[-1] for task in tasks}) == len(tasks)
