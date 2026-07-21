from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from console.config import load_config
from console.scheduler import EvaluationScheduler
from console.server_b import create_app


class Client:
    def json(self, _path): raise OSError("offline")
    def post(self, _path, payload=None): return {"forwarded": payload}


def test_offline_status_and_forwarding(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"; cfg_path.write_text(f"project_root: {tmp_path.as_posix()}\nserver:\n  state_dir: state\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    scheduler = EvaluationScheduler(cfg, client=Client())
    client = TestClient(create_app(cfg, scheduler))
    assert client.get("/api/status").json()["agent"]["status"] == "OFFLINE"
    page = client.get("/")
    assert page.status_code == 200
    assert "cdn" not in page.text.lower()
    assert "<svg" in page.text
    assert client.post("/api/start", json={"overrides": {"updates": 2}}).status_code == 200
