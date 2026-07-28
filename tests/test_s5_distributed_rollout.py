from __future__ import annotations

import io
import json
import threading
import urllib.error
from http.server import ThreadingHTTPServer

import pytest

from console.rollout_worker import RolloutWorkerHandler, RolloutWorkerServer
from rl.distributed_rollout import RemoteRolloutClient, RemoteRolloutError, trajectory_step_from_dict, trajectory_step_to_dict
from rl.types import TrajectoryStep


def test_trajectory_step_round_trip() -> None:
    step = TrajectoryStep(
        feature_values=(1.0, 2.0),
        legal_mask=(True, False, True),
        action=2,
        old_log_prob=-0.4,
        value=0.3,
        reward=1.0,
        done=True,
        policy_version="policy-7",
    )

    restored = trajectory_step_from_dict(trajectory_step_to_dict(step))

    assert restored == step


class _FakeWorker:
    def execute(self, payload):
        task = payload["tasks"][0]
        return {
            "protocol_version": 1,
            "policy_checksum": payload["policy_checksum"],
            "completed_games": 1,
            "steps": [{
                "feature_values": [1.0],
                "legal_mask": [True],
                "action": 0,
                "old_log_prob": 0.0,
                "value": 0.0,
                "reward": 1.0,
                "done": True,
                "policy_version": task["policy_generation"],
            }],
        }


def test_rollout_worker_http_authentication() -> None:
    server = RolloutWorkerServer(("127.0.0.1", 0), _FakeWorker(), "secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RemoteRolloutClient(f"http://127.0.0.1:{server.server_port}", token="wrong")
        with pytest.raises(RemoteRolloutError, match="HTTP 401"):
            client._post("/v1/rollout", {"tasks": []})
        assert RemoteRolloutClient(f"http://127.0.0.1:{server.server_port}").health()["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_client_retries_transient_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class _Response:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return io.BytesIO(b'{"ok":true}').read()

    def fake_urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError(ConnectionResetError(10054, "connection reset"))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RemoteRolloutClient(
        "http://worker.invalid",
        max_attempts=2,
        retry_delay_seconds=0,
    )

    assert client._post("/v1/rollout", {"hello": "world"}) == {"ok": True}
    assert attempts == 2
