"""Standalone HTTP service that runs production rollout batches on machine B."""
from __future__ import annotations

import argparse
import gzip
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

from rl.distributed_rollout import RolloutWorker


class RolloutWorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, worker: RolloutWorker, token: str = "") -> None:
        super().__init__(address, RolloutWorkerHandler)
        self.worker = worker
        self.token = token


class RolloutWorkerHandler(BaseHTTPRequestHandler):
    server: RolloutWorkerServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "service": "s5-rollout-worker"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/rollout":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 512 * 1024 * 1024:
                raise ValueError("invalid request size")
            content = self.rfile.read(length)
            if self.headers.get("Content-Encoding") == "gzip":
                content = gzip.decompress(content)
            payload = json.loads(content.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("request body must be an object")
            result = self.server.worker.execute(payload)
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        token = self.server.token
        return not token or self.headers.get("Authorization") == f"Bearer {token}"

    def _json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        compressed = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024
        if compressed:
            body = gzip.compress(body, compresslevel=1)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the S5 distributed rollout worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--token", default="")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker = RolloutWorker(args.project_root, device=args.device, cache_dir=args.cache_dir)
    server = RolloutWorkerServer((args.host, args.port), worker, args.token)
    print(f"S5 rollout worker listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
