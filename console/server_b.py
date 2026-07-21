from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from console.config import ConsoleConfig
from console.scheduler import EvaluationScheduler
from console.storage import read_jsonl


def create_app(config: ConsoleConfig, scheduler: EvaluationScheduler | None = None) -> FastAPI:
    managed = scheduler or EvaluationScheduler(config)
    stop = threading.Event()
    def loop():
        while not stop.is_set():
            try:
                managed.sync_metrics(); managed.sync_checkpoints(); managed.evaluate_next()
            except Exception as exc:
                from console.storage import append_jsonl
                from datetime import datetime, timezone
                append_jsonl(managed.events, {"timestamp": datetime.now(timezone.utc).isoformat(), "level": "error", "message": str(exc)})
            stop.wait(config.evaluation.poll_seconds)
    @asynccontextmanager
    async def lifespan(_app):
        thread = threading.Thread(target=loop, daemon=True); thread.start()
        yield
        stop.set(); thread.join(timeout=2)
    app = FastAPI(title="S5 Console B", lifespan=lifespan)
    static = Path(__file__).parent / "static/index.html"
    @app.get("/")
    def index(): return FileResponse(static)
    @app.get("/api/status")
    def status():
        try: agent = managed.client.json("/status")
        except Exception as exc: agent = {"status": "OFFLINE", "error": str(exc)}
        return {
            "agent": agent, "evaluator_paused": managed.thermal_paused,
            "server_telemetry": read_jsonl(managed.telemetry, limit=8640),
        }
    @app.get("/api/history")
    def history(limit: int = 2000): return read_jsonl(managed.metrics, limit=min(max(limit, 1), 10000))
    @app.get("/api/evaluations")
    def evaluations(): return read_jsonl(managed.evals, limit=1000)
    @app.get("/api/events")
    def events(): return read_jsonl(managed.events, limit=1000)
    @app.post("/api/start")
    def start(payload: dict[str, object]):
        try:
            result = managed.client.post("/start", payload)
            managed.event("info", "training_start_requested")
            return result
        except Exception as exc:
            raise HTTPException(502, str(exc)) from exc
    @app.post("/api/pause")
    def pause():
        try:
            result = managed.client.post("/pause")
            managed.event("info", "training_pause_requested")
            return result
        except Exception as exc:
            raise HTTPException(502, str(exc)) from exc
    return app
