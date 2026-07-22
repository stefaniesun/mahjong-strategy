from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from console.config import ConsoleConfig
from console.storage import atomic_write_json, read_json, read_jsonl, safe_path, sha256_file
from console.telemetry import agent_pause_reason, collect_telemetry, record_telemetry

STATES = {"IDLE", "RUNNING", "PAUSING", "PAUSED", "COMPLETED", "ERROR"}


class StartRequest(BaseModel):
    overrides: dict[str, object] = Field(default_factory=dict)
    resume: bool = True


def _pid_command(pid: int) -> str | None:
    try:
        if platform.system() == "Windows":
            command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"]
        else:
            command = ["ps", "-p", str(pid), "-o", "args="]
        value = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _tail(path: Path, lines: int, block_size: int = 8192) -> str:
    if not path.exists():
        return ""
    wanted, chunks = min(max(lines, 1), 2000), []
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position, count = stream.tell(), 0
        while position > 0 and count <= wanted:
            size = min(block_size, position)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            count += chunk.count(b"\n")
    return "\n".join(b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()[-wanted:])


class AgentManager:
    def __init__(
        self,
        config: ConsoleConfig,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        pid_command: Callable[[int], str | None] = _pid_command,
        telemetry_collector: Callable[[Path], object] = collect_telemetry,
    ) -> None:
        self.config, self.popen, self.pid_command = config, popen, pid_command
        self.telemetry_collector = telemetry_collector
        self.root, self.output = config.project_root, config.output_dir
        self.state_dir = config.agent_state_dir
        self.state_file, self.stop_file = self.state_dir / "state.json", self.state_dir / "stop.flag"
        self.metrics_file, self.log_file = self.output / "metrics.jsonl", self.state_dir / "training.log"
        self.telemetry_file = self.state_dir / "telemetry.jsonl"
        self._lock = threading.RLock()
        self._stop_health = threading.Event()
        self._generation = 0
        self._adopted_pid: int | None = None
        self.process: subprocess.Popen | None = None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            loaded = read_json(self.state_file, {})
        except (OSError, ValueError):
            loaded = {}
        self.state: dict[str, object] = loaded if isinstance(loaded, dict) else {}
        self._recover_state()

    def _recover_state(self) -> None:
        status, pid, command = self.state.get("status"), self.state.get("pid"), self.state.get("command")
        if status in {"RUNNING", "PAUSING"} and isinstance(pid, int) and isinstance(command, list):
            actual = self.pid_command(pid)
            expected = str(self.config.resolve_path(self.config.training.script)).lower()
            if actual and expected in actual.lower() and all(token in actual.lower() for token in ("--output-dir", str(self.output).lower())):
                self.state["status"] = status
                self._adopted_pid = pid
                self._save()
                threading.Thread(target=self._monitor_adopted, args=(pid,), daemon=True).start()
                return
        self.state = {"status": "PAUSED" if self.latest_checkpoint().exists() else "IDLE"}
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.state_file, self.state)

    def latest_checkpoint(self) -> Path:
        return self.output / "checkpoints" / "latest.pt"

    def _monitor_adopted(self, pid: int) -> None:
        while self._adopted_pid == pid and self.pid_command(pid) is not None:
            time.sleep(min(self.config.health.poll_seconds, 5.0))
        with self._lock:
            if self._adopted_pid != pid:
                return
            self._adopted_pid = None
            requested = self.state.get("status") == "PAUSING"
            self.state.update(status="PAUSED" if requested else "COMPLETED")
            self._save()

    def _managed_running(self) -> bool:

        if self.process is not None:
            return self.process.poll() is None
        pid = self.state.get("pid")
        command = self.state.get("command")
        if not isinstance(pid, int) or not isinstance(command, list):
            return False
        actual = self.pid_command(pid)
        expected = str(self.config.resolve_path(self.config.training.script)).lower()
        return bool(actual and expected in actual.lower() and "--output-dir" in actual.lower() and str(self.output).lower() in actual.lower())

    def command(self, overrides: Mapping[str, object], resume: bool) -> list[str]:
        values = self.config.training_values(overrides)
        script = self.config.resolve_path(str(values.pop("script")))
        if not script.is_file():
            raise ValueError(f"training script does not exist: {script}")
        python = str(values.pop("python"))
        command = [python, "-S", str(script), "--mode", "train", "--output-dir", str(self.output)]
        flags = {"updates": "--updates", "episodes_per_update": "--episodes-per-update", "arena_games": "--arena-games", "seed": "--seed", "device": "--device", "max_game_steps": "--max-game-steps"}
        for name, flag in flags.items():
            command.extend((flag, str(values[name])))
        command.extend(("--stop-file", str(self.stop_file), "--metrics-file", str(self.metrics_file)))
        latest = self.latest_checkpoint()
        if resume and latest.is_file():
            command.extend(("--resume", str(latest)))
        elif not resume and (latest.exists() or self.metrics_file.exists()):
            raise ValueError("resume=false requires an empty output directory")
        return command

    def start(self, overrides: Mapping[str, object], resume: bool = True) -> dict[str, object]:
        with self._lock:
            if self._managed_running():
                raise RuntimeError("training is already running")
            command = self.command(overrides, resume)
            self.stop_file.unlink(missing_ok=True)
            self.output.mkdir(parents=True, exist_ok=True)
            log = self.log_file.open("a", encoding="utf-8")
            try:
                process = self.popen(command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT, shell=False)
            except OSError as exc:
                log.close()
                self.state = {"status": "ERROR", "error": str(exc)}
                self._save()
                raise RuntimeError(f"failed to start training: {exc}") from exc
            self._generation += 1
            generation = self._generation
            self.process = process
            self.state = {"status": "RUNNING", "pid": process.pid, "command": command, "generation": generation}
            self._save()
            threading.Thread(target=self._monitor, args=(process, log, generation), daemon=True).start()
            return self.status()

    def _monitor(self, process: subprocess.Popen, log, generation: int) -> None:
        code = process.wait()
        with self._lock:
            log.close()
            if generation != self._generation or self.process is not process:
                return
            requested = self.state.get("status") == "PAUSING"
            self.process = None
            self.state.update(status="PAUSED" if requested and code == 0 else "COMPLETED" if code == 0 else "ERROR", exit_code=code)
            if code != 0:
                self.state["error"] = _tail(self.log_file, 20)
            self._save()

    def pause(self, reason: str = "operator") -> dict[str, object]:
        with self._lock:
            if self.state.get("status") == "PAUSING" and self._managed_running():
                return self.status()
            if self.state.get("status") != "RUNNING" or not self._managed_running():
                raise RuntimeError("training is not running")
            self.stop_file.touch()
            self.state.update(status="PAUSING", pause_reason=reason)
            self._save()
            return self.status()

    def sample_health(self) -> dict[str, object]:
        sample = self.telemetry_collector(self.output)
        record_telemetry(self.telemetry_file, sample, self.config.health.retention_hours)
        reason = agent_pause_reason(sample, self.config.health)
        if reason:
            with self._lock:
                if self.state.get("status") == "RUNNING" and self._managed_running():
                    self.pause(reason)
        return asdict(sample)

    def health_loop(self) -> None:
        while not self._stop_health.wait(self.config.health.poll_seconds):
            try:
                self.sample_health()
            except Exception as exc:
                with self._lock:
                    self.state["telemetry_error"] = str(exc)
                    self._save()

    def start_health_loop(self) -> None:
        self._stop_health.clear()
        threading.Thread(target=self.health_loop, daemon=True).start()

    def stop_health_loop(self) -> None:
        self._stop_health.set()

    def status(self) -> dict[str, object]:
        result = dict(self.state)
        result["metrics"] = read_jsonl(self.metrics_file, limit=200)
        result["telemetry"] = read_jsonl(self.telemetry_file, limit=8640)
        return result

    def checkpoints(self) -> list[dict[str, object]]:
        root = (self.output / "checkpoints").resolve()
        if not root.exists():
            return []
        rows = []
        for path in sorted(root.rglob("*.pt")):
            try:
                resolved = path.resolve()
                if root not in resolved.parents or path.name == "diagnostic.pt" or not path.is_file():
                    continue
                stat = path.stat()
                rows.append({"name": path.relative_to(root).as_posix(), "size": stat.st_size, "sha256": sha256_file(path), "mtime": stat.st_mtime})
            except OSError:
                continue
        return rows


def create_app(config: ConsoleConfig, manager: AgentManager | None = None) -> FastAPI:
    managed = manager or AgentManager(config)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        managed.start_health_loop()
        yield
        managed.stop_health_loop()
    app = FastAPI(title="S5 Agent A", lifespan=lifespan)
    app.state.manager = managed

    @app.post("/start")
    def start(request: StartRequest):
        try:
            return managed.start(request.overrides, request.resume)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/pause")
    def pause():
        try:
            return managed.pause()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/status")
    def status():
        return managed.status()

    @app.get("/checkpoints")
    def checkpoints():
        return managed.checkpoints()

    @app.get("/checkpoints/{name:path}")
    def checkpoint(name: str):
        root = (managed.output / "checkpoints").resolve()
        try:
            path = safe_path(root, name)
        except ValueError as exc:
            raise HTTPException(404) from exc
        if root not in path.parents or not path.is_file() or path.name == "diagnostic.pt":
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/log_tail", response_class=PlainTextResponse)
    def log_tail(lines: int = 200):
        return _tail(managed.log_file, lines)

    return app
