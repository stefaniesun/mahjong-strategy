from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from console.config import ConsoleConfig
from console.evaluator import evaluate_checkpoint
from console.storage import append_jsonl, atomic_write_json, read_json, read_jsonl, sha256_file
from console.telemetry import collect_telemetry, evaluator_paused, record_telemetry


class AgentClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def json(self, path: str) -> object:
        with urllib.request.urlopen(self.base_url + path, timeout=10) as response:
            return json.load(response)

    def post(self, path: str, payload: object | None = None) -> object:
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def download(self, name: str, destination: Path) -> None:
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in Path(name).parts)
        with urllib.request.urlopen(self.base_url + "/checkpoints/" + encoded, timeout=60) as response, destination.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)


class EvaluationScheduler:
    def __init__(
        self, config: ConsoleConfig, *, client: AgentClient | None = None,
        evaluator: Callable = evaluate_checkpoint, telemetry_collector: Callable = collect_telemetry,
    ) -> None:
        self.config, self.client, self.evaluator = config, client or AgentClient(config.server.agent_url), evaluator
        self.telemetry_collector = telemetry_collector
        self.state_dir = config.server_state_dir
        self.archive, self.metrics = self.state_dir / "checkpoints", self.state_dir / "metrics.jsonl"
        self.catalog = self.state_dir / "checkpoint_catalog.json"
        self.evals, self.events, self.telemetry = self.state_dir / "evals.jsonl", self.state_dir / "events.jsonl", self.state_dir / "telemetry.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.thermal_paused = False
        self._active_checkpoints: set[Path] = set()

    def event(self, level: str, message: str, **details: object) -> None:
        append_jsonl(self.events, {
            "timestamp": datetime.now(timezone.utc).isoformat(), "level": level,
            "message": message, **details,
        })


    def completed_ids(self) -> set[str]:
        return {str(row["task_id"]) for row in read_jsonl(self.evals) if "task_id" in row}

    @staticmethod
    def task_id(checksum: str, kind: str, games: int, seed: int, day: date | None = None) -> str:
        suffix = f":{day.isoformat()}" if kind == "formal" and day else ""
        return f"{checksum}:{kind}:{games}:{seed}{suffix}"

    def sync_metrics(self) -> dict[str, object]:
        status = self.client.json("/status")
        if not isinstance(status, dict):
            raise ValueError("agent status must be an object")
        known = {(row.get("global_step"), row.get("timestamp")) for row in read_jsonl(self.metrics)}
        for row in status.get("metrics", []):
            if isinstance(row, dict) and (row.get("global_step"), row.get("timestamp")) not in known:
                append_jsonl(self.metrics, row)
                known.add((row.get("global_step"), row.get("timestamp")))
        return status

    def sync_checkpoints(self) -> list[Path]:
        rows = self.client.json("/checkpoints")
        if not isinstance(rows, list):
            raise ValueError("checkpoint list must be an array")
        catalog = read_json(self.catalog, {})
        catalog = catalog if isinstance(catalog, dict) else {}
        synced: list[Path] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            checksum, size = str(row.get("sha256", "")), int(row.get("size", -1))
            if len(checksum) != 64 or size < 0:
                continue
            target = self.archive / f"{checksum}.pt"
            if not (target.exists() and target.stat().st_size == size and sha256_file(target) == checksum):
                if shutil.disk_usage(self.archive).free / 1024 ** 3 < self.config.evaluation.min_disk_free_gib:
                    raise RuntimeError("evaluation disk space is below safety threshold")
                fd, temp_name = tempfile.mkstemp(prefix=".download-", suffix=".pt", dir=self.archive)
                os.close(fd)
                temp = Path(temp_name)
                try:
                    self.client.download(str(row["name"]), temp)
                    if temp.stat().st_size != size or sha256_file(temp) != checksum:
                        raise ValueError("downloaded checkpoint verification failed")
                    os.replace(temp, target)
                    self.event("info", "checkpoint_synced", checkpoint=checksum, remote_name=row.get("name"))
                finally:
                    temp.unlink(missing_ok=True)
            catalog[checksum] = {
                "remote_name": row.get("name"), "mtime": float(row.get("mtime", 0)),
                "size": size, "sha256": checksum,
            }
            synced.append(target)
        atomic_write_json(self.catalog, catalog)
        self._retain(catalog)
        return synced

    def _retain(self, catalog: dict[str, object]) -> None:
        files = sorted(
            self.archive.glob("*.pt"),
            key=lambda path: float(catalog.get(path.stem, {}).get("mtime", 0)) if isinstance(catalog.get(path.stem), dict) else 0,
            reverse=True,
        )
        milestone_days: set[str] = set()
        retained: set[Path] = set(files[:self.config.evaluation.keep_checkpoints]) | self._active_checkpoints
        for path in files:
            item = catalog.get(path.stem, {})
            timestamp = float(item.get("mtime", 0)) if isinstance(item, dict) else 0
            day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
            if day not in milestone_days:
                retained.add(path)
                milestone_days.add(day)
        for path in files:
            if path in retained:
                continue
            path.unlink(missing_ok=True)
            catalog.pop(path.stem, None)
        atomic_write_json(self.catalog, catalog)

    def pending(self, checkpoints: list[Path], today: date | None = None) -> list[tuple[Path, str, int, int, str]]:
        if not checkpoints:
            return []
        day, completed = today or datetime.now(timezone.utc).date(), self.completed_ids()
        catalog = read_json(self.catalog, {})
        catalog = catalog if isinstance(catalog, dict) else {}
        latest = max(
            checkpoints,
            key=lambda path: float(catalog.get(path.stem, {}).get("mtime", 0)) if isinstance(catalog.get(path.stem), dict) else path.stat().st_mtime,
        )
        tasks: list[tuple[Path, str, int, int, str]] = []
        quick_id = self.task_id(latest.stem, "quick", self.config.evaluation.quick_games, 0)
        if quick_id not in completed:
            tasks.append((latest, "quick", self.config.evaluation.quick_games, 0, quick_id))
        formal_id = self.task_id(latest.stem, "formal", self.config.evaluation.formal_games, self.config.evaluation.formal_seed, day)
        if formal_id not in completed:
            tasks.append((latest, "formal", self.config.evaluation.formal_games, self.config.evaluation.formal_seed, formal_id))
        return tasks

    def evaluate_next(self) -> dict[str, object] | None:
        sample = self.telemetry_collector(self.state_dir)
        record_telemetry(self.telemetry, sample, self.config.health.retention_hours)
        self.thermal_paused = evaluator_paused(sample, self.config.health, self.thermal_paused)
        if self.thermal_paused:
            return None
        tasks = self.pending(list(self.archive.glob("*.pt")))
        if not tasks:
            return None
        checkpoint, kind, games, seed, task_id = tasks[0]
        self._active_checkpoints.add(checkpoint)
        self.event("info", "evaluation_started", checkpoint=checkpoint.stem, kind=kind)
        last_sample_at = 0.0
        def progress(_completed: int, _total: int) -> None:
            nonlocal last_sample_at
            now = datetime.now(timezone.utc).timestamp()
            if now - last_sample_at < self.config.health.poll_seconds:
                return
            last_sample_at = now
            health = self.telemetry_collector(self.state_dir)
            record_telemetry(self.telemetry, health, self.config.health.retention_hours)
            was_paused = self.thermal_paused
            self.thermal_paused = evaluator_paused(health, self.config.health, self.thermal_paused)
            if self.thermal_paused and not was_paused:
                self.event("warning", "evaluation_overheat", cpu_temp_c=health.cpu_temp_c)
            if self.thermal_paused:
                raise RuntimeError("evaluation interrupted by CPU overheat")
        try:
            result = self.evaluator(
                checkpoint, games=games, seed=seed, kind=kind,
                s4_policy_path=self.config.resolve_path(self.config.evaluation.s4_policy_path),
                s4_belief_path=self.config.resolve_path(self.config.evaluation.s4_belief_path),
                progress_callback=progress,
            )
        finally:
            self._active_checkpoints.discard(checkpoint)
        result.update(task_id=task_id, timestamp=datetime.now(timezone.utc).isoformat())
        append_jsonl(self.evals, result)
        self.event("info", "evaluation_completed", checkpoint=checkpoint.stem, kind=kind, task_id=task_id)
        return result
