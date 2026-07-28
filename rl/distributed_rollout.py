"""HTTP transport and worker runtime for distributed S5 rollout sampling."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from rl.curriculum import ObservationCurriculum
from rl.league import OpponentLeague, OpponentKind, SnapshotMetadata
from rl.production_adapter import ProductionGameTask, RolloutRuntimeState, run_production_game
from rl.types import TrajectoryStep

PROTOCOL_VERSION = 1


class RemoteRolloutError(RuntimeError):
    """A remote worker rejected or failed a rollout batch."""


def trajectory_step_to_dict(step: TrajectoryStep) -> dict[str, object]:
    return {
        "feature_values": list(step.feature_values),
        "legal_mask": list(step.legal_mask),
        "action": step.action,
        "old_log_prob": step.old_log_prob,
        "value": step.value,
        "reward": step.reward,
        "done": step.done,
        "policy_version": step.policy_version,
    }


def trajectory_step_from_dict(payload: Mapping[str, object]) -> TrajectoryStep:
    return TrajectoryStep(
        feature_values=payload["feature_values"],
        legal_mask=payload["legal_mask"],
        action=int(payload["action"]),
        old_log_prob=float(payload["old_log_prob"]),
        value=float(payload["value"]),
        reward=float(payload["reward"]),
        done=bool(payload["done"]),
        policy_version=str(payload["policy_version"]),
    )


def serialize_policy(model) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save({"model_config": asdict(model.config), "model_state": model.state_dict()}, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def file_asset(path: Path, checksum: str | None = None) -> dict[str, str]:
    content = path.read_bytes()
    digest = checksum or hashlib.sha256(content).hexdigest()
    return {"checksum": digest, "content": base64.b64encode(content).decode("ascii")}


def referenced_snapshot_assets(league: OpponentLeague) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in league.entries:
        if entry.kind not in (OpponentKind.CURRENT, OpponentKind.SNAPSHOT) or entry.snapshot is None:
            continue
        path = Path(entry.snapshot.checkpoint_path)
        key = entry.snapshot.checksum or str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        assets.append(file_asset(path, entry.snapshot.checksum))
    return assets


class RemoteRolloutClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = 900.0,
        max_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds

    def rollout(
        self,
        *,
        model,
        tasks: Sequence[ProductionGameTask],
        league: OpponentLeague,
        curriculum: ObservationCurriculum,
        policy_generation: str,
        policy_checksum: str,
        max_game_steps: int,
    ) -> tuple[TrajectoryStep, ...]:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_generation": policy_generation,
            "policy_checksum": policy_checksum,
            "policy": serialize_policy(model),
            "tasks": [asdict(task) for task in tasks],
            "league": league.to_dict(),
            "curriculum": curriculum.to_dict(),
            "snapshot_assets": referenced_snapshot_assets(league),
            "max_game_steps": max_game_steps,
        }
        response = self._post("/v1/rollout", payload)
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise RemoteRolloutError("remote rollout protocol version mismatch")
        if response.get("policy_checksum") != policy_checksum:
            raise RemoteRolloutError("remote rollout policy checksum mismatch")
        raw_steps = response.get("steps")
        if not isinstance(raw_steps, list):
            raise RemoteRolloutError("remote rollout response has no steps")
        steps = tuple(trajectory_step_from_dict(item) for item in raw_steps)
        if any(step.policy_version != policy_generation for step in steps):
            raise RemoteRolloutError("remote rollout returned stale trajectories")
        return steps

    def health(self) -> dict[str, object]:
        request = urllib.request.Request(f"{self.base_url}/health", headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 10.0)) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RemoteRolloutError(f"remote rollout health check failed: {exc}") from exc

    def _post(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        body = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), compresslevel=1)
        headers = self._headers()
        headers["Content-Encoding"] = "gzip"
        headers["Accept-Encoding"] = "gzip"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )
        result: object | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    content = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        content = gzip.decompress(content)
                    result = json.loads(content.decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RemoteRolloutError(f"remote rollout HTTP {exc.code}: {detail}") from exc
            except json.JSONDecodeError as exc:
                raise RemoteRolloutError(f"remote rollout returned invalid JSON: {exc}") from exc
            except (OSError, urllib.error.URLError) as exc:
                if attempt >= self.max_attempts:
                    raise RemoteRolloutError(
                        "remote rollout request failed after "
                        f"{self.max_attempts} attempts: {exc}"
                    ) from exc
                time.sleep(self.retry_delay_seconds * attempt)
        if not isinstance(result, dict):
            raise RemoteRolloutError("remote rollout response must be an object")
        return result

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


class RolloutWorker:
    """Loads immutable policy/runtime state and executes a remote game batch."""

    def __init__(self, project_root: Path, *, device: str = "cpu", cache_dir: Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.device = device
        self.cache_dir = Path(cache_dir or self.project_root / ".console" / "rollout_worker_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def execute(self, payload: Mapping[str, object]) -> dict[str, object]:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("unsupported rollout protocol version")
        with self._lock:
            return self._execute_locked(payload)

    def _execute_locked(self, payload: Mapping[str, object]) -> dict[str, object]:
        import torch

        from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
        from rl.opponent_resolver import OpponentResolver
        from rl.rollout import FrozenBeliefProvider, LearnerDecision

        policy_generation = str(payload["policy_generation"])
        policy_checksum = str(payload["policy_checksum"])
        model_payload = torch.load(io.BytesIO(base64.b64decode(str(payload["policy"]))), map_location="cpu", weights_only=True)
        model = PolicyValueNet(PolicyValueNetConfig(**model_payload["model_config"]))
        model.load_state_dict(model_payload["model_state"], strict=True)
        model.to(self.device).eval()

        paths = self._store_assets(payload.get("snapshot_assets", []))
        league_data = json.loads(json.dumps(payload["league"]))
        for entry in [league_data["current"], *league_data["history"]]:
            snapshot = entry.get("snapshot")
            if snapshot is not None:
                checksum = snapshot.get("checksum")
                if checksum not in paths:
                    raise ValueError(f"missing snapshot asset: {checksum}")
                snapshot["checkpoint_path"] = str(paths[checksum])
        league = OpponentLeague.from_dict(league_data)
        curriculum = ObservationCurriculum.from_dict(payload["curriculum"])
        belief_provider = FrozenBeliefProvider.from_checkpoint(
            str(self.project_root / "training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/belief_s4.pt")
        )
        resolver = OpponentResolver(model_factory=lambda: PolicyValueNet(model.config), belief_provider=belief_provider)

        def decide(view):
            with torch.no_grad():
                output = model(
                    torch.tensor([view.feature_values], dtype=torch.float32, device=self.device),
                    torch.tensor([view.legal_mask], dtype=torch.bool, device=self.device),
                )
                probabilities = torch.softmax(output.action_logits, dim=-1).cpu()[0]
                generator = torch.Generator(device="cpu").manual_seed(self._sampling_seed(view.decision_seed))
                action = int(torch.multinomial(probabilities, 1, generator=generator).item())
                return LearnerDecision(action, float(torch.log(probabilities[action]).item()), float(output.values.item()))

        runtime = RolloutRuntimeState(
            league=league,
            curriculum=curriculum,
            policy_generation=policy_generation,
            policy_checksum=policy_checksum,
            learner_decider=decide,
            opponent_resolver=resolver,
            belief_provider=belief_provider,
            max_game_steps=int(payload["max_game_steps"]),
        )
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("rollout tasks must be a non-empty list")
        steps: list[TrajectoryStep] = []
        for raw_task in raw_tasks:
            steps.extend(run_production_game(ProductionGameTask(**raw_task), runtime).steps)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "policy_checksum": policy_checksum,
            "completed_games": len(raw_tasks),
            "steps": [trajectory_step_to_dict(step) for step in steps],
        }

    def _store_assets(self, raw_assets: object) -> dict[str, Path]:
        if not isinstance(raw_assets, list):
            raise TypeError("snapshot_assets must be a list")
        result: dict[str, Path] = {}
        for asset in raw_assets:
            checksum = str(asset["checksum"])
            if not checksum or any(character not in "0123456789abcdef" for character in checksum.lower()):
                raise ValueError("invalid snapshot checksum")
            content = base64.b64decode(str(asset["content"]), validate=True)
            digest = hashlib.sha256(content).hexdigest()
            path = self.cache_dir / f"{checksum}.pt"
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
            result[checksum] = path
            result[digest] = path
        return result

    @staticmethod
    def _sampling_seed(seed: int) -> int:
        return int.from_bytes(hashlib.sha256(str(seed).encode("ascii")).digest()[:8], "big") % (2**63 - 1)
