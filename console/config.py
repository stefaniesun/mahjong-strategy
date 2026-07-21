from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, TypeVar

import yaml


ALLOWED_TRAINING_OVERRIDES = frozenset({"updates", "episodes_per_update", "arena_games", "seed", "device", "max_game_steps"})
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    python: str = "python"
    script: str = "tools/cloud_train_s5.py"
    output_dir: str = "training_artifacts/S5/local_console"
    updates: int = 100
    episodes_per_update: int = 32
    arena_games: int = 100
    seed: int = 20260721
    device: str = "auto"
    max_game_steps: int = 4000


@dataclass(frozen=True, slots=True)
class HealthConfig:
    poll_seconds: float = 10.0
    agent_cpu_pause_c: float = 85.0
    agent_disk_pause_gib: float = 10.0
    evaluator_cpu_pause_c: float = 90.0
    evaluator_cpu_resume_c: float = 80.0
    retention_hours: int = 24

    def __post_init__(self) -> None:
        if self.poll_seconds <= 0 or self.agent_disk_pause_gib < 0 or self.retention_hours <= 0:
            raise ValueError("health intervals, disk threshold and retention must be positive")
        if self.evaluator_cpu_resume_c >= self.evaluator_cpu_pause_c:
            raise ValueError("evaluator resume temperature must be below pause temperature")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    quick_games: int = 100
    formal_games: int = 500
    formal_seed: int = 90000
    poll_seconds: float = 30.0
    keep_checkpoints: int = 20


@dataclass(frozen=True, slots=True)
class AgentConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    state_dir: str = ".console/agent-a"


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    agent_url: str = "http://127.0.0.1:8765"
    state_dir: str = ".console/server-b"


@dataclass(frozen=True, slots=True)
class ConsoleConfig:
    project_root: Path
    training: TrainingConfig = field(default_factory=TrainingConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def training_values(self, overrides: Mapping[str, object] | None = None) -> dict[str, object]:
        values = {item.name: getattr(self.training, item.name) for item in fields(self.training)}
        for key, value in (overrides or {}).items():
            if key not in ALLOWED_TRAINING_OVERRIDES:
                raise ValueError(f"training override is not allowed: {key}")
            values[key] = value
        _validate_training(values)
        return values

    def resolve_path(self, value: str | Path) -> Path:
        raw = Path(value)
        candidate = (self.project_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise ValueError(f"path escapes project_root: {value}")
        return candidate

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.training.output_dir)

    @property
    def agent_state_dir(self) -> Path:
        return self.resolve_path(self.agent.state_dir)

    @property
    def server_state_dir(self) -> Path:
        return self.resolve_path(self.server.state_dir)


def _strict_dataclass(cls: type[T], value: object) -> T:
    if value is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    names = {item.name for item in fields(cls)}
    unknown = set(value) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**value)  # type: ignore[arg-type]


def _positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(name: str, value: object, *, minimum: float = 0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")


def _validate_training(values: Mapping[str, object]) -> None:
    for key in ("updates", "episodes_per_update", "arena_games", "max_game_steps"):
        _positive_int(key, values[key])
    if not isinstance(values["seed"], int) or isinstance(values["seed"], bool):
        raise ValueError("seed must be an integer")
    for key in ("python", "script", "output_dir"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if values["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")


def load_config(path: Path) -> ConsoleConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("console config must be a mapping")
    allowed = {"project_root", "training", "health", "evaluation", "agent", "server"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown console config keys: {sorted(unknown)}")
    raw_root = Path(raw.get("project_root", path.parent.parent)).expanduser()
    root = (path.parent / raw_root).resolve() if not raw_root.is_absolute() else raw_root.resolve()
    result = ConsoleConfig(
        project_root=root,
        training=_strict_dataclass(TrainingConfig, raw.get("training")),
        health=_strict_dataclass(HealthConfig, raw.get("health")),
        evaluation=_strict_dataclass(EvaluationConfig, raw.get("evaluation")),
        agent=_strict_dataclass(AgentConfig, raw.get("agent")),
        server=_strict_dataclass(ServerConfig, raw.get("server")),
    )
    _validate_training(result.training_values())
    for name, value in (
        ("poll_seconds", result.health.poll_seconds),
        ("agent_cpu_pause_c", result.health.agent_cpu_pause_c),
        ("agent_disk_pause_gib", result.health.agent_disk_pause_gib),
        ("evaluator_cpu_pause_c", result.health.evaluator_cpu_pause_c),
        ("evaluator_cpu_resume_c", result.health.evaluator_cpu_resume_c),
    ):
        _finite_number(name, value)
    _positive_int("retention_hours", result.health.retention_hours)
    for name, value in (
        ("quick_games", result.evaluation.quick_games),
        ("formal_games", result.evaluation.formal_games),
        ("keep_checkpoints", result.evaluation.keep_checkpoints),
    ):
        _positive_int(name, value)
    for name, endpoint in (("agent", result.agent), ("server", result.server)):
        if not isinstance(endpoint.host, str) or not endpoint.host.strip():
            raise ValueError(f"{name}.host must be a nonempty string")
        if not isinstance(endpoint.port, int) or isinstance(endpoint.port, bool) or not 1 <= endpoint.port <= 65535:
            raise ValueError(f"{name}.port must be between 1 and 65535")
    result.resolve_path(result.training.script)
    result.output_dir
    result.agent_state_dir
    result.server_state_dir
    return result
