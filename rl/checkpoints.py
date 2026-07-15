"""Atomic, reproducible S5 training checkpoints."""

from __future__ import annotations

import os
import random
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from rl.curriculum import ObservationCurriculum
from rl.league import OpponentLeague


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Restored training state needed to continue at the next rollout."""

    global_step: int
    next_rollout_seed: int
    league: OpponentLeague
    curriculum: ObservationCurriculum
    config: dict[str, object]
    metrics: dict[str, object]
    frozen_s4_provenance: dict[str, object]


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, object]) -> None:
    if set(state).difference({"python", "torch", "torch_cuda"}):
        raise ValueError("checkpoint RNG state has unexpected fields")
    if "python" not in state or "torch" not in state:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("checkpoint torch RNG state must be a tensor")
    torch.set_rng_state(torch_state)
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_state = state["torch_cuda"]
        if not isinstance(cuda_state, (list, tuple)) or not all(isinstance(item, torch.Tensor) for item in cuda_state):
            raise TypeError("checkpoint CUDA RNG state must be tensors")
        torch.cuda.set_rng_state_all(list(cuda_state))


def _require_json_mapping(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    """Return a portable mapping or reject it before a checkpoint is written."""
    candidate = dict(value)
    try:
        json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable") from exc
    return candidate


def _fsync_parent(directory: Path) -> None:
    """Best-effort directory durability (not available on every Windows FS)."""
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported S5 checkpoint format")
    required = {
        "model_state_dict", "optimizer_state_dict", "global_step", "next_rollout_seed", "rng_state",
        "league", "curriculum", "config", "metrics", "frozen_s4_provenance",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint missing field: {sorted(missing)[0]}")
    if not isinstance(payload["model_state_dict"], Mapping) or not isinstance(payload["optimizer_state_dict"], Mapping):
        raise TypeError("checkpoint model and optimizer state must be mappings")
    global_step, next_seed = payload["global_step"], payload["next_rollout_seed"]
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0 or not isinstance(next_seed, int) or isinstance(next_seed, bool) or next_seed < 0:
        raise ValueError("checkpoint step values must be non-negative integers")
    if not isinstance(payload["rng_state"], Mapping):
        raise TypeError("checkpoint RNG state must be a mapping")
    for name in ("config", "metrics", "frozen_s4_provenance"):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"checkpoint {name} must be a mapping")
    if not payload["frozen_s4_provenance"]:
        raise ValueError("checkpoint frozen_s4_provenance must be non-empty")
    if not isinstance(payload["league"], Mapping) or not isinstance(payload["curriculum"], Mapping):
        raise TypeError("checkpoint league/curriculum must be mappings")
    return payload


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    next_rollout_seed: int,
    league: OpponentLeague,
    curriculum: ObservationCurriculum,
    config: Mapping[str, object],
    metrics: Mapping[str, object],
    frozen_s4_provenance: Mapping[str, object],
) -> Path:
    """Atomically save complete, portable S5 state to ``path``.

    The file is written to a sibling temporary name and then replaced, so an
    interrupted process can leave at most a stale temporary file, never a
    half-written checkpoint at the canonical path.
    """
    if global_step < 0 or next_rollout_seed < 0:
        raise ValueError("global_step and next_rollout_seed must be non-negative")
    if not isinstance(frozen_s4_provenance, Mapping) or not frozen_s4_provenance:
        raise ValueError("frozen_s4_provenance must be a non-empty mapping")
    if not isinstance(league, OpponentLeague) or not isinstance(curriculum, ObservationCurriculum):
        raise TypeError("league and curriculum must be S5 state machines")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "next_rollout_seed": next_rollout_seed,
        "rng_state": _capture_rng_state(),
        "league": league.to_dict(),
        "curriculum": curriculum.to_dict(),
        "config": dict(config),
        "metrics": dict(metrics),
        "frozen_s4_provenance": _require_json_mapping(frozen_s4_provenance, name="frozen_s4_provenance"),
    }
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    """Restore model/optimizer and return the next-rollout orchestration state."""
    try:
        payload = _validate_payload(torch.load(Path(path), map_location=map_location, weights_only=True))
    except Exception as exc:
        if isinstance(exc, (ValueError, TypeError)):
            raise
        raise ValueError("checkpoint is not a safe S5 checkpoint payload") from exc
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    global_step = payload["global_step"]
    next_seed = payload["next_rollout_seed"]
    return TrainingCheckpoint(
        global_step=global_step,
        next_rollout_seed=next_seed,
        league=OpponentLeague.from_dict(payload["league"]),
        curriculum=ObservationCurriculum.from_dict(payload["curriculum"]),
        config=dict(payload["config"]),
        metrics=dict(payload["metrics"]),
        frozen_s4_provenance=dict(payload["frozen_s4_provenance"]),
    )
