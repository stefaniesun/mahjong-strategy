"""S5 local training orchestration with injectable CPU-smoke dependencies."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from rl.checkpoints import load_checkpoint, save_checkpoint
from rl.curriculum import ObservationCurriculum
from rl.league import DualArenaMetrics, OpponentEntry, OpponentLeague, SnapshotMetadata
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
from rl.ppo_trainer import PPOBatch, PPOConfig, PPOHealth, ppo_update
from rl.types import TrajectoryStep


@dataclass(frozen=True, slots=True)
class TrainingRuntimeState:
    """The current League/Curriculum state passed explicitly to every adapter."""

    league: OpponentLeague
    curriculum: ObservationCurriculum
    policy_generation: str
    policy_checksum: str


RolloutFactory = Callable[[PolicyValueNet, "S5TrainingConfig", int, int, TrainingRuntimeState], Sequence[TrajectoryStep]]
ArenaEvaluator = Callable[[PolicyValueNet, OpponentEntry, TrainingRuntimeState], DualArenaMetrics]


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """Fail-fast limits that make a bad S5 update diagnosable and resumable."""

    min_entropy: float = 0.0
    max_kl: float = 10.0
    max_value_loss: float = 1_000_000.0
    min_s3_perfect_win_rate: float = 0.0
    min_s3_degraded_win_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.min_entropy < 0.0 or self.max_kl < 0.0 or self.max_value_loss < 0.0:
            raise ValueError("health thresholds must be non-negative")
        if not 0.0 <= self.min_s3_perfect_win_rate <= 1.0 or not 0.0 <= self.min_s3_degraded_win_rate <= 1.0:
            raise ValueError("S3 health thresholds must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class S5TrainingConfig:
    """Configuration for a resumable local/cloud S5 run.

    Frozen S4 artifacts are intentionally required even in injected smoke jobs:
    every training report and checkpoint must prove the exact S4 dependency.
    """

    output_dir: Path
    frozen_s4_belief_path: Path
    frozen_s4_policy_path: Path
    frozen_s4_provenance: Mapping[str, object]
    updates: int = 1
    rollout_seed_start: int = 0
    snapshot_interval: int = 1
    device: str = "cpu"
    # None means "derive this from the frozen S4 PolicyNet checkpoint".  The
    # old 80/64 defaults were placeholders and silently disagreed with the
    # archived S4-v1 policy (263/637).
    feature_size: int | None = None
    action_size: int | None = None
    hidden_size: int | None = None
    residual_blocks: int | None = None
    dropout: float | None = None
    learning_rate: float = 3e-4
    ppo: PPOConfig = field(default_factory=PPOConfig)
    kl_start_coef: float | None = None
    kl_end_coef: float | None = None
    kl_schedule_total_updates: int | None = None
    health_thresholds: HealthThresholds = field(default_factory=HealthThresholds)
    resume_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "frozen_s4_belief_path", Path(self.frozen_s4_belief_path))
        object.__setattr__(self, "frozen_s4_policy_path", Path(self.frozen_s4_policy_path))
        if self.updates <= 0 or self.snapshot_interval <= 0:
            raise ValueError("updates and snapshot_interval must be positive")
        if self.rollout_seed_start < 0:
            raise ValueError("rollout_seed_start must be non-negative")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        for name, value in (
            ("feature_size", self.feature_size),
            ("action_size", self.action_size),
            ("hidden_size", self.hidden_size),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.residual_blocks is not None and (
            not isinstance(self.residual_blocks, int)
            or isinstance(self.residual_blocks, bool)
            or self.residual_blocks < 0
        ):
            raise ValueError("residual_blocks must be a non-negative integer or None")
        if self.dropout is not None and (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be a finite value in [0, 1) or None")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.ppo, PPOConfig):
            raise TypeError("ppo must be PPOConfig")
        if self.kl_schedule_total_updates is not None and (
            not isinstance(self.kl_schedule_total_updates, int)
            or isinstance(self.kl_schedule_total_updates, bool)
            or self.kl_schedule_total_updates <= 0
        ):
            raise ValueError("kl_schedule_total_updates must be a positive integer or None")
        if not isinstance(self.health_thresholds, HealthThresholds):
            raise TypeError("health_thresholds must be HealthThresholds")
        for name, value in (("kl_start_coef", self.kl_start_coef), ("kl_end_coef", self.kl_end_coef)):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be a finite non-negative number or None")
        if not isinstance(self.frozen_s4_provenance, Mapping) or not self.frozen_s4_provenance:
            raise ValueError("frozen_s4_provenance is required")
        try:
            json.dumps(dict(self.frozen_s4_provenance), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("frozen_s4_provenance must be JSON serializable") from exc
        for artifact in (self.frozen_s4_belief_path, self.frozen_s4_policy_path):
            if not artifact.is_file():
                raise FileNotFoundError(f"required frozen S4 artifact is missing: {artifact}")
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))

    def serializable_dict(self, *, resolved_kl_schedule_total_updates: int | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "output_dir": str(self.output_dir),
            "frozen_s4_belief_path": str(self.frozen_s4_belief_path),
            "frozen_s4_policy_path": str(self.frozen_s4_policy_path),
            "frozen_s4_provenance": dict(self.frozen_s4_provenance),
            "updates": self.updates,
            "rollout_seed_start": self.rollout_seed_start,
            "snapshot_interval": self.snapshot_interval,
            "device": self.device,
            "feature_size": self.feature_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "residual_blocks": self.residual_blocks,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "ppo": asdict(self.ppo),
            "kl_start_coef": self.kl_start_coef,
            "kl_end_coef": self.kl_end_coef,
            "kl_schedule_total_updates": self.kl_schedule_total_updates,
            "health_thresholds": asdict(self.health_thresholds),
            "resume_checkpoint": str(self.resume_checkpoint) if self.resume_checkpoint else None,
        }
        if resolved_kl_schedule_total_updates is not None:
            result["resolved_kl_schedule_total_updates"] = resolved_kl_schedule_total_updates
        return result


@dataclass(frozen=True, slots=True)
class S5TrainingDependencies:
    """Narrow injectable boundary for tests and eventual engine adapters."""

    model_factory: Callable[[S5TrainingConfig], PolicyValueNet] | None = None
    optimizer_factory: Callable[[PolicyValueNet], torch.optim.Optimizer] | None = None
    rollout_factory: RolloutFactory | None = None
    arena_evaluator: ArenaEvaluator | None = None
    league_factory: Callable[[], OpponentLeague] | None = None
    curriculum_factory: Callable[[], ObservationCurriculum] | None = None
    reference_model_factory: Callable[[PolicyValueNet, S5TrainingConfig], PolicyValueNet] | None = None


@dataclass(frozen=True, slots=True)
class S5TrainingResult:
    global_step: int
    next_rollout_seed: int
    report_path: Path
    markdown_path: Path
    publication_manifest_path: Path
    checkpoint_path: Path
    report: dict[str, object]
    league: OpponentLeague
    curriculum: ObservationCurriculum


def _resolve_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def _artifact_provenance(config: S5TrainingConfig) -> dict[str, object]:
    def item(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"declared": dict(config.frozen_s4_provenance), "belief": item(config.frozen_s4_belief_path), "policy": item(config.frozen_s4_policy_path)}


def _model_checksum(model: PolicyValueNet) -> str:
    """Hash the exact learner tensors used for one rollout generation."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _S4PolicyArchitecture:
    """Architecture recorded by the frozen S4 PolicyNet artifact."""

    feature_size: int
    action_size: int
    hidden_size: int
    residual_blocks: int
    dropout: float


def _load_frozen_s4_policy_payload(path: Path) -> Mapping[str, object]:
    """Load a trusted-tensors-only S4 artifact before inspecting its metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("frozen S4 policy checkpoint must be a mapping")
    return payload


def _frozen_s4_policy_architecture(payload: Mapping[str, object]) -> _S4PolicyArchitecture:
    """Read complete, type-checked architecture metadata from an S4 artifact."""
    raw = payload.get("model_config")
    if not isinstance(raw, Mapping):
        raise ValueError("frozen S4 policy checkpoint must include a model_config mapping")

    def positive_int(key: str) -> int:
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"frozen S4 policy model_config.{key} must be a positive integer")
        return value

    residual_blocks = raw.get("residual_blocks")
    if not isinstance(residual_blocks, int) or isinstance(residual_blocks, bool) or residual_blocks < 0:
        raise ValueError("frozen S4 policy model_config.residual_blocks must be a non-negative integer")
    dropout = raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("frozen S4 policy model_config.dropout must be a finite value in [0, 1)")
    return _S4PolicyArchitecture(
        feature_size=positive_int("input_size"),
        action_size=positive_int("action_size"),
        hidden_size=positive_int("hidden_size"),
        residual_blocks=residual_blocks,
        dropout=float(dropout),
    )


def _resolved_s5_model_config(config: S5TrainingConfig, architecture: _S4PolicyArchitecture) -> PolicyValueNetConfig:
    """Infer S5 dimensions from S4, rejecting every conflicting override."""
    requested = {
        "feature_size": config.feature_size,
        "action_size": config.action_size,
        "hidden_size": config.hidden_size,
        "residual_blocks": config.residual_blocks,
        "dropout": config.dropout,
    }
    actual = {
        "feature_size": architecture.feature_size,
        "action_size": architecture.action_size,
        "hidden_size": architecture.hidden_size,
        "residual_blocks": architecture.residual_blocks,
        "dropout": architecture.dropout,
    }
    source_name = {"feature_size": "input_size", "action_size": "action_size"}
    for name, value in requested.items():
        if value is not None and value != actual[name]:
            raise ValueError(
                f"{name}={value} does not match frozen S4 policy "
                f"{source_name.get(name, name)}={actual[name]}"
            )
    return PolicyValueNetConfig(
        input_size=architecture.feature_size,
        action_size=architecture.action_size,
        hidden_size=architecture.hidden_size,
        residual_blocks=architecture.residual_blocks,
        dropout=architecture.dropout,
    )


def _default_model(config: S5TrainingConfig) -> PolicyValueNet:
    payload = _load_frozen_s4_policy_payload(config.frozen_s4_policy_path)
    architecture = _frozen_s4_policy_architecture(payload)
    model = PolicyValueNet(_resolved_s5_model_config(config, architecture))
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise ValueError("frozen S4 policy checkpoint has no model state dictionary")
    try:
        model.load_s4_policy_state_dict(state)
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise ValueError("frozen S4 policy weights are incompatible with its model_config") from exc
    return model


def _batch_from_steps(steps: Sequence[TrajectoryStep], reference_model: PolicyValueNet) -> PPOBatch:
    if not steps:
        raise ValueError("rollout factory returned no learner trajectory steps")
    features = torch.tensor([step.feature_values for step in steps], dtype=torch.float32)
    mask = torch.tensor([step.legal_mask for step in steps], dtype=torch.bool)
    actions = torch.tensor([step.action for step in steps], dtype=torch.long)
    reference_device = next(reference_model.parameters()).device
    with torch.no_grad():
        reference_logits = reference_model(
            features.to(reference_device), mask.to(reference_device)
        ).action_logits.detach().cpu()
    return PPOBatch(
        features=features,
        legal_mask=mask,
        actions=actions,
        old_log_probs=torch.tensor([step.old_log_prob for step in steps], dtype=torch.float32),
        old_values=torch.tensor([step.value for step in steps], dtype=torch.float32),
        rewards=torch.tensor([step.reward for step in steps], dtype=torch.float32),
        dones=torch.tensor([step.done for step in steps], dtype=torch.bool),
        reference_logits=reference_logits,
    )


def _evaluate(
    evaluator: ArenaEvaluator,
    model: PolicyValueNet,
    runtime: TrainingRuntimeState,
) -> dict[str, DualArenaMetrics]:
    """Evaluate the candidate against every current, historical and fixed entry."""
    results: dict[str, DualArenaMetrics] = {}
    for entry in runtime.league.entries:
        metric = evaluator(model, entry, runtime)
        if not isinstance(metric, DualArenaMetrics):
            raise TypeError("arena_evaluator must return DualArenaMetrics for every league entry")
        results[entry.key] = metric
    return results


def _health_is_valid(health: PPOHealth) -> bool:
    return all(math.isfinite(value) for value in asdict(health).values())


def _scheduled_ppo(config: S5TrainingConfig, global_step: int, schedule_total_updates: int) -> PPOConfig:
    """Schedule KL by durable global progress, including after a resume."""
    start = config.ppo.kl_coef if config.kl_start_coef is None else config.kl_start_coef
    end = start if config.kl_end_coef is None else config.kl_end_coef
    fraction = 1.0 if schedule_total_updates == 1 else min(global_step, schedule_total_updates - 1) / (schedule_total_updates - 1)
    return replace(config.ppo, kl_coef=start + (end - start) * fraction)


def _resolved_kl_schedule_horizon(config: S5TrainingConfig, restored_config: Mapping[str, object] | None = None) -> int:
    """Return one persisted KL horizon instead of deriving it per invocation.

    A fresh short smoke run still needs a two-point schedule so that splitting
    it after one update and resuming does not turn both updates into the end
    coefficient. Production callers can always provide a larger explicit
    horizon.
    """
    if restored_config is None:
        return config.kl_schedule_total_updates if config.kl_schedule_total_updates is not None else max(config.updates, 2)
    stored = restored_config.get("resolved_kl_schedule_total_updates")
    if not isinstance(stored, int) or isinstance(stored, bool) or stored <= 0:
        raise ValueError("resume checkpoint has no valid resolved KL schedule horizon")
    if config.kl_schedule_total_updates is not None and config.kl_schedule_total_updates != stored:
        raise ValueError("requested KL schedule horizon does not match the resume checkpoint")
    return stored


def _fsync_parent(directory: Path) -> None:
    """Best-effort directory durability for an atomic artifact replacement."""
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _report_manifest_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.manifest.json")


def _digest_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_report(path: Path, markdown_path: Path, report: Mapping[str, object]) -> Path:
    """Publish a matched JSON/Markdown pair by committing its manifest last.

    A reader must use :func:`load_published_s5_report`; a crash between either
    artifact and the final manifest can then only leave a pair that is rejected,
    never a silently mixed-generation report.
    """
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    arena = report["arena"]
    health = report["health"]
    markdown = (
        "# S5 Training Report\n\n"
        f"- Global step: {report['global_step']}\n"
        f"- Device: {report['device']}\n"
        f"- Per-opponent perfect/degraded metrics: `{json.dumps(arena['by_opponent'], sort_keys=True)}`\n"
        f"- S3 comparison: `{json.dumps(arena['s3_comparison'], sort_keys=True)}`\n"
        f"- PPO updates: {len(health)}\n"
    )
    _atomic_write_text(path, serialized)
    _atomic_write_text(markdown_path, markdown)
    manifest_path = _report_manifest_path(path)
    manifest = {
        "format_version": 1,
        "artifacts": {
            "json": {"name": path.name, "sha256": _digest_text(serialized)},
            "markdown": {"name": markdown_path.name, "sha256": _digest_text(markdown)},
        },
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return manifest_path


def load_published_s5_report(path: str | Path) -> dict[str, object]:
    """Load a committed S5 JSON report and reject missing/mixed report pairs."""
    report_path = Path(path)
    manifest_path = _report_manifest_path(report_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("S5 report publication manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("S5 report publication manifest is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("S5 report publication manifest is invalid")
    json_artifact, markdown_artifact = artifacts.get("json"), artifacts.get("markdown")
    if not isinstance(json_artifact, dict) or not isinstance(markdown_artifact, dict):
        raise ValueError("S5 report publication manifest is invalid")
    markdown_path = report_path.with_name(str(markdown_artifact.get("name", "")))
    expected = ((report_path, json_artifact), (markdown_path, markdown_artifact))
    for artifact_path, descriptor in expected:
        if descriptor.get("name") != artifact_path.name or not isinstance(descriptor.get("sha256"), str):
            raise ValueError("S5 report publication manifest is invalid")
        try:
            content = artifact_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("S5 report artifacts do not match the committed manifest") from exc
        if _digest_text(content) != descriptor["sha256"]:
            raise ValueError("S5 report artifacts do not match the committed manifest")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("committed S5 report JSON is invalid") from exc
    if not isinstance(report, dict):
        raise ValueError("committed S5 report JSON must be an object")
    return report


def _atomic_write_text(destination: Path, content: str) -> None:
    """Publish a complete text artifact or preserve the previous canonical one."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_diagnostic(
    path: Path, *, reason: str, stage: str, model: PolicyValueNet, optimizer: torch.optim.Optimizer,
    global_step: int, next_seed: int, league: OpponentLeague, curriculum: ObservationCurriculum,
    config: S5TrainingConfig, schedule_total_updates: int, health_rows: Sequence[Mapping[str, float]], provenance: Mapping[str, object],
) -> None:
    save_checkpoint(
        path, model=model, optimizer=optimizer, global_step=global_step, next_rollout_seed=next_seed,
        league=league, curriculum=curriculum,
        config=config.serializable_dict(resolved_kl_schedule_total_updates=schedule_total_updates),
        metrics={"health": list(health_rows), "alert": reason, "stage": stage, "reason": reason}, frozen_s4_provenance=provenance,
    )


def _health_alert(health: PPOHealth, thresholds: HealthThresholds) -> str | None:
    if not _health_is_valid(health):
        return "nonfinite_health"
    if health.entropy < thresholds.min_entropy:
        return "entropy_collapse"
    if health.kl > thresholds.max_kl:
        return "kl_explosion"
    if health.value_loss > thresholds.max_value_loss:
        return "value_loss_divergence"
    return None


def _s3_alert(metrics: Mapping[str, DualArenaMetrics], thresholds: HealthThresholds) -> str | None:
    s3 = metrics["s3"]
    if s3.perfect_win_rate < thresholds.min_s3_perfect_win_rate or s3.degraded_win_rate < thresholds.min_s3_degraded_win_rate:
        return "s3_arena_regression"
    return None


def _write_immutable_snapshot(
    directory: Path, *, model: PolicyValueNet, optimizer: torch.optim.Optimizer, global_step: int,
    next_seed: int, league: OpponentLeague, curriculum: ObservationCurriculum, config: S5TrainingConfig,
    schedule_total_updates: int, health_rows: Sequence[Mapping[str, float]], provenance: Mapping[str, object],
) -> SnapshotMetadata:
    path = directory / f"s5-step-{global_step}.pt"
    if path.exists():
        raise FileExistsError(f"immutable snapshot already exists: {path}")
    save_checkpoint(
        path, model=model, optimizer=optimizer, global_step=global_step, next_rollout_seed=next_seed,
        league=league, curriculum=curriculum,
        config=config.serializable_dict(resolved_kl_schedule_total_updates=schedule_total_updates),
        metrics={"health": list(health_rows)}, frozen_s4_provenance=provenance,
    )
    return SnapshotMetadata(
        snapshot_id=f"s5-{global_step}", policy_version=f"s5-step-{global_step}",
        checkpoint_path=str(path), training_step=global_step,
        checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def run_s5_training(config: S5TrainingConfig, *, dependencies: S5TrainingDependencies | None = None) -> S5TrainingResult:
    """Run a finite, resumable S5 update budget and emit auditable artifacts.

    Real engine rollout and arena adapters are deliberately injected: the
    orchestration layer owns PPO/checkpoint invariants without ever needing to
    reach through the learner information boundary.
    """
    if not isinstance(config, S5TrainingConfig):
        raise TypeError("config must be S5TrainingConfig")
    dependencies = dependencies or S5TrainingDependencies()
    if dependencies.rollout_factory is None or dependencies.arena_evaluator is None:
        raise ValueError("run_s5_training requires rollout_factory and arena_evaluator adapters")
    device = _resolve_device(config.device)
    model = (dependencies.model_factory or _default_model)(config).to(device)
    if not isinstance(model, PolicyValueNet):
        raise TypeError("model_factory must return PolicyValueNet")
    optimizer = (dependencies.optimizer_factory or (lambda current: torch.optim.Adam(current.parameters(), lr=config.learning_rate)))(model)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_factory must return a torch Optimizer")
    provenance = _artifact_provenance(config)
    reference = (dependencies.reference_model_factory(model, config) if dependencies.reference_model_factory else copy.deepcopy(model)).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    league = dependencies.league_factory() if dependencies.league_factory else None
    curriculum = dependencies.curriculum_factory() if dependencies.curriculum_factory else None
    if not isinstance(league, OpponentLeague) or not isinstance(curriculum, ObservationCurriculum):
        raise ValueError("run_s5_training requires league_factory and curriculum_factory adapters")

    global_step, next_seed = 0, config.rollout_seed_start
    schedule_total_updates = _resolved_kl_schedule_horizon(config)
    if config.resume_checkpoint is not None:
        restored = load_checkpoint(config.resume_checkpoint, model=model, optimizer=optimizer, restore_rng=True, map_location=device)
        if restored.frozen_s4_provenance != provenance:
            raise ValueError("resume checkpoint frozen S4 provenance does not match this run")
        global_step, next_seed, league, curriculum = restored.global_step, restored.next_rollout_seed, restored.league, restored.curriculum
        schedule_total_updates = _resolved_kl_schedule_horizon(config, restored.config)

    checkpoints = config.output_dir / "checkpoints"
    reports = config.output_dir / "reports"
    latest = checkpoints / "latest.pt"
    diagnostic = checkpoints / "diagnostic.pt"
    health_rows: list[dict[str, float]] = []
    if config.resume_checkpoint is not None:
        restored_health = restored.metrics.get("health", [])
        if not isinstance(restored_health, list) or not all(isinstance(row, Mapping) for row in restored_health):
            raise ValueError("resume checkpoint health metrics must be a list of mappings")
        health_rows = [dict(row) for row in restored_health]
    latest_arena: dict[str, DualArenaMetrics] | None = None
    for update in range(config.updates):
        stage = "rollout"
        diagnostic_reason: str | None = None
        runtime = TrainingRuntimeState(
            league=league,
            curriculum=curriculum,
            policy_generation=f"s5-step-{global_step}",
            policy_checksum=_model_checksum(model),
        )
        try:
            steps = dependencies.rollout_factory(model, config, update, next_seed, runtime)
            stage = "batch"
            batch = _batch_from_steps(steps, reference)
            stage = "ppo_update"
            effective_ppo = _scheduled_ppo(config, global_step, schedule_total_updates)
            health = ppo_update(model, batch, optimizer, effective_ppo)
            health_row = asdict(health)
            health_rows.append(health_row)
            global_step += 1
            next_seed += 1
            stage = "health"
            health_reason = _health_alert(health, config.health_thresholds)
            if health_reason is not None:
                diagnostic_reason = health_reason
                raise RuntimeError(f"PPO health alert: {health_reason}")
            stage = "evaluation"
            latest_arena = _evaluate(dependencies.arena_evaluator, model, runtime)
            s3_reason = _s3_alert(latest_arena, config.health_thresholds)
            if s3_reason is not None:
                diagnostic_reason = s3_reason
                raise RuntimeError(f"PPO health alert: {s3_reason}")
            if global_step % config.snapshot_interval == 0:
                stage = "snapshot"
                candidate = _write_immutable_snapshot(
                    checkpoints / "snapshots", model=model, optimizer=optimizer, global_step=global_step,
                    next_seed=next_seed, league=league, curriculum=curriculum, config=config,
                    schedule_total_updates=schedule_total_updates, health_rows=health_rows, provenance=provenance,
                )
                stage = "league"
                league.promote_candidate(candidate, latest_arena)
            stage = "curriculum"
            curriculum.advance(latest_arena["s3"])
            stage = "latest_checkpoint"
            save_checkpoint(
                latest, model=model, optimizer=optimizer, global_step=global_step, next_rollout_seed=next_seed,
                league=league, curriculum=curriculum,
                config=config.serializable_dict(resolved_kl_schedule_total_updates=schedule_total_updates),
                metrics={"health": health_rows}, frozen_s4_provenance=provenance,
            )
        except Exception as exc:
            reason = diagnostic_reason or f"{stage}_exception"
            try:
                _save_diagnostic(diagnostic, reason=reason, stage=stage, model=model, optimizer=optimizer, global_step=global_step, next_seed=next_seed, league=league, curriculum=curriculum, config=config, schedule_total_updates=schedule_total_updates, health_rows=health_rows, provenance=provenance)
            except Exception:
                # Never replace the original training failure with diagnostic I/O.
                pass
            raise

    assert latest_arena is not None
    report = {
        "global_step": global_step,
        "next_rollout_seed": next_seed,
        "device": device.type,
        "frozen_s4_provenance": provenance,
        "health": health_rows,
        "arena": {
            "by_opponent": {key: asdict(metric) for key, metric in latest_arena.items()},
            "s3_comparison": asdict(latest_arena["s3"]),
        },
        "league": league.to_dict(),
        "curriculum": curriculum.to_dict(),
    }
    report_path, markdown_path = reports / "s5_training_report.json", reports / "s5_training_report.md"
    try:
        publication_manifest_path = _write_report(report_path, markdown_path, report)
    except Exception:
        try:
            _save_diagnostic(diagnostic, reason="report_exception", stage="report", model=model, optimizer=optimizer, global_step=global_step, next_seed=next_seed, league=league, curriculum=curriculum, config=config, schedule_total_updates=schedule_total_updates, health_rows=health_rows, provenance=provenance)
        except Exception:
            pass
        raise
    return S5TrainingResult(global_step, next_seed, report_path, markdown_path, publication_manifest_path, latest, report, league, curriculum)
