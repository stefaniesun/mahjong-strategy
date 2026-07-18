"""Reproducible S5 cloud package builder and bounded training entry point.

``smoke`` is deliberately a tiny controlled bootstrap diagnosis.  ``train``
uses complete S1 games through S2 observations, frozen accepted S4 belief,
S3/league opponents, curriculum degradation, and real dual-track arena games.
Only the latter is an S5 training run; neither mode is an acceptance claim by
itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import sysconfig
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S4_ARCHIVE = Path("training_artifacts/S4/v5_20260718_encoder_v4")
S4_BELIEF = S4_ARCHIVE / "checkpoints/belief_s4.pt"
S4_POLICY = S4_ARCHIVE / "checkpoints/policy_s4.pt"
S4_RELEASE = "S4/v5_20260718_encoder_v4"
S4_V5_COLD_START_POLICY_VERSION = "s4-v5-encoder-v4-cold-start"
S4_REQUIRED_SHA256 = {
    "checkpoints/belief_s4.pt": "caa9775e65070e6196a2b020fc013783bbd818c54f763911286a0165915f3e01",
    "checkpoints/policy_s4.pt": "3c23f0b7841298ff0cd9fd531a6f261fdb0a07436b08f9896bf32feb090fd8e2",
}
PACKAGE_FORMAT_VERSION = 2
PACKAGE_MANIFEST_NAME = "s5_cloud_package_manifest.json"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class S5CloudRunConfig:
    output_dir: Path
    mode: str = "smoke"
    device: str = "auto"
    updates: int = 1
    seed: int = 20260715
    episodes_per_update: int = 4
    arena_games: int | None = None
    max_game_steps: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.arena_games is None:
            object.__setattr__(self, "arena_games", 4 if self.mode == "smoke" else 1000)
        if self.mode not in {"smoke", "train"}:
            raise ValueError("mode must be smoke or train")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if not isinstance(self.updates, int) or isinstance(self.updates, bool) or self.updates <= 0:
            raise ValueError("updates must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, value in (("episodes_per_update", self.episodes_per_update), ("arena_games", self.arena_games), ("max_game_steps", self.max_game_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.mode == "smoke" and self.updates != 1:
            raise ValueError("smoke mode requires updates=1")
        if self.mode == "smoke" and (self.episodes_per_update != 4 or self.arena_games != 4):
            raise ValueError("smoke mode does not accept formal training episode or arena budgets")


@dataclass(frozen=True, slots=True)
class S5CloudTrainingResult:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    checkpoint_path: Path
    device: str
    global_step: int


@dataclass(frozen=True, slots=True)
class S5LocalPrepSmokeResult:
    """Evidence emitted by the bounded, real v5 local preparation smoke.

    This is intentionally separate from the tiny cloud ``smoke`` mode: it
    uses complete S1 games and real frozen v5 S4 assets, but has a fixed small
    budget and is never a formal S5 training or playing-strength claim.
    """

    output_dir: Path
    completed_games: int
    trajectory_steps: int
    illegal_actions: int
    zero_sum_failures: int
    ppo_updates: int
    resume_updates: int
    losses: tuple[float, ...]
    entropies: tuple[float, ...]
    kls: tuple[float, ...]
    resume_curve_matches: bool
    v5_snapshot_in_league: bool
    effective_sampling_weights: dict[str, float]
    opponents_used_perfect_observation: bool
    learner_used_curriculum_degradation: bool
    checkpoint_path: Path
    evidence_path: Path
    benchmark_path: Path
    artifact_paths: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_paths(root: Path) -> tuple[Path, ...]:
    """Return the complete, intentionally small source set needed by S5."""
    paths: list[Path] = []
    for directory in ("engine", "state", "policies", "rl", "learning"):
        paths.extend(path for path in (root / directory).rglob("*.py") if "__pycache__" not in path.parts)
    for filename in ("config.py", "requirements.txt", "strategy_S5_rl_spec.md", "S5_expert_behavior_checklist.md", "S5_CLOUD_TRAINING_README.md", "docs/concepts.md"):
        path = root / filename
        if path.is_file():
            paths.append(path)
    paths.append(root / "tools" / "cloud_train_s5.py")
    paths.extend(path for path in (root / "configs").rglob("*") if path.is_file())
    archive = root / S4_ARCHIVE
    for relative in (
        *(Path(name) for name in S4_REQUIRED_SHA256),
        Path("reports/s4_training_report.json"),
        Path("reports/s4_training_report.md"), Path("reports/belief_bucket_report.md"),
    ):
        path = archive / relative
        if not path.is_file():
            raise FileNotFoundError(f"required archived S4 runtime asset is missing: {path}")
        paths.append(path)
    return tuple(sorted(set(paths), key=lambda path: path.relative_to(root).as_posix()))


def _verify_s4_archive(root: Path) -> None:
    """Reject a package build/run if its accepted S4 archive was modified."""
    for relative, expected_sha256 in S4_REQUIRED_SHA256.items():
        asset = root / S4_ARCHIVE / relative
        if not asset.is_file() or _sha256(asset) != expected_sha256:
            raise ValueError(f"accepted S4 checksum mismatch: {relative}")


def verify_s5_cloud_package(root: Path = PROJECT_ROOT) -> None:
    """Verify the package manifest when executing from an extracted ZIP.

    Source-tree execution has no package manifest and is permitted for local
    development.  A manifest that is present is authoritative and must cover
    exactly every delivered file before either smoke or train starts.
    """
    base = Path(root).resolve()
    manifest_path = base / PACKAGE_MANIFEST_NAME
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["files"]
        inventory = manifest["inventory"]
        s4_artifacts = manifest["s4_artifacts"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("package manifest is missing or invalid") from exc
    if (
        manifest.get("format_version") != PACKAGE_FORMAT_VERSION
        or not isinstance(records, list)
        or not isinstance(inventory, dict)
        or not isinstance(s4_artifacts, dict)
        or s4_artifacts.get("release") != S4_RELEASE
        or s4_artifacts.get("cold_start_policy_version") != S4_V5_COLD_START_POLICY_VERSION
    ):
        raise ValueError("package manifest is missing or invalid")
    listed_paths = inventory.get("all_files")
    if (
        inventory.get("policy") != "closed"
        or inventory.get("self_file") != PACKAGE_MANIFEST_NAME
        or inventory.get("self_hash_policy") != "manifest-is-verified-structurally"
        or inventory.get("runtime_outputs") != "must-be-outside-package-root"
        or not isinstance(listed_paths, list)
        or any(not isinstance(name, str) for name in listed_paths)
        or len(listed_paths) != len(set(listed_paths))
        or listed_paths != sorted(listed_paths)
    ):
        raise ValueError("package manifest is missing or invalid")
    expected: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("package manifest is missing or invalid")
        name, digest, size = record.get("path"), record.get("sha256"), record.get("bytes")
        if not isinstance(name, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise ValueError("package manifest is missing or invalid")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in expected:
            raise ValueError("package manifest is missing or invalid")
        asset = base / relative
        if not asset.is_file() or asset.stat().st_size != size or _sha256(asset) != digest:
            raise ValueError(f"package manifest verification failed: {name}")
        expected.add(name)
    if set(listed_paths) != expected | {PACKAGE_MANIFEST_NAME}:
        raise ValueError("package manifest is missing or invalid")
    actual: set[str] = set()
    for candidate in base.rglob("*"):
        relative = candidate.relative_to(base).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"unexpected package file: {relative}")
        if candidate.is_file():
            actual.add(relative)
    unexpected = sorted(actual - set(listed_paths))
    missing = sorted(set(listed_paths) - actual)
    if unexpected:
        raise ValueError(f"unexpected package file: {unexpected[0]}")
    if missing:
        raise ValueError(f"package manifest verification failed: {missing[0]}")
    required = {
        "tools/cloud_train_s5.py",
        S4_BELIEF.as_posix(),
        S4_POLICY.as_posix(),
    }
    if not required.issubset(expected):
        raise ValueError("package manifest is missing required runtime assets")


def _package_manifest(root: Path, files: Iterable[Path]) -> bytes:
    listed = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in files
    ]
    s4_root = root / S4_ARCHIVE
    file_paths = [record["path"] for record in listed]
    data = {
        "format_version": PACKAGE_FORMAT_VERSION,
        "package_kind": "s5_cloud_training",
        "deterministic_zip_timestamp_utc": "1980-01-01T00:00:00Z",
        "files": listed,
        "inventory": {
            "policy": "closed",
            "all_files": sorted([*file_paths, PACKAGE_MANIFEST_NAME]),
            "self_file": PACKAGE_MANIFEST_NAME,
            "self_hash_policy": "manifest-is-verified-structurally",
            "runtime_outputs": "must-be-outside-package-root",
        },
        "s4_artifacts": {
            "release": S4_RELEASE,
            "cold_start_policy_version": S4_V5_COLD_START_POLICY_VERSION,
            "belief": {"path": S4_BELIEF.as_posix(), "sha256": _sha256(s4_root / "checkpoints/belief_s4.pt")},
            "policy": {"path": S4_POLICY.as_posix(), "sha256": _sha256(s4_root / "checkpoints/policy_s4.pt")},
        },
        "omissions": ["S4 raw decision data (data/s4_decisions.jsonl)", "prior cloud ZIPs", "runtime outputs", "__pycache__"],
    }
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _writestr(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compresslevel=9)


def build_s5_cloud_package(output_path: Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Build byte-for-byte reproducible cloud ZIP without raw S4 training data."""
    root = Path(project_root).resolve()
    _verify_s4_archive(root)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = _runtime_paths(root)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    manifest = _package_manifest(root, files)
    entries = [(path.relative_to(root).as_posix(), path.read_bytes()) for path in files]
    entries.append((PACKAGE_MANIFEST_NAME, manifest))
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for name, contents in sorted(entries, key=lambda item: item[0]):
            _writestr(archive, name, contents)
    os.replace(temporary, destination)
    return destination


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _torch_sampling_seed(seed: int) -> int:
    """Map S5's SHA-256 decision seed into torch's signed 63-bit domain."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("sampling seed must be an integer")
    return seed % (2**63 - 1)


def _controlled_dependencies(seed: int):
    """Bounded real-PPO adapter used only for smoke/bootstrap execution."""
    import torch

    from rl.curriculum import CurriculumConfig, CurriculumStage, DegradationProfile, ObservationCurriculum
    from rl.league import DualArenaMetrics, OpponentLeague, SnapshotMetadata
    from rl.train_rl import S5TrainingDependencies
    from rl.types import TrajectoryStep

    def rollout(model, _config, update: int, next_seed: int, _runtime):
        generator = torch.Generator(device="cpu").manual_seed(seed + next_seed + update)
        feature_size, action_size = model.config.input_size, model.config.action_size
        features = torch.rand((2, feature_size), generator=generator).tolist()
        actions = (next_seed + update) % action_size, (next_seed + update + 1) % action_size
        masks: list[list[bool]] = []
        for action in actions:
            mask = [False] * action_size
            mask[action] = True
            mask[(action + 1) % action_size] = True
            masks.append(mask)
        with torch.no_grad():
            outputs = model(torch.tensor(features, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool))
            log_probs = torch.log_softmax(outputs.action_logits, dim=-1)
        return tuple(
            TrajectoryStep(tuple(features[index]), tuple(masks[index]), actions[index], float(log_probs[index, actions[index]]), float(outputs.values[index]), 0.0 if index == 0 else 0.1, index == 1, S4_V5_COLD_START_POLICY_VERSION)
            for index in range(2)
        )

    def arena(_model, opponent, _runtime):
        # Deterministic, explicitly controlled bootstrap check.  It exercises
        # the dual-track persistence/admission plumbing but is not a strength claim.
        offset = (sum(ord(char) for char in opponent.key) % 7) / 1000.0
        return DualArenaMetrics(0.55 + offset, 0.53 + offset, 4, 4, 0, 0)

    def league() -> OpponentLeague:
        return OpponentLeague(current_policy=SnapshotMetadata(
            S4_V5_COLD_START_POLICY_VERSION,
            S4_V5_COLD_START_POLICY_VERSION,
            "cold-start.pt",
            0,
        ))

    def curriculum() -> ObservationCurriculum:
        return ObservationCurriculum(CurriculumConfig((
            CurriculumStage("perfect", DegradationProfile.perfect(), 0.0, 0.0),
            CurriculumStage("light_noise", DegradationProfile("light_noise", 0.03, 0.0), 0.0, 0.0),
        )))

    return S5TrainingDependencies(rollout_factory=rollout, arena_evaluator=arena, league_factory=league, curriculum_factory=curriculum)


def _formal_dependencies(root: Path, cloud_config: S5CloudRunConfig):
    """Build actual S1/S2/S3 training adapters for formal ``--mode train``.

    This is intentionally concrete rather than a callback-shaped synthetic
    source: each learner record is produced by :func:`run_rollout_game`; every
    reported win rate comes from full engine arena games.  The frozen S4 belief
    asset is shared read-only across all learner and evaluation decisions.
    """
    import torch

    from rl.curriculum import CurriculumConfig, CurriculumStage, DegradationProfile, ObservationCurriculum
    from rl.league import OpponentLeague, SnapshotMetadata
    from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
    from rl.opponent_resolver import OpponentResolver
    from rl.production_adapter import (
        ProductionGameTask,
        RolloutRuntimeState,
        run_production_arena,
        run_production_game,
    )
    from rl.rollout import FrozenBeliefProvider, LearnerDecision
    from rl.train_rl import S5TrainingDependencies

    belief_provider = FrozenBeliefProvider.from_checkpoint(str(root / S4_BELIEF))
    frozen_policy_path = root / S4_POLICY
    frozen_policy_payload = torch.load(frozen_policy_path, map_location="cpu", weights_only=True)
    raw_model_config = frozen_policy_payload.get("model_config") if isinstance(frozen_policy_payload, dict) else None
    if not isinstance(raw_model_config, dict):
        raise ValueError("frozen S4 policy checkpoint must include model_config")
    model_config = PolicyValueNetConfig(**raw_model_config)
    opponent_resolver = OpponentResolver(
        model_factory=lambda: PolicyValueNet(model_config),
        belief_provider=belief_provider,
    )

    def league() -> OpponentLeague:
        return OpponentLeague(current_policy=SnapshotMetadata(
            S4_V5_COLD_START_POLICY_VERSION,
            S4_V5_COLD_START_POLICY_VERSION,
            str(frozen_policy_path),
            0,
            _sha256(frozen_policy_path),
        ))

    curriculum_state = ObservationCurriculum(CurriculumConfig((
        CurriculumStage("perfect", DegradationProfile.perfect(), 0.0, 0.0),
        CurriculumStage("vision_noise", DegradationProfile("vision_noise", 0.03, 0.0), 0.0, 0.0),
        CurriculumStage("partial_history", DegradationProfile("partial_history", 0.06, 0.25), 0.0, 0.0),
    )))

    def learner_decider(model, view):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                output = model(
                    torch.tensor([view.feature_values], dtype=torch.float32, device=device),
                    torch.tensor([view.legal_mask], dtype=torch.bool, device=device),
                )
                probabilities = torch.softmax(output.action_logits, dim=-1).cpu()[0]
                generator = torch.Generator(device="cpu").manual_seed(_torch_sampling_seed(view.decision_seed))
                action = int(torch.multinomial(probabilities, 1, generator=generator).item())
                log_prob = float(torch.log(probabilities[action]).item())
                value = float(output.values.item())
        finally:
            model.train(was_training)
        return LearnerDecision(action, log_prob, value)

    def runtime_state(model, state, *, arena_seed: int = 0):
        return RolloutRuntimeState(
            league=state.league,
            curriculum=state.curriculum,
            policy_generation=state.policy_generation,
            policy_checksum=state.policy_checksum,
            learner_decider=lambda view: learner_decider(model, view),
            opponent_resolver=opponent_resolver,
            belief_provider=belief_provider,
            arena_seed=arena_seed,
            max_game_steps=cloud_config.max_game_steps,
        )

    def rollout(model, _training_config, update: int, next_seed: int, state):
        steps = []
        runtime = runtime_state(model, state)
        for episode in range(cloud_config.episodes_per_update):
            episode_seed = cloud_config.seed + next_seed * 10_000 + episode
            task = ProductionGameTask(
                round_id=next_seed,
                game_index=episode,
                policy_generation=state.policy_generation,
                policy_checksum=state.policy_checksum,
                environment_seed=episode_seed,
                learner_seed=episode_seed ^ 0x5A17,
                opponent_seed=episode_seed,
                learner_seat=episode % 4,
            )
            steps.extend(run_production_game(task, runtime).steps)
        return tuple(steps)

    def arena(model, entry, state):
        return run_production_arena(
            model,
            entry,
            runtime_state(model, state, arena_seed=cloud_config.seed),
            cloud_config.arena_games,
        )

    return S5TrainingDependencies(
        rollout_factory=rollout,
        arena_evaluator=arena,
        league_factory=league,
        curriculum_factory=lambda: curriculum_state,
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Durably publish a run manifest only after every referenced artifact exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            parent = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_package_output_outside_root(root: Path, output_dir: Path) -> None:
    """Keep generated checkpoints and reports out of the closed source inventory."""
    if not (root / PACKAGE_MANIFEST_NAME).is_file():
        return
    try:
        Path(output_dir).resolve().relative_to(root)
    except ValueError:
        return
    raise ValueError("package runtime output directory must be outside package root")


def _validate_prep_trajectory(steps, *, feature_size: int, action_size: int) -> None:
    """Reject incomplete or non-portable learner trajectory evidence early."""
    if not steps:
        raise RuntimeError("completed v5 rollout recorded no learner trajectory")
    for index, step in enumerate(steps):
        if len(step.feature_values) != feature_size or len(step.legal_mask) != action_size:
            raise RuntimeError("v5 rollout trajectory schema does not match the loaded policy")
        if not 0 <= step.action < action_size or not step.legal_mask[step.action]:
            raise RuntimeError("v5 rollout trajectory contains an illegal action")
        if not all(math.isfinite(float(value)) for value in (*step.feature_values, step.old_log_prob, step.value, step.reward)):
            raise RuntimeError("v5 rollout trajectory contains non-finite values")
        terminal = index == len(steps) - 1
        if step.done != terminal or (not terminal and step.reward != 0.0):
            raise RuntimeError("v5 rollout trajectory violates terminal-reward schema")


def _write_local_prep_benchmark(
    path: Path,
    *,
    root: Path,
    games: int,
    elapsed_seconds: float,
    seed: int,
    torch_threads: int,
) -> None:
    """Write the measured, bounded rollout ledger without retaining decisions."""
    rate = games / elapsed_seconds * 60.0 if elapsed_seconds > 0.0 else 0.0
    content = (
        "# S5 Local v5 Rollout Benchmark\n\n"
        "This is bounded pre-training evidence, not formal S5 training or a strength result.\n\n"
        f"- Command: `python -c \"from tools.cloud_train_s5 import run_local_v5_prep_smoke; run_local_v5_prep_smoke()\"`\n"
        f"- Seed: `{seed}`\n"
        f"- Completed real S1 rollouts: `{games}`\n"
        f"- Rollout elapsed seconds: `{elapsed_seconds:.3f}`\n"
        f"- Throughput: `{rate:.3f}` games/minute\n"
        f"- OS: `{platform.platform()}`\n"
        f"- CPU logical cores: `{os.cpu_count()}`\n"
        f"- Python: `{sys.version.split()[0]}`\n"
        f"- PyTorch threads: `{torch_threads}`\n"
        f"- v5 belief SHA256: `{_sha256(root / S4_BELIEF)}`\n"
        f"- v5 policy SHA256: `{_sha256(root / S4_POLICY)}`\n"
    )
    _atomic_write_json(path.with_suffix(".json"), {
        "kind": "s5_local_v5_rollout_benchmark",
        "seed": seed,
        "completed_games": games,
        "elapsed_seconds": elapsed_seconds,
        "games_per_minute": rate,
        "os": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch_threads": torch_threads,
        "policy_sha256": _sha256(root / S4_POLICY),
        "belief_sha256": _sha256(root / S4_BELIEF),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def run_local_v5_prep_smoke(
    *,
    output_dir: Path | None = None,
    games: int = 50,
    updates: int = 10,
    seed: int = 20260718,
    project_root: Path = PROJECT_ROOT,
) -> S5LocalPrepSmokeResult:
    """Run the required bounded, real-v5 local preparation validation.

    The only persisted records are aggregate evidence, health curves and a
    resumable checkpoint.  In particular, per-decision observations/actions
    are never serialized, so this cannot accidentally become a raw training
    data archive.
    """
    if not isinstance(games, int) or isinstance(games, bool) or games < 50:
        raise ValueError("local v5 preparation smoke requires at least 50 completed games")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 10:
        raise ValueError("local v5 preparation smoke requires at least 10 PPO updates")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    import torch

    from policies.opponent_pool import GreedyPolicy, RandomPolicy
    from policies.rule_policy import RulePolicy
    from rl.checkpoints import load_checkpoint, save_checkpoint
    from rl.curriculum import CurriculumConfig, CurriculumStage, DegradationProfile, ObservationCurriculum
    from rl.league import OpponentLeague, SnapshotMetadata
    from rl.ppo_trainer import PPOConfig, ppo_update
    from rl.rollout import FrozenBeliefProvider, LearnerDecision, RolloutConfig, run_rollout_game
    from rl.train_rl import _batch_from_steps, _default_model, _frozen_reference_policy
    from state.observation_degradation import DegradationPipeline, VisionNoise

    root = Path(project_root).resolve()
    _verify_s4_archive(root)
    destination = (root / "training_artifacts" / "S5" / "prep_20260718") if output_dir is None else Path(output_dir)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = destination / "checkpoints"
    evidence_path = destination / "local_prep_evidence.json"
    benchmark_path = destination / "rollout_benchmark.md"

    # Use exactly the accepted v5 policy architecture/weights for both learner
    # cold start and frozen KL reference.  ``_default_model`` validates the
    # current encoder contract before loading it.
    from rl.train_rl import S5TrainingConfig

    provenance = {
        "release": S4_RELEASE,
        "encoder_version": "s2.v4.encoder.v4",
        "cold_start_policy_version": S4_V5_COLD_START_POLICY_VERSION,
        "policy_sha256": _sha256(root / S4_POLICY),
        "belief_sha256": _sha256(root / S4_BELIEF),
    }
    model_config = S5TrainingConfig(
        output_dir=destination,
        frozen_s4_belief_path=root / S4_BELIEF,
        frozen_s4_policy_path=root / S4_POLICY,
        frozen_s4_provenance=provenance,
        updates=updates,
        rollout_seed_start=seed,
        device="cpu",
    )
    torch.manual_seed(seed)
    learner_model = _default_model(model_config).cpu()
    reference_model = _frozen_reference_policy(learner_model).cpu()
    belief_provider = FrozenBeliefProvider.from_checkpoint(str(root / S4_BELIEF))
    optimizer = torch.optim.Adam(learner_model.parameters(), lr=model_config.learning_rate)
    ppo_config = PPOConfig(kl_coef=0.05)

    profile = DegradationProfile("prep_light_noise", vision_miss_rate=0.03, mid_game_ratio=0.0)
    curriculum = ObservationCurriculum(CurriculumConfig((
        CurriculumStage("prep_light_noise", profile, 1.0, 1.0),
    )))
    current = SnapshotMetadata(
        snapshot_id=S4_V5_COLD_START_POLICY_VERSION,
        policy_version=S4_V5_COLD_START_POLICY_VERSION,
        checkpoint_path=str(root / S4_POLICY),
        training_step=0,
        checksum=_sha256(root / S4_POLICY),
    )
    league = OpponentLeague(current_policy=current)
    opponents = (RulePolicy(), GreedyPolicy(), RandomPolicy(seed=seed ^ 0xA5A5))
    degradation_calls = 0

    def learner_degrader(state, degradation_seed: int):
        nonlocal degradation_calls
        degradation_calls += 1
        return DegradationPipeline([VisionNoise(miss_rate=profile.vision_miss_rate, seed=degradation_seed)]).apply(state)

    def learner_decider(view):
        learner_model.eval()
        with torch.no_grad():
            output = learner_model(
                torch.tensor([view.feature_values], dtype=torch.float32),
                torch.tensor([view.legal_mask], dtype=torch.bool),
            )
            probabilities = torch.softmax(output.action_logits[0], dim=-1)
            generator = torch.Generator(device="cpu").manual_seed(_torch_sampling_seed(view.decision_seed))
            action = int(torch.multinomial(probabilities, 1, generator=generator).item())
            return LearnerDecision(action, float(torch.log(probabilities[action]).item()), float(output.values[0].item()))

    all_steps = []
    illegal_actions = 0
    zero_sum_failures = 0
    rollout_started = time.perf_counter()
    for game_index in range(games):
        result = run_rollout_game(
            learner_decider,
            opponents,
            config=RolloutConfig(
                seed=seed + game_index,
                learner_rng_seed=(seed << 16) + game_index,
                learner_seat=game_index % 4,
                max_steps=1000,
                observation_noise=profile.vision_miss_rate,
            ),
            policy_version=S4_V5_COLD_START_POLICY_VERSION,
            belief_provider=belief_provider,
            observation_degrader=learner_degrader,
        )
        _validate_prep_trajectory(
            result.steps,
            feature_size=learner_model.config.input_size,
            action_size=learner_model.config.action_size,
        )
        illegal_actions += int(result.illegal_action)
        zero_sum_failures += int(sum(result.scores) != 0)
        all_steps.extend(result.steps)
    rollout_elapsed = time.perf_counter() - rollout_started
    if illegal_actions or zero_sum_failures:
        raise RuntimeError("v5 local preparation rollout invariants failed")

    batch = _batch_from_steps(tuple(all_steps), reference_model)

    # Resume evidence must compare two *identical* PPO executions.  Keep all
    # captured inputs in memory only: the published artefact remains aggregate
    # evidence and never becomes a decision-level training-data archive.
    initial_model_state = copy.deepcopy(learner_model.state_dict())
    initial_batch_tensors = {
        name: getattr(batch, name).detach().clone()
        for name in (
            "features", "legal_mask", "actions", "old_log_probs", "old_values",
            "rewards", "dones", "reference_logits",
        )
    }

    def exact_state_matches(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor):
            return (
                isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and torch.equal(left, right)
            )
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(
                exact_state_matches(value, right[name]) for name, value in left.items()
            )
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(
                exact_state_matches(value, other) for value, other in zip(left, right)
            )
        return type(left) is type(right) and left == right

    def make_comparison_model():
        candidate = _default_model(model_config).cpu()
        candidate.load_state_dict(initial_model_state)
        return candidate

    # Construct both paths from cloned, validated cold-start state.  The frozen
    # references are retained as explicit evidence that the same S4 policy
    # produced the immutable KL targets in both paths.
    uninterrupted_model = make_comparison_model()
    split_model = make_comparison_model()
    uninterrupted_reference = _frozen_reference_policy(uninterrupted_model).cpu()
    split_reference = _frozen_reference_policy(split_model).cpu()
    uninterrupted_optimizer = torch.optim.Adam(uninterrupted_model.parameters(), lr=model_config.learning_rate)
    split_optimizer = torch.optim.Adam(split_model.parameters(), lr=model_config.learning_rate)
    same_initial_model_state = exact_state_matches(uninterrupted_model.state_dict(), split_model.state_dict())
    same_initial_optimizer_state = exact_state_matches(
        uninterrupted_optimizer.state_dict(), split_optimizer.state_dict()
    )
    same_reference_policy_state = (
        exact_state_matches(reference_model.state_dict(), uninterrupted_reference.state_dict())
        and exact_state_matches(uninterrupted_reference.state_dict(), split_reference.state_dict())
    )

    def validate_health(health, *, phase: str) -> None:
        values = (health.total_loss, health.policy_loss, health.value_loss, health.entropy, health.kl, health.grad_norm)
        if not all(math.isfinite(value) for value in values) or health.entropy <= 0.0 or health.kl < 0.0:
            raise RuntimeError(f"{phase} v5 local preparation PPO health is non-finite or unreasonable")

    # Run the uninterrupted control all the way to 15 updates with the same
    # batch and explicit seed as the split/resume path below.
    comparison_seed = seed ^ 0x5A17
    torch.manual_seed(comparison_seed)
    health_rows = []
    uninterrupted_health = []
    for step in range(1, updates + 6):
        health = ppo_update(uninterrupted_model, batch, uninterrupted_optimizer, ppo_config)
        validate_health(health, phase="uninterrupted")
        uninterrupted_health.append(health)
        if step <= updates:
            health_rows.append({"global_step": step, "loss": health.total_loss, "entropy": health.entropy, "kl": health.kl})

    # The split path deliberately repeats the same first ten update inputs and
    # seed, then serializes the complete optimizer state for a true resume.
    torch.manual_seed(comparison_seed)
    split_health = []
    for _step in range(1, updates + 1):
        health = ppo_update(split_model, batch, split_optimizer, ppo_config)
        validate_health(health, phase="pre-checkpoint")
        split_health.append(health)

    checkpoint_path = checkpoints / f"v5-prep-step-{updates}.pt"
    save_checkpoint(
        checkpoint_path,
        model=split_model,
        optimizer=split_optimizer,
        global_step=updates,
        next_rollout_seed=seed + games,
        league=league,
        curriculum=curriculum,
        config={"kind": "bounded_local_v5_prep", "games": games, "updates": updates},
        metrics={"health": health_rows},
        frozen_s4_provenance=provenance,
    )

    resumed_model = _default_model(model_config).cpu()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=model_config.learning_rate)
    restored = load_checkpoint(checkpoint_path, model=resumed_model, optimizer=resumed_optimizer, map_location="cpu")
    resume_health = []
    resumed_health = []
    for offset in range(1, 6):
        health = ppo_update(resumed_model, batch, resumed_optimizer, ppo_config)
        validate_health(health, phase="resumed")
        resumed_health.append(health)
        global_step = restored.global_step + offset
        resume_health.append({"global_step": global_step, "loss": health.total_loss, "entropy": health.entropy, "kl": health.kl})
    same_update_inputs = all(
        torch.equal(getattr(batch, name), expected)
        for name, expected in initial_batch_tensors.items()
    )
    model_state_matches = exact_state_matches(uninterrupted_model.state_dict(), resumed_model.state_dict())
    optimizer_state_matches = exact_state_matches(
        uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict()
    )
    pre_checkpoint_health_matches = all(
        exact_state_matches(asdict(control), asdict(split))
        for control, split in zip(uninterrupted_health[:updates], split_health)
    )
    resume_health_matches = all(
        exact_state_matches(asdict(control), asdict(resumed))
        for control, resumed in zip(uninterrupted_health[updates:], resumed_health)
    )
    resume_equivalence = {
        "comparison_seed": comparison_seed,
        "uninterrupted_updates": updates + 5,
        "split_updates_before_checkpoint": updates,
        "split_updates_after_resume": 5,
        "same_initial_model_state": same_initial_model_state,
        "same_initial_optimizer_state": same_initial_optimizer_state,
        "same_reference_policy_state": same_reference_policy_state,
        "same_update_inputs": same_update_inputs,
        "pre_checkpoint_health_matches": pre_checkpoint_health_matches,
        "resume_health_matches": resume_health_matches,
        "model_state_matches": model_state_matches,
        "optimizer_state_matches": optimizer_state_matches,
        "strict_tolerance": 0.0,
    }
    resume_curve_matches = (
        restored.global_step == updates
        and restored.next_rollout_seed == seed + games
        and all(
            resume_equivalence[name]
            for name in (
                "same_initial_model_state",
                "same_initial_optimizer_state",
                "same_reference_policy_state",
                "same_update_inputs",
                "pre_checkpoint_health_matches",
                "resume_health_matches",
                "model_state_matches",
                "optimizer_state_matches",
            )
        )
    )
    if not resume_curve_matches:
        raise RuntimeError("v5 preparation checkpoint does not reproduce the uninterrupted PPO state")

    # With no historical snapshots, configured history mass is intentionally
    # unavailable; record the normalized effective distribution used now.
    weights = {
        "current": league.config.latest_weight,
        "s3": league.config.s3_weight,
        "greedy": league.config.greedy_random_weight / 2.0,
        "random": league.config.greedy_random_weight / 2.0,
    }
    total_weight = sum(weights.values())
    effective_weights = {key: value / total_weight for key, value in weights.items()}
    sampled = league.sample(3, seed=seed)
    v5_snapshot_in_league = (
        league.current_entry.snapshot == current
        and current.checkpoint_path == str(root / S4_POLICY)
        and all(weight > 0.0 for weight in effective_weights.values())
        and len(sampled) == 3
    )
    opponents_used_perfect_observation = (
        curriculum.opponent_profile.is_perfect
        and all(not hasattr(opponent, "degradation") for opponent in opponents)
    )
    learner_used_curriculum_degradation = degradation_calls > 0 and not curriculum.learner_profile.is_perfect
    if not v5_snapshot_in_league or not opponents_used_perfect_observation or not learner_used_curriculum_degradation:
        raise RuntimeError("v5 local preparation league or observation-isolation check failed")

    _write_local_prep_benchmark(
        benchmark_path,
        root=root,
        games=games,
        elapsed_seconds=rollout_elapsed,
        seed=seed,
        torch_threads=torch.get_num_threads(),
    )
    evidence = {
        "kind": "bounded_local_s5_v5_preparation_smoke",
        "warning": "Not formal S5 training and not a playing-strength result.",
        "seed": seed,
        "completed_games": games,
        "trajectory_steps": len(all_steps),
        "illegal_actions": illegal_actions,
        "zero_sum_failures": zero_sum_failures,
        "ppo_updates_before_resume": updates,
        "ppo_updates_after_resume": 5,
        "resume_curve_matches": resume_curve_matches,
        "resume_equivalence": resume_equivalence,
        "v5_snapshot_in_league": v5_snapshot_in_league,
        "effective_sampling_weights": effective_weights,
        "opponents_used_perfect_observation": opponents_used_perfect_observation,
        "learner_used_curriculum_degradation": learner_used_curriculum_degradation,
        "rollout_elapsed_seconds": rollout_elapsed,
        "rollout_games_per_minute": games / rollout_elapsed * 60.0 if rollout_elapsed > 0.0 else 0.0,
        "health": health_rows,
        "resume_health": resume_health,
        "frozen_s4_provenance": provenance,
        "raw_decisions_written": False,
    }
    _atomic_write_json(evidence_path, evidence)
    artifacts = (checkpoint_path, evidence_path, benchmark_path, benchmark_path.with_suffix(".json"))
    return S5LocalPrepSmokeResult(
        destination, games, len(all_steps), illegal_actions, zero_sum_failures, updates, 5,
        tuple(row["loss"] for row in health_rows),
        tuple(row["entropy"] for row in health_rows),
        tuple(row["kl"] for row in health_rows),
        resume_curve_matches, v5_snapshot_in_league, effective_weights,
        opponents_used_perfect_observation, learner_used_curriculum_degradation,
        checkpoint_path, evidence_path, benchmark_path, artifacts,
    )


def run_s5_cloud_training(config: S5CloudRunConfig, *, project_root: Path = PROJECT_ROOT) -> S5CloudTrainingResult:
    """Run a bounded S5 diagnostic or formal engine-backed training job."""
    if not isinstance(config, S5CloudRunConfig):
        raise TypeError("config must be S5CloudRunConfig")
    root = Path(project_root).resolve()
    verify_s5_cloud_package(root)
    _require_package_output_outside_root(root, config.output_dir)
    _verify_s4_archive(root)
    device = _resolve_device(config.device)
    # The package root is made importable only after its closed inventory has
    # been checked.  Everything above this line is stdlib-only verification.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rl.train_rl import S5TrainingConfig, load_published_s5_report, run_s5_training

    belief, policy = root / S4_BELIEF, root / S4_POLICY
    dependencies = _controlled_dependencies(config.seed) if config.mode == "smoke" else _formal_dependencies(root, config)
    frozen_s4_provenance = {
        "release": S4_RELEASE,
        "cold_start_policy_version": S4_V5_COLD_START_POLICY_VERSION,
        "frozen_artifact_sha256": dict(S4_REQUIRED_SHA256),
    }
    result = run_s5_training(S5TrainingConfig(
        output_dir=config.output_dir,
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance=frozen_s4_provenance,
        updates=config.updates,
        rollout_seed_start=config.seed,
        snapshot_interval=1,
        device=device,
    ), dependencies=dependencies)
    # The report pair must be committed and mutually hash-verified before the
    # outer run manifest points at it.
    load_published_s5_report(result.report_path)
    manifest = {
        "format_version": 1,
        "run_kind": "formal_s5_engine_training" if config.mode == "train" else "controlled_s5_smoke",
        "warning": (
            "Formal runs use real engine rollout and arena games; independent acceptance evaluation is still required."
            if config.mode == "train" else
            "Smoke is a controlled bootstrap diagnosis, not training or playing-strength evidence."
        ),
        "resolved_config": {**asdict(config), "output_dir": str(config.output_dir), "resolved_device": device},
        "s4_artifacts": {
            "release": S4_RELEASE,
            "cold_start_policy_version": S4_V5_COLD_START_POLICY_VERSION,
            "belief": {"path": str(belief), "sha256": _sha256(belief)},
            "policy": {"path": str(policy), "sha256": _sha256(policy)},
        },
        "frozen_s4_provenance": frozen_s4_provenance,
        "artifacts": {
            "report_publication_manifest": {"path": str(result.publication_manifest_path), "sha256": _sha256(result.publication_manifest_path)},
            "checkpoint": {"path": str(result.checkpoint_path), "sha256": _sha256(result.checkpoint_path)},
        },
        "dual_track_arena": result.report["arena"]["s3_comparison"],
        "global_step": result.global_step,
    }
    manifest_path = config.output_dir / "s5_cloud_run_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return S5CloudTrainingResult(config.output_dir, manifest_path, result.report_path, result.checkpoint_path, device, result.global_step)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run the reproducible S5 cloud training package")
    parser.add_argument("--build-package", type=Path, help="write a deterministic S5 cloud ZIP and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("cloud_outputs/s5_smoke"))
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--episodes-per-update", type=int, default=4, help="formal train: completed S1 games per PPO update")
    parser.add_argument("--arena-games", type=int, default=None, help="full games per opponent and arena track (train default: 1000; smoke: 4)")
    parser.add_argument("--max-game-steps", type=int, default=1000)
    return parser.parse_args(argv)


def _require_no_site_startup() -> bool:
    """Refuse a CLI process whose import startup could run ``sitecustomize``.

    This module deliberately imports only the standard library before this
    boundary.  In particular, a package extracted beside the script must be
    started with ``-S`` so Python skips ``site`` and cannot execute an
    attacker-supplied ``tools/sitecustomize.py`` before the closed-inventory
    verifier runs.
    """
    if sys.flags.no_site:
        return True
    print(
        "Refusing unsafe Python startup: run `python -S tools/cloud_train_s5.py ...` "
        "so sitecustomize is disabled before package verification.",
        file=sys.stderr,
    )
    return False


def _enable_installed_dependencies_without_site() -> None:
    """Expose the selected interpreter's installed wheels without importing ``site``.

    ``-S`` intentionally omits site-packages, including from an activated
    virtual environment.  Adding existing wheel directories directly keeps
    that startup protection intact: unlike ``site.addsitedir`` it does not
    process ``.pth`` files or execute their import statements.
    """
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    ]
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        environment = Path(virtual_env)
        candidates.extend((
            str(environment / "Lib" / "site-packages"),
            str(environment / "lib" / version / "site-packages"),
        ))
    executable_environment = Path(sys.executable).resolve().parent.parent
    candidates.extend((
        str(executable_environment / "Lib" / "site-packages"),
        str(executable_environment / "lib" / version / "site-packages"),
    ))
    for candidate in candidates:
        if not candidate:
            continue
        directory = Path(candidate)
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.append(str(directory))


def main(argv: list[str] | None = None) -> int:
    if not _require_no_site_startup():
        return 2
    _enable_installed_dependencies_without_site()
    args = parse_args(argv)
    if args.build_package is not None:
        print(build_s5_cloud_package(args.build_package))
        return 0
    result = run_s5_cloud_training(S5CloudRunConfig(
        args.output_dir, args.mode, args.device, args.updates, args.seed,
        args.episodes_per_update, args.arena_games, args.max_game_steps,
    ))
    print(json.dumps({"manifest": str(result.manifest_path), "device": result.device, "global_step": result.global_step}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
