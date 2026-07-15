"""Reproducible S5 cloud package builder and bounded training entry point.

``smoke`` is deliberately a tiny controlled bootstrap diagnosis.  ``train``
uses complete S1 games through S2 observations, frozen accepted S4 belief,
S3/league opponents, curriculum degradation, and real dual-track arena games.
Only the latter is an S5 training run; neither mode is an acceptance claim by
itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S4_ARCHIVE = Path("training_artifacts/S4/v1_20260711_repaired_cuda")
S4_BELIEF = S4_ARCHIVE / "checkpoints/belief_s4.pt"
S4_POLICY = S4_ARCHIVE / "checkpoints/policy_s4.pt"
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
        Path("LOCAL_ARCHIVE_README.txt"), Path("manifest.json"), Path("local_file_checksums.json"),
        Path("checkpoints/belief_s4.pt"), Path("checkpoints/policy_s4.pt"),
        Path("reports/s4_training_report.json"), Path("reports/belief_s4_test_evaluation.json"),
        Path("reports/s4_training_report.md"),
    ):
        path = archive / relative
        if not path.is_file():
            raise FileNotFoundError(f"required archived S4 runtime asset is missing: {path}")
        paths.append(path)
    return tuple(sorted(set(paths), key=lambda path: path.relative_to(root).as_posix()))


def _verify_s4_archive(root: Path) -> None:
    """Reject a package build/run if its accepted S4 archive was modified."""
    checksum_path = root / S4_ARCHIVE / "local_file_checksums.json"
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        records = payload["verified_files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("accepted S4 checksum manifest is missing or invalid") from exc
    required = {"manifest.json", "checkpoints/belief_s4.pt", "checkpoints/policy_s4.pt"}
    found: set[str] = set()
    if not isinstance(records, list):
        raise ValueError("accepted S4 checksum manifest is missing or invalid")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            raise ValueError("accepted S4 checksum manifest is missing or invalid")
        relative = Path(record["path"])
        if relative.as_posix() not in required:
            continue
        asset = root / S4_ARCHIVE / relative
        if not asset.is_file() or _sha256(asset) != record["sha256"]:
            raise ValueError(f"accepted S4 checksum mismatch: {relative.as_posix()}")
        found.add(relative.as_posix())
    if found != required:
        raise ValueError("accepted S4 checksum manifest lacks required frozen assets")


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
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("package manifest is missing or invalid") from exc
    if manifest.get("format_version") != PACKAGE_FORMAT_VERSION or not isinstance(records, list) or not isinstance(inventory, dict):
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
        "training_artifacts/S4/v1_20260711_repaired_cuda/checkpoints/belief_s4.pt",
        "training_artifacts/S4/v1_20260711_repaired_cuda/checkpoints/policy_s4.pt",
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
            TrajectoryStep(tuple(features[index]), tuple(masks[index]), actions[index], float(log_probs[index, actions[index]]), float(outputs.values[index]), 0.0 if index == 0 else 0.1, index == 1, "s4-v1-cold-start")
            for index in range(2)
        )

    def arena(_model, opponent, _runtime):
        # Deterministic, explicitly controlled bootstrap check.  It exercises
        # the dual-track persistence/admission plumbing but is not a strength claim.
        offset = (sum(ord(char) for char in opponent.key) % 7) / 1000.0
        return DualArenaMetrics(0.55 + offset, 0.53 + offset, 4, 4, 0, 0)

    def league() -> OpponentLeague:
        return OpponentLeague(current_policy=SnapshotMetadata("s4-v1-cold-start", "s4-v1-cold-start", "cold-start.pt", 0))

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
            "s4-v1-cold-start",
            "s4-v1-cold-start",
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
    result = run_s5_training(S5TrainingConfig(
        output_dir=config.output_dir,
        frozen_s4_belief_path=belief,
        frozen_s4_policy_path=policy,
        frozen_s4_provenance={"release": "S4/v1_20260711_repaired_cuda", "archive_manifest_sha256": _sha256(root / S4_ARCHIVE / "manifest.json")},
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
            "belief": {"path": str(belief), "sha256": _sha256(belief)},
            "policy": {"path": str(policy), "sha256": _sha256(policy)},
        },
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
