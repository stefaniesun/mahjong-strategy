"""Reproducible S4-v5 belief evaluation by wall-count and observation profile.

This is deliberately an evaluation-only tool: it loads the frozen v5 belief
checkpoint and never changes encoder or model weights.  It reuses the S4
dataset-builder and belief evaluators so labels, degradation, and metrics are
identical to the training/evaluation pipeline.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from learning.datasets.dataset_builder import DatasetBuildConfig, build_belief_sample
from learning.eval.eval_belief import evaluate_belief_model, evaluate_prior_belief_samples
from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed
from selfplay.data_recorder import DecisionRecord, run_recorded_selfplay_game
from state.encoder import ENCODER_VERSION
from tools.cloud_train_s4_50k_cached import _split_name


PROFILES = ("perfect", "light_noise", "midgame", "heavy")
PHASE_BUCKETS = ("opening", "midgame", "endgame")
DEFAULT_SEED = 20260716
DEFAULT_TARGET_VALIDATION_RECORDS = 20_000
DEFAULT_CANDIDATE_GAMES = 3_000
DEFAULT_CHECKPOINT = Path("training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/belief_s4.pt")
DEFAULT_REPORT = Path("training_artifacts/S4/v5_20260718_encoder_v4/reports/belief_bucket_report.md")


@dataclass(frozen=True)
class Acceptance:
    passed: bool
    reasons: tuple[str, ...]


def phase_bucket_from_wall_count(wall_count: int) -> str:
    """Classify exactly as the S4 supplement specifies."""
    if wall_count > 40:
        return "opening"
    if wall_count >= 20:
        return "midgame"
    return "endgame"


def select_validation_records(records: Iterable[DecisionRecord], *, seed: int) -> list[DecisionRecord]:
    """Use the cloud trainer's game-level split, never a record-level split."""
    return [record for record in records if _split_name(record.game_id, seed) == "val"]


def evaluate_acceptance(
    metrics: dict[str, dict[str, dict[str, float]]],
    *,
    source_records: int | float | None = None,
    target_validation_records: int | float | None = None,
) -> Acceptance:
    reasons: list[str] = []
    for profile in PROFILES:
        buckets = metrics.get(profile, {})
        for phase in PHASE_BUCKETS:
            values = buckets.get(phase)
            if not values:
                reasons.append(f"{profile}/{phase}: no validation records")
                continue
            numeric: dict[str, float] = {}
            for field in ("samples", "model_tile_log_loss", "prior_tile_log_loss", "gain"):
                try:
                    numeric[field] = float(values.get(field, 1.0 if field == "samples" else float("nan")))
                except (TypeError, ValueError):
                    numeric[field] = float("nan")
                if not math.isfinite(numeric[field]):
                    reasons.append(f"{profile}/{phase}: {field} is not finite")
            if not math.isfinite(numeric["samples"]) or numeric["samples"] <= 0:
                reasons.append(f"{profile}/{phase}: no validation records")
                continue
            if not all(math.isfinite(numeric[field]) for field in ("model_tile_log_loss", "prior_tile_log_loss", "gain")):
                continue
            if numeric["model_tile_log_loss"] >= numeric["prior_tile_log_loss"]:
                reasons.append(f"{profile}/{phase}: model tile log-loss is not below prior")
        opening = buckets.get("opening")
        endgame = buckets.get("endgame")
        if opening and endgame:
            try:
                opening_samples = float(opening.get("samples", 1.0))
                endgame_samples = float(endgame.get("samples", 1.0))
                opening_gain = float(opening["gain"])
                endgame_gain = float(endgame["gain"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in (opening_samples, endgame_samples, opening_gain, endgame_gain)) and opening_samples > 0 and endgame_samples > 0 and endgame_gain <= opening_gain:
                reasons.append(f"{profile}: endgame gain is not greater than opening gain")
    if source_records is not None or target_validation_records is not None:
        try:
            source = float(source_records)
            target = float(target_validation_records)
        except (TypeError, ValueError):
            source = target = float("nan")
        if not math.isfinite(source) or not math.isfinite(target):
            reasons.append("source/target validation record counts are not finite")
        elif source < target:
            reasons.append(f"validation source records {int(source)} are below target {int(target)}")
    return Acceptance(passed=not reasons, reasons=tuple(reasons))


def render_report(report: dict[str, Any]) -> str:
    """Render deterministically; the report is an auditable gate artifact."""
    acceptance = evaluate_acceptance(
        report["metrics"],
        source_records=report["source"]["records"],
        target_validation_records=report["settings"]["target_validation_records"],
    )
    checkpoint = report["checkpoint"]
    source = report["source"]
    settings = report["settings"]
    lines = [
        "# S4 v5 Belief Bucket Evaluation",
        "",
        "## Provenance",
        "",
        f"- checkpoint: `{checkpoint['path']}`",
        f"- checkpoint SHA256: `{checkpoint['sha256']}`",
        f"- encoder version: `{checkpoint['encoder_version']}`",
        f"- source: `{source['kind']}`",
        f"- validation records: {source['records']}",
        f"- validation game-id range: `{source['game_id_range']}`",
        f"- source games: {source.get('games', 'n/a')}",
        f"- candidate seed range: `{source.get('candidate_seed_range', 'n/a')}`",
        f"- selected validation seeds: {source.get('selected_val_seeds', 'n/a')}",
        f"- split: cloud `cloud_train_s4_50k_cached._split_name(game_id, seed) == val`",
        f"- seed: {settings['seed']}",
        f"- target validation records: {settings['target_validation_records']}",
        "- profiles: perfect, light_noise, midgame, heavy",
        "- wall buckets: opening >40; midgame 20..40; endgame <20",
        "",
        "## Tile log-loss",
        "",
        "| Profile | Wall bucket | Samples | Model | Prior | Gain (prior - model) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in PROFILES:
        for phase in PHASE_BUCKETS:
            values = report["metrics"].get(profile, {}).get(phase)
            if values is None:
                lines.append(f"| {profile} | {phase} | 0 | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {profile} | {phase} | {int(values.get('samples', 0))} | "
                    f"{values['model_tile_log_loss']:.6f} | {values['prior_tile_log_loss']:.6f} | {values['gain']:.6f} |"
                )
    lines.extend(["", "## Acceptance"])
    if acceptance.passed:
        lines.extend(["", "**PASS** - all profile/bucket model losses beat prior and every endgame gain exceeds opening gain."])
    else:
        lines.extend(["", "**EXCEPTION** - do not treat this as an S4 pass; inspect labels/data before proceeding.", ""])
        lines.extend(f"- {reason}" for reason in acceptance.reasons)
    return "\n".join(lines) + "\n"


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wall_count(record: DecisionRecord) -> int:
    try:
        return int(record.state["facts"]["wall_count"]["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"record {record.game_id}/{record.step} has no observed wall_count") from error


def _record_range(records: Sequence[DecisionRecord]) -> str:
    game_ids = sorted({record.game_id for record in records})
    return "n/a" if not game_ids else f"{game_ids[0]}..{game_ids[-1]}"


def _load_checkpoint(path: Path) -> tuple[BeliefNet, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    encoder_version = str(payload.get("encoder_version", "missing"))
    if encoder_version != ENCODER_VERSION:
        raise ValueError(f"checkpoint encoder version {encoder_version!r} does not match required {ENCODER_VERSION!r}")
    model = BeliefNet(BeliefNetConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, encoder_version


def _records_from_package(
    zip_path: Path, *, target_validation_records: int, split_seed: int
) -> list[DecisionRecord]:
    """Read the original package's compressed shards without writing raw data."""
    with zipfile.ZipFile(zip_path) as archive:
        manifests = sorted(name for name in archive.namelist() if name.endswith("manifest.json"))
        if not manifests:
            raise ValueError(f"package has no manifest: {zip_path}")
        manifest_name = next((name for name in manifests if "/data/" in f"/{name}"), manifests[0])
        manifest = json.loads(archive.read(manifest_name))
        if manifest.get("schema") != "s2.v4":
            raise ValueError(f"unsupported package schema: {manifest.get('schema')}")
        records: list[DecisionRecord] = []
        for shard in manifest["shards"]:
            data_file = str(shard["data_file"])
            candidates = [name for name in archive.namelist() if name.endswith("/" + data_file) or name == data_file]
            if len(candidates) != 1:
                raise ValueError(f"cannot uniquely locate shard {data_file} in {zip_path}")
            info = archive.getinfo(candidates[0])
            if info.file_size != int(shard["bytes"]):
                raise ValueError(f"shard size mismatch: {data_file}")
            with archive.open(info, mode="r") as compressed, gzip.GzipFile(fileobj=compressed, mode="rb") as file:
                for line in file:
                    if line.strip():
                        payload = json.loads(line)
                        game_id = payload.get("game_id")
                        if not isinstance(game_id, str):
                            raise ValueError(f"record in {data_file} has no string game_id")
                        if _split_name(game_id, split_seed) != "val":
                            continue
                        records.append(DecisionRecord.from_dict(payload))
                        if len(records) >= target_validation_records:
                            return records
    return records


def _regenerate_records(
    *,
    seed: int,
    games: int,
    progress_every: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    for game_offset in range(games):
        game_seed = seed + game_offset
        _, game_records = run_recorded_selfplay_game(
            game_id=f"belief-bucket-{game_seed}", seed=game_seed, max_steps=1_000
        )
        records.extend(game_records)
        completed = game_offset + 1
        if on_progress is not None and (completed % progress_every == 0 or completed == games):
            on_progress(completed, games)
    return records


def _regenerate_validation_records(
    *,
    seed: int,
    games: int,
    split_seed: int,
    progress_every: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[DecisionRecord]:
    """Regenerate deterministically while retaining only the cloud-val split.

    A record payload is large.  Filtering per game is equivalent to
    ``select_validation_records(_regenerate_records(...))`` because the cloud
    split is game-id based, but avoids keeping the 95% non-validation payload.
    """
    validation_records: list[DecisionRecord] = []
    for game_offset in range(games):
        game_seed = seed + game_offset
        game_id = f"belief-bucket-{game_seed}"
        _, game_records = run_recorded_selfplay_game(game_id=game_id, seed=game_seed, max_steps=1_000)
        if _split_name(game_id, split_seed) == "val":
            validation_records.extend(game_records)
        completed = game_offset + 1
        if on_progress is not None and (completed % progress_every == 0 or completed == games):
            on_progress(completed, games)
    return validation_records


def _candidate_val_seeds(*, start_seed: int, candidate_games: int, split_seed: int) -> tuple[int, ...]:
    """Return cloud-validation game seeds in ascending candidate order."""
    return tuple(
        game_seed
        for game_seed in range(start_seed, start_seed + candidate_games)
        if _split_name(f"belief-bucket-{game_seed}", split_seed) == "val"
    )


def _regenerate_selected_validation_records(
    seeds: Sequence[int],
    *,
    progress_every: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[DecisionRecord]:
    """Generate only preselected validation games, preserving candidate order."""
    validation_records: list[DecisionRecord] = []
    for index, game_seed in enumerate(seeds, 1):
        _, game_records = run_recorded_selfplay_game(
            game_id=f"belief-bucket-{game_seed}", seed=game_seed, max_steps=1_000
        )
        validation_records.extend(game_records)
        if on_progress is not None and (index % progress_every == 0 or index == len(seeds)):
            on_progress(index, len(seeds))
    return validation_records


def _find_local_package(root: Path) -> Path | None:
    packages = sorted(root.rglob("s4_50k_cloud_training_package_*.zip"))
    return packages[0] if packages else None


def build_report(
    *,
    checkpoint: Path,
    seed: int,
    target_validation_records: int,
    regenerate_games: int,
    package: Path | None = None,
) -> dict[str, Any]:
    """Load data, evaluate all required profiles, and return only JSON-safe values."""
    set_torch_seed(seed)
    model, encoder_version = _load_checkpoint(checkpoint)
    if package is not None:
        validation_records = _records_from_package(
            package, target_validation_records=target_validation_records, split_seed=seed
        )
        source_kind = "local_50k_package_shards"
        source_games = "package"
    else:
        candidate_start = seed
        candidate_games = regenerate_games
        selected_seeds = _candidate_val_seeds(
            start_seed=candidate_start, candidate_games=candidate_games, split_seed=seed
        )
        validation_records = _regenerate_selected_validation_records(
            selected_seeds,
            on_progress=lambda completed, total: print(f"[regenerate] {completed}/{total} validation games", flush=True),
        )
        while len(validation_records) < target_validation_records:
            extension_start = candidate_start + candidate_games
            extension_games = max(500, candidate_games // 4)
            extension_seeds = _candidate_val_seeds(
                start_seed=extension_start, candidate_games=extension_games, split_seed=seed
            )
            extension_records = _regenerate_selected_validation_records(
                extension_seeds,
                on_progress=lambda completed, total: print(
                    f"[regenerate extension] {completed}/{total} validation games", flush=True
                ),
            )
            selected_seeds += extension_seeds
            validation_records.extend(extension_records)
            candidate_games += extension_games
        source_kind = "deterministic_regeneration"
        source_games = candidate_games
    if not validation_records:
        raise RuntimeError("no validation records selected; increase deterministic regeneration games")
    # Preserve the deterministic stream order and cap only if source exceeds the requested target.
    validation_records = validation_records[:target_validation_records]
    by_phase: dict[str, list[DecisionRecord]] = {phase: [] for phase in PHASE_BUCKETS}
    for record in validation_records:
        by_phase[phase_bucket_from_wall_count(_wall_count(record))].append(record)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for profile in PROFILES:
        config = DatasetBuildConfig(seed=seed, degradation_profile=profile)
        profile_samples = [build_belief_sample(record, config) for record in validation_records]
        samples_by_phase: dict[str, list[Any]] = {phase: [] for phase in PHASE_BUCKETS}
        for record, sample in zip(validation_records, profile_samples):
            samples_by_phase[phase_bucket_from_wall_count(_wall_count(record))].append(sample)
        profile_metrics: dict[str, dict[str, float]] = {}
        for phase in PHASE_BUCKETS:
            records = by_phase[phase]
            if not records:
                profile_metrics[phase] = {"samples": 0.0, "model_tile_log_loss": float("inf"), "prior_tile_log_loss": float("inf"), "gain": float("nan")}
                continue
            samples = samples_by_phase[phase]
            model_report = evaluate_belief_model(model, samples)
            prior_report = evaluate_prior_belief_samples(records, samples, config)
            profile_metrics[phase] = {
                "samples": float(len(records)),
                "model_tile_log_loss": float(model_report.tile_log_loss),
                "prior_tile_log_loss": float(prior_report.tile_log_loss),
                "gain": float(prior_report.tile_log_loss - model_report.tile_log_loss),
            }
        metrics[profile] = profile_metrics
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": _checkpoint_sha256(checkpoint), "encoder_version": encoder_version},
        "source": {
            "kind": source_kind,
            "records": len(validation_records),
            "game_id_range": _record_range(validation_records),
            "games": source_games,
            "candidate_seed_range": f"{seed}..{seed + source_games - 1}" if package is None else "n/a",
            "selected_val_seeds": len(selected_seeds) if package is None else "n/a",
        },
        "settings": {"seed": seed, "target_validation_records": target_validation_records, "regenerate_games": regenerate_games},
        "metrics": metrics,
    }


def _render_exception_report(error: Exception) -> str:
    return "\n".join((
        "# S4 v5 Belief Bucket Evaluation",
        "",
        "## Acceptance",
        "",
        f"**EXCEPTION** - evaluation failed: {type(error).__name__}: {error}",
        "",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen S4-v5 belief model by wall-count bucket.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-validation-records", type=int, default=DEFAULT_TARGET_VALIDATION_RECORDS)
    parser.add_argument("--regenerate-games", type=int, default=DEFAULT_CANDIDATE_GAMES, help="candidate seed interval size")
    parser.add_argument("--package", type=Path, default=None, help="Optional local s4_50k package ZIP; auto-discovered otherwise.")
    args = parser.parse_args()
    if args.regenerate_games <= 0 or args.target_validation_records <= 0:
        raise SystemExit("regenerate-games and target-validation-records must be positive")
    package = args.package or _find_local_package(Path.cwd())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build_report(
            checkpoint=args.checkpoint,
            seed=args.seed,
            target_validation_records=args.target_validation_records,
            regenerate_games=args.regenerate_games,
            package=package,
        )
        rendered = render_report(report)
        acceptance = evaluate_acceptance(
            report["metrics"],
            source_records=report["source"]["records"],
            target_validation_records=report["settings"]["target_validation_records"],
        )
    except Exception as error:
        rendered = _render_exception_report(error)
        args.report.write_text(rendered, encoding="utf-8")
        print(rendered, end="", file=sys.stderr)
        raise SystemExit(1) from error
    args.report.write_text(rendered, encoding="utf-8")
    if not acceptance.passed:
        print(rendered, end="", file=sys.stderr)
        raise SystemExit(1)
    print(rendered, end="")


if __name__ == "__main__":
    main()
