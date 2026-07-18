"""Cached-pipeline variant of cloud_train_s4_50k.py.

与原版逐位等价的加速版:
- 每个 epoch 先用多进程按分片并行完成「解析 JSON -> 退化 -> 构建样本 -> 编码」,
  以 npz 数组落盘;训练循环按分片顺序流式读取缓存,行级切片成 batch。
- 退化采样按 (seed, epoch, record) 稳定哈希确定,与并行顺序无关,数值与原
  流式实现完全一致;batch 构造为逐行堆叠,切片等价。
- 策略阶段的 inference_batch_size 分组与组内洗牌逻辑逐位复刻(game_id 随行携带)。
- 原流式路径保留为 --pipeline stream,供对拍与回退。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from learning.datasets.dataset_builder import (
    BeliefSample,
    DatasetBuildConfig,
    PolicySample,
    _clean_action,
    _degrade,
    _mask_from_actions,
    _record_state,
    build_belief_sample,
)
from learning.eval.eval_belief import evaluate_belief_model, evaluate_prior_belief_records
from learning.eval.eval_policy import evaluate_policy_samples
from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed
from learning.models.policy_net import PolicyNet, PolicyNetConfig
from learning.training.train_belief import BeliefBatch, belief_batch_from_samples
from learning.training.train_policy import TrainPolicyConfig, policy_batch_from_samples
from selfplay.data_recorder import DecisionRecord
from state.action_space import action_space_size, action_to_index
from state.encoder import ENCODER_VERSION, encode_state
from state.tile_belief import LearnedBelief


PROFILES = ("perfect", "light_noise", "midgame", "heavy")
PROFILE_WEIGHTS = (0.10, 0.30, 0.35, 0.25)


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    manifest: Path
    output_dir: Path
    seed: int = 20260716
    batch_size: int = 512
    inference_batch_size: int = 1024
    hidden_size: int = 256
    residual_blocks: int = 3
    dropout: float = 0.05
    learning_rate: float = 3e-4
    belief_epochs: int = 3
    policy_epochs: int = 5
    eval_samples: int = 20000
    device: str = "auto"
    workers: int = 0  # 0 = auto (cpu_count - 2)
    pipeline: str = "cached"  # cached | stream
    cache_dir: Path | None = None
    extra_data_dirs: tuple[str, ...] = ()  # 附加数据集根目录(含 shards/ 与 manifest.json),如 DAgger


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _stable_int(*values: object) -> int:
    text = "|".join(str(value) for value in values)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _split_name(game_id: str, seed: int) -> str:
    bucket = _stable_int("split", seed, game_id) % 100
    return "train" if bucket < 90 else "val" if bucket < 95 else "test"


def _profile_config(record: DecisionRecord, seed: int, epoch: int) -> DatasetBuildConfig:
    rng = random.Random(_stable_int("profile", seed, epoch, record.game_id, record.step, record.player))
    profile = rng.choices(PROFILES, weights=PROFILE_WEIGHTS, k=1)[0]
    degradation_seed = _stable_int("degradation", seed, epoch, record.game_id) % (2**31)
    return DatasetBuildConfig(seed=degradation_seed, degradation_profile=profile)


def _load_one_manifest(data_dir: Path, manifest_path: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("games", 0) <= 0:
        raise ValueError(f"manifest has no games: {manifest_path}")
    if manifest.get("schema") != "s2.v4":
        raise ValueError(f"unsupported dataset schema: {manifest.get('schema')}")
    paths: list[Path] = []
    for shard in manifest.get("shards", []):
        path = data_dir / shard["data_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"shard size mismatch: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"manifest has no shards: {manifest_path}")
    return paths, manifest


def _load_manifest(config: TrainingConfig) -> tuple[list[Path], dict[str, Any]]:
    """主数据集 + 任意个附加数据集(如 DAgger)按声明顺序拼接为统一训练流。"""
    paths, manifest = _load_one_manifest(config.data_dir, config.manifest)
    extra_manifests: list[dict[str, Any]] = []
    for extra_dir in config.extra_data_dirs:
        extra_dir = Path(extra_dir)
        extra_paths, extra_manifest = _load_one_manifest(extra_dir / "shards", extra_dir / "manifest.json")
        paths.extend(extra_paths)
        extra_manifests.append(extra_manifest)
    if extra_manifests:
        merged = dict(manifest)
        merged["games"] = manifest["games"] + sum(m["games"] for m in extra_manifests)
        merged["decision_records"] = manifest["decision_records"] + sum(m["decision_records"] for m in extra_manifests)
        merged["compressed_bytes"] = manifest["compressed_bytes"] + sum(m["compressed_bytes"] for m in extra_manifests)
        merged["dataset_fingerprint"] = hashlib.sha256(
            "|".join([manifest["dataset_fingerprint"]] + [m["dataset_fingerprint"] for m in extra_manifests]).encode()
        ).hexdigest()
        merged["extra_datasets"] = [
            {key: m.get(key) for key in ("kind", "dagger_iteration", "games", "decision_records", "dataset_fingerprint")}
            for m in extra_manifests
        ]
        manifest = merged
    return paths, manifest


def _iter_records(paths: Sequence[Path]) -> Iterator[DecisionRecord]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield DecisionRecord.from_dict(json.loads(line))


def _iter_split(paths: Sequence[Path], split: str, seed: int) -> Iterator[DecisionRecord]:
    for record in _iter_records(paths):
        if _split_name(record.game_id, seed) == split:
            yield record


def _batched(items: Iterator[DecisionRecord], size: int) -> Iterator[list[DecisionRecord]]:
    batch: list[DecisionRecord] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _belief_samples(records: Sequence[DecisionRecord], seed: int, epoch: int) -> list[BeliefSample]:
    return [build_belief_sample(record, _profile_config(record, seed, epoch)) for record in records]


def _policy_samples(
    records: Sequence[DecisionRecord], belief: LearnedBelief, seed: int, epoch: int
) -> list[PolicySample]:
    configs = [_profile_config(record, seed, epoch) for record in records]
    states = [_degrade(_record_state(record), config, record) for record, config in zip(records, configs)]
    predictions = belief.infer_batch(states)
    samples: list[PolicySample] = []
    for record, config, state, beliefs in zip(records, configs, states, predictions):
        encoded = encode_state(replace(state, beliefs=beliefs))
        action = _clean_action(record.action)
        legal_actions = [_clean_action(item) for item in record.legal_actions]
        mask = _mask_from_actions(record.legal_actions)
        action_index = action_to_index(action)
        if not mask[action_index]:
            raise ValueError("record action is not included in legal actions")
        legal_kinds = {item["kind"] for item in legal_actions}
        samples.append(
            PolicySample(
                game_id=record.game_id,
                step=record.step,
                player=record.player,
                phase=record.phase,
                encoded=encoded,
                action_index=action_index,
                legal_mask=mask,
                degradation_profile=config.degradation_profile,
                action_kind=str(action["kind"]),
                legal_action_count=sum(mask),
                is_pong_pass_decision={"pong", "pass"}.issubset(legal_kinds),
            )
        )
    return samples


def _sample_weight(sample_kind: str, legal_action_count: int, is_pong_pass: bool) -> float:
    weight = 0.1 if legal_action_count == 1 else 1.0
    weight *= {"discard": 1.5, "swap_three": 2.0, "declare_void": 1.5}.get(sample_kind, 1.0)
    if is_pong_pass:
        weight *= 2.0
    return weight


# ---------------------------------------------------------------------------
# 并行缓存管线
# ---------------------------------------------------------------------------

_WORKER_BELIEF: LearnedBelief | None = None
_WORKER_CHUNK = 8192  # worker 内分块处理,控制峰值内存


def _worker_init(belief_payload_path: str | None) -> None:
    torch.set_num_threads(1)
    global _WORKER_BELIEF
    if belief_payload_path:
        payload = torch.load(belief_payload_path, map_location="cpu")
        model = BeliefNet(BeliefNetConfig(**payload["model_config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        _WORKER_BELIEF = LearnedBelief(model=model)


def _shard_train_records(path: Path, seed: int) -> Iterator[DecisionRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = DecisionRecord.from_dict(json.loads(line))
                if _split_name(record.game_id, seed) == "train":
                    yield record


def _encode_shard_belief(args: tuple[str, int, int, str]) -> tuple[str, int]:
    shard_path, seed, epoch, cache_file = args
    chunks: list[dict[str, np.ndarray]] = []
    buffer: list[DecisionRecord] = []

    def flush() -> None:
        if not buffer:
            return
        batch = belief_batch_from_samples(_belief_samples(buffer, seed, epoch))
        chunks.append(
            {
                "features": batch.features.numpy(),
                "tile_targets": batch.tile_location_targets.numpy(),
                "tile_mask": batch.tile_location_mask.numpy(),
                "tenpai_targets": batch.opponent_tenpai_targets.numpy(),
                "tenpai_mask": batch.opponent_tenpai_mask.numpy(),
                "danger_targets": batch.discard_danger_targets.numpy(),
                "danger_mask": batch.discard_danger_mask.numpy(),
            }
        )
        buffer.clear()

    for record in _shard_train_records(Path(shard_path), seed):
        buffer.append(record)
        if len(buffer) >= _WORKER_CHUNK:
            flush()
    flush()
    if chunks:
        merged = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    else:
        merged = {}
    np.savez(cache_file, **merged)
    return cache_file, int(merged["features"].shape[0]) if merged else 0


def _encode_shard_policy(args: tuple[str, int, int, str]) -> tuple[str, int]:
    shard_path, seed, epoch, cache_file = args
    assert _WORKER_BELIEF is not None, "policy worker requires belief model"
    chunks: list[dict[str, np.ndarray]] = []
    buffer: list[DecisionRecord] = []

    def flush() -> None:
        if not buffer:
            return
        samples = _policy_samples(buffer, _WORKER_BELIEF, seed, epoch)
        chunks.append(
            {
                "features": np.asarray([s.encoded.values for s in samples], dtype=np.float32),
                "action_targets": np.asarray([s.action_index for s in samples], dtype=np.int64),
                "legal_mask": np.asarray([s.legal_mask for s in samples], dtype=np.bool_),
                "weights": np.asarray(
                    [_sample_weight(s.action_kind, s.legal_action_count, s.is_pong_pass_decision) for s in samples],
                    dtype=np.float32,
                ),
                "game_ids": np.asarray([s.game_id for s in samples], dtype=np.str_),
            }
        )
        buffer.clear()

    for record in _shard_train_records(Path(shard_path), seed):
        buffer.append(record)
        if len(buffer) >= _WORKER_CHUNK:
            flush()
    flush()
    if chunks:
        merged = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    else:
        merged = {}
    np.savez(cache_file, **merged)
    return cache_file, int(merged["features"].shape[0]) if merged else 0


def _build_epoch_cache(
    paths: Sequence[Path],
    seed: int,
    epoch: int,
    workers: int,
    cache_dir: Path,
    kind: str,
    belief_payload_path: str | None = None,
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), seed, epoch, str(cache_dir / f"{kind}_e{epoch}_{index:03d}.npz"))
        for index, path in enumerate(paths)
    ]
    worker_fn = _encode_shard_belief if kind == "belief" else _encode_shard_policy
    started = time.perf_counter()
    with mp.Pool(processes=workers, initializer=_worker_init, initargs=(belief_payload_path,)) as pool:
        results = list(pool.imap(worker_fn, tasks, chunksize=1))
    total = sum(count for _, count in results)
    print(
        f"[cache] {kind} epoch {epoch}: {len(results)} shards, {total} samples, "
        f"{time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return [Path(file) for file, _ in results]


def _delete_cache(files: Sequence[Path]) -> None:
    for file in files:
        try:
            file.unlink()
        except FileNotFoundError:
            pass


def _iter_cached_rows(files: Sequence[Path], keys: Sequence[str]) -> Iterator[dict[str, np.ndarray]]:
    """按分片顺序流式产出行级数据(每次一个分片的数组字典)。"""
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            if "features" not in data:
                continue
            yield {key: data[key] for key in keys}


def _belief_batches_from_cache(files: Sequence[Path], batch_size: int) -> Iterator[BeliefBatch]:
    keys = ("features", "tile_targets", "tile_mask", "tenpai_targets", "tenpai_mask", "danger_targets", "danger_mask")
    pending: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    pending_rows = 0

    def emit(arrays: dict[str, np.ndarray], start: int, end: int) -> BeliefBatch:
        return BeliefBatch(
            features=torch.from_numpy(arrays["features"][start:end]),
            tile_location_targets=torch.from_numpy(arrays["tile_targets"][start:end]),
            tile_location_mask=torch.from_numpy(arrays["tile_mask"][start:end]),
            opponent_tenpai_targets=torch.from_numpy(arrays["tenpai_targets"][start:end]),
            opponent_tenpai_mask=torch.from_numpy(arrays["tenpai_mask"][start:end]),
            discard_danger_targets=torch.from_numpy(arrays["danger_targets"][start:end]),
            discard_danger_mask=torch.from_numpy(arrays["danger_mask"][start:end]),
        )

    for shard in _iter_cached_rows(files, keys):
        for key in keys:
            pending[key].append(shard[key])
        pending_rows += shard["features"].shape[0]
        while pending_rows >= batch_size:
            merged = {key: np.concatenate(pending[key]) if len(pending[key]) > 1 else pending[key][0] for key in keys}
            yield emit(merged, 0, batch_size)
            rest = {key: merged[key][batch_size:] for key in keys}
            pending = {key: [rest[key]] if rest[key].shape[0] else [] for key in keys}
            pending_rows -= batch_size
    if pending_rows:
        merged = {key: np.concatenate(pending[key]) if len(pending[key]) > 1 else pending[key][0] for key in keys}
        yield emit(merged, 0, pending_rows)


def _policy_groups_from_cache(
    files: Sequence[Path], group_size: int
) -> Iterator[dict[str, np.ndarray]]:
    """复刻原实现:全局训练流按 inference_batch_size 分组,组内数据随行携带 game_id。"""
    keys = ("features", "action_targets", "legal_mask", "weights", "game_ids")
    pending: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    pending_rows = 0
    for shard in _iter_cached_rows(files, keys):
        for key in keys:
            pending[key].append(shard[key])
        pending_rows += shard["features"].shape[0]
        while pending_rows >= group_size:
            merged = {key: np.concatenate(pending[key]) if len(pending[key]) > 1 else pending[key][0] for key in keys}
            yield {key: merged[key][:group_size] for key in keys}
            rest = {key: merged[key][group_size:] for key in keys}
            pending = {key: [rest[key]] if rest[key].shape[0] else [] for key in keys}
            pending_rows -= group_size
    if pending_rows:
        merged = {key: np.concatenate(pending[key]) if len(pending[key]) > 1 else pending[key][0] for key in keys}
        yield merged


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------


def _train_belief(
    paths: Sequence[Path], config: TrainingConfig, device: torch.device, checkpoint_dir: Path, cache_dir: Path
) -> tuple[BeliefNet, list[dict[str, Any]]]:
    first_record = next(_iter_split(paths, "train", config.seed))
    input_size = len(_belief_samples([first_record], config.seed, 0)[0].encoded.values)
    model_config = BeliefNetConfig(input_size, config.hidden_size, config.residual_blocks, config.dropout)
    model = BeliefNet(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    for epoch in range(config.belief_epochs):
        if config.pipeline == "cached":
            cache_files = _build_epoch_cache(paths, config.seed, epoch, config.workers, cache_dir, "belief")
            batch_iter: Iterator[BeliefBatch] = _belief_batches_from_cache(cache_files, config.batch_size)
        else:
            cache_files = []
            batch_iter = (
                belief_batch_from_samples(_belief_samples(records, config.seed, epoch))
                for records in _batched(_iter_split(paths, "train", config.seed), config.batch_size)
            )
        model.train()
        totals = {"loss": 0.0, "tile_location_loss": 0.0, "opponent_tenpai_loss": 0.0, "discard_danger_loss": 0.0}
        samples_seen = 0
        started = time.perf_counter()
        for batch in batch_iter:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.features)
            tile_per = -(batch.tile_location_targets * torch.log_softmax(output.tile_location_logits, dim=-1)).sum(-1)
            tile_loss = tile_per[batch.tile_location_mask].mean()
            tenpai_loss = F.binary_cross_entropy_with_logits(
                output.opponent_tenpai_logits[batch.opponent_tenpai_mask],
                batch.opponent_tenpai_targets[batch.opponent_tenpai_mask],
            )
            danger_loss = F.binary_cross_entropy_with_logits(
                output.discard_danger_logits[batch.discard_danger_mask],
                batch.discard_danger_targets[batch.discard_danger_mask],
            )
            loss = tile_loss + tenpai_loss + danger_loss
            loss.backward()
            optimizer.step()
            count = int(batch.features.shape[0])
            samples_seen += count
            for key, value in (("loss", loss), ("tile_location_loss", tile_loss), ("opponent_tenpai_loss", tenpai_loss), ("discard_danger_loss", danger_loss)):
                totals[key] += float(value.detach().cpu()) * count
            if samples_seen % (config.batch_size * 100) < config.batch_size:
                print(f"belief epoch {epoch + 1}/{config.belief_epochs}: {samples_seen} samples", flush=True)
        _delete_cache(cache_files)
        metrics = {key: value / samples_seen for key, value in totals.items()}
        metrics.update(epoch=epoch + 1, samples=samples_seen, elapsed_seconds=round(time.perf_counter() - started, 2))
        history.append(metrics)
        torch.save(
            {
                "model_config": asdict(model.config), "encoder_version": ENCODER_VERSION,
                "state_dict": model.state_dict(), "training_config": _config_dict(config),
                "metrics": metrics, "history": history,
            }, checkpoint_dir / "belief_s4_latest.pt"
        )
    return model, history


def _train_policy(
    paths: Sequence[Path], config: TrainingConfig, device: torch.device, belief_model: BeliefNet,
    checkpoint_dir: Path, cache_dir: Path,
) -> tuple[PolicyNet, list[dict[str, Any]]]:
    belief = LearnedBelief(model=belief_model)
    first_records = [next(_iter_split(paths, "train", config.seed))]
    first_sample = _policy_samples(first_records, belief, config.seed, 0)[0]
    model_config = PolicyNetConfig(len(first_sample.encoded.values), action_space_size(), config.hidden_size, config.residual_blocks, config.dropout)
    model = PolicyNet(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []

    belief_payload_path = None
    if config.pipeline == "cached":
        belief_payload_path = str(cache_dir / "belief_for_workers.pt")
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_config": asdict(belief_model.config), "state_dict": {k: v.cpu() for k, v in belief_model.state_dict().items()}},
            belief_payload_path,
        )

    for epoch in range(config.policy_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        samples_seen = 0
        started = time.perf_counter()

        if config.pipeline == "cached":
            cache_files = _build_epoch_cache(
                paths, config.seed, epoch, config.workers, cache_dir, "policy", belief_payload_path
            )
            group_iter = _policy_groups_from_cache(cache_files, config.inference_batch_size)
        else:
            cache_files = []

            def _stream_groups() -> Iterator[dict[str, np.ndarray]]:
                for records in _batched(_iter_split(paths, "train", config.seed), config.inference_batch_size):
                    samples = _policy_samples(records, belief, config.seed, epoch)
                    yield {
                        "features": np.asarray([s.encoded.values for s in samples], dtype=np.float32),
                        "action_targets": np.asarray([s.action_index for s in samples], dtype=np.int64),
                        "legal_mask": np.asarray([s.legal_mask for s in samples], dtype=np.bool_),
                        "weights": np.asarray(
                            [_sample_weight(s.action_kind, s.legal_action_count, s.is_pong_pass_decision) for s in samples],
                            dtype=np.float32,
                        ),
                        "game_ids": np.asarray([s.game_id for s in samples], dtype=np.str_),
                    }

            group_iter = _stream_groups()

        for group in group_iter:
            size = int(group["features"].shape[0])
            order = list(range(size))
            random.Random(_stable_int(config.seed, epoch, str(group["game_ids"][0]))).shuffle(order)
            order_index = np.asarray(order, dtype=np.int64)
            features = torch.from_numpy(group["features"][order_index])
            targets = torch.from_numpy(group["action_targets"][order_index])
            legal = torch.from_numpy(group["legal_mask"][order_index])
            weights_all = torch.from_numpy(group["weights"][order_index])
            for start in range(0, size, config.batch_size):
                end = min(start + config.batch_size, size)
                feats = features[start:end].to(device)
                tgt = targets[start:end].to(device)
                msk = legal[start:end].to(device)
                weights = weights_all[start:end].to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(feats, legal_mask=msk)
                per_sample = F.cross_entropy(output.logits, tgt, reduction="none")
                loss = (per_sample * weights).sum() / weights.sum()
                loss.backward()
                optimizer.step()
                count = end - start
                samples_seen += count
                total_loss += float(loss.detach().cpu()) * count
                total_correct += int((output.logits.argmax(-1) == tgt).sum().detach().cpu())
            if samples_seen % (config.inference_batch_size * 50) < config.inference_batch_size:
                print(f"policy epoch {epoch + 1}/{config.policy_epochs}: {samples_seen} samples", flush=True)
        _delete_cache(cache_files)
        metrics = {
            "epoch": epoch + 1, "samples": samples_seen, "loss": total_loss / samples_seen,
            "top1_accuracy": total_correct / samples_seen,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        history.append(metrics)
        torch.save(
            {
                "model_config": asdict(model.config), "encoder_version": ENCODER_VERSION,
                "state_dict": model.state_dict(), "training_config": _config_dict(config),
                "metrics": metrics, "history": history, "belief_metadata": {"source": "learned", "checkpoint": "belief_s4.pt"},
            }, checkpoint_dir / "policy_s4_latest.pt"
        )
    return model, history


def _collect_records(paths: Sequence[Path], split: str, seed: int, limit: int) -> list[DecisionRecord]:
    result: list[DecisionRecord] = []
    for record in _iter_split(paths, split, seed):
        result.append(record)
        if len(result) >= limit:
            break
    return result


def _config_dict(config: TrainingConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def run(config: TrainingConfig) -> dict[str, Any]:
    if config.workers <= 0:
        config = replace(config, workers=max(1, (os.cpu_count() or 2) - 2))
    cache_dir = config.cache_dir or (config.output_dir / "cache")
    set_torch_seed(config.seed)
    device = _resolve_device(config.device)
    paths, manifest = _load_manifest(config)
    checkpoint_dir = config.output_dir / "checkpoints"
    report_dir = config.output_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    belief_model, belief_history = _train_belief(paths, config, device, checkpoint_dir, cache_dir)
    validation_records = _collect_records(paths, "val", config.seed, config.eval_samples)
    validation_samples = _belief_samples(validation_records, config.seed, config.belief_epochs)
    belief_eval = evaluate_belief_model(belief_model, validation_samples)
    prior_eval = evaluate_prior_belief_records(validation_records, DatasetBuildConfig(seed=config.seed, degradation_profile="light_noise"))
    belief_checkpoint = checkpoint_dir / "belief_s4.pt"
    torch.save(
        {
            "model_config": asdict(belief_model.config), "encoder_version": ENCODER_VERSION,
            "state_dict": belief_model.state_dict(), "training_config": _config_dict(config),
            "history": belief_history, "eval": asdict(belief_eval), "prior_eval": asdict(prior_eval),
            "data_fingerprint": manifest["dataset_fingerprint"], "execution_device": device.type,
        }, belief_checkpoint
    )

    policy_model, policy_history = _train_policy(paths, config, device, belief_model, checkpoint_dir, cache_dir)
    test_records = _collect_records(paths, "test", config.seed, config.eval_samples)
    policy_samples = _policy_samples(test_records, LearnedBelief(model=belief_model), config.seed, config.policy_epochs)
    policy_eval = evaluate_policy_samples(policy_model, policy_samples)
    policy_checkpoint = checkpoint_dir / "policy_s4.pt"
    torch.save(
        {
            "model_config": asdict(policy_model.config), "encoder_version": ENCODER_VERSION,
            "state_dict": policy_model.state_dict(), "training_config": _config_dict(config),
            "history": policy_history, "eval": asdict(policy_eval),
            "belief_metadata": {"source": "learned", "checkpoint": belief_checkpoint.name},
            "data_fingerprint": manifest["dataset_fingerprint"], "execution_device": device.type,
        }, policy_checkpoint
    )
    shutil.rmtree(cache_dir, ignore_errors=True)
    report = {
        "config": _config_dict(config),
        "data": {key: manifest[key] for key in ("games", "decision_records", "compressed_bytes", "dataset_fingerprint")},
        "execution": {"device": device.type, "torch_version": torch.__version__, "pipeline": config.pipeline, "workers": config.workers},
        "profile_mix": dict(zip(PROFILES, PROFILE_WEIGHTS)),
        "belief_history": belief_history, "belief_metrics": asdict(belief_eval), "prior_belief_metrics": asdict(prior_eval),
        "policy_history": policy_history, "policy_metrics": asdict(policy_eval),
        "checkpoints": {"belief": str(belief_checkpoint), "policy": str(policy_checkpoint)},
    }
    (report_dir / "s4_training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (report_dir / "s4_training_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    belief = report["belief_metrics"]
    prior = report["prior_belief_metrics"]
    policy = report["policy_metrics"]
    return "\n".join([
        "# S4 50K Training Report", "", f"- games: {report['data']['games']}",
        f"- decision records: {report['data']['decision_records']}", f"- device: {report['execution']['device']}",
        f"- pipeline: {report['execution']['pipeline']} (workers={report['execution']['workers']})",
        f"- data fingerprint: `{report['data']['dataset_fingerprint']}`", "", "## Belief", "",
        f"- model tile log-loss: {belief['tile_log_loss']:.6f}", f"- prior tile log-loss: {prior['tile_log_loss']:.6f}",
        f"- opponent tenpai ECE: {belief['opponent_tenpai_ece']:.6f}", f"- discard danger ECE: {belief['discard_danger_ece']:.6f}",
        "", "## Policy", "", f"- top-1 accuracy: {policy['top1_accuracy']:.6f}",
        f"- non-forced accuracy: {policy['non_forced_accuracy']}", f"- illegal argmax count: {policy['illegal_argmax_count']}",
        f"- illegal probability mass: {policy['illegal_probability_mass']:.8f}", "",
    ])


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Cached-pipeline S4 training on the verified 50K dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cloud_outputs/s4_50k"))
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--belief-epochs", type=int, default=3)
    parser.add_argument("--policy-epochs", type=int, default=5)
    parser.add_argument("--eval-samples", type=int, default=20000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pipeline", choices=("cached", "stream"), default="cached")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--extra-data-dir", action="append", default=[], dest="extra_data_dirs",
        help="附加数据集根目录(含 shards/ 与 manifest.json),可多次指定,如 DAgger 数据",
    )
    args = vars(parser.parse_args())
    args["extra_data_dirs"] = tuple(args["extra_data_dirs"])
    return TrainingConfig(**args)


if __name__ == "__main__":
    print(json.dumps(run(_parse_args())["data"], ensure_ascii=False, indent=2))
