from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from learning.datasets.dataset_builder import DatasetBuildConfig, build_belief_sample, build_policy_sample
from learning.eval.eval_belief import evaluate_belief_model, evaluate_prior_belief_records
from learning.eval.eval_policy import evaluate_policy_samples
from learning.device import resolve_device
from learning.models.belief_net import BeliefNetConfig
from learning.models.policy_net import PolicyNetConfig
from learning.training.train_belief import TrainBeliefConfig, train_belief_epoch
from learning.training.train_policy import TrainPolicyConfig, save_policy_checkpoint, train_policy

from selfplay.data_recorder import DecisionRecord, load_jsonl, run_recorded_selfplay_game, write_jsonl
from state.action_space import action_space_size
from state.encoder import ENCODER_VERSION
from state.tile_belief import LearnedBelief
from learning.datasets.splits import split_records_by_game




@dataclass(frozen=True)
class CloudTrainingConfig:
    output_dir: Path = Path("cloud_outputs/s4")
    input_jsonl: Path | None = None
    games: int = 1000
    max_steps: int = 1000
    seed: int = 20260709
    degradation_profile: str = "perfect"
    batch_size: int = 128
    hidden_size: int = 128
    residual_blocks: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    sample_limit: int | None = None
    device: str = "auto"
    max_epochs: int = 10
    patience: int = 3
    min_delta: float = 1e-4
    forced_action_weight: float = 0.1
    discard_weight: float = 1.5
    swap_three_weight: float = 2.0
    declare_missing_suit_weight: float = 1.5
    pong_pass_weight: float = 2.0



@dataclass(frozen=True)
class CloudTrainingResult:
    records: int
    samples: int
    data_path: Path
    belief_checkpoint: Path
    policy_checkpoint: Path
    json_report: Path
    markdown_report: Path


def run_cloud_training(config: CloudTrainingConfig) -> CloudTrainingResult:
    output_dir = Path(config.output_dir)
    data_dir = output_dir / "data"
    checkpoint_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    records = _load_or_generate_records(config, data_dir)
    if config.sample_limit is not None:
        records = records[: config.sample_limit]
    if not records:
        raise ValueError("no decision records available for S4 training")

    splits = split_records_by_game(records, seed=config.seed, ratios=(0.8, 0.1, 0.1))
    if any(not splits[name] for name in ("train", "val", "test")):
        raise ValueError("S4 training requires enough games for non-empty train/val/test splits")
    dataset_config = DatasetBuildConfig(seed=config.seed, degradation_profile=config.degradation_profile)
    belief_train_samples = [build_belief_sample(record, dataset_config) for record in splits["train"]]
    belief_val_samples = [build_belief_sample(record, dataset_config) for record in splits["val"]]

    belief_model_config = BeliefNetConfig(
        input_size=len(belief_train_samples[0].encoded.values),
        hidden_size=config.hidden_size,
        residual_blocks=config.residual_blocks,
        dropout=config.dropout,
    )
    belief_train_config = TrainBeliefConfig(
        model=belief_model_config,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        device=device.type,
    )
    belief_model, belief_train_metrics = train_belief_epoch(belief_train_samples, belief_train_config)
    belief_eval = evaluate_belief_model(belief_model, belief_val_samples)
    prior_eval = evaluate_prior_belief_records(splits["val"], dataset_config)

    data_path = config.input_jsonl or data_dir / "s4_decisions.jsonl"
    fingerprint = data_fingerprint(data_path)
    belief_checkpoint = checkpoint_dir / "belief_s4.pt"
    torch.save(
        {
            "model_config": asdict(belief_model.config),
            "encoder_version": ENCODER_VERSION,
            "training_config": asdict(belief_train_config),
            "state_dict": belief_model.state_dict(),
            "metrics": dict(belief_train_metrics),
            "eval": asdict(belief_eval),
            "prior_eval": asdict(prior_eval),
            "data_fingerprint": fingerprint,
            "execution_device": device.type,
        },
        belief_checkpoint,
    )

    learned_belief = LearnedBelief(model=belief_model)
    policy_splits = {
        name: [build_policy_sample(record, dataset_config, belief=learned_belief) for record in split_records]
        for name, split_records in splits.items()
    }
    policy_model_config = PolicyNetConfig(
        input_size=len(policy_splits["train"][0].encoded.values),
        action_size=action_space_size(),
        hidden_size=config.hidden_size,
        residual_blocks=config.residual_blocks,
        dropout=config.dropout,
    )
    policy_train_config = TrainPolicyConfig(
        model=policy_model_config,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        device=device.type,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        forced_action_weight=config.forced_action_weight,
        discard_weight=config.discard_weight,
        swap_three_weight=config.swap_three_weight,
        declare_missing_suit_weight=config.declare_missing_suit_weight,
        pong_pass_weight=config.pong_pass_weight,
    )
    policy_model, policy_train_metrics = train_policy(policy_splits["train"], policy_splits["val"], policy_train_config)
    policy_eval = evaluate_policy_samples(policy_model, policy_splits["test"])
    split_summary = {name: {"records": len(items), "games": len({record.game_id for record in items})} for name, items in splits.items()}
    policy_checkpoint = checkpoint_dir / "policy_s4.pt"
    save_policy_checkpoint(
        policy_checkpoint,
        policy_model,
        policy_train_config,
        policy_train_metrics,
        data_fingerprint=fingerprint,
        belief_metadata={"source": "learned", "checkpoint": belief_checkpoint.name},
        split_summary=split_summary,
    )
    policy_samples = policy_splits["train"] + policy_splits["val"] + policy_splits["test"]


    json_report = report_dir / "s4_training_report.json"
    markdown_report = report_dir / "s4_training_report.md"
    report = {
        "config": _jsonable_config(config),
        "execution": {"device": device.type},
        "data": {
            "records": len(records),
            "samples": len(policy_samples),
            "fingerprint": fingerprint,
            "path": str(data_path),
            "splits": split_summary,
            "split_game_ids": {name: sorted({record.game_id for record in items}) for name, items in splits.items()},
        },

        "checkpoints": {"belief": str(belief_checkpoint), "policy": str(policy_checkpoint)},
        "belief_train_metrics": belief_train_metrics,
        "belief_metrics": asdict(belief_eval),
        "prior_belief_metrics": asdict(prior_eval),
        "policy_metrics": asdict(policy_eval),
        "policy_train_metrics": policy_train_metrics,
    }
    json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    markdown_report.write_text(_markdown_report(report), encoding="utf-8")

    return CloudTrainingResult(
        records=len(records),
        samples=len(policy_samples),
        data_path=data_path,
        belief_checkpoint=belief_checkpoint,
        policy_checkpoint=policy_checkpoint,
        json_report=json_report,
        markdown_report=markdown_report,
    )


def data_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_or_generate_records(config: CloudTrainingConfig, data_dir: Path) -> list[DecisionRecord]:
    if config.input_jsonl is not None:
        return load_jsonl(config.input_jsonl)

    records: list[DecisionRecord] = []
    for index in range(config.games):
        _, game_records = run_recorded_selfplay_game(
            game_id=f"cloud-s4-{config.seed}-{index}",
            seed=config.seed + index,
            max_steps=config.max_steps,
        )
        records.extend(game_records)
    write_jsonl(records, data_dir / "s4_decisions.jsonl")
    return records


def _jsonable_config(config: CloudTrainingConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _markdown_report(report: dict[str, Any]) -> str:
    belief = report["belief_metrics"]
    prior = report["prior_belief_metrics"]
    policy = report["policy_metrics"]
    return "\n".join(
        [
            "# S4 Cloud Training Report",
            "",
            f"- records: {report['data']['records']}",
            f"- execution device: {report['execution']['device']}",
            f"- data fingerprint: `{report['data']['fingerprint']}`",
            f"- belief checkpoint: `{report['checkpoints']['belief']}`",
            f"- policy checkpoint: `{report['checkpoints']['policy']}`",
            "",
            "## Belief",
            "",
            f"- train loss: {report['belief_train_metrics']['loss']:.6f}",
            f"- model tile log-loss: {belief['tile_log_loss']:.6f}",
            f"- prior tile log-loss: {prior['tile_log_loss']:.6f}",
            f"- opponent tenpai ECE: {belief['opponent_tenpai_ece']:.6f}",
            f"- discard danger ECE: {belief['discard_danger_ece']:.6f}",
            "",
            "## Policy",
            "",
            f"- best epoch: {report['policy_train_metrics']['best_epoch']}",
            f"- epochs trained: {report['policy_train_metrics']['epochs_trained']}",
            f"- eval top-1 accuracy: {policy['top1_accuracy']:.6f}",
            f"- forced action rate: {policy['forced_rate']:.6f}",
            f"- non-forced accuracy: {policy['non_forced_accuracy'] if policy['non_forced_accuracy'] is not None else 'n/a'}",

            f"- illegal argmax count: {policy['illegal_argmax_count']}",
            f"- illegal probability mass: {policy['illegal_probability_mass']:.6f}",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S4 belief + policy cloud training smoke or full job.")
    parser.add_argument("--output-dir", type=Path, default=Path("cloud_outputs/s4"))
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--degradation-profile", default="perfect", choices=["perfect", "light_noise", "midgame", "heavy"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--forced-action-weight", type=float, default=0.1)
    parser.add_argument("--discard-weight", type=float, default=1.5)
    parser.add_argument("--swap-three-weight", type=float, default=2.0)
    parser.add_argument("--declare-missing-suit-weight", type=float, default=1.5)
    parser.add_argument("--pong-pass-weight", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_cloud_training(
        CloudTrainingConfig(
            output_dir=args.output_dir,
            input_jsonl=args.input_jsonl,
            games=args.games,
            max_steps=args.max_steps,
            seed=args.seed,
            degradation_profile=args.degradation_profile,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            sample_limit=args.sample_limit,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            forced_action_weight=args.forced_action_weight,
            discard_weight=args.discard_weight,
            swap_three_weight=args.swap_three_weight,
            declare_missing_suit_weight=args.declare_missing_suit_weight,
            pong_pass_weight=args.pong_pass_weight,

        )
    )
    print(json.dumps({key: str(value) for key, value in asdict(result).items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
