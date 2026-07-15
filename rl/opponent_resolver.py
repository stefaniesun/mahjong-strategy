"""Resolve portable S5 league entries into isolated read-only CPU policies."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol


import torch
from torch import nn

from engine.actions import Action
from policies.base_policy import BasePolicy
from policies.opponent_pool import GreedyPolicy, RandomPolicy
from policies.protocol_actions import action_from_protocol, validate_legal_mask
from policies.rule_policy import RulePolicy
from rl.curriculum import DegradationProfile
from rl.league import OpponentEntry, OpponentKind
from state.action_space import index_to_action
from state.encoder import encode_state
from state.observation_degradation import DegradationPipeline, MidGameSnapshot, VisionNoise
from state.protocol import S2ProtocolState


class BeliefProvider(Protocol):
    def apply(self, state: S2ProtocolState) -> S2ProtocolState: ...


ModelFactory = Callable[[], nn.Module]
CheckpointLoader = Callable[[bytes], object]


def _safe_checkpoint_loader(contents: bytes) -> object:
    return torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)



class ModelPolicy(BasePolicy):
    """Perfect-S2 policy backed by a permanently frozen CPU policy/value model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        belief_provider: BeliefProvider,
        degradation: DegradationProfile | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if degradation is not None and not isinstance(degradation, DegradationProfile):
            raise TypeError("degradation must be DegradationProfile or None")
        if not callable(getattr(belief_provider, "apply", None)):
            raise TypeError("belief_provider must expose apply(state)")
        self.model = model.cpu()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.belief_provider = belief_provider
        self.degradation = degradation
        self.seed = seed

    def choose_action(
        self,
        protocol_state: S2ProtocolState,
        legal_mask: Sequence[bool],
    ) -> Action:
        self.model.cpu()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        validate_legal_mask(legal_mask)

        state = protocol_state
        if self.degradation is not None:
            operators = [
                VisionNoise(
                    miss_rate=self.degradation.vision_miss_rate,
                    seed=self.seed,
                )
            ]
            if self.degradation.mid_game_ratio > 0.0:
                operators.insert(
                    0,
                    MidGameSnapshot(int(108 * self.degradation.mid_game_ratio)),
                )
            state = DegradationPipeline(operators).apply(state)
        state = self.belief_provider.apply(state)
        encoded = encode_state(state)
        with torch.inference_mode():
            output = self.model(
                torch.tensor([encoded.values], dtype=torch.float32, device="cpu"),
                torch.tensor([legal_mask], dtype=torch.bool, device="cpu"),
            )
            action_index = int(output.action_logits.argmax(dim=-1).item())
        return action_from_protocol(index_to_action(action_index))


class OpponentResolver:
    """Construct fixed baselines or load immutable model opponents from metadata."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        belief_provider: BeliefProvider,
        checkpoint_loader: CheckpointLoader = _safe_checkpoint_loader,
    ) -> None:
        if not callable(model_factory):
            raise TypeError("model_factory must be callable")
        if not callable(checkpoint_loader):
            raise TypeError("checkpoint_loader must be callable")
        if not callable(getattr(belief_provider, "apply", None)):
            raise TypeError("belief_provider must expose apply(state)")
        self._model_factory = model_factory
        self._belief_provider = belief_provider
        self._checkpoint_loader = checkpoint_loader

    def resolve(self, entry: OpponentEntry, *, seed: int) -> BasePolicy:
        if not isinstance(entry, OpponentEntry):
            raise TypeError("entry must be OpponentEntry")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if entry.kind is OpponentKind.S3:
            return RulePolicy()
        if entry.kind is OpponentKind.GREEDY:
            return GreedyPolicy()
        if entry.kind is OpponentKind.RANDOM:
            return RandomPolicy(seed=seed)
        if entry.kind not in (OpponentKind.CURRENT, OpponentKind.SNAPSHOT):
            raise ValueError(f"unsupported league opponent kind: {entry.kind}")

        snapshot = entry.snapshot
        if snapshot is None:
            raise ValueError("model league entry is missing snapshot metadata")
        checkpoint_path = Path(snapshot.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"league snapshot is unavailable: {checkpoint_path}")
        if snapshot.checksum is None:
            raise ValueError("league snapshot checksum is required")
        checkpoint_bytes = checkpoint_path.read_bytes()
        if hashlib.sha256(checkpoint_bytes).hexdigest() != snapshot.checksum:
            raise ValueError(f"league snapshot checksum mismatch: {checkpoint_path}")

        try:
            payload = self._checkpoint_loader(checkpoint_bytes)

        except Exception as exc:
            if isinstance(exc, (TypeError, ValueError)):
                raise
            raise ValueError("league snapshot is not a safe checkpoint") from exc
        state = (
            payload.get("model_state_dict", payload.get("state_dict"))
            if isinstance(payload, Mapping)
            else None
        )
        if not isinstance(state, Mapping):
            raise ValueError("league snapshot has no safe model state")
        model = self._model_factory()
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a torch module")
        model = model.cpu()
        try:
            if "model_state_dict" in payload:
                model.load_state_dict(state)
            elif callable(getattr(model, "load_s4_policy_state_dict", None)):
                model.load_s4_policy_state_dict(state)
            else:
                model.load_state_dict(state)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("league snapshot model state is incompatible") from exc

        return ModelPolicy(
            model,
            belief_provider=self._belief_provider,
            seed=seed,
        )
