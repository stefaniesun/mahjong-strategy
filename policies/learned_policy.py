from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from engine.actions import Action
from learning.models.policy_net import PolicyNet, PolicyNetConfig
from policies.base_policy import BasePolicy
from policies.protocol_actions import action_from_protocol, validate_legal_mask
from state.action_space import action_space_size, index_to_action
from state.encoder import ENCODER_VERSION, encode_state
from state.protocol import S2ProtocolState
from state.tile_belief import LearnedBelief, TileBelief, with_prior_beliefs



class LearnedPolicy(BasePolicy):
    def __init__(self, model_path: str | Path, belief_model_path: str | Path | None = None) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")

        checkpoint_version = checkpoint.get("encoder_version")
        if checkpoint_version != ENCODER_VERSION:
            raise ValueError(
                f"checkpoint encoder version {checkpoint_version!r} is incompatible with {ENCODER_VERSION!r}"
            )
        config_data = checkpoint.get("model_config")
        if config_data is None:
            raise ValueError("checkpoint must include model_config")
        config = PolicyNetConfig(**config_data)
        if config.action_size != action_space_size():
            raise ValueError(
                f"checkpoint action size {config.action_size} is incompatible with {action_space_size()}"
            )
        self.model = PolicyNet(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        belief_metadata = checkpoint.get("belief_metadata") or {}
        source = belief_metadata.get("source", "legacy")
        if source == "learned" and belief_model_path is None:
            raise ValueError("belief_model_path is required for a learned-belief policy checkpoint")
        self.belief: TileBelief | None = LearnedBelief(model_path=str(belief_model_path)) if belief_model_path is not None else None

    def choose_action(self, protocol_state: S2ProtocolState, legal_mask: Sequence[bool]) -> Action:
        validate_legal_mask(legal_mask)
        policy_state = with_prior_beliefs(protocol_state, self.belief) if self.belief is not None else protocol_state
        encoded = encode_state(policy_state)

        if encoded.size != self.model.config.input_size:
            raise ValueError(
                f"encoded state size {encoded.size} is incompatible with {self.model.config.input_size}"
            )
        features = torch.tensor([encoded.values], dtype=torch.float32)
        mask = torch.tensor([legal_mask], dtype=torch.bool)
        with torch.no_grad():
            action_index = int(self.model(features, legal_mask=mask).logits.argmax(dim=-1).item())
        return action_from_protocol(index_to_action(action_index))
