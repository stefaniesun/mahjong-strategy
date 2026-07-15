from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from engine.actions import Action
from state.protocol import S2ProtocolState


class BasePolicy(ABC):
    @abstractmethod
    def choose_action(self, protocol_state: S2ProtocolState, legal_mask: Sequence[bool]) -> Action:
        raise NotImplementedError

