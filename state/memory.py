from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


@dataclass(frozen=True)
class ObservationMemory:
    observation_start: int
    events: list[dict[str, Any]]
    exchange_tracking: ObservedValue[dict[str, Any]]

    @classmethod
    def from_state(cls, state: S2ProtocolState) -> "ObservationMemory":
        observation_start = state.observation_start.value if state.observation_start.status is not ObservationStatus.UNKNOWN else 0
        events = []
        if state.facts.event_history.status is not ObservationStatus.UNKNOWN:
            for event in state.facts.event_history.value or []:
                if _event_seq(event) >= observation_start:
                    events.append(dict(event))
        return cls(
            observation_start=int(observation_start),
            events=events,
            exchange_tracking=state.facts.exchange_tracking,
        )

    def update(self, event: dict[str, Any]) -> "ObservationMemory":
        if _event_seq(event) < self.observation_start:
            return self
        return ObservationMemory(
            observation_start=self.observation_start,
            events=[dict(item) for item in self.events] + [dict(event)],
            exchange_tracking=self.exchange_tracking,
        )

    def to_observed_value(self) -> ObservedValue[dict[str, Any]]:
        return ObservedValue.observed(
            {
                "observation_start": self.observation_start,
                "events": [dict(event) for event in self.events],
                "exchange_tracking": self.exchange_tracking,
            }
        )


def _event_seq(event: dict[str, Any]) -> int:
    return int(event.get("seq", 0))
