from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ObservedValue(Generic[T]):
    value: T | None
    status: ObservationStatus
    confidence: float

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, ObservationStatus) else ObservationStatus(self.status)
        object.__setattr__(self, "status", status)
        # 归一化契约保留(嵌套的 JSON 形状字典自动还原为 ObservedValue),
        # 但标量值直接跳过整个递归——绝大多数字段是标量,这是构造热点。
        value = self.value
        if isinstance(value, (dict, list)):
            object.__setattr__(self, "value", _from_jsonable(value))
        if not 0.0 <= self.confidence <= 1.0:

            raise ValueError("confidence must be in [0, 1]")
        if status is ObservationStatus.OBSERVED and self.confidence != 1.0:
            raise ValueError("observed values must have confidence 1.0")
        if status is ObservationStatus.UNKNOWN:
            if self.value is not None:
                raise ValueError("unknown values must not carry a value")
            if self.confidence != 0.0:
                raise ValueError("unknown values must have confidence 0.0")

    @classmethod
    def observed(cls, value: T) -> "ObservedValue[T]":
        # 快路径:observed 的校验恒成立(confidence=1.0),绕过 dataclass __init__ 机制。
        # 该方法是协议构造热点(每局被调数千次),语义与常规构造完全一致。
        self = object.__new__(cls)
        if isinstance(value, (dict, list)):
            value = _from_jsonable(value)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "status", ObservationStatus.OBSERVED)
        object.__setattr__(self, "confidence", 1.0)
        return self

    @classmethod
    def estimated(cls, value: T, confidence: float) -> "ObservedValue[T]":
        return cls(value=value, status=ObservationStatus.ESTIMATED, confidence=confidence)

    @classmethod
    def unknown(cls) -> "ObservedValue[Any]":
        # unknown 是无状态的不可变值,进程内共享同一个单例即可(相等性按值比较,行为不变)。
        return _UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _to_jsonable(self.value),
            "status": self.status.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservedValue[Any]":
        return cls(
            value=_from_jsonable(data.get("value")),
            status=ObservationStatus(data["status"]),
            confidence=float(data["confidence"]),
        )


# unknown 的进程级单例(不可变,值相等语义下与新建实例完全等价)
_UNKNOWN: ObservedValue[Any] = ObservedValue(value=None, status=ObservationStatus.UNKNOWN, confidence=0.0)


@dataclass(frozen=True)
class Facts:
    players: ObservedValue[list[dict[str, Any]]] = field(default_factory=lambda: ObservedValue.observed([]))
    dealer: ObservedValue[int] = field(default_factory=ObservedValue.unknown)
    dealer_relative_position: ObservedValue[int] = field(default_factory=ObservedValue.unknown)
    is_dealer: ObservedValue[bool] = field(default_factory=ObservedValue.unknown)
    wall_count: ObservedValue[int] = field(default_factory=ObservedValue.unknown)
    is_last_tile: ObservedValue[bool] = field(default_factory=ObservedValue.unknown)
    pending_discard: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    pending_rob_kong: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    exchange_tracking: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    event_history: ObservedValue[list[dict[str, Any]]] = field(default_factory=lambda: ObservedValue.observed([]))
    revealed_win_hands: ObservedValue[dict[int, list[str]]] = field(default_factory=lambda: ObservedValue.observed({}))
    seen_counts: ObservedValue[list[float]] = field(default_factory=lambda: ObservedValue.observed([0] * 27))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_observed_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Facts":
        return cls(**_observed_fields_from_dict(data))


@dataclass(frozen=True)
class Statistics:
    remaining_tile_counts: ObservedValue[list[float]] = field(default_factory=lambda: ObservedValue.observed([4] * 27))
    unknown_pool_breakdown: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    own_hand_analysis: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    candidate_action_features: ObservedValue[list[dict[str, Any]]] = field(default_factory=lambda: ObservedValue.observed([]))
    dingque_constraints: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_observed_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Statistics":
        return cls(**_observed_fields_from_dict(data))


@dataclass(frozen=True)
class Beliefs:
    source: ObservedValue[str] = field(default_factory=lambda: ObservedValue.observed("prior"))
    tile_location_beliefs: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    opponent_tenpai_beliefs: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    discard_danger: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_observed_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Beliefs":
        return cls(**_observed_fields_from_dict(data))


@dataclass(frozen=True)
class S2ProtocolState:
    perspective_player: int
    phase: ObservedValue[str]
    current_player: ObservedValue[int]
    current_player_relative: ObservedValue[int]
    facts: Facts
    statistics: Statistics = field(default_factory=Statistics)
    beliefs: Beliefs = field(default_factory=Beliefs)
    legal_actions: ObservedValue[list[dict[str, Any]]] = field(default_factory=lambda: ObservedValue.observed([]))
    observation_start: ObservedValue[int] = field(default_factory=lambda: ObservedValue.observed(0))
    rule_config: ObservedValue[dict[str, Any]] = field(default_factory=ObservedValue.unknown)
    version: str = "s2.v4"

    def __post_init__(self) -> None:
        if self.version != "s2.v4":
            raise ValueError("unsupported protocol version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "perspective_player": self.perspective_player,
            "phase": self.phase.to_dict(),
            "current_player": self.current_player.to_dict(),
            "current_player_relative": self.current_player_relative.to_dict(),
            "facts": self.facts.to_dict(),
            "statistics": self.statistics.to_dict(),
            "beliefs": self.beliefs.to_dict(),
            "legal_actions": self.legal_actions.to_dict(),
            "observation_start": self.observation_start.to_dict(),
            "rule_config": self.rule_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "S2ProtocolState":
        return cls(
            version=data["version"],
            perspective_player=data["perspective_player"],
            phase=ObservedValue.from_dict(data["phase"]),
            current_player=ObservedValue.from_dict(data["current_player"]),
            current_player_relative=ObservedValue.from_dict(data["current_player_relative"]),
            facts=Facts.from_dict(data["facts"]),
            statistics=Statistics.from_dict(data["statistics"]),
            beliefs=Beliefs.from_dict(data["beliefs"]),
            legal_actions=ObservedValue.from_dict(data["legal_actions"]),
            observation_start=ObservedValue.from_dict(data["observation_start"]),
            rule_config=ObservedValue.from_dict(data["rule_config"]),
        )


def _dataclass_observed_to_dict(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name).to_dict() for name in obj.__dataclass_fields__}


def _observed_fields_from_dict(data: dict[str, Any]) -> dict[str, ObservedValue[Any]]:
    return {name: ObservedValue.from_dict(value) for name, value in data.items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, ObservedValue):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _from_jsonable(value: Any) -> Any:
    # 性能注意:只对 dict/list 子项递归,标量与 ObservedValue 子项原样保留,
    # 顶层容器仍返回新对象(保持原有的复制语义,防止调用方后续改动引起别名问题)。
    if isinstance(value, dict):
        if set(value) == {"value", "status", "confidence"}:
            return ObservedValue.from_dict(value)
        return {
            key: _from_jsonable(item) if isinstance(item, (dict, list)) else item
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_from_jsonable(item) if isinstance(item, (dict, list)) else item for item in value]
    return value




