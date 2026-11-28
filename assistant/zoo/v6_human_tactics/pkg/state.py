"""辅助器输入状态:某位玩家视角下牌桌上已知的一切。

与 S2 协议无关,只放辅助器做确定性计算需要的公开信息和自己的手牌。
约定(沿用项目规则契约):暗杠的牌面对所有人可见;胡牌者只亮出胡的那一张,手牌扣着。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.config import RuleConfig, active_rule_config

SEATS = 4


@dataclass(frozen=True)
class MeldView:
    kind: str  # "pong" 或 "kong"
    tile: str  # 如 "5W"
    concealed: bool = False  # 暗杠
    from_seat: int | None = None
    added: bool = False  # 补杠(碰后加杠,不收杠钱)
    income: int | None = None  # 杠的那一刻实际收到的杠钱(已胡离场的人不付);None 表示不知道

    def __post_init__(self) -> None:
        if self.kind not in ("pong", "kong"):
            raise ValueError(f"unsupported meld kind: {self.kind}")

    @property
    def size(self) -> int:
        return 4 if self.kind == "kong" else 3


@dataclass(frozen=True)
class WinRecord:
    seat: int
    tile: str | None  # 亮出的胡牌张;无法确定时为 None
    self_draw: bool
    from_seat: int | None = None  # 放炮者,或抢杠胡时的杠牌者
    robbing_kong: bool = False


@dataclass(frozen=True)
class PlayerView:
    seat: int
    melds: tuple[MeldView, ...] = ()
    river: tuple[str, ...] = ()  # 现在还留在牌河里的牌(被碰杠走的已进入对方副露)
    discards: tuple[str, ...] = ()  # 打出过的全部牌,按顺序,含后来被碰杠走的
    void_suit: str | None = None  # "W"/"T"/"B";未知时为 None
    won: bool = False
    hidden_count: int = 0  # 视角玩家看不到的暗手牌张数;已胡者不含胡牌张;视角玩家自己为 0
    discard_tsumogiri: tuple[bool, ...] = ()  # 与 discards 对齐: True=摸切, False=手切(未知时可为空)
    passed_since_tedashi: tuple[str, ...] = ()  # 对手自上次手切/副露(手牌变化)以来放过未胡的牌


@dataclass(frozen=True)
class SwapInfo:
    given: tuple[str, ...]
    given_to: int
    received: tuple[str, ...] = ()
    received_from: int | None = None


@dataclass(frozen=True)
class AssistantState:
    perspective: int
    phase: str  # "swap_three" / "declare_void" / "play" / "finished"
    own_hand: tuple[str, ...]
    players: tuple[PlayerView, ...]
    wins: tuple[WinRecord, ...] = ()
    wall_count: int = 0
    current_player: int = 0
    pending_discard: tuple[int, str] | None = None  # (放炮者, 牌)
    pending_after_kong: bool = False  # 待响应的这张是杠后打出的
    pending_last_wall: bool = False  # 待响应的这张是海底打出的
    pending_rob_kong: tuple[int, str] | None = None  # (补杠者, 牌)
    draw_after_kong: bool = False  # 自己刚摸的是杠后补牌
    draw_was_last: bool = False  # 自己刚摸的是海底最后一张
    passed_hu_lock: bool = False
    passed_fan: int = 0
    swap: SwapInfo | None = None
    rule_config: RuleConfig = field(default_factory=active_rule_config)
    swap_direction: int | None = None
    dealer: int | None = None

    def __post_init__(self) -> None:
        if len(self.players) != SEATS or any(view.seat != seat for seat, view in enumerate(self.players)):
            raise ValueError("players must be four PlayerView objects in seat order")
        if not 0 <= self.perspective < SEATS:
            raise ValueError("perspective must be a seat index 0..3")

    @property
    def me(self) -> PlayerView:
        return self.players[self.perspective]

    def opponents(self) -> list[PlayerView]:
        return [view for view in self.players if view.seat != self.perspective]

    def active_opponents(self) -> list[PlayerView]:
        return [view for view in self.opponents() if not view.won]

    def win_of(self, seat: int) -> WinRecord | None:
        for record in self.wins:
            if record.seat == seat:
                return record
        return None
