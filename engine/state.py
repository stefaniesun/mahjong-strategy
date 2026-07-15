from __future__ import annotations

from dataclasses import dataclass, field

from engine.gang import GangRecord
from engine.hand import Hand
from engine.tiles import Suit, Tile, tile_to_str



@dataclass(frozen=True)
class PendingDiscard:
    discarder: int
    tile: Tile


@dataclass(frozen=True)
class PendingRobKong:
    kong_player: int
    tile: Tile
    winners: list[int]


@dataclass
class GameState:

    hands: list[Hand]
    wall: list[Tile]
    dealer: int = 0
    current_player: int = 0
    phase: str = "swap_three"
    void_suits: list[Suit | None] = field(default_factory=lambda: [None, None, None, None])
    scores: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    won: list[bool] = field(default_factory=lambda: [False, False, False, False])
    win_order: list[int] = field(default_factory=list)
    rivers: list[list[Tile]] = field(default_factory=lambda: [[], [], [], []])
    swap_choices: list[tuple[Tile, ...] | None] = field(default_factory=lambda: [None, None, None, None])
    swap_direction: int = 1
    dice: tuple[int, int] | None = None
    pending_discard: PendingDiscard | None = None

    pending_winners: list[int] = field(default_factory=list)
    pending_passers: list[int] = field(default_factory=list)
    pending_rob_kong: PendingRobKong | None = None

    passed_hu_lock: list[bool] = field(default_factory=lambda: [False, False, False, False])

    passed_fan: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    gang_records: list[GangRecord] = field(default_factory=list)
    next_gang_id: int = 1
    last_transferable_gang_id: int | None = None
    after_kong_discard_player: int | None = None
    current_draw_after_kong: bool = False
    current_draw_last_wall: bool = False
    pending_discard_after_kong: bool = False
    pending_discard_last_wall: bool = False
    finished: bool = False

    next_dealer: int | None = None
    first_win_multi_discarder: int | None = None



    def to_dict(self) -> dict:
        return {
            "dealer": self.dealer,
            "current_player": self.current_player,
            "phase": self.phase,
            "wall_count": len(self.wall),
            "void_suits": [suit.value if suit else None for suit in self.void_suits],
            "scores": list(self.scores),
            "won": list(self.won),
            "win_order": list(self.win_order),
            "rivers": [[tile_to_str(tile) for tile in river] for river in self.rivers],
            "swap_direction": self.swap_direction,
            "dice": list(self.dice) if self.dice is not None else None,
            "dice_sum": sum(self.dice) if self.dice is not None else None,
            "pending_discard": None

            if self.pending_discard is None
            else {
                "discarder": self.pending_discard.discarder,
                "tile": tile_to_str(self.pending_discard.tile),
            },
            "pending_winners": list(self.pending_winners),
            "pending_passers": list(self.pending_passers),
            "pending_rob_kong": None

            if self.pending_rob_kong is None
            else {
                "kong_player": self.pending_rob_kong.kong_player,
                "tile": tile_to_str(self.pending_rob_kong.tile),
                "winners": list(self.pending_rob_kong.winners),
            },
            "passed_hu_lock": list(self.passed_hu_lock),

            "passed_fan": list(self.passed_fan),
            "gang_records": [
                {
                    "gang_id": record.gang_id,
                    "gang_player": record.gang_player,
                    "gang_type": record.gang_type.value,
                    "payments": list(record.payments),
                    "total_amount": record.total_amount,
                    "transferred_to": list(record.transferred_to),
                    "transfer_count": record.transfer_count,
                    "refunded": record.refunded,
                }
                for record in self.gang_records
            ],
            "next_gang_id": self.next_gang_id,
            "last_transferable_gang_id": self.last_transferable_gang_id,
            "after_kong_discard_player": self.after_kong_discard_player,
            "current_draw_after_kong": self.current_draw_after_kong,
            "current_draw_last_wall": self.current_draw_last_wall,
            "pending_discard_after_kong": self.pending_discard_after_kong,
            "pending_discard_last_wall": self.pending_discard_last_wall,
            "finished": self.finished,


            "next_dealer": self.next_dealer,
            "first_win_multi_discarder": self.first_win_multi_discarder,
        }

