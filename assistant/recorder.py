"""对局公开事件记录器。

引擎本身不记录胡牌张:点炮胡的那张留在放炮者牌河里,自摸胡的那张混在胡牌者手里。
所以通过它来推进对局,顺手记下:每家打出过的全部牌、胡牌事件(胡牌张、自摸/点炮/抢杠)、补杠。
自摸的胡牌张 = 胡牌者最近摸到的那张(摸牌只发生在牌墙顶,由谁摸由引擎推进后的当前玩家决定)。
"""
from __future__ import annotations

from engine.actions import Action, ActionKind
from engine.game import Game
from engine.gang import GangKind
from engine.tiles import tile_to_str

from assistant.state import WinRecord


class PublicRecorder:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.discards: list[list[str]] = [[] for _ in range(4)]
        self.discard_tsumogiri: list[list[bool]] = [[] for _ in range(4)]
        self.passed_since_tedashi: list[set[str]] = [set() for _ in range(4)]
        self.wins: list[WinRecord] = []
        self.last_drawn: list[str | None] = [None] * 4
        self.draw_counts: list[int] = [0] * 4
        self.drew_since_meld: list[bool] = [False] * 4
        self.added_kongs: set[tuple[int, str]] = set()
        self.kong_income: dict[tuple[int, str], int] = {}  # (杠者, 牌) -> 杠的那一刻实际收到的杠钱
        self.events: list[dict] = []

    def step(self, player: int, action: Action) -> None:
        state = self.game.state
        assert state is not None
        wall_before = len(state.wall)
        wall_top = state.wall[-1] if state.wall else None
        pending = state.pending_discard
        rob = state.pending_rob_kong
        was_tsumogiri = (
            action.kind is ActionKind.DISCARD
            and action.tile is not None
            and self.drew_since_meld[player]
            and self.last_drawn[player] == tile_to_str(action.tile)
        )

        self.game.step(player, action)
        state = self.game.state
        kind = action.kind

        if kind is ActionKind.DISCARD and action.tile is not None:
            t_str = tile_to_str(action.tile)
            self.discards[player].append(t_str)
            self.discard_tsumogiri[player].append(was_tsumogiri)
            if not was_tsumogiri:
                self.passed_since_tedashi[player].clear()
            # 若此张牌打出后直接无人进入待响应(或 pending_winners 不含某在场玩家), 说明其他在场玩家当前手牌不能胡这张
            pending_win_seats = set(state.pending_winners) if state.pending_discard is not None else set()
            for s in range(4):
                if s != player and not state.won[s] and s not in pending_win_seats:
                    self.passed_since_tedashi[s].add(t_str)
            self.events.append({"type": "discard", "seat": player, "tile": t_str, "tsumogiri": was_tsumogiri})
        elif kind is ActionKind.PASS and pending is not None:
            self.passed_since_tedashi[player].add(tile_to_str(pending.tile))
        elif kind is ActionKind.PONG and action.tile is not None:
            self.drew_since_meld[player] = False
            self.passed_since_tedashi[player].clear()
            self.events.append({"type": "pong", "seat": player, "tile": tile_to_str(action.tile)})
        elif kind is ActionKind.KONG and action.tile is not None:
            self.passed_since_tedashi[player].clear()
            if action.kong_kind is GangKind.ADDED:
                self.added_kongs.add((player, tile_to_str(action.tile)))
            if state.gang_records and state.gang_records[-1].gang_player == player:
                self.kong_income[(player, tile_to_str(action.tile))] = state.gang_records[-1].total_amount
            self.events.append(
                {"type": "kong", "seat": player, "tile": tile_to_str(action.tile), "kong_kind": action.kong_kind.value if action.kong_kind else None}
            )
        elif kind is ActionKind.WIN and pending is not None:
            self.wins.append(WinRecord(player, tile_to_str(pending.tile), False, pending.discarder))
            self.events.append({"type": "win", "seat": player, "tile": tile_to_str(pending.tile), "from": pending.discarder})
        elif kind is ActionKind.ROB_KONG_WIN and rob is not None:
            self.wins.append(WinRecord(player, tile_to_str(rob.tile), False, rob.kong_player, robbing_kong=True))
            self.events.append({"type": "rob_kong_win", "seat": player, "tile": tile_to_str(rob.tile), "from": rob.kong_player})
        elif kind is ActionKind.SELF_WIN:
            tile = self.last_drawn[player] if self.drew_since_meld[player] else None
            self.wins.append(WinRecord(player, tile, True, None))
            self.events.append({"type": "self_win", "seat": player, "tile": tile})

        if wall_top is not None and len(state.wall) == wall_before - 1:
            drawer = state.current_player
            self.last_drawn[drawer] = tile_to_str(wall_top)
            self.draw_counts[drawer] += 1
            self.drew_since_meld[drawer] = True
