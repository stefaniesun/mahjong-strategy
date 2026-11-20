"""人机对局:你坐 0 号位,另外三家由 AI 打;每次轮到你做决定时附上辅助器报告。

推进顺序与 assistant/driver.py、learning/eval/arena.py 一致。发给前端的只有你这个视角能看到的信息,
牌局结束后才摊开所有人的手牌。
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from dataclasses import asdict
from typing import Any

from engine.actions import Action, ActionKind
from engine.game import Game
from engine.gang import GangKind
from engine.meld import MeldKind
from engine.tiles import Suit, parse_tile, tile_to_str
from policies.base_policy import BasePolicy
from policies.decision_boundary import choose_policy_action

from assistant.adapters.from_engine import build_state
from assistant.analyzer import analyze
from assistant.recorder import PublicRecorder
from assistant.tiles import label, seat_name, suit_label

HUMAN = 0
KONG_NAMES = {GangKind.CONCEALED: "暗杠", GangKind.EXPOSED: "明杠", GangKind.ADDED: "补杠"}
SUIT_BY_VALUE = {suit.value: suit for suit in Suit}


def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "tiles": [tile_to_str(t) for t in action.tiles],
        "tile": None if action.tile is None else tile_to_str(action.tile),
        "suit": None if action.suit is None else action.suit.value,
        "kong_kind": None if action.kong_kind is None else action.kong_kind.value,
    }


def action_from_dict(data: dict[str, Any]) -> Action:
    return Action(
        ActionKind(data["kind"]),
        tiles=tuple(parse_tile(t) for t in data.get("tiles") or ()),
        tile=None if data.get("tile") is None else parse_tile(data["tile"]),
        suit=None if data.get("suit") is None else SUIT_BY_VALUE[data["suit"]],
        kong_kind=None if data.get("kong_kind") is None else GangKind(data["kong_kind"]),
    )


def button_label(action: Action) -> str:
    kind = action.kind
    if kind is ActionKind.PONG:
        return "碰"
    if kind is ActionKind.KONG:
        return f"{KONG_NAMES.get(action.kong_kind, '杠')} {label(tile_to_str(action.tile))}"
    if kind is ActionKind.WIN:
        return "胡"
    if kind is ActionKind.SELF_WIN:
        return "自摸"
    if kind is ActionKind.ROB_KONG_WIN:
        return "抢杠胡"
    if kind is ActionKind.PASS:
        return "过"
    return kind.value


def next_actor(game: Game) -> int | None:
    """下一个该行动的玩家;牌局结束返回 None。"""
    state = game.state
    if state is None or state.finished:
        return None
    if state.phase == "swap_three":
        return next(p for p in range(4) if state.swap_choices[p] is None)
    if state.phase == "declare_void":
        return next(p for p in range(4) if state.void_suits[p] is None)
    if state.pending_rob_kong is not None:
        return state.pending_rob_kong.winners[0]
    if state.pending_discard is not None:
        if state.pending_winners:
            return state.pending_winners[0]
        discarder = state.pending_discard.discarder
        for offset in range(1, 4):
            player = (discarder + offset) % 4
            if not state.won[player] and player not in state.pending_passers:
                return player
        raise RuntimeError("pending discard has no responder")
    return state.current_player


class HandLog:
    """一局的完整记录:种子、庄家和按顺序的每一步,足以原样复现牌局。"""

    def __init__(self, seed: int, dealer: int) -> None:
        self.seed = seed
        self.dealer = dealer
        self.steps: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "dealer": self.dealer, "steps": list(self.steps)}


def replay_hand(seed: int, dealer: int, steps: Sequence[dict[str, Any]]) -> tuple[Game, PublicRecorder]:
    """按记录把牌局重放到记录末尾,用于复现标记的局面。"""
    game = Game(seed=seed, dealer=dealer)
    game.reset()
    recorder = PublicRecorder(game)
    for step in steps:
        recorder.step(step["player"], action_from_dict(step["action"]))
    return game, recorder


class GameSession:
    def __init__(self, ai: BasePolicy, *, ai_name: str, log_dir: Path, rng_seed: int | None = None) -> None:
        self.ai = ai
        self.ai_name = ai_name
        self.rng = random.Random(rng_seed)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.totals = [0, 0, 0, 0]
        self.hand_no = 0
        self.dealer = 0
        self.history: list[dict[str, Any]] = []
        self.new_hand()

    # ---- 牌局推进 ----

    def new_hand(self, seed: int | None = None) -> None:
        if getattr(self, "game", None) is not None and self.game.state.next_dealer is not None:
            self.dealer = self.game.state.next_dealer
        self.hand_no += 1
        self.seed = seed if seed is not None else self.rng.randrange(1, 2**31)
        self.game = Game(seed=self.seed, dealer=self.dealer)
        self.game.reset()
        self.recorder = PublicRecorder(self.game)
        self.log = HandLog(self.seed, self.dealer)
        self.events: list[dict[str, Any]] = []
        self.settled = False
        self._report_cache: tuple[int, dict | None] | None = None

    @property
    def state(self):
        return self.game.state

    def human_legal(self) -> list[Action]:
        return self.game.legal_actions(HUMAN) if next_actor(self.game) == HUMAN else []

    def waiting_for_human(self) -> bool:
        return len(self.human_legal()) > 1

    def _step(self, player: int, action: Action) -> None:
        self.recorder.step(player, action)
        self.log.steps.append({"player": player, "action": action_to_dict(action)})
        event = self._event(player, action)
        if event is not None:
            self.events.append(event)
        if self.state.finished and not self.settled:
            self._settle()

    def _event(self, player: int, action: Action) -> dict[str, Any] | None:
        kind = action.kind
        name = seat_name(player, HUMAN)
        tile = None if action.tile is None else tile_to_str(action.tile)
        if kind is ActionKind.PASS:
            return None
        if kind is ActionKind.SWAP_THREE:
            text = f"{name}选好了换出的三张" if player != HUMAN else "你换出 " + "、".join(label(tile_to_str(t)) for t in action.tiles)
        elif kind is ActionKind.DECLARE_VOID:
            text = f"{name}定缺好了" if player != HUMAN else f"你定缺 {suit_label(action.suit.value)}"
        elif kind is ActionKind.DISCARD:
            text = f"{name}打出 {label(tile)}"
        elif kind is ActionKind.PONG:
            text = f"{name}碰 {label(tile)}"
        elif kind is ActionKind.KONG:
            text = f"{name}{KONG_NAMES.get(action.kong_kind, '杠')} {label(tile)}" if action.kong_kind is not GangKind.CONCEALED or player == HUMAN else f"{name}暗杠"
        elif kind in (ActionKind.WIN, ActionKind.SELF_WIN, ActionKind.ROB_KONG_WIN):
            record = self.state.win_records[player]
            tile = None if record is None or record.tile is None else tile_to_str(record.tile)
            how = {ActionKind.WIN: "胡", ActionKind.SELF_WIN: "自摸", ActionKind.ROB_KONG_WIN: "抢杠胡"}[kind]
            text = f"{name}{how}" + (f" {label(tile)}" if tile else "")
        else:
            text = f"{name} {kind.value}"
        return {"seat": player, "kind": kind.value, "tile": tile, "text": text, "n": len(self.log.steps)}

    def advance(self, max_steps: int = 400) -> list[dict[str, Any]]:
        """让 AI(以及你只有一个选项时的自动"过")往前走,直到出现一个看得见的动作、轮到你做决定或牌局结束。"""
        before = len(self.events)
        for _ in range(max_steps):
            actor = next_actor(self.game)
            if actor is None:
                break
            if actor == HUMAN:
                legal = self.game.legal_actions(HUMAN)
                if len(legal) > 1:
                    break
                self._step(HUMAN, legal[0])
            else:
                decision = choose_policy_action(self.state, actor, self.ai)
                self._step(actor, decision.action)
            if len(self.events) > before:
                break
        return self.events[before:]

    def act(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        legal = self.human_legal()
        if len(legal) <= 1:
            raise ValueError("现在不需要你做决定")
        action = self._match(payload, legal)
        before = len(self.events)
        self._step(HUMAN, action)
        return self.events[before:]

    def _match(self, payload: dict[str, Any], legal: list[Action]) -> Action:
        if "index" in payload:
            index = int(payload["index"])
            if not 0 <= index < len(legal):
                raise ValueError("动作编号无效")
            return legal[index]
        kind = ActionKind(payload["kind"])
        if kind is ActionKind.SWAP_THREE:
            wanted = sorted(parse_tile(t).index for t in payload.get("tiles") or ())
            for action in legal:
                if action.kind is kind and sorted(t.index for t in action.tiles) == wanted:
                    return action
            raise ValueError("换三张必须是同一花色的三张手牌")
        candidate = action_from_dict(payload)
        if candidate not in legal:
            raise ValueError("这个动作现在不合法")
        return candidate

    def _settle(self) -> None:
        self.settled = True
        for seat in range(4):
            self.totals[seat] += self.state.scores[seat]
        path = self.log_dir / f"hand{self.hand_no:03d}_seed{self.seed}.json"
        record = self.log.to_dict() | {
            "hand_no": self.hand_no,
            "scores": list(self.state.scores),
            "ai": self.ai_name,
            "rules": asdict(self.game.config),
        }
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        self.history.append({"hand_no": self.hand_no, "scores": list(self.state.scores)})

    # ---- 辅助器与标记 ----

    def report(self) -> dict[str, Any] | None:
        n = len(self.log.steps)
        if self._report_cache is not None and self._report_cache[0] == n:
            return self._report_cache[1]
        result = None
        if not self.state.finished:
            result = analyze(build_state(self.state, self.recorder, HUMAN)).to_dict()
        self._report_cache = (n, result)
        return result

    def mark(self, note: str) -> dict[str, Any]:
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hand_no": self.hand_no,
            "note": note,
            "replay": self.log.to_dict(),
            "view": self.view(include_report=False),
            "report": self.report(),
        }
        with (self.log_dir / "marks.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"ok": True, "count": sum(1 for _ in (self.log_dir / "marks.jsonl").open(encoding="utf-8"))}

    # ---- 发给前端的视图(只含你能看到的信息) ----

    def view(self, include_report: bool = True) -> dict[str, Any]:
        state = self.state
        finished = state.finished
        actor = next_actor(self.game)
        legal = self.human_legal()
        seats = [self._seat_view(seat, finished) for seat in range(4)]

        hand = [tile_to_str(t) for t in state.hands[HUMAN].tiles()]
        drawn = None
        if (
            state.phase == "play"
            and actor == HUMAN
            and state.pending_discard is None
            and state.pending_rob_kong is None
            and self.recorder.drew_since_meld[HUMAN]
            and self.recorder.last_drawn[HUMAN] in hand
        ):
            drawn = self.recorder.last_drawn[HUMAN]
            hand.remove(drawn)

        pending = None
        if state.pending_discard is not None:
            pending = {"seat": state.pending_discard.discarder, "tile": tile_to_str(state.pending_discard.tile)}
        elif state.pending_rob_kong is not None:
            pending = {"seat": state.pending_rob_kong.kong_player, "tile": tile_to_str(state.pending_rob_kong.tile), "rob": True}

        decision = None
        if len(legal) > 1:
            decision = {
                "phase": state.phase,
                "discard": sorted({tile_to_str(a.tile) for a in legal if a.kind is ActionKind.DISCARD}),
                "buttons": [
                    {"index": i, "label": button_label(a), "kind": a.kind.value}
                    for i, a in enumerate(legal)
                    if a.kind not in (ActionKind.DISCARD, ActionKind.SWAP_THREE, ActionKind.DECLARE_VOID)
                ],
                "swap": any(a.kind is ActionKind.SWAP_THREE for a in legal),
                "void": [a.suit.value for a in legal if a.kind is ActionKind.DECLARE_VOID],
            }

        swap_info = None
        if state.phase != "swap_three" and state.swap_choices[HUMAN] is not None:
            giver_to = (HUMAN - state.swap_direction) % 4
            received_from = next(p for p in range(4) if (p - state.swap_direction) % 4 == HUMAN)
            swap_info = {
                "given": [tile_to_str(t) for t in state.swap_choices[HUMAN]],
                "given_to": giver_to,
                "received": [tile_to_str(t) for t in state.swap_choices[received_from]],
                "received_from": received_from,
            }

        result = {
            "hand_no": self.hand_no,
            "seed": self.seed,
            "ai_name": self.ai_name,
            "rules": {"max_fan": self.game.config.max_fan, "self_draw_mode": self.game.config.self_draw_mode},
            "phase": state.phase,
            "finished": finished,
            "dealer": state.dealer,
            "wall": len(state.wall),
            "dice": list(state.dice) if state.dice else None,
            "swap_direction": state.swap_direction,
            "current": actor,
            "seats": seats,
            "hand": hand,
            "drawn": drawn,
            "pending": pending,
            "decision": decision,
            "swap_info": swap_info,
            "events": self.events[-40:],
            "totals": list(self.totals),
            "history": self.history[-20:],
        }
        if include_report:
            result["report"] = self.report() if decision is not None or state.phase == "play" else None
        return result

    def _seat_view(self, seat: int, finished: bool) -> dict[str, Any]:
        state = self.state
        hand = state.hands[seat]
        record = state.win_records[seat]
        reveal = finished or seat == HUMAN
        melds = []
        for meld in hand.melds:
            concealed = meld.kind is MeldKind.KONG and not meld.exposed
            melds.append({"kind": "pong" if meld.kind is MeldKind.PONG else "kong", "tile": tile_to_str(meld.tiles[0]), "concealed": concealed})
        win = None
        if state.won[seat] and record is not None:
            win = {
                "tile": None if record.tile is None else tile_to_str(record.tile),
                "how": {"discard": "胡", "self_draw": "自摸", "rob_kong": "抢杠胡"}[record.kind],
                "from": record.from_player,
                "order": state.win_order.index(seat) + 1,
            }
        known_tiles: list[str] = []
        if seat != HUMAN and state.phase != "swap_three" and state.swap_choices[HUMAN] is not None:
            giver_to = (HUMAN - state.swap_direction) % 4
            if seat == giver_to:
                known_tiles = [tile_to_str(t) for t in state.swap_choices[HUMAN]]
        hidden = hand.size - len(known_tiles)
        if win is not None and win["tile"] is not None and record.kind == "self_draw":
            hidden -= 1  # 自摸的那张亮在外面
        void = state.void_suits[seat]
        show_void = seat == HUMAN or state.phase in ("play", "finished")
        return {
            "seat": seat,
            "name": seat_name(seat, HUMAN),
            "is_ai": seat != HUMAN,
            "void": void.value if void is not None and show_void else None,
            "hidden": hidden,
            "known_tiles": known_tiles,
            "melds": melds,
            "river": [tile_to_str(t) for t in state.rivers[seat]],
            "won": state.won[seat],
            "win": win,
            "score": state.scores[seat],
            "total": self.totals[seat] + (0 if self.settled else state.scores[seat]),
            "hand": [tile_to_str(t) for t in hand.tiles()] if reveal else None,
        }
