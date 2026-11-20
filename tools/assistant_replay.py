"""引擎对局 -> 逐手辅助报告 -> 自包含 HTML(双击即可打开,不依赖服务器和 CDN)。

用法:
    python tools/assistant_replay.py --seed 7 --player 0
    python tools/assistant_replay.py --seed 7 --player 0 --out outputs/assistant/replay_7_p0.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import Action, ActionKind
from engine.gang import GangKind
from engine.tiles import tile_to_str

from assistant.adapters.from_engine import build_state
from assistant.analyzer import analyze
from assistant.driver import run_game
from assistant.state import AssistantState
from assistant.tiles import label, seat_name, suit_label, tidx

TEMPLATE = Path(__file__).resolve().parents[1] / "assistant" / "viewer" / "template.html"
KONG_NAMES = {GangKind.CONCEALED: "暗杠", GangKind.EXPOSED: "明杠", GangKind.ADDED: "补杠"}


def action_text(action: Action) -> str:
    kind = action.kind
    if kind is ActionKind.DISCARD:
        return f"打 {label(tile_to_str(action.tile))}"
    if kind is ActionKind.PONG:
        return f"碰 {label(tile_to_str(action.tile))}"
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
    if kind is ActionKind.SWAP_THREE:
        return "换出 " + "、".join(label(tile_to_str(t)) for t in action.tiles)
    if kind is ActionKind.DECLARE_VOID:
        return f"定缺 {suit_label(action.suit.value)}"
    return kind.value


def table_view(state: AssistantState, scores: list[int]) -> dict:
    me = state.perspective
    seats = []
    for view in state.players:
        record = state.win_of(view.seat)
        win = None
        if record is not None:
            how = "自摸" if record.self_draw else ("抢杠胡" if record.robbing_kong else f"点炮胡(放炮:{seat_name(record.from_seat, me)})")
            win = {"tile": label(record.tile) if record.tile else "?", "how": how}
        seats.append(
            {
                "seat": view.seat,
                "name": seat_name(view.seat, me),
                "void": suit_label(view.void_suit) if view.void_suit else None,
                "won": view.won,
                "win": win,
                "melds": [
                    {"tile": label(m.tile), "kind": ("暗杠" if m.concealed else ("补杠" if m.added else ("明杠" if m.kind == "kong" else "碰"))), "size": m.size}
                    for m in view.melds
                ],
                "river": [label(t) for t in view.river],
                "hidden": view.hidden_count,
                "score": scores[view.seat],
            }
        )
    pending = None
    if state.pending_discard is not None:
        pending = f"{seat_name(state.pending_discard[0], me)}打出 {label(state.pending_discard[1])}"
    elif state.pending_rob_kong is not None:
        pending = f"{seat_name(state.pending_rob_kong[0], me)}补杠 {label(state.pending_rob_kong[1])}"
    return {
        "wall": state.wall_count,
        "phase": {"swap_three": "换三张", "declare_void": "定缺", "play": "对局", "finished": "结束"}.get(state.phase, state.phase),
        "current": seat_name(state.current_player, me),
        "own_hand": [label(t) for t in sorted(state.own_hand, key=tidx)],
        "pending": pending,
        "seats": seats,
    }


def build_replay(seed: int, player: int) -> dict:
    frames: list[dict] = []

    def hook(game, recorder, actor, decision):
        if actor != player:
            return
        state = build_state(game.state, recorder, actor)
        report = analyze(state)
        frames.append(
            {
                "table": table_view(state, list(game.state.scores)),
                "action": action_text(decision.action),
                "legal": [action_text(a) for a in decision.legal_actions],
                "report": report.to_dict(),
            }
        )

    game, recorder = run_game(seed, on_decision=hook)
    final = game.state
    return {
        "seed": seed,
        "player": player,
        "frames": frames,
        "summary": {
            "scores": {seat_name(s, player): final.scores[s] for s in range(4)},
            "wins": [
                {"name": seat_name(r.seat, player), "tile": label(r.tile) if r.tile else "?", "self_draw": r.self_draw}
                for r in recorder.wins
            ],
        },
    }


def render_html(replay: dict) -> str:
    data = json.dumps(replay, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成辅助器回放网页")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--player", type=int, default=0, choices=range(4))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path("outputs/assistant") / f"replay_seed{args.seed}_p{args.player}.html"
    replay = build_replay(args.seed, args.player)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(replay), encoding="utf-8")
    print(f"已生成 {out}:{len(replay['frames'])} 个决策点")


if __name__ == "__main__":
    main()
