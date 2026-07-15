from __future__ import annotations

import random

from engine.actions import Action, ActionKind, legal_declare_void_actions, legal_swap_actions, swap_direction_from_dice_sum
from engine.fan_calc import WinContext, calculate_fan
from engine.gang import GangKind
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.settlement import apply_kong_transfer, record_kong_payment, settle_drawn_game, settle_win

from engine.state import GameState, PendingDiscard, PendingRobKong


from engine.tiles import Tile, full_wall
from engine.win_check import can_win


class Game:
    def __init__(self, seed: int | None = None, dealer: int = 0):
        self.seed = seed
        self.dealer = dealer
        self.rng = random.Random(seed)
        self.state: GameState | None = None

    def reset(self) -> GameState:
        wall = full_wall()
        self.rng.shuffle(wall)
        hands = [Hand() for _ in range(4)]
        for player in range(4):
            draw_count = 14 if player == self.dealer else 13
            for _ in range(draw_count):
                hands[player].add(wall.pop())
        dice = (self.rng.randint(1, 6), self.rng.randint(1, 6))
        dice_sum = sum(dice)
        self.state = GameState(
            hands=hands,
            wall=wall,
            dealer=self.dealer,
            current_player=self.dealer,
            swap_direction=swap_direction_from_dice_sum(dice_sum),
            dice=dice,
        )

        return self.state

    def legal_actions(self, player: int) -> list[Action]:
        state = self._require_state()
        if state.finished:
            return []
        if state.phase == "swap_three":
            if state.swap_choices[player] is not None:
                return []
            return legal_swap_actions(state.hands[player])
        if state.phase == "declare_void":
            if state.void_suits[player] is not None:
                return []
            return legal_declare_void_actions()

        if state.phase == "play":
            if state.pending_rob_kong is not None:
                if player in state.pending_rob_kong.winners:
                    return [Action(ActionKind.ROB_KONG_WIN), Action(ActionKind.PASS)]
                return []
            if state.pending_discard is not None:

                pending = state.pending_discard
                if state.pending_winners:
                    if player in state.pending_winners:
                        return [Action(ActionKind.WIN), Action(ActionKind.PASS)]
                    return []
                if player == pending.discarder:
                    return []
                if player in state.pending_passers:

                    return []
                actions = []
                if not state.won[player]:

                    if state.hands[player].count(pending.tile) >= 2:
                        actions.append(Action(ActionKind.PONG, tile=pending.tile))
                    if len(state.wall) > 1 and state.hands[player].count(pending.tile) >= 3:
                        actions.append(Action(ActionKind.KONG, tile=pending.tile, kong_kind=GangKind.EXPOSED))
                    actions.append(Action(ActionKind.PASS))

                return actions

            if player != state.current_player or state.won[player]:
                return []
            actions = [Action(ActionKind.DISCARD, tile=tile) for tile in state.hands[player].tiles()]
            if can_win(state.hands[player], state.void_suits[player]):
                actions.append(Action(ActionKind.SELF_WIN))
            if len(state.wall) > 1:
                for tile in set(state.hands[player].tiles()):
                    if state.hands[player].count(tile) == 4:
                        actions.append(Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.CONCEALED))
                for meld in state.hands[player].melds:
                    if meld.kind is MeldKind.PONG and state.hands[player].count(meld.tiles[0]) >= 1:
                        actions.append(Action(ActionKind.KONG, tile=meld.tiles[0], kong_kind=GangKind.ADDED))

            return actions


        return []

    def step(self, player: int, action: Action) -> GameState:
        state = self._require_state()
        if state.finished:
            raise RuntimeError("game is finished")
        if state.phase not in {"swap_three", "declare_void", "play"}:
            raise RuntimeError(f"unknown phase: {state.phase}")
        if action not in self.legal_actions(player):
            raise ValueError(f"illegal action for player {player}: {action}")
        if state.phase == "swap_three":
            self._step_swap(player, action)

        elif state.phase == "declare_void":
            self._step_declare_void(player, action)
        elif state.phase == "play":
            if state.pending_rob_kong is not None:
                self._step_rob_kong_response(player, action)
            else:
                self._step_play(player, action)

        else:
            raise RuntimeError(f"unknown phase: {state.phase}")
        return state

    def _step_swap(self, player: int, action: Action) -> None:
        state = self._require_state()
        if action.kind is not ActionKind.SWAP_THREE or len(action.tiles) != 3:
            raise ValueError("swap phase requires swap_three action with three tiles")
        if len({tile.suit for tile in action.tiles}) != 1:
            raise ValueError("swap tiles must be same suit")
        for tile in action.tiles:
            state.hands[player].remove(tile)
        state.swap_choices[player] = action.tiles
        if all(choice is not None for choice in state.swap_choices):
            choices = [choice for choice in state.swap_choices if choice is not None]
            for giver, tiles in enumerate(choices):
                receiver = (giver - state.swap_direction) % 4
                for tile in tiles:
                    state.hands[receiver].add(tile)

            state.phase = "declare_void"

    def _step_declare_void(self, player: int, action: Action) -> None:
        state = self._require_state()
        if action.kind is not ActionKind.DECLARE_VOID or action.suit is None:
            raise ValueError("declare_void phase requires declare_void action")
        state.void_suits[player] = action.suit
        if all(suit is not None for suit in state.void_suits):
            state.phase = "play"
            state.current_player = state.dealer

    def _step_play(self, player: int, action: Action) -> None:
        state = self._require_state()
        if state.pending_discard is not None:
            self._step_response(player, action)
            return
        if player != state.current_player:
            raise ValueError("only current player can act")
        if action.kind is ActionKind.SELF_WIN:
            self._settle_self_win(player)
            return
        if action.kind is ActionKind.KONG:
            self._step_self_kong(player, action)
            return
        if action.kind is not ActionKind.DISCARD or action.tile is None:
            raise ValueError("play phase requires discard, kong, or self_win")

        state.hands[player].remove(action.tile)
        state.rivers[player].append(action.tile)
        state.pending_discard_after_kong = state.current_draw_after_kong
        state.pending_discard_last_wall = state.current_draw_last_wall
        state.current_draw_after_kong = False
        state.current_draw_last_wall = False
        winners = self._discard_winners(player, action.tile)
        state.pending_discard = PendingDiscard(discarder=player, tile=action.tile)
        state.pending_winners = winners

        if not state.win_order and len(winners) >= 2:
            state.first_win_multi_discarder = player


    def _step_response(self, player: int, action: Action) -> None:
        state = self._require_state()
        pending = state.pending_discard
        if pending is None:
            raise RuntimeError("no pending discard")
        if action.kind is ActionKind.WIN:
            if player not in state.pending_winners:
                raise ValueError("player cannot win this discard")
            state.hands[player].add(pending.tile)
            settle_win(
                state.scores,
                winner=player,
                payer=pending.discarder,
                hand=state.hands[player],
                context=self._pending_discard_win_context(),
            )

            self._apply_kong_transfer_if_needed(player)
            state.hands[player].remove(pending.tile)

            self._mark_winner(player)
            state.pending_winners.remove(player)
            if not state.pending_winners:
                self._clear_pending_discard()
                if not state.finished:
                    self._advance_after_resolved_discard(player)
            return

        if action.kind is ActionKind.PASS:
            if player in state.pending_winners:
                state.passed_hu_lock[player] = True
                state.passed_fan[player] = self._discard_win_fan(player, pending.tile)
                state.pending_winners.remove(player)
            elif player != pending.discarder:
                state.pending_passers.append(player)
                discarder = pending.discarder
                if all(
                    candidate == discarder
                    or state.won[candidate]
                    or candidate in state.pending_passers
                    for candidate in range(4)
                ):
                    self._clear_pending_discard()
                    self._advance_after_resolved_discard(discarder)
                return

            return


        if action.kind is ActionKind.PONG:
            self._step_pong(player, action)
            return
        if action.kind is ActionKind.KONG:
            self._step_exposed_kong(player, action)
            return
        raise ValueError("pending discard requires win, pong, kong, or pass")


    def _step_pong(self, player: int, action: Action) -> None:
        state = self._require_state()
        pending = state.pending_discard
        if pending is None or action.tile is None:
            raise RuntimeError("no pending discard to pong")
        if player == pending.discarder or state.hands[player].count(action.tile) < 2:
            raise ValueError("player cannot pong this discard")
        for _ in range(2):
            state.hands[player].remove(action.tile)
        state.hands[player].add_meld(Meld(MeldKind.PONG, (action.tile,) * 3, exposed=True, from_player=pending.discarder))
        self._remove_last_river_tile(pending.discarder, action.tile)
        self._clear_pending_discard()
        state.current_player = player

    def _step_exposed_kong(self, player: int, action: Action) -> None:
        state = self._require_state()
        pending = state.pending_discard
        if pending is None or action.tile is None or action.kong_kind is not GangKind.EXPOSED:
            raise RuntimeError("no pending discard to kong")
        if len(state.wall) <= 1:
            raise ValueError("cannot kong unless at least two wall tiles remain")

        if player == pending.discarder or state.hands[player].count(action.tile) < 3:
            raise ValueError("player cannot kong this discard")
        for _ in range(3):
            state.hands[player].remove(action.tile)
        state.hands[player].add_meld(Meld(MeldKind.KONG, (action.tile,) * 4, exposed=True, from_player=pending.discarder))
        self._remove_last_river_tile(pending.discarder, action.tile)
        record = record_kong_payment(
            state.scores,
            gang_id=state.next_gang_id,
            kong_player=player,
            kind=GangKind.EXPOSED,
            from_player=pending.discarder,
        )
        state.next_gang_id += 1
        state.gang_records.append(record)
        self._clear_pending_discard()
        state.current_player = player
        self._draw_replacement_after_kong(player, record.gang_id)

    def _step_self_kong(self, player: int, action: Action) -> None:
        state = self._require_state()
        if action.tile is None:
            raise ValueError("kong tile is required")
        if len(state.wall) <= 1:
            raise ValueError("cannot kong unless at least two wall tiles remain")

        if action.kong_kind is GangKind.CONCEALED:
            if state.hands[player].count(action.tile) != 4:
                raise ValueError("player cannot concealed kong this tile")
            for _ in range(4):
                state.hands[player].remove(action.tile)
            state.hands[player].add_meld(Meld(MeldKind.KONG, (action.tile,) * 4, exposed=False))
            record = record_kong_payment(
                state.scores,
                gang_id=state.next_gang_id,
                kong_player=player,
                kind=GangKind.CONCEALED,
            )
        elif action.kong_kind is GangKind.ADDED:
            if state.hands[player].count(action.tile) < 1:
                raise ValueError("player cannot added kong this tile")
            pong_index = self._find_pong_meld_index(player, action.tile)
            if pong_index is None:
                raise ValueError("added kong requires an existing pong meld")
            robbers = self._rob_kong_winners(player, action.tile)
            if robbers:
                state.pending_rob_kong = PendingRobKong(kong_player=player, tile=action.tile, winners=robbers)
                return
            state.hands[player].remove(action.tile)
            old_meld = state.hands[player].melds[pong_index]
            state.hands[player].melds[pong_index] = Meld(
                MeldKind.KONG,
                (action.tile,) * 4,
                exposed=True,
                from_player=old_meld.from_player,
            )
            record = record_kong_payment(
                state.scores,
                gang_id=state.next_gang_id,
                kong_player=player,
                kind=GangKind.ADDED,
            )

        else:
            raise ValueError("unsupported self kong kind")
        state.next_gang_id += 1
        state.gang_records.append(record)
        self._draw_replacement_after_kong(player, record.gang_id)

    def _find_pong_meld_index(self, player: int, tile: Tile) -> int | None:
        state = self._require_state()
        for index, meld in enumerate(state.hands[player].melds):
            if meld.kind is MeldKind.PONG and meld.tiles[0] == tile:
                return index
        return None

    def _rob_kong_winners(self, kong_player: int, tile: Tile) -> list[int]:
        state = self._require_state()
        winners: list[int] = []
        for player in range(4):
            if player == kong_player or state.won[player]:
                continue
            trial = Hand(counts=list(state.hands[player].counts), melds=list(state.hands[player].melds))
            try:
                trial.add(tile)
            except ValueError:
                continue
            if can_win(trial, state.void_suits[player]) and self._can_win_under_pass_lock(
                player,
                trial,
                WinContext(robbing_kong=True),
            ):
                winners.append(player)
        return winners

    def _step_rob_kong_response(self, player: int, action: Action) -> None:
        state = self._require_state()
        pending = state.pending_rob_kong
        if pending is None:
            raise RuntimeError("no pending rob kong")
        if action.kind is ActionKind.ROB_KONG_WIN:
            if player not in pending.winners:
                raise ValueError("player cannot rob this kong")
            state.hands[player].add(pending.tile)
            settle_win(
                state.scores,
                winner=player,
                payer=pending.kong_player,
                hand=state.hands[player],
                context=WinContext(robbing_kong=True),
            )
            state.hands[player].remove(pending.tile)
            self._mark_winner(player)
            pending.winners.remove(player)
            if not pending.winners:
                state.pending_rob_kong = None
                if not state.finished:
                    self._advance_after_resolved_discard(pending.kong_player)
            return
        if action.kind is ActionKind.PASS:
            if player not in pending.winners:
                raise ValueError("player has no rob kong response")
            state.passed_hu_lock[player] = True
            state.passed_fan[player] = self._rob_kong_win_fan(player, pending.tile)
            pending.winners.remove(player)
            if not pending.winners:
                state.pending_rob_kong = None
                self._complete_added_kong(pending.kong_player, pending.tile)
            return
        raise ValueError("pending rob kong requires rob_kong_win or pass")

    def _rob_kong_win_fan(self, player: int, tile: Tile) -> int:
        state = self._require_state()
        trial = Hand(counts=list(state.hands[player].counts), melds=list(state.hands[player].melds))
        trial.add(tile)
        return calculate_fan(trial, WinContext(robbing_kong=True)).fan

    def _complete_added_kong(self, player: int, tile: Tile) -> None:
        state = self._require_state()
        pong_index = self._find_pong_meld_index(player, tile)
        if pong_index is None:
            raise ValueError("added kong requires an existing pong meld")
        state.hands[player].remove(tile)
        old_meld = state.hands[player].melds[pong_index]
        state.hands[player].melds[pong_index] = Meld(MeldKind.KONG, (tile,) * 4, exposed=True, from_player=old_meld.from_player)
        record = record_kong_payment(state.scores, gang_id=state.next_gang_id, kong_player=player, kind=GangKind.ADDED)
        state.next_gang_id += 1
        state.gang_records.append(record)
        self._draw_replacement_after_kong(player, record.gang_id)

    def _draw_replacement_after_kong(self, player: int, gang_id: int | None) -> None:

        state = self._require_state()
        if not state.wall:
            raise ValueError("cannot draw replacement from empty wall")
        state.current_draw_after_kong = True
        state.current_draw_last_wall = len(state.wall) == 1
        state.hands[player].add(state.wall.pop())
        state.current_player = player

        state.last_transferable_gang_id = gang_id
        state.after_kong_discard_player = player

    def _remove_last_river_tile(self, player: int, tile: Tile) -> None:
        state = self._require_state()
        if not state.rivers[player] or state.rivers[player][-1] != tile:
            raise RuntimeError("discard tile is not at the end of river")
        state.rivers[player].pop()

    def _settle_self_win(self, player: int) -> None:
        state = self._require_state()
        settle_win(

            state.scores,
            winner=player,
            payer=None,
            hand=state.hands[player],
            context=WinContext(
                self_draw=True,
                after_kong=state.current_draw_after_kong,
                haidi=state.current_draw_last_wall,
            ),
            active_players=self._active_players(),

        )
        self._mark_winner(player)
        if not state.finished:
            self._advance_after_resolved_discard(player)

    def _discard_winners(self, discarder: int, tile: Tile) -> list[int]:
        state = self._require_state()
        winners: list[int] = []
        for player in range(4):
            if player == discarder or state.won[player]:
                continue
            trial = Hand(counts=list(state.hands[player].counts), melds=list(state.hands[player].melds))
            try:
                trial.add(tile)
            except ValueError:
                continue
            if can_win(trial, state.void_suits[player]) and self._can_win_under_pass_lock(
                player,
                trial,
                self._pending_discard_win_context(),
            ):
                winners.append(player)
        return winners

    def _can_win_under_pass_lock(
        self,
        player: int,
        hand: Hand,
        context: WinContext,
    ) -> bool:
        state = self._require_state()
        if not state.passed_hu_lock[player]:
            return True
        fan = calculate_fan(hand, context).fan
        return fan > state.passed_fan[player]



    def _discard_win_fan(self, player: int, tile: Tile) -> int:
        state = self._require_state()
        trial = Hand(counts=list(state.hands[player].counts), melds=list(state.hands[player].melds))
        trial.add(tile)
        return calculate_fan(trial, self._pending_discard_win_context()).fan

    def _pending_discard_win_context(self) -> WinContext:
        state = self._require_state()
        return WinContext(
            after_kong=state.pending_discard_after_kong,
            haidi=state.pending_discard_last_wall,
        )


    def _apply_kong_transfer_if_needed(self, winner: int) -> None:
        state = self._require_state()
        if state.last_transferable_gang_id is None:
            return
        for index, record in enumerate(state.gang_records):
            if record.gang_id == state.last_transferable_gang_id:
                state.gang_records[index] = apply_kong_transfer(state.scores, record, winner)
                return


    def _mark_winner(self, player: int) -> None:

        state = self._require_state()
        if not state.won[player]:
            state.won[player] = True
            state.win_order.append(player)
        self._check_finished()

    def _check_finished(self) -> None:
        state = self._require_state()
        if state.finished:
            return
        active_count = sum(not won for won in state.won)
        if active_count <= 1 or not state.wall:
            if not state.wall and active_count > 1:
                state.gang_records = settle_drawn_game(
                    state.scores,
                    hands=state.hands,
                    won=state.won,
                    void_suits=state.void_suits,
                    gang_records=state.gang_records,
                )
            state.finished = True
            state.phase = "finished"
            if state.first_win_multi_discarder is not None:
                state.next_dealer = state.first_win_multi_discarder
            elif state.win_order:
                state.next_dealer = state.win_order[0]
            else:
                state.next_dealer = state.dealer




    def _advance_after_resolved_discard(self, from_player: int) -> None:
        state = self._require_state()
        self._check_finished()
        if state.finished:
            return
        next_player = self._next_active_player(from_player)
        state.current_player = next_player
        if not state.wall:
            self._check_finished()
            return
        state.current_draw_after_kong = False
        state.current_draw_last_wall = len(state.wall) == 1
        state.hands[next_player].add(state.wall.pop())
        state.passed_hu_lock[next_player] = False

        state.passed_fan[next_player] = 0

    def _next_active_player(self, player: int) -> int:
        state = self._require_state()
        candidate = (player + 1) % 4
        while state.won[candidate]:
            candidate = (candidate + 1) % 4
        return candidate

    def _active_players(self) -> list[int]:
        state = self._require_state()
        return [player for player in range(4) if not state.won[player]]

    def _clear_pending_discard(self) -> None:
        state = self._require_state()
        state.pending_discard = None
        state.pending_winners = []
        state.pending_passers = []
        state.last_transferable_gang_id = None

        state.after_kong_discard_player = None
        state.pending_discard_after_kong = False
        state.pending_discard_last_wall = False



    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("game has not been reset")
        return self.state
