from __future__ import annotations

from dataclasses import replace

from engine.config import RuleConfig
from engine.fan_calc import WinContext, calculate_fan
from engine.gang import GangKind, GangRecord
from engine.hand import Hand
from engine.tiles import Suit
from engine.ting_check import max_ting_fan, ting_tiles




def assert_zero_sum(scores: list[int]) -> None:
    total = sum(scores)
    if total != 0:
        raise AssertionError(f"scores must be zero-sum, got total={total}: {scores}")


def _transfer(scores: list[int], payer: int, receiver: int, amount: int) -> None:
    scores[payer] -= amount
    scores[receiver] += amount


def settle_win(
    scores: list[int],
    winner: int,
    payer: int | None,
    hand: Hand,
    context: WinContext,
    active_players: list[int] | None = None,
    config: RuleConfig | None = None,
) -> int:
    result = calculate_fan(hand, context, config)
    if context.self_draw:
        players = active_players if active_players is not None else list(range(len(scores)))
        for player in players:
            if player != winner:
                _transfer(scores, player, winner, result.score_per_payer)
    else:
        if payer is None:
            raise ValueError("payer is required for discard win")
        _transfer(scores, payer, winner, result.score_per_payer)
    assert_zero_sum(scores)
    return result.score_per_payer


def _normalize_kong_kind(kind: str | GangKind) -> GangKind:
    if isinstance(kind, GangKind):
        return kind
    return GangKind(kind)


def record_kong_payment(
    scores: list[int],
    gang_id: int,
    kong_player: int,
    kind: str | GangKind,
    from_player: int | None = None,
    config: RuleConfig | None = None,
) -> GangRecord:
    config = config or RuleConfig()
    gang_kind = _normalize_kong_kind(kind)
    amount = config.base_score * 2
    payments: list[tuple[int, int]] = []
    if gang_kind is GangKind.ADDED:
        assert_zero_sum(scores)
        return GangRecord(gang_id=gang_id, gang_player=kong_player, gang_type=gang_kind, payments=())
    if gang_kind is GangKind.EXPOSED:
        if from_player is None:
            raise ValueError("from_player is required for exposed kong")
        _transfer(scores, from_player, kong_player, amount)
        payments.append((from_player, amount))
    elif gang_kind is GangKind.CONCEALED:
        for player in range(len(scores)):
            if player != kong_player:
                _transfer(scores, player, kong_player, amount)
                payments.append((player, amount))
    else:
        raise ValueError(f"unknown kong kind: {kind}")
    assert_zero_sum(scores)
    return GangRecord(gang_id=gang_id, gang_player=kong_player, gang_type=gang_kind, payments=tuple(payments))


def apply_kong_transfer(scores: list[int], record: GangRecord, winner: int) -> GangRecord:
    if record.total_amount <= 0:
        assert_zero_sum(scores)
        return record
    _transfer(scores, record.gang_player, winner, record.total_amount)
    assert_zero_sum(scores)
    return replace(
        record,
        transferred_to=record.transferred_to + (winner,),
        transfer_count=record.transfer_count + 1,
    )


def _has_void_suit_tiles(hand: Hand, void_suit: Suit | None) -> bool:
    return void_suit is not None and any(tile.suit is void_suit for tile in hand.tiles())


def _is_ting_for_drawn_game(hand: Hand, void_suit: Suit | None) -> bool:
    if _has_void_suit_tiles(hand, void_suit):
        return False
    return bool(ting_tiles(hand, void_suit))



def _refund_kong_record(scores: list[int], record: GangRecord) -> GangRecord:
    if record.refunded or record.total_amount <= 0:
        return record
    for winner in record.transferred_to:
        _transfer(scores, winner, record.gang_player, record.total_amount)
    for payer, amount in record.payments:
        _transfer(scores, record.gang_player, payer, amount)
    return replace(record, refunded=True)


def settle_drawn_game(
    scores: list[int],
    hands: list[Hand],
    won: list[bool],
    void_suits: list[Suit | None],
    gang_records: list[GangRecord],
    config: RuleConfig | None = None,
) -> list[GangRecord]:
    config = config or RuleConfig()
    ting_players = [
        player
        for player, player_hand in enumerate(hands)
        if not won[player] and _is_ting_for_drawn_game(player_hand, void_suits[player])
    ]
    not_ting_players = [player for player, has_won in enumerate(won) if not has_won and player not in ting_players]

    for payer in not_ting_players:
        for receiver in ting_players:
            fan = max_ting_fan(hands[receiver], config, void_suits[receiver])

            _transfer(scores, payer, receiver, config.base_score * (2 ** fan))

    updated_records: list[GangRecord] = []
    for record in gang_records:
        if record.gang_player in not_ting_players:
            updated_records.append(_refund_kong_record(scores, record))
        else:
            updated_records.append(record)
    assert_zero_sum(scores)
    return updated_records





def apply_kong_payment(
    scores: list[int],
    kong_player: int,
    kind: str,
    from_player: int | None = None,
    config: RuleConfig | None = None,
) -> int:
    record = record_kong_payment(
        scores,
        gang_id=-1,
        kong_player=kong_player,
        kind=kind,
        from_player=from_player,
        config=config,
    )
    return record.total_amount

