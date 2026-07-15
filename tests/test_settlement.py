from engine.fan_calc import WinContext
from engine.gang import GangKind
from engine.hand import Hand
from engine.settlement import (
    apply_kong_payment,
    apply_kong_transfer,
    record_kong_payment,
    settle_drawn_game,
    settle_win,
    assert_zero_sum,
)
from engine.tiles import Suit




def hand(text: str) -> Hand:
    return Hand.from_strings(text.split())


def test_discard_win_settlement_is_zero_sum():
    scores = [0, 0, 0, 0]
    winner_hand = hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W")

    settle_win(scores, winner=1, payer=0, hand=winner_hand, context=WinContext())

    assert scores == [-4, 4, 0, 0]
    assert_zero_sum(scores)


def test_self_draw_all_unwon_players_pay():
    scores = [0, 0, 0, 0]
    winner_hand = hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W")

    settle_win(scores, winner=2, payer=None, hand=winner_hand, context=WinContext(self_draw=True), active_players=[0, 1, 2, 3])

    assert scores == [-5, -5, 15, -5]
    assert_zero_sum(scores)


def test_exposed_and_concealed_kong_payments():
    scores = [0, 0, 0, 0]

    apply_kong_payment(scores, kong_player=1, kind="exposed", from_player=0)
    assert scores == [-2, 2, 0, 0]
    apply_kong_payment(scores, kong_player=2, kind="concealed")
    assert scores == [-4, 0, 6, -2]
    assert_zero_sum(scores)


def test_added_kong_has_no_payment():
    scores = [0, 0, 0, 0]

    apply_kong_payment(scores, kong_player=1, kind="added")

    assert scores == [0, 0, 0, 0]
    assert_zero_sum(scores)


def test_record_kong_payment_returns_auditable_record():
    scores = [0, 0, 0, 0]

    record = record_kong_payment(scores, gang_id=7, kong_player=2, kind=GangKind.CONCEALED)

    assert record.gang_id == 7
    assert record.gang_player == 2
    assert record.gang_type is GangKind.CONCEALED
    assert record.payments == ((0, 2), (1, 2), (3, 2))
    assert record.total_amount == 6
    assert scores == [-2, -2, 6, -2]
    assert_zero_sum(scores)


def test_added_kong_record_has_no_payments():
    scores = [0, 0, 0, 0]

    record = record_kong_payment(scores, gang_id=8, kong_player=1, kind=GangKind.ADDED)

    assert record.payments == ()
    assert record.total_amount == 0
    assert scores == [0, 0, 0, 0]
    assert_zero_sum(scores)


def test_kong_transfer_moves_full_amount_to_winner():
    scores = [0, 0, 0, 0]
    record = record_kong_payment(scores, gang_id=9, kong_player=2, kind=GangKind.CONCEALED)

    updated = apply_kong_transfer(scores, record, winner=1)

    assert scores == [-2, 4, 0, -2]
    assert updated.transferred_to == (1,)
    assert updated.transfer_count == 1
    assert_zero_sum(scores)


def test_kong_transfer_can_copy_to_multiple_winners():
    scores = [0, 0, 0, 0]
    record = record_kong_payment(scores, gang_id=10, kong_player=2, kind=GangKind.EXPOSED, from_player=0)

    record = apply_kong_transfer(scores, record, winner=1)
    record = apply_kong_transfer(scores, record, winner=3)

    assert scores == [-2, 2, -2, 2]
    assert record.transferred_to == (1, 3)
    assert record.transfer_count == 2
    assert_zero_sum(scores)


def test_drawn_game_chajia_pays_ting_players_and_refunds_untilted_kongs():
    scores = [0, 0, 0, 0]
    record = record_kong_payment(scores, gang_id=11, kong_player=0, kind=GangKind.CONCEALED)
    hands = [
        hand("1B 2B 3B 4W 5W 6W 7W 8W 9W 2T 3T 4T 5T"),
        hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W"),
        hand("1W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 4T 5T"),
        hand("1W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 4T 5T"),
    ]

    updated_records = settle_drawn_game(
        scores,
        hands=hands,
        won=[False, False, False, False],
        void_suits=[Suit.BING, None, None, None],
        gang_records=[record],
    )

    assert scores == [-4, 12, -4, -4]
    assert updated_records[0].refunded is True
    assert_zero_sum(scores)



def test_drawn_game_refund_also_reverts_kong_transfer():
    scores = [0, 0, 0, 0]
    record = record_kong_payment(scores, gang_id=12, kong_player=0, kind=GangKind.CONCEALED)
    record = apply_kong_transfer(scores, record, winner=1)
    hands = [
        hand("1B 2B 3B 4W 5W 6W 7W 8W 9W 2T 3T 4T 5T"),
        hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W"),
        hand("1W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 4T 5T"),
        hand("1W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 4T 5T"),
    ]

    updated_records = settle_drawn_game(
        scores,
        hands=hands,
        won=[False, False, False, False],
        void_suits=[Suit.BING, None, None, None],
        gang_records=[record],
    )

    assert scores == [-4, 12, -4, -4]
    assert updated_records[0].refunded is True
    assert_zero_sum(scores)



