from __future__ import annotations

from engine.hand import Hand
from engine.meld import MeldKind
from engine.tiles import Suit, Tile


def _has_void_suit(hand: Hand, void_suit: Suit | None) -> bool:
    if void_suit is None:
        return False
    for index, count in enumerate(hand.counts):
        if count and Tile.from_index(index).suit is void_suit:
            return True
    return False


def is_seven_pairs(hand: Hand) -> bool:
    if hand.melds:
        return False
    if hand.size != 14:
        return False
    return sum(count == 2 for count in hand.counts) + sum(count == 4 for count in hand.counts) * 2 == 7


def _can_split_melds(counts: tuple[int, ...]) -> bool:
    try:
        first = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return True

    work = list(counts)
    if work[first] >= 3:
        work[first] -= 3
        if _can_split_melds(tuple(work)):
            return True
        work[first] += 3

    rank = first % 9
    if rank <= 6 and first // 9 == (first + 1) // 9 == (first + 2) // 9:
        if work[first + 1] > 0 and work[first + 2] > 0:
            work[first] -= 1
            work[first + 1] -= 1
            work[first + 2] -= 1
            if _can_split_melds(tuple(work)):
                return True

    return False


def is_standard_win(hand: Hand) -> bool:
    fixed_meld_count = sum(1 for meld in hand.melds if meld.kind in {MeldKind.CHOW, MeldKind.PONG, MeldKind.KONG})
    needed_hand_melds = 4 - fixed_meld_count
    if needed_hand_melds < 0:
        return False
    if hand.size != needed_hand_melds * 3 + 2:
        return False

    for pair_index, count in enumerate(hand.counts):
        if count < 2:
            continue
        work = list(hand.counts)
        work[pair_index] -= 2
        if _can_split_melds(tuple(work)):
            return True
    return False


def can_win(hand: Hand, void_suit: Suit | None = None) -> bool:
    if _has_void_suit(hand, void_suit):
        return False
    return is_seven_pairs(hand) or is_standard_win(hand)
