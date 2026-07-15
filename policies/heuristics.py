from __future__ import annotations

from itertools import combinations

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit, Tile, parse_tile, tile_to_str

from policies.shanten import shanten, useful_tiles
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


def visible_hand_and_void_suit(protocol_state: S2ProtocolState) -> tuple[Hand, Suit | None]:
    players = protocol_state.facts.players
    if players.status is ObservationStatus.UNKNOWN or players.value is None:
        raise ValueError("protocol state does not expose player facts")
    own_player = next((player for player in players.value if player.get("relative_position") == 0), None)
    if own_player is None:
        raise ValueError("protocol state does not contain the perspective player")
    concealed_hand = own_player.get("concealed_hand")
    melds = own_player.get("melds")
    void_suit = own_player.get("void_suit")
    if not isinstance(concealed_hand, ObservedValue) or concealed_hand.status is ObservationStatus.UNKNOWN or concealed_hand.value is None:
        raise ValueError("protocol state does not expose the perspective player's concealed hand")
    if not isinstance(melds, ObservedValue) or melds.status is ObservationStatus.UNKNOWN or melds.value is None:
        raise ValueError("protocol state does not expose the perspective player's melds")
    if not isinstance(void_suit, ObservedValue) or void_suit.status is ObservationStatus.UNKNOWN:
        raise ValueError("protocol state does not expose the perspective player's void suit")
    hand = Hand.from_strings(concealed_hand.value)
    for meld in melds.value:
        hand.add_meld(
            Meld(
                kind=MeldKind(meld["kind"]),
                tiles=tuple(parse_tile(text) for text in meld["tiles"]),
                exposed=bool(meld.get("exposed", False)),
                from_player=meld.get("from_player"),
            )
        )
    return hand, None if void_suit.value is None else Suit(void_suit.value)



def choose_void_suit(hand: Hand) -> Suit:

    scores = []
    for suit in SUITS:
        suited = [tile for tile in hand.tiles() if tile.suit is suit]
        scores.append((len(suited), _suit_connection_score(hand, suit), SUITS.index(suit), suit))
    return min(scores)[3]


def choose_swap_tiles(hand: Hand) -> tuple[Tile, Tile, Tile]:
    preferred = choose_void_suit(hand)
    legal_suits = [preferred] + [suit for suit in SUITS if suit is not preferred]
    best_combo: tuple[Tile, Tile, Tile] | None = None
    best_score: tuple[int, int, int, tuple[str, ...]] | None = None
    for suit in legal_suits:
        suited = [tile for tile in hand.tiles() if tile.suit is suit]
        if len(suited) < 3:
            continue
        for combo in set(combinations(suited, 3)):
            score = _swap_combo_score(hand, combo, preferred)
            if best_score is None or score < best_score:
                best_combo = combo
                best_score = score
        if best_combo is not None and suit is preferred:
            break
    if best_combo is None:
        raise ValueError("hand has no legal swap-three choice")
    return tuple(sorted(best_combo))


def choose_discard(hand: Hand, void_suit: Suit | None = None) -> Tile:
    void_tiles = [tile for tile in hand.tiles() if void_suit is not None and tile.suit is void_suit]
    if void_tiles:
        return min(void_tiles, key=lambda tile: _discard_tiebreak_score(hand, tile, void_suit))

    candidates = {text for text in _best_discard_texts(hand, void_suit)}
    candidate_tiles = [tile for tile in set(hand.tiles()) if tile_to_str(tile) in candidates]
    return max(candidate_tiles, key=lambda tile: _discard_value_after_removal(hand, tile, void_suit))


def should_pong(hand: Hand, tile: Tile, void_suit: Suit | None = None) -> bool:
    if hand.count(tile) < 2:
        return False
    before = shanten(hand, void_suit=void_suit)
    trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
    trial.remove(tile)
    trial.remove(tile)
    trial.add_meld(Meld(MeldKind.PONG, (tile, tile, tile), exposed=True))
    return shanten(trial, void_suit=void_suit) < before


def _best_discard_texts(hand: Hand, void_suit: Suit | None) -> list[str]:
    from policies.shanten import best_discards

    return best_discards(hand, void_suit=void_suit)


def _discard_value_after_removal(hand: Hand, tile: Tile, void_suit: Suit | None) -> tuple[int, int, int, int, int]:
    trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
    trial.remove(tile)
    useful_count = len(useful_tiles(trial, void_suit=void_suit))
    return (
        useful_count,
        _isolation_score(hand, tile),
        -_duplicate_count(hand, tile),
        -_suit_count(hand, tile.suit),
        -tile.index,
    )


def _discard_tiebreak_score(hand: Hand, tile: Tile, void_suit: Suit | None) -> tuple[int, int, int, int]:
    trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
    trial.remove(tile)
    return (shanten(trial, void_suit=void_suit), -_isolation_score(hand, tile), _duplicate_count(hand, tile), tile.index)


def _swap_combo_score(hand: Hand, combo: tuple[Tile, Tile, Tile], preferred: Suit) -> tuple[int, int, int, tuple[str, ...]]:
    pair_breaks = sum(1 for tile in set(combo) if hand.count(tile) >= 2 and combo.count(tile) < hand.count(tile))
    kept_suit_penalty = 0 if combo[0].suit is preferred else 1
    connection = sum(_tile_connection_score(hand, tile) for tile in combo)
    duplicates = sum(_duplicate_count(hand, tile) for tile in combo)
    return (kept_suit_penalty, pair_breaks + duplicates, connection, tuple(tile_to_str(tile) for tile in sorted(combo)))


def _suit_connection_score(hand: Hand, suit: Suit) -> int:
    return sum(_tile_connection_score(hand, tile) for tile in hand.tiles() if tile.suit is suit)


def _tile_connection_score(hand: Hand, tile: Tile) -> int:
    score = 0
    for delta in (-2, -1, 1, 2):
        rank = tile.rank + delta
        if 1 <= rank <= 9 and hand.count(Tile(tile.suit, rank)) > 0:
            score += 2 if abs(delta) == 1 else 1
    score += 3 * max(0, hand.count(tile) - 1)
    return score


def _isolation_score(hand: Hand, tile: Tile) -> int:
    return -_tile_connection_score(hand, tile)


def _duplicate_count(hand: Hand, tile: Tile) -> int:
    return max(0, hand.count(tile) - 1)


def _suit_count(hand: Hand, suit: Suit) -> int:
    return sum(count for index, count in enumerate(hand.counts) if Tile.from_index(index).suit is suit)
