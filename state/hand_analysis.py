from __future__ import annotations

from functools import lru_cache
from typing import Any

from engine.hand import Hand
from engine.tiles import SUITS, Suit, Tile, parse_tile, tile_to_str

from engine.ting_check import ting_tiles
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


def analyze_own_hand(state: S2ProtocolState) -> ObservedValue[dict[str, Any]]:
    player = _self_player(state)
    concealed = player.get("concealed_hand")
    if not isinstance(concealed, ObservedValue) or concealed.status is ObservationStatus.UNKNOWN:
        return ObservedValue.unknown()

    hand = Hand.from_strings(concealed.value or [])
    void_suit = _void_suit(player)
    suit_counts = _suit_counts(hand)
    value = {
        "tile_count": hand.size,
        "suit_counts": {suit.value: suit_counts[suit] for suit in SUITS},
        "void_suit": None if void_suit is None else void_suit.value,
        "void_tile_count": 0 if void_suit is None else suit_counts[void_suit],
        "pairs": sum(1 for count in hand.counts if count >= 2),
        "triplets": sum(1 for count in hand.counts if count >= 3),
        "quads": sum(1 for count in hand.counts if count >= 4),
        "singleton_tiles": [tile_to_str(tile) for tile in hand.tiles() if hand.count(tile) == 1],
        "ting_tiles": [tile_to_str(tile) for tile in sorted(ting_tiles(hand, void_suit))],

    }
    if concealed.status is ObservationStatus.ESTIMATED:
        return ObservedValue.estimated(value, concealed.confidence)
    return ObservedValue.observed(value)


def shanten(hand: Hand, void_suit: Suit | None = None) -> int:
    melds = len(hand.melds)
    counts = tuple(_effective_counts(hand, void_suit))
    standard = _standard_shanten(counts, melds)
    seven_pairs = _seven_pairs_shanten(counts) if melds == 0 else standard
    return min(standard, seven_pairs)


# 结果缓存:两个函数的输出仅由(手牌计数, 副露数, 定缺)决定,同一手牌在
# 候选弃牌枚举与跨巡决策中被反复查询。返回副本防止调用方改动污染缓存。
_BEST_DISCARDS_CACHE: dict[tuple, list[str]] = {}
_USEFUL_TILES_CACHE: dict[tuple, list[str]] = {}
_ANALYSIS_CACHE_LIMIT = 1 << 18


def best_discards(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    key = (tuple(hand.counts), len(hand.melds), void_suit)
    cached = _BEST_DISCARDS_CACHE.get(key)
    if cached is None:
        cached = _best_discards_uncached(hand, void_suit)
        if len(_BEST_DISCARDS_CACHE) < _ANALYSIS_CACHE_LIMIT:
            _BEST_DISCARDS_CACHE[key] = cached
    return list(cached)


def _best_discards_uncached(hand: Hand, void_suit: Suit | None) -> list[str]:
    void_discards = [tile_to_str(tile) for tile in hand.tiles() if void_suit is not None and tile.suit is void_suit]
    if void_discards:
        return _sort_tile_texts(set(void_discards))

    candidates: list[tuple[int, str]] = []
    for tile in hand.tiles():
        trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
        trial.remove(tile)
        candidates.append((shanten(trial, void_suit), tile_to_str(tile)))
    if not candidates:
        return []
    best = min(score for score, _ in candidates)
    return _sort_tile_texts({tile for score, tile in candidates if score == best})


def useful_tiles(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    key = (tuple(hand.counts), len(hand.melds), void_suit)
    cached = _USEFUL_TILES_CACHE.get(key)
    if cached is None:
        cached = _useful_tiles_uncached(hand, void_suit)
        if len(_USEFUL_TILES_CACHE) < _ANALYSIS_CACHE_LIMIT:
            _USEFUL_TILES_CACHE[key] = cached
    return list(cached)


def _useful_tiles_uncached(hand: Hand, void_suit: Suit | None) -> list[str]:
    current = shanten(hand, void_suit)
    useful: list[str] = []
    for tile in _all_tiles():
        if void_suit is not None and tile.suit is void_suit:
            continue
        if hand.count(tile) >= 4:
            continue
        trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
        trial.add(tile)
        if shanten(trial, void_suit) < current:
            useful.append(tile_to_str(tile))
    return _sort_tile_texts(useful)


def _self_player(state: S2ProtocolState) -> dict[str, Any]:
    return state.facts.players.value[0]



def _suit_counts(hand: Hand) -> dict[Suit, int]:
    counts = {suit: 0 for suit in SUITS}
    for tile in hand.tiles():
        counts[tile.suit] += 1
    return counts


def _void_suit(player: dict[str, Any]) -> Suit | None:
    value = player.get("void_suit")
    if not isinstance(value, ObservedValue) or value.status is ObservationStatus.UNKNOWN or value.value is None:
        return None
    return Suit(value.value)


def _effective_counts(hand: Hand, void_suit: Suit | None) -> list[int]:
    counts = list(hand.counts)
    if void_suit is None:
        return counts
    for tile in _all_tiles():
        if tile.suit is void_suit:
            counts[tile.index] = 0
    return counts


_ALL_TILES: tuple[Tile, ...] = tuple(Tile(suit, rank) for suit in SUITS for rank in range(1, 10))


def _all_tiles() -> tuple[Tile, ...]:
    return _ALL_TILES


def _seven_pairs_shanten(counts: tuple[int, ...]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - unique)


@lru_cache(maxsize=1 << 18)
def _standard_shanten(counts: tuple[int, ...], fixed_melds: int) -> int:
    best = 8
    pair_indexes: list[int | None] = [None]
    pair_indexes.extend(index for index, count in enumerate(counts) if count >= 2)
    for pair_index in pair_indexes:
        trial = list(counts)
        pair = 0
        if pair_index is not None:
            trial[pair_index] -= 2
            pair = 1
        melds, taatsu = _best_blocks(tuple(trial))
        total_melds = min(4, fixed_melds + melds)
        usable_taatsu = min(taatsu, 4 - total_melds)
        best = min(best, 8 - total_melds * 2 - usable_taatsu - pair)
    return best


@lru_cache(maxsize=None)
def _best_blocks(counts: tuple[int, ...]) -> tuple[int, int]:
    best = (0, 0)
    for start in (0, 9, 18):
        melds, taatsu = _best_suit_blocks(counts[start : start + 9])
        best = _add_blocks(best, (melds, taatsu))
    return best


@lru_cache(maxsize=None)
def _best_suit_blocks(counts: tuple[int, ...]) -> tuple[int, int]:
    first = next((index for index, count in enumerate(counts) if count > 0), None)
    if first is None:
        return (0, 0)

    best = (0, 0)
    reduced = list(counts)
    reduced[first] -= 1
    best = max(best, _best_suit_blocks(tuple(reduced)), key=_block_score)

    if counts[first] >= 3:
        reduced = list(counts)
        reduced[first] -= 3
        best = max(best, _add_blocks((1, 0), _best_suit_blocks(tuple(reduced))), key=_block_score)

    if first <= 6 and counts[first + 1] > 0 and counts[first + 2] > 0:
        reduced = list(counts)
        reduced[first] -= 1
        reduced[first + 1] -= 1
        reduced[first + 2] -= 1
        best = max(best, _add_blocks((1, 0), _best_suit_blocks(tuple(reduced))), key=_block_score)

    if counts[first] >= 2:
        reduced = list(counts)
        reduced[first] -= 2
        best = max(best, _add_blocks((0, 1), _best_suit_blocks(tuple(reduced))), key=_block_score)

    for second in (first + 1, first + 2):
        if second < 9 and counts[second] > 0:
            reduced = list(counts)
            reduced[first] -= 1
            reduced[second] -= 1
            best = max(best, _add_blocks((0, 1), _best_suit_blocks(tuple(reduced))), key=_block_score)

    return best


def _add_blocks(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return (left[0] + right[0], left[1] + right[1])


def _block_score(blocks: tuple[int, int]) -> tuple[int, int]:
    melds, taatsu = blocks
    return (melds * 2 + min(taatsu, 4 - melds), melds)


def _sort_tile_texts(tiles: set[str] | list[str]) -> list[str]:
    return sorted(tiles, key=lambda value: (SUITS.index(Suit(value[-1])), int(value[:-1])))

