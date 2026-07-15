# S3 Shanten Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the S3 shanten foundation by making `state/hand_analysis.py` the single source of truth for shanten, best discards, and useful tiles, then exposing a thin `policies/shanten.py` adapter.

**Architecture:** S2 owns the actual hand-analysis algorithm so later S3/S4/S5 users do not diverge. S3 imports the S2 functions through a small adapter and never reimplements shanten logic. Tests first lock behavior for standard hands, seven pairs, void-suit constraints, useful-tile counting, and S3/S2 parity.

**Tech Stack:** Python 3.10+, pytest, existing `engine.hand.Hand`, `engine.tiles`, and `engine.ting_check` conventions.

---

## File Structure

- Modify: `state/hand_analysis.py`
  - Add public `shanten(hand, void_suit=None)`, `best_discards(hand, void_suit=None)`, and `useful_tiles(hand, void_suit=None)`.
  - Keep `analyze_own_hand(state)` as existing S2 protocol feature extraction.
  - Implement standard-form and seven-pairs shanten with memoized suit decomposition.
- Create: `policies/__init__.py`
  - Mark `policies` as importable package.
- Create: `policies/shanten.py`
  - Thin S3 adapter that delegates to `state.hand_analysis`.
- Create: `tests/test_s3_shanten.py`
  - TDD coverage for source implementation and adapter parity.

---

### Task 1: Lock S2 Shanten Source Behavior

**Files:**
- Test: `tests/test_s3_shanten.py`
- Modify: `state/hand_analysis.py`

- [ ] **Step 1: Write failing tests for shanten basics**

Create `tests/test_s3_shanten.py` with these initial tests:

```python
from engine.hand import Hand
from engine.tiles import Suit
from state.hand_analysis import best_discards, shanten, useful_tiles


def hand(tiles: list[str]) -> Hand:
    return Hand.from_strings(tiles)


def test_complete_standard_hand_is_minus_one_shanten():
    assert shanten(hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "9W", "5T", "5T"])) == -1


def test_ready_standard_hand_is_zero_shanten():
    assert shanten(hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "9W", "5T"])) == 0


def test_one_away_standard_hand_is_one_shanten():
    assert shanten(hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])) == 1


def test_complete_seven_pairs_is_minus_one_shanten():
    assert shanten(hand(["1W", "1W", "2W", "2W", "3W", "3W", "4T", "4T", "5T", "5T", "6B", "6B", "9B", "9B"])) == -1


def test_ready_seven_pairs_is_zero_shanten():
    assert shanten(hand(["1W", "1W", "2W", "2W", "3W", "3W", "4T", "4T", "5T", "5T", "6B", "6B", "9B"])) == 0


def test_void_suit_tiles_do_not_count_as_progress():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "9W", "5T"])
    assert shanten(candidate) == 0
    assert shanten(candidate, void_suit=Suit.W) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s3_shanten.py -q
```

Expected: import failure because `shanten`, `best_discards`, and `useful_tiles` are not defined.

- [ ] **Step 3: Implement minimal public API in `state/hand_analysis.py`**

Add imports near the top:

```python
from functools import lru_cache
from itertools import combinations
from engine.tiles import Tile
```

Add these public functions after `analyze_own_hand`:

```python
def shanten(hand: Hand, void_suit: Suit | None = None) -> int:
    counts = _effective_counts(hand, void_suit)
    standard = _standard_shanten(tuple(counts))
    seven_pairs = _seven_pairs_shanten(counts)
    return min(standard, seven_pairs)


def best_discards(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for tile in hand.tiles():
        trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
        trial.remove(tile)
        candidates.append((shanten(trial, void_suit), tile_to_str(tile)))
    if not candidates:
        return []
    best = min(score for score, _ in candidates)
    return sorted({tile for score, tile in candidates if score == best})


def useful_tiles(hand: Hand, void_suit: Suit | None = None) -> list[str]:
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
    return sorted(useful, key=lambda value: (value[-1], int(value[:-1])))
```

Add private helpers below existing helpers:

```python
def _effective_counts(hand: Hand, void_suit: Suit | None) -> list[int]:
    counts = list(hand.counts)
    if void_suit is None:
        return counts
    for tile in _all_tiles():
        if tile.suit is void_suit:
            counts[tile.index] = 0
    return counts


def _all_tiles() -> list[Tile]:
    return [parse_tile(f"{rank}{suit.value}") for suit in SUITS for rank in range(1, 10)]


def _seven_pairs_shanten(counts: list[int]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - unique)


def _standard_shanten(counts: tuple[int, ...]) -> int:
    melds = 0
    taatsu = 0
    for meld in hand_melds_from_counts(counts):
        if len(meld.tiles) == 3:
            melds += 1
    best = 8
    for pair_index in [None] + [index for index, count in enumerate(counts) if count >= 2]:
        trial = list(counts)
        pairs = 0
        if pair_index is not None:
            trial[pair_index] -= 2
            pairs = 1
        extra_melds, extra_taatsu = _best_blocks(tuple(trial))
        total_melds = melds + extra_melds
        usable_taatsu = min(extra_taatsu, 4 - total_melds)
        best = min(best, 8 - total_melds * 2 - usable_taatsu - pairs)
    return best
```

Then replace the draft `_standard_shanten` if needed with an implementation that directly decomposes concealed counts and treats existing melds as fixed melds. Ensure no helper references an undefined function.

- [ ] **Step 4: Run tests and fix algorithm details**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s3_shanten.py -q
```

Expected after fixing: all tests in `tests/test_s3_shanten.py` pass.

---

### Task 2: Add Best Discards and Useful Tiles Coverage

**Files:**
- Test: `tests/test_s3_shanten.py`
- Modify: `state/hand_analysis.py`

- [ ] **Step 1: Add failing tests for discard and useful-tile behavior**

Append:

```python

def test_best_discards_keep_lowest_shanten_after_discard():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])
    discards = best_discards(candidate)
    assert "5T" in discards
    assert "9B" not in discards


def test_best_discards_prioritize_void_suit_when_present():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])
    discards = best_discards(candidate, void_suit=Suit.W)
    assert set(discards).issubset({"1W", "2W", "3W", "7W", "8W"})


def test_useful_tiles_are_tiles_that_reduce_shanten():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])
    useful = useful_tiles(candidate)
    assert "5T" in useful
    assert "9W" in useful


def test_useful_tiles_exclude_void_suit():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])
    useful = useful_tiles(candidate, void_suit=Suit.W)
    assert all(not tile.endswith("W") for tile in useful)
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s3_shanten.py -q
```

Expected: failures only where implementation needs tie-breaking or useful-tile details.

- [ ] **Step 3: Make minimal implementation changes**

Ensure `best_discards` has this void-suit priority before shanten scoring:

```python
void_discards = [tile_to_str(tile) for tile in hand.tiles() if void_suit is not None and tile.suit is void_suit]
if void_discards:
    return sorted(set(void_discards), key=lambda value: (value[-1], int(value[:-1])))
```

Ensure `useful_tiles` returns only legal 1-9 suited tiles, excludes void suit, excludes exhausted four-of-a-kind, and includes tiles that strictly reduce `shanten`.

- [ ] **Step 4: Run focused tests again**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s3_shanten.py -q
```

Expected: pass.

---

### Task 3: Add S3 Thin Adapter

**Files:**
- Create: `policies/__init__.py`
- Create: `policies/shanten.py`
- Test: `tests/test_s3_shanten.py`

- [ ] **Step 1: Add failing adapter parity tests**

Append to `tests/test_s3_shanten.py`:

```python
from policies import shanten as s3_shanten


def test_s3_shanten_adapter_matches_s2_source():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B"])
    assert s3_shanten.shanten(candidate) == shanten(candidate)
    assert s3_shanten.best_discards(candidate) == best_discards(candidate)
    assert s3_shanten.useful_tiles(candidate) == useful_tiles(candidate)


def test_s3_shanten_adapter_forwards_void_suit():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])
    assert s3_shanten.shanten(candidate, void_suit=Suit.W) == shanten(candidate, void_suit=Suit.W)
    assert s3_shanten.best_discards(candidate, void_suit=Suit.W) == best_discards(candidate, void_suit=Suit.W)
```

- [ ] **Step 2: Run test to verify adapter package is missing**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s3_shanten.py -q
```

Expected: import failure for `policies`.

- [ ] **Step 3: Create adapter files**

Create `policies/__init__.py`:

```python
"""Rule and baseline policy package for S3+."""
```

Create `policies/shanten.py`:

```python
from __future__ import annotations

from engine.hand import Hand
from engine.tiles import Suit
from state.hand_analysis import best_discards as _best_discards
from state.hand_analysis import shanten as _shanten
from state.hand_analysis import useful_tiles as _useful_tiles


def shanten(hand: Hand, void_suit: Suit | None = None) -> int:
    return _shanten(hand, void_suit=void_suit)


def best_discards(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    return _best_discards(hand, void_suit=void_suit)


def useful_tiles(hand: Hand, void_suit: Suit | None = None) -> list[str]:
    return _useful_tiles(hand, void_suit=void_suit)
```

- [ ] **Step 4: Run adapter tests**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s3_shanten.py -q
```

Expected: pass.

---

### Task 4: Regression and Cleanup

**Files:**
- Modify if needed: `state/hand_analysis.py`
- Modify if needed: `tests/test_s3_shanten.py`

- [ ] **Step 1: Run S2 and S3 focused tests**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest tests/test_s2_unknown_aware_features.py tests/test_s2_end_to_end.py tests/test_s3_shanten.py -q
```

Expected: all pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
Set-Location d:/sichuan-mahjong-engine; python -m pytest -q
```

Expected: all pass.

- [ ] **Step 3: Check lints for touched files**

Check:

- `state/hand_analysis.py`
- `policies/shanten.py`
- `tests/test_s3_shanten.py`

Expected: zero diagnostics or only pre-existing unrelated diagnostics.

---

## Self-Review

- Spec coverage: This plan implements S3 Task 1 only: S2-source shanten, best discards, useful tiles, and S3 thin adapter. It does not implement S3 heuristic policy, selfplay, data recorder, or opponent pool.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: Public signatures consistently use `Hand`, `Suit | None`, `int`, and `list[str]` across S2 source and S3 adapter.
