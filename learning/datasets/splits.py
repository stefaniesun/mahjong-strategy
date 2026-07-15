from __future__ import annotations

import random
from collections.abc import Sequence

from selfplay.data_recorder import DecisionRecord


SplitRecords = dict[str, list[DecisionRecord]]


def split_records_by_game(
    records: Sequence[DecisionRecord],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> SplitRecords:
    if len(ratios) != 3:
        raise ValueError("ratios must be a train/val/test triple")
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("ratios must be non-negative")
    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")

    game_ids = sorted({record.game_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(game_ids)

    train_count, val_count = _split_counts(len(game_ids), ratios)
    train_games = set(game_ids[:train_count])
    val_games = set(game_ids[train_count : train_count + val_count])
    test_games = set(game_ids[train_count + val_count :])

    return {
        "train": [record for record in records if record.game_id in train_games],
        "val": [record for record in records if record.game_id in val_games],
        "test": [record for record in records if record.game_id in test_games],
    }


def _split_counts(game_count: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    if game_count == 0:
        return 0, 0

    total = sum(ratios)
    normalized = [ratio / total for ratio in ratios]
    raw = [game_count * ratio for ratio in normalized]
    counts = [int(value) for value in raw]
    remainder = game_count - sum(counts)

    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], ratios[index]), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    positive_splits = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if game_count >= len(positive_splits):
        for index in positive_splits:
            if counts[index] == 0:
                donor = max((item for item in positive_splits if counts[item] > 1), key=lambda item: counts[item])
                counts[donor] -= 1
                counts[index] += 1

    return counts[0], counts[1]
