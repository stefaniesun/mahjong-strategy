from tools.random_playout import run_many, run_random_game


def test_random_game_finishes_and_is_zero_sum():
    result = run_random_game(seed=1, max_steps=500)

    assert result.finished is True
    assert sum(result.scores) == 0
    assert result.steps <= 500


def test_run_many_seeded_games_are_zero_sum():
    results = run_many(games=20, seed=10, max_steps=500)

    assert len(results) == 20
    assert all(result.finished for result in results)
    assert all(sum(result.scores) == 0 for result in results)
