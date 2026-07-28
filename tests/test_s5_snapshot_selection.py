from __future__ import annotations

from tools.evaluate_s5_snapshots import CandidateResult, _selection_key, _wilson95


def _candidate(*, step: int, score: float, confidence: float, win_rate: float = 0.5, illegal: int = 0) -> CandidateResult:
    return CandidateResult(
        checkpoint=f"s5-step-{step}.pt",
        global_step=step,
        games=100,
        wins=round(win_rate * 100),
        win_rate=win_rate,
        win_rate_confidence95=0.1,
        average_score=score,
        score_confidence95=confidence,
        unfinished=0,
        illegal_actions=illegal,
        zero_sum_violations=0,
        seat_games=[25, 25, 25, 25],
        seat_win_rates=[win_rate] * 4,
        seat_average_scores=[score] * 4,
    )


def test_selection_prefers_healthy_lower_confidence_bound() -> None:
    stable = _candidate(step=100, score=2.0, confidence=0.5)
    noisy = _candidate(step=200, score=3.0, confidence=2.0)
    illegal = _candidate(step=300, score=10.0, confidence=0.1, illegal=1)
    assert max([noisy, illegal, stable], key=_selection_key) is stable


def test_wilson_confidence_half_width_is_bounded() -> None:
    assert _wilson95(0, 0) == 0.0
    assert 0.0 < _wilson95(50, 100) < 0.2
