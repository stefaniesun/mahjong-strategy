# S1 Complete Game Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Turn the current S1 rule-engine skeleton into a deterministic full-round loop that can run random games and verify zero-sum scoring.

**Architecture:** Keep the existing small modules. Extend `GameState` with discard river, swap/void tracking, last discard response context, and terminal flags. Implement simple deterministic `Game.step()` progression for opening actions, discard, self-draw win, discard win/pass, and draw advancement. Keep complex kong transfer rules in later S1 hardening.

**Tech Stack:** Python 3.10+, pytest, existing `engine` package.

---

### Task 1: State and action protocol for full-round loop

**Files:**
- Modify: `engine/state.py`
- Modify: `engine/actions.py`
- Test: `tests/test_game_loop_full.py`

- [ ] Add tests for state serialization fields and legal play actions.
- [ ] Extend `GameState` with `rivers`, `swap_choices`, `swap_direction`, `pending_discard`, `pending_winners`, `passed_hu_lock`, `passed_fan`, `finished`, and `next_dealer`.
- [ ] Add `DRAW`, `SELF_WIN`, and richer `PASS`/`WIN` action support.
- [ ] Run `python -m pytest tests/test_game_loop_full.py -v`.

### Task 2: Opening phase step loop

**Files:**
- Modify: `engine/game.py`
- Test: `tests/test_game_loop_full.py`

- [ ] Add tests for completing swap-three and declare-void via `step()`.
- [ ] Implement `Game.step()` for `SWAP_THREE` and `DECLARE_VOID`.
- [ ] Ensure swap is synchronous and applies dice direction.
- [ ] Transition to `play` with dealer as `current_player`.

### Task 3: Play phase discard/draw/win loop

**Files:**
- Modify: `engine/game.py`
- Test: `tests/test_game_loop_full.py`

- [ ] Add tests for discard removing a tile, response pass drawing next tile, self-draw win, and discard win.
- [ ] Implement discard response detection with `can_win()`.
- [ ] Implement `PASS`, `WIN`, and `SELF_WIN` settlement flow.
- [ ] Mark winners, append `win_order`, and end round when unwon players <= 1 or wall is empty.

### Task 4: Random playout CLI and zero-sum smoke

**Files:**
- Modify: `tools/random_playout.py`
- Test: `tests/test_random_playout.py`

- [ ] Add tests for running multiple seeded random games without crash and with zero-sum scores.
- [ ] Implement `run_random_game(seed, max_steps)` with simple random legal action policy.
- [ ] Implement CLI args `--games`, `--seed`, `--max-steps`.
- [ ] Run `python -m pytest -v` and `python tools/random_playout.py --games 100 --seed 1`.
