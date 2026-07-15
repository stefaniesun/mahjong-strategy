# S1 Kong Transfer and Pass Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement over-pass locking and kong-after-discard transfer scoring while keeping the existing random full-round loop stable and zero-sum.

**Architecture:** Extend the existing state/settlement modules with a small kong ledger and transient after-kong discard context. Keep game flow changes localized in `engine/game.py`: filter discard-win responses through pass-lock rules, record pass-lock fan, apply `after_kong` context and optional kong transfer when resolving a discard win.

**Tech Stack:** Python 3.10+, pytest, existing `engine` package.

---

### Task 1: Kong ledger and transfer settlement

**Files:**
- Modify: `engine/gang.py`
- Modify: `engine/settlement.py`
- Modify: `tests/test_settlement.py`

- [ ] Add tests for recording exposed/concealed/added kong payments and transfer.
- [ ] Extend `GangRecord` with transfer/refund audit fields.
- [ ] Add `record_kong_payment()` and `apply_kong_transfer()`.
- [ ] Keep `apply_kong_payment()` compatible with existing tests.
- [ ] Run `python -m pytest tests\test_settlement.py -v`.

### Task 2: State fields for pass lock and after-kong context

**Files:**
- Modify: `engine/state.py`
- Modify: `tests/test_game_loop_full.py`

- [ ] Add serialization assertions for kong ledger and after-kong context.
- [ ] Add `gang_records`, `next_gang_id`, `last_transferable_gang_id`, and `after_kong_discard_player` to `GameState`.
- [ ] Run `python -m pytest tests\test_game_loop_full.py::test_state_serializes_full_loop_fields -v`.

### Task 3: Over-pass lock behavior

**Files:**
- Modify: `engine/game.py`
- Modify: `tests/test_game_loop_full.py`

- [ ] Add tests for pass-lock set, lock filtering, draw clearing, and bigger-fan unlock.
- [ ] Compute discard-win fan with the same trial hand used by `can_win()`.
- [ ] Filter legal `WIN` responses if `passed_hu_lock[player]` is active and current fan is not greater than `passed_fan[player]`.
- [ ] On `PASS` from a player who could win, store the abandoned fan.
- [ ] Clear pass lock when that player draws.
- [ ] Run `python -m pytest tests\test_game_loop_full.py -v`.

### Task 4: After-kong discard scoring and transfer

**Files:**
- Modify: `engine/game.py`
- Modify: `tests/test_game_loop_full.py`

- [ ] Add tests for after-kong point increase, transferable kong money, added-kong no-transfer, and multi-win transfer copy semantics.
- [ ] Resolve discard wins with `WinContext(after_kong=True)` when the discard matches after-kong context.
- [ ] Apply `apply_kong_transfer()` for each winner only when `last_transferable_gang_id` references a paid kong.
- [ ] Clear after-kong context after the discard response window resolves.
- [ ] Run `python -m pytest tests\test_game_loop_full.py tests\test_settlement.py -v`.

### Task 5: Final verification

**Files:**
- Verify all touched files.

- [ ] Run `python -m pytest -v`.
- [ ] Run `python tools\random_playout.py --games 100 --seed 1 --max-steps 1000`.
- [ ] Confirm no linter errors in `engine` and `tests` touched by this change.
