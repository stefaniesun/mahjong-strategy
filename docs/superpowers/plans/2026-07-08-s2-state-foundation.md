# S2 State Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the S2 `s2.v4` protocol foundation and S1 engine adapter for player-perspective observations.

**Architecture:** Add a separate `state/` package so S2 state representation stays decoupled from the S1 engine. The first batch implements protocol dataclasses, JSON round-trip, visibility-safe player views, deterministic seen-count extraction, and protocol documentation.

**Tech Stack:** Python 3.10+, dataclasses, stdlib JSON-compatible dicts, pytest, existing `engine.*` domain classes.

---

## File Structure

- `state/__init__.py`: package marker and public exports.
- `state/protocol.py`: `ObservedValue`, protocol dataclasses, schema validation, JSON-compatible serialization/deserialization.
- `state/visibility.py`: relative-seat helpers and visibility-safe conversion primitives.
- `state/adapters/__init__.py`: adapter package marker.
- `state/adapters/from_engine.py`: convert S1 `GameState` into `S2ProtocolState` for one player.
- `tests/test_s2_protocol.py`: protocol observability and round-trip tests.
- `tests/test_s2_from_engine.py`: engine adapter and visibility red-line tests.
- `docs/protocol_spec.md`: field semantics, observability states, and JSON examples.

### Task 1: Protocol tests

**Files:**
- Create: `tests/test_s2_protocol.py`
- Create later: `state/protocol.py`

- [ ] **Step 1: Write failing tests** for `ObservedValue` semantics and round-trip.
- [ ] **Step 2: Run** `python -m pytest tests/test_s2_protocol.py -q`; expect import failure for missing `state.protocol`.
- [ ] **Step 3: Implement minimal protocol dataclasses** with `to_dict()` / `from_dict()`.
- [ ] **Step 4: Re-run** `python -m pytest tests/test_s2_protocol.py -q`; expect pass.

### Task 2: Engine adapter tests

**Files:**
- Create: `tests/test_s2_from_engine.py`
- Create later: `state/visibility.py`
- Create later: `state/adapters/from_engine.py`

- [ ] **Step 1: Write failing tests** proving player relative order, own hand visibility, opponent hidden hands, revealed winning hands, and `seen_counts` behavior.
- [ ] **Step 2: Run** `python -m pytest tests/test_s2_from_engine.py -q`; expect import failure for missing adapter.
- [ ] **Step 3: Implement visibility helpers and engine adapter**.
- [ ] **Step 4: Re-run** adapter tests; expect pass.

### Task 3: Documentation and regression

**Files:**
- Create: `docs/protocol_spec.md`

- [ ] **Step 1: Document** `ObservedValue`, facts/statistics/beliefs, full observation, mid-game unknowns, and estimated noisy fields.
- [ ] **Step 2: Run** `python -m pytest tests/test_s2_protocol.py tests/test_s2_from_engine.py -q`.
- [ ] **Step 3: Run** `python -m pytest -q` to confirm S1 remains green.
