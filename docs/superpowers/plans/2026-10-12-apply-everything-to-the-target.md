# Target-Directed Search with a Loop-Only Permission Rule — Implementation Plan

> **⚠ SUPERSEDED (Tasks 4–7 only):** the design was simplified to a single engine — see `docs/superpowers/plans/2026-10-12-single-engine.md`. Tasks 1–3 below are complete, reviewed, and unchanged.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach `simplify(expr)` (the target) using every move we have — formulas in both orientations, anywhere in the tree, methods as leaps — **without designing the path**: no cost steering, no expression-quality pruning. The only permission rule is that **no step may land on a state this route has already shown** (whole-expression history **and** node lineage). When nothing reaches the target, record a gap carrying everything we tried.

**Architecture:** `_target_route` becomes a plain **breadth-first search** (FIFO, shortest route first) over a global `visited` set — the visited set is what keeps it finite, since we no longer prune by quality. **The target is a parameter**: `simplify(expr)` is the default source, and the caller may supply one instead (`POST /api/solve {"target": ...}`) for anything `simplify` does not answer — geometry, a specified normal form, an exercise where the answer is dictated. `_build_steps` prefers the classroom route if it already lands on the target, else the BFS path, else the classroom route plus a gap. `_SolveState` gains **node lineage**: each displayed step links the subtree values it introduced to the values it removed, and a move that would swap a node back onto a linked earlier value is refused — the term-level version of "don't land on an earlier step". The old `blocked()` guard ("formula output may not be rewritten") is deleted; history replaces it. SymPy keeps three jobs only: it *supplies the target when we don't*, *verifies* steps (`simplify(a-b)==0` via the gate), and appears as a *hint* move at the last depth.

**Tech Stack:** Python 3 / SymPy, SQLite knowledge base (`entries`, `group_members`, `function_index`, `gaps`), FastAPI, React + TypeScript, pytest.

**Project rules that override the skill defaults:**
- **No commits unless the user explicitly asks.** Every "Commit" step below must be skipped unless the user has said to commit.
- **No cost-based steering anywhere.** `cost()`/`count_ops` may not rank, prune, or choose. The one retained exception is pre-existing: when *nothing* reaches the target, the fallback still compares TEACH vs THINK by cost (see Task 6, "retained deliberately"). Removing that is a separate one-line decision, not part of this plan.
- If an existing test fails after a change: **stop and report it** — never edit an assertion without approval.
- Never kill running servers; never rebuild `data/knowledge.db` except where Task 7 says so.

---

## File structure

| File | Responsibility |
|---|---|
| `src/knowledge/ops.py` | Method-op catalogue. **Task 1 (done).** |
| `data/samples/algebra.jsonl` | Shipped formulas/methods. **Task 2 (row present, unverified).** |
| `src/algebra/steps.py` | `_SolveState` (history + **lineage**), `_apply_once`, `_iter_changes` (both orientations, no `blocked`), `_target_route` (**BFS + method leaps**), `_build_steps` (teach → BFS → gap). |
| `tests/test_knowledge.py` | All tests. |

---

### Task 1: Two new method operations — `to_sin_cos`, `together` — ✅ DONE, spec-reviewed, quality-approved (130 passed)

- [x] All steps complete. `src/knowledge/ops.py` gains `together` → "Take the LCM" and `to_sin_cos` → "Write in terms of sin and cos"; `tests/test_knowledge.py::test_elementary_trig_operations_exist` passes.

---

### Task 2: The elementary method row (data the user owns) — ✅ DONE, verified (131 passed)

**Files:**
- `data/samples/algebra.jsonl` (one appended line — **already on disk from an interrupted run**)
- Test: `tests/test_knowledge.py` (**test already on disk, unverified**)

- [x] **Step 1: Verify what is already on disk** → `1 passed` (2026-10-12)

Run: `python -m pytest tests/test_knowledge.py::test_all_in_sin_and_cos_method_reaches_two -q` (workdir `D:\MathsLang`)

- If **`1 passed`** → go to Step 3.
- If **FAIL or not found** → apply Step 2, then re-run.

- [ ] **Step 2: (only if Step 1 failed) Apply the spec**

Test (append to `tests/test_knowledge.py`):

```python
def test_all_in_sin_and_cos_method_reaches_two(sample_formulas):
    """A stored method, not a SymPy jump, lands on SymPy's answer."""
    from src.algebra.latex import to_latex

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    rows = store.find_methods(store.get_connection(), expr, for_ways=False)
    row = next((r for r in rows if r["name"] == "All in sin and cos"), None)
    assert row is not None, "the method must be offered for this expression"

    steps, final = runner.run_method(expr, row["steps"], with_final=True)
    assert [s["name"] for s in steps] == [
        "Initial expression",
        "Write in terms of sin and cos",
        "Take the LCM",
        "Simplify",
    ]
    assert to_latex(final) == "2"
```

Append to `data/samples/algebra.jsonl` (after the `Tidy Trigonometry` line):

```json
{"kind":"method","name":"All in sin and cos","steps":[{"op":"to_sin_cos"},{"op":"together"},{"op":"simplify"}],"origin":"hand"}
```

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass (130 before Task 2's test; 131 with it). On any failure → **stop and report**.

- [ ] **Step 4: Commit (ONLY if the user asked)**

---

### Task 3: Formulas in both directions, from every source

**Files:** `src/algebra/steps.py` (`_stored_candidates` ~654, `_stored_hits` ~668); test in `tests/test_knowledge.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_stored_formulas_are_offered_backwards():
    """A = B must be usable as B -> A wherever the right-hand side shows up."""
    from src.algebra.latex import to_latex
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    expr = parse_expression("sin(x)/cos(x)")
    state = _SolveState(expr)   # the teaching walk: with_derived=False

    hits = [(name, new) for new, name, _f, _r in
            _iter_changes(expr, SIMPLIFY_RULES, True, state, with_derived=False)]

    assert any(name == "Tangent as a Quotient (backwards)" for name, _ in hits)
    assert any(to_latex(new) == "\\tan{\\left(x \\right)}" for _, new in hits)


def test_a_formula_can_fire_inside_a_larger_expression():
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    expr = parse_expression("2*(sin(y)^2 + cos(y)^2)")
    state = _SolveState(expr)

    names = [name for _new, name, _f, _r in
             _iter_changes(expr, SIMPLIFY_RULES, True, state, with_derived=False)]

    assert "Pythagorean Identity" in names
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_knowledge.py::test_stored_formulas_are_offered_backwards tests/test_knowledge.py::test_a_formula_can_fire_inside_a_larger_expression -q`
Expected: first FAILS (`Tangent as a Quotient (backwards)` absent); second may already pass (child recursion exists) — keep it as the guard for that behaviour.

- [ ] **Step 3: Widen the candidate set** — replace `_stored_candidates` in `src/algebra/steps.py` with:

```python
def _stored_candidates(expr: Expr) -> list:
    """Rows that could apply here: shape match, family, then functions.

    Each source gets its OWN budget. A shared cap let the shape matches
    (14 rows for an expression full of sin/cos) leave two slots for the
    function pass -- which is how a row reachable only from its other side
    (sin(x)/cos(x) -> "Tangent as a Quotient") went missing.

    The function pass is read with a wide limit and chosen in stable id
    order, so this source does not change contents depending on how often
    earlier solves happened to bump other rows (usage ordering made the
    same input produce different candidates in run 1 and run 10).
    """
    connection = knowledge_store.get_connection()
    candidates = list(knowledge_store.find_identities(connection, expr))
    if len(candidates) > _STORED_ROW_CAP:
        candidates = candidates[:_STORED_ROW_CAP]

    seen_ids = {row["id"] for row in candidates}
    for row in knowledge_store.find_by_groups(connection, classify(expr)):
        if len(candidates) >= _STORED_ROW_CAP:
            break
        if row["id"] not in seen_ids:
            candidates.append(row)
            seen_ids.add(row["id"])

    matching = knowledge_store.find_by_functions(
        connection, mentioned_functions(expr), limit=_FUNCTION_FETCH)
    added = 0
    for row in sorted(matching, key=lambda r: r["id"]):
        if added >= _FUNCTION_ROW_CAP:
            break
        if row["id"] in seen_ids:
            continue
        candidates.append(row)
        seen_ids.add(row["id"])
        added += 1
    return candidates
```

Add the budgets beside the existing `_DERIVE_ROW_CAP` (~line 691):

```python
# Per-source budgets: a later source may not be starved by an earlier one.
_STORED_ROW_CAP = 16     # shape + family matches
_FUNCTION_ROW_CAP = 8    # rows found only through their functions
_FUNCTION_FETCH = 500    # rows read before choosing (a bound, not a rank)
```

- [ ] **Step 4: Apply both directions, with forwards keeping precedence**

Three linked pieces. The direction must be threaded through the whole tree: per-node ordering is not enough, because a backwards move at an *earlier* sibling would otherwise shadow a forward move at a later one (observed: `2*sin(x)^2 + cos(2*x)` losing its `Double Angle for Cosine` opening).

**4a. `_iter_changes` gains `direction`.** Signature becomes:

```python
def _iter_changes(expr: Expr, rules: list[Rule], include_db: bool,
                  state: _SolveState, with_derived: bool,
                  direction: str = "both"):
```

It passes `direction` to `_stored_hits(expr, state, direction)` and unchanged through **both** recursive calls (the product sum-factors block and the "one level deeper" block). Add `from itertools import chain` to the module imports.

**4b. `_stored_hits` takes the direction:**

```python
def _stored_hits(expr: Expr, state: _SolveState, direction: str = "both"):
    """Applicable STORED formulas, in trust order -- BOTH directions.

    A = B is two moves:  A -> B  (the stored orientation) and  B -> A
    (the same row, applied the other way). `direction` selects which pass
    this is: "forward", "backwards", or "both" (forwards then backwards)
    when a caller does not care. The backwards move gets its own name so
    bans and "other ways" can refer to it.
    """
    try:
        candidates = _stored_candidates(expr)
    except Exception:
        return  # no database: no formula changes (the rules still work)

    parsed = []
    for row in candidates:
        try:
            left, right = knowledge_patterns.parse_pair(row["left"], row["right"])
        except Exception:
            continue
        parsed.append((row, left, right))

    phases = {"forward": (False,), "backwards": (True,),
              "both": (False, True)}[direction]
    for backwards in phases:
        for row, left, right in parsed:
            source, target, name = (
                (right, left, f"{row['name']} (backwards)") if backwards
                else (left, right, row["name"])
            )
            if name in state.banned:
                continue
            try:
                new_expr = knowledge_patterns.apply_identity(expr, source, target)
            except Exception:
                continue
            if new_expr is None or to_latex(new_expr) == to_latex(expr):
                continue
            if any(isinstance(hole, Wild) for hole in new_expr.free_symbols):
                continue  # a hole stayed unfilled: not a real rewrite
            yield new_expr, name, row["statement"], row["id"]
```

**4c. The three call sites enumerate forwards first, then backwards.** Each builds its iterable as:

```python
    phases = chain(
        _iter_changes(expr, rules, include_db, state, with_derived=...,
                      direction="forward"),
        _iter_changes(expr, rules, include_db, state, with_derived=...,
                      direction="backwards"),
    )
```

— in `_apply_once` (`with_derived=False`), in `_best_improvement`, and as `candidates` in `_target_route`'s inner loop. Only those loop headers change; bodies stay as they are. `_apply_once` checks **history only** in this task — the `lineage_forbids` half arrives with Task 4.

Add `Wild` to the module's sympy import (line 31): `from sympy import Add, Mul, Wild, cancel, simplify, srepr, trigsimp`

- [ ] **Step 5: Run the two tests** → Expected `2 passed`

Run: `python -m pytest tests/test_knowledge.py::test_stored_formulas_are_offered_backwards tests/test_knowledge.py::test_a_formula_can_fire_inside_a_larger_expression -q`

- [ ] **Step 6: Risk step — whole suite**

Run: `python -m pytest -q`
Expected: all pass. On failure → **stop and report** the test name and what the new move changed.

- [ ] **Step 7: Commit (ONLY if asked)**

---

### Task 4: History is the only permission rule (delete `blocked`, add node lineage)

**Files:** `src/algebra/steps.py` — `_SolveState` (~59-134), `_apply_once` (~513), `_teach` (~425), `_iter_changes` (~529-630); tests.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_formula_made_parts_are_not_off_limits():
    """History decides what may rewrite a part -- not who produced it."""
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    original = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    state = _SolveState(original)   # sin(x)+cos(x)-1 has never been shown

    node = parse_expression("sin(x) + cos(x) - 1")   # what an LCM step produced
    names = [name for _new, name, _f, _r in
             _iter_changes(node, SIMPLIFY_RULES, True, state, with_derived=False)]
    assert "Combine a sin + b cos" in names


def test_a_step_may_not_return_to_an_earlier_line():
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _apply_once

    state = _SolveState(parse_expression("sin(2*x)"))   # already an earlier line
    hit = _apply_once(parse_expression("2*sin(x)*cos(x)"),
                      SIMPLIFY_RULES, True, state)

    if hit is not None:
        assert not state.revisits(hit[0])   # the natural move here is sin(2x)


def test_teach_walk_gets_past_formula_output():
    """The reported bug: the walk died with sin+cos-1 untouched."""
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _teach

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    steps, _final = _teach(expr, SIMPLIFY_RULES, True, frozenset())

    assert len(steps) >= 12   # it stopped at exactly 11 lines before this fix


def test_a_node_may_not_be_rewritten_back_to_a_value_it_had():
    """Node lineage: a term may not swap back onto a value it held earlier,
    even when the rest of the expression has moved on (so whole-expression
    history cannot see the loop)."""
    from src.algebra.steps import _SolveState

    state = _SolveState(parse_expression("sin(x)^2 + cos(x)^2 + z"))
    state.record(parse_expression("1 + z"))   # the route displayed: ... -> 1 + z

    going_back = parse_expression("sin(x)^2 + cos(x)^2 + w")
    unrelated = parse_expression("1 + w")

    assert state.lineage_forbids(parse_expression("1 + z"), going_back) is True
    assert state.lineage_forbids(parse_expression("1 + z"), unrelated) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_knowledge.py -k "formula_made_parts or earlier_line or past_formula or node_may_not" -q`
Expected: `4 failed` (`_SolveState` has no `lineage_forbids`, the guard still blocks, the walk still stops at 11, `_apply_once` still returns `sin(2x)`).

- [ ] **Step 3: Node lineage in `_SolveState`**

In `__init__`, after `self.history_srepr: set = set()` add:

```python
        self.lineage: dict = {}   # subtree srepr -> subtree values it has swapped with
```

Add the two methods (after `revisits`):

```python
    def record(self, expr: Expr, display_latex: str = "") -> None:
        """Remember a state this route has shown, and link the term values
        it swapped (the node-level trail: what replaced what)."""
        previous = self.history[-1] if self.history else None

        self.history.append(expr)
        self.history_latex.add(to_latex(expr))
        self.history_srepr.add(srepr(expr))
        if display_latex:
            self.history_latex.add(display_latex)

        if previous is None:
            return
        introduced = _sreprs(expr) - _sreprs(previous)
        removed = _sreprs(previous) - _sreprs(expr)
        for new_value in introduced:
            for old_value in removed:
                self.lineage.setdefault(new_value, set()).add(old_value)
                self.lineage.setdefault(old_value, set()).add(new_value)

    def lineage_forbids(self, previous: Expr, candidate: Expr) -> bool:
        """Would this move swap a term back onto an earlier value it held?

        Whole-expression history cannot see this loop: the rest of the
        expression may have moved on, so only the term itself returns.
        """
        introduced = _sreprs(candidate) - _sreprs(previous)
        removed = _sreprs(previous) - _sreprs(candidate)
        return any(
            old in self.lineage.get(new, ())
            for new in introduced
            for old in removed
        )
```

(`_sreprs` is already defined in this module at ~line 316; a module-level call from a method is fine.)

**Then delete the whole `blocked` method** (~113-134, docstring included) — nothing may call it any more.

- [ ] **Step 4: `_apply_once` enforces history + lineage**

Task 3 already restructured `_apply_once` into the chained forward/backwards phases (with a commented "Task 4 insertion point" where the lineage check goes). The final form — replace it with:

```python
def _apply_once(expr: Expr, rules: list[Rule], include_db: bool,
                state: _SolveState):
    """The first change the permission rule allows.

    History and node lineage are the ONLY gates: a step may rewrite
    anything -- including parts a formula produced -- as long as it does
    not land on an earlier line of this route, and no term swaps back onto
    a value it already held. Refused candidates are SKIPPED, not fatal:
    the walk continues with the next move. Two phases: every forward
    orientation in the whole tree, then every backwards one.
    """
    phases = chain(
        _iter_changes(expr, rules, include_db, state, with_derived=False,
                      direction="forward"),
        _iter_changes(expr, rules, include_db, state, with_derived=False,
                      direction="backwards"),
    )
    for change in phases:
        if state.revisits(change[0]):
            continue  # lands on an earlier line
        if state.lineage_forbids(expr, change[0]):
            continue  # a term swaps back onto an earlier value
        return change
    return None
```

Delete the "Task 4 insertion point" comment that Task 3 left behind.

- [ ] **Step 5: Remove the now-redundant check in `_teach`** (~431) so it reads:

```python
        hit = _apply_once(current, rules, include_db, state)
        if hit is None:
            break  # nothing left to do -- we are done

        new_expr, name, formula, row_id = hit
        step: Step = {"name": name, "latex": to_latex(new_expr)}
```

- [ ] **Step 6: Stop gating moves on `formula_made_this` in `_iter_changes`** (~529)

Delete `formula_made_this = state.blocked(expr)` and change these five sites:

| site | becomes |
|---|---|
| `if include_db and not formula_made_this:` | `if include_db:` |
| `if with_derived and not formula_made_this and _is_trig_node(expr) \` | `if with_derived and _is_trig_node(expr) \` |
| `if not formula_made_this:` | drop the `if`, unindent its body (it precedes the `for rule in shape_rules:` loop) |
| `if with_derived and include_db and not formula_made_this \` | `if with_derived and include_db \` |
| `if with_derived and not formula_made_this:` | `if with_derived:` |

- [ ] **Step 7: Run the four new tests** → Expected `4 passed`

Run: `python -m pytest tests/test_knowledge.py -k "formula_made_parts or earlier_line or past_formula or node_may_not" -q`

- [ ] **Step 8: Risk step — whole suite**

Run: `python -m pytest -q`
Expected: all pass. On failure → **stop and report** (most likely a route that changed because formula output is now rewritable, or a lineage refusal that a walk depended on).

- [ ] **Step 9: Commit (ONLY if asked)**

---

### Task 5: Methods as leaps inside the search

**Files:** `src/algebra/steps.py` (new `_method_leaps`, wired into `_target_route`); tests.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_a_method_leaps_straight_to_the_target(sample_formulas):
    """One method move reaches the target; its own sub-steps are what we show."""
    from src.algebra.steps import _method_leaps

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    leaps = _method_leaps(expr, frozenset())

    named = {leap["name"]: leap for leap in leaps}
    assert "All in sin and cos" in named
    assert to_latex(named["All in sin and cos"]["final"]) == "2"
    # the leap must carry the steps we would display for it
    assert [s["name"] for s in named["All in sin and cos"]["steps"]] == [
        "Initial expression", "Write in terms of sin and cos",
        "Take the LCM", "Simplify",
    ]
```

(`to_latex` is used elsewhere in this file — import it locally in the test: `from src.algebra.latex import to_latex`.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_knowledge.py::test_a_method_leaps_straight_to_the_target -q`
Expected: FAIL with `ImportError`/`NameError` (`_method_leaps` missing).

- [ ] **Step 3: Implement** — add after `_method_ways` (~219):

```python
def _method_leaps(expr: Expr, banned: "frozenset[str]") -> list:
    """Applicable stored methods run as ONE candidate move each.

    A method compresses a long procedure into a single edge of the search
    -- that is how a deep elementary route still fits the depth limit.
    The sub-steps ride along so that choosing this leap shows the method's
    own named lines instead of one opaque jump.

    Each leap: {"name", "final", "steps", "row_id"}.
    """
    try:
        connection = knowledge_store.get_connection()
        rows = knowledge_store.find_methods(connection, expr, for_ways=False,
                                            limit=8)
    except Exception:
        return []

    leaps: list = []
    for row in rows:
        if row["name"] in banned:
            continue
        try:
            steps, final = _method_runner.run_method(expr, row["steps"],
                                                     with_final=True)
        except Exception:
            continue  # this method does not fit this expression
        if len(steps) < 2 or to_latex(final) == to_latex(expr):
            continue  # it did nothing
        leaps.append({"name": row["name"], "final": final, "steps": steps,
                      "row_id": row["id"]})
    return leaps
```

Note: `store.find_methods(connection, expr, for_ways=False)` already has a `limit` parameter (default 20); passing `limit=8` is a bound, not a preference.

- [ ] **Step 4: Run the test** → Expected `1 passed`

- [ ] **Step 5: Risk step — whole suite** → all pass, else **stop and report**.

- [ ] **Step 6: Commit (ONLY if asked)**

---

### Task 6: Breadth-first search, target-first selection, gaps that carry what we tried

**Files:** `src/algebra/steps.py` (`_target_route` ~338, `_build_steps` ~235, `_distance` ~327 to delete); tests.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_search_reaches_the_target_without_a_quality_prune(sample_formulas):
    """BFS, not a cost heuristic, is what gets us there."""
    from src.algebra.steps import _target_route

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    target = simplify(expr)
    moves, final, tried = _target_route(expr, target, SIMPLIFY_RULES, True,
                                        frozenset())

    assert moves is not None
    assert to_latex(final) == "2"


def test_main_route_shows_the_methods_steps(sample_formulas):
    steps = steps_for("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    assert [s["name"] for s in steps] == [
        "Initial expression",
        "Write in terms of sin and cos",
        "Take the LCM",
        "Simplify",
    ]
    assert steps[-1]["latex"] == "2"


def test_gap_still_records_what_was_tried(sample_formulas):
    known = {row["target"] for row in store.list_gaps()}
    steps_for("(x^2 - 1)*(x^2 - 4)/(x^2 - 2*x + 1)")
    fresh = [row for row in store.list_gaps() if row["target"] not in known]

    if fresh:   # reached -> no gap is correct too; only check the content
        assert fresh[0]["tried"]          # something must be listed
```

(`SIMPLIFY_RULES` needs a local import: `from src.algebra.rules import SIMPLIFY_RULES`; `simplify` already imported at module level of this test file? if not: `from sympy import simplify`.)

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_knowledge.py -k "quality_prune or methods_steps or still_records" -q`
Expected: FAIL (route still chosen by the old cost-ordered beam / `Simplify trigonometry`).

- [ ] **Step 3: Replace the beam with breadth-first search** — rewrite `_target_route` (~338) as:

```python
# Search limits (limits, never preferences): how far we look and how many
# candidates we may expand before giving up and recording a gap.
TARGET_DEPTH = 5
SEARCH_BUDGET = 1200

# SymPy's own moves are a HINT, explored only at the last depth so that
# our named strategies always get first chance at every level.
_SYMPY_HINTS = frozenset({
    "Simplify with SymPy",
    "Simplify trigonometry",
    "Write with sin/cos and cancel",
})


def _target_route(expr: Expr, target: Expr, rules: list[Rule],
                  include_db: bool, banned: "frozenset[str]"
                  ) -> "tuple[list | None, Expr | None, set]":
    """Breadth-first search from expr to the target's form.

    No quality ordering: candidates are expanded in the order they are
    generated (formulas and rules first, then method leaps), shallow
    routes before deep ones. The global `visited` set is what keeps this
    finite -- never expand a state twice, anywhere in the search.

    Returns (moves, final, tried): moves are the Step dicts after the
    initial line (None when nothing reached the target), tried is every
    strategy name attempted -- formulas, methods and hints alike.
    """
    from collections import deque

    state = _SolveState(expr, banned)
    tried: set = set()
    target_latex = to_latex(target)
    target_srepr = srepr(target)

    visited = {to_latex(expr), to_latex_as_written(expr), srepr(expr)}
    queue = deque([(expr, [], 0)])   # (expression, moves so far, depth)
    budget = SEARCH_BUDGET

    while queue and budget > 0:
        node, moves, depth = queue.popleft()
        if depth >= TARGET_DEPTH:
            continue

        candidates = list(_iter_changes(node, rules, include_db, state,
                                        with_derived=True))
        if include_db:
            candidates.extend(
                (leap["final"], leap["name"], None, leap["row_id"])
                for leap in _method_leaps(node, banned)
            )

        for new_expr, name, formula, row_id in candidates:
            budget -= 1
            if budget <= 0:
                break
            tried.add(name)

            if name in _SYMPY_HINTS and depth < TARGET_DEPTH - 1:
                continue   # hints wait; named strategies go first

            latex = to_latex(new_expr)
            if latex in visited or srepr(new_expr) in visited:
                continue   # the loop rule, applied to the whole search

            if latex == target_latex or srepr(new_expr) == target_srepr:
                # choosing a method leap: show the method's own lines
                if name in {leap["name"] for leap in
                            (_method_leaps(node, banned) if include_db else [])}:
                    steps = _method_leap_steps(node, name)
                    route = moves + steps if steps else \
                        moves + [{"name": name, "latex": latex,
                                  "formula": formula}]
                else:
                    route = moves + [{"name": name, "latex": latex,
                                      "formula": formula}]
                return route, new_expr, tried

            visited.add(latex)
            visited.add(srepr(new_expr))
            queue.append((new_expr, moves + [{"name": name, "latex": latex,
                                              "formula": formula,
                                              "_row": row_id}], depth + 1))

    return None, None, tried
```

Add the helper used above (cache `_method_leaps` results per node by calling it once and passing the list through — restructure the loop as follows to avoid calling it twice: compute `leaps = _method_leaps(node, banned) if include_db else []` **once** before `candidates`, keep a `leap_names = {leap["name"]: leap for leap in leaps}`, extend candidates from it, and at the hit use `leap_names[name]["steps"][1:]` for the route):

```python
        # inside the hit branch, for a method leap:
        leap = leap_names.get(name)
        if leap is not None:
            sub_steps = [{"name": s["name"], "latex": s["latex"],
                          **({"formula": s["formula"]} if s.get("formula") else {})}
                         for s in leap["steps"][1:]]   # drop its Initial line
            route = moves + sub_steps
        else:
            route = moves + [{"name": name, "latex": latex,
                              "formula": formula}]
```

Finally **delete `_distance`** (~327) and the `cost`-based sort it served — they are dead once the queue is FIFO.

- [ ] **Step 4: Rework `_build_steps`** (~235) so cost never picks the path:

```python
def _build_steps(expr: Expr, rules: list[Rule], include_db: bool = True,
                 banned: "frozenset[str]" = frozenset()
                 ) -> "tuple[list[Step], Expr]":
    """Returns (steps, final expr): the best route we can explain.

    Target-first, with no cost steering:
      A. the classroom route, if it already lands on SymPy's target;
      B. otherwise the breadth-first path to the target (shortest first);
      C. otherwise the classroom route, plus a gap naming everything tried.

    No target (SymPy agrees the input is already the answer): TEACH vs
    THINK as before -- that comparison is display selection between two
    routes neither of which reaches anything, and is retained unchanged.
    """
    teaching_steps, teaching_final = _teach(expr, rules, include_db, banned)

    if include_db:
        target = simplify(expr)
        if to_latex(target) != to_latex(expr):
            teach_reached = (
                to_latex(teaching_final) == to_latex(target)
                or srepr(teaching_final) == srepr(target)
            )
            if not teach_reached:
                route_moves, target_final, tried = _target_route(
                    expr, target, rules, include_db, banned)
                if route_moves is not None:
                    steps = [_initial_step(expr)] + [
                        {k: v for k, v in move.items() if not k.startswith("_")}
                        for move in route_moves
                    ]
                    if (len(route_moves) == 1
                            and route_moves[0]["name"] == "Simplify with SymPy"):
                        _record_gap(target, expr, tried)
                    return steps, target_final
                if len(tried) >= 3:
                    _record_gap(target, expr, tried)

    thinking_steps, thinking_final = _think(expr, rules, include_db, banned)

    if _route_wins(thinking_steps, thinking_final,
                   teaching_steps, teaching_final):
        if _same_answer(thinking_final, expr):   # soundness net
            return thinking_steps, thinking_final
    return teaching_steps, teaching_final
```

What disappears: `teach_is_better = cost(teaching_final) < cost(target_final)` — cost may no longer decide between a reaching route and the classroom route.

- [ ] **Step 5: Accept a supplied target (when `simplify` is not the answer)**

The target becomes a parameter all the way down. In `src/algebra/steps.py`:

```python
def simplify_steps(expr: Expr, include_db: bool = True,
                   target: "Expr | None" = None) -> list[Step]:
    """Give the list of steps needed to simplify this expression.

    target: the answer to reach. Defaults to SymPy's simplify(expr); pass
    one explicitly when the answer is not something simplify() knows
    (another domain, a specified form, a dictated exercise answer).
    """
    steps, _ = _build_steps(expr, SIMPLIFY_RULES, include_db, target=target)
    return steps
```

In `_build_steps`, widen the signature to `..., target: "Expr | None" = None)` and replace the head of the target block:

```python
    if include_db:
        if target is None:
            target = simplify(expr)   # unless the caller supplied one
        if to_latex(target) != to_latex(expr):
```

and thread it through `alternative_ways(expr, include_db=True, limit=4, target=None)` → `_build_steps(expr, SIMPLIFY_RULES, include_db, target=target)`.

In `api/main.py`:

```python
class SolveRequest(BaseModel):
    expression: str  # what the user typed, e.g. "2*x + 3*x + 5 - 1"
    mode: str = "simplify"  # "simplify" or "solve"
    target: str | None = None  # the answer to reach; defaults to simplify()
```

and in `solve()`:

```python
        elif request.mode == "simplify":
            expression = parse_expression(request.expression)
            target = (parse_expression(request.target)
                      if request.target else None)
            steps = simplify_steps(expression, target=target)
            # Other routes that finish at the same answer (simplify only).
            ways = alternative_ways(expression, target=target)
```

Tests (append):

```python
def test_a_supplied_target_overrides_sympys_answer(sample_formulas):
    """SymPy would say sin(2x) is already simple; the caller may instead
    ask for 2*sin(x)*cos(x) -- and the search must reach THAT."""
    from src.algebra.latex import to_latex
    from src.algebra.steps import simplify_steps

    expr = parse_expression("sin(2*x)")
    target = parse_expression("2*sin(x)*cos(x)")
    steps = simplify_steps(expr, target=target)

    assert steps[-1]["latex"] == to_latex(target)
    assert [s["name"] for s in steps] == [
        "Initial expression", "Double Angle for Sine"]
```

and in `tests/test_api.py` (where the TestClient lives):

```python
def test_solve_accepts_an_explicit_target():
    response = client.post(
        "/api/solve",
        json={"expression": "sin(2*x)", "mode": "simplify",
              "target": "2*sin(x)*cos(x)"},
    )
    assert response.status_code == 200
    assert response.json()["steps"][-1]["name"] == "Double Angle for Sine"
```

Run: `python -m pytest tests/test_knowledge.py::test_a_supplied_target_overrides_sympys_answer tests/test_api.py::test_solve_accepts_an_explicit_target -q`
Expected: `2 passed`

- [ ] **Step 6: Run the three original new tests** → Expected `3 passed`

- [ ] **Step 7: Risk step — whole suite** → all pass, else **stop and report** (expected suspects: a route that now reaches the target through BFS where a test pinned the old route; the gap test if BFS now reaches the rational target).

- [ ] **Step 8: Commit (ONLY if asked)**

---

### Task 7: Full verification

- [ ] **Step 1:** `python -m pytest -q` → all passed (report the count).
- [ ] **Step 2:** `npx tsc --noEmit` → exits 0.
- [ ] **Step 3:** `npm run build` → `✓ built in ...`.
- [ ] **Step 4: Sync the live database** (the user has approved re-imports; `data/samples/algebra.jsonl` now carries the new method row):

```powershell
Copy-Item -LiteralPath data\knowledge.db -Destination data\knowledge.db.bak-2026-10-12b -Force
python scripts/import_formulas.py data/samples/algebra.jsonl data/samples/trigonometry.jsonl data/samples/trigonometric_equations.jsonl data/samples/trigonometry_reference.jsonl
```

Expected: `imported 1` (the method row), `rejected 0`, total grows by 1; the 12 existing methods and gap rows survive.

- [ ] **Step 5: Live smoke** — start `npm run dev`, wait for both ports, then:

```powershell
$r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/solve" -ContentType "application/json" -Body '{"expression":"(1 + cot(x) - csc(x)) * (1 + tan(x) + sec(x))","mode":"simplify"}'
$r.final    # expect: 2
$r.steps | ForEach-Object { "$($_.name) -> $($_.latex)" }
# expect: Initial expression, Write in terms of sin and cos, Take the LCM, Simplify
```

Also confirm `1 + tan(x) + sec(x)` still returns a route (walk no longer dies early — expect MORE than the old 3 lines).

- [ ] **Step 6:** Stop only the dev servers this session started (never kill unrelated uvicorn/vite).
- [ ] **Step 7:** Report to the user: test count, smoke output, anything that changed unexpectedly. **No commit unless asked.**

---

## Retained deliberately / out of scope

- **TEACH vs THINK cost comparison in the no-target fallback** — pre-existing behaviour, not path steering toward a target. Removing it is a one-line change if you want the pure version.
- **Cost/count_ops as compass** — removed entirely: no ranking, no pruning, no meter (per spec).
- A `right_signature` column so SQL selects rows by their right-hand shape directly; backwards reach currently comes from the function pass (`_STORED_ROW_CAP`).
- Node lineage applies to the teaching walk (where churn happens); the BFS is protected by its global `visited` set, where a term returning in a genuinely new context is allowed exploration.
