# Single-Engine Simplification — Implementation Plan (Tasks 4–7, revision 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps for tracking.
> **Supersedes Tasks 4–7 of** `2026-10-12-apply-everything-to-the-target.md` (Tasks 1–3 there are DONE and unchanged).

**Goal:** One engine, four rules — *moves* = every formula (both ways, anywhere in the tree) + tidy/shape rules + methods + SymPy sub-pipelines; *permission* = never land on a state this route has already shown; *goal* = the caller's target, else `simplify(expr)` **only when it changes the input**; *exploration* = greedy walk stopping on the goal, then breadth-first if it misses, else a gap. **No goal → one suggestion step, never a walk.** Everything that was patching a second engine gets deleted.

**Tech Stack:** Python 3 / SymPy, SQLite, FastAPI, React + TS, pytest.

**Project rules:** no commits unless the user asks; **never edit an existing assertion without the user's approval** — stop and report (the anticipated re-scopes are listed in Task 7); never kill running servers; never rebuild `data/knowledge.db` except where Task 7 says.

---

## Deletion list (the point of this revision)

| Delete | Where | Why it's safe |
|---|---|---|
| `_think`, `_best_improvement`, `_collapse` | `src/algebra/steps.py` | the cost-gated second engine |
| `_route_wins` | same | only compared the two engines |
| `_branch_sreprs`, `lineage_forbids`, `self.lineage`, the linkage in `record()` | same | added this session for the greedy walk; BFS's `visited` + history suffice |
| `_distance` + the `next_beam.sort(...)` | same (`_target_route`) | the heuristic steering we removed by design |
| `_SYMPY_HINTS` set + its `if name in ... continue` gate | same | hints are ordinary candidates now |
| `blocked()` / `formula_made_this` | same | already deleted in the earlier Task 4 attempt — verify gone |
| `src/algebra/cost.py` | module | **only if** a grep shows no remaining importers (tests may import it — report, don't guess) |

---

### Task 4′: One permission rule + "no goal → one suggestion step"

**Files:** `src/algebra/steps.py`, `tests/test_knowledge.py`

- [ ] **Step 1: Reconcile the four tests added earlier this session**

- Keep (still valid): `test_formula_made_parts_are_not_off_limits`, `test_a_step_may_not_return_to_an_earlier_line`, `test_teach_walk_gets_past_formula_output`.
- **Delete** `test_a_node_may_not_be_rewritten_back_to_a_value_it_had` — it asserts `lineage_forbids`, which this revision deletes. (Test removal, listed for approval in Task 7.)

- [ ] **Step 2: Add the no-goal rule** — new helper in `steps.py`:

```python
def _one_suggestion(expr: Expr, rules: list[Rule], include_db: bool,
                    banned: "frozenset[str]") -> "tuple[list[Step], Expr]":
    """No goal to reach: show what ONE move can do, then stop.

    Without a target we have no way to tell a step forward from a step
    sideways, so we never walk -- one applicable move, clearly a
    suggestion, is the honest display.
    """
    state = _SolveState(expr, banned, to_latex_as_written(expr))
    hit = _apply_once(expr, rules, include_db, state)
    if hit is None:
        return [_initial_step(expr)], expr
    new_expr, name, formula, _row_id = hit
    step = {"name": name, "latex": to_latex(new_expr)}
    if formula:
        step["formula"] = formula
    return [_initial_step(expr), step], new_expr
```

- [ ] **Step 3: Run** `python -m pytest tests/test_knowledge.py -k "formula_made_parts or earlier_line or past_formula" -q` → `3 passed`

- [ ] **Step 4: Risk step** — `python -m pytest -q` → report failures (the lineage test should be the only new failure if Step 1 was done; anything else must be reported).

---

### Task 5′: Delete THINK, cost comparisons, and hint gating

**Files:** `src/algebra/steps.py`, `tests/*`

- [ ] **Step 1: Delete** `_think`, `_best_improvement`, `_collapse`, `_route_wins` (bodies gone entirely; no stubs).
- [ ] **Step 2: Delete `_SYMPY_HINTS`** and the gate `if name in _SYMPY_HINTS and depth < TARGET_DEPTH - 1: continue` from `_target_route` — SymPy moves become ordinary candidates.
- [ ] **Step 3: Rewrite `_build_steps` to the four-rule shape** (no cost anywhere):

```python
def _build_steps(expr: Expr, rules: list[Rule], include_db: bool = True,
                 banned: "frozenset[str]" = frozenset(),
                 target: "Expr | None" = None) -> "tuple[list[Step], Expr]":
    """One engine: walk to the goal, else search, else gap.

    goal   = the caller's target, else simplify(expr) if that changes it
    walk   = greedy (_teach), which STOPS on the goal
    search = breadth-first when the walk misses
    no goal = one suggestion step, never a walk
    """
    if target is None and include_db:
        candidate = simplify(expr)
        if to_latex(candidate) != to_latex(expr):
            target = candidate

    if target is None:
        return _one_suggestion(expr, rules, include_db, banned)

    teaching_steps, teaching_final = _teach(expr, rules, include_db, banned,
                                            target)
    if to_latex(teaching_final) == to_latex(target):
        return teaching_steps, teaching_final     # the walk arrived

    if include_db:
        route_moves, final, tried = _target_route(expr, target, rules,
                                                  include_db, banned)
        if route_moves is not None:
            steps = [_initial_step(expr)] + [
                {k: v for k, v in move.items() if not k.startswith("_")}
                for move in route_moves]
            return steps, final
        _record_gap(target, expr, tried)

    # nothing reached the goal: show how far the walk got
    return teaching_steps, teaching_final
```

(`simplify_steps` and `alternative_ways` keep their `target=None` signatures from the earlier spec.)

- [ ] **Step 4: Risk step** — `python -m pytest -q`; **STOP and report** every failure (think-dependent tests are expected casualties — list them, do not edit).

---

### Task 6′: Breadth-first search — FIFO, no heuristic, methods as leaps

**Files:** `src/algebra/steps.py`, tests

- [ ] **Step 1: Rewrite `_target_route` as BFS** (this supersedes the beam):

```python
# Search bounds -- limits, never preferences.
TARGET_DEPTH = 5
SEARCH_BUDGET = 1200


def _target_route(expr: Expr, target: Expr, rules: list[Rule],
                  include_db: bool, banned: "frozenset[str]"
                  ) -> "tuple[list | None, Expr | None, set]":
    """Breadth-first walk to the target's form. No quality ordering:
    shallow routes first, candidates in generation order (formulas and
    rules, then method leaps). The global visited set is what keeps it
    finite -- never expand a state twice, anywhere.
    """
    from collections import deque

    state = _SolveState(expr, banned)
    tried: set = set()
    target_latex = to_latex(target)
    target_srepr = srepr(target)
    visited = {to_latex(expr), to_latex_as_written(expr), srepr(expr)}
    queue = deque([(expr, [], 0)])
    budget = SEARCH_BUDGET

    while queue and budget > 0:
        node, moves, depth = queue.popleft()
        if depth >= TARGET_DEPTH:
            continue

        candidates = list(_iter_changes(node, rules, include_db, state,
                                        with_derived=True,
                                        direction="forward"))
        candidates += list(_iter_changes(node, rules, include_db, state,
                                         with_derived=True,
                                         direction="backwards"))
        if include_db:
            leap_names = {}
            for leap in _method_leaps(node, banned):
                leap_names[leap["name"]] = leap
                candidates.append((leap["final"], leap["name"], None,
                                   leap["row_id"]))
        else:
            leap_names = {}

        for new_expr, name, formula, row_id in candidates:
            budget -= 1
            if budget <= 0:
                break
            tried.add(name)
            latex = to_latex(new_expr)
            if latex in visited or srepr(new_expr) in visited:
                continue                       # the loop rule, search-wide

            if latex == target_latex or srepr(new_expr) == target_srepr:
                leap = leap_names.get(name)
                if leap is not None:
                    route = moves + [{"name": s["name"], "latex": s["latex"],
                                      **({"formula": s["formula"]}
                                         if s.get("formula") else {})}
                                     for s in leap["steps"][1:]]
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

- [ ] **Step 2: `_method_leaps`** (if not already present) — applicable methods run as one candidate each, carrying their sub-steps:

```python
def _method_leaps(expr: Expr, banned: "frozenset[str]") -> list:
    """Stored methods run as ONE candidate move each: a long procedure
    compressed into a single edge, so deep routes still fit the depth
    limit. Each leap: {"name", "final", "steps", "row_id"}."""
    try:
        connection = knowledge_store.get_connection()
        rows = knowledge_store.find_methods(connection, expr, for_ways=False,
                                            limit=8)
    except Exception:
        return []
    leaps = []
    for row in rows:
        if row["name"] in banned:
            continue
        try:
            steps, final = _method_runner.run_method(expr, row["steps"],
                                                     with_final=True)
        except Exception:
            continue
        if len(steps) < 2 or to_latex(final) == to_latex(expr):
            continue
        leaps.append({"name": row["name"], "final": final, "steps": steps,
                      "row_id": row["id"]})
    return leaps
```

- [ ] **Step 3: Tests**

```python
def test_bfs_reaches_the_target_with_named_steps(sample_formulas):
    from src.algebra.steps import _target_route
    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    moves, final, tried = _target_route(expr, simplify(expr), SIMPLIFY_RULES,
                                        True, frozenset())
    assert moves is not None and to_latex(final) == "2"


def test_main_route_shows_the_methods_steps(sample_formulas):
    steps = steps_for("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    assert [s["name"] for s in steps] == [
        "Initial expression", "Write in terms of sin and cos",
        "Take the LCM", "Simplify"]
    assert steps[-1]["latex"] == "2"


def test_no_goal_means_one_suggestion_not_a_walk():
    expr = parse_expression("sin(x) + cos(x)")   # simplify() leaves this alone
    steps = steps_for("sin(x) + cos(x)")
    assert len(steps) == 2                       # initial + exactly one move
    assert steps[1]["name"] == "Combine a sin + b cos"


def test_a_supplied_target_overrides_sympys_answer(sample_formulas):
    from src.algebra.steps import simplify_steps
    expr, target = parse_expression("sin(2*x)"), parse_expression("2*sin(x)*cos(x)")
    steps = simplify_steps(expr, target=target)
    assert steps[-1]["latex"] == to_latex(target)
```

- [ ] **Step 4: Risk step** — `python -m pytest -q`; **STOP and report**.

---

### Task 7′: Verification, approved re-scopes, live smoke

- [ ] **Step 1:** `python -m pytest -q` → report count.
- [ ] **Step 2 (approval gate):** list every failing assertion with before → after, get the user's approval, then make only those edits. **Anticipated:** `test_takes_the_lcm_and_finishes_the_chain` (final `sec²` → `1/cos²`), plus anything flagged in Task 5′/6′ risk steps.
- [ ] **Step 3:** `npx tsc --noEmit` → clean; `npm run build` → `✓ built`.
- [ ] **Step 4:** live DB sync (backup first):

```powershell
Copy-Item -LiteralPath data\knowledge.db -Destination data\knowledge.db.bak-2026-10-12c -Force
python scripts/import_formulas.py data/samples/algebra.jsonl data/samples/trigonometry.jsonl data/samples/trigonometric_equations.jsonl data/samples/trigonometry_reference.jsonl
```

- [ ] **Step 5: smoke** — `npm run dev`, then: `(1+cot−csc)(1+tan+sec)` → final `2` with the four named steps; `sin(x)+cos(x)` → exactly one suggestion step; `2*x+3*x+5-1` → reaches `5 x + 4`.
- [ ] **Step 6:** stop only this session's dev servers; report; **no commit unless asked**.

---

## Task 8: Strategies (authored routes: form -> ordered steps)

**User decisions this implements:** (1) dispatch lives in `steps.py` (`if form then run strategy`), strategy bodies in a NEW file `src/algebra/strategies.py`; (2) a termwise pass uses our tidy rules + stored formulas + the SymPy hint; (3) phase 1 depends on the question's form/subpart; (4) a goal always exists (SymPy's, else the caller's -- Task 9 wires the UI); (5) if two strategies reach, the SHORTER path is the main route, the other becomes an alternative way.

- [ ] **Step 1: Create `src/algebra/strategies.py`** -- every strategy is `{name, matches(expr), run(expr, target, banned, include_db) -> (steps, final) | None}`; exported list `STRATEGIES`. First strategy:

```python
STRATEGY_BUDGET = 60   # candidate moves one strategy run may spend


def _is_product_of_sums(expr: Expr) -> bool:
    """Form: (a + b + c)(d + e) -- a product carrying at least two sums
    (Mul is flattened by normalisation, so nested products count too)."""
    return expr.is_Mul and sum(1 for a in expr.args if isinstance(a, Add)) >= 2


def expand_then_termwise(expr, target, banned, include_db):
    """Form (a+b+c)(d+e): multiply out FIRST, then simplify term by term.

    Each loop iteration prefers an expand (a bracket that reappeared wins
    over any other move); when there is nothing left to expand, ONE
    termwise pass runs -- every child left to right, then the root -- and
    the loop repeats until nothing changes, the goal is reached, history
    would repeat, or the budget runs out.
    """
    state = _SolveState(expr, banned, to_latex_as_written(expr))
    current = expr
    budget = STRATEGY_BUDGET

    while budget > 0:
        if to_latex(current) == to_latex(target):
            return _steps_so_far(...), current      # reached

        expand_hit = _first_move_named(current, "Expand brackets", ...)
        if expand_hit:
            ...append step, record, current = new, budget -= 1
            continue

        term_hit = _termwise_move(current, state, include_db, ...)  # defined below
        if term_hit is None:
            break                                   # fixed point
        ...append step, record, current = new, budget -= 1

    return None if to_latex(current) != to_latex(target) else (...), current
```

Helper `_termwise_move(root, state, include_db, budget)` -- **one** legal move for a full pass:
```python
    # children first, left to right: the first legal change under any child
    for index, child in enumerate(root.args):
        for new_child, name, formula, _row in _iter_changes(
                child, SIMPLIFY_RULES, include_db, state, with_derived=True):
            rebuilt = _rebuild(root, index, new_child)   # same rebuild the walk uses
            if state.revisits(rebuilt):
                continue
            return rebuilt, name, formula, ...
    # then one move at the root itself (so combines/fraction work fire)
    hit = _apply_once(root, SIMPLIFY_RULES, include_db, state)
    if hit: return rebuilt-equivalent
    return None
```
(`with_derived=True` = tidy + formulas + SymPy hint, per decision 2. `_rebuild` mirrors `_iter_changes`'s `expr.func(*args, evaluate=False)` + `normalize`. Strategies never touch method leaps.)

- [ ] **Step 2: Dispatch in `steps.py`** -- new `_strategy_routes(expr, target, include_db, banned)`; import `from .strategies import STRATEGIES` **inside** the function (strategies.py imports steps.py helpers at module level -- lazy import breaks the cycle). Returns reaching routes sorted by `len(steps)` ascending. Wired into `_build_steps` immediately after the goal is known, BEFORE `_teach`:

```python
    if target is not None:
        strategy_routes = _strategy_routes(expr, target, include_db, banned)
        if strategy_routes:
            steps, final = strategy_routes[0]        # (5) shortest first
            return steps, final
```
Non-reaching strategy runs are discarded (walk/BFS still handle those inputs).

- [ ] **Step 3: Others as options** -- in `alternative_ways`, prepend every reaching strategy route EXCEPT the chosen one as ways: `{"name": <strategy name>, "steps": <its steps>, "final": <its final>}` (existing `_same_answer` filtering still applies). The chosen strategy must not appear among its own alternatives.

- [ ] **Step 4: Tests** (append; use `monkeypatch.setattr(strategies, "STRATEGIES", [fake])` for the synthetic ones)

```python
def test_product_of_sums_form_is_detected():
    from src.algebra.strategies import _is_product_of_sums
    assert _is_product_of_sums(parse_expression("(a + b + c)*(d + e)"))
    assert _is_product_of_sums(parse_expression("(x + 1)*(y + 1)"))
    assert not _is_product_of_sums(parse_expression("2*x + 3"))
    assert not _is_product_of_sums(parse_expression("sin(x) + cos(x)"))


def test_expand_strategy_reaches_an_expanded_target():
    expr = parse_expression("(a + b + c)*(d + e)")
    target = parse_expression("a*d + a*e + b*d + b*e + c*d + c*e")
    steps = simplify_steps(expr, target=target)
    assert [s["name"] for s in steps] == ["Initial expression", "Expand brackets"]
    assert steps[-1]["latex"] == to_latex(target)


def test_termwise_pass_fires_root_work_after_children():
    from src.algebra.strategies import _termwise_move
    from src.algebra.steps import _SolveState
    expr = parse_expression("2*x + 3*x + 5 - 1")      # children alone cannot change
    state = _SolveState(expr, frozenset(), to_latex_as_written(expr))
    hit = _termwise_move(expr, state, True)
    assert hit is not None
    assert hit[1] == "Evaluate numbers"               # root-level tidy wins first


def test_a_non_reaching_strategy_never_hijacks_the_route():
    # fake strategy matching everything but never reaching the goal
    fake = make_fake_strategy(name="Always Late", reaches=False)
    monkeypatch.setattr(strategies, "STRATEGIES", [fake])
    steps = steps_for("2*x + 3*x + 5 - 1")
    assert [s["name"] for s in steps] == [
        "Initial expression", "Evaluate numbers", "Combine like terms"]


def test_shortest_reaching_strategy_is_the_main_route(monkeypatch):
    short, long = make_fake_strategy(name="Short One", reaches=True, size=1), \
                  make_fake_strategy(name="Long One", reaches=True, size=3)
    monkeypatch.setattr(strategies, "STRATEGIES", [long, short])
    steps = steps_for("sin(x) + cos(x)")
    assert steps[1]["name"] == "Short One"
    ways = alternative_ways(parse_expression("sin(x) + cos(x)"))
    assert "Long One" in [w["name"] for w in ways]     # (5) others as options
```
(`make_fake_strategy` is a small local test helper: `matches=lambda e: True`, `run` returning `reached final` for `sin(x)+cos(x)`'s simplify answer or a non-reaching sentinel; for `reaches=True` the fake's final must equal `to_latex(simplify(parse_expression("sin(x) + cos(x)")))`. If SymPy does not change that input, use `2*x + 3*x + 5 - 1` with final `5 x + 4` instead -- pick the variant that works and say which.)

- [ ] **Step 5: Risk step** -- `python -m pytest -q` -> **146 passed** (141 + 5). Any failure: STOP and report; no assertion edits.

- [ ] **Step 6:** NO COMMIT.

---

## Task 9: The UI can provide the goal (decision 4: "there will be a goal always")

**Files:** `src/App.tsx` (+ nothing else -- the API field exists).

- [ ] Add an optional **Target** input next to the expression box in the solve tab (label "Target (optional) -- the form you want"). Send it as `target` in the `/api/solve` body only when non-empty (and in `/api/ask` if that path can reach simplify; otherwise note it stays solve-tab only).
- [ ] Show it back in the result header as `Target: <latex>` when supplied, so the goal is visible.
- [ ] Verify: `npx tsc --noEmit` clean; `npm run build` OK; live smoke -- with `Target = 2*sin(x)*cos(x)` on `sin(2*x)` the UI route ends `Double Angle for Sine`; with no target, behaviour unchanged.

---

## Task 8b: Strategies become DATA + Rule B + hide alternatives

**User decisions:** (i) strategies must be editable without code -> rows in the DB (`kind:strategy`), phases `{do, scope, until}` executed by one small interpreter in `strategies.py`; (ii) **Rule B**: if a strategy matches the form, ITS route is the route (reaching or not, with the gap rules); walk/BFS/fallbacks run only when NO strategy matches; (iii) the `a sin + b cos` method stays as-is (already data; only two arithmetic `compute` functions remain in `auxiliary.py`); (iv) hide "Other ways" in the UI but keep the engine for later; (v) the first data strategy is the user's own four-phase route.

- [ ] **Step 1: strategy rows.** New file `data/samples/strategies.jsonl`:
```json
{"kind":"strategy","name":"Expand then reduce","form":"product_of_sums","phases":[
 {"do":"expand","scope":"whole","until":"stuck"},
 {"do":"simplify","scope":"term","until":"stuck"},
 {"do":"collect","scope":"whole","until":"stuck"},
 {"do":"simplify","scope":"whole","until":"once"}]}
```
Importer: new branch for `kind:strategy` -> store the doc in `steps` (JSON), `applies_to = form`, `verified = 1` (no proof needed: no left/right), validate `form` against the known-forms registry and every phase against `{do: known op, scope: whole|term, until: stuck|once}` (reject on violation, like methods). `store.find_strategies(connection, form)`; add `strategy` to the `Entry.kind` comment (api) and the TS union (App) so it lists in the Formulas tab.

- [ ] **Step 2: interpreter replaces the hardcoded function.** `strategies.py` keeps `Strategy`/`STRATEGIES` shape for dispatch but loads rows via `find_strategies(form)`; form registry `_FORMS = {"product_of_sums": _is_product_of_sums}`. Run semantics: goal check every iteration; per phase, repeat until it cannot change anything (`until: stuck`) or once (`until: once`); then next phase; history (`state.revisits`) is the ONLY permission; `STRATEGY_BUDGET` total. Phase executor:
  - `scope: whole` -> `run_operation(do, current)` (labels come from `ops.py`: expand->"Expand brackets", simplify->"Simplify", collect->"Collect like terms", ...); for `do: collect` only, if the op changed nothing, fall back to the tidy rule names `{"Evaluate numbers","Combine like terms"}` via `_iter_changes` (root, `with_derived=False`) so `1 - cot + tan - 1` becomes `tan - cot`.
  - `scope: term` -> for each child left to right: `run_operation(do, child)`, rebuild with `normalize(root.func(*args, evaluate=False))`, accept the first child that changes AND passes `state.revisits` on the rebuilt root.
  - **No `with_derived` inside strategies** (the user's "apply sympy.simplify on each term" IS the hint, as a direct op) -- this is also the performance fix: strategies must stop burning derived/proof time.
  - Return `(steps, final)` ALWAYS when `len(steps) >= 2` (Rule B: non-reaching routes are kept); dispatcher decides.

- [ ] **Step 3: Rule B in `steps.py`.** `_strategy_routes` returns `{"name","steps","final"}` for every MATCHING strategy, sorted: reaching first (shortest), else most steps (furthest progress). In `_build_steps`, when the goal exists: **if any strategy matched -> show its best route and return** (gap recorded when `final != target`, or when reached with a raw-SymPy final hop via `_UNEXPLAINED_HOPS`; `tried` for the gap = the route's step names). **Only when nothing matched** does the existing chain run: `_teach` -> `_target_route` -> gap. No strategy may be bypassed by a fallback.

- [ ] **Step 4: hide alternatives.** `src/App.tsx`: add `const SHOW_ALTERNATIVE_WAYS = false;` and gate the "Other ways to reach the same answer" section on it (do not delete the rendering code -- held for future). API/engine untouched so no test changes.

- [ ] **Step 5: pending review fixes from Task 8's review** (that round was interrupted): module + `_build_steps` docstrings updated to mention STRATEGIES in the flow; widen `_strategy_routes`' try/except to cover the result unpacking (a broken/garbage strategy must not break a solve); MAX_STEPS comment noting a strategy route may exceed it by one line (we show the honest route, not a slice).

- [ ] **Step 6: tests.** New: strategy row imports + `find_strategies` lookup; the interpreter executes the user's route on `(1 + tan(x))*(1 - cot(x))` with goal `tan(x) - 1/tan(x)` and produces `Expand brackets -> (per-term) Simplify -> Evaluate numbers -> Simplify` reaching the goal; Rule B: a matching-but-failing strategy is STILL the displayed route (fallbacks do not run); no-match input keeps today's fallback behaviour unchanged. **Expected to fail -- STOP and report, do not edit:** `test_main_route_shows_the_methods_steps` (the trig product matches the form, so its route becomes the strategy's) and possibly `test_gap_is_recorded_when_no_strategy_reaches_the_target` (gap `tried` becomes the strategy's step names, not BFS names). Both need user approval to re-scope -- print the new route/tried and ask.

---

## Task 8c (SUPERSEDES 8b): Ladder X — data strategies, walk as a phase, one solution

**Confirmed decisions:** (1) strategies are DB rows (`kind:strategy`), edited as questions arrive; (2) ladder X: strategy matched -> its ONE route; none matched -> last resort `simplify whole` + gap; (3) the walk is NOT a top-level engine — it is a phase strategies invoke ("when to use walk to search for formulas and methods"); (4) methods are searched BY the walk phase; (5) the user's first strategy row stays exactly as written, walk phases come later as they author; (6) one solution only — alternatives not computed/returned/shown, code held for v2; (7) SymPy runs only when instructed (goal fetch, an op phase, or the last resort).

- [ ] **X1 — strategy rows.** `data/samples/strategies.jsonl` with exactly:
```json
{"kind":"strategy","name":"Expand then reduce","form":"product_of_sums","phases":[
 {"do":"expand","scope":"whole","until":"stuck"},
 {"do":"simplify","scope":"term","until":"stuck"},
 {"do":"collect","scope":"whole","until":"stuck"},
 {"do":"simplify","scope":"whole","until":"once"}]}
```
Importer branch `kind:strategy`: doc JSON -> `steps`, `applies_to = form`, `verified = 1` (no proof: no left/right), validate `form` against the registry and each phase against `{do: known op or "walk", scope: whole|term, until: stuck|once}` — reject like methods. `store.find_strategies(connection, form)`. `strategy` added to Entry kind comment (api) + TS union (App).

- [ ] **X2 — interpreter in `strategies.py`** (replaces the hardcoded function): form registry `_FORMS = {"product_of_sums": _is_product_of_sums}`; `matching_strategies(expr)` = module `STRATEGIES` (test fakes) **+** DB rows for matching forms; phase executor with goal-check every iteration, total `STRATEGY_BUDGET`, history as the only permission:
  - **`{do:"walk", until:...}`** = one legal move at a time from `_iter_changes(..., with_derived=True)` **minus names in `_SYMPY_MOVES`**, plus `_method_leaps` (authored moves), history-gated, until stuck. (This is the old walk, demoted and SymPy-free.)
  - **op, scope whole** = `run_operation(do, root)`; for `do:"collect"` fall back to tidy-rule names `{"Evaluate numbers","Combine like terms"}` when the op changes nothing.
  - **op, scope term** = per child left-to-right `run_operation(do, child)`, rebuild via `normalize(root.func(*args, evaluate=False))`, first child that changes AND passes `revisits`.
  - Returns `(steps, final)` whenever `len(steps) >= 2` — **non-reaching routes are kept (Rule X)**; `None` only when it produced nothing.

- [ ] **X3 — ladder in `_build_steps`:** goal known? matched strategies (best = reaching-shortest, else most steps) -> return route + gap rules (didn't reach -> gap with route names as `tried`; reached with final in `_UNEXPLAINED_HOPS` -> gap). No matched strategy/no route -> **last resort**: `run_operation("simplify", expr)`; if its latex == goal -> `[Initial, {"name": "Simplify", ...}]` + gap; else -> `_one_suggestion` + gap `no strategy reaches this target`. No goal -> `_one_suggestion` (unchanged). **Delete** `_target_route`, `TARGET_DEPTH`, `SEARCH_BUDGET`, and all calls to `_teach` from `_build_steps` (`_teach` itself stays — `alternative_ways`/v2 use it).

- [ ] **X4 — one solution.** `api/main.py`: solve() stops calling `alternative_ways` (returns `alternatives: []`); `src/App.tsx`: `const SHOW_ALTERNATIVE_WAYS = false;` gating the section. Engine function untouched (v2).

- [ ] **X5 — tests (new):** strategy row imports + `find_strategies`; interpreter executes the user's 4 phases on `(1 + tan(x))*(1 - cot(x))` (goal `tan(x) - 1/tan(x)`) yielding `Expand brackets -> Simplify -> Evaluate numbers -> Simplify` reaching the goal; walk-phase unit: `_walk_phase` applies `Pythagorean Identity` on `sin(x)^2 + cos(x)^2 + y`; ladder: no-match input -> `[Initial, Simplify]` + gap; matching-but-failing strategy still shown; user-target-unreachable -> suggestion + gap. **Old tests failing = the re-scoping program: STOP, group the failures, print each actual new route, present to the user — never edit an assertion.** Known groups: classroom chains/step-lists, endpoint alternatives, supplied-target reach tests, BFS tests, gap `tried` content, `test_main_route_shows_the_methods_steps` (trig product now strategy-owned).
