"""
steps.py -- builds the step list: the ladder, one step at a time.

The flow (LADDER X):

    1. GOAL:   the caller's target if given, else simplify(expr) when
       that changes the input. (Its answer defines "there".)
    2. STRATEGY: a form-matched strategy runs its ONE route -- reaching
       or not. The chosen route is the best produced: reaching first
       (shortest), else most steps (furthest progress). A route that
       misses the goal, or lands on it with a raw SymPy hop, is recorded
       as a gap -- the strategy/formula to plug next.
    3. LAST RESORT: with no strategy route at all, one instructed
       whole-expression simplify: when that equals the goal we show it
       (and record a gap -- the landing is not ours), otherwise fall
       through.
    4. GAP + SUGGESTION: record "no strategy reaches this target" and
       show exactly one applicable move -- a suggestion, never a walk.
    5. NO GOAL: strategies may still run (D1(a)): the best route they
       produced (most steps = furthest progress, ties keep the
       strategies' own order) is the display -- no landing append and no
       gap, because there is no goal to miss. When no strategy produced
       a route, exactly one applicable move -- a suggestion, never a
       walk.

    D3(b): when a goal exists and the chosen route misses it, the
    landing is appended ONLY when the goal came from our own simplify()
    fetch (a user-supplied target is unverified by construction, so its
    route stays as-is with the unreachable gap).

    The walk is NOT an engine: it is a PHASE strategies invoke
    (strategies.py). SymPy runs only when instructed -- the goal fetch,
    an op phase, or the last resort. One solution only: alternate routes
    are not computed for the main display (alternative_ways is kept for
    a later version).

Guards that never change:
    * no step may lead back to an earlier state of its route (history:
      canonical latex + structure -- value checks are meaningless here
      because every simplify step preserves value)
    * history is the only permission rule
    * MAX_STEPS caps the display; STRATEGY_BUDGET caps one strategy run
"""

from itertools import chain

from sympy import Add, Mul, Wild, cancel, simplify, srepr, trigsimp
from sympy.core.expr import Expr

from ..knowledge import derive as derive_mod
from ..knowledge import patterns as knowledge_patterns
from ..knowledge import runner as _method_runner
from ..knowledge import store as knowledge_store
from ..knowledge.classify import classify, mentioned_functions
from ..knowledge.ops import run_operation
from .latex import to_latex, to_latex_as_written
from .parser import normalize
from .rules import SHAPE_RULES, SIMPLIFY_RULES, Rule

# Hard cap: never show the user more than this many steps. Strategy
# routes may exceed it by one (see strategies.py) -- we show the honest
# route rather than slice it.
MAX_STEPS = 20

# Memo for _strategy_routes: (as-written latex, srepr, target latex,
# banned, include_db) -> (the STRATEGIES list it was computed with,
# routes). Same-input solves run the strategies twice (build + ways);
# the cache makes the second run free. The as-written latex is part of
# the key because srepr canonicalises Add argument order:
# srepr("2*x+3*x-1+5") == srepr("2*x+3*x+5-1"), so an srepr-only key
# served one spelling the other's cached route (and its wrong first
# line). Callers only read the routes, so returning the stored object
# itself is fine (no copy needed). `banned` must be hashable -- it is a
# frozenset everywhere it is passed. Each entry pins the exact
# STRATEGIES list it used: the hit checks `is`, and holding the
# reference means a freed list's address can never be recycled under a
# live entry (a replaced/monkeypatched list simply misses).
_STRATEGY_CACHE: dict = {}
_STRATEGY_CACHE_LIMIT = 32

# How many method rows we read before choosing (a bound, not a rank).
_LEAP_FETCH = 32

# What the web API returns: a list of {"name": ..., "latex": ...}.
Step = dict


class _SolveState:
    """Everything one run of the solver needs to remember.

    banned          step names we promised not to use (alternative routes)
    history*        every STATE this route has displayed -- the progress
                    guard: no action may lead back to an earlier step
                    (checked by canonical latex first, then structure;
                    value-equivalence is meaningless here because every
                    simplify step preserves value by design)
    simplified      nodes already handed to SymPy's simplify (once each)
    """

    def __init__(self, original: Expr, banned: "frozenset[str]" = frozenset(),
                 display_latex: str = ""):
        self.banned = banned
        self.simplified: set = set()   # nodes already given to simplify()
        self.sym_seen: set = set()     # (which sub-pipeline, node) done already
        self.history: list = []           # the exprs this route displayed
        self.history_latex: set = set()   # canonical latex of those states
        self.history_srepr: set = set()   # structural fingerprints of those

        # The starting point counts as an "earlier step": a route may
        # never wander back to where it began (in any spelling).
        self.record(original, display_latex or to_latex(original))

    def record(self, expr: Expr, display_latex: str = "") -> None:
        """Remember a state this route has shown."""
        self.history.append(expr)
        self.history_latex.add(to_latex(expr))
        self.history_srepr.add(srepr(expr))
        if display_latex:
            self.history_latex.add(display_latex)

    def revisits(self, expr: Expr) -> bool:
        """Would this step lead back to an earlier step of this route?

        Tier 1: same displayed maths (canonical latex -- spelling-proof).
        Tier 2: same structure (srepr) -- catches wording-only changes.
        (Value-equivalence is NOT used: in a simplify route every state
        equals every other, so it would reject everything.)
        """
        if to_latex(expr) in self.history_latex:
            return True
        return srepr(expr) in self.history_srepr


def simplify_steps(expr: Expr, include_db: bool = True,
                   target: "Expr | None" = None) -> list[Step]:
    """Give the list of steps needed to simplify this expression."""
    steps, _ = _build_steps(expr, SIMPLIFY_RULES, include_db, target=target)
    return steps


def alternative_ways(expr: Expr, include_db: bool = True,
                     limit: int = 4,
                     target: "Expr | None" = None) -> list[dict]:
    """OTHER routes that finish at the SAME maths as the normal route.

    The exploring runs use the TEACHING walk only: it is fast, and its
    steps carry the formula names people want to compare ("Cos 2A as
    1 - 2sin^2" vs "Double Angle for Cosine" vs ...).
    """
    main_steps, main_final = _build_steps(expr, SIMPLIFY_RULES, include_db,
                                          target=target)
    if len(main_steps) < 2:
        return []  # nothing happened, so nothing to compare

    ways: list[dict] = []
    goal = _effective_target(expr, target, include_db)
    for route in _strategy_routes(expr, goal, include_db, frozenset()):
        if route["steps"] == main_steps:
            continue                      # that one IS the main route
        if len(route["steps"]) < 3:
            continue  # one move is not a route: the teach loop below
                      # lists single moves, so a 2-line strategy route
                      # would only steal their slot
        if not _same_answer(route["final"], main_final):
            continue  # a route elsewhere is not a WAY to this answer
        ways.append({"name": route["name"], "steps": route["steps"],
                     "final": route["final"]})

    banned: set = {main_steps[1]["name"]}  # the move we already used
    collected: set = {main_steps[1]["name"],
                      *(way["name"] for way in ways)}
    iterations = 0

    # Keep exploring until we have `limit` DISTINCT ways (or the banned
    # names run out). Same-named ways are skipped -- they are the same way.
    while len(ways) < limit and iterations < limit * 4:
        iterations += 1
        steps, final = _teach(expr, SIMPLIFY_RULES, include_db,
                              frozenset(banned))
        if len(steps) < 2:
            break  # no other move exists

        banned.add(steps[1]["name"])

        if steps[1]["name"] in collected:
            continue  # same way as before -- look for a different start

        if _same_answer(final, main_final):
            # "final" is the SymPy expression (tests use it to prove the
            # routes are equal maths); the API's response model drops it.
            ways.append({"name": steps[1]["name"], "steps": steps,
                         "final": final})
            collected.add(steps[1]["name"])

    # Methods with an explicit shape count as ways too -- e.g. the
    # step-by-step WORKING of the a*sin+b*cos trick, which is the long way
    # round to the same answer the direct formula gives in one step.
    ways.extend(_method_ways(expr, main_final,
                             {main_steps[1]["name"],
                              *(way["name"] for way in ways)},
                             include_db))

    return ways


def _method_ways(expr: Expr, main_final: Expr, used_names: set,
                 include_db: bool) -> list:
    """Run shape-specific methods as alternative routes to the same answer."""
    if not include_db:
        return []

    try:
        connection = knowledge_store.get_connection()
        rows = knowledge_store.find_methods(connection, expr, for_ways=True)
    except Exception:
        return []

    ways: list = []
    for row in rows:
        if row["name"] in used_names:
            continue
        try:
            steps, final = _method_runner.run_method(expr, row["steps"],
                                                     with_final=True)
        except Exception:
            continue  # this method does not fit this expression
        if len(steps) < 2:
            continue
        if not _same_answer(final, main_final):
            continue  # a different answer is a different method, not a way
        ways.append({"name": row["name"], "steps": steps, "final": final})
    return ways


def _method_leaps(expr: Expr, banned: "frozenset[str]") -> list:
    """Stored methods run as ONE candidate move each: a long procedure
    compressed into a single edge, so deep routes still fit the depth
    limit. The sub-steps ride along, so choosing this leap shows the
    method's own named lines instead of one opaque jump.
    Each leap: {"name", "final", "steps", "row_id"}.

    Rows are read wide and chosen in id order (never usage order), so
    leap membership and sequence stay stable across runs as usage grows.
    """
    try:
        connection = knowledge_store.get_connection()
        rows = knowledge_store.find_methods(connection, expr, for_ways=False,
                                            limit=_LEAP_FETCH)
    except Exception:
        return []
    leaps = []
    for row in sorted(rows, key=lambda r: r["id"])[:8]:
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


def _same_answer(a: Expr, b: Expr) -> bool:
    """Do these two expressions mean the same maths?"""
    if to_latex(a) == to_latex(b):
        return True
    try:
        from sympy import simplify
        return simplify(a - b) == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Building one route
# ---------------------------------------------------------------------------
def _effective_target(expr: Expr, target: "Expr | None",
                      include_db: bool) -> "Expr | None":
    """The goal a solve chases: the caller's target if given, else
    simplify(expr) when that changes the input (None when no goal).

    Shared by the main route and the alternatives so the two can never
    disagree about what "there" means."""
    if target is not None:
        return target
    if not include_db:
        return None   # never ask SymPy for a goal when the db is off
    candidate = simplify(expr)
    if to_latex(candidate) != to_latex(expr):
        return candidate
    return None


def _strategy_routes(expr: Expr, target: "Expr | None", include_db: bool,
                     banned: "frozenset[str]") -> list:
    """Every route the applicable strategies produced, best one first.

    Ladder X shows ONE solution: the routes are candidates for that one
    slot, not alternatives. Sort: reaching first (shortest), else most
    steps (furthest progress) -- a non-reaching route that got further
    is the better thing to show. With NO goal (D1(a): target is None)
    nothing can be "reaching", so the sort is simply most steps first --
    furthest progress -- and ties keep the strategies' own order (the
    sort is stable). Routes with fewer than two lines (the strategy
    produced nothing) are dropped; a broken strategy (or garbage it
    returns, including a bad unpack) never breaks a solve.

    Memoised: _build_steps already ran the same strategies for the same
    (expr, target, banned, include_db) during the solve that precedes
    alternative_ways, so the second run would cost the same again.
    The key carries the as-written latex AND the srepr: the srepr alone
    is not enough, because it canonicalises Add argument order (two
    spellings of the same maths hash identically and would share one
    cached route -- the wrong first line for the second spelling).
    Key components are all hashable -- `banned` is a frozenset (asserted
    by use: it is a dict key here). Each entry pins the STRATEGIES list
    it was computed with, so a swapped-in list (tests monkeypatch it)
    misses instead of serving stale routes. Callers only read the
    returned routes, so handing back the same object is safe. When the
    cache grows past the limit it is cleared wholesale (simple eviction,
    not LRU -- the cost is one re-run at most)."""
    if not include_db:
        return []
    from .strategies import STRATEGIES, matching_strategies
    key = (to_latex_as_written(expr), srepr(expr),
           to_latex(target) if target is not None else None,
           banned, include_db)
    cached = _STRATEGY_CACHE.get(key)
    if cached is not None and cached[0] is STRATEGIES:
        return cached[1]

    target_latex = to_latex(target) if target is not None else None
    try:
        candidates = matching_strategies(expr)
    except Exception:
        candidates = []   # a broken lookup must never break a solve

    routes = []
    for strategy in candidates:
        try:
            if not strategy.matches(expr):
                continue
            steps, final = strategy.run(expr, target, banned, include_db)
            if steps is None or len(steps) < 2:
                continue  # nothing happened: not a route
        except Exception:
            continue   # a broken strategy (or garbage it returns) must
                        # never break a solve
        routes.append({"name": strategy.name, "steps": steps,
                       "final": final})

    if target_latex is None:
        # D1(a): no goal to reach -- furthest progress wins; ties keep
        # the order matching_strategies produced (list.sort is stable).
        routes.sort(key=lambda route: -len(route["steps"]))
    else:
        routes.sort(key=lambda route: (
            (0, len(route["steps"]))                # reaching: shortest
            if to_latex(route["final"]) == target_latex
            else (1, -len(route["steps"]))))        # missing: furthest
    if len(_STRATEGY_CACHE) > _STRATEGY_CACHE_LIMIT:
        _STRATEGY_CACHE.clear()
    _STRATEGY_CACHE[key] = (STRATEGIES, routes)
    return routes


def _build_steps(expr: Expr, rules: list[Rule], include_db: bool = True,
                 banned: "frozenset[str]" = frozenset(),
                 target: "Expr | None" = None) -> "tuple[list[Step], Expr]":
    """Ladder X: one strategy route, else the last resort, else a
    suggestion.

    goal       = the caller's target, else simplify(expr) if that
                 changes it
    strategy   = a form-matched strategy's ONE route, reaching or not
                 (a miss, or a final raw-SymPy hop, is a gap; a miss on
                 a simplify()-fetched goal also earns the appended
                 landing -- D3(b))
    last resort= one instructed whole-expression simplify, only when
                 that lands on the goal (the landing is not ours: gap)
    otherwise  = a gap ("no strategy reaches this target") and exactly
                 one suggestion -- never a walk
    no goal    = strategies may still run (D1(a)): the best route (most
                 steps, ties keep the strategies' own order) is the
                 display -- no landing, no gap; without one route, one
                 suggestion step, never a walk
    include_db=False means we never ask SymPy for a goal: with no caller
    target there is deliberately no goal at all -- and no strategy run
    either (the db is what feeds it) -- one suggestion, never a walk.
    """
    user_supplied = target is not None
    target = _effective_target(expr, target, include_db)

    if target is None:
        # D1(a): no goal, but strategies may still run -- their best
        # route is the display. No landing append and no gap: there is
        # no goal the route could miss.
        routes = _strategy_routes(expr, None, include_db, banned)
        if routes:
            chosen = routes[0]
            return chosen["steps"], chosen["final"]
        return _one_suggestion(expr, rules, include_db, banned)

    routes = _strategy_routes(expr, target, include_db, banned)
    if routes:
        chosen = routes[0]
        steps, final = chosen["steps"], chosen["final"]
        if to_latex(final) != to_latex(target) and not user_supplied:
            # D3(b): the goal is our own simplify() answer, so the
            # landing is shown as the final step (the route copy is
            # rebuilt -- the cache owns `steps`). A user-supplied target
            # is unverified by construction: its route stays as-is.
            steps = [*steps,
                     {"name": "Simplify", "latex": to_latex(target)}]
            final = target
        if (to_latex(final) != to_latex(target)
                or steps[-1]["name"] in _UNEXPLAINED_HOPS):
            # Either the route missed the goal, or its final hop is raw
            # SymPy (the D3(b) landing is a "Simplify" hop, so the gap
            # still records it) -- both are exactly the gap a new
            # formula/strategy should close.
            _record_gap(target, expr, [s["name"] for s in steps])
        return steps, final

    # last resort: one instructed whole-expression simplify
    result = run_operation("simplify", expr)
    if result is not None and to_latex(result[0]) == to_latex(target):
        _record_gap(target, expr, [result[1]])
        return [_initial_step(expr),
                {"name": result[1], "latex": to_latex(result[0])}], result[0]

    _record_gap(target, expr, ["no strategy reaches this target"])
    return _one_suggestion(expr, rules, include_db, banned)


def _initial_step(expr: Expr) -> Step:
    return {"name": "Initial expression", "latex": to_latex_as_written(expr)}


def _record_gap(target: Expr, expr: Expr, tried: set) -> None:
    """Remember a simplification we cannot (yet) explain step by step."""
    try:
        knowledge_store.record_gap(
            to_latex(target), to_latex(expr),
            # Keep most of what we tried: an alphabetical cap cut off names
            # that mattered. 24 covers a normal solve; a fully exhausted
            # search can exceed it, so this is a display bound, not a
            # completeness guarantee.
            ", ".join(sorted(tried)[:24]),
        )
    except Exception:
        pass  # gap bookkeeping must never break a solve


# SymPy's own moves. The WALK is not an engine: strategies.py skips
# these names, so a walk phase can never pick SymPy by itself -- SymPy
# runs only when instructed (the goal fetch, an op phase, or the ladder's
# last resort). Where the moves are enumerated as candidates they still
# come after ours (an ordering convention, not a quality judgement).
_SYMPY_MOVES = frozenset({
    "Simplify with SymPy",
    "Simplify trigonometry",
    "Write with sin/cos and cancel",
})

# A route that lands on the target with one of these as its LAST step has
# not demonstrated the landing with our own formulas: the gap still
# records it (user's rule: gap = the final hop is not ours).
_UNEXPLAINED_HOPS = _SYMPY_MOVES | {"Simplify"}


def _teach(expr: Expr, rules: list[Rule], include_db: bool,
           banned: "frozenset[str]",
           target: "Expr | None" = None) -> "tuple[list[Step], Expr]":
    """The classroom walk -- any change, one visible step at a time.

    When `target` is given, the walk stops the moment it lands on that
    exact form -- reaching the answer ends the walk; we never wander
    past it.
    """
    steps: list[Step] = [{"name": "Initial expression",
                          "latex": to_latex_as_written(expr)}]
    # The state's history starts with this line (as written AND canonical),
    # so a later fold-back that spells the original differently is still
    # recognised as "we have been here before".
    state = _SolveState(expr, banned, steps[0]["latex"])
    current = expr

    while len(steps) < MAX_STEPS:
        hit = _apply_once(current, rules, include_db, state)
        if hit is None:
            break  # nothing left to do -- we are done

        new_expr, name, formula, row_id = hit

        step: Step = {"name": name, "latex": to_latex(new_expr)}
        if formula:
            # The formula that was applied (from the database), so the
            # screen can show WHICH rule of maths was used.
            step["formula"] = formula
        steps.append(step)
        state.record(new_expr, step["latex"])
        current = new_expr
        _bump(row_id)

        if target is not None and to_latex(new_expr) == to_latex(target):
            break  # we have arrived: stop instead of wandering past it

    return steps, current


def _one_suggestion(expr: Expr, rules: list[Rule], include_db: bool,
                    banned: "frozenset[str]") -> "tuple[list[Step], Expr]":
    """No goal to reach: show what ONE move can do, then stop.

    Without a target we cannot tell a step forward from a step sideways,
    so we never walk -- one applicable move, clearly a suggestion, is the
    honest display.
    """
    state = _SolveState(expr, banned, to_latex_as_written(expr))
    hit = _apply_once(expr, rules, include_db, state)
    if hit is None:
        return [_initial_step(expr)], expr
    new_expr, name, formula, row_id = hit
    step = {"name": name, "latex": to_latex(new_expr)}
    if formula:
        step["formula"] = formula
    _bump(row_id)   # a shown suggestion counts like any other step
    return [_initial_step(expr), step], new_expr


def _bump(row_id) -> None:
    """Count that a formula actually fired (so it sorts higher next time)."""
    if row_id is None:
        return
    try:
        connection = knowledge_store.get_connection()
        knowledge_store.bump_usage(connection, row_id)
        connection.commit()
    except Exception:
        pass  # usage counts are a nicety, never a reason to fail a solve


# ---------------------------------------------------------------------------
# The move generator: the first applicable change (as always, in order:
# tidy rules -> database -> expanding/factoring -> children, sums/products
# first).
# ---------------------------------------------------------------------------
def _is_trig_node(expr: Expr) -> bool:
    """Does this node involve trig functions? (SymPy's simplify shines
    there -- algebra is left to our own rules and derived formulas.)"""
    try:
        from ..knowledge.classify import mentioned_functions
        return bool(mentioned_functions(expr))
    except Exception:
        return False


def _apply_once(expr: Expr, rules: list[Rule], include_db: bool,
                state: _SolveState):
    """The first change the permission rule allows.

    History is the ONLY gate: a step may rewrite anything -- including
    parts a formula produced -- as long as it does not land on an
    earlier line of this route. Refused candidates are SKIPPED, not
    fatal: the walk continues with the next move. Two phases: every
    forward orientation in the whole tree, then every backwards one.
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
        return change
    return None


def _iter_changes(expr: Expr, rules: list[Rule], include_db: bool,
                  state: _SolveState, with_derived: bool,
                  direction: str = "both"):
    """Yield every single change, as (new_root, name, formula, row_id).

    Same order the solver has always used, so the teaching route is
    unchanged; with_derived simply keeps looking past the first hit.
    """
    tidy_rules = [rule for rule in rules if rule not in SHAPE_RULES]
    shape_rules = [rule for rule in rules if rule in SHAPE_RULES]

    # 1) tidy-up rules work everywhere
    for rule in tidy_rules:
        hit = rule(expr)
        if hit is None:
            continue
        new_expr, name = hit
        if name in state.banned:
            continue
        if to_latex(new_expr) == to_latex(expr):
            continue  # not visible on screen
        yield new_expr, name, None, None

    # 2) SUM-FACTORS OF A PRODUCT GO FIRST: in a product like
    #    (sin^2 + cos^2)(sin^4 + cos^4) the factors are where reductions
    #    live (Pythagorean: sin^2+cos^2 = 1). If we tried the top-level
    #    folds first they would consume that factor before it could be
    #    reduced to 1.
    pre_visited: set = set()
    if isinstance(expr, Mul):
        for index, part in enumerate(expr.args):
            if isinstance(part, Add) and part.args:
                pre_visited.add(index)
                for child_root, name, formula, row_id in _iter_changes(
                        part, rules, include_db, state, with_derived,
                        direction):
                    args = list(expr.args)
                    args[index] = child_root
                    rebuilt = normalize(expr.func(*args, evaluate=False))
                    if to_latex(rebuilt) == to_latex(expr):
                        continue
                    yield rebuilt, name, formula, row_id

    # 3) STORED database formulas (their own names, highest trust)
    if include_db:
        for change in _stored_hits(expr, state, direction):
            yield change

    # 3b) One level of a difference of squares: a^(2k) - b^(2k) peels to
    #     (a^k - b^k)(a^k + b^k) -- ONE level at a time, and only AFTER the
    #     stored formulas of this node had their turn, so a formula
    #     (sin^2 + cos^2 = 1, sin^4 + cos^4 = ...) can fire between the
    #     factor levels. Full factor() stays available (shape rules below)
    #     for everything this one-level peel does not cover.
    if direction in ("forward", "both"):
        hit = run_operation("factor_once", expr)
        if hit is not None:
            new_expr, name = hit
            if name not in state.banned and to_latex(new_expr) != to_latex(expr):
                yield new_expr, name, None, None

    # 4) one honest SymPy shot at this node (search route, trig only).
    #    Placed BEFORE deriving, because derive() pays a proof for every
    #    variant while simplify() is one call -- a quick win must not be
    #    buried under the derivation cost.
    if with_derived and _is_trig_node(expr) \
            and "Simplify with SymPy" not in state.banned \
            and srepr(expr) not in state.simplified:
        state.simplified.add(srepr(expr))
        try:
            simplified = simplify(expr)
        except Exception:
            simplified = None
        if simplified is not None and to_latex(simplified) != to_latex(expr):
            yield simplified, "Simplify with SymPy", None, None

    # 5) expanding / factoring / reducing
    for rule in shape_rules:
        hit = rule(expr)
        if hit is None:
            continue
        new_expr, name = hit
        if name in state.banned:
            continue
        if to_latex(new_expr) == to_latex(expr):
            continue
        yield new_expr, name, None, None

    # 6) THE TRIG SUB-PIPELINES (search route only): SymPy's own
    #     functions as individually named, individually judged steps --
    #     one call each, no steering: the move list alone decides what
    #     we try next.
    #       trigsimp                       -> "Simplify trigonometry"
    #       rewrite(sin,cos) then cancel   -> "Write with sin/cos and cancel"
    #     (the rewrite alone usually grows the expression, so the pair is
    #      judged as ONE step, exactly like our denominator rewrite+cancel)
    if with_derived and include_db \
            and _is_trig_node(expr):
        from ..knowledge.gate import _to_sin_cos

        if "Simplify trigonometry" not in state.banned \
                and ("trigsimp", srepr(expr)) not in state.sym_seen:
            state.sym_seen.add(("trigsimp", srepr(expr)))
            try:
                new_expr = trigsimp(expr)
            except Exception:
                new_expr = None
            if new_expr is not None and to_latex(new_expr) != to_latex(expr):
                yield new_expr, "Simplify trigonometry", None, None

        if "Write with sin/cos and cancel" not in state.banned \
                and ("rewrite", srepr(expr)) not in state.sym_seen:
            state.sym_seen.add(("rewrite", srepr(expr)))
            try:
                new_expr = cancel(_to_sin_cos(expr))
            except Exception:
                new_expr = None
            if new_expr is not None and to_latex(new_expr) != to_latex(expr):
                yield new_expr, "Write with sin/cos and cancel", None, None

    # 7) DERIVED rearrangements of stored formulas (search route only) --
    #    last, because they are the expensive ones to produce.
    if with_derived:
        for change in _derived_hits(expr, state):
            yield change

    # 8) one level deeper -- sums and products first (the numerator of a
    #    fraction is the interesting part, denominators come after)
    interesting = [i for i, part in enumerate(expr.args)
                   if isinstance(part, (Add, Mul)) and part.args]
    others = [i for i, part in enumerate(expr.args) if i not in interesting]

    for index in interesting + others:
        if index in pre_visited:
            continue  # that sum-factor was already explored in step 2
        child = expr.args[index]
        for child_root, name, formula, row_id in _iter_changes(
                child, rules, include_db, state, with_derived, direction):
            args = list(expr.args)
            args[index] = child_root
            rebuilt = normalize(expr.func(*args, evaluate=False))
            if to_latex(rebuilt) == to_latex(expr):
                continue
            yield rebuilt, name, formula, row_id


def _expands_variable_angle(source: Expr) -> bool:
    """Does this pattern side EXPAND a compact trig call of a variable angle?

    The source of  sin(?a + ?b),  cos(2*?a), ...  is one trig function
    applied to an angle that still carries a symbol -- applying it spells
    the compact call OUT (an expansion). A deterministic route (the
    apply_formula phase) must not expand what a previous formula step just
    compressed (the R-form ping-pong the history gate alone stops only a
    full cycle later). Expansions stay available to the search walk and to
    suggestions; exact values keep a constant angle (sin(pi/10) = ...) and
    are reductions, so they pass.
    """
    from ..knowledge.classify import _TRIG

    return (source.func in _TRIG and len(source.args) == 1
            and bool(source.args[0].free_symbols))


def formula_applies_here(expr: Expr, include_db: bool = True) -> bool:
    """Can any verified stored identity rewrite this node (either way)?

    The factor_once op consults it: when a formula owns a node, the named
    formula step is the better move than a bare factor peel -- and peeling
    would run PAST a recognizable form (sin^2 - cos^2 is the double-angle
    row's, the book never factors it to (sin - cos)(sin + cos)).
    """
    if not include_db:
        return False
    try:
        candidates = _stored_candidates(expr)
    except Exception:
        return False

    for row in candidates:
        try:
            left, right = knowledge_patterns.parse_pair(row["left"], row["right"])
        except Exception:
            continue
        for source, target in ((left, right), (right, left)):
            try:
                new_expr = knowledge_patterns.apply_identity(expr, source, target)
            except Exception:
                continue
            if new_expr is None or to_latex(new_expr) == to_latex(expr):
                continue
            if any(isinstance(hole, Wild) for hole in new_expr.free_symbols):
                continue  # a hole stayed unfilled: not a real rewrite
            return True
    return False


def first_formula_move(expr: Expr, state: _SolveState,
                       include_db: bool):
    """The first STORED-identity rewrite available in this node's subtree,
    as (new_expr, name, formula, row_id) -- or None when no stored formula
    applies.

    This is the op-phase twin of the walk's formula step: the same
    candidate generator (formulas forward, then backwards, recursing into
    children) with everything that is not a stored formula skipped --
    rules, factor peels, SymPy moves and derived rearrangements all carry
    a None/derived formula and are filtered by `formula is not None`
    (derived ones never come up at all: with_derived=False). A strategy
    phase of  {"do": "apply_formula"}  uses it, so a route can apply one
    formula at a time -- deterministically, with the formula's name and
    statement shown -- instead of letting a free walk wander.

    Formula directions that EXPAND a compact trig call of a variable
    angle are skipped (see _expands_variable_angle): a route must not
    spell out what its own previous step just compressed.
    """
    if not include_db:
        return None

    blocked: set = set()
    try:
        candidates = _stored_candidates(expr)
    except Exception:
        candidates = []
    for row in candidates:
        try:
            left, right = knowledge_patterns.parse_pair(row["left"], row["right"])
        except Exception:
            continue
        for backwards in (False, True):
            source = right if backwards else left
            if _expands_variable_angle(source):
                blocked.add(f"{row['name']} (backwards)" if backwards
                            else row["name"])

    for new_expr, name, formula, row in _iter_changes(
            expr, SIMPLIFY_RULES, include_db, state,
            with_derived=False, direction="both"):
        if formula is None:
            continue  # a rule/factor move, not a stored formula
        if name in blocked:
            continue  # an expansion of a compact call: the walk's job
        return new_expr, name, formula, row
    return None


def _stored_candidates(expr: Expr) -> list:
    """Rows that could apply here: shape match, family, then functions.

    Each source gets its OWN budget. A shared cap let the shape matches
    (14 rows for an expression full of sin/cos) leave two slots for the
    function pass -- which is how a row reachable only from its other side
    (sin(x)/cos(x) -> "Tangent as a Quotient") went missing.

    The function pass includes ALL function-matching rows: _FUNCTION_FETCH
    is a bound on how many rows we READ, not a rank on which ones we keep,
    and the fetched list is never sliced. A row imported later (a higher
    id) must never be structurally invisible -- with a top-N cut, every
    formula added after N older rows mentioning the same functions
    disappears from _iter_changes for good (the gap batch: "Tan plus Cot
    is Two over Sin Two" sat at id position 61 of 62 and was offered to
    nobody).
    Determinism comes from sorting the full set ascending by id -- the same
    input always yields the same candidates in the same order -- not from
    cutting the list: id order already made this source independent of how
    often earlier solves happened to bump other rows (usage ordering made
    the same input produce different candidates in run 1 and run 10).
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
    for row in sorted(matching, key=lambda r: r["id"]):
        if row["id"] in seen_ids:
            continue
        candidates.append(row)
        seen_ids.add(row["id"])
    return candidates


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


# Per-source budgets: a later source may not be starved by an earlier one.
_STORED_ROW_CAP = 16     # shape + family matches
_FUNCTION_FETCH = 500    # rows read before choosing (a bound, not a rank)

# Never derive from more than this many rows at one node: every derivation
# costs a SymPy proof, and the most useful rows sort first anyway.
_DERIVE_ROW_CAP = 8


def _derived_hits(expr: Expr, state: _SolveState):
    """Rearrangements WE derive from stored rows (search route only).

    Rows come from THREE sources, because the row that unlocks this node
    may not shape-match it at all:
      1. rows that shape-matched or share the node's family
      2. rows mentioning the same FUNCTIONS ("1 - sin^2" -> sin -> the
         Pythagorean row -> derived "1 - sin^2 = cos^2")
    """
    try:
        derived_rows = _stored_candidates(expr)
        connection = knowledge_store.get_connection()
        seen_ids = {row["id"] for row in derived_rows}
        for row in knowledge_store.find_by_groups(connection, classify(expr)):
            if row["id"] not in seen_ids:
                derived_rows.append(row)
                seen_ids.add(row["id"])
        for row in knowledge_store.find_by_functions(
                connection, mentioned_functions(expr)):
            if row["id"] not in seen_ids:
                derived_rows.append(row)
                seen_ids.add(row["id"])
                if len(derived_rows) >= _DERIVE_ROW_CAP:
                    break
    except Exception:
        return  # no database: no derived candidates

    for row in derived_rows[:_DERIVE_ROW_CAP]:
        if f"Rearrange: {row['name']}" in state.banned:
            continue
        try:
            variants = derive_mod.derive_variants(
                *knowledge_patterns.parse_pair(row["left"], row["right"]))
        except Exception:
            continue
        for variant in variants:
            try:
                new_expr = knowledge_patterns.apply_identity(
                    expr, variant["left"], variant["right"])
            except Exception:
                continue
            if new_expr is None or to_latex(new_expr) == to_latex(expr):
                continue
            if any(isinstance(hole, Wild) for hole in new_expr.free_symbols):
                continue  # a hole stayed unfilled: not a real rewrite
            yield (new_expr, f"Rearrange: {row['name']}",
                   variant["statement"], row["id"])
