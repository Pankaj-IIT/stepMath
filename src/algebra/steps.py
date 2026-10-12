"""
steps.py -- runs the rules again and again to build the step list.

The idea:

    start with what you typed
        -> try every rule on it
        -> keep the FIRST change any rule produces
        -> show that change as one step
        -> repeat until no rule can improve it any more

Three safety nets keep this sensible:

    1. a step only counts when the displayed maths actually changed
       (so a rule cannot bounce the expression back and forth)
    2. MAX_STEPS caps the total number of steps
    3. big moves -- expanding and database formulas -- only touch parts a
       formula CREATED. Parts you typed (and parts that merely contain
       letters you typed) stay open for business, so
           2*sin(x)^2 + cos(2*x)  can still reach  1
       via the Pythagorean identity after cos(2*x) was rewritten.

For alternative routes through the same maths see alternative_ways().
"""

from sympy import srepr
from sympy import Add, Mul
from sympy.core.expr import Expr

from ..knowledge import patterns as knowledge_patterns
from ..knowledge import store as knowledge_store
from ..knowledge import runner as _method_runner
from ..knowledge.classify import classify
from .latex import to_latex, to_latex_as_written
from .parser import normalize
from .rules import SHAPE_RULES, SIMPLIFY_RULES, Rule

# Hard cap: never show the user more than this many steps.
MAX_STEPS = 20

# What the web API returns: a list of {"name": ..., "latex": ...}.
Step = dict


class _SolveState:
    """Everything one run of the solver needs to remember.

    banned   step names we promised not to use (alternative routes)
    original the srepr (structural fingerprint) of every node the user
             typed -- the one honest way to tell "your maths" apart from
             "maths a formula produced". (Object identity does NOT work:
             SymPy caches and reuses nodes, so the same object can appear
             on both sides.)
    """

    def __init__(self, original: Expr, banned: "frozenset[str]" = frozenset()):
        self.banned = banned
        self.original: set = set()

        pending = [original]
        while pending:
            node = pending.pop()
            self.original.add(srepr(node))
            pending.extend(node.args)

    def blocked(self, expr: Expr) -> bool:
        """Hold back the big moves (expanding, taking common, database)?

        Big moves are held back on a part that contains NONE of the
        structure the user typed -- i.e. it is pure formula output.
        Rewriting such a part again is how two formulas undo each other.

        As soon as any piece you typed survives inside the part, it stays
        open. That hand-over between formulas is exactly what makes

            -2*sin(x)^2 - cos(2*x)
              -> (Double Angle) -2*sin^2 - cos^2
              -> (take out -1)  -(sin^2 + cos^2)
              -> (Pythagorean)  -(-1)  ->  -1

        reach its answer instead of stalling half-way.

        Tidy-up rules (numbers, like terms) are never blocked -- they
        only clean up, they cannot start an argument between formulas.
        """
        saw_structure = False

        pending = [expr]
        while pending:
            node = pending.pop()
            if not node.args:
                continue  # leaves (numbers, symbols) carry no structure
            saw_structure = True
            if srepr(node) in self.original:
                return False  # your maths is still in here -- stay open
            pending.extend(node.args)

        return saw_structure  # no parts at all, or entirely new structure


def simplify_steps(expr: Expr, include_db: bool = True) -> list[Step]:
    """Give the list of steps needed to simplify this expression.

    include_db=False skips the formula database (handy in tests, and it is
    what keeps the app working when the database file is missing).
    """
    steps, _ = _build_steps(expr, SIMPLIFY_RULES, include_db)
    return steps


def alternative_ways(expr: Expr, include_db: bool = True,
                     limit: int = 3) -> list[dict]:
    """OTHER routes that finish at the SAME answer as the normal route.

    The normal route takes the first move it finds. Here we ask: what if
    we had taken a different first move instead? Each run bans one more
    first move, so a different route surfaces each time. Routes that end
    somewhere else are dropped -- only ways that reach the same answer
    are kept (that is what a student wants to compare).

    Returns a list of {"name": first move, "steps": the whole route}.
    """
    main_steps, main_final = _build_steps(expr, SIMPLIFY_RULES, include_db)
    if len(main_steps) < 2:
        return []  # nothing happened, so nothing to compare

    ways: list[dict] = []
    banned: set = {main_steps[1]["name"]}  # the move we already used

    for _ in range(limit):
        steps, final = _build_steps(expr, SIMPLIFY_RULES, include_db,
                                    frozenset(banned))
        if len(steps) < 2:
            break  # no other move exists

        banned.add(steps[1]["name"])

        if _same_answer(final, main_final):
            # "final" is the SymPy expression (tests use it to prove the
            # routes are equal maths); the API's response model drops it.
            ways.append({"name": steps[1]["name"], "steps": steps,
                         "final": final})

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


def _same_answer(a: Expr, b: Expr) -> bool:
    """Do these two expressions mean the same maths?"""
    if to_latex(a) == to_latex(b):
        return True
    try:
        from sympy import simplify
        return simplify(a - b) == 0
    except Exception:
        return False


def _build_steps(expr: Expr, rules: list[Rule], include_db: bool = True,
                 banned: "frozenset[str]" = frozenset()
                 ) -> "tuple[list[Step], Expr]":
    """Walk the rules until nothing changes. Returns (steps, final expression)."""
    # The first line shows the expression exactly as it was typed.
    steps: list[Step] = [{"name": "Initial expression", "latex": to_latex_as_written(expr)}]
    seen = {steps[0]["latex"]}  # remember what we have already displayed
    current = expr

    state = _SolveState(expr, banned)

    for _ in range(MAX_STEPS):
        result = _apply_once(current, rules, include_db, state)
        if result is None:
            break  # nothing left to do -- we are done

        # The ban list only forces a different FIRST move. Once we have
        # one, the route may use every formula again -- otherwise an
        # alternative could never finish (it would need the very formula
        # we banned to complete its second half).
        state.banned = frozenset()

        current, name, formula = result
        latex = to_latex(current)

        if latex in seen:
            break  # we are going around in circles -- stop

        seen.add(latex)

        step: Step = {"name": name, "latex": latex}
        if formula:
            # The formula that was applied (from the database), so the
            # screen can show WHICH rule of maths was used.
            step["formula"] = formula
        steps.append(step)

    return steps, current


def _apply_once(expr: Expr, rules: list[Rule], include_db: bool,
                state: _SolveState) -> "tuple[Expr, str, str | None] | None":
    """Make exactly one change anywhere in the expression.

    Returns (new expression, step name, formula used) -- the formula is
    None for our own code rules (they are not in the database) and the
    statement of the row for database identities.

    Order, most trustworthy first:
        1. our tidy-up rules      (numbers, zeros, like terms, powers)
        2. an identity from the database
        3. expanding / taking out a common factor
        4. one level deeper into the expression

    Database formulas go BEFORE expanding and factoring so a step keeps the
    formula's real name ("Difference of Squares") instead of the generic
    "Take out the common factor".

    Big moves (2 and 3) are held back on parts that contain none of the
    structure the user typed -- otherwise two formulas undo each other
    (expand, un-expand, expand...). Tidy-up rules are harmless, so they
    may keep working anywhere, and any part still containing your own
    maths stays open: that is what lets  sin^2 + cos^2 = 1  finish the job
    after cos(2x) was rewritten.
    """
    big_moves_held_back = state.blocked(expr)
    tidy_rules = [rule for rule in rules if rule not in SHAPE_RULES]
    shape_rules = [rule for rule in rules if rule in SHAPE_RULES]

    # 1) Tidy-up rules work everywhere -- they only clean up.
    for rule in tidy_rules:
        hit = rule(expr)
        if hit is None:
            continue

        new_expr, name = hit
        if name in state.banned:
            continue  # saved for an alternative route
        if to_latex(new_expr) == to_latex(expr):
            continue  # looks the same on screen -- not worth a step

        return new_expr, name, None

    # 2) Does the formula database know an identity for this shape?
    if include_db and not big_moves_held_back:
        hit = _database_identity(expr, state)
        if hit is not None:
            return hit

    # 3) Expanding and taking out a common factor (SymPy's factor()).
    if not big_moves_held_back:
        for rule in shape_rules:
            hit = rule(expr)
            if hit is None:
                continue

            new_expr, name = hit
            if name in state.banned:
                continue
            if to_latex(new_expr) == to_latex(expr):
                continue

            return new_expr, name, None

    # 4) Otherwise try each child. Sums and products go FIRST: in a
    #    fraction like (sin^2 + cos^2)/cos^2 the numerator (an Add) is the
    #    interesting part, and it must get its formula before we start
    #    rewriting the denominator (a Pow).
    interesting = [i for i, part in enumerate(expr.args)
                   if isinstance(part, (Add, Mul)) and part.args]
    others = [i for i, part in enumerate(expr.args) if i not in interesting]

    for index in interesting + others:
        part = expr.args[index]
        hit = _apply_once(part, rules, include_db, state)
        if hit is None:
            continue

        new_part, name, formula = hit
        args = list(expr.args)
        args[index] = new_part

        # Put the changed part back into the parent, keeping the parent
        # exactly as written (evaluate=False) so nothing else jumps ahead.
        # normalize() strips any stray "1 *" this creates (e.g. after
        # sin^2+cos^2 became 1 inside  1/cos^2), so the result matches the
        # shapes the formula database stores.
        rebuilt = normalize(expr.func(*args, evaluate=False))
        if to_latex(rebuilt) == to_latex(expr):
            continue

        # The formula stays with the step, even though we rebuilt a parent.
        return rebuilt, name, formula

    return None


def _database_identity(expr: Expr, state: _SolveState
                       ) -> "tuple[Expr, str, str] | None":
    """Look for a matching identity in data/knowledge.db.

    Returns (new expression, step name, the formula itself) so the screen
    can show which formula was used. Any problem here (file missing, table
    empty, formula unreadable) simply means "no identity found" -- the
    database never breaks the solver.
    """
    try:
        connection = knowledge_store.get_connection()
        candidates = list(knowledge_store.find_identities(connection, expr))

        # The group route (classify(node) -> families -> members) runs
        # beside the shape index: it catches families the node only matches
        # by MEANING (e.g. the expanded spelling of a two-angle sum).
        tags = classify(expr)
        if tags:
            seen = {row["id"] for row in candidates}
            for row in knowledge_store.find_by_groups(connection, tags):
                if row["id"] not in seen:
                    candidates.append(row)

        for row in candidates:
            if row["name"] in state.banned:
                continue  # saved for an alternative route

            left, right = knowledge_patterns.parse_pair(row["left"], row["right"])
            new_expr = knowledge_patterns.apply_identity(expr, left, right)
            if new_expr is None:
                continue
            if to_latex(new_expr) == to_latex(expr):
                continue

            # Count the hit so this formula sorts higher next time.
            knowledge_store.bump_usage(connection, row["id"])
            connection.commit()

            return new_expr, row["name"], row["statement"]

    except Exception:
        return None

    return None
