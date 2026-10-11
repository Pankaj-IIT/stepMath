"""
steps.py -- runs the rules again and again to build the step list.

The idea:

    start with what you typed
        -> try every rule on it
        -> keep the FIRST change any rule produces
        -> show that change as one step
        -> repeat until no rule can improve it any more

Two safety nets stop this loop from running forever:

    1. a rule only counts when the displayed maths actually changed
       (so a rule cannot bounce the expression back and forth)
    2. MAX_STEPS caps the total number of steps
"""

from sympy.core.expr import Expr

from ..knowledge import patterns as knowledge_patterns
from ..knowledge import store as knowledge_store
from .latex import to_latex, to_latex_as_written
from .rules import SIMPLIFY_RULES, Rule

# Hard cap: never show the user more than this many steps.
MAX_STEPS = 20

# What the web API returns: a list of {"name": ..., "latex": ...}.
Step = dict


def simplify_steps(expr: Expr, include_db: bool = True) -> list[Step]:
    """Give the list of steps needed to simplify this expression.

    include_db=False skips the formula database (handy in tests, and it is
    what keeps the app working when the database file is missing).
    """
    return _build_steps(expr, SIMPLIFY_RULES, include_db)


def _build_steps(expr: Expr, rules: list[Rule], include_db: bool = True) -> list[Step]:
    # The first line shows the expression exactly as it was typed.
    steps: list[Step] = [{"name": "Initial expression", "latex": to_latex_as_written(expr)}]
    seen = {steps[0]["latex"]}  # remember what we have already displayed
    current = expr

    for _ in range(MAX_STEPS):
        result = _apply_once(current, rules, include_db)
        if result is None:
            break  # nothing left to do -- we are done

        current, name = result
        latex = to_latex(current)

        if latex in seen:
            break  # we are going around in circles -- stop

        seen.add(latex)
        steps.append({"name": name, "latex": latex})

    return steps


def _apply_once(expr: Expr, rules: list[Rule],
                include_db: bool = True) -> "tuple[Expr, str] | None":
    """Make exactly one change anywhere in the expression.

    Order, most trustworthy first:
        1. our own code rules   (rules.py)
        2. identities from the formula database (looked up by shape)
        3. one level deeper into the expression
    """
    # 1) Can a code rule improve THIS piece of maths?
    for rule in rules:
        hit = rule(expr)
        if hit is None:
            continue

        new_expr, name = hit
        if to_latex(new_expr) == to_latex(expr):
            continue  # looks the same on screen -- not worth a step

        return new_expr, name

    # 2) Does the formula database know an identity for this shape?
    if include_db:
        hit = _database_identity(expr)
        if hit is not None:
            return hit

    # 3) Otherwise try each child (the parts inside the brackets, so to speak).
    for index, part in enumerate(expr.args):
        hit = _apply_once(part, rules, include_db)
        if hit is None:
            continue

        new_part, name = hit
        args = list(expr.args)
        args[index] = new_part

        # Put the changed part back into the parent, keeping the parent
        # exactly as written (evaluate=False) so nothing else jumps ahead.
        rebuilt = expr.func(*args, evaluate=False)
        if to_latex(rebuilt) == to_latex(expr):
            continue

        return rebuilt, name

    return None


def _database_identity(expr: Expr) -> "tuple[Expr, str] | None":
    """Look for a matching identity in data/knowledge.db.

    Any problem here (file missing, table empty, formula unreadable) simply
    means "no identity found" -- the database never breaks the solver.
    """
    try:
        connection = knowledge_store.get_connection()
        candidates = knowledge_store.find_identities(connection, expr)

        for row in candidates:
            left, right = knowledge_patterns.parse_pair(row["left"], row["right"])
            new_expr = knowledge_patterns.apply_identity(expr, left, right)
            if new_expr is None:
                continue
            if to_latex(new_expr) == to_latex(expr):
                continue

            # Count the hit so this formula sorts higher next time.
            knowledge_store.bump_usage(connection, row["id"])
            connection.commit()
            return new_expr, row["name"]

    except Exception:
        return None

    return None
