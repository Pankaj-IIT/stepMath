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

from .latex import to_latex, to_latex_as_written
from .rules import SIMPLIFY_RULES, Rule

# Hard cap: never show the user more than this many steps.
MAX_STEPS = 20

# What the web API returns: a list of {"name": ..., "latex": ...}.
Step = dict


def simplify_steps(expr: Expr) -> list[Step]:
    """Give the list of steps needed to simplify this expression."""
    return _build_steps(expr, SIMPLIFY_RULES)


def _build_steps(expr: Expr, rules: list[Rule]) -> list[Step]:
    # The first line shows the expression exactly as it was typed.
    steps: list[Step] = [{"name": "Initial expression", "latex": to_latex_as_written(expr)}]
    seen = {steps[0]["latex"]}  # remember what we have already displayed
    current = expr

    for _ in range(MAX_STEPS):
        result = _apply_once(current, rules)
        if result is None:
            break  # nothing left to do -- we are done

        current, name = result
        latex = to_latex(current)

        if latex in seen:
            break  # we are going around in circles -- stop

        seen.add(latex)
        steps.append({"name": name, "latex": latex})

    return steps


def _apply_once(expr: Expr, rules: list[Rule]) -> "tuple[Expr, str] | None":
    """Make exactly one change anywhere in the expression.

    We look at the outside first, and only go one level deeper when no rule
    can improve this level. One change per call keeps every step easy to read.
    """
    # 1) Can a rule improve THIS piece of maths?
    for rule in rules:
        hit = rule(expr)
        if hit is None:
            continue

        new_expr, name = hit
        if to_latex(new_expr) == to_latex(expr):
            continue  # looks the same on screen -- not worth a step

        return new_expr, name

    # 2) Otherwise try each child (the parts inside the brackets, so to speak).
    for index, part in enumerate(expr.args):
        hit = _apply_once(part, rules)
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
