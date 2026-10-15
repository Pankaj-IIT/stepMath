"""
prove.py -- the PROVE handler: show HOW two sides are equal.

Strategy (the classic one): move everything to one side and drive the
result to zero, showing each transformation as a step:

    prove that sin(x)^2 + cos(x)^2 = 1
        1. To prove ..................  sin(x)^2 + cos(x)^2 = 1
        2. Bring everything to one side  sin(x)^2 + cos(x)^2 - 1
        3. Simplify ..................  0
        4. Everything cancels: the statement is TRUE

The ladder of attempts is fixed and bounded -- never more than one step
per attempt, and we never claim a proof we could not reach: a non-zero
remainder ends with an honest "cannot prove" step.
"""

from typing import Tuple

from sympy import Expr, expand, expand_trig, simplify, trigsimp

from ..algebra.latex import to_latex
from ..knowledge.gate import _to_sin_cos
from ..knowledge.patterns import _evaluated

Step = dict

# One attempt per step, cheapest first. "simplify" is the catch-all last.
_LADDER: list = [
    ("Expand brackets", expand),
    ("Expand the trigonometry", expand_trig),
    ("Simplify the trigonometry", trigsimp),
    ("Write in terms of sin and cos", lambda e: simplify(_to_sin_cos(e))),
    ("Simplify", simplify),
]


def prove_steps(lhs: Expr, rhs: Expr) -> Tuple[list, bool]:
    """Walk the ladder for  lhs = rhs.  Returns (steps, proved?)."""
    steps: list = [{
        "name": "To prove",
        "latex": f"{to_latex(lhs)} = {to_latex(rhs)}",
    }]

    # Everything on one side: proving  lhs = rhs  is proving  lhs - rhs = 0.
    current = _evaluated(lhs - rhs)
    steps.append({
        "name": "Bring everything to one side",
        "latex": to_latex(current),
    })

    if current == 0:
        steps.append({"name": "Both sides are equal", "latex": "0"})
        return steps, True

    seen = {to_latex(current)}
    for name, attempt in _LADDER:
        try:
            new_expr = attempt(current)
        except Exception:
            continue  # a failed attempt is simply skipped

        latex = to_latex(new_expr)
        if latex in seen:
            continue  # this attempt added nothing new
        seen.add(latex)
        steps.append({"name": name, "latex": latex})
        current = new_expr

        if current == 0:
            break

    if current == 0:
        steps.append({
            "name": "Everything cancels: the statement is TRUE",
            "latex": "0",
        })
        return steps, True

    # Honest ending: we did NOT reach zero, so we do not claim a proof.
    steps.append({
        "name": "Cannot prove: not zero yet",
        "latex": to_latex(current),
    })
    return steps, False
