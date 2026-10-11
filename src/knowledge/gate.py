"""
gate.py -- the machine checker for formulas before we trust them.

Every AI-generated identity must prove itself:

    "is  left - right  really zero for ALL values?"

We ask SymPy. Three answers are possible:

    "pass"      SymPy proved it is zero           -> verified = 1
    "fail"      SymPy proved it is NOT zero       -> rejected, we show why
    "unknown"   SymPy could not decide            -> verified = 0,
                waits in the review queue for a human

A second, cheaper check runs when the proof fails: put a few random numbers
into the holes. If even one number makes the two sides differ, the formula
is definitely wrong.
"""

from typing import Tuple

from sympy import Expr, simplify, trigsimp

from .patterns import _evaluated


def verify_identity(left: Expr, right: Expr) -> Tuple[str, str]:
    """Check one identity. Returns (verdict, detail)."""
    difference = left - right
    attempts = []

    for attempt in (lambda e: simplify(e),
                    lambda e: trigsimp(e),
                    lambda e: simplify(trigsimp(e))):
        try:
            result = attempt(difference)
        except Exception:
            continue

        attempts.append(result)
        if result == 0:
            return "pass", "SymPy proved both sides are equal"

    # The two sides differ by a fixed amount (e.g. 1) -> definitely wrong.
    for result in attempts:
        if result.is_Number and result != 0:
            return "fail", f"counter-example: the two sides differ by {result}"

    # Could not prove it -- look for a concrete counter-example.
    counterexample = _find_counterexample(difference)
    if counterexample:
        return "fail", f"counter-example: {counterexample}"

    return "unknown", "SymPy could not prove or disprove it"


def _find_counterexample(expr: Expr) -> str:
    """Fill the holes with random numbers; return the first clash we find."""
    from sympy import Symbol, Wild  # local imports keep the module top clean
    from sympy.utilities.lambdify import lambdify
    import random

    wilds = [s for s in expr.free_symbols if isinstance(s, Wild)]
    if not wilds:
        return ""

    # lambdify cannot print a Wild, so we hand it ordinary symbols instead.
    plain = [Symbol(w.name) for w in wilds]
    try:
        values = lambdify(plain, _evaluated(expr).xreplace(dict(zip(wilds, plain))),
                          "math")
    except Exception:
        return ""

    for _ in range(8):
        assignment = {w: random.choice([-3, -2, -1.5, -1, 0.5, 1, 2, 3]) for w in wilds}
        try:
            result = values(*[assignment[w] for w in wilds])
        except Exception:
            continue
        if isinstance(result, complex):
            continue
        if abs(float(result)) > 1e-9:
            numbers = ", ".join(f"{w} = {assignment[w]}" for w in wilds)
            return f"{numbers} gives {result}"

    return ""
