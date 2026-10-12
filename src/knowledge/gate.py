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

from sympy import Expr, Symbol, Wild, expand_trig, simplify, trigsimp

from .patterns import _evaluated


def verify_identity(left: Expr, right: Expr) -> Tuple[str, str]:
    """Check one identity. Returns (verdict, detail)."""
    difference = left - right
    attempts = []

    # Tried in order, from quick to heavy. The last one rewrites everything
    # into sin/cos, which is what finally cracks cotangent and tangent
    # formulas like  cot(A+B) = (cotA cotB - 1)/(cotB + cotA).
    for attempt in (lambda e: simplify(e),
                    lambda e: trigsimp(e),
                    lambda e: simplify(trigsimp(e)),
                    lambda e: simplify(expand_trig(e)),
                    lambda e: simplify(_to_sin_cos(e))):
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
    counterexample = _sample_failure(difference, set())
    if counterexample:
        return "fail", f"counter-example: {counterexample}"

    return "unknown", "SymPy could not prove or disprove it"


def verify_solution(left: Expr, right: Expr, unknown: "Symbol",
                    value: Expr) -> Tuple[str, str]:
    """Does this answer really solve the equation?

    We put the answer into the equation and simplify the difference:

        sin(?t) = 0   with   ?t = ?n*pi   ->   sin(n*pi) - 0  ->  0  (proved)

    Holes that appear ONLY in the answer (?n here) are treated as integers,
    because that is what "n" means in these formulas (n in Z). The other
    holes (like the alpha in sin(?t) = sin(?a)) stay free.
    """
    equation_holes = set(left.free_symbols) | set(right.free_symbols)
    extra_holes = [s for s in value.free_symbols
                   if isinstance(s, Wild) and s not in equation_holes]

    integer_map = {hole: Symbol(hole.name, integer=True) for hole in extra_holes}
    bound = value.subs(integer_map)
    check = left.subs({unknown: bound}) - right

    attempts = []
    for attempt in (lambda e: simplify(e),
                    lambda e: trigsimp(e),
                    lambda e: simplify(expand_trig(e)),
                    lambda e: simplify(_to_sin_cos(e))):
        try:
            result = attempt(check)
        except Exception:
            continue
        attempts.append(result)
        if result == 0:
            return "pass", "SymPy proved the answer satisfies the equation"

    for result in attempts:
        if result.is_Number and result != 0:
            return "fail", f"counter-example: the answer gives {result}"

    # Formulas with (-1)^n: SymPy cannot pull (-1)^n out of a sine, so we
    # check n even and n odd separately -- between them they cover every
    # integer, which is exactly how these formulas are proved by hand.
    if _prove_by_parity(check, set(integer_map.values())):
        return "pass", "SymPy proved it for even and odd n"

    counterexample = _sample_failure(check, set(integer_map.values()))
    if counterexample:
        return "fail", f"counter-example: {counterexample}"

    return "unknown", "SymPy could not prove or disprove the answer"


def _prove_by_parity(expr: Expr, integer_symbols: set) -> bool:
    """Prove something about integers by trying even and odd cases."""
    from itertools import product

    symbols = [s for s in integer_symbols if s in expr.free_symbols]
    if not symbols or len(symbols) > 2:
        return False  # keep it simple: one or two integer holes

    for choices in product((0, 1), repeat=len(symbols)):
        replacements = {}
        for symbol, choice in zip(symbols, choices):
            fresh = Symbol(f"{symbol.name}_parity", integer=True)
            replacements[symbol] = 2 * fresh + choice

        try:
            result = simplify(expr.subs(replacements))
        except Exception:
            return False
        if result != 0:
            return False

    return True


def _sample_failure(expr: Expr, integer_symbols: set) -> str:
    """Try concrete numbers in the holes; return the first clash we find."""
    from sympy import Symbol, Wild
    from sympy.utilities.lambdify import lambdify
    import random

    wilds = [s for s in expr.free_symbols if isinstance(s, Wild)]
    if wilds:
        # Lambdify cannot print a Wild, so use ordinary symbols instead.
        plain_wilds = [Symbol(s.name) for s in wilds]
        expr = expr.xreplace(dict(zip(wilds, plain_wilds)))
        free = plain_wilds + [s for s in expr.free_symbols if s not in plain_wilds]
    else:
        free = list(expr.free_symbols)

    if not free:
        return ""

    try:
        values = lambdify(free, _evaluated(expr), "math")
    except Exception:
        return ""

    integers = [s for s in free if s in integer_symbols]
    others = [s for s in free if s not in integers]

    for _ in range(8):
        assignment = {}
        for symbol in integers:
            assignment[symbol] = random.choice([-3, -2, -1, 0, 1, 2, 3])
        for symbol in others:
            assignment[symbol] = random.choice([-3, -2, -1.5, -1, 0.5, 1, 2, 3])

        try:
            result = values(*[assignment[symbol] for symbol in free])
        except Exception:
            continue

        if isinstance(result, complex):
            continue
        if abs(float(result)) > 1e-9:
            numbers = ", ".join(f"{s} = {assignment[s]}" for s in free)
            return f"{numbers} gives {result}"

    return ""


def _to_sin_cos(expr: Expr) -> Expr:
    """Rewrite tan/cot/sec/csc away, leaving only sin and cos."""
    from sympy import cos, csc, cot, sec, sin, tan

    return (
        expr
        .replace(cot, lambda t: cos(t) / sin(t))
        .replace(tan, lambda t: sin(t) / cos(t))
        .replace(sec, lambda t: 1 / cos(t))
        .replace(csc, lambda t: 1 / sin(t))
    )
