"""
ops.py -- the small catalogue of ready-made maths operations.

A method step can say  {"op": "factor"}  and this is where "factor" is
defined: a name, a human label for the step, and the SymPy function that
does the work.

Keep this list small and obvious. Every entry must be a pure expression
transformation: expression in, expression out.
"""

from typing import Optional, Tuple

from sympy import Add, Expr, Mul, Pow, S, cancel, collect, expand, expand_trig, \
    factor, powsimp, simplify, together, trigsimp

from ..algebra.parser import normalize
from ..algebra.rules import changed  # the one shared "did it change?" check
from .gate import _to_sin_cos
from .patterns import _evaluated, _exact_root

def _signed_parts(term: Expr) -> "Tuple[int, Expr]":
    """(sign, core): pull a leading minus out of one evaluated term.

        -cos(x)**8  ->  (-1, cos(x)**8)
        -4          ->  (-1, 4)
    """
    if term.is_Number:
        if term.is_negative:
            return -1, -term
        return 1, term

    sign = 1
    parts = []
    for part in (term.args if term.is_Mul else [term]):
        if part == S.NegativeOne:
            sign = -1
        elif part != S.One:
            parts.append(part)
    return sign, Mul(*parts) if parts else S.One


def _root_of_term(term: Expr, total: int, half: int) -> "Expr | None":
    """The base whose `half`-th power rebuilds `term`, or None.

    `total` is the full (even) exponent the term carries:
        sin(x)^8 (total 8, half 4)  ->  sin(x)^4
        x^2       (total 2, half 1) ->  x
        4         (total 2)         ->  2   (exact whole-number root only)
    """
    if term.is_Pow:
        base = term.base
        return base if half == 1 else Pow(base, half, evaluate=False)

    if term.is_Number:
        return _exact_root(term, total)  # None when no exact whole root

    return None


def factor_once(expr: Expr) -> Optional[Tuple[Expr, str]]:
    """Peel exactly ONE level of a difference of squares.

        sin(x)^8 - cos(x)^8  ->  (sin(x)^4 - cos(x)^4)*(sin(x)^4 + cos(x)^4)
        x^2 - 4              ->  (x - 2)*(x + 2)

    Only a two-term sum  a^(2k) - b^(2k)  (k >= 1, opposite signs)
    qualifies, so one call factors exactly one level.  The next level
    happens on the caller's NEXT move (a phase's `until: stuck` loop, or
    the walk), which is what lets a stored formula fire between the
    levels -- the book's "factor, apply a formula, factor again".
    """
    if not (expr.is_Add and len(expr.args) == 2):
        return None

    signed = [_signed_parts(_evaluated(term)) for term in expr.args]
    signs = [sign for sign, _ in signed]
    if sorted(signs) != [-1, 1]:
        return None  # not "one positive term minus one positive term"

    positive = next(core for sign, core in signed if sign == 1)
    negative = next(core for sign, core in signed if sign == -1)

    if not (positive.is_Pow and positive.exp.is_Integer and positive.exp >= 2):
        return None
    total = int(positive.exp)
    if total % 2 != 0:
        return None
    half = total // 2

    # The negative term must carry the same total power (or be its exact root).
    if negative.is_Pow:
        if not (negative.exp.is_Integer and int(negative.exp) == total):
            return None
    elif not negative.is_Number:
        return None

    base_a = _root_of_term(positive, total, half)
    base_b = _root_of_term(negative, total, half)
    if base_a is None or base_b is None:
        return None
    if base_a == base_b:
        return None  # a^(2k) - a^(2k) is just 0; nothing to show

    # A stored formula that owns this node is the better move: it shows the
    # formula's name and statement, and peeling would run past a
    # recognizable form (sin^2 - cos^2 belongs to the double-angle row).
    from ..algebra.steps import formula_applies_here  # lazy: steps imports ops
    if formula_applies_here(expr):
        return None

    # -base_b (evaluated) keeps numbers clean:  x^2 - 4  ->  (x - 2), not (x - 1*2).
    left = Add(base_a, -base_b, evaluate=False)
    right = Add(base_a, base_b, evaluate=False)
    new_expr = normalize(Mul(left, right, evaluate=False))
    if not changed(expr, new_expr):
        return None
    return new_expr, "Factor as difference of squares"


def _factor_once_op(expr: Expr) -> Expr:
    """The catalogue form of factor_once: a bare expression out (the
    (expression, label) tuple stays inside, for run_operation's label)."""
    hit = factor_once(expr)
    return hit[0] if hit is not None else expr


def _apply_formula_op(expr: Expr) -> Expr:
    """The catalogue form of apply_formula: a bare expression out.

    Stateless (its own throwaway state), because run_operation calls the
    catalogue function with the expression alone. The move itself is the
    shared steps.first_formula_move -- the first verified stored identity
    that applies in the node's subtree, forwards then backwards."""
    from ..algebra.latex import to_latex
    from ..algebra.steps import _SolveState, first_formula_move

    state = _SolveState(expr, frozenset(), to_latex(expr))
    move = first_formula_move(expr, state, True)
    return move[0] if move is not None else expr


# name -> (label shown to the user, function)
OPERATIONS: dict = {
    "expand": ("Expand brackets", expand),
    "factor": ("Factorise", factor),
    "factor_once": ("Factor as difference of squares", _factor_once_op),
    "simplify": ("Simplify", simplify),
    "cancel": ("Cancel common factors", cancel),
    "trigsimp": ("Simplify trigonometry", trigsimp),
    "powsimp": ("Combine powers", powsimp),
    "expand_trig": ("Expand trigonometry", expand_trig),
    "collect": ("Collect like terms", lambda e: _collect(e)),
    # The elementary trig route: write everything in sin/cos, then gather
    # one denominator. Together these are what a student does by hand
    # before the Pythagorean identity can fire.
    "together": ("Take the LCM", together),
    "to_sin_cos": ("Write in terms of sin and cos", _to_sin_cos),
    # One stored-identity rewrite at this node. A strategy phase of
    # {"do": "apply_formula"} goes through strategies._phase_move instead,
    # where the same move is shown with the formula's name and statement.
    "apply_formula": ("Apply formula", _apply_formula_op),
}


def _collect(expr: Expr) -> Expr:
    """Collect terms by every variable in the expression."""
    variables = sorted(expr.free_symbols, key=str)
    return collect(expr, variables) if variables else expr


def known_operation(name: str) -> bool:
    """Is this op name in our catalogue?"""
    return name in OPERATIONS


def run_operation(name: str, expr: Expr) -> Optional[Tuple[Expr, str]]:
    """Run one catalogue operation.

    Returns (new_expression, step label) when it changed something,
    otherwise None -- so a method step that does nothing is simply skipped.
    """
    if name not in OPERATIONS:
        raise ValueError(f'Unknown operation "{name}". Known: {", ".join(sorted(OPERATIONS))}')

    label, function = OPERATIONS[name]
    new_expr = function(expr)

    if not changed(expr, new_expr):
        return None
    return new_expr, label
