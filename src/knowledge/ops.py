"""
ops.py -- the small catalogue of ready-made maths operations.

A method step can say  {"op": "factor"}  and this is where "factor" is
defined: a name, a human label for the step, and the SymPy function that
does the work.

Keep this list small and obvious. Every entry must be a pure expression
transformation: expression in, expression out.
"""

from typing import Optional, Tuple

from sympy import Expr, cancel, collect, expand, expand_trig, factor, powsimp, \
    simplify, together, trigsimp

from ..algebra.rules import changed  # the one shared "did it change?" check
from .gate import _to_sin_cos

# name -> (label shown to the user, function)
OPERATIONS: dict = {
    "expand": ("Expand brackets", expand),
    "factor": ("Factorise", factor),
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
