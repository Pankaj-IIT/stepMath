"""
ops.py -- the small catalogue of ready-made maths operations.

A method step can say  {"op": "factor"}  and this is where "factor" is
defined: a name, a human label for the step, and the SymPy function that
does the work.

Keep this list small and obvious. Every entry must be a pure expression
transformation: expression in, expression out.
"""

from typing import Callable, Optional, Tuple

from sympy import Expr, cancel, collect, expand, expand_trig, factor, powsimp, simplify, srepr, trigsimp

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

    if srepr(new_expr) == srepr(expr):
        return None
    return new_expr, label
