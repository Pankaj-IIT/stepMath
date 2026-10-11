"""
signature.py -- the "shape" of an expression.

To find a matching formula without reading 10 million rows, we describe each
expression by its skeleton only, ignoring the actual names and numbers:

    sin(x)^2 + cos(x)^2   ->   Add(Pow(sin(_),#), Pow(cos(_),#))

    _   stands for any symbol   (x, y, theta, ?a ... all look the same)
    #   stands for any number   (2, 3.5 ... all look the same)
    the rest are the operations and function names

A stored formula and the expression you are simplifying get the same
skeleton when they belong together, so we can look up candidates with one
indexed query instead of scanning the whole table.
"""

from sympy import Expr, Symbol, Wild


def signature_of(expr: Expr) -> str:
    """Return the shape string for this expression."""
    # Numbers: 5, 3.14, -2 ...
    if expr.is_Number:
        return "#"

    # Names: x, y, and the ?a holes in a pattern
    if isinstance(expr, (Symbol, Wild)):
        return "_"

    # Anything with parts inside: Add, Mul, Pow, sin, cos ...
    if expr.args:
        name = expr.func.__name__
        inner = ",".join(signature_of(part) for part in expr.args)
        return f"{name}({inner})"

    # A named leaf with no parts (rare) -- keep the name, e.g. "pi"
    return expr.func.__name__
