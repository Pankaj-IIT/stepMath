"""
signature.py -- the "shape" of an expression.

To find a matching formula without reading 10 million rows, we describe each
expression by its skeleton only, ignoring the actual names and numbers:

    sin(x)^2 + cos(x)^2   ->   Add(Pow(sin(_),#), Pow(cos(_),#))

    _   stands for any symbol   (x, y, theta ... all look the same)
    #   stands for any number   (2, 3.14 ... all look the same)
    %   stands for a HOLE (?a) when signing a formula -- a hole can be
        filled with anything, even a whole subtree like (x + y), so the
        stored shape must be allowed to match more than it shows
    the rest are the operations and function names

The stored shape is used as a SQL LIKE pattern (that is why % and _ work as
wildcards there), so a formula  sin(?c)  is stored as  sin(%)  and found
for  sin(x)  as well as  sin(x + y).

A stored formula and the expression you are simplifying get a matching
skeleton when they belong together, so we can look up candidates with one
indexed query instead of scanning the whole table.
"""

from sympy import Add, Expr, Mul, Pow, Symbol, Wild


def signature_of(expr: Expr) -> str:
    """Return the shape string for this expression (see module docstring)."""
    # NOTE: we deliberately do NOT fully evaluate here. Evaluating patterns
    # that contain holes behaves differently from evaluating your typed
    # maths (SymPy pulls the sign out of sin(?a - ?b) but not of sin(x-y)),
    # and the two shapes would stop lining up. Only the harmless case below
    # is normalised -- negative-power nesting.
    if isinstance(expr, Pow) and expr.base.is_Pow:
        # (cos(x)^2)^-1 and cos(x)^-2 are the same maths written two ways;
        # fold the exponents when both are plain numbers.
        base, exponent = expr.base, expr.exp
        while isinstance(base, Pow) and base.exp.is_Number and exponent.is_Number:
            base, exponent = base.base, base.exp * exponent
        expr = Pow(base, exponent)

    return _shape(expr)


def _shape(expr: Expr) -> str:
    """Shape of one expression (recursion helper for signature_of)."""
    # A hole: it may be filled with anything, even a whole subtree, so it
    # becomes '%' -- which, in a LIKE comparison, swallows any length.
    if isinstance(expr, Wild):
        return "%"

    # Numbers: 5, 3.14, -2 ...
    if expr.is_Number:
        return "#"

    # Plain names: x, y, pi ...
    if isinstance(expr, Symbol):
        return "_"

    # Anything with parts inside: Add, Mul, Pow, sin, cos ...
    if expr.args:
        name = expr.func.__name__
        parts = [signature_of(part) for part in expr.args]

        # Addition and multiplication may be written in any order, so the
        # parts are sorted --  x + 2*pi  and  2*pi + x  must get the SAME
        # shape, otherwise the index would miss one of them.
        #
        # The sort key treats % (a hole) and _ (a symbol) as the same
        # character. Without that, a pattern part "%" sorts BEFORE "Mul("
        # while the matching node part "_" sorts AFTER it -- the two sides
        # would list their parts in opposite orders and never match.
        if isinstance(expr, (Add, Mul)):
            parts.sort(key=lambda part: part.replace("%", "_"))

        return f"{name}({','.join(parts)})"

    # A named leaf with no parts (rare) -- keep the name, e.g. "pi"
    return expr.func.__name__
