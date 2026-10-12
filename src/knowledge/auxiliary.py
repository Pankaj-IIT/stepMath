"""
auxiliary.py -- the "a*sin + b*cos" method from the notes (R-formula).

The picture's procedure for   a cos(t) + b sin(t) = c :

    1. divide everything by   R = sqrt(a^2 + b^2)
    2. note the equation only has a solution when |c| <= R
    3. the left side is now   cos(phi)*sin(t) + sin(phi)*cos(t)
       with  phi = atan2(...)   -- i.e. it IS  sin(t + phi)
    4. solve the simple equation  sin(t + phi) = c/R

Everything here is COMPUTED, not pattern-matched: the intermediates
contain real numbers (3/5, 4/3, ...), and a stored pattern could never
reliably recognise an expression SymPy has already evaluated. So these are
small functions that inspect the maths directly and rewrite it -- used by
the method runner (expression steps) and by the solver (equation steps).
"""

from typing import Optional, Tuple

from sympy import Add, Expr, Mul, atan2, cos, sin, sqrt
from sympy.core.numbers import Number

from .classify import flatten

# Which side the terms are written in:  c*sin(t) + s*cos(t)  or the reverse.
SinFirst = Tuple[Expr, Expr, Expr]  # (sin coefficient, cos coefficient, t)


def _unwrap(term: Expr) -> "Tuple[Optional[Number], Optional[Expr]]":
    """Split  K*sin(x)  into  (K, sin(x)).

    A subtraction like  -4*cos(x)  parses as a NESTED product
    (-1 * (4*cos(x))), so flatten() (shared with classify.py) opens it up.
    A bare sin(x) counts as 1*sin(x).
    """
    if term.func in (sin, cos):
        return Number(1), term
    if not term.is_Mul:
        return None, None

    numbers = []
    functions = []
    other = []
    for part in flatten(term):
        if part.is_number:
            numbers.append(part)
        elif part.func in (sin, cos):
            functions.append(part)
        else:
            other.append(part)

    if other or len(functions) != 1:
        return None, None  # not "a number times ONE trig function"
    if not numbers:
        return Number(1), functions[0]
    return Mul(*numbers), functions[0]  # Mul evaluates: -1*4 -> -4


def _ratio_sum(expr: Expr) -> Optional[SinFirst]:
    """Find  c*sin(t) + s*cos(t)  (either writing order) with NUMBER c, s.

    Returns (c, s, t): the coefficient OF SINE, the coefficient OF COSINE,
    and the shared angle. Only the pair itself counts -- anything else in
    the sum is ignored by the caller.
    """
    terms = expr.args if expr.is_Add else [expr]
    sin_coeff = cos_coeff = None
    angle = None

    for term in terms:
        coefficient, function = _unwrap(term)
        if coefficient is None:
            continue
        if function.func is sin and sin_coeff is None:
            sin_coeff, angle = coefficient, function.args[0]
        elif function.func is cos and cos_coeff is None:
            if angle is not None and function.args[0] != angle:
                return None  # two different angles: not the method
            cos_coeff, angle = coefficient, function.args[0]
        else:
            return None  # a second sine/cos term we cannot handle

    if sin_coeff is None or cos_coeff is None:
        return None
    return sin_coeff, cos_coeff, angle


def _phi(sin_coeff: Expr, cos_coeff: Expr) -> Expr:
    """phi with  cos(phi) = sin_coeff  and  sin(phi) = cos_coeff.

    atan2(sin(phi), cos(phi)) = atan2(cos_coeff, sin_coeff) -- correct for
    every sign, unlike a bare tan^-1 which assumes the first quadrant.
    """
    return atan2(cos_coeff, sin_coeff)


# ---------------------------------------------------------------------------
# Expression steps (used by the method runner)
# ---------------------------------------------------------------------------
def auxiliary_divide(expr: Expr) -> "Optional[Tuple[Expr, str]]":
    """Take out  R = sqrt(a^2 + b^2)  from the two trig terms.

        3*sin(x) + 4*cos(x) + 5
          ->  5*(3*sin(x)/5 + 4*cos(x)/5) + 5        "Take out R = sqrt(a^2+b^2)"
    """
    if not expr.is_Add:
        return None

    pair = _ratio_sum(expr)
    if pair is None:
        return None

    sin_coeff, cos_coeff, angle = pair
    radius = sqrt(sin_coeff ** 2 + cos_coeff ** 2)
    if radius == 1 or not radius.is_number:
        return None  # nothing to take out (or cannot: keep looking)

    # R * (c/R * sin + s/R * cos), keeping whatever else was in the sum.
    inner = Add(
        Mul(sin_coeff / radius, sin(angle), evaluate=False),
        Mul(cos_coeff / radius, cos(angle), evaluate=False),
        evaluate=False,
    )
    combined = Mul(radius, inner, evaluate=False)

    rebuilt = []
    placed = False
    for term in expr.args:
        coefficient, function = _unwrap(term)
        if coefficient is not None and function is not None:
            # Both trig terms are now inside `combined`; keep their place
            # once and drop the second one.
            if not placed:
                rebuilt.append(combined)
                placed = True
            continue
        rebuilt.append(term)

    return Add(*rebuilt, evaluate=False), "Take out R = sqrt(a^2 + b^2)"


def auxiliary_as_sin(expr: Expr) -> "Optional[Tuple[Expr, str]]":
    """Recognise  c*sin(t) + s*cos(t)  (with c^2 + s^2 = 1) as sin(t + phi).

        5*(3*sin(x)/5 + 4*cos(x)/5)
          ->  5*sin(x + atan(4/3))    "Write as R sin(theta + phi)"

    Works on the whole expression or on one term of a sum, so a carried
    "+ 5" from the divide step does not hide the pattern.
    """
    found, label = _as_sin_node(expr)
    if found is not None:
        return found, label

    if expr.is_Add:
        for index, term in enumerate(expr.args):
            found, label = _as_sin_node(term)
            if found is not None:
                args = list(expr.args)
                args[index] = found
                return Add(*args, evaluate=False), label

    return None


def _as_sin_node(node: Expr) -> "Tuple[Optional[Expr], Optional[str]]":
    """Turn this node into its sin(t + phi) form, or (None, None)."""
    # Form 1: the ratios sit behind a factor:  R*(c*sin + s*cos)
    if node.is_Mul and len(node.args) == 2 and node.args[0].is_number:
        radius, inner = node.args
        pair = _ratio_sum(inner) if inner.is_Add else _ratio_sum(
            Add(inner, evaluate=False))
        if pair is not None and _is_unit_pair(*pair[:2]):
            sin_coeff, cos_coeff, angle = pair
            return (Mul(radius, sin(angle + _phi(sin_coeff, cos_coeff)),
                        evaluate=False),
                    "Write as R sin(theta + phi)")

    # Form 2: already the plain ratio sum (the picture's divided left side)
    if node.is_Add:
        pair = _ratio_sum(node)
        if pair is not None and _is_unit_pair(*pair[:2]):
            sin_coeff, cos_coeff, angle = pair
            return (sin(angle + _phi(sin_coeff, cos_coeff)),
                    "Write as sin(theta + phi)")

    return None, None


def _is_unit_pair(sin_coeff: Expr, cos_coeff: Expr) -> bool:
    """True when c^2 + s^2 = 1 -- the ratios must not change the value."""
    from sympy import simplify
    try:
        return simplify(sin_coeff ** 2 + cos_coeff ** 2 - 1) == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Equation steps (used by the solver)
# ---------------------------------------------------------------------------
def divide_sides(left: Expr, right: Expr
                 ) -> "Tuple[Expr, Expr, str] | str | None":
    """Divide both sides by R = sqrt(a^2+b^2) (the picture's step 2).

    Returns (new_left, new_right, label) on success, the string
    "no-solution" when |c| > R (the picture's condition), or None when
    this equation is not of the a*sin + b*cos form.
    """
    if not left.is_Add and _ratio_sum(left) is None:
        return None
    pair = _ratio_sum(left)
    if pair is None or not right.is_number:
        return None

    # The notes' equation has ONLY the two trig terms on the left; anything
    # else (+ 5, another variable, ...) means "not this method".
    if left.is_Add and any(_unwrap(term)[0] is None for term in left.args):
        return None

    sin_coeff, cos_coeff, angle = pair
    radius = sqrt(sin_coeff ** 2 + cos_coeff ** 2)
    if not radius.is_number or radius == 1:
        return None
    if abs(right) > radius:
        # The picture: "equation (i) has the solution only if |c| <= R"
        return "no-solution"

    new_left = Add(
        Mul(sin_coeff / radius, sin(angle), evaluate=False),
        Mul(cos_coeff / radius, cos(angle), evaluate=False),
        evaluate=False,
    )
    new_right = right / radius
    return new_left, new_right, "Divide both sides by sqrt(a^2 + b^2)"


def left_as_sin(expr: Expr) -> "Optional[Tuple[Expr, str]]":
    """Turn the divided left side into sin(t + phi) (the picture's step 3)."""
    pair = _ratio_sum(expr) if expr.is_Add else None
    if pair is None or not _is_unit_pair(*pair[:2]):
        return None
    sin_coeff, cos_coeff, angle = pair
    return (sin(angle + _phi(sin_coeff, cos_coeff)),
            "Write the left side as sin(theta + phi)")


# The registry the method runner dispatches  {"compute": ...}  steps through.
COMPUTES: dict = {
    "auxiliary_divide": auxiliary_divide,
    "auxiliary_as_sin": auxiliary_as_sin,
}


def run_compute(name: str, expr: Expr) -> "Optional[Tuple[Expr, str]]":
    """Run one named compute-step of a method."""
    if name not in COMPUTES:
        raise ValueError(
            f'Unknown compute step "{name}". Known: {", ".join(sorted(COMPUTES))}'
        )
    return COMPUTES[name](expr)
