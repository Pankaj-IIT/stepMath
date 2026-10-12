"""
rules.py -- the list of algebra tricks we know.

Each rule is one small function with the same shape:

    def my_rule(expr):
        ...
        return new_expression, "Step name"   # when it improved the maths
        return None                           # when it has nothing to do

A rule never decides WHEN it runs -- steps.py does that. A rule only answers:
"can I improve this piece of maths, and what should we call that step?"

The rules are listed at the bottom of this file, in the order they should be
tried. Order matters: do the easy clean-ups first (numbers, zeros), and only
then the bigger moves (expanding brackets).
"""

from typing import Callable, Optional, Tuple

from sympy import (Add, Mul, Pow, S, collect, csc, cos, cot, expand,
                   expand_trig, factor, pi, sec, simplify, srepr, sin, tan,
                   together)
from sympy.core.expr import Expr

# A rule takes an expression and either returns (new_expression, step_name)
# or None when it cannot help.
Rule = Callable[[Expr], Optional[Tuple[Expr, str]]]


def changed(before: Expr, after: Expr) -> bool:
    """Did the maths really change? Compares the full structure, not just looks."""
    return srepr(before) != srepr(after)


def _calculate(expr: Expr) -> Expr:
    """Rebuild an expression from the bottom up, so SymPy works out the numbers.

    Example: the unevaluated piece  5 - 1*1   becomes   4.
    """
    if not expr.args:
        return expr
    return expr.func(*[_calculate(part) for part in expr.args])


# ---------------------------------------------------------------------------
# Rule 1 -- zeros
# ---------------------------------------------------------------------------
def cancel_zero(expr: Expr):
    """0 * anything is 0, and adding 0 changes nothing.

        0 * y          ->  0                     "Multiply by zero"
        0 * y + 4      ->  0 + 4                 "Multiply by zero"
        0 + 4          ->  4                     "Add with zero"
    """
    # A multiplication containing a plain zero collapses to zero.
    if expr.is_Mul and any(factor == S.Zero for factor in expr.args):
        return S.Zero, "Multiply by zero"

    if expr.is_Add:
        # Inside a sum, first collapse any term like 0*y ...
        for index, term in enumerate(expr.args):
            if term.is_Mul and any(factor == S.Zero for factor in term.args):
                terms = list(expr.args)
                terms[index] = S.Zero
                return Add(*terms, evaluate=False), "Multiply by zero"

        # ... then a plain + 0 can simply be dropped: x + 0 = x
        if any(term == S.Zero for term in expr.args):
            kept = [term for term in expr.args if term != S.Zero]
            return Add(*kept, evaluate=False), "Add with zero"

    return None


# ---------------------------------------------------------------------------
# Rule 2 -- plain number arithmetic
# ---------------------------------------------------------------------------
def eval_numbers(expr: Expr):
    """Do the arithmetic on plain numbers.

        5 - 1          ->  4                     "Evaluate numbers"
        2*3*x          ->  6*x                   "Evaluate numbers"
        (5 - 1) + x    ->  4 + x                 "Evaluate numbers"
    """
    if not (expr.is_Add or expr.is_Mul):
        return None

    numbers = [term for term in expr.args if term.is_number]
    others = [term for term in expr.args if not term.is_number]

    if not numbers:
        return None

    # Work out the number part properly (rules above may have left it rough).
    done = [_calculate(term) for term in numbers]
    total = Add(*done) if expr.is_Add else Mul(*done)

    # A "number" that still has parts -- cos(1/2), sqrt(2) -- cannot be
    # worked out any further, so combining it would only shuffle the
    # display around. Leave it alone.
    if not total.is_Number:
        return None

    # Nothing useful to do: adding 0, or multiplying by 1, changes nothing
    # (and rebuilding would only shuffle the display around).
    if (expr.is_Add and total == 0) or (expr.is_Mul and total == 1):
        return None

    # Build the result in reading order:
    #   sums   -> numbers last    (2*x + 3*x + 4)
    #   products -> number first  (6*x, not x 6)
    if expr.is_Add:
        new = Add(*others, total, evaluate=False)
    else:
        new = Mul(total, *others, evaluate=False)

    if not changed(expr, new):
        return None
    return new, "Evaluate numbers"


# ---------------------------------------------------------------------------
# Rule 3 -- like terms
# ---------------------------------------------------------------------------
def combine_like_terms(expr: Expr):
    """Terms with the same variable can be merged.

        2*x + 3*x      ->  5*x                   "Combine like terms"
        4*x + 2*x - 1  ->  6*x - 1               "Combine like terms"
    """
    if not expr.is_Add:
        return None

    # collect() needs to know which variables to gather: "x + x" -> "2*x"
    variables = sorted(expr.free_symbols, key=str)
    if not variables:
        return None  # only numbers here -- rule 2 deals with those

    new = collect(expr, variables)
    if not changed(expr, new):
        return None
    return new, "Combine like terms"


# ---------------------------------------------------------------------------
# Rule 4 -- repeated factors
# ---------------------------------------------------------------------------
def tidy_powers(expr: Expr):
    """A factor written twice can be written as a power.

        x * x          ->  x^2                   "Tidy powers"
        2*x*x          ->  2*x^2                 "Tidy powers"
    """
    if not expr.is_Mul:
        return None

    # Nothing repeats, e.g. 2*x has two different factors.
    if len(set(expr.args)) == len(expr.args):
        return None

    # Rebuilding WITH evaluation lets SymPy collect the repeats.
    new = Mul(*expr.args)
    if not changed(expr, new):
        return None
    return new, "Tidy powers"


# ---------------------------------------------------------------------------
# Rule 5 -- brackets
# ---------------------------------------------------------------------------
def _has_denominator(expr: Expr) -> bool:
    """Is there a division anywhere in here (a power with a minus sign)?"""
    if isinstance(expr, Pow) and expr.exp.is_Number and expr.exp.is_negative:
        return True
    return any(_has_denominator(part) for part in expr.args)


def expand_brackets(expr: Expr):
    """Multiply a bracket out.

        2 * (x + 3)    ->  2*x + 6               "Expand brackets"
        (x + 1)^2      ->  x^2 + 2*x + 1         "Expand brackets"
    """
    # Never expand across a division: splitting  (x+y)/(x*y)  into
    # x/(x*y) + y/(x*y) is a step backwards.
    if _has_denominator(expr):
        return None

    new = expand(expr)
    if not changed(expr, new):
        return None
    return new, "Expand brackets"


def take_lcm(expr: Expr):
    """Put fractions over one common denominator -- SymPy's together().

        sin^2/cos^2 + 1  ->  (sin^2 + cos^2)/cos^2   "Take the LCM"
        1/y + x/y        ->  (x + 1)/y

    That common denominator is what lets the next formula fire, e.g. the
    numerator above becomes  sin^2 + cos^2  =  1.

    Like expanding and factoring this reshapes the maths, so steps.py only
    allows it on parts that still contain what you typed.
    """
    # Only when there IS a division: together() also factors plain sums
    # (it turns 2*x + 4 into 2*(x + 2)), which would steal the step name
    # from take_common below.
    if not _has_denominator(expr):
        return None

    new = together(expr)
    if not changed(expr, new):
        return None
    return new, "Take the LCM"


def take_common(expr: Expr):
    """Take out a common factor -- SymPy's factor() does the work.

        2*x + 4        ->  2*(x + 2)             "Take out the common factor"
        -sin^2 - cos^2 ->  -(sin^2 + cos^2)      (which then lets the
                                                  Pythagorean identity
                                                  finish the job)

    Like expanding, this is a "big move": it re-shapes the maths, so
    steps.py only allows it on parts that still contain what you typed.
    """
    new = factor(expr)
    if not changed(expr, new):
        return None
    return new, "Take out the common factor"


# ---------------------------------------------------------------------------
# The order rules are tried in: cheap clean-ups first, big moves last.
#
# The two groups are kept apart on purpose:
#
#   TIDY_RULES  are safe anywhere -- they only ever clean up (numbers,
#               zeros, repeated factors, like terms).
#   SHAPE_RULES change the shape of the maths (expanding), so steps.py only
#               lets them work on the expression the user TYPED. Otherwise
#               an expansion could keep re-expanding parts that an identity
#               just created, and the steps would churn back and forth.
# ---------------------------------------------------------------------------
TIDY_RULES = [
    cancel_zero,
    eval_numbers,
    combine_like_terms,
    tidy_powers,
]

def reduce_angle(expr: Expr):
    """Move a constant (pi-based or negative) angle out: SymPy does it all.

        sin(x + pi/2)  ->  cos(x)        "Reduce the angle"
        sin(x + 2*pi)  ->  sin(x)
        cos(x + pi)    ->  -cos(x)
        sin(pi - x)    ->  sin(x)
        sin(-x)        ->  -sin(x)

    Replaces the whole allied-angle/period/negative-angle formula family:
    there is no point storing rows for what SymPy already computes exactly.
    Only constants IN THE ANGLE qualify, so two-angle sums (sin(x+y)) still
    go through their own group.
    """
    if expr.func not in (sin, cos, tan, cot, sec, csc) or len(expr.args) != 1:
        return None

    angle = expr.args[0]
    constant_shift = isinstance(angle, Add) and angle.has(pi)
    negative_angle = angle.is_Mul and angle.could_extract_minus_sign()
    if not (constant_shift or negative_angle):
        return None  # not a constant shift (sin(x+y) has its own group)

    new = simplify(expand_trig(expr))
    if not changed(expr, new):
        return None
    return new, "Reduce the angle"


SHAPE_RULES = [
    reduce_angle,   # first: its condition is narrow and specific
    expand_brackets,
    take_lcm,      # before factoring, so a fraction step gets the right name
    take_common,
]

SIMPLIFY_RULES = TIDY_RULES + SHAPE_RULES
