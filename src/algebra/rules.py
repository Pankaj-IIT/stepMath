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

from sympy import Add, Mul, S, collect, expand, srepr
from sympy.core.expr import Expr

# A rule takes an expression and either returns (new_expression, step_name)
# or None when it cannot help.
Rule = Callable[[Expr], Optional[Tuple[Expr, str]]]


def _changed(before: Expr, after: Expr) -> bool:
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

    # Build the result in reading order:
    #   sums   -> numbers last    (2*x + 3*x + 4)
    #   products -> number first  (6*x, not x 6)
    if expr.is_Add:
        new = Add(*others, total, evaluate=False)
    else:
        new = Mul(total, *others, evaluate=False)

    if not _changed(expr, new):
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
    if not _changed(expr, new):
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
    if not _changed(expr, new):
        return None
    return new, "Tidy powers"


# ---------------------------------------------------------------------------
# Rule 5 -- brackets
# ---------------------------------------------------------------------------
def expand_brackets(expr: Expr):
    """Multiply a bracket out.

        2 * (x + 3)    ->  2*x + 6               "Expand brackets"
        (x + 1)^2      ->  x^2 + 2*x + 1         "Expand brackets"
    """
    new = expand(expr)
    if not _changed(expr, new):
        return None
    return new, "Expand brackets"


# ---------------------------------------------------------------------------
# The order rules are tried in: cheap clean-ups first, big moves last.
# steps.py walks this list for every piece of the expression.
# ---------------------------------------------------------------------------
SIMPLIFY_RULES = [
    cancel_zero,
    eval_numbers,
    combine_like_terms,
    tidy_powers,
    expand_brackets,
]
