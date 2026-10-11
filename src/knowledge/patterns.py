"""
patterns.py -- reading and applying formula patterns.

A pattern is a formula written with holes:

    left :   sin(?a)^2 + cos(?a)^2
    right:   1

"?a" means "anything can sit here". SymPy has the same idea, called Wild,
so we translate our ?a into SymPy's Wild and let SymPy do the matching:

    expr.match(pattern)      ->  {a_: x}       (the hole filled with x)
    pattern.subs(that_dict)  ->  the original expr again

Two safety rules live here:

    1. We only accept a match when re-substituting gives back the original
       expression -- this stops SymPy's matcher from inventing nonsense
       bindings.
    2. We compare LaTeX before and after, so a "change" nobody can see is
       never reported as a step (that is checked in steps.py).
"""

import re
from typing import Optional, Tuple

from sympy import Add, Expr, Mul, Pow, Symbol, Wild
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..algebra.parser import normalize

_TRANSFORMATIONS = (
    tuple(standard_transformations)
    + (convert_xor, implicit_multiplication_application)
)

# Our hole syntax:  ?a  ?rest  ?x1
_HOLE = re.compile(r"\?([A-Za-z_]\w*)")

# Which characters a pattern may contain (a superset of normal maths,
# plus "?" for holes). Anything else is refused.
_PATTERN_CHARACTERS = re.compile(r"^[a-zA-Z0-9_+\-*/^().\s?]+$")


def _holes(*texts: str) -> dict:
    """Collect every ?name mentioned in these texts as a SymPy Wild.

    The key must be the CLEANED name (_a) because that is what appears in
    the text we are about to parse; the Wild itself keeps the original
    name (a), which is the key we use in the bindings dictionary.
    """
    names = set()
    for text in texts:
        names.update(_HOLE.findall(text))
    return {f"_{name}": Wild(name) for name in names}


def _clean(text: str) -> str:
    """Turn our holes into names SymPy understands:  ?a  ->  _a"""
    return _HOLE.sub(lambda m: "_" + m.group(1), text)


def parse_pattern(text: str) -> Expr:
    """Read one side of a formula, holes and all."""
    if not _PATTERN_CHARACTERS.match(text or ""):
        raise ValueError(f"Not a valid pattern: {text!r}")

    wilds = _holes(text)
    parsed = parse_expr(
        _clean(text),
        transformations=_TRANSFORMATIONS,
        evaluate=False,
        local_dict=wilds,
    )
    return normalize(parsed)


def parse_pair(left: str, right: str) -> Tuple[Expr, Expr]:
    """Read both sides of a formula so the holes are shared objects.

    Sharing matters: the ?a in the left side and the ?a in the right side
    must be the SAME hole, otherwise filling one does not fill the other.
    """
    wilds = _holes(left, right)
    parsed_left = parse_expr(_clean(left), transformations=_TRANSFORMATIONS,
                             evaluate=False, local_dict=wilds)
    parsed_right = parse_expr(_clean(right), transformations=_TRANSFORMATIONS,
                              evaluate=False, local_dict=wilds)
    return normalize(parsed_left), normalize(parsed_right)


def _evaluated(expr: Expr) -> Expr:
    """Rebuild an expression with SymPy's normal (eager) evaluation.

    Our engine keeps expressions exactly as written, but SymPy's matcher
    works better on the normal form, so we try both.
    """
    if not expr.args:
        return expr
    return expr.func(*[_evaluated(part) for part in expr.args])


def _exact_root(number: Expr, exponent: int) -> Optional[Expr]:
    """The exact whole-number root, when there is one: 4 -> 2, 1/4 -> 1/2.

    Returns None for anything that would leave a root behind (the square
    root of 2, say) -- we only bind a hole to an honest number.
    """
    from sympy import root as exact_root

    value = exact_root(number, exponent)
    if value.is_Rational and value.is_number:
        return value
    return None


def _same(a: Expr, b: Expr) -> bool:
    """Are these two the same maths? Compares the normal (evaluated) form,
    so the order things were written in does not matter."""
    return _evaluated(a) == _evaluated(b)


def _match(pattern: Expr, expr: Expr, bindings: dict) -> Optional[dict]:
    """Fill the pattern's holes so it rebuilds this expression.

    A tiny structural matcher. SymPy has one built in, but it gives up on
    cases we need (like  x^2 - y^2  against  ?a^2 - ?b^2), and this one is
    short enough to read in one sitting.

    Rules:
        ?hole            takes anything; the same hole used twice must take
                         the same thing
        numbers          must be equal (2 matches 2, not 3)
        plain names      must have the same name (x matches x, not y)
        Add / Mul        may be written in any order, so we try the
                         arrangements until one fits
        everything else  matches part by part, in order
    """
    # 1. A hole swallows anything in front of it.
    #    Keyed by the hole object itself, because that is what SymPy's
    #    substitution understands (a plain "a" symbol would not match it).
    if isinstance(pattern, Wild):
        if pattern in bindings:
            return bindings if _same(bindings[pattern], expr) else None
        bindings[pattern] = expr
        return bindings

    # 2. A pattern power against a plain number:  ?b^2  can stand for 4,
    #    with ?b = 2.  Without this,  x^2 - 4  would never match
    #    ?a^2 - ?b^2, because 4 is written as a number, not as 2^2.
    if (isinstance(pattern, Pow) and expr.is_Number
            and pattern.exp.is_Integer and int(pattern.exp) >= 2):
        root = _exact_root(expr, int(pattern.exp))
        if root is not None:
            return _match(pattern.base, root, bindings)
        return None

    # 2. Plain numbers and plain names must agree.
    if pattern.is_Number or expr.is_Number:
        if pattern.is_Number and expr.is_Number and pattern == expr:
            return bindings
        return None

    if isinstance(pattern, Symbol):
        if isinstance(expr, Symbol) and not isinstance(expr, Wild) \
                and pattern.name == expr.name:
            return bindings
        return None

    # 3. Different kind of node (sin vs cos, Add vs Mul, ...) -> no match.
    if pattern.func != expr.func:
        return None

    # 4. Addition and multiplication can be written in any order.
    if isinstance(pattern, (Add, Mul)):
        return _match_in_any_order(pattern, expr, bindings)

    # 5. Powers, functions, etc: same parts, same order.
    if len(pattern.args) != len(expr.args):
        return None

    for pattern_part, expr_part in zip(pattern.args, expr.args):
        bindings = _match(pattern_part, expr_part, bindings)
        if bindings is None:
            return None
    return bindings


def _match_in_any_order(pattern: Expr, expr: Expr, bindings: dict) -> Optional[dict]:
    """Match Add/Mul where the parts may be written in a different order."""
    from itertools import permutations

    pattern_parts = pattern.args
    expr_parts = expr.args

    if len(pattern_parts) != len(expr_parts):
        return None

    # Up to 4 parts we try every arrangement; above that, just two sensible
    # guesses (in order and reversed) so we never explode combinatorially.
    if len(pattern_parts) <= 4:
        orders = list(permutations(range(len(pattern_parts))))
    else:
        orders = [tuple(range(len(pattern_parts))),
                  tuple(reversed(range(len(pattern_parts))))]

    for order in orders:
        trial = dict(bindings)
        for position, pattern_index in enumerate(order):
            trial = _match(pattern_parts[pattern_index], expr_parts[position], trial)
            if trial is None:
                break
        else:
            return trial  # every part found a home

    return None


def match_or_none(expr: Expr, pattern: Expr) -> "Optional[Tuple[dict, list]]":
    """Try to match this expression against a pattern.

    Returns (hole_fillings, leftover_positions) or None.

    leftover_positions lists the parts of the expression the pattern did NOT
    use. Usually that is [] (the whole expression matched). It is not empty
    when a formula is hiding inside a bigger sum, e.g. the identity

        sin(?a)^2 + cos(?a)^2  =  1

    inside   sin(z)^2 + cos(z)^2 + 4.   There we match the first two parts
    and hand back [2] (the "+ 4"), so the caller can rebuild
    "1 + 4" instead of losing the 4.
    """
    for candidate_expr, candidate_pattern in (
        (expr, pattern),
        (_evaluated(expr), _evaluated(pattern)),
    ):
        bindings = _match(candidate_pattern, candidate_expr, {})

        # Final soundness check: does the pattern really rebuild it?
        if bindings is not None and _same(candidate_pattern.subs(bindings),
                                          candidate_expr):
            return bindings, []

    # A formula may also cover only PART of a sum or product.
    return _match_part_of(expr, pattern)


def _match_part_of(expr: Expr, pattern: Expr) -> "Optional[Tuple[dict, list]]":
    """Match a smaller pattern against part of a bigger Add or Mul."""
    from itertools import combinations

    if not (isinstance(expr, (Add, Mul)) and isinstance(pattern, (Add, Mul))):
        return None
    if expr.func != pattern.func:
        return None
    if len(pattern.args) >= len(expr.args):
        return None  # not a "part of" situation

    # Try every group of parts that is the same size as the pattern.
    for chosen in combinations(range(len(expr.args)), len(pattern.args)):
        piece = expr.func(*[expr.args[i] for i in chosen], evaluate=False)

        bindings = _match(pattern, piece, {})
        if bindings is None:
            continue
        if not _same(pattern.subs(bindings), piece):
            continue

        leftover = [i for i in range(len(expr.args)) if i not in chosen]
        return bindings, leftover

    return None


def apply_identity(expr: Expr, left: Expr, right: Expr) -> Optional[Expr]:
    """Try to rewrite this expression using  left = right.

    Returns the new expression, or None when the formula does not apply.
    """
    matched = match_or_none(expr, left)
    if matched is None:
        return None

    bindings, leftover = matched
    replacement = right.subs(bindings)

    if not leftover:
        return replacement

    # The formula only covered part of a sum/product: put the result where
    # the first used part was, and keep every untouched part.
    used_first = min(i for i in range(len(expr.args)) if i not in leftover)
    rebuilt = []
    for position, part in enumerate(expr.args):
        if position == used_first:
            rebuilt.append(replacement)
        elif position in leftover:
            rebuilt.append(part)

    return expr.func(*rebuilt, evaluate=False)
