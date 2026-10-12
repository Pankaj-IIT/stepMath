"""
parser.py -- turns the text you type into something SymPy understands.

Two important jobs happen here:

1. SAFETY.
   SymPy's text parser is powerful enough to run Python code, so we first
   check the input against a whitelist of allowed characters. Anything else
   (like @ or ;) is refused before SymPy ever sees it.

2. KEEPING YOUR TEXT EXACTLY AS WRITTEN.
   We parse with evaluate=False. That stops SymPy from quietly doing the
   maths for us at parse time:

       "2*x + 3*x"   normally becomes   "5*x"   immediately (no step to show!)

   With evaluate=False the expression stays as written, so rules.py can
   show the user each change as its own step.
"""

import re

from sympy import Eq, Expr, Mul
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# What we allow users to type: letters, digits, maths symbols, and "="
# (the "=" is needed for equations; parse_expression rejects it separately).
# Everything else is refused before SymPy sees it.
_ALLOWED_CHARACTERS = re.compile(r"^[a-zA-Z0-9_+\-*/^().=\s]+$")

# How SymPy should read the text (shared with knowledge/patterns.py, so
# formulas and typed maths are read by exactly the same rules):
#   standard_transformations     - the usual reading rules
#   convert_xor                  - make "^" mean "to the power of" (not XOR)
#   implicit_multiplication_application - allow "2x" as well as "2*x"
TRANSFORMATIONS = (
    tuple(standard_transformations)
    + (convert_xor, implicit_multiplication_application)
)


def _check_input(text: str) -> None:
    """Refuse anything that is not plain maths text."""
    if not _ALLOWED_CHARACTERS.match(text):
        raise ValueError(
            "Only letters, numbers and maths symbols (+ - * / ^ ( )) are allowed."
        )


# ---------------------------------------------------------------------------
# People write functions WITHOUT brackets:  sinx, cos2x, sin x ...
# Without help, "sinx" reads as TWO variables (sin times x), so no formula
# in the database would ever match it. This step gives those calls brackets.
# ---------------------------------------------------------------------------

# Longest names first, so "cosec" is found before "cos".
_BARE_FUNCTIONS = (
    "arcsin", "arccos", "arctan", "cosec", "sinh", "cosh", "tanh",
    "csc", "sec", "tan", "sin", "cos", "cot", "exp", "log", "ln", "sqrt",
)

# Spellings people use that SymPy names differently.
_ALIASES = {"cosec": "csc"}


def _function_at(text: str, index: int) -> "str | None":
    """Is a known function name starting exactly at this position?"""
    for name in _BARE_FUNCTIONS:
        if text.startswith(name, index):
            return name
    return None


def _may_start_here(text: str, index: int) -> bool:
    """May a function name begin at this position?

    Yes at the start, after an operator or bracket -- and after a digit
    (so 2sinx is 2*sin(x)). NOT in the middle of another word, which keeps
    a variable called "asinx" untouched.
    """
    if index == 0:
        return True

    previous = text[index - 1]
    if previous.isdigit():
        return True
    if previous.isalnum() or previous == "_":
        return False
    return True


def expand_bare_functions(text: str) -> str:
    """Give bracket-less function calls their brackets.

        -2sinx^2   ->  -2sin(x)^2        (your reported case)
        cos2x      ->  cos(2x)
        sin x + 1  ->  sin(x) + 1
        sin(x)^2   ->  unchanged (already has brackets)

    One caution: a variable whose name merely STARTS with a function name
    ("sine") is read as sin(e) -- write sin(e) if that is what you mean.
    """
    out: list[str] = []
    index = 0

    while index < len(text):
        function = None
        # A function may also start right after a call we just rewrote:
        # 2sinxcosx  ->  the "cos" follows the ")" of "sin(x)".
        just_closed_call = bool(out) and out[-1].endswith(")")
        if _may_start_here(text, index) or just_closed_call:
            function = _function_at(text, index)

        if function is None:
            out.append(text[index])
            index += 1
            continue

        after_name = index + len(function)

        # Already a proper call: sin(...) -- leave it exactly as written.
        if after_name < len(text) and text[after_name] == "(":
            out.append(_ALIASES.get(function, function))
            index = after_name
            continue

        # Find the argument: first the identifier run after the name,
        # skipping at most the spaces people put there (sin x).
        argument_start = after_name
        while argument_start < len(text) and text[argument_start] == " ":
            argument_start += 1

        argument_end = argument_start
        while argument_end < len(text) and (
            text[argument_end].isalnum() or text[argument_end] == "_"
        ):
            argument_end += 1

        # Never swallow a second function: in 2sinxcosx the argument of
        # sin is just "x", and "cosx" is handled next.
        argument_end = _next_function(text, argument_start, argument_end)

        if argument_end == argument_start:
            # No argument follows ("sin + 1"): keep the bare name.
            out.append(_ALIASES.get(function, function))
            index = after_name
            continue

        out.append(f"{_ALIASES.get(function, function)}"
                   f"({text[argument_start:argument_end]})")
        index = argument_end

    return "".join(out)


def _next_function(text: str, start: int, end: int) -> int:
    """Position of a second function name inside [start, end), else `end`."""
    position = start + 1
    while position < end:
        if _function_at(text, position) is not None:
            return position
        position += 1
    return end


def normalize(expr: Expr) -> Expr:
    """Tidy up the noise that unevaluated parsing leaves behind.

    With evaluate=False, "1 / x" is kept as  "1 * (1/x)"  -- maths-wise it
    is fine, but on screen it would read  "1 1/x".  We drop stray
    multiplies-by-one everywhere, keeping everything else exactly as
    written (so the steps can still be shown one at a time).
    """
    if not expr.args:
        return expr

    parts = [normalize(part) for part in expr.args]

    if expr.is_Mul:
        kept = [part for part in parts if part != 1]
        if kept and len(kept) != len(parts):
            if len(kept) == 1:
                return kept[0]
            return Mul(*kept, evaluate=False)

    if parts == list(expr.args):
        return expr
    return expr.func(*parts, evaluate=False)


def parse_expression(text: str) -> Expr:
    """Turn "2*x + 3*x + 5 - 1" into a SymPy expression, kept exactly as typed."""
    _check_input(text)

    if "=" in text:
        raise ValueError(
            'This box is for expressions. For equations like "2*x + 6 = 14", '
            "switch the mode to Solve."
        )

    if not text.strip():
        raise ValueError("Please type something first.")

    # Give bracket-less calls their brackets:  sinx  ->  sin(x)
    text = expand_bare_functions(text)

    parsed = parse_expr(
        text,
        transformations=TRANSFORMATIONS,
        evaluate=False,  # do NOT solve it yet -- we want to show the steps
    )
    return normalize(parsed)


def parse_equation(text: str) -> tuple[Expr, Expr]:
    """Turn "2*x + 6 = 14" into its two sides: (2*x + 6, 14)."""
    _check_input(text)

    if "=" not in text:
        raise ValueError('An equation needs "=", for example "2*x + 6 = 14".')

    left_text, right_text = text.split("=", 1)

    if not left_text.strip() or not right_text.strip():
        raise ValueError('Both sides of "=" need maths on them.')

    return parse_expression(left_text), parse_expression(right_text)


def equation(left: Expr, right: Expr) -> Eq:
    """Build an equation object for display: keeps the "=" between the sides."""
    return Eq(left, right, evaluate=False)
