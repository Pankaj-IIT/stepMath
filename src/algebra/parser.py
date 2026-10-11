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

from sympy import Eq, Expr
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

# How SymPy should read the text:
#   standard_transformations     - the usual reading rules
#   convert_xor                  - make "^" mean "to the power of" (not XOR)
#   implicit_multiplication_application - allow "2x" as well as "2*x"
_TRANSFORMATIONS = (
    tuple(standard_transformations)
    + (convert_xor, implicit_multiplication_application)
)


def _check_input(text: str) -> None:
    """Refuse anything that is not plain maths text."""
    if not _ALLOWED_CHARACTERS.match(text):
        raise ValueError(
            "Only letters, numbers and maths symbols (+ - * / ^ ( )) are allowed."
        )


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

    return parse_expr(
        text,
        transformations=_TRANSFORMATIONS,
        evaluate=False,  # do NOT solve it yet -- we want to show the steps
    )


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
