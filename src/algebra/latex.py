"""
latex.py -- turns maths into text that KaTeX can draw.

SymPy can print maths two useful ways:
    str(expr)    "2*x + 3*x + 4"      good for programmers
    latex(expr)  "2 x + 3 x + 4"      good for KaTeX on the web page

We always use the LaTeX version, because that is what the React app renders.
"""

from sympy import Expr, latex


def to_latex(expr: Expr) -> str:
    """Give me the LaTeX spelling of this expression.

    This is SymPy's own ordering, which reads well for maths it produced,
    e.g. "5 x + 4".
    """
    return latex(expr)


def to_latex_as_written(expr: Expr) -> str:
    """Same, but keeps the terms in the order the user typed them.

    Used for the very first line, so it matches the input box:
    "2 x + 3 x + 5 - 1" instead of SymPy's "... - 1 + 5".
    """
    return latex(expr, order="none")
