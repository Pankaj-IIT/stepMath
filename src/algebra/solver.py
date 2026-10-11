"""
solver.py -- step-by-step solving of simple equations (version 1: linear only).

Example:   2*x + 6 = 14

    1. "Initial equation"            2 x + 6 = 14
    2. "Subtract 6 from both sides"  2 x = 8
    3. "Divide both sides by 2"      x = 4

Everything here is done the way a person would do it on paper: move the
numbers across, then remove the multiplying. SymPy is used to build the
expressions so we never do arithmetic wrong ourselves.
"""

from sympy import Poly, Symbol
from sympy.core.expr import Expr

from .latex import to_latex, to_latex_as_written
from .parser import equation
from .steps import Step


def solve_equation(left: Expr, right: Expr) -> list[Step]:
    """Solve one equation and return the steps a student would write down."""
    steps: list[Step] = [
        # First line: exactly as the user typed it.
        {"name": "Initial equation", "latex": to_latex_as_written(equation(left, right))}
    ]

    variable = _the_variable(left, right)

    # Version 1 only knows how to move numbers across. So the variable has to
    # sit on one side -- e.g. "x + 3 = 2*x - 1" is the natural next feature.
    if variable in left.free_symbols and variable in right.free_symbols:
        raise ValueError(
            f"The variable {variable} appears on both sides, which is not "
            "supported yet. Try moving it to one side yourself first."
        )

    # Put the side with the variable on the left, so the steps read naturally.
    if variable in right.free_symbols:
        left, right = right, left

    coefficient, constant = _linear_parts(left, variable)  # left is a*x + b

    # Step: move the constant number to the other side.
    #   2*x + 6 = 14   ->   2*x = 8
    if constant != 0:
        right = right - constant
        left = coefficient * variable
        if constant > 0:
            name = f"Subtract {constant} from both sides"
        else:
            name = f"Add {-constant} to both sides"
        steps.append({"name": name, "latex": to_latex(equation(left, right))})

    # Step: remove the multiplying in front of the variable.
    #   2*x = 8   ->   x = 4
    if coefficient != 1:
        right = right / coefficient
        left = variable
        steps.append(
            {"name": f"Divide both sides by {coefficient}",
             "latex": to_latex(equation(left, right))}
        )

    # Finish with the answer, unless the last step already shows it.
    answer = equation(variable, right)
    if steps[-1]["latex"] != to_latex(answer):
        steps.append({"name": "Solution", "latex": to_latex(answer)})

    return steps


def _the_variable(left: Expr, right: Expr) -> Symbol:
    """Find the one variable we are solving for, or explain why we cannot."""
    free = left.free_symbols | right.free_symbols

    if not free:
        raise ValueError("There is no variable here to solve for.")
    if len(free) > 1:
        raise ValueError(
            "Only equations with one variable are supported so far "
            f"(I found: {', '.join(sorted(str(s) for s in free))})."
        )
    return free.pop()


def _linear_parts(expr: Expr, variable: Symbol) -> tuple[Expr, Expr]:
    """Read one side as (coefficient, constant) for the form  a*x + b.

    Raises a friendly error when the equation is not simple and linear.
    """
    try:
        polynomial = Poly(expr, variable)
    except Exception:
        raise ValueError(
            "I can only solve simple equations so far, like 2*x + 6 = 14."
        )

    if polynomial.degree() > 1:
        raise ValueError(
            "I can only solve equations where the variable is to the power of 1, "
            "like 2*x + 6 = 14."
        )

    coefficient = polynomial.coeff_monomial(variable)  # the "a" in a*x
    constant = polynomial.coeff_monomial(1)  # the "b" in a*x + b
    return coefficient, constant
