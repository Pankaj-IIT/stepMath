"""
solver.py -- step-by-step solving of equations.

Two kinds of equations are handled, in this order:

1. TRIGONOMETRIC equations with a stored general solution:
       sin(x) = 0    ->   x = n*pi        (from the formula database)
   These are recognised first, because you cannot move numbers across a
   sine the way you do with ordinary algebra.

2. LINEAR equations, done the way a person would on paper:
       2*x + 6 = 14  ->  2*x = 8  ->  x = 4
   SymPy builds the expressions so we never do arithmetic wrong ourselves.
"""

from sympy import Poly, Symbol, Wild
from sympy.core.expr import Expr

from ..knowledge import auxiliary
from ..knowledge import patterns as knowledge_patterns
from ..knowledge import store as knowledge_store
from .latex import to_latex, to_latex_as_written
from .parser import equation
from .steps import Step


def solve_equation(left: Expr, right: Expr) -> list[Step]:
    """Solve one equation and return the steps a student would write down."""
    steps: list[Step] = [
        # First line: exactly as the user typed it.
        {"name": "Initial equation", "latex": to_latex_as_written(equation(left, right))}
    ]

    # 1) A stored general solution wins, if one fits this equation.
    branches, formulas, pairs = _stored_general_solutions(left, right)
    if branches:
        steps.append(_general_step(branches, formulas))

        if pairs:
            # e.g.  x + atan(3/4) = (4n+1)*pi/2  -- ordinary algebra can
            # still isolate x from here.
            left, right = pairs[0]
            try:
                return _finish_linear(steps, left, right)
            except ValueError:
                pass  # not isolatable: the general solution already answers
        return steps

    # 2) The auxiliary-angle method from the notes:  a*sin + b*cos = c.
    divided = auxiliary.divide_sides(left, right)
    if divided == "no-solution":
        steps.append(_no_solution_step())
        return steps

    if isinstance(divided, tuple):
        left, right, label = divided
        steps.append({"name": label, "latex": to_latex(equation(left, right))})

    as_sin = auxiliary.left_as_sin(left)
    if as_sin is not None:
        left, label = as_sin
        # |c| <= 1 is still required when the divide step was not needed
        # (the coefficients already had R = 1).
        if right.is_number and abs(right) > 1:
            steps.append(_no_solution_step())
            return steps
        steps.append({"name": label, "latex": to_latex(equation(left, right))})

        # The left side is now sin(t + phi): stored solutions may finish it.
        branches, formulas, pairs = _stored_general_solutions(left, right)
        if branches:
            steps.append(_general_step(branches, formulas))
            if pairs:
                left, right = pairs[0]
                try:
                    return _finish_linear(steps, left, right)
                except ValueError:
                    pass
        # Otherwise stop exactly where the picture stops:
        #   "sin(t + phi) = c/R ... Now this equation can be solved easily."
        return steps

    if isinstance(divided, tuple):
        return steps  # divided, but not recognised: show the reduced form

    # 3) Otherwise treat it as an ordinary linear equation.
    return _finish_linear(steps, left, right)


def _general_step(branches: list, formulas: list) -> Step:
    step: Step = {"name": "General solution",
                  "latex": " \\text{ or } ".join(branches)}
    if formulas:
        # The formula(s) the answer came from, e.g.  sin(t) = 0
        step["formula"] = " \\text{ or } ".join(formulas)
    return step


def _no_solution_step() -> Step:
    """The notes' condition: a*sin + b*cos = c needs |c| <= sqrt(a^2+b^2)."""
    return {
        "name": "No solution",
        "latex": "\\text{No solution: } \\left|c\\right| "
                 "\\le \\sqrt{a^2 + b^2} \\text{ is false}",
    }


def _finish_linear(steps: list, left: Expr, right: Expr) -> list:
    """The ordinary 'move numbers across' ending of a solve."""
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
    # A "General solution" step may list SEVERAL branches (the +/- forms) --
    # it is already the answer; appending one single branch would hide the
    # other one.
    answer = equation(variable, right)
    if steps[-1]["name"] != "General solution" \
            and steps[-1]["latex"] != to_latex(answer):
        steps.append({"name": "Solution", "latex": to_latex(answer)})

    return steps


def _the_variable(left: Expr, right: Expr) -> Symbol:
    """Find the one variable we are solving for, or explain why we cannot."""
    # Integer-assumed symbols (the n of a general solution, e.g. (4n+1)*pi/2)
    # are answer bookkeeping, not unknowns -- the real unknown is the only
    # free symbol left after filtering them out.
    free = {symbol for symbol in (left.free_symbols | right.free_symbols)
            if not symbol.is_integer}

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


def _stored_general_solutions(left: Expr, right: Expr
                              ) -> "tuple[list, list, list]":
    """(answer branches, formulas used, raw equations) for this equation.

    The raw equations are the solved forms themselves, e.g.
    (x + atan(3/4), (4n+1)*pi/2) -- ordinary algebra can isolate x from
    them afterwards. Empty lists when no stored solution fits.
    The equation is tried as written and with the sides swapped, so both
    "sin(x) = 0" and "0 = sin(x)" find their answer.
    """
    try:
        connection = knowledge_store.get_connection()
    except Exception:
        return [], [], []  # no database: fall back to the linear path

    branches: list = []
    formulas: list = []
    pairs: list = []
    found_any = False

    try:
        for side_a, side_b in ((left, right), (right, left)):
            for row in knowledge_store.find_solutions(connection, side_a):
                fitted = _fit_solution(side_a, side_b, row)
                if not fitted:
                    continue
                branch_latex, formula_latex, solved_left, solved_right = fitted

                if branch_latex not in branches:
                    branches.append(branch_latex)
                    pairs.append((solved_left, solved_right))
                    found_any = True
                if formula_latex and formula_latex not in formulas:
                    formulas.append(formula_latex)
                knowledge_store.bump_usage(connection, row["id"])
            if branches:
                break  # matched with this side order; do not swap too
    except Exception:
        found_any = bool(branches)

    if found_any:
        connection.commit()
    return branches, formulas, pairs


def _fit_solution(side_a: Expr, side_b: Expr, row):
    """Does this stored solution fit?

    Returns (answer latex, formula latex, solved left, solved right),
    or None.  The last two are the solution as an equation, e.g.
    (x + atan(3/4), (4n+1)*pi/2).
    """
    try:
        row_left, row_right, unknown, value = knowledge_patterns.parse_solution(
            row["left"], row["right"], row["solution"]
        )
    except Exception:
        return None  # a damaged row is simply skipped

    match_a = knowledge_patterns.match_or_none(side_a, row_left)
    match_b = knowledge_patterns.match_or_none(side_b, row_right)
    if not match_a or not match_b:
        return None
    if match_a[1] or match_b[1]:
        return None  # a side must match completely, not just in part

    bindings = knowledge_patterns.merge_bindings(match_a[0], match_b[0])
    if bindings is None:
        return None  # a hole would need two values at once

    # Holes that appear only in the answer (?n) keep their integer meaning;
    # every other hole is filled from the match (?a -> pi/3).
    integer_holes = {
        hole: Symbol(hole.name, integer=True)
        for hole in value.free_symbols
        if isinstance(hole, Wild) and hole not in bindings
    }
    solved_left = unknown.subs(bindings)
    solved_right = value.subs({**integer_holes, **bindings})

    formula = f"{to_latex(row_left)} = {to_latex(row_right)}"
    answer = f"{to_latex(solved_left)} = {to_latex(solved_right)}"
    return answer, formula, solved_left, solved_right
