"""
runner.py -- runs a method's steps, one screen at a time.

A method is ONE database row whose "steps" column holds a small JSON list:

    [ {"op": "expand"},
      {"left": "?a*x^2 + ?b*x", "right": "...", "name": "Rewrite as brackets"},
      {"op": "factor"} ]

Every step is written out in full right here -- the runner never has to go
looking for anything in the database halfway through.

It runs the steps in order and returns one display line per step. If a
pattern step does not match (or SymPy throws), it stops immediately with
"Method stopped at step N: ..." rather than leaving a half-finished result.
"""

import json
from typing import List, Tuple

from sympy import Expr

from ..algebra.latex import to_latex, to_latex_as_written
from .ops import known_operation, run_operation
from .patterns import apply_identity, parse_pair

# Never show more than this many lines for one method.
MAX_STEPS = 30

Step = dict  # {"name": ..., "latex": ...}


class MethodError(Exception):
    """A method could not finish. Says which step went wrong and why."""

    def __init__(self, step_number: int, reason: str):
        super().__init__(f"Method stopped at step {step_number}: {reason}")
        self.step_number = step_number
        self.reason = reason


def load_steps(raw: str) -> list:
    """Turn the JSON stored in the database into a list of step objects."""
    try:
        steps = json.loads(raw or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"steps is not valid JSON: {error}")
    validate_steps(steps)
    return steps


def validate_steps(steps) -> None:
    """Check a method's steps are usable. Raises ValueError with a reason."""
    if not isinstance(steps, list) or not steps:
        raise ValueError("a method needs a non-empty list of steps")

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} must be an object")

        if "op" in step:
            if not known_operation(step["op"]):
                raise ValueError(f"step {index}: unknown operation {step['op']!r}")
        elif "left" in step and "right" in step:
            try:
                parse_pair(step["left"], step["right"])
            except Exception as error:
                raise ValueError(f"step {index} could not be read: {error}")
        else:
            raise ValueError(
                f'step {index} needs either "op" or "left" + "right"'
            )


def run_method(expr: Expr, raw_steps: str) -> List[Step]:
    """Run every step of a method and return the lines to display.

    Raises MethodError when a pattern step does not fit the expression.
    """
    steps = load_steps(raw_steps)

    output: List[Step] = [
        {"name": "Initial expression", "latex": to_latex_as_written(expr)}
    ]
    seen = {output[0]["latex"]}
    current = expr

    for number, step in enumerate(steps, start=1):
        if len(output) >= MAX_STEPS:
            break

        if "op" in step:
            try:
                result = run_operation(step["op"], current)
            except Exception as error:
                raise MethodError(number, str(error))
            if result is None:
                continue  # operation had nothing to do -- quietly move on
            new_expr, label = result
            name = step.get("name", label)

        else:
            left, right = parse_pair(step["left"], step["right"])
            new_expr = apply_identity(current, left, right)
            if new_expr is None:
                raise MethodError(
                    number,
                    step.get("name", f'pattern "{step["left"]}" did not match'),
                )
            name = step.get("name", "Rewrite")

        latex = to_latex(new_expr)
        if latex in seen:
            continue  # nothing visible changed -- do not show it twice

        seen.add(latex)
        output.append({"name": name, "latex": latex})
        current = new_expr

    return output


def first_step_shape(raw_steps: str) -> str:
    """How a method announces what it can start on.

    First step is a pattern -> the shape of that pattern (indexed).
    First step is a plain operation -> 'any' (always offered).
    """
    steps = load_steps(raw_steps)
    first = steps[0]

    if "op" in first:
        return "any"

    left, _ = parse_pair(first["left"], first["right"])
    from .signature import signature_of  # local import keeps the module flat
    return signature_of(left)
