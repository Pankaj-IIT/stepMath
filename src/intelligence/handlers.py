"""
handlers.py -- the intent registry: which handler runs for which question.

    read_question() says WHAT it is; this file says WHAT TO DO about it.
    Adding a direction later = adding one function here and one entry in
    HANDLERS. The two stubs (conditional, optimize) are already registered
    so those questions get an honest "not built yet" instead of nonsense.
"""

from . import prove
from .reading import NotSupported

Step = dict


def _simplify(reading: dict) -> dict:
    """Handler #1: tidy an expression (the pipeline we already had)."""
    from ..algebra.parser import parse_expression
    from ..algebra.steps import alternative_ways, simplify_steps

    expression = parse_expression(reading["expression"])
    steps = simplify_steps(expression)
    return {
        "steps": steps,
        "final": steps[-1]["latex"],
        "alternatives": alternative_ways(expression),
    }


def _prove(reading: dict) -> dict:
    """Handler #2: show how two sides are equal (or say we cannot)."""
    from ..algebra.parser import parse_expression

    lhs = parse_expression(reading["lhs"])
    rhs = parse_expression(reading["rhs"])
    steps, proven = prove.prove_steps(lhs, rhs)
    return {
        "steps": steps,
        "final": steps[-1]["latex"],
        "alternatives": [],
        "proven": proven,
    }


def _solve(reading: dict) -> dict:
    """Handler #3: solve an equation (same engine as the Solve toggle)."""
    from ..algebra.parser import parse_expression
    from ..algebra.solver import solve_equation

    left = parse_expression(reading["lhs"])
    right = parse_expression(reading["rhs"])
    steps = solve_equation(left, right)
    return {"steps": steps, "final": steps[-1]["latex"], "alternatives": []}


def _conditional(reading: dict) -> dict:
    raise NotSupported(
        "Reading a conditional proof (\"if ..., prove ...\") is not built "
        "yet -- it arrives with the global constraints work."
    )


def _optimize(reading: dict) -> dict:
    raise NotSupported(
        "Maximum/minimum questions are not built yet -- they arrive with "
        "the global constraints work."
    )


# intent -> handler. Unknown intents fall back to the safest one.
HANDLERS = {
    "simplify": _simplify,
    "prove": _prove,
    "solve": _solve,
    "conditional": _conditional,
    "optimize": _optimize,
}


def dispatch(reading: dict) -> dict:
    """Run the handler for this reading and tag it with the reading info."""
    handler = HANDLERS.get(reading["intent"], _simplify)
    result = handler(reading)
    result.setdefault("intent", reading["intent"])
    result.setdefault("explanation", reading.get("explanation", ""))
    return result
