"""
reading.py -- reads the question text and says what kind it is.

Purely deterministic rules, in priority order:

    1. "prove that ... = ..." / "show ..." / "verify ..."   -> prove
    2. "if <condition>, prove ..."                          -> conditional
       (recognised now, handled when the conditions backlog lands)
    3. "maximum / minimum / greatest / least ..."           -> optimize
       (recognised now, handled when the constraints backlog lands)
    4. anything with an equals sign                         -> solve
    5. everything else                                      -> simplify

Every reading carries an "explanation" in plain English -- the UI shows it
so the direction is never a mystery.
"""

import re

# Command words that turn a statement into a proof request.
_PROVE = re.compile(
    r"^\s*(?:prove|show|verify|establish|demonstrate)\b[\s:,]*"
    r"(?:that\b[\s:]*)?(?P<rest>.+)$",
    re.IGNORECASE,
)

# "If A + B + C = 180, prove that ..." -- recognised, not yet solved.
_CONDITIONAL = re.compile(
    r"^\s*if\b(?P<given>[^,]+),?\s*(?:prove|show|verify)\b[\s:,]*"
    r"(?:that\b[\s:]*)?(?P<rest>.+)$",
    re.IGNORECASE,
)

# Maximum / minimum questions (backlog: needs the constraints work).
_OPTIMIZE = re.compile(
    r"\b(?:maximum|minimum|greatest|least|greatest value|least value)\b",
    re.IGNORECASE,
)


class NotSupported(Exception):
    """The direction was understood, but its handler is not built yet."""


def read_question(text: str) -> dict:
    """Turn the typed question into {intent, payload, explanation}.

    Raises ValueError when the text cannot be read at all.
    """
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Please type something first.")

    # 1) conditional proof: "if ..., prove ..." (recognised, not solved yet)
    match = _CONDITIONAL.match(stripped)
    if match:
        return {
            "intent": "conditional",
            "given": match.group("given").strip(),
            "expression": match.group("rest").strip(),
            "explanation": "it starts with 'if ...' followed by 'prove'",
        }

    # 2) plain proof: "prove that <lhs> = <rhs>"
    match = _PROVE.match(stripped)
    if match:
        rest = match.group("rest").strip()
        if "=" not in rest:
            raise ValueError(
                'A proof needs an equals sign, e.g. "prove that sin(x)^2 + cos(x)^2 = 1".'
            )
        lhs_text, rhs_text = rest.split("=", 1)
        if not lhs_text.strip() or not rhs_text.strip():
            raise ValueError("Both sides of the '=' are needed to prove something.")
        return {
            "intent": "prove",
            "lhs": lhs_text.strip(),
            "rhs": rhs_text.strip(),
            "explanation": "it starts with 'prove/show/verify' and contains an equals sign",
        }

    # 3) optimisation questions (backlog handler)
    if _OPTIMIZE.search(stripped):
        return {
            "intent": "optimize",
            "expression": stripped,
            "explanation": "it asks about a maximum or minimum",
        }

    # 4) a plain equation: solve it
    if "=" in stripped:
        if stripped.count("=") != 1:
            raise ValueError("Only one '=' is supported per question.")
        lhs_text, rhs_text = stripped.split("=", 1)
        if not lhs_text.strip() or not rhs_text.strip():
            raise ValueError("Both sides of the '=' are needed.")
        return {
            "intent": "solve",
            "lhs": lhs_text.strip(),
            "rhs": rhs_text.strip(),
            "explanation": "it contains an equals sign, so it is an equation",
        }

    # 5) anything else is an expression to simplify
    return {
        "intent": "simplify",
        "expression": stripped,
        "explanation": "it is a plain expression, so it is simplified",
    }
