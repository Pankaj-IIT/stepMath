"""
One test per algebra rule, plus a few full "show me the steps" tests.

Run them with:      pytest

Each test follows the same pattern: type an expression in, check the steps
and the final answer that come out.
"""

import pytest

from conftest import steps_for
from src.algebra.parser import parse_expression


def names_of(text: str) -> list[str]:
    """Helper: just the step names, so tests read like English."""
    return [step["name"] for step in steps_for(text)]


def final_of(text: str) -> str:
    """Helper: the last (fully simplified) line."""
    return steps_for(text)[-1]["latex"]


# --- Rule 1: zeros ---------------------------------------------------------
def test_multiply_by_zero():
    assert final_of("0 * y") == "0"


def test_zero_term_inside_a_sum():
    assert "Multiply by zero" in names_of("0 * y + 4")
    assert final_of("0 * y + 4") == "4"


def test_add_with_zero():
    assert final_of("x + 0") == "x"


# --- Rule 2: numbers -------------------------------------------------------
def test_evaluate_numbers():
    assert final_of("5 - 1") == "4"


def test_evaluate_numbers_keeps_the_rest():
    assert final_of("2 * 3 * x") == "6 x"


def test_numbers_next_to_variables():
    assert final_of("2 * x + 3 * x + 5 - 1") == "5 x + 4"


# --- Rule 3: like terms ----------------------------------------------------
def test_combine_like_terms():
    assert final_of("2*x + 3*x") == "5 x"


def test_combine_like_terms_with_numbers():
    assert final_of("4*x + 2*x - 1") == "6 x - 1"


# --- Rule 4: powers --------------------------------------------------------
def test_tidy_powers():
    assert final_of("x * x") == "x^{2}"


# --- Rule 5: brackets ------------------------------------------------------
def test_expand_brackets():
    assert final_of("2 * (x + 3)") == "2 x + 6"


def test_expand_square_of_sum():
    assert final_of("(x + 1)^2") == "x^{2} + 2 x + 1"


# --- Parsing ---------------------------------------------------------------
def test_shorthand_multiplication():
    assert final_of("2x + 3x") == "5 x"


def test_bare_function_names_get_their_brackets():
    """sinx must read as sin(x), not as two variables called sin and x."""
    from sympy import srepr

    from src.algebra.parser import parse_expression

    pairs = [
        ("-2sinx^2 - cos2x + 5", "-2*sin(x)^2 - cos(2*x) + 5"),
        ("sin x + 1", "sin(x) + 1"),
        ("2sinxcosx", "2*sin(x)*cos(x)"),
        ("cosecx", "csc(x)"),          # our spelling, SymPy's name
        ("asinx", "asinx"),            # a variable, not a function call
        ("sin(x)^2", "sin(x)^2"),      # already correct: untouched
    ]
    for typed, expected in pairs:
        assert srepr(parse_expression(typed)) == srepr(parse_expression(expected)), typed


def test_bare_function_input_solves_the_same_as_bracketed(sample_formulas):
    """The reported case: both spellings must reach the same answer."""
    bare = steps_for("-2sinx^2 - cos2x + 5")
    bracketed = steps_for("-2*sin(x)^2 - cos(2*x) + 5")

    assert bare[-1]["latex"] == bracketed[-1]["latex"] == "4"


def test_caret_means_power():
    assert final_of("x^2") == "x^{2}"


def test_bad_character_is_refused():
    with pytest.raises(ValueError):
        parse_expression("2 + @")


def test_equation_in_simplify_mode_is_refused():
    with pytest.raises(ValueError):
        parse_expression("2*x = 4")


# --- Whole step lists ------------------------------------------------------
def test_first_line_matches_what_you_typed():
    steps = steps_for("2*x + 3*x + 5 - 1")
    assert steps[0]["latex"] == "2 x + 3 x + 5 - 1"


def test_full_step_list_for_a_typical_question():
    steps = steps_for("2*x + 3*x + 5 - 1")
    assert [step["name"] for step in steps] == [
        "Initial expression",
        "Evaluate numbers",
        "Combine like terms",
    ]
    assert steps[-1]["latex"] == "5 x + 4"


def test_steps_never_repeat_the_same_screen():
    steps = steps_for("2*(x + 3) + 4*x + 5 - 2")
    latexes = [step["latex"] for step in steps]
    assert len(latexes) == len(set(latexes))


def test_already_simple_expression_has_one_step():
    steps = steps_for("x + 1")
    assert len(steps) == 1


def test_steps_are_capped():
    # A silly expression that could keep changing must still stop.
    steps = steps_for("(x + 1) * (x + 1) * (x + 1) * (x + 1)")
    assert len(steps) <= 21
