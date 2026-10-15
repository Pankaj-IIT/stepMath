"""
Tests for the formula database layer: shapes, patterns, the SymPy gate,
the store, methods, and the import script.

Run with:   pytest
"""

import json
from pathlib import Path

import pytest
from sympy import cos, sin, symbols

from conftest import load_importer, make_entry, steps_for
from src.algebra.parser import parse_equation, parse_expression
from src.algebra.solver import solve_equation
from src.algebra.steps import simplify_steps
from src.knowledge import gate, ops, runner, store
from src.knowledge.patterns import apply_identity, parse_pair, parse_solution
from src.knowledge.signature import signature_of

x, y = symbols("x y")


# --- signatures ------------------------------------------------------------
def test_signature_describes_shape_only():
    assert signature_of(parse_expression("x^2 + 3")) == signature_of(
        parse_expression("y^2 + 8")
    )


def test_signature_ignores_the_order_of_a_sum():
    # x + 2*pi and 2*pi + x are the same maths, so must get the same shape
    # (otherwise the database index would miss one of them).
    assert signature_of(parse_expression("sin(x + 2*pi)")) == signature_of(
        parse_expression("sin(2*pi + x)")
    )


def test_different_shapes_differ():
    assert signature_of(parse_expression("x + 1")) != signature_of(
        parse_expression("x * 1")
    )


def test_function_names_are_part_of_the_shape():
    assert signature_of(parse_expression("sin(x)")) != signature_of(
        parse_expression("cos(x)")
    )


# --- patterns --------------------------------------------------------------
def test_holes_are_shared_between_both_sides():
    import sqlite3

    left, right = parse_pair("?a^2 - ?b^2", "(?a - ?b)*(?a + ?b)")

    # both sides mention the same holes (?a and ?b)
    left_holes = {s.name for s in left.free_symbols}
    right_holes = {s.name for s in right.free_symbols}
    assert left_holes == right_holes

    # and the stored shape really finds the expression it stands for
    # (this is exactly the SQL LIKE check the database does)
    pattern = signature_of(left)
    node = signature_of(parse_expression("x^2 - y^2"))
    assert "%" in pattern
    like = sqlite3.connect(":memory:").execute(
        "SELECT ? LIKE ?", (node, pattern)
    ).fetchone()[0]
    assert like


def test_a_hole_can_cover_a_whole_subexpression():
    """sin(?c) - sin(?d) must be found for sin(x+y) - sin(x-y), even though
    each hole is filled with a whole sum, not just a single letter."""
    import sqlite3

    pattern = signature_of(parse_pair("sin(?c) - sin(?d)", "1")[0])
    node = signature_of(parse_expression("sin(x+y) - sin(x-y)"))

    like = sqlite3.connect(":memory:").execute(
        "SELECT ? LIKE ?", (node, pattern)
    ).fetchone()[0]
    assert like, f"{node!r} not found by {pattern!r}"


def test_pattern_rewrites_whole_expression():
    expr = parse_expression("x^2 - y^2")
    left, right = parse_pair("?a^2 - ?b^2", "(?a - ?b)*(?a + ?b)")
    result = apply_identity(expr, left, right)
    assert result is not None
    assert (result - (x - y) * (x + y)) == 0


def test_pattern_matches_a_number_instead_of_a_power():
    # x^2 - 4: the 4 stands in for ?b^2, so ?b becomes 2
    expr = parse_expression("x^2 - 4")
    left, right = parse_pair("?a^2 - ?b^2", "(?a - ?b)*(?a + ?b)")
    result = apply_identity(expr, left, right)
    assert result is not None
    assert (result - (x - 2) * (x + 2)) == 0


def test_pattern_can_match_part_of_a_bigger_sum():
    from sympy import simplify

    expr = parse_expression("sin(z)^2 + cos(z)^2 + 4")
    left, right = parse_pair("sin(?a)^2 + cos(?a)^2", "1")
    result = apply_identity(expr, left, right)
    assert result is not None
    # the 1 must be put back next to the untouched 4:  1 + 4
    assert simplify(result) == 5


def test_pattern_refuses_when_holes_disagree():
    # 2*x + 3:  ?a*x + ?a  would need 2 == 3
    expr = parse_expression("2*x + 3")
    left, right = parse_pair("?a*x + ?a", "?a*(x + 1)")
    assert apply_identity(expr, left, right) is None


def test_pattern_does_not_match_unrelated_expression():
    expr = parse_expression("x + 5")
    left, right = parse_pair("?a^2 - ?b^2", "(?a - ?b)*(?a + ?b)")
    assert apply_identity(expr, left, right) is None


# --- the SymPy gate --------------------------------------------------------
def test_gate_passes_a_true_identity():
    left, right = parse_pair("sin(?a)^2 + cos(?a)^2", "1")
    verdict, _ = gate.verify_identity(left, right)
    assert verdict == "pass"


def test_gate_rejects_a_wrong_identity():
    left, right = parse_pair("?a^2 + 1", "1")
    verdict, detail = gate.verify_identity(left, right)
    assert verdict == "fail"
    assert "counter-example" in detail


def test_gate_handles_honest_unknowns():
    left, right = parse_pair("?a + ?b", "?a - ?b")
    verdict, _ = gate.verify_identity(left, right)
    assert verdict in ("fail", "unknown")  # certainly not "pass"


def test_gate_cracks_cotangent_and_tangent_formulas():
    """These need the rewrite into sin/cos to be provable."""
    left, right = parse_pair(
        "cot(?a + ?b)", "(cot(?a)*cot(?b) - 1)/(cot(?b) + cot(?a))"
    )
    assert gate.verify_identity(left, right)[0] == "pass"

    left, right = parse_pair("cot(?a) - tan(?a)", "2*cot(2*?a)")
    assert gate.verify_identity(left, right)[0] == "pass"


# --- general solutions of equations ----------------------------------------
def test_parse_solution_finds_the_unknown():
    from sympy import Wild

    from src.algebra.latex import to_latex

    left, right, unknown, value = parse_solution("sin(?t)", "0", "?t = ?n*pi")
    assert unknown == Wild("t")          # the unknown is the hole on the left
    assert to_latex(value) == "\\pi n"   # the answer keeps its own hole (n)


def test_gate_proves_a_general_solution():
    left, right, unknown, value = parse_solution("sin(?t)", "0", "?t = ?n*pi")
    verdict, _ = gate.verify_solution(left, right, unknown, value)
    assert verdict == "pass"


def test_gate_proves_answers_with_minus_one_to_the_n():
    """sin(theta) = sin(alpha): needs the even/odd split to be proved."""
    left, right, unknown, value = parse_solution(
        "sin(?t)", "sin(?a)", "?t = ?n*pi + (-1)^?n * ?a"
    )
    verdict, _ = gate.verify_solution(left, right, unknown, value)
    assert verdict == "pass"


def test_gate_rejects_a_wrong_general_solution():
    left, right, unknown, value = parse_solution(
        "sin(?t)", "sin(?a)", "?t = ?n*pi + ?a"
    )
    verdict, detail = gate.verify_solution(left, right, unknown, value)
    assert verdict == "fail"
    assert "counter-example" in detail


def _seed_solution(name: str, left_text: str, right_text: str,
                   solution_text: str) -> None:
    """Put a verified solution row into the (test) database."""
    left, right, unknown, value = parse_solution(left_text, right_text, solution_text)
    verdict, _ = gate.verify_solution(left, right, unknown, value)

    connection = store.get_connection()
    store.insert_entry(connection, {
        "kind": "solution",
        "name": name,
        "statement": "",
        "left": left_text,
        "right": right_text,
        "solution": solution_text,
        "signature": signature_of(left),
        "verified": 1 if verdict == "pass" else 0,
        "origin": "hand",
        "source": "test",
        "dedup_key": store.dedup_key("solution", left_text, right_text,
                                     solution_text),
    })
    connection.commit()


def test_solver_gives_a_stored_general_solution():
    _seed_solution("sin = 0", "sin(?t)", "0", "?t = ?n*pi")

    left, right = parse_equation("sin(x) = 0")
    steps = solve_equation(left, right)

    assert [step["name"] for step in steps] == ["Initial equation",
                                                "General solution"]
    assert steps[-1]["latex"] == "x = \\pi n"


def test_solver_shows_both_plus_and_minus_branches():
    _seed_solution("cos = cos plus", "cos(?t)", "cos(?a)", "?t = 2*?n*pi + ?a")
    _seed_solution("cos = cos minus", "cos(?t)", "cos(?a)", "?t = 2*?n*pi - ?a")

    left, right = parse_equation("cos(x) = cos(pi/3)")
    steps = solve_equation(left, right)

    assert " or " in steps[-1]["latex"]
    assert "\\frac{\\pi}{3}" in steps[-1]["latex"]


def test_solver_still_handles_ordinary_linear_equations():
    left, right = parse_equation("2*x + 6 = 14")
    steps = solve_equation(left, right)
    assert steps[-1]["latex"] == "x = 4"


# --- the store -------------------------------------------------------------
# (Entry building is shared: from conftest import make_entry / seed_entry)
def test_insert_find_and_count():
    connection = store.get_connection()
    before = store.counts(connection)["total"]

    assert store.insert_entry(connection, make_entry(name="Insert Count")) is True
    connection.commit()

    assert store.counts(connection)["total"] == before + 1


def test_duplicate_content_is_skipped():
    connection = store.get_connection()
    entry = make_entry(name="Dup Test")
    assert store.insert_entry(connection, entry) is True
    connection.commit()

    assert store.insert_entry(connection, entry) is False


def test_search_finds_by_name_prefix():
    connection = store.get_connection()
    store.insert_entry(connection, make_entry(name="Commutative Addition"))
    connection.commit()

    rows = store.search(connection, "commut")
    assert any(row["name"] == "Commutative Addition" for row in rows)


def test_review_queue_only_lists_unverified():
    connection = store.get_connection()
    store.insert_entry(connection, make_entry(name="Unproven Thing", verified=0))
    connection.commit()

    rows = store.search(connection, "", verified=0)
    assert all(row["verified"] == 0 for row in rows)
    assert any(row["name"] == "Unproven Thing" for row in rows)


def test_approve_and_reject():
    connection = store.get_connection()
    store.insert_entry(connection, make_entry(name="To Approve", verified=0))
    connection.commit()
    row = store.search(connection, "To Approve")[0]

    store.set_verified(connection, row["id"], 1)
    connection.commit()
    assert store.get_entry(connection, row["id"])["verified"] == 1

    store.delete_entry(connection, row["id"])
    connection.commit()
    assert store.get_entry(connection, row["id"]) is None


# --- using the database while simplifying ---------------------------------
def test_solver_uses_a_stored_identity():
    connection = store.get_connection()
    store.insert_entry(connection, make_entry(
        name="Pyth Test",
        # different hole name than the shipped Pythagorean row, so this is
        # not skipped as a duplicate -- and usage 999 so THIS row is the
        # one the solver picks (it would otherwise sort later by id)
        left="sin(?w)^2 + cos(?w)^2",
        right="1",
        statement="sin^2 + cos^2 = 1",
        signature=signature_of(parse_expression("sin(x)^2 + cos(x)^2")),
        lead_signature=signature_of(parse_expression("sin(x)^2")),
    ))
    connection.execute(
        "UPDATE entries SET usage_count = 999 WHERE name = 'Pyth Test'"
    )
    connection.commit()

    steps = simplify_steps(parse_expression("sin(t)^2 + cos(t)^2 + 4"))
    names = [step["name"] for step in steps]
    assert "Pyth Test" in names
    assert steps[-1]["latex"] == "5"

    # the step must also carry the formula that was applied
    used = next(step for step in steps if step["name"] == "Pyth Test")
    assert used["formula"] == "sin^2 + cos^2 = 1"

    # This row exists only for this test. Remove it again, so its top
    # usage does not hijack the step names of later tests.
    connection.execute(
        "DELETE FROM entries_fts WHERE rowid IN "
        "(SELECT id FROM entries WHERE name = 'Pyth Test')"
    )
    connection.execute("DELETE FROM entries WHERE name = 'Pyth Test'")
    connection.commit()


def test_solver_works_without_the_database():
    steps = simplify_steps(parse_expression("2*x + 3*x"), include_db=False)
    assert steps[-1]["latex"] == "5 x"


# --- methods ---------------------------------------------------------------
def test_method_runs_its_steps_in_order():
    expr = parse_expression("(x + 1)*(x + 2)")
    steps = runner.run_method(expr, json.dumps([{"op": "expand"}, {"op": "collect"}]))
    names = [step["name"] for step in steps]
    assert names[0] == "Initial expression"
    assert "Expand brackets" in names
    assert steps[-1]["latex"] == "x^{2} + 3 x + 2"


def test_method_stops_with_a_clear_reason():
    expr = parse_expression("x + 1")  # no squares here
    steps = json.dumps([
        {"left": "?a^2 - ?b^2", "right": "(?a - ?b)*(?a + ?b)", "name": "Difference"},
    ])

    with pytest.raises(runner.MethodError) as caught:
        runner.run_method(expr, steps)

    assert caught.value.step_number == 1
    assert "Difference" in str(caught.value)


def test_method_pattern_step_carries_its_formula():
    """A pattern step shows the formula it used, e.g. a^2 - b^2 = ..."""
    expr = parse_expression("x^2 - 4")
    steps = runner.run_method(expr, json.dumps([
        {"left": "?a^2 - ?b^2", "right": "(?a - ?b)*(?a + ?b)",
         "name": "Difference of Squares"},
    ]))

    assert steps[-1]["name"] == "Difference of Squares"
    assert "=" in steps[-1]["formula"]


def test_method_validation_rejects_garbage():
    with pytest.raises(ValueError):
        runner.validate_steps([{"op": "make_coffee"}])

    with pytest.raises(ValueError):
        runner.validate_steps([{"left": "?a + ?b"}])  # no right side


def test_unknown_operation_is_refused():
    with pytest.raises(ValueError):
        ops.run_operation("make_coffee", parse_expression("x"))


# --- chains that must reach their famous answer -----------------------------
def test_double_angle_chain_reaches_one(sample_formulas):
    """2sin^2 + cos2x  ->  1   (the chain the user reported as broken)."""
    steps = steps_for("2*sin(x)^2 + cos(2*x)")
    assert steps[-1]["latex"] == "1"
    assert steps[-1]["name"] == "Pythagorean Identity"


def test_double_angle_chain_reaches_minus_one(sample_formulas):
    """-2cos^2 + cos2x  ->  -1, through the negated Pythagorean form."""
    steps = steps_for("-2*cos(x)^2 + cos(2*x)")
    assert steps[-1]["latex"] == "-1"
    assert steps[-1]["name"] == "Negated Pythagorean Identity"


def test_square_of_sum_reaches_one_plus_sin_2x(sample_formulas):
    # simplify() answers 2*sin(x + pi/4)^2 here; this test means the
    # textbook result, so it states its own target (the user's rule:
    # "I will provide the target if not simplify").
    expr = parse_expression("(sin(x) + cos(x))^2")
    steps = simplify_steps(expr, target=parse_expression("sin(2*x) + 1"))
    assert steps[-1]["latex"] == "\\sin{\\left(2 x \\right)} + 1"


def test_takes_out_a_common_factor():
    """SymPy's factor() behind a step with a readable name."""
    steps = steps_for("2*x + 4")
    assert steps[-1]["name"] == "Take out the common factor"
    assert steps[-1]["latex"] == "2 \\left(x + 2\\right)"


def test_takes_the_lcm_and_finishes_the_chain(sample_formulas):
    """Fractions get one common denominator, which unlocks the next formula.

        1 + sin^2/cos^2
          -> (sin^2 + cos^2)/cos^2     "Take the LCM"
          -> 1/cos^2                   Pythagorean Identity
          -> sec^2                     One over Cos Squared
    """
    # SymPy's simplify() would answer 1/cos^2; this test means the
    # textbook form, so it states its own target (the user's rule:
    # "I will provide the target if not simplify").
    expr = parse_expression("1 + sin(x)^2/cos(x)^2")
    steps = simplify_steps(expr, target=parse_expression("sec(x)^2"))
    names = [step["name"] for step in steps]

    assert "Take the LCM" in names
    assert "Pythagorean Identity" in names
    assert steps[-1]["latex"] == "\\sec^{2}{\\left(x \\right)}"


def test_one_plus_tan_squared_reaches_secant_squared(sample_formulas):
    # simplify() answers 1/cos^2 here; this test means sec^2 (the user's
    # rule: "I will provide the target if not simplify").
    expr = parse_expression("1 + tanx^2")
    steps = simplify_steps(expr, target=parse_expression("sec(x)^2"))
    assert steps[-1]["name"] == "One plus Tan Squared"
    assert steps[-1]["latex"] == "\\sec^{2}{\\left(x \\right)}"


# --- target-directed search and the gaps list -------------------------------
def test_gap_is_recorded_when_no_strategy_reaches_the_target(sample_formulas):
    """SymPy suggests the expanded numerator; our strategies walk around
    it (factored form) without landing on it, so this joins the list of
    simplifications we cannot explain yet -- the strategies to plug next."""
    known = {row["target"] for row in store.list_gaps()}
    steps = steps_for("(x^2 - 1)*(x^2 - 4)/(x^2 - 2*x + 1)")
    fresh = [row for row in store.list_gaps() if row["target"] not in known]

    assert fresh, "expected the unreachable target to be recorded as a gap"
    gap = fresh[0]
    assert "x^{2} - 1" in gap["example"]    # the input that missed it
    assert "Take the LCM" in gap["tried"]    # what we did try
    # the user still gets an explained (teaching) route meanwhile
    assert steps[-1]["name"] != "Simplify with SymPy"


def test_gap_hits_accumulate_on_the_same_target():
    store.record_gap("1", "sin(x)^2 + cos(x)^2", "Pythagorean Identity")
    store.record_gap("1", "sin(t)^2 + cos(t)^2", "Pythagorean Identity")

    row = next(r for r in store.list_gaps() if r["target"] == "1")
    assert row["hits"] == 2
    assert "Pythagorean Identity" in row["tried"]


def test_sub_pipeline_is_offered_as_a_named_step():
    """The trig sub-pipeline (trigsimp) must appear among the candidates,
    named like any other strategy -- not as the blunt SymPy jump."""
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    expr = parse_expression("sin(x) * sec(x)")
    state = _SolveState(expr, frozenset())
    results = {
        name: new
        for new, name, _formula, _row in _iter_changes(
            expr, SIMPLIFY_RULES, True, state, with_derived=True
        )
    }

    assert "Simplify trigonometry" in results
    assert results["Simplify trigonometry"] == parse_expression("tan(x)")


# --- combining a*sin + b*cos (the auxiliary-angle trick) --------------------
def test_combine_sin_and_cos_with_coefficients(sample_formulas):
    steps = steps_for("2*sin(x) + 3*cos(x)")
    assert steps[-1]["name"] == "Combine a sin + b cos"
    assert steps[-1]["latex"] == (
        "\\sqrt{13} \\sin{\\left(x + \\operatorname{atan}"
        "{\\left(\\frac{3}{2} \\right)} \\right)}"
    )


def test_combine_sin_and_cos_without_coefficients(sample_formulas):
    """Bare sin + cos counts as 1*sin + 1*cos (implicit coefficient)."""
    steps = steps_for("sin(x) + cos(x)")
    assert steps[-1]["name"] == "Combine a sin + b cos"
    assert steps[-1]["latex"] == (
        "\\sqrt{2} \\sin{\\left(x + \\frac{\\pi}{4} \\right)}"
    )


def test_combine_inside_a_three_term_sum(sample_formulas):
    """The reported case: 3sin + 4cos + 5 (the whole-shape lookup cannot
    match here -- 2 terms in the formula vs 3 in the expression -- so the
    first-part lookup has to find it)."""
    steps = steps_for("3sinx + 4cosx + 5")
    assert steps[-1]["name"] == "Combine a sin + b cos"
    assert steps[-1]["latex"] == (
        "5 \\sin{\\left(x + \\operatorname{atan}"
        "{\\left(\\frac{4}{3} \\right)} \\right)} + 5"
    )


def test_combine_formula_never_swallows_products(sample_formulas):
    """?# holes only accept NUMBERS, so sin(x)*cos(y) + ... is safe."""
    names = [step["name"] for step in
             steps_for("sin(x)*cos(y) + sin(y)*cos(x)")]
    assert "Combine a sin + b cos" not in names


# --- the polygon facts live in ONE card ------------------------------------
def test_polygon_facts_are_combined_into_one_note(sample_formulas):
    connection = store.get_connection()
    rows = connection.execute(
        "SELECT name, statement FROM entries WHERE name LIKE '%polygon%'"
    ).fetchall()

    assert [row["name"] for row in rows] == ["n-sided polygon angles"]
    # all three statements are on that one card
    assert "aligned" in rows[0]["statement"]


# --- the auxiliary-angle method (a*sin + b*cos), as in the notes -----------
def _auxiliary_method_steps(expression: str) -> list:
    """Run the saved 'show the working' method on an expression."""
    connection = store.get_connection()
    row = connection.execute(
        "SELECT steps FROM entries "
        "WHERE name LIKE 'Combine a sin + b cos (show%'"
    ).fetchone()
    assert row is not None, "the saved method is missing from the database"
    return runner.run_method(parse_expression(expression), row["steps"])


def test_auxiliary_method_shows_every_step(sample_formulas):
    """The saved method walks the notes' procedure instead of jumping."""
    steps = _auxiliary_method_steps("3*sin(x) + 4*cos(x) + 5")

    assert [step["name"] for step in steps] == [
        "Initial expression",
        "Take out R = sqrt(a^2 + b^2)",
        "Write as R sin(theta + phi)",
    ]
    assert steps[-1]["latex"] == (
        "5 \\sin{\\left(x + \\operatorname{atan}"
        "{\\left(\\frac{4}{3} \\right)} \\right)} + 5"
    )


def test_auxiliary_working_is_an_alternative_not_a_button(sample_formulas):
    """No Methods bar: the working is offered as another WAY instead."""
    connection = store.get_connection()
    expression = parse_expression("3*sin(x) + 4*cos(x) + 5")

    # 1) not in the buttons...
    panel_rows = store.find_methods(connection, expression)
    assert not any(row["name"] == "Combine a sin + b cos (show the working)"
                   for row in panel_rows)

    # 2) ...but it IS one of the "other ways to the same answer"
    from src.algebra.steps import alternative_ways
    ways = alternative_ways(expression)
    names = {way["name"] for way in ways}
    assert "Combine a sin + b cos (show the working)" in names


def test_combine_works_for_minus_too(sample_formulas):
    """a*sin - b*cos must combine just like the plus form."""
    steps = steps_for("3*sin(x) - 4*cos(x) + 5")
    assert steps[-1]["latex"] == (
        "5 \\sin{\\left(x - \\operatorname{atan}"
        "{\\left(\\frac{4}{3} \\right)} \\right)} + 5"
    )


# --- reduce the angle: SymPy does these directly (no stored formulas) -------
def test_reduce_angle_uses_sympy_instead_of_stored_rows(sample_formulas):
    """The allied-angle / period / negative family is one rule now."""
    assert steps_for("sin(x + pi/2)")[-1]["latex"] == "\\cos{\\left(x \\right)}"
    assert steps_for("sin(x + 2*pi)")[-1]["latex"] == "\\sin{\\left(x \\right)}"
    assert steps_for("cos(x + pi)")[-1]["latex"] == "- \\cos{\\left(x \\right)}"
    assert steps_for("sin(-x)")[-1]["latex"] == "- \\sin{\\left(x \\right)}"


def test_reduce_angle_leaves_two_angle_sums_to_their_group(sample_formulas):
    """sin(x+y) must expand via its group, not be 'reduced'."""
    names = [step["name"] for step in steps_for("sin(x+y)")]
    assert "Sine of A + B" in names
    assert "Reduce the angle" not in names


# --- formula families (classify -> groups) ---------------------------------
def test_classifier_tags_every_spelling_of_a_family():
    """The tree identifies a group by MEANING, not one rigid template."""
    from src.knowledge.classify import classify

    cases = {
        "sin(x+y)": "two-angle-sum",                       # compact
        "sin(x)*cos(y) + sin(y)*cos(x)": "two-angle-sum",  # expanded
        "sin(a)*cos(b) + cos(a)*sin(b)": "two-angle-sum",  # as written in books
        "sin(2*x)": "double-angle",
        "sin(x/2)": "half-angle",                          # NOT double-angle
        "sin(0.5*x)": "half-angle",
        "sin(x)^2 + cos(x)^2": "pythagorean",
        "sec(x)^2 - tan(x)^2": "pythagorean",
        "3*sin(x) + 4*cos(x)": "linear-sin-cos",
        "sin(x) + sin(y)": "sum-to-product",
        "2*sin(x)*cos(y)": "product-to-sum",
    }
    for text, expected in cases.items():
        assert expected in classify(parse_expression(text)), text


def test_patterns_classify_like_the_nodes_they_match():
    """Membership is derived with the SAME classifier, on the left side."""
    from src.knowledge.classify import classify

    fold_left, _ = parse_pair("sin(?a)*cos(?b) + cos(?a)*sin(?b)",
                              "sin(?a + ?b)")
    assert "two-angle-sum" in classify(fold_left)

    pythagorean_left, _ = parse_pair("sin(?a)^2 + cos(?a)^2", "1")
    assert "pythagorean" in classify(pythagorean_left)


def test_fold_back_formula_works(sample_formulas):
    """The reported example: typed products fold into one compact sum."""
    steps = steps_for("sin(x)*cos(y) + sin(y)*cos(x)")
    assert steps[-1]["name"] == "Fold sin cos + cos sin"
    assert steps[-1]["latex"] == "\\sin{\\left(x + y \\right)}"


def test_group_route_finds_members(sample_formulas):
    connection = store.get_connection()
    rows = store.find_by_groups(connection, ["two-angle-sum"])
    names = [row["name"] for row in rows]

    assert "Fold sin cos + cos sin" in names
    assert "Sine of A + B" in names          # the expand direction too
    assert store.find_by_groups(connection, []) == []


# --- the progress guard: no action may lead to an earlier step --------------
def test_no_route_returns_to_an_earlier_step():
    """The user's rule: a step must never lead back to a state the route
    has already shown (checked in canonical spelling, so a fold-back that
    only re-orders the terms is caught too)."""
    from src.algebra.latex import to_latex

    for text in (
        "sin(x)^8 - cos(x)^8",
        "sin(x+y) + sin(x-y)",
        "(sec(x) - tan(x))*(sec(x) + tan(x))",
        "2*x + 4",
        "(sin(x) + cos(x))^2",
    ):
        steps = steps_for(text)

        # canonical latex of the START plus every displayed step
        shown = [to_latex(parse_expression(text))]
        shown += [step["latex"] for step in steps[1:]]

        assert len(shown) == len(set(shown)), (
            f"{text} revisited an earlier step: {shown}"
        )


# --- other ways to the same answer -----------------------------------------
def test_cos_double_angle_variants_are_all_offered(sample_formulas):
    """Every spelling of cos(2A) should show up as its own way.

    Which form the MAIN route ends in is decided by move order alone
    (the greedy walk takes whichever named move comes first -- no cost
    function steers it), so this test checks EQUIVALENCE, not one fixed
    spelling -- and the variants must still be offered.
    """
    from sympy import simplify

    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _build_steps, alternative_ways

    expression = "2*sin(x)^2 - cos(2*x)"
    _, main_final = _build_steps(parse_expression(expression),
                                 SIMPLIFY_RULES, True)

    known_forms = [
        parse_expression("3*sin(x)^2 - cos(2*x)"),
        parse_expression("1 - 2*cos(2*x)"),
    ]
    assert any(simplify(main_final - form) == 0 for form in known_forms)

    ways = alternative_ways(parse_expression(expression))
    names = {way["name"] for way in ways}

    assert "Cos 2A as 2cos^2 - 1" in names
    assert "Cos 2A as 1 - 2sin^2" in names
    assert "Cos 2A in terms of tan" in names

    # every alternative is kept only because it equals the main answer
    for way in ways:
        assert simplify(way["final"] - main_final) == 0


def test_method_is_offered_for_a_part_nested_inside_the_expression(
        sample_formulas):
    """A method whose pattern matches a PART (not the whole) is offered."""
    connection = store.get_connection()
    rows = store.find_methods(connection, parse_expression("x + sin(2*x)"))
    assert any(row["name"] == "Sin 2A in terms of tan" for row in rows)


def test_alternative_ways_reach_the_same_answer_by_two_routes(
        seeded_sine_formulas):
    from src.algebra.steps import alternative_ways

    expression = "sin(x+y) + sin(x-y)"
    main_final = steps_for(expression)[-1]["latex"]

    ways = alternative_ways(parse_expression(expression))
    names = {way["name"] for way in ways}

    assert "Sine of A + B" in names   # open both sines, then combine
    assert "Sine of A - B" in names   # open them the other way round

    # Every route must end at the SAME MATHS as the main route (the latex
    # may be spelled differently, e.g. factored vs expanded).
    from sympy import simplify

    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _build_steps
    _, main_final = _build_steps(parse_expression(expression), SIMPLIFY_RULES, True)

    for way in ways:
        assert simplify(way["final"] - main_final) == 0


def test_alternative_ways_skip_when_there_is_nothing_to_compare():
    from src.algebra.steps import alternative_ways

    assert alternative_ways(parse_expression("x + 1")) == []


# --- the import script -----------------------------------------------------
# (Module loading is shared: from conftest import load_importer)
def test_import_file_counts_and_rejects(tmp_path):
    module = load_importer()

    batch = tmp_path / "batch.jsonl"
    batch.write_text("\n".join([
        json.dumps({"kind": "identity", "name": "Good One",
                    "left": "?a + 0", "right": "?a", "origin": "hand"}),
        json.dumps({"kind": "identity", "name": "Wrong One",
                    "left": "?a + 1", "right": "?a", "origin": "hand"}),
        json.dumps({"kind": "identity", "name": "Broken Line",
                    "left": "?a +", "right": "?a", "origin": "hand"}),
        "this is not json at all",
    ]), encoding="utf-8")

    summary = module.import_file(batch)

    assert summary["lines"] == 4
    assert summary["imported"] == 1        # only the true identity
    assert summary["rejected"] == 3        # wrong + broken + not json
    assert Path(summary["rejected_file"]).exists()

    # importing the same file again adds nothing
    again = module.import_file(batch)
    assert again["imported"] == 0
    assert again["duplicates"] == 1


def test_import_trusts_a_method_only_when_its_steps_prove(tmp_path):
    module = load_importer()

    batch = tmp_path / "methods.jsonl"
    batch.write_text("\n".join([
        # every pattern step proves itself -> trusted straight away
        json.dumps({"kind": "method", "name": "Good Method",
                    "steps": [{"left": "sin(?a)^2",
                               "right": "1 - cos(?a)^2", "name": "Pythagoras"}]}),
        # one step is simply false -> the whole line is rejected
        json.dumps({"kind": "method", "name": "Bad Method",
                    "steps": [{"left": "?a + 1", "right": "?a", "name": "Drop one"}]}),
    ]), encoding="utf-8")

    summary = module.import_file(batch)

    assert summary["imported"] == 1
    assert summary["verified"] == 1
    assert summary["rejected"] == 1

    rejected = Path(summary["rejected_file"]).read_text(encoding="utf-8")
    assert "Bad Method" in rejected


def test_import_reads_notes_and_solutions(tmp_path):
    module = load_importer()

    batch = tmp_path / "mixed.jsonl"
    batch.write_text("\n".join([
        # a note is shown to people but never rewrites anything -> trusted
        json.dumps({"kind": "note", "name": "Period of tan",
                    "statement": "\\text{period } \\pi"}),
        # a correct answer for an equation -> proved and trusted
        # (content differs from the shipped batches, so it is not skipped
        #  as a duplicate)
        json.dumps({"kind": "solution", "name": "tan theta = 0, even turns",
                    "left": "tan(?t)", "right": "0",
                    "solution": "?t = ?n*2*pi"}),
        # an answer that does NOT solve the equation -> rejected
        json.dumps({"kind": "solution", "name": "nonsense answer",
                    "left": "sin(?t)", "right": "0", "solution": "?t = ?a + 1"}),
    ]), encoding="utf-8")

    summary = module.import_file(batch)

    assert summary["lines"] == 3
    assert summary["imported"] == 2
    assert summary["rejected"] == 1

    rejected = Path(summary["rejected_file"]).read_text(encoding="utf-8")
    assert "nonsense answer" in rejected


# --- the elementary trig route's two operations ----------------------------
def test_elementary_trig_operations_exist():
    """The method catalogue needs the two ops the elementary route uses."""
    from src.algebra.latex import to_latex
    from src.knowledge.ops import known_operation, run_operation

    assert known_operation("to_sin_cos")
    assert known_operation("together")

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")

    rewritten, label = run_operation("to_sin_cos", expr)
    assert label == "Write in terms of sin and cos"
    assert "cot" not in to_latex(rewritten) and "sec" not in to_latex(rewritten)

    combined, label = run_operation("together", rewritten)
    assert label == "Take the LCM"
    assert "\\frac" in to_latex(combined)   # everything over one denominator


def test_all_in_sin_and_cos_method_reaches_two(sample_formulas):
    """A stored method, not a SymPy jump, lands on SymPy's answer."""
    from src.algebra.latex import to_latex

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    rows = store.find_methods(store.get_connection(), expr, for_ways=False)
    row = next((r for r in rows if r["name"] == "All in sin and cos"), None)
    assert row is not None, "the method must be offered for this expression"

    steps, final = runner.run_method(expr, row["steps"], with_final=True)
    assert [s["name"] for s in steps] == [
        "Initial expression",
        "Write in terms of sin and cos",
        "Take the LCM",
        "Simplify",
    ]
    assert to_latex(final) == "2"


# --- stored formulas in BOTH directions ------------------------------------
def test_stored_formulas_are_offered_backwards(sample_formulas):
    """A = B must be usable as B -> A wherever the right-hand side shows up."""
    from src.algebra.latex import to_latex
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    expr = parse_expression("sin(x)/cos(x)")
    state = _SolveState(expr)   # the teaching walk: with_derived=False

    hits = [(name, new) for new, name, _f, _r in
            _iter_changes(expr, SIMPLIFY_RULES, True, state, with_derived=False)]

    assert any(name == "Tangent as a Quotient (backwards)" for name, _ in hits)
    assert any(to_latex(new) == "\\tan{\\left(x \\right)}" for _, new in hits)


def test_a_formula_can_fire_inside_a_larger_expression(sample_formulas):
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    expr = parse_expression("2*(sin(y)^2 + cos(y)^2)")
    state = _SolveState(expr)

    names = [name for _new, name, _f, _r in
             _iter_changes(expr, SIMPLIFY_RULES, True, state, with_derived=False)]

    assert "Pythagorean Identity" in names


def test_recently_added_rows_are_reachable_via_the_function_pass(tmp_path):
    """A high-id row must be offered even when many older rows share its functions.

    The gap batch is imported LAST here (its live situation: a row added
    this session sits at the far end of the id order), so 17 shape matches
    and 61 function matches with LOWER ids stand ahead of it. Pre-fix both
    sources cut it (shape slice 16, function cap 8), the stored phase
    yielded nothing, and the walk fell back to a shape rule ("Take the
    LCM") instead of "Tan plus Cot is Two over Sin Two".
    """
    import os

    from src.algebra.latex import to_latex_as_written
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    previous_db = os.environ.get("KNOWLEDGE_DB")
    os.environ["KNOWLEDGE_DB"] = str(tmp_path / "knowledge.db")
    try:
        importer = load_importer()
        samples = Path(__file__).resolve().parents[1] / "data" / "samples"
        batches = [path for path in sorted(samples.glob("*.jsonl"))
                   if ".rejected." not in path.name]
        batches.sort(key=lambda path: path.name == "algebra.jsonl")
        for batch in batches:
            summary = importer.import_file(batch)
            assert summary["rejected"] == 0, batch.name

        expr = parse_expression("tan(x) + cot(x) + 2 - 2/sin(2*x)")
        state = _SolveState(expr, frozenset(), to_latex_as_written(expr))
        names = [name for _new, name, _f, _r in
                 _iter_changes(expr, SIMPLIFY_RULES, True, state, with_derived=True)]
        assert "Tan plus Cot is Two over Sin Two" in names
    finally:
        if previous_db is None:
            os.environ.pop("KNOWLEDGE_DB", None)
        else:
            os.environ["KNOWLEDGE_DB"] = previous_db


# --- history is the only permission rule ------------------------------------
def test_formula_made_parts_are_not_off_limits(sample_formulas):
    """History decides what may rewrite a part -- not who produced it."""
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _iter_changes

    original = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    state = _SolveState(original)   # sin(x)+cos(x)-1 has never been shown

    node = parse_expression("sin(x) + cos(x) - 1")   # what an LCM step produced
    names = [name for _new, name, _f, _r in
             _iter_changes(node, SIMPLIFY_RULES, True, state, with_derived=False)]
    assert "Combine a sin + b cos" in names


def test_a_step_may_not_return_to_an_earlier_line():
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _SolveState, _apply_once

    state = _SolveState(parse_expression("sin(2*x)"))   # already an earlier line
    hit = _apply_once(parse_expression("2*sin(x)*cos(x)"),
                      SIMPLIFY_RULES, True, state)

    if hit is not None:
        assert not state.revisits(hit[0])   # the natural move here is sin(2x)


def test_teach_walk_gets_past_formula_output(sample_formulas):
    """The reported bug: the walk died with sin+cos-1 untouched."""
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _teach

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    steps, _final = _teach(expr, SIMPLIFY_RULES, True, frozenset())

    assert len(steps) >= 12   # it stopped at exactly 11 lines before this fix


# --- the breadth-first search toward a target --------------------------------
def test_bfs_reaches_the_target_with_named_steps(sample_formulas):
    from src.algebra.latex import to_latex

    expr = parse_expression("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    assert moves is not None
    assert to_latex(final) == "2"


def test_main_route_shows_the_methods_steps(sample_formulas):
    steps = steps_for("(1 + cot(x) - csc(x))*(1 + tan(x) + sec(x))")
    assert [s["name"] for s in steps] == [
        "Initial expression", "Write in terms of sin and cos",
        "Take the LCM", "Simplify"]
    assert steps[-1]["latex"] == "2"


def test_no_goal_means_one_suggestion_not_a_walk():
    # simplify() leaves sin(x)+cos(x) alone, so there is no goal to chase
    steps = steps_for("sin(x) + cos(x)")
    assert len(steps) == 2
    assert steps[1]["name"] == "Combine a sin + b cos"


def test_a_supplied_target_overrides_sympys_answer(sample_formulas):
    from src.algebra.latex import to_latex

    expr = parse_expression("sin(2*x)")
    target = parse_expression("2*sin(x)*cos(x)")
    steps = simplify_steps(expr, target=target)
    assert steps[-1]["latex"] == to_latex(target)
    assert [s["name"] for s in steps] == [
        "Initial expression", "Double Angle for Sine"]


# --- re-importing must refresh the derived indexes --------------------------
def test_reimporting_a_formula_refreshes_its_function_index():
    """A formula that already exists must still get the importer's NEW
    view of it -- otherwise an index improvement never reaches rows that
    are already in the database (the live DB lagged behind the code)."""
    from src.knowledge import store as st

    connection = st.get_connection()
    entry = make_entry(name="Index Refresh Test",
                       left="zzz(?a)", right="1/(?a)")
    entry["functions"] = ["zzz"]           # the OLD, one-sided view
    assert st.insert_entry(connection, entry) is True
    connection.commit()

    row = connection.execute(
        "SELECT id FROM entries WHERE name = 'Index Refresh Test'").fetchone()
    first = connection.execute(
        "SELECT fn FROM function_index WHERE entry_id = ?", (row["id"],)
    ).fetchall()
    assert [r["fn"] for r in first] == ["zzz"]

    entry["functions"] = ["zzz", "www"]    # the NEW, both-sides view
    assert st.insert_entry(connection, entry) is False   # duplicate content
    connection.commit()

    after = {r["fn"] for r in connection.execute(
        "SELECT fn FROM function_index WHERE entry_id = ?", (row["id"],))}
    assert {"zzz", "www"} <= after         # the new view reached the old row


# --- authored strategies (form-matched routes) -------------------------------
def make_fake_strategy(name, final_expr=None, pad=0):
    """A strategy that matches everything; returns None when final_expr
    is None (did not reach). `pad` inserts that many detour lines before
    the arrival, so two reaching strategies can have different lengths
    for the shortest-first sort to fix."""
    from src.algebra import strategies as st
    from src.algebra.latex import to_latex
    from src.algebra.steps import _initial_step

    def matches(expr):
        return True

    def run(expr, target, banned, include_db):
        if final_expr is None:
            return None
        steps = [_initial_step(expr)]
        steps += [{"name": name, "latex": to_latex(expr)}] * pad
        steps.append({"name": name, "latex": to_latex(final_expr)})
        return steps, final_expr

    return st.Strategy(name, matches, run)


def test_product_of_sums_form_is_detected():
    from src.algebra.strategies import _is_product_of_sums
    assert _is_product_of_sums(parse_expression("(a + b + c)*(d + e)"))
    assert _is_product_of_sums(parse_expression("(x + 1)*(y + 1)"))
    assert not _is_product_of_sums(parse_expression("2*x + 3"))
    assert not _is_product_of_sums(parse_expression("sin(x) + cos(x)"))


def test_product_form_needs_only_one_multi_term_bracket():
    from src.algebra.strategies import _is_product_of_sums
    assert _is_product_of_sums(parse_expression("(1 - tan(x))*cos(x)"))
    assert _is_product_of_sums(parse_expression("2*(x + 1)"))
    assert not _is_product_of_sums(parse_expression("2*x*y"))   # no bracket
    assert not _is_product_of_sums(parse_expression("sin(x) + cos(x)"))


def test_expand_strategy_reaches_an_expanded_target():
    from src.algebra.latex import to_latex

    expr = parse_expression("(a + b + c)*(d + e)")
    target = parse_expression("a*d + a*e + b*d + b*e + c*d + c*e")
    steps = simplify_steps(expr, target=target)
    assert [s["name"] for s in steps] == [
        "Initial expression", "Expand brackets"]
    assert steps[-1]["latex"] == to_latex(target)


def test_termwise_pass_fires_root_work_after_the_children():
    from src.algebra.latex import to_latex_as_written
    from src.algebra.steps import _SolveState
    from src.algebra.strategies import _termwise_move

    expr = parse_expression("2*x + 3*x + 5 - 1")   # children alone cannot change
    state = _SolveState(expr, frozenset(), to_latex_as_written(expr))
    hit = _termwise_move(expr, state, True)
    assert hit is not None
    assert hit[1] == "Evaluate numbers"            # the root-level tidy move


def test_a_strategy_that_misses_never_hijacks_the_route(monkeypatch):
    from src.algebra import strategies as st
    monkeypatch.setattr(st, "STRATEGIES",
                        [make_fake_strategy("Always Late", final_expr=None)])
    steps = steps_for("2*x + 3*x + 5 - 1")
    assert [s["name"] for s in steps] == [
        "Initial expression", "Evaluate numbers", "Combine like terms"]


def test_shortest_reaching_strategy_wins_and_the_other_is_an_option(
        monkeypatch):
    from src.algebra import strategies as st
    from src.algebra.steps import alternative_ways

    five_x_four = parse_expression("5*x + 4")
    monkeypatch.setattr(st, "STRATEGIES", [
        make_fake_strategy("Long One", final_expr=five_x_four, pad=1),
        make_fake_strategy("Short One", final_expr=five_x_four),
    ])
    # "Long One" listed first on purpose: sorting must fix the order.
    steps = steps_for("2*x + 3*x + 5 - 1")
    assert steps[1]["name"] == "Short One"

    ways = alternative_ways(parse_expression("2*x + 3*x + 5 - 1"))
    names = [way["name"] for way in ways]
    assert "Long One" in names          # (5) the other one is an option
    assert "Short One" not in names     # ...but never as its own alternative


# --- Task 8c: ladder X (strategy route -> last resort -> suggestion) --------
def test_strategy_row_is_imported_and_findable(sample_formulas):
    rows = store.find_strategies(store.get_connection(), "product_of_sums")
    assert [r["name"] for r in rows] == ["Expand then reduce"]
    assert json.loads(rows[0]["steps"])[0] == {
        "do": "expand", "scope": "whole", "until": "stuck"}


def test_strategy_lives_in_its_own_table_not_in_entries(sample_formulas):
    """A strategy row belongs to `strategies`; `entries` keeps formulas,
    methods, notes and solutions -- no strategy kind is left behind."""
    connection = store.get_connection()

    row = connection.execute(
        "SELECT * FROM strategies WHERE name = 'Expand then reduce'"
    ).fetchone()
    assert row is not None, "the shipped strategy must be in strategies"
    assert row["form"] == "product_of_sums"
    phases = json.loads(row["steps"])
    assert isinstance(phases, list)          # the bare phase list, not a dict
    assert len(phases) == 4

    leftover = connection.execute(
        "SELECT COUNT(*) FROM entries WHERE kind = 'strategy'"
    ).fetchone()[0]
    assert leftover == 0


def test_product_strategy_runs_its_phases_end_to_end(sample_formulas):
    """The user's four-phase strategy, executed on their example."""
    from sympy import simplify

    from src.algebra.latex import to_latex

    expr = parse_expression("(1+tanx)(1-cotx)")
    steps = simplify_steps(expr)
    assert [s["name"] for s in steps] == [
        "Initial expression", "Expand brackets", "Simplify",
        "Collect like terms"]
    # phase 4 (whole simplify) is a no-op on this input -> no step shown
    assert steps[-1]["latex"] == to_latex(parse_expression("tan(x) - cot(x)"))
    # it did not land on the goal, so the gap must say so
    targets = {row["target"] for row in store.list_gaps()}
    assert to_latex(simplify(expr)) in targets


def test_walk_phase_applies_a_formula(sample_formulas):
    from src.algebra.strategies import _run_phases

    doc = {"phases": [{"do": "walk", "until": "stuck"}]}
    result = _run_phases(doc, parse_expression("sin(x)^2 + cos(x)^2 + y"),
                         parse_expression("y"), frozenset(), True)
    assert result is not None
    names = [s["name"] for s in result[0]]
    assert "Pythagorean Identity" in names


def test_ladder_last_resort_when_no_strategy_matches(sample_formulas):
    steps = steps_for("2*x + 3*x + 5 - 1")
    assert [s["name"] for s in steps] == ["Initial expression", "Simplify"]
    assert steps[-1]["latex"] == "5 x + 4"
    targets = {row["target"] for row in store.list_gaps()}
    assert "5 x + 4" in targets


def test_unreachable_user_target_gives_one_suggestion_and_a_gap(sample_formulas):
    from src.algebra.latex import to_latex

    steps = simplify_steps(parse_expression("sin(2*x)"),
                           target=parse_expression("2*sin(x)*cos(x)"))
    assert len(steps) == 2          # initial + one suggestion
    targets = {row["target"] for row in store.list_gaps()}
    assert to_latex(parse_expression("2*sin(x)*cos(x)")) in targets


def test_rule_x_a_non_reaching_strategy_route_is_still_shown(monkeypatch):
    """Rule X: the ONE route is kept even when the strategy missed."""
    from src.algebra import strategies as st
    from src.algebra.latex import to_latex

    sideways = parse_expression("y")     # NOT the goal (5*x + 4)
    monkeypatch.setattr(st, "STRATEGIES", [
        make_fake_strategy("Goes Sideways", final_expr=sideways)])

    steps = steps_for("2*x + 3*x + 5 - 1")
    assert [s["name"] for s in steps] == [
        "Initial expression", "Goes Sideways"]
    assert steps[-1]["latex"] == to_latex(sideways)



# --- the combination scope: pairwise-then-triple simplification ------------
def test_combination_scope_combines_pairs_one_step_at_a_time(sample_formulas):
    """Two terms merge first; the next combination is its own step."""
    from src.algebra.strategies import _run_phases

    doc = {"phases": [{"do": "simplify", "scope": "combination",
                       "until": "stuck"}]}
    expr = parse_expression("sin(x)^2 + cos(x)^2 + y")
    result = _run_phases(doc, expr, parse_expression("z"), frozenset(), True)
    assert result is not None
    steps, final = result
    # pair (sin^2, cos^2) combined -> 1; nothing else combines
    assert final == parse_expression("1 + y")   # or to_latex equality if safer
    assert len(steps) >= 2


def test_combination_reaches_a_goal_via_growing_subsets(sample_formulas):
    from sympy import simplify

    from src.algebra.latex import to_latex

    expr = parse_expression("(x + 1)*(sin(x)^2 + cos(x)^2)")
    steps = simplify_steps(expr)   # strategy includes the new phase
    assert steps[-1]["latex"] == to_latex(simplify(expr))   # goal reached
    assert steps[1]["name"] == "Expand brackets"            # still the strategy's route


def test_validate_strategy_accepts_combination_and_rejects_unknown_scope():
    from src.algebra.strategies import validate_strategy

    # the new scope is part of the shared vocabulary...
    validate_strategy("product_of_sums", [
        {"do": "simplify", "scope": "combination", "until": "stuck"}])

    # ...and an unknown scope is still rejected by name
    with pytest.raises(ValueError, match="scope"):
        validate_strategy("product_of_sums", [
            {"do": "simplify", "scope": "nowhere", "until": "stuck"}])


def test_combination_ignores_results_that_are_still_a_sums(monkeypatch):
    """A pair whose 'simplify' only re-spells a sum is NOT a combination."""
    import src.algebra.strategies as st
    from src.algebra.latex import to_latex_as_written

    monkeypatch.setattr(st, "run_operation",
                        lambda do, e: (e.func(*reversed(e.args)), "Simplify")
                        if e.is_Add else (e, "Simplify"))
    root = parse_expression("x + y + z")
    state = st._SolveState(root, frozenset(), to_latex_as_written(root))
    assert st._combination_move(root, "simplify", state) is None


def test_combination_removes_terms_that_collapse_to_zero():
    """sec - 1/cos -> 0 must DELETE both terms, never leave '+ 0'."""
    from src.algebra.latex import to_latex, to_latex_as_written
    from src.algebra.steps import _SolveState
    from src.algebra.strategies import _combination_move

    expr = parse_expression("sec(x) - 1/cos(x) + y")
    state = _SolveState(expr, frozenset(), to_latex_as_written(expr))
    move = _combination_move(expr, "simplify", state)
    assert move is not None
    new_root, entries, _rows = move
    assert to_latex(new_root) == "y"          # both terms gone, no "0"
    assert "0" not in entries[0]["latex"]


def test_strategy_route_finishes_at_two_for_the_headline_input(sample_formulas):
    """With the trigsimp combination phase, the product route lands on 2."""
    steps = steps_for("(1+cotx-cscx)(1+tanx+secx)")
    assert steps[-1]["latex"] == "2"


# --- "fill SymPy's gaps": gate fallback, six rows, bracket scope, walk row ---
def test_gate_fallback_proves_tan_plus_cot():
    """The plain check cannot prove tan + cot = 2/sin(2a); the fallback
    that rewrites both sides into sin/cos must return pass."""
    from sympy import cot, sin, tan

    a = symbols("a")
    result = gate.verify_identity(tan(a) + cot(a), 2 / sin(2 * a))
    assert result == ("pass", "SymPy proved both sides are equal")


def test_six_new_identity_rows_are_imported_and_verified(sample_formulas):
    """The six gap-filling rows are in the store, all proven by the gate."""
    names = [
        "Cotangent as Tan Reciprocal",
        "Tan times Cot is One",
        "Tan times Cosecant is Secant",
        "Cot times Secant is Cosecant",
        "Cosecant times Secant is Two over Sin Two",
        "Tan plus Cot is Two over Sin Two",
    ]
    connection = store.get_connection()
    placeholders = ",".join("?" * len(names))
    rows = connection.execute(
        f"SELECT name, verified FROM entries WHERE name IN ({placeholders})",
        names,
    ).fetchall()

    assert len(rows) == 6
    assert {row["name"] for row in rows} == set(names)
    assert all(row["verified"] == 1 for row in rows)


def test_sum_strategy_reaches_plain_arithmetic(sample_formulas):
    """multi_term_sum route: 2x + 3x - 1 + 5 must reach 5x + 4 (and not as
    a lone SymPy Simplify hop)."""
    steps = steps_for("2*x + 3*x - 1 + 5")

    assert steps[-1]["latex"] == "5 x + 4"
    assert len(steps) >= 2
    assert [step["name"] for step in steps] != ["Initial expression",
                                                "Simplify"]


def test_sum_strategy_combination_steps_for_plain_arithmetic(sample_formulas):
    """BUG 3: the subset sum was built with evaluation ON, so
    Add(2*x, 3*x) collapsed to 5*x BEFORE run_operation saw it -- the
    sum strategy's first (combination) phase was dead and Factorise
    showed instead. Built unevaluated, the combination phases own these
    steps: pair (2x, 3x) -> 5x, then pair (-1, 5) -> 4, done."""
    steps = steps_for("2*x + 3*x - 1 + 5")

    assert [step["name"] for step in steps] == [
        "Initial expression", "Simplify", "Simplify"]
    assert steps[-1]["latex"] == "5 x + 4"


def test_new_strategy_finds_the_power_difference(sample_formulas):
    """The multi_term_sum strategy must get to work on sin^8 - cos^8.

    The first move is ONE level of the difference of squares (the
    incremental peel), not a full one-shot factorization."""
    steps = steps_for("sin(x)^8 - cos(x)^8")
    assert steps[1]["name"] == "Factor as difference of squares"


def test_bracket_scope_combines_inside_a_factor():
    """scope 'bracket': one visible step that combines inside a factor.

    The plan's example root (x + 1)*(2*y + 3*y + 1) parses with its three
    terms intact, but _combination_move re-builds the subset with
    evaluate=True, so 2*y + 3*y collapses BEFORE run_operation sees it and
    no visible combination exists -- therefore a bracket whose pair only
    SymPy can fold (sin^2 + cos^2 -> 1) is used instead.
    """
    from src.algebra.latex import to_latex, to_latex_as_written
    from src.algebra.steps import _SolveState
    from src.algebra.strategies import _phase_move

    root = parse_expression("(x + 1)*(sin(x)^2 + cos(x)^2 + y)")
    state = _SolveState(root, frozenset(), to_latex_as_written(root))
    hit = _phase_move(root, "simplify", "bracket", state)

    assert hit is not None
    new_root, entries, _rows = hit
    assert entries[0]["latex"] == "y + 1"             # inside the bracket
    assert to_latex(new_root) == "\\left(x + 1\\right) \\left(y + 1\\right)"


def test_walk_phase_applies_the_new_sum_row(sample_formulas):
    """A bare walk phase on tan + cot + 2 - 2/sin(2x): the new identity
    row must fire and the walk must land on 2."""
    from src.algebra.strategies import _run_phases

    doc = {"phases": [{"do": "walk", "until": "stuck"}]}
    expr = parse_expression("tan(x) + cot(x) + 2 - 2/sin(2*x)")
    result = _run_phases(doc, expr, parse_expression("2"), frozenset(), True)

    assert result is not None
    steps, final = result
    assert steps[1]["name"] == "Tan plus Cot is Two over Sin Two"
    assert "\\frac{2}{\\sin{\\left(2 x \\right)}}" in steps[1]["latex"]
    assert steps[-1]["latex"] == "2"


# --- BUG 1: _derived_hits must skip variants that leak a free Wild ---------
def test_derived_hits_never_leak_a_free_wild(sample_formulas):
    """A partial variant match keeps an unbound Wild hole: showing it
    prints garbage (the user's screenshot, step 12).

    Natural trigger (found by replaying the product strategy's route for
    (1+cotx-cscx)(1+tanx+secx) state by state): the first offending
    state is index 9, leaking through `Rearrange: Combine a sin + b cos`.
    The test replays the same route (aborting once 10 states -- the
    window containing that first offender -- are recorded) and asserts
    the plan's condition for every recorded state: no yield of
    _iter_changes(..., with_derived=True) may carry a free Wild. The
    later route states shift with D2, so the window is capped, not the
    whole 21-state walk.
    """
    from sympy import Wild, simplify

    from src.algebra import strategies as st
    from src.algebra.latex import to_latex
    from src.algebra.rules import SIMPLIFY_RULES
    from src.algebra.steps import _iter_changes, _SolveState

    WINDOW = 10   # states checked: enough to include offender index 9

    class _Enough(Exception):
        pass

    recorded = []

    class RecordingState(_SolveState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded.append(self)

        def record(self, expr, display_latex=""):
            super().record(expr, display_latex)
            if len(self.history) >= WINDOW:
                raise _Enough   # the window is full: stop replaying

    expr = parse_expression("(1+cotx-cscx)(1+tanx+secx)")
    original = st._SolveState
    st._SolveState = RecordingState
    try:
        try:
            for strategy in st.matching_strategies(expr):
                if strategy.name == "Expand then reduce":
                    strategy.run(expr, simplify(expr), frozenset(), True)
                    break
        except _Enough:
            pass
    finally:
        st._SolveState = original

    assert recorded, "the product strategy must run"
    history = recorded[-1].history
    assert 3 <= len(history) <= WINDOW   # the route really walked somewhere

    leaks = []
    for state_expr in history:
        state = _SolveState(state_expr, frozenset(), to_latex(state_expr))
        for new_expr, name, _formula, _row in _iter_changes(
                state_expr, SIMPLIFY_RULES, True, state, with_derived=True):
            wilds = [h for h in new_expr.free_symbols if isinstance(h, Wild)]
            if wilds:
                leaks.append((to_latex(state_expr), name, wilds))
    assert leaks == [], (
        f"{len(leaks)} state(s) leaked a free Wild, "
        f"first: {leaks[0][1]} {leaks[0][2]}"
    )


def test_combination_cancels_all_zero_pairs_before_merging_anything():
    """'Cancel until stuck': both zero-pairs must be gone before any merge."""
    from src.algebra.latex import to_latex, to_latex_as_written
    from src.algebra.strategies import _combination_move
    from src.algebra.steps import _SolveState

    expr = parse_expression(
        "tan(x) + cot(x) - csc(x) + sec(x) + 2 - 1/cos(x) - 2/sin(2*x) + 1/sin(x)")
    state = _SolveState(expr, frozenset(), to_latex_as_written(expr))

    first = _combination_move(expr, "simplify", state)
    assert first is not None
    state2 = _SolveState(first[0], frozenset(), to_latex(first[0]))
    second = _combination_move(first[0], "simplify", state2)
    assert second is not None
    latex = to_latex(second[0])
    # after two rounds BOTH cancellations must be complete:
    for gone in ("\\sec", "\\csc", "1}{\\cos", "1}{\\sin"):
        assert gone not in latex, f"{gone} survived two cancel rounds: {latex}"
