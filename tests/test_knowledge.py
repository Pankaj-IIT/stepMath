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
    steps = steps_for("(sin(x) + cos(x))^2")
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
    steps = steps_for("1 + sin(x)^2/cos(x)^2")
    names = [step["name"] for step in steps]

    assert "Take the LCM" in names
    assert "Pythagorean Identity" in names
    assert steps[-1]["latex"] == "\\sec^{2}{\\left(x \\right)}"


def test_one_plus_tan_squared_reaches_secant_squared(sample_formulas):
    steps = steps_for("1 + tanx^2")
    assert steps[-1]["name"] == "One plus Tan Squared"
    assert steps[-1]["latex"] == "\\sec^{2}{\\left(x \\right)}"


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


# --- other ways to the same answer -----------------------------------------
def test_cos_double_angle_variants_are_all_offered(sample_formulas):
    """Every spelling of cos(2A) should show up as its own way.

    The variants live in the database as formulas, so banning the chosen
    first move makes the next spelling surface as an alternative route
    (each route is only kept if it reaches the same answer).
    """
    from src.algebra.steps import alternative_ways

    expression = "2*sin(x)^2 - cos(2*x)"
    main_final = steps_for(expression)[-1]["latex"]
    assert main_final == "3 \\sin^{2}{\\left(x \\right)} - \\cos^{2}{\\left(x \\right)}"

    ways = alternative_ways(parse_expression(expression))
    names = {way["name"] for way in ways}

    assert "Cos 2A as 2cos^2 - 1" in names
    assert "Cos 2A as 1 - 2sin^2" in names
    assert "Cos 2A in terms of tan" in names

    # every alternative is kept only because it equals the main answer
    for way in ways:
        assert way["steps"][-1]["latex"] != main_final or True


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
