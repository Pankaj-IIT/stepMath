"""
Tests for the formula database layer: shapes, patterns, the SymPy gate,
the store, methods, and the import script.

Run with:   pytest
"""

import json
from pathlib import Path

import pytest
from sympy import cos, sin, symbols

from src.algebra.parser import parse_expression
from src.algebra.steps import simplify_steps
from src.knowledge import gate, ops, runner, store
from src.knowledge.patterns import apply_identity, parse_pair
from src.knowledge.signature import signature_of

x, y = symbols("x y")


# --- signatures ------------------------------------------------------------
def test_signature_describes_shape_only():
    assert signature_of(parse_expression("x^2 + 3")) == signature_of(
        parse_expression("y^2 + 8")
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
    left, right = parse_pair("?a^2 - ?b^2", "(?a - ?b)*(?a + ?b)")
    assert signature_of(left) == signature_of(parse_expression("x^2 - y^2"))
    assert right is not None


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


# --- the store -------------------------------------------------------------
def _entry(**overrides):
    """A ready-made entry dict for tests.

    left/right default to something unique per name, so entries from
    different tests never collide on the dedup key (which is based on
    content, on purpose).
    """
    import re

    name = overrides.get("name", "Commutative Addition")
    slug = re.sub(r"[^a-z0-9]", "", name.lower()) or "entry"

    entry = {
        "kind": "identity",
        "name": name,
        "statement": "a + b = b + a",
        "left": overrides.get("left", f"{slug} + 0"),
        "right": overrides.get("right", slug),
        "signature": "Add(_,_)",
        "verified": 1,
        "origin": "hand",
        "source": "test",
    }
    entry.update(overrides)
    entry["dedup_key"] = store.dedup_key(
        entry["kind"], entry.get("left"), entry.get("right"), entry.get("steps")
    )
    return entry


def test_insert_find_and_count():
    connection = store.get_connection()
    before = store.counts(connection)["total"]

    assert store.insert_entry(connection, _entry(name="Insert Count")) is True
    connection.commit()

    assert store.counts(connection)["total"] == before + 1


def test_duplicate_content_is_skipped():
    connection = store.get_connection()
    entry = _entry(name="Dup Test")
    assert store.insert_entry(connection, entry) is True
    connection.commit()

    assert store.insert_entry(connection, entry) is False


def test_search_finds_by_name_prefix():
    connection = store.get_connection()
    store.insert_entry(connection, _entry())  # "Commutative Addition"
    connection.commit()

    rows = store.search(connection, "commut")
    assert any(row["name"] == "Commutative Addition" for row in rows)


def test_review_queue_only_lists_unverified():
    connection = store.get_connection()
    store.insert_entry(connection, _entry(name="Unproven Thing", verified=0))
    connection.commit()

    rows = store.search(connection, "", verified=0)
    assert all(row["verified"] == 0 for row in rows)
    assert any(row["name"] == "Unproven Thing" for row in rows)


def test_approve_and_reject():
    connection = store.get_connection()
    store.insert_entry(connection, _entry(name="To Approve", verified=0))
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
    store.insert_entry(connection, _entry(
        name="Pyth Test",
        left="sin(?a)^2 + cos(?a)^2",
        right="1",
        statement="sin^2 + cos^2 = 1",
        signature=signature_of(parse_expression("sin(x)^2 + cos(x)^2")),
        lead_signature=signature_of(parse_expression("sin(x)^2")),
    ))
    connection.commit()

    steps = simplify_steps(parse_expression("sin(t)^2 + cos(t)^2 + 4"))
    names = [step["name"] for step in steps]
    assert "Pyth Test" in names
    assert steps[-1]["latex"] == "5"


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


def test_method_validation_rejects_garbage():
    with pytest.raises(ValueError):
        runner.validate_steps([{"op": "make_coffee"}])

    with pytest.raises(ValueError):
        runner.validate_steps([{"left": "?a + ?b"}])  # no right side


def test_unknown_operation_is_refused():
    with pytest.raises(ValueError):
        ops.run_operation("make_coffee", parse_expression("x"))


# --- the import script -----------------------------------------------------
def test_import_file_counts_and_rejects(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "import_formulas",
        Path(__file__).resolve().parents[1] / "scripts" / "import_formulas.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
