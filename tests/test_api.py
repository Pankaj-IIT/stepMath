"""
End-to-end tests: post a request to the web API the way React does.

Run with:   pytest
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simplify_endpoint():
    response = client.post(
        "/api/solve",
        json={"expression": "2*x + 3*x + 5 - 1", "mode": "simplify"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["final"] == "5 x + 4"
    assert body["steps"][0]["name"] == "Initial expression"
    assert len(body["steps"]) == 3


def test_solve_endpoint_shows_the_moves():
    response = client.post(
        "/api/solve",
        json={"expression": "2*x + 6 = 14", "mode": "solve"},
    )
    assert response.status_code == 200
    names = [step["name"] for step in response.json()["steps"]]
    assert names == [
        "Initial equation",
        "Subtract 6 from both sides",
        "Divide both sides by 2",
    ]
    assert response.json()["final"] == "x = 4"


def test_bad_input_returns_a_friendly_error():
    response = client.post(
        "/api/solve",
        json={"expression": "2 + @", "mode": "simplify"},
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_non_linear_equation_is_refused_clearly():
    response = client.post(
        "/api/solve",
        json={"expression": "x^2 = 16", "mode": "solve"},
    )
    assert response.status_code == 400
    assert "power of 1" in response.json()["error"]


def test_simplify_returns_other_ways_to_the_same_answer(seeded_sine_formulas):
    """The UI shows alternative routes; the API must supply them."""
    response = client.post(
        "/api/solve",
        json={"expression": "sin(x+y) + sin(x-y)", "mode": "simplify"},
    )

    assert response.status_code == 200
    body = response.json()

    names = {way["name"] for way in body["alternatives"]}
    assert "Sine of A + B" in names
    assert "Sine of A - B" in names

    # Every alternative is a complete route of its own (its maths-equality
    # with the main route is proven by the engine test in test_knowledge).
    for way in body["alternatives"]:
        assert way["steps"][0]["name"] == "Initial expression"
        assert len(way["steps"]) >= 2


def test_auxiliary_angle_equation_flows_like_the_notes(seeded_sine_formulas):
    """The picture's procedure, step by step, for 3cos + 4sin = 5."""
    response = client.post(
        "/api/solve",
        json={"expression": "3*cos(x) + 4*sin(x) = 5", "mode": "solve"},
    )

    assert response.status_code == 200
    body = response.json()
    names = [step["name"] for step in body["steps"]]

    assert names == [
        "Initial equation",
        "Divide both sides by sqrt(a^2 + b^2)",
        "Write the left side as sin(theta + phi)",
        "General solution",
        "Subtract atan(3/4) from both sides",
    ]
    assert body["final"].startswith("x =")


def test_no_solution_condition_from_the_notes(seeded_sine_formulas):
    """|c| > sqrt(a^2+b^2) -> the equation has no solution."""
    response = client.post(
        "/api/solve",
        json={"expression": "3*sin(x) + 4*cos(x) = 10", "mode": "solve"},
    )

    assert response.status_code == 200
    steps = response.json()["steps"]
    assert steps[-1]["name"] == "No solution"


def test_minus_form_also_flows_like_the_notes(seeded_sine_formulas):
    """a*sin - b*cos = c takes the same road (atan2 handles the sign)."""
    response = client.post(
        "/api/solve",
        json={"expression": "3*sin(x) - 4*cos(x) = 5", "mode": "solve"},
    )

    assert response.status_code == 200
    names = [step["name"] for step in response.json()["steps"]]

    assert names[1] == "Divide both sides by sqrt(a^2 + b^2)"
    assert names[2] == "Write the left side as sin(theta + phi)"
    assert "General solution" in names
    assert response.json()["final"].startswith("x =")


def test_entries_carry_their_formula_families(seeded_sine_formulas):
    """Every card can show which groups it belongs to."""
    response = client.get("/api/entries", params={"q": "sin plus"})

    assert response.status_code == 200
    matching = response.json()["entries"]
    assert matching, "the Sin plus Sin row should be findable"
    assert any("sum-to-product" in entry["groups"] for entry in matching)


def test_family_filter_returns_only_members(seeded_sine_formulas):
    response = client.get("/api/entries", params={"group": "pythagorean"})

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert entries, "pythagorean family must not be empty"
    for entry in entries:
        assert "pythagorean" in entry["groups"]


def test_groups_endpoint_lists_families_with_counts(seeded_sine_formulas):
    response = client.get("/api/groups")

    assert response.status_code == 200
    groups = {group["name"]: group for group in response.json()["groups"]}
    assert "two-angle-sum" in groups
    assert groups["two-angle-sum"]["count"] > 0
    assert groups["two-angle-sum"]["description"]


def test_browse_shows_more_than_the_old_fifty(seeded_sine_formulas):
    """The list used to cap at 50 rows, hiding most of the database."""
    response = client.get("/api/entries", params={"q": ""})
    assert response.status_code == 200
    body = response.json()
    assert len(body["entries"]) > 50
    assert len(body["entries"]) == body["counts"]["total"]


def test_trig_equation_gets_its_general_solution():
    """sin(x) = 0 is answered from the formula database, not by algebra."""
    from src.knowledge import gate, store
    from src.knowledge.patterns import parse_solution
    from src.knowledge.signature import signature_of

    left, right, unknown, value = parse_solution("sin(?t)", "0", "?t = ?n*pi")
    connection = store.get_connection()
    store.insert_entry(connection, {
        "kind": "solution",
        "name": "API test sin theta = 0",
        "statement": "",
        "left": "sin(?t)",
        "right": "0",
        "solution": "?t = ?n*pi",
        "signature": signature_of(left),
        "verified": 1 if gate.verify_solution(left, right, unknown, value)[0] == "pass" else 0,
        "origin": "hand",
        "source": "test",
        "dedup_key": store.dedup_key("solution", "sin(?t)", "0", "?t = ?n*pi"),
    })
    connection.commit()

    response = client.post(
        "/api/solve",
        json={"expression": "sin(x) = 0", "mode": "solve"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["steps"][-1]["name"] == "General solution"
    assert body["final"] == "x = \\pi n"
    # the step also shows the formula it came from, e.g. sin(t) = 0
    assert "\\sin" in body["steps"][-1]["formula"]
