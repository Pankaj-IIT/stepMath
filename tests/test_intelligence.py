"""
Tests for the question layer: reading (what kind of question is this?),
the handlers (what do we do about it?), and the /api/ask endpoint.

Run with:   pytest
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.intelligence.handlers import dispatch
from src.intelligence.reading import NotSupported, read_question

client = TestClient(app)


# --- reading: deterministic direction picking -------------------------------
def test_read_plain_prove():
    reading = read_question("prove that sin(x)^2 + cos(x)^2 = 1")

    assert reading["intent"] == "prove"
    assert reading["lhs"] == "sin(x)^2 + cos(x)^2"
    assert reading["rhs"] == "1"
    assert "prove" in reading["explanation"]


def test_read_show_that_variant():
    reading = read_question("Show that 2 + 2 = 4")

    assert reading["intent"] == "prove"
    assert reading["lhs"] == "2 + 2"


def test_read_conditional_proof():
    reading = read_question(
        "if A + B + C = 180, prove that tanA + tanB + tanC = tanAtanBtanC"
    )

    assert reading["intent"] == "conditional"
    assert reading["given"] == "A + B + C = 180"


def test_read_optimize_question():
    reading = read_question("find the maximum of 3*sin(x) + 4*cos(x)")
    assert reading["intent"] == "optimize"


def test_read_plain_equation():
    reading = read_question("2*x + 6 = 14")
    assert reading["intent"] == "solve"
    assert reading["lhs"] == "2*x + 6"


def test_read_bare_expression_is_simplify():
    reading = read_question("2*x + 3*x - 4")
    assert reading["intent"] == "simplify"


def test_read_prove_without_equals_is_refused():
    with pytest.raises(ValueError):
        read_question("prove that sinx is nice")


# --- handlers ---------------------------------------------------------------
def test_prove_reaches_zero_for_a_true_identity():
    result = dispatch(read_question("prove that sin(x)^2 + cos(x)^2 = 1"))

    assert result["intent"] == "prove"
    assert result["proven"] is True
    assert result["steps"][0]["name"] == "To prove"
    assert "TRUE" in result["steps"][-1]["name"]


def test_prove_reaches_zero_for_a_product_identity():
    result = dispatch(
        read_question("prove that 2*sin(x)*cos(y) = sin(x+y) + sin(x-y)")
    )
    assert result["proven"] is True
    assert result["final"] == "0"


def test_prove_is_honest_when_it_cannot():
    result = dispatch(read_question("prove that 2 + 2 = 5"))

    assert result["proven"] is False
    assert result["steps"][-1]["name"].startswith("Cannot prove")


def test_conditional_and_optimize_are_registered_stubs():
    for question in (
        "if A + B + C = 180, prove that tanA + tanB + tanC = tanAtanBtanC",
        "find the maximum of 3*sin(x) + 4*cos(x)",
    ):
        with pytest.raises(NotSupported) as caught:
            dispatch(read_question(question))
        assert "not built yet" in str(caught.value)


def test_simplify_handler_keeps_the_alternatives(sample_formulas):
    result = dispatch(read_question("sin(x+y) + sin(x-y)"))

    assert result["intent"] == "simplify"
    assert result["final"] == "2 \\sin{\\left(x \\right)} \\cos{\\left(y \\right)}"
    assert any(way["name"] == "Sine of A - B" for way in result["alternatives"])


# --- the endpoint -----------------------------------------------------------
def test_ask_endpoint_proves():
    response = client.post("/api/ask", json={
        "question": "prove that sin(x)^2 + cos(x)^2 = 1",
    })

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "prove"
    assert body["proven"] is True
    assert body["explanation"]


def test_ask_endpoint_simplifies_a_bare_expression():
    response = client.post("/api/ask", json={"question": "2*x + 3*x - 4"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "simplify"
    assert body["final"] == "5 x - 4"


def test_ask_endpoint_explains_unbuilt_directions():
    response = client.post("/api/ask", json={
        "question": "find the maximum of 3*sin(x) + 4*cos(x)",
    })

    assert response.status_code == 400
    assert "not built yet" in response.json()["error"]


def test_ask_endpoint_refuses_unreadable_questions():
    response = client.post("/api/ask", json={"question": "prove that"})

    assert response.status_code == 400
    assert response.json()["error"]
