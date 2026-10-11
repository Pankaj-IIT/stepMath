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
