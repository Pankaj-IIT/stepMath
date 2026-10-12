"""
Tests for the formula-database endpoints (browse, review, methods).

Run with:   pytest
"""

from fastapi.testclient import TestClient

from api.main import app
from conftest import seed_entry
from src.knowledge import store

client = TestClient(app)


def _seed(**overrides):
    """Put one entry into the (test) database and return its id.

    Content (left/right/steps) is unique per name, because duplicates are
    skipped on purpose -- see store.insert_entry. The building of the row
    itself lives in conftest.make_entry, shared with the other test files.
    """
    return seed_entry(**{"statement": "x = x", **overrides})


def test_entries_endpoint_returns_list_and_counts():
    _seed(name="Browse Me")
    response = client.get("/api/entries", params={"q": "browse"})

    assert response.status_code == 200
    body = response.json()
    assert any(entry["name"] == "Browse Me" for entry in body["entries"])
    assert body["counts"]["total"] >= 1


def test_review_queue_endpoint():
    entry_id = _seed(name="Needs An Eye", verified=0)

    response = client.get("/api/entries", params={"verified": 0})
    assert response.status_code == 200
    assert any(entry["id"] == entry_id for entry in response.json()["entries"])


def test_approve_moves_it_out_of_the_queue():
    entry_id = _seed(name="Approve Me", verified=0)

    assert client.post(f"/api/entries/{entry_id}/approve").json() == {"ok": True}
    row = store.get_entry(store.get_connection(), entry_id)
    assert row["verified"] == 1


def test_reject_deletes_it():
    entry_id = _seed(name="Reject Me", verified=0)

    assert client.post(f"/api/entries/{entry_id}/reject").json() == {"ok": True}
    assert store.get_entry(store.get_connection(), entry_id) is None


def test_approving_a_missing_entry_is_a_friendly_error():
    response = client.post("/api/entries/999999/approve")
    assert response.status_code == 400
    assert "No such formula" in response.json()["error"]


def test_applicable_methods_for_an_expression():
    _seed(
        kind="method",
        name="A Method For Squares",
        left=None,
        right=None,
        steps='[{"op": "expand"}]',
        signature="",
        applies_to="any",
        statement="",
    )

    response = client.get("/api/methods/applicable",
                          params={"expression": "2*(x + 3)"})
    assert response.status_code == 200
    assert any(entry["name"] == "A Method For Squares"
               for entry in response.json()["entries"])


def test_running_a_method_returns_its_steps():
    entry_id = _seed(
        kind="method",
        name="Run Me",
        left=None,
        right=None,
        # unique content: the step's own name makes it differ from the
        # shipped "Expand then Collect" batch row (duplicates are skipped)
        steps='[{"op": "expand"}, {"op": "collect", "name": "Gather terms"}]',
        signature="",
        applies_to="any",
        statement="",
    )

    response = client.post("/api/methods/run",
                           json={"expression": "2*(x + 3)", "id": entry_id})

    assert response.status_code == 200
    body = response.json()
    assert body["steps"][0]["name"] == "Initial expression"
    assert body["final"] == "2 x + 6"


def test_running_a_method_that_does_not_fit_says_so():
    entry_id = _seed(
        kind="method",
        name="Square Method",
        left=None,
        right=None,
        steps='[{"left": "?a^2 - ?b^2", "right": "(?a - ?b)*(?a + ?b)", "name": "Difference of Squares"}]',
        signature="",
        applies_to="any",
        statement="",
    )

    response = client.post("/api/methods/run",
                           json={"expression": "x + 1", "id": entry_id})

    assert response.status_code == 400
    assert "Method stopped at step 1" in response.json()["error"]


def test_running_an_identity_as_a_method_is_refused():
    entry_id = _seed(name="Not A Method")

    response = client.post("/api/methods/run",
                           json={"expression": "x", "id": entry_id})

    assert response.status_code == 400
    assert "not a method" in response.json()["error"]


def test_method_starting_with_a_pattern_is_offered_by_shape():
    """x^2 - 4 has a slightly different shape than ?a^2 - ?b^2 (the 4 is a
    number, not a written-out power), and must still be offered the method."""
    from src.algebra.parser import parse_expression
    from src.knowledge.signature import signature_of

    _seed(
        kind="method",
        name="Squares Method",
        left=None,
        right=None,
        steps='[{"left": "?a^2 - ?b^2", "right": "(?a - ?b)*(?a + ?b)", "name": "Difference"}]',
        signature="",
        applies_to=signature_of(parse_expression("x^2 - y^2")),
        lead_signature=signature_of(parse_expression("x^2")),
        statement="",
    )

    response = client.get("/api/methods/applicable",
                          params={"expression": "x^2 - 4"})

    assert response.status_code == 200
    names = [entry["name"] for entry in response.json()["entries"]]
    assert "Squares Method" in names
