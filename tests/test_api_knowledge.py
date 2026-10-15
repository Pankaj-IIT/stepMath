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


def test_gaps_endpoint_lists_strategies_we_cannot_reach():
    store.record_gap("x^5 + x^5", "x^5 + x^5", "Combine like terms")

    response = client.get("/api/gaps")

    assert response.status_code == 200
    gaps = response.json()["gaps"]
    assert any(
        gap["target"] == "x^5 + x^5" and "Combine like terms" in gap["tried"]
        for gap in gaps
    )


# --- the Strategies tab: list, vocabulary, create, edit, delete -------------
def test_strategies_endpoint_lists_the_shipped_strategy(sample_formulas):
    response = client.get("/api/strategies")

    assert response.status_code == 200
    strategies = response.json()["strategies"]
    shipped = [s for s in strategies if s["name"] == "Expand then reduce"]
    assert shipped, "the shipped strategy must appear in the list"
    assert shipped[0]["form"] == "product_of_sums"
    assert len(shipped[0]["phases"]) == 4


def test_strategy_meta_describes_the_vocabulary(sample_formulas):
    response = client.get("/api/strategy-meta")

    assert response.status_code == 200
    body = response.json()
    assert "product_of_sums" in body["forms"]
    assert "walk" in body["dos"]
    assert "simplify" in body["dos"]
    assert body["scopes"] == ["whole", "term"]
    assert body["untils"] == ["stuck", "once"]


def test_a_new_strategy_can_be_created_and_deleted(sample_formulas):
    created = client.post("/api/strategies", json={
        "name": "Scratch Walk First",
        "form": "product_of_sums",
        "phases": [{"do": "walk", "scope": "whole", "until": "stuck"}],
    })
    assert created.status_code == 200
    entry_id = created.json()["id"]

    names = [s["name"]
             for s in client.get("/api/strategies").json()["strategies"]]
    assert "Scratch Walk First" in names

    deleted = client.delete(f"/api/strategies/{entry_id}")
    assert deleted.status_code == 200

    names = [s["name"]
             for s in client.get("/api/strategies").json()["strategies"]]
    assert "Scratch Walk First" not in names


def test_post_with_an_invalid_phase_names_the_phase(sample_formulas):
    response = client.post("/api/strategies", json={
        "name": "Bad Phase Strategy",
        "form": "product_of_sums",
        "phases": [{"do": "nope", "scope": "whole", "until": "stuck"}],
    })

    assert response.status_code == 400
    assert "phase 1" in response.json()["error"]


def test_put_updates_a_strategy_without_duplicating_it(sample_formulas):
    created = client.post("/api/strategies", json={
        "name": "Scratch To Rename",
        "form": "product_of_sums",
        "phases": [{"do": "expand", "scope": "whole", "until": "stuck"}],
    })
    assert created.status_code == 200
    entry_id = created.json()["id"]
    before = len(client.get("/api/strategies").json()["strategies"])

    updated = client.put(f"/api/strategies/{entry_id}", json={
        "name": "Scratch Renamed",
        "form": "product_of_sums",
        "phases": [
            {"do": "expand", "scope": "whole", "until": "stuck"},
            {"do": "collect", "scope": "whole", "until": "once"},
        ],
    })
    assert updated.status_code == 200

    after = client.get("/api/strategies").json()["strategies"]
    assert len(after) == before            # updated in place, no duplicate
    mine = [s for s in after if s["id"] == entry_id]
    assert mine and mine[0]["name"] == "Scratch Renamed"
    assert len(mine[0]["phases"]) == 2

    client.delete(f"/api/strategies/{entry_id}")   # leave no scratch behind
