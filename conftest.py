"""
pytest finds this file automatically.

It does three jobs:

1. Puts the project folder on Python's import path, so tests can simply
   write:  from src.algebra.rules import eval_numbers

2. Points KNOWLEDGE_DB at a throwaway file, so tests never touch the real
   data/knowledge.db. Each test session gets its own empty database.

3. Holds the shared test helpers (steps_for, load_importer, make_entry,
   seed_entry) so they are written once instead of per test file.
"""

import os

import pytest


# --- shared helpers (import with: from conftest import steps_for) ----------
def steps_for(text: str) -> list:
    """Expression text -> the steps the app would show."""
    from src.algebra.parser import parse_expression
    from src.algebra.steps import simplify_steps
    return simplify_steps(parse_expression(text))


def load_importer():
    """Load scripts/import_formulas.py as a module we can call."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "import_formulas",
        Path(__file__).resolve().parent / "scripts" / "import_formulas.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_entry(**overrides) -> dict:
    """A ready-made database row (an identity by default).

    left/right default to something derived from the NAME, so entries from
    different tests never collide on the content-based dedup key.
    """
    import re

    name = overrides.get("name", "Test formula")
    slug = re.sub(r"[^a-z0-9]", "", name.lower()) or "entry"

    entry = {
        "kind": "identity",
        "name": name,
        "statement": "",
        "left": overrides.get("left", f"{slug} + 0"),
        "right": overrides.get("right", slug),
        "signature": "Add(_,_)",
        "verified": 1,
        "origin": "hand",
        "source": "test",
    }
    entry.update(overrides)

    from src.knowledge import store
    entry["dedup_key"] = store.dedup_key(
        entry["kind"], entry.get("left"), entry.get("right"), entry.get("steps")
    )
    return entry


def seed_entry(**overrides) -> int:
    """Insert one entry into the (test) database; returns its id."""
    from src.knowledge import store

    connection = store.get_connection()
    store.insert_entry(connection, make_entry(**overrides))
    connection.commit()

    row = connection.execute(
        "SELECT id FROM entries WHERE name = ?",
        (overrides.get("name", "Test formula"),),
    ).fetchone()
    return row["id"]


@pytest.fixture(scope="session", autouse=True)
def knowledge_db(tmp_path_factory):
    """Give the whole test session its own formula database."""
    path = tmp_path_factory.mktemp("knowledge") / "knowledge.db"
    os.environ["KNOWLEDGE_DB"] = str(path)
    yield path
    os.environ.pop("KNOWLEDGE_DB", None)


@pytest.fixture(scope="session")
def sample_formulas(knowledge_db):
    """Load the shipped formula batches (data/samples/*.jsonl) once.

    Chain tests need the real formulas (Pythagorean, Double Angle, ...);
    this also exercises the importer and its SymPy proof gate for real.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "import_formulas", root / "scripts" / "import_formulas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for batch in sorted((root / "data" / "samples").glob("*.jsonl")):
        # Never re-import a reject file (algebra.rejected.jsonl etc.) --
        # its stem would grow ".rejected" once per session.
        if ".rejected." in batch.name:
            continue
        summary = module.import_file(batch)
        assert summary["rejected"] == 0, f"rejected rows in {batch.name}"


@pytest.fixture
def seeded_sine_formulas(sample_formulas):
    """The three sine formulas the alternative-route tests need are already
    in the shipped batches -- this fixture just guarantees they are present
    (duplicates are skipped by content, so re-adding is harmless)."""
    rows = [
        ("Sin plus Sin", "sin(?c) + sin(?d)",
         "2*sin((?c + ?d)/2)*cos((?c - ?d)/2)"),
        ("Sine of A + B", "sin(?a + ?b)",
         "sin(?a)*cos(?b) + cos(?a)*sin(?b)"),
        ("Sine of A - B", "sin(?a - ?b)",
         "sin(?a)*cos(?b) - cos(?a)*sin(?b)"),
    ]
    for name, left_text, right_text in rows:
        seed_entry(name=name, left=left_text, right=right_text)
