"""
pytest finds this file automatically.

It does two jobs:

1. Puts the project folder on Python's import path, so tests can simply
   write:  from src.algebra.rules import eval_numbers

2. Points KNOWLEDGE_DB at a throwaway file, so tests never touch the real
   data/knowledge.db. Each test session gets its own empty database.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def knowledge_db(tmp_path_factory):
    """Give the whole test session its own formula database."""
    path = tmp_path_factory.mktemp("knowledge") / "knowledge.db"
    os.environ["KNOWLEDGE_DB"] = str(path)
    yield path
    os.environ.pop("KNOWLEDGE_DB", None)
