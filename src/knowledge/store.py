"""
store.py -- all the reading and writing of data/knowledge.db.

One table holds everything:

    entries(
        id, kind ('identity' | 'method'), name, statement (LaTeX for show),
        left, right      -- identity: the formula itself
        steps            -- method: its steps, ALL inside this one row
        signature        -- shape of the left side   (how we find it fast)
        applies_to       -- method: shape of its first step, or 'any'
        verified         -- 1 = trustworthy (passed the gate / approved)
        origin, source   -- 'ai' | 'hand', and which batch file it came from
        usage_count      -- how often it fired (we try popular ones first)
        dedup_key        -- stops the same formula being imported twice
    )

Plus a small full-text table for the search box (name + statement only).

The database file is created automatically the first time we need it, so a
missing database is never an error -- there is simply nothing in it yet.
"""

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from sympy import Add, Mul, Expr

from .signature import signature_of

# Where the file lives (data/knowledge.db in the project folder).
# The KNOWLEDGE_DB environment variable overrides it (used by tests).
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,              -- 'identity' | 'method'
    name        TEXT NOT NULL,
    statement   TEXT NOT NULL,              -- LaTeX shown to people
    left        TEXT,                       -- identity: pattern side
    right       TEXT,                       -- identity: result side
    steps       TEXT,                       -- method: JSON with ALL steps
    signature   TEXT NOT NULL,              -- indexed shape, for finding it
    lead_signature TEXT NOT NULL DEFAULT '',-- for sum/product patterns: the
                                             -- shape of their first part, so
                                             -- we can find a formula hiding
                                             -- inside a bigger sum
    applies_to  TEXT NOT NULL DEFAULT '',   -- method: first-step shape or 'any'
    verified    INTEGER NOT NULL DEFAULT 0, -- 1 = trusted
    origin      TEXT NOT NULL DEFAULT 'ai', -- 'ai' | 'hand'
    source      TEXT,                       -- batch file name
    usage_count INTEGER NOT NULL DEFAULT 0,
    dedup_key   TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entries_lookup
    ON entries(kind, verified, signature, usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_entries_lead
    ON entries(kind, verified, lead_signature, usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_entries_methods
    ON entries(kind, verified, applies_to, usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_entries_review
    ON entries(verified, id);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
    USING fts5(name, statement);
"""


def db_path() -> Path:
    """Path of the database file for this process."""
    return Path(os.environ.get("KNOWLEDGE_DB", str(DEFAULT_DB_PATH)))


def connect() -> sqlite3.Connection:
    """Open (and create if needed) the database."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: the web server answers requests on several
    # threads, and we share one connection (see get_connection).
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(SCHEMA)
    return connection


# One open connection per database file, so a normal solve does not have to
# re-open the file for every little lookup.
_CONNECTIONS: dict = {}


def get_connection() -> sqlite3.Connection:
    """A shared connection to the current database file."""
    key = str(db_path())
    if key not in _CONNECTIONS:
        _CONNECTIONS[key] = connect()
    return _CONNECTIONS[key]


def dedup_key(kind: str, left: Optional[str], right: Optional[str],
              steps: Optional[str]) -> str:
    """A fingerprint of a formula's content, so imports can skip duplicates."""
    payload = f"{kind}|{left or ''}|{right or ''}|{steps or ''}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def insert_entry(connection: sqlite3.Connection, entry: dict) -> bool:
    """Add one entry. Returns False when we already have it (duplicate)."""
    try:
        cursor = connection.execute(
            """
            INSERT INTO entries
                (kind, name, statement, left, right, steps, signature,
                 lead_signature, applies_to, verified, origin, source, dedup_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["kind"], entry["name"], entry["statement"],
                entry.get("left"), entry.get("right"), entry.get("steps"),
                entry["signature"], entry.get("lead_signature", ""),
                entry.get("applies_to", ""),
                int(entry.get("verified", 0)), entry.get("origin", "ai"),
                entry.get("source"), entry["dedup_key"],
            ),
        )
    except sqlite3.IntegrityError:
        return False  # already have this exact formula

    connection.execute(
        "INSERT INTO entries_fts(rowid, name, statement) VALUES (?, ?, ?)",
        (cursor.lastrowid, entry["name"], entry["statement"]),
    )
    return True


def find_identities(connection: sqlite3.Connection, expr: Expr,
                    limit: int = 50) -> List[sqlite3.Row]:
    """Trusted identities that could apply to this expression.

    Two lookups, cheapest first:

        1. exact shape --  sin(?a)^2 + cos(?a)^2  against exactly that
        2. only when the node is a sum/product and step 1 found nothing:
           look for formulas whose FIRST part has the shape of one of our
           parts.  That is how "sin^2 + cos^2 = 1" gets found inside the
           bigger sum  "sin(z)^2 + cos(z)^2 + 4"  (the matcher then proves
           the rest).

    Ordered most-used first, so the popular formulas get tried before the
    long tail of 10 million AI-generated ones.
    """
    shape = signature_of(expr)

    rows = connection.execute(
        """
        SELECT * FROM entries
        WHERE kind = 'identity' AND verified = 1 AND signature = ?
        ORDER BY usage_count DESC, id
        LIMIT ?
        """,
        (shape, limit),
    ).fetchall()
    if rows or not isinstance(expr, (Add, Mul)):
        return rows

    part_shapes = [signature_of(part) for part in expr.args]
    placeholders = ",".join("?" * len(part_shapes))

    return connection.execute(
        f"""
        SELECT * FROM entries
        WHERE kind = 'identity' AND verified = 1
          AND lead_signature IN ({placeholders})
          AND signature != ?
        ORDER BY usage_count DESC, id
        LIMIT ?
        """,
        (*part_shapes, shape, limit),
    ).fetchall()


def find_methods(connection: sqlite3.Connection, expr: Expr,
                 limit: int = 20) -> List[sqlite3.Row]:
    """Trusted methods that could start on this expression.

    A method is offered when
        * its first step is a plain operation (like "expand") -> always
        * its first step is a pattern whose WHOLE shape matches
        * its first step is a pattern whose FIRST part matches one of our
          parts (that is how "use the difference of squares" gets offered
          for  x^2 - 4, whose shape is not identical to ?a^2 - ?b^2)
    """
    shape = signature_of(expr)

    params = [shape]
    lead_clause = ""
    if isinstance(expr, (Add, Mul)):
        part_shapes = [signature_of(part) for part in expr.args]
        placeholders = ",".join("?" * len(part_shapes))
        lead_clause = f" OR lead_signature IN ({placeholders})"
        params.extend(part_shapes)

    return connection.execute(
        f"""
        SELECT * FROM entries
        WHERE kind = 'method' AND verified = 1
          AND (applies_to = ? OR applies_to = 'any'{lead_clause})
        ORDER BY (applies_to = ?) DESC, usage_count DESC, id
        LIMIT ?
        """,
        (*params, shape, limit),
    ).fetchall()


def search(connection: sqlite3.Connection, query: str = "",
           verified: Optional[int] = None, limit: int = 50) -> List[sqlite3.Row]:
    """Search box: full-text on name/statement, or browse when empty.

    Anything that the full-text engine cannot read (users like typing
    things such as "x^2") falls back to a simple contains-search, so the
    box never throws an error at the user.
    """
    if query:
        # Full-text search matches whole words, so we make every word a
        # prefix: typing "square" then also finds "Difference of Squares".
        # Anything the full-text engine cannot read (a query like "x^2")
        # simply falls through to the contains-search below.
        words = re.findall(r"[A-Za-z0-9_]+", query)
        fts_query = " ".join(f"{word}*" for word in words)

        rows = []
        if fts_query:
            try:
                rows = connection.execute(
                    """
                    SELECT e.* FROM entries_fts f
                    JOIN entries e ON e.id = f.rowid
                    WHERE entries_fts MATCH ?
                      AND (? IS NULL OR e.verified = ?)
                    ORDER BY e.usage_count DESC, e.id
                    LIMIT ?
                    """,
                    (fts_query, verified, verified, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            rows = connection.execute(
                """
                SELECT * FROM entries
                WHERE (name LIKE ? OR statement LIKE ?)
                  AND (? IS NULL OR verified = ?)
                ORDER BY usage_count DESC, id
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", verified, verified, limit),
            ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT * FROM entries
            WHERE (? IS NULL OR verified = ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (verified, verified, limit),
        ).fetchall()
    return rows


def get_entry(connection: sqlite3.Connection, entry_id: int) -> Optional[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM entries WHERE id = ?", (entry_id,)
    ).fetchone()


def set_verified(connection: sqlite3.Connection, entry_id: int, verified: int) -> None:
    """Approve (1) or reject-by-unapproving (0) an entry."""
    connection.execute(
        "UPDATE entries SET verified = ? WHERE id = ?", (int(verified), entry_id)
    )


def delete_entry(connection: sqlite3.Connection, entry_id: int) -> None:
    """Throw an entry away (reject it)."""
    connection.execute("DELETE FROM entries_fts WHERE rowid = ?", (entry_id,))
    connection.execute("DELETE FROM entries WHERE id = ?", (entry_id,))


def bump_usage(connection: sqlite3.Connection, entry_id: int) -> None:
    """Count that this formula actually fired, so it sorts higher next time."""
    connection.execute(
        "UPDATE entries SET usage_count = usage_count + 1 WHERE id = ?",
        (entry_id,),
    )


def counts(connection: sqlite3.Connection) -> dict:
    """Handy summary for the import report / UI."""
    row = connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(verified = 1) AS verified,
               SUM(verified = 0) AS needs_review
        FROM entries
        """
    ).fetchone()
    return {
        "total": row["total"] or 0,
        "verified": row["verified"] or 0,
        "needs_review": row["needs_review"] or 0,
    }
