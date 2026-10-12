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

CREATE TABLE IF NOT EXISTS group_members (
    group_name TEXT NOT NULL,   -- a family from classify.py, e.g. two-angle-sum
    entry_id   INTEGER NOT NULL,
    PRIMARY KEY (group_name, entry_id)
);

CREATE INDEX IF NOT EXISTS idx_group_members ON group_members(group_name);

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
    _ensure_columns(connection)
    connection.commit()
    return connection


def _ensure_columns(connection: sqlite3.Connection) -> None:
    """Tiny migration: add columns that newer code expects.

    CREATE TABLE IF NOT EXISTS does nothing on an existing file, so when we
    grow the schema we patch older databases here instead.
    """
    existing = {row[1] for row in connection.execute("PRAGMA table_info(entries)")}

    for name, declaration in (
        ("solution", "TEXT NOT NULL DEFAULT ''"),  # kind='solution' rows
        ("panel", "INTEGER NOT NULL DEFAULT 1"),   # 0 = hide from the buttons
    ):
        if name not in existing:
            connection.execute(
                f"ALTER TABLE entries ADD COLUMN {name} {declaration}"
            )


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
                (kind, name, statement, left, right, steps, solution, signature,
                 lead_signature, applies_to, verified, origin, source,
                 dedup_key, panel)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["kind"], entry["name"], entry["statement"],
                entry.get("left"), entry.get("right"), entry.get("steps"),
                entry.get("solution") or "",
                entry["signature"], entry.get("lead_signature", ""),
                entry.get("applies_to", ""),
                int(entry.get("verified", 0)), entry.get("origin", "ai"),
                entry.get("source"), entry["dedup_key"],
                int(entry.get("panel", 1)),
            ),
        )
    except sqlite3.IntegrityError as error:
        if "dedup_key" not in str(error):
            raise  # some OTHER problem -- do not hide it behind "duplicate"
        return False  # we already have this exact formula

    # Which formula families this row belongs to (see classify.py).
    for group_name in entry.get("groups") or []:
        connection.execute(
            "INSERT OR IGNORE INTO group_members (group_name, entry_id) "
            "VALUES (?, ?)",
            (group_name, cursor.lastrowid),
        )

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

    Ordered best first:
        1. formulas whose whole shape matches
        2. formulas whose FIRST part matches one of our parts -- that is
           how "sin^2 + cos^2 = 1" is found inside the bigger sum
           "sin(z)^2 + cos(z)^2 + 4", and how a loose exact match (a
           pattern with a greedy %) can never hide the better formula:
           the caller tries every candidate until one actually applies.

    The stored signature is used as a SQL LIKE pattern: a formula's hole
    (stored as %) may then cover a whole subtree, e.g.  sin(%)  is found
    for  sin(x)  and for  sin(x + y)  alike.
    """
    shape = signature_of(expr)

    params = [shape]
    lead_clause = ""
    if isinstance(expr, (Add, Mul)):
        part_shapes = [signature_of(part) for part in expr.args]
        matches = " OR ".join("? LIKE lead_signature" for _ in part_shapes)
        lead_clause = f" OR ({matches})"
        params.extend(part_shapes)

    return connection.execute(
        f"""
        SELECT * FROM entries
        WHERE kind = 'identity' AND verified = 1
          AND (? LIKE signature {lead_clause})
        ORDER BY (? LIKE signature) DESC, usage_count DESC, id
        LIMIT ?
        """,
        (*params, shape, limit),
    ).fetchall()


def find_by_groups(connection: sqlite3.Connection, tags: "list | set",
                   limit: int = 40) -> List[sqlite3.Row]:
    """Trusted identities belonging to ANY of these families.

    This is the group route: classify(node) -> tags -> members. It is
    bounded by the group size instead of scanning the whole table, and it
    finds families by MEANING (e.g. the expanded spelling of a two-angle
    sum), which the shape index alone cannot do.
    """
    tags = list(tags)
    if not tags:
        return []

    placeholders = ",".join("?" * len(tags))
    return connection.execute(
        f"""
        SELECT DISTINCT e.* FROM group_members gm
        JOIN entries e ON e.id = gm.entry_id
        WHERE gm.group_name IN ({placeholders})
          AND e.kind = 'identity' AND e.verified = 1
        ORDER BY e.usage_count DESC, e.id
        LIMIT ?
        """,
        (*tags, limit),
    ).fetchall()


def find_methods(connection: sqlite3.Connection, expr: Expr,
                 limit: int = 20, for_ways: bool = False) -> List[sqlite3.Row]:
    """Trusted methods that could start on this expression.

    A method is offered when
        * its first step is a plain operation (like "expand") -> always
        * its first step is a pattern whose shape matches the WHOLE
          expression (the stored shape is a LIKE pattern, so a hole % may
          cover a whole subtree)
        * its first step is a pattern that matches one of our PARTS --
          directly (applies_to) or via that part's first piece
          (lead_signature).

    panel=False rows are hidden from the buttons (the UI has none now
    anyway; alternatives collect them instead).

    for_ways=True is the collector for "other ways to the same answer":
    it skips the always-offered ('any') methods, which would clutter every
    answer, and keeps both explicit-shape and panel=False rows.
    """
    shape = signature_of(expr)

    params = [shape]
    part_clause = ""
    if isinstance(expr, (Add, Mul)):
        part_shapes = [signature_of(part) for part in expr.args]
        matches = " OR ".join(
            "(? LIKE lead_signature OR ? LIKE applies_to)"
            for _ in part_shapes
        )
        part_clause = f" OR ({matches})"
        params.extend(part_shapes)  # first ? of every pair
        params.extend(part_shapes)  # second ? of every pair

    if for_ways:
        head = "applies_to != 'any' AND (? LIKE applies_to"
    else:
        head = "panel = 1 AND (applies_to = 'any' OR ? LIKE applies_to"

    return connection.execute(
        f"""
        SELECT * FROM entries
        WHERE kind = 'method' AND verified = 1
          AND {head} {part_clause})
        ORDER BY (? LIKE applies_to) DESC, usage_count DESC, id
        LIMIT ?
        """,
        (*params, shape, limit),
    ).fetchall()


def search(connection: sqlite3.Connection, query: str = "",
           verified: Optional[int] = None, limit: int = 50,
           group: Optional[str] = None) -> List[sqlite3.Row]:
    """Search box: full-text on name/statement, or browse when empty.

    Anything that the full-text engine cannot read (users like typing
    things such as "x^2") falls back to a simple contains-search, so the
    box never throws an error at the user.

    group="two-angle-sum" limits the result to that family's members.
    """
    group_clause = ""
    group_params = []
    if group:
        group_clause = (" AND id IN (SELECT entry_id FROM group_members"
                        " WHERE group_name = ?)")
        group_params = [group]

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
                    f"""
                    SELECT e.* FROM entries_fts f
                    JOIN entries e ON e.id = f.rowid
                    WHERE entries_fts MATCH ?
                      AND (? IS NULL OR e.verified = ?)
                      {group_clause.replace(' id ', ' e.id ')}
                    ORDER BY e.usage_count DESC, e.id
                    LIMIT ?
                    """,
                    (fts_query, verified, verified, *group_params, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            rows = connection.execute(
                f"""
                SELECT * FROM entries
                WHERE (name LIKE ? OR statement LIKE ?)
                  AND (? IS NULL OR verified = ?)
                  {group_clause}
                ORDER BY usage_count DESC, id
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", verified, verified,
                 *group_params, limit),
            ).fetchall()
    else:
        rows = connection.execute(
            f"""
            SELECT * FROM entries
            WHERE (? IS NULL OR verified = ?)
            {group_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            (verified, verified, *group_params, limit),
        ).fetchall()
    return rows


def list_groups(connection: sqlite3.Connection) -> List[dict]:
    """Every formula family with its description and member count."""
    from .classify import GROUPS

    counts = {
        row["group_name"]: row["n"]
        for row in connection.execute(
            "SELECT group_name, COUNT(*) AS n FROM group_members "
            "GROUP BY group_name")
    }

    result = [
        {"name": name, "description": description,
         "count": counts.get(name, 0)}
        for name, description in GROUPS.items()
    ]
    known = set(GROUPS)
    for name, count in counts.items():        # families not (yet) in GROUPS
        if name not in known:
            result.append({"name": name, "description": "", "count": count})
    return result


def groups_for_rows(connection: sqlite3.Connection,
                    rows: List) -> dict:
    """{entry id: [family names]} for a page of rows (one query)."""
    if not rows:
        return {}
    ids = [row["id"] for row in rows]
    placeholders = ",".join("?" * len(ids))
    tags: dict = {}
    for entry in connection.execute(
            f"SELECT entry_id, group_name FROM group_members "
            f"WHERE entry_id IN ({placeholders}) ORDER BY group_name",
            ids):
        tags.setdefault(entry["entry_id"], []).append(entry["group_name"])
    return tags


def find_solutions(connection: sqlite3.Connection, expr: Expr) -> List[sqlite3.Row]:
    """Verified solution rows whose equation side has this shape.

    Used by the solver: given the left side of an equation (say sin(x)),
    find rows like "sin(?t) = 0  ->  ?t = ?n*pi". The stored shape is a
    LIKE pattern, so the hole in  sin(%)  also fits  sin(x + y).
    """
    shape = signature_of(expr)
    return connection.execute(
        """
        SELECT * FROM entries
        WHERE kind = 'solution' AND verified = 1 AND ? LIKE signature
        ORDER BY usage_count DESC, id
        LIMIT 50
        """,
        (shape,),
    ).fetchall()


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
