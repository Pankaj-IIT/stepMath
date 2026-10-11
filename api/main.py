"""
The web server -- the door between the React app and the maths engine.

Run it from the project folder:

    uvicorn api.main:app --reload

Then the React dev server can ask it, through the /api proxy in vite.config.ts:

    POST /api/solve
    { "expression": "2*x + 3*x + 5 - 1", "mode": "simplify" }

    200 -> { "steps": [ { "name": ..., "latex": ... }, ... ], "final": "..." }
    400 -> { "error": "..." }   (the message shown under the input box)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.algebra.latex import to_latex
from src.algebra.parser import parse_equation, parse_expression
from src.algebra.solver import solve_equation
from src.algebra.steps import simplify_steps
from src.knowledge import runner as method_runner
from src.knowledge import store

app = FastAPI(title="Math Step Solver API", version="1.0")

# Allows the browser to call us directly too (handy if the Vite proxy is off).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- What the browser sends and receives ----------------------------------
class SolveRequest(BaseModel):
    expression: str  # what the user typed, e.g. "2*x + 3*x + 5 - 1"
    mode: str = "simplify"  # "simplify" or "solve"


class Step(BaseModel):
    name: str  # e.g. "Combine like terms"
    latex: str  # e.g. "5 x + 4"  (KaTeX draws this)


class SolveResponse(BaseModel):
    steps: list[Step]
    final: str  # the latex of the last step


class Entry(BaseModel):
    """One row of the formula database, as the UI needs to show it."""
    id: int
    kind: str  # 'identity' | 'method'
    name: str
    statement: str  # LaTeX, may be empty for methods of plain operations
    steps: str | None = None  # methods only: their steps as JSON text
    verified: int  # 1 = trusted, 0 = waiting for review
    origin: str  # 'ai' | 'hand'
    source: str | None = None
    usage_count: int


class EntryList(BaseModel):
    entries: list[Entry]
    counts: dict  # totals, so the UI can show "12 need review"


class MethodRunRequest(BaseModel):
    expression: str
    id: int  # which method (entry id) to run


@app.get("/api/health")
def health() -> dict:
    """Used by the UI to check that the Python server is running."""
    return {"status": "ok"}


@app.post("/api/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> "SolveResponse | JSONResponse":
    try:
        if request.mode == "solve":
            left, right = parse_equation(request.expression)
            steps = solve_equation(left, right)
        elif request.mode == "simplify":
            expression = parse_expression(request.expression)
            steps = simplify_steps(expression)
        else:
            raise ValueError(f'Unknown mode "{request.mode}" (use simplify or solve).')

        return SolveResponse(steps=steps, final=steps[-1]["latex"])

    except Exception as error:  # any problem becomes a friendly 400 message
        return _error(error)


def _error(error: Exception) -> JSONResponse:
    """Every problem leaves the API the same way: a 400 with a readable text."""
    return JSONResponse(
        status_code=400,
        content={"error": str(error) or "Something went wrong."},
    )


def _entry_out(row) -> Entry:
    """Turn one database row into what the UI asked for."""
    return Entry(
        id=row["id"],
        kind=row["kind"],
        name=row["name"],
        statement=row["statement"],
        steps=row["steps"],
        verified=row["verified"],
        origin=row["origin"],
        source=row["source"],
        usage_count=row["usage_count"],
    )


# ---------------------------------------------------------------------------
# The formula database: browse, search, and review
# ---------------------------------------------------------------------------
@app.get("/api/entries", response_model=EntryList)
def list_entries(q: str = "", verified: int | None = None,
                 limit: int = 50) -> EntryList:
    """Search the database, or browse the newest rows when q is empty.

    verified=1 shows only trusted formulas, verified=0 shows the review
    queue, and leaving it out shows everything.
    """
    connection = store.get_connection()
    rows = store.search(connection, q, verified, limit)
    return EntryList(
        entries=[_entry_out(row) for row in rows],
        counts=store.counts(connection),
    )


@app.post("/api/entries/{entry_id}/approve")
def approve_entry(entry_id: int) -> dict:
    """A human says "this one is right" -> it starts being used."""
    connection = store.get_connection()
    if store.get_entry(connection, entry_id) is None:
        return _error(ValueError("No such formula."))
    store.set_verified(connection, entry_id, 1)
    connection.commit()
    return {"ok": True}


@app.post("/api/entries/{entry_id}/reject")
def reject_entry(entry_id: int) -> dict:
    """A human says "this one is wrong" -> it is deleted."""
    connection = store.get_connection()
    if store.get_entry(connection, entry_id) is None:
        return _error(ValueError("No such formula."))
    store.delete_entry(connection, entry_id)
    connection.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Methods: which ones could start here, and running one
# ---------------------------------------------------------------------------
@app.get("/api/methods/applicable", response_model=EntryList)
def applicable_methods(expression: str) -> EntryList:
    """Which complete methods could be started on this expression?"""
    try:
        parsed = parse_expression(expression)
        rows = store.find_methods(store.get_connection(), parsed)
    except Exception as error:
        return _error(error)

    return EntryList(
        entries=[_entry_out(row) for row in rows],
        counts=store.counts(store.get_connection()),
    )


@app.post("/api/methods/run", response_model=SolveResponse)
def run_method(request: MethodRunRequest) -> "SolveResponse | JSONResponse":
    """Run one method's steps on an expression and show each step."""
    try:
        connection = store.get_connection()
        row = store.get_entry(connection, request.id)

        if row is None:
            raise ValueError("No such method.")
        if row["kind"] != "method":
            raise ValueError("That entry is an identity, not a method.")

        parsed = parse_expression(request.expression)
        steps = method_runner.run_method(parsed, row["steps"])

        store.bump_usage(connection, row["id"])
        connection.commit()

        return SolveResponse(steps=steps, final=steps[-1]["latex"])

    except Exception as error:
        return _error(error)
