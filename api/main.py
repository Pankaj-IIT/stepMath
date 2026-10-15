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

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.algebra import strategies as strategy_module
from src.algebra.parser import parse_equation, parse_expression
from src.algebra.solver import solve_equation
from src.algebra.steps import simplify_steps
from src.intelligence.handlers import dispatch
from src.intelligence.reading import read_question
from src.knowledge import runner as method_runner
from src.knowledge import store
from src.knowledge.ops import OPERATIONS

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
    target: str | None = None  # the answer to reach; defaults to simplify()


class Step(BaseModel):
    name: str  # e.g. "Combine like terms"
    latex: str  # e.g. "5 x + 4"  (KaTeX draws this)
    formula: str | None = None  # the formula used, e.g. "sin^2 + cos^2 = 1"


class AlternativeWay(BaseModel):
    """Another route to the same answer: its first move, and the full path."""
    name: str  # the move that starts this route, e.g. "Sine of A + B"
    steps: list[Step]


class SolveResponse(BaseModel):
    steps: list[Step]
    final: str  # the latex of the last step
    alternatives: list[AlternativeWay] = []  # other ways to the same answer


class Entry(BaseModel):
    """One row of the formula database, as the UI needs to show it."""
    id: int
    kind: str  # 'identity' | 'method' | 'note' | 'solution'
    name: str
    statement: str  # LaTeX, may be empty for methods of plain operations
    steps: str | None = None  # methods only: their steps as JSON text
    verified: int  # 1 = trusted, 0 = waiting for review
    origin: str  # 'ai' | 'hand'
    source: str | None = None
    usage_count: int
    groups: list[str] = []  # formula families, e.g. ['two-angle-sum']


class EntryList(BaseModel):
    entries: list[Entry]
    counts: dict  # totals, so the UI can show "12 need review"


class GroupList(BaseModel):
    """The formula families, for the filter chips in the Formulas tab."""
    groups: list[dict]  # {name, description, count}


class GapList(BaseModel):
    """Simplifications SymPy found that our strategies cannot reach yet."""
    gaps: list[dict]  # {target, example, tried, hits}


class StrategyIn(BaseModel):
    """What the Strategies tab saves (both add and edit)."""
    name: str
    form: str
    phases: list[dict]  # validated by strategies.validate_strategy, shared
                        # with the importer, so the two can never disagree


class StrategyOut(BaseModel):
    """One strategy row, as the Strategies tab shows it."""
    id: int
    name: str
    form: str
    phases: list[dict]
    verified: int
    usage_count: int


class StrategyList(BaseModel):
    strategies: list[StrategyOut]


class StrategyVocabulary(BaseModel):
    """The choices the phases builder offers (from the interpreter)."""
    forms: list[str]
    dos: list[str]      # "walk" plus every known operation
    scopes: list[str]
    untils: list[str]


class MethodRunRequest(BaseModel):
    expression: str
    id: int  # which method (entry id) to run


class AskRequest(BaseModel):
    question: str  # the whole typed question, e.g. "prove that sin(x)^2 + ... = 1"


class AskResponse(SolveResponse):
    """What the question layer answers: the direction plus its steps."""
    intent: str  # simplify | prove | solve | ...
    explanation: str  # WHY it read the question that way
    proven: bool | None = None  # only for intent=prove


@app.get("/api/health")
def health() -> dict:
    """Used by the UI to check that the Python server is running."""
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> "AskResponse | JSONResponse":
    """The question layer: read the question, pick a direction, run it.

    The reading is deterministic (no AI): plain rules say whether this is
    a simplify/prove/solve question and WHY, then the matching handler
    produces the steps.
    """
    try:
        reading = read_question(request.question)
        result = dispatch(reading)
        return AskResponse(
            steps=result["steps"],
            final=result["final"],
            alternatives=[AlternativeWay(**way)
                          for way in result.get("alternatives", [])],
            intent=result["intent"],
            explanation=result["explanation"],
            proven=result.get("proven"),
        )
    except Exception as error:  # ValueError, NotSupported, parse errors...
        return _error(error)


@app.post("/api/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> "SolveResponse | JSONResponse":
    try:
        ways: list = []

        if request.mode == "solve":
            left, right = parse_equation(request.expression)
            steps = solve_equation(left, right)
        elif request.mode == "simplify":
            expression = parse_expression(request.expression)
            target = (parse_expression(request.target)
                      if request.target else None)
            steps = simplify_steps(expression, target=target)
            # alternatives held for a later version (user: one solution for now)
        else:
            raise ValueError(f'Unknown mode "{request.mode}" (use simplify or solve).')

        return SolveResponse(
            steps=steps,
            final=steps[-1]["latex"],
            alternatives=[AlternativeWay(**way) for way in ways],
        )

    except Exception as error:  # any problem becomes a friendly 400 message
        return _error(error)


def _error(error: Exception) -> JSONResponse:
    """Every problem leaves the API the same way: a 400 with a readable text."""
    return JSONResponse(
        status_code=400,
        content={"error": str(error) or "Something went wrong."},
    )


def _entry_out(row, tags: "list | None" = None) -> Entry:
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
        groups=tags or [],
    )


def _strategy_out(row) -> StrategyOut:
    """One strategy row as the Strategies tab needs to show it.

    `steps` is either the bare phase list (everything written since the
    strategies table existed) or a whole document dict with a "phases"
    key -- the same duality _run_phases accepts at solve time, kept here
    as a defensive unwrap.

    `verified` is a compatibility shim: strategies are trusted on save
    (they have no left/right to prove), and the strategies table has no
    review-queue column, so every row answers 1.
    """
    try:
        doc = json.loads(row["steps"] or "[]")
    except (TypeError, json.JSONDecodeError):
        doc = None  # a broken row still lists (its phases render empty)
    phases = doc.get("phases") if isinstance(doc, dict) else doc
    if not isinstance(phases, list):
        phases = []
    return StrategyOut(
        id=row["id"],
        name=row["name"],
        form=row["form"],
        phases=phases,
        verified=1,
        usage_count=row["usage_count"],
    )


# ---------------------------------------------------------------------------
# The formula database: browse, search, families, and review
# ---------------------------------------------------------------------------
@app.get("/api/entries", response_model=EntryList)
def list_entries(q: str = "", verified: int | None = None,
                 limit: int = 500, group: str = "") -> EntryList:
    """Search the database, or browse everything when q is empty.

    verified=1 shows only trusted formulas, verified=0 shows the review
    queue, and leaving it out shows everything. group="two-angle-sum"
    limits the list to one formula family.
    """
    connection = store.get_connection()
    rows = store.search(connection, q, verified, limit, group or None)
    tags = store.groups_for_rows(connection, rows)
    return EntryList(
        entries=[_entry_out(row, tags.get(row["id"])) for row in rows],
        counts=store.counts(connection),
    )


@app.get("/api/groups", response_model=GroupList)
def list_groups() -> GroupList:
    """The formula families, each with a description and member count."""
    return GroupList(groups=store.list_groups(store.get_connection()))


@app.get("/api/gaps", response_model=GapList)
def gaps() -> GapList:
    """The "plug the strategy here" list: targets we cannot explain yet."""
    return GapList(gaps=[dict(row) for row in store.list_gaps()])


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
# Strategies: view, create, edit, delete (the Strategies tab).
# Saving is INSTANT -- validate, write, done (no review queue: a strategy
# has no left/right to prove, and the interpreter itself refuses rows
# whose vocabulary it does not know).
# ---------------------------------------------------------------------------
@app.get("/api/strategies", response_model=StrategyList)
def list_strategies() -> StrategyList:
    """Every strategy, in id order (what the Strategies tab lists)."""
    rows = store.list_strategies(store.get_connection())
    return StrategyList(strategies=[_strategy_out(row) for row in rows])


@app.get("/api/strategy-meta", response_model=StrategyVocabulary)
def strategy_meta() -> StrategyVocabulary:
    """The vocabulary a phase may use: forms, ops, scopes, untils."""
    return StrategyVocabulary(
        forms=sorted(strategy_module._FORMS),
        dos=["walk", *sorted(OPERATIONS)],
        scopes=["whole", "term", "combination", "bracket"],
        untils=["stuck", "once"],
    )


@app.post("/api/strategies")
def create_strategy(request: StrategyIn) -> dict:
    """Instant save: validate with the shared checker, then insert."""
    try:
        name = request.name.strip()
        if not name:
            raise ValueError("a strategy needs a name")
        form = request.form.strip()
        strategy_module.validate_strategy(form, request.phases)

        steps_json = json.dumps(request.phases, ensure_ascii=False)
        connection = store.get_connection()
        # A duplicate (same form + phases) raises the friendly ValueError
        # here, which the except below turns into the 400 the POST has
        # always answered with.
        strategy_id = store.insert_strategy(connection, name, form,
                                             steps_json, source="ui")
        connection.commit()
        return {"id": strategy_id}

    except Exception as error:  # ValueError from the shared validator...
        return _error(error)


@app.put("/api/strategies/{entry_id}")
def update_strategy(entry_id: int, request: StrategyIn) -> dict:
    """Instant save over an existing row: update, never re-insert."""
    try:
        name = request.name.strip()
        if not name:
            raise ValueError("a strategy needs a name")
        form = request.form.strip()
        strategy_module.validate_strategy(form, request.phases)

        connection = store.get_connection()
        row = store.get_strategy(connection, entry_id)
        if row is None:
            return _error(ValueError("No such strategy."))

        steps_json = json.dumps(request.phases, ensure_ascii=False)
        store.update_strategy(connection, entry_id, name, form, steps_json)
        connection.commit()
        return {"ok": True, "id": entry_id}

    except Exception as error:
        return _error(error)


@app.delete("/api/strategies/{entry_id}")
def delete_strategy(entry_id: int) -> dict:
    """Gone: the row is deleted for good."""
    connection = store.get_connection()
    if store.get_strategy(connection, entry_id) is None:
        return _error(ValueError("No such strategy."))
    store.delete_strategy(connection, entry_id)
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
