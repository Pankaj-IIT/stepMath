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
        return JSONResponse(
            status_code=400,
            content={"error": str(error) or "Could not read that expression."},
        )
