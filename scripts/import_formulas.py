"""
Import formulas from a JSONL file into data/knowledge.db.

    python scripts/import_formulas.py data/samples/algebra.jsonl

One JSON object per line. Five kinds are accepted:

  {"kind":"identity","name":"Sum of Squares",
   "left":"?a^2 - ?b^2","right":"(?a - ?b)*(?a + ?b)"}

  {"kind":"method","name":"Factorise completely",
   "steps":[{"op":"factor"}]}

  {"kind":"strategy","name":"Expand then reduce","form":"product_of_sums",
   "phases":[{"do":"expand","scope":"whole","until":"stuck"}, ...]}

  {"kind":"note","name":"Period of tan",
   "statement":"\\text{period } \\pi"}          (reference only, never rewrites)

  {"kind":"solution","name":"sin theta = 0",
   "left":"sin(?t)","right":"0","solution":"?t = ?n*pi"}

What happens to each line:

    unreadable / unknown kind           -> rejected.jsonl (with the reason)
    identity that SymPy proves wrong    -> rejected.jsonl (with a counter-example)
    identity SymPy proves right         -> verified = 1
    identity SymPy cannot decide        -> verified = 0  (review queue)
    method with a provably wrong step   -> rejected.jsonl (with the reason)
    method whose steps all prove right  -> verified = 1
    method with a step SymPy cannot tell-> verified = 0  (review queue)
    note (reference text, never used
         to rewrite anything)           -> verified = 1
    solution that does NOT solve its
         equation                       -> rejected.jsonl (with a counter-example)
    solution SymPy proves               -> verified = 1
    solution SymPy cannot decide        -> verified = 0  (review queue)
    strategy with an unknown form or a
    phase the interpreter cannot run   -> rejected.jsonl (with the reason)
    strategy whose form and phases are
    all in the interpreter's vocabulary-> verified = 1  (no proof: no
                                              left/right to prove)
    exact duplicate of something we have-> skipped

Running the same file twice is harmless -- duplicates are detected by
content, not by file name.
"""

import json
import sys
from pathlib import Path

# Make "src..." importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sympy import Expr  # noqa: E402

from src.algebra.latex import to_latex  # noqa: E402
from src.knowledge import gate, runner, store  # noqa: E402
from src.knowledge.classify import classify, mentioned_functions  # noqa: E402
from src.knowledge.patterns import parse_pair, parse_solution  # noqa: E402
from src.knowledge.signature import signature_of  # noqa: E402

KINDS = ("identity", "method", "strategy", "note", "solution")


def _lead_of(left: Expr) -> str:
    """Shape stored for the "first part" lookup of a sum/product pattern.

    Two adjustments, both about surviving a LIKE lookup against whatever
    your typed maths looks like:

    * a hole-coefficient is stripped: the pattern part  ?a*sin(?t)  is
      stored as  sin(%)  so it also finds the BARE term  sin(x)
      (implicit coefficient 1)
    * a leading % is added, so the same stored shape also finds the
      WRAPPED term  Mul(#, sin(_))  i.e.  3*sin(x)  -- that is what makes
      "combine a sin + b cos" fire inside a three-term sum like
      3*sin(x) + 4*cos(x) + 5, where the whole-shape lookup cannot help
      (2 terms in the formula, 3 in the expression).

    The matcher still proves every candidate afterwards, so the looser
    lookup only ever costs a little time, never correctness.
    """
    from sympy import Add, Mul, Wild

    if not isinstance(left, (Add, Mul)) or not left.args:
        return ""

    first = left.args[0]
    while isinstance(first, Mul) and len(first.args) == 2 \
            and isinstance(first.args[0], Wild):
        first = first.args[1]  # ?a*X  ->  X

    return "%" + signature_of(first)


def _method_statement(steps: list) -> str:
    """LaTeX for a method: its first pattern step, when it has one."""
    for step in steps:
        if "left" in step:
            left, right = parse_pair(step["left"], step["right"])
            return f"{to_latex(left)} = {to_latex(right)}"
    return ""  # a method of plain operations has no formula to show


def _prepare(line: dict, source: str) -> dict:
    """Turn one JSON line into the fields the database expects.

    Raises ValueError with a human-readable reason when the line is bad.
    """
    kind = line.get("kind")
    name = (line.get("name") or "").strip()

    if kind not in KINDS:
        raise ValueError(f'kind must be one of {", ".join(KINDS)}, got {kind!r}')
    if not name:
        raise ValueError("a formula needs a name")

    origin = line.get("origin", "ai")
    entry = {
        "kind": kind,
        "name": name,
        "origin": origin,
        "source": source,
    }

    if kind == "identity":
        left_text = (line.get("left") or "").strip()
        right_text = (line.get("right") or "").strip()
        if not left_text or not right_text:
            raise ValueError("an identity needs both 'left' and 'right'")

        left, right = parse_pair(left_text, right_text)  # raises when unreadable

        verdict, detail = gate.verify_identity(left, right)
        if verdict == "fail":
            raise ValueError(f"SymPy disproved it: {detail}")

        # lead_signature: for sum/product formulas, the shape of the first
        # part -- this is what lets us find a formula hiding inside a
        # bigger sum (see store.find_identities). Hole-coefficients are
        # stripped so bare terms match too (see _lead_of).
        lead = _lead_of(left)

        entry.update(
            left=left_text,
            right=right_text,
            # statement (the LaTeX we show people) is optional in the file:
            # we can build it from the two sides.
            statement=line.get("statement") or f"{to_latex(left)} = {to_latex(right)}",
            signature=signature_of(left),
            lead_signature=lead,
            # Which formula families this belongs to -- computed by running
            # the SAME classifier the solver uses on nodes, so membership
            # and lookup can never drift apart.
            groups=sorted(classify(left)),
            # BOTH sides: a row must be findable from whichever side
            # resembles the expression (that is what lets a backwards
            # direction be offered at all).
            functions=sorted(mentioned_functions(left)
                             | mentioned_functions(right)),
            verified=1 if verdict == "pass" else 0,
        )
        return entry

    # --- a note -----------------------------------------------------------
    # Reference facts (domain/range/period tables, polygon formulas, advice
    # on how to solve a type of equation). They are shown in the Formulas
    # tab and never used to rewrite anything, so nothing to prove here.
    if kind == "note":
        statement = (line.get("statement") or "").strip()
        if not statement:
            raise ValueError("a note needs a 'statement' (the maths to show)")

        entry.update(
            statement=statement,
            signature="",
            verified=1,
        )
        return entry

    # --- a solution -------------------------------------------------------
    # The general solution of an equation, e.g.  sin(?t) = 0  ->  ?t = ?n*pi.
    # Holes that only appear in the answer (?n) are read as integers.
    if kind == "solution":
        left_text = (line.get("left") or "").strip()
        right_text = (line.get("right") or "").strip()
        solution_text = (line.get("solution") or "").strip()
        if not left_text or not right_text or not solution_text:
            raise ValueError("a solution needs 'left', 'right' and 'solution'")

        left, right, unknown, value = parse_solution(
            left_text, right_text, solution_text
        )  # raises when unreadable

        verdict, detail = gate.verify_solution(left, right, unknown, value)
        if verdict == "fail":
            raise ValueError(f"SymPy disproved the answer: {detail}")

        entry.update(
            left=left_text,
            right=right_text,
            solution=solution_text,
            statement=line.get("statement") or (
                f"{to_latex(left)} = {to_latex(right)}\\;\\Rightarrow\\;"
                f"{to_latex(unknown)} = {to_latex(value)}"
            ),
            signature=signature_of(left),
            verified=1 if verdict == "pass" else 0,
        )
        return entry

    # --- a strategy -------------------------------------------------------
    # An authored route: a FORM (which expressions it applies to) plus the
    # PHASES the interpreter runs for it. There is no left/right to prove,
    # so the row is trusted straight away -- but the form and every phase
    # must be something strategies.py actually knows how to run, or the
    # route would fail silently at solve time.
    if kind == "strategy":
        from src.algebra import strategies as strategy_module  # lazy: cycle

        form = (line.get("form") or "").strip()
        phases = line.get("phases")
        # One shared validator: the Strategies tab runs every row through
        # the SAME checks, so the importer and the UI can never disagree.
        strategy_module.validate_strategy(form, phases)

        entry.update(
            # The phases go in exactly as authored, so the interpreter
            # can run them unchanged (the form travels as its own field --
            # strategies have their own table, with no formula columns).
            steps=json.dumps(phases, ensure_ascii=False),
            form=form,
            verified=1,
        )
        return entry

    # --- a method -------------------------------------------------------
    steps = line.get("steps")
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except json.JSONDecodeError as error:
            raise ValueError(f"steps is not valid JSON: {error}")

    runner.validate_steps(steps)  # raises with the reason when unusable

    # Machine-check the method: every pattern step must be provable, or the
    # whole method waits for a human. (A wrong step would silently break
    # people's work, so we do not trust anyone -- including ourselves --
    # without a proof.)
    needs_review = False
    for step in steps:
        if "left" not in step:
            continue  # a plain operation like "expand" is always safe

        step_left, step_right = parse_pair(step["left"], step["right"])
        verdict, detail = gate.verify_identity(step_left, step_right)
        if verdict == "fail":
            raise ValueError(
                f'step "{step.get("name", step["left"])}" is wrong: {detail}'
            )
        if verdict != "pass":
            needs_review = True

    import json as _json  # JSON we store back into the row
    steps_json = _json.dumps(steps, ensure_ascii=False)

    # Where should this method be offered? An explicit "for" pattern wins
    # (use it when the steps are computations, which carry no pattern of
    # their own); otherwise the first pattern step decides.
    from sympy import Add, Mul
    applies_to, lead = "any", ""
    for_text = (line.get("for") or "").strip()
    if for_text:
        from src.knowledge.patterns import parse_pattern
        first = parse_pattern(for_text)  # raises when unreadable
        applies_to = signature_of(first)
        lead = _lead_of(first)
    else:
        for step in steps:
            if "left" in step:
                first, _ = parse_pair(step["left"], step["right"])
                applies_to = signature_of(first)
                lead = _lead_of(first)
                break

    # "panel": false keeps a method out of any buttons; it stays available
    # as an alternative way (the ways collector uses for_ways=True).
    entry["panel"] = 0 if line.get("panel") is False else 1

    entry.update(
        steps=steps_json,
        statement=line.get("statement") or _method_statement(steps),
        signature=applies_to,
        applies_to=applies_to,
        lead_signature=lead,
        verified=0 if needs_review else 1,
    )
    return entry


def import_file(path: Path) -> dict:
    """Import every line of one JSONL file. Returns the summary counts."""
    source = path.name
    rejected_path = path.with_name(f"{path.stem}.rejected.jsonl")

    summary = {"lines": 0, "imported": 0, "duplicates": 0,
               "verified": 0, "needs_review": 0, "rejected": 0}
    rejected = []

    connection = store.connect()

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            text = raw_line.strip()
            if not text or text.startswith("#"):
                continue
            summary["lines"] += 1

            try:
                line = json.loads(text)
                if not isinstance(line, dict):
                    raise ValueError("each line must be a JSON object")
                entry = _prepare(line, source)

                # Strategies have their own table and their own fingerprint
                # (form::phases, computed by store.insert_strategy), so
                # only the entries kinds are fingerprinted here:
                #   identity  -> its two sides
                #   method    -> its list of steps
                #   solution  -> its answer text (both +/- branches count)
                #   note      -> its name AND its statement (two notes may
                #                legitimately show the same maths)
                if entry["kind"] != "strategy":
                    if entry["kind"] == "identity":
                        content = ""
                    elif entry["kind"] == "method":
                        content = entry.get("steps")
                    elif entry["kind"] == "solution":
                        content = entry.get("solution")
                    else:
                        content = f'{entry["name"]} :: {entry.get("statement")}'

                    entry["dedup_key"] = store.dedup_key(
                        entry["kind"], entry.get("left"), entry.get("right"),
                        content,
                    )
            except Exception as error:
                summary["rejected"] += 1
                rejected.append(json.dumps(
                    {"reason": str(error), "line": text}, ensure_ascii=False))
                continue

            if entry["kind"] == "strategy":
                # Into the strategies table. A row with the SAME form and
                # phases is a duplicate (counted as one, not re-imported);
                # insert_strategy reports it as the same friendly ValueError
                # the API's POST answers with.
                try:
                    store.insert_strategy(
                        connection, entry["name"], entry["form"],
                        entry["steps"],
                        origin=entry.get("origin", "hand"),
                        source=entry.get("source"),
                    )
                except ValueError:
                    summary["duplicates"] += 1
                else:
                    summary["imported"] += 1
                    if entry["verified"]:
                        summary["verified"] += 1
                    else:
                        summary["needs_review"] += 1
                continue

            if store.insert_entry(connection, entry):
                summary["imported"] += 1
                if entry["verified"]:
                    summary["verified"] += 1
                else:
                    summary["needs_review"] += 1
            else:
                summary["duplicates"] += 1

        connection.commit()
    finally:
        connection.close()

    # Keep the rejects so nothing an AI produced is silently lost
    # (but do not create an empty file when everything was fine).
    if rejected:
        rejected_path.write_text("\n".join(rejected) + "\n", encoding="utf-8")
    summary["rejected_file"] = str(rejected_path) if rejected else None
    return summary


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    for name in argv[1:]:
        path = Path(name)
        if not path.exists():
            print(f"not found: {path}")
            return 1

        summary = import_file(path)
        print(
            f"{path.name}: lines {summary['lines']} | "
            f"imported {summary['imported']} "
            f"(verified {summary['verified']}, "
            f"needs review {summary['needs_review']}) | "
            f"duplicates {summary['duplicates']} | "
            f"rejected {summary['rejected']}"
        )

    print("database:", store.counts(store.connect()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
