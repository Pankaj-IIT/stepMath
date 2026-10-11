"""
Import formulas from a JSONL file into data/knowledge.db.

    python scripts/import_formulas.py data/samples/algebra.jsonl

One JSON object per line. Two kinds are accepted:

  {"kind":"identity","name":"Sum of Squares",
   "left":"?a^2 - ?b^2","right":"(?a - ?b)*(?a + ?b)"}

  {"kind":"method","name":"Factorise completely",
   "steps":[{"op":"factor"}]}

What happens to each line:

    unreadable / unknown kind           -> rejected.jsonl (with the reason)
    identity that SymPy proves wrong    -> rejected.jsonl (with a counter-example)
    identity SymPy proves right         -> verified = 1
    identity SymPy cannot decide        -> verified = 0  (review queue)
    method written by a human           -> verified = 1  (we trust our own)
    method written by AI                -> verified = 0  (review queue)
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

from src.algebra.latex import to_latex  # noqa: E402
from src.knowledge import gate, runner, store  # noqa: E402
from src.knowledge.patterns import parse_pair  # noqa: E402
from src.knowledge.signature import signature_of  # noqa: E402


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

    if kind not in ("identity", "method"):
        raise ValueError(f'kind must be "identity" or "method", got {kind!r}')
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
        # bigger sum (see store.find_identities).
        from sympy import Add, Mul
        lead = ""
        if isinstance(left, (Add, Mul)) and left.args:
            lead = signature_of(left.args[0])

        entry.update(
            left=left_text,
            right=right_text,
            statement=f"{to_latex(left)} = {to_latex(right)}",
            signature=signature_of(left),
            lead_signature=lead,
            verified=1 if verdict == "pass" else 0,
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

    import json as _json  # JSON we store back into the row
    steps_json = _json.dumps(steps, ensure_ascii=False)

    # For a method that starts with a pattern step, record the shape of its
    # whole first step (applies_to) AND of that step's first part
    # (lead_signature) -- together they let us offer the method for an
    # expression that only looks similar, like  x^2 - 4.
    from sympy import Add, Mul
    applies_to, lead = "any", ""
    for step in steps:
        if "left" in step:
            first, _ = parse_pair(step["left"], step["right"])
            applies_to = signature_of(first)
            if isinstance(first, (Add, Mul)) and first.args:
                lead = signature_of(first.args[0])
            break

    entry.update(
        steps=steps_json,
        statement=_method_statement(steps),
        signature=applies_to,
        applies_to=applies_to,
        lead_signature=lead,
        # A method cannot be "proved" like an identity can, so AI methods
        # wait for a human. Ours (origin 'hand') we vouch for.
        verified=1 if origin == "hand" else 0,
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
                entry["dedup_key"] = store.dedup_key(
                    entry["kind"], entry.get("left"), entry.get("right"),
                    entry.get("steps"),
                )
            except Exception as error:
                summary["rejected"] += 1
                rejected.append(json.dumps(
                    {"reason": str(error), "line": text}, ensure_ascii=False))
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

    # Keep the rejects so nothing an AI produced is silently lost.
    rejected_path.write_text("\n".join(rejected) + ("\n" if rejected else ""),
                             encoding="utf-8")
    summary["rejected_file"] = str(rejected_path)
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
