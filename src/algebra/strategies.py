"""
strategies.py -- the strategy interpreter: a strategy document says IF
the expression has this FORM, THEN run these phases, in this order.

A strategy is DATA (a row in the database's `strategies` table): a form
key plus a list of phases such as

    {"do": "expand" | "simplify" | ... | "walk",
     "scope": "whole" | "term" | "combination" | "bracket",
     "until": "stuck" | "once"}

The combination scope is the strict one: only a collapse that reaches
ONE single term counts -- "it combines two terms but what is the use as
it does not convert to one single term so we should omit such results".
A subset whose result is still a sum (however re-spelled) or that would
leave a junk "+ 0" standing is omitted; every other part of a run is
unchanged.

This file holds what a strategy DOES -- the interpreter that runs one
document. The walk is NOT an engine: it is a phase strategies invoke,
and SymPy runs only when instructed (the goal fetch, an op phase, or the
ladder's last resort in steps.py). Every run spends the same budget and
obeys the SAME permission rule as everything else -- it may never land on
a state the route already showed. One solution only: a run returns its
ONE route, reaching or not (ladder X keeps the furthest honest route and
records a gap when it did not land).
"""

import json
from itertools import combinations

from sympy import Add
from sympy.core.expr import Expr

from ..knowledge.ops import known_operation, run_operation
from ..knowledge import store as knowledge_store
from .latex import to_latex, to_latex_as_written
from .parser import normalize
from .rules import SIMPLIFY_RULES
from .steps import (_SYMPY_MOVES, _SolveState, _bump, _initial_step,
                    _iter_changes, _method_leaps, first_formula_move)

# One strategy run's move limit -- moves one run may spend (a limit,
# never a preference). A form that MATCHES but cannot reach the goal must
# not hold the solver hostage. Note: 1 + STRATEGY_BUDGET can exceed
# MAX_STEPS (20) by one line; we show the honest route rather than slice it.
STRATEGY_BUDGET = 20

# One combination MOVE's attempt limit: how many run_operation calls the
# selection of a single move may spend before it reports stuck (a LIMIT,
# never a preference). It is also what bounds subset enumeration for a
# large root: subsets are produced lazily in order (pairs, then triples,
# then k=4, ...) and this cap alone says how far that walk may go.
COMBINATION_ATTEMPTS = 40


def _is_product_of_sums(expr: Expr) -> bool:
    """Form: a product carrying at least one multi-term bracket.

    (1 + tan(x))*(1 - cot(x)) matches, and so does (1 - tan(x))*cos(x):
    one bracket with more than one term is enough. Single terms around it
    (cos(x), a symbol, a number) do not disqualify the form. Mul is
    flattened by normalisation, so nested products count too.
    """
    return expr.is_Mul and any(
        isinstance(a, Add) and len(a.args) >= 2 for a in expr.args)


def _is_multi_term_sum(expr: Expr) -> bool:
    """Form: any sum or difference of two or more terms.

    Disjoint from product_of_sums (which needs a Mul): every Add matches,
    however simple -- 2*x + 3*x - 1 + 5 and sin(x)^8 - cos(x)^8 alike.
    """
    return expr.is_Add


# The known forms; a strategy row whose form is not a key here is
# rejected by the importer (unknown forms cannot be matched at solve time).
_FORMS = {"product_of_sums": _is_product_of_sums,
          "multi_term_sum": _is_multi_term_sum}


def validate_strategy(form, phases) -> None:
    """Check one strategy document against the interpreter's vocabulary.

    The JSONL importer and the Strategies tab both run every row through
    THIS function, so a strategy the UI accepts can never be one the
    importer rejects (and vice versa). Raises ValueError with a message
    naming the offending phase.
    """
    if not form:
        raise ValueError("a strategy needs a 'form' (the shape it matches)")
    if form not in _FORMS:
        known = ", ".join(sorted(_FORMS))
        raise ValueError(f"unknown form {form!r} -- known forms: {known}")

    if not isinstance(phases, list) or not phases:
        raise ValueError("a strategy needs a non-empty list of 'phases'")

    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            raise ValueError(f"phase {index} must be an object")
        do = phase.get("do")
        if do != "walk" and not known_operation(str(do or "")):
            raise ValueError(
                f"phase {index}: unknown operation {do!r} "
                f"-- use a known op or 'walk'"
            )
        if phase.get("scope") not in ("whole", "term", "combination",
                                      "bracket"):
            raise ValueError(
                f"phase {index}: scope must be 'whole', 'term', "
                f"'combination' or 'bracket', "
                f"got {phase.get('scope')!r}"
            )
        if phase.get("until") not in ("stuck", "once"):
            raise ValueError(
                f"phase {index}: until must be 'stuck' or 'once', "
                f"got {phase.get('until')!r}"
            )


class Strategy:
    """One authored route: a form test plus the run that executes it."""

    def __init__(self, name, matches, run):
        self.name = name
        self.matches = matches      # matches(expr) -> bool
        self.run = run              # run(expr, target, banned, include_db)
                                    #   -> (steps, final) | None


# The walk excludes SymPy moves: import _SYMPY_MOVES from .steps. SymPy
# runs only when instructed -- a walk phase may never pick one itself.
_SYMPY_FREE = _SYMPY_MOVES

# What the collect fallback accepts when collect() itself changes nothing
# (the tidy rules are what turn 1 - cot + tan - 1 into tan - cot).
_COLLECT_FALLBACK = {"Evaluate numbers", "Combine like terms"}


def _rebuild(root: Expr, index: int, new_child: Expr) -> Expr:
    """Put one changed child back into its parent, unevaluated then
    normalised, so the display stays in reading order.

    A product/sum root is also FLATTENED: a child that is itself a
    product/sum is opened up, so its factors/terms stay visible as
    children of the root for the next phase (an unflattened
    Mul((a, b), c) would hide a and b from scope-"term" phases)."""
    args = list(root.args)
    args[index] = new_child

    if root.is_Mul:
        flat = [p for a in args for p in (a.args if a.is_Mul else [a])]
        args = flat
    elif root.is_Add:
        flat = [p for a in args for p in (a.args if a.is_Add else [a])]
        args = flat

    return normalize(root.func(*args, evaluate=False))


def _combination_move(root, do, state, attempt_cap=COMBINATION_ATTEMPTS):
    """One pairwise-then-triple-etc. move over the root's children.

    The children (the terms of the current root) are combined SMALLEST
    SUBSETS FIRST: every pair left to right (i < j in index order), then
    every triple, then k=4, ... while the attempt cap allows.

    CANCEL BEFORE MERGE -- "cancel until stuck": the scan runs TWICE over
    that SAME enumeration order (no scoring, no best-candidate ranking --
    only which category is tried first). The first pass keeps a candidate
    only when the op's result is a PLAIN NUMBER (result.is_number --
    typically 0, the cancellations), so every subset that cancels is
    tried before any subset that merely merges. Only when the first pass
    found nothing does the second pass run the old rule below: the first
    single-term result that is not a sum.

    ACCEPTANCE RULE -- a subset only counts when run_operation(do, sum
    of those children) collapses it to ONE single term: "it combines two
    terms but what is the use as it does not convert to one single term
    so we should omit such results". So a result that is STILL an Add is
    rejected however it is spelled (re-bracketing a sum is not a real
    combination), and a result of 0 REMOVES those children outright --
    never a "+ 0" left standing. When the removal would leave nothing
    (the whole root vanished), the subset is rejected too. Otherwise the
    single term takes the subset's leftmost place (the rest of the
    subset is gone, every other child keeps its place), and the rebuilt
    root must still pass the visibility and history gates.

    The FIRST subset of the winning pass that reaches its accepted form
    AND passes the history gate wins -- one move per call, so each
    combination lands as its own visible step, exactly like every other
    phase (the phase's `until` loops it).

    Every run_operation call spends one attempt -- across BOTH passes;
    at attempt_cap the selection stops and reports stuck (None). That cap
    is a LIMIT, never a preference -- for a large n it is what keeps the
    enumeration bounded, since subsets are generated lazily in the order
    above.
    """
    children = list(root.args)
    if len(children) < 2:
        return None                      # nothing to pair up

    attempts = 0
    for cancellations_only in (True, False):
        for size in range(2, len(children) + 1):
            for subset in combinations(range(len(children)), size):
                if attempts >= attempt_cap:
                    return None              # the cap is a limit: stuck
                attempts += 1                # every run_operation call counts
                # Unevaluated: with evaluation on, Add(2*x, 3*x) collapses to
                # 5*x BEFORE the op sees it, so the op reports "no change" and
                # the combination phase is dead (the sum strategy fell through
                # to Factorise). The op gets the raw sum and must do the work;
                # an op that merely evaluates it returns an Add -> rejected by
                # the single-term rule below, so acceptance is unchanged.
                hit = run_operation(
                    do, Add(*(children[i] for i in subset), evaluate=False))
                if hit is None:
                    continue                 # this subset did not combine
                new_child, label = hit
                if cancellations_only:
                    if not new_child.is_number:
                        continue         # pass 1: cancellations only
                elif new_child.is_Add:
                    continue             # still a sum: no real combination
                picked = set(subset)
                collapse_to_zero = new_child == 0
                args = []
                for index, child in enumerate(children):
                    if index in picked:
                        if not collapse_to_zero and index == subset[0]:
                            # a genuine single term takes the subset's own
                            # (leftmost) place; a collapse to 0 deletes the
                            # subset instead (never a "+ 0" on screen), and
                            # every other child keeps its place.
                            args.append(new_child)
                        continue
                    args.append(child)
                if not args:
                    continue                 # the root vanished: nothing new
                rebuilt = normalize(root.func(*args, evaluate=False))
                if to_latex(rebuilt) == to_latex(root):
                    continue                 # not visible on screen
                if state.revisits(rebuilt):
                    continue
                return rebuilt, [{"name": label, "latex": to_latex(rebuilt)}], []
    return None


def _phase_move(root, do, scope, state, include_db=True):
    """One op move at this node -> (new_root, entries, rows) | None.

    do "apply_formula": one stored-identity rewrite somewhere in this
                        node's subtree (first applicable, forwards then
                        backwards), shown with the formula's OWN name and
                        statement -- the deterministic, no-wandering twin
                        of the walk's formula step (see steps.first_formula_move).
    scope "whole"      : the op runs on the whole expression.
    scope "term"       : the op runs on each child left to right; the
                         first child that changes the root AND passes the
                         history gate wins (only that term gets a step).
    scope "combination": the op runs on the smallest subset of children
                         that collapses to ONE single term -- every pair
                         first, then every triple, ... -- and only that
                         subset's move is accepted (one combination per
                         step). Only collapses that reach a single term
                         count; a result that is still a sum (however
                         re-spelled) or a collapse that would leave a
                         "+ 0" behind is omitted.
    scope "bracket"    : the op works INSIDE the brackets of a product.
                         The root must be a Mul (nothing factored -> phase
                         stuck); each factor is visited left to right, one
                         visible step per call: a bracket with more than
                         two terms gets a combination move (the same
                         acceptance rules as scope "combination"), any
                         other factor gets one plain operation -- when it
                         actually changes that factor. The phase's
                         `until:stuck` loop does the rest.
        entries = the display lines this move produced (a leap shows its
    sub-steps, never one opaque line); rows = usage ids to bump.
    """
    if do == "apply_formula":
        move = first_formula_move(root, state, include_db)
        if move is None:
            return None
        new_expr, name, formula, row = move
        if to_latex(new_expr) == to_latex(root) or state.revisits(new_expr):
            return None
        entry = {"name": name, "latex": to_latex(new_expr)}
        if formula:
            entry["formula"] = formula
        return new_expr, [entry], [row] if row else []

    if scope == "bracket":
        if not root.is_Mul:
            return None              # nothing factored -> phase stuck
        for index, factor in enumerate(root.args):
            if isinstance(factor, Add) and len(factor.args) > 2:
                move = _combination_move(factor, do, state)   # >2 terms
                if move is None:
                    continue
                new_factor, entries, rows = move
                rebuilt = _rebuild(root, index, new_factor)
            else:
                hit = run_operation(do, factor)      # <=2 terms: once
                if hit is None:
                    continue
                new_factor, label = hit
                if to_latex(new_factor) == to_latex(factor):
                    continue       # not visible on screen
                rebuilt = _rebuild(root, index, new_factor)
                entries = [{"name": label, "latex": to_latex(rebuilt)}]
                rows = []
            if state.revisits(rebuilt):
                continue
            return rebuilt, entries, rows
        return None

    if scope == "combination":
        return _combination_move(root, do, state)

    if scope == "term":
        for index, child in enumerate(root.args):
            result = run_operation(do, child)
            if result is None:
                continue
            new_child, label = result
            rebuilt = _rebuild(root, index, new_child)
            if to_latex(rebuilt) == to_latex(root):
                continue
            if state.revisits(rebuilt):
                continue
            return rebuilt, [{"name": label, "latex": to_latex(rebuilt)}], []
        return None

    result = run_operation(do, root)
    if result is not None:
        new_expr, label = result
        if to_latex(new_expr) != to_latex(root) and not state.revisits(new_expr):
            return new_expr, [{"name": label, "latex": to_latex(new_expr)}], []

    if do == "collect":
        # collect() itself rarely folds plain arithmetic; the tidy rules
        # are what turn 1 - cot + tan - 1 into tan - cot.
        for new_expr, name, formula, row in _iter_changes(
                root, SIMPLIFY_RULES, False, state,
                with_derived=False, direction="forward"):
            if name not in _COLLECT_FALLBACK:
                continue
            if state.revisits(new_expr):
                continue
            entry = {"name": name, "latex": to_latex(new_expr)}
            if formula:
                entry["formula"] = formula
            return new_expr, [entry], [row] if row else []
    return None


def _walk_phase(root, state, include_db):
    """One legal walk move -> (new_root, entries, rows) | None.

    Candidates come from the move generator (formulas + rules), skipping
    any _SYMPY_FREE name -- the walk may never pick SymPy by itself.
    First non-revisiting candidate wins; when none, an authored method
    leap runs as one move, shown as its own sub-steps.
    """
    for new_expr, name, formula, row in _iter_changes(
            root, SIMPLIFY_RULES, include_db, state, with_derived=True):
        if name in _SYMPY_FREE:
            continue
        if state.revisits(new_expr):
            continue
        entry = {"name": name, "latex": to_latex(new_expr)}
        if formula:
            entry["formula"] = formula
        return new_expr, [entry], [row] if row else []

    if include_db:
        for leap in _method_leaps(root, frozenset()):
            if state.revisits(leap["final"]):
                continue
            entries = [dict(step) for step in leap["steps"][1:]]  # no Initial
            if not entries:
                continue
            return leap["final"], entries, [leap["row_id"]]
    return None


def _run_phases(doc, expr, target, banned, include_db):
    """Run one strategy document: its phases, in order, under one budget.

    The goal check sits at the top of every iteration: the run returns
    the moment it lands on the target. It stops at the goal, at a fixed
    point (a phase cannot move), at the budget, and returns (steps,
    final) EVEN WHEN it did not reach the goal -- ladder X keeps that
    route and records the gap in steps.py. None only when nothing
    happened (fewer than two lines).
    """
    state = _SolveState(expr, banned, to_latex_as_written(expr))
    steps = [_initial_step(expr)]
    current = expr
    budget = STRATEGY_BUDGET
    # No goal (D1(a)): the run simply executes every phase -- there is
    # no exact form to stop on.
    target_latex = to_latex(target) if target is not None else None
    # A doc is either {"phases": [...]} or the bare phase list a
    # strategy row stores in its `steps` column.
    phases = doc.get("phases") if isinstance(doc, dict) else doc

    for phase in phases or []:
        do = phase.get("do")
        scope = phase.get("scope", "whole")
        until = phase.get("until", "stuck")

        while budget > 0:
            if target_latex is not None and to_latex(current) == target_latex:
                return steps, current    # goal check at the top: we arrived

            if do == "walk":
                move = _walk_phase(current, state, include_db)
            else:
                move = _phase_move(current, do, scope, state, include_db)
            if move is None:
                break                    # fixed point: this phase is done

            new_expr, entries, rows = move
            if not entries:
                break                    # nothing to show: never spin
            budget -= 1
            if state.revisits(new_expr):
                break                    # belt and braces: no shown repeat

            steps.extend(entries)
            state.record(new_expr, steps[-1]["latex"])
            for row in rows:
                _bump(row)
            current = new_expr

            if until == "once":
                break

        if target_latex is not None and to_latex(current) == target_latex:
            return steps, current
        if budget <= 0:
            break                        # budget spent: keep what we have

    if len(steps) < 2:
        return None                      # the strategy produced nothing
    return steps, current


def matching_strategies(expr) -> list:
    """Every strategy that applies here.

    The union of the module-level STRATEGIES list (tests monkeypatch it,
    so the seam stays) and the database rows of every form in _FORMS
    whose matcher accepts `expr`. A row wraps into a Strategy whose run
    loads its document and hands it to _run_phases (the doc is bound as
    a default argument, so late binding can never swap it out).
    """
    strategies = list(STRATEGIES)

    for form, matcher in _FORMS.items():
        try:
            if not matcher(expr):
                continue
            rows = knowledge_store.find_strategies(
                knowledge_store.get_connection(), form)
        except Exception:
            continue      # no database: the module list still works
        for row in rows:
            try:
                doc = json.loads(row["steps"])
            except Exception:
                continue  # a broken row never breaks a solve
            strategies.append(Strategy(
                row["name"], matcher,
                lambda expr, target, banned, include_db, doc=doc:
                    _run_phases(doc, expr, target, banned, include_db)))
    return strategies


# The module-level seam: the shipped strategies live in the database
# (data/samples/strategies.jsonl); tests may swap this list in.
STRATEGIES = []
