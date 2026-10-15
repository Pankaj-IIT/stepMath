"""
derive.py -- rearrangements we derive from the formulas we already have.

We do not store every rearrangement of every identity. Instead, when the
solver is stuck, it takes a STORED identity and soundly rewrites it, the
way a student rearranges a formula on paper:

    swap        A = B                ->  B = A
    isolate     A + B = C            ->  A = C - B        (an addend)
                A * B = C            ->  A = C / B        (a factor)
                (sides are FACTORED first, so  sec^2 - tan^2 = 1
                 becomes (sec-tan)(sec+tan) = 1 and then yields
                 the textbook rewrite  sec - t = 1/(sec+t) )

Every derived variant must PASS THE SymPy GATE before it may be used --
we only "think" with proven steps, never with guesses. Variants are
memoised per (left, right), so each identity is expanded at most once.
"""

from sympy import Add, Expr, Mul, factor

from ..algebra.latex import to_latex
from ..knowledge import gate, store
from ..knowledge.patterns import _evaluated

# memo: (left srepr, right srepr) -> [variants]
_MEMO: dict = {}
_MEMO_LIMIT = 4000

# Never derive more than this many variants from one identity.
_MAX_VARIANTS = 8


def derive_variants(left: Expr, right: Expr) -> list:
    """Proven rearrangements of the identity  left = right.

    Each variant: {"left": expr, "right": expr, "statement": "lhs = rhs"}.
    """
    key = (srepr_of(left), srepr_of(right))
    if key in _MEMO:
        return _MEMO[key]

    variants: list = []

    def offer(new_left, new_right):
        """Accept only variants SymPy can prove (verdicts are cached on
        disk, so each variant is proved once ever -- not once per run)."""
        if len(variants) >= _MAX_VARIANTS:
            return

        key = _proof_key(new_left, new_right)
        verdict = store.cached_verdict(key)
        if verdict is None:
            try:
                verdict, _detail = gate.verify_identity(new_left, new_right)
            except Exception:
                verdict = "unknown"
            store.remember_verdict(key, verdict)

        if verdict != "pass":
            return  # not proven -> never used
        if srepr_of(new_left) == srepr_of(left) and \
                srepr_of(new_right) == srepr_of(right):
            return  # nothing changed
        variants.append({
            "left": new_left,
            "right": new_right,
            "statement": f"{to_latex(new_left)} = {to_latex(new_right)}",
        })

    # 1) the flipped direction
    offer(right, left)

    # 2) isolate one addend:  A + B = C  ->  A = C - B
    if left.is_Add:
        for index, term in enumerate(left.args):
            rest = Add(*[part for i, part in enumerate(left.args)
                         if i != index], evaluate=False)
            offer(term, _evaluated(right - rest))

    # 3) factor a side, then isolate one factor:  A * B = C  ->  A = C / B
    for side, other in ((left, right), (right, left)):
        factored = factor(side)
        if not factored.is_Mul or len(factored.args) < 2:
            continue  # nothing to isolate
        for index, part in enumerate(factored.args):
            rest = Mul(*[p for i, p in enumerate(factored.args)
                         if i != index], evaluate=False)
            if side is left:
                offer(part, _evaluated(other / rest))
            else:
                # the factor sits on the right:  C = A * B  ->  A = C / B
                offer(_evaluated(other / rest), part)

    if len(_MEMO) < _MEMO_LIMIT:
        _MEMO[key] = variants
    return variants


def srepr_of(expr: Expr) -> str:
    """Structural fingerprint used as the memo key."""
    from sympy import srepr
    return srepr(expr)


def _proof_key(left: Expr, right: Expr) -> str:
    """Cache key for the proof of one variant (stable across processes)."""
    import hashlib
    payload = f"{srepr_of(left)} = {srepr_of(right)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
