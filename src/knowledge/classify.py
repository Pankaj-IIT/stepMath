"""
classify.py -- reads one node of the expression tree and says which
formula FAMILIES it belongs to.

This is how "the tree identifies the group": a node is tagged by what the
maths MEANS, not by one rigid template -- the same family is recognised in
every spelling it can have:

    sin(x + y)                    -> two-angle-sum   (compact spelling)
    sin(x)*cos(y) + sin(y)*cos(x) -> two-angle-sum   (expanded spelling)
    sin(2*x)                      -> double-angle
    sin(x/2)                      -> half-angle      (2x and x/2 differ!)
    sin(x)^2 + cos(x)^2           -> pythagorean

At import the SAME classifier runs on each formula's "left" pattern, so
membership is always consistent -- nobody has to tag anything by hand.
Holes (?a) behave like coefficients/angles, so patterns classify too.

classify() must be CHEAP: it looks only at the node itself (the solver
already visits every child), and every check bails out early.
"""

from sympy import Add, Expr, Mul, Rational, Wild, cos, csc, cot, pi as _PI, \
    sec, sin, tan

from .patterns import _evaluated

# The families we know. Kept here (not in the database) because the
# classifier code defines them -- the descriptions can feed the UI later.
GROUPS = {
    "two-angle-sum": "a trig function of a sum/difference: sin(a+b), "
                     "and its expanded form sin(a)cos(b) + cos(a)sin(b)",
    "double-angle": "a trig function of twice an angle: sin(2x), cos(2x)",
    "half-angle": "a trig function of half an angle: sin(x/2)",
    "multiple-angle": "a trig function of 3x, 4x, ...",
    "pythagorean": "the squared pairs: sin^2+cos^2, sec^2-tan^2, csc^2-cot^2",
    "sum-to-product": "a sum of two bare trig functions: sin(C) + sin(D)",
    "product-to-sum": "a product of trig functions: 2*sin(A)*cos(B)",
    "linear-sin-cos": "a*sin(t) + b*cos(t) (the auxiliary-angle family)",
}

_TRIG = (sin, cos, tan, cot, sec, csc)
_PYTHAGOREAN_PAIRS = ({sin, cos}, {sec, tan}, {csc, cot})


def classify(expr: Expr) -> set:
    """Tags for this node (see GROUPS). Empty set = unknown family."""
    tags: set = set()

    if expr.func in _TRIG and expr.args:
        _classify_function(expr, tags)
    elif expr.is_Add:
        _classify_sum(expr, tags)
    elif expr.is_Mul:
        _classify_product(expr, tags)

    return tags


def _classify_function(expr: Expr, tags: set) -> None:
    """sin(...) etc.: compact spellings -- function of a sum, 2x, x/2 ..."""
    if len(expr.args) != 1:
        return
    arg = expr.args[0]

    if isinstance(arg, Add):
        # sin(x + y) is a two-angle sum. Constants involving pi
        # (sin(x + pi/2), sin(x + 2*pi)) get NO family tag: SymPy reduces
        # those directly via the "Reduce the angle" rule, so there are no
        # formulas for them to belong to.
        if not any(part.has(_PI) for part in arg.args):
            tags.add("two-angle-sum")
        return

    if not isinstance(arg, Mul):
        return

    # A single angle times a number: 2*x, x/2, 3*x ...
    numbers, others = [], []
    for factor in flatten(arg):
        if factor.is_number:
            numbers.append(factor)
        else:
            others.append(factor)
    if len(others) != 1 or not numbers:
        return  # not "k * angle" (e.g. x*y)

    # Evaluate for real: with our unevaluated parser x/2 holds an
    # unevaluated Pow(2,-1), whose repr says 1/2 but is not yet a Rational.
    k = _evaluated(Mul(*numbers))
    if k == 2:
        tags.add("double-angle")
    elif k.is_Float and k == 0.5:      # typed as 0.5*x
        tags.add("half-angle")
    elif isinstance(k, Rational) and abs(k.p) == 1 and k.q >= 2:
        tags.add("half-angle")
    elif isinstance(k, Rational) and abs(k.p) >= 3 and k.q == 1:
        tags.add("multiple-angle")


def flatten(expr: Expr) -> list:
    """All factors of a (possibly nested, unevaluated) product.

    The one place that opens up a product -- shared with auxiliary.py, so
    "-1 * (4 * cos(x))" is read the same way everywhere.
    """
    if not expr.is_Mul:
        return [expr]
    factors = []
    for part in expr.args:
        factors.extend(flatten(part))
    return factors


def _trig_term(term: Expr):
    """Describe one summand:  (coefficients, [(fn, angle), ...]).

    Coefficients may be numbers OR holes (?a) -- patterns classify too.
    Returns None when the term is not made of trig functions and numbers.
    """
    numeric = []
    trig = []
    for factor in flatten(term):
        if factor.is_number or isinstance(factor, Wild):
            numeric.append(factor)
        elif factor.func in _TRIG and len(factor.args) == 1:
            trig.append((factor.func, factor.args[0]))
        else:
            return None

    coefficient = Mul(*numeric) if numeric else None
    return coefficient, trig


def _classify_sum(expr: Expr, tags: set) -> None:
    """Add nodes: expanded spellings, bare sums, squared pairs."""
    # Pythagorean first: its terms are POWERS, which _trig_term rejects,
    # so it must not depend on the term description below.
    squared = _squared_functions(expr)
    for pair in _PYTHAGOREAN_PAIRS:
        if pair <= squared:
            tags.add("pythagorean")
            break

    terms = [_trig_term(term) for term in expr.args]
    if any(term is None for term in terms):
        return  # ordinary terms (numbers, x-products) present

    # --- a sum/difference of TWO bare trig functions: sin(C) + sin(D) ---
    #     and the auxiliary-angle pair: a*sin(t) + b*cos(t)
    if len(terms) == 2 and all(len(trig) == 1 for _, trig in terms):
        functions = {trig[0][0] for _, trig in terms}
        if len(functions) == 1:
            tags.add("sum-to-product")
        elif functions == {sin, cos}:
            tags.add("linear-sin-cos")

    # --- expanded two-angle-sum: sin(a)cos(b) +/- cos(a)sin(b) ---
    if len(terms) == 2 and all(len(trig) == 2 for _, trig in terms):
        if _expanded_two_angle(terms):
            tags.add("two-angle-sum")


def _expanded_two_angle(terms: list) -> bool:
    """Do the two product terms spell out  sin(a)cos(b) +/- cos(a)sin(b)?

    (or the cosine variant). The angles of term 1 must reappear SWAPPED in
    term 2 -- that crosswise match distinguishes it from a random sum of
    products.
    """
    (_coeff1, trig1), (_coeff2, trig2) = terms
    map1 = {fn: angle for fn, angle in trig1}
    map2 = {fn: angle for fn, angle in trig2}
    if len(map1) != 2 or len(map2) != 2:
        return False
    if set(map1) != {sin, cos} or set(map2) != {sin, cos}:
        return False
    return (map1[sin] == map2[cos]) and (map1[cos] == map2[sin])


def _squared_functions(expr: Expr) -> set:
    """Function names appearing squared among this sum's terms.

    Signs and coefficients are stripped first (via flatten), so
    -2*sin(x)^2 counts just like sin(x)^2.
    """
    found = set()
    for term in expr.args:
        parts = [part for part in flatten(term) if not part.is_number]
        if len(parts) != 1:
            continue  # not "one thing, maybe signed"
        core = parts[0]
        if getattr(core, "is_Pow", False) and core.exp == 2 \
                and core.base.func in _TRIG:
            found.add(core.base.func)
    return found


def mentioned_functions(expr: Expr) -> set:
    """Names of the trig functions this expression uses (as strings).

    Feeds the function index: a node containing sin() can only be helped
    by formulas whose left side also mentions sin() -- that is how the
    thinker finds candidate rows even when their SHAPE differs from the
    node (e.g. deriving  1 - sin^2 = cos^2  from the Pythagorean row
    while looking at the node  1 - sin^2).
    """
    found = set()
    pending = [expr]
    while pending:
        node = pending.pop()
        if node.func in _TRIG and len(node.args) == 1:
            found.add(node.func.__name__)
        pending.extend(node.args)
    return found


def _classify_product(expr: Expr, tags: set) -> None:
    """Mul nodes: products of trig functions (product-to-sum family)."""
    described = _trig_term(expr)
    if described is None:
        return
    _coefficient, trig = described
    if len(trig) == 2 and {fn for fn, _angle in trig} == {sin, cos}:
        tags.add("product-to-sum")
