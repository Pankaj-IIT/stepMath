"""
knowledge package -- our formula/trick database.

One SQLite file (data/knowledge.db) holds:
    * identities   -- single-step rewrites like  sin(?a)^2 + cos(?a)^2 = 1
    * methods      -- multi-step tricks stored as ONE self-contained row
                      (all their steps inline, never looked up elsewhere)

What each file does:

    signature.py   the "shape" of an expression -- how we index 10M rows
    patterns.py    reading patterns like  ?a*x  and matching them
    gate.py        SymPy proof-checks a formula before we trust it
    ops.py         the small catalogue of ready-made SymPy operations
                   (expand, factor, simplify, ...) that methods can call
    runner.py      runs a method's steps one at a time
    store.py       all the SQLite reading and writing
"""
