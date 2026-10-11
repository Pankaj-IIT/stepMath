"""
algebra package -- the maths brain of the app.

What each file is for:

    parser.py   your typed text -> a SymPy object
    latex.py    a SymPy object  -> LaTeX text the browser can draw
    rules.py    the list of algebra tricks we know
    steps.py    runs the rules over and over to build the step list
    solver.py   step-by-step solving of simple equations

The React app never talks to these files directly -- it asks the small web
server in api/main.py, and that server calls into this package.
"""
