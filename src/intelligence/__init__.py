"""
intelligence package -- the layer that READS the question and picks a
direction.

    read_question(text)  ->  {intent, payload, explanation}   (reading.py)
    dispatch(reading)    ->  {steps, final, ...}              (handlers.py)

Right now the reading is fully deterministic (no AI): a handful of clear,
offline rules decide what kind of question was asked, and the explanation
says WHY, so the app can always show "Reading this as: PROVE -- ...".
An AI reader can be slotted in behind read_question() later; its output
would be validated by the same parser and gate as everything else.
"""
