"""Motion planning: getting from A to B. How the game moves her, step by step (body), her physics and
where a jump or fall lands (physics), the moves that link platforms and the quickest route over them
(route), and several moves turned into one continuous key timeline (flow). The closed loops that carry
these plans out are control.navigator and control.mover.

May import: localization, and the shared modules.
"""
