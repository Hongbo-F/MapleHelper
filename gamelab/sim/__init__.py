"""An offline stand-in for the game, to run gamelab's navigation, fighting, looting and tours without it.

world.py simulates the game on a real map; fakes.py puts it behind the seams gamelab already has (Eyes,
gamelab.control.client, time); scenarios.py runs gamelab's entry points on it; golden.py records their
traces and checks the code still does the same (`python -m gamelab.sim.golden record|check`). Nothing
here sends a key to the game or writes under gamelab/lab during a run: the stack runs only inside
`with scenarios.Sim(model, scenario) as sim:`, where the fakes are in place, and opening any file under
gamelab/lab raises SimViolation. (A telemetry file opened before the run, as `python -m gamelab sim`
does, keeps receiving records for the mini-map.)
"""
