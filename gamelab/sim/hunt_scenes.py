"""Where she hunts: the journeys of the tours of 2026-11-05 put in the world (tests/fixtures/hunt_journeys_2026-11-05.json, README: "Where she
hunts"), each from a stop's fight or a stop's end where it set off, with the monsters the perception service's looks had on the way, the piles it went
for, her round and its rate at take weight 0.5 frozen, nobody else and nothing back within it. She decides the rest: whether the journey
goes as it went live, or (behavior.targeting.stay_or_go) where a landing inside its chain calls for her.

Not in scenarios.SCENARIOS: no golden, no effect on the others (as sim.no_hit_scenes)."""

import dataclasses
import json
import time
from pathlib import Path

from ..attack import danger
from ..attack.spots import Spot
from ..behavior.hunt import Visit
from ..behavior.ledger import Pile
from ..behavior.patrol import Plan

JOURNEYS = Path(__file__).parents[2] / "tests" / "fixtures" / "hunt_journeys_2026-11-05.json"
WORLD = ("zero-walls", "near-misses", "walkers", "spawn-fade", "map-drops", "drop-takers", "near-misses-exact", "live-floors")  # (the thief's)
AFTER_S = 8.0  # s of a scene past what the journey took live


def cases() -> dict[str, dict]:
    return {c["name"]: c for c in json.loads(JOURNEYS.read_text(encoding="utf-8"))["cases"]}


def scenario_of(case: dict, seed: int = 1):
    """Journey `case` in world `seed`: the monsters on the platforms it passed, and where it's after a target that target's platform's."""
    from ..behavior.grind import Grind
    from . import harness, scenarios
    her, seconds = harness.WORLDS["live-floors"], case["seconds"] + AFTER_S
    base = harness.scenario("loop-grind", seed, "crowd", seconds / 60, 0.0, scenarios.TREE, WORLD, (), "choice")
    pile, round_ = case["kind"] == "loot", case.get("stop") or case["round"] or (case["to"][0],)
    keep = {*case["passed"], *(() if case["target"] is None else (case["target"]["on"],))}

    def run(sim):
        h = sim.hunter(potion_hp=0.45, her=danger.Her(her["max_hp"], her["level"], her["defence"]))
        g = Grind(sim.tour(h, sim.navigator(h, fight=True)), log=sim.log)
        p, (here, x) = g.patrol, case["start"]
        p.plan = Plan(tuple(p.stop_on(q) for q in round_), (case["rate"] or 0.0) / 60.0, 60.0, 30.0, 30.0, 0.0, 0)
        p.replan = lambda t, force=False: None
        own = g._next_ground

        def at_a_stop(s, heading=None):  # (her first look: at a stop she has got to, none of the run's start)
            g._next_ground = own
            g.visit = Visit(here, Spot(here, x, 1.0, sim.model.safe(here)))
            g.visit.arrived = time.monotonic()
            g.stop, g.obj = p.plan.stops[0], None
        g._next_ground = at_a_stop
        if pile:
            q, at = case["to"]
            h.drops.piles.append(Pile(at, sim.model.platform(q).y_at(at), q, time.monotonic(), hers=True, mesos=True, status="seen", last_seen=time.monotonic(), xs=(at,),
                                      items=("mesos",), worth=30.0, dearest=30.0, each=(30.0,)))
        return str(g.run(minutes=seconds / 60))
    return dataclasses.replace(base, name=f"journey-{case['name']}:{seed}", fn=run, start=tuple(case["start"]), limit=seconds, monsters=[tuple(m) for m in case["mobs"] if m[1] in keep],
                               drops=(("mesos", *case["to"]),) if pile else (), realism=dataclasses.replace(base.realism, camps=(), respawn_tick=0.0, respawn=600.0))


def run_journey(case: dict, seed: int = 1) -> dict:
    """Journey `case` run as harness.run_one a hunt: the platforms she stood on in turn (`path`), the platform of her first attack (`first_attack`), the
    landing stops of her chains (`asked`), whether she got to where the journey went (`got_there`), her kills and touches, and the log."""
    from .. import telemetry
    from . import fakes, harness, scenarios
    costs, before = harness.COSTS["measured"], {}
    try:
        for k, v in costs.items():
            before[k] = getattr(fakes, k)
            setattr(fakes, k, v)
        with telemetry.captured() as records:
            trace, sim = scenarios.run_sim(scenario_of(case, seed))
    finally:
        for k, v in before.items():
            setattr(fakes, k, v)
    path, first, at = [], None, None
    for r in records:
        if r.get("topic") == "scene" and (h := r.get("her") or {}).get("platform") is not None and h.get("mode") in ("ground", None):
            at = h["platform"]
            if not path or path[-1] != at:
                path.append(at)
        elif r.get("topic") == "attack" and first is None:
            first = at
    return {"path": path, "first_attack": first, "got_there": case["to"][0] in path, "kills": sum(k[3] == "her" for k in sim.world.kill_log),
            "asked": sum("at a landing" in x for x in trace["log"]), "touches": sim.world.touches, "log": trace["log"]}
