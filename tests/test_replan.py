"""Replanning at every state while a chain runs (control.navigator.REPLAN, 2026-10-30; the user: "ideally we probably want
to regenerate motion plan at every frame so any hit/new spawn will cause motion plan to react immediately"): a monster
coming into a leg still to come replanned within a state and spliced in, no key let go; after a touch the next chain
planned at once from where the knock lands her; better ways switched to with hysteresis, never to the same way; where it
takes over from (a flight, a takeoff's key gone); the magician's casts from the ground only kept."""

import dataclasses

import pytest

from gamelab.control import navigator as nv
from gamelab.localization.mapdata import Platform, Segment
from gamelab.motion.bodies import Arc
from gamelab.motion.flow import Body, Leg, Span
from gamelab.motion.physics import JUMP_LAG, STOP_S
from gamelab.motion.route import Move
from gamelab.sim import harness, scenarios
from gamelab.sim.world import Realism

REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data")


def _platforms():
    a, b = Platform(1, [Segment(0, 500, 400, 500)]), Platform(2, [Segment(450, 440, 800, 440)])
    model = type("M", (), {"platform": staticmethod(lambda pid: {1: a, 2: b}[pid])})()
    return a, b, model


def _running(legs):
    return nv._Running("p", legs, 100.0, False)


def test_the_splice_point_is_where_she_runs_or_where_a_flight_or_a_takeoff_whose_key_went_lands_her():
    """Navigator._splice_point: on the ground in a run, the run's place and its arrow held (Body.run) at its speed; in a
    jump's flight, or from its takeoff's key on (JUMP_LAG before the lift-off) less KEY_COMMIT_S (the game has it before
    new keys could act), where it lands her, its end; a walk-off's from STOP_S before she leaves the edge (her stop's
    slide); standing in a leg, there; past the plan's end, none."""
    a, b, model = _platforms()
    nav = nv.Navigator.__new__(nv.Navigator)
    nav.model = model
    run = Span(1, 0.0, 1.0, 100.0, 249.0)
    jump = Leg(Move("jump", 1, 2, 249.0, 500.0, "right"), Body(0.0, 100.0, a), Body(1.6, 500.0, b, run=1, landed=True),
               [run], [Arc(1.0, 1.5, 249.0, 500.0, 480.0, 440.0, -555.0)], stands=[Span(2, 1.5, 1.6, 480.0, 500.0)])
    r = _running([jump])
    t, body = nav._splice_point(r, 0.5)
    assert t == 0.5 and body.t == 0.0 and body.platform is a and body.run == 1
    assert body.x == pytest.approx(174.5) and body.v == pytest.approx(149.0)
    off = 1.0 - JUMP_LAG - nv.KEY_COMMIT_S  # the takeoff's key is the game's from here on
    assert nav._splice_point(r, off - 0.01)[0] == pytest.approx(off - 0.01)  # (still on the run)
    for t in (off + 0.01, 1.2, 1.55):  # its key gone, in the air, landing: from where it lands her
        at, body = nav._splice_point(r, t)
        assert at == 1.6 and body.t == 0.0 and body.platform is b and body.x == 500.0 and body.landed
    assert nav._splice_point(r, 1.7) is None
    walk = Leg(Move("walk_off", 1, 2, 400.0, 460.0, "right"), Body(0.0, 300.0, a), Body(1.4, 470.0, b, run=1),
               [Span(1, 0.0, 0.7, 300.0, 400.0)], [Arc(0.7, 1.3, 400.0, 500.0, 460.0, 440.0, 0.0)])
    assert nav._splice_point(_running([walk]), 0.7 - STOP_S - nv.KEY_COMMIT_S + 0.01)[0] == 1.4
    assert nav._splice_point(_running([walk]), 0.2)[0] == 0.2


def test_the_same_way_is_the_same_moves_of_the_route_wherever_they_take_off():
    """navigator._same_moves: a replan whose route goes the running one's way (each move's kind, from and onto, as far
    as both go) is the same way, however far apart its takeoffs (the flow's variety moves them from chain to chain):
    on the user's own play a replan one state later took the same route 99.2% of the time."""
    a = Move("jump", 1, 2, 240.0, 500.0, "right")
    assert nv._same_moves([a, Move("drop", 2, 3, 600.0, 610.0, "")], [Move("jump", 1, 2, 300.0, 520.0, "right")])
    assert not nv._same_moves([a], [Move("walk_off", 1, 2, 400.0, 460.0, "right")])


def _run(sc, fn):
    return scenarios.run_sim(dataclasses.replace(sc, fn=fn))


def _evade_layout(**kw):
    return scenarios.Scenario("replan touch", None, (21, 60.0), monsters=[("Green Mushroom", 21, 130.0)], seed=2, limit=60,
                              map_id=scenarios.TREE, attack="choice", her=scenarios.SCENARIOS["nav_tree_evade"].her,
                              realism=Realism(zero_walls=True, chase=harness.CHASE_TREE,
                                              invulnerable=harness.INVULNERABLE_LIVE), **kw)


@REAL
def test_a_touch_mid_chain_is_replanned_at_once_from_where_the_knock_lands_her(monkeypatch):
    """KNOCK_REPLAN: a chain cut by a touch (nav_tree_evade's: a Green Mushroom walking at her on 21) has the next
    chain planned from the first state that has her on the ground again, not after she has stood still and a look of
    its own (recover): it goes out within a few states of the knock's landing (KNOCK_AIR_S after the popup), sooner than
    with KNOCK_REPLAN off, and no `recover` is asked; she's on a platform where it starts."""
    calls = []

    def go(sim):
        nav = sim.navigator(sim.hunter())
        real = nav.recover
        nav.recover = lambda *a, **k: calls.append(sim.clock.now) or real(*a, **k)
        return nav.go(19, seconds=60)

    def touch_to_next(trace):
        log = trace["log"]
        i = next(i for i, line in enumerate(log) if "plan cut short" in line and "touched" in line)
        cut_at = float(log[i].split()[0])
        sent = [p[0] for p in trace["plans"] if p[1] == "flow" and p[0] > cut_at - 1e-6]
        return cut_at, log[i], sent[0] - cut_at

    trace, sim = _run(_evade_layout(), go)
    cut_at, line, gap = touch_to_next(trace)
    assert "the knock over" in line and "replanning from there at once" in line
    assert not [t for t in calls if cut_at - 1e-6 <= t - sim.clock.start <= cut_at + gap]  # (no stand-still wait)
    assert gap <= nv.KNOCK_AIR_S + 3 * nv.STATE_S + 0.02, gap
    assert trace["result"]["result"] == "arrived"
    monkeypatch.setattr(nv, "KNOCK_REPLAN", False)  # as before: she stands still first, then a look
    trace_b, _ = _run(_evade_layout(), go)
    cut_b, line_b, gap_b = touch_to_next(trace_b)
    assert "the knock over" not in line_b and gap_b > gap


def _spawn_into_landing(state: dict, platform: int | None = None):
    """A navigator's follow wrapped so that, on the first chain whose run along her platform lasts 0.6 s or more, a
    Green Mushroom comes into being where its second leg lands (past the first one's), 0.3 s after the chain goes
    (World.call_at), while she still runs: a new spawn into a leg still to come, with a way short of it."""
    def wrap(sim, nav):
        real = nav.follow

        def follow(legs, keys, *a, **k):
            if not state and legs and legs[0].spans and legs[0].spans[-1].t1 >= 0.6:
                land = ([leg for leg in legs if leg.air or leg.rope][1:2] or [None])[0]
                if land is not None:
                    at = sim.clock.now + 0.3
                    state.update(at=at, x=land.end.x, platform=land.end.platform.id)

                    def spawn(world):
                        before = len(world.monsters)
                        world._spawn([("Green Mushroom", land.end.platform.id, land.end.x)])
                        for i, m in enumerate(world.monsters[before:]):
                            m.idx = before + i
                    sim.world.call_at(at, spawn)
            return real(legs, keys, *a, **k)
        nav.follow = follow
    return wrap


@REAL
def test_a_new_spawn_into_a_leg_is_replanned_within_a_state_and_spliced_with_no_key_let_go():
    """A monster that comes into being where a later leg of the running chain lands (a new spawn: no look the chain was
    planned from had it) is met within a state of its first sighting: the chain replanned from where she'll be when new
    keys can act, against the whole state, and spliced in (the fake player as the real one: client.splice), the
    running plan answered spliced, its held arrow not let go; nothing stops (no cut for it)."""
    state: dict = {}
    hook = _spawn_into_landing(state)

    def go(sim):
        nav = sim.navigator(sim.hunter())
        hook(sim, nav)
        return nav.go(6, seconds=90)  # (up the tower from the floor's right: a run along 33 first)

    sc = scenarios.Scenario("replan spawn", None, (33, 300.0), monsters=None, seed=1, limit=120, map_id=scenarios.TREE)
    trace, sim = _run(sc, go)
    assert state, "no chain with a later landing to spawn into"
    splices = trace["splices"]
    assert splices, trace["log"][:40]
    first = splices[0]  # [t, its index, the index of the plan it takes over from, at_ms, taken]
    assert state["at"] - sim.clock.start <= first[0] + 1e-6 <= state["at"] - sim.clock.start + 2 * nv.STATE_S + 0.05
    old = sim.player.order[first[2]]
    assert first[4] and old.state == "spliced"
    assert not any("plan cut short" in line and nv.COMING in line for line in trace["log"])
    assert any("replanned from" in line and nv.COMING in line for line in trace["log"])
    assert trace["result"]["replans"] >= 1


@REAL
def test_better_ways_are_switched_to_with_hysteresis_never_dithering(monkeypatch):
    """With no monster about, every replan goes the running chain's way (the same moves of the route) and no chain stops
    short: nothing is switched to. Two ways whose costs swap from one look to the next (the route planner's, made to
    alternate by 0.5 s, more than SWITCH_GAIN_S, and every replan taken for another way): a switch SWITCH_MIN_S after
    the last at the soonest, never one a state."""
    from gamelab.motion import route
    sc = scenarios.Scenario("replan dither", None, (6, 250.0), monsters=None, seed=1, limit=120, map_id=scenarios.TREE)

    def go(sim):
        return sim.navigator(sim.hunter()).go(33, seconds=90)
    trace, sim = _run(sc, go)
    assert trace["result"]["result"] == "arrived" and trace["result"]["replans"] == 0, trace["log"][:30]
    flips = {"n": 0}
    real = route.Planner.quickest

    def quickest(self, *a, **k):
        way = real(self, *a, **k)
        flips["n"] += 1
        return None if way is None else dataclasses.replace(way, cost=way.cost + (0.0 if flips["n"] % 2 else 0.5))

    def through(self, src, x, dst, via, *a, **k):
        way = real(self, src, x, dst, *a, **k)
        return None if way is None else dataclasses.replace(way, cost=way.cost + 0.5)
    monkeypatch.setattr(route.Planner, "quickest", quickest)
    monkeypatch.setattr(route.Planner, "through", through)
    monkeypatch.setattr(nv, "_same_moves", lambda a, b: False)
    trace, sim = _run(sc, go)
    at = [sp[0] for sp in trace["splices"]]
    assert at, "no switch at all"
    gaps = [b - a for a, b in zip(at, at[1:])]
    assert all(g >= nv.SWITCH_MIN_S - 1e-6 for g in gaps), gaps


@REAL
def test_her_casts_go_from_the_ground_only_and_replanned_chains_carry_her_moves_alone():
    """The magician's Magic Claw from the ground only (Actions.ground_only, the user's "no mid-jump casts": no attack
    key in a plan with a jump) holds with chains replanned as they run: every plan of the navigator's, sent or spliced,
    carries her moves' keys and her pick-up alone, never an attack's, and no plan of the run has an attack key and a
    jump together (her fights' throws from the ground, as before: whether she's on the ground when one is keyed is
    theirs, unchanged, on and off alike)."""
    def go(sim):
        make = sim.hunter

        def hunter(**kw):
            h = make(**kw)
            h.actions.ground_only = True  # (the magician's: Hunter's spell without `air`)
            return h
        sim.hunter = hunter
        return scenarios.SCENARIOS["loop_tree_23"].fn(sim)

    sc = dataclasses.replace(scenarios.SCENARIOS["loop_tree_23"], limit=90.0, name="replan casts")
    trace, sim = _run(sc, go)
    k = sim.keys
    moves = {k.left.name, k.right.name, k.jump.name, k.up.name, k.down.name, k.pick_up.name}
    attacks = {key.name for key in (k.attack, getattr(k, "lucky_seven", None), getattr(k, "magic_claw", None)) if key}
    flows = [p for p in trace["plans"] if p[1] == "flow"]
    assert flows and all({e[1] for e in p[2]} <= moves for p in flows)
    keyed = [{e[1] for e in p[2]} for p in trace["plans"]]
    assert any(ks & attacks for ks in keyed)  # (she fought)
    assert not any(ks & attacks and k.jump.name in ks for ks in keyed)
