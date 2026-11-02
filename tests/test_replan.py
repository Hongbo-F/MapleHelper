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
from gamelab.sim.world import HER_HALF, HER_HEIGHT, Realism

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


def test_the_splice_point_holds_the_arrow_the_running_plans_keys_hold_not_the_way_she_slides():
    """Its review's fix (2026-10-30): a splice point on a span she slides along after letting go of the arrow (a
    landing's run-out) was taken for a run that way (the span's dx), the new chain spliced in holding that arrow, and
    its keys, made as if no arrow were held, never let it go: up the tower from 13 she ran off 11's end and out of the
    map. The arrow is the running plan's keys' (_Running.arrows: _arrow_at), and a chain spliced in starts with it held
    (flow.chain's `held`), letting it go where it doesn't run on."""
    a, b, model = _platforms()
    nav = nv.Navigator.__new__(nv.Navigator)
    nav.model = model
    slide = Span(1, 0.6, 0.9, 200.0, 186.0)  # (a landing's run-out leftward, no arrow held from 0.64 s)
    leg = Leg(Move("jump", 1, 1, 249.0, 200.0, "left"), Body(0.0, 300.0, a), Body(0.9, 186.0, a), [slide], [])
    held_left = ((0, -1, True), (640, -1, False), (893, 1, True))
    t, body = nav._splice_point(nv._Running("p", [leg], 100.0, False, arrows=held_left), 0.7)
    assert t == 0.7 and body.run == 0 and body.v == 0.0
    assert nav._splice_point(nv._Running("p", [leg], 100.0, False, arrows=((0, -1, True),)), 0.7)[1].run == -1
    assert nav._splice_point(_running([leg]), 0.7)[1].run == -1  # (no keys known: by the span, as before)
    assert [nv._arrow_at(held_left, t) for t in (0.0, 0.5, 0.7, 0.9)] == [-1, -1, 0, 1]
    assert nv._arrow_at(((0, -1, True), (100, 1, True)), 0.2) == 1  # (both held: the one pressed last)
    from gamelab.motion.flow import Keys
    k = Keys({"left": "left", "right": "right"})
    k.arrow = -1  # (a splice's start with left held: turning right lets it go first)
    k.set_arrow(0.25, 1)
    assert k.events == [[250, "left", "up"], [251, "right", "down"]]


def test_the_splice_point_on_a_run_from_standing_is_where_the_chains_physics_has_her(monkeypatch):
    """SPLICE_PHYSICS (the review of 2026-10-31): on a leg's run from its start with its arrow held all along, the splice
    point is where the chain's own physics has her (flow._ground: the arrow seen LAGS.on after its key, her speeding up
    from standing), not the run's span (a straight line at its mean speed for monster checks): at 0.06 s she hasn't
    moved yet, at 0.5 s she runs at full speed 149 (0.119 s behind a standing start's); the span had her 174.5 at 149
    there, 100 + 5 at 0.06. Off: the span's, as before; with no keys known, the span's too."""
    from gamelab.motion import body as engine
    from gamelab.motion.flow import LAGS, SHOE
    a, b, model = _platforms()
    nav = nv.Navigator.__new__(nv.Navigator)
    nav.model = model
    run = Span(1, 0.0, 1.0, 100.0, 249.0)
    jump = Leg(Move("jump", 1, 2, 249.0, 500.0, "right"), Body(0.0, 100.0, a), Body(1.6, 500.0, b, run=1, landed=True),
               [run], [Arc(1.0, 1.5, 249.0, 500.0, 480.0, 440.0, -555.0)], stands=[Span(2, 1.5, 1.6, 480.0, 500.0)])
    r = nv._Running("p", [jump], 100.0, False, arrows=((0, 1, True), (935, 1, False)))
    _, early = nav._splice_point(r, 0.06)
    assert early.x == pytest.approx(100.0) and early.v == pytest.approx(0.0) and early.run == 1
    _, mid = nav._splice_point(r, 0.5)
    assert mid.v == pytest.approx(149.0) and mid.x == pytest.approx(
        100.0 + engine.run_x(0.5 - LAGS.on, 0.0, 1, SHOE)) and mid.x < 174.5 - 15
    # A leg from a time whose press rounds up to the next whole ms (flow.ms: 0.4006 s, down at 401 ms): the arrow is
    # looked for as of that ms (sp.t0 + 1e-4 missed it, and the span's x was taken: the review of 2026-11-01).
    from gamelab.motion.flow import _ground, ms
    for t0 in (0.4, 0.4003, 0.4006, 0.4009):
        start = Body(t0, 100.0, a)
        x1 = _ground(start, 1, t0 + 1.0)[0]
        leg = Leg(Move("walk", 1, 1, 100.0, x1, "right"), start, Body(t0 + 1.0, x1, a, run=1),
                  [Span(1, t0, t0 + 1.0, 100.0, x1)], [])
        _, body = nav._splice_point(nv._Running("p", [leg], 100.0, False, arrows=((ms(t0), 1, True),)), t0 + 0.3)
        assert body.x == pytest.approx(_ground(start, 1, t0 + 0.3)[0]) and body.run == 1
    monkeypatch.setattr(nv, "SPLICE_PHYSICS", False)
    assert nav._splice_point(r, 0.5)[1].x == pytest.approx(174.5)


def test_after_a_touch_the_next_chain_waits_for_the_knocks_slide_to_end(monkeypatch):
    """KNOCK_STILL (the review of 2026-10-31): the knock lands her sliding (~140 units a second, braking), and a chain
    planned from the first state on the ground has her standing: 18 of 28 trip landings elsewhere than planned in the
    harness's standard grind came right after one. Navigator._knock_landing takes the first state on the ground within
    KNOCK_STILL_DX of the last other state on the ground (the same state glanced again isn't one), from KNOCK_AIR_S
    after the touch; off, the first on the ground."""
    from types import SimpleNamespace as NS
    # (states 15 a second from the touch at 0: in the air, then down at x 100 sliding 9, 5, 2 units a state, then still)
    xs = [None, None, None, 100.0, 109.0, 114.0, 116.0, 116.0, 116.0]
    clock = {"t": 0.0}
    looked = []

    class Senses:
        glanced = None

        def wait_frame(self, s):
            clock["t"] += 1.0 / 30  # (a glance every half state: each state glanced twice)

        def glance(self):
            i = min(int(clock["t"] * 15), len(xs) - 1)
            x = xs[i]
            self.glanced = NS(seq=i, t=i / 15, placed=True, mode="air" if x is None else "ground",
                              where=(x if x is not None else 90.0 + 5 * i, 400.0, None if x is None else 1, None))

        def look(self, mobs, wide, after):
            looked.append(after)
            return NS(where=NS(platform=1, x=self.glanced.where[0]), tag_seen=False, ground=None, t=after)

    nav = nv.Navigator.__new__(nv.Navigator)
    nav.senses, nav._mode = Senses(), nv.AVOID
    monkeypatch.setattr(nv.time, "monotonic", lambda: clock["t"])
    s = nav._knock_landing({"at": 0.0})
    assert s.where.x == 116.0 and looked == [6 / 15]  # (109 -> 114 is 5 units on, 114 -> 116 is 2: over)
    clock["t"], looked[:] = 0.0, []
    monkeypatch.setattr(nv, "KNOCK_STILL", False)
    assert nav._knock_landing({"at": 0.0}).where.x == 100.0 and looked == [3 / 15]


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
                              realism=Realism(zero_walls=True, chase=True, invulnerable=harness.INVULNERABLE_LIVE),
                              **kw)  # (the mushroom hit before the trip and chasing her, as nav_tree_evade's: go's aggro)


@REAL
def test_a_touch_mid_chain_is_replanned_at_once_from_where_the_knock_lands_her(monkeypatch):
    """KNOCK_REPLAN: a chain cut by a touch (nav_tree_evade's: a Green Mushroom walking at her on 21) has the next
    chain planned from the first state that has her on the ground again, not after she has stood still and a look of
    its own (recover): it goes out within a few states of the knock's landing (KNOCK_AIR_S after the popup), sooner than
    with KNOCK_REPLAN off, and no `recover` is asked; she's on a platform where it starts."""
    calls = []

    def go(sim):
        sim.world.monsters[0].aggro(sim.world.t)
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


def _spawn_run(wrap=None, realism=None, think: float = 0.0, reads: bool = False):
    """The spawn into a later landing (_spawn_into_landing) up the tower from the floor's right, `wrap(sim, nav)` on the
    navigator first (`think`, `reads`: the clock's, fakes.Clock): (trace, sim, the spawn's state)."""
    state: dict = {}
    hook = _spawn_into_landing(state)

    def go(sim):
        nav = sim.navigator(sim.hunter())
        hook(sim, nav)
        if wrap is not None:
            wrap(sim, nav)
        return nav.go(6, seconds=90)

    sc = scenarios.Scenario("replan spawn", None, (33, 300.0), monsters=None, seed=1, limit=120, map_id=scenarios.TREE,
                            realism=realism, think=think, think_reads=reads)
    trace, sim = _run(sc, go)
    assert state, "no chain with a later landing to spawn into"
    return trace, sim, state


@REAL
def test_a_monster_coming_she_cant_act_on_yet_is_asked_about_again_at_the_next_state():
    """The review of 2026-10-30: the monster a running chain's legs now meet was marked reacted-to (REACT_AGAIN_S) before
    `_switch` was asked, so where nothing could be done (in a flight, past a takeoff's key, not where the plan has her,
    and no room to stop) it went unheeded for 3 s, against _switch's "the next state asks again". Now it's marked only
    when acted on (a switch, a cut) or replanned: here the first ask is made to find nothing to do, and the next state
    asks about it again and splices."""
    asks: list = []

    def wrap(sim, nav):
        real_switch, real_room = nav._switch, nav._room_to_stop

        def switch(st, running, dst, why, kind):
            if kind == "monster":
                asks.append(sim.clock.now - sim.clock.start)
                if len(asks) == 1:
                    nav._no_room = True
                    return None  # (as in a flight: nothing replanned)
            return real_switch(st, running, dst, why, kind)

        def room(running, t):
            if getattr(nav, "_no_room", False):
                nav._no_room = False
                return False
            return real_room(running, t)
        nav._switch, nav._room_to_stop = switch, room
    trace, sim, state = _spawn_run(wrap)
    assert len(asks) >= 2 and asks[1] - asks[0] <= 2 * nv.STATE_S + 1e-6, asks
    assert any(sp[0] >= asks[1] - 1e-6 for sp in trace["splices"]), (asks, trace["splices"])


@REAL
def test_a_replan_done_too_late_for_its_splice_point_is_dropped():
    """The review of 2026-10-30: nothing checked a replan's deadline once it was computed, so one slower than about 108
    ms would have been spliced in past the running plan's next takeoff key (let go by the switch while the new plan has
    her running on). A replan the player would take more than SPLICE_LATE_S past its plan time 0 is dropped: said with
    what comes of it (for a monster coming her stop for it or the next state's ask, for a better way the next look's),
    nothing spliced for it (here every replan's chain made to take 0.15 s of the clock)."""
    def wrap(sim, nav):
        real_switch, real_chain = nav._switch, nav.flow.chain
        inside = {"on": False}

        def chain(*a, **k):
            if inside["on"]:
                sim.clock.sleep(0.15)
            return real_chain(*a, **k)

        def switch(*a, **k):
            inside["on"] = True
            try:
                return real_switch(*a, **k)
            finally:
                inside["on"] = False
        nav.flow.chain, nav._switch = chain, switch
    trace, sim, state = _spawn_run(wrap)
    late = [line for line in trace["log"] if "replanned too late" in line]
    assert late, trace["log"][:60]
    assert not trace["splices"]
    monster = [line for line in late if nv.COMING in line]
    again = f"dropped, asked about again {nv.LATE_AGAIN_S:.1f} s on (dropped late {nv.LATE_RETRIES + 1} times in a row)"
    assert monster and all(line.endswith(("dropped, she stops for it", "dropped, the next state asks again", again))
                           for line in monster), monster
    better = f"dropped, the next look for a better way asks again ({nv.BETTER_EVERY_S:.2f} s on)"
    assert all(" a better way" in line and line.endswith(better) for line in late if line not in monster), late


def _late_once(asks: list, room: bool | None = None, kind: str = "monster"):
    """A navigator wrap (_spawn_run's) whose first replan of `kind` to plan a chain takes 0.15 s of the clock there (too
    late: SPLICE_LATE_S), with room to stop or none right after it as `room` says (None: as it is); every ask of `kind`
    (its sim time, and whether it was the late one) in `asks`."""
    def wrap(sim, nav):
        real_switch, real_chain, real_room = nav._switch, nav.flow.chain, nav._room_to_stop
        on = {"inside": False, "slowed": False, "this": False, "room": None}

        def chain(*a, **k):
            if on["inside"] and not on["slowed"]:
                on["slowed"] = on["this"] = True
                sim.clock.sleep(0.15)
            return real_chain(*a, **k)

        def switch(st, running, dst, why, k):
            if k != kind:
                return real_switch(st, running, dst, why, k)
            on["inside"], on["this"] = True, False
            t = sim.clock.now - sim.clock.start
            try:
                return real_switch(st, running, dst, why, k)
            finally:
                on["inside"] = False
                asks.append((t, on["this"]))
                if on["this"]:
                    on["room"] = room

        def room_to_stop(running, t):
            forced, on["room"] = on["room"], None
            return real_room(running, t) if forced is None else forced
        nav.flow.chain, nav._switch, nav._room_to_stop = chain, switch, room_to_stop
    return wrap


@REAL
def test_a_monster_whose_replan_was_dropped_late_with_no_room_to_stop_is_asked_about_again_at_the_next_state():
    """The latency analysis of 2026-11-01: a "monster" replan dropped as too late left its monster marked replanned
    (`_replanned` set before the deadline's check) for REACT_AGAIN_S, so with no room to stop she ran on into it while
    the log said the next state asks again. Now a late one isn't taken as handled: said so, the next state asks about it
    again (here in time: spliced), and nothing cut."""
    asks: list = []
    # (her body for touches as when this was written, 15 either side: with the client's line, world.HER_TOUCH, a touch
    # it took on the trip's way up no longer comes, and a later chain is cut for another monster coming)
    trace, sim, state = _spawn_run(_late_once(asks, room=False), realism=Realism(her_box=(HER_HALF, HER_HEIGHT)))
    i = next(i for i, (_, late) in enumerate(asks) if late)
    assert len(asks) > i + 1 and asks[i + 1][0] - asks[i][0] <= 0.15 + 2 * nv.STATE_S + 1e-6, asks
    said = [line for line in trace["log"] if "replanned too late" in line]
    assert len(said) == 1 and nv.COMING in said[0] and said[0].endswith("dropped, the next state asks again"), said
    assert any(sp[0] >= asks[i + 1][0] - 1e-6 and sp[4] for sp in trace["splices"]), (asks, trace["splices"])
    assert not any("plan cut short" in line and nv.COMING in line for line in trace["log"])


def _late_always(asks: list):
    """A navigator wrap (_spawn_run's) whose every "monster" replan takes 0.15 s of the clock (too late:
    SPLICE_LATE_S) with no room to stop after it; every such ask (its sim time) in `asks`."""
    def wrap(sim, nav):
        real_switch, real_chain = nav._switch, nav.flow.chain
        on = {"inside": False}

        def chain(*a, **k):
            if on["inside"]:
                sim.clock.sleep(0.15)
            return real_chain(*a, **k)

        def switch(st, running, dst, why, k):
            if k != "monster":
                return real_switch(st, running, dst, why, k)
            on["inside"] = True
            asks.append(sim.clock.now - sim.clock.start)
            try:
                return real_switch(st, running, dst, why, k)
            finally:
                on["inside"] = False
        nav.flow.chain, nav._switch, nav._room_to_stop = chain, switch, lambda running, t: False
    return wrap


@REAL
def test_a_monster_whose_replans_keep_coming_late_is_asked_about_again_only_after_a_while():
    """The review of 2026-11-01, its finding 3: a monster whose replan is dropped as too late with no room to stop is
    asked about again at the next state LATE_RETRIES times in a row; dropped late again, it's marked for LATE_AGAIN_S
    (said so), and not asked about again before then: under the load that makes a replan late, every state paid a whole
    replan while the monster came."""
    asks: list = []
    trace, sim, state = _spawn_run(_late_always(asks))
    said = [line for line in trace["log"] if "replanned too late" in line and nv.COMING in line]
    assert said and said[0].endswith("dropped, the next state asks again"), said
    again = f"dropped, asked about again {nv.LATE_AGAIN_S:.1f} s on (dropped late {nv.LATE_RETRIES + 1} times in a row)"
    k = next(k for k, line in enumerate(said) if line.endswith(again))
    assert k == nv.LATE_RETRIES, said
    assert len(asks) > k + 1, asks
    assert asks[k] - asks[k - 1] <= 0.15 + 2 * nv.STATE_S + 1e-6  # (the retry: the next state's)
    assert asks[k + 1] - asks[k] >= nv.LATE_AGAIN_S - 1e-6, asks  # (then a while)
    assert not any("plan cut short" in line and nv.COMING in line for line in trace["log"])


@REAL
def test_a_monster_whose_replan_was_dropped_late_with_room_to_stop_has_her_stop_for_it():
    """... and with room to stop, the dropped replan's monster cuts the plan (REACT's cut, as without REPLAN): said as
    her stop for it, the cut for it next, and it's handled (not asked about again at the next state)."""
    asks: list = []
    trace, sim, state = _spawn_run(_late_once(asks, room=True))
    i = next(i for i, (_, late) in enumerate(asks) if late)
    log = trace["log"]
    j = next(j for j, line in enumerate(log) if "replanned too late" in line)
    assert nv.COMING in log[j] and log[j].endswith("dropped, she stops for it"), log[j]
    assert "plan cut short" in log[j + 1] and nv.COMING in log[j + 1], log[j:j + 2]
    assert not [t for t, _ in asks[i + 1:] if t - asks[i][0] <= 0.15 + 2 * nv.STATE_S + 1e-6], asks


@REAL
def test_a_player_without_splices_has_a_replanned_chain_cut_and_sent():
    """The elevated player live until it's restarted on this code takes no splices (client.can_splice; the simulator's
    Realism.splices False, the harness's `--world no-splices`): the spawn's replan cuts the running plan and sends the
    new chain at once, its keys from its plan time 0 on; she still gets there, nothing spliced."""
    trace, sim, state = _spawn_run(realism=Realism(splices=False))
    assert not sim.player.splices and not trace["splices"]
    assert any("replanned from" in line and "cut and sent" in line for line in trace["log"]), trace["log"][:60]
    assert trace["result"]["replans"] >= 1 and trace["result"]["result"] == "arrived"


def test_a_monster_standing_is_planned_either_way_a_while_and_one_set_off_since_isnt_as_planned():
    """The review of 2026-10-30: a monster in a stand or flinch frame was planned within TURN_SLACK and 20 units a second
    of its spot (45 units at 1 s), where one sets off at about 0.61 a second. Now as one whose heading isn't known (vx
    None), either way at its speed STAND_UNSURE_S at the most (MonsterTrack.unsure, Navigator._walking); and the chain's
    watch takes one it planned standing that walks now, or planned walking that turned or stopped, as not the one it
    planned against (_as_planned: REACT and the replan look at it again)."""
    from types import SimpleNamespace as NS

    from gamelab.localization.monsters import MonsterTrack
    a, b, model = _platforms()
    stands = MonsterTrack(1, 200.0, 100.0, 0, 30.0, None, 0.0, unsure=nv.STAND_UNSURE_S)
    assert stands.reach(0.5, a) == (150.0, 250.0) and stands.reach(3.0, a) == (100.0, 300.0)
    assert MonsterTrack(1, 200.0, 100.0, 0, 30.0, 0.0, 0.0).reach(1.0, a) == (155.0, 245.0)  # (as it was)
    nav = nv.Navigator.__new__(nv.Navigator)
    s = NS(tracks=[NS(id=7, vx=0.0), NS(id=8, vx=-100.0)])
    looked = [MonsterTrack(1, 200.0, 100.0, 1, 30.0, 100.0, 0.0), MonsterTrack(1, 300.0, 100.0, 0, 30.0, None, 0.0)]
    still, walks = nav._walking(s, looked, [NS(track=7), NS(track=8)])
    assert (still.vx, still.heading, still.unsure, still.track) == (None, 0, nv.STAND_UNSURE_S, 7)
    assert (walks.vx, walks.heading, walks.unsure) == (-100.0, -1, None)
    walking = MonsterTrack(1, 200.0, 100.0, 1, 30.0, 100.0, 0.0)
    assert nv._as_planned(stands, NS(vx=0.0)) and not nv._as_planned(stands, NS(vx=100.0))
    assert nv._as_planned(walking, NS(vx=100.0))
    assert not nv._as_planned(walking, NS(vx=-100.0)) and not nv._as_planned(walking, NS(vx=0.0))
    assert nv._as_planned(MonsterTrack(1, 200.0, 100.0, 0, 30.0, None, 0.0), NS(vx=100.0))  # (planned either way)


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
    # A better way's replan dropped as too late (its chain made to take 0.15 s of the clock): as before, nothing taken,
    # no monster marked; its next look BETTER_EVERY_S on, as the log says.
    marks, looks = [], []

    def go_late(sim):
        nav = sim.navigator(sim.hunter())
        real_switch, real_chain = nav._switch, nav.flow.chain
        inside = {"on": False}

        def chain(*a, **k):
            if inside["on"]:
                sim.clock.sleep(0.15)
            return real_chain(*a, **k)

        def switch(st, running, dst, why, kind):
            inside["on"] = kind == "better"
            looks.append(sim.clock.now - sim.clock.start)
            try:
                return real_switch(st, running, dst, why, kind)
            finally:
                inside["on"] = False
                marks.append(dict(nav._reacted))
        nav.flow.chain, nav._switch = chain, switch
        return nav.go(33, seconds=90)
    trace, sim = _run(sc, go_late)
    late = [line for line in trace["log"] if "replanned too late" in line]
    assert late and all(" a better way (" in line for line in late), late[:3]
    assert all(line.endswith(f"dropped, the next look for a better way asks again ({nv.BETTER_EVERY_S:.2f} s on)")
               for line in late), late[:3]
    assert not trace["splices"] and not any(marks)
    assert all(b - a >= nv.BETTER_EVERY_S - 1e-6 for a, b in zip(looks, looks[1:])), looks


@REAL
def test_with_the_clock_read_as_it_goes_a_slow_replan_is_late_by_its_own_computing(monkeypatch):
    """The harness's --think-clock (fakes.Clock.reads): the stack's computing charged at every read of the clock too, so
    a replan's own computing makes it late. The spawn's replans made to take 0.2 s of this computer's (its real clock
    faked, moving only while their chains are planned): dropped as too late, nothing spliced. With --think alone
    (charged at the fakes' calls) the deadline can't fire: the same 0.2 s, and the splice goes, the player taking it
    late (the latency analysis of 2026-11-01)."""
    from gamelab.sim import fakes
    real = {"t": 0.0}
    monkeypatch.setitem(fakes._REAL, "perf_counter", lambda: real["t"])

    def wrap(sim, nav):
        real_switch, real_chain = nav._switch, nav.flow.chain
        inside = {"on": False}

        def chain(*a, **k):
            if inside["on"]:
                real["t"] += 0.2
            return real_chain(*a, **k)

        def switch(*a, **k):
            inside["on"] = True
            try:
                return real_switch(*a, **k)
            finally:
                inside["on"] = False
        nav.flow.chain, nav._switch = chain, switch
    trace, sim, state = _spawn_run(wrap, think=1.0, reads=True)
    assert any("replanned too late (200 ms" in line for line in trace["log"]), trace["log"][:60]
    assert not trace["splices"]
    trace, sim, state = _spawn_run(wrap, think=1.0)
    assert not any("replanned too late" in line for line in trace["log"])
    assert trace["splices"] and max(p.spliced["late_ms"] for p in sim.player.order if p.spliced) >= 100


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
