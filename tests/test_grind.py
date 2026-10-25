"""Grinding (behavior.patrol, behavior.grind) on synthetic maps: the loop it chooses against camping, the next stop
along it and what pulls her off it, other players' areas, no flip-flopping between loops, the belief of each spawn
point, and kills in passing on another platform. No game data, no simulator: route times are stand-ins."""

import time
from types import SimpleNamespace

import pytest

from gamelab.behavior import patrol
from gamelab.behavior.fight import FightState
from gamelab.behavior.grind import Grind
from gamelab.behavior.hunt import Objective
from gamelab.localization import mapdata
from gamelab.localization.memory import Memory
from gamelab.localization.monsters import Monster
from gamelab.localization.players import Player
from gamelab.perception.mobs import Mob

HORNY, GREEN = "horny-mushroom", "green-mushroom"


def _map(plats, spawns):
    """A map of flat platforms (x from, x to, y; each its own foothold group: none walks onto another) and spawn
    points ((x, y, kind): "h" a Horny Mushroom, 41 EXP, "g" a Green Mushroom, 26)."""
    fh = {str(i): {str(i): {"x1": x1, "y1": y, "x2": x2, "y2": y, "prev": 0, "next": 0}}
          for i, (x1, x2, y) in enumerate(plats, 1)}
    life = {str(k): {"type": "m", "id": {"h": "1", "g": "2"}[mob], "x": x, "y": y, "cy": y, "rx0": x - 40,
                     "rx1": x + 40} for k, (x, y, mob) in enumerate(spawns)}
    tree = {"info": {"VRLeft": -500, "VRTop": -3000, "VRRight": 4000, "VRBottom": 1000}, "foothold": {"1": fh},
            "life": life, "miniMap": {"width": 4500, "height": 4000, "centerX": 500, "centerY": 3000, "mag": 4}}
    return mapdata.parse(1, tree, "Grind test", {}, {"1": "Horny Mushroom", "2": "Green Mushroom"},
                         {"1": 41, "2": 26})


def _pid(model, x, y):
    return model.platform_under(x, y).id


def _patrol(model, seconds):
    """A Patrol on `model` whose route time between two platforms is `seconds(a, b)` (platform ids) plus a walk."""
    def travel(a, x, b, bx):
        return seconds(a, b) if a != b else abs(bx - x) / 149.0
    return patrol.Patrol(model, {HORNY, GREEN}, {HORNY: 41.0, GREEN: 26.0}, travel)


def _sight(model, kind, x, y):
    """A monster seen at (x, y), placed on the platform there."""
    return Monster(Mob(kind, 0, 0, 0, 0, 1.0), x, y, _pid(model, x, y))


# A column of four platforms 200 units apart (no star reaches from one to another), one Horny Mushroom on each, 2 s
# from one to the next; and a fifth far off (30 s).
COLUMN = [(0, 300, 0), (0, 300, -200), (0, 300, -400), (0, 300, -600), (2000, 2300, 0)]
COLUMN_SPAWNS = [(150, 0, "h"), (150, -200, "h"), (150, -400, "h"), (150, -600, "h"), (2150, 0, "h")]


def _column():
    model = _map(COLUMN, COLUMN_SPAWNS)
    ids = [_pid(model, 150, y) for y in (0, -200, -400, -600)]
    far = _pid(model, 2150, 0)

    def seconds(a, b):
        if far in (a, b):
            return 30.0
        return 2.0 * abs(ids.index(a) - ids.index(b))
    return model, ids, far, _patrol(model, seconds)


def test_a_loop_timed_to_respawns_beats_camping():
    model, ids, far, p = _column()
    assert [len(s.cover) for s in p.stops] == [1] * 5  # each platform's own point, nothing across
    camp, loop = p.camping(0.0), p.best(0.0)
    # Camping at one point gets a kill each time it's back (17.8 s on average: REFILL_D0 + REFILL_TAU), and stands
    # idle the rest of the time.
    assert len(camp.stops) == 1 and camp.rate == pytest.approx(41 / (patrol.REFILL_D0 + patrol.REFILL_TAU), rel=0.1)
    # The round of the column: each point passed again about when it's back (its cycle past the refill's median), for
    # a good deal more EXP a minute; the stop 30 s away isn't worth the trip.
    assert set(loop.stops) == {p.stop_on(i) for i in ids} and p.stop_on(far) not in loop.stops
    assert loop.cycle >= patrol.refill_median(1)
    assert loop.rate > 1.5 * camp.rate
    # A round's worth is what comes back in it: the same round in a world refilling 4 times as slowly is worth less.
    p.belief.refills = [4.0] * 100
    assert p.rate(loop.stops, 0.0).rate < 0.7 * loop.rate


def test_the_next_stop_goes_round_the_loop_and_skips_what_she_has_just_emptied():
    model, ids, far, p = _column()
    m = Memory(model)
    p.belief.update(0.0, m)
    p.replan(0.0)
    order = list(p.plan.stops)
    first, info = p.choose(0.0, ids[0], 150.0, current=p.stop_on(ids[0]))
    assert first in order and info["why"] == "the loop"
    # Round the loop from there, in its order.
    nxt, _ = p.choose(1.0, p.platform_of(first), 150.0, current=first)
    assert nxt == order[(order.index(first) + 1) % len(order)]
    # Her kill emptied the stop after that a moment ago, and her memory saw it go: skipped until it's due.
    after = order[(order.index(nxt) + 1) % len(order)]
    q = next(i for i, _ in p.stops[after].cover)
    p.belief.empty_since[q] = 1.5
    skip, info = p.choose(2.0, p.platform_of(nxt), 150.0, current=nxt)
    assert skip != after and [s["stop"] for s in info["skipped"]] == [p.platform_of(after)]
    assert info["skipped"][0]["due"] == pytest.approx(1.5 + patrol.refill_median(1) - 2.0, abs=0.1)


def test_a_pile_seen_off_the_loop_pulls_the_route():
    # Two areas side by side: the loop's on the left (three platforms of one Horny Mushroom each), a fourth platform
    # off to the right with two Horny Mushroom points and a Green's.
    model = _map([(0, 300, 0), (0, 300, -200), (0, 300, -400), (1000, 1400, -200)],
                 [(150, 0, "h"), (150, -200, "h"), (150, -400, "h"), (1100, -200, "h"), (1300, -200, "h"),
                  (1200, -200, "g")])
    left = [_pid(model, 150, y) for y in (0, -200, -400)]
    right = _pid(model, 1200, -200)
    p = _patrol(model, lambda a, b: 6.0 if right in (a, b) else 2.0 * abs(left.index(a) - left.index(b)))
    m = Memory(model)
    p.belief.update(0.0, m)
    p._take(p.rate([p.stop_on(i) for i in left], 0.0), 0.0, "test")
    here = p.stop_on(left[0])
    nxt, info = p.choose(0.5, left[0], 150.0, current=here)
    assert p.platform_of(nxt) in left and info["why"] == "the loop"
    # Three monsters seen on the right: worth the trip off the loop.
    pile = [_sight(model, HORNY, 1100, -200), _sight(model, HORNY, 1290, -200), _sight(model, GREEN, 1200, -200)]
    m.update(1.0, pile, None)
    p.belief.update(1.0, m)
    nxt, info = p.choose(1.0, left[0], 150.0, current=here)
    assert p.platform_of(nxt) == right and info["why"] == "pulled" and info["seen"] > 90
    assert info["instead_of"]["stop"] in left
    # One alone isn't a pile: the loop goes on.
    m2 = Memory(model)
    m2.update(1.0, [_sight(model, GREEN, 1200, -200)], None)
    p.belief = patrol.Belief(model, p.points)
    p.belief.update(1.0, m2)
    nxt, info = p.choose(1.0, left[0], 150.0, current=here)
    assert p.platform_of(nxt) in left and info["why"] == "the loop"


def _two_areas():
    """Two like areas far apart, each two platforms of two Horny Mushroom points, 2 s between them, 60 s across."""
    model = _map([(0, 300, 0), (0, 300, -200), (2000, 2300, 0), (2000, 2300, -200)],
                 [(100, 0, "h"), (200, 0, "h"), (100, -200, "h"), (200, -200, "h"),
                  (2100, 0, "h"), (2200, 0, "h"), (2100, -200, "h"), (2200, -200, "h")])
    a = [_pid(model, 150, 0), _pid(model, 150, -200)]
    b = [_pid(model, 2150, 0), _pid(model, 2150, -200)]

    def seconds(x, y):
        return 2.0 if (x in a) == (y in a) else 60.0
    return model, a, b, _patrol(model, seconds)


def test_an_area_with_another_player_is_avoided():
    model, a, b, p = _two_areas()
    m = Memory(model)
    other = Player(1, "stranger", 150.0, -200.0, a[1], (0.0, 0.0), 0.0, 0.0)
    p.belief.update(0.0, m, [other])
    by, below = [q for q in p.points if q.platform == a[1]], [q for q in p.points if q.platform == a[0]]
    assert [p.belief.share(q.id, 0.0) for q in by] == pytest.approx([1 - patrol.TAKE_PRIOR] * 2)
    assert [p.belief.share(q.id, 0.0) for q in below] == [1.0, 1.0]  # 200 units under them: not by them
    loop = p.best(0.0)
    assert {p.platform_of(i) for i in loop.stops} == set(b)
    # Learned: another player by them took 6 Horny Mushrooms she saw, and left her none: they take that kind.
    for k in range(6):
        e = Monster(Mob(HORNY, 0, 0, 0, 0, 1.0), 100.0, 0.0, a[0])
        m.update(1.0 + k, [e], None, [Player(1, "stranger", 100.0, 0.0, a[0], (0.0, 0.0), 1.0, 1.0)])
        m.entries[-1].state, m.entries[-1].ended = "gone", 1.0 + k
        m.events.append(("gone", 1.0 + k, m.entries[-1]))
        m.counts["gone"] += 1
    p.belief.update(8.0, m, [other])
    assert p.belief.take(HORNY) > 0.75
    # Long after the other player was last by it, the area is hers again.
    p.belief.update(8.0 + patrol.RIVAL_KEEP_S + 1, m, [])
    assert p.belief.share(by[0].id, 8.0 + patrol.RIVAL_KEEP_S + 1) == 1.0


def test_no_flip_flopping_between_loops():
    model, a, b, p = _two_areas()
    m = Memory(model)
    p.belief.update(0.0, m, [Player(1, "stranger", 150.0, -200.0, a[1], (0.0, 0.0), 0.0, 0.0)])
    assert p.replan(0.0) == "the first"
    first = set(p.plan.stops)
    # The other player moves to the other area: the areas swap, but not within LOOP_MIN_S of taking hers up ...
    t = patrol.REPLAN_S + 1
    p.belief.update(t, m, [Player(1, "stranger", 2150.0, -200.0, b[1], (0.0, 0.0), t, t)])
    p.belief.rival_at = {q.id: t for q in p.points if q.platform in b}
    assert p.replan(t) is None and set(p.plan.stops) == first
    # ... after it, the better one is taken up.
    t = patrol.LOOP_MIN_S + 1
    p.belief.rival_at = {q.id: t for q in p.points if q.platform in b}
    assert p.replan(t) is not None and set(p.plan.stops) != first
    # Back and forth by a little (under LOOP_GAIN) changes nothing, however often it's looked at.
    second, loops = set(p.plan.stops), p.loops
    for k in range(10):
        t += patrol.LOOP_MIN_S + patrol.REPLAN_S
        p.belief.rival_at = {}
        p.belief.refills = [1.0 + 0.02 * (k % 2)]
        p.replan(t)
    assert set(p.plan.stops) == second and p.loops == loops


def test_the_belief_of_a_point_her_kill_emptied_refills_and_learns_how_fast():
    model, ids, far, p = _column()
    b, m = p.belief, Memory(model)
    q = next(q for q in p.points if q.platform == ids[0])
    seen = _sight(model, HORNY, 150, 0)
    m.update(0.0, [seen], None)
    b.update(0.0, m)
    assert b.stock(q.id, 0.0) == pytest.approx(1.0, abs=0.01)
    # Her kill: the point is empty, nothing back within REFILL_D0, even odds at the median.
    m._killed(m.entries[0], 1.0, 150.0)
    b.update(1.0, m)
    assert b.stock(q.id, 1.0 + patrol.REFILL_D0 - 0.1) == 0.0
    assert b.stock(q.id, 1.0 + patrol.refill_median(1)) == pytest.approx(0.5)
    assert p.expect(p.stop_on(ids[0]), 2.0)[0] == 0.0
    # Watched back 7 s after the kill (a look saw it empty a moment before): refills come about twice as fast here.
    b.empty_seen[q.id] = 7.5
    m.update(8.0, [_sight(model, HORNY, 140, 0)], None)
    b.update(8.0, m)
    assert q.id not in b.empty_since and len(b.refills) == 1
    assert b.refills[0] == pytest.approx((7.75 - 1.0) / patrol.refill_median(1))
    assert b.scale < 1.0
    # A point never seen: Q_UNKNOWN.
    other = next(q for q in p.points if q.platform == ids[1])
    assert b.stock(other.id, 8.0) == patrol.Q_UNKNOWN


def test_the_danger_hook_is_a_cost_and_learns_from_her_hp():
    model, a, b, p = _two_areas()
    # By default a platform with Horny Mushroom points (stance.KEEP_OFF) costs HP_S_PER_SPAWN a point a second.
    assert p.danger(a[0], 150.0) == pytest.approx(2 * patrol.HP_S_PER_SPAWN)
    before = p.rate([p.stop_on(i) for i in a], 0.0)
    # She has seen a bar of HP go a minute on the first (as in the simulator): that area's rounds are worth less, a
    # cost, not a ban.
    p.learn_hp(a[0], 5.0, 300.0)
    after = p.rate([p.stop_on(i) for i in a], 0.0)
    assert after.rate < before.rate and after.hp > before.hp > 0
    assert {p.platform_of(i) for i in p.best(0.0).stops} == set(b)


def _grind(model, kinds):
    """A Grind with only what its targets and the shooting in passing read."""
    g = object.__new__(Grind)
    g.h = SimpleNamespace(model=model, report=SimpleNamespace(exp_kills=0), kinds_told=lambda n: [None] * n,
                          looter=SimpleNamespace(pending=[]), actions=SimpleNamespace(facing="right"))
    g.model, g.kinds, g.fs = model, kinds, FightState()
    g._tops, g._halves = {}, {}
    return g


def test_she_shoots_in_passing_what_a_star_reaches_on_another_platform():
    # On her way along A (y 500) to a stop on C, a Horny Mushroom on B, 50 units up and 220 across: a target, and the
    # leg stops for it (rule 4) though B isn't the stop's.
    model = _map([(0, 400, 500), (300, 700, 450), (2000, 2400, 500)],
                 [(200, 500, "h"), (520, 450, "h"), (2200, 500, "h")])
    A, B, C = _pid(model, 200, 500), _pid(model, 520, 450), _pid(model, 2200, 500)
    g = _grind(model, {HORNY})
    g.patrol = _patrol(model, lambda a, b: 3.0)
    g.stop = g.patrol.stop_on(C)
    g.visit = SimpleNamespace(ground=C)
    g.obj = Objective("spot", C, 2200.0, ground=C)
    up = _sight(model, HORNY, 520, 450)
    s = SimpleNamespace(platform=model.platform(A), where=SimpleNamespace(x=300.0, y=500.0), monsters=[up], t=0.0)
    targets, visible, hidden = g._targets(s, time.monotonic())
    assert up in targets and g._can_pass_shoot(s, targets)
    # From A's far end it's out of a star's reach (500 across): no target on the way.
    s.where = SimpleNamespace(x=20.0, y=500.0)
    targets, _, _ = g._targets(s, time.monotonic())
    assert up not in targets and not g._can_pass_shoot(s, targets)
    # The stop's own monsters are targets wherever she is, but on the way only those in reach are shot at.
    there = _sight(model, HORNY, 2300, 500)
    s.monsters = [there]
    targets, _, _ = g._targets(s, time.monotonic())
    assert there in targets and not g._can_pass_shoot(s, targets)
