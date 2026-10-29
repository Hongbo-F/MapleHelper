"""Grinding (behavior.patrol, behavior.grind): the loop it chooses against camping, the next stop
along it and what pulls her off it, other players' areas, no flip-flopping between loops, the belief of each spawn
point, kills in passing on another platform, a stop's place by the danger, and a pause while looks fail. On synthetic
maps, route times stand-ins; the pause in the simulator on Tree Dungeon (its game data)."""

import math
import time
from types import SimpleNamespace

import pytest

from gamelab.attack import reach
from gamelab.attack.danger import Her
from gamelab.behavior import grind, patrol
from gamelab.behavior.fight import FightState
from gamelab.behavior.grind import Grind
from gamelab.behavior.hunt import Objective
from gamelab.behavior.ledger import Kill
from gamelab.localization import mapdata
from gamelab.localization.memory import Cover, Memory
from gamelab.localization.monsters import Monster
from gamelab.localization.players import Player
from gamelab.perception.mobs import Mob
from gamelab.sim import scenarios

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


TOUCH = {HORNY: 0.164, GREEN: 0.08}  # the share of her HP a touch takes (Tree Dungeon's, patrol.touch_shares)


def _patrol(model, seconds):
    """A Patrol on `model` whose route time between two platforms is `seconds(a, b)` (platform ids) plus a walk."""
    def travel(a, x, b, bx):
        return seconds(a, b) if a != b else abs(bx - x) / 149.0
    return patrol.Patrol(model, {HORNY, GREEN}, {HORNY: 41.0, GREEN: 26.0}, travel, touch=TOUCH)


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


def _beside(off: float):
    """Two areas side by side: the loop's on the left (three platforms of one Horny Mushroom each, 2 s apart), a fourth
    platform `off` s to the right with two Horny Mushroom points and a Green's; the loop taken up, her at its first
    stop."""
    model = _map([(0, 300, 0), (0, 300, -200), (0, 300, -400), (1000, 1400, -200)],
                 [(150, 0, "h"), (150, -200, "h"), (150, -400, "h"), (1100, -200, "h"), (1300, -200, "h"),
                  (1200, -200, "g")])
    left = [_pid(model, 150, y) for y in (0, -200, -400)]
    right = _pid(model, 1200, -200)
    p = _patrol(model, lambda a, b: off if right in (a, b) else 2.0 * abs(left.index(a) - left.index(b)))
    p.belief.update(0.0, Memory(model))
    p._take(p.rate([p.stop_on(i) for i in left], 0.0), 0.0, "test")
    return model, left, right, p


def _seen(model, p, sights, t=1.0):
    """The belief afresh, from a memory that saw `sights` ((kind, x, y)) at `t`."""
    m = Memory(model)
    m.update(t, [_sight(model, k, x, y) for k, x, y in sights], None)
    p.belief = patrol.Belief(model, p.points)
    p.belief.update(t, m)


def test_a_pile_seen_off_the_loop_pulls_her_when_it_pays_for_its_detour():
    model, left, right, p = _beside(6.0)
    here = p.stop_on(left[0])
    nxt, info = p.choose(0.5, left[0], 150.0, current=here)
    assert p.platform_of(nxt) in left and info["why"] == "the loop"
    rate = p.plan.rate  # the loop's long-run EXP a second (4.5: its points not all back each round)
    assert 4.0 < rate < 5.0
    # Three monsters seen on the right, 6 s off (95 EXP expected there): going there, shooting them and on back to the
    # loop's next stop, 2 s from her, is a 20 s detour, in which the loop gives about as much: not PULL_GAIN times, so
    # the loop goes on (until 2026-10-29 a pile pulled at 1.5 times the next stop's own EXP a second, and these did).
    three = [(HORNY, 1100, -200), (HORNY, 1290, -200), (GREEN, 1200, -200)]
    _seen(model, p, three)
    r, after = p.stop_on(right), nxt
    exp = p.expect(r, 7.0)[0]
    straight = p.travel(left[0], 150.0, p.platform_of(after), p.stops[after].x)
    d = p.detour(r, after, 6.0, straight, exp)
    assert exp > 90 and straight == 2.0 and d == pytest.approx(exp * p.kill_s + patrol.STOP_S + 6.0 + 6.0 - 2.0)
    assert 1.0 < exp / (rate * d) < patrol.PULL_GAIN
    nxt, info = p.choose(1.0, left[0], 150.0, current=here)
    assert nxt == after and info["why"] == "the loop"
    # Five there: they pay for theirs.
    _seen(model, p, three + [(HORNY, 1150, -200), (HORNY, 1250, -200)])
    nxt, info = p.choose(1.0, left[0], 150.0, current=here)
    assert p.platform_of(nxt) == right and info["why"] == "pulled" and info["seen"] > 150
    assert info["instead_of"]["stop"] == p.platform_of(after) and info["loop_rate"] == round(rate, 2)
    assert info["expected"] >= patrol.PULL_GAIN * rate * info["detour"] and 20 < info["detour"] < 30
    # The three, with the right 3 s off: a 13.6 s detour, worth it.
    model, left, right, p = _beside(3.0)
    _seen(model, p, three)
    nxt, info = p.choose(1.0, left[0], 150.0, current=p.stop_on(left[0]))
    assert p.platform_of(nxt) == right and info["why"] == "pulled" and info["detour"] == pytest.approx(13.6, abs=0.3)
    assert info["ahead"] == info["instead_of"]["stop"]
    # One Green Mushroom alone, 6 s off, isn't worth leaving the loop for.
    model, left, right, p = _beside(6.0)
    _seen(model, p, [(GREEN, 1200, -200)])
    nxt, info = p.choose(1.0, left[0], 150.0, current=p.stop_on(left[0]))
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
    # Learned: another player by them took 6 Horny Mushrooms she saw (looks forgot them), and left her none: they take
    # that kind.
    for k in range(6):
        e = Monster(Mob(HORNY, 0, 0, 0, 0, 1.0), 100.0, 0.0, a[0])
        m.update(1.0 + k, [e], None, [Player(1, "stranger", 100.0, 0.0, a[0], (0.0, 0.0), 1.0, 1.0)])
        m.entries[-1].state, m.entries[-1].ended, m.entries[-1].missed = "gone", 1.5 + k, 1.5 + k
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
    assert b.refills[0] == pytest.approx((7.75 - 1.0) / patrol.refill_mean(1))
    assert b.scale < 1.0
    # A point never seen: Q_UNKNOWN.
    other = next(q for q in p.points if q.platform == ids[1])
    assert b.stock(other.id, 8.0) == patrol.Q_UNKNOWN


def test_the_danger_hook_is_a_cost_and_learns_from_her_hp():
    model, a, b, p = _two_areas()
    # By default a platform's danger is the touch shares of its points of kinds over DANGER_MIN: two Horny Mushrooms,
    # DANGER_HP_S of her HP a second for each unit of it.
    assert p.danger(a[0], 150.0) == pytest.approx(2 * TOUCH[HORNY])
    assert p.hp_rate(a[0], 150.0) == pytest.approx(patrol.DANGER_HP_S * 2 * TOUCH[HORNY])
    weak = patrol.Patrol(model, {HORNY}, {HORNY: 41.0}, p.travel, touch={HORNY: patrol.DANGER_MIN / 2})
    assert weak.danger(a[0], 150.0) == 0.0  # a kind whose touch takes little: no danger
    # Tree Dungeon's from the game data (its touch damage), where it's on this computer.
    if scenarios.available(10002073):
        shares = patrol.touch_shares(mapdata.load(10002073), Her(752.0, 24.0))  # (her HUD's at level 24)
        assert shares["horny-mushroom"] == pytest.approx(0.164, abs=0.01) and shares["slime"] < patrol.DANGER_MIN
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


def test_a_world_refilling_as_the_model_says_is_learned_as_the_model():
    # Watched refills drawn from the model itself: their scale comes out at 1 (against the model's median it came out at
    # 1.25, the mean being 1.28 times the median).
    import math
    import random
    model, ids, far, p = _column()
    rng, n = random.Random(1), 1
    tau = n * patrol.REFILL_TAU
    p.belief.refills = [(patrol.REFILL_D0 - tau * math.log(1 - rng.random())) / patrol.refill_mean(n)
                        for _ in range(2000)]
    assert p.belief.scale == pytest.approx(1.0, abs=0.05)


def test_a_point_looks_saw_empty_is_believed_empty_however_long_since_it_was_emptied():
    model, ids, far, p = _column()
    b, m = p.belief, Memory(model)
    q = next(q for q in p.points if q.platform == ids[0])
    look = Cover(0.0, ((-300.0, 100.0),), -200.0, 600.0)  # all of the bottom platform searched
    m.update(10.0, [], Cover(10.0, look.rows, look.x0, look.x1))
    b.update(10.0, m)
    assert b.empty_since[q.id] == 10.0 and b.stock(q.id, 10.0) == 0.0
    # Seen empty again 50 s later: not back now, whatever the clock from the first look says (0.97 back by it) ...
    m.update(60.0, [], Cover(60.0, look.rows, look.x0, look.x1))
    b.update(60.0, m)
    assert b.empty_since[q.id] == 10.0 and b.stock(q.id, 60.0) == 0.0
    # ... and from then on as a point just watched empty: the clock past REFILL_D0 has no memory.
    tau = patrol.REFILL_TAU
    assert b.stock(q.id, 60.0 + tau) == pytest.approx(1 - math.exp(-1), abs=1e-6)
    assert b.due(q.id, 60.0) == pytest.approx(tau * math.log(2))
    # Never watched since its kill, a point is due at the model's median after it, and is overdue after that.
    b.empty_seen.pop(q.id)
    assert b.due(q.id, 60.0) == pytest.approx(10.0 + patrol.refill_median(1) - 60.0)


def test_a_point_seen_full_again_takes_a_fresh_clock_when_it_is_next_emptied():
    model, ids, far, p = _column()
    b, m = p.belief, Memory(model)
    q = next(q for q in p.points if q.platform == ids[0])
    look = ((-300.0, 100.0),)
    m.update(0.0, [_sight(model, HORNY, 150, 0)], Cover(0.0, look, -200.0, 600.0))
    b.update(0.0, m)
    # Missed by looks for 2.2 s (under SURE): seen empty ...
    for t in (1.0, 2.2):
        m.update(t, [], Cover(t, look, -200.0, 600.0))
        b.update(t, m)
    assert q.id in b.empty_since
    # ... then seen again (the same monster), until 60 s: the point is full, its clock gone.
    for t in (3.0, 30.0, 60.0):
        m.update(t, [_sight(model, HORNY, 150, 0)], Cover(t, look, -200.0, 600.0))
        b.update(t, m)
    assert q.id not in b.empty_since
    # Taken by someone else: empty from the look that saw it gone, not full by a clock from 2.2 s.
    for t in (61.0, 63.0, 66.0):
        m.update(t, [], Cover(t, look, -200.0, 600.0))
        b.update(t, m)
    assert b.empty_since[q.id] >= 61.0 and b.stock(q.id, 66.0) == 0.0
    assert p.expect(p.stop_on(ids[0]), 66.0)[0] == 0.0


def test_a_look_empties_only_the_points_whose_monsters_it_searched():
    # One platform from x 0 to 1200 with a Horny Mushroom point at 300 and one at 1100; a look whose screen ends at
    # x 700: the point at 1100 wasn't seen, so it isn't seen empty (half the platform searched isn't all of it).
    model = _map([(0, 1200, 0)], [(300, 0, "h"), (1100, 0, "h")])
    p = _patrol(model, lambda a, b: 1.0)
    b, m = p.belief, Memory(model)
    m.update(5.0, [], Cover(5.0, ((-300.0, 100.0),), -400.0, 700.0))
    b.update(5.0, m)
    near, far_point = sorted(p.points, key=lambda q: q.x)
    assert near.id in b.empty_since and far_point.id not in b.empty_since
    assert b.stock(far_point.id, 5.0) == patrol.Q_UNKNOWN
    assert m.covered_at(near.platform) is None


def test_others_take_is_learned_from_monsters_looked_at_again_only():
    model, a, b, p = _two_areas()
    m = Memory(model)
    by = [Player(1, "stranger", 100.0, 0.0, a[0], (0.0, 0.0), 0.0, 0.0)]
    # A Horny Mushroom by another player, seen once and never looked at again: neither left to her nor taken ...
    m.update(0.0, [Monster(Mob(HORNY, 0, 0, 0, 0, 1.0), 100.0, 0.0, a[0])], None, by)
    for t in (5.0, 12.0):
        m.update(t, [], None, by)
        p.belief.update(t, m, by)
    assert p.belief.kept_by[HORNY] == 0 and p.belief.take(HORNY) == patrol.TAKE_PRIOR
    # ... nor once forgotten by age (MEMORY_S: no look covered its place).
    m.update(61.0, [], None, by)
    p.belief.update(61.0, m, by)
    assert m.counts["aged"] == 1 and p.belief.gone_by[HORNY] == 0
    # Seen there again 10 s after it was first seen: left to her.
    m.update(70.0, [Monster(Mob(HORNY, 0, 0, 0, 0, 1.0), 100.0, 0.0, a[0])], None, by)
    m.update(80.0, [Monster(Mob(HORNY, 0, 0, 0, 0, 1.0), 110.0, 0.0, a[0])], None, by)
    p.belief.update(80.0, m, by)
    assert p.belief.kept_by[HORNY] == 1 and p.belief.take(HORNY) < patrol.TAKE_PRIOR


def test_one_monster_seen_close_by_pulls_her_from_a_loop_stop_far_off():
    # On the column's bottom platform, the loop's next stop 30 s off and never seen; one Horny Mushroom seen on the
    # platform above, 2 s off: a pile of one is worth the trip (two kills' worth had been asked).
    model, ids, far, p = _column()
    m = Memory(model)
    p._take(p.rate([p.stop_on(ids[0]), p.stop_on(far)], 0.0), 0.0, "test")
    m.update(1.0, [_sight(model, HORNY, 150, -200)], None)
    p.belief.update(1.0, m)
    nxt, info = p.choose(1.0, ids[0], 150.0, current=p.stop_on(ids[0]))
    assert p.platform_of(nxt) == ids[1] and info["why"] == "pulled" and info["instead_of"]["stop"] == far
    # On her way to the far stop, the same: the stop she's heading to stands in for the loop's next ...
    p.at_loop = 0
    nxt, info = p.choose(1.0, ids[0], 150.0, heading=p.stop_on(far), peek=True)
    assert p.platform_of(nxt) == ids[1] and info["why"] == "pulled"
    assert p.at_loop == 0  # (a peek moves nothing)
    # ... and with nothing seen she goes on to it.
    p.belief = patrol.Belief(model, p.points)
    p.belief.update(1.0, Memory(model))
    nxt, info = p.choose(1.0, ids[0], 150.0, heading=p.stop_on(far), peek=True)
    assert nxt == p.stop_on(far) and info["why"] == "on her way"


def _tower():
    """A tower: the loop's three platforms at the top (two Horny Mushroom points each, 1.5 s apart), a shelf on the way
    up (one Horny Mushroom point) 12 s from the bottom and 13 from the top, and the bottom's three platforms far apart
    (one Green Mushroom point each, 2 s apart), 25 s from the top. The loop round the top taken up, its rate as
    Tree Dungeon's live (447 EXP a minute, 7.45 EXP/s: the magician's of 2026-10-29)."""
    from dataclasses import replace
    model = _map([(0, 300, -2000), (0, 300, -1800), (0, 300, -1600), (0, 300, -800), (0, 300, 0), (1000, 1300, 0),
                  (2000, 2300, 0)],
                 [(100, -2000, "h"), (200, -2000, "h"), (100, -1800, "h"), (200, -1800, "h"), (100, -1600, "h"),
                  (200, -1600, "h"), (150, -800, "h"), (150, 0, "g"), (1150, 0, "g"), (2150, 0, "g")])
    top = [_pid(model, 150, y) for y in (-2000, -1800, -1600)]
    shelf = _pid(model, 150, -800)
    bottom = [_pid(model, x, 0) for x in (150, 1150, 2150)]

    def seconds(a, b):
        for group, step in ((top, 1.5), (bottom, 2.0)):
            if a in group and b in group:
                return step * abs(group.index(a) - group.index(b))
        return {frozenset("tb"): 25.0, frozenset("sb"): 12.0, frozenset("ts"): 13.0}[
            frozenset("t" if a in top else "s" if a == shelf else "b") | frozenset("t" if b in top else "s" if b == shelf
                                                                                    else "b")]
    p = _patrol(model, seconds)
    p.belief.update(0.0, Memory(model))
    p._take(p.rate([p.stop_on(i) for i in top], 0.0), 0.0, "test")
    p.plan = replace(p.plan, rate=447 / 60)
    return model, top, shelf, bottom, p


def test_far_from_the_loop_a_pile_by_her_pays_only_for_its_detour():
    # The magician live on Tree Dungeon (2026-10-29, mage-1029-0111.log): left at the bottom by a Lie Detector's
    # retreat, every pile seen at the bottom pulled her, at 1.5 times the EXP a second of the loop's next stop with its
    # 27 s climb counted, and she never climbed. Here: at the bottom, the loop's next stop 25 s up.
    model, top, shelf, bottom, p = _tower()
    here = p.stop_on(bottom[0])
    nxt, info = p.choose(1.0, bottom[0], 150.0, current=here, peek=True)
    assert p.platform_of(nxt) in top and info["why"] == "the loop" and info["travel"] == 25.0
    climb = info["gain"]  # the next stop's own EXP a second, its climb counted (1.0)
    assert climb < 1.5
    # Two Green Mushrooms seen on the next platform along the bottom, 2 s off (50 EXP expected at even odds or better):
    # 6.9 EXP a second there, its travel and kills counted, 7 times the climb's; but going there and on up adds 2 s to
    # the climb, a 7.2 s detour, in which the loop gives 54 EXP: not worth it. Nor are three (75 EXP, 9.3 s, 1.07 times
    # the loop's 70).
    for n in (2, 3):
        _seen(model, p, [(GREEN, 1150 + 20 * k, 0) for k in range(n)])
        nxt, info = p.choose(1.0, bottom[0], 150.0, current=here, peek=True)
        assert p.platform_of(nxt) in top and info["why"] == "the loop", n
        b1 = p.stop_on(bottom[1])
        far, (exp, seen, shoot, _) = 2.0, p.expect(b1, 3.0)
        assert seen == pytest.approx(26 * n, rel=0.05) and exp / (far + shoot + patrol.STOP_S) > 4 * climb
        assert exp < patrol.PULL_GAIN * p.plan.rate * p.detour(b1, nxt, far, 25.0, exp)
    # Two Horny Mushrooms seen on the shelf, on her way up (12 s, and 13 on: no longer than the climb; 63 EXP expected
    # by the time she's there): its detour is their kills and a stop, 6.3 s, which they pay for 1.34 times: she goes
    # up by the shelf.
    _seen(model, p, [(HORNY, 120, -800), (HORNY, 180, -800)])
    nxt, info = p.choose(1.0, bottom[0], 150.0, current=here, peek=True)
    assert p.platform_of(nxt) == shelf and info["why"] == "pulled"
    assert info["detour"] == pytest.approx(info["expected"] * p.kill_s + patrol.STOP_S, abs=0.1)
    assert info["instead_of"]["stop"] in top


def test_after_a_pile_the_loop_goes_on_from_the_stop_its_detour_went_on_to():
    # Round the left (the loop), the right 3 s off. Leaving the first stop for the loop's next ...
    model, left, right, p = _beside(3.0)
    here = p.stop_on(left[0])
    nxt, info = p.choose(0.5, left[0], 150.0, current=here)
    assert info["why"] == "the loop"
    # ... three seen on the right on the way there: pulled, the detour costed on to that loop stop ...
    _seen(model, p, [(HORNY, 1100, -200), (HORNY, 1290, -200), (GREEN, 1200, -200)])
    j, info = p.choose(1.0, left[0], 160.0, heading=nxt)
    assert p.platform_of(j) == right and info["why"] == "pulled" and info["ahead"] == p.platform_of(nxt)
    # ... and from the pile on to it (until 2026-10-29 the loop went on from the stop after it: a pull on a trip had
    # her miss the stop she'd been heading to for a round).
    p.belief.empty_since.update({q.id: 2.0 for q in p.points if q.platform == right})
    after, info = p.choose(8.0, right, 1200.0, current=j)
    assert after == nxt and info["why"] == "the loop"


def test_on_her_way_to_a_pile_another_must_beat_it_with_the_margin():
    # Her on a ledge (H), two ledges near her with two Horny Mushroom points each (P 2 s off, Q 2.3, 0.5 s apart, both
    # on her way up: 23 and 22.7 s on), the loop's three platforms 25 s up (its rate Tree Dungeon's live, 7.45 EXP/s).
    # Three Horny Mushrooms seen on each.
    from dataclasses import replace
    model = _map([(0, 300, -2000), (0, 300, -1800), (0, 300, -1600), (0, 300, 0), (1000, 1300, 0), (2000, 2300, 0)],
                 [(100, -2000, "h"), (200, -2000, "h"), (100, -1800, "h"), (200, -1800, "h"), (100, -1600, "h"),
                  (200, -1600, "h"), (1100, 0, "h"), (1200, 0, "h"), (2100, 0, "h"), (2200, 0, "h")])
    top = [_pid(model, 150, y) for y in (-2000, -1800, -1600)]
    H, P, Q = (_pid(model, x, 0) for x in (150, 1150, 2150))
    near = {frozenset((H, P)): 2.0, frozenset((H, Q)): 2.3, frozenset((P, Q)): 0.5}

    def seconds(a, b):
        if a in top and b in top:
            return 1.5 * abs(top.index(a) - top.index(b))
        if a in top or b in top:
            return {P: 23.0, Q: 22.7}.get(a if b in top else b, 25.0)
        return near[frozenset((a, b))]
    p = _patrol(model, seconds)
    p.belief.update(0.0, Memory(model))
    p._take(p.rate([p.stop_on(i) for i in top], 0.0), 0.0, "test")
    p.plan = replace(p.plan, rate=447 / 60)
    _seen(model, p, [(HORNY, 1100 + 50 * k, 0) for k in range(3)] + [(HORNY, 2100 + 50 * k, 0) for k in range(3)])
    sp, sq = p.stop_on(P), p.stop_on(Q)
    (fp, ep), (fq, eq) = [(f, p.expect(s, 1.0 + f)[0]) for s, f in ((sp, 2.0), (sq, 2.3))]
    assert ep == pytest.approx(3 * 41, rel=0.1) and eq == pytest.approx(3 * 41, rel=0.1)
    rate = p.plan.rate
    # Each, measured against going to the other (the rule's first build), paid for its detour: she turned from one to
    # the other at every look (loop_tree_grind_jump_key, 30 ms apart).
    assert patrol.pays(eq, patrol.detour_s(eq, p.kill_s, fq, 0.5, fp), rate)
    assert patrol.pays(ep, patrol.detour_s(ep, p.kill_s, fp, 0.5, fq), rate)
    # Now each is costed to the loop's next stop, and on her way to one the other must leave more with the margin:
    # on her way to either, she keeps to it.
    for going in (sp, sq):
        nxt, info = p.choose(1.0, H, 150.0, heading=going, peek=True)
        assert nxt == going and info["why"] == "on her way"
    # Not on her way to either: the better of them.
    nxt, info = p.choose(1.0, H, 150.0, peek=True)
    assert nxt == sp and info["why"] == "pulled" and info["ahead"] in top


def _in_line():
    """Three platforms side by side, C between A and B (A to C 2 s, C to B 2 s, A to B 4 s), a Horny Mushroom point on
    each; the loop A, B, C taken up (4.5 EXP a second), her at A (the review of 2026-10-29's probe:
    D:/MapleHelper-scratch/pull/review/probe_loopstop.py)."""
    model = _map([(0, 300, 0), (2000, 2300, 0), (1000, 1300, 0)], [(150, 0, "h"), (2150, 0, "h"), (1150, 0, "h")])
    A, B, C = _pid(model, 150, 0), _pid(model, 2150, 0), _pid(model, 1150, 0)
    legs = {frozenset((A, B)): 4.0, frozenset((A, C)): 2.0, frozenset((B, C)): 2.0}
    p = _patrol(model, lambda a, b: legs[frozenset((a, b))])
    p.belief.update(0.0, Memory(model))
    p._take(p.rate([p.stop_on(i) for i in (A, B, C)], 0.0), 0.0, "test")
    return model, (A, B, C), p


def test_after_a_pile_at_a_stop_of_the_loop_further_on_she_goes_on_to_the_stop_it_was_costed_to():
    model, (A, B, C), p = _in_line()
    sa, sb, sc = (p.stop_on(i) for i in (A, B, C))
    # Five seen on C, on her way from A to the loop's next, B: pulled, the detour costed on to B ...
    _seen(model, p, [(HORNY, 1100 + 20 * k, 0) for k in range(5)])
    j, info = p.choose(1.0, A, 150.0, current=sa)
    assert j == sc and info["why"] == "pulled" and info["ahead"] == B and p.pulled == (sc, sb)
    nxt, info = p.choose(1.5, A, 290.0, heading=sc, peek=True)
    assert nxt == sc and info["why"] == "on her way"
    # ... and leaving C, on to B. Until 2026-10-29 the loop went on from C, a stop of the loop itself, the next after
    # it being A, the stop she'd just left: B missed for a round.
    cq = next(q.id for q in p.points if q.platform == C)
    p.belief.at.pop(cq)
    p.belief.empty_since[cq] = p.belief.empty_seen[cq] = 6.0
    nxt, info = p.choose(8.0, C, 1150.0, current=sc)
    assert nxt == sb and info["why"] == "the loop" and p.pulled is None


def test_on_her_way_to_a_pile_that_no_longer_pays_she_goes_back_to_the_loop():
    model, (A, B, C), p = _in_line()
    sa, sb, sc = (p.stop_on(i) for i in (A, B, C))
    _seen(model, p, [(HORNY, 1100 + 20 * k, 0) for k in range(5)])
    j, _ = p.choose(1.0, A, 150.0, current=sa)
    assert j == sc
    # On her way there, C's monsters shot in passing from A's end: C no longer pays for its detour, even without the
    # margin: on to the loop's next stop, the one its detour went on to (until 2026-10-29 she went on to the empty pile
    # and left it with nothing: loop_tree_grind_rope_top's 15, 7 of 26 stays in her 01:11 run) ...
    cq = next(q.id for q in p.points if q.platform == C)
    p.belief.at.pop(cq)
    p.belief.empty_since[cq] = p.belief.empty_seen[cq] = 1.5
    nxt, info = p.choose(2.0, A, 290.0, heading=sc, peek=True)
    assert nxt == sb and info["why"] == "back to the loop" and info["dropped"]["stop"] == C
    assert info["dropped"]["over"] < 0 and "no longer pays" in patrol.choice_text(info)
    nxt, info = p.choose(2.0, A, 290.0, heading=sc)
    assert nxt == sb and p.pulled is None and p.at_loop == p.plan.stops.index(sb)
    # ... and on her way to it, the pile pulls her again only by paying with the margin: as it is, it doesn't; seen
    # again, it does.
    nxt, info = p.choose(2.5, A, 290.0, heading=sb, peek=True)
    assert nxt == sb and info["why"] == "on her way"
    _seen(model, p, [(HORNY, 1100 + 20 * k, 0) for k in range(5)], t=3.0)
    nxt, info = p.choose(3.0, A, 290.0, heading=sb, peek=True)
    assert nxt == sc and info["why"] == "pulled"


def test_a_piles_touches_count_against_it_as_a_loops_do(monkeypatch):
    from dataclasses import replace
    model, left, right, p = _beside(3.0)
    here, r = p.stop_on(left[0]), p.stop_on(right)
    _seen(model, p, [(HORNY, 1100, -200), (HORNY, 1290, -200), (GREEN, 1200, -200)])
    nxt, info = p.choose(1.0, left[0], 150.0, current=here, peek=True)
    assert nxt == r and info["why"] == "pulled"
    # She has seen 4 bars of her HP a minute go there: what touches cost her while she shoots it (HP_PRICE EXP a bar,
    # as the loop's rate has them at its stops) leaves the three's EXP short of paying for their detour.
    p.learn_hp(right, 4.0, 60.0)
    hp = p.hp_rate(right, p.stops[r].x)
    assert p.net(r, 100.0) == pytest.approx(100.0 * (1 - patrol.HP_PRICE * p.kill_s * hp)) and p.net(r, 100.0) < 75
    nxt, info = p.choose(1.0, left[0], 150.0, current=here, peek=True)
    assert p.platform_of(nxt) in left and info["why"] == "the loop"
    # A loop worth nothing by the belief (its touches costing what it gives): no bar for a pile to beat (until
    # 2026-10-29 every pile seen paid then, the biggest however far), she keeps to it.
    p.hp_seen.clear()
    p.hp_time.clear()
    p.plan = replace(p.plan, rate=0.0)
    nxt, info = p.choose(1.0, left[0], 150.0, current=here, peek=True)
    assert p.platform_of(nxt) in left and info["why"] == "the loop"


def test_her_loop_found_again_in_another_order_keeps_its_order_and_takes_its_rate_now():
    model, a, b, p = _two_areas()
    p.belief.update(0.0, Memory(model))
    found = p.best(0.0)
    mine = found.stops[::-1]
    p._take(p.rate(mine, 0.0), 0.0, "test")
    before = p.plan.rate
    # Refills watched four times as slow as the model's: the same round, its rate now lower; found again in its other
    # order, hers keeps her order (her place in it) and takes the rate (until 2026-10-29 it kept the one it was taken
    # up at: a 2-stop round's two orders are one round, so the pulls' bar and a stop's stayed where they started).
    p.belief.refills = [4.0] * 100
    t = patrol.REPLAN_S + 1
    assert set(p.best(t).stops) == set(mine) and p.best(t).stops != mine
    assert p.replan(t) is None and p.plan.stops == mine
    assert p.plan.rate == pytest.approx(p.rate(mine, t).rate) and p.plan.rate < 0.8 * before


# Her live pulls of 2026-10-29 01:11-01:18 (mage-1029-0111.log's 'next:' lines, the telemetry's 'grind' records):
# (the stop pulled to, the stop pulled from, the EXP expected there, of it what touches there leave (Patrol.net: her HP
# fed as grinding feeds it), her kill time then (s a point of EXP, from the record's EXP a second), s to it, s from it
# on to the other (the planner's route), s to the other from where she was (the planner's, from her latest look), the
# loop's EXP a second as the code had it then (Patrol.replan every REPLAN_S, the belief fed from the run's looks: the
# 'loop' record's rate is written only when the loop changes, 7.45 and 7.36, and was in this table until the review of
# 2026-10-29), the pile's EXP a second with its travel and kills and the other's:
# D:/MapleHelper-scratch/pull/fix/replay_fix.py). The other is the loop's next stop in 25 of them; in 3 (from 32, 28
# and 20) a pile she was on her way to, which by this rule she wouldn't have been (none of the others pays).
LIVE_PULLS = [
    (30, 13, 45.7, 45.1, 0.0854, 5.8, 20.2, 25.5, 7.562, 4.27, 1.04),
    (29, 10, 31.5, 31.1, 0.0796, 3.7, 22.8, 26.2, 7.562, 4.37, 1.34),
    (31, 10, 13.2, 12.8, 0.0899, 1.5, 26.0, 23.2, 7.989, 3.58, 1.48),
    (29, 10, 70.2, 67.3, 0.0938, 3.7, 22.8, 26.3, 7.457, 6.22, 1.32),
    (30, 10, 48.7, 48.0, 0.1018, 2.8, 24.0, 26.7, 7.235, 5.56, 1.3),
    (32, 10, 45.2, 44.1, 0.0956, 1.2, 26.8, 23.9, 6.812, 6.93, 1.43),
    (30, 32, 35.9, 35.5, 0.0938, 4.2, 1.3, 1.8, 6.812, 4.19, 2.1),
    (33, 10, 15.0, 14.7, 0.0924, 1.5, 27.7, 24.3, 7.242, 3.86, 1.42),
    (32, 10, 15.9, 15.5, 0.0952, 1.3, 26.8, 26.6, 7.323, 4.17, 1.3),
    (30, 8, 34.3, 33.6, 0.1024, 3.3, 26.0, 29.2, 7.323, 4.39, 1.68),
    (28, 8, 46.8, 46.2, 0.1009, 4.7, 26.9, 29.9, 6.778, 4.49, 1.65),
    (31, 8, 47.6, 46.4, 0.1002, 1.1, 28.1, 24.9, 6.835, 6.93, 1.91),
    (29, 6, 47.8, 45.1, 0.1037, 3.7, 26.2, 29.7, 6.835, 4.95, 1.12),
    (30, 6, 27.3, 26.8, 0.1024, 2.1, 27.4, 26.2, 6.7, 4.63, 1.26),
    (32, 6, 24.3, 23.8, 0.0996, 1.2, 30.2, 27.3, 6.7, 5.26, 1.22),
    (30, 6, 48.6, 47.5, 0.1067, 3.0, 27.4, 30.2, 6.7, 5.29, 1.11),
    (28, 6, 47.0, 46.1, 0.1073, 4.4, 28.2, 31.0, 6.7, 4.5, 1.08),
    (30, 28, 48.9, 47.6, 0.1092, 2.7, 1.0, 3.2, 6.7, 5.41, 3.51),
    (24, 6, 26.0, 24.6, 0.1059, 7.6, 20.7, 27.6, 6.446, 2.29, 1.2),
    (23, 14, 16.0, 15.1, 0.1063, 1.3, 11.0, 12.2, 6.528, 4.0, 0.68),
    (20, 14, 16.6, 16.2, 0.106, 3.6, 7.9, 10.9, 6.528, 2.61, 0.76),
    (23, 20, 28.4, 27.0, 0.1065, 1.5, 3.8, 2.4, 6.385, 5.14, 3.19),
    (29, 9, 72.9, 68.7, 0.0987, 2.8, 23.8, 26.5, 6.972, 6.63, 1.62),
    (30, 7, 34.7, 33.6, 0.101, 4.0, 27.0, 30.8, 7.428, 4.08, 1.54),
    (32, 7, 48.8, 46.5, 0.1116, 1.3, 29.9, 27.0, 7.211, 6.3, 1.68),
    (23, 7, 72.4, 69.4, 0.1134, 11.3, 19.0, 29.8, 7.211, 3.53, 1.55),
    (24, 7, 65.2, 61.7, 0.1125, 1.1, 20.3, 21.0, 6.528, 6.91, 2.03),
    (23, 7, 100.6, 96.5, 0.1124, 3.1, 19.0, 21.9, 6.528, 6.53, 1.97),
]


def test_none_of_her_live_pulls_at_the_bottom_pays_for_its_detour():
    # Each pulled her by the old rule (1.5 times the other's EXP a second); by the detour's, none: the most, 69 EXP (net
    # of touches) 2.8 s off on her way up from 31 to 29, and 97 on 23 on her way up to 7, 1.18 times what the loop gave
    # over their 8.3 and 12.6 s. With less margin those on her way would, and only those (going there and on up adds 0.5
    # s at most to the climb): 5 at 1.1, 10 with none, each on her way up.
    ratios = []
    for _, _, exp, net, kill_s, far, on, far_next, rate, gain, other in LIVE_PULLS:
        assert gain >= 1.5 * other and net < exp
        d = patrol.detour_s(exp, kill_s, far, on, far_next)
        assert not patrol.pays(net, d, rate)
        ratios.append((net / (rate * d), far + on - far_next))
    assert max(r for r, _ in ratios) == pytest.approx(1.18, abs=0.01)
    for gain, n in ((1.1, 5), (1.0, 10)):
        kept = [extra for r, extra in ratios if r >= gain]
        assert len(kept) == n and max(kept) <= 0.5


def test_a_stop_counts_only_the_monsters_she_gets_at_from_it():
    # A (x 0-400, y 500) and B 50 units up (x 300-1300). From A a star reaches B's left end, not its right one.
    model = _map([(0, 400, 500), (300, 1300, 450)], [(200, 500, "h"), (500, 450, "h"), (1200, 450, "h")])
    A, B = _pid(model, 200, 500), _pid(model, 1200, 450)
    p = _patrol(model, lambda a, b: 2.0)
    a, b = p.stop_on(A), p.stop_on(B)
    assert any(p.points[q].platform == B for q, c in p.stops[a].cover)  # A's stop gets at some of B
    m = Memory(model)
    # Three seen at B's right end: A's stop gets at none of them (the share of B it reaches had counted them) ...
    m.update(1.0, [_sight(model, HORNY, 1150, 450), _sight(model, HORNY, 1200, 450), _sight(model, HORNY, 1250, 450)],
             None)
    p.belief.update(1.0, m)
    assert p.expect(a, 1.0)[1] == 0.0 and p.expect(b, 1.0)[1] == pytest.approx(3 * 41)
    p._take(p.rate([a, b], 1.0), 1.0, "test")
    nxt, info = p.choose(1.0, A, 200.0)
    assert nxt == b  # pulled to the stop that gets at them, not bounced to A's
    # ... one at B's left end, A's does.
    m.update(2.0, [_sight(model, HORNY, 480, 450)], None)
    p.belief.update(2.0, m)
    assert p.expect(a, 2.0)[1] == pytest.approx(41)
    # Left with nothing in reach there, a stop doesn't count again what she'd been there to see; a monster seen
    # since, it does.
    p.left_at[a] = 2.5
    assert p.expect(a, 3.0)[1] == 0.0
    m.update(3.0, [_sight(model, HORNY, 490, 450)], None)
    p.belief.update(3.0, m)
    assert p.expect(a, 3.0)[1] == pytest.approx(41)


def test_a_better_loop_is_taken_up_only_when_it_is_loop_gain_better(monkeypatch):
    model, a, b, p = _two_areas()
    m = Memory(model)
    p.belief.update(0.0, m)
    A, B = [p.stop_on(i) for i in a], [p.stop_on(i) for i in b]
    p._take(p.rate(B, 0.0), 0.0, "test")
    t = patrol.LOOP_MIN_S + 1
    # Another player by B's points, taking a little of them (7%: learned from 25 monsters left to her): A's round is
    # the best, but not LOOP_GAIN better than hers: kept ...
    p.belief.kept_by[HORNY] = 25
    p.belief.rival_at = {q.id: t for q in p.points if q.platform in b}
    ratio = p.rate(A, t).rate / p.rate(B, t).rate
    assert 1.0 < ratio < 1 + patrol.LOOP_GAIN and set(p.best(t).stops) == set(A)
    assert p.replan(t) is None and set(p.plan.stops) == set(B)
    with monkeypatch.context() as mp:  # (with no margin it would have changed)
        mp.setattr(patrol, "LOOP_GAIN", 0.0)
        assert p.replan(t, force=True) is not None and set(p.plan.stops) == set(A)
    p._take(p.rate(B, t), 0.0, "test")
    # ... taking half of them (the prior, nothing learned): more than LOOP_GAIN better, taken up.
    p.belief.kept_by[HORNY] = 0
    assert p.rate(A, t).rate / p.rate(B, t).rate > 1 + patrol.LOOP_GAIN
    assert p.replan(t + patrol.REPLAN_S) is not None and set(p.plan.stops) == set(A)


def _at_stop(model, p, pid, x, monsters, t):
    """A Grind at the stop on platform `pid`, arrived, standing at x with `monsters` in her look at time t, her memory
    and the belief updated with them."""
    g = _grind(model, {HORNY, GREEN})
    g.patrol, g.stop = p, p.stop_on(pid)
    g.h.memory = Memory(model)
    g.tour = SimpleNamespace(exp={HORNY: 41.0, GREEN: 26.0})
    g._skip, g._alt, g._reroute = {}, None, False
    g.visit = SimpleNamespace(ground=pid, arrived=time.monotonic(), empty_since=None)
    plat = model.platform(pid)
    s = SimpleNamespace(platform=plat, where=SimpleNamespace(x=x, y=plat.y_at(x)), monsters=monsters, t=t, dot=None,
                        players=[])
    g.h.memory.update(t, monsters, None)
    p.belief.update(t, g.h.memory)
    return g, s


def test_at_a_stop_she_stays_while_it_beats_going_on_and_never_waits_for_respawns(monkeypatch):
    from dataclasses import replace
    model, ids, far, p = _column()
    p.replan(0.0, force=True)
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    # A Horny Mushroom on her platform 100 units off, in reach: its 41 EXP over its kill (11.8 EXP/s) beats the loop's
    # rate: she stays.
    m = _sight(model, HORNY, 250, 0)
    g, s = _at_stop(model, p, ids[0], 150.0, [m], 1.0)
    assert g._held(s) and g._stay_gain(s, g._held(s)) > p.plan.rate
    assert g._leaving(s, g._held(s)) is None
    # The loop worth more than that (say 20 EXP/s): she goes on, its EXP left to her passing again.
    p.plan = replace(p.plan, rate=20.0)
    for dt in (0.0, grind.LEAVE_CONFIRM_S + 0.01):
        clock[0] += dt
        reason = g._leaving(s, g._held(s))
    assert reason == "11.8 EXP/s here, going on gives 20.0"
    # Her kill emptied the point a moment ago: nothing comes back within REFILL_D0, so she doesn't wait for it; and
    # watched empty after that, its respawn's few EXP a second (Patrol.wait_gain) don't hold her either.
    p.replan(0.0, force=True)
    g, s = _at_stop(model, p, ids[0], 150.0, [], 2.0)
    q = next(i for i, _ in p.stops[g.stop].cover)
    p.belief.empty_since[q] = p.belief.killed[q] = p.belief.empty_seen[q] = 1.9
    assert p.wait_gain(g.stop, 2.0) == 0.0
    assert g._leaving(s, []) is None  # (LEAVE_CONFIRM_S first)
    clock[0] += grind.LEAVE_CONFIRM_S + 0.01
    assert g._leaving(s, []) == "nothing left in reach"
    p.belief.empty_seen[q] = 30.0
    assert 0 < p.wait_gain(g.stop, 30.0) < 0.5 * p.plan.rate


def test_on_her_way_to_a_stop_a_pile_seen_elsewhere_takes_her_there(monkeypatch):
    from gamelab.behavior.hunt import Hunt
    model, ids, far, p = _column()
    p._take(p.rate([p.stop_on(ids[0]), p.stop_on(far)], 0.0), 0.0, "test")
    monkeypatch.setattr(Hunt, "_interrupt", lambda self, s: False)  # none of the hunt loop's reasons
    g, s = _at_stop(model, p, ids[0], 150.0, [], 1.0)
    g.stop, g.visit = p.stop_on(far), SimpleNamespace(ground=far, arrived=None, empty_since=None)
    g.obj = Objective("spot", far, p.stops[p.stop_on(far)].x, ground=far)
    assert g._interrupt(s) is False and not g._reroute  # nothing seen: on to the far stop
    # One seen on the platform above (2 s off, the far stop 30): a stop on the way's worth more.
    g.h.memory.update(1.5, [_sight(model, HORNY, 150, -200)], None)
    s.t, g._alt = 1.5, None
    assert g._interrupt(s) is True and g._reroute
    # Rerouted to it at the trip's next pass (Patrol.choose on her way to the far stop) ...
    g._reroute = False
    j, info = p.choose(1.5, ids[0], 150.0, heading=p.stop_on(far))
    assert p.platform_of(j) == ids[1] and info["why"] == "pulled"
    g.stop, g.visit = j, SimpleNamespace(ground=ids[1], arrived=None, empty_since=None)
    g.obj = Objective("spot", ids[1], p.stops[j].x, ground=ids[1])
    # ... and its monster shot in passing on the way: back to the far stop, the loop's, at the next look.
    q = next(q.id for q in p.points if q.platform == ids[1])
    g.h.memory = Memory(model)
    p.belief.empty_since[q] = p.belief.empty_seen[q] = 1.8
    s.t, g._alt = 2.0, None
    assert g._interrupt(s) is True and g._reroute
    assert g._alt[2] == p.stop_on(far) and g._alt[3]["why"] == "back to the loop"


def test_a_stop_is_the_safest_place_that_reaches_as_much(monkeypatch):
    # One platform with two Horny Mushroom points (250, 350): she gets at both wherever she stands on it, so with no
    # danger along it (the default's is the platform's) its stop is its middle; standing clear of its points
    # (patrol.STOP_SLACK, off by default), past where her attack from the ground surely takes a respawn there
    # (reach.ground_sure: 105 for a mushroom), as near its middle as that goes (465). With a danger hook highest at 250
    # (the positioning model's shape: Danger.at), the safest place reaching as much, its far end.
    model = _map([(0, 600, 0), (2000, 2600, 0)], [(250, 0, "h"), (350, 0, "h"), (2300, 0, "h")])
    a = _pid(model, 300, 0)
    flat = _patrol(model, lambda a, b: 5.0)
    assert flat.stops[flat.stop_on(a)].x == pytest.approx(300.0, abs=patrol.STAND_STEP)
    monkeypatch.setattr(patrol, "STOP_SLACK", 0.05)
    x = _patrol(model, lambda a, b: 5.0).stops[flat.stop_on(a)].x
    assert x - 350 >= reach.ground_sure(28.0)[0] and x == pytest.approx(465.0)
    monkeypatch.undo()
    p = patrol.Patrol(model, {HORNY}, {HORNY: 41.0}, flat.travel,
                      danger=lambda pid, x: 0.3 * math.exp(-abs(x - 250) / 100) if pid == a else 0.0)
    plat = model.platform(a)
    assert p.stops[p.stop_on(a)].x == pytest.approx(plat.right - reach.EDGE_MARGIN)
    assert p.stops[p.stop_on(a)].cover == flat.stops[flat.stop_on(a)].cover  # (as much)


def test_a_pause_moves_grindings_clocks_on():
    # Looks failing their checks for 30 s (Hunt.run's pause: Senses.regain): the hunt loop's clocks move on (the stop's
    # stay, STOP_MAX_S's), and grinding's own: her EXP a second over the run, where she'd go instead looked at afresh,
    # and the HP gone over the pause not charged to a platform.
    from gamelab.attack.spots import Spot
    from gamelab.behavior.hunt import Visit
    g, now = object.__new__(Grind), time.monotonic()
    g._progress, g.obj, g.stack, g._idle_since, g._hanging, g._rope_rest = now, None, [], None, None, 0.0
    g.visit = Visit(23, Spot(23, 8.0, 1.0, False), arrived=now - 5.0)
    g._start, g._alt, g._hp = now - 60.0, (now, None, 3, {}), (now, 0.9, 23)
    g._paused(30.0)
    assert g._start == pytest.approx(now - 30.0) and g._alt is None and g._hp is None
    assert g.visit.arrived == pytest.approx(now + 25.0) and g._progress == pytest.approx(now + 30.0)


def test_a_grind_stands_still_while_its_looks_fail_and_grinds_on_after():
    """Grinding is the hunt loop's subclass, so its pause is the hunt loop's (Hunt.run: LookLost, Senses.regain): from
    20 s to 35 s of loop_tree_grind's 90 s every look fails its checks; it stands still, says why, sends no plan from
    LOST_S after the first failure until the looks pass, and grinds on after, its stop not left for the time stood."""
    import dataclasses

    from gamelab.perception.framecheck import BadLook, Problem
    from gamelab.sim.scenarios import _grind as grinding
    if scenarios.real_map(10002073) is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data")
    window = (20.0, 35.0)

    def run(sim):
        sim.fail_frames(window[0], window[1],  # (perception's frames: every look's)
                        Problem("offset", "the capture is offset by (+26, +26) px from the game's client area"))
        return grinding(1.5)(sim)

    sc = dataclasses.replace(scenarios.SCENARIOS["loop_tree_grind"], name="loop_tree_grind_lost", fn=run, expects=())
    trace = scenarios.run_scenario(sc)
    log = trace["log"]
    assert any("standing still: the capture is offset" in line for line in log), log
    assert any("the game looks as expected again" in line for line in log)
    failed = next(float(line.split()[0]) for line in log if "a look failed its checks" in line)
    assert [p for p in trace["plans"] if failed + 0.5 <= p[0] < window[1]] == []
    assert any(p[0] >= window[1] for p in trace["plans"])  # grinding on after
    after = [line for line in log if float(line.split()[0]) >= window[1]]
    assert not any(f"{grind.STOP_MAX_S:.0f} s here" in line for line in after[:20]), after[:20]
    assert trace["result"]["hunt_stop"] == "time up"


def test_her_kill_and_another_players_seen_dead_in_one_look_both_reach_the_belief():
    """Her kill (the ledger's, on its track) and another player's the feed found dead (Ledger.dead, Ledger.theirs) in one
    look: both points emptied, her kill's timing a refill (`killed`), the other's not; by another player, hers is one
    left to her (kept_by), the other's one taken (gone_by). (Counted as 'dead', the other's event pushed hers out of
    what the belief read, and was taken for hers: in loop_tree_grind the share another player takes of Green Mushrooms
    went from master's 0.8 to the prior 0.5: review of 2026-10-27, review-behaviour/memcount.py.)"""
    model = _map([(0, 400, 0)], [(100, 0, "h"), (300, 0, "h")])
    pid = _pid(model, 200, 0)
    p = _patrol(model, lambda a, b: 2.0)
    kills, dead, theirs = [], {}, set()
    m = Memory(model, kills=lambda: kills, dead=lambda: dead, theirs=lambda: theirs)
    stranger = [Player(1, "stranger", 300.0, 0.0, pid, (0.0, 0.0), 1.0, 1.0)]
    mob = Mob(HORNY, 0, 0, 0, 0, 1.0)
    m.update(1.0, [Monster(mob, 100.0, 0.0, pid, None, 0, "", 11), Monster(mob, 300.0, 0.0, pid, None, 0, "", 12)],
             None, stranger)
    p.belief.update(1.0, m, stranger)
    kills.append(Kill(1.5, HORNY, pid, 100.0, 0.0, pid, HORNY, 1.3, 11))
    dead.update({11: 1.5, 12: 1.5})
    theirs.add(12)
    m.update(1.5, [], None, stranger)
    p.belief.update(1.5, m, stranger)
    hers, other = sorted((q for q in p.points), key=lambda q: q.x)
    b = p.belief
    assert [w for w, _, _ in m.events] == ["new", "new", "killed", "taken"]
    assert b.empty_since[hers.id] == b.empty_since[other.id] == 1.5
    assert hers.id in b.killed and other.id not in b.killed
    assert b.kept_by[HORNY] == 1 and b.gone_by[HORNY] == 1
