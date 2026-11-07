"""The loot chain (behavior.lootchain; the user, 2026-11-04: "plan for a few loot targets in advance so you don't need to have this
'pauses'"): the plan's shape from frozen scenes on Tree Dungeon, the walk legs it's made of (motion.flow), the way through them (the
navigator's `through`), a monster in the way, what a stand told the hunt told by a run instead, and the switch."""

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gamelab.behavior import grind as grind_mod
from gamelab.behavior import lootchain as lc
from gamelab.behavior.grind import TO_TARGET, Grind
from gamelab.behavior.hunt import PILE_TRIES, Objective
from gamelab.behavior.ledger import Pile
from gamelab.control import navigator as nv
from gamelab.control.navigator import Navigator, TripState, _arrows_in, _ran_over
from gamelab.localization.monsters import MonsterTrack
from gamelab.motion import flow as fl
from gamelab.motion.flow import Body, Flow
from gamelab.motion.route import Move, Planner
from gamelab.sim import scenarios

KEYS = {"left": "L", "right": "R", "jump": "J", "up": "U", "down": "D", "pick": "P"}


@pytest.fixture(scope="module")
def tree():
    return scenarios.real_map(10002073)


@pytest.fixture(scope="module")
def planner(tree):
    return Planner(tree)


def _pile(x, platform=9, xs=None, tries=0, items=None, status="seen"):
    xs = (x,) if xs is None else tuple(xs)
    p = Pile(x, 0.0, platform, 0.0, hers=True, status=status, xs=xs, items=tuple(items or ("mesos",) * len(xs)), tries=tries, last_seen=0.0)
    return p


def _chain(*piles, dest=None):
    return lc.LootChain([lc.Stretch(p, lc.bunches(p.xs), (), p.tries > 0) for p in piles], dest)


def _enter(a, ax, b, lo, hi):
    return lo  # (from the left, whatever: the stand-in's landing)


# --- the plan's shape -------------------------------------------------------------------------------------------------

def test_drops_closer_than_the_pile_gap_are_a_bunch():
    assert lc.bunches([10.0, 12.0, 60.0, 40.0]) == [(11.0, 2), (40.0, 1), (60.0, 1)]


def test_a_run_gives_two_presses_she_stands_only_for_the_rest():
    assert lc.stand_s(1) == 0.0 and lc.stand_s(2) == 0.0
    assert lc.stand_s(3) == pytest.approx(lc.PRESS_S)
    assert lc.stand_s(4) == pytest.approx(2 * lc.PRESS_S)
    assert lc.stand_s(1, again=True) == pytest.approx(2 * lc.PRESS_S)  # (a drop still there after a run: a stand's presses)


def test_three_piles_along_her_platform_are_one_walk_to_the_last():
    way = _chain(_pile(50.0, xs=(40.0, 60.0)), _pile(100.0), _pile(148.0)).way(9, 0.0, _enter)
    assert way == [(9, 148.0, 0.0)]


def test_it_goes_at_the_drops_as_seen_not_at_the_piles_middle():
    p = _pile(0.0, xs=(25.0, 70.0))  # (the kill died at 0: its pile's middle is no drop's place)
    assert _chain(p).way(9, 0.0, _enter) == [(9, 70.0, 0.0)]
    assert _chain(p).way(9, 100.0, _enter) == [(9, 25.0, 0.0)]  # (from the other side, the far end's the other drop)


def test_a_turn_between_piles_is_two_walks_with_no_stand():
    assert _chain(_pile(100.0), _pile(-50.0)).way(9, 0.0, _enter) == [(9, 100.0, 0.0), (9, -50.0, 0.0)]


def test_a_bunch_of_three_is_stood_at_for_its_third_press():
    way = _chain(_pile(50.0, xs=(48.0, 50.0, 52.0)), _pile(100.0)).way(9, 0.0, _enter)
    assert way == [(9, 50.0, pytest.approx(lc.PRESS_S)), (9, 100.0, 0.0)]


def test_a_pile_on_another_platform_starts_at_its_end_nearer_her_landing():
    way = _chain(_pile(100.0), _pile(0.0, platform=8, xs=(-40.0, 40.0))).way(9, 0.0, lambda a, ax, b, lo, hi: 38.0)
    assert way == [(9, 100.0, 0.0), (8, 40.0, 0.0), (8, -40.0, 0.0)]


def test_where_she_goes_on_to_is_the_last_of_the_way():
    assert _chain(_pile(100.0), dest=(8, 30.0)).way(9, 0.0, _enter) == [(9, 100.0, 0.0), (8, 30.0, 0.0)]
    assert _chain(_pile(100.0), dest=(9, 130.0)).way(9, 0.0, _enter) == [(9, 130.0, 0.0)]  # (on past it, the same way)


def test_under_her_already_it_presses_all_the_same():
    assert _chain(_pile(0.0)).way(9, 0.0, _enter) == [(9, 0.0, pytest.approx(lc.PRESS_S))]
    assert _chain(_pile(0.0), _pile(-50.0)).way(9, 0.0, _enter) == [(9, -50.0, 0.0)]  # (with a walk on, that walk's first press)


def test_a_stretch_run_over_is_off_the_way_and_what_was_cut_short_stays():
    c = _chain(_pile(50.0), _pile(100.0))
    c.book([(9, 0.0, 60.0)])
    assert c.stretches[0].done and not c.stretches[1].done and not c.done
    assert c.way(9, 60.0, _enter) == [(9, 100.0, 0.0)]
    c.book([(8, 60.0, 200.0)])  # (another platform's run takes nothing)
    assert not c.stretches[1].done
    c.book([(9, 60.0, 90.0)])  # (15 of it: within reach)
    assert c.done


def test_the_log_says_it_in_a_line():
    c = _chain(_pile(-33.0), _pile(100.0, platform=8), _pile(148.0, platform=8))
    c.est_s = 2.9
    assert c.text() == "3 stops, x -33 on 9, x 100 on 8, x 148 on 8: 2.9 s"
    assert c.record()["planned"] == 3 and c.record()["done"] == 0


# --- the walk legs -------------------------------------------------------------------------------------------------

def _flow(m):
    return Flow(m, KEYS)


def _walk(pid, x, stand=0.0):
    return Move("walk", pid, pid, x, x, "right", stand)


def _arrows(k):
    return [e for e in k.sorted() if e[1] in ("L", "R")]


def test_a_chain_of_walks_holds_the_arrow_through_them_pick_up_pressed_all_the_way(tree):
    f, p = _flow(tree), tree.platform(10)
    a = p.left + 20.0
    legs, k, why = f.chain([_walk(10, a + 40.0), _walk(10, a + 90.0), _walk(10, a + 130.0)], Body(0.0, a, p))
    assert why == "done" and [lg.end.x for lg in legs][:2] == [pytest.approx(a + 40.0), pytest.approx(a + 90.0)]
    arrows = _arrows(k)
    assert [e[2] for e in arrows] == ["down", "up"]  # (no letting go between them)
    assert legs[-1].end.run == 0 and abs(legs[-1].end.x - (a + 130.0)) < 1.5  # (she stands at the last)
    picks = [e[0] for e in k.sorted() if e[1] == "P" and e[2] == "down"]
    assert len(picks) >= int(legs[-1].end.t / fl.WALK_PICK_EVERY) - 1
    assert max(b - a_ for a_, b in zip(picks, picks[1:])) <= 1000 * fl.WALK_PICK_EVERY + 2


def test_a_turn_is_the_other_arrow_at_once_no_stop_first(tree):
    f, p = _flow(tree), tree.platform(10)
    a = (p.left + p.right) / 2
    legs, k, why = f.chain([_walk(10, a + 50.0), _walk(10, a - 20.0)], Body(0.0, a, p))
    assert why == "done"
    downs = [e[1] for e in _arrows(k) if e[2] == "down"]
    assert downs == ["R", "L"]
    assert legs[0].end.run == 1  # (still running at the first)


def test_a_stand_at_a_bunch_stops_there_for_its_presses(tree):
    f, p = _flow(tree), tree.platform(10)
    a = p.left + 20.0
    legs, k, why = f.chain([_walk(10, a + 60.0, 0.24), _walk(10, a + 120.0)], Body(0.0, a, p))
    assert why == "done" and legs[0].end.run == 0
    stand = legs[0].spans[-1]
    assert stand.x0 == stand.x1 and stand.t1 - stand.t0 == pytest.approx(0.24)
    picks = [e[0] for e in k.sorted() if e[1] == "P" and e[2] == "down"]
    assert sum(stand.t0 * 1000 - 1 <= t < stand.t1 * 1000 for t in picks) == 2


def test_a_run_through_a_drop_by_the_platforms_end_stops_rather_than_runs_off(tree):
    f, p = _flow(tree), tree.platform(10)
    goal = p.right - 5.0
    legs, k, why = f.chain([_walk(10, goal), _walk(10, goal - 60.0)], Body(0.0, goal - 80.0, p))
    assert why == "done" and legs[0].end.run == 0 and p.covers(legs[0].end.x)


def _stays_on(legs, p):
    """Every span and where a leg's run stops (its turn, its slide) on platform `p` or the platform it's on."""
    for leg in legs:
        for sp in leg.spans:
            on = tree_of(leg, sp.platform)
            assert on.left <= min(sp.x0, sp.x1) and max(sp.x0, sp.x1) <= on.right, (leg.move, sp)
        e = leg.end
        if e.run:
            stop = e.x + e.run * fl.body.stop_distance(e.v, fl.SHOE, fl.LAGS)
            assert e.platform.left <= stop <= e.platform.right, (leg.move, stop)


def tree_of(leg, pid):
    return leg.start.platform if leg.start.platform.id == pid else leg.end.platform


@pytest.mark.parametrize("gap", range(10, 61, 5))
@pytest.mark.parametrize("side", [1, -1], ids=["right", "left"])
def test_a_walk_then_a_walk_stays_on_the_platform_however_near_its_end_the_first_goal(tree, gap, side):
    """(The safety review of 2026-11-05: at WALK_ON_ROOM 20 the plan itself ran 0.6-4.6 past the end with a first goal 20-24 from it.)"""
    f, p = _flow(tree), tree.platform(10)
    end = p.right if side > 0 else p.left
    goal = end - side * gap
    for second in (goal - side * 60.0, goal + side * min(gap - 3.0, 5.0)):  # (back from it, and on towards the end)
        legs, _, why = f.chain([_walk(10, goal), _walk(10, second)], Body(0.0, goal - side * 80.0, p))
        assert legs and why in ("done", "unpredictable")
        _stays_on(legs, p)


@pytest.mark.parametrize("gap", range(10, 61, 10))
def test_a_landing_running_into_a_walk_stays_on_the_platform_however_near_its_end_the_goal(tree, planner, gap):
    """Her route's jump, then a walk near the end of the platform it lands on, ahead of her landing and behind it."""
    f = _flow(tree)
    moves = planner.plan(9, 0.0, 8)
    q = tree.platform(8)
    start = Body(0.0, 0.0, tree.platform(9))
    for goal in (q.right - gap, q.left + gap):
        legs, _, why = f.chain([*moves, _walk(8, goal)], start)
        assert legs
        _stays_on(legs, q)


def test_running_with_no_room_for_her_turn_a_walk_back_isnt_on(tree):
    """A landing running at the end: a walk the other way isn't a way from there (the leg before is done the way that stops, `follows`)."""
    f, p = _flow(tree), tree.platform(10)
    b = Body(0.0, p.right - 12.0, p, run=1, landed=True, v=fl.WALK_SPEED)
    assert f.leg(_walk(10, p.right - 100.0), b, fl.Keys(KEYS), "stop") is None
    assert f.leg(_walk(10, p.right - 100.0), b, fl.Keys(KEYS), "run") is None
    far = Body(0.0, p.left + 200.0, p, run=1, landed=True, v=fl.WALK_SPEED)
    assert f.leg(_walk(10, p.left + 100.0), far, fl.Keys(KEYS), "stop") is not None


def test_a_monster_in_the_way_of_the_second_walk_ends_the_chain_short_of_it(tree):
    """Her whole box over the whole way: a walk's legs are checked as any leg (flow.danger): the chain stops short, standing clear."""
    f, p = _flow(tree), tree.platform(10)
    a = p.left + 20.0
    mo = MonsterTrack(10, a + 240.0, 0.0, 0, 20.0, 0.0, 0.0, 40.0, 0, "horny-mushroom", 40.0)
    legs, k, why = f.chain([_walk(10, a + 40.0), _walk(10, a + 110.0), _walk(10, a + 220.0)], Body(0.0, a, p), [mo], hops=False)
    assert why == "monster"
    assert 1 <= len(legs) <= 2 and legs[-1].end.x <= a + 110.0 + 15.0


# --- the way through (the navigator) ------------------------------------------------------------------------------

def _nav(m, planner):
    nav = Navigator.__new__(Navigator)
    nav.model, nav.planner = m, planner
    nav._weights = {}
    nav.weight = lambda move: 1.0
    return nav


def test_the_way_through_is_the_routes_moves_with_a_walk_to_each_stop(tree, planner):
    nav = _nav(tree, planner)
    st = TripState(8, stops=[(9, 60.0, 0.0), (8, 100.0, 0.0)], through=lambda s: [])
    path = nav._way_through(st, 9, 0.0)
    assert path[0].kind == "walk" and path[0].land_x == 60.0 and path[-1].kind == "walk" and path[-1].land_x == 100.0
    assert all(m.kind != "walk" for m in path[1:-1]) and len(path) >= 3
    assert path[1].src == 9 and path[-2].dst == 8


def test_a_route_with_her_walk_on_from_its_landing_counted(tree, planner):
    """`Planner.plan`'s dst_x: the quickest with her walk on from where it lands her, never dearer than the quickest onto the platform and the walk."""
    src, dst, x, goal = 9, 8, 0.0, 200.0
    first = planner.plan(src, x, dst)
    moves = planner.plan(src, x, dst, dst_x=goal)
    assert moves and moves[-1].dst == dst
    cost = lambda ms: sum(m.seconds for m in ms)
    assert cost(moves) + abs(moves[-1].land_x - goal) / fl.WALK_SPEED <= cost(first) + abs(first[-1].land_x - goal) / fl.WALK_SPEED + 1.0


def test_a_walk_made_is_its_stop_off_the_trips_way():
    st = TripState(8, stops=[(10, 60.0, 0.0), (8, 100.0, 0.0)], through=lambda s: [])
    Navigator._gone_by(st, [Move("jump", 9, 8, 0.0, 0.0), _walk(10, 60.0)])
    assert st.stops == [(8, 100.0, 0.0)]


def test_a_twins_pile_is_booked_and_its_stop_made_by_the_platform_it_lies_on():
    """(Map 10004021 has twins; her moves and spans have the platform they lie on, `MapModel.place_of`.)"""
    c = _chain(_pile(50.0, platform=77))
    c.book([(7, 0.0, 60.0)])
    assert not c.done
    c.book([(7, 0.0, 60.0)], lambda p: 7 if p == 77 else p)
    assert c.done
    nav = Navigator.__new__(Navigator)
    nav.model = SimpleNamespace(place_of=lambda p: 7 if p == 77 else p)
    nav._penalty = lambda st: None
    st = TripState(7, stops=[(77, 60.0, 0.0)], through=lambda s: [])
    path = nav._way_through(st, 77, 0.0)
    assert path[0].src == 7
    Navigator._gone_by(st, path)
    assert st.stops == []


def test_a_stand_where_she_is_leaves_her_facing_as_it_was(tree):
    """No arrow in the plan: `Actions.facing` isn't told of a move, and the throw's retap stands (the safety review's note)."""
    f, p = _flow(tree), tree.platform(10)
    a = (p.left + p.right) / 2
    still, _, _ = f.chain([_walk(10, a, 0.12)], Body(0.0, a, p))
    assert not _arrows_in(still)
    moved, _, _ = f.chain([_walk(10, a + 60.0)], Body(0.0, a, p))
    assert _arrows_in(moved)


def test_a_replanned_chain_never_taken_ran_her_over_nothing(tree):
    f, p = _flow(tree), tree.platform(10)
    a = p.left + 20.0
    legs, _, _ = f.chain([_walk(10, a + 50.0), _walk(10, a + 120.0)], Body(0.0, a, p))
    assert _ran_over(legs, (legs, 0.0)) == []


def test_what_a_chain_ran_her_over_cut_short_is_only_as_far_as_then(tree):
    f, p = _flow(tree), tree.platform(10)
    a = p.left + 20.0
    legs, _, _ = f.chain([_walk(10, a + 50.0), _walk(10, a + 120.0)], Body(0.0, a, p))
    assert _ran_over(legs)[-1][2] == pytest.approx(legs[-1].end.x)
    cut_at = legs[1].spans[0].t0 + 0.1
    ran = _ran_over(legs, (legs, cut_at))
    assert a + 50.0 < ran[-1][2] < a + 120.0


# --- the hunt's end of it -----------------------------------------------------------------------------------------

def _grind():
    g = Grind.__new__(Grind)
    g.log = lambda text: None
    g.said = []
    g.tp = SimpleNamespace(chain=True, screen=True, choice=None, counts=__import__("collections").Counter(), piles_wait=lambda: False,
                           pile_first=lambda p, stop, now: True)
    g.h = SimpleNamespace(looter=SimpleNamespace(wanted=lambda near: near, unpickable=[], can_take=lambda k: True, stood_on=lambda kinds, pile: g.said.append((kinds, pile)),
                                                 pending=[], cant_pick=lambda stuck: g.said.append(("stuck", stuck)), inventory_full=False),
                          report=SimpleNamespace(loot_runs=0), drops=SimpleNamespace(tried=lambda p: setattr(p, "tries", p.tries + 1)))
    g.report = SimpleNamespace(piles=0)
    g.stack, g.obj = [], None
    g._stood, g._pile_of = {}, {}
    g._chain_off = -1e18
    g._idle_since = 1.0
    g.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(left=-1000.0, right=1000.0), place_of=lambda pid: pid)
    g._respawn_guard = lambda s, p, at: None
    return g


def _stop(p, leave=1.0, guards=(), lands=0.0):
    return SimpleNamespace(spot=SimpleNamespace(key=p.key, guards=guards, lands=lands), leave=leave, arrive=max(0.0, leave - 0.2))


def _look(loot=()):
    return SimpleNamespace(loot=list(loot), platform=SimpleNamespace(id=9), where=SimpleNamespace(x=0.0))


def _with_itinerary(g, piles, stops=None, dest=None):
    stops = stops or [_stop(p, 1.0 + i) for i, p in enumerate(piles)]
    g._sweeper = SimpleNamespace(current=SimpleNamespace(stops=stops), dest=dest)
    g._pile_of = {p.key: p for p in piles}


def test_the_plan_takes_the_itinerarys_first_few_piles():
    g = _grind()
    piles = [_pile(50.0 * i) for i in range(1, 7)]
    _with_itinerary(g, piles)
    c = g._loot_plan(_look())
    assert [st.pile for st in c.stretches] == piles[:lc.MOST_PILES] and c.est_s == 1.0 + lc.MOST_PILES - 1


def test_the_plan_ends_short_of_a_monster_to_shoot_first():
    g = _grind()
    piles = [_pile(50.0), _pile(100.0), _pile(150.0)]
    _with_itinerary(g, piles, [_stop(piles[0]), _stop(piles[1], guards=(object(),)), _stop(piles[2])])
    assert [st.pile for st in g._loot_plan(_look()).stretches] == piles[:1]


def test_a_pile_not_seen_yet_or_not_landed_is_the_plans_last_to_its_middle():
    for odd in ({"status": "expected"}, {"lands": 0.5}):
        g = _grind()
        piles = [_pile(50.0), _pile(100.0, status=odd.get("status", "seen"), xs=()), _pile(150.0)]
        _with_itinerary(g, piles, [_stop(piles[0]), _stop(piles[1], lands=odd.get("lands", 0.0)), _stop(piles[2])], dest=(8, 0.0))
        g._target_place = (8, 0.0)
        c = g._loot_plan(_look())
        assert [st.pile for st in c.stretches] == piles[:2] and c.dest is None
        assert c.stretches[1].bunches == [(100.0, 1)] and not c.stretches[1].seen


def test_the_plan_ends_short_of_a_later_pile_the_spawn_guard_would_hold():
    """(The safety review: the guard asked only of a chain's first, at `_go_for`.)"""
    g = _grind()
    piles = [_pile(50.0), _pile(100.0), _pile(150.0)]
    _with_itinerary(g, piles)
    g._respawn_guard = lambda s, p, at: "a pass" if p is piles[1] else None
    assert [st.pile for st in g._loot_plan(_look()).stretches] == piles[:1]


def test_a_pile_not_seen_yet_by_a_platforms_end_is_walked_to_short_of_it():
    g = _grind()
    g.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(left=-100.0, right=100.0))
    p = _pile(98.0, status="expected", xs=())
    _with_itinerary(g, [p])
    assert g._loot_plan(_look()).stretches[0].bunches == [(100.0 - lc.END_ROOM, 1)]


def test_before_her_target_only_the_piles_that_may_go_before_it():
    g = _grind()
    piles = [_pile(50.0), _pile(100.0), _pile(150.0)]
    _with_itinerary(g, piles)
    g.tp.choice = SimpleNamespace(primary=object())
    g.tp.pile_first = lambda p, stop, now: stop.leave <= 2.0
    assert [st.pile for st in g._loot_plan(_look()).stretches] == piles[:2]
    g._sweeper.dest = (8, 0.0)  # (on her way to it: all that are on her way)
    assert [st.pile for st in g._loot_plan(_look()).stretches] == piles


def _drop(x, kind="mesos"):
    return SimpleNamespace(x=x, platform=9, mesos=kind == "mesos", item=SimpleNamespace(kind=kind))


def test_a_drop_by_a_platforms_end_is_walked_to_short_of_it():
    g = _grind()
    g.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(left=0.0, right=100.0))
    assert g._stretch(_look(), _pile(50.0, xs=(98.0,))).bunches == [(100.0 - lc.END_ROOM, 1)]


def test_a_stretch_goes_by_the_drops_as_the_store_has_them():
    g = _grind()
    p = _pile(50.0, xs=(30.0, 70.0), items=("mesos", "horny-cap"))
    st = g._stretch(_look(), p)
    assert st.bunches == [(30.0, 1), (70.0, 1)] and st.kinds == ("horny-cap",)


def test_a_stretch_leaves_out_a_full_tabs_items_and_says_so_with_nothing_left():
    g = _grind()
    g.h.looter.can_take = lambda k: False
    p = _pile(50.0, xs=(30.0, 70.0), items=("mesos", "horny-cap"))
    assert g._stretch(_look(), p).bunches == [(30.0, 1)]
    q = _pile(50.0, xs=(30.0,), items=("horny-cap",))
    assert g._stretch(_look(), q) is None and q.tries == 1


def test_a_drop_she_has_been_at_twice_that_stays_is_told_as_a_stand_tells_it():
    g = _grind()
    p = _pile(50.0, xs=(50.0,), tries=2)
    g._stood[id(p)] = (p, [50.0, 52.0])
    d = _drop(50.0)
    assert g._stretch(_look([d]), p) is None
    assert ("stuck", [d]) in g.said


def test_a_pile_a_look_sees_empty_is_no_stop(monkeypatch):
    g = _grind()
    p = _pile(50.0)
    p.empty_since = 1.0
    monkeypatch.setattr(grind_mod.Hunt, "_candidate", lambda self, p, s=None: True)
    assert not g._candidate(p)
    p.empty_since = None
    assert g._candidate(p)
    q = _pile(50.0, status="expected", xs=())  # (a kill's no look has seen: `_none_landed` says, from by it)
    q.empty_since = 1.0
    assert g._candidate(q)


def test_the_plan_goes_on_past_a_pile_with_nothing_to_take():
    g = _grind()
    g.h.looter.can_take = lambda k: False
    piles = [_pile(50.0, xs=(50.0,), items=("cap",)), _pile(100.0)]
    _with_itinerary(g, piles)
    assert [st.pile for st in g._loot_plan(_look()).stretches] == piles[1:] and piles[0].tries == 1


def _over(g, chain, ran, result="in position", why="done", blocked=False):
    g.obj = Objective("pile", chain.stretches[0].platform, chain.stretches[0].pile.x, pile=chain.stretches[0].pile, why="sweep", chain=chain)
    g.obj.trip = SimpleNamespace(ran=ran, chain_why=why, walk_blocked=blocked)
    return g._chain_over(g.obj, result)


def test_over_each_stretch_run_over_is_told_as_a_stand_tells_and_the_objective_done():
    g = _grind()
    a, b = _pile(50.0, xs=(40.0, 60.0)), _pile(100.0)
    chain = lc.LootChain([lc.Stretch(a, lc.bunches(a.xs), ("cap",)), lc.Stretch(b, lc.bunches(b.xs))])
    assert _over(g, chain, [(9, 0.0, 100.0)]) is True
    assert a.tries == b.tries == 1 and g.h.report.loot_runs == 2 and g.obj is None
    assert (["cap"], a) in g.said
    assert g.tp.counts["loot stops done"] == 2 and g._idle_since is None


def test_a_run_through_isnt_a_stand_towards_cant_pick_up_nor_one_of_her_tries():
    """(The safety review of 2026-11-05: one run and one stand said "can't pick up", the backpack's push after it, where master took two
    stands; RUN_PRESSES isn't proven live.) A run, then two stands a drop stays under: only then `cant_pick`, and the pile's still one to
    go for at that look, as after two stands it has always been."""
    g = _grind()
    a = _pile(50.0, xs=(50.0,), items=("cap",))
    drop = SimpleNamespace(platform=9, x=50.0)

    def through():
        st = g._stretch(_look([drop]), a)
        chain = lc.LootChain([st])
        assert _over(g, chain, [(9, 0.0, 60.0)]) is True
        return st

    first = through()
    assert not first.again and a.tries == 1 and a.runs == 1 and id(a) not in g._stood
    assert through().again and a.tries == 2 and g._stood[id(a)][1] == [50.0]
    assert not any(s[0] == "stuck" for s in g.said)
    assert through().again and a.tries == 3 and g._stood[id(a)][1] == [50.0, 50.0]
    assert not any(s[0] == "stuck" for s in g.said)
    assert a.tries - a.runs < PILE_TRIES  # (Hunt._candidate's: the look that has it stuck still comes)
    assert g._stretch(_look([drop]), a) is None and any(s[0] == "stuck" for s in g.said)
    assert _pile(9.0, tries=PILE_TRIES).tries - _pile(9.0).runs >= PILE_TRIES  # (no run: as ever)


def test_over_short_of_its_end_a_part_of_a_stretch_isnt_a_try_and_the_objective_is_done_with_all_the_same():
    g = _grind()
    a, b = _pile(50.0, xs=(40.0, 90.0)), _pile(200.0)
    chain = lc.LootChain([lc.Stretch(a, lc.bunches(a.xs)), lc.Stretch(b, lc.bunches(b.xs))])
    assert _over(g, chain, [(9, 0.0, 45.0)]) is True
    assert g.obj is None and a.tries == 0 and b.tries == 0


def test_a_monster_in_its_first_walk_sends_her_the_old_way_for_a_moment():
    g = _grind()
    a = _pile(50.0)
    assert _over(g, lc.LootChain([lc.Stretch(a, lc.bunches(a.xs))]), [], blocked=True) is True
    assert g.obj is None and g._chain_off > time.monotonic() and a.tries == 0


def test_a_failed_trip_is_as_any_piles():
    g = _grind()
    a = _pile(50.0)
    assert _over(g, lc.LootChain([lc.Stretch(a, lc.bunches(a.xs))]), [], "too many misses", why="") is False


# --- the switch ---------------------------------------------------------------------------------------------------

def test_the_switch(monkeypatch):
    monkeypatch.setenv("GAMELAB_LOOT_CHAIN", "0")
    assert not lc.on()
    monkeypatch.setenv("GAMELAB_LOOT_CHAIN", "1")
    assert lc.on()
    monkeypatch.delenv("GAMELAB_LOOT_CHAIN")
    assert lc.on()


def test_with_it_off_by_a_kills_drops_not_seen_yet_or_with_a_guard_her_drops_go_the_old_way(monkeypatch):
    g = _grind()
    g.tp.chain = False
    called = []
    monkeypatch.setattr(grind_mod.Hunt, "_sweep_act", lambda self, *a: called.append("old"))
    monkeypatch.setattr(grind_mod.Hunt, "_at_pile", lambda self, s, p: True)
    p = _pile(50.0)
    _with_itinerary(g, [p])
    g._sweep_act(_look(), _stop(p), [], [], 0.0)
    assert called == ["old"]
    g.tp.chain = True
    for stop, status in ((_stop(p, guards=(object(),)), "seen"), (_stop(p, lands=0.3), "seen"), (_stop(p), "expected")):
        called.clear()
        p.status = status
        g._sweep_act(_look(), stop, [], [], 0.0)
        assert called == ["old"]


def test_over_a_stretch_not_seen_yet_isnt_a_try_nor_a_stand():
    g = _grind()
    a = _pile(50.0, status="expected", xs=())
    chain = lc.LootChain([lc.Stretch(a, [(50.0, 1)], seen=False)])
    assert _over(g, chain, [(9, 0.0, 50.0)]) is True
    assert a.tries == 0 and id(a) not in g._stood and g.h.report.loot_runs == 0


# --- in the simulator, the whole way -------------------------------------------------------------------------------

def _grind_scene(drops, start=(10, -120.0), limit=40.0):
    sc = replace(scenarios.SCENARIOS["loop_tree_grind"], name="loot_chain_scene", start=start, monsters=None, drops=drops, limit=limit,
                 realism=None, expects=(), noise=None)
    trace, sim = scenarios.run_sim(sc)
    return trace, sim.world


@pytest.mark.parametrize("switch", ["1", "0"])
def test_her_drops_along_a_platform_she_runs_through_or_stops_at_each(monkeypatch, switch):
    """No monster about: her drops are all she has to do. With the chain, one run through them: no stand from the first drop she picks up to
    the last. Without, a stop at each."""
    monkeypatch.setenv("GAMELAB_LOOT_CHAIN", switch)
    drops = (("mesos", 10, -60.0), ("mesos", 10, 0.0), ("mesos", 10, 90.0))
    trace, world = _grind_scene(drops)
    taken = sorted(d.taken_at for d in world.drops if d.taken_by == "her")
    assert len(taken) == 3
    still = [t for t in world.still_log if taken[0] <= t <= taken[-1]]
    if switch == "1":
        assert any("loot: " in line for line in trace["log"])
        assert not still and taken[-1] - taken[0] < 1.6  # (150 units at her 125-150 a second)
    else:
        assert still and taken[-1] - taken[0] > 1.6


def test_a_move_and_two_stops_are_one_plan():
    """(The drop on 8 she runs over early on, on a trip: pick-up pressed on every run. The two on 10, once they're past another's ownership (OWN_S): from
    11, a climb and two walks, one plan, with no stand between them.)"""
    trace, world = _grind_scene((("mesos", 10, -60.0), ("mesos", 10, 60.0), ("mesos", 8, 100.0)))
    taken = sorted(d.taken_at for d in world.drops if d.taken_by == "her")
    assert len(taken) == 3
    assert any("loot: 2 stops" in line for line in trace["log"]) and any("climb->10, walk->10, walk->10" in line for line in trace["log"])
    assert not any(taken[-2] <= t <= taken[-1] for t in world.still_log)


# --- a monster in the way of a chain (the reviews of 2026-11-05: 101 "nothing in reach" in 126 s of loop_tree_remembered_keys) --------

def _nav_for_way(wont=False, throws=(0, 0), unshot=None):
    nav = Navigator.__new__(Navigator)
    nav._mode = "transit"
    calls = []
    counts = iter(throws)
    nav.escort = SimpleNamespace(in_way=lambda s, x: SimpleNamespace(track=8), wont_have=lambda s, m: wont,
                                 throws=lambda: next(counts), clear_way=lambda s, x, finish=True: calls.append(x) or 0)
    nav._unshot = dict(unshot or {})
    return nav, calls


def _trip_through():
    st = TripState(9, stops=[(9, 60.0, 0.0)], through=lambda s: [], mode="transit")
    st.trip = SimpleNamespace(kills=0)
    return st


def test_no_fight_in_a_chains_way_at_a_monster_no_fight_will_have():
    nav, calls = _nav_for_way(wont=True)
    assert nav._clear_the_way(_trip_through(), object(), 60.0) is False and not calls


def test_a_fight_in_a_chains_way_that_threw_nothing_isnt_tried_again_for_a_moment():
    nav, calls = _nav_for_way(throws=(3, 3))
    st = _trip_through()
    assert nav._clear_the_way(st, object(), 60.0) is True and calls == [60.0] and nav._unshot[8] > time.monotonic()
    assert nav._clear_the_way(st, object(), 60.0) is False and calls == [60.0]


def test_a_fight_in_a_chains_way_that_threw_is_no_reason_to_hold_off():
    nav, calls = _nav_for_way(throws=(3, 4, 4, 5))
    st = _trip_through()
    assert nav._clear_the_way(st, object(), 60.0) is True and 8 not in nav._unshot
    assert nav._clear_the_way(st, object(), 60.0) is True and calls == [60.0, 60.0]


def test_a_chain_sent_regardless_goes_as_far_as_the_routes_moves_never_its_walks():
    nav = Navigator.__new__(Navigator)
    asked = []
    nav._chain = lambda st, s, path, *a, **k: asked.append(list(path)) or ([1], None, "done")
    nav._pass_record = lambda *a, **k: {}
    look = SimpleNamespace(where=SimpleNamespace(platform=SimpleNamespace(id=9)))
    st = _trip_through()
    jump, walk = Move("jump", 9, 8, 0.0, 0.0), _walk(8, 60.0)
    assert nav._regardless(st, look, [jump, walk, _walk(8, 90.0)], [], 0.0, None, "why")[0] == [1]
    assert asked == [[jump]]
    asked.clear()
    assert nav._regardless(st, look, [walk], [], 0.0, None, "why", blind=True) is None and not asked
    st.through = None
    assert nav._regardless(st, look, [jump, walk], [], 0.0, None, "why")[0] == [1] and asked == [[jump, walk]]


def test_a_monster_set_aside_is_no_fight_on_her_way_with_the_whole_screen(tree):
    from gamelab.behavior import fight
    from gamelab.perception.mobs import Mob
    from gamelab.localization.monsters import Monster
    p = tree.platform(10)
    a = (p.left + p.right) / 2
    m = Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), a + 40.0, p.y_at(a + 40.0), 10, track=8)
    s = SimpleNamespace(where=SimpleNamespace(x=a), platform=p, monsters=[m])
    esc = fight.FightOnTheWay(SimpleNamespace(attacks=SimpleNamespace(), ledger=None, model=tree))
    assert esc.in_way(s, a + 100.0) is m and not esc.wont_have(s, m)
    esc.set_aside = lambda: {8: 1e18}
    assert esc.wont_have(s, m)


def test_a_kill_bound_to_a_track_that_lives_on_says_dying_only_for_its_death(monkeypatch):
    """(ledger.py's `kills[-8:]` had no limit: a live horny-mushroom walked on, no target, for 70 s of loop_tree_remembered_keys.)"""
    from gamelab.behavior.ledger import BOUND_LASTS_S, Kill, dying
    from gamelab.perception.mobs import Mob
    from gamelab.localization.monsters import Monster
    m = Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), 40.0, 0.0, 10, track=8)
    k = Kill(100.0, "horny-mushroom", 10, 40.0, 0.0, 10, track=8)
    die = {"horny-mushroom": 0.7}
    monkeypatch.delenv("GAMELAB_SCREEN", raising=False)
    monkeypatch.delenv("GAMELAB_TARGET", raising=False)
    assert dying([k], m, 100.0 + 0.7 + BOUND_LASTS_S - 0.01, [m], die)
    assert not dying([k], m, 100.0 + 0.7 + BOUND_LASTS_S + 0.01, [m], die)
    assert dying([k], m, 500.0, [m], die, dead={8})  # (the feed has it dead: however long)
    assert dying([k], m, 500.0, [], die)  # (only her memory has it: till 8 more kills, as ever)
    for off in ("GAMELAB_SCREEN", "GAMELAB_TARGET"):
        monkeypatch.setenv(off, "0")
        assert dying([k], m, 500.0, [m], die)  # (the switch off: till 8 more kills, as master)
        monkeypatch.delenv(off)


def test_a_trip_to_her_targets_place_lands_for_that_place():
    """(The words review's m4: 633 trips of the clip's replay, 0.12 s over the estimate on average, 9 by 1 s or more.)"""
    nav = Navigator.__new__(Navigator)
    asked = {}
    nav.planner = SimpleNamespace(plan=lambda src, x, dst, **k: asked.update(k) or [])
    nav.weight = None
    nav._penalty = lambda st: None
    st = TripState(8, 120.0, vary=False, land_near=True)
    nav._route(st, 9, 0.0, 8)
    assert asked["dst_x"] == 120.0
    nav._route(replace(st, land_near=False), 9, 0.0, 8)
    assert asked["dst_x"] is None
    g = Grind.__new__(Grind)
    assert g._land_near(SimpleNamespace(why=TO_TARGET)) and not g._land_near(SimpleNamespace(why="sweep"))
