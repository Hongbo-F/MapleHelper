"""The hunt loop's pile rules (behavior.hunt), on stand-ins: no map, no simulator. The loop itself runs
end to end in the simulator's loop_* scenarios."""

import time
from types import SimpleNamespace

from gamelab.attack.spots import Spot
from gamelab.behavior import hunt
from gamelab.behavior.hunt import Hunt, Objective, Visit
from gamelab.behavior.ledger import Pile


def _hunt(piles, kills=(), travel=5.0):
    """A Hunt with only what the pile rules read: the drop store, the looter, route times, the visit."""
    h = object.__new__(Hunt)
    h.h = SimpleNamespace(drops=SimpleNamespace(known=list(piles)), looter=SimpleNamespace(inventory_full=False))
    h.nav = SimpleNamespace(planner=SimpleNamespace(travel=lambda src, x, dst: travel))
    h.visit = Visit(1, Spot(1, 0.0, 1.0, True), kills=list(kills))
    h._exit_piles = None
    return h


def _scene():
    return SimpleNamespace(platform=SimpleNamespace(id=1), where=SimpleNamespace(x=0.0))


def _pile(age, platform=2, hers=True, **kw):
    return Pile(100.0, 0.0, platform, time.monotonic() - age, hers, **kw)


def test_a_pile_is_urgent_only_when_its_time_is_nearly_up_and_it_is_close():
    # 5 s away (+1 s to land, +0.6 s on it) with the 30 s margin: urgent once under 36.6 s of its life is left.
    old, young = _pile(hunt.DROP_LIFE - 30), _pile(10)
    assert _hunt([young, old])._urgent_pile(_scene()) is old
    assert _hunt([young])._urgent_pile(_scene()) is None
    far = _hunt([old], travel=hunt.URGENT_TRAVEL + 1)  # farther ones wait until she leaves the ground
    assert far._urgent_pile(_scene()) is None
    # Too late to make it: 5 s of life left, 6.6 s to get it.
    assert _hunt([_pile(hunt.DROP_LIFE - 5)])._urgent_pile(_scene()) is None


def test_piles_no_kill_of_hers_explains_wait_and_a_pile_is_tried_twice_at_most():
    h = _hunt([])
    assert not h._fresh(_pile(10, hers=False)) and h._fresh(_pile(hunt.OTHERS_AFTER + 1, hers=False))
    assert h._fresh(_pile(10, tries=1)) and not h._fresh(_pile(10, tries=hunt.PILE_TRIES))
    assert not h._fresh(_pile(hunt.DROP_LIFE + 1)) and not h._fresh(_pile(10, status="gone"))
    h.h.looter.inventory_full = True  # items can't be picked up any more; mesos still can
    assert not h._fresh(_pile(10)) and h._fresh(_pile(10, mesos=True))


def test_lull_loot_only_while_a_respawn_of_hers_is_due_and_back_in_time():
    pile = _pile(10)
    now = time.monotonic()
    assert _hunt([pile])._lull_pile(_scene()) is None  # nothing due back: the ground is empty, she leaves
    # Due back in 8 s; the pile takes 6.6 s (5 s there, 1 s to land, 0.6 s on it).
    assert _hunt([pile], kills=[now - hunt.RESPAWN + 8])._lull_pile(_scene()) is pile
    assert _hunt([pile], kills=[now - hunt.RESPAWN + 6])._lull_pile(_scene()) is None


def test_leaving_takes_only_the_piles_left_when_she_decided():
    a, b = _pile(10), _pile(10, platform=3)
    h = _hunt([a])
    h._exit_piles = [a]
    h.h.drops.known.append(b)  # a kill on the way to a: its pile isn't gone for before she leaves
    assert h._best_pile(_scene(), near_only=False, among=h._exit_piles) is a
    a.tries = hunt.PILE_TRIES
    assert h._best_pile(_scene(), near_only=False, among=h._exit_piles) is None


def test_an_objective_s_clock_runs_while_it_is_suspended():
    o = Objective("spot", 1, 0.0)
    o.started -= hunt.OBJECTIVE_S
    assert o.age() >= hunt.OBJECTIVE_S


def test_a_visit_s_length_is_blended_with_a_prior_so_an_empty_ground_doesnt_look_best():
    h = _hunt([])
    h.tour = SimpleNamespace(stats={"36": {"visits": 3, "exp": 40.0, "timed": 1, "seconds": 0.6}})
    assert h._visit_s(36) == (hunt.PRIOR * hunt.VISIT_S + 0.6) / (hunt.PRIOR + 1)  # 40 s, not 0.6 s
    assert h._visit_s(17) == hunt.VISIT_S  # never timed


def test_a_pile_objective_no_longer_worth_it_ends_the_trip_to_it():
    pile = _pile(10, status="gone")  # seen gone on the way
    h = _hunt([pile])
    h.h.low_hp = 0.3
    h.obj = Objective("pile", 2, 100.0, pile=pile)
    s = SimpleNamespace(hp=1.0, platform=SimpleNamespace(id=1), where=SimpleNamespace(x=0.0, y=0.0), monsters=[])
    assert h._interrupt(s)
    pile.status = "seen"  # still there: the trip goes on
    h.kinds = None
    assert not h._interrupt(s)


def test_kills_count_for_the_visit_only_on_its_ground_or_from_it_and_its_spot():
    from gamelab.behavior.hunt import HuntReport
    from gamelab.behavior.ledger import Kill
    h = _hunt([])
    now = time.monotonic()
    h.visit = Visit(36, Spot(36, 540.0, 1.0, False), arrived=now - 30)
    h.tour = SimpleNamespace(exp={"pig": 13, "slime": 12})
    h.kinds = {"pig", "slime"}
    h.report, h.obj, h._fs_key, h._gone, h._kills_seen = HuntReport(), None, None, 0, 0
    h.h.report = SimpleNamespace(exp_kills=0)
    h.h.drops.gone_tried = 0
    h.h.ledger = SimpleNamespace(kills=[Kill(now, "pig", 36, 500.0, 575.0, 36),   # on the ground: hers
                                        Kill(now, "slime", 17, 900.0, -335.0, 17)])  # on 17, while she's on 17
    h._book(SimpleNamespace(platform=SimpleNamespace(id=17)))
    assert h.visit.exp == 13 and len(h.visit.kills) == 1 and h.report.on_the_way == 1 and h.report.exp == 25


def test_potions_are_out_only_after_two_in_a_row_bring_no_hp_back(monkeypatch):
    monkeypatch.setattr(hunt.time, "sleep", lambda s: None)
    h = _hunt([])
    hp = iter([0.40, 0.70, 0.40, 0.40])  # a miss, a potion that worked, then two misses
    h.h.look = lambda mobs=True: SimpleNamespace(hp=next(hp))
    h.h.potion_hp = 0.45
    h.log, h._potions_out, h._potion_misses = (lambda m: None), False, 0
    for _ in range(3):
        h._check_potion(0.40)
    assert not h._potions_out
    h._check_potion(0.40)
    assert h._potions_out and h.h.potion_hp == 0.45  # she still presses the key: a false alarm costs nothing


def test_the_ground_is_judged_empty_only_from_looks_that_cover_it_for_a_while():
    h = _hunt([])
    h.visit = Visit(17, Spot(16, 636.0, 0.35, True), arrived=time.monotonic() - 30)
    h.kinds, h._skip, h._idle_since = None, {}, None
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=17, left=861.0, right=1389.0))
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: [])

    def look(pid, strips):
        return SimpleNamespace(platform=SimpleNamespace(id=pid), looked=strips, monsters=[],
                               where=SimpleNamespace(x=0.0, y=0.0))
    assert h._leaving(look(36, [(36, 129, 951)])) is None  # down on 36, 17 out of view: no verdict
    assert h._leaving(look(20, [(17, 855, 886)])) is None and h.visit.empty_since is None  # a sliver of it
    on_spot = look(16, [(16, 501, 669), (17, 861, 1300)])  # 83% of 17
    assert h._leaving(on_spot) is None and h.visit.empty_since is not None  # empty: to be confirmed
    assert h._leaving(look(16, [(16, 501, 669)])) is None and h.visit.empty_since is not None  # band look: kept
    h.visit.empty_since -= hunt.EMPTY_CONFIRM_S
    assert h._leaving(on_spot) == "nothing left here, and nothing due back" and 17 in h._skip
