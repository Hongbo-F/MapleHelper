"""The hunt loop's pile rules (behavior.hunt), on stand-ins: no map, no simulator. The loop itself runs
end to end in the simulator's loop_* scenarios."""

import time
from types import SimpleNamespace

from gamelab.attack.spots import Spot
from gamelab.behavior import hunt
from gamelab.behavior.fight import FightState
from gamelab.behavior.hunt import Hunt, Objective, Visit
from gamelab.behavior.ledger import Pile


def _hunt(piles, kills=(), travel=5.0):
    """A Hunt with only what the pile rules read: the drop store, the looter, route times, the visit (and no
    monster she has hurt)."""
    h = object.__new__(Hunt)
    h.h = SimpleNamespace(drops=SimpleNamespace(known=list(piles)), looter=SimpleNamespace(inventory_full=False),
                          wounds=SimpleNamespace(seen_within=lambda now, seconds, hurt_by=None: []))
    h.fs = FightState()
    h.nav = SimpleNamespace(planner=SimpleNamespace(travel=lambda src, x, dst: travel,
                                                    arrive=lambda src, x, dst, dst_x=None: travel))
    h.visit = Visit(1, Spot(1, 0.0, 1.0, True), kills=list(kills))
    h._exit_piles, h.obj = None, None
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


def test_she_finishes_what_she_hurt_before_she_sets_off_for_another_platform(monkeypatch):
    from gamelab.behavior import fight
    hurt = SimpleNamespace(kind="pig", x=-200.0, platform=4, bar=0.4)
    monkeypatch.setattr(fight, "owed", lambda h, s, st=None, hurt_by=None: [hurt])
    monkeypatch.setattr(fight, "finish_act", lambda h, s, wounds, around: ("throw", "left") if wounds else None)
    monkeypatch.setattr(fight, "finish_targets", lambda s, wounds, around: list(wounds))
    h = _hunt([])
    s = _scene()
    h.obj = Objective("spot", 7, 0.0)            # going to another platform: she finishes it first
    assert h._owed(s, []) == [hurt]
    h.obj = Objective("pile", 1, 50.0)           # along her own platform she leaves nothing behind
    assert h._owed(s, []) is None
    h.obj = Objective("spot", 7, 0.0)
    h.obj.finishing = hunt.FINISH_S              # done finishing for this one: off she goes
    assert h._owed(s, []) is None
    h.obj = None                                 # at the ground, shooting goes on as before
    assert h._owed(s, []) is None


def test_in_passing_she_steps_out_from_under_a_target_off_her_level(monkeypatch):
    # From 22 a pig on 21 near its right end is under the star's search: a step along 22 brings it in, and that's
    # a reason to stop a trip (and shoot in passing), as one in reach is; a long walk isn't.
    acts = {"act": ("back", "right", 80.0)}
    monkeypatch.setattr(hunt, "stance", lambda *a, **k: acts["act"])
    h = _hunt([])
    h.model, h._tops, h._halves = None, {}, {}
    h.h.actions = SimpleNamespace(facing=None)
    h.h.low_hp = 0.3
    h.kinds = None
    h._hits = lambda s, m: False
    h._close_quarters = lambda s, monsters: False
    h._standable = lambda pid: False
    h.obj = Objective("spot", 19, 36.0)
    pig = SimpleNamespace(kind="pig", x=-86.0, y=-85.0, platform=21, bar=None)
    s = SimpleNamespace(hp=1.0, platform=SimpleNamespace(id=22), where=SimpleNamespace(x=-10.0, y=-25.0), monsters=[pig])
    assert h._can_pass_shoot(s, [pig]) and h._interrupt(s)
    acts["act"] = ("back", "right", hunt.PASS_STEP + 1)
    assert not h._can_pass_shoot(s, [pig]) and not h._interrupt(s)
    acts["act"] = ("approach", "left", 60.0)  # closing in on one isn't for passing through
    assert not h._can_pass_shoot(s, [pig])


def test_in_passing_she_throws_at_her_targets_only_and_never_at_a_kill_fading_out():
    # The wanted monsters in view that aren't targets (out of reach) are only stepped out from under; and one seen
    # fading out where this look's kill died (in `visible`, which was taken before the kill was counted) not even
    # that: thrown at, it took a star, then its "hidden" ghost more.
    h = _hunt([])
    h.kinds, h.obj = None, Objective("spot", 19, 36.0)
    dead, above, pig = (SimpleNamespace(kind=kind, x=x, y=-85.0, platform=21, bar=None)
                        for kind, x in (("slime", -60.0), ("orange-mushroom", -86.0), ("pig", -300.0)))
    h.fs.dying = [(dead, time.monotonic())]
    got = {}
    h._can_pass_shoot = lambda s, targets, others=(): got.update(can=(list(targets), list(others))) or True
    h._passing_shot = lambda s, targets, others, visible, hidden, now: got.update(shot=(list(targets), list(others)))
    s = SimpleNamespace(hp=1.0, platform=SimpleNamespace(id=22), where=SimpleNamespace(x=-10.0, y=-25.0),
                        monsters=[dead, above, pig])
    h._while_travelling(s, [pig], [dead, above, pig], [], time.monotonic())
    assert got["can"] == got["shot"] == ([pig], [above])


def test_a_valuable_drop_is_gone_for_at_once_and_only_so_often():
    bubble = _pile(5, items=("slime-bubble", "mesos"), worth=65.0, dearest=50.0)
    coins = _pile(5, items=("mesos", "mesos", "leather"), worth=40.0, dearest=15.0)  # worth more than 30, but cheap drops
    h = _hunt([coins, bubble])
    assert h._valuable_pile(_scene()) is bubble
    assert _hunt([coins])._valuable_pile(_scene()) is None
    assert _hunt([bubble], travel=hunt.VALUABLE_TRAVEL + 1)._valuable_pile(_scene()) is None
    h.obj = Objective("pile", 2, 100.0, pile=bubble, why="valuable pile")  # going for it already
    assert h._valuable_pile(_scene()) is None
    h.obj = None
    bubble.rushed = hunt.VALUABLE_TRIES  # gone for at once that often (a monster by it): a pile like the others
    assert h._valuable_pile(_scene()) is None


def _detour_hunt(piles, times):
    """A Hunt whose route times are `times`: {(from platform, to platform): s} (Planner.arrive's)."""
    h = _hunt(piles)
    h.nav.planner.arrive = lambda src, x, dst, dst_x=None: times.get((src, dst))
    h._standable = lambda pid: True
    return h


def _at(pid, x=0.0, monsters=()):
    return SimpleNamespace(platform=SimpleNamespace(id=pid), where=SimpleNamespace(x=x), monsters=list(monsters))


def test_a_pile_a_little_off_her_way_is_picked_up_on_the_way():
    # From 13 to 17's spot on 16, 5.7 s; by coins on 19 (2.0 s there, 0.6 on them, 4.4 on to 16): 1.3 s more.
    near, far = _pile(20, platform=19), _pile(20, platform=12)
    times = {(13, 16): 5.7, (13, 19): 2.0, (19, 16): 4.4, (13, 12): 3.0, (12, 16): 5.0}  # 12: 2.9 s more
    h = _detour_hunt([far, near], times)
    h.obj = Objective("spot", 16, 636.0)
    assert h._detour(_at(13)) == (2.0 + hunt.PICK_S + 4.4 - 5.7, near)
    assert h._detour(_at(13, monsters=[SimpleNamespace(platform=19, x=120.0)])) is None  # guarded
    # Coins on 12, 3 s away, 1.1 s more: past DETOUR_REACH, they'd make the trip their own.
    times.update({(12, 16): 3.2, (13, 19): 3.0})
    assert hunt.DETOUR_REACH < 3.0 and h._detour(_at(13)) is None
    times.update({(13, 19): 2.0})
    h.obj.detours = hunt.DETOURS                    # bounded: so many on one trip, then on her way
    assert h._detour(_at(13)) is None
    for why in ("on the way", "valuable pile", "urgent pile"):  # not from a pile on the way, nor to one that can't wait
        h.obj = Objective("pile", 16, 600.0, why=why)
        assert h._detour(_at(13)) is None


def test_the_wait_for_drops_to_land_counts_in_what_a_pile_on_the_way_costs():
    times = {(13, 16): 5.7, (13, 19): 0.5, (19, 16): 5.4}  # she's right by it: 0.5 + 0.6 + 5.4 - 5.7 = 0.8 s
    h = _detour_hunt([_pile(1.0, platform=19, status="expected")], times)  # lands 1.0 s from now: 0.5 s waited
    h.obj = Objective("spot", 16, 636.0)
    extra, _ = h._detour(_at(13))
    assert abs(extra - (0.8 + 0.5)) < 0.05
    h.h.drops.known[0].since = time.monotonic()  # killed just now: 1.5 s to wait, 2.3 s in all
    assert h._detour(_at(13)) is None
