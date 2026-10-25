"""The hunt loop's pile rules (behavior.hunt), on stand-ins: no map, no simulator (but where a hunter and its
Hunt take their map's worth, on Tree Dungeon's from the game data). The loop itself runs end to end in the
simulator's loop_* scenarios."""

import random
import time
from types import SimpleNamespace

import pytest

from gamedata.paths import COPY
from gamelab.attack.spots import Spot
from gamelab.behavior import hunt
from gamelab.behavior.fight import FightState
from gamelab.behavior.hunt import Hunt, Objective, Visit
from gamelab.behavior.ledger import Pile
from gamelab.localization import mapdata

from test_sim import KEYS, tree


class Remembers:
    """Hunter.memory's stand-in: the monsters it has (`seen`), as `there` gives them."""

    def __init__(self, seen=()):
        self.seen = list(seen)

    def there(self, t, platform=None, kinds=None, min_p=0.5):
        return [m for m in self.seen if (platform is None or m.platform == platform)
                and (kinds is None or m.kind in kinds)]


def _hunt(piles, kills=(), travel=5.0):
    """A Hunt with only what the pile rules read: the drop store, the looter, route times, the visit (and no
    monster she has hurt, none remembered)."""
    h = object.__new__(Hunt)
    h.h = SimpleNamespace(drops=SimpleNamespace(known=list(piles)), looter=SimpleNamespace(inventory_full=False),
                          wounds=SimpleNamespace(seen_within=lambda now, seconds, hurt_by=None: []),
                          danger=None, memory=Remembers())  # (the map's danger: none here)
    h.fs = FightState()
    h.nav = SimpleNamespace(planner=SimpleNamespace(travel=lambda src, x, dst: travel,
                                                    arrive=lambda src, x, dst, dst_x=None, keep_off=(): travel))
    h.visit = Visit(1, Spot(1, 0.0, 1.0, True), kills=list(kills))
    h._exit_piles, h.obj = None, None
    h._seen = None  # no look yet (so no other player by any pile)
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


def test_grounds_about_as_good_go_to_the_one_she_was_at_longest_ago(monkeypatch):
    """_next_ground: the best EXP a second, travel included; with GROUND_TIE, of the grounds within it of the best, the
    one she left longest ago, a ground she never got to first. 1, 4 and 2 as from the basin by the spawn data (16.8,
    16.5 and 16.2 EXP a visit, 20 s away; no random 0.75-1.25 here)."""
    from gamelab.behavior.hunt import HuntReport
    values = {1: 16.8, 4: 16.5, 2: 16.2}
    h = object.__new__(Hunt)
    h.tour = SimpleNamespace(ground_values=lambda: values, hunting_grounds=lambda visit_s=None: list(values), stats={})
    h.nav = SimpleNamespace(planner=SimpleNamespace(travel=lambda src, x, dst: 20.0))
    h.model, h.kinds, h.rng, h.log = None, None, SimpleNamespace(uniform=lambda a, b: 1.0), lambda text: None
    h._skip, h._left, h.visit, h.stack, h.obj, h.report = {}, {}, None, [], None, HuntReport()
    h.h = SimpleNamespace(danger=None)  # (the map's danger: none here)
    monkeypatch.setattr(hunt.spots, "best_spot", lambda model, g, kinds, danger=None: Spot(g, 0.0, 1.0, True))
    monkeypatch.setattr(hunt.spots, "spot", lambda model, g, where, travel, kinds, danger=None: Spot(g, 0.0, 1.0, True))
    s = SimpleNamespace(where=SimpleNamespace(platform=SimpleNamespace(id=36), x=300.0), players=[])  # nobody else

    def nxt(left: dict) -> int:
        h._left, h.visit = dict(left), None
        h._next_ground(s)
        return h.visit.ground
    assert nxt({}) == 1 and nxt({1: 50.0}) == 1  # GROUND_TIE 0: the best
    monkeypatch.setattr(hunt, "GROUND_TIE", 0.05)
    assert nxt({}) == 1                              # none visited: the best of them
    assert nxt({1: 50.0}) == 4 and nxt({1: 50.0, 4: 60.0}) == 2 and nxt({1: 70.0, 4: 60.0, 2: 80.0}) == 4
    monkeypatch.setattr(hunt, "GROUND_TIE", 0.02)    # 2 is 3.6% under 1: not as good
    assert nxt({1: 50.0, 4: 60.0}) == 1
    from gamelab import __main__ as cli  # `tour --loop --ground-tie F` sets it for a run (the in-game check)
    assert cli.build_parser().parse_args(["tour", "--loop", "--ground-tie", "0.1"]).ground_tie == 0.1
    assert cli.build_parser().parse_args(["tour", "--loop"]).ground_tie is None


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
    h.kinds, h._skip, h._idle_since = None, {4: time.monotonic() + 60}, None  # another ground to go back to
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=17, left=861.0, right=1389.0))
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: [])

    def look(pid, strips):
        return SimpleNamespace(platform=SimpleNamespace(id=pid), looked=strips, monsters=[],
                               where=SimpleNamespace(x=0.0, y=0.0), t=time.monotonic())
    assert h._leaving(look(36, [(36, 129, 951)])) is None  # down on 36, 17 out of view: no verdict
    assert h._leaving(look(20, [(17, 855, 886)])) is None and h.visit.empty_since is None  # a sliver of it
    on_spot = look(16, [(16, 501, 669), (17, 861, 1300)])  # 83% of 17
    assert h._leaving(on_spot) is None and h.visit.empty_since is not None  # empty: to be confirmed
    assert h._leaving(look(16, [(16, 501, 669)])) is None and h.visit.empty_since is not None  # band look: kept
    h.visit.empty_since -= hunt.EMPTY_CONFIRM_S
    assert h._leaving(on_spot) == "nothing left here, and nothing due back" and 17 in h._skip


def test_she_doesnt_leave_a_ground_for_itself_she_waits_there():
    """With the ground she's at the only one she'd go to (23 alone, live on 2026-10-24), neither an empty ground nor
    her patience running out is a reason to leave: she stays (waiting), says so once, and it's progress as a new ground
    was. With another ground to go to, or one skipped to come back to, she leaves as before."""
    lines = []
    h = _hunt([])
    h.log = lines.append
    h.visit = Visit(23, Spot(24, 231.0, 0.95, False), arrived=time.monotonic() - 30)
    h.kinds, h._skip, h._idle_since, h._stay_said, h._progress = None, {}, None, None, 0.0
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=23, left=-197.0, right=186.0))
    grounds = [23]
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: list(grounds))
    empty = SimpleNamespace(platform=SimpleNamespace(id=24), looked=[(23, -197, 186)], monsters=[], players=[],
                            where=SimpleNamespace(x=231.0, y=-430.0), t=time.monotonic())
    assert h._leaving(empty) is None and h.visit.empty_since is not None  # empty: to be confirmed
    h.visit.empty_since -= hunt.EMPTY_CONFIRM_S
    assert h._leaving(empty) is None and 23 not in h._skip and h._progress > 0  # stays, not skipped
    assert h._leaving(empty) is None and len(lines) == 1 and "staying at platform 23" in lines[0]  # said once
    # Out of view of the ground (not covered), her patience run out: the same.
    away = SimpleNamespace(platform=SimpleNamespace(id=24), looked=[], monsters=[], players=[],
                           where=SimpleNamespace(x=231.0, y=-430.0), t=time.monotonic())
    h._idle_since = time.monotonic() - 10
    assert h._leaving(away) is None
    h.visit.throws += 1  # a throw since: said again
    assert h._leaving(away) is None and len(lines) == 2
    h._skip = {20: time.monotonic() + 60}  # another ground, skipped for now: the soonest back, so she goes
    assert h._leaving(away) == "nothing came into reach"
    h._skip, grounds[:] = {}, [23, 20]  # another ground to go to
    assert h._leaving(away) == "nothing came into reach"
    assert h._leaving(empty) == "nothing left here, and nothing due back" and 23 in h._skip


def test_a_monster_her_memory_has_on_the_ground_keeps_it_from_being_judged_empty():
    # The look covering 17 sees nothing on it, but her memory has a pig there (hidden under damage numbers a moment,
    # or seen by the last slice of the sweep): not empty. Once the memory lets it go, the clock starts.
    h = _hunt([])
    h.visit = Visit(17, Spot(16, 636.0, 0.35, True), arrived=time.monotonic() - 30)
    h.kinds, h._skip, h._idle_since = None, {}, None
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=17, left=861.0, right=1389.0))
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: [])
    on_spot = SimpleNamespace(platform=SimpleNamespace(id=16), looked=[(16, 501, 669), (17, 861, 1300)], monsters=[],
                              where=SimpleNamespace(x=0.0, y=0.0), t=time.monotonic())
    h.h.memory.seen = [SimpleNamespace(platform=17, x=1000.0, kind="pig")]
    assert h._leaving(on_spot) is None and h.visit.empty_since is None
    h.h.memory.seen = []
    assert h._leaving(on_spot) is None and h.visit.empty_since is not None


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


def _shooter(piles, kill_s=4.0, exp=40.0, travel=2.0, players=()):
    """_hunt with what EXP before loot reads too: a wanted monster takes `kill_s` to kill and gives `exp`, a star from
    where she stands hits any monster off her platform, her log kept (`said`)."""
    h = _hunt(piles, travel=travel)
    h.kinds, h.tour = None, SimpleNamespace(exp={"horny-mushroom": exp})
    h.h.attacks = SimpleNamespace(kill_s=lambda m: kill_s)
    h._hits = lambda s, m: True
    h._loot_said, h.said = {}, []
    h.log = h.said.append
    h._seen = SimpleNamespace(players=list(players))
    return h


def _horny(platform=3):
    return SimpleNamespace(kind="horny-mushroom", platform=platform, x=200.0, y=-60.0)


def test_exp_before_loot_a_pile_that_will_still_be_there_waits_for_what_is_in_reach():
    """The user, 2026-10-24: "The hunt should be driven by maximizing the exp gain". A valuable pile 10 s old (its life
    DROP_LIFE, 90 s), nobody by it: with a Horny Mushroom in reach she shoots first; with nothing in reach it goes."""
    ore = _pile(10, items=("sapphire-ore",), worth=120.0, dearest=120.0)
    h = _shooter([ore])
    s = _scene()
    assert h._at_risk(s, ore, 4.0) == 0.0
    assert not h._loot_first(s, ore, [_horny()])
    assert ("sapphire-ore on platform 2 at x 100 (worth 120, at risk 0%) against 10.0 EXP a second in reach for 4 s, "
            "6.6 s there and back: shooting first") in h.said[-1]  # (3 s to it: the planner's 2, and 1 to get off)
    assert h._loot_first(s, ore, [])  # nothing in reach: at once
    h.kinds = {"horny-mushroom"}  # a monster not worth hunting in reach doesn't hold it up
    assert h._loot_first(s, ore, [SimpleNamespace(kind="slime", platform=3, x=0.0, y=0.0)])


def test_exp_before_loot_a_pile_at_risk_goes_first_when_it_is_worth_more_than_the_trip_s_exp():
    # 10 s of its life left: gone if she shoots 11 s first, then goes (3 s there and PICK_S): no URGENT_MARGIN to spare
    ore = _pile(hunt.DROP_LIFE - 10, items=("sapphire-ore",), worth=120.0, dearest=120.0)
    s = _scene()
    slow = _shooter([ore], kill_s=11.0)  # the regular attack at a Horny Mushroom: 3.6 EXP a second
    assert slow._at_risk(s, ore, 11.0) == 1.0
    assert slow._at_risk(s, ore, 0.0) == pytest.approx((3.6 + 30 - 10) / 30, abs=0.01)  # going now: some to spare
    assert slow._at_risk(s, _pile(hunt.DROP_LIFE - 3), 0.0) == 0.0  # gone before she's there anyway
    assert slow._loot_first(s, ore, [_horny()])  # 120 x 0.5 = 60 against 3.6 x 6.6 = 24
    fast = _shooter([ore], kill_s=3.3)  # Lucky Seven's: 12.1 EXP a second, 80 over the trip
    assert not fast._loot_first(s, ore, [_horny()])


def test_exp_before_loot_another_player_by_a_pile_puts_it_at_risk_once_it_is_not_hers_alone():
    rival = SimpleNamespace(rival=True, near=lambda x, y, dx, dy: True, x=120.0)
    s = _scene()
    young = _pile(2)
    h = _shooter([young], players=[rival])
    assert h._at_risk(s, young, 4.0) == 0.0  # hers alone until OWN_S: 2 + 4 + 3 + 0.6 s in
    assert h._at_risk(s, young, 10.0) == hunt.RIVAL_TAKES
    theirs = _pile(40, hers=False)
    assert h._at_risk(s, theirs, 0.0) == hunt.RIVAL_TAKES
    assert _shooter([theirs])._at_risk(s, theirs, 0.0) == 0.0  # nobody by it


def test_a_trip_for_a_pile_that_went_first_is_called_off_for_what_goes_first_now():
    ore = _pile(10, items=("sapphire-ore",), worth=120.0, dearest=120.0, rushed=1)
    h = _shooter([ore])
    spot = Objective("spot", 1, 0.0, ground=1)
    h.stack, h.obj = [spot], Objective("pile", 2, 100.0, pile=ore, why="valuable pile")
    s = SimpleNamespace(hp=1.0, platform=SimpleNamespace(id=1), where=SimpleNamespace(x=0.0, y=0.0), monsters=[],
                        players=[])
    h._travel = lambda rule: rule
    assert h._while_travelling(s, [], [], [], time.monotonic()) == "valuable pile"  # nothing in reach: on her way
    assert h._while_travelling(s, [_horny()], [], [], time.monotonic()) is None
    assert h.obj is spot and ore.rushed == 0 and ore.tries == 0 and ore in h.h.drops.known
    assert "sapphire-ore on platform 2 at x 100: later, shooting what's in reach first" in h.said[-1]


def test_picking_up_yields_to_what_is_in_reach_while_she_hunts():
    from gamelab.behavior.looting import DropSpot, Looter
    said = []
    s = SimpleNamespace(where=SimpleNamespace(x=0.0), platform=SimpleNamespace(id=1), monsters=[], loot=[])
    hunter = SimpleNamespace(look=lambda **kw: s, log=said.append)
    looter = Looter(hunter)
    looter.yield_to = lambda look: True
    drops = DropSpot(10.0, 0.0, 1, time.monotonic(), True)
    assert looter.collect(drops) is False and looter.pending == [drops]
    assert said == ["  something in reach first: the drops later"]


def test_on_another_map_a_drop_is_valuable_at_twice_its_bronze_coin():
    # Tree Dungeon's monsters drop 25-53 mesos (looting.coins_for: a bronze coin there 33.4): a Slime Bubble (50) is
    # under two of them, a Topaz Ore (80) not; on the first map a bubble is 3 of its coins (15), valuable.
    from gamelab.behavior.looting import Worth
    bubble = _pile(5, items=("slime-bubble", "mesos"), worth=83.4, dearest=50.0)
    topaz = _pile(5, items=("topaz-ore",), worth=80.0, dearest=80.0)
    h = _hunt([bubble])
    assert h.valuable == hunt.VALUABLE and h._valuable_pile(_scene()) is bubble
    h.valuable = Worth({"bronze": 33.4, "gold": 51.5}).valuable
    assert h._valuable_pile(_scene()) is None
    h.h.drops.known.append(topaz)
    assert h._valuable_pile(_scene()) is topaz


@pytest.mark.skipif(not (mapdata.CACHE / "10002073.json").exists() and not (COPY / "Map").is_dir(),
                    reason="needs Tree Dungeon's map data")
def test_a_hunter_on_tree_dungeon_and_its_hunt_go_by_the_maps_worth():
    # The hunter takes its map's worth (looting.worth_for), and its Hunt rule 6a's line from it: 66.8 on Tree Dungeon,
    # twice its bronze coin's 33.4 (on the first map 30). Its monsters' drops are all known: nothing said of them.
    from gamelab.behavior.hunter import Hunter
    model, said = mapdata.load(10002073), []
    h = Hunter(SimpleNamespace(), KEYS, model, log=said.append)
    assert h.worth.coins["bronze"] == pytest.approx(33.4, abs=0.05) and h.drops.value is h.worth
    hunting = Hunt(SimpleNamespace(h=h, nav=SimpleNamespace(model=model), rng=random.Random(1), worth=None),
                   log=said.append)
    assert hunting.valuable == h.worth.valuable == pytest.approx(66.8, abs=0.1) and said == []


def test_a_hunter_on_a_map_whose_monsters_drops_are_not_known_says_so_once():
    # perception.drops.MONSTERS doesn't know a Snail off the first map (whose drops are pinned): its drops (but potions
    # and mesos) aren't looked for there.
    from gamelab.behavior.hunter import Hunter
    said = []
    Hunter(SimpleNamespace(), KEYS, mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2}), log=said.append)
    assert len(said) == 1 and said[0].startswith("drops not known for Snail ")


def _detour_hunt(piles, times):
    """A Hunt whose route times are `times`: {(from platform, to platform): s} (Planner.arrive's), or keeping off
    platforms, {(from, to, keep_off): s}."""
    h = _hunt(piles)
    h.nav.planner.arrive = lambda src, x, dst, dst_x=None, keep_off=(): times.get((src, dst, tuple(keep_off)),
                                                                                  times.get((src, dst)))
    h._standable = lambda pid: True
    return h


def _at(pid, x=0.0, monsters=()):
    return SimpleNamespace(platform=SimpleNamespace(id=pid), where=SimpleNamespace(x=x), monsters=list(monsters),
                           t=time.monotonic())


def test_a_pile_a_little_off_her_way_is_picked_up_on_the_way():
    # From 13 to 17's spot on 16, 5.7 s; by coins on 19 (2.0 s there, 0.6 on them, 4.4 on to 16): 1.3 s more.
    near, far = _pile(20, platform=19), _pile(20, platform=12)
    times = {(13, 16): 5.7, (13, 19): 2.0, (19, 16): 4.4, (13, 12): 3.0, (12, 16): 5.0}  # 12: 2.9 s more
    h = _detour_hunt([far, near], times)
    h.obj = Objective("spot", 16, 636.0)
    assert h._detour(_at(13)) == (2.0 + hunt.PICK_S + 4.4 - 5.7, near)
    h.h.memory.seen = [SimpleNamespace(platform=19, x=120.0, kind="pig")]  # remembered by it (off her band):
    assert h._detour(_at(13)) is None                                      # guarded
    h.h.memory.seen = []
    # Coins on 12, 3 s away, 1.1 s more: past DETOUR_REACH, they'd make the trip their own.
    times.update({(12, 16): 3.2, (13, 19): 3.0})
    assert hunt.DETOUR_REACH < 3.0 and h._detour(_at(13)) is None
    times.update({(13, 19): 2.0})
    h.obj.detours = hunt.DETOURS                    # bounded: so many on one trip, then on her way
    assert h._detour(_at(13)) is None
    for why in ("on the way", "valuable pile", "urgent pile"):  # not from a pile on the way, nor to one that can't wait
        h.obj = Objective("pile", 16, 600.0, why=why)
        assert h._detour(_at(13)) is None


def test_a_pile_on_the_way_is_weighed_by_the_route_the_trip_takes():
    # A trip to a spot off its ground keeps off the ground (hunt._keep_off, route.KEEP_OFF_S), and her ways to the
    # spot, from here and from the pile, are weighed so; her way to the pile is the stop's own trip's, which keeps off
    # nothing. From 36 at x 0 to 17's spot on 16, a pile on 36 0.5 s away: by the plain routes (up the right side,
    # past 17) 15.0 s from here and 15.2 from the pile, 1.3 s more; by the trip's (up the left), 18.0 and 21.3: 4.4 s.
    pile = _pile(20, platform=36)
    h = _detour_hunt([pile], {})
    asked = []

    def arrive(src, x, dst, dst_x=None, keep_off=()):
        asked.append((dst, tuple(keep_off)))
        if dst == 36:
            return 0.5
        return (18.0 if x == 0.0 else 21.3) if tuple(keep_off) == (17,) else (15.0 if x == 0.0 else 15.2)
    h.nav.planner.arrive = arrive
    h.obj = Objective("spot", 16, 636.0, ground=17)
    assert h._detour(_at(36)) is None
    assert sorted(set(asked)) == [(16, (17,)), (36, ())]
    h.obj = Objective("spot", 16, 636.0, ground=16)  # a spot on its ground: nothing to keep off
    assert h._detour(_at(36)) == (0.5 + hunt.PICK_S + 15.2 - 15.0, pile)


def test_the_wait_for_drops_to_land_counts_in_what_a_pile_on_the_way_costs():
    times = {(13, 16): 5.7, (13, 19): 0.5, (19, 16): 5.4}  # she's right by it: 0.5 + 0.6 + 5.4 - 5.7 = 0.8 s
    h = _detour_hunt([_pile(1.0, platform=19, status="expected")], times)  # lands 1.0 s from now: 0.5 s waited
    h.obj = Objective("spot", 16, 636.0)
    extra, _ = h._detour(_at(13))
    assert abs(extra - (0.8 + 0.5)) < 0.05
    h.h.drops.known[0].since = time.monotonic()  # killed just now: 1.5 s to wait, 2.3 s in all
    assert h._detour(_at(13)) is None


def test_a_pile_by_another_player_is_left_to_them_unless_it_is_valuable_or_hers(monkeypatch):
    """PLAYER_PILES (off by default): a pile no kill of hers explains, with another player by it in the latest look
    (400 units across, 100 up or down), isn't worth a trip, however old, unless a drop in it is worth VALUABLE; her own
    are."""
    from gamelab.localization.players import Player
    monkeypatch.setattr(hunt, "PLAYER_PILES", True)
    theirs = _pile(hunt.OTHERS_AFTER + 1, hers=False)
    dear = _pile(hunt.OTHERS_AFTER + 1, hers=False, dearest=hunt.VALUABLE, worth=hunt.VALUABLE)
    mine = _pile(10)
    h = _hunt([theirs, dear, mine])
    assert h._fresh(theirs) and h._fresh(dear) and h._fresh(mine)
    for kind, left in (("stranger", True), ("party", False)):  # her party doesn't hunt against her
        h._seen = SimpleNamespace(players=[Player(1, kind, 300.0, 0.0, 2, (0.0, 0.0), 0.0, 1.0, seen=2)])
        assert h._fresh(theirs) != left and h._fresh(dear) and h._fresh(mine)
    h._seen = SimpleNamespace(players=[Player(1, "stranger", 600.0, 0.0, 2, (0.0, 0.0), 0.0, 1.0, seen=2)])
    assert h._fresh(theirs)  # 500 units off it
