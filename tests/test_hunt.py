"""The hunt loop's rules (behavior.hunt), its loot sweep's among them, on stand-ins: no map, no simulator (but where a
hunter and its Hunt take their map's worth, on Tree Dungeon's from the game data). The loop itself runs end to end in
the simulator's loop_* scenarios."""

import random
import time
from types import SimpleNamespace

import pytest

from gamedata.paths import COPY
from gamelab.attack.spots import Spot
from gamelab.behavior import hunt, sweep
from gamelab.behavior.fight import FightState
from gamelab.behavior.hunt import Hunt, Objective, Visit
from gamelab.behavior.ledger import DROP_LANDS, EXPIRE_PASS_S, EXPIRE_S, Pile
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
    h._exit_reason, h.obj, h.stack = None, None, []
    h._seen = None  # no look yet (so no other player by any pile)
    return h


def _scene():
    return SimpleNamespace(platform=SimpleNamespace(id=1), where=SimpleNamespace(x=0.0))


def _pile(age, platform=2, hers=True, **kw):
    return Pile(100.0, 0.0, platform, time.monotonic() - age, hers, **kw)










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


def test_the_hunt_loop_grinds_by_default_and_camps_with_camp(monkeypatch):
    # Since 2026-10-27 `tour --loop` grinds (behavior.grind); `--camp` hunts one ground at a time (Hunt); `--grind` is
    # still taken, and not with `--camp`.
    from gamelab import __main__ as cli
    p = cli.build_parser()
    assert p.parse_args(["tour", "--loop"]).grind and p.parse_args(["tour", "--loop", "--grind"]).grind
    assert not p.parse_args(["tour", "--loop", "--camp"]).grind
    with pytest.raises(SystemExit):
        p.parse_args(["tour", "--loop", "--grind", "--camp"])
    from contextlib import nullcontext

    from gamelab.behavior import grind, retreat, tour as tour_mod
    ran = []

    class Ran:
        def __init__(self, tour, log):
            self.name = type(self).__name__

        def run(self, minutes):
            ran.append(self.name)
            return "ok"
    monkeypatch.setattr(grind, "Grind", type("Grind", (Ran,), {}))
    monkeypatch.setattr(hunt, "Hunt", type("Hunt", (Ran,), {}))
    monkeypatch.setattr(retreat, "on_alarm", lambda hunter: nullcontext())
    monkeypatch.setattr(tour_mod, "Tour", lambda hunter, nav, log, rng: SimpleNamespace())
    monkeypatch.setattr(cli, "_record", lambda *a: None)
    monkeypatch.setattr(cli, "_to_game", lambda: True)
    monkeypatch.setattr(cli, "_hunter", lambda args, **kw: SimpleNamespace(navigator=lambda **kw: None))
    for argv in (["tour", "--loop"], ["tour", "--loop", "--camp"]):
        assert cli.cmd_tour(p.parse_args(argv)) == 0
    assert ran == ["Grind", "Hunt"]


def test_a_pile_objective_no_longer_worth_it_ends_the_trip_to_it():
    pile = _pile(10, status="gone")  # seen gone on the way
    h = _hunt([pile])
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


def test_the_ground_is_judged_empty_only_from_looks_that_cover_it_for_a_while():
    h = _hunt([])
    h.visit = Visit(17, Spot(16, 636.0, 0.35, True), arrived=time.monotonic() - 30)
    h.kinds, h._skip, h._idle_since = None, {4: time.monotonic() + 60}, None  # another ground to go back to
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=17, left=861.0, right=1389.0, y_at=lambda x: -335.0))
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: [])

    def look(pid, span):  # (span: the map x its search for monsters took in at 17's level, Scene.searched_x)
        return SimpleNamespace(platform=SimpleNamespace(id=pid), searched_x=lambda y: span, monsters=[],
                               where=SimpleNamespace(x=0.0, y=0.0), t=time.monotonic())
    assert h._leaving(look(36, None)) is None  # down on 36, 17's level not searched: no verdict
    assert h._leaving(look(20, (300, 886))) is None and h.visit.empty_since is None  # a sliver of it
    on_spot = look(16, (501, 1300))  # 83% of 17
    assert h._leaving(on_spot) is None and h.visit.empty_since is not None  # empty: to be confirmed
    assert h._leaving(look(16, None)) is None and h.visit.empty_since is not None  # band look: kept
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
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=23, left=-197.0, right=186.0, y_at=lambda x: -448.0))
    grounds = [23]
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: list(grounds))
    empty = SimpleNamespace(platform=SimpleNamespace(id=24), searched_x=lambda y: (-658.0, 708.0), monsters=[],
                            players=[], where=SimpleNamespace(x=231.0, y=-430.0), t=time.monotonic())
    assert h._leaving(empty) is None and h.visit.empty_since is not None  # empty: to be confirmed
    h.visit.empty_since -= hunt.EMPTY_CONFIRM_S
    assert h._leaving(empty) is None and 23 not in h._skip and h._progress > 0  # stays, not skipped
    assert h._leaving(empty) is None and len(lines) == 1 and "staying at platform 23" in lines[0]  # said once
    # Out of view of the ground (not covered), her patience run out: the same.
    away = SimpleNamespace(platform=SimpleNamespace(id=24), searched_x=lambda y: None, monsters=[], players=[],
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
    h.model = SimpleNamespace(platform=lambda pid: SimpleNamespace(id=17, left=861.0, right=1389.0, y_at=lambda x: -335.0))
    h.tour = SimpleNamespace(stats={}, ground_values=lambda: {}, hunting_grounds=lambda visit_s=None: [])
    on_spot = SimpleNamespace(platform=SimpleNamespace(id=16), searched_x=lambda y: (501.0, 1300.0), monsters=[],
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
    h._hits = h._star_hits = lambda s, m: True
    h._loot_said, h.said = {}, []
    h.log = h.said.append
    h._seen = SimpleNamespace(players=list(players))
    return h


def _horny(platform=3):
    return SimpleNamespace(kind="horny-mushroom", platform=platform, x=200.0, y=-60.0)


def test_exp_before_loot_a_pile_that_will_still_be_there_waits_for_what_is_in_reach():
    """The user, 2026-10-24: "The hunt should be driven by maximizing the exp gain". A valuable pile 10 s old (how long it
    lies: ledger.SURVIVAL), nobody by it: with a Horny Mushroom in reach she shoots first; with nothing in reach it
    goes."""
    ore = _pile(10, items=("sapphire-ore",), worth=120.0, dearest=120.0)
    h = _shooter([ore])
    s = _scene()
    assert 0.0 < h._at_risk(s, ore, 4.0) < 0.01  # (by her drops' survival: ledger.SURVIVAL, hers alone OWN_S)
    assert not h._loot_first(s, ore, [_horny()])
    assert ("sapphire-ore on platform 2 at x 100 (worth 120, at risk 0%) against 10.0 EXP a second in reach for 4 s, "
            "6.6 s there and back: shooting first") in h.said[-1]  # (3 s to it: the planner's 2, and 1 to get off)
    assert h._loot_first(s, ore, [])  # nothing in reach: at once
    h.kinds = {"horny-mushroom"}  # a monster not worth hunting in reach doesn't hold it up
    assert h._loot_first(s, ore, [SimpleNamespace(kind="slime", platform=3, x=0.0, y=0.0)])




def test_exp_before_loot_another_player_by_a_pile_puts_it_at_risk_once_it_is_not_hers_alone():
    rival = SimpleNamespace(rival=True, near=lambda x, y, dx, dy: True, x=120.0)
    s = _scene()
    young = _pile(2)
    h = _shooter([young], players=[rival])
    assert h._at_risk(s, young, 4.0) < 0.05  # hers alone until OWN_S: 2 + 4 + 3 + 0.6 s in (its survival alone)
    assert h._at_risk(s, young, 10.0) == hunt.RIVAL_TAKES
    theirs = _pile(40, hers=False)
    assert h._at_risk(s, theirs, 0.0) == hunt.RIVAL_TAKES
    assert _shooter([theirs])._at_risk(s, theirs, 0.0) == 0.0  # nobody by it
    # A valuable pile at risk so goes first where it's worth more than what's in reach would give over the trip: 120 x
    # 0.5 x 0.5 = 30 against the regular attack's 3.6 EXP a second x 6.6 s (24); not against Lucky Seven's 12.1 (80)
    ore = _pile(20, hers=False, items=("sapphire-ore",), worth=120.0, dearest=120.0)
    assert _shooter([ore], kill_s=11.0, players=[rival])._loot_first(s, ore, [_horny()])
    assert not _shooter([ore], kill_s=3.3, players=[rival])._loot_first(s, ore, [_horny()])
    assert not _shooter([ore], kill_s=11.0)._loot_first(s, ore, [_horny()])  # nobody by it: its survival alone


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




def test_a_trip_stops_for_what_is_in_reach_only_where_the_pass_would_shoot_it():
    """In reach from a jump alone, a monster off the ground and off her platform isn't one of the pass's targets
    (fight._wanted: a star from where she stands must hit it): it stopped every trip on its first look, and the pass
    sent her on again, 2,373 times in 107 s in loop_tree_walls (found offline, 2026-10-25). On the ground, on her
    platform, or in a star's reach, it stops the trip as before."""
    h = _shooter([])
    h.model, h._tops, h._halves = None, {}, {}
    h._close_quarters = lambda s, monsters: False
    h._owed = lambda s, monsters: None
    h._standable = lambda pid: False
    h._sweep_calls = lambda s: False
    h._pass_step = lambda s, targets: None
    h._star_hits = lambda s, m: False  # a jump's attack reaches it, a star from the ground doesn't
    h.obj = Objective("spot", 19, 36.0)  # her time in passing not spent
    s = SimpleNamespace(hp=1.0, platform=SimpleNamespace(id=5), where=SimpleNamespace(x=0.0, y=0.0),
                        monsters=[_horny(3)], players=[])
    assert not h._interrupt(s)
    s.monsters = [_horny(1)]  # on the ground she hunts (the visit's, 1)
    assert h._interrupt(s)
    s.monsters = [_horny(5)]  # on her platform
    assert h._interrupt(s)
    h._star_hits = lambda s, m: True
    s.monsters = [_horny(3)]
    assert h._interrupt(s)


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


def test_a_pile_is_rushed_and_put_first_only_for_what_she_can_take():
    """The independent check of 2026-11-01: rule 6a went by the pile's dearest drop and its whole worth, and EXP before
    loot by its whole worth, a full tab's items too: with her Etc tab full she rushed a pile of a coin and an ore for
    the coin. Now by the dearest drop she can take (Hunt._dearest: looting.Looter.can_take) and her worth of it
    (Hunt._worth): on Tree Dungeon (a bronze coin 33.4, valuable from 66.8) a coin and a Silver Ore (80) with Etc full
    is neither rushed nor put ahead of a Horny Mushroom in reach; with room, both; a Magic Potion (Use) still is."""
    from gamelab.behavior.looting import Worth, tab_of
    tree_worth = Worth({"bronze": 33.4, "gold": 51.5})
    ore = _pile(20, hers=False, items=("mesos", "silver-ore"), worth=113.4, dearest=80.0, each=(33.4, 80.0), mesos=True)
    rival = SimpleNamespace(rival=True, near=lambda x, y, dx, dy: True, x=120.0)
    for full, rushed in ((set(), True), ({"Etc"}, False)):
        h = _shooter([ore], kill_s=11.0, players=[rival])
        h.valuable, h.h.drops.value = tree_worth.valuable, tree_worth
        h.h.looter = SimpleNamespace(inventory_full=bool(full), items_full=False,
                                     can_take=lambda k, full=full: k == "mesos" or tab_of(k) not in full)
        assert (h._valuable_pile(_scene()) is ore) == rushed and h._dearest(ore) == (80.0 if rushed else 33.4)
        assert h._loot_first(_scene(), ore, [_horny()]) == rushed
    potion = _pile(5, items=("mesos", "magic-potion"), worth=283.4, dearest=250.0, each=(33.4, 250.0), mesos=True)
    h.h.drops.known = [potion]
    assert h._valuable_pile(_scene()) is potion and h._dearest(potion) == 250.0


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
    assert hunting.valuable == h.worth.valuable == pytest.approx(66.8, abs=0.1)
    assert [line for line in said if not line.startswith("safety stop: ")] == []  # (her reserve, said at her start)


def test_a_hunter_on_a_map_whose_monsters_drops_are_not_known_says_so_once():
    # perception.drops.MONSTERS doesn't know a Snail off the first map (whose drops are pinned): its drops (but potions
    # and mesos) aren't looked for there.
    from gamelab.behavior.hunter import Hunter
    said = []
    Hunter(SimpleNamespace(), KEYS, mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2}), log=said.append)
    said = [line for line in said if not line.startswith("safety stop: ")]  # (her reserve, said at her start)
    assert len(said) == 1 and said[0].startswith("drops not known for Snail ")












def test_a_pile_by_another_player_is_left_to_them_unless_it_is_valuable_or_hers(monkeypatch):
    """PLAYER_PILES (off by default): a pile no kill of hers explains, with another player by it in the latest look
    (400 units across, 100 up or down), isn't worth a trip, however old, unless a drop in it is worth VALUABLE; her own
    are."""
    from gamelab.localization.players import Player
    monkeypatch.setattr(hunt, "PLAYER_PILES", True)
    theirs = _pile(hunt.OWN_S + 1, hers=False)
    dear = _pile(hunt.OWN_S + 1, hers=False, dearest=hunt.VALUABLE, worth=hunt.VALUABLE)
    mine = _pile(10)
    h = _hunt([theirs, dear, mine])
    assert h._candidate(theirs) and h._candidate(dear) and h._candidate(mine)
    for kind, left in (("stranger", True), ("party", False)):  # her party doesn't hunt against her
        h._seen = SimpleNamespace(players=[Player(1, kind, 300.0, 0.0, 2, (0.0, 0.0), 0.0, 1.0, seen=2)])
        assert h._candidate(theirs) != left and h._candidate(dear) and h._candidate(mine)
    h._seen = SimpleNamespace(players=[Player(1, "stranger", 600.0, 0.0, 2, (0.0, 0.0), 0.0, 1.0, seen=2)])
    assert h._candidate(theirs)  # 500 units off it


# --- the map's danger for her pile tests (test_danger's kinds) ------------------------------------------------------------

def _ground():
    """Tree Dungeon's 23 and 24, made up (test_danger's kinds): the ground (500..900, y 500) with a Horny Mushroom's spawn
    point at 800, and her spot's platform beside it (900..1100, y 480)."""
    from test_danger import build
    m = build([(500, 500, 900, 500), (900, 480, 1100, 480)], [("19", 800, 500)])
    ground, spot = (m.platform_under(700, 500).id, m.platform_under(1000, 480).id)
    return m, ground, spot


def _priced(model, piles, memory=(), kills=(), land=880.0, travel=1.0):
    """_shooter with the map's danger for her (test_danger's HER), her memory's monsters (`memory`, as the memory's
    entries), her kills (`kills`: ledger.Kill), a route onto any platform landing at map x `land` taking `travel` s."""
    from gamelab.attack.danger import Danger
    from test_danger import HER
    h = _shooter(piles, travel=travel)
    h.h.danger, h.model = Danger(model, HER), model
    h.h.memory = SimpleNamespace(alive=lambda t, min_p=0.0: list(memory), there=lambda *a, **k: [])
    h.h.ledger = SimpleNamespace(kills=list(kills))
    h.nav.planner.plan = lambda src, x, dst: [SimpleNamespace(land_x=land)]
    h._unseen_for, h._unseen_list, h._landings, h._block_said = None, [], (None, {}), {}
    return h


def _on(pid, x, model, monsters=()):
    return SimpleNamespace(platform=model.platform(pid), where=SimpleNamespace(x=x, y=model.platform(pid).y_at(x)),
                           monsters=list(monsters), swept=[], tracks=[], t=time.monotonic(), players=[])


def _remembered(kind, pid, x, age, model):
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    m = Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, model.platform(pid).y_at(x), pid)
    return SimpleNamespace(kind=kind, platform=pid, x=x, seen=time.monotonic() - age, monster=m)


# --- the loot sweep in the hunt loop (from 2026-10-30: behavior.sweep) ----------------------------------------------------

def test_with_her_etc_tab_full_a_pile_of_etc_items_alone_is_no_candidate_and_one_with_coins_is_worth_the_coins():
    """Her inventory by tab (looting.Looter.full, from 2026-10-31): her Etc tab full, a pile seen holding a Horny
    Mushroom's cap alone holds nothing she can take: not a candidate, and one she's going for is worthless; one with
    coins and a cap is, worth its coins (`_worth`); one not seen yet, while her Use tab has room, still is."""
    from gamelab.behavior.looting import FIRST, Looter
    h = _hunt([])
    looter = Looter(SimpleNamespace(log=lambda m: None))
    looter.full = {"Etc": (time.monotonic(), "test")}
    h.h.looter, h.h.drops.value = looter, FIRST
    cap = _pile(10, items=("horny-mushroom-cap",), worth=FIRST("horny-mushroom-cap"), status="seen")
    both = _pile(10, mesos=True, items=("mesos", "horny-mushroom-cap"), status="seen",
                 worth=FIRST("mesos", "bronze") + FIRST("horny-mushroom-cap"))
    assert not h._candidate(cap) and h._worthless(cap)
    assert h._candidate(both) and h._worth(both) == pytest.approx(FIRST("mesos", "bronze"))
    assert h._candidate(_pile(10))  # (not seen yet: it may hold a Use item, or mesos)
    looter.full["Use"] = (time.monotonic(), "test")
    assert not h._candidate(_pile(10)) and looter.items_full


def test_with_her_etc_tab_full_a_pile_not_seen_yet_is_worth_its_kinds_drops_but_its_etc_items():
    """The review of 2026-10-31: a kill's pile no look has seen yet was priced at all its kind's expected drops
    (looting.expected_worth) with her Etc tab full, its Etc items too; now at those she can take (Looter.can_take: the
    mesos, a kind she holds a stack of), and all of it again once the tab has room."""
    from gamelab.behavior.looting import Looter, Worth, expected_worth, tab_of
    tree = Worth({"bronze": 33.4, "gold": 51.5})
    h = _hunt([])
    looter = Looter(SimpleNamespace(log=lambda m: None))
    h.h.looter, h.h.drops.value = looter, tree
    pile = _pile(10, kind="horny-mushroom", unseen=expected_worth("horny-mushroom", tree))
    assert h._worth(pile) == pytest.approx(pile.value) and pile.value == pytest.approx(23.5, abs=0.1)
    looter.full = {"Etc": (time.monotonic(), "test")}
    no_etc = expected_worth("horny-mushroom", tree, lambda k: tab_of(k) != "Etc")  # (its mesos and potions)
    assert h._worth(pile) == pytest.approx(no_etc) and no_etc < pile.value - 5.0
    looter.held.add("horny-mushroom-cap")  # (a stack of caps of hers: still taken)
    assert no_etc + 5.0 < h._worth(pile) <= pile.value


def test_a_pile_is_a_candidate_by_its_survival_its_owner_and_its_tries():
    """Not a fixed life (DROP_LIFE, 90 s, until 2026-10-30): a pile of hers 100 s old, never seen since her kill, lies
    there still 93% of the time (ledger.SURVIVAL, by the live evidence since 2026-10-31: hers alone for OWN_S, others'
    0.05 a minute after) and is gone for, one 170 s old too (the raw curve until then, 3%, under SURVIVE_MIN from about
    163 s), and one 150 s old seen 10 s ago; just past EXPIRE_S too (its expiry pass may not have come yet), past
    EXPIRE_S and EXPIRE_PASS_S not. Another's once past its killer's ownership (OWN_S,
    where OTHERS_AFTER waited 30 s); one stood on or gone for PILE_TRIES times, not; nor one seen gone."""
    h = _hunt([])
    seen = _pile(150, status="seen", last_seen=time.monotonic() - 10.0)
    assert h._candidate(_pile(10)) and h._candidate(_pile(100)) and h._candidate(_pile(170)) and h._candidate(seen)
    assert h._candidate(_pile(EXPIRE_S + 1)) and not h._candidate(_pile(EXPIRE_S + EXPIRE_PASS_S + 1))
    assert not h._candidate(_pile(10, hers=False)) and h._candidate(_pile(hunt.OWN_S + 1, hers=False))
    assert not h._candidate(_pile(10, tries=hunt.PILE_TRIES)) and h._candidate(_pile(10, tries=hunt.PILE_TRIES - 1))
    assert not h._candidate(_pile(10, status="gone"))
    h.h.looter.inventory_full = True
    assert not h._candidate(_pile(10)) and h._candidate(_pile(10, mesos=True))


def test_a_pile_of_hers_by_her_is_left_to_her_pets_for_a_moment(monkeypatch):
    """PET_LOOT (the user, 2026-11-01: her pets "will pick up whatever thing when the pet walks past those loot"; off by
    default from the review of that day): a pile of hers on her platform 40 units off her is no candidate until she has
    been by it PET_WAIT s, from its drops' landing at the soonest, then one; 190 units off (over PET_NEAR), on another
    platform, or another's past its ownership, one at once; her time by it counted afresh once she has been farther;
    with PET_LOOT off, one at once."""
    assert not hunt.PET_LOOT
    monkeypatch.setattr(hunt, "PET_LOOT", True)
    h = _hunt([])
    h._s = SimpleNamespace(platform=SimpleNamespace(id=2), where=SimpleNamespace(x=60.0))  # (the piles at x 100, on 2)
    fresh, landing = _pile(DROP_LANDS), _pile(0.5)  # (its drops just landed; landing in 1.5 s)
    assert not h._candidate(fresh) and not h._candidate(landing)
    assert h._pet_by[landing.key] == pytest.approx(landing.since + DROP_LANDS)  # (counted from the landing)
    h._pet_by[fresh.key] -= hunt.PET_WAIT  # (PET_WAIT s on)
    assert h._candidate(fresh) and not h._candidate(landing)
    far = Pile(250.0, 0.0, 2, time.monotonic() - DROP_LANDS, True)
    assert h._candidate(far) and h._candidate(_pile(DROP_LANDS, platform=3))
    assert h._candidate(_pile(hunt.OWN_S + 1, hers=False))
    h._s.where.x = 300.0  # (farther than PET_NEAR from it: counted afresh when she's back by it)
    assert h._candidate(fresh) and fresh.key not in h._pet_by
    h._s.where.x = 60.0
    assert not h._candidate(fresh)
    monkeypatch.setattr(hunt, "PET_LOOT", False)
    assert h._candidate(fresh) and h._candidate(landing)


def test_a_pile_left_to_her_pets_is_judged_from_the_look_asked_about_and_counted_from_her_own_looks(monkeypatch):
    """The review of 2026-11-01: `_pets_first` read where she is from her last pass's look whatever look the candidate
    was asked about (the navigator's on a trip, through _interrupt and _sweep_calls, and the made-up ones), so a pile
    was judged by her from where her trip began. Now from the look asked about: one 40 units off where a look on her
    trip has her is hers to wait for, though her pass's had her 300 off; one made up from another (Scene.what_if)
    reads her time by it and starts or resets nothing."""
    monkeypatch.setattr(hunt, "PET_LOOT", True)
    h = _hunt([])
    h._s = SimpleNamespace(platform=SimpleNamespace(id=2), where=SimpleNamespace(x=400.0))  # (the pass's: 300 off)
    fresh = _pile(DROP_LANDS)
    on_trip = SimpleNamespace(platform=SimpleNamespace(id=2), where=SimpleNamespace(x=60.0))
    assert h._candidate(fresh) and not h._candidate(fresh, on_trip) and fresh.key in h._pet_by
    made_up = SimpleNamespace(platform=SimpleNamespace(id=2), where=SimpleNamespace(x=400.0), what_if=True)
    assert h._candidate(fresh, made_up) and fresh.key in h._pet_by  # (farther there: not reset)
    other = _pile(DROP_LANDS)
    near = SimpleNamespace(platform=SimpleNamespace(id=2), where=SimpleNamespace(x=60.0), what_if=True)
    assert not h._candidate(other, near) and other.key not in h._pet_by  # (by it there: not started)


def _line_hunt(piles):
    """_hunt with the sweep's reads: route times on a line of platforms (a walk, and 1.5 s a platform between)."""
    h = _hunt(piles)

    def arrive(src, x, dst, dst_x=None, keep_off=()):
        return abs((dst_x if dst_x is not None else x) - x) / 149.0 + 1.5 * abs(dst - src)
    h.nav.planner.arrive = arrive
    h.nav.planner.travel = lambda src, x, dst, keep_off=(): 1.5 * abs(dst - src)
    h.log, h.said = (lambda line: h.said.append(line)), []
    h.model = None
    return h


def _on_line(pid, x):
    return SimpleNamespace(platform=SimpleNamespace(id=pid), where=SimpleNamespace(x=x), monsters=[], swept=[],
                           t=time.monotonic(), players=[])


def test_the_sweep_takes_a_pile_within_what_it_may_add_to_her_way():
    """From x 0 on platform 1 to x 600 on 3, coins on 2 at x 300 (on her way: 0.24 s more for its stand): taken fresh
    and old (on her way: ALLOW_OLD), and said; coins 150 behind her on 1 (2.0 s there and back, and the stand) taken
    fresh (ALLOW_FRESH 3 s), not 70 s old (ALLOW_OLD 1 s: no late trips for old far piles), whatever her time's
    worth (the user, 2026-11-01: "piles should be picked up soon after killing")."""
    coins = _pile(5, platform=2, items=("mesos",), worth=33.4, xs=(300.0,), status="seen")
    coins.x, coins.last_seen = 300.0, time.monotonic()
    h = _line_hunt([coins])
    stop = h._sweep(_on_line(1, 0.0), (3, 600.0))
    assert stop is not None and h._pile_of[stop.spot.key] is coins and stop.spot.stand == pytest.approx(0.24)
    assert stop.spot.allow == hunt.ALLOW_FRESH and h.said[-1].startswith("  sweep: mesos on 2 at x 300: ")
    coins.since -= 70.0
    assert _line_hunt([coins])._sweep(_on_line(1, 0.0), (3, 600.0)).spot.allow == hunt.ALLOW_OLD
    behind = _pile(5, platform=1, items=("mesos",), worth=33.4, xs=(-150.0,), status="seen")
    behind.x, behind.last_seen = -150.0, time.monotonic()
    assert _line_hunt([behind])._sweep(_on_line(1, 0.0), (1, 0.0)) is not None
    behind.since -= 70.0
    assert _line_hunt([behind])._sweep(_on_line(1, 0.0), (1, 0.0)) is None


def test_the_sweep_waits_for_a_kills_drops_to_land_within_what_the_pile_may_add_to_her_way(monkeypatch):
    """By her kill of a Horny Mushroom (its pile expected: looting.expected_worth's 23.5 on Tree Dungeon's worth), its
    drops 1.0 s from landing (DROP_LANDS after the kill), about 1.7 s in all with its stand and back: she waits for them
    within a fresh pile's ALLOW_FRESH, not within 1.5 s."""
    kill = _pile(DROP_LANDS - 1.0, platform=1)
    kill.x, kill.unseen = 20.0, 23.5
    stop = _line_hunt([kill])._sweep(_on_line(1, 0.0), (1, 0.0))
    assert stop is not None and stop.spot.lands == pytest.approx(1.0, abs=0.05)
    monkeypatch.setattr(hunt, "ALLOW_FRESH", 1.5)
    assert _line_hunt([kill])._sweep(_on_line(1, 0.0), (1, 0.0)) is None


def _walker(kind, pid, x, facing, model, pose="move"):
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, model.platform(pid).y_at(x), pid, facing=facing, pose=pose)


def test_a_monster_walking_at_where_she_would_stand_is_shot_first_or_the_pile_left():
    """The user, 2026-10-30: "in general you should kill all the mobs that are in the way, try not to hit by mobs just
    for picking up loots". Drops at x 745-755 on the ground (500..900), she on her spot beside it (her trip there 1.1 s:
    the planner's 0.1 and 1 to get off): a Horny Mushroom at 890 walking left at its 100 units a second is by them as she
    stands there: shot first where a star from her spot hits it (its kill time and EXP the pile's), else the pile is
    left; one at 600 walking left, away, guards nothing; one standing at 870 may set off either way any time (its
    heading not known: sweep.UNSURE_S of walking, the review of 2026-10-30, where until then it stayed put and guarded
    nothing): a guard too; one the look didn't find (about: her memory's) leaves it, never thrown at."""
    m, ground, spot = _ground()
    pile = Pile(750.0, 500.0, ground, time.monotonic() - 3.0, True, status="seen", xs=(745.0, 755.0),
                last_seen=time.monotonic())
    s = _on(spot, 1000.0, m, [_walker("horny-mushroom", ground, 890.0, -1, m)])
    h = _priced(m, [pile], travel=0.1)
    guards, why = h._guards(s, pile, h._travel_to_pile(s, pile), h._stand_s(pile))
    assert why is None and [g.monster.x for g in guards] == [890.0] and guards[0].kill_s == 4.0
    h._hits = lambda s, m: False
    assert h._guards(s, pile, 1.1, h._stand_s(pile)) == ((), f"a horny-mushroom at x 890 on {ground} would meet her there")
    away = _walker("horny-mushroom", ground, 600.0, -1, m)
    assert _priced(m, [pile], travel=0.1)._guards(_on(spot, 1000.0, m, [away]), pile, 1.1, 0.5) == ((), None)
    stands = _walker("horny-mushroom", ground, 870.0, 0, m, "stand")
    standing = _priced(m, [pile], travel=0.1)
    assert standing._reach_of(stands)[0] is None  # (either way, as one whose heading isn't known)
    # Its middle turns where the game turns a walker, 20 along its end footholds from the ground's ends (500..900:
    # monsters.walk_edges, since master 072e1ec the stack's danger's too), where until then the ends themselves.
    from gamelab.localization.monsters import WALK_EDGE
    assert standing._reach_of(stands)[3:] == (500.0 + WALK_EDGE, 900.0 - WALK_EDGE)
    assert [g.monster.x for g in standing._guards(_on(spot, 1000.0, m, [stands]), pile, 1.1, 0.5)[0]] == [870.0]
    about = _priced(m, [pile], memory=[_remembered("horny-mushroom", ground, 765.0, 1.0, m)], travel=0.1)
    assert about._guards(_on(spot, 1000.0, m), pile, 1.1, 0.5)[1].startswith("a horny-mushroom unseen at x 765")
    first = _priced(m, [pile], travel=0.1)  # where nothing counts as a danger (the first map): nothing guards
    first.h.danger = None
    assert first._guards(s, pile, 1.1, 0.5) == ((), None)


def test_a_monster_on_the_platform_just_under_a_pile_guards_it_as_one_on_the_pile_s_own(monkeypatch):
    """Her full box against theirs (the review of 2026-10-31: BESIDE_GUARDS): a Horny Mushroom walking left at 880 on a
    platform 30 under the ground (its top, 51, reaches into her standing there) is by the drops at 745-755 as she
    stands there, as one on the ground would be: shot first, or the pile left. One on a platform 70 under it walks
    under her feet: nothing. Until then the pile's own platform's alone guarded it. And rule 1, with her fights'
    BESIDE (on by default since 2026-10-31), counts the one under her, too close (inside its keep-off), as one on her
    platform."""
    from test_danger import build
    m = build([(500, 500, 900, 500), (900, 480, 1100, 480), (400, 530, 900, 530), (400, 570, 900, 570)],
              [("19", 800, 500)])
    ground, spot = m.platform_under(700, 500).id, m.platform_under(1000, 480).id
    low, deep = m.platform_under(600, 531).id, m.platform_under(600, 571).id
    pile = Pile(750.0, 500.0, ground, time.monotonic() - 3.0, True, status="seen", xs=(745.0, 755.0),
                last_seen=time.monotonic())
    s = _on(spot, 1000.0, m, [_walker("horny-mushroom", low, 880.0, -1, m)])
    h = _priced(m, [pile], travel=0.1)
    assert [g.monster.x for g in h._guards(s, pile, 1.1, h._stand_s(pile))[0]] == [880.0]
    h._hits = lambda s, m: False
    assert h._guards(s, pile, 1.1, h._stand_s(pile)) == ((), f"a horny-mushroom at x 880 on {low} would meet her there")
    deeper = _on(spot, 1000.0, m, [_walker("horny-mushroom", deep, 880.0, -1, m)])
    assert _priced(m, [pile], travel=0.1)._guards(deeper, pile, 1.1, 0.5) == ((), None)
    monkeypatch.setattr(hunt, "BESIDE_GUARDS", False)
    assert _priced(m, [pile], travel=0.1)._guards(s, pile, 1.1, 0.5) == ((), None)
    # Rule 1 with her fights' BESIDE (on by default): the one under her 120 off (its keep-off 169) is too close; 70
    # under her it isn't on her level.
    from gamelab.attack import stance
    assert stance.BESIDE
    h = _priced(m, [], travel=0.1)
    h._halves = {"horny-mushroom": 28.0}
    h.h.actions = SimpleNamespace(facing="right", lock_until=0.0, attack_at=None)
    assert h._close_quarters(_on(ground, 760.0, m), [_walker("horny-mushroom", low, 880.0, -1, m)])
    assert not h._close_quarters(_on(ground, 760.0, m), [_walker("horny-mushroom", deep, 880.0, -1, m)])


def test_her_own_kill_dying_by_its_drops_guards_nothing():
    """Live on 2026-10-30 12-28% of the piles' waits (PILE_CLEAR) were for the pile's own monster, drawn dying where it
    dropped them: her kill placed on it (ledger.dying) guards nothing, nor a track the feed has taken for dead."""
    from gamelab.behavior.ledger import Kill
    m, ground, spot = _ground()
    now = time.monotonic()
    pile = Pile(750.0, 500.0, ground, now - 0.3, True, xs=(750.0,), status="seen", last_seen=now)
    corpse = _walker("horny-mushroom", ground, 752.0, 0, m, "stand")
    from dataclasses import replace
    s = _on(spot, 1000.0, m, [corpse])
    s.t = now
    assert _priced(m, [pile], travel=0.1)._guards(s, pile, 1.1, 0.5)[0]  # (a live one there: a guard)
    dead = _priced(m, [pile], kills=[Kill(now - 0.3, "horny-mushroom", ground, 750.0, 500.0, ground)], travel=0.1)
    assert dead._guards(s, pile, 1.1, 0.5) == ((), None)
    s.monsters = [replace(corpse, track=7)]
    gone = _priced(m, [pile], travel=0.1)
    gone.h.ledger.dead = {7: now}
    assert gone._guards(s, pile, 1.1, 0.5) == ((), None)


class _Acts:
    """Her actions' stand-in: the walks and pick-ups, in order."""

    def __init__(self):
        self.done = []

    def walk(self, side, units):
        self.done.append(("walk", side, round(units)))

    def pick_here(self, presses):
        self.done.append(("pick", presses))


def test_a_stand_walks_to_each_bunch_presses_once_a_drop_and_leaves_what_she_stood_on_twice():
    from gamelab.behavior.looting import Looter
    now = time.monotonic()
    pile = Pile(130.0, 0.0, 1, now - 5, True, status="seen", xs=(100.0, 105.0, 160.0), last_seen=now)
    h = _hunt([pile])
    said = []
    h.h.looter = Looter(SimpleNamespace(log=said.append))
    h.h.actions, h.h.report = _Acts(), SimpleNamespace(loot_runs=0)
    h.h.drops.tried = lambda p: setattr(p, "tries", p.tries + 1)
    h.report, h.model, h._stood = hunt.HuntReport(), None, {}
    coins = [SimpleNamespace(platform=1, x=x, mesos=True, item=SimpleNamespace(kind="mesos")) for x in (100.0, 105.0, 160.0)]
    s = _on_line(1, 90.0)
    s.loot = list(coins)
    h._stand(s, pile)
    assert h.h.actions.done == [("walk", "right", 12), ("pick", 3), ("walk", "right", 58), ("pick", 2)]
    assert pile.tries == 1 and h._stood[id(pile)][1] == [102.5, 160.0]
    h.h.actions.done.clear()
    h._stand(s, pile)                         # still there: stood on again (from the far end: she's at 90)
    assert [a for a in h.h.actions.done if a[0] == "pick"] == [("pick", 3), ("pick", 2)] and pile.tries == 2
    h.h.actions.done.clear()
    h._stand(s, pile)                         # stood on twice and still there: someone else's
    assert h.h.actions.done == [] and pile.tries == 3 and said[-1].startswith("  can't pick up 3 mesos at x 100, 105, 160")


def test_a_stand_is_not_made_where_a_monster_would_meet_her_now():
    m, ground, spot = _ground()
    now = time.monotonic()
    pile = Pile(750.0, 500.0, ground, now - 5, True, status="seen", xs=(750.0,), last_seen=now)
    h = _priced(m, [pile], travel=0.1)
    h.h.actions = _Acts()
    h._hits = lambda s, m: False
    s = _on(ground, 740.0, m, [_walker("horny-mushroom", ground, 800.0, -1, m)])
    s.loot = []
    h._stand(s, pile)
    assert h.h.actions.done == [] and pile.tries == 0


def test_leaving_a_ground_the_sweep_has_her_drops_on_her_way_to_the_next_first():
    h = _hunt([])
    asked, acted = [], []
    h._next_dest = lambda s: (7, -95.0)
    h._sweep = lambda s, dest, say=True: asked.append(dest) or "a stop"
    h._sweep_act = lambda s, stop, visible, hidden, now: acted.append(stop)
    h._leave(_scene(), "nothing left here")
    assert asked == [(7, -95.0)] and acted == ["a stop"]
    h._sweep = lambda s, dest, say=True: None
    h._next_ground = lambda s: acted.append("next ground")
    h._travel = lambda rule: acted.append(rule)
    h.log = lambda line: acted.append(line)
    h._leave(_scene(), "nothing left here")
    assert acted[1:] == ["  leaving platform 1: nothing left here", "next ground", "advance"]


def test_on_her_way_to_a_sweeps_pile_she_goes_on_while_it_is_first_else_takes_what_is():
    """The objective a pile of the sweep's: solved again from every look on her way on (Objective.after: leaving, the next
    ground; else the objective under it, else her spot); the same pile first, on her trip there; another, or none, the
    objective under it and what the sweep has now."""
    pile, other = _pile(5), _pile(5, platform=3)
    h = _hunt([pile, other])
    h.stack, h.obj = [Objective("spot", 9, 50.0)], Objective("pile", 2, 100.0, pile=pile, why="sweep", after=(7, 0.0))
    first, asked, acted = [pile], [], []
    h._pile_of = {pile.key: pile, other.key: other}

    def solve(s, dest, say=True):
        asked.append(dest)
        return SimpleNamespace(spot=SimpleNamespace(key=first[0].key, guards=()))
    h._sweep = solve
    h._at_pile = lambda s, p: False
    h._travel = lambda rule: rule
    h._sweep_act = lambda s, stop, visible, hidden, now: acted.append(stop.spot.key)
    assert h._on_sweep(_scene(), [], [], 0.0) == "sweep" and asked == [(7, 0.0)]
    first[0] = other
    h._on_sweep(_scene(), [], [], 0.0)
    assert acted == [other.key] and h.obj.kind == "spot"


def test_a_made_up_look_leaves_the_sweeps_itinerary_as_it_was():
    """The navigator asks the hunt's `until` on a look made up from another, her put where a chain ends (its
    _stop_calls, _look_ahead: Scene.what_if): the sweep's answer there is given, but its itinerary (sweep.Sweeper), its
    piles and what it said stay as they were, and its "later, ..." isn't said from a place she isn't at (the review of
    2026-10-30)."""
    coins = _pile(5, platform=2, items=("mesos",), worth=33.4, xs=(300.0,), status="seen")
    coins.x, coins.last_seen = 300.0, time.monotonic()
    h = _line_hunt([coins])
    h.obj = SimpleNamespace(kind="spot", why="travel", platform=3, x=600.0)
    real = _on_line(1, 0.0)
    assert h._sweep_calls(real)  # (the pile on her way)
    before = (h._sweeper.current, h._sweeper.dest, dict(h._pile_of), h._swept_for, h._sweep_said, list(h.said))
    there = _on_line(3, 590.0)  # (where a chain ends: past the pile)
    there.what_if = True
    h._sweep_calls(there)
    assert (h._sweeper.current, h._sweeper.dest, dict(h._pile_of), h._swept_for, h._sweep_said, list(h.said)) == before


def test_the_grinds_way_back_to_its_stop_is_never_cut_for_a_pile_on_it():
    """The grind's way back to its stop (behavior.grind's `_back`, 'back to spot'): the sweep doesn't call her off it for
    a pile on her way (`_sweep_calls`), as it does a trip's; until 2026-10-31 it did, and the pass after the cut didn't
    sweep on that way, so she set off again, cut again, about 900 times in a row on the harness's seed 9."""
    coins = _pile(5, platform=2, items=("mesos",), worth=33.4, xs=(300.0,), status="seen")
    coins.x, coins.last_seen = 300.0, time.monotonic()
    h = _line_hunt([coins])
    h.obj = SimpleNamespace(kind="spot", why="travel", platform=3, x=600.0)
    assert h._sweep_calls(_on_line(1, 0.0))
    h.obj = SimpleNamespace(kind="spot", why="back to spot", platform=3, x=600.0)
    assert not h._sweep_calls(_on_line(1, 0.0))


def test_leaving_a_ground_the_sweep_has_its_budgets_over_the_whole_way_out():
    """DEPART_BUDGETS: the piles on her way out within twice the sweep's budget over the whole way from when she
    decided (sweep.budget_for of the straight way then), less the straight way on from where she is, not a fresh
    budget at each solve (the review of 2026-10-30: master's "she never left" guard gone, tour_tree_run 534 -> 628 s);
    spent, nothing more is worth a detour; staying after all, as before."""
    h = _line_hunt([])
    h.visit = None
    s = _on_line(1, 0.0)
    h._exit_reason = "nothing came into reach"
    decided = time.monotonic() - 3.0  # (she decided 3 s ago, her way out then 1.5 s straight: 1.5 + 2 x 2.0 = 5.5 s)
    h._depart = (decided, 1.5 + hunt.DEPART_BUDGETS * sweep.budget_for(1.5))
    budget, floor = h._sweep_terms(s, (2, 0.0))  # (1.5 s straight on from here)
    assert budget == pytest.approx(5.5 - 3.0 - 1.5, abs=0.05) and floor == 0.0
    h._depart = (decided - 10.0, h._depart[1])
    assert h._sweep_terms(s, (2, 0.0))[0] == 0.0
    h._exit_reason = None  # (not leaving: its piles' allowances alone)
    assert h._sweep_terms(s, (2, 0.0)) == (None, 0.0)


def test_a_pile_may_add_to_her_way_by_its_age_and_less_where_it_is_more_dangerous():
    """`_allow` (the user, 2026-11-01: "piles should be picked up soon after killing. If a pile is about to expire, it
    probably means they aren't important enough to get picked up anyway"): ALLOW_FRESH to FRESH_S, ALLOW_OLD from OLD_S,
    straight between; and ALLOW_OLD at most for a pile where the danger is SAFER_MARGIN over where she stands ("try not
    to hit by mobs just for picking up loots"): on Tree Dungeon's made-up 23 and 24, fresh drops by the Horny Mushroom's
    spawn point on the ground, she on her spot beside it, against drops on her spot's platform. A bar of her HP is still
    its potions' mesos at MESO_EXP (bar_exp: the HP price of her touches, not of loot)."""
    h = _hunt([])
    now = time.monotonic()
    assert h._allow(_pile(5), now) == h._allow(_pile(hunt.FRESH_S), now) == hunt.ALLOW_FRESH
    assert h._allow(_pile(hunt.OLD_S + 1), now) == hunt.ALLOW_OLD
    mid = (hunt.FRESH_S + hunt.OLD_S) / 2
    assert h._allow(_pile(mid), now) == pytest.approx((hunt.ALLOW_FRESH + hunt.ALLOW_OLD) / 2, abs=0.01)
    m, ground, spot = _ground()
    by_point = Pile(790.0, 500.0, ground, now - 3.0, True, status="seen", xs=(785.0, 795.0), last_seen=now)
    by_her = Pile(960.0, 480.0, spot, now - 3.0, True, status="seen", xs=(955.0, 965.0), last_seen=now)
    h = _priced(m, [by_point, by_her], travel=0.1)
    s = _on(spot, 1060.0, m)
    d = h.h.danger
    assert d.at(ground, 790.0, []) > d.at(spot, 1060.0, []) + hunt.SAFER_MARGIN >= d.at(spot, 960.0, [])
    found, out = h._spots(s)
    allow = {h._pile_of[sp.key] is by_point: sp.allow for sp in found}
    assert allow == {True: hunt.ALLOW_OLD, False: hunt.ALLOW_FRESH} and not out
    assert hunt.bar_exp(774.0) == 774.0 / hunt.HP_PER_MESO * hunt.MESO_EXP == 193.5
