"""The fight loop (behavior.fight.clear) on prepared looks and a fake clock: hidden monsters, kills
from her EXP, the kill count, far targets, the star's path, what the shot log keeps (and which throw a
kill is credited to); and the hunter counting EXP gains."""

import time
from types import SimpleNamespace

import numpy as np

from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import reach
from gamelab.behavior import fight as fight_mod
from gamelab.behavior.fight import EMPTY_PATIENCE, HIDDEN_FOR
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.looting import LOOT_REACH, DropSpot
from gamelab.localization import mapdata
from gamelab.perception.eyes import Observation

from test_attack import PLAT, target

EMPTY_MAP = {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
             "miniMap": {"centerX": 0, "centerY": 0, "mag": 4}}


class _Eyes:
    """Hands out prepared looks (no game needed)."""
    want_ropes = False

    def __init__(self, looks):
        self.looks = iter(looks)

    def look(self):
        return next(self.looks)


def hunter(eyes=None) -> Hunter:
    return Hunter(eyes or _Eyes([]), GameKeys.__new__(GameKeys), mapdata.parse(1, EMPTY_MAP, "Test", {}),
                  log=lambda m: None)


def test_a_kill_is_confirmed_by_her_exp_number_changing():
    before = np.zeros((26, 140), bool)
    after = before.copy()
    after[5:15, 100:104] = True  # a digit changed
    looks = [Observation(t=t, frame=np.zeros((10, 10, 3), np.uint8), tag=None, hp=1.0, mp=1.0, exp=e)
             for t, e in ((1.0, before), (1.5, before), (2.0, after), (2.5, after))]
    h = hunter(_Eyes(looks))
    for _ in looks:
        h.look(mobs=False)
    assert h.exp_events == [2.0] and h.report.exp_kills == 1
    assert not h.take_exp(since=2.5)   # gains before she started throwing aren't this target's
    assert h.take_exp(since=1.8) and h.exp_events == []


def _fighter(monkeypatch, looks, on_look=None):
    """A Hunter whose looks come from `looks` (lists of monsters), 0.3 s apart on a fake clock; her
    throws are recorded instead of pressed."""
    h = hunter()
    clock, seq, throws = [0.0], iter(looks), []

    def look(*a, **k):
        clock[0] += 0.3
        if on_look:
            on_look(h, clock[0])
        return SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT, hp=1.0, monsters=next(seq, []))

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h.look, h.health_check = look, lambda s: None
    h.actions.throw = h.actions.jump_throw = h.actions.jump_back = lambda side: throws.append((round(clock[0], 1), side))
    h.looter.sweep_through = lambda s, deaths: None
    h.looter.mop_up = lambda stood=None: None  # looking for loot is tested on its own (test_loot.py)
    return h, throws


def test_she_keeps_throwing_at_a_monster_hidden_after_a_hit(monkeypatch):
    h, throws = _fighter(monkeypatch, [[target(700)]])  # seen once, then lost under its damage numbers
    r = h.clear(patience=2.0)
    assert len(throws) == 1 + round(HIDDEN_FOR / 0.3 - 1) and {side for _, side in throws} == {RIGHT}
    assert r.stop == "nothing left in reach"


def test_her_exp_going_up_ends_the_throws_at_a_hidden_monster(monkeypatch):
    def kill_at(h, t):
        if abs(t - 0.9) < 1e-6:
            h.report.exp_kills += 1
    h, throws = _fighter(monkeypatch, [[target(700)]], on_look=kill_at)
    r = h.clear(patience=2.0)
    assert len(throws) == 2 and r.kills == {"pig": 1}


def test_with_no_monster_in_sight_she_moves_on_soon(monkeypatch):
    h, throws = _fighter(monkeypatch, [])
    h.clear(patience=2.0)
    assert time.monotonic() <= 0.3 + EMPTY_PATIENCE + 0.3 and not throws
    # One seen, just out of reach: she waits the full patience for it to walk in.
    far = [[target(980, y=200)]] * 20
    h, throws = _fighter(monkeypatch, far)
    h.clear(patience=2.0, approach=False)
    assert time.monotonic() >= 2.0 and not throws


def test_drops_are_swept_once_she_lands_or_left_to_the_tour(monkeypatch):
    h, _ = _fighter(monkeypatch, [])
    swept = []
    h.looter.sweep_through = lambda s, deaths: swept.append([d.x for d in deaths])
    air = SimpleNamespace(where=None, platform=None)
    ground = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    looks = iter([air, air, ground])
    h.look = lambda *a, **k: next(looks)
    deaths = [DropSpot(600, 500, 0, 0.0), DropSpot(700, 560, 1, 0.0)]  # one on her platform, one below
    h.looter.sweep_after(deaths)
    assert swept == [[600, 700]] and [d.x for d in h.looter.pending] == [700]
    h.looter.pending, swept[:] = [], []
    h.look = lambda *a, **k: air  # never lands (on a rope, say): the tour collects them later
    h.looter.sweep_after(deaths)
    assert not swept and len(h.looter.pending) == 2


def test_she_stands_on_each_pile_pressing_pick_up(monkeypatch):
    h, _ = _fighter(monkeypatch, [])
    del h.looter.sweep_through  # the real one
    did = []
    h.actions.walk = lambda direction, units: did.append(("walk", direction, round(units)))
    h.actions.pick_here = lambda presses=4: did.append(("pick",))
    s = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    h.looter.sweep_through(s, [DropSpot(600, 500, 0, 0.0), DropSpot(400, 500, 0, 0.0)])
    # Nearer end first (both 100 away: the first found), over each pile, stopping on it.
    walks = [d for d in did if d[0] == "walk"]
    assert did.count(("pick",)) == 2
    assert sum(u for _, d, u in walks if d == RIGHT) - sum(u for _, d, u in walks if d == LEFT) in (
        600 + LOOT_REACH - 500, 400 - LOOT_REACH - 500)


def test_a_kill_after_she_moved_on_to_the_next_target_still_gets_its_drops(monkeypatch):
    # She threw at a pig on the right, then turned to one on the left; the right one's death (EXP up)
    # comes in after that, when the last throw aimed only left.
    def kill_at(h, t):
        if abs(t - 1.2) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700)], [target(450)], [target(450)], [target(450)], [target(450)]]
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    swept = []
    h.looter.sweep_after = lambda deaths: swept.extend(d.x for d in deaths)
    h.clear(max_kills=1, patience=2.0)
    assert [side for _, side in throws][:2] == [RIGHT, LEFT]  # then at the one on her left
    assert h.report.kills == {"pig": 1} and swept == [700]   # not the one still in plain sight


def test_after_the_kill_count_she_finishes_what_is_in_reach(monkeypatch):
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700), target(250)]] * 2 + [[target(250)]] * 3 + [[]] * 5
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.looter.sweep_after = lambda deaths: None
    r = h.clear(max_kills=1, patience=2.0, finish=True)
    assert len(throws) >= 3 and LEFT in [side for _, side in throws]  # the one on the left too
    assert r.stop == "kill limit"
    h2, throws2 = _fighter(monkeypatch, looks, on_look=kill_at)
    h2.looter.sweep_after = lambda deaths: None
    h2.clear(max_kills=1, patience=2.0)  # without finish: stops at the kill
    assert len(throws2) < len(throws)


def test_a_fight_finishes_one_she_has_hurt_before_it_ends(monkeypatch):
    # The kill count is reached with a pig she has hurt still in reach (Hunter.wounds): without `finish` the
    # fight goes on until that one is dead too (she used to leave monsters half hunted).
    def kills(h, t):
        if abs(t - 0.6) < 1e-6 or abs(t - 1.5) < 1e-6:
            h.report.exp_kills += 1
    hurt = target(250, bar=0.5)
    looks = [[target(700), hurt]] * 2 + [[hurt]] * 3 + [[]] * 5
    h, throws = _fighter(monkeypatch, looks, on_look=kills)
    h.looter.sweep_after = lambda deaths: None
    h.wounds.seen_within = lambda now, seconds, hurt_by=None: [hurt] if h.report.exp_kills < 2 else []
    h.clear(max_kills=1, patience=2.0)
    assert [side for _, side in throws][-3:] == [LEFT] * 3 and h.report.kills == {"pig": 2}
    finished = len(throws)
    # Out of reach but a short walk away, she closes in on it even in a fight that doesn't (approach off).
    walks = []
    h, throws = _fighter(monkeypatch, [[target(950, bar=0.5)]] * 4)
    h.actions.walk = lambda direction, units: walks.append((direction, round(units)))
    h.wounds.seen_within = lambda now, seconds, hurt_by=None: [target(950, bar=0.5)]
    h.clear(seconds=1.0, approach=False)
    assert walks and walks[0][0] == RIGHT and walks[0][1] <= fight_mod.FINISH_WALK and not throws
    # Farther than that: she's free to go.
    walks.clear()
    far = target(500 + reach.SHOT_RANGE + fight_mod.FINISH_WALK + 60, 560, platform=1, bar=0.5)  # 60 below
    h, throws = _fighter(monkeypatch, [[far]] * 4)
    h.actions.walk = lambda direction, units: walks.append((direction, round(units)))
    h.wounds.seen_within = lambda now, seconds, hurt_by=None: [far]
    r = h.clear(seconds=1.0, patience=0.3, approach=False)
    assert not walks and r.stop == "nothing left in reach"
    # Without finish_wounds (a probe's fight off one close to her: the wounds are the probe's targets) it ends at
    # its kill count, and doesn't close in on one.
    h, throws = _fighter(monkeypatch, looks, on_look=kills)
    h.looter.sweep_after = lambda deaths: None
    h.wounds.seen_within = lambda now, seconds, hurt_by=None: [hurt] if h.report.exp_kills < 2 else []
    h.clear(max_kills=1, patience=2.0, finish_wounds=False)
    assert h.report.kills == {"pig": 1} and len(throws) < finished
    walks.clear()
    h, throws = _fighter(monkeypatch, [[target(950, bar=0.5)]] * 4)
    h.actions.walk = lambda direction, units: walks.append((direction, round(units)))
    h.wounds.seen_within = lambda now, seconds, hurt_by=None: [target(950, bar=0.5)]
    h.clear(seconds=1.0, patience=0.3, approach=False, finish_wounds=False)
    assert not walks and not throws


def test_a_probe_fights_off_one_close_to_her_without_finishing_its_own_targets(monkeypatch):
    # probe._keep_clear's fights (probe._fight): one off a monster close to her on her platform, up to one kill,
    # and a clear of her platform first, both leaving the monsters she has hurt alone (the probe's own targets:
    # its stars at them would be outside the probe's log).
    from gamelab.behavior import probe
    asked = {}
    h = SimpleNamespace(log=lambda *a: None, clear=lambda kinds, **kw: asked.update(kw))
    p = probe.PROBES["step-down"]
    s = SimpleNamespace(where=SimpleNamespace(x=50.0, y=0.0), platform=SimpleNamespace(id=2),
                        monsters=[target(70, 0.0, platform=2, kind="orange-mushroom")])
    report = probe.ProbeReport(p.name)
    assert probe._keep_clear(h, p, s, probe._Guard(False, p.x), report)
    assert asked["finish_wounds"] is False and asked["max_kills"] == 1
    asked.clear()
    assert probe._keep_clear(h, p, s, probe._Guard(True, p.x), report)  # clears_first: step-down from 2
    assert asked["finish_wounds"] is False and asked["max_kills"] == probe.CLEAR_KILLS and report.fights == 2


def test_the_escorts_fight_in_her_way_finishes_wounds_only_when_asked():
    # FightOnTheWay.clear_way: a fight off the monster in her way, up to two kills, finishing what she has hurt
    # (clear's finish_wounds) unless `finish` is off, as the navigator has it in transit.
    asked = {}
    h = SimpleNamespace(log=lambda *a: None, clear=lambda kinds=None, **kw: asked.update(kw),
                        report=SimpleNamespace(exp_kills=0))
    escort = fight_mod.FightOnTheWay(h)
    s = SimpleNamespace(where=SimpleNamespace(x=50.0, y=0.0), platform=SimpleNamespace(id=2),
                        monsters=[target(200, 0.0, platform=2)])
    assert escort.clear_way(s, 300.0) == 0 and asked["finish_wounds"] is True and asked["max_kills"] == 2
    assert escort.clear_way(s, 300.0, finish=False) == 0 and asked["finish_wounds"] is False
    asked.clear()
    assert escort.clear_way(s, -400.0) is None and not asked  # nothing in her way that way


def test_a_kill_seen_while_she_sweeps_up_still_counts(monkeypatch):
    # Her EXP goes up again during the sweep after a kill (a star in flight killed the other pig), seen by a
    # look of the sweep's: the fight ends at its kill count there, and that kill is still told apart (the
    # tour's stops sum up the kills by kind, and missed it).
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700), target(800)], [target(800)]] + [[]] * 20
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    swept = []

    def sweep(s, deaths):
        swept.append([d.x for d in deaths])
        h.report.exp_kills += 1
    h.looter.sweep_through = sweep
    h.looter.sweep_after = lambda deaths: swept.append([d.x for d in deaths])
    r = h.clear(max_kills=2, patience=2.0)
    assert r.exp_kills == 2 and r.kills == {"pig": 2} and r.stop == "kill limit"
    assert swept == [[700], [800]]  # the second one's drops are swept up too
    # The same seen by the sweep after the fight (at its kill count after the first): counted, its drops left.
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.looter.sweep_through = lambda s, deaths: None
    swept = []

    def sweep_after(deaths):
        swept.append([d.x for d in deaths])
        if len(swept) == 1:
            h.report.exp_kills += 1
    h.looter.sweep_after = sweep_after
    r = h.clear(max_kills=1, patience=2.0)
    assert r.exp_kills == 2 and r.kills == {"pig": 2} and swept == [[700]]
    assert h.looter.pending == []  # its drops are left (going after them made loot trip after loot trip)


def test_with_far_on_she_shoots_monsters_on_any_platform_in_reach(monkeypatch):
    far_pig = target(700, y=560, platform=7)  # another platform, 60 below and 200 off: in reach
    h, throws = _fighter(monkeypatch, [[far_pig]] * 3)
    h.clear(ground={3}, patience=0.5, far=True)
    assert throws and throws[0][1] == RIGHT
    h, throws = _fighter(monkeypatch, [[far_pig]] * 3)
    h.clear(ground={3}, patience=0.5)  # without far: not one of hers
    assert not throws


def test_a_kill_is_matched_to_the_first_monster_in_the_star_path(monkeypatch):
    # A snail (not hunted) stands between her and the pig she throws at: the star hits the snail,
    # and the EXP that comes in is the snail's, not the pig's.
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    snail, pig = target(650, kind="snail"), target(780)
    looks = [[snail, pig], [pig], [pig]] + [[]] * 6
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.looter.sweep_after = lambda deaths: None
    h.clear(kinds={"pig"}, max_kills=1, patience=0.5)
    assert h.report.kills == {"snail": 1}


def test_a_jump_throw_aims_at_the_platform_above_and_over_a_snail_beside_her():
    # She's on PLAT (y 500) at x 500; a platform 60 above runs to her left. A jump throw's attack picks
    # its target ~49 units up: the slimes up there are in its area at any distance a star reaches (on
    # the ground, one 60 above needs 117 sideways), and the snail 30 away on her own platform is under it.
    from gamelab.behavior.fight import _in_path
    from gamelab.motion.physics import JUMP_ATTACK_MS
    late = 100  # the attack keyed 100 ms after the jump (now JUMP_ATTACK_MS is 40: see the end)
    s = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    snail = target(470, kind="snail")
    near_slime, far_slime = target(400, y=440, platform=5, kind="slime"), target(300, y=440, platform=5, kind="slime")
    too_far = target(30, y=440, platform=5, kind="slime")  # 470 away: past the star's reach
    seen = [snail, near_slime, far_slime, too_far]
    assert _in_path(s, LEFT, seen, late) == [near_slime, far_slime]
    assert _in_path(s, LEFT, seen) == [snail, far_slime]  # a throw from the ground: as before
    # A pig beside her is tall enough for the jump throw's attack.
    pig = target(470)
    assert _in_path(s, LEFT, [pig, near_slime], late) == [pig, near_slime]
    # Keyed 40 ms after the jump, she's low enough that the snail beside her is in the attack's path first.
    assert _in_path(s, LEFT, seen, JUMP_ATTACK_MS)[0] is snail


def test_a_monster_just_hit_is_in_the_path_as_tall_as_it_flinches():
    # A pig 60 below her and 130 along: walking (46 tall) the star's search from the ground picks it only
    # from 151, but just hit it flinches 54 tall for 0.6 s and is picked from 111.
    from gamelab.behavior.fight import FightState, _in_path
    s = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    pig = target(630, 560, platform=1)
    assert _in_path(s, RIGHT, [pig]) == [] and _in_path(s, RIGHT, [pig], flinching=[pig]) == [pig]
    # Just hit: thrown at, and its HP bar new or lower than when it was last seen (not one whose bar stayed).
    st = FightState(thrown=[[target(630, 560, platform=1), 1.0], [target(700, 560, platform=1, bar=0.5), 1.0],
                            [target(300, bar=0.6), 1.0]])
    seen = [target(632, 560, platform=1, bar=0.7), target(702, 560, platform=1, bar=0.5), target(300, bar=0.3)]
    assert st.just_hit(seen) == [seen[0], seen[2]]


def test_no_kill_is_matched_behind_a_monster_whose_star_is_stopped(monkeypatch):
    # test_attack's ledge map: from H at 100 the search picks a pig on L at 330 first (230 off, band 6), in her
    # floor's shadow, so her floor stops its star (the v83 line, reach's); the client doesn't refill its pick, so
    # the pig on L at 420 (320 off, band 11, its own line clear) can't have taken that throw's attack either.
    # Neither the fight's path nor the ledger's has one: a kill then isn't put on either. Alone, the one at 420 is
    # in both.
    from gamelab.behavior.fight import _in_path
    from gamelab.behavior.ledger import in_path
    from test_attack import ledge_map, shield_map
    model, h, low, up = ledge_map()
    shade, far = target(330, 560, platform=low.id), target(420, 560, platform=low.id)
    s = SimpleNamespace(where=SimpleNamespace(x=100, y=500), platform=h, monsters=[shade, far])
    assert _in_path(s, RIGHT, [shade, far]) == [] and in_path(s, RIGHT, [shade, far]) == []
    assert _in_path(s, RIGHT, [far]) == [far] and in_path(s, RIGHT, [far]) == [far]
    # And behind a wall, on the level line: from A at 10, the search picks a pig up on B at 215 first (band 5),
    # and the step wall stops its star; the pig on L at 170 (band 6, its line clear by the level line) gets none.
    monkeypatch.setattr(reach, "LINE", "level")
    model, a, b, low = shield_map()
    up, under = target(215, 440, platform=b.id), target(170, 560, platform=low.id)
    s = SimpleNamespace(where=SimpleNamespace(x=10, y=500), platform=a, monsters=[up, under])
    assert _in_path(s, RIGHT, [up, under]) == [] and in_path(s, RIGHT, [up, under]) == []
    assert _in_path(s, RIGHT, [under]) == [under] and in_path(s, RIGHT, [under]) == [under]


def test_a_kill_is_left_alone_as_long_as_its_death_animation_runs():
    # Seen fading out where it died for its die1 frames' delays (game data): a pig 0.66 s, a snail 1.26; a
    # kind the data doesn't have, FADING. Counted from the look that saw her EXP go up.
    from gamelab.behavior.fight import FADING, FightState, fade_s
    assert (fade_s("pig"), fade_s("slime"), fade_s("snail"), fade_s("dragon")) == (0.66, 0.84, 1.26, FADING)
    st = FightState(dying=[(target(600), 10.0), (target(700, kind="snail"), 10.0)])
    st.forget_faded(10.7)
    assert [d.kind for d, _ in st.dying] == ["snail"]
    st.forget_faded(11.3)
    assert st.dying == []


def test_a_kill_on_the_platform_above_during_jump_throws_is_placed_there(monkeypatch):
    # Jump-throwing at a pig beside her (cornered: another pig behind her), the attack picks its target
    # ~49 units up, so a slime on the platform above, 100 to the side, is in its path (from the ground one
    # 60 above needs 117): when it dies, the kill is placed there, not on the pig still in plain sight.
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    pig, behind, slime = target(470), target(600), target(400, y=440, platform=5, kind="slime")
    h, throws = _fighter(monkeypatch, [[pig, behind, slime]] + [[pig, behind]] * 5, on_look=kill_at)
    h.looter.sweep_after = lambda deaths: None
    logged = []
    h.log = logged.append
    h.clear(max_kills=1, patience=0.5)
    assert any("jump-throwing left" in line for line in logged) and throws[0][1] == LEFT
    assert h.report.kills == {"slime": 1}


def test_hidden_monsters_are_told_by_the_one_time_read_per_look(monkeypatch):
    # A clock that moves on with every read: a monster seen in this look must not count as hidden
    # (the fight reads the time once per look; a second read would make every monster a "hidden" ghost).
    h, throws = _fighter(monkeypatch, [[target(700)]] * 3 + [[]] * 10)
    ticks = [0.0]

    def monotonic():
        ticks[0] += 0.001
        return ticks[0]
    looks = iter([[target(700)]] * 3)
    h.look = lambda *a, **k: SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT, hp=1.0,
                                             monsters=next(looks, []))
    monkeypatch.setattr(time, "monotonic", monotonic)
    h.clear(patience=0.01, seconds=0.5)
    assert throws and all(side == RIGHT for _, side in throws)


def test_the_shot_log_keeps_where_she_and_the_target_were_and_the_wall_rule():
    # From below the step (A, at x 100) at a monster up on B: a wall of her group in the way. From B down
    # at one on A: clear. Recorded whether or not her decisions keep to the rule (reach.WALLS_BLOCK).
    from gamelab.behavior.fight import _shot
    from test_attack import step_map
    model, a, b, _ = step_map()
    up = _shot(SimpleNamespace(where=SimpleNamespace(x=100, y=500), platform=a, tag_seen=True, monsters=[]),
               target(300, 440, platform=b.id), 200, -60, model)
    assert (up["x"], up["tx"], up["her_group"], up["clear"]) == (100, 300, [1, 1], False)
    down = _shot(SimpleNamespace(where=SimpleNamespace(x=300, y=440), platform=b, tag_seen=True, monsters=[]),
                 target(100, 500, platform=a.id), -200, 60, model)
    assert down["clear"] and down["her"] == b.id and down["target"] == a.id


def test_the_shot_log_keeps_what_was_in_front_then_and_at_the_pick():
    # Every monster on the side she throws, at the throw and at the first look that places her after it (about
    # when the attack picked its target), whether the target is seen in it or not (a hit one is often hidden
    # under its damage numbers), and when that look was, so which one the search could pick can be fitted
    # from the log alone.
    from gamelab.behavior.fight import _note_pick, _shot
    from test_attack import step_map
    model, a, b, _ = step_map()
    pig, behind, snail = (target(300, 440, platform=b.id), target(20, 500, platform=a.id, kind="snail"),
                          target(150, 500, platform=a.id, kind="snail"))
    s = SimpleNamespace(where=SimpleNamespace(x=100, y=500), platform=a, tag_seen=True, monsters=[pig, behind, snail])
    sh = _shot(s, pig, 200, -60, model)
    assert [(m["kind"], m["x"], m.get("target", False)) for m in sh["front"]] == [("pig", 300, True),
                                                                                  ("snail", 150, False)]
    assert sh["y"] == 500 and "pick" not in sh
    _note_pick([sh], SimpleNamespace(where=None, monsters=[], t=sh["key"] + 0.05), None)
    assert "pick" not in sh  # a look that doesn't place her
    later = SimpleNamespace(where=SimpleNamespace(x=104, y=500), monsters=[target(290, 440, platform=b.id)],
                            t=sh["key"] + 0.1)
    _note_pick([sh], later, None)  # the target not seen in it
    assert sh["pick"]["x"] == 104 and sh["pick"]["t"] == 0.1
    assert sh["pick"]["front"] == [{"kind": "pig", "x": 290, "y": 440, "platform": b.id, "bar": None}]
    first = sh["pick"]
    _note_pick([sh], SimpleNamespace(where=SimpleNamespace(x=200, y=500), monsters=[], t=sh["key"] + 0.9), None)
    assert sh["pick"] is first  # the first look after the throw only


def _throw_at(key: float, due: float, bar=None) -> dict:
    """A throw in flight for the shot log's scoring (fight._shot's fields that it goes by)."""
    return {"kind": "pig", "dx": 200, "dy": 0, "bar": bar, "key": key, "before_t": key, "due": due,
            "looks": [(key, bar)]}


def _logged(path) -> list[dict]:
    import json
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_kill_is_credited_to_one_throw_in_the_shot_log(monkeypatch, tmp_path):
    # Three throws in flight when the target is gone and her EXP has gone up: the last one whose star can have
    # landed by then is the kill; the one before it may have hit it too (undecided); the one whose damage
    # can't show yet (under 0.47 s after its key) missed it. (The old log, lab/shots.jsonl, made every throw
    # due by then a hit.) Seen without the HP bar it had, it was dying already: the kill is the last throw
    # that can have landed before that.
    from gamelab import paths
    from gamelab.behavior import fight
    log = tmp_path / "shots2.jsonl"
    monkeypatch.setattr(paths, "SHOT_LOG", log)
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h = hunter()
    h.take_exp = lambda since: True
    h.looter.collect = lambda spot: None

    def gone(flying: list, **kw) -> list:
        st = fight.KillState("pig", (700, 500), 0, first_throw=7.0, flying=flying, **kw)
        n = len(_logged(log))
        assert fight._target_gone(h, st) == "killed"
        return [r["hit"] for r in _logged(log)[n:]]

    assert gone([_throw_at(k, k + 0.8) for k in (8.0, 8.8, 9.6)]) == [None, True, False]
    assert gone([_throw_at(k, k + 0.7) for k in (7.6, 8.4, 9.2)], had_bar=True, dying_since=9.0) == [None, True, False]
    # The last one that can have landed was a far throw, not due yet when it was seen dying: its star may have
    # killed it (the simulator lands a star's damage at its release) or found it dead (if the game shows the
    # damage when the star gets there). Neither is credited. Alone, it's the kill.
    assert gone([_throw_at(8.0, 8.6), _throw_at(8.8, 9.95)], had_bar=True, dying_since=9.4) == [None, None]
    assert gone([_throw_at(8.8, 9.95)], had_bar=True, dying_since=9.4) == [True]
    row = _logged(log)[-1]
    assert (row["due_s"], row["decided_s"], row["bars"]) == (1.15, 0.6, [[0.0, None]])
    assert not {"key", "due", "looks", "decided", "before_t"} & set(row)


def test_a_bar_drop_is_credited_to_the_throw_whose_star_made_it(monkeypatch, tmp_path):
    # Two throws in flight at an unhurt pig. The first star lands (the bar shows, 70% left) before the second
    # one's damage can show: the second is scored from that bar, so when the bar stays at 70% it missed (once
    # it's been seen 0.3 s past its due time). Scored from the bar at its throw (none), it took the first
    # one's hit.
    from gamelab import paths
    from gamelab.behavior import fight
    log = tmp_path / "shots2.jsonl"
    monkeypatch.setattr(paths, "SHOT_LOG", log)
    flying = [_throw_at(0.0, 1.0)]
    fight._score_shots(flying, None, now=0.05)
    flying.append(_throw_at(0.85, 1.7))
    for t, bar in ((1.2, 0.7), (1.85, 0.7)):
        fight._score_shots(flying, bar, now=t)
    assert [r["hit"] for r in _logged(log)] == [True]  # the second not seen long enough after it was due
    fight._score_shots(flying, 0.7, now=2.7)
    rows = _logged(log)
    assert [r["hit"] for r in rows] == [True, False] and all("hit" in sh for sh in flying)
    assert rows[1]["bars"] == [[0.0, None], [0.35, 0.7], [1.0, 0.7], [1.85, 0.7]]


def test_one_bar_drop_two_throws_that_may_have_made_it_are_undecided(monkeypatch, tmp_path):
    # A far throw that missed, then one that hit, the fight looking once a throw (right after each key): the
    # look after the first was due came before its star had surely landed (0.3 s past due), and the next one
    # sees the second's hit. Either star can have made that drop (shown at the release, or when it gets
    # there): both undecided. (The hit wasn't bounded in time: the miss was logged a hit, 15 of 93 misses
    # over two scenarios' seeds.) Seen unchanged long enough after it was due, a throw missed whatever else
    # was in flight.
    from gamelab import paths
    from gamelab.behavior import fight
    log = tmp_path / "shots2.jsonl"
    monkeypatch.setattr(paths, "SHOT_LOG", log)
    for last in (0.67, None):
        flying = [_throw_at(0.0, 0.965)]
        fight._score_shots(flying, None, now=0.05)
        flying.append(_throw_at(0.8, 1.765))
        for t, bar in ((0.85, None), (1.65, last)):
            fight._score_shots(flying, bar, now=t)
        if last is not None:
            assert [r["hit"] for r in _logged(log)] == [None, None]
    assert [r["hit"] for r in _logged(log)[2:]] == [False]
