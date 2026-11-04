"""The target-first planner (attack.target, behavior.targeting; GAMELAB_TARGET): frozen scenes on Tree Dungeon's game data, for
both characters where it differs. Which monster, the next, a respawn, a monster coming at her, the plan's place, her drops
against her target, and the backpack's push (control.notify.BackpackPush)."""

import contextlib
import time
from types import SimpleNamespace

import pytest

from gamelab import character, telemetry
from gamelab.attack import magic, reach
from gamelab.attack import target as tg
from gamelab.attack.stance import NO_PACE, Pace, _sizes, _stand_limits
from gamelab.behavior import grind as grind_mod
from gamelab.behavior import looting, sweep, targeting
from gamelab.behavior.grind import Grind
from gamelab.behavior.ledger import DROP_LANDS, EXPIRE_S, LANDED_S, OWN_S, REVIVE_S, DropStore, Pile
from gamelab.behavior.looting import Looter
from gamelab.control import notify
from gamelab.localization.monsters import Monster, bodies, touch_box
from gamelab.motion import physics
from gamelab.motion.route import Planner
from gamelab.perception.mobs import Mob
from gamelab.sim import scenarios

TREE = 10002073
HORNY, GREEN = "horny-mushroom", "green-mushroom"


@pytest.fixture(scope="module")
def tree():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m, Planner(m)


def _as(magician: bool):
    """Her reach's constants (attack.magic) for the block, or the thief's as they are."""
    return magic.swapped(character.load("magician")) if magician else contextlib.nullcontext()


def _bodies(m):
    tops, halves = {}, {}
    for k, b in bodies(m).items():
        halves[k], tops[k] = touch_box(k, b)
    return tops, halves


def _mon(m, kind, x, pf, bar=None, track=None, fading=None):
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9, fading=fading), x, m.platform(pf).y_at(x), pf, bar=bar, track=track)


def _her(m, pf, x, around, magician, lead=0.0, inv=0.0, facing="right", pace=NO_PACE, can_jump=True):
    p = m.platform(pf)
    tops, halves = _bodies(m)
    lo, hi = _stand_limits(p, crowded=True)
    return tg.Her(p, x, p.y_at(x), lead, inv, facing, not magician, list(around), _sizes(tops, halves) if magician else None,
                  halves, None, pace, lo, hi, can_jump=can_jump)


def _foe(mon, exp=41.0, kill=2.5, vx=None, chasing=False, solid=0.0, hit=0.0, born=False, p=1.0):
    return tg.Foe(mon, exp, kill, p, vx, chasing, solid, hit, born)


both = pytest.mark.parametrize("magician", [True, False], ids=["her", "thief"])


# --- which monster ---------------------------------------------------------------------------------------------------------

@both
def test_just_topped_out_at_minus_80_she_takes_the_hurt_chaser_at_once_and_the_walker_next(tree, magician):
    """Just topped out at the rope at -80 on 10, her invulnerability left, a hurt Horny Mushroom chasing on her left and one
    walking on her right: the hurt one first (its kill the soonest, no term of its own for that), the walker the next; no
    walk: she takes it from where she stands (a jump attack or a step to range where it's too close)."""
    m, _ = tree
    with _as(magician):
        hurt = _mon(m, HORNY, -113.0, 10, bar=0.18, track=1)
        walker = _mon(m, HORNY, -30.0, 10, track=2)
        her = _her(m, 10, -80.0, [hurt, walker], magician, inv=0.65, facing="left")
        c = tg.Chooser().choose(her, [_foe(hurt, kill=0.6, vx=60.0, chasing=True), _foe(walker, kill=2.5, vx=-60.0)])
    assert c.primary.m is hurt and c.next.m is walker
    assert c.primary.way.how != "walk" and c.primary.way.place is None


@both
def test_a_monster_too_close_for_her_attack_is_a_jump_attack_or_a_step_to_range(tree, magician):
    """Too close for her attack from the ground: the thief's two-key jump attack (right after take-off), her step to range
    (no casts mid-jump): how soon either lands is what the score goes by."""
    m, _ = tree
    with _as(magician):
        near = _mon(m, GREEN, 5.0 if magician else 30.0, 10, track=1)
        her = _her(m, 10, 0.0, [near], magician)
        way = tg.soonest(her, _foe(near, exp=30.0))
    assert way.how == "close"
    if magician:
        assert way.s >= tg.STEP_SETTLE_S
    else:
        assert way.s == pytest.approx(tg.JUMP_S)


def test_with_no_monster_in_view_there_is_no_target(tree):
    m, _ = tree
    her = _her(m, 10, -80.0, [], False)
    c = tg.Chooser().choose(her, [])
    assert c.primary is None and c.next is None and not c.why


@both
def test_of_two_monsters_the_one_killed_sooner_goes_first_though_farther(tree, magician):
    """One whole at -120 and one almost dead at 240, both in reach from where she stands: the one that gives its EXP sooner."""
    m, _ = tree
    with _as(magician):
        whole = _mon(m, HORNY, -120.0, 10, track=1)
        hurt = _mon(m, GREEN, 240.0, 10, bar=0.1, track=2)
        her = _her(m, 10, 0.0, [whole, hurt], magician)
        reachable = reach.can_hit(her.platform, 0.0, her.y, hurt, others=[hurt], sizes=her.sizes)
        c = tg.Chooser().choose(her, [_foe(whole, exp=41.0, kill=2.5), _foe(hurt, exp=30.0, kill=0.4)])
    assert reachable and c.primary.m is hurt and c.next.m is whole


def test_the_target_is_kept_until_another_beats_it_by_the_margin(tree):
    """No flip for a small difference; a clear one changes it, said why."""
    m, _ = tree
    a, b = _mon(m, HORNY, 120.0, 10, track=1), _mon(m, HORNY, -120.0, 10, track=2)
    her = _her(m, 10, 0.0, [a, b], False)
    ch = tg.Chooser()
    assert ch.choose(her, [_foe(a, kill=2.0), _foe(b, kill=2.1)]).primary.m is a
    kept = ch.choose(her, [_foe(a, kill=2.0), _foe(b, kill=1.8)])
    assert kept.primary.m is a and not kept.why
    took = ch.choose(her, [_foe(a, kill=2.0), _foe(b, kill=1.0)])
    assert took.primary.m is b and took.how == "beaten"


# --- the next ----------------------------------------------------------------------------------------------------------------

def test_target_killed_mid_plan_the_next_on_the_same_pass(tree):
    """The primary dead (gone from the look's foes) as she's about to walk for it: the choice of that very look is the next,
    with no pass of its own, and its next comes with it."""
    m, _ = tree
    a, b, d = _mon(m, GREEN, 150.0, 10, bar=0.1, track=1), _mon(m, HORNY, -150.0, 10, track=2), _mon(m, GREEN, -200.0, 10, track=3)
    her = _her(m, 10, 0.0, [a, b, d], False)
    ch = tg.Chooser()
    first = ch.choose(her, [_foe(a, exp=30.0, kill=0.3), _foe(b, kill=2.5), _foe(d, exp=30.0, kill=2.0)])
    assert first.primary.m is a and first.next is not None
    was_next = first.next.m
    her2 = _her(m, 10, 0.0, [b, d], False)
    after = ch.choose(her2, [f for f in (_foe(b, kill=2.5), _foe(d, exp=30.0, kill=2.0))])
    assert after.primary.m is was_next and after.how == "next"
    assert after.next is not None and after.next.m is not was_next


def test_the_next_gone_when_the_primary_dies_she_chooses_afresh(tree):
    m, _ = tree
    a, b, d = _mon(m, GREEN, 150.0, 10, bar=0.1, track=1), _mon(m, HORNY, -150.0, 10, track=2), _mon(m, GREEN, -200.0, 10, track=3)
    her = _her(m, 10, 0.0, [a, b], False)
    ch = tg.Chooser()
    first = ch.choose(her, [_foe(a, exp=30.0, kill=0.3), _foe(b, kill=2.5)])
    assert first.next.m is b
    later = ch.choose(_her(m, 10, 0.0, [d], False), [_foe(d, exp=30.0, kill=2.0)])  # (both gone: another's come)
    assert later.primary.m is d and later.how == "gone"


def test_the_next_is_scored_from_where_she_will_stand_with_the_kill_time_gone(tree):
    """A monster walking towards where she'll be is where it'll be when the primary is dead."""
    m, _ = tree
    a, b = _mon(m, GREEN, 120.0, 10, track=1), _mon(m, HORNY, -300.0, 10, track=2)
    her = _her(m, 10, 0.0, [a, b], False)
    primary = tg.terms_of(her, _foe(a, exp=30.0, kill=2.0))
    still = tg.after(her, primary, [_foe(a, exp=30.0, kill=2.0), _foe(b, kill=2.5)])
    coming = tg.after(her, primary, [_foe(a, exp=30.0, kill=2.0), _foe(b, kill=2.5, vx=60.0)])
    assert still.m is b and coming.m is b and coming.score >= still.score


# --- a respawn ----------------------------------------------------------------------------------------------------------------

def _targeting(m, exp=None, attacks=None):
    h = SimpleNamespace(attacks=attacks or SimpleNamespace(picks=False, chance=lambda m: None, reckon=lambda m: None,
                                                           kill_s=lambda m: 2.0),
                        aggro=SimpleNamespace(chasers={}), actions=SimpleNamespace(facing="right", attack_at=0.0), log=lambda text: None,
                        senses=SimpleNamespace(invulnerable_until=lambda: 0.0), ledger=SimpleNamespace(dead={}),
                        drops=DropStore())
    g = SimpleNamespace(h=h, model=m, kinds=None, danger=None, tour=SimpleNamespace(exp=exp or {HORNY: 41.0, GREEN: 30.0}),
                        _avg_exp=lambda: 30.0)
    return targeting.Targeting(g)


def test_a_monster_fading_in_beside_her_enters_the_choice_at_its_first_look(tree):
    """Its 'spawned' event (the frame's) to the choice of that look: no delay; with when it can be hit and when it can touch
    her from its alpha."""
    m, _ = tree
    tp = _targeting(m)
    newborn = _mon(m, HORNY, 120.0, 10, track=7, fading=0.1)
    s = SimpleNamespace(platform=m.platform(10), where=SimpleNamespace(x=0.0, y=m.platform(10).y_at(0.0)), t=50.0, monsters=[newborn],
                        tracks=[])
    pace = Pace(0.0)
    foes = tp.foes_of(s, [newborn], [newborn], pace)
    assert foes[0].born and foes[0].solid_s == pytest.approx((1.0 - 0.1) / targeting.FADE_PER_S)
    assert foes[0].hit_s == pytest.approx((tg.HIT_ALPHA - 0.1) / targeting.FADE_PER_S)


def test_a_newborn_that_beats_her_target_by_the_margin_changes_it_at_once_said_a_respawn(tree):
    m, _ = tree
    far = _mon(m, HORNY, 300.0, 10, track=1)
    born = _mon(m, GREEN, 150.0, 10, track=2, bar=None, fading=0.4)
    her = _her(m, 10, 0.0, [far], False)
    ch = tg.Chooser()
    assert ch.choose(her, [_foe(far, kill=3.0)]).primary.m is far
    her2 = _her(m, 10, 0.0, [far, born], False)
    c = ch.choose(her2, [_foe(far, kill=3.0), _foe(born, exp=30.0, kill=0.5, born=True, solid=0.5)])
    assert c.primary.m is born and c.how == "respawn" and c.why.startswith("respawn at 150")


def _track(m, id_, kind, x, pf, vx=0.0, fading=None, solid_at=None, state="live"):
    return SimpleNamespace(id=id_, kind=kind, platform=pf, x=x, y=m.platform(pf).y_at(x), mob=Mob(kind, 0, 0, 30, 30, 0.9, fading=fading),
                           bar=None, facing=0, pose="", vx=vx, state=state, fading=fading, solid_at=solid_at)


def _in_her_lock(m, primary_x=-250.0):
    """A choice made at the look after her key: a whole Horny Mushroom far on her left, she on 10 at 0 in her lock."""
    tp = _targeting(m)
    t = _mon(m, HORNY, primary_x, 10, track=1)
    tp.her = _her(m, 10, 0.0, [t], False, lead=0.5)
    tp.chooser.margin = tg.MARGIN
    tp.choice = tp.chooser.choose(tp.her, [_foe(t, kill=3.0)])
    tp._chose_at = time.monotonic()
    return tp


def test_while_her_lock_lasts_each_frame_is_read_a_newborn_that_beats_her_target_calls_the_attack_off(tree):
    """The frame's 'spawned' event, not drained yet (no look since her key): taken in on that frame, and it calls her attack
    off (the next pass decides at once). One that doesn't beat it by the margin changes nothing."""
    m, _ = tree
    tp = _in_her_lock(m)
    tp.h.attacks = SimpleNamespace(picks=False, chance=lambda m: None, reckon=lambda m: None, kill_s=lambda m: 0.5)
    born = _track(m, 7, GREEN, 120.0, 10, fading=0.2, solid_at=50.65)
    st = SimpleNamespace(t=50.0, tracks=[born])
    event = SimpleNamespace(kind="spawned", track=7, t=50.0, data={"platform": 10, "x": 120})
    assert tp._changes(st, [event], 0.4)
    assert tp.counts["lock: respawn"] == 1
    weak = _in_her_lock(m, primary_x=-100.0)
    weak.h.attacks = SimpleNamespace(picks=False, chance=lambda m: None, reckon=lambda m: None, kill_s=lambda m: 9.0)
    assert not weak._changes(SimpleNamespace(t=50.0, tracks=[born]), [event], 0.4)


def test_while_her_lock_lasts_a_monster_coming_at_her_calls_the_attack_off_unless_it_is_her_target(tree):
    m, _ = tree
    tp = _in_her_lock(m)
    coming = _track(m, 2, GREEN, 150.0, 10, vx=-100.0)
    assert tp._changes(SimpleNamespace(t=50.0, tracks=[coming]), [], 0.4)
    assert tp.counts["lock: coming"] == 1
    away = _track(m, 3, GREEN, 150.0, 10, vx=100.0)
    assert not tp._changes(SimpleNamespace(t=50.0, tracks=[away]), [], 0.4)
    target = _track(m, 1, HORNY, -150.0, 10, vx=100.0)
    assert not tp._changes(SimpleNamespace(t=50.0, tracks=[target]), [], 0.4)


def test_a_monster_arriving_after_she_has_finished_the_primary_does_not_call_her_attack_off(tree):
    """The kill's time from the look already holds her lock (Way.s): it's not added again."""
    m, _ = tree
    tp = _in_her_lock(m)
    c = tp.choice
    finish = c.primary.way.s + c.primary.foe.kill_s
    left = 0.4
    gap = 150.0 - tg.half_of(GREEN, tp.her.halves) - tg.HER_HALF
    arrives = finish + left / 2  # (after she's finished; before finish + left, the double count)
    assert not tp._changes(SimpleNamespace(t=50.0, tracks=[_track(m, 2, GREEN, 150.0, 10, vx=-gap / arrives)]), [], left)
    assert tp._changes(SimpleNamespace(t=50.0, tracks=[_track(m, 2, GREEN, 150.0, 10, vx=-gap / (finish * 0.8))]), [], left)


def test_the_watch_of_her_lock_waits_it_out_and_goes_when_nothing_changes(tree):
    m, _ = tree
    tp = _in_her_lock(m)
    now = time.monotonic()
    tp.h.actions = SimpleNamespace(lock_until=now + 0.05, facing="right")
    st = SimpleNamespace(t=now, tracks=[])
    tp.h.senses = SimpleNamespace(fresh_after=lambda: 0.0,
                                  feed=SimpleNamespace(newest=lambda after: st if st.t >= after else None, pending_events=lambda t: []))
    assert tp.watch_lock() is True
    assert time.monotonic() >= now + 0.05
    tp.h.actions.lock_until = time.monotonic() + 5.0
    st.tracks = [_track(m, 2, GREEN, 150.0, 10, vx=-100.0)]
    st.t = time.monotonic()
    assert tp.watch_lock() is False  # (at once: the next pass decides)


# --- a monster coming at her -----------------------------------------------------------------------------------------------------

@pytest.fixture
def coming_scene(tree):
    m, _ = tree
    target = _mon(m, HORNY, -150.0, 10, track=1)
    walking = _mon(m, GREEN, 150.0, 10, track=2)
    return m, target, walking


def test_a_monster_walking_at_her_while_she_attacks_another_takes_over_said_why(coming_scene):
    """It reaches her before she's done with the primary, her invulnerability over by then."""
    m, target, walking = coming_scene
    her = _her(m, 10, 0.0, [target, walking], False)
    ch = tg.Chooser()
    assert ch.choose(her, [_foe(target, kill=1.0)]).primary.m is target
    c = ch.choose(her, [_foe(target, kill=3.0), _foe(walking, exp=30.0, kill=3.0, vx=-100.0)])
    assert c.primary.m is walking and c.how == "coming" and c.why.startswith("coming at her: ")
    assert c.coming.foe.m is walking and 0.0 < c.coming.arrives < 3.0


def test_the_same_when_she_is_already_attacking_it_is_no_change(coming_scene):
    m, target, walking = coming_scene
    her = _her(m, 10, 0.0, [walking], False)
    ch = tg.Chooser()
    assert ch.choose(her, [_foe(walking, exp=30.0, kill=3.0)]).primary.m is walking
    c = ch.choose(her, [_foe(walking, exp=30.0, kill=3.0, vx=-100.0)])
    assert c.primary.m is walking and c.how == "" and not c.why


def test_one_that_arrives_inside_her_invulnerability_can_t_knock_her_and_changes_nothing(coming_scene):
    m, target, walking = coming_scene
    her = _her(m, 10, 0.0, [target, walking], False, inv=1.45)
    ch = tg.Chooser()
    assert ch.choose(her, [_foe(target, kill=1.0)]).primary.m is target
    c = ch.choose(her, [_foe(target, kill=1.0), _foe(walking, exp=30.0, kill=3.0, vx=-200.0)])
    assert c.coming is None and c.primary.m is target


def test_one_that_arrives_after_she_has_finished_changes_nothing(coming_scene):
    m, target, walking = coming_scene
    her = _her(m, 10, 0.0, [target, walking], False)
    ch = tg.Chooser()
    ch.choose(her, [_foe(target, kill=0.5)])
    c = ch.choose(her, [_foe(target, kill=0.5), _foe(walking, exp=30.0, kill=3.0, vx=-20.0)])
    assert c.coming is None and c.primary.m is target


def test_a_monster_fading_in_on_her_counts_from_when_it_is_solid(tree):
    m, _ = tree
    born = _mon(m, GREEN, 20.0, 10, track=2, fading=0.2)
    t = _mon(m, HORNY, -150.0, 10, track=1)
    her = _her(m, 10, 0.0, [t, born], False)
    assert tg.coming(her, [_foe(t, kill=3.0), _foe(born, solid=0.65, born=True)], 3.0).arrives == pytest.approx(0.65)
    assert tg.coming(her, [_foe(t, kill=0.3), _foe(born, solid=0.65, born=True)], 0.3) is None


# --- the plan's place -----------------------------------------------------------------------------------------------------------

def test_the_places_for_a_monster_are_the_nearest_each_side_the_nearer_first(tree):
    """Her attack from the ground lands from either side of it: one place a side, the nearest of its stretch, checked; the
    walk's the nearer, and a walk her speed from the stats panel takes (read at each ask: physics.set_speed)."""
    m, _ = tree
    far = _mon(m, GREEN, 300.0, 10, track=1)
    her = _her(m, 10, -300.0, [far], False)
    opts = tg.places(her, far)
    assert opts and opts == sorted(opts, key=lambda g: abs(g - her.x)) and len(opts) <= 2
    way = tg.soonest(her, _foe(far, exp=30.0))
    assert way.how == "walk" and way.place == opts[0]


def test_her_walk_is_read_when_asked_not_when_imported(tree, monkeypatch):
    """Her SPEED from the stats panel (physics.set_speed at a hunter's start) counts in the time to get at a monster."""
    m, _ = tree
    far = _mon(m, GREEN, 300.0, 10, track=1)
    her = _her(m, 10, -300.0, [far], False)
    was = tg.soonest(her, _foe(far, exp=30.0))
    monkeypatch.setattr(physics, "WALK_SPEED", physics.WALK_SPEED * 2.0)
    now = tg.soonest(her, _foe(far, exp=30.0))
    assert now.place == was.place and now.s < was.s


def _look_at(m, x=-300.0, t=50.0):
    p = m.platform(10)
    return SimpleNamespace(platform=p, where=SimpleNamespace(x=x, y=p.y_at(x)), t=t, monsters=[], tracks=[], looked=[], in_view=lambda pid, x: False)


def test_the_walk_step_takes_the_nearer_clear_place_with_the_look_s_tracks_taken_once(tree):
    m, _ = tree
    tp = _targeting(m)
    far = _mon(m, GREEN, 0.0, 10, track=1)
    tp.her = _her(m, 10, -300.0, [far], False)
    tp.choice = tp.chooser.choose(tp.her, [_foe(far, exp=30.0)])
    asked = []
    tp.g.nav = SimpleNamespace(first_clear=lambda s, x0, goals: asked.append(list(goals)) or goals[-1])
    s = _look_at(m)
    s.platform = tp.her.platform
    side, units, place = tp.walk_step(s)
    assert len(asked) == 1 and place == asked[0][-1] and side == ("right" if place > -300.0 else "left") and units <= tg.WALK_STEP
    tp.g.nav = SimpleNamespace(first_clear=lambda s, x0, goals: None)  # (none clear: the nearer all the same)
    assert tp.walk_step(s)[2] == asked[0][0]


def test_a_rule_only_asking_what_the_stance_would_do_sets_nothing_aside(tree):
    """A wait for an answer alone (Hunt._pass's close quarters, no approach) doesn't set the target aside; acting on it does,
    said in the log."""
    m, _ = tree
    tp = _targeting(m)
    said = []
    tp.h.log = said.append
    t = _mon(m, GREEN, 100.0, 10, track=4)
    tp.her = _her(m, 10, 0.0, [t], False)
    tp.choice = tp.chooser.choose(tp.her, [_foe(t, exp=30.0)])
    s = _look_at(m, x=0.0)
    assert tp.plan(s, ("wait",), False, False) == ("wait",) and not tp.no_way and not said
    assert tp.plan(s, ("wait",), False, True) == ("wait",) and tp.no_way == {4: s.t + targeting.NO_WAY_S}
    assert len(said) == 1 and "#4" in said[0] and "set aside" in said[0]


def test_a_spot_trip_ends_on_reaching_its_platform_with_the_planner_only():
    """No walk on to a precomputed x: at the -80 top she takes a target at once. With the switch off, master's walk."""
    g = SimpleNamespace(tp=object())
    assert Grind._ends_on_arrival(g, SimpleNamespace(kind="spot")) and not Grind._ends_on_arrival(g, SimpleNamespace(kind="pile"))
    assert not Grind._ends_on_arrival(SimpleNamespace(tp=None), SimpleNamespace(kind="spot"))


# --- her drops --------------------------------------------------------------------------------------------------------------------

def _pile(age, platform=10, x=100.0, hers=True, **kw):
    return Pile(x, 0.0, platform, time.monotonic() - age, hers, **kw)


def _choice_tp(m, how="walk", reach_s=2.0, kill=2.0, coming=False, primary=True):
    tp = _targeting(m)
    if primary:
        mon = _mon(m, GREEN, 200.0, 10, track=1)
        foe = _foe(mon, exp=30.0, kill=kill)
        t = tg.Terms(foe, tg.Way(reach_s, how), 1.0)
        tp.choice = tg.Choice(t, None, "coming at her: 0.5 s" if coming else "", (t,), tg.Coming(foe, 0.5) if coming else None,
                              "coming" if coming else "")
    return tp


def test_her_drops_beside_her_go_first_with_no_monster_in_reach_and_she_takes_them_whatever_they_add_to_her_way(tree):
    """Nothing in reach: a pile of hers nearer than the target's place goes first, and on her platform it may add what it
    takes (she doesn't leave it with her drops on it); the sweep has it where master's allowances wouldn't."""
    m, _ = tree
    tp = _choice_tp(m, how="walk", reach_s=2.0)
    p = _pile(5.0, status="seen")
    assert tp.pile_first(p, SimpleNamespace(leave=1.0), time.monotonic())
    s = SimpleNamespace(platform=m.platform(10))
    assert tp.loot_allow(p, s, time.monotonic()) >= targeting.LOOT_ALL
    far_walk = sweep.Spot("far", 10, 900.0, 30.0, 0.5, allow=tp.loot_allow(p, s, time.monotonic()))
    it = sweep.plan([far_walk], (10, 0.0), (7, 0.0), lambda a, ax, b, bx: abs(ax - bx) / 150.0 + (a != b) * 5.0, most=None)
    assert it.keys() == ("far",)
    capped = sweep.Spot("far", 10, 900.0, 30.0, 0.5, allow=1.0)
    assert not sweep.plan([capped], (10, 0.0), (7, 0.0), lambda a, ax, b, bx: abs(ax - bx) / 150.0 + (a != b) * 5.0).stops


def test_her_drops_wait_for_the_target_in_reach_while_they_can(tree):
    m, _ = tree
    tp = _choice_tp(m, how="now", reach_s=0.0)
    assert not tp.pile_first(_pile(5.0), SimpleNamespace(leave=1.0), time.monotonic())


def test_drops_with_little_life_left_go_before_the_next_target(tree):
    m, _ = tree
    tp = _choice_tp(m, how="now", reach_s=0.0, kill=3.0)
    old = _pile(EXPIRE_S - targeting.LOOT_DUE_S - 2.0)
    assert tp.pile_first(old, SimpleNamespace(leave=1.0), time.monotonic())


def test_a_monster_coming_at_her_goes_before_any_drops(tree):
    m, _ = tree
    tp = _choice_tp(m, how="now", reach_s=0.0, coming=True)
    assert not tp.pile_first(_pile(EXPIRE_S - 5.0), SimpleNamespace(leave=1.0), time.monotonic())


def test_a_far_pile_waits_for_her_way_then_goes_whatever_it_takes_near_the_end_of_its_life(tree):
    """On another platform: LOOT_FRESH s on her way, whatever it takes near its end."""
    m, _ = tree
    tp = _targeting(m)
    s = SimpleNamespace(platform=m.platform(10))
    now = time.monotonic()
    assert tp.loot_allow(_pile(5.0, platform=7), s, now) == pytest.approx(targeting.LOOT_FRESH)
    assert tp.loot_allow(_pile(1.0), s, now) >= targeting.LOOT_ALL  # (on her platform, its drops still landing)
    assert tp.loot_allow(_pile(9.0), s, now) == pytest.approx(targeting.LOOT_FRESH)  # (long landed, no look has seen any)
    assert tp.loot_allow(_pile(OWN_S + 50.0, platform=7), s, now) == pytest.approx(targeting.LOOT_FRESH)
    assert tp.loot_allow(_pile(EXPIRE_S - 5.0, platform=7), s, now) >= targeting.LOOT_ALL


LANDED = DROP_LANDS + LANDED_S  # (s after a kill from which a search of its place can say it dropped nothing)


def _strip(x0=0.0, x1=200.0, at=None, sure=True, mesos_only=False):
    return SimpleNamespace(platform=10, x0=x0, x1=x1, at=time.monotonic() if at is None else at, sure=sure, mesos_only=mesos_only)


def _stood(strips, t=None):
    """A look with her standing and these strips of drops."""
    return SimpleNamespace(strips=strips, looked=[(10, 0.0, 200.0)], in_view=lambda pid, x: pid == 10 and 0.0 <= x <= 200.0,
                           t=time.monotonic() if t is None else t, ground=(0.0, 100.0), geo=SimpleNamespace(fpx=lambda px: px))


def _grind_with(m):
    tp = _targeting(m)
    said = []
    return SimpleNamespace(tp=tp, h=tp.h, log=said.append), said


def test_a_kill_that_dropped_nothing_has_no_stand(tree):
    """At a kill's pile no look has seen: until its drops have landed she waits (no walk over its spread); then a search there
    sure of it with nothing in it says it's gone, said in the log and as an event, a later sighting reviving it. From their
    landing, not looting.SETTLE's: keeping each pile open to then cost her 20 EXP a minute. Off, master's."""
    m, _ = tree
    g, said = _grind_with(m)
    s = _stood([_strip()])
    landing = _pile(1.0)
    assert Grind._none_landed(g, s, landing) and landing.status == "expected"
    assert LANDED < looting.SETTLE
    landed = _pile((LANDED + looting.SETTLE) / 2)
    with telemetry.captured() as recs:
        assert Grind._none_landed(g, s, landed) and landed.status == "gone" and landed.gone_at == s.t and landed.none_landed
    assert g.tp.counts["none landed"] == 1 and len(said) == 1 and "none landed" in said[0]
    assert [r for r in recs if r.get("kind") == "none_landed" and r["platform"] == 10 and r["x"] == 100]
    assert not Grind._none_landed(SimpleNamespace(tp=None), s, _pile(LANDED + 0.5))


def test_no_mark_on_a_search_that_cannot_say_so(tree):
    """A strip not sure of it, one for mesos alone, one from before its drops landed, or her in the air: no mark. She waits a
    little for the rotation's next (SEARCH_S), then walks its spread as before."""
    m, _ = tree
    g, _ = _grind_with(m)
    now = time.monotonic()
    for s in (_stood([_strip(sure=False)]), _stood([_strip(mesos_only=True)]), _stood([_strip(at=now - 5.0)]), _stood([_strip(x0=300.0, x1=500.0)])):
        p = _pile(LANDED + 0.5)
        if s.strips[0].x0 == 300.0:
            s.in_view = lambda pid, x: False
            s.looked = []
        assert not g.h.drops.searched_empty(s, p)
        assert Grind._none_landed(g, s, p) and p.status == "expected"  # (the next look may have it)
        late = _pile(LANDED + grind_mod.SEARCH_S + 0.5)
        assert not Grind._none_landed(g, s, late) and late.status == "expected"  # (her walk over its spread)
    air = _stood([_strip()])
    air.ground = None
    assert not g.h.drops.searched_empty(air, _pile(LANDED + 0.5))
    own = _stood([])  # (no strips: the look's own search)
    assert g.h.drops.searched_empty(own, _pile(LANDED + 0.5))


def _seen_at(x, platform=10):
    return SimpleNamespace(item=SimpleNamespace(kind="horny-mushroom-cap", coin=None), x=x, y=0.0, platform=platform, mesos=False)


def test_a_pile_seen_gone_comes_back_when_seen_however_late_with_the_planner_and_is_counted():
    """Off (REVIVE_S), a sighting past it makes a pile of another's; with the planner (revive_s: the whole life) it's hers,
    her age kept, the revive counted and an event for the first live check."""
    t0 = time.monotonic()
    late = REVIVE_S + 40.0
    for revive_s, hers in ((REVIVE_S, False), (EXPIRE_S, True)):
        store = DropStore(revive_s=revive_s)
        p = Pile(100.0, 0.0, 10, t0, True, status="gone", gone_at=t0 + 5.0, none_landed=True)
        store.piles.append(p)
        with telemetry.captured() as recs:
            store._sighting(t0 + 5.0 + late, [_seen_at(105.0)], lambda pid, x: True, True)
        if hers:
            assert p.status == "seen" and p.gone_at is None and store.revived == 1 and len(store.piles) == 1
            assert [r for r in recs if r.get("kind") == "pile_revived" and r["none_landed"] and r["platform"] == 10]
        else:
            assert p.status == "gone" and store.revived == 0 and not store.piles[-1].hers


def test_the_set_piece_counts_a_knock_off_10_to_anywhere_once_a_climb():
    """A fall off 10 costs her the climb wherever it lands (13 only the longest of them); a second knock after the same climb,
    one past 6 s, or one off another platform, adds none."""
    tr = {"result": {"climbs": [10.0, 20.0, 30.0], "knocks": [(12.0, 10, 12), (13.0, 10, 13), (27.0, 10, 13), (31.0, 8, 13)]}}
    assert scenarios._knock_offs(tr) == 1
    tr["result"]["knocks"].append((33.0, 10, 9))
    assert scenarios._knock_offs(tr) == 2


def _said_for(m, c, tp, x=0.0):
    said = []
    tp.h.log = said.append
    tp.choice = c
    tp.her = _her(m, 10, x, [], False)
    tp._say(_look_at(m, x=x), ("throw", "right"))
    return said


def test_the_log_says_what_is_known_of_the_monster_it_left(tree):
    """Dead, set aside by the plan, or only not in this look (a look misses one now and then): the line tells them apart, with
    her distance and its score; no bar is "no HP bar", not unhurt."""
    m, _ = tree
    a, b = _mon(m, GREEN, 150.0, 10, track=1), _mon(m, GREEN, 170.0, 10, track=2)
    for known, word in (("dead", "#1 killed"), ("aside", "#1 set aside"), ("", "#1 not in this look")):
        tp = _targeting(m)
        her = _her(m, 10, 0.0, [a, b], False)
        tp.chooser.choose(her, [_foe(a, exp=30.0), _foe(b, exp=30.0)])
        if known == "dead":
            tp.h.ledger.dead[1] = 50.0
        if known == "aside":
            tp.no_way[1] = 1e9
        c = tp.chooser.choose(_her(m, 10, 0.0, [b], False), [_foe(b, exp=30.0)])
        said = _said_for(m, c, tp)
        assert len(said) == 1 and word in said[0], said
        assert "at +170" in said[0] and "no HP bar" in said[0] and "score " in said[0] and "unhurt" not in said[0]


def test_a_new_next_alone_says_nothing(tree):
    """One line when her target or her plan changes: not for the next changing under them."""
    m, _ = tree
    a, b, d = _mon(m, GREEN, 150.0, 10, track=1), _mon(m, GREEN, 170.0, 10, track=2), _mon(m, GREEN, -170.0, 10, track=3)
    tp = _targeting(m)
    her = _her(m, 10, 0.0, [a, b, d], False)
    said = _said_for(m, tp.chooser.choose(her, [_foe(a, exp=30.0), _foe(b, exp=30.0)]), tp)
    assert len(said) == 1
    c = tp.chooser.choose(her, [_foe(a, exp=30.0), _foe(d, exp=30.0)])
    assert c.next.m is d
    tp.choice = c
    tp._say(_look_at(m, x=0.0), ("throw", "right"))
    assert len(said) == 1


def test_a_monster_standing_on_a_pile_is_simply_her_target(tree):
    """Seen on her platform by the pile, it guards it (the sweep leaves the pile out for this look) and it's a foe like any:
    her target. One on another platform by a pile there, or one the look missed, doesn't keep her from going."""
    m, _ = tree
    on_pile = _mon(m, GREEN, 100.0, 10, track=1)
    g = SimpleNamespace(tp=_targeting(m))
    here = SimpleNamespace(platform=m.platform(10))
    pile = _pile(5.0)
    assert Grind._guards_it(g, here, pile, on_pile, True)
    assert not Grind._guards_it(g, here, pile, on_pile, False)
    assert not Grind._guards_it(g, SimpleNamespace(platform=m.platform(7)), pile, on_pile, True)
    assert Grind._guards_it(SimpleNamespace(tp=None), SimpleNamespace(platform=m.platform(7)), pile, on_pile, False)
    c = tg.Chooser().choose(_her(m, 10, 0.0, [on_pile], False), [_foe(on_pile, exp=30.0, kill=2.0)])
    assert c.primary.m is on_pile


# --- her backpack --------------------------------------------------------------------------------------------------------------------

def _loot(kind, x=100.0, platform=10):
    return SimpleNamespace(item=SimpleNamespace(kind=kind, coin=None), x=x, y=0.0, platform=platform,
                           mesos=kind == looting.MESOS)


def _looter(messages):
    h = SimpleNamespace(log=lambda text: None, senses=SimpleNamespace(eyes=SimpleNamespace(messages=lambda: messages[0])))
    return Looter(h)


FULL_LINE = "You can't get anymore items."


def test_backpack_full_one_regular_push_naming_the_tab_none_again_until_it_clears_and_mesos_still_taken():
    sent = []
    messages = [FULL_LINE]
    lo = _looter(messages)
    lo.push = notify.BackpackPush(lambda title, text: sent.append((title, text)), clock=lambda fmt: "12:00:00")
    cap = _loot("horny-mushroom-cap")
    lo.cant_pick([cap])
    assert len(sent) == 1 and "Etc" in sent[0][1] and "her" not in sent[0][1].split() and sent[0][0] == "Backpack full"
    assert "Etc" in lo.full and lo.can_take(looting.MESOS) and not lo.can_take("green-mushroom-cap")
    lo.cant_pick([_loot("green-mushroom-cap")])
    lo.full.clear()  # (its TAB_RETRY_S: a guess that it may have room isn't the game saying so)
    lo.cant_pick([cap])
    assert len(sent) == 1
    lo.full.clear()  # (tried again: one that didn't fit earned after the refusal, the game showing it has room)
    messages[0] = FULL_LINE + "\nGreen Mushroom Cap x1 earned. (Etc)"
    lo.refused.clear()
    lo.cant_pick([_loot("green-mushroom-cap")])
    assert "Etc" not in lo.push.sent and "Etc" not in lo.full
    messages[0] = FULL_LINE
    lo.cant_pick([_loot("topaz-ore")])
    assert len(sent) == 2 and "Etc" in lo.full
    assert lo.can_take(looting.MESOS)


def test_a_new_kind_picked_up_clears_the_backpack_push_too():
    sent = []
    lo = _looter([FULL_LINE])
    lo.push = notify.BackpackPush(lambda title, text: sent.append(title))
    lo.cant_pick([_loot("horny-mushroom-cap")])
    assert lo.push.sent == {"Etc"}
    gone = SimpleNamespace(status="gone", last_seen=None, items=())
    lo._stands.append((gone, {"topaz-ore"}, time.monotonic()))
    lo.on_look(SimpleNamespace())
    assert not lo.push.sent and "topaz-ore" in lo.held


def test_items_staying_under_her_without_the_games_word_mark_the_tab_and_send_seems_full_once():
    """The refusal's line lives seconds and fights come between her stands: with no word from the game the guess pushes too
    (the tab's items aren't stood on again until TAB_RETRY_S, so nothing else would tell)."""
    sent = []
    lo = _looter([""])
    lo.push = notify.BackpackPush(lambda title, text: sent.append(title))
    for k in ("horny-mushroom-cap", "green-mushroom-cap", "topaz-ore"):
        lo.cant_pick([_loot(k)])
    assert "Etc" in lo.full and sent == ["Backpack seems full"]
    lo.cant_pick([_loot("blue-mushroom-cap")])
    assert sent == ["Backpack seems full"]


def test_the_backpack_push_is_bark_at_level_active_with_no_ring_no_text_no_critical_sound(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "send_bark", lambda cfg, title, text, level, call, sound="alarm": calls.append((title, level, call, sound)))
    mailed = []
    monkeypatch.setattr(notify, "mail", lambda *a, **k: mailed.append(a))
    bark_file = tmp_path / "bark.json"
    bark_file.write_text('{"key": "SECRETKEY"}', encoding="utf-8")
    p = notify.BackpackPush.from_lab(bark_path=bark_file)
    assert p.full("Etc") and not p.full("Etc")
    for _ in range(100):
        if calls:
            break
        time.sleep(0.01)
    assert calls == [("Backpack full", "active", False, None)] and not mailed
    url = notify.bark_url({"server": "https://x", "key": "K"}, "t", "b", "active", False, None)
    assert "level=active" in url and "sound" not in url and "volume" not in url and "call" not in url


def test_the_backpack_push_sends_nothing_under_no_push_and_never_logs_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv(notify.NO_PUSH_ENV, "1")
    calls = []
    monkeypatch.setattr(notify, "send_bark", lambda *a, **k: calls.append(a))
    assert notify.BackpackPush.from_lab().full("Etc")
    time.sleep(0.02)
    assert not calls
    said = []

    def boom(cfg, title, text, level, call, sound="alarm"):
        raise OSError(f"no way to https://x/{cfg['key']}/y")
    monkeypatch.setattr(notify, "send_bark", boom)
    t = notify.bark({"server": "https://x", "key": "SECRETKEY", "ring": False}, "t", "b", "active", log=said.append, sound=None)
    t.join(2.0)
    assert said and "SECRETKEY" not in said[0] and "<key>" in said[0]


def test_the_switch_is_on_by_default_since_its_gate_and_off_with_0(monkeypatch):
    monkeypatch.delenv("GAMELAB_TARGET", raising=False)
    assert targeting.on()
    monkeypatch.setenv("GAMELAB_TARGET", "1")
    assert targeting.on()
    monkeypatch.setenv("GAMELAB_TARGET", "0")
    assert not targeting.on()
    with targeting.forced_on():
        assert targeting.on()
    assert not targeting.on()
    monkeypatch.setenv("GAMELAB_TARGET_SETPIECE", "0")
    monkeypatch.setenv("GAMELAB_TARGET", "1")
    with targeting.forced_on():
        assert not targeting.on()
    assert targeting.on()
