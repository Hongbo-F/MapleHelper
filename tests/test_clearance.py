"""Clearance kept ahead of time, and jump throws on a slope (from 2026-10-27): where a monster will be when her next
attack starts (attack.stance.Pace, behavior.fight.pace_of), what each throw from where she stands surely takes then
(stance._from_here: from a jump by the star's search alone, reach.AIR_MODEL, on the slope as it is: reach.air_sure),
the step to where her next throw will (stance._clear_step), and the simulator's world as fitted to the live jump
throws (sim.world: AIR_PICK 'search', Noise.air_short, air_under and air_turn_lost, the Lucky Seven key's claw)."""

from types import SimpleNamespace

import pytest

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import reach
from gamelab.attack import stance as st
from gamelab.behavior import fight
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import Monster
from gamelab.motion import physics
from gamelab.perception.mobs import Mob
from gamelab.sim import world as world_mod
from gamelab.sim.world import Noise

from test_sim import model, plat, still, tap, world  # noqa: F401 (model: the module's fixture)

FLAT = Platform(0, [Segment(0, 500, 1000, 500)])
# 400 wide, rising 40 over its middle 200 (Tree Dungeon's platforms rise 11-36 along one, 0.38-0.58 a unit at the
# steepest: the slopes' diagnosis)
SLOPE = Platform(5, [Segment(0, 540, 100, 540), Segment(100, 540, 300, 500), Segment(300, 500, 400, 500)])
TOPS, HALVES = {"green-mushroom": 51.0}, {"green-mushroom": 28.0}
GREEN = "green-mushroom"


def mon(x, y=500.0, platform=0, kind=GREEN):
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, y, platform)


def walking(vx):
    return st.Pace(0.0, lambda m: vx, {GREEN: 100.0})


def stance(x, ms, p=FLAT, facing=RIGHT, pace=st.NO_PACE, **kw):
    return st.stance(x, p.y_at(x), p, ms, False, facing, tops=TOPS, halves=HALVES, pace=pace, **kw)


def test_who_is_too_close_goes_by_where_it_will_be_when_her_attack_starts():
    # A Green 150 off walking at her at 95 units/s, her keys free in 0.6 s (her lock and the look's age): when a throw
    # from the ground would start (0.67 s on) it's 86 off, inside its claw's edge (105): too close; standing, not.
    green = mon(650)
    assert not st.too_close(green, 500, HALVES)
    assert st.too_close(green, 500, HALVES, pace=st.Pace(0.6, lambda m: -95.0))
    assert not st.too_close(green, 500, HALVES, pace=st.Pace(0.6, lambda m: 0.0))
    # Walking away it's further off by then; its walk not known, it walks at her at its kind's speed (WALK_AT).
    assert not st.too_close(mon(600), 500, HALVES, pace=st.Pace(0.6, lambda m: 95.0))
    assert st.too_close(green, 500, HALVES, pace=st.Pace(0.6, lambda m: None, {GREEN: 100.0}))
    # So the stance: from the ground it would be a claw swipe, but a jump in place keyed 40 ms after the jump key
    # surely takes it (83 off at that attack's start): the jump, not the throw it was decided as live.
    assert stance(500, [green], pace=st.Pace(0.6, lambda m: -95.0)) == ("jump_throw", RIGHT)
    assert stance(500, [green], pace=st.Pace(0.6, lambda m: 0.0)) == ("throw", RIGHT)


def test_up_close_the_throw_is_one_whose_attack_surely_takes_it():
    # A Green standing on her level: past its claw's edge (105) a throw from the ground; nearer, a jump in place,
    # whose attack 40 ms into the jump picks it from 44 (54 with CLEAR_MARGIN); nearer still, a step to where the next
    # one will (to 55 off: from 20 off, 35 back).
    assert stance(500, [mon(615)]) == ("throw", RIGHT)
    assert stance(500, [mon(560)]) == ("jump_throw", RIGHT)
    assert stance(500, [mon(554)]) == ("jump_throw", RIGHT) and stance(500, [mon(553)])[:2] == ("back", LEFT)
    assert stance(500, [mon(520)]) == ("back", LEFT, 35)
    # Facing away: an air turn's attack (120 ms into the jump, her feet 52-56 up) picks one on her level only from 144,
    # so she taps the arrow on the ground first and jumps keyed 40 ms after the jump key (JumpPlan air_turn False).
    tap_first = ("jump_throw", RIGHT, st.JumpPlan(physics.JUMP_ATTACK_MS, 0, False, "tap first"))
    assert stance(500, [mon(580)], facing=LEFT) == tap_first
    # Cornered (at her platform's end, 50 kept with monsters about): no step; within CLAW_ONLY (15: on her) the claw from
    # the ground (a swipe, no MP), beyond it the jump in place all the same (live 16% of such casts killed one 20-30
    # ahead, 37% 30-40).
    assert stance(60, [mon(70)]) == ("throw", RIGHT)
    assert stance(60, [mon(80)]) == ("jump_throw", RIGHT)


def test_against_one_walking_at_her_no_step_clears_it():
    # She walks at 149 units/s, it at 95: a step gains 54 units for each 149, less what it walks while she starts and
    # stops. Point-blank and walking at her, no step within STEP_AWAY makes her next throw sure: a jump back (it buys
    # room) where there's room, else the jump in place.
    green = mon(530)
    assert stance(500, [green], pace=walking(-95.0)) == ("jump_back", RIGHT)
    assert stance(500, [green], pace=walking(0.0)) == ("back", LEFT, 25)
    assert st._clear_step(500, 500, green, LEFT, 400, 51, 28, FLAT, [green], HALVES, walking(-95.0)) is None


def test_on_a_slope_a_jump_throw_at_one_lower_down_is_keyed_so_its_attack_picks_it():
    # Downhill: her at 260 (y 508), a Green 70 off at 190 (y 522, 14 lower). Keyed 40 ms after the jump key her
    # feet are 21-26 up: its top 51 over its own feet is under the first band's window then (12 over hers), and the
    # second band takes it only from 85 ahead. So not that jump: a step back uphill, to where it does.
    y_her, y_it = SLOPE.y_at(260), SLOPE.y_at(190)
    assert y_it - y_her == pytest.approx(14.0)
    assert not reach.air_sure(70 - st.CLEAR_MARGIN, y_it - y_her, 51, 28, physics.JUMP_ATTACK_MS)
    act = stance(260, [mon(190, y_it, 5)], p=SLOPE, facing=LEFT)
    assert act[:2] == ("back", RIGHT)
    # The same 70 off on level ground: the jump.
    assert stance(260, [mon(190)], facing=LEFT) == ("jump_throw", LEFT)
    # Uphill (she's the lower one), the jump in place picks it all the more.
    assert stance(190, [mon(260, y_her, 5)], p=SLOPE, facing=RIGHT) == ("jump_throw", RIGHT)


def test_pace_of_reads_a_monsters_walk_off_its_track():
    # A monster of the look matched to its track (kind, platform, within TRACK_MATCH): the track's speed once seen in
    # TRACK_SEEN frames, not known before; one the look didn't see (hidden, remembered) stays where it was seen; with no
    # feed, its pose.
    seen = mon(600)
    hidden = mon(700)
    track = SimpleNamespace(kind=GREEN, platform=0, x=605.0, vx=-90.0, seen=5, state="live")
    young = SimpleNamespace(kind=GREEN, platform=0, x=800.0, vx=-90.0, seen=1, state="live")
    s = SimpleNamespace(t=10.0, monsters=[seen, mon(800)], tracks=[track, young])
    h = SimpleNamespace(actions=SimpleNamespace(lock_until=0.0), model=None)
    pace = fight.pace_of(h, s)
    assert pace.walk(seen) == -90.0 and pace.walk(s.monsters[1]) is None and pace.walk(hidden) == 0.0
    s.tracks = []
    moving = Monster(Mob(GREEN, 0, 0, 30, 30, 0.9), 600, 500, 0, facing=1, pose="move")
    s.monsters = [moving]
    assert fight.pace_of(h, s).walk(moving) > 0


# --- the world ------------------------------------------------------------------------------------------------------

def _jump_throw(w, gap):
    w.invulnerable_ms = 10_000
    tap(w, 0.0, "alt", 0.04)
    tap(w, gap, "f", 0.04)
    w.advance(1.0)


def test_from_a_jump_the_world_picks_by_the_star_search_alone(model, monkeypatch):  # noqa: F811
    # AIR_PICK 'search': keyed 40 ms after the jump key, a pig 30 off (its far side 64, short of the first band) is
    # picked by nothing; 120 off, its top over the second band's window's floor, it takes the star.
    c = plat(model, 500, 400)
    assert world_mod.AIR_PICK == "search"
    w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 330.0)], facing="right"))
    _jump_throw(w, 0.04)
    assert w.monsters[0].hp == w.monsters[0].hp_max and any("no hit" in line for line in w.log)
    w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 420.0)], facing="right"))
    _jump_throw(w, 0.04)
    assert w.monsters[0].hp < w.monsters[0].hp_max
    # With Noise.air_short, one on her platform short of the first band takes it that often (drawn per attack).
    hits = 0
    for seed in range(40):
        w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 330.0)], facing="right", seed=seed,
                        noise=Noise(air_short=0.3)))
        _jump_throw(w, 0.04)
        hits += w.monsters[0].hp < w.monsters[0].hp_max
    assert 6 <= hits <= 20


def test_an_arrow_in_the_air_may_not_turn_her(model):  # noqa: F811
    # Noise.air_turn_lost: the arrow toward her back acted on in the air doesn't turn her (her frame shows it): every
    # time at 1.0, and the attack goes the way she faced.
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 420.0)], facing="left",
                    noise=Noise(air_turn_lost=1.0)))
    w.invulnerable_ms = 10_000
    tap(w, 0.0, "alt", 0.04)
    tap(w, 0.06, "right", 0.04)
    tap(w, 0.12, "f", 0.04)
    w.advance(1.0)
    assert w.air_turns_lost == 1 and w.facing == "left" and w.monsters[0].hp == w.monsters[0].hp_max


def test_the_lucky_seven_key_swipes_up_close_with_no_cast(model):  # noqa: F811
    # On the ground the claw swipes a monster in its stance's box whichever key: the Lucky Seven key's cast is given
    # back (its MP and stars: World._uncast), the swipe's damage the claw's (Her.claw); 120 off it's a cast.
    from dataclasses import replace
    from maplehelper.keys import parse as K
    from gamelab.sim.world import World
    from test_sim import KEYS
    keys = replace(KEYS, lucky_seven=K("g"))
    her = replace(world_mod.LEVEL_25, skill=world_mod.LUCKY_SEVEN, spread=True)
    c = plat(model, 500, 400)
    for dx, swiped in ((50.0, 1), (120.0, 0)):
        w = still(World(model, keys, (c.id, 300.0), monsters=[("Pig", c.id, 300.0 + dx)], facing="right",
                        character=her))
        w.invulnerable_ms = 10_000
        mp = w.mp
        tap(w, 0.0, "g", 0.04)
        w.advance(1.0)
        assert w.swiped_casts == swiped and w.casts == 1 - swiped, dx
        assert (w.mp >= mp) == bool(swiped) and w.monsters[0].hp < w.monsters[0].hp_max, dx
