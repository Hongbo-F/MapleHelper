"""What her attack key would pick, asked of the game's search as it goes down (reach.key_pick, Hunter._key_pick): a telemetry
'key_pick' record, nothing changed in her play (the user, 2026-11-05, on the no-hit recordings)."""

import time
from types import SimpleNamespace as NS

import pytest

from gamelab import telemetry
from gamelab.attack import reach
from gamelab.behavior import fight
from gamelab.behavior.hunter import KEY_PICK_S, Hunter
from gamelab.localization import mapdata
from gamelab.localization.monsters import Monster
from gamelab.motion import physics
from gamelab.perception.mobs import Mob
from gamelab.sim import no_hit_scenes as nh, scenarios

from test_jump_reach import LEDGE, build, pig, plat, seg

FLOOR = plat(LEDGE, 500, 100)
REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")


def _placed(*monsters):
    body = reach.body_of("pig")
    return [(m, m.x, m.y, *body) for m in monsters]


def test_the_search_says_the_claw_a_star_a_stopped_star_or_nothing():
    near, far = pig(740, 500, FLOOR.id), pig(900, 500, FLOOR.id)
    assert reach.key_pick(FLOOR, 700, 500, 1, _placed(near, far)) == ("claw", [near])  # (both stances reach it)
    assert reach.key_pick(FLOOR, 700, 500, 1, _placed(far)) == ("star", [far])
    assert reach.key_pick(FLOOR, 700, 500, -1, _placed(near, far)) == ("nothing", [])  # (behind her)
    assert reach.key_pick(FLOOR, 700, 500, 1, []) == ("nothing", [])
    # In the air no claw: under the search's first band it's picked by nothing, whatever's past it
    assert reach.key_pick(FLOOR, 700, 475, 1, _placed(pig(720, 500, FLOOR.id)), ground=False) == ("nothing", [])
    assert reach.key_pick(FLOOR, 700, 475, 1, _placed(near, far), ground=False)[0] == "star"
    # Her group's wall at x 600 stops the line to one past it, and no other takes it
    walled = build({"1": [seg(1, 0, 500, 600, 500, 0, 2), seg(2, 600, 500, 600, 250, 1, 0)], "2": [seg(3, 610, 470, 1000, 470)]})
    floor, block = plat(walled, 500, 300), plat(walled, 470, 700)
    m = pig(700, 470, block.id)
    assert reach.key_pick(floor, 550, 500, 1, _placed(m)) == ("stopped", [m])


def _track(id_, x, state="live", killed=False, last_seen=0.0, platform=FLOOR.id, facing=0, pose=""):
    return NS(id=id_, kind="pig", state=state, killed=killed, x=x, y=500.0, platform=platform, bar=None, facing=facing,
              pose=pose, last_seen=last_seen, mob=Mob("pig", 0, 0, 30, 30, 0.9))


def _seen(x, track, facing=0, pose=""):
    return Monster(Mob("pig", 0, 0, 30, 30, 0.9), x, 500.0, FLOOR.id, None, facing, pose, track)


def _asked(monsters=(), tracks=(), where=(700.0, 500.0, FLOOR.id), key_is=(None, 0, 0, False), target=None, side="right",
           ms_on=100.0, st=True):
    """A key `ms_on` ms from the state's frame, as Hunter._key_pick takes it: its record."""
    asked = []
    now = time.monotonic()
    state = NS(t=now, where=where, monsters=list(monsters), tracks=list(tracks)) if st else None

    def newest(after):
        asked.append(after)
        return state

    when = now + ms_on / 1000
    me = NS(senses=NS(feed=NS(newest=newest)), actions=NS(key_is=key_is), model=LEDGE,
            attacks=NS(picks=False, thrown=(when, NS(track=target))))
    with telemetry.captured() as records:
        Hunter._key_pick(me, when, side)
    assert asked == [-float("inf")]  # (whatever its age: `_still` has nothing to say of a frame no newer than the look)
    (rec,) = records
    assert rec["topic"] == "key_pick" and rec["cost_ms"] >= 0.0
    return rec


def test_its_record_says_its_target_or_another_and_what_it_used():
    rec = _asked([_seen(900.0, 7), _seen(910.0, 8)], target=8)
    assert rec["answer"] == "star" and rec["on"] == "target" and {p[0] for p in rec["picks"]} == {7, 8}
    assert (rec["x"], rec["feet"], rec["platform"], rec["seen"], rec["known"]) == (700.0, 500.0, FLOOR.id, 2, 2)
    rec = _asked([_seen(900.0, 7)], target=8)
    assert rec["answer"] == "star" and rec["on"] == "another" and rec["picks"][0][:2] == [7, "pig"]
    rec = _asked(target=8)
    assert rec["answer"] == "nothing" and rec["on"] is None and rec["picks"] == []


def test_one_walking_is_asked_where_its_heading_has_it_at_the_pick():
    # 30 ahead of the claw's end, walking away at a pig's speed, or standing: where its body is at the pick says
    speed = fight.speeds(LEDGE)["pig"]
    at = 0.2 + KEY_PICK_S
    stands = _asked([_seen(800.0, 1)], ms_on=200.0)
    walks = _asked([_seen(800.0, 1, 1, "move")], ms_on=200.0)
    assert stands["picks"][0][2] == 100.0 and walks["picks"][0][2] == pytest.approx(100.0 + speed * at, abs=0.1)


def test_a_track_dead_by_its_own_record_is_out_and_a_live_one_the_frame_missed_is_in():
    gone = _asked([_seen(900.0, 7)], [_track(7, 900.0, "dying", True)], target=7)
    assert gone["answer"] == "nothing" and gone["known"] == 0
    other = _asked([_seen(900.0, 7), _seen(1000.0, 8)], [_track(7, 900.0, "dead", True), _track(8, 1000.0)], target=7)
    assert other["answer"] == "star" and other["on"] == "another"
    # Under her body in this frame, its track live: it's there. Hidden past its time, or dead, isn't
    now = time.monotonic()
    held = _asked([], [_track(9, 900.0, last_seen=now)], target=9)
    assert held["answer"] == "star" and held["on"] == "target" and (held["seen"], held["known"]) == (0, 1)
    assert _asked([], [_track(9, 900.0, "hidden", last_seen=now)], target=9)["answer"] == "nothing"


def test_her_jump_has_her_where_its_arc_does_and_no_claw():
    d = physics.jump_of("up").arc(0.04 + KEY_PICK_S)
    assert d[0] == 0.0 and d[1] == pytest.approx(physics.jump_rise(40 + KEY_PICK_S * 1000))
    rec = _asked([_seen(740.0, 7)], key_is=("up", 0, 40, False), target=7)
    assert rec["feet"] == pytest.approx(500.0 - d[1], abs=0.1) and rec["answer"] != "claw"
    forward = _asked([_seen(1000.0, 7)], key_is=("forward", 1, 40, False), target=7)
    assert forward["x"] == pytest.approx(700.0 + physics.jump_of("forward", 1).arc(0.04 + KEY_PICK_S)[0], abs=0.1)
    back = _asked([_seen(1000.0, 7)], key_is=("back", -1, 120, False), target=7)
    assert back["x"] < 700.0
    tapped = _asked([_seen(1000.0, 7)], key_is=(None, 0, 0, True), target=7, side="left")
    assert tapped["x"] == 694.0 and tapped["answer"] == "nothing"  # (the tap's walk; left, where nothing is)


def test_with_no_place_to_ask_from_it_says_so_and_whatever_goes_wrong_is_the_records():
    assert _asked(st=False)["answer"] is None
    assert _asked(where=None)["answer"] is None
    assert _asked(where=(700.0, 440.0, None))["answer"] is None
    rec = _asked(where=(700.0, 440.0, 999))  # (a platform of no map she's on: a state from before a map's change)
    assert rec["answer"] is None and "IndexError" in rec["error"]


def test_nothing_is_asked_with_no_recorder_or_of_the_magicians_casts():
    def never(after):
        raise AssertionError
    me = NS(senses=NS(feed=NS(newest=never)), attacks=NS(picks=False))
    Hunter._key_pick(me, 0.0, "left")
    me.attacks.picks = True
    with telemetry.captured() as records:
        Hunter._key_pick(me, 0.0, "left")
    assert not records


def test_every_kind_of_her_keys_tells_what_it_is():
    from test_choice import KEYS7, _look
    from test_fight import EMPTY_MAP, _Eyes
    h = Hunter(_Eyes([]), KEYS7, mapdata.parse(1, EMPTY_MAP, "Test", {}), attack="choice")
    h.attacks.on_look(_look([("green-mushroom", 250.0, None)]))
    h.attacks.touched = lambda s: True
    h.actions.play = lambda events, name, kind=None: None
    h.actions.wait_lock = lambda: None
    h.actions._lock_for_attack = lambda: True
    h.actions.still = None
    h.actions.facing = "right"
    assert h.actions.throw("right") and h.actions.key_is == (None, 0, 0, False)
    h.actions.facing = "left"
    assert h.actions.throw("right") and h.actions.key_is == (None, 0, 0, True)
    assert h.actions.jump_throw("right", 40) and h.actions.key_is == ("up", 0, 40, False)
    h.actions.facing = "left"
    assert h.actions.jump_throw("right", 40, air_turn=True) and h.actions.key_is[:2] == ("up", 0) and not h.actions.key_is[3]
    h.actions.facing = "left"
    assert h.actions.jump_throw("right", 40, air_turn=False) and h.actions.key_is == ("up", 0, 40, True)
    assert h.actions.jump_throw("left", 40, 1) and h.actions.key_is == ("forward", 1, 40, False)
    assert h.actions.jump_back("right", 120) and h.actions.key_is == ("back", -1, 120, False)


@REAL
def test_in_the_world_every_attack_has_its_key_asked_and_she_plays_as_she_did():
    trace, sim, out = nh.run_scene("walked_past_080032", 1)
    keys = [r for r in sim.records if r["topic"] == "key_pick"]
    attacks = [r for r in sim.records if r["topic"] == "sim_attack"]
    assert attacks and len(keys) >= len(attacks)
    assert all(min(abs(k["key"] - a["key"]) for k in keys) < 0.05 for a in attacks)
    assert all(k["answer"] is not None and k["age_ms"] < 200 for k in keys)


def test_the_no_hit_event_has_each_attacks_key_as_a_record_has_its_time(monkeypatch):
    from gamelab.control import alarm
    monkeypatch.setattr(alarm, "keep_recent_after", lambda *a: None)
    me, now = NS(log=lambda text: None, _no_hit_clips=0, _no_hit_kept=None), time.monotonic()
    wall = time.time()
    thrown = [(now - 3.0, 7), (now - 2.0, 7), (now - 1.0, None), (now - 0.5, 8)]
    with telemetry.captured() as records:
        Hunter._no_hit(me, NS(t=now, where=None, platform=None), thrown, {7, 8})
    keys = next(r for r in records if r.get("kind") == "no_hit")["keys"]
    assert keys == pytest.approx([wall - 3.0, wall - 2.0, wall - 1.0, wall - 0.5], abs=0.05)
