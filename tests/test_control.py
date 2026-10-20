"""Her actions as key plans (control.actions), without the player."""

from types import SimpleNamespace

from maplehelper.engine import RIGHT

from gamelab.control.actions import STAR_DELAY_MS, Actions
from gamelab.motion.flow import JUMP_LAG
from gamelab.motion.physics import LAGS, air_time


def actions(sent: list) -> Actions:
    key = lambda name: SimpleNamespace(name=name)
    a = Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"),
                                lucky_seven=None, pick_up=key("z"), hp_potion=None, chair=None))
    a.play = lambda events, name: sent.append((name, events))
    return a


def test_jump_back_turns_and_throws_just_before_landing():
    sent = []
    actions(sent).jump_back(RIGHT)  # the monster is on her right: she jumps left
    (_, events), = sent
    assert [e[0] for e in events] == sorted(e[0] for e in events)  # the player refuses unsorted plans
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    lift = at("alt", "down") + 1000 * JUMP_LAG
    land = lift + 1000 * air_time(0)
    apex = lift + 1000 * air_time(0) / 2
    assert at("left", "down") < at("alt", "down")              # moving away when she jumps
    assert lift < at("right", "down") + 1000 * LAGS.on         # the game sees the turn once she's off the ground
    assert at("right", "down") < at("f", "down") < apex        # turned in the air, then the key
    leaves = at("f", "down") + STAR_DELAY_MS                    # ... the star leaves ~520 ms later
    assert 0 <= land - leaves <= 40                            # just before landing
    assert at("left", "up") <= at("right", "down")             # no longer holding away when she lands


def test_the_attack_keys_come_as_long_after_the_jump_keys_as_attack_reach_reckons():
    # attack.reach works out what an attack from a jump hits from these gaps (motion.physics): the key
    # plans must keep to them.
    from gamelab.motion.physics import JUMP_ATTACK_MS, JUMP_BACK_ATTACK_MS
    for method, gap in (("jump_throw", JUMP_ATTACK_MS), ("jump_back", JUMP_BACK_ATTACK_MS)):
        sent = []
        a = actions(sent)
        a.facing = RIGHT
        getattr(a, method)(RIGHT)
        (_, events), = sent
        at = lambda k: next(t for t, name, h_ in events if name == k and h_ == "down")
        assert at("f") - at("alt") == gap, method


def test_a_throw_turns_her_only_when_she_faces_the_other_way():
    sent = []
    a = actions(sent)
    a.throw(RIGHT)
    a.lock_until = 0.0
    a.throw(RIGHT)
    turned, again = sent[0][1], sent[1][1]
    assert [e[1] for e in turned] == ["right", "right", "f", "f"] and [e[1] for e in again] == ["f", "f"]
    assert a.facing == RIGHT


def test_no_potion_key_means_no_potion():
    sent = []
    assert actions(sent).drink() is False and not sent


def test_a_plan_with_a_jump_waits_out_the_attack_lock_and_one_without_doesnt(monkeypatch):
    """A jump (or attack) pressed inside the 730 ms attack lock is ignored, so every plan that presses one
    is held back just enough: a jump 200 ms into the plan, with 0.5 s of lock left, waits 0.3 s."""
    from gamelab.control import actions as actions_mod
    now, slept = [100.0], []
    monkeypatch.setattr(actions_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(actions_mod.time, "sleep", lambda s: slept.append(round(s, 3)))
    a = actions([])
    a.lock_until = 100.5
    a.hold_back([[0, "right", "down"], [200, "alt", "down"], [240, "alt", "up"]])
    a.hold_back([[0, "right", "down"], [400, "right", "up"]])        # no jump: nothing to wait for
    a.hold_back([[0, "right", "down"], [600, "alt", "down"]])        # its jump comes after the lock anyway
    assert slept == [0.3]


def test_the_movers_walks_press_pick_up_as_every_walk_does():
    """Walking to an x (after a trip, before a jump up or a drop) presses pick-up every 150 ms and on
    through the slide, like Actions.walk: anything she passes over is looted."""
    from gamelab.control.actions import WALK_PICK_AFTER_MS, WALK_PICK_EVERY_MS
    from gamelab.control.mover import Mover
    events, _ = Mover(None, actions([]), None).walk_events(150.0, tolerance=8)
    picks = [e[0] for e in events if e[1] == "z" and e[2] == "down"]
    release = next(e[0] for e in events if e[1] == "right" and e[2] == "up")
    assert picks[0] == 0 and picks[1] == WALK_PICK_EVERY_MS and release < picks[-1] < release + WALK_PICK_AFTER_MS


def test_no_jump_in_her_key_plans_is_keyed_in_the_kicks_blind_spot():
    """A jump keyed 55-117 ms after the arrow, from standing, finds her just below full speed: no kick, and
    she leaves at 119-149 units/s (three of the five jump-backs recorded at 60 fps, keyed 80 ms after the
    arrow, left at 121-127)."""
    from gamelab.control import actions as actions_mod
    from gamelab.control.mover import MIN_RUN
    from gamelab.motion import body
    from gamelab.motion.physics import LAGS, SHOE, run_ms, sure_jump_ms
    sent = []
    actions(sent).jump_back(RIGHT)
    (_, events), = sent
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    gap = at("alt", "down") - at("left", "down")
    assert gap == actions_mod.JB_RUN_MS and sure_jump_ms(gap) == gap and body.short_takeoff_chance(gap, SHOE, LAGS) <= 0.01
    # The mover's shortest run-up is keyed as its length says (not put off), outside the blind spot.
    assert run_ms(MIN_RUN) == round(MIN_RUN / 0.149 + 46) and body.jump_window_ok(run_ms(MIN_RUN), SHOE, LAGS)


def test_a_jump_throw_that_turns_her_jumps_once_the_turns_tap_has_stopped_walking_her():
    from maplehelper.game import MOVEMENT
    from gamelab.control.actions import JUMP_THROW_TURN_MS
    from gamelab.motion import body
    from gamelab.motion.physics import SHOE
    sent = []
    a = actions(sent)
    a.facing = "left"
    a.jump_throw(RIGHT)
    (_, events), = sent
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    assert at("right", "down") == 0 and at("right", "up") == MOVEMENT.tap_turn_ms
    lift = at("alt", "down") / 1000 + JUMP_LAG
    assert at("alt", "down") == JUMP_THROW_TURN_MS and lift > body.hold_time(MOVEMENT.tap_turn_ms, SHOE, LAGS) + 0.03
    # Facing the right way already, no tap, and the jump goes at once.
    sent.clear()
    a.lock_until = 0.0
    a.jump_throw(RIGHT)
    assert [e[1] for e in sent[0][1]] == ["alt", "alt", "f", "f"] and sent[0][1][0][0] == 0


def test_the_movers_drop_steps_off_a_spot_where_it_wouldnt_go_through(monkeypatch):
    """Within 20 units of the move's spot does for the mover's drop, unless a drop there wouldn't go through
    (on a foothold that forbids it, or by a rope's top): then it walks to the nearest spot that would."""
    from gamelab.control import mover as mover_mod
    from gamelab.localization import mapdata
    from gamelab.motion.route import Move
    from test_navigation import by_y, tree
    monkeypatch.setattr(mover_mod.time, "sleep", lambda s: None)
    t = tree()
    t["foothold"]["1"]["1"]["2"]["forbidFallDown"] = 1  # A's 100..200
    m = mapdata.parse(1, t, "Test Map", {})
    a, c = by_y(m, 300, 100), by_y(m, 500, 400)
    sent = []
    acts = actions(sent)
    acts.keys.down = SimpleNamespace(name="down")
    mover = mover_mod.Mover(None, acts, m)
    for here, walks in ((104.0, {"left"}), (90.0, set())):
        sent.clear()
        mover._drop(Move("drop", a.id, c.id, 92.0, 92.0), SimpleNamespace(where=SimpleNamespace(x=here)))
        (_, events), = sent
        assert {name for _, name, how in events if name in ("left", "right") and how == "down"} == walks, here
