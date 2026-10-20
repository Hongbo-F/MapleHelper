"""Her actions as key plans (control.actions), without the player."""

from types import SimpleNamespace

from maplehelper.engine import RIGHT

from gamelab.control.actions import STAR_DELAY_MS, Actions
from gamelab.motion.flow import JUMP_LAG
from gamelab.motion.physics import air_time


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
    assert lift < at("right", "down") < at("f", "down") < apex  # turned in the air, then the key
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
