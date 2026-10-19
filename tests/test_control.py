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
