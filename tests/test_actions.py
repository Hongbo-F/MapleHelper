"""maplehelper.actions: the moves press the right keys for the measured lengths."""

import threading

import pytest

from maplehelper import actions, keys
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import LEFT, RIGHT, Ctx, _Silent
from maplehelper.game import MOVEMENT, RECOVERY

K = keys.parse
HER_KEYS = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f"), lucky_seven=K("g"),
                    up=K("up"), down=K("down"), pick_up=K("z"), chair=K("d"), sit=K("x"),
                    hp_potion=K("r"), mp_potion=K("t"))


class Clock(Ctx):
    def __init__(self, keys=HER_KEYS, facing=RIGHT):
        self.events, self.now = [], 0
        super().__init__(lambda k, d: self.events.append((self.now, k.name, "down" if d else "up")),
                         keys, Timing(), facing, False, threading.Event(), _Silent(), "")

    def wait(self, ms):
        self.now += ms

    def held(self, name):
        """How long `name` was held each time it was pressed."""
        downs, out = {}, []
        for t, k, kind in self.events:
            if k != name:
                continue
            if kind == "down":
                downs[k] = t
            else:
                out.append(t - downs.pop(k))
        return out


def test_walk_ms_matches_the_measured_walks():
    for hold, px in [(300, 75), (600, 146), (1000, 221)]:
        assert actions.walk_ms(px) == pytest.approx(hold, abs=60)
    assert actions.walk_ms(0) == 0
    assert actions.walk_ms(5) == MOVEMENT.tap_turn_ms  # too short to walk: just a tap


def test_walk_holds_the_arrow_then_lets_her_stop():
    c = Clock()
    actions.walk(c, LEFT, 200)
    assert c.held("left") == [actions.walk_ms(200)]
    assert c.now == actions.walk_ms(200) + MOVEMENT.stop_slide_ms


def test_running_jump_keeps_the_arrow_held_through_the_air():
    c = Clock()
    actions.jump(c, RIGHT)
    [(t_right, _, _), (t_jump, _, _)] = [e for e in c.events if e[2] == "down"]
    assert t_jump - t_right == actions.RUN_JUMP_AFTER_MS  # already moving at takeoff, and at full speed
    assert c.held("right")[0] >= MOVEMENT.reaction_ms + MOVEMENT.jump_airtime_ms


def test_ledge_reach_shrinks_with_height():
    assert actions.ledge_takeoff_px(0) > actions.ledge_takeoff_px(50) > actions.ledge_takeoff_px(90) > 0
    assert actions.ledge_takeoff_px(MOVEMENT.jump_height_px) == 0
    assert actions.ledge_takeoff_px(86) < 140  # the measured miss: 86 px ledge from ~140 px away


def test_jump_to_ledge_runs_and_jumps_without_stopping():
    c = Clock()
    actions.jump_to_ledge(c, LEFT, ledge_px=90, height_px=70, own_edge_px=60)
    assert len(c.held("left")) == 1  # one continuous hold: no stop before the jump
    jump_at = next(t for t, k, kind in c.events if k == "alt" and kind == "down")
    m = MOVEMENT
    takeoff_px = (jump_at - m.reaction_ms) * m.walk_px_per_ms + m.walk_px_per_ms * m.reaction_ms
    assert takeoff_px == pytest.approx(60 - actions.EDGE_MARGIN_PX, abs=1)  # just short of her own edge


def test_jump_to_ledge_never_keys_the_jump_before_she_has_reached_full_speed():
    # Taking off within ~21 px, it keyed the jump 100 ms after the arrow, where by gamelab's engine a
    # quarter of jumps leave short: too fast for the kick, short of full speed.
    c = Clock()
    actions.jump_to_ledge(c, LEFT, ledge_px=40, height_px=60, own_edge_px=25)
    arrow = next(t for t, k, kind in c.events if k == "left" and kind == "down")
    jump = next(t for t, k, kind in c.events if k == "alt" and kind == "down")
    assert jump - arrow == actions.RUN_JUMP_AFTER_MS >= 118


def test_jump_to_ledge_refuses_what_it_cant_make():
    with pytest.raises(ValueError, match="too high"):
        actions.jump_to_ledge(Clock(), LEFT, ledge_px=50, height_px=150, own_edge_px=40)
    with pytest.raises(ValueError, match="too far"):
        actions.jump_to_ledge(Clock(), LEFT, ledge_px=200, height_px=86, own_edge_px=20)


def test_climb_rope_jumps_with_up_held():
    c = Clock()
    actions.climb_rope(c, 300)
    order = [(k, kind) for _, k, kind in c.events]
    assert order[:3] == [("up", "down"), ("alt", "down"), ("alt", "up")]
    assert c.held("up")[0] >= 300 / MOVEMENT.climb_px_per_ms


def test_pick_up_presses_once_per_item():
    c = Clock()
    actions.pick_up(c, 3)
    assert len(c.held("z")) == 3


def test_rest_sits_long_enough_then_stands_up_facing_the_same_way():
    c = Clock(facing=LEFT)
    actions.rest(c, 120)
    assert actions.rest_ms(120) == 30_000  # 50 HP per 10 s tick -> 3 ticks
    assert c.held("d") and c.held("left") == [MOVEMENT.tap_turn_ms]  # tap the way she already faces
    assert actions.rest_ms(0) == 0
    assert RECOVERY.on_chair_hp_per_10s == 50


def test_attacks_respect_the_attack_lock():
    c = Clock()
    actions.lucky_seven(c, 2)
    downs = [t for t, k, kind in c.events if k == "g" and kind == "down"]
    assert downs[1] - downs[0] >= 730


def test_unbound_action_is_explained():
    c = Clock(keys=GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f")))
    with pytest.raises(ValueError, match="'pick_up' isn't bound"):
        actions.pick_up(c)


def test_every_named_action_runs():
    sample = {str: "right", float: 60.0, int: 1}
    for name, (fn, converters) in actions.ACTIONS.items():
        fn(Clock(), *[sample[conv] for conv in converters])
