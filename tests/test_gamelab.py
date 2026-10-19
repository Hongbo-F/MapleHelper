"""gamelab parts that don't need the game: plan validation, timelines and the image analysis."""

import numpy as np
import pytest

from maplehelper import keys
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import LEFT, RIGHT

from gamelab.camera import camera_x
from gamelab.player import allowed_keys, validate
from gamelab.timeline import chain, combo_events
from gamelab.track import airborne, find_name_tag

K = keys.parse
GAME_KEYS = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f"), lucky_seven=K("g"))
ALLOWED = allowed_keys(GAME_KEYS)


def plan(events=(), **capture):
    return {"events": [list(e) for e in events], "capture": {"fps": 30, "duration_ms": 1000, **capture}}


def test_allowed_keys_are_her_bound_actions_only():
    assert sorted(ALLOWED) == ["alt", "f", "g", "left", "right"]
    no_l7 = allowed_keys(GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f")))
    assert "g" not in no_l7


def test_valid_plan_passes():
    validate(plan([[0, "left", "down"], [40, "alt", "down"], [80, "alt", "up"], [80, "left", "up"]]), ALLOWED)


@pytest.mark.parametrize("bad, message", [
    (plan([[0, "r", "down"]]), "not one of her bound keys"),           # e.g. a potion key
    (plan([[50, "alt", "down"], [10, "alt", "up"]]), "sorted"),
    (plan([[0, "alt", "press"]]), "bad event kind"),
    (plan([[0, "alt"]]), "bad event"),
    (plan(fps=60, duration_ms=20_000), "900 frames"),
    (plan(fps=120), "fps"),
    (plan(crop=[0, 0, -5, 10]), "crop"),
    (plan([[0, "alt", "down"]] * 401), "at most"),
])
def test_bad_plans_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        validate(bad, ALLOWED)


def test_timeline_plays_the_real_combo_code():
    events, end = combo_events("jump_back_shot", RIGHT, 0, GAME_KEYS, Timing())
    t = Timing()
    keys_only = [(k, d) for _, k, d in events]
    assert keys_only == [("left", "up"), ("right", "up"),  # clean start
                         ("left", "down"), ("alt", "down"), ("alt", "up"), ("left", "up"),
                         ("right", "down"), ("g", "down"), ("g", "up"), ("right", "up")]
    assert events[3][0] == t.run_up  # jump after the run-up
    assert end == t.run_up + t.key_press + t.air_delay + t.turn_lead + t.key_press + t.throw_gap


def test_chain_starts_each_copy_when_the_previous_ends():
    events, starts, end = chain("jump_back_shot", LEFT, 3, GAME_KEYS, Timing())
    one = combo_events("jump_back_shot", LEFT, 0, GAME_KEYS, Timing())[1]
    assert starts == [0, one, 2 * one] and end == 3 * one
    assert [e[0] for e in events] == sorted(e[0] for e in events)


def test_airborne_spans_from_name_tag_heights_ignoring_glitches():
    # a jump with one mis-tracked frame in the middle, then a 1-frame blip on the ground
    ys = [100] * 5 + [90, 70, 60, 100, 50, 60, 70, 90] + [100] * 5 + [80] + [100] * 5
    rows = [(i * 30.0, 50, y, 1.0) for i, y in enumerate(ys)]
    assert airborne(rows) == [(150.0, 390.0)]


def test_camera_x_follows_a_scrolling_platform():
    rng = np.random.default_rng(1)
    world = (rng.random((40, 600)) * 255).astype(np.uint8)
    frames, rows = [], []
    for shift in range(0, 30, 3):  # camera moves right 3 px per frame
        f = np.zeros((200, 300, 3), np.uint8)
        f[110:150] = world[:, shift:shift + 300, None]
        frames.append(f)
        rows.append((0.0, 150, 100, 1.0))  # name tag resting at y=100, platform just below
    cam = camera_x(frames, rows)
    assert cam[-1] == pytest.approx(27, abs=1)


def test_find_name_tag_prefers_the_one_near_the_middle():
    frame = np.full((400, 600, 3), (200, 120, 150), np.uint8)  # pinkish sky
    for cx, cy in [(300, 230), (60, 380)]:  # her tag near the middle, another near a corner
        frame[cy - 9:cy + 9, cx - 40:cx + 40] = 20                     # dark box
        frame[cy - 5:cy + 5, cx - 32:cx + 32:3] = 255                  # white "text" strokes
    assert find_name_tag(frame) == pytest.approx((300, 230), abs=3)
