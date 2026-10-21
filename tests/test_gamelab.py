"""gamelab parts that don't need the game: plan validation, timelines and the image analysis."""

import numpy as np
import pytest

from maplehelper import keys
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import LEFT, RIGHT

from gamelab.analysis.scroll import camera_x
from gamelab.control.player import allowed_keys, validate
from gamelab.control.timeline import chain, combo_events
from gamelab.analysis.track import airborne, find_name_tag
from gamelab.control import client
import time

K = keys.parse
GAME_KEYS = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f"), lucky_seven=K("g"))
ALLOWED = allowed_keys(GAME_KEYS)


def plan(events=(), **capture):
    return {"events": [list(e) for e in events], "capture": {"fps": 30, "duration_ms": 1000, **capture}}


def test_allowed_keys_are_her_bound_actions_plus_exploring_keys():
    assert sorted(ALLOWED) == sorted(["alt", "f", "g", "left", "right", "\\", "esc"])
    no_l7 = allowed_keys(GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f")))
    assert "g" not in no_l7
    with_loot = allowed_keys(GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f"),
                                      up=K("up"), pick_up=K("z"), chair=K("d")))
    assert {"up", "z", "d"} <= set(with_loot)


def test_enter_is_never_allowed_even_if_bound():
    keys_with_enter = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("enter"))
    assert "enter" not in allowed_keys(keys_with_enter, extra=("enter",))


def test_valid_plan_passes():
    validate(plan([[0, "left", "down"], [40, "alt", "down"], [80, "alt", "up"], [80, "left", "up"],
                   [100, "mouse", 960, 540]]), ALLOWED)


@pytest.mark.parametrize("bad, message", [
    (plan([[0, "r", "down"]]), "not allowed"),           # e.g. a potion key
    (plan([[0, "enter", "down"]]), "not allowed"),       # would send the chat box
    (plan([[0, "mouse", -1, 5]]), "bad mouse position"),
    (plan([[0, "mouse", 5, 5, "click"]]), "bad event"),  # there is no clicking
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


def test_explore_steps_become_plan_events():
    from gamelab.__main__ import parse_steps
    events, end = parse_steps(["right*800", "alt", "500", "right+", "alt", "right-", "@960,540", "\\"])
    assert events == [
        [0, "right", "down"], [800, "right", "up"],
        [800, "alt", "down"], [840, "alt", "up"],
        [1340, "right", "down"],
        [1340, "alt", "down"], [1380, "alt", "up"],
        [1380, "right", "up"],
        [1380, "mouse", 960, 540],
        [1380, "\\", "down"], [1420, "\\", "up"],
    ]
    assert end == 1420


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


def test_a_result_file_still_locked_is_read_again(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    calls = []
    real = Path.read_text

    def flaky(self, *a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "Permission denied")  # Windows, just after the player's replace
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert client._read_result(path) == {"ok": True} and len(calls) == 3


def _game_frame(tag=None, size=(1080, 1920)):
    """A full-size frame of the game (a plain sky), with her name tag (a dark box with white strokes) centred at
    `tag` if given."""
    frame = np.full((*size, 3), (200, 120, 150), np.uint8)
    if tag is not None:
        cx, cy = tag
        frame[cy - 9:cy + 9, cx - 40:cx + 40] = 20
        frame[cy - 5:cy + 5, cx - 32:cx + 32:3] = 255
    return frame


def _calibration(crop):
    from gamelab.analysis import calibration, track
    half = _game_frame((1000, 600))[::2, ::2]
    return calibration.Calibration(list(crop), 2, track.make_template(half, (500, 300), scale=2))


def test_a_recording_is_cropped_around_her_name_tag_where_it_is_when_it_starts():
    # Calibrate's crop is around wherever she stood then: on 2026-10-21 a sweep's 16 `act --record` runs never had
    # her in theirs. The crop is taken around her name tag as the game shows it at the start (her tag at the
    # crop's 400, 300, as calibrate puts it), saying so when that isn't the stored one; where the tag isn't
    # found, the stored crop, with a warning.
    from gamelab.analysis import calibration
    stale = _calibration([0, 482, 800, 400])
    assert calibration.match_tag(_game_frame((1000, 600)), stale.template)[0] == pytest.approx((1000, 600), abs=2)
    crop, note = calibration.crop_on_her(stale, _game_frame((1000, 600)), 1920, 1080)
    assert crop == pytest.approx([600, 300, 800, 400], abs=2)
    assert note.startswith("her name tag is at ") and note.endswith("(the stored [0, 482, 800, 400] hasn't her)")
    crop, note = calibration.crop_on_her(stale, _game_frame((1800, 1000)), 1920, 1080)  # kept inside the client
    assert crop[0] == 1120 and crop[1] == 680
    crop, note = calibration.crop_on_her(stale, _game_frame(), 1920, 1080)
    assert crop == [0, 482, 800, 400] and "wasn't found" in note and "calibrate" in note
    # Her tag where calibrate put it in the stored crop, or within TAG_SLACK of it: that crop is kept, unsaid.
    fresh = _calibration(calibration.crop_around((1000, 600), 1920, 1080))
    for tag in ((1000, 600), (1030, 580)):
        assert calibration.crop_on_her(fresh, _game_frame(tag), 1920, 1080) == (fresh.crop, "")
    crop, note = calibration.crop_on_her(fresh, _game_frame((1100, 600)), 1920, 1080)
    assert crop == pytest.approx([700, 300, 800, 400], abs=2) and note.endswith("had her too)")


def test_act_record_records_the_crop_around_her_not_the_stored_one(monkeypatch, tmp_path, capsys):
    from gamelab import __main__ as cli
    from gamelab import window
    from gamelab.analysis import calibration, recordings
    frame, plans, recorded = _game_frame((1000, 600)), [], {}

    def submit(events=(), **capture):  # the player: one capture of the game as it is, keys or not
        plans.append(capture)
        crop = capture.get("crop")
        shot = frame if crop is None else frame[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]]
        return {"frames": [shot[::capture.get("scale", 1), ::capture.get("scale", 1)]],
                "client": {"width": 1920, "height": 1080}, "times": [0], "sent": [], "problems": []}

    def record(events, **kw):
        recorded.update(kw)
        return recordings.Recording(kw["name"], np.zeros((1, 200, 400, 3), np.uint8), np.zeros(1))
    monkeypatch.setattr(window, "bring_to_front", lambda *a, **k: True)
    monkeypatch.setattr(client, "submit", submit)
    monkeypatch.setattr(calibration, "load", lambda: _calibration([0, 482, 800, 400]))
    monkeypatch.setattr(recordings, "record", record)
    monkeypatch.setattr(recordings, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(cli, "LAB", tmp_path)
    args = cli.build_parser().parse_args(["act", "right+", "52", "alt", "700", "right-", "--record"])
    assert cli.cmd_act(args) == 0
    assert recorded["crop"] == pytest.approx([600, 300, 800, 400], abs=2) and recorded["scale"] == 2
    assert plans[0].get("crop") is None and plans[0]["scale"] == 1  # the first capture found her, full size
    assert "recording [" in capsys.readouterr().out
