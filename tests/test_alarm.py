"""Lie Detector auto-stop: spotting it, stopping the player, and refusing to run until cleared."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.control import alarm, client
from gamelab.perception import lie_detector
from gamelab.perception.lie_detector import LieDetectorSpotter
from gamelab.__main__ import LIE_DETECTOR_EXIT, main
from gamelab.control.player import LIE_DETECTOR_ERROR, Player, Watchdog, _sleep_until

FIXTURES = Path(__file__).parent / "fixtures"


def img(name):
    return np.asarray(Image.open(FIXTURES / name).convert("RGB"))


SCENE = img("game_scene.png")            # daffodils and other yellow scenery
WINDOW = img("lie_detector_window.png")  # the real Lie Detector countdown window (top part)
NEAR_MISS = img("near_miss.png")         # the key bindings window's orange CANCEL button


def scene_with_window():
    frame = np.zeros((700, 1100, 3), np.uint8)
    frame[:SCENE.shape[0], :SCENE.shape[1]] = SCENE
    frame[300:300 + WINDOW.shape[0], 600:600 + WINDOW.shape[1]] = WINDOW
    return frame


@pytest.fixture
def alarm_file(tmp_path, monkeypatch):
    from gamelab.control import player
    path = tmp_path / "ALARM.json"
    for mod in (alarm, client, player):
        monkeypatch.setattr(mod, "ALARM", path)
    return path


def test_spots_the_lie_detector_anywhere_on_screen():
    spotter = LieDetectorSpotter()
    assert spotter.seen(scene_with_window())
    assert spotter.score(WINDOW) > 0.95


def test_spots_the_mouse_mini_game_and_the_truth_booth():
    # The mini-game was only captured at half size: enlarge it back to the game's size.
    half = img("lie_detector_minigame_half.png")
    mini_game = np.asarray(Image.fromarray(half).resize((half.shape[1] * 2, half.shape[0] * 2), Image.BILINEAR))
    spotter = LieDetectorSpotter()
    assert spotter.scores(mini_game)["mini-game"] >= 1
    assert spotter.seen(mini_game)
    assert spotter.scores(img("truth_booth_timer.png"))["truth booth"] >= 1


SEEN_IN_GAME = sorted((FIXTURES / "lie_detector").glob("*.png"))  # <date>_<time>_<form>.png, see its README
FORMS = {"countdown": "countdown", "mini_game": "mini-game", "truth_booth": "truth booth"}


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_spots_every_lie_detector_seen_in_game(png):
    form = FORMS[png.stem.split("_", 2)[2]]
    assert LieDetectorSpotter().scores(img(png))[form] >= 1


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_spots_every_lie_detector_seen_in_game_at_the_game_s_own_size_too(png):
    # Since 2026-10-23 the game's window is 1366x768, pixel for pixel: the 1920x1080 screens shrunk to it.
    form = FORMS[png.stem.split("_", 2)[2]]
    native = np.asarray(Image.fromarray(img(png)).resize((1366, 768), Image.BOX))
    assert LieDetectorSpotter().scores(native)[form] >= 1


def test_the_lie_detectors_seen_in_game_are_all_there():
    assert len(SEEN_IN_GAME) >= 6


@pytest.mark.parametrize("frame", [SCENE, NEAR_MISS, np.zeros((540, 960, 3), np.uint8)])
def test_ignores_yellow_scenery_and_other_windows(frame):
    assert not LieDetectorSpotter().seen(frame)


def numpy_yellow(rgb):
    """lie_detector.yellow as it was: numpy's tests on int16 copies."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r > 180) & (g > 140) & (b < 110) & (r - b > 110)).astype(np.float32)


def test_the_title_yellow_is_the_same_as_numpys_on_every_colour():
    g, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
    for r0 in range(0, 256, 16):
        picture = np.empty((16 * 256, 256, 3), np.uint8)
        picture[..., 0] = np.repeat(np.arange(r0, r0 + 16, dtype=np.uint8), 256)[:, None]
        picture[..., 1], picture[..., 2] = np.tile(g, (16, 1)), np.tile(b, (16, 1))
        new = lie_detector.yellow(picture)
        assert new.dtype == np.float32 and np.array_equal(new, numpy_yellow(picture))


def test_spotter_scores_are_the_same_as_with_numpys_yellow(monkeypatch):
    half = img("lie_detector_minigame_half.png")
    mini_game = np.asarray(Image.fromarray(half).resize((half.shape[1] * 2, half.shape[0] * 2), Image.BILINEAR))
    frames = [scene_with_window(), WINDOW, mini_game, img("truth_booth_timer.png"), SCENE, NEAR_MISS,
              np.zeros((540, 960, 3), np.uint8)]
    new = [LieDetectorSpotter().scores(f) for f in frames]
    monkeypatch.setattr(lie_detector, "yellow", numpy_yellow)
    old = LieDetectorSpotter()
    assert new == [old.scores(f) for f in frames]


def test_watchdog_grabs_the_game_as_an_rgb_picture_of_its_own(monkeypatch):
    from gamelab.control import player as player_mod
    rng = np.random.default_rng(0)
    bgra = rng.integers(0, 256, (1080, 1920, 4), np.uint8)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    on_top = {"game": True}
    monkeypatch.setattr(player_mod.window, "shows", lambda hwnd, area: on_top["game"])

    class Screen:
        def grab(self, box):
            return bgra

    dog = Watchdog("MapleStory", threading.Event(), spotter=object())
    frame = dog._grab(Screen())
    assert frame.flags.c_contiguous and np.array_equal(frame, bgra[:, :, 2::-1])
    # Another window over some of the game: what's on top is still handed over (checked for the Lie Detector,
    # which may show round it), said to be covered.
    on_top["game"] = False
    frame, why = dog._grab(Screen())
    assert why == "covered" and np.array_equal(frame, bgra[:, :, 2::-1])


def test_watchdog_says_why_it_has_no_picture_the_game_minimized_or_gone(monkeypatch):
    # Minimized, the game's client area isn't 0 x 0 (432x54 on 2026-10-24): Windows is asked.
    from gamelab.control import player as player_mod
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: True)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": -32000, "top": -32000, "width": 432,
                                                                        "height": 54})
    dog = Watchdog("MapleStory", threading.Event(), spotter=object())
    assert dog._grab(None) == "minimized"

    def no_game(title):
        raise RuntimeError("no window titled 'MapleStory'")

    monkeypatch.setattr(player_mod.window, "find_game", no_game)
    assert dog._grab(None) == "no window"


def test_watchdog_sets_the_alarm_first_then_reports_once(tmp_path):
    frames = iter([SCENE, SCENE, scene_with_window(), scene_with_window(), SCENE])
    flag, reports, said = threading.Event(), [], []
    dog = Watchdog("MapleStory", flag, grab=lambda: next(frames), on_alarm=lambda frame: reports.append(flag.is_set()),
                   folder=tmp_path, log=said.append)
    for t in range(5):
        dog.tick(t * 0.25)
    assert flag.is_set() and reports == [True]  # the alarm was already up when it reported, and once a cycle
    assert dog.recording is not None and dog.recording.grabs == 3  # and the watchdog goes on, recording it
    dog.recording.finish("stopped")
    assert dog.recording.join(timeout=10)


def test_watchdog_keeps_watching_while_all_is_well(tmp_path):
    flag = threading.Event()
    dog = Watchdog("MapleStory", flag, grab=lambda: SCENE, interval=0.01, on_alarm=lambda f: None, folder=tmp_path,
                   log=lambda m: None)
    dog.start()
    time.sleep(0.1)
    assert dog.is_alive() and not flag.is_set()
    dog.stopped.set()
    dog.join(timeout=2)


def test_player_refuses_plans_once_the_alarm_is_up():
    flag = threading.Event()
    flag.set()
    # Returns before looking for the game window or pressing anything.
    assert Player({}, "no such window", flag).run({"events": [[0, "f", "down"]]}) == {"error": LIE_DETECTOR_ERROR}


def test_waits_end_early_when_the_alarm_goes_off():
    flag = threading.Event()
    threading.Timer(0.05, flag.set).start()
    started = time.perf_counter()
    assert _sleep_until(time.perf_counter() + 5, flag) is False
    assert time.perf_counter() - started < 0.5
    assert _sleep_until(time.perf_counter() + 0.01) is True


def test_raised_alarm_blocks_gamelab_until_cleared(alarm_file, monkeypatch, tmp_path):
    from gamelab.control import lie_recorder
    monkeypatch.setattr(lie_recorder, "LIE_RECORDINGS", tmp_path / "lie_detector")  # (clear-alarm reads them)
    pages = []
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append(title))
    info = alarm.raise_alarm(scene_with_window())
    assert pages == ["MapleStory Lie Detector on screen"]
    assert json.loads(alarm_file.read_text())["reason"] == "lie_detector"
    assert Path(info["screenshot"]).exists()
    Path(info["screenshot"]).unlink()

    with pytest.raises(client.LieDetectorAlarm):
        client.submit([], name="x")
    assert main(["act", "100"]) == LIE_DETECTOR_EXIT  # refused before touching the game
    assert main(["map"]) == LIE_DETECTOR_EXIT         # its look would bring the game to the front
    assert main(["player", "--elevate"]) == LIE_DETECTOR_EXIT  # nor is the player started again

    assert main(["clear-alarm"]) == 0
    assert not alarm_file.exists()
    client.check_alarm()  # no longer raises


def test_a_plan_can_be_cut_short_and_lets_go_of_its_keys(tmp_path, monkeypatch):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    sent = []
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: sent.append((key.name, down)))
    p = Player({"f": parse("f")}, "Game")
    threading.Timer(0.1, lambda: (tmp_path / "plan1.cancel").touch()).start()
    started = time.perf_counter()
    result = p.run({"id": "plan1", "events": [[0, "f", "down"], [3000, "f", "up"]], "capture": {"lead_ms": 0}})
    assert time.perf_counter() - started < 1.0
    assert result["problems"] == ["cancelled"]
    assert sent == [("f", True), ("f", False)]           # pressed, then let go when cut short
    assert not (tmp_path / "plan1.cancel").exists()       # the signal is used up


def test_a_plan_can_last_past_its_last_key(tmp_path, monkeypatch):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: None)
    p = Player({"f": parse("f")}, "Game")
    started = time.perf_counter()
    p.run({"id": "p2", "events": [[0, "f", "down"], [10, "f", "up"]], "end_ms": 300, "capture": {"lead_ms": 0}})
    assert 0.28 <= time.perf_counter() - started < 0.6
    with pytest.raises(ValueError):
        player_mod.validate({"events": [[0, "f", "down"], [500, "f", "up"]], "end_ms": 100}, {"f": parse("f")})


# --- what still goes with the alarm up: her pet's food and her retreat (control.alarm.PLAN_KINDS) -------------------

def _keys(pet="u"):
    from maplehelper.config import GameKeys
    from maplehelper.keys import parse
    return GameKeys(left=parse("left"), right=parse("right"), jump=parse("alt"), attack=parse("ctrl"),
                    lucky_seven=parse("shift"), up=parse("up"), down=parse("down"), pick_up=parse("z"),
                    pet_food=parse(pet) if pet else None)


def _tap(key, t=0):
    return [[t, key, "down"], [t + 40, key, "up"]]


def test_with_the_alarm_up_a_pet_food_plan_is_one_tap_of_its_key_and_nothing_else():
    from gamelab.control.player import alarm_refusal
    keys = _keys()

    def food(events, **kw):
        return {"kind": "pet_food", "events": events, **kw}
    assert alarm_refusal(food(_tap("u")), keys, 30.0, [], 100.0) is None
    for events in (_tap("u") + _tap("u", 100), _tap("u") + _tap("left", 100), _tap("ctrl"), [[0, "u", "down"]],
                   [[0, "u", "down"], [500, "u", "up"]], [[0, "mouse", 5, 5]] + _tap("u"), _tap("lbutton"), []):
        assert alarm_refusal(food(events), keys, 30.0, [], 100.0) is not None, events
    assert "no mouse" in alarm_refusal(food([[0, "mouse", 5, 5]] + _tap("u")), keys, 30.0, [], 100.0)
    assert alarm_refusal(food(_tap("u"), capture={"fps": 30}), keys, 30.0, [], 100.0) is not None
    assert alarm_refusal(food(_tap("u"), window_client=[1366, 768]), keys, 30.0, [], 100.0) is not None
    assert alarm_refusal(food(_tap("u")), _keys(pet=None), 30.0, [], 100.0) == "no pet food key is bound"
    # A few a minute: the fourth within 60 s is refused, the next once the first is a minute old.
    assert "a minute at most" in alarm_refusal(food(_tap("u")), keys, 30.0, [50.0, 70.0, 90.0], 100.0)
    assert alarm_refusal(food(_tap("u")), keys, 30.0, [40.0, 70.0, 90.0], 100.0) is None
    # Any plan without a kind, or of another, is refused as before.
    assert alarm_refusal({"events": _tap("u")}, keys, 1.0, [], 100.0) is not None
    assert alarm_refusal({"kind": "throw", "events": _tap("u")}, keys, 1.0, [], 100.0) is not None


def test_with_the_alarm_up_a_plan_let_through_holds_the_player_no_longer_than_it_has_to():
    """Review, 2026-10-24: a pet food plan with a lead of 1.5 s was taken and tapped at 1.5 s, one with end_ms 2000
    held the player 2 s (and `validate` bounds a lead only by the frames recorded). Whole within TAP_MAX_MS, from its
    start: its lead, its tap, its end; a retreat's lead too (its moves end with its window)."""
    from gamelab.control.player import TAP_MAX_MS, alarm_refusal
    keys = _keys()

    def food(events=None, **kw):
        return {"kind": "pet_food", "events": events or _tap("u"), **kw}
    for ok in (food(), food(capture={"lead_ms": 0}), food(capture={"lead_ms": TAP_MAX_MS}), food(end_ms=TAP_MAX_MS),
               food(_tap("u", TAP_MAX_MS - 40))):
        assert alarm_refusal(ok, keys, 30.0, [], 100.0) is None, ok
    for late in (food(capture={"lead_ms": 1500}), food(capture={"lead_ms": -1}), food(capture={"lead_ms": "0"}),
                 food(end_ms=2000), food(_tap("u", 19_000))):
        assert alarm_refusal(late, keys, 30.0, [], 100.0) is not None, late
    run = [[0, "left", "down"], [1500, "left", "up"]]
    assert alarm_refusal({"kind": "retreat", "events": run, "end_ms": 3000}, keys, 1.0, [], 100.0) is None
    assert "starts within" in alarm_refusal({"kind": "retreat", "events": run, "capture": {"lead_ms": 5000}}, keys,
                                            1.0, [], 100.0)


def test_with_the_alarm_up_a_retreat_is_her_moves_alone_for_its_window_with_every_key_let_go():
    from gamelab.control import alarm as alarm_mod
    from gamelab.control.player import alarm_refusal
    keys = _keys()
    run = [[0, "left", "down"], [100, "alt", "down"], [140, "alt", "up"], [600, "left", "up"],
           [700, "up", "down"], [1500, "up", "up"], [1600, "down", "down"], [1700, "down", "up"]]

    def retreat(events):
        return {"kind": "retreat", "events": events}
    assert alarm_refusal(retreat(run), keys, 0.5, [], 100.0) is None
    assert alarm_refusal(retreat(run), keys, alarm_mod.RETREAT_WINDOW_S - 0.01, [], 100.0) is None
    assert "are over" in alarm_refusal(retreat(run), keys, alarm_mod.RETREAT_WINDOW_S + 0.01, [], 100.0)
    assert "isn't known" in alarm_refusal(retreat(run), keys, None, [], 100.0)
    for other in ("ctrl", "shift", "z", "u", "lbutton"):  # no attack, Lucky Seven, pick-up, pet food or click
        assert alarm_refusal(retreat(run + _tap(other, 1800)), keys, 1.0, [], 100.0) is not None, other
    assert "held at its end" in alarm_refusal(retreat(run[:-1]), keys, 1.0, [], 100.0)
    assert "no mouse" in alarm_refusal(retreat([[0, "mouse", 10, 10]] + run), keys, 1.0, [], 100.0)
    assert alarm_mod.RETREAT_WINDOW_S > alarm_mod.RETREAT_S  # her 10 s retreat fits


def test_the_alarm_keeps_when_the_lie_detector_was_first_seen():
    from gamelab.control.player import Alarm
    flag = Alarm()
    assert flag.at is None and not flag.is_set()
    before = time.monotonic()
    flag.set()
    first = flag.at
    assert flag.is_set() and before <= first <= time.monotonic()
    flag.set()  # seen again: the sighting it counts from stays the first
    assert flag.at == first


class _Game:
    """The player's way to the game faked: the window there and in front, the keys it sends kept."""

    def __init__(self, monkeypatch, tmp_path):
        from gamelab.control import player as player_mod
        self.sent, self.fronted = [], []
        monkeypatch.setattr(player_mod, "INBOX", tmp_path)
        monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
        monkeypatch.setattr(player_mod.window, "client_rect",
                            lambda hwnd: {"left": 0, "top": 0, "width": 1366, "height": 768})
        monkeypatch.setattr(player_mod.window, "bring_to_front", lambda title: self.fronted.append(title))
        monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
        monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: self.sent.append((key.name, down)))
        monkeypatch.setattr(player_mod.win32, "send_mouse_button", lambda name, down: self.sent.append((name, down)))


def _alarmed_player(seen_ago: float = 1.0):
    from gamelab.control.player import Alarm, allowed_keys
    flag, said = Alarm(), []
    flag.set()
    flag.at = time.monotonic() - seen_ago
    keys = _keys()
    return Player(allowed_keys(keys), "Game", flag, keys=keys, log=said.append), said


def _plan(pid, events, kind="pet_food"):
    return {"id": pid, "kind": kind, "events": events, "focus": True, "capture": {"lead_ms": 0}}


def test_with_the_alarm_up_the_player_feeds_her_pet_and_refuses_the_rest(monkeypatch, tmp_path):
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player()
    result = p.run(_plan("food1", _tap("u")))
    assert "error" not in result and game.sent == [("u", True), ("u", False)]
    assert game.fronted == []  # never brought forward with the alarm up: whoever is at the PC is answering it
    assert said == ["Lie Detector alarm up: her pet's food (food1), 1 in the last minute"]
    # The pet food key with anything else, a hunt's plan, a click: refused before anything is pressed.
    for pid, events, kind in (("food2", _tap("u") + _tap("ctrl", 100), "pet_food"), ("throw", _tap("ctrl"), None),
                              ("click", _tap("lbutton"), "pet_food")):
        assert p.run(_plan(pid, events, kind))["error"].startswith(LIE_DETECTOR_ERROR), pid
    assert p.run({"id": "throw", "events": _tap("ctrl")}) == {"error": LIE_DETECTOR_ERROR}  # as before
    assert game.sent == [("u", True), ("u", False)]
    assert [line.split(": ")[1] for line in said[1:]] == ["refused food2 (pet_food)", "refused click (pet_food)"]
    # A few a minute.
    for i in range(2, 4):
        assert "error" not in p.run(_plan(f"food{i + 1}", _tap("u")))
    assert "a minute at most" in p.run(_plan("food5", _tap("u")))["error"]


def test_with_the_alarm_up_a_pet_food_plan_that_would_hold_the_player_is_refused_at_once(monkeypatch, tmp_path):
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player()
    started = time.perf_counter()
    late = dict(_plan("food1", _tap("u")), capture={"lead_ms": 1500})
    held = dict(_plan("food2", _tap("u")), end_ms=2000)
    for plan in (late, held):
        assert p.run(plan)["error"].startswith(LIE_DETECTOR_ERROR)
    assert time.perf_counter() - started < 0.2 and game.sent == [] and p.fed == []


def test_with_the_alarm_up_her_retreat_is_played_until_its_window_ends_and_refused_after(monkeypatch, tmp_path):
    from gamelab.control import alarm as alarm_mod
    from gamelab.control.player import RETREAT_OVER
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player(seen_ago=alarm_mod.RETREAT_WINDOW_S - 0.2)  # 0.2 s of it left
    started = time.perf_counter()
    result = p.run(_plan("flow1", [[0, "left", "down"], [1500, "left", "up"]], "retreat"))
    assert 0.1 < time.perf_counter() - started < 0.6  # cut at the window's end, not played its 1.5 s
    assert "error" not in result and result["problems"] == [RETREAT_OVER]
    assert game.sent == [("left", True), ("left", False)]  # let go of when cut
    assert game.fronted == [] and "her retreat's plan flow1" in said[0]
    late = p.run(_plan("flow2", _tap("left"), "retreat"))
    assert late["error"].startswith(LIE_DETECTOR_ERROR) and "are over" in late["error"]
    assert game.sent == [("left", True), ("left", False)]


def test_with_the_alarm_file_up_but_unseen_by_the_player_its_kinds_go_by_the_alarm_s_rules(monkeypatch, tmp_path,
                                                                                           alarm_file):
    """Review, 2026-10-24: a plan with a kind gets past the client's check of the alarm file, and a player whose own
    watchdog hadn't seen the Lie Detector (`watch --lie-detector` raised it) played it as any: the game brought to the
    front while the test was answered, no limit a minute, no window, no key check."""
    from gamelab.control import alarm as alarm_mod
    from gamelab.control import player as player_mod
    from gamelab.control.player import Alarm, allowed_keys
    game = _Game(monkeypatch, tmp_path)
    keys, said = _keys(), []
    p = Player(allowed_keys(keys), "Game", Alarm(), keys=keys, log=said.append)  # its own alarm not set
    alarm_file.write_text(json.dumps({"reason": "lie_detector (seen by watch)", "at": time.time() - 2.0}),
                          encoding="utf-8")
    for i in range(3):
        assert "error" not in p.run(_plan(f"food{i}", _tap("u")))
    assert "a minute at most" in p.run(_plan("food3", _tap("u")))["error"]
    assert "one tap of 'u'" in p.run(_plan("food4", _tap("u") + _tap("ctrl", 100)))["error"]
    assert "error" not in p.run(_plan("flow", [[0, "left", "down"], [100, "left", "up"]], "retreat"))
    assert "isn't one of her moves' keys" in p.run(_plan("flow2", _tap("ctrl"), "retreat"))["error"]
    assert game.fronted == [] and game.sent == [("u", True), ("u", False)] * 3 + [("left", True), ("left", False)]
    assert said[0] == "Lie Detector alarm up (its file's; this player hasn't seen it): her pet's food (food0), 1 in " \
                      "the last minute"
    # Seen past the window by the file's `at`, or when isn't known: no retreat.
    alarm_file.write_text(json.dumps({"at": time.time() - alarm_mod.RETREAT_WINDOW_S - 1}), encoding="utf-8")
    assert "are over" in p.run(_plan("flow3", _tap("left"), "retreat"))["error"]
    alarm_file.write_text(json.dumps({"reason": "lie_detector"}), encoding="utf-8")
    assert "isn't known" in p.run(_plan("flow4", _tap("left"), "retreat"))["error"]
    # The file gone, a plan of a kind plays as any (the game brought forward if it asks).
    alarm_file.unlink()
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game" if game.fronted else "other")
    assert "error" not in p.run(_plan("food5", _tap("u")))
    assert game.fronted == ["Game"]


def test_a_plan_running_when_the_alarm_goes_off_is_stopped_whatever_its_kind(monkeypatch, tmp_path):
    from gamelab.control.player import Alarm, allowed_keys
    game = _Game(monkeypatch, tmp_path)
    flag, keys = Alarm(), _keys()
    p = Player(allowed_keys(keys), "Game", flag, keys=keys, log=lambda m: None)
    threading.Timer(0.1, flag.set).start()
    started = time.perf_counter()
    result = p.run(_plan("flow", [[0, "left", "down"], [3000, "left", "up"]], "retreat"))
    assert time.perf_counter() - started < 1.0 and result["error"] == LIE_DETECTOR_ERROR
    assert game.sent == [("left", True), ("left", False)]


def test_the_client_sends_her_pet_food_and_her_retreat_with_the_alarm_up_and_nothing_else(alarm_file):
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 21:02:00",
                                      "at": time.time() - 2.0}), encoding="utf-8")
    with pytest.raises(client.LieDetectorAlarm):
        client.send(_tap("ctrl"), name="throw")
    with pytest.raises(ValueError):
        client.send(_tap("u"), name="x", kind="throw")
    pid = client.send(_tap("u"), name="pet_food", kind="pet_food")
    plan = json.loads((client.INBOX / f"{pid}.json").read_text(encoding="utf-8"))
    assert plan["kind"] == "pet_food" and plan["events"] == _tap("u")
    assert client.alarm_up() and 1.9 < client.alarm_age() < 5.0
    # Waited for, it isn't withdrawn as a plan without a kind is (the player decides): no player, no answer.
    with pytest.raises(client.PlayerError, match="no result"):
        client.wait(pid, timeout=0.05)
    alarm_file.unlink()
    assert not client.alarm_up() and client.alarm_age() is None


def test_a_plan_refused_for_the_lie_detector_raises_its_alarm_even_before_its_file_is_written(alarm_file):
    for error in (LIE_DETECTOR_ERROR, f"{LIE_DETECTOR_ERROR}: 3 pet foods a minute at most with the alarm up"):
        pid = client.send([], name="x")
        client.OUTBOX.mkdir(parents=True, exist_ok=True)
        (client.OUTBOX / f"{pid}.json").write_text(json.dumps({"error": error}), encoding="utf-8")
        with pytest.raises(client.LieDetectorAlarm, match="Lie Detector on screen"):
            client.wait(pid, timeout=1)
