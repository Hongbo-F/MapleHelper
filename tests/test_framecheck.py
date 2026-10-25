"""Looks that aren't of the game as the eyes expect it (perception.framecheck, localization.placecheck): refused and
said why, recovered from where that can be done here, and never acted on (localization.senses: LookLost, and a hunt
standing still until the game looks right again). Saved frames of Tree Dungeon (tests/fixtures/capture) and copies of
them shifted, shrunk and covered; a fake game window that moves, lags and changes size between looks."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.analysis import bench
from gamelab.localization import mapdata, placecheck
from gamelab.localization import senses as senses_module
from gamelab.localization.pose import Where
from gamelab.perception import framecheck
from gamelab.perception.framecheck import BadLook, LookLost, Problem
from gamelab.perception.minimap import MinimapPicture
from gamelab.perception.nametag import RECUT_AFTER, NameTag
from gamelab.screen import NATIVE, STRETCHED

FIXTURES = Path(__file__).parent / "fixtures"
TAG = FIXTURES / "native" / "tag_template_1366x768.npy"
TREE = 10002073


def frame(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


GOOD = frame(FIXTURES / "capture" / "tree_135315.png")
OFFSET = frame(FIXTURES / "capture" / "tree_164845_offset.png")
ROCKY = [frame(FIXTURES / "native" / n) for n in ("n28_000.png", "n28_008.png", "native_first.png")]


def shifted(full: np.ndarray, dx: int, dy: int, fill: int = 255) -> np.ndarray:
    """The frame as a capture offset from the client area shows it: the game's picture moved dx right, dy down."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(full, m, (full.shape[1], full.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(fill,) * 3)


def tree_model():
    try:
        return mapdata.load(TREE)
    except (OSError, KeyError):
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data")


def tree_picture():
    picture = mapdata.minimap_picture(TREE)
    if picture is None:
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    return picture


# --- the frame's own checks ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("i", range(4))
def test_good_frames_of_either_map_pass(i):
    assert framecheck.frame_problem(([GOOD] + ROCKY)[i], NATIVE) is None


def test_the_frame_of_16_48_45_is_offset_as_the_window_moved():
    """Saved while the user moved the game window: +26,+26 (the window's own title bar in the frame's top rows)."""
    p = framecheck.frame_problem(OFFSET, NATIVE)
    assert p.kind == "offset" and "(+26, +26)" in p.why


@pytest.mark.parametrize("dx,dy", [(1, 0), (-1, 0), (0, 1), (0, -1), (-5, 6), (26, 26), (-26, -26), (40, -40)])
def test_a_capture_offset_any_way_is_refused_and_says_by_how_much(dx, dy):
    p = framecheck.frame_problem(shifted(GOOD, dx, dy), NATIVE)
    assert p.kind == "offset" and f"({dx:+d}, {dy:+d})" in p.why


def test_a_capture_of_half_size_isnt_the_game():
    canvas = np.full_like(GOOD, 240)
    canvas[:384, :683] = cv2.resize(GOOD, (683, 384), interpolation=cv2.INTER_AREA)
    p = framecheck.frame_problem(canvas, NATIVE)
    assert p.kind == "offset" and "doesn't show the game" in p.why


def test_another_windows_strip_over_the_top_left_hides_the_minimap_title():
    covered = GOOD.copy()
    covered[:, :26] = covered[:26] = (250, 250, 250)
    assert framecheck.frame_problem(covered, NATIVE).kind == "minimap"


def test_the_minimap_closed_is_said():
    closed = GOOD.copy()
    closed[0:230, 0:270] = GOOD[300:530, 500:770]  # the play area where the minimap window was
    p = framecheck.frame_problem(closed, NATIVE)
    assert p.kind == "minimap" and "minimap" in p.why


def test_the_hud_covered_is_said_but_the_games_menu_over_part_of_it_isnt():
    covered = GOOD.copy()
    covered[690:712, 200:1100] = (90, 90, 90)
    assert framecheck.frame_problem(covered, NATIVE).kind == "hud"
    menu = GOOD.copy()
    menu[540:720, 925:1010] = (40, 120, 200)  # the game's menu, as on nm_menu_btn.png: 0.883 of the border left
    assert framecheck.hud_share(menu) > framecheck.HUD_SHARE and framecheck.frame_problem(menu, NATIVE) is None


def test_the_minimap_closed_and_an_end_of_the_hud_covered_isnt_taken_for_an_offset_capture():
    """The reviewer's frame: the minimap closed and a game window over the HUD border's right end was "offset by (+0,
    +0)", which reopened the capture and handed it to mss for nothing. Covering the left end instead, the border's
    other end tells where it is."""
    closed = GOOD.copy()
    closed[0:230, 0:270] = GOOD[300:530, 500:770]
    for x0, x1 in ((1010, 1090), (230, 300)):
        covered = closed.copy()
        covered[540:720, x0:x1] = (40, 120, 200)
        assert framecheck.find_offset(covered) == (0, 0)
        assert framecheck.frame_problem(covered, NATIVE).kind == "minimap"
    assert framecheck.find_offset(shifted(covered, 26, 26)) == (26, 26)


def test_frames_of_the_stretched_window_arent_checked():
    assert framecheck.frame_problem(np.zeros((1080, 1920, 3), np.uint8), STRETCHED) is None


def test_the_window_checks(monkeypatch):
    state = {"iconic": False, "front": "MapleStory", "shows": True, "on screen": True}
    grids = []
    monkeypatch.setattr(framecheck.window, "minimized", lambda hwnd: state["iconic"])
    monkeypatch.setattr(framecheck.window, "foreground_title", lambda: state["front"])
    monkeypatch.setattr(framecheck.window, "shows", lambda hwnd, area, grid=None: grids.append(grid) or state["shows"])
    monkeypatch.setattr(framecheck.window, "on_screen", lambda area: state["on screen"])
    area = {"left": 2446, "top": 146, "width": 1366, "height": 768}
    assert framecheck.window_problem(1, area, "MapleStory", NATIVE) is None and grids == [framecheck.COVER_GRID]
    resized = framecheck.window_problem(1, dict(area, width=1396, height=1952), "MapleStory", NATIVE)
    assert resized.kind == "size" and "1396x1952, not 1366x768" in resized.why and "isn't resized" in resized.why
    state["shows"] = False
    assert framecheck.window_problem(1, area, "MapleStory", NATIVE).why == "another window is over the game"
    state["front"] = "Claude"
    assert framecheck.window_problem(1, area, "MapleStory", NATIVE).why == "the game isn't in front"
    state["on screen"] = False  # off the desktop: not taken for a window over it
    off = framecheck.window_problem(1, area, "MapleStory", NATIVE)
    assert off.kind == "offscreen" and "partly off the screen" in off.why and "isn't moved" in off.why
    state["iconic"] = True
    assert framecheck.window_problem(1, area, "MapleStory", NATIVE).kind == "minimized"
    assert {"size", "minimized", "gone", "offscreen"} == set(framecheck.WAIT_KINDS)


REAL_SHOWS, REAL_ON_SCREEN = framecheck.window.shows, framecheck.window.on_screen  # (before conftest refuses them)


class User32:
    """Windows' answers about the screen: which top window is at a point (`over`: boxes of another window), and whether
    a point is on a monitor (`monitor`: the desktop's box)."""

    def __init__(self, game: int, over=(), monitor=(0, 0, 3840, 2160)):
        self.game, self.over, self.monitor, self.asked = game, list(over), monitor, []

    def WindowFromPoint(self, point):  # noqa: N802 (Windows' names)
        self.asked.append((point.x, point.y))
        return 2 if any(x0 <= point.x < x1 and y0 <= point.y < y1 for x0, y0, x1, y1 in self.over) else self.game

    def GetAncestor(self, hwnd, flags):  # noqa: N802
        return hwnd

    def MonitorFromPoint(self, point, flags):  # noqa: N802
        x0, y0, x1, y1 = self.monitor
        return 1 if x0 <= point.x < x1 and y0 <= point.y < y1 else None


def test_another_window_over_part_of_the_game_is_seen_at_the_grids_points(monkeypatch):
    """A box over x 300-660, y 60-360 of the client (the reviewer's) passes the five points of before (the middle and
    the corners), not the grid's; a game window hanging off the desktop is said, not taken for a window over it."""
    area = {"left": 2446, "top": 146, "width": 1366, "height": 768}
    box = (area["left"] + 300, area["top"] + 60, area["left"] + 660, area["top"] + 360)
    user32 = User32(1, over=[box])
    monkeypatch.setattr(framecheck.window, "user32", user32)
    assert REAL_SHOWS(1, area)  # the five points miss it
    assert not REAL_SHOWS(1, area, grid=framecheck.COVER_GRID)
    user32.over, user32.asked = [], []
    assert REAL_SHOWS(1, area, grid=framecheck.COVER_GRID) and len(user32.asked) == 35
    right, bottom = area["left"] + area["width"] - 9, area["top"] + area["height"] - 9  # (8 px in)
    corners = [(area["left"] + 8, area["top"] + 8), (right, bottom)]
    assert all(c in user32.asked for c in corners)
    assert REAL_ON_SCREEN(area) and not REAL_ON_SCREEN(dict(area, left=3000))
    assert not REAL_ON_SCREEN(dict(area, left=-20000))


def test_a_window_moved_while_captured_and_an_old_frame_are_said():
    a = {"left": 2446, "top": 146, "width": 1366, "height": 768}
    assert framecheck.moved_problem(a, dict(a)) is None
    moved = framecheck.moved_problem(a, dict(a, left=2472, top=172))
    assert moved.kind == "moved" and "(2446, 146) -> (2472, 172)" in moved.why
    assert framecheck.stale_problem(100.0, 100.0 + framecheck.STALE_S - 0.01) is None
    assert framecheck.stale_problem(100.0, 103.0).kind == "stale"
    assert framecheck.stale_problem(None, 103.0) is None  # (mss: no frame times)


# --- her place on the map -----------------------------------------------------------------------------------------

def test_the_hunt_of_16_46_50_put_her_off_the_map_from_its_first_look():
    model = tree_model()
    for x, y in [(949.6, -909.5), (725.0, -900.0), (501.6, -1341.5), (1365.6, -493.5)]:  # its telemetry
        assert "off the map" in placecheck.off_map(model, Where(x, y, None, (0.0, 0.0)))
    for x, y in [(-218.4, -47.1), (310.0, 247.8), (31.3, -451.3), (77.3, -1677.5)]:  # good runs of that day
        assert placecheck.off_map(model, Where(x, y, None, (0.0, 0.0))) is None


# --- eyes on a fake game window --------------------------------------------------------------------------------------

class Desktop:
    """The screen (capture.Grabber's stand-in): a desktop with the game's frame where its window is (`at`). Until the
    duplication is opened afresh (`reopen`) or mss takes over (`fall_back`), it may show the game where the window was
    (`shown`: a frame from before the window moved)."""

    def __init__(self, full: np.ndarray, at=(100, 50), size=(1700, 1000)):
        self.full, self.at, self.shown, self.size = full, at, at, size
        self.reopened, self.fell_back, self.shots = 0, [], 0

    def move(self, at, lag: bool = False) -> None:
        self.at = at
        if not lag:
            self.shown = at

    def shot(self, area, after=None, at=None):
        self.shots += 1
        desk = np.full((self.size[1], self.size[0], 4), 240, np.uint8)
        x, y = self.shown
        h, w = self.full.shape[:2]
        desk[y:y + h, x:x + w] = cv2.cvtColor(self.full, cv2.COLOR_RGB2BGRA)
        return bench.SavedShot(desk, area)

    def reopen(self) -> None:
        self.reopened += 1
        self.shown = self.at

    def fall_back(self, why: str) -> None:
        self.fell_back.append(why)
        self.shown = self.at


class GameWindow(bench.SavedEyes):
    """Eyes on the Desktop's game window: where it is, its client size, whether it's in front (no real window). A
    move scripted for the next look happens between reading where the window is and the capture's check after it."""

    def __init__(self, desktop: Desktop, picture: np.ndarray | None = None):
        super().__init__(np.load(TAG), detector=object(), geo=NATIVE)
        self.screen, self.client = desktop, (1366, 768)
        self.name_tag.full = np.load(TAG)
        self.minimap = MinimapPicture(picture, NATIVE) if picture is not None else None
        self.front, self.during, self.gone, self.iconic = True, None, False, False
        self.fronted = 0  # times a look brought the game to the front (or would have)
        self.said: list[str] = []
        self.log = self.said.append

    def _client_area(self):
        if self.gone:
            raise RuntimeError("no window titled 'MapleStory'; is the game running?")
        if self.during is not None and self.screen.shots:  # read after the shot: the window moved meanwhile
            self.screen.move(self.during)
            self.during = None
        return {"left": self.screen.at[0], "top": self.screen.at[1], "width": self.client[0], "height": self.client[1]}

    def _window_problem(self, area):
        if self.iconic:
            return Problem("minimized", "the game is minimized: waiting for it to be restored")
        if (area["width"], area["height"]) != self.geo.client:
            return framecheck.size_problem((area["width"], area["height"]), self.geo)
        return None if self.front else Problem("window", "the game isn't in front")

    def _to_front(self):
        self.fronted += 1
        self.iconic = False  # (bring_to_front restores a minimized game)
        if self.gone:
            raise RuntimeError("no window titled 'MapleStory'; is the game running?")


def test_a_look_at_a_window_that_stays_put_passes_and_reads_her():
    eyes = GameWindow(Desktop(GOOD), tree_picture())
    obs = eyes.look()
    assert obs.bad is None and obs.tag == (689, 505) and obs.picture[:2] == (87, 17)


def test_a_window_dragged_during_the_look_is_refused_and_the_next_look_passes():
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    eyes.look()
    eyes.during, desk.shots = (126, 76), 0
    with pytest.raises(BadLook) as e:
        eyes.look()
    assert e.value.kind == "moved" and "(100, 50) -> (126, 76)" in e.value.why
    assert eyes.look().bad is None
    assert eyes.said == ["  the game window moved: (100, 50) -> (126, 76)"]


def test_a_capture_lagging_behind_the_window_is_refused_then_duplicated_afresh():
    """The 16:48:45 case: the window read where it is, the frame showing it where it was. The first recovery opens the
    duplication afresh; a look after it shows the window where it is."""
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    eyes.look()
    desk.move((126, 76), lag=True)
    with pytest.raises(BadLook) as e:
        eyes.look()
    assert e.value.kind == "offset" and "(-26, -26)" in e.value.why
    eyes.recover(e.value.problem)
    assert desk.reopened == 1 and eyes.minimap.corner is None and eyes.name_tag._last is None
    assert eyes.look().bad is None and eyes.minimap.corner == (87, 17)


def test_a_capture_still_off_falls_back_to_mss_and_both_are_tried_again_while_it_lasts(monkeypatch):
    from gamelab.perception import eyes as eyes_module
    clock = FakeTime()
    monkeypatch.setattr(eyes_module, "time", clock)
    eyes = GameWindow(Desktop(GOOD))
    eyes.recover(Problem("moved", "moved"))  # a move is waited out: the capture isn't at fault
    assert eyes.screen.reopened == 0 and eyes.screen.fell_back == []
    eyes.recover(Problem("offset", "off"))
    clock.sleep(0.2)
    eyes.recover(Problem("stale", "old"))
    assert eyes.screen.reopened == 1 and eyes.screen.fell_back == ["old"]
    for _ in range(int(eyes_module.CAPTURE_AGAIN_S / 0.2) - 1):  # nothing more for CAPTURE_AGAIN_S ...
        clock.sleep(0.2)
        eyes.recover(Problem("offset", "off"))
    assert eyes.screen.reopened == 1 and eyes.screen.fell_back == ["old"]
    clock.sleep(0.3)  # ... then the two again, in turn
    eyes.recover(Problem("offset", "off"))
    assert eyes.screen.reopened == 2
    clock.sleep(eyes_module.CAPTURE_AGAIN_S + 0.1)
    eyes.recover(Problem("offset", "off"))
    assert eyes.screen.fell_back == ["old", "off"]
    assert eyes.look().bad is None and eyes._fixes == 0  # a frame that passes: the next failure starts afresh
    eyes.recover(Problem("offset", "off"))
    assert eyes.screen.reopened == 3


def test_a_capture_lagging_after_a_drag_that_first_failed_as_moved_is_still_duplicated_afresh(monkeypatch):
    """The reviewer's repro_attempts.py: the drag failed twice as "moved", then the frame lagged behind the window. The
    reopen went by the failures of any kind in a row, so it never came, and the hunt stood still for good."""
    clock = FakeTime()
    monkeypatch.setattr(senses_module, "time", clock)
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
    senses.log = eyes.log = (lambda text: None)
    senses.look(mobs=False)
    real = eyes.look
    script = [BadLook(Problem("moved", "the game window moved while it was captured: (100, 50) -> (113, 63)")),
              BadLook(Problem("moved", "the game window moved while it was captured: (113, 63) -> (126, 76)"))]

    def look():
        if script:
            raise script.pop(0)
        return real()

    eyes.look = look
    desk.move((126, 76), lag=True)  # the drag over, the duplication still showing the window where it was
    s = senses.look(mobs=False)
    assert s.bad is None and desk.reopened == 1 and clock.now - 1000.0 < 4 * senses_module.RETRY_GAP_S


def test_a_resized_client_is_refused_without_a_capture_never_brought_forward_or_resized(monkeypatch):
    """The real window checks (Eyes._window_problem, framecheck.window_problem) on a client resized by Chrome Remote
    Desktop: said, no capture, the game not brought to the front, and never resized."""
    from gamelab.perception.eyes import Eyes
    desk = Desktop(GOOD)
    eyes = GameWindow(desk)
    eyes._window_problem = Eyes._window_problem.__get__(eyes)  # the real one, on a fake window
    eyes._to_front = Eyes._to_front.__get__(eyes)
    area = {"left": 100, "top": 50, "width": 1396, "height": 1952}
    eyes._client_area = lambda: dict(area)
    monkeypatch.setattr(framecheck.window, "find_game", lambda title: 1)
    monkeypatch.setattr(framecheck.window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(framecheck.window, "foreground_title", lambda: "Claude")  # the user in another program
    monkeypatch.setattr(framecheck.window, "resize_client", lambda *a: pytest.fail("resized the game window"))
    for _ in range(3):
        with pytest.raises(BadLook) as e:
            eyes.look()
        assert e.value.kind == "size" and "1396x1952, not 1366x768" in e.value.why and desk.shots == 0
    assert eyes.window_wait().kind == "size"  # (bring_to_front, on_screen and shows refuse in tests: never called)


def test_the_game_not_in_front_is_refused():
    eyes = GameWindow(Desktop(GOOD))
    eyes.front = False
    with pytest.raises(BadLook, match="isn't in front"):
        eyes.look()


def test_a_map_whose_picture_isnt_in_the_minimap_is_refused():
    """Another map's picture (West Rocky Mountain II's frame, Tree Dungeon's picture): her dot can't be read."""
    eyes = GameWindow(Desktop(ROCKY[0]), tree_picture())
    with pytest.raises(BadLook) as e:
        eyes.look()
    assert e.value.kind == "picture" and "best match" in e.value.why


def test_a_glance_at_an_offset_capture_has_no_dot_and_keeps_the_last_hp():
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    dot, hp = eyes.glance()
    assert dot is not None and hp > 0.5
    desk.move((126, 76), lag=True)
    eyes._glanced = -1.0
    assert eyes.glance() == (None, hp) and eyes.glances_failed == 1
    eyes.front, eyes._glanced = False, -1.0
    assert eyes.glance() == (None, hp) and eyes.glances_failed == 2


# --- Senses: never handing on a bad look ------------------------------------------------------------------------------

class FakeTime:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s

    def strftime(self, fmt: str) -> str:
        return "00:00:00"


def test_looks_failing_a_moment_are_taken_again_and_the_listeners_see_only_the_good_one(monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(senses_module, "time", clock)
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
    said, seen = [], []
    senses.log = said.append
    senses.listeners.append(seen.append)
    real = eyes.look
    bad = iter([BadLook(Problem("offset", "the capture is offset by (+26, +26) px")),
                BadLook(Problem("moved", "the game window moved while it was captured"))])

    def look():
        for e in bad:
            raise e
        return real()

    eyes.look = look
    s = senses.look(mobs=False)
    assert s.bad is None and s.where.platform.id == 23 and seen == [s]
    assert clock.now == pytest.approx(1000.0 + 2 * senses_module.RETRY_GAP_S)
    assert said[0] == "  a look failed its checks: the capture is offset by (+26, +26) px"
    assert said[1] == "  ... then: the game window moved while it was captured"
    assert said[2].startswith("  the game looks as expected again")
    assert senses.failed == {"looks": 2, "lost": 0}


def test_the_16_46_50_hunt_would_have_stood_still(monkeypatch):
    """Eyes without the map's minimap picture, as in that process: her dot puts her off the map at every look, which
    is never handed on; after LOST_S, LookLost, and nothing ever acted on her place."""
    clock = FakeTime()
    monkeypatch.setattr(senses_module, "time", clock)
    eyes = GameWindow(Desktop(GOOD))  # no picture
    senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
    seen, learned = [], []
    senses.log, senses.listeners = (lambda text: None), [seen.append]
    senses.camera.learn = lambda *a, **k: learned.append(a)
    with pytest.raises(LookLost) as e:
        senses.look(mobs=False)
    assert e.value.kind == "place" and "(678, -1341), off the map" in e.value.why
    assert clock.now - 1000.0 >= senses_module.LOST_S and seen == [] and learned == []  # the camera learns nothing
    with pytest.raises(LookLost):  # while it lasts, at once
        senses.look(mobs=False)
    assert clock.now - 1000.0 < senses_module.LOST_S + 2 * senses_module.RETRY_GAP_S


def test_regain_stands_still_until_a_look_passes(monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(senses_module, "time", clock)
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    eyes.client = (1396, 1952)  # Chrome Remote Desktop resized it
    senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
    said = []
    senses.log = said.append
    with pytest.raises(LookLost) as e:
        senses.look(mobs=False)
    real_sleep = clock.sleep

    def sleep(s):  # the user sets it back 10 s on
        real_sleep(s)
        if clock.now >= 1015.0:
            eyes.client = (1366, 768)

    clock.sleep = sleep
    s = senses.regain(e.value, until=clock.now + 60)
    assert s is not None and s.where.platform.id == 23 and 1015.0 <= clock.now < 1016.0
    assert said[0].startswith("  a look failed its checks: the game's client is 1396x1952, not 1366x768")
    assert any(line.startswith("  standing still: the game's client is 1396x1952") for line in said)
    assert said[-1].startswith("  the game looks as expected again")
    clock.sleep, eyes.client = real_sleep, (1396, 1952)
    with pytest.raises(LookLost) as e:
        senses.look(mobs=False)
    assert senses.regain(e.value, until=clock.now + 5) is None  # still lost when the time is up
    assert said[-1].startswith("  still lost when the time was up: the game's client is 1396x1952")


def test_standing_still_on_a_resized_window_never_brings_the_game_forward_or_captures(monkeypatch):
    """The reviewer's focus.py: standing still on a client resized by Chrome Remote Desktop, a look was taken twice a
    second, each bringing the game to the front over whatever the user was doing (136 times in 63 s). Now a look asks
    the window alone until it's the size again; the look that then passes brings it forward once."""
    clock = FakeTime()
    monkeypatch.setattr(senses_module, "time", clock)
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    eyes.client = (1396, 1952)
    senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
    senses.log = lambda text: None
    with pytest.raises(LookLost) as e:
        senses.look(mobs=False)
    real_sleep = clock.sleep

    def sleep(s):  # the user sets it back a minute on
        real_sleep(s)
        if clock.now >= 1063.0:
            eyes.client = (1366, 768)

    clock.sleep = sleep
    assert senses.regain(e.value, until=clock.now + 120) is not None
    assert eyes.fronted == 1 and desk.shots == 1 and clock.now >= 1063.0


def test_a_game_minimized_while_the_user_sets_it_right_isnt_restored_but_one_minimized_otherwise_is():
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    eyes.iconic = True  # minimized, nothing waited for: restored, as a look always did
    assert eyes.look().bad is None and eyes.fronted == 1
    eyes.client = (1396, 1952)
    with pytest.raises(BadLook, match="1396x1952"):
        eyes.look()
    eyes.iconic = True  # ... and minimized meanwhile: waited for too
    for _ in range(3):
        with pytest.raises(BadLook) as e:
            eyes.look()
        assert e.value.kind == "minimized" and eyes.fronted == 1
    eyes.iconic, eyes.client = False, (1366, 768)
    assert eyes.look().bad is None and eyes.fronted == 2


def test_the_game_window_gone_makes_a_hunt_stand_still_not_stop(monkeypatch):
    """The reviewer's nowindow.py: a game closed or crashed raised a plain RuntimeError, which ended the hunt."""
    clock = FakeTime()
    monkeypatch.setattr(senses_module, "time", clock)
    desk = Desktop(GOOD)
    eyes = GameWindow(desk, tree_picture())
    senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
    senses.log = lambda text: None
    eyes.gone = True
    with pytest.raises(LookLost) as e:
        senses.look(mobs=False)
    assert e.value.kind == "gone" and "is the game running" in e.value.why and eyes.fronted == 0
    real_sleep = clock.sleep

    def sleep(s):  # the game started again
        real_sleep(s)
        eyes.gone = clock.now < 1030.0

    clock.sleep = sleep
    assert senses.regain(e.value, until=clock.now + 60) is not None and eyes.fronted == 1


def test_a_refused_frame_keeps_the_last_good_hp_and_the_feeds_step_reads_nothing_off_it():
    """The reviewer's feedhp.py: the 16:48:45 offset frame read 0% HP, and the feed's world step made a hit of it."""
    from gamelab.localization.worldstate import Schedule, WorldStep
    eyes = GameWindow(Desktop(GOOD), tree_picture())
    step = WorldStep(senses_module.Senses(eyes, tree_model(), band=(100, 100)),
                     schedule=Schedule(band_s=None, wide_s=None))
    good = step.step(lambda ropes: eyes.look_at(GOOD, 1.0, ropes), 1.0, True)
    assert good.where is not None and good.hp > 0.5
    bad = step.step(lambda ropes: eyes.look_at(OFFSET, 1.1, ropes), 1.1, True)
    assert bad.where is None and bad.hp == good.hp and bad.events == []


# --- a hunt in the simulator, its looks failing for a while ------------------------------------------------------------

def _lost_hunt(window: tuple[float, float], minutes: float, problem: Problem) -> dict:
    """The hunt loop on Tree Dungeon (loop_tree_run's start, `minutes`), every look failing its checks with `problem`
    from window[0] to window[1] s of it: its trace."""
    import dataclasses

    from gamelab.sim import scenarios
    from gamelab.sim.scenarios import _loop
    if scenarios.real_map(TREE) is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data")

    def run(sim):
        real = sim.eyes.look

        def look():
            if window[0] <= sim.clock.elapsed < window[1]:
                sim.clock.sleep(0.03)  # a capture's time
                raise BadLook(problem)
            return real()

        sim.eyes.look = look
        return _loop(minutes)(sim)

    sc = dataclasses.replace(scenarios.SCENARIOS["loop_tree_run"], name="loop_tree_lost", fn=run, expects=())
    return scenarios.run_scenario(sc)


def test_a_hunt_sends_nothing_while_its_looks_fail_and_hunts_on_after():
    """From 20 s to 35 s of a 90 s hunt every look fails its checks (the capture offset). It stands still, says why,
    sends no plan from LOST_S after the first failure until the looks pass again, and then hunts on."""
    window = (20.0, 35.0)
    trace = _lost_hunt(window, 1.5, Problem("offset", "the capture is offset by (+26, +26) px from the game's client "
                                                      "area"))
    log = trace["log"]
    assert any("standing still: the capture is offset by (+26, +26)" in line for line in log), log
    assert any("the game looks as expected again" in line for line in log)
    # (a plan sent as the looks began to fail may still play out: the last before the window was at 20.49 s)
    quiet = [p for p in trace["plans"] if window[0] + 0.5 <= p[0] < window[1]]
    assert quiet == [], quiet
    assert any(p[0] >= window[1] for p in trace["plans"])  # hunting on after
    assert trace["result"]["hunt_stop"] == "time up"


def test_a_pause_longer_than_stuck_s_is_no_time_without_progress():
    """Looks failing for 125 s (the client resized, the user taking a while), longer than hunt.STUCK_S: when they pass
    again the hunt goes on, the pause not counted as time without progress or on its objective (before, the first
    look after it ended the hunt as stuck, or dropped the objective at once)."""
    from gamelab.behavior import hunt
    window = (25.0, 150.0)
    trace = _lost_hunt(window, 3.0, framecheck.size_problem((1396, 1952), NATIVE))
    assert window[1] - window[0] > hunt.STUCK_S
    after = [line for line in trace["log"] if float(line.split()[0]) >= window[1]]
    assert after and after[0].split(None, 1)[1].startswith("the game looks as expected again")
    assert not any("no kill" in line or "suspensions included: dropped" in line for line in after[:20]), after[:20]
    assert trace["result"]["hunt_stop"] == "time up"
    assert sum(p[0] >= window[1] for p in trace["plans"]) > 20


# --- the name tag, the minimap picture and the capture -----------------------------------------------------------------

def test_a_name_tag_template_that_no_longer_matches_is_cut_again(tmp_path):
    full = ROCKY[0]
    small = np.load(TAG)
    tag = NameTag(small, tmp_path / "tag_full.npy", NATIVE)
    tag.full = np.full_like(small, 128)  # a template that matches nothing
    tag.full[::2] = 0
    for _ in range(RECUT_AFTER - 1):
        assert tag.find(full) is None
    assert tag.find(full) == (685, 528) and tag.recuts == 1 and (tmp_path / "tag_full.npy").exists()
    assert tag.find(full) == (685, 528)


def test_a_minimap_picture_that_failed_to_load_is_tried_again(monkeypatch):
    import gamedata.canvas
    calls = []
    real = gamedata.canvas.outlinked

    def outlinked(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("a passing failure")
        return real(path)

    monkeypatch.setattr(gamedata.canvas, "outlinked", outlinked)
    monkeypatch.setattr(mapdata, "_PICTURES", {})
    assert mapdata.minimap_picture(TREE) is None
    tree_picture()  # (skips without the game data)
    assert len(calls) == 2


def test_a_command_stops_saying_why_if_the_maps_picture_wont_load(monkeypatch, capsys):
    from types import SimpleNamespace

    from gamelab import __main__ as cli

    def fail(map_id):
        raise KeyError("miniMap")

    monkeypatch.setattr(mapdata, "load_minimap_picture", fail)
    monkeypatch.setattr(cli, "MINIMAP_RETRY_S", 0.0)
    with pytest.raises(RuntimeError, match="minimap picture didn't load"):
        cli._minimap_picture(SimpleNamespace(id=TREE, minimap={"width": 548}))
    assert cli._minimap_picture(SimpleNamespace(id=1, minimap={"width": None})) is None
    monkeypatch.setattr(mapdata, "load_minimap_picture", lambda map_id: np.zeros((183, 67, 3), np.uint8))
    assert cli._minimap_picture(SimpleNamespace(id=TREE, minimap={"width": 548})).shape == (183, 67, 3)
    assert "the minimap picture of map 10002073: 67x183 game px" in capsys.readouterr().out


def test_bench_look_reads_tree_dungeons_frames_and_says_at_once_of_one_that_fails(monkeypatch, tmp_path, capsys):
    """bench-look (offline) on Tree Dungeon's frames: its eyes had no minimap picture, so every look was off the map, and
    after senses.LOST_S of retrying it stopped. A frame that fails its checks is said at once, with why."""
    import time
    from types import SimpleNamespace

    from gamelab import __main__ as cli
    from gamelab import paths
    tree_model(), tree_picture()
    monkeypatch.setattr(cli, "_calibration", lambda geo: SimpleNamespace(template=np.load(TAG)))
    monkeypatch.setattr(paths, "TAG_FULL", tmp_path / "tag.npy")
    good, offset = (str(FIXTURES / "capture" / n) for n in ("tree_135315.png", "tree_164845_offset.png"))
    assert cli.main(["bench-look", "--map", str(TREE), "--reps", "1", good]) == 0
    begun = time.monotonic()
    assert cli.main(["bench-look", "--map", str(TREE), "--reps", "1", good, offset]) == 1
    assert time.monotonic() - begun < senses_module.LOST_S
    assert "tree_164845_offset.png: the capture is offset by (+26, +26)" in capsys.readouterr().out


def test_the_scene_record_says_where_the_minimap_picture_was():
    from gamelab.localization.scene import Scene
    s = Scene(t=0.0, frame=GOOD, tag=None, hp=1.0, mp=1.0, geo=NATIVE, picture=(87, 17, 0.645))
    assert s.record()["minimap"] == [87, 17, 0.645]
    assert "minimap" not in Scene(t=0.0, frame=GOOD, tag=None, hp=1.0, mp=1.0, geo=NATIVE).record()


def test_hunt_look_every_says_each_look_and_stands_still_through_a_lost_one(monkeypatch, capsys):
    """`hunt look --every S` (no keys): a line a look, and a LookLost waited out as a hunt does (Senses.regain)."""
    from types import SimpleNamespace

    from gamelab import __main__ as cli
    clock = FakeTime()
    monkeypatch.setattr(cli, "time", clock)
    here = SimpleNamespace(where=Where(31.3, -451.3, SimpleNamespace(id=23), (0.0, 0.0)), how="camera")
    here.platform = here.where.platform
    looks = iter([here, LookLost(Problem("size", "the game's client is 1396x1952, not 1366x768")), here])
    regained = []

    def look(mobs=True):
        out = next(looks)
        if isinstance(out, Exception):
            raise out
        return out

    def regain(lost, until):
        regained.append(lost.why)
        clock.now += 5.0
        return here

    hunter = SimpleNamespace(look=look, senses=SimpleNamespace(regain=regain))
    assert cli._hunt_looks(hunter, every=1.0, seconds=2.5) == 0
    out = capsys.readouterr().out
    assert out.count("her: map (31, -451) on platform 23 (camera)") == 1 and regained == [
        "the game's client is 1396x1952, not 1366x768"]
