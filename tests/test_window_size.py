"""A client of the wrong size (or the game drawing off it) set right as the user does it, with Alt+Enter twice: full
screen and back (control.window_size). A fake game window that switches when an Alt+Enter is played, on a fake clock,
and does nothing with one that comes too soon after a switch (2026-10-25 23:41); Senses handing its looks to the fix;
the commands' start; the perception service's frames through the switch. No key, window or capture of the real game."""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gamelab import __main__ as cli
from gamelab import telemetry, window
from gamelab.control import client, player, window_size
from gamelab.control.window_size import ALT_ENTER, FULL_HOLD_S, SETTLE_S, SizeFix, refit
from gamelab.localization import senses as senses_module
from gamelab.perception.framecheck import BadLook, LookLost, Problem, size_problem
from gamelab.screen import NATIVE

from test_framecheck import FIXTURES, GOOD, Desktop, FakeTime, GameWindow, frame, tree_model, tree_picture

MONITOR = {"left": 0, "top": 0, "width": 3840, "height": 2160}
BACK = {"left": 2454, "top": 107, "width": 1366, "height": 768}  # where an Alt+Enter from full screen put it (23:41)
WRONG = size_problem((1350, 700), NATIVE)
DRAWN = Problem("drawn", "the game draws its picture (-18, -85) px off its client area")


def NO_LD() -> bool:  # noqa: N802 (the Lie Detector looked for, never there)
    return False


class Clock:
    """time.monotonic and time.sleep, a fake's; time.time moves with it (the hour's tries)."""

    def __init__(self):
        self.now, self.wall = 1000.0, time.time()

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.wall + self.now - 1000.0

    def sleep(self, s: float) -> None:
        self.now += s

    def strftime(self, fmt: str) -> str:
        return "00:00:00"


class Game:
    """The game's window as Windows tells it: its client area (full screen: the monitor's), whether it's in front. An
    Alt+Enter the player plays switches it `lag` s later, windowed to full screen and full screen to BACK (or to each
    of `then` in turn), unless it's `deaf`, or comes within `busy_for` s of the last switch landing (the game still
    setting itself up: at 23:41 one 3.5 s after the first wasn't taken). `after_plan` runs after each plan."""

    def __init__(self, clock: Clock, size=(1350, 700), full=False):
        self.clock = clock
        self.area = dict(MONITOR) if full else {"left": 100, "top": 50, "width": size[0], "height": size[1]}
        self.lag, self.deaf, self.busy_for, self.front, self.then = 1.2, False, 0.0, True, None
        self.plans: list[dict] = []
        self.landing: tuple[float, dict] | None = None
        self.landed = -1e18
        self.after_plan = None
        self.refuse: str | None = None  # the player's error for a plan
        self.taken = 0

    def install(self, monkeypatch) -> "Game":
        monkeypatch.setattr(window_size, "time", self.clock)
        monkeypatch.setattr(window, "find_game", lambda title: 1)
        monkeypatch.setattr(window, "minimized", lambda hwnd: False)
        monkeypatch.setattr(window, "client_rect", self.client_rect)
        monkeypatch.setattr(window, "full_screen", lambda hwnd, area: area == MONITOR)
        monkeypatch.setattr(window, "bring_to_front", lambda title, wait=0.0: self.front)
        monkeypatch.setattr(window, "foreground_title", lambda: "MapleStory" if self.front else "Claude")
        monkeypatch.setattr(window, "dpi_awareness", lambda: 2)
        monkeypatch.setattr(client, "submit", self.submit)
        return self

    def submit(self, events=(), **kw) -> dict:
        if self.refuse:
            raise client.PlayerError(self.refuse)
        self.client_rect(1)  # (a switch due lands first)
        self.plans.append({"t": self.clock.now, "events": [list(e) for e in events], **kw})
        if not self.deaf and self.landing is None and self.clock.now - self.landed >= self.busy_for:
            to = self.then.pop(0) if self.then else dict(BACK) if self.area == MONITOR else dict(MONITOR)
            self.landing = (self.clock.now + self.lag, to)
            self.taken += 1
        if self.after_plan:
            self.after_plan()
        return {"sent": [[e[0], *e[1:]] for e in events], "problems": []}

    def client_rect(self, hwnd) -> dict:
        if self.landing is not None and self.clock.now >= self.landing[0]:
            self.area, self.landed, self.landing = self.landing[1], self.landing[0], None
        return dict(self.area)


@pytest.fixture
def lab(tmp_path):
    """The alarm file, the hour's tries and the lock are the test's own (tests/conftest.py)."""
    return tmp_path


def spotter(seen=False):
    looks = []

    def spot():
        looks.append(1)
        return seen
    spot.looks = looks
    return spot


# --- the plan --------------------------------------------------------------------------------------------------------

def test_alt_enter_is_alt_held_round_a_tap_of_enter_and_the_player_takes_it():
    events = [list(e) for e in ALT_ENTER]
    assert events == [[0, "alt", "down"], [50, "enter", "down"], [90, "enter", "up"], [140, "alt", "up"]]
    assert cli.parse_steps(["alt+", "50", "enter", "50", "alt-"]) == (events, 140)  # as the explore steps say it
    player.validate({"events": events}, player.allowed_keys())


# --- one try ---------------------------------------------------------------------------------------------------------

def test_a_client_of_another_size_is_set_right_by_alt_enter_twice_the_second_held_for_full_screen(monkeypatch, lab):
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    said, spot = [], spotter()
    with telemetry.captured() as records:
        out = refit("MapleStory", log=said.append, spot=spot)
    assert out.ok and out.before == (1350, 700) and out.after == (1366, 768) and not out.full and out.pressed == 2
    assert [p["events"] for p in game.plans] == [[list(e) for e in ALT_ENTER]] * 2
    assert all(p["focus"] and p["name"] == "alt_enter" for p in game.plans)
    # The second not before FULL_HOLD_S after the first (full screen came at 1.2 s and stayed SETTLE_S: 2.7 s), and
    # the size read once it had settled too; the window's settle said, and when.
    assert game.plans[1]["t"] - game.plans[0]["t"] == pytest.approx(FULL_HOLD_S, abs=0.11)
    assert clock.now - game.plans[1]["t"] == pytest.approx(game.lag + SETTLE_S, abs=0.11)
    assert out.took == pytest.approx([1.2, 1.2], abs=0.11) and len(spot.looks) == 2
    assert out.settled == pytest.approx(clock.now, abs=0.11)
    assert said[0] == "  the client is 1350x700, not 1366x768: Alt+Enter twice"
    assert said[1].startswith("  the client is 1366x768 again (Alt+Enter 2 times; the window settled 1.2, 1.")
    assert [r["text"] for r in records if r["topic"] == "event" and r["kind"] == "window"] == [s[2:] for s in said]


def test_the_23_41_timing_a_second_alt_enter_too_soon_after_full_screen_isnt_taken(monkeypatch, lab):
    """2026-10-25 23:41: the first Alt+Enter put the game full screen; a second 3.5 s after it wasn't taken; one about
    8 s after it was. As built on work/window-recover, the second went once full screen had stood SETTLE_S (2.7 s in):
    not taken, and the try ended with the game left full screen. Now it waits FULL_HOLD_S, and one still not taken is
    pressed again (BACK_PRESSES at most)."""
    clock = Clock()
    game = Game(clock, size=(1366, 768)).install(monkeypatch)
    game.busy_for = 4.0  # (taken again 5.2 s after the first)
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
    assert out.ok and out.pressed == 2 and game.taken == 2
    game2 = Game(Clock(), size=(1366, 768)).install(monkeypatch)
    game2.busy_for = 9.0  # still busy at 8 s: that one isn't taken, the next CHANGE_S on is
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert out.ok and out.pressed == 3 and game2.taken == 2
    assert game2.plans[2]["t"] - game2.plans[1]["t"] == pytest.approx(window_size.CHANGE_S, abs=0.11)
    game3 = Game(Clock(), size=(1366, 768)).install(monkeypatch)
    game3.busy_for = 1e9  # never takes another: left full screen, said
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert not out.ok and out.full and out.pressed == 1 + window_size.BACK_PRESSES
    assert said[-1].startswith("  the client is 3840x2160 (full screen), not 1366x768 (Alt+Enter 4 times")
    assert said[-1].endswith(": the window didn't change within 8 s of Alt+Enter")


def test_a_game_full_screen_already_gets_only_the_way_back(monkeypatch, lab):
    game = Game(Clock(), full=True).install(monkeypatch)
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert out.ok and out.pressed == 1 and len(game.plans) == 1
    assert said[0] == "  the client is 3840x2160, full screen: Alt+Enter to set it back"


def test_a_window_set_right_by_the_first_alt_enter_gets_no_second(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)
    game.then = [dict(BACK)]
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
    assert out.ok and out.pressed == 1 and len(game.plans) == 1


def test_an_alt_enter_that_changes_nothing_gets_no_second(monkeypatch, lab):
    """A second would put a game that ignored the first full screen, and leave it so."""
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    game.deaf = True
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert not out.ok and out.pressed == 1 and len(game.plans) == 1 and out.after == (1350, 700)
    assert clock.now - game.plans[0]["t"] == pytest.approx(window_size.CHANGE_S, abs=0.11)
    assert said[-1] == ("  the client is 1350x700, not 1366x768 (Alt+Enter 1 time): the window didn't change within "
                        "8 s of Alt+Enter")


def test_a_first_alt_enter_that_leaves_it_windowed_at_another_size_gets_no_second(monkeypatch, lab):
    """A game full screen that wasn't told so, say: a second would put it full screen, and leave it so."""
    game = Game(Clock()).install(monkeypatch)
    game.then = [dict(BACK, width=1300, height=650), dict(MONITOR)]
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert not out.ok and out.pressed == 1 and len(game.plans) == 1 and out.after == (1300, 650) and not out.full
    assert said[-1].endswith(": the first Alt+Enter didn't put it full screen (a second would)")


def test_a_window_back_at_another_size_is_said(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)
    game.then = [dict(MONITOR), dict(BACK, width=1350, height=700)]
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert not out.ok and out.pressed == 2 and out.after == (1350, 700)
    assert said[-1].startswith("  the client is 1350x700, not 1366x768 (Alt+Enter 2 times; the window settled")


def test_no_alt_enter_with_the_lie_detector_alarm_up_or_the_lie_detector_on_screen(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)
    client.ALARM.write_text('{"time": "19:55:00"}', encoding="utf-8")
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)  # (not raised: the command goes on as the alarm has it)
    assert game.plans == [] and out.held and out.why == "the Lie Detector alarm is up"
    client.ALARM.unlink()
    out = refit("MapleStory", log=lambda text: None, spot=spotter(seen=True))
    assert game.plans == [] and out.pressed == 0 and out.why == "the Lie Detector is on screen"


def test_the_alarm_up_while_its_full_screen_stops_the_second(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)
    game.after_plan = lambda: client.ALARM.write_text('{"time": "19:55:01"}', encoding="utf-8")
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert len(game.plans) == 1 and out.full and not out.ok
    assert said[-1].endswith(": the Lie Detector alarm is up (the game may be left full screen)")


def test_the_lie_detector_on_screen_while_its_full_screen_stops_the_second_too(monkeypatch, lab):
    """No alarm yet (the player's watchdog doesn't see it at another size): the look before the second sees it."""
    game = Game(Clock()).install(monkeypatch)
    looks = iter([False, True])
    said = []
    out = refit("MapleStory", log=said.append, spot=lambda: next(looks))
    assert len(game.plans) == 1 and out.pressed == 1 and out.full and not out.ok
    assert said[-1].endswith("): the Lie Detector is on screen")


def test_no_first_alt_enter_if_the_screen_cant_be_looked_at_for_the_lie_detector(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)

    def blind():
        raise BadLook(Problem("perception", "the perception service isn't reached"))
    out = refit("MapleStory", log=lambda text: None, spot=blind)
    assert game.plans == [] and out.why == ("the screen couldn't be looked at for the Lie Detector (the perception "
                                            "service isn't reached)")
    out = refit("MapleStory", log=lambda text: None)  # (nothing to look with)
    assert game.plans == [] and "couldn't be looked at" in out.why
    looks = []

    def blind_once_full_screen():  # full screen by the first: the second goes out all the same (left so: worse)
        looks.append(1)
        if len(looks) == 2:
            raise BadLook(Problem("perception", "no frame from the perception service within 1.0 s"))
        return False
    out = refit("MapleStory", log=lambda text: None, spot=blind_once_full_screen)
    assert out.ok and out.pressed == 2 and len(game.plans) == 2


def test_no_alt_enter_while_a_plan_is_queued_or_played_or_another_process_is_trying(monkeypatch, lab):
    from gamelab.localization.service import only_one
    game = Game(Clock()).install(monkeypatch)
    client.INBOX.mkdir(parents=True)
    plan = client.INBOX / "195500_chain_ab12.running"
    plan.touch()
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
    assert out.held and out.pressed == 0 and game.plans == [] and "a plan is waiting" in out.why
    plan.unlink()
    held = only_one(window_size.LOCK)  # another process's try (a named mutex: let go of when it ends, even killed)
    try:
        out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
        assert out.held and game.plans == [] and out.why == "another process is setting it right"
    finally:
        window_size._unlock(held)
    assert refit("MapleStory", log=lambda text: None, spot=NO_LD).ok and len(game.plans) == 2
    assert only_one(window_size.LOCK) is not None  # (let go of after the try)


def test_no_alt_enter_unless_the_game_is_in_front_or_if_the_player_didnt_press_it(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)
    game.front = False
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
    assert game.plans == [] and out.why == "the game didn't come to the front ('Claude' is in front)"
    game.front, game.refuse = True, "game not in front (foreground: 'Claude')"  # the player's own check, before a key
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
    assert out.pressed == 0 and out.why == "the player didn't play Alt+Enter: game not in front (foreground: 'Claude')"
    game.refuse = None
    monkeypatch.setattr(client, "submit", lambda events=(), **kw: {"sent": [[0, "alt", "down"]], "problems": [
        "game left the foreground; stopped pressing keys"]})
    out = refit("MapleStory", log=lambda text: None, spot=NO_LD)
    assert out.pressed == 0 and out.why == ("the player didn't press Enter (game left the foreground; stopped pressing "
                                            "keys)")


def test_a_game_gone_in_the_middle_of_a_try_is_said_not_raised(monkeypatch, lab):
    game = Game(Clock()).install(monkeypatch)

    def gone(title):
        raise RuntimeError("no window titled 'MapleStory'; is the game running?")
    game.after_plan = lambda: monkeypatch.setattr(window, "find_game", gone)
    said = []
    out = refit("MapleStory", log=said.append, spot=NO_LD)
    assert not out.ok and out.pressed == 1 and out.after is None and len(game.plans) == 1
    assert said[-1] == ("  the game window is gone (Alt+Enter 1 time): the window didn't settle within 20 s of "
                        "Alt+Enter")


# --- the Lie Detector looked for at another size ---------------------------------------------------------------------

@pytest.mark.parametrize("png, size, seen", [
    ("lie_detector/2026-10-21_012504_mini_game_notice.png", (1396, 1952), True),  # (Chrome Remote Desktop's, once)
    ("lie_detector/2026-10-24_235129_mini_game.png", (3840, 2160), True),  # (full screen on the 4K monitor)
    ("lie_detector/2026-10-24_235129_mini_game.png", (1280, 720), True),
    ("lie_detector/2026-10-25_002828_quiz.jpg", (2560, 1440), True),
    ("lie_detector/2026-10-25_002758_quiz_warning.jpg", (1350, 700), True),
    ("lie_detector/2026-10-18_204759_truth_booth.png", (1600, 900), True),
    ("lie_detector/2026-10-19_004720_countdown.png", (3840, 2160), True),
    ("lie_detector/2026-10-19_004720_countdown.png", (1350, 700), False),  # not yet: its letters resampled too far
    ("game_scene.png", (1350, 700), False),  # (daffodils and other yellow scenery)
    ("capture/tree_135315.png", (3840, 2160), False),
])
def test_the_lie_detector_is_looked_for_on_the_services_picture_of_a_client_of_another_size(monkeypatch, png, size,
                                                                                               seen):
    """The game's own view stretched to a client of `size`, as perception.frames.glimpse hands it (1920x1080, the
    spotter's templates' size): seen there, where a 1366x768 picture of it didn't see the countdown at any size."""
    from gamelab.analysis import bench
    from gamelab.perception import frames
    view = frame(FIXTURES / png)
    if view.shape[1] != 1366:
        view = cv2.resize(view, (1366, 768), interpolation=cv2.INTER_AREA)
    shown = cv2.resize(view, size, interpolation=cv2.INTER_LINEAR)
    screen = bench.SavedScreen()
    screen.bgra = cv2.cvtColor(shown, cv2.COLOR_RGB2BGRA)
    monkeypatch.setattr(frames.window, "find_game", lambda title: 1)
    monkeypatch.setattr(frames.window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(frames.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": size[0],
                                                                    "height": size[1]})
    picture = frames.glimpse("MapleStory", screen)
    assert picture.shape == (1080, 1920, 3) and window_size.lie_detector_seen(picture) == seen


# --- when a look has nothing to act on for it ------------------------------------------------------------------------

class Tries:
    """refit's stand-in for SizeFix: the times it was asked, each answered from `results` (True: set right)."""

    def __init__(self, game: Game, results=()):
        self.game, self.results, self.at = game, list(results), []

    def __call__(self, title, want, log, spot):
        self.at.append(self.game.clock.now)
        ok = self.results.pop(0) if self.results else False
        if ok == "held":
            return window_size.Refit(before=(1350, 700), why="a plan is waiting for the player or being played",
                                     held=True)
        if ok:
            self.game.area = dict(BACK)
        return window_size.Refit(before=(1350, 700), after=window_size.size_of(self.game.area), ok=ok, pressed=2,
                                 settled=self.game.clock.now)


def test_a_fault_is_tried_once_it_has_stood_then_once_more_then_the_user_is_waited_for(monkeypatch, lab):
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    tries = Tries(game)
    monkeypatch.setattr(window_size, "refit", tries)
    said = []
    fix = SizeFix("MapleStory", log=said.append, spot=NO_LD)

    def at(t, problem=WRONG):
        clock.now = t
        return fix(problem)

    assert not at(1000.0) and not at(1004.9) and tries.at == []  # the same fault not yet STEADY_S
    assert not at(1005.0, Problem("window", "the game isn't in front")) and tries.at == []  # (not one it mends)
    assert not at(1005.0) and tries.at == [1005.0]
    assert not at(1034.0) and tries.at == [1005.0]  # AGAIN_S after
    assert not at(1035.0) and tries.at == [1005.0, 1035.0]
    assert said == ["  Alt+Enter didn't set it right in 2 tries: waiting for the user to set the client back to "
                    "1366x768"]
    assert not at(1100.0) and not at(2000.0) and len(tries.at) == 2 and len(said) == 1  # said once, never a loop
    fix.passed()  # the user set it back; later it goes wrong again
    tries.results = [True]
    game.area = dict(game.area, width=1340, height=690)
    assert not at(2100.0)
    game.area = dict(game.area, width=1350, height=700)  # still being dragged: the wait starts again
    assert not at(2103.0) and not at(2107.9) and len(tries.at) == 2
    assert at(2108.0) and tries.at[-1] == 2108.0 and fix.settled == 2108.0


def test_the_game_drawing_off_its_client_is_tried_though_the_size_is_right_and_never_in_a_loop(monkeypatch, lab):
    """2026-10-25 23:40: the client 1366x768, the game's picture drawn 18 px left and 85 up. Alt+Enter twice set it
    right then; a try that set the size right with the looks after still drawn off is followed by one more AGAIN_S
    later, then the user is waited for: a passed look alone starts it afresh."""
    clock = Clock()
    game = Game(clock, size=(1366, 768)).install(monkeypatch)
    tries = Tries(game, results=[True, True])
    monkeypatch.setattr(window_size, "refit", tries)
    said = []
    fix = SizeFix("MapleStory", log=said.append, spot=NO_LD)
    assert not fix(DRAWN) and tries.at == []
    clock.now += window_size.STEADY_S
    assert fix(DRAWN) and len(tries.at) == 1  # set right, by its size: the looks after say if it's drawn right
    clock.now += 1.0
    assert not fix(DRAWN) and len(tries.at) == 1  # still drawn off: not before AGAIN_S
    clock.now += window_size.AGAIN_S
    assert fix(DRAWN) and len(tries.at) == 2
    clock.now += window_size.AGAIN_S
    assert not fix(DRAWN) and len(tries.at) == 2 and said[-1].startswith("  Alt+Enter didn't set it right in 2 tries")
    clock.now += window_size.HOUR_S
    assert not fix(DRAWN) and len(tries.at) == 2  # given up until a look passes
    fix.passed()
    clock.now += 1.0
    assert not fix(DRAWN)
    clock.now += window_size.STEADY_S
    assert fix(DRAWN) is False and len(tries.at) == 3  # (afresh; this one didn't, `results` spent)


def test_tries_an_hour_are_bounded_across_processes_and_a_try_held_off_isnt_one(monkeypatch, lab):
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    tries = Tries(game, results=["held", "held"])
    monkeypatch.setattr(window_size, "refit", tries)
    said = []
    fix = SizeFix("MapleStory", log=said.append, spot=NO_LD)
    fix(WRONG, wait=False)
    fix(WRONG, wait=False)
    assert len(tries.at) == 2 and fix.tries == 0 and not window_size.WINDOW_TRIES.exists()
    for t in (1000.0, 1100.0, 1200.0):  # three tries of other processes in the hour
        window_size.book(window_size.WINDOW_TRIES, clock.wall + t - 1000.0)
    clock.now = 1200.0
    fix(WRONG, wait=False)  # the fourth
    assert len(tries.at) == 3 and len(window_size.tries_within(window_size.WINDOW_TRIES, clock.time())) == 4
    fix.passed()
    clock.now = 1300.0
    assert not fix(WRONG, wait=False) and len(tries.at) == 3
    assert said[-1] == ("  Alt+Enter was tried 4 times in the last hour (4 at most): waiting for the user to set the "
                        "client back to 1366x768")
    fix.passed()
    clock.now = 1000.0 + window_size.HOUR_S + 1  # the first of them an hour old
    tries.results = [True]
    assert fix(WRONG, wait=False) and len(tries.at) == 4


def test_why_a_try_isnt_made_is_said_once_each_time_it_goes_wrong(monkeypatch, lab):
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    tries = Tries(game, results=["held"] * 3)
    monkeypatch.setattr(window_size, "refit", tries)
    said = []
    fix = SizeFix("MapleStory", log=said.append, spot=NO_LD)
    for _ in range(3):  # (a look every 0.2 s)
        assert not fix(WRONG, wait=False)
    assert len(tries.at) == 3 and said == ["  no Alt+Enter: a plan is waiting for the player or being played"]
    monkeypatch.setattr(window, "minimized", lambda hwnd: True)
    assert not fix(WRONG, wait=False) and not fix(WRONG, wait=False)
    assert said[1:] == ["  no Alt+Enter: the game is minimized"]
    monkeypatch.setattr(window, "minimized", lambda hwnd: False)
    client.ALARM.write_text('{"time": "23:59:00"}', encoding="utf-8")  # (none with the alarm up: said, not raised)
    assert not fix(WRONG, wait=False) and len(tries.at) == 3
    assert said[2:] == ["  no Alt+Enter: the Lie Detector alarm is up"]
    fix.passed()  # set back; later it goes wrong again: said again
    assert not fix(WRONG, wait=False) and said[3:] == ["  no Alt+Enter: the Lie Detector alarm is up"]


def test_a_first_try_is_pending_from_the_fault_until_it_is_made_or_held_off_too_long(monkeypatch, lab):
    """What a command that stops on looks failing waits for before it stops (Senses._failed's hold)."""
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    tries = Tries(game, results=["held", "held", False])
    monkeypatch.setattr(window_size, "refit", tries)
    fix = SizeFix("MapleStory", log=lambda text: None, spot=NO_LD)
    assert not fix.pending()  # (no fault seen yet)
    fix(WRONG)
    assert fix.pending() and tries.at == []
    clock.now = 1000.0 + window_size.STEADY_S
    fix(WRONG)
    fix(WRONG)  # held off twice: still to come
    assert fix.pending() and len(tries.at) == 2
    clock.now = 1000.0 + window_size.STEADY_S + window_size.HOLD_S - 0.1
    assert fix.pending()
    fix(WRONG)  # made, and it didn't set it right
    assert len(tries.at) == 3 and fix.tries == 1 and not fix.pending()
    fix.passed()
    fix(WRONG)
    assert fix.pending()
    clock.now += window_size.STEADY_S + window_size.HOLD_S
    assert not fix.pending()  # held off (or still being dragged) that long: the command stops


def test_a_game_minimized_gone_or_the_size_already_right_isnt_tried(monkeypatch, lab):
    clock = Clock()
    game = Game(clock).install(monkeypatch)
    tries = Tries(game)
    monkeypatch.setattr(window_size, "refit", tries)
    fix = SizeFix("MapleStory", log=lambda text: None, spot=NO_LD)
    monkeypatch.setattr(window, "minimized", lambda hwnd: True)
    assert not fix(WRONG, wait=False)
    monkeypatch.setattr(window, "minimized", lambda hwnd: False)
    game.area = dict(BACK)
    assert not fix(WRONG, wait=False)  # (a size fault, the size right meanwhile)

    def gone(title):
        raise RuntimeError("no window titled 'MapleStory'; is the game running?")
    monkeypatch.setattr(window, "find_game", gone)
    assert not fix(WRONG, wait=False) and tries.at == []


# --- Senses: the looks handed to the fix -----------------------------------------------------------------------------

class WallTime(FakeTime):
    def time(self) -> float:
        return self.now


class Resized:
    """A game window resized to 1350x700 (GameWindow on a Desktop, test_framecheck's), with the size fix a command that
    acts gives its Senses, on a fake clock: its tries (`refit`'s stand-in) set the client right if `mends`, and are
    held off by a plan until `busy_until`."""

    def __init__(self, monkeypatch, mends: bool = True, busy_until: float = 0.0):
        self.mends, self.busy_until = mends, busy_until
        self.clock = WallTime()
        monkeypatch.setattr(senses_module, "time", self.clock)
        monkeypatch.setattr(window_size, "time", self.clock)
        self.desk = Desktop(GOOD)
        self.eyes = eyes = GameWindow(self.desk, tree_picture())
        eyes.client = (1350, 700)
        monkeypatch.setattr(window, "find_game", lambda title: 1)
        monkeypatch.setattr(window, "minimized", lambda hwnd: False)
        monkeypatch.setattr(window, "client_rect", lambda hwnd: {"left": 100, "top": 50, "width": eyes.client[0],
                                                                 "height": eyes.client[1]})
        monkeypatch.setattr(window, "full_screen", lambda hwnd, area: False)
        self.tried: list[float] = []
        monkeypatch.setattr(window_size, "refit", self.refit)
        self.fix = window_size.SizeFix("MapleStory", log=eyes.said.append)
        self.senses = senses_module.Senses(eyes, tree_model(), band=(100, 100))
        self.senses.window_fix = self.fix
        self.said: list[str] = []
        self.senses.log = self.said.append

    def refit(self, title, want, log, spot):
        if self.clock.now < self.busy_until:
            return window_size.Refit(before=self.eyes.client, why="a plan is waiting for the player or being played",
                                     held=True)
        self.tried.append(self.clock.now)
        if self.mends:
            self.eyes.client = want
        return window_size.Refit(before=(1350, 700), after=self.eyes.client, ok=self.mends, pressed=2,
                                 settled=self.clock.now)


def test_a_resized_window_is_set_right_with_alt_enter_once_it_has_stood_and_a_hunt_goes_on(monkeypatch):
    """The hunt loop (`regains`: Hunt.run) stands still as before (LookLost at LOST_S); while it stands still (regain),
    once the same wrong size has stood STEADY_S, Alt+Enter twice (a stand-in that sets it right here), and the next
    look passes. Only that look captures."""
    game = Resized(monkeypatch)
    clock, desk, senses, tried = game.clock, game.desk, game.senses, game.tried
    senses.regains = True
    with pytest.raises(LookLost) as e:
        senses.look(mobs=False)
    assert tried == [] and clock.now - 1000.0 < window_size.STEADY_S
    s = senses.regain(e.value, until=clock.now + 60)
    assert s is not None and s.where.platform.id == 23
    assert len(tried) == 1 and tried[0] - 1000.0 >= window_size.STEADY_S and clock.now - tried[0] < 1.0
    assert desk.shots == 1 and game.fix.tries == 0  # (a passed look: afresh)
    assert senses.settled == tried[0] and game.said[-1].startswith("  the game looks as expected again")


def test_a_command_that_stops_on_looks_failing_waits_for_the_fix_and_goes_on(monkeypatch):
    """LookLost (LOST_S, 3 s) comes before the fix's STEADY_S (5 s): a command that doesn't stand still on it (all but
    the hunt loop) would stop untried. Its looks wait for the first try instead, and the looks after a try that set it
    right have LOST_S afresh (here one refused as the window moved while it was captured)."""
    game = Resized(monkeypatch)
    clock, eyes, senses, tried = game.clock, game.eyes, game.senses, game.tried
    real, moved = eyes.look, []

    def look():
        if tried and not moved:
            moved.append(clock.now)
            raise BadLook(Problem("moved", "the game window moved while it was captured"))
        return real()

    eyes.look = look
    s = senses.look(mobs=False)
    assert s.where.platform.id == 23 and senses.failed["lost"] == 0
    assert len(tried) == 1 and tried[0] - 1000.0 >= window_size.STEADY_S
    assert moved and moved[0] - 1000.0 > senses_module.LOST_S  # (it stopped there before)
    assert game.said[0].startswith("  a look failed its checks: the game's client is 1350x700, not 1366x768")
    assert game.said[1] == "  ... then: the game window moved while it was captured"
    assert game.said[-1].startswith("  the game looks as expected again")


def test_a_command_that_stops_on_looks_failing_stops_after_a_try_that_didnt_set_it_right(monkeypatch):
    game = Resized(monkeypatch, mends=False)
    with pytest.raises(LookLost) as e:
        game.senses.look(mobs=False)
    assert e.value.kind == "size" and len(game.tried) == 1 and game.clock.now - game.tried[0] < 1.0
    with pytest.raises(LookLost):  # at once, while it lasts: the second try is AGAIN_S on (a hunt standing still's)
        game.senses.look(mobs=False)
    assert len(game.tried) == 1


def test_a_command_that_stops_on_looks_failing_waits_a_while_for_a_plan_holding_the_try_off(monkeypatch):
    game = Resized(monkeypatch, busy_until=1008.0)  # a chain's plan playing on 8 s
    assert game.senses.look(mobs=False).where.platform.id == 23
    assert len(game.tried) == 1 and 1008.0 <= game.tried[0] < 1008.5
    assert game.eyes.said[0] == "  no Alt+Enter: a plan is waiting for the player or being played"
    game = Resized(monkeypatch, busy_until=1e9)  # one that never ends
    with pytest.raises(LookLost):
        game.senses.look(mobs=False)
    assert game.tried == [] and game.clock.now - 1000.0 == pytest.approx(window_size.STEADY_S + window_size.HOLD_S,
                                                                         abs=0.5)
    assert game.eyes.said == ["  no Alt+Enter: a plan is waiting for the player or being played"]  # (said once)


def test_no_look_or_glance_after_the_fix_is_of_a_frame_shown_before_the_window_settled(monkeypatch):
    """The perception service's states of frames from before the switch (or from its middle) are never acted on: a
    look's and a glance's frames are shown after the window settled at the soonest (Senses.settled)."""
    from gamelab.localization.feed import Feed
    clock = WallTime()
    monkeypatch.setattr(senses_module, "time", clock)
    asked = {"look": [], "glance": []}
    fixed = []

    class Service(Feed):  # (a feed with nothing to act on: the client the wrong size, then the frames gone)
        def look(self, senses, mobs, wide, loot, mesos_only, after=None):
            asked["look"].append((clock.now, after))

        def glance(self, after, timeout=0.25):
            asked["glance"].append(after)

        def problem(self):
            return WRONG if not fixed else Problem("perception", "no state from perception in time")

    class Fix:
        settled = None

        def __call__(self, problem):
            if clock.now < 1006.0:
                return False
            fixed.append(clock.now)
            self.settled = clock.now - 0.5  # (the window settled a moment before the try ended)
            return True

        def pending(self):
            return not fixed

        def passed(self):
            pass
    senses = senses_module.Senses(GameWindow(Desktop(GOOD)), tree_model(), band=(100, 100), feed=Service())
    senses.log, senses.window_fix = (lambda text: None), Fix()
    with pytest.raises(LookLost):
        senses.look(mobs=False)
    after_fix = [a for t, a in asked["look"] if t >= fixed[0]]
    assert fixed and all(a >= senses.window_fix.settled for a in after_fix) and len(after_fix) > 3
    assert any(a < senses.window_fix.settled for t, a in asked["look"] if t < fixed[0])  # (before it: MAX_AGE_S's)
    assert clock.now - fixed[0] >= senses_module.LOST_S  # (LOST_S afresh from the try)
    senses._usable = clock.now
    assert senses.glance() == (None, 1.0) and asked["glance"][-1] >= senses.window_fix.settled
    asked["look"].clear()
    with pytest.raises(LookLost):  # (an explicit `after`, Navigator.recover's, is floored the same)
        senses.look(mobs=False, after=1.0)
    assert asked["look"][0][1] == senses.settled


# --- the commands ----------------------------------------------------------------------------------------------------

class Viewer:
    """The perception service's client for `window --recover`: its picture of the client, its newest problem."""

    def __init__(self, name, title="", log=None, problem=None):
        self.problem_ = problem

    def ready(self, timeout):
        return True

    def problem(self):
        return self.problem_

    def full_frame(self, after, timeout=1.0, any_size=False):
        assert any_size
        return np.zeros((768, 1366, 3), np.uint8)

    alive, why, served, refused, connects, gaps = True, "", 0, 0, 1, 0

    def close(self):
        pass


def test_window_recover_says_the_size_before_and_after(monkeypatch, lab, capsys):
    from gamelab.localization import service
    game = Game(Clock()).install(monkeypatch)
    monkeypatch.setattr(service, "ServiceClient", Viewer)
    monkeypatch.setattr(cli, "_game_title", lambda: "MapleStory")
    assert cli.main(["window", "--recover"]) == 0 and len(game.plans) == 2
    out = capsys.readouterr().out
    assert "the game's client: 1350x700 at (100, 50) (desktop px); DPI awareness 2" in out
    assert "the client is 1350x700, not 1366x768: Alt+Enter twice" in out
    assert "the game's client now: 1366x768 at (2454, 107)" in out
    assert not window_size.WINDOW_TRIES.exists()  # (by hand: not one of the hunts' tries an hour)
    assert cli.main(["window", "--recover"]) == 0 and len(game.plans) == 2  # 1366x768 already
    assert "it's 1366x768 already, drawn where it is: nothing to do" in capsys.readouterr().out
    assert cli.main(["window", "--recover", "--force"]) == 0 and len(game.plans) == 4
    monkeypatch.setattr(service, "ServiceClient", lambda name, **kw: Viewer(name, problem=DRAWN))
    assert cli.main(["window", "--recover"]) == 0 and len(game.plans) == 6  # drawn off: tried at 1366x768 too
    assert "the perception service says: the game draws its picture (-18, -85)" in capsys.readouterr().out
    monkeypatch.setattr(service, "ServiceClient", Viewer)
    game.deaf = True
    game.area = dict(game.area, width=1350, height=700)
    assert cli.main(["window", "--recover"]) == 1 and len(game.plans) == 7
    client.INBOX.mkdir(parents=True)
    (client.INBOX / "195500_pet_food_ab12.json").touch()  # another process's plan waiting for the player
    capsys.readouterr()
    assert cli.main(["window", "--recover"]) == 1 and len(game.plans) == 7
    assert "no Alt+Enter: a plan is waiting for the player or being played" in capsys.readouterr().out
    assert cli.main(["window"]) == 1 and "give the size" in capsys.readouterr().out


def test_a_command_that_acts_started_on_a_full_screen_client_sets_it_right_and_goes_on(monkeypatch, lab, capsys):
    """23:21 on 2026-10-25: full screen at 3840x2160, and a command that acts stopped with "not calibrated at
    3840x2160 yet". Now it tries at once, and waits (said once) while that doesn't do it; it goes on once the client is
    a size the eyes are for."""
    from gamelab.screen import geometry
    clock = Clock()
    game = Game(clock, full=True).install(monkeypatch)
    monkeypatch.setattr(cli, "time", clock)
    monkeypatch.setattr(cli, "_game_geometry", lambda title, restore=False: geometry(*window_size.size_of(game.area)))
    monkeypatch.setattr(cli, "_calibrated", lambda size: size in ((1366, 768), (1920, 1080)))
    fix = SizeFix("MapleStory", log=cli._log, spot=NO_LD)
    cli._fit_start("MapleStory", fix)
    assert len(game.plans) == 1 and window_size.size_of(game.area) == (1366, 768)
    game.area, game.deaf = dict(MONITOR), True  # full screen again, and the game deaf to Alt+Enter
    fix.passed()
    sleeps = []
    real = clock.sleep

    def sleep(s):  # the user sets it back by hand a while on
        sleeps.append(s)
        real(s)
        if clock.now >= 2000.0:
            game.area = dict(BACK)
    clock.sleep = sleep
    capsys.readouterr()
    cli._fit_start("MapleStory", fix)  # (no exit: it waits)
    out = capsys.readouterr().out
    assert clock.now >= 2000.0 and len(game.plans) == 1 + 2 * window_size.BACK_PRESSES  # its two tries, then waits
    assert out.count("a size the eyes aren't calibrated for: waiting for it to be set back") == 1
    assert "going on" in out.splitlines()[-1]
    game.area = dict(MONITOR)
    client.ALARM.write_text('{"time": "23:59:00"}', encoding="utf-8")
    with pytest.raises(client.LieDetectorAlarm):  # the alarm stops it, as at any such command's start
        cli._fit_start("MapleStory", fix)


@pytest.mark.parametrize("argv", [["hunt", "kill"], ["hunt", "clear"], ["nav", "23"], ["tour"], ["tour", "--loop"]])
def test_the_commands_that_act_record_their_telemetry_before_their_hunter_sets_the_size_right(monkeypatch, lab, argv):
    calls = []
    monkeypatch.setattr(cli, "_record", lambda command, map_id: calls.append("record"))

    def hunter(args, **kw):
        calls.append("hunter")
        raise RuntimeError("stopped here")
    monkeypatch.setattr(cli, "_hunter", hunter)
    assert cli.main(argv) == 1 and calls == ["record", "hunter"]
    calls.clear()
    assert cli.main(["hunt", "look"]) == 1 and calls == ["hunter"]  # (sends no keys, records nothing)


# --- the perception service through the switch -----------------------------------------------------------------------

def test_the_services_frames_go_through_a_full_screen_switch_and_say_the_game_drawn_off(monkeypatch):
    """The service's frames (perception.frames.ScreenFrames) while the client goes 1366x768 -> full screen 3840x2160 ->
    1366x768, the capture failing outright once as it comes back (mss refused mid-switch: it ended the frame loop, the
    service with it): each frame says what it is, nothing is raised, and the frames after are read as before. And a
    frame the game drew at its window's corner (the frame of 23:40) is "drawn", with the window's frame known."""
    from gamelab.analysis import bench
    from gamelab.perception.frames import ScreenFrames
    state = {"area": dict(BACK), "fail": False, "picture": GOOD}
    monkeypatch.setattr(window, "find_game", lambda title: 1)
    monkeypatch.setattr(window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(window, "client_rect", lambda hwnd: dict(state["area"]))
    monkeypatch.setattr(window, "on_screen", lambda area: True)
    monkeypatch.setattr(window, "foreground_title", lambda: "MapleStory")
    monkeypatch.setattr(window, "shows", lambda hwnd, area, grid=None: True)
    monkeypatch.setattr(window, "inset", lambda hwnd, area: (18, 85))  # the window's frame at 300%

    class Screen(bench.SavedScreen):
        def shot(self, area, after=None, at=None):
            if state["fail"]:
                state["fail"] = False
                raise OSError("gdi32.GetDIBits() failed")
            desk = np.zeros((2160, 3840, 4), np.uint8)
            x, y = state["area"]["left"], state["area"]["top"]
            desk[y:y + 768, x:x + 1366] = cv2.cvtColor(state["picture"], cv2.COLOR_RGB2BGRA)
            return bench.SavedShot(desk, area)
    frames = ScreenFrames("MapleStory", grabber=Screen(), geo=NATIVE, log=lambda text: None)
    assert frames.next(time.monotonic()).visible
    state["area"] = dict(MONITOR)
    f = frames.next(time.monotonic())
    assert not f.visible and f.problem.kind == "size" and "3840x2160" in f.why
    state["area"], state["fail"] = dict(BACK), True
    f = frames.next(time.monotonic())
    assert not f.visible and f.problem.kind == "capture" and "GetDIBits" in f.why
    assert frames.next(time.monotonic()).visible  # (and the window where it was: no move said)
    state["picture"] = frame(FIXTURES / "capture" / "tree_234040_drawn.png")
    f = frames.next(time.monotonic())
    assert not f.visible and f.problem.kind == "drawn" and "(-18, -85)" in f.why


def test_a_full_ask_for_any_size_is_answered_with_the_clients_picture_while_its_size_is_wrong(monkeypatch):
    """The size fix's look for the Lie Detector and the pet's for the HUD (ServiceClient.full_frame with any_size): while
    the frames aren't fit to read for a fault Alt+Enter mends, the service answers with the client's picture as it is
    (perception.frames.glimpse), down the pipe; a plain full ask, or another fault, with why there's none."""
    from gamelab.localization import service
    from gamelab.localization.worldstate import WorldState
    from test_service import _joining, _svc
    picture = np.full((1080, 1920, 3), 7, np.uint8)
    glimpses = []
    svc = _svc(glimpse=lambda: glimpses.append(1) or picture)
    said = []
    c = _joining(svc, 1, said)
    asks = svc._take(c, ("full", 1, 10.0, True), None)
    svc._take(c, ("full", 2, 10.0), asks)
    wrong = WorldState(5, 10.2, False, why=WRONG.why, bad=WRONG)
    svc._answer_fulls(None, wrong)
    answers = {m[2]: m[3] for m in said if m[1] == "full"}
    assert answers[1]["frame"] is picture and answers[1]["bad"] is None and glimpses == [1]
    assert answers[2]["bad"] == WRONG and "frame" not in answers[2]
    said.clear()
    svc._take(c, ("full", 3, 10.0, True), None)
    hidden = Problem("window", "the game isn't in front")
    svc._answer_fulls(None, WorldState(6, 10.3, False, why=hidden.why, bad=hidden))
    assert said[0][3]["bad"] == hidden and "frame" not in said[0][3] and glimpses == [1]
    # The command's side: the picture handed over as it came.
    me = service.ServiceClient("tour", acts=False, start=False, ensure=None)

    class Pipe:
        def send(self, msg):
            assert msg[0] == "full" and msg[3] is True
            me._take(("full", msg[1], {"seq": None, "slot": None, "t": 10.4, "epoch": 0, "bad": None,
                                       "frame": picture}))
    me._conn, me.alive = Pipe(), True
    assert me.full_frame(10.0, any_size=True) is picture


def test_the_service_answers_a_full_ask_for_any_size_with_no_run_yet(monkeypatch):
    """The service started with the game full screen has no run (no geometry to look with): a full ask for any size
    still gets the client's picture."""
    from test_service import _joining, _svc
    picture = np.zeros((1080, 1920, 3), np.uint8)
    svc = _svc(glimpse=lambda: picture)
    svc.geometry = lambda: (None, size_problem((3840, 2160), NATIVE))
    said = []
    c = _joining(svc, 1, said)
    svc._take(c, ("full", 1, 0.0, True), None)
    assert not svc._build()
    svc._why = size_problem((3840, 2160), NATIVE)
    svc._stop = True  # (its wait ends at once)
    svc._not_visible()
    assert [m[3].get("frame") is picture for m in said if m[1] == "full"] == [True]


def test_the_pet_and_hunt_look_get_no_size_fix_and_a_command_at_1920x1080_none_on_its_looks(monkeypatch):
    """Alt+Enter sets the game's own 1366x768: the eyes of the stretched 1920x1080 window keep theirs as they are."""
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.localization import mapdata, service
    from gamelab.screen import STRETCHED
    from test_worldstate import small_map

    class Client:
        def __init__(self, name, **kw):
            self.welcome, self.why = {"pid": 1, "epoch": 1, "map": kw["map_id"]}, ""

        def ready(self, timeout):
            return True
    geo = {"geo": NATIVE}
    monkeypatch.setattr(service, "ServiceClient", Client)
    monkeypatch.setattr(cli, "_eyes", lambda model, title, perception, restore: SimpleNamespace(log=print,
                                                                                               geo=geo["geo"]))
    monkeypatch.setattr(cli, "_fit_start", lambda title, fix: None)
    monkeypatch.setattr(mapdata, "load", lambda map_id: small_map())
    monkeypatch.setattr(hunter_mod, "Hunter", lambda eyes, keys, model, log, feed, **kw: SimpleNamespace(
        senses=SimpleNamespace(window_fix=None)))
    args = cli.build_parser().parse_args(["tour", "--loop"])
    assert isinstance(cli._hunter(args).senses.window_fix, SizeFix)
    assert cli._hunter(args, throws=False).senses.window_fix is None  # (the pet: PetWait, behavior.pet)
    assert cli._hunter(args, viewer=True).senses.window_fix is None
    geo["geo"] = STRETCHED
    assert cli._hunter(args).senses.window_fix is None
    cli._PERCEPTION.clear()


# --- whether it's full screen ----------------------------------------------------------------------------------------

def test_full_screen_is_a_client_that_is_the_whole_of_its_monitor(monkeypatch):
    import importlib.util

    from ctypes import wintypes

    from gamelab import paths
    spec = importlib.util.spec_from_file_location("window_copy", paths.REPO / "gamelab" / "window.py")
    win = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(win)  # a copy of its own: the tests' window module refuses every call

    class User32:
        def __init__(self, monitor, rect=(0, 0, 0, 0)):
            self.monitor, self.rect = monitor, rect

        def MonitorFromWindow(self, hwnd, flags):  # noqa: N802 (Windows' names)
            return 7

        def GetMonitorInfoW(self, monitor, info):  # noqa: N802
            info._obj.rcMonitor = wintypes.RECT(*self.monitor)
            return 1

        def GetWindowRect(self, hwnd, rect):  # noqa: N802
            rect._obj.left, rect._obj.top, rect._obj.right, rect._obj.bottom = self.rect
            return 1
    monkeypatch.setattr(win, "user32", User32((0, 0, 3840, 2160), rect=(2436, 22, 3838, 893)))
    assert win.full_screen(1, dict(MONITOR)) and not win.full_screen(1, dict(BACK))
    assert win.inset(1, dict(BACK)) == (18, 85)  # (23:40's: the window's frame at 300%)
    monkeypatch.setattr(win, "user32", User32((-1366, 0, 0, 768)))  # the game's own mode on a monitor to the left
    assert win.full_screen(1, {"left": -1366, "top": 0, "width": 1366, "height": 768})
