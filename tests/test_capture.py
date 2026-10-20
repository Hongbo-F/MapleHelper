"""Capturing the screen (gamelab.capture) without the screen: desktop duplication's COM calls are faked, so
what's tested is what's made of their answers: the black first frame, frames shown after a look began, a
duplication lost and started again, and the fall back to mss. And Eyes taking a glance's two boxes from one
frame, at the old pace."""

import time

import numpy as np
import pytest

from gamelab import capture
from gamelab.capture import (DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_INVALID_CALL, DXGI_ERROR_WAIT_TIMEOUT, FIRST_FRAME_S,
                             FRESH_WAIT_S, RETRY_S, S_OK, CaptureError, Grabber)
from gamelab.perception import eyes as eyes_module
from gamelab.perception.eyes import GLANCE_EVERY_S, Eyes

AREA = {"left": 0, "top": 0, "width": 1920, "height": 1080}
TIMEOUT = (DXGI_ERROR_WAIT_TIMEOUT, 0, None)


class Clock:
    """time.monotonic and time.sleep, made up."""

    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def picture(value: int, size=(1080, 1920)) -> np.ndarray:
    return np.full((*size, 4), value, np.uint8)


class FakeDxgi:
    """Stands in for capture.Dxgi: each AcquireNextFrame gives the next of `script`: (HRESULT, present time in
    ms ticks, picture). A timeout (or the script running out) takes as long as it was given."""

    def __init__(self, clock: Clock, script: list, desktop=(0, 0, 1920, 1080)):
        self.clock, self.script, self.desktop = clock, list(script), desktop
        self.picture, self.timeouts, self.closed = None, [], False

    def acquire(self, timeout_ms: int):
        self.timeouts.append(timeout_ms)
        hr, present, pic = self.script.pop(0) if self.script else TIMEOUT
        if hr == DXGI_ERROR_WAIT_TIMEOUT:
            self.clock.now += timeout_ms / 1000
        elif hr == S_OK and present:
            self.picture = pic
        return hr, present

    def to_monotonic(self, ticks: int) -> float:
        return ticks / 1000

    def read(self, boxes, convert=None):
        views = [self.picture[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
        return [np.array(v) if convert is None else convert(v) for v in views]

    def close(self):
        self.closed = True


class Opener:
    """Opens FakeDxgi duplications, one script each (an exception in the list: that opening fails)."""

    def __init__(self, clock: Clock, *scripts, desktop=(0, 0, 1920, 1080)):
        self.clock, self.scripts, self.desktop = clock, list(scripts), desktop
        self.opened: list[FakeDxgi] = []
        self.points = []

    def __call__(self, point):
        self.points.append(point)
        script = self.scripts.pop(0) if self.scripts else []
        if isinstance(script, Exception):
            raise script
        com = FakeDxgi(self.clock, script, self.desktop)
        self.opened.append(com)
        return com


class FakeMss:
    def __init__(self, value: int = 7):
        self.value, self.grabs = value, []

    def grab(self, box):
        self.grabs.append(box)
        return picture(self.value, (box["height"], box["width"]))

    def close(self):
        pass


def grabber(clock, opener, method="dxgi") -> Grabber:
    g = Grabber(method, opener=opener, clock=clock, sleep=clock.sleep)
    g._mss = FakeMss()
    return g


def first_frames(t_ms: int, value: int = 50) -> list:
    """What duplicating gives first: a black frame with no present time, then the screen."""
    return [(S_OK, 0, picture(0)), (S_OK, t_ms, picture(value))]


def test_the_black_first_frame_doesnt_count():
    clock = Clock()
    g = grabber(clock, Opener(clock, first_frames(100_500)))
    shot = g.shot(AREA)
    assert isinstance(shot, capture.DxgiShot) and shot.t == 100.5
    assert (shot.crop(0, 0, 1920, 1080) == 50).all()
    assert g.dup.frames == 1 and g.counts["dxgi"] == 1


def test_with_nothing_new_a_shot_is_the_last_frame_at_once():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500))
    g = grabber(clock, opener)
    g.shot(AREA)
    clock.now = 101.0
    shot = g.shot(AREA)  # nothing new: DXGI times out at once
    assert shot.t == 100.5 and (shot.crop(10, 10, 4, 4) == 50).all()
    assert opener.opened[0].timeouts[-1] == 0 and clock.now == 101.0


def test_pointer_only_frames_keep_the_frame_and_its_time():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500) + [(S_OK, 0, picture(0))])
    g = grabber(clock, opener)
    g.shot(AREA)
    shot = g.shot(AREA)
    assert shot.t == 100.5 and (shot.crop(0, 0, 8, 8) == 50).all()


def test_a_look_waits_for_a_frame_shown_after_it_began():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500) + [(S_OK, 100_990, picture(60)), (S_OK, 101_008, picture(70))])
    g = grabber(clock, opener)
    g.shot(AREA)
    clock.now = 101.0
    shot = g.shot(AREA, after=101.0)  # the newest frame was shown before: the next one
    assert shot.t == 101.008 and (shot.crop(0, 0, 4, 4) == 70).all()
    assert opener.opened[0].timeouts[-2:] == [0, round(1000 * FRESH_WAIT_S)]


def test_with_nothing_shown_after_a_look_began_the_screen_is_the_frame_held():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500))
    g = grabber(clock, opener)
    g.shot(AREA)
    clock.now = 101.0
    shot = g.shot(AREA, after=101.0)
    assert shot.t == 100.5 and (shot.crop(0, 0, 4, 4) == 50).all()
    assert 101.0 + FRESH_WAIT_S <= clock.now < 101.0 + FRESH_WAIT_S + 0.002  # waited, but no longer


@pytest.mark.parametrize("hr", [DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_INVALID_CALL])
def test_a_lost_duplication_is_started_again(hr):
    clock = Clock()
    opener = Opener(clock, first_frames(100_500) + [(hr, 0, None)], first_frames(101_200, 80))
    g = grabber(clock, opener)
    g.shot(AREA)
    clock.now = 101.1
    shot = g.shot(AREA)  # the black first frame of the new duplication is skipped too
    assert shot.t == 101.2 and (shot.crop(0, 0, 4, 4) == 80).all()
    assert len(opener.opened) == 2 and opener.opened[0].closed
    assert g.counts["duplicated again"] == 1 and g.counts["failed"] == 0


def test_lost_again_right_away_it_falls_back_to_mss_and_tries_dxgi_later():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500) + [(DXGI_ERROR_ACCESS_LOST, 0, None)],
                    [(DXGI_ERROR_ACCESS_LOST, 0, None)], first_frames(115_000, 90))
    g = grabber(clock, opener)
    g.shot(AREA)
    shot = g.shot(AREA)
    assert isinstance(shot, capture.MssShot) and (shot.crop(0, 0, 4, 4) == 7).all()
    assert g.counts["failed"] == 1 and "0x887A0026" in g.error and g.dup is None and opener.opened[1].closed
    clock.now += RETRY_S - 1
    assert isinstance(g.shot(AREA), capture.MssShot) and len(opener.opened) == 2  # not yet
    clock.now += 1
    shot = g.shot(AREA)
    assert isinstance(shot, capture.DxgiShot) and (shot.crop(0, 0, 4, 4) == 90).all()


def test_a_duplication_that_cant_open_falls_back_to_mss():
    clock = Clock()
    opener = Opener(clock, CaptureError("DuplicateOutput failed: 0x80070005"), first_frames(111_000))
    g = grabber(clock, opener)
    assert isinstance(g.shot(AREA), capture.MssShot) and "0x80070005" in g.error
    clock.now += RETRY_S
    assert isinstance(g.shot(AREA), capture.DxgiShot) and g.counts == {
        "dxgi": 1, "mss": 1, "failed": 1, "duplicated again": 0}


def test_no_frame_after_duplicating_falls_back_to_mss():
    clock = Clock()
    opener = Opener(clock, [(S_OK, 0, picture(0))])  # only the black frame, then nothing
    g = grabber(clock, opener)
    start = clock.now
    assert isinstance(g.shot(AREA), capture.MssShot) and "no frame" in g.error
    assert clock.now - start == pytest.approx(FIRST_FRAME_S, abs=0.002)


def test_other_failures_fall_back_to_mss():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500) + [(0x887A0005, 0, None)])  # DXGI_ERROR_DEVICE_REMOVED
    g = grabber(clock, opener)
    g.shot(AREA)
    assert isinstance(g.shot(AREA), capture.MssShot) and "0x887A0005" in g.error


def test_mss_alone_never_duplicates():
    clock = Clock()
    g = grabber(clock, Opener(clock, CaptureError("must not be opened")), method="mss")
    shot = g.shot(AREA, after=clock.now)
    assert [b.shape for b in shot.crops([(0, 95, 300, 205), (707, 1064, 147, 1)])] == [(205, 300, 4), (1, 147, 4)]
    assert g._mss.grabs[1] == {"left": 707, "top": 1064, "width": 147, "height": 1} and g.counts["failed"] == 0


def test_a_window_across_two_monitors_is_grabbed_with_mss():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500))
    g = grabber(clock, opener)
    across = {"left": 500, "top": 0, "width": 1920, "height": 1080}  # its centre on the first monitor
    assert isinstance(g.shot(across), capture.MssShot) and g.counts["failed"] == 0
    assert isinstance(g.shot(AREA), capture.DxgiShot) and len(opener.opened) == 1


def test_a_window_moved_to_another_monitor_is_duplicated_there():
    clock = Clock()
    opener = Opener(clock, first_frames(100_500), first_frames(100_600, 40), desktop=(0, 0, 1920, 1080))
    g = grabber(clock, opener)
    g.shot(AREA)
    opener.desktop = (1920, 0, 3840, 1080)
    shot = g.shot({"left": 1920, "top": 0, "width": 1920, "height": 1080})
    assert isinstance(shot, capture.DxgiShot) and opener.points[-1] == (2880, 540) and opener.opened[0].closed
    assert (shot.crop(0, 0, 4, 4) == 40).all()


def test_boxes_are_cut_in_the_area_and_converted():
    clock = Clock()
    pic = np.arange(1080 * 1920 * 4, dtype=np.uint32).astype(np.uint8).reshape(1080, 1920, 4)
    opener = Opener(clock, [(S_OK, 0, picture(0)), (S_OK, 100_500, pic)])
    g = grabber(clock, opener)
    area = {"left": 100, "top": 50, "width": 800, "height": 600}
    a, b = g.shot(area).crops([(10, 20, 30, 40), (0, 0, 5, 1)], lambda v: v[:, :, :3].copy())
    assert np.array_equal(a, pic[70:110, 110:140, :3]) and np.array_equal(b, pic[50:51, 100:105, :3])


def test_a_shots_boxes_are_copied_before_the_next_shot():
    clock = Clock()
    g = grabber(clock, Opener(clock, first_frames(100_500)))
    old = g.shot(AREA)
    g.shot(AREA)
    with pytest.raises(RuntimeError, match="before the grabber's next shot"):
        old.crop(0, 0, 4, 4)


def test_a_shot_at_a_time_waits_till_then():
    clock = Clock()
    g = grabber(clock, Opener(clock, first_frames(100_500)))
    g.shot(AREA, at=100.25)
    assert clock.now >= 100.25


def test_present_times_are_on_monotonics_clock():
    # QueryPerformanceCounter read here (a clock, not the screen)
    freq = capture.qpc_frequency()
    offset = capture.qpc_offset(freq)
    assert abs(capture.qpc() / freq + offset - time.monotonic()) < 0.002


def test_tests_never_capture_the_real_screen():
    with pytest.raises(AssertionError, match="desktop duplication"):
        Grabber("dxgi").shot(AREA)
    with pytest.raises(AssertionError, match="mss"):
        Grabber("mss").shot(AREA).crop(0, 0, 4, 4)


@pytest.mark.parametrize("text,method", [("", "dxgi"), ("[capture]\n", "dxgi"), ('[capture]\nmethod = "mss"\n', "mss"),
                                         ('[capture]\nmethod = "dxgi"\n', "dxgi")])
def test_the_method_comes_from_config(tmp_path, text, method):
    (tmp_path / "config.toml").write_text(text)
    assert capture.configured_method(tmp_path / "config.toml") == method


def test_a_bad_method_in_config_is_refused(tmp_path):
    (tmp_path / "config.toml").write_text('[capture]\nmethod = "gdi"\n')
    with pytest.raises(ValueError, match="method must be one of"):
        capture.configured_method(tmp_path / "config.toml")
    assert capture.configured_method(tmp_path / "none.toml") == "dxgi"


def test_the_repos_config_says_dxgi():
    assert capture.configured_method() == "dxgi"


# --- Eyes on a grabber ---------------------------------------------------------------------------------

class RecordingScreen:
    """A screen whose shots record how they were asked for."""

    def __init__(self, clock: Clock):
        self.clock, self.asked = clock, []
        self.bgra = np.zeros((1080, 1920, 4), np.uint8)
        self.bgra[150:156, 100:106] = (40, 230, 240, 255)  # her yellow minimap dot (BGR)
        self.bgra[1062:1066, 707:800] = (30, 30, 220, 255)  # some HP left

    def shot(self, area, after=None, at=None):
        self.asked.append((after, at))
        if at is not None:
            self.clock.now = max(self.clock.now, at)
        from gamelab.analysis.bench import SavedShot
        return SavedShot(self.bgra, area)


class GameEyes(Eyes):
    def _client_area(self):
        return dict(AREA)

    def _to_front(self):
        pass


def test_a_glance_takes_both_boxes_from_one_shot_at_the_old_pace(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(eyes_module.time, "monotonic", clock)
    screen = RecordingScreen(clock)
    eyes = GameEyes("MapleStory", np.zeros((10, 36), np.uint8), detector=object(), screen=screen)
    dot, hp = eyes.glance()
    assert dot is not None and 0 < hp < 1 and len(screen.asked) == 1 and screen.asked[0][1] <= clock.now
    clock.now += 0.03  # the navigator's sleep between glances
    eyes.glance()
    assert screen.asked[1] == (None, pytest.approx(100.0 + GLANCE_EVERY_S))
    clock.now += 1.0  # a glance long after: at once
    begun = clock.now
    eyes.glance()
    assert screen.asked[2][1] <= begun


def test_a_look_asks_for_a_frame_shown_after_it_began(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(eyes_module.time, "monotonic", clock)
    screen = RecordingScreen(clock)
    eyes = GameEyes("MapleStory", np.zeros((10, 36), np.uint8), detector=object(), screen=screen)
    full = eyes._capture()
    assert screen.asked == [(100.0, None)] and full.shape == (1080, 1920, 3) and full.flags.c_contiguous
    assert eyes.seen_at is not None


def test_eyes_share_the_processs_grabber():
    a = Eyes("MapleStory", np.zeros((10, 36), np.uint8), detector=object())
    b = Eyes("MapleStory", np.zeros((10, 36), np.uint8), detector=object())
    assert a.screen is b.screen is capture.shared()


def test_bench_capture_times_both_ways_and_compares_their_pixels():
    from gamelab.analysis import bench
    clock = Clock()
    dxgi = grabber(clock, Opener(clock, [(S_OK, 0, picture(0)), (S_OK, 100_500, picture(7))]))
    gdi = grabber(clock, Opener(clock, CaptureError("must not be opened")), method="mss")
    timer = bench.Timer()
    bench.capture_stages(AREA, dxgi, gdi, timer)
    assert timer.counts == {"looks": 1, "... DXGI and mss the same on every pixel": 1,
                            "... pixels that differed (in all)": 0}
    assert {"dxgi: a look's capture (RGB)", "mss: a glance (two grabs)"} <= set(timer.samples)
