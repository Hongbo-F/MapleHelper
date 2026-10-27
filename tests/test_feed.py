"""The continuous feed (localization.feed): the ring of frames in shared memory never holds more than its
slots, a look is never handed a frame from before it was asked for nor one that failed its checks, every event is
drained once, what's asked is done on the next frame (its searches off the frame loop, back their cost later), and
Senses on it: every look and glance from it, standing still (LookLost) while it has nothing to act on, and going on
after. The perception service that runs it live is test_service's."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gamelab.localization import feed
from gamelab.localization import senses as senses_mod
from gamelab.localization.feed import Asks, Ring, SyncFeed, fresh, run
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import Schedule, SearchThread, WorldState, WorldStep
from gamelab.perception.frames import Frame, ListFrames
from gamelab.sim.fakes import FakeDetector

from test_worldstate import Drawn, pig, small_map

FPS = 30


def test_the_ring_holds_its_slots_frames_and_no_more():
    ring = Ring(slots=4, shape=(6, 8, 3))
    try:
        for seq in range(1, 11):
            assert ring.put(seq, 100.0 + seq, np.full((6, 8, 3), seq, np.uint8)) == seq % 4
        assert ring.held() == [7, 8, 9, 10]
        assert ring.get(10 % 4, 10)[0, 0, 0] == 10 and ring.get(6 % 4, 6) is None  # 6 was overwritten by 10
        assert ring.put(11, 0.0, np.zeros((5, 8, 3), np.uint8)) is None  # not its shape: not kept
        assert ring.slot_of(9) == 9 % 4 and ring.slot_of(6) is None and ring.slot_of(11) is None
        other = Ring(ring.name, slots=4, shape=(6, 8, 3))  # the child's side, by name
        assert other.get(9 % 4, 9)[0, 0, 0] == 9
        other.close()
    finally:
        ring.close(unlink=True)


def state(seq, t, **kw) -> WorldState:
    kw.setdefault("tag", (480, 308))
    kw.setdefault("where", (1000.0, 500.0, 0, (0.0, 0.0)))
    return WorldState(seq, t, True, **kw)


def test_a_look_is_never_handed_a_frame_from_before_it_asked(monkeypatch):
    want = Asks(after=10.0)
    assert not fresh(state(1, 9.99), want) and fresh(state(2, 10.0), want)
    assert not fresh(WorldState(3, 10.5, False), want)  # the game wasn't on screen
    band = Asks(after=10.0, band=True)
    assert fresh(state(4, 10.1, band_at=10.0), band) and not fresh(state(5, 10.4, band_at=10.05), band)
    # Reading the feed (ASK_ONLY_STALE, the default): any search of her band within MOBS_FRESH_S of the frame will do.
    assert fresh(state(4, 10.1, band_at=9.95), band) and not fresh(state(5, 10.1, band_at=9.75), band)
    monkeypatch.setattr(feed, "ASK_ONLY_STALE", False)  # asking for everything: the search it asked for, on a frame since
    assert not fresh(state(5, 10.1, band_at=9.95), band)
    assert not fresh(state(6, 10.1, band_at=10.1), Asks(after=10.0, wide=True))
    loot = Asks(after=10.0, loot=True)
    assert not fresh(state(7, 10.1), loot) and fresh(state(8, 10.1, loot=[]), loot)
    assert not fresh(state(9, 10.1, loot=[]), Asks(after=10.0, loot_wide=True))
    assert not fresh(state(10, 10.1, loot=[], mesos_only=True), loot)  # it looked for mesos only
    assert fresh(state(11, 10.1, tag=None), band)  # she isn't placed: as good as a look would be
    # A frame that failed its checks is never fresh, placed or not (it was served as an unplaced look, and the mover
    # tried 40 times, nudging the mouse, then stopped the command).
    from gamelab.perception.framecheck import Problem
    for kind in ("offset", "minimap", "hud", "picture", "tag", "place"):
        assert not fresh(state(12, 10.1, band_at=10.1, bad=Problem(kind, kind)), band)
        assert not fresh(state(12, 10.1, tag=None, where=None, bad=Problem(kind, kind)), Asks(after=10.0))


def test_a_look_asks_for_what_the_next_frame_would_have_stale_though_its_after_is_older():
    """A look's `after` is often before the newest frame (Senses.fresh_after: up to MAX_AGE_S before now). What it asks
    is judged by the next frame after the newest: the whole screen searched 0.5 s before the newest frame (WIDE_FRESH_S)
    is stale by the next, so it's asked, though a frame at `after` would have had it fresh. (Judged by `after` alone it
    went unasked: in the simulator a hunt's wide look waited FEED_WAIT_S for nothing and failed its checks.)"""
    newest = state(1, 7.733, wide_at=7.233, band_at=7.6)
    ask = feed.to_ask(Asks(after=7.654, wide=True), newest)
    assert ask.wide
    assert not feed.to_ask(Asks(after=7.654, wide=True), state(1, 7.733, wide_at=7.5)).wide  # fresh by the next
    assert feed.to_ask(Asks(after=7.654, band=True), state(1, 7.733, band_at=7.45)).band  # 0.3 s: MOBS_FRESH_S
    assert not feed.to_ask(Asks(after=7.654, band=True), newest).band
    assert feed.to_ask(Asks(after=8.0, band=True), newest).band  # (after the newest: by `after`, as before)


def test_a_bad_newest_state_gives_no_look_at_once_and_says_why(sync):
    # SyncFeed (and the service's client) as the frames fail their checks: no look given, at once (not after
    # FEED_WAIT_S), and `problem` says why; a glance has none either.
    from gamelab.perception.framecheck import Problem
    f, clock, eyes = sync
    clock.sleep(0.5)
    real = f._look
    problem = Problem("offset", "the capture is offset by (+26, +26) px")

    def look(ropes):
        obs = real(ropes)
        obs.bad, obs.tag, obs.dot = problem, None, None
        return obs
    f._look = look
    clock.sleep(0.1)
    asked = clock.now
    assert f.latest(Asks(asked - 0.05)) is None and clock.now - asked < 0.1 and f.problem() == problem
    assert f.glance(asked - 0.05, timeout=0.0) is None
    f._look = real
    clock.sleep(0.1)
    assert f.latest(Asks(clock.now - 0.05)) is not None and f.problem() is None


class Clock:
    """time.monotonic and time.sleep for the feed module, a frame of the game every 1/60 s calling `tick`."""

    def __init__(self, t: float = 100.0):
        self.now, self.tick, self.next = t, None, t

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        end = self.now + max(0.0, s)
        while self.next <= end + 1e-12:
            self.now = self.next
            if self.tick:
                self.tick(self.now)
            self.next += 1 / feed.GAME_HZ
        self.now = end


@pytest.fixture
def sync(monkeypatch):
    model = small_map()
    eyes = Drawn(model)
    clock = Clock()
    monkeypatch.setattr(feed, "time", clock)
    monkeypatch.setattr(senses_mod, "time", clock)  # (a look's freshness goes by it: Senses.fresh_after)
    step = WorldStep(Senses(eyes, model, band=(100, 100)))
    f = SyncFeed(step, lambda ropes: eyes.frame(clock.now, ropes))
    clock.tick = f.tick
    return f, clock, eyes


def test_the_sync_feed_hands_a_look_the_frame_after_it_asked_its_cost_later(sync, monkeypatch):
    monkeypatch.setattr(feed, "ASK_ONLY_STALE", False)  # a look asking for all it wants, its band searched on its frame
    f, clock, eyes = sync
    eyes.mobs = [pig(1100)]
    clock.sleep(0.5)
    c = feed.COSTS
    for _ in range(20):
        asked = clock.now
        st = f.latest(Asks(asked, band=True))
        # (a frame shown when asked counts, to 1e-9 s: the clock's 100.49999999999997 is 100.5's frame)
        assert st is not None and st.t >= asked - 1e-9 and fresh(st, Asks(asked, band=True))
        assert st.band_at >= asked  # her band searched on a frame shown after the look asked: its own search
        # ready only once worked out: the searched frame's step, then the search on its thread
        assert clock.now >= st.band_at + c.capture + c.look + c.band - 1e-9
        assert clock.now - asked < 0.25  # (at worst a search of the whole screen was out, 0.11 s)
        clock.sleep(0.037)
    assert f.counts["frames"] >= 30  # it went on looking between the looks, 30 times a second


def test_a_look_reads_the_feeds_world_state_without_waiting_for_a_search_of_its_own(sync):
    # Reading (ASK_ONLY_STALE, the default): the next frame placing her, its monsters those the feed follows (re-found in
    # it) and a search of her band within MOBS_FRESH_S; it waits for that frame's step alone, not for a search.
    f, clock, eyes = sync
    eyes.mobs = [pig(1100)]
    clock.sleep(0.5)  # (the pig found, and followed since)
    c, waits, searched = feed.COSTS, [], 0
    for _ in range(20):
        asked = clock.now
        st = f.latest(Asks(asked, band=True))
        # (a frame shown when asked counts, to 1e-9 s: the clock's 100.49999999999997 is 100.5's frame)
        assert st is not None and st.t >= asked - 1e-9 and fresh(st, Asks(asked, band=True))
        assert [m.kind for m in st.monsters] == ["pig"] and st.t - st.band_at <= feed.MOBS_FRESH_S + 1e-9
        waits.append(clock.now - asked)
        searched += st.band_at >= asked
        clock.sleep(0.037)
    # The next game frame (1/60 s at most) and its step (capture, her place, the pig re-found), now and then a frame's
    # more while a search was out; never the band's own search on top (c.band, 26 ms), as asking had it.
    assert max(waits) <= 2 / feed.GAME_HZ + c.capture + c.look + c.refind + c.beside + 1e-9
    assert sorted(waits)[len(waits) // 2] < c.capture + c.look + c.band
    assert searched < 20  # most looks read a search from before they asked


def test_the_sync_feeds_searches_come_back_their_cost_later_merged_as_of_their_frame(sync, monkeypatch):
    monkeypatch.setattr(feed, "ASK_ONLY_STALE", False)  # (the look waits for the search it asked for)
    f, clock, eyes = sync
    clock.sleep(0.5)  # (the first frame's search of the whole screen is back)
    eyes.mobs = [pig(1100)]
    asked = clock.now
    st = f.latest(Asks(asked, band=True))
    c = feed.COSTS
    searched = [(ready, s) for ready, s in f.states if s.late and s.t >= asked]
    assert searched and searched[0][1].work["band"] == 1
    ready, late = searched[0]
    back = late.t + c.capture + c.look + c.band  # its frame's step, then the search on its thread
    # Merged as it came (the frame loop idle), or by the frame it was on (placing her, then the re-finds)
    assert back - 1e-9 <= ready <= back + c.capture + c.look + 2 * c.refind + 1e-9
    assert [m.kind for m in late.monsters] == ["pig"] and [e.kind for e in late.events] == ["appeared"]
    assert clock.now == pytest.approx(ready) and st.band_at == late.t  # the look had waited for it
    clock.sleep(0.1)
    # The frames stepped while it was out don't have the pig; the first after it does, re-found
    between = [s for r, s in f.states if not s.late and late.t < s.t and r < ready - 1e-9]
    after = [s for r, s in f.states if not s.late and s.t > late.t and r >= ready - 1e-9]
    assert all(s.monsters == [] and (s.band_at or 0) < late.t for s in between)
    assert after and [m.kind for m in after[0].monsters] == ["pig"] and after[0].band_at == late.t
    (tr,) = f.step.tracker.tracks.values()
    assert tr.first_seen == late.t


def test_every_event_is_drained_once(sync):
    f, clock, eyes = sync
    eyes.mobs = [pig(1100)]
    clock.sleep(0.5)
    eyes.exp = 10
    clock.sleep(0.2)
    first = f.drain_events()
    assert [e.kind for e in first] == ["appeared", "exp_gain"]
    clock.sleep(0.5)
    assert f.drain_events() == []


def test_what_a_look_asks_is_searched_on_the_next_frame(sync):
    f, clock, eyes = sync
    clock.sleep(0.3)
    eyes.drops = []
    asked = clock.now
    st = f.latest(Asks(asked, loot=True))
    assert st is not None and st.loot == [] and st.t - asked <= 2 / feed.GAME_HZ + 1e-9  # the next it could take
    assert f.counts["asked"] == 1


def test_asks_waiting_together_keep_what_each_wants_and_the_newest_mesos_only(sync):
    older, newer = Asks(10.0, band=True, loot=True), Asks(10.2, loot=True, mesos_only=True)
    merged = older.merge(newer)
    assert (merged.after, merged.band, merged.loot, merged.mesos_only) == (10.0, True, True, True)
    assert not newer.merge(Asks(10.3, loot=True)).mesos_only  # her inventory freed up meanwhile
    # The sync feed merges a look's ask into the one still waiting, the newer last.
    f, clock, eyes = sync
    f.asks = Asks(clock.now + 1.0, loot=True)
    assert f.latest(Asks(clock.now + 1.0, loot=True, mesos_only=True), timeout=0.0) is None
    assert f.asks.loot and f.asks.mesos_only


def test_the_loop_does_what_was_asked_on_the_first_frame_at_its_time():
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(band_s=None, wide_s=None))
    frames = [np.zeros((4, 4, 3), np.uint8)] * 12

    class Eyes2:
        def look_at(self, full, t, ropes):
            return eyes.frame(t, ropes)
    runner = feed.Runner(ListFrames(frames, loop=False), Eyes2(), step)
    out, asks = [], {2: Asks(0.0, band=True), 5: Asks(0.0, wide=True)}  # asked once 2 (5) frames are out
    run(runner, 200.0, lambda wait: (asks.pop(sum(not s.late for s in out), None), False),
        lambda st, fr: out.append(st))
    assert [s.seq for s in out if not s.late] == list(range(1, 13))
    late = [s for s in out if s.late]  # each frame searched, given again as its searches came back
    assert [s.seq for s in late if s.work["band"]] == [3] and [s.seq for s in late if s.work["wide"]] == [6]
    assert out.index(late[0]) < out.index(next(s for s in out if s.seq == 4))  # back before the next frame's


def test_a_rate_gone_up_takes_from_the_next_frame():
    """The perception service's frame loop at its idle rate when a command that hunts attaches (Service._hello wakes its
    poll): the next frame comes the hunt's period after the last, not the idle one's; a rate gone down takes after the
    frame due."""
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(band_s=None, wide_s=None))

    class Eyes2:
        def look_at(self, full, t, ropes):
            return eyes.frame(t, ropes)
    runner = feed.Runner(ListFrames([np.zeros((4, 4, 3), np.uint8)] * 7, loop=False), Eyes2(), step)
    out, rate = [], {"hz": 2.0}

    def emit(st, fr):
        out.append(st.t)
        rate["hz"] = {2: 20.0, 5: 4.0}.get(len(out), rate["hz"])  # a hunt attached after the 2nd frame, gone after the 5th

    def poll(wait):  # woken every 10 ms, as the service's poll is by a hello
        time.sleep(min(wait, 0.01))
        return None, False
    run(runner, lambda: rate["hz"], poll, emit)
    gaps = np.diff(out)
    assert len(out) == 7 and gaps[0] >= 0.45 and all(g < 0.15 for g in gaps[1:5]), gaps
    assert gaps[5] >= 0.2 and gaps[4] < 0.15, gaps  # (gone down: the frame due at the old rate first)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' priority classes")
def test_a_worker_started_at_idle_priority_stays_there_and_one_at_normal_goes_below_it():
    """feed.quiet (the harness's workers): a background job started at idle (D:/MapleHelper-scratch/cpu/hunt/
    bg_quiet.cmd, every child inheriting it) had its workers go up to below normal."""
    import subprocess
    code = ("import ctypes; from gamelab.localization import feed; feed.quiet(); k = ctypes.windll.kernel32; "
            "k.GetCurrentProcess.restype = ctypes.c_void_p; k.GetPriorityClass.argtypes = (ctypes.c_void_p,); "
            "print(k.GetPriorityClass(k.GetCurrentProcess()))")
    classes = []
    for start in (feed.IDLE_CLASS, 0x20):  # (IDLE_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                             creationflags=start, cwd=Path(__file__).parent.parent)
        classes.append(int(out.stdout.split()[-1]))
    assert classes == [feed.IDLE_CLASS, feed.BELOW_NORMAL]


def test_frames_the_game_isnt_showing_say_so_and_are_not_read():
    model = small_map()
    step = WorldStep(Senses(Drawn(model), model, band=(100, 100)))

    class Hidden:
        def next(self, after):
            return None if self.n == 0 else self._one()

        def _one(self):
            self.n -= 1
            return Frame(None, time.monotonic(), False, why="the game isn't in front")
    hidden = Hidden()
    hidden.n = 3
    out = []
    run(feed.Runner(hidden, None, step), 30.0, lambda wait: (None, False), lambda st, fr: out.append(st))
    assert [(s.visible, s.why) for s in out] == [(False, "the game isn't in front")] * 3


class _Look:
    """The drawn frames' eyes, as the service's Runner needs them (look_at a frame), with a pig walking."""

    def __init__(self, eyes):
        self.eyes, self.t0 = eyes, None

    def look_at(self, full, t, ropes):
        self.t0 = self.t0 if self.t0 is not None else t
        self.eyes.mobs = [pig(900 + 60 * (t - self.t0))]
        return self.eyes.frame(t, ropes)


class Slow(FakeDetector):
    """A detector whose searches take `s` seconds (of the real clock); re-finds are as quick as ever."""

    def __init__(self, s: float):
        super().__init__()
        self.s = s

    def detect(self, frame, rows=None):
        time.sleep(self.s)
        return super().detect(frame, rows)

    def detect_wide(self, frame, rows, band=120, step=90):
        time.sleep(self.s)
        return super().detect_wide(frame, rows, band, step)


def drawn_runner(n: int = 0, slow: float = 0.0) -> feed.Runner:
    """A service's runner for its tests (test_service): a list of blank frames (a stand-in for the screen), read by eyes
    that draw a pig walking (built in the service's process, from this module); its searches on their thread, as the
    service's, each taking `slow` s."""
    model = small_map()
    eyes = Drawn(model)
    if slow:
        eyes.detector = Slow(slow)
    senses = Senses(eyes, model, band=(100, 100))
    step = WorldStep(senses, searches=SearchThread(senses))
    return feed.Runner(ListFrames([np.zeros((4, 4, 3), np.uint8)] * 3), _Look(eyes), step)


def test_screen_frames_never_bring_the_game_forward_and_say_when_it_isnt_showing(monkeypatch):
    from gamelab import window
    from gamelab.analysis import bench
    from gamelab.perception import frames as frames_module
    from gamelab.perception.frames import ScreenFrames
    game = {"front": "MapleStory", "shows": True, "size": (1920, 1080), "running": True, "iconic": False,
            "on": True, "at": (0, 0), "drag": None}

    def find_game(title):
        if not game["running"]:
            raise RuntimeError("no window")
        return 7

    def client_rect(hwnd):
        rect = {"left": game["at"][0], "top": game["at"][1], "width": game["size"][0], "height": game["size"][1]}
        if game["drag"] is not None:  # read again after the shot: moved meanwhile
            game["at"], game["drag"] = game["drag"], None
        return rect
    monkeypatch.setattr(window, "find_game", find_game)
    monkeypatch.setattr(window, "minimized", lambda hwnd: game["iconic"])
    monkeypatch.setattr(window, "client_rect", client_rect)
    monkeypatch.setattr(window, "on_screen", lambda area: game["on"])
    monkeypatch.setattr(window, "foreground_title", lambda: game["front"])
    monkeypatch.setattr(window, "shows", lambda hwnd, area, grid=None: game["shows"])
    # window.bring_to_front raises here (tests/conftest.py): a frame source that called it would fail
    screen = bench.SavedScreen()
    screen.bgra = np.zeros((1200, 2000, 4), np.uint8)  # (the desktop: the window moves on it)
    said = []
    frames = ScreenFrames("MapleStory", grabber=screen, log=said.append)
    f = frames.next(time.monotonic())
    assert f.visible and f.full.shape == (1080, 1920, 3) and f.problem is None
    for change, why, kind in ((("front", "Notepad"), "in front", "window"), (("shows", False), "another window", "window"),
                              (("size", (1280, 720)), "1280x720", "size"), (("running", False), "isn't running", "gone"),
                              (("iconic", True), "minimized", "minimized"), (("on", False), "off the screen",
                                                                            "offscreen")):
        before = game[change[0]]
        game[change[0]] = change[1]
        f = frames.next(time.monotonic())
        assert not f.visible and f.full is None and why in f.why and f.problem.kind == kind, (change, f)
        game[change[0]] = before
    # The window dragged while it was captured: not read (its picture isn't where the window is); the next frame is,
    # and the move is said once.
    game["drag"] = (26, 26)
    f = frames.next(time.monotonic())
    assert not f.visible and f.problem.kind == "moved" and "(0, 0) -> (26, 26)" in f.why
    assert frames.next(time.monotonic()).visible and said == ["  the game window moved: (0, 0) -> (26, 26)"]
    # A frame the capture held for STALE_S or more: the duplication stopped handing over frames.
    old, real_shot = time.monotonic() - frames_module.framecheck.STALE_S - 1.0, screen.shot
    monkeypatch.setattr(screen, "shot", lambda area, after=None, at=None: _aged(real_shot(area), old))
    f = frames.next(time.monotonic())
    assert not f.visible and f.problem.kind == "stale"


def _aged(shot, t):
    shot.t = t
    return shot


def test_senses_look_and_glance_through_the_feed_and_stand_still_while_it_has_nothing(sync, monkeypatch):
    # Every look and glance is the feed's (no eyes' look of their own); while it has nothing to act on (here: down),
    # a look is asked again RETRY_GAP_S apart and raises LookLost after LOST_S, a glance has no dot and the last HP,
    # and raises LookLost after GLANCE_LOST_S; once it's back, the look passes, said so.
    from gamelab.perception.eyes import GLANCE_EVERY_S
    from gamelab.perception.framecheck import LookLost
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    said = []
    senses.log = said.append
    eyes.mobs = [pig(1100)]
    clock.sleep(0.5)
    asked = clock.now
    s = senses.look()
    assert s.t >= asked - senses_mod.MAX_AGE_S and s.searched and [m.kind for m in s.monsters] == ["pig"]
    assert [e.kind for e in s.events] == ["appeared"]  # what the feed saw since: once
    assert senses.look(mobs=False).events == [] and getattr(eyes, "looks", 0) == 0
    a, b = senses.glance(), senses.glance()
    assert senses.counts["glances"] == 2 and senses._glanced - asked >= GLANCE_EVERY_S - 1e-9
    real = f.latest
    monkeypatch.setattr(f, "latest", lambda want, timeout=feed.FEED_WAIT_S: (clock.sleep(timeout), None)[1])
    hp = a[1]
    assert senses.glance() == (None, hp)  # (0.25 s waited: no dot, the last HP)
    with pytest.raises(LookLost):
        senses.glance()  # none for GLANCE_LOST_S: a chain running is cut (control.navigator)
    lost = clock.now
    with pytest.raises(LookLost):
        senses.look()
    assert clock.now - lost >= senses_mod.LOST_S and getattr(eyes, "looks", 0) == 0
    assert said[0].startswith("  a look failed its checks: ")
    monkeypatch.setattr(f, "latest", real)
    back = senses.regain(LookLost(f.problem() or senses_mod.Problem("perception", "down")), clock.now + 5.0)
    assert back is not None and said[-1].startswith("  the game looks as expected again")
    assert senses.failed["lost"] == 1


def test_a_command_standing_still_has_the_game_brought_forward_after_lost_s_too(sync, monkeypatch):
    # Every look without a state to act on has the feed bring the game forward (the service's client does it once a second
    # at most, and never for what only the user sets right), after LOST_S too: a hunt standing still on "the game isn't in
    # front" (Senses.regain) brings it forward again. Before, LookLost was raised first: once looks had failed for LOST_S,
    # none did.
    from gamelab.perception.framecheck import LookLost, Problem
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    senses.log = lambda m: None
    behind = Problem("window", "the game isn't in front")
    fronted = []
    monkeypatch.setattr(f, "latest", lambda want, timeout=feed.FEED_WAIT_S: (clock.sleep(timeout), None)[1])
    monkeypatch.setattr(f, "problem", lambda: behind)
    monkeypatch.setattr(f, "bring_forward", fronted.append)
    with pytest.raises(LookLost):
        senses.look()
    first = len(fronted)
    assert first >= 3 and set(fronted) == {behind} and first == senses.failed["looks"]  # every look that failed
    with pytest.raises(LookLost):
        senses.look()  # at once now: the game brought forward first all the same
    assert len(fronted) == first + 1
    assert senses.regain(LookLost(behind), clock.now + 3.0) is None
    assert len(fronted) == senses.failed["looks"] >= first + 3  # (each of regain's looks, PAUSE_POLL_S apart)


def test_a_look_is_the_newest_frame_newer_than_the_last_looks_and_after_her_keys_effect(sync, monkeypatch):
    # Senses.fresh_after: (a) never the last look's frame again, (b) shown KEY_EFFECT_S after her last key at the
    # soonest, (c) at most MAX_AGE_S old: a look after a pause takes the newest frame in, without waiting for another.
    # KEYS_WAIT "moves" or "none": a look that isn't the mover's doesn't wait for the keys' effect.
    from gamelab.localization import senses as senses_mod
    f, clock, eyes = sync
    monkeypatch.setattr(senses_mod, "time", clock)
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(1.0)
    asked = clock.now
    first = senses.look(mobs=False)
    assert clock.now == asked and asked - senses_mod.MAX_AGE_S <= first.t <= asked  # (c): the newest in, no wait
    again = senses.look(mobs=False)
    assert again.t > first.t and clock.now > asked  # (a): the next frame, waited for
    keyed = clock.now
    senses.keyed = lambda: keyed  # a key just went
    after_keys = senses.look(mobs=False)
    assert after_keys.t >= keyed + senses_mod.KEY_EFFECT_S - 1e-9  # (b): what came of it shows
    assert clock.now - keyed < senses_mod.KEY_EFFECT_S + 2 / feed.HZ  # ... the frame after that, no later
    clock.sleep(0.5)
    asked = clock.now
    assert senses.look(mobs=False).t >= asked - senses_mod.MAX_AGE_S and clock.now == asked  # long after: (c) again
    monkeypatch.setattr(senses_mod, "KEYS_WAIT", "moves")
    keyed = clock.now
    newest = senses.look(mobs=False)
    assert newest.t < keyed + senses_mod.KEY_EFFECT_S and clock.now - keyed < 2 / feed.HZ  # no wait for it
    assert senses.fresh_after(keys=True) >= keyed + senses_mod.KEY_EFFECT_S  # (the mover's still does: Mover.locate)


def test_a_look_for_monsters_after_a_throw_at_one_is_of_the_frame_that_shows_its_kill(sync, monkeypatch):
    # Senses.shows (the hunter's: Attacks.shown_by, KILL_SHOWN_S after her last throw at a monster's key): a look for
    # monsters, the next throw decided on it, is of a frame shown then, where that's SHOWS_WAIT_S off at most (she's
    # locked until 0.79 s after the key); a look without monsters, or one asked for a frame of its own (`after`), isn't.
    f, clock, eyes = sync
    monkeypatch.setattr(senses_mod, "time", clock)
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(1.0)
    keyed = clock.now
    senses.keyed = lambda: keyed
    senses.shows = lambda: keyed + 0.4  # a throw at a monster keyed just now
    assert senses.look(mobs=False).t < keyed + 0.4  # (her HP: no wait)
    s = senses.look()
    assert s.t >= keyed + 0.4 - 1e-9 and clock.now - keyed < 0.4 + 2 / feed.HZ  # its frame, no later
    keyed = clock.now  # another throw
    assert senses.look(after=keyed).t < keyed + 2 / feed.HZ  # (a look asked of a frame of its own: as asked)
    keyed = clock.now
    senses.shows = lambda: keyed + senses_mod.SHOWS_WAIT_S + 0.1  # too far off: not waited for
    assert senses.look().t < keyed + senses_mod.KEY_EFFECT_S + 2 / feed.HZ


def test_the_movers_look_after_a_move_sees_what_her_keys_did(sync, monkeypatch):
    # Mover.locate (placing her after a move) is of a frame shown KEY_EFFECT_S after her last key, KEYS_WAIT "moves" too.
    from types import SimpleNamespace

    from gamelab.control.mover import Mover
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(1.0)
    keyed = clock.now
    senses.keyed = lambda: keyed
    monkeypatch.setattr(senses_mod, "KEYS_WAIT", "moves")
    s = Mover(senses, SimpleNamespace(nudge_mouse=lambda size: None), f.step.model).locate()
    assert s.t >= keyed + senses_mod.KEY_EFFECT_S - 1e-9 and s.where is not None


def test_her_last_key_is_her_last_plans_end_less_what_it_ran_on_after_it(monkeypatch):
    # Actions.last_key: a plan played (Actions.play) ends with its last key, as its result is read; a navigator chain's
    # runs on past its last key to her landing (end_ms), and cut short every key is let go then.
    from types import SimpleNamespace

    from gamelab.control import actions as actions_mod, client, navigator
    from gamelab.control.actions import Actions
    now = [50.0]
    monkeypatch.setattr(actions_mod, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda s: None))
    monkeypatch.setattr(client, "submit", lambda *a, **k: now.__setitem__(0, now[0] + 0.4))
    monkeypatch.setattr(Actions, "hold_back", lambda self, events: None)
    a = Actions(SimpleNamespace(), skill=False)
    assert a.last_key < 0
    a.play([[0, "left", "down"], [400, "left", "up"]], "walk")
    assert a.last_key == 50.4
    nav = navigator.Navigator.__new__(navigator.Navigator)
    nav.actions, nav._tails = a, {"chain": 0.3}
    now[0] = 60.0
    nav._ended("chain")
    assert a.last_key == pytest.approx(59.7)
    nav._tails["cut"] = 0.3
    now[0] = 61.0
    nav._ended("cut", cut=True)
    assert a.last_key == 61.0 and nav._tails == {}


def test_a_look_after_a_glance_saw_her_still_takes_that_frame_and_waits_for_none(sync):
    # Navigator.recover: once a glance from the feed has seen her standing still, the look placing her may be of that
    # very frame (Senses.look's `after`); the next look is of a newer frame (Senses.fresh_after).
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(0.5)
    senses.glance()
    seen, asked = senses.glanced_at, clock.now
    s = senses.look(mobs=False, after=seen)
    assert s.t == seen and clock.now == asked and getattr(eyes, "looks", 0) == 0
    s = senses.look(mobs=False)
    assert s.t > seen and clock.now > asked


def test_a_band_look_from_the_feed_has_what_it_follows_off_her_band(sync):
    # Her memory takes what the feed follows off her band (Scene.swept), with the whole screen as searched while the
    # feed's last search of it is fresh.
    from gamelab.localization.memory import Memory
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    memory = Memory(f.step.model)
    senses.listeners.append(memory.on_look)
    eyes.mobs = [pig(1100), pig(1200, 300)]  # one in her band, one on the platform 200 above (off it)
    clock.sleep(1.25)  # (the whole screen searched at 101.03, back by 101.2: the one above followed since)
    s = senses.look()
    above = f.step.model.platform_under(1200, 300).id
    assert [m.x for m in s.monsters] == [pytest.approx(1100, abs=3)]
    assert [(m.x, m.platform) for m in s.swept] == [(pytest.approx(1200, abs=3), above)]
    assert len(s.rows) == 2 and s.rows[1][0] < s.rows[0][0] and s.rows[1][1] >= s.rows[0][1]  # her band, the screen
    assert len(memory.alive(s.t)) == 2
    clock.sleep(0.4)  # the whole screen's search over 0.6 s old now: her band's rows alone
    assert len(senses.look().rows) == 1


def test_glances_from_the_feed_come_as_often_as_its_frames_or_63_ms_apart(sync, monkeypatch):
    """Senses.glance from the feed, a frame waited for before each (as the navigator glances): with
    FEED_GLANCE_EVERY_S 0, every frame it looks at (30 a second), with GLANCE_EVERY_S every other one; each glance's
    frame time in `glanced_at`. At the feed's rate no frame is asked for on their account; 63 ms apart each glance
    asks for its frame, the next one coming too soon."""
    from gamelab.localization import senses as senses_mod
    from gamelab.perception.eyes import GLANCE_EVERY_S
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(0.5)
    for every, apart in ((0.0, 1 / feed.HZ), (GLANCE_EVERY_S, 2 / feed.HZ)):
        monkeypatch.setattr(senses_mod, "FEED_GLANCE_EVERY_S", every)
        senses.glance()
        times, asked = [], f.counts["asked"]
        for _ in range(8):
            senses.wait_frame(0.03)
            senses.glance()
            times.append(senses.glanced_at)
        assert [round(b - a, 4) for a, b in zip(times, times[1:])] == [round(apart, 4)] * 7
        assert f.counts["asked"] - asked == (0 if every == 0.0 else 8)
    assert senses.counts["glances"] == 18


def test_looks_from_the_feed_teach_the_stacks_camera(sync):
    # Placed by the feed's own step, a look skips Senses' placing; its camera learns from it all the same (where it
    # keeps her: Senses.place on the same look would have learned the same).
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    alone = Senses(eyes, f.step.model, band=(100, 100))
    clock.sleep(0.5)
    before = senses.camera.offset
    s = senses.look(mobs=False)
    assert s.tag_seen and senses.counts["looks"] == 1 and senses.camera.offset != before
    alone.place(eyes.look())  # the same look placed here: the camera learns the same
    assert alone.camera.offset == pytest.approx(senses.camera.offset)


def test_a_look_takes_the_feeds_events_up_to_its_frame_and_the_later_ones_go_with_the_next(sync, monkeypatch):
    # A look may be handed a state older than the newest (the frame searched for drops): an EXP gain of a later
    # frame goes with the next look, whose monsters no longer show the one it killed.
    from dataclasses import replace

    from gamelab.localization.tracking import Event
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(0.5)
    s = senses.look(mobs=False)
    t = s.t
    f.events.extend([Event("exp_gain", t + 0.4), Event("hp_drop", t + 0.5), Event("exp_gain", t + 0.6)])
    served = iter([replace(s, t=t + 0.5), replace(s, t=t + 0.7)])
    monkeypatch.setattr(f, "look", lambda *a: next(served))
    assert [(e.kind, e.t) for e in senses.look().events] == [("exp_gain", t + 0.4), ("hp_drop", t + 0.5)]
    assert [(e.kind, e.t) for e in senses.look().events] == [("exp_gain", t + 0.6)]
