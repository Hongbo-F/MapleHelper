"""The continuous feed (localization.feed): the ring of frames in shared memory never holds more than its
slots, a look is never handed a frame from before it was asked for nor one that failed its checks, every event is
drained once, one pass a frame at the one rate (localization.rate.HZ), what's asked done on the next frame due (no
frame of its own: HZ a second whatever is asked), and Senses on it: every look and glance from it, standing still (LookLost) while it has nothing to act on, and going on after. The
perception service that runs it live is test_service's."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gamelab.localization import feed
from gamelab.localization import senses as senses_mod
from gamelab.localization.feed import Asks, Ring, SyncFeed, fresh, run
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import Schedule, WorldState, WorldStep
from gamelab.perception.frames import Frame, ListFrames
from gamelab.sim.fakes import FakeDetector

from test_worldstate import Drawn, pig, small_map

FPS = feed.HZ


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


def test_a_look_is_never_handed_a_frame_from_before_it_asked():
    want = Asks(after=10.0)
    assert not fresh(state(1, 9.99), want) and fresh(state(2, 10.0), want)
    assert not fresh(WorldState(3, 10.5, False), want)  # the game wasn't on screen
    # Her band and the whole screen are every state's (since 2026-10-30): asking for them asks for nothing more.
    for band in (Asks(after=10.0, band=True), Asks(after=10.0, wide=True)):
        assert fresh(state(4, 10.1, band_at=10.1, wide_at=10.1), band)
    loot = Asks(after=10.0, loot=True)
    assert not fresh(state(7, 10.1), loot) and fresh(state(8, 10.1, loot=[]), loot)
    assert not fresh(state(9, 10.1, loot=[]), Asks(after=10.0, loot_wide=True))
    assert not fresh(state(10, 10.1, loot=[], mesos_only=True), loot)  # it looked for mesos only
    assert fresh(state(11, 10.1, tag=None), loot)  # she isn't placed: as good as a look would be
    # A frame that failed its checks is never fresh, placed or not (it was served as an unplaced look, and the mover
    # tried 40 times, nudging the mouse, then stopped the command).
    from gamelab.perception.framecheck import Problem
    for kind in ("offset", "minimap", "hud", "picture", "tag", "place"):
        assert not fresh(state(12, 10.1, bad=Problem(kind, kind)), want)
        assert not fresh(state(12, 10.1, tag=None, where=None, bad=Problem(kind, kind)), Asks(after=10.0))


def test_a_look_asks_for_a_frame_at_its_time_and_the_drops_it_wants_on_it():
    """feed.to_ask: a frame shown at `after` or later and the drops wanted searched on it; never her band or the whole
    screen (every state has them: until 2026-10-30 they were asked for when the newest state's would be stale)."""
    ask = feed.to_ask(Asks(after=7.654, band=True, wide=True, loot=True, mesos_only=True))
    assert (ask.after, ask.band, ask.wide, ask.loot, ask.loot_wide, ask.mesos_only) == (7.654, False, False, True,
                                                                                         False, True)
    assert feed.adds(Asks(loot_wide=True), Asks(loot=True)) and not feed.adds(Asks(band=True), Asks())


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
    clock.sleep(0.15)
    asked = clock.now
    assert f.latest(Asks(asked - 0.07)) is None and clock.now - asked < 0.1 and f.problem() == problem
    assert f.glance(asked - 0.07, timeout=0.0) is None
    f._look = real
    clock.sleep(0.15)
    assert f.latest(Asks(clock.now - 0.07)) is not None and f.problem() is None


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


def test_the_sync_feed_looks_hz_times_a_second_each_state_its_pass_later(sync):
    """Nothing asked: a frame every 1/HZ s (every 4th of the game's 60 at 15), each state ready its pass's cost after
    its frame (Costs.of: the capture, her look and the whole screen at once, the bars, the strips of drops)."""
    f, clock, eyes = sync
    eyes.mobs = [pig(1100)]
    clock.sleep(2.0)
    frames = [st.t for _, st in f.states]
    assert f.counts["frames"] == round(2.0 * FPS) + 1 and f.counts["asked"] == 0
    assert all(round(b - a, 6) == round(1 / FPS, 6) for a, b in zip(frames, frames[1:]))
    c = f.costs
    for ready, st in f.states:
        assert ready == pytest.approx(st.t + c.of(st))
        assert c.of(st) == pytest.approx(c.capture + max(c.look, c.wide) + c.beside + c.bar * st.work["bars"]
                                         + c.loot_strip * st.work["strips"])
        assert [m.kind for m in st.monsters] == ["pig"] and st.band_at == st.wide_at == st.t


def test_a_frame_asked_for_is_the_next_frame_due_never_one_of_its_own(sync):
    """An ask (a look's: Senses.fresh_after, KEY_EFFECT_S after her key) waits for the frame due, wherever it falls
    between two (the user, 2026-10-30: "I meant everything at 15 fps"): the frames stay on their grid, HZ a second
    whatever is asked. Until the review of 2026-10-30 an ask well before the frame due had a frame of its own at once and
    the frames after re-phased to it: 15.7-15.9 a second in the harness's standard grind."""
    f, clock, eyes = sync
    t0 = clock.now
    clock.sleep(1.0)
    last = max(st.t for _, st in f.states)
    for k, early in enumerate((2, 3, 1)):  # 33, 50 and 17 ms after the frame before
        asked = last + early / feed.GAME_HZ
        clock.sleep(asked - clock.now - 1e-6)
        st = f.latest(Asks(asked, loot=True))
        assert st.t == pytest.approx(last + 1 / FPS) and st.loot is not None and f.counts["asked"] == k + 1
        last = st.t
    clock.sleep(0.5)
    frames = sorted(s.t for _, s in f.states)
    assert all(round(b - a, 6) == round(1 / FPS, 6) for a, b in zip(frames, frames[1:]))  # (on the grid throughout)
    assert f.counts["frames"] == int((clock.now - t0) * FPS + 1e-6) + 1  # (HZ a second, the asked ones among them)


def test_a_look_reads_every_monster_on_screen_off_any_frame(sync):
    """Every state has its own frame's whole screen (since 2026-10-30): a look waits for its frame's pass alone, never
    for a search (until then its band or the screen were searched on a thread, back their cost later)."""
    f, clock, eyes = sync
    eyes.mobs = [pig(1100), pig(1200, 300)]
    clock.sleep(0.5)
    c, waits = f.costs, []
    for _ in range(20):
        asked = clock.now
        st = f.latest(Asks(asked, band=True, wide=True))
        assert st is not None and st.t >= asked - 1e-9 and sorted(round(m.y) for m in st.monsters) == [300, 500]
        waits.append(clock.now - asked)
        clock.sleep(0.037)
    assert max(waits) <= 1 / FPS + 1 / feed.GAME_HZ + c.of(st) + 2 * c.bar + 1e-9  # (the frame due, its pass)


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
    assert st is not None and st.loot == [] and st.t - asked <= 1 / FPS + 1e-9  # (the next frame due)
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
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(drops_strips=0))
    frames = [np.zeros((4, 4, 3), np.uint8)] * 12

    class Eyes2:
        def look_at(self, full, t, ropes):
            return eyes.frame(t, ropes)
    runner = feed.Runner(ListFrames(frames, loop=False), Eyes2(), step)
    out, asks = [], {2: Asks(0.0, loot=True), 5: Asks(0.0, loot_wide=True)}  # asked once 2 (5) frames are out
    run(runner, 200.0, lambda wait: (asks.pop(len(out), None), False), lambda st, fr: out.append(st))
    assert [s.seq for s in out] == list(range(1, 13))
    assert [(s.seq, s.loot_wide) for s in out if s.loot is not None] == [(3, False), (6, True)]


def test_the_loop_takes_what_was_asked_on_the_frame_due_and_keeps_its_pace():
    """feed.run: at 10 a second, an ask for a frame 30 ms after the last one waits for the frame due (no frame of its
    own: HZ a second whatever is asked), which carries it; the frames keep their period."""
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(drops_strips=0))

    class Eyes2:
        def look_at(self, full, t, ropes):
            return eyes.frame(t, ropes)
    runner = feed.Runner(ListFrames([np.zeros((4, 4, 3), np.uint8)] * 6, loop=False), Eyes2(), step)
    out, sent = [], []

    def poll(wait):
        time.sleep(min(wait, 0.005))
        if len(out) == 2 and not sent:
            sent.append(Asks(out[-1] + 0.03, loot=True))
            return sent[-1], False
        return None, False
    took = []
    run(runner, 10.0, poll, lambda st, fr: (out.append(st.t), took.append(st.seq) if st.loot is not None else None))
    gaps = np.diff(out)
    assert len(out) == 6 and all(0.09 <= g <= 0.14 for g in gaps), gaps  # (the asked frame the one due: its period)
    assert took == [3]


def test_a_rate_gone_up_takes_from_the_next_frame():
    """The perception service's frame loop at its idle rate when a command that hunts attaches (Service._hello wakes its
    poll): the next frame comes the hunt's period after the last, not the idle one's; a rate gone down takes after the
    frame due."""
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(drops_strips=0))

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
    """A detector whose searches take `s` seconds (of the real clock)."""

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
    that draw a pig walking (built in the service's process, from this module); the whole screen in every pass, as the
    service's, each search taking `slow` s."""
    model = small_map()
    eyes = Drawn(model)
    if slow:
        eyes.detector = Slow(slow)
    senses = Senses(eyes, model, band=(100, 100))
    step = WorldStep(senses)
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
    a = senses.glance()
    first = senses.glanced_at
    b = senses.glance()
    assert senses.counts["glances"] == 2 and senses.glanced_at > first  # (the next frame: a glance never takes one twice)
    real = f.latest
    monkeypatch.setattr(f, "latest", lambda want, timeout=feed.FEED_WAIT_S, ask=True: (clock.sleep(timeout), None)[1])
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
    monkeypatch.setattr(f, "latest", lambda want, timeout=feed.FEED_WAIT_S, ask=True: (clock.sleep(timeout), None)[1])
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
    # Her memory takes the monsters off her band (Scene.swept), with the whole screen searched in every frame's pass:
    # every look says it searched her band and the screen.
    from gamelab.localization.memory import Memory
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    memory = Memory(f.step.model)
    senses.listeners.append(memory.on_look)
    eyes.mobs = [pig(1100), pig(1200, 300)]  # one in her band, one on the platform 200 above (off it)
    clock.sleep(0.25)
    s = senses.look()
    above = f.step.model.platform_under(1200, 300).id
    assert [m.x for m in s.monsters] == [pytest.approx(1100, abs=3)]
    assert [(m.x, m.platform) for m in s.swept] == [(pytest.approx(1200, abs=3), above)]
    assert len(s.rows) == 2 and s.rows[1][0] < s.rows[0][0] and s.rows[1][1] >= s.rows[0][1]  # her band, the screen
    assert len(memory.alive(s.t)) == 2
    for _ in range(20):
        clock.sleep(0.137)
        assert len(senses.look().rows) == 2


def test_glances_from_the_feed_come_with_every_frame_asking_for_none(sync, monkeypatch):
    """Senses.glance from the feed, a frame waited for before each (as the navigator glances): with FEED_GLANCE_EVERY_S
    0, every frame it looks at (HZ a second: 15, the pace glances had at 30 frames a second, GLANCE_EVERY_S 63 ms), each
    glance's frame time in `glanced_at`; no frame is asked for on their account (a frame's period between them would
    skip a frame shown a little early, and halve them)."""
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    clock.sleep(0.5)
    assert senses_mod.FEED_GLANCE_EVERY_S == 0.0
    senses.glance()
    times, asked = [], f.counts["asked"]
    for _ in range(8):
        senses.wait_frame(0.1)
        senses.glance()
        times.append(senses.glanced_at)
    assert [round(b - a, 4) for a, b in zip(times, times[1:])] == [round(1 / feed.HZ, 4)] * 7
    assert f.counts["asked"] == asked and senses.counts["glances"] == 9


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
