"""The continuous feed (localization.feed): the ring of frames in shared memory never holds more than its
slots, a look is never handed a frame from before it was asked for, every event is drained once, what's
asked is done on the next frame, and the child process itself, started and stopped, with a list of frames
for a screen. When the child dies, the stack's side says so and hands out nothing more."""

import functools
import sys
import time

import numpy as np
import pytest

from gamelab.localization import feed
from gamelab.localization.feed import Asks, FeedClient, Ring, SyncFeed, fresh, run
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import Schedule, WorldState, WorldStep
from gamelab.perception.frames import Frame, ListFrames

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
    band = Asks(after=10.0, band=True)
    assert fresh(state(4, 10.1, band_at=9.95), band) and not fresh(state(5, 10.1, band_at=9.8), band)
    assert not fresh(state(6, 10.1, band_at=10.1), Asks(after=10.0, wide=True))
    loot = Asks(after=10.0, loot=True)
    assert not fresh(state(7, 10.1), loot) and fresh(state(8, 10.1, loot=[]), loot)
    assert not fresh(state(9, 10.1, loot=[]), Asks(after=10.0, loot_wide=True))
    assert not fresh(state(10, 10.1, loot=[], mesos_only=True), loot)  # it looked for mesos only
    assert fresh(state(11, 10.1, tag=None), band)  # she isn't placed: as good as a look would be


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
    step = WorldStep(Senses(eyes, model, band=(100, 100)))
    f = SyncFeed(step, lambda ropes: eyes.frame(clock.now, ropes))
    clock.tick = f.tick
    return f, clock, eyes


def test_the_sync_feed_hands_a_look_the_frame_after_it_asked_its_cost_later(sync):
    f, clock, eyes = sync
    eyes.mobs = [pig(1100)]
    clock.sleep(0.5)
    for _ in range(20):
        asked = clock.now
        st = f.latest(Asks(asked, band=True))
        assert st is not None and st.t >= asked and fresh(st, Asks(asked, band=True))
        assert clock.now >= st.t + feed.COSTS.of(st) - 1e-9  # ready only once it's been worked out
        assert clock.now - asked < 0.2  # (at worst the frame searched the whole screen, 0.14 s)
        clock.sleep(0.037)
    assert f.counts["frames"] >= 30  # it went on looking between the looks, 30 times a second


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


def test_the_loop_does_what_was_asked_on_the_first_frame_at_its_time():
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(band_every=1000, wide_s=None))
    frames = [np.zeros((4, 4, 3), np.uint8)] * 12

    class Eyes2:
        def look_at(self, full, t, ropes):
            return eyes.frame(t, ropes)
    runner = feed.Runner(ListFrames(frames, loop=False), Eyes2(), step)
    out, asks = [], {2: Asks(0.0, band=True), 5: Asks(0.0, wide=True)}  # asked once 2 (5) frames are out
    run(runner, 200.0, lambda wait: (asks.pop(len(out), None), False), lambda st, fr: out.append(st))
    assert len(out) == 12
    assert [s.seq for s in out if s.work["band"]] == [3] and [s.seq for s in out if s.work["wide"]] == [6]


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


def test_the_stacks_side_asks_and_waits_and_gives_up(monkeypatch):
    client = FeedClient(lambda: None, start=False)
    client.alive = True
    sent = []
    monkeypatch.setattr(client, "_send", lambda *msg: sent.append(msg))
    now = time.monotonic()
    client.states.append(state(1, now - 1.0))
    assert client.latest(Asks(now, band=True), timeout=0.05) is None
    assert sent and sent[0][0] == "ask" and sent[0][1].band
    client.states.append(state(2, now + 0.01, band_at=now))
    assert client.latest(Asks(now, band=True), timeout=0.05).seq == 2
    client.states.append(WorldState(3, now + 0.02, False, why="the game isn't in front"))
    t0 = time.monotonic()
    assert client.latest(Asks(now + 0.015), timeout=1.0) is None and time.monotonic() - t0 < 0.5
    client.alive = False
    assert client.latest(Asks(0.0), timeout=1.0) is None


# --- the child process itself ----------------------------------------------------------------------

class _Look:
    """The drawn frames' eyes, as the child's Runner needs them (look_at a frame), with a pig walking."""

    def __init__(self, eyes):
        self.eyes, self.t0 = eyes, None

    def look_at(self, full, t, ropes):
        self.t0 = self.t0 if self.t0 is not None else t
        self.eyes.mobs = [pig(900 + 60 * (t - self.t0))]
        return self.eyes.frame(t, ropes)


def drawn_runner(n: int = 0) -> feed.Runner:
    """The child's runner for the smoke test: a list of blank frames (a stand-in for the screen), read by eyes
    that draw a pig walking (built in the child, from this module)."""
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)))
    return feed.Runner(ListFrames([np.zeros((4, 4, 3), np.uint8)] * 3), _Look(eyes), step)


def test_the_child_runs_below_normal_and_hands_out_states_frames_and_events():
    client = FeedClient(functools.partial(drawn_runner, 0), hz=60.0)
    try:
        assert client.alive, client.why
        if sys.platform == "win32":
            assert client.hello["priority"] == feed.BELOW_NORMAL
        after = time.monotonic()
        st = client.latest(Asks(after, band=True), timeout=5.0)
        assert st is not None and st.t >= after and st.placed
        assert [m.kind for m in st.monsters] == ["pig"]
        assert client.frame_of(st).shape == feed.FRAME_SHAPE
        assert client.wait_frame(1.0)
        time.sleep(0.3)
        first = client.drain_events()
        assert [e.kind for e in first].count("appeared") == 1
        assert "appeared" not in [e.kind for e in client.drain_events()]
        assert len(client.ring.held()) <= feed.RING_SLOTS
    finally:
        client.close()
    assert client.proc.exitcode is not None and not client.alive
    assert client.bye is not None and client.bye["frames"] > 10


def test_when_the_child_dies_the_stack_is_told_and_gets_nothing_more():
    client = FeedClient(functools.partial(drawn_runner, 0), hz=60.0)
    try:
        assert client.latest(Asks(time.monotonic()), timeout=5.0) is not None
        client.proc.terminate()
        client.proc.join(5.0)
        deadline = time.monotonic() + 5.0
        while client.alive and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not client.alive and client.why
        assert client.latest(Asks(time.monotonic()), timeout=0.5) is None
        assert not client.wait_frame(0.1)
    finally:
        client.close()


def test_a_child_that_cant_start_says_why():
    client = FeedClient(functools.partial(int, "not a number"), hz=60.0)
    try:
        assert not client.alive and "couldn't start" in client.why
    finally:
        client.close()


def test_screen_frames_never_bring_the_game_forward_and_say_when_it_isnt_showing(monkeypatch):
    from gamelab import window
    from gamelab.analysis import bench
    from gamelab.perception.frames import ScreenFrames
    game = {"front": "MapleStory", "shows": True, "size": (1920, 1080), "running": True}

    def find_game(title):
        if not game["running"]:
            raise RuntimeError("no window")
        return 7
    monkeypatch.setattr(window, "find_game", find_game)
    monkeypatch.setattr(window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": game["size"][0],
                                                             "height": game["size"][1]})
    monkeypatch.setattr(window, "foreground_title", lambda: game["front"])
    monkeypatch.setattr(window, "shows", lambda hwnd, area: game["shows"])
    # window.bring_to_front raises here (tests/conftest.py): a frame source that called it would fail
    screen = bench.SavedScreen()
    screen.bgra = np.zeros((1080, 1920, 4), np.uint8)
    frames = ScreenFrames("MapleStory", grabber=screen)
    f = frames.next(time.monotonic())
    assert f.visible and f.full.shape == (1080, 1920, 3)
    for change, why in ((("front", "Notepad"), "in front"), (("shows", False), "in front"),
                        (("size", (1280, 720)), "1280x720"), (("running", False), "isn't running")):
        before = game[change[0]]
        game[change[0]] = change[1]
        f = frames.next(time.monotonic())
        assert not f.visible and f.full is None and why in f.why
        game[change[0]] = before


def test_senses_look_through_the_feed_and_look_themselves_once_its_down(sync):
    from gamelab.perception.eyes import GLANCE_EVERY_S
    f, clock, eyes = sync
    senses = Senses(eyes, f.step.model, band=(100, 100), feed=f)
    eyes.mobs = [pig(1100)]
    clock.sleep(0.5)
    asked = clock.now
    s = senses.look()
    assert s.t >= asked and s.searched and [m.kind for m in s.monsters] == ["pig"]
    assert [e.kind for e in s.events] == ["appeared"]  # what the feed saw since: once
    assert senses.look(mobs=False).events == [] and getattr(eyes, "looks", 0) == 0
    a, b = senses.glance(), senses.glance()
    assert senses.counts["glances from the feed"] == 2 and senses._glanced - asked >= GLANCE_EVERY_S - 1e-9
    f.alive = False  # the feed's gone: the look is taken here, as without one
    s = senses.look()
    assert eyes.looks == 1 and s.events is None and senses.counts == {
        "from the feed": 2, "taken here": 1, "glances from the feed": 2}
    senses.glance()
    assert eyes.glances == 1


def test_watch_publishes_what_it_sees_and_can_neither_send_keys_nor_bring_the_game_forward(monkeypatch, tmp_path):
    from maplehelper import win32

    from gamelab import __main__ as cli, telemetry, window
    from gamelab.control import client as player_client
    from gamelab.localization import worldstate
    touched = []
    for obj, name in ((win32, "send_key"), (window, "bring_to_front"), (window, "move_cursor"),
                      (player_client, "send"), (player_client, "submit"), (player_client, "cancel")):
        monkeypatch.setattr(obj, name, lambda *a, _n=name, **k: touched.append(_n))
    monkeypatch.setattr(telemetry, "FOLDER", tmp_path)
    monkeypatch.setattr(feed, "quiet", lambda: None)  # not the test process's priority
    model = small_map()
    eyes = Drawn(model)

    def runner(args):
        step = WorldStep(Senses(eyes, model, band=(100, 100)), throws=False, schedule=worldstate.WATCH)
        frames = ListFrames([np.zeros((4, 4, 3), np.uint8)])
        return feed.Runner(frames, _Look(eyes), step)
    monkeypatch.setattr(cli, "_watch_runner", runner)
    assert cli.main(["watch", "--minutes", "0.03", "--hz", "30"]) == 0
    assert touched == []
    (path,) = tmp_path.glob("*-watch.jsonl")
    records = telemetry.read(path)
    topics = {r["topic"] for r in records}
    assert {"session", "scene", "tracks", "log"} <= topics
    assert records[0]["mode"] == "watch"
    scenes = [r for r in records if r["topic"] == "scene"]
    assert 5 <= len(scenes) <= 25 and scenes[-1]["monsters"][0]["kind"] == "pig"  # 10 a second for ~1.8 s
    assert [t["kind"] for t in next(r for r in records if r["topic"] == "tracks")["list"]] == ["pig"]


def test_hunting_commands_take_a_feed_with_feed_and_go_without_it_when_it_cant_start(monkeypatch):
    from gamelab import __main__ as cli
    from gamelab.analysis import calibration
    for cmd in (["hunt", "clear", "--feed"], ["nav", "35", "--feed"], ["tour", "--loop", "--feed"]):
        assert cli.build_parser().parse_args(cmd).feed
    assert not cli.build_parser().parse_args(["tour"]).feed
    started = []

    class Client:
        def __init__(self, make, alive):
            self.make, self.alive, self.why, self.hz, self.bye = make, alive, "" if alive else "no game", 30.0, None
            self.served = self.refused = 0
            self.hello = {"pid": 1, "priority": feed.BELOW_NORMAL}
            started.append(self)

        def close(self):
            self.alive = False
    monkeypatch.setattr(calibration, "load", lambda: type("Cal", (), {"template": np.zeros((10, 36), np.uint8)}))
    monkeypatch.setattr(feed, "FeedClient", lambda make: Client(make, True))
    model = small_map()
    client = cli._feed(model, "MapleStory")
    assert client is started[0] and cli._FEEDS == [client] and client.make.func is feed.screen_world
    cli._close_feeds()
    assert cli._FEEDS == [] and not client.alive
    monkeypatch.setattr(feed, "FeedClient", lambda make: Client(make, False))
    assert cli._feed(model, "MapleStory") is None and cli._FEEDS == []
