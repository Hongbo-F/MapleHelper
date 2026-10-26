"""The perception service (localization.service): one process looking at the game all the time, any number of
commands attached to it as clients. In-process: what it does with what its clients say (asks merged, her throws known
while a command tells them, one pace and schedule whoever is attached, a client's queue dropping its oldest); and the
real thing on a test pipe, the drawn frames of test_feed for a screen: two clients attached and detached, one that never
reads leaving the frame loop at its rate, the service killed and started again under a client, a client killed."""

import multiprocessing
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab.localization import service
from gamelab.localization.service import HUNTING, SCHEDULE, Hello, Service, ServiceClient, Writer
from gamelab.localization.worldstate import Asks, SearchThread, WorldStep
from gamelab.localization.senses import Senses
from gamelab.perception.frames import ListFrames
from gamelab.screen import STRETCHED

from test_feed import _Look, drawn_runner
from test_worldstate import Drawn, small_map

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the service's pipe is a Windows named pipe")


# --- what it does with what its clients say ---------------------------------------------------------------------

def _svc(**kw) -> Service:
    return Service(lambda map_id, geo: drawn_runner(), 1, lambda: (STRETCHED, None), address="unused",
                   log=lambda m: None, **kw)


def _attach(svc: Service, **hello) -> service._Client:
    c = service._Client(len(svc.clients) + 1, SimpleNamespace(send_bytes=lambda b: None, close=lambda: None))
    c.hello = Hello(**hello)
    svc.clients[c.id] = c
    return c


def test_asks_from_every_client_are_merged_and_mesos_only_if_each_asked_only_for_mesos():
    svc = _svc()
    a, b = _attach(svc, name="hunt", acts=True), _attach(svc, name="watch")
    merged = svc._take(a, ("ask", Asks(10.2, band=True, loot=True, mesos_only=True)), None)
    merged = svc._take(b, ("ask", Asks(10.1, loot=True)), merged)
    assert (merged.after, merged.band, merged.loot, merged.mesos_only) == (10.1, True, True, False)
    first = svc._take(a, ("ask", Asks(10.2, loot=True, mesos_only=True)), None)
    both = svc._take(b, ("ask", Asks(10.3, loot=True, mesos_only=True)), first)
    assert both.mesos_only  # (to_scene filters what each wants of a frame searched for all)
    # A full ask is a frame asked for at its time, its searches what the others asked
    asked = svc._take(a, ("full", 7, 10.5), merged)
    assert asked.after == 10.1 and asked.band and svc._fulls == [(a, 7, 10.5, False)]
    svc._take(a, ("full", 8, 10.6, True), asked)  # (one that takes the client at any size: ServiceClient.full_frame)
    assert svc._fulls[-1] == (a, 8, 10.6, True)


def test_one_pace_and_searches_whoever_is_attached_the_drops_asked_for_by_a_hunt_and_her_throws_follow_it():
    # The user, 2026-10-26: "Run the perception at the same rate between hunt and idle. The idea is perception is
    # running as a constant thing and hunt is only a consumer of it." Until then a command that hunts had the frames at
    # 30 a second and the hunt's schedule while it was attached, and 10 a second with watch's slower searches otherwise.
    # The drops are searched on schedule only while no command hunts: a hunt asks for those it wants on its frames.
    svc = _svc()
    svc.runner = drawn_runner()
    try:
        for name, kw in (("watch", {}), ("pet", {"acts": True}), ("tour", {"acts": True, "throws": True, "fast": True})):
            _attach(svc, name=name, **kw)
            svc._tune()
            assert svc.rate() == service.HZ == 30.0
            assert svc.runner.step.schedule is (HUNTING if name == "tour" else SCHEDULE)
            assert svc.runner.step.tracker.throws_known == (name == "tour")  # (the one that tells her throws)
        del svc.clients[max(svc.clients)]
        svc._tune()
        assert svc.rate() == 30.0 and svc.runner.step.schedule is SCHEDULE and not svc.runner.step.tracker.throws_known
        assert svc.status()["hz"] == 30.0
    finally:
        svc.runner.step.close()
    # Her band every 0.2 s and the screen every second either way; the drops at watch's rates, or when asked
    assert (SCHEDULE.band_s, SCHEDULE.wide_s, SCHEDULE.loot_s, SCHEDULE.loot_wide_s) == (0.2, 1.0, 0.5, 2.0)
    assert (HUNTING.band_s, HUNTING.wide_s, HUNTING.loot_s, HUNTING.loot_wide_s) == (0.2, 1.0, None, None)
    assert not hasattr(service, "IDLE_HZ") and not hasattr(service, "ACTIVE")


def test_a_command_that_hunts_attaching_changes_nothing_of_the_frame_loops_pace():
    """Its frames come at the one pace: its hello wakes nothing (until 2026-10-26 it brought the next frame forward to
    the hunt's pace); `fast` only holds the map (test_a_second_command_that_hunts_on_another_map_is_refused...)."""
    svc = _svc()
    svc.map_id = 5
    said = []
    svc._hello(_joining(svc, 1, said), Hello(name="map", map=5))
    svc._hello(_joining(svc, 2, said), Hello(name="tour", acts=True, throws=True, fast=True, map=5))
    assert not svc._woken and [m[1] for m in said] == ["welcome", "welcome"]


def test_a_clients_queue_keeps_its_newest_drops_the_oldest_and_says_a_gap():
    sent, gate = [], threading.Event()

    class Stuck:
        def send_bytes(self, data):
            gate.wait(5.0)  # a client that doesn't read: the pipe's buffer full
            sent.append(pickle.loads(data))
    w = Writer(Stuck(), keep=4)
    for k in range(10):
        w.put(pickle.dumps(("state", k)))
    time.sleep(0.05)
    assert w.dropped >= 5 and len(w.q) <= 4  # (the writer may have one on its way to the stuck pipe)
    gate.set()
    deadline = time.monotonic() + 2.0
    while (not sent or sent[-1] != ("state", 9)) and time.monotonic() < deadline:
        time.sleep(0.01)
    w.close()
    states = [m[1] for m in sent if m[0] == "state"]
    assert sum(m[1] for m in sent if m[0] == "gap") == w.dropped  # every drop said
    assert states[-3:] == [7, 8, 9] and len(states) == 10 - w.dropped and states == sorted(states)


def _joining(svc: Service, cid: int, said: list) -> service._Client:
    c = service._Client(cid, SimpleNamespace(send_bytes=lambda b: None, close=lambda: None))
    svc.clients[cid] = c
    c.send = lambda *msg: said.append((cid, *msg))
    return c


def test_a_second_command_that_hunts_on_another_map_is_refused_and_one_alone_moves_the_service():
    svc = _svc()
    svc.map_id = 5
    hunt = _attach(svc, name="tour", acts=True, throws=True, fast=True, map=5)
    said = []
    other = _joining(svc, 9, said)
    svc._hello(other, Hello(name="nav", acts=True, throws=True, fast=True, map=6))
    assert said[0][1] == "refused" and "map 5 for tour" in said[0][2] and svc._rebuild is None
    viewer = _joining(svc, 10, said)
    svc._hello(viewer, Hello(name="map", map=6))  # a viewer on another map: welcomed, the hunt keeps its map
    assert said[-1][1] == "welcome" and svc._rebuild is None
    del svc.clients[hunt.id], svc.clients[9]
    svc._hello(other, Hello(name="nav", acts=True, throws=True, fast=True, map=6))  # alone now: its map next
    assert said[-1][1] == "welcome" and svc._rebuild == 6


def test_the_pets_babysitter_holds_no_map_and_a_hunt_on_another_moves_the_service():
    # `pet --every 1` (the babysitter's, outside hunts) sends keys but doesn't hunt, and reads only whole frames: it
    # holds no map. A hunt on another map attaching while it runs moves the service to its map (before, it was refused
    # and ended at its start: the babysitter stops the pet only once it sees a hunt); the pet is never refused, and once
    # no command that hunts is left, the service follows the map of the newest command that acts.
    svc = _svc()
    svc.map_id = 5
    said = []
    pet = _joining(svc, 1, said)
    svc._hello(pet, Hello(name="pet", acts=True, map=5))
    hunt = _joining(svc, 2, said)
    svc._hello(hunt, Hello(name="tour", acts=True, throws=True, fast=True, map=6))
    assert [m[1] for m in said] == ["welcome", "welcome"] and svc._rebuild == 6
    svc.map_id, svc._rebuild = 6, None  # (the run on 6 made)
    late = _joining(svc, 3, said)
    svc._hello(late, Hello(name="pet", acts=True, map=5))  # a pet on 5 while the hunt holds 6: welcomed, nothing moves
    assert said[-1][1] == "welcome" and svc._rebuild is None
    svc._drop(hunt)  # the hunt over: the newest command that acts names the map
    assert svc._rebuild == 5
    svc.map_id, svc._rebuild = 5, None  # (the run on 5 made)
    svc._drop(late)
    svc._drop(pet)  # (with none left the map stays)
    assert svc._rebuild is None and svc.map_id == 5


def test_a_command_that_acts_is_handed_only_states_of_its_maps_run():
    # A reconnect welcomes a command to whatever run the service is on (one the babysitter started on another map, say):
    # its looks get none of that run's states until its own map's run is welcomed, and none of a run before the one it's
    # welcomed to (they can come after the new run's welcome, which is sent first). A viewer takes whatever it looks at.
    from gamelab.localization.worldstate import WorldState
    hunt = ServiceClient("tour", acts=True, throws=True, map_id=2, start=False, ensure=None)
    viewer = ServiceClient("watch", map_id=2, start=False, ensure=None)
    now = time.monotonic()
    for c in (hunt, viewer):
        c._take(("welcome", {"epoch": 4, "pid": 7, "map": 1, "next": 2, "ring": None}))
        c._take(("state", _state(10, now + 0.01, epoch=4)))
        c._take(("state", _state(11, now + 0.02, epoch=3)))  # (the run before's)
    assert [st.seq for st in hunt.states] == [10] and hunt.stale == 1
    assert hunt.latest(Asks(now), timeout=0.05) is None and "waiting for its run on 2" in hunt.problem().why
    assert viewer.latest(Asks(now), timeout=0.05).seq == 10
    hunt._take(("epoch", {"epoch": 5, "pid": 7, "map": 2, "next": 2, "ring": None}))
    hunt._take(("state", _state(12, now + 0.03, epoch=4)))  # (queued before the new run's welcome went)
    hunt._take(("state", _state(1, now + 0.04, epoch=5)))
    assert hunt.latest(Asks(now), timeout=0.05).seq == 1 and hunt.stale == 2 and hunt.problem() is None


# --- the real thing on a test pipe ------------------------------------------------------------------------------

def full_runner():
    """drawn_runner with frames of the 1920x1080 window's size (a `full` ask's frame is the screen's)."""
    model = small_map()
    eyes = Drawn(model)
    senses = Senses(eyes, model, band=(100, 100))
    step = WorldStep(senses, searches=SearchThread(senses))
    frames = [np.full((1080, 1920, 3), k, np.uint8) for k in (10, 20, 30)]
    from test_feed import feed
    return feed.Runner(ListFrames(frames), _Look(eyes), step)


class Hanging:
    """Frames that stop coming after `n` (the screen's capture hung): the frame loop waits on the next for ever."""

    def __init__(self, frames, n: int):
        self.frames, self.n = frames, n

    def next(self, after: float):
        self.n -= 1
        if self.n < 0:
            time.sleep(3600)
        return self.frames.next(after)


def serve(address: str, info: str, seconds: float, hz: float, full: bool = False,
          slow: float = 0.0, hang_after: int | None = None, stuck_s: float | None = None) -> None:
    """A service in a process of its own, on the drawn frames (their searches `slow` s each; none after `hang_after`
    frames), with a watchdog of `stuck_s` (None: none)."""
    def make(map_id, geo):
        runner = full_runner() if full else drawn_runner(0, slow)
        if hang_after is not None:
            runner.frames = Hanging(runner.frames, hang_after)
        return runner
    service.settle()
    svc = Service(make, 1, lambda: (STRETCHED, None), address=address, info=Path(info), log=lambda m: None,
                  hz=hz, stuck_s=stuck_s)
    svc.serve(until=time.monotonic() + seconds)


class Running:
    """A service process on a pipe of the test's own."""

    def __init__(self, tmp_path: Path, name: str, seconds: float = 30.0, hz: float = 30.0, full: bool = False,
                 at: "Running | None" = None, slow: float = 0.0, hang_after: int | None = None,
                 stuck_s: float | None = None):
        """`at`: where another one ran (started again there, as the babysitter starts it at the same address)."""
        pipe = rf"\\.\pipe\gamelab-perceive-test-{os.getpid()}-{name}-{time.monotonic_ns()}"
        self.address, self.info = (at.address, at.info) if at else (pipe, tmp_path / f"{name}.json")
        ctx = multiprocessing.get_context("spawn")
        self.proc = ctx.Process(target=serve, args=(self.address, str(self.info), seconds, hz, full, slow, hang_after,
                                                    stuck_s), daemon=True)
        self.proc.start()

    def client(self, name: str, **kw) -> ServiceClient:
        return ServiceClient(name, address=self.address, info=self.info, ensure=None, log=lambda m: None, **kw)

    def stop(self) -> None:
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc.join(5.0)


def _rate(client: ServiceClient, seconds: float) -> float:
    first = max(st.seq for st in client.states)
    t0 = time.monotonic()
    time.sleep(seconds)
    return (max(st.seq for st in client.states) - first) / (time.monotonic() - t0)


def test_two_commands_attach_are_both_served_and_detach(tmp_path):
    svc = Running(tmp_path, "two")
    hunt = svc.client("tour", acts=True, throws=True)
    viewer = svc.client("watch")
    try:
        assert hunt.ready(30.0) and viewer.ready(5.0), (hunt.why, viewer.why)
        assert viewer.wait_frame(5.0)
        first = viewer.states[0]
        after = time.monotonic()
        st = hunt.latest(Asks(after, band=True), timeout=5.0)
        assert st is not None and st.t >= after and st.placed and [m.kind for m in st.monsters] == ["pig"]
        assert hunt.frame_of(st).shape == (540, 960, 3) and hunt.frame_of(st).any()
        seen = viewer.latest(Asks(after), timeout=5.0)
        assert seen is not None and seen.seq >= st.seq - 2 and seen.epoch == st.epoch == hunt.epoch
        status = viewer.status()
        assert sorted(c["name"] for c in status["clients"]) == ["tour", "watch"] and status["hz"] == 30.0
        assert _rate(viewer, 1.0) > 20  # one pace, the tour attached ...
        hunt.close()
        time.sleep(0.3)
        status = viewer.status()
        assert [c["name"] for c in status["clients"]] == ["watch"] and status["hz"] == 30.0
        assert _rate(viewer, 1.0) > 20  # ... or not (until 2026-10-26, 10 a second without it)
        assert all(e.t >= first.t for e in viewer.drain_events())  # its own events, from when it attached
    finally:
        hunt.close()
        viewer.close()
        svc.stop()


def test_a_client_that_never_reads_leaves_the_frame_loop_at_its_rate(tmp_path):
    import multiprocessing.connection
    svc = Running(tmp_path, "stuck")
    reader = svc.client("watch")
    try:
        assert reader.ready(30.0), reader.why
        info = service.read_info(svc.info)
        stuck = multiprocessing.connection.Client(svc.address, family="AF_PIPE", authkey=bytes.fromhex(info["authkey"]))
        stuck.send(("hello", Hello("stuck", 1).__dict__))  # attached, and never reads a thing
        time.sleep(1.0)
        assert _rate(reader, 3.0) > 25  # its pipe full after a few states, the frames go on at 30 a second
        status = reader.status()
        assert {c["name"]: c["dropped"] for c in status["clients"]}["stuck"] > 0  # its oldest dropped
        stuck.close()
    finally:
        reader.close()
        svc.stop()


def test_a_command_stands_still_while_the_service_is_gone_and_goes_on_with_the_new_one(tmp_path):
    svc = Running(tmp_path, "killed")
    hunt = svc.client("tour", acts=True, throws=True)
    again = None
    try:
        assert hunt.ready(30.0), hunt.why
        assert hunt.latest(Asks(time.monotonic()), timeout=5.0) is not None
        first = hunt.run
        svc.stop()  # killed mid-run
        deadline = time.monotonic() + 5.0
        while hunt.alive and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not hunt.alive and ("ended" in hunt.why or "can't be reached" in hunt.why), hunt.why
        assert hunt.latest(Asks(time.monotonic()), timeout=0.3) is None  # nothing to act on
        assert hunt.problem().kind == "perception"
        again = Running(tmp_path, "killed", at=svc)  # started again (the babysitter, or the client's `ensure`)
        assert hunt.ready(30.0), hunt.why
        st = hunt.latest(Asks(time.monotonic(), band=True), timeout=5.0)
        assert st is not None and hunt.run != first and hunt.frame_of(st).any() and hunt.connects == 2
    finally:
        hunt.close()
        svc.stop()
        if again is not None:
            again.stop()


def test_a_service_whose_frame_loop_hangs_ends_itself_and_the_hunt_goes_on_with_the_next(tmp_path):
    # Its capture hung (the frames stop coming), the process and its pipe up: its watchdog ends it (EXIT_STUCK), the
    # command stands still meanwhile, and goes on with the one started next (the babysitter's or ensure's). Before,
    # nothing noticed: every look found nothing to act on, and a hunt stood still to its end.
    svc = Running(tmp_path, "hung", hang_after=30, stuck_s=2.0)
    hunt = svc.client("tour", acts=True, throws=True)
    again = None
    try:
        assert hunt.ready(30.0), hunt.why
        assert hunt.latest(Asks(time.monotonic()), timeout=5.0) is not None
        svc.proc.join(30.0)
        assert svc.proc.exitcode == service.EXIT_STUCK
        assert hunt.latest(Asks(time.monotonic()), timeout=0.3) is None and hunt.problem().kind == "perception"
        again = Running(tmp_path, "hung", at=svc)
        assert hunt.ready(30.0), hunt.why
        assert hunt.latest(Asks(time.monotonic()), timeout=5.0) is not None and hunt.connects == 2
    finally:
        hunt.close()
        svc.stop()
        if again is not None:
            again.stop()


def test_a_client_that_hears_nothing_from_the_service_reconnects(tmp_path, monkeypatch):
    # The service's frame loop hung with no watchdog (its pipe still up): the client drops its connection after
    # SILENT_S without a word, says why, and connects again; nothing is handed out to act on meanwhile.
    monkeypatch.setattr(service, "SILENT_S", 1.0)
    svc = Running(tmp_path, "silent", hang_after=30)
    said = []
    hunt = ServiceClient("tour", acts=True, throws=True, address=svc.address, info=svc.info, ensure=None,
                         log=said.append)
    try:
        assert hunt.ready(30.0), hunt.why
        assert hunt.latest(Asks(time.monotonic()), timeout=5.0) is not None
        deadline = time.monotonic() + 15.0
        while hunt.connects < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert hunt.connects >= 2 and any("nothing from the perception service for 1 s" in m for m in said), said
        assert hunt.latest(Asks(time.monotonic()), timeout=0.3) is None
    finally:
        hunt.close()
        svc.stop()


def test_a_client_closed_between_its_poll_and_its_read_ends_quietly(monkeypatch):
    # `close` closes the connection from another thread: between the reader's poll and its read the handle is gone
    # (multiprocessing raises TypeError then: ReadFile on None). Closing, that's the end; otherwise it's a bug, raised.
    client = ServiceClient("tour", acts=True, start=False, ensure=None, log=lambda m: None)
    bug = []

    class Conn:
        def send(self, msg):
            pass

        def poll(self, timeout):
            return True

        def recv(self):
            if bug:
                raise TypeError("a bug")
            client._closing = True  # (`close` on another thread, as the read began)
            raise TypeError("ReadFile() argument 1 must be int, not None")

        def close(self):
            pass
    monkeypatch.setattr(client, "_connect", lambda: Conn())
    client._run()  # (returns)
    assert not client.alive and client.why == "closed"
    client._closing = False
    bug.append(True)
    with pytest.raises(TypeError):
        client._run()


def test_the_watchdog_says_a_loop_that_hasnt_gone_round_or_a_search_out_too_long():
    svc = _svc(stuck_s=10.0)
    now = time.monotonic()
    svc._went_round()
    assert svc.stuck(now + 5.0) is None and "hasn't gone round for 11" in svc.stuck(now + 11.0)
    svc._went_round(service.BUILD_STUCK_S)  # making a run (the map's templates loaded): longer
    assert svc.stuck(now + 30.0) is None
    svc._went_round()
    svc.runner = SimpleNamespace(step=SimpleNamespace(searches=SimpleNamespace(since=now - 10.5)))
    assert "a search has been out for 10" in svc.stuck(now)
    svc.runner.step.searches.since = None
    assert svc.stuck(now) is None


def test_perceive_on_saved_frames_has_a_pipe_mutex_and_info_file_of_its_own(tmp_path, monkeypatch):
    # `perceive --frames` (offline) never takes the live service's mutex, pipe or info file: the live one starts and
    # is reached beside it. A command reaches it with GAMELAB_PERCEIVE=saved, and starts none.
    from PIL import Image

    from gamelab import __main__ as cli
    Image.fromarray(np.zeros((768, 1366, 3), np.uint8)).save(tmp_path / "a.png")
    held, made = [], []
    monkeypatch.setattr(service, "only_one", lambda name=service.MUTEX: held.append(name) or 1)
    monkeypatch.setattr(service, "settle", lambda: None)

    class Made:
        def __init__(self, make, map_id, where, **kw):
            made.append(kw)

        def serve(self, until=None):
            pass

        def status(self):
            return {}
    monkeypatch.setattr(service, "Service", Made)
    assert cli.main(["perceive", "--frames", str(tmp_path), "--minutes", "0.01"]) == 0
    assert held == [service.SAVED_MUTEX] and made[0]["address"] == service.SAVED_ADDRESS
    assert made[0]["info"].name == "perceive-saved.json" and made[0]["stuck_s"] == service.STUCK_S
    assert service.SAVED_MUTEX != service.MUTEX and service.SAVED_ADDRESS != service.ADDRESS
    monkeypatch.setenv(service.WHICH, "saved")
    c = ServiceClient("watch", start=False)
    assert (c.address, c.info.name) == (service.SAVED_ADDRESS, "perceive-saved.json")
    monkeypatch.setattr(service, "_start_detached", lambda *a: pytest.fail("started a service"))
    monkeypatch.setattr(service, "_ENSURED", [-1e18])
    assert service.ensure(1, log=lambda m: None) is False
    monkeypatch.delenv(service.WHICH)
    c = ServiceClient("watch", start=False)
    assert (c.address, c.info.name) == (service.ADDRESS, "perceive.json")


def test_a_looks_frame_gone_from_the_ring_comes_out_black_and_is_counted():
    """A look's frame is copied out of the service's ring as its state is taken: one overwritten since (by the frame
    RING_SLOTS later) comes out black, counted (`lost`, said as the command ends); her heading never learns from it
    (test_heading). The ring holds 16 since the native hot paths made 30 frames a second: 533 ms, as 8 did at 15."""
    from gamelab.localization.feed import Ring
    from gamelab.localization.worldstate import WorldState
    assert service.RING_SLOTS * 1000 / 30 >= 500  # (at 30 frames a second: ACTIVE_HZ here, Branch 1's HZ)
    w, h = STRETCHED.frame_size
    ring = Ring(slots=2, shape=(h, w, 3))
    c = ServiceClient("tour", start=False, ensure=None)
    try:
        c.ring, c.epoch = ring, 1
        states = [WorldState(k, float(k), True, geo=STRETCHED, epoch=1,
                             slot=ring.put(k, float(k), np.full((h, w, 3), k + 1, np.uint8))) for k in range(3)]
        assert (c.frame_of(states[2]) == 3).all() and (c.frame_of(states[1]) == 2).all() and c.lost == 0
        gone = c.frame_of(states[0])  # (frame 2 went into its slot)
        assert gone.shape == (h, w, 3) and not gone.any() and c.lost == 1
    finally:
        c.ring = None
        ring.close(unlink=True)


def test_the_service_says_whether_gamelabs_c_is_in_use():
    """A DLL that couldn't be built leaves perception to Python and OpenCV, slower: the service's console says which
    as each run starts (and the session record has it), and every 10 s line while the C isn't all in use."""
    from gamelab import native
    said = []
    pub = service.Publisher((100, 100), log=said.append)
    runner = drawn_runner()
    pub.started(runner, 1)
    assert said == [f"perception: gamelab's C pieces {native.state()}"] and ("C pieces" in pub.status(0.0)) == pub.slow
    assert pub.slow == (native.lib() is None or not native.masked_in_c())
    runner.step.close()


def _client_process(address: str, info: str) -> None:
    c = ServiceClient("doomed", address=address, info=Path(info), ensure=None, log=lambda m: None)
    c.ready(30.0)
    time.sleep(60)


def test_a_client_killed_is_dropped(tmp_path):
    svc = Running(tmp_path, "client")
    viewer = svc.client("watch")
    ctx = multiprocessing.get_context("spawn")
    doomed = ctx.Process(target=_client_process, args=(svc.address, str(svc.info)), daemon=True)
    try:
        assert viewer.ready(30.0), viewer.why
        doomed.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and "doomed" not in [c["name"] for c in (viewer.status() or {}).get(
                "clients", [])]:
            time.sleep(0.1)
        assert "doomed" in [c["name"] for c in viewer.status()["clients"]]
        doomed.terminate()
        doomed.join(5.0)
        time.sleep(0.5)
        assert [c["name"] for c in viewer.status()["clients"]] == ["watch"]
        assert viewer.latest(Asks(time.monotonic()), timeout=2.0) is not None  # the service goes on
    finally:
        viewer.close()
        if doomed.is_alive():
            doomed.terminate()
        svc.stop()


def test_a_full_frame_is_the_screens_next_frame_at_full_size(tmp_path):
    svc = Running(tmp_path, "full", full=True)
    pet = svc.client("pet", acts=True)
    try:
        assert pet.ready(30.0), pet.why
        after = time.monotonic()
        full = pet.full_frame(after)
        assert full.shape == (1080, 1920, 3) and int(full[0, 0, 0]) in (10, 20, 30)
    finally:
        pet.close()
        svc.stop()


def test_the_frames_keep_their_rate_while_a_slow_search_is_out(tmp_path, monkeypatch):
    # Each search takes 0.15 s: the frame loop goes on beside it (its searches on a thread of their own), a look that
    # asks for a search of its own waits for it.
    svc = Running(tmp_path, "slow", slow=0.15)
    hunt = svc.client("tour", acts=True, throws=True)
    monkeypatch.setattr(service.feed, "ASK_ONLY_STALE", False)  # (the look asks for her band on a frame of its own)
    try:
        assert hunt.ready(30.0), hunt.why
        assert hunt.latest(Asks(time.monotonic(), band=True), timeout=5.0) is not None
        t0 = time.monotonic()
        asked = hunt.latest(Asks(t0, band=True), timeout=2.0)
        waited = time.monotonic() - t0
        assert asked is not None and asked.band_at >= t0 and 0.15 <= waited < 0.6
        assert _rate(hunt, 1.0) > 15  # (30 a second asked for; with the searches in the frame loop under 7)
    finally:
        hunt.close()
        svc.stop()


def test_the_commands_side_asks_and_waits_and_gives_up(monkeypatch):
    from gamelab.localization.worldstate import WorldState
    from gamelab.perception.framecheck import Problem
    client = ServiceClient("tour", acts=True, start=False, ensure=None)
    client.alive = True
    sent = []
    monkeypatch.setattr(client, "_send", lambda *msg: sent.append(msg) or True)
    now = time.monotonic()
    client.states.append(_state(1, now - 1.0))
    assert client.latest(Asks(now, band=True), timeout=0.05) is None
    assert sent and sent[0][0] == "ask" and sent[0][1].band
    client.states.append(_state(2, now + 0.01, band_at=now))
    assert client.latest(Asks(now, band=True), timeout=0.05).seq == 2
    client.states.append(WorldState(3, now + 0.02, False, why="the game isn't in front",
                                    bad=Problem("window", "the game isn't in front")))
    t0 = time.monotonic()
    assert client.latest(Asks(now + 0.015), timeout=1.0) is None and time.monotonic() - t0 < 0.5  # at once
    assert client.problem().kind == "window"
    client.alive = False
    assert client.latest(Asks(0.0), timeout=0.1) is None and client.problem().kind == "perception"


def _state(seq, t, **kw):
    from gamelab.localization.worldstate import WorldState
    kw.setdefault("tag", (480, 308))
    kw.setdefault("where", (1000.0, 500.0, 0, (0.0, 0.0)))
    return WorldState(seq, t, True, **kw)


def test_the_service_publishes_what_it_sees_as_watch_did(tmp_path, monkeypatch):
    # watch's records, now the service's (Publisher): a session, a scene 10 times a second (every monster, the drops
    # last searched and where), the tracks twice a second, her EXP gains; it sends no key and brings nothing forward.
    from maplehelper import win32

    from gamelab import telemetry, window
    from gamelab.control import client as player_client
    from gamelab.localization import worldstate
    from gamelab.perception.exp_number import ExpReading
    from test_feed import feed
    touched = []
    for obj, name in ((win32, "send_key"), (window, "bring_to_front"), (window, "move_cursor"),
                      (player_client, "send"), (player_client, "submit"), (player_client, "cancel")):
        monkeypatch.setattr(obj, name, lambda *a, _n=name, **k: touched.append(_n))
    model = small_map()
    model.mob_exp["Pig"] = 13  # (as on Thicket Around the Beach II: a pig's kill)
    eyes = Drawn(model)

    class Gaining(_Look):  # the number drawn as data: 421 EXP, then a pig's 13 more 0.5 s in
        def look_at(self, full, t, ropes):
            obs = super().look_at(full, t, ropes)
            eyes.exp = 10 if t - self.t0 >= 0.5 else 0
            obs.exp, obs.exp_box = eyes.frame(t, ropes).exp, np.full((1, 1), eyes.exp, np.uint8)
            return obs
    monkeypatch.setattr(worldstate.exp_number, "read",
                        lambda box: ExpReading(421 + (13 if box[0, 0] else 0), 2.5, (19,)))

    def make(map_id, geo):
        step = WorldStep(Senses(eyes, model, band=(100, 100)), throws=False, schedule=SCHEDULE)
        return feed.Runner(ListFrames([np.zeros((4, 4, 3), np.uint8)]), Gaining(eyes), step)
    assert SCHEDULE is worldstate.WATCH  # (the drops searched at watch's rates, published)
    telemetry.record_to("perceive", tmp_path)
    address = rf"\\.\pipe\gamelab-perceive-test-pub-{os.getpid()}-{time.monotonic_ns()}"
    svc = Service(make, 1, lambda: (STRETCHED, None), address=address, log=lambda m: None,
                  publish=service.Publisher((100, 100), log=lambda m: None))
    svc.serve(until=time.monotonic() + 1.8)
    assert touched == []
    (path,) = tmp_path.glob("*-perceive.jsonl")
    records = telemetry.read(path)
    topics = {r["topic"] for r in records}
    assert {"session", "scene", "tracks", "exp"} <= topics and records[0]["command"] == "perceive"
    scenes = [r for r in records if r["topic"] == "scene"]
    assert 8 <= len(scenes) <= 25 and scenes[-1]["monsters"][0]["kind"] == "pig"  # 10 a second for ~1.8 s
    # Where the last search for drops looked (a late state's: the drops come back with it), her platform among them.
    assert scenes[-1]["her"]["platform"] in {lk[0] for lk in scenes[-1]["looked"]}
    assert [t["kind"] for t in [r for r in records if r["topic"] == "tracks"][-1]["list"]] == ["pig"]
    exp = [r for r in records if r["topic"] == "exp"]
    assert [(r["exp"], r["level"], r["gain"], r["kinds"], r["total"]) for r in exp] == [(434, 19, 13, ["pig"], 13)]


def test_the_commands_attach_as_what_they_are_and_none_has_a_feed_or_sweep_flag(monkeypatch):
    # The hunting commands send keys and tell her throws (the hunt's pace); `pet` sends keys but doesn't hunt; `hunt
    # look` is a viewer. No command looks for itself any more: --feed/--no-feed and --sweep are gone.
    from gamelab import __main__ as cli
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.localization import mapdata
    for cmd in (["hunt", "clear", "--no-feed"], ["tour", "--loop", "--sweep"], ["nav", "35", "--feed"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(cmd)
    made = []

    class Client:
        def __init__(self, name, **kw):
            made.append((name, kw))
            self.welcome, self.why = {"pid": 1, "epoch": 1, "map": kw["map_id"]}, ""

        def ready(self, timeout):
            return True

        def full_frame(self, after):
            raise AssertionError("no frame wanted here")
    monkeypatch.setattr(service, "ServiceClient", Client)
    monkeypatch.setattr(cli, "_eyes", lambda model, title, perception, restore: SimpleNamespace(log=print,
                                                                                               geo=STRETCHED))
    fits = []  # (the commands that hunt set a client of the wrong size right as they start: test_window_size)
    monkeypatch.setattr(cli, "_fit_start", lambda title, fix: fits.append(type(fix).__name__))
    monkeypatch.setattr(mapdata, "load", lambda map_id: small_map())
    monkeypatch.setattr(hunter_mod, "Hunter", lambda eyes, keys, model, log, feed, **kw: SimpleNamespace(feed=feed))
    for argv, viewer, throws, want in ((["tour", "--loop"], False, True, (True, True)),
                                       (["pet"], False, False, (True, False)),
                                       (["hunt", "look"], True, True, (False, False))):
        args = cli.build_parser().parse_args(argv)
        cli._hunter(args, viewer=viewer, throws=throws)
        name, kw = made[-1]
        assert (kw["acts"], kw["throws"]) == want, argv
    assert fits == ["SizeFix"]  # the tour's alone: the pet and `hunt look` send no Alt+Enter as they start
    cli._PERCEPTION.clear()


def test_watch_counts_the_frames_of_a_new_run_numbered_from_1_again(monkeypatch):
    # A run's frames are numbered from 1 (the map changed, or the service started again): watch goes on counting them
    # and saying what they show. Before, it waited for the new numbers to pass the old (0 frames a second meanwhile).
    from collections import deque

    from gamelab import __main__ as cli
    from gamelab.localization.worldstate import WorldState
    from gamelab.perception.framecheck import Problem
    logged = []
    monkeypatch.setattr(cli, "_log", logged.append)
    script = [((1, 1), WorldState(k, 0.0, True, tag=(480, 308), where=(1000.0, 500.0, 0, (0.0, 0.0))))
              for k in range(1, 40)]
    script += [((2, 1), WorldState(k, 0.0, False, why="the game is minimized",
                                   bad=Problem("minimized", "the game is minimized"))) for k in range(1, 5)]

    class Perception:  # the service's client: a state at each wait, then the service started again
        def __init__(self):
            self.states, self.run = deque(maxlen=8), None

        def wait_frame(self, timeout):
            time.sleep(0.002)
            if script:
                run, st = script.pop(0)
                if run != self.run:
                    self.run = run
                    self.states.clear()
                self.states.append(st)
            return True

        def status(self):
            return {}
    monkeypatch.setattr(cli, "_perception", lambda model, title, name, acts: Perception())
    assert cli.main(["watch", "--minutes", "0.01", "--map", "10000012"]) == 0
    assert logged[:2] == ["the game is on screen: watching", "not watching: the game is minimized"]


def test_a_command_on_another_map_waits_for_the_services_run_on_it(tmp_path):
    # The service starts on map 1; a command that acts on map 2 attaches: it's welcomed, and ready() returns only once
    # the service's run on map 2 has begun (a new run: its rings and epoch), so no state of map 1's run reaches it.
    svc = Running(tmp_path, "maps")
    hunt = svc.client("tour", acts=True, throws=True, map_id=2)
    try:
        assert hunt.ready(30.0), hunt.why
        assert hunt.welcome["map"] == 2 and hunt.epoch == 2 and all(st.epoch == 2 for st in hunt.states)
        assert hunt.latest(Asks(time.monotonic()), timeout=5.0) is not None
    finally:
        hunt.close()
        svc.stop()
