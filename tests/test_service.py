"""The perception service (localization.service): one process looking at the game all the time, any number of
commands attached to it as clients. In-process: what it does with what its clients say (asks merged, her throws known
while a command tells them, the hunt's pace while one hunts, a client's queue dropping its oldest); and the real thing
on a test pipe, the drawn frames of test_feed for a screen: two clients attached and detached, one that never reads
leaving the frame loop at its rate, the service killed and started again under a client, a client killed."""

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
from gamelab.localization.service import IDLE, ACTIVE, Hello, Service, ServiceClient, Writer
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
    assert asked.after == 10.1 and asked.band and svc._fulls == [(a, 7, 10.5)]


def test_the_hunts_pace_and_her_throws_follow_the_commands_attached():
    svc = _svc(idle_hz=10.0, active_hz=30.0)
    svc.runner = drawn_runner()
    try:
        watch = _attach(svc, name="watch")
        svc._tune()
        assert svc.rate() == 10.0 and svc.runner.step.schedule == IDLE and not svc.runner.step.tracker.throws_known
        pet = _attach(svc, name="pet", acts=True)  # it sends keys, but doesn't hunt nor tell her throws
        svc._tune()
        assert svc.rate() == 10.0 and svc.runner.step.schedule == IDLE and not svc.runner.step.tracker.throws_known
        hunt = _attach(svc, name="tour", acts=True, throws=True, fast=True)
        svc._tune()
        assert svc.rate() == 30.0 and svc.runner.step.schedule == ACTIVE and svc.runner.step.tracker.throws_known
        del svc.clients[hunt.id]
        svc._tune()
        assert svc.rate() == 10.0 and svc.runner.step.schedule == IDLE and not svc.runner.step.tracker.throws_known
        assert {watch.id, pet.id} == set(svc.clients)
    finally:
        svc.runner.step.close()
    # The idle schedule searches her band and the screen less often, and the drops at watch's rates
    assert (IDLE.band_s, IDLE.wide_s, IDLE.loot_s, IDLE.loot_wide_s) == (0.5, 2.0, 0.5, 2.0)
    assert (ACTIVE.band_s, ACTIVE.wide_s, ACTIVE.loot_s) == (0.2, 1.0, None)


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


def test_a_second_command_that_acts_on_another_map_is_refused_and_one_alone_moves_the_service():
    svc = _svc()
    svc.map_id = 5
    hunt = _attach(svc, name="tour", acts=True, map=5)
    other = service._Client(9, SimpleNamespace(send_bytes=lambda b: None, close=lambda: None))
    svc.clients[9] = other
    said = []
    other.send = lambda *msg: said.append(msg)
    svc._hello(other, Hello(name="nav", acts=True, map=6))
    assert said[0][0] == "refused" and "map 5 for tour" in said[0][1] and svc._rebuild is None
    viewer = service._Client(10, SimpleNamespace(send_bytes=lambda b: None, close=lambda: None))
    viewer.send = lambda *msg: said.append(msg)
    svc._hello(viewer, Hello(name="map", map=6))  # a viewer on another map: welcomed, the hunt keeps its map
    assert said[-1][0] == "welcome" and svc._rebuild is None
    del svc.clients[hunt.id], svc.clients[9]
    svc._hello(other, Hello(name="nav", acts=True, map=6))  # alone now: the service looks at its map next
    assert said[-1][0] == "welcome" and svc._rebuild == 6


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


def serve(address: str, info: str, seconds: float, idle_hz: float, active_hz: float, full: bool = False,
          slow: float = 0.0) -> None:
    """A service in a process of its own, on the drawn frames (their searches `slow` s each)."""
    make = (lambda map_id, geo: full_runner()) if full else (lambda map_id, geo: drawn_runner(0, slow))
    service.settle()
    svc = Service(make, 1, lambda: (STRETCHED, None), address=address, info=Path(info), log=lambda m: None,
                  idle_hz=idle_hz, active_hz=active_hz)
    svc.serve(until=time.monotonic() + seconds)


class Running:
    """A service process on a pipe of the test's own."""

    def __init__(self, tmp_path: Path, name: str, seconds: float = 30.0, idle_hz: float = 10.0,
                 active_hz: float = 30.0, full: bool = False, at: "Running | None" = None, slow: float = 0.0):
        """`at`: where another one ran (started again there, as the babysitter starts it at the same address)."""
        pipe = rf"\\.\pipe\gamelab-perceive-test-{os.getpid()}-{name}-{time.monotonic_ns()}"
        self.address, self.info = (at.address, at.info) if at else (pipe, tmp_path / f"{name}.json")
        ctx = multiprocessing.get_context("spawn")
        self.proc = ctx.Process(target=serve, args=(self.address, str(self.info), seconds, idle_hz, active_hz, full,
                                                    slow), daemon=True)
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
        assert _rate(viewer, 1.0) > 20  # the hunt's pace while the tour is attached
        hunt.close()
        time.sleep(0.3)
        status = viewer.status()
        assert [c["name"] for c in status["clients"]] == ["watch"] and status["hz"] == 10.0
        assert _rate(viewer, 1.0) < 13  # idle again
        assert all(e.t >= first.t for e in viewer.drain_events())  # its own events, from when it attached
    finally:
        hunt.close()
        viewer.close()
        svc.stop()


def test_a_client_that_never_reads_leaves_the_frame_loop_at_its_rate(tmp_path):
    import multiprocessing.connection
    svc = Running(tmp_path, "stuck", idle_hz=30.0)
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
        step = WorldStep(Senses(eyes, model, band=(100, 100)), throws=False, schedule=IDLE)
        return feed.Runner(ListFrames([np.zeros((4, 4, 3), np.uint8)]), Gaining(eyes), step)
    telemetry.record_to("perceive", tmp_path)
    address = rf"\\.\pipe\gamelab-perceive-test-pub-{os.getpid()}-{time.monotonic_ns()}"
    svc = Service(make, 1, lambda: (STRETCHED, None), address=address, log=lambda m: None,
                  publish=service.Publisher((100, 100), log=lambda m: None), idle_hz=30.0)
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
    monkeypatch.setattr(cli, "_eyes", lambda model, title, perception, restore: SimpleNamespace(log=print))
    monkeypatch.setattr(mapdata, "load", lambda map_id: small_map())
    monkeypatch.setattr(hunter_mod, "Hunter", lambda eyes, keys, model, log, feed, **kw: SimpleNamespace(feed=feed))
    for argv, viewer, throws, want in ((["tour", "--loop"], False, True, (True, True)),
                                       (["pet"], False, False, (True, False)),
                                       (["hunt", "look"], True, True, (False, False))):
        args = cli.build_parser().parse_args(argv)
        cli._hunter(args, viewer=viewer, throws=throws)
        name, kw = made[-1]
        assert (kw["acts"], kw["throws"]) == want, argv
    cli._PERCEPTION.clear()
