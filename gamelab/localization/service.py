"""The perception service: the game looked at every frame by one process that runs at all times (`python -m gamelab
perceive`, kept up by the babysitter), the latest of what it saw handed to every command that asks, and published for
the mini-map. (The user, 2026-10-25: "perception should just always run and the other modules just work based on
latest perception result"; and, asked how: "run one shared perception process at all times".)

What it runs is the feed's frame loop (localization.feed.run: the world step on every frame, its searches on a thread
of their own, localization.worldstate) on the screen (perception.frames.ScreenFrames, the process's one DXGI
duplication: capture.shared). It never sends a key and never brings the game forward; it waits for the game window
when there's none. Only one runs on the PC (a named mutex, MUTEX). Its rate and searches follow who's attached
(`Clients`): with a command that hunts, ACTIVE_HZ and the feed's schedule (her band every 0.2 s, the whole screen every
second); with none, IDLE_HZ and IDLE (her band every 0.5 s, the screen every 2 s, the drops at `watch`'s old rates).
Her throws are known to the tracker while a command that tells them is attached. Its telemetry is `watch`'s was (a
scene 10 times a second, the tracks twice, the events, her EXP: `Publisher`), in <time>-perceive.jsonl.

The commands are its clients (`ServiceClient`, a localization.feed.Feed: what Senses reads), attaching and detaching
as they start and end, any number at once, none of them its children:
- a named pipe (multiprocessing.connection, ADDRESS, an authkey kept in lab/perceive.json beside its pid) to each:
  down it each frame's state (plain data, a few KB, pickled once) by a writer thread of the client's own, so a client
  that doesn't read never holds up the frame loop (its queue keeps the newest QUEUE_KEEP; the older are dropped,
  counted, and a `gap` said); up it what the command asks: `hello` (its name, pid, whether it sends keys, tells her
  throws, wants the hunt's pace, its map), `ask` (what to search on the next frame: localization.worldstate.Asks,
  merged with the others'), `throw`, `full` (the next frame at full size: the pet's panels, the pick-up log's OCR),
  `status`, `bye`;
- the frames in shared memory (localization.feed.Ring: RING_SLOTS look frames, FULL_SLOTS full-size ones), each
  run's own (`epoch`): when the map changes (a command that hunts names another, or one that acts while none hunts:
  Service._hello) or the game is first seen, a new run with new rings, told to every client.
A client gone (its pipe closed) is dropped. A client of a service gone (or silent SILENT_S) reconnects every
RECONNECT_S, and starts one (`ensure`: detached, through WMI as the babysitter does, so it outlives the command) at
most every ENSURE_EVERY_S. A service whose frame loop hangs ends itself (STUCK_S), for another to be started.

Measured before it was built (D:/MapleHelper-scratch/perceive: README, The perception service): a state reaches a
client 0.13-0.25 ms after it goes, a frame copies out of the ring in 1-1.6 ms, and a client that never read held the
frame loop up for 10.5 s with the writes in the loop, none with a writer thread per client.
"""

import ctypes
import json
import multiprocessing.connection
import os
import pickle
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .. import paths, telemetry, window
from ..perception.framecheck import BadLook, Problem
from ..screen import Geometry, geometry
from . import feed
from .feed import Feed, Ring, Runner, frame_shape, newest_fresh, newest_frame, to_ask, adds
from .tracking import Event
from .worldstate import (LOOT_EVERY_S, LOOT_WIDE_EVERY_S, Asks, Schedule, SearchThread, WorldState, WorldStep, to_scene)

ADDRESS = r"\\.\pipe\gamelab-perceive"
MUTEX = "Local\\gamelab-perceive"
# `perceive --frames` (saved frames, offline) has a pipe, mutex and info file of its own: the live service starts and is
# reached beside it (with the live ones, a run of it on the PC kept the service from starting, and a command from the
# same checkout acted on saved frames). A command reaches it with GAMELAB_PERCEIVE=saved (`which`), and starts none.
SAVED_ADDRESS = ADDRESS + "-saved"
SAVED_MUTEX = MUTEX + "-saved"
WHICH = "GAMELAB_PERCEIVE"
RING_SLOTS = 8        # look frames kept in shared memory: at 30 Hz the newest 267 ms (25 MB at 1366x768)
FULL_SLOTS = 2        # full-size frames, for the `full` asks (at 1920x1080 a look frame is half size)
ACTIVE_HZ = 30.0      # frames a second with a command that hunts attached ...
IDLE_HZ = 10.0        # ... and with none (`watch`'s rate in the babysitter: 1.21 cores on the saved frames, the
                      # searches the most of it)
ACTIVE = Schedule()   # the feed's: her band every 0.2 s, the whole screen every second, the drops when asked
IDLE = Schedule(band_s=0.5, wide_s=2.0, loot_s=LOOT_EVERY_S, loot_wide_s=LOOT_WIDE_EVERY_S)
QUEUE_KEEP = 64       # states a client's writer holds at most (2 s at 30 Hz): older ones are dropped, a gap said
KEEP = 8              # states a client keeps (feed.KEEP)
EVENTS_KEPT = 10_000  # events a client keeps for its next drain at most
RECONNECT_S = 0.5     # s between a client's tries to reach the service
ENSURE_EVERY_S = 10.0  # s at least between two starts of the service by a client (ensure)
HELLO_WAIT_S = 3.0    # s a command waits at its start for the service's welcome before it says why there's none
FULL_WAIT_S = 1.0     # s a `full` ask waits for its frame
STATUS_S = 10.0       # s between the service's lines on the console
# A service whose frame loop hasn't gone round for STUCK_S (BUILD_STUCK_S while it makes a run: the map's templates
# loaded), or whose search has been out that long, ends its process (EXIT_STUCK): its clients' `ensure` and the
# babysitter start another. With the process and its pipe up and the loop hung, every look found nothing to act on and a
# hunt stood still to its end, nothing restarting it. The loop goes round at least every 1/IDLE_HZ s, and every
# NOT_VISIBLE_S without the game; a search takes 0.03-0.3 s. A client that hears nothing for SILENT_S (a state comes
# every 0.1 s at the least, every 0.25 s without the game) drops its connection and reconnects.
STUCK_S = 10.0
BUILD_STUCK_S = 60.0
EXIT_STUCK = 3
SILENT_S = 10.0
# watch's records (now the service's): scenes 10 times a second, tracks twice a second (a scene is ~400 bytes, the
# tracks ~850: about 20 MB an hour, which telemetry's 2 GB hold 4 days of)
SCENE_S = 0.1
TRACKS_S = 0.5
EVENTS = ("died", "exp_gain")  # tracker events published as telemetry 'event' records (and hits, new ones)


def saved(which: str | None = None) -> bool:
    """Whether the service meant is the one on saved frames (`perceive --frames`): `which`, by default GAMELAB_PERCEIVE's."""
    return (os.environ.get(WHICH, "") if which is None else which) == "saved"


def pipe_address(saved_frames: bool | None = None) -> str:
    """The pipe of the live service, or of the one on saved frames (`saved`)."""
    return SAVED_ADDRESS if (saved() if saved_frames is None else saved_frames) else ADDRESS


def info_path(saved_frames: bool | None = None) -> Path:
    """Where the running service says how to reach it (its pid, the pipe, the authkey): lab/perceive.json, or
    lab/perceive-saved.json for the one on saved frames (`saved`)."""
    on_saved = saved() if saved_frames is None else saved_frames
    return paths.LAB / ("perceive-saved.json" if on_saved else "perceive.json")


def log_path() -> Path:
    """Where a service started by `ensure` writes its console."""
    return paths.LAB / "perceive.log"


# --- one at a time, and started when none runs ---------------------------------------------------------------

def only_one(name: str = MUTEX):
    """A named mutex held for as long as this process runs, so only one service runs on the PC: its handle, or None if
    another service holds it (control.rolling.only_one's way)."""
    if sys.platform != "win32":
        return True
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        k32.CloseHandle(handle)
        return None
    return handle


def running(name: str = MUTEX) -> bool:
    """Whether a service holds its mutex now (one is running)."""
    if sys.platform != "win32":
        return False
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenMutexW.restype = wintypes.HANDLE
    k32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.OpenMutexW(0x00100000, False, name)  # SYNCHRONIZE
    if handle:
        k32.CloseHandle(handle)
        return True
    return False


_ENSURED = [-1e18]  # when ensure last started one


def ensure(map_id: int | None = None, log: Callable[[str], None] = print) -> bool:
    """Start the service (`python -m gamelab perceive`, from this checkout, its console in lab/perceive.log) unless one
    runs, at most every ENSURE_EVERY_S: detached through WMI, as the babysitter starts its programs, so it outlives the
    command that started it and a restart of whatever started that. True if one was started. None is started for a
    command reaching the one on saved frames (GAMELAB_PERCEIVE=saved: it's started by hand)."""
    now = time.monotonic()
    if saved() or now - _ENSURED[0] < ENSURE_EVERY_S or running():
        return False
    _ENSURED[0] = now
    command = (f'cmd.exe /c set PYTHONUNBUFFERED=1&& set OPENBLAS_NUM_THREADS=1&& "{sys.executable}" -u -X utf8 -m '
               f'gamelab perceive' + ("" if map_id is None else f" --map {int(map_id)}") + f' >> "{log_path()}" 2>&1')
    try:
        _start_detached(command, str(paths.REPO))
    except (OSError, subprocess.SubprocessError) as e:
        log(f"perception: couldn't start the service ({type(e).__name__}: {e})")
        return False
    log(f"perception: started the service (map {map_id}); its console: {log_path()}")
    return True


def _start_detached(command: str, folder: str) -> None:
    """Run `command` in `folder` as a process of WMI's (Win32_Process.Create), hidden: not this one's child."""
    script = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
              "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$env:GAMELAB_CMD; "
              "CurrentDirectory=$env:GAMELAB_DIR; ProcessStartupInformation=$si} | Out-Null")
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], check=True, timeout=60,
                   env=dict(os.environ, GAMELAB_CMD=command, GAMELAB_DIR=folder), capture_output=True)


def settle() -> None:
    """The service's process: normal priority (it isn't starved when the machine is busy: at below normal it didn't
    start within 30 s beside 22 busy processes, at normal it looked at 14 frames a second), OpenCV on one thread (the
    monster search's own threads do the work: feed.quiet), and Windows' timers at 1 ms (its waits ended on 15.6 ms ticks
    otherwise: a 33 ms wait took 47)."""
    import cv2
    cv2.setNumThreads(1)
    if sys.platform == "win32":
        ctypes.WinDLL("winmm").timeBeginPeriod(1)


# --- the game on screen ---------------------------------------------------------------------------------------

def screen_world(map_id: int, title: str, template: np.ndarray, band: tuple[float, float],
                 schedule: Schedule = IDLE, geo: Geometry = None, log: Callable[[str], None] = print) -> Runner:
    """The game on screen (perception.frames.ScreenFrames on the process's grabber: never brought to the front), with
    monster templates for map `map_id` and her name tag (`template`, calibration's), for a game window of geometry
    `geo`. Raises RuntimeError if the map has a minimap picture that won't load."""
    from ..perception.eyes import Eyes
    from ..perception.frames import ScreenFrames
    from ..perception.sprites import detector_for
    from . import mapdata
    from .senses import Senses
    model = mapdata.load(map_id)
    frames = ScreenFrames(title, geo=geo, log=log)
    try:  # as the commands that act load it: the service doesn't place her with the first map's origin
        picture = mapdata.load_minimap_picture(model.id)
    except (KeyError, OSError, ValueError) as e:
        if model.minimap.get("width"):  # the map has one: without it her dot is read with the first map's origin
            raise RuntimeError(f"map {model.id}'s minimap picture didn't load ({e!r})") from None
        picture = None
    eyes = Eyes(title, template, detector_for(model, geo), screen=frames.grabber, geo=geo, minimap_picture=picture)
    eyes.log = log
    senses = Senses(eyes, model, band)
    return Runner(frames, eyes, WorldStep(senses, False, schedule, SearchThread(senses)), title)


def serve_saved(files: list[str], map_id: int, template: np.ndarray, band: tuple[float, float], address: str,
                info: str, seconds: float | None = None, hold: int = 30, quiet: bool = True) -> None:
    """A service on saved frames (feed.saved_world, each `hold` frames in a row, round and round) at a pipe of its own
    (`address`, `info`), for `seconds` (None: until killed): a process's target, offline (sim.soak's, the measurements').
    `quiet`: nothing said."""
    from PIL import Image

    from ..screen import of_frame
    settle()
    geo = of_frame(np.asarray(Image.open(files[0]).convert("RGB")))
    svc = Service(lambda m, g: feed.saved_world(files, m, template, band, hold=hold, schedule=IDLE), map_id,
                  lambda: (geo, None), address=address, info=Path(info), log=(lambda m: None) if quiet else print,
                  stuck_s=STUCK_S)
    svc.serve(None if seconds is None else time.monotonic() + seconds)


def window_geometry(title: str) -> tuple[Geometry | None, Problem | None]:
    """The game window's geometry now (screen.Geometry, from its client size), or why there's none to read: gone,
    minimized (its client area isn't the game's then), a size no templates are for."""
    from ..perception import framecheck
    try:
        hwnd = window.find_game(title)
        if window.minimized(hwnd):
            return None, Problem("minimized", "the game is minimized: waiting for it to be restored")
        area = window.client_rect(hwnd)
    except RuntimeError as e:
        return None, Problem("gone", f"the game isn't running ({e})")
    size = (area["width"], area["height"])
    if size not in ((1366, 768), (1920, 1080)):
        return None, framecheck.size_problem(size, geometry(1366, 768))
    return geometry(*size), None


# --- the service ------------------------------------------------------------------------------------------------

@dataclass
class Hello:
    """What a client said of itself."""
    name: str = "?"
    pid: int = 0
    acts: bool = False    # it sends keys (the game is brought forward for it; its map is followed)
    throws: bool = False  # it tells her throws (the tracker's kills are hers only after one her way)
    fast: bool = False    # it hunts: the frames at ACTIVE_HZ and the feed's schedule while it's attached
    map: int | None = None


class Writer:
    """A client's pipe, written by a thread of its own from a queue: the newest QUEUE_KEEP states at most, the older
    dropped (counted, and a `gap` said before the next), so a client that doesn't read never holds anything up (a pipe's
    buffer is 8 KB, and a write waits until it's read). What's said to the client alone (`put` with `keep`: its
    welcome, a new run, a full frame's answer) is never dropped, and goes first."""

    def __init__(self, conn, keep: int = QUEUE_KEEP, name: str = "perceive-writer"):
        self.conn, self.keep = conn, keep
        self.q: deque[bytes] = deque()
        self.said: deque[bytes] = deque()  # (never dropped)
        self.cond = threading.Condition()
        self.dropped, self.gap, self.alive = 0, 0, True
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def put(self, data: bytes, keep: bool = False) -> None:
        with self.cond:
            if not self.alive:
                return
            if keep:
                self.said.append(data)
                self.cond.notify()
                return
            if len(self.q) >= self.keep:
                self.q.popleft()
                self.dropped += 1
                self.gap += 1
            self.q.append(data)
            self.cond.notify()

    def close(self) -> None:
        with self.cond:
            self.alive = False
            self.cond.notify()

    def _run(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.q or self.said or not self.alive)
                if not self.alive:
                    return
                if self.said:
                    gap, data = 0, self.said.popleft()
                else:
                    gap, self.gap = self.gap, 0
                    data = self.q.popleft()
            try:
                if gap:
                    self.conn.send_bytes(pickle.dumps(("gap", gap)))
                self.conn.send_bytes(data)
            except (OSError, EOFError, ValueError):
                with self.cond:
                    self.alive = False
                return


class _Client:
    """One command attached: its pipe, its writer, what it said of itself."""

    def __init__(self, cid: int, conn):
        self.id, self.conn = cid, conn
        self.hello: Hello | None = None
        self.writer = Writer(conn, name=f"perceive-writer-{cid}")
        self.since = time.monotonic()

    def send(self, *msg) -> None:
        """A message to this client alone (never dropped)."""
        self.writer.put(pickle.dumps(msg), keep=True)


class Service:
    """The perception service's process (see the top). `make(map_id, geo)` builds what it looks with (screen_world, or
    saved frames: feed.saved_world); `geometry()` says the game window's geometry or why there's none (window_geometry;
    saved frames: theirs). Clients reach it at `address` with `authkey` (written to `info` for them). `publish`: its
    telemetry (Publisher), or None. `stuck_s`: its watchdog's (STUCK_S: the process ends if the frame loop hangs that
    long; None, the tests' in-process services: none)."""

    def __init__(self, make: Callable[[int, Geometry], Runner], map_id: int,
                 geometry: Callable[[], tuple[Geometry | None, Problem | None]], *, address: str = ADDRESS,
                 authkey: bytes | None = None, info: Path | None = None, log: Callable[[str], None] = print,
                 publish: "Publisher | None" = None, idle_hz: float = IDLE_HZ, active_hz: float = ACTIVE_HZ,
                 stuck_s: float | None = None):
        self.make, self.map_id, self.geometry, self.log = make, map_id, geometry, log
        self.address, self.authkey = address, authkey or secrets.token_bytes(32)
        self.info = info
        self.publish = publish
        self.idle_hz, self.active_hz = idle_hz, active_hz
        self.stuck_s = stuck_s
        self._beat = (time.monotonic(), BUILD_STUCK_S)  # when the frame loop last went round, and how long it may not
        self.runner: Runner | None = None
        self.geo: Geometry | None = None
        self.epoch = 0
        self.ring: Ring | None = None
        self.full_ring: Ring | None = None
        self.clients: dict[int, _Client] = {}
        self._next_id = 1
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._inbox: deque = deque()      # (client, message) for the frame loop: asks, throws, full asks
        self._fulls: list = []            # full asks waiting for their frame: (client, request id, after)
        self._full_seq = 0
        self._stop = False
        self._rebuild: int | None = None  # a map to look at next (an acting client's)
        self._seq = 0                     # states made while there's no runner
        self.counts = {"connected": 0, "states": 0, "dropped": 0, "fulls": 0, "epochs": 0, "refused": 0}
        self._status = time.monotonic()
        self._frames_since = 0
        self._listener = None

    # --- clients ---------------------------------------------------------------------------------------------

    def listen(self) -> None:
        """Open the pipe, say how to reach it (`info`), and accept clients on a thread of its own."""
        self._listener = multiprocessing.connection.Listener(self.address, family="AF_PIPE", authkey=self.authkey)
        if self.info is not None:
            write_info(self.info, {"pid": os.getpid(), "address": self.address, "authkey": self.authkey.hex(),
                                   "started": time.time(), "map": self.map_id})
        threading.Thread(target=self._accept, name="perceive-accept", daemon=True).start()

    def _accept(self) -> None:
        while not self._stop:
            try:
                conn = self._listener.accept()
            except (OSError, EOFError, multiprocessing.AuthenticationError):
                if self._stop:
                    return
                continue
            with self._lock:
                c = _Client(self._next_id, conn)
                self._next_id += 1
                self.clients[c.id] = c
                self.counts["connected"] += 1
            threading.Thread(target=self._read, args=(c,), name=f"perceive-reader-{c.id}", daemon=True).start()

    def _read(self, c: _Client) -> None:
        """A client's requests, as they come: what it says of itself and `status` answered here; asks, throws and full
        asks handed to the frame loop."""
        try:
            while True:
                msg = c.conn.recv()
                kind = msg[0]
                if kind == "hello":
                    self._hello(c, Hello(**msg[1]))
                elif kind == "status":
                    c.send("status", self.status())
                elif kind == "bye":
                    break
                else:
                    with self._cond:
                        self._inbox.append((c, msg))
                        self._cond.notify_all()
        except (EOFError, OSError, pickle.UnpicklingError, TypeError, ValueError):
            pass
        self._drop(c)

    def _hello(self, c: _Client, hello: Hello) -> None:
        """A client said what it is: welcomed with the run's rings, or refused (a second command that hunts, on another
        map). The map looked at is held by the commands that hunt (`fast`): one on another map than the one looked at
        has the service look at its map (a new run); any other client (a viewer, or `pet`, which sends keys but reads
        only whole frames) moves it only while none holds it, and is never refused. (The pet's babysitter, `pet --every
        1` on its map outside hunts, held it as a command that acts, and a hunt on any other map was refused at its
        start: the babysitter stops the pet only once it sees a hunt.)"""
        with self._lock:
            holding = [o for o in self.clients.values() if o is not c and o.hello is not None and o.hello.fast
                       and o.hello.map is not None]
            wanted = self._rebuild if self._rebuild is not None else self.map_id
            if hello.map is not None and hello.map != wanted:
                held = [o for o in holding if o.hello.map == wanted]
                if held and hello.fast:
                    self.counts["refused"] += 1
                    c.send("refused", f"the perception service is looking at map {wanted} for {held[0].hello.name} "
                                      f"(pid {held[0].hello.pid}): stop it first, or run this on map {wanted}")
                    c.hello = hello
                    return
                if not held:
                    self._rebuild = hello.map
                    self._cond.notify_all()
            c.hello = hello
            c.send("welcome", self._welcome())
        self.log(f"perception: {hello.name} (pid {hello.pid}) attached" + (", sends keys" if hello.acts else "")
                 + (f", map {hello.map}" if hello.map is not None else ""))

    def _drop(self, c: _Client) -> None:
        with self._lock:
            if self.clients.pop(c.id, None) is None:
                return
            c.writer.close()
            self.counts["dropped"] += c.writer.dropped
            self._fulls = [f for f in self._fulls if f[0] is not c]
            self._follow()
        try:
            c.conn.close()
        except OSError:
            pass
        if c.hello is not None:
            self.log(f"perception: {c.hello.name} (pid {c.hello.pid}) detached")

    def _follow(self) -> None:
        """A client gone: with no command that hunts left to hold the map, the newest command that acts on another map
        has the service look at its map (its states wait for that run: ServiceClient.on_map)."""
        hellos = [o.hello for o in self.clients.values() if o.hello is not None and o.hello.map is not None]
        if any(h.fast for h in hellos):
            return
        wanted = self._rebuild if self._rebuild is not None else self.map_id
        acting = [h for h in hellos if h.acts]
        if acting and acting[-1].map != wanted:
            self._rebuild = acting[-1].map
            self._cond.notify_all()

    def _welcome(self) -> dict:
        """What a client needs to read this run's frames (None for the rings while the game hasn't been seen)."""
        return {"epoch": self.epoch, "pid": os.getpid(), "map": self.map_id,
                "next": self._rebuild if self._rebuild is not None else self.map_id,  # (a new run due: its map)
                "ring": None if self.ring is None else self.ring.name, "slots": RING_SLOTS,
                "shape": None if self.ring is None else list(self.ring.shape),
                "full": None if self.full_ring is None else self.full_ring.name, "full_slots": FULL_SLOTS,
                "full_shape": None if self.full_ring is None else list(self.full_ring.shape)}

    def _broadcast(self, *msg) -> None:
        data = pickle.dumps(msg)
        with self._lock:
            targets = [c for c in self.clients.values() if c.hello is not None]
        for c in targets:
            c.writer.put(data, keep=True)

    def attached(self) -> list[Hello]:
        with self._lock:
            return [c.hello for c in self.clients.values() if c.hello is not None]

    def fast(self) -> bool:
        """Whether a command that hunts is attached (ACTIVE_HZ, ACTIVE)."""
        return any(h.fast for h in self.attached())

    def rate(self) -> float:
        return self.active_hz if self.fast() else self.idle_hz

    def status(self) -> dict:
        with self._lock:
            return {"pid": os.getpid(), "epoch": self.epoch, "map": self.map_id,
                    "frames": None if self.runner is None else self.runner.step.counts["frames"],
                    "clients": [{"name": c.hello.name, "pid": c.hello.pid, "acts": c.hello.acts,
                                 "dropped": c.writer.dropped} for c in self.clients.values() if c.hello is not None],
                    "hz": self.rate(), **self.counts}

    # --- the frame loop ----------------------------------------------------------------------------------------

    def serve(self, until: float | None = None) -> None:
        """Look at the game until `until` (time.monotonic) or `stop`: the frame loop (feed.run) on the run of now,
        a new run when the map changes or the game is first seen; while there's nothing to look with (no game window
        yet), a state every feed.NOT_VISIBLE_S saying why."""
        if self._listener is None:
            self.listen()
        if self.stuck_s is not None:
            threading.Thread(target=self._watchdog, name="perceive-watchdog", daemon=True).start()
        try:
            while not self._stop and (until is None or time.monotonic() < until):
                if self.runner is None or self._rebuild is not None:
                    if not self._build():
                        self._not_visible()
                        continue
                self._tune()
                feed.run(self.runner, self.rate, self._poll, self._emit, until, self._shown)
        finally:
            self.close()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _went_round(self, limit: float | None = None) -> None:
        """The frame loop went round (its next turn due within `limit` s, by default `stuck_s`)."""
        self._beat = (time.monotonic(), limit if limit is not None else (self.stuck_s or STUCK_S))

    def stuck(self, now: float | None = None) -> str | None:
        """Why the service is stuck (the frame loop not gone round in time, a search out too long), or None."""
        now = time.monotonic() if now is None else now
        at, limit = self._beat
        if now - at > limit:
            return f"the frame loop hasn't gone round for {now - at:.1f} s"
        searches = None if self.runner is None else self.runner.step.searches
        since = getattr(searches, "since", None)
        if since is not None and now - since > (self.stuck_s or STUCK_S):
            return f"a search has been out for {now - since:.1f} s"
        return None

    def _watchdog(self) -> None:
        """The process ended (EXIT_STUCK) once it's `stuck`, so that another is started (the clients' `ensure`, the
        babysitter): its pipe and mutex go with it."""
        while not self._stop:
            time.sleep(min(1.0, self.stuck_s / 4))
            if self._stop:
                return
            if (why := self.stuck()) is not None:
                self.log(f"perception: stuck ({why}): the service ends, for another to be started")
                for out in (sys.stdout, sys.stderr):
                    try:
                        out.flush()
                    except (OSError, ValueError, AttributeError):
                        pass
                os._exit(EXIT_STUCK)

    def _build(self) -> bool:
        """A new run: the map to look at (an acting client's, or the one before) on the window's geometry, new rings,
        told to every client. False while the window can't be read (no game, minimized, another size)."""
        self._went_round(BUILD_STUCK_S)
        geo, problem = (self.geo, None) if self.geo is not None else self.geometry()
        if geo is None:
            self._why = problem
            return False
        with self._lock:
            map_id = self._rebuild if self._rebuild is not None else self.map_id
        try:
            runner = self.make(map_id, geo)
        except Exception as e:  # (its minimap picture won't load: the run before goes on, if there's one)
            self.log(f"perception: can't look at map {map_id}: {type(e).__name__}: {e}")
            with self._lock:
                self._rebuild = None
            self._why = Problem("picture", f"the perception service can't look at map {map_id}: {e}")
            return self.runner is not None
        self._went_round()
        runner.step.searches.wake = self.wake  # (the frame loop may be waiting for a request or its next frame)
        old = (self.runner, self.ring, self.full_ring)
        with self._lock:
            self.runner, self.geo, self.map_id, self._rebuild = runner, geo, map_id, None
            self.epoch += 1
            self.counts["epochs"] += 1
            self.ring = Ring(slots=RING_SLOTS, shape=frame_shape(geo))
            self.full_ring = Ring(slots=FULL_SLOTS, shape=(geo.client[1], geo.client[0], 3))
            welcome = self._welcome()
        self._broadcast("epoch", welcome)
        if self.publish is not None:
            self.publish.started(runner, map_id)
        self.log(f"perception: looking at map {map_id}, the game's client {geo.client[0]}x{geo.client[1]} "
                 f"(run {self.epoch})")
        if old[0] is not None:
            old[0].step.close()
        for ring in old[1:]:
            if ring is not None:
                ring.close(unlink=True)
        return True

    def _tune(self) -> None:
        """The step as the clients attached want it: the hunt's schedule while one hunts (ACTIVE) or IDLE, her throws
        known while one tells them."""
        attached = self.attached()
        step = self.runner.step
        step.schedule = ACTIVE if any(h.fast for h in attached) else IDLE
        step.tracker.throws_known = any(h.throws for h in attached)

    def _not_visible(self) -> None:
        """No run to look with: a state saying why, to every client, then a wait (asks are dropped meanwhile). Its
        number goes on from the last state's (a run's frames are numbered from 1, the run before's carry on here)."""
        self._went_round()
        self._seq += 1
        why = getattr(self, "_why", None) or Problem("gone", "the game isn't running")
        st = WorldState(self._seq, time.monotonic(), False, why=why.why, bad=why, epoch=self.epoch)
        self._send_state(st)
        self._answer_fulls(None, st)
        with self._cond:
            self._cond.wait_for(lambda: self._stop, feed.NOT_VISIBLE_S)
            self._inbox.clear()

    def _poll(self, wait: float) -> tuple[Asks | None, bool]:
        """feed.run's: what the clients asked (merged), and whether to stop (asked to, or a new run is due), waiting up
        to `wait` s for a request or the searches' answer."""
        self._went_round()
        deadline = time.monotonic() + wait
        asks = None
        with self._cond:
            while True:
                while self._inbox:
                    c, msg = self._inbox.popleft()
                    asks = self._take(c, msg, asks)
                if self._stop or self._rebuild is not None:
                    return asks, True
                left = deadline - time.monotonic()
                if asks is not None or left <= 0 or self._woken:
                    self._woken = False
                    break
                self._cond.wait(left)
        self._tune()
        return asks, False

    _woken = False

    def wake(self) -> None:
        """The searches' answer is in: the frame loop merges it (it may be waiting for a request or its next frame)."""
        with self._cond:
            self._woken = True
            self._cond.notify_all()

    def _take(self, c: _Client, msg: tuple, asks: Asks | None) -> Asks | None:
        """A request for the frame loop: an ask merged into `asks` (mesos only if each client asked only for mesos), a
        throw told to the step, a full ask kept for its frame (and its time asked of the frames)."""
        kind = msg[0]
        if kind == "ask":
            ask: Asks = msg[1]
            if asks is None:
                return ask
            merged = asks.merge(ask)
            return replace(merged, mesos_only=asks.mesos_only and ask.mesos_only)
        if kind == "throw" and self.runner is not None:
            self.runner.step.throw(msg[1], msg[2])
        elif kind == "full":
            self._fulls.append((c, msg[1], msg[2]))
            ask = Asks(msg[2])
            return ask if asks is None else replace(asks.merge(ask), mesos_only=asks.mesos_only)
        return asks

    def _shown(self, frame) -> None:
        """A frame showing the game, as it's taken: the full asks waiting for it answered with it."""
        self._answer_fulls(frame, None)

    def _answer_fulls(self, frame, st: WorldState | None) -> None:
        with self._lock:
            due = [f for f in self._fulls if (frame.t if frame is not None else st.t) >= f[2] - 1e-9]
            self._fulls = [f for f in self._fulls if f not in due]
        for c, rid, _ in due:
            if frame is not None and self.full_ring is not None and frame.full is not None:
                self._full_seq += 1
                slot = self.full_ring.put(self._full_seq, frame.t, frame.full)
                self.counts["fulls"] += 1
                c.send("full", rid, {"seq": self._full_seq, "slot": slot, "t": frame.t, "epoch": self.epoch,
                                     "bad": None if slot is not None else Problem("size", "another size")})
            else:
                bad = st.bad if st is not None else Problem("gone", "no frame")
                c.send("full", rid, {"seq": None, "slot": None, "t": st.t if st is not None else time.monotonic(),
                                     "epoch": self.epoch, "bad": bad})

    def _emit(self, st: WorldState, frame) -> None:
        """feed.run's: each frame's state into the ring and to every client (pickled once), and published."""
        if st.frame is not None and not st.late:
            st.slot = self.ring.put(st.seq, st.t, st.frame)
        elif st.frame is not None and st.slot is None:
            st.slot = self.ring.slot_of(st.seq)  # (feed.serve's: a late state copied before its frame went in)
        if not st.visible:
            self._answer_fulls(None, st)
        self._send_state(replace(st, frame=None, epoch=self.epoch), st)
        if not st.late:
            self._frames_since += 1
        self._say()

    def _send_state(self, out: WorldState, st: WorldState | None = None) -> None:
        if not out.late:
            self._seq = max(self._seq, out.seq)  # (`_not_visible`'s go on from it)
        data = pickle.dumps(("state", out))
        with self._lock:
            targets = [c for c in self.clients.values() if c.hello is not None]
        for c in targets:
            c.writer.put(data)
        self.counts["states"] += 1
        if self.publish is not None and st is not None:
            self.publish.state(st)

    def _say(self) -> None:
        now = time.monotonic()
        if now - self._status < STATUS_S:
            return
        rate = self._frames_since / max(1e-9, now - self._status)
        self._status, self._frames_since = now, 0
        names = ", ".join(f"{h.name}{'*' if h.acts else ''}" for h in self.attached()) or "nobody"
        line = f"{rate:.1f} frames a second (asked {self.rate():.0f}); attached: {names}"
        if self.publish is not None:
            line += self.publish.status(now)
        self.log(line)

    def close(self) -> None:
        self._stop = True
        with self._lock:
            clients = list(self.clients.values())
        for c in clients:
            self._drop(c)
        if self.runner is not None:
            self.runner.step.close()
        for ring in (self.ring, self.full_ring):
            if ring is not None:
                ring.close(unlink=True)
        self.ring = self.full_ring = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self.publish is not None:
            self.publish.close()


def write_info(path: Path, info: dict) -> None:
    """lab/perceive.json, written whole (a client never reads half of it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(info), encoding="utf-8")
    tmp.replace(path)


def read_info(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- what the service publishes (watch's records) --------------------------------------------------------------

class Publisher:
    """The service's telemetry, as `watch` published it (README, Continuous perception): a `scene` SCENE_S apart (every
    monster on screen, the drops last searched), the `tracks` TRACKS_S apart, an `event` for each death, EXP gain and
    new hit, and an `exp` record for each gain (her EXP as read, the kinds it was, EXP a minute), into the telemetry
    file it was made with (telemetry.record_to). And the console's line every STATUS_S (`status`)."""

    def __init__(self, band: tuple[float, float], log: Callable[[str], None] = print):
        self.band, self.log = band, log
        self.runner: Runner | None = None
        self.model = None
        self.w = {"scene": -1e9, "tracks": -1e9, "visible": None, "loot": [], "looked": []}

    def started(self, runner: Runner, map_id: int) -> None:
        self.runner, self.model = runner, runner.step.model
        telemetry.emit("session", map=map_id, command="perceive", mode="watch")

    def state(self, st: WorldState) -> None:
        if self.runner is None:
            return
        w, now = self.w, time.monotonic()
        if st.late:  # a frame searched, given again with what its searches found: its drops, its events
            if st.loot is not None:
                w["loot"], w["looked"] = st.loot, st.looked
            for e in st.events:
                if e.kind in EVENTS:
                    telemetry.emit("event", kind=e.kind, text=e.text(), track=e.track)
            return
        if st.visible != w["visible"]:
            w["visible"] = st.visible
            self.log("the game is on screen: watching" if st.visible else f"not watching: {st.why} (nothing is read)")
        if st.visible and st.bad is None:
            for e in st.events:
                if e.kind in EVENTS or (e.kind == "hp_drop" and e.data.get("new")):
                    telemetry.emit("event", kind=e.kind, text=e.text(), track=e.track)
                if e.kind == "died":
                    by = e.data.get("by")
                    self.log(f"  {e.data['kind']} #{e.track} died on platform {e.data['platform']}: "
                             + ("hers" if e.data["cause"] == "hers" else "someone else's"
                                + (f" (another player's, #{by})" if by is not None else "")))
            if st.loot is not None:
                w["loot"], w["looked"] = st.loot, st.looked
            if now - w["scene"] >= SCENE_S:
                w["scene"] = now
                s = to_scene(st, self.model, self.band, mobs=True, wide=True)
                s.loot, s.looked = w["loot"], w["looked"]  # as of the last frame searched for drops
                telemetry.emit("scene", **s.record())
            if now - w["tracks"] >= TRACKS_S:
                w["tracks"] = now
                telemetry.emit("tracks", list=[tr.record() for tr in st.tracks], frames=st.seq)
        meter = self.runner.step.meter
        while self.runner.step.gains:  # each gain as the hunter records it, for the mini-map's EXP line
            g = self.runner.step.gains.popleft()
            if meter.at is not None:
                rate = meter.per_minute(now)
                telemetry.emit("exp", exp=meter.at[1], level=meter.at[0], gain=g.exp, kinds=list(g.kinds),
                               total=meter.total, per_minute=None if rate is None else round(rate, 1))

    def status(self, now: float) -> str:
        """The console line's part on what it saw: the monsters followed, deaths, her EXP a minute."""
        if self.runner is None:
            return ""
        step = self.runner.step
        c, meter = step.tracker.counts, step.meter
        for note in meter.notes:
            self.log(f"  {note}")
        meter.notes.clear()
        live = [tr for tr in step.tracker.tracks.values() if tr.state != "dead"]
        per_minute = meter.per_minute(now)
        exp = "" if meter.at is None else f"; EXP {meter.at[1]:,}" + (
            "" if per_minute is None else f", {per_minute:.0f} a minute")
        return (f"; {len(live)} monsters followed; deaths seen: {c['died hers']} hers, {c['died other']} others'"
                f"{exp}")

    def close(self) -> None:
        telemetry.stop()


# --- the commands' side -----------------------------------------------------------------------------------------

class ServiceClient(Feed):
    """The perception service from a command's side (a localization.feed.Feed: what Senses reads). `name`, `acts` (it
    sends keys: the game is brought forward for it, `bring_forward`), `throws` (it tells her throws), `fast` (it hunts:
    the service's hunting pace while it's attached; by default as `acts` and `throws`), `map_id` (the map it decides
    on): what it says of itself. It connects (and reconnects, every RECONNECT_S, and after SILENT_S without a word from
    the service) on a thread of its own; while it can't, `ensure` (service.ensure; None: never) starts the service.
    `address`, `info`: where to reach it (by default the live service's, or with GAMELAB_PERCEIVE=saved the one on saved
    frames: `pipe_address`, `info_path`; tests: their own). `title`: the game window's, for `bring_forward`. A command that
    acts on a map is handed only states of the service's run on that map (`on_map`), and of the run it was last
    welcomed to (a state of the run before can come after the new run's welcome, which goes first)."""

    def __init__(self, name: str, *, acts: bool = False, throws: bool = False, fast: bool | None = None,
                 map_id: int | None = None, title: str = "", address: str | None = None, info: Path | None = None,
                 ensure: Callable[..., bool] | None = ensure, log: Callable[[str], None] = print,
                 start: bool = True):
        self.hello = Hello(name, os.getpid(), acts, throws, acts and throws if fast is None else fast, map_id)
        self.title, self.address, self.info = title, address or pipe_address(), info or info_path()
        self._ensure, self.log = ensure, log
        self.states: deque[WorldState] = deque(maxlen=KEEP)
        self.events: deque[Event] = deque(maxlen=EVENTS_KEPT)
        self.dropped = 0           # events lost because nothing drained them
        self.gaps = 0              # states the service dropped for this client (it read too slowly)
        self.served = self.refused = 0
        self.stale = 0             # states of a run before the one it was welcomed to, left out
        self.on_state: Callable[[WorldState], None] | None = None
        self.alive, self.why = False, "not connected yet"
        self.welcome: dict | None = None
        self.refusal: str | None = None
        self.epoch = 0
        self.run: tuple[int, int] | None = None  # the service's pid and run its rings are of
        self.ring: Ring | None = None
        self.full_ring: Ring | None = None
        self.connects = 0          # times it connected
        self._fulls: dict[int, dict] = {}
        self._rid = 0
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._conn = None
        self._closing = False
        self._fronted = -1e18
        self._thread = None
        if start:
            self.start()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="perceive-client", daemon=True)
        self._thread.start()

    # --- the connection ---------------------------------------------------------------------------------------

    def _run(self) -> None:
        said = False
        while not self._closing:
            conn = self._connect()
            if conn is None:
                if self._ensure is not None:
                    self._ensure(self.hello.map, log=self.log)
                with self._cond:
                    self._cond.wait_for(lambda: self._closing, RECONNECT_S)
                continue
            said, silent = False, False
            try:
                self._send_on(conn, ("hello", self.hello.__dict__))
                while not self._closing:
                    if not conn.poll(SILENT_S):  # (a service whose frame loop hangs: its pipe up, nothing said)
                        silent = True
                        break
                    self._take(conn.recv())
                    if self.refusal is not None:
                        break
            except (EOFError, OSError, pickle.UnpicklingError, ValueError):
                pass
            except TypeError:  # (closed by `close` between the poll and the read: its handle None)
                if not self._closing:
                    raise
            with self._cond:
                self._conn = None
                down = self.alive
                self.alive = False
                self.why = self.refusal or ("closed" if self._closing else
                                            f"nothing from the perception service for {SILENT_S:g} s" if silent else
                                            "the perception service's process ended")
                self._cond.notify_all()
            try:
                conn.close()
            except OSError:
                pass
            if self.refusal is not None:
                return
            if down and not said:
                said = True
                self.log(f"perception: {self.why}; reconnecting")

    def _connect(self):
        info = read_info(self.info)
        if info is None or info.get("address", self.address) != self.address:
            self.why = f"the perception service isn't running (no {self.info})"
            return None
        try:
            conn = multiprocessing.connection.Client(self.address, family="AF_PIPE",
                                                     authkey=bytes.fromhex(info["authkey"]))
        except (OSError, EOFError, ValueError, KeyError, multiprocessing.AuthenticationError) as e:
            self.why = f"the perception service can't be reached ({type(e).__name__}: {e})"
            return None
        with self._cond:
            self._conn = conn
            self.connects += 1
        return conn

    def _take(self, msg) -> None:
        kind = msg[0]
        with self._cond:
            if kind == "state":
                st = msg[1]
                if st.epoch != self.epoch:  # (the run before's, queued before the new run's welcome went: it goes first)
                    self.stale += 1
                    return
                self.states.append(st)
                if self.on_state is not None:
                    self.on_state(st)
                if self.on_map():  # (another map's run's events aren't this command's)
                    room = EVENTS_KEPT - len(self.events)
                    self.dropped += max(0, len(st.events) - room)
                    self.events.extend(st.events)
            elif kind in ("welcome", "epoch"):
                self._open(msg[1])
                self.alive, self.why = True, ""
            elif kind == "full":
                self._fulls[msg[1]] = msg[2]
            elif kind == "gap":
                self.gaps += msg[1]
            elif kind == "refused":
                self.refusal = msg[1]
            elif kind == "status":
                self._fulls[-1] = msg[1]
            self._cond.notify_all()

    def _open(self, welcome: dict) -> None:
        """A new run's rings (a new epoch: its frames and track ids are new)."""
        self.welcome = welcome
        run = (welcome["pid"], welcome["epoch"])  # (a service started again counts its runs from 1 again)
        if run == self.run and (self.ring is not None or not welcome.get("ring")):
            return
        for ring in (self.ring, self.full_ring):
            if ring is not None:
                ring.close()
        self.ring = self.full_ring = None
        self.run, self.epoch = run, welcome["epoch"]
        if welcome.get("ring"):
            self.ring = Ring(welcome["ring"], slots=welcome["slots"], shape=tuple(welcome["shape"]))
            self.full_ring = Ring(welcome["full"], slots=welcome["full_slots"], shape=tuple(welcome["full_shape"]))
        self.states.clear()  # (the run before's: other frames, other track ids)

    def _send_on(self, conn, msg) -> None:
        with self._send_lock:
            conn.send(msg)

    def _send(self, *msg) -> bool:
        conn = self._conn
        if conn is None:
            return False
        try:
            self._send_on(conn, msg)
            return True
        except (OSError, ValueError):
            return False

    def ready(self, timeout: float = HELLO_WAIT_S) -> bool:
        """Wait (`timeout` s at most) to be welcomed, and if the service is to look at another map for it, for that run
        (no state of the map before is handed out then). Raises RuntimeError if the service refused this command
        (another command that acts holds it on another map)."""
        def welcomed():
            w = self.welcome or {}
            return self.refusal is not None or (self.alive and w.get("next", w.get("map")) == w.get("map"))
        with self._cond:
            self._cond.wait_for(welcomed, timeout)
        if self.refusal is not None:
            raise RuntimeError(self.refusal)
        return self.alive

    def status(self, timeout: float = 2.0) -> dict | None:
        """What the service says of itself (its clients, frames, epoch), or None."""
        with self._cond:
            self._fulls.pop(-1, None)
        if not self._send("status"):
            return None
        with self._cond:
            self._cond.wait_for(lambda: -1 in self._fulls, timeout)
            return self._fulls.pop(-1, None)

    # --- the Feed -------------------------------------------------------------------------------------------

    def on_map(self) -> bool:
        """Whether the states it's handed are of its map: a command that acts on one is handed only those of the
        service's run on it (once a reconnect has it welcomed to a service on another map, the babysitter's say, that
        service's run is moving to its map: its states till then are the other map's). Any other: whatever it looks at."""
        if not self.hello.acts or self.hello.map is None:
            return True
        w = self.welcome or {}
        return w.get("map") == w.get("next", w.get("map")) == self.hello.map

    def latest(self, want: Asks, timeout: float = feed.FEED_WAIT_S) -> WorldState | None:
        deadline, asked, seen = time.monotonic() + timeout, None, None
        with self._cond:
            while True:
                live = self.alive and self.on_map()
                st = newest_fresh(self.states, want) if live else None
                if st is not None:
                    self.served += 1
                    return st
                left = deadline - time.monotonic()
                newest = newest_frame(self.states) if live else None
                if (newest is not None and newest.t >= want.after and not newest.usable) or left <= 0:
                    self.refused += 1
                    return None
                if live:
                    seq = newest.seq if newest is not None else 0
                    ask = to_ask(want, newest) if asked is None or (feed.ASK_ONLY_STALE and seq != seen) else None
                    if ask is not None and (asked is None or adds(ask, asked)):
                        if self._send("ask", ask):
                            asked = ask if asked is None else asked.merge(ask)
                    seen = seq
                self._cond.wait(left)

    def problem(self) -> Problem | None:
        """Why there's no state to act on now: the newest one's (the game not on screen or not as expected), or the
        service not reached, or its run not on this command's map yet; None if the newest is usable."""
        with self._cond:
            if not self.alive:
                return Problem("perception", self.why or "the perception service isn't reached")
            if not self.on_map():
                w = self.welcome or {}
                return Problem("perception", f"the perception service looks at map {w.get('map')}, not "
                                             f"{self.hello.map}: waiting for its run on {self.hello.map}")
            newest = newest_frame(self.states)
            if newest is None:
                return Problem("perception", "no state from the perception service yet")
            return newest.bad

    def bring_forward(self, problem: Problem | None) -> None:
        """A command that sends keys brings the game to the front when it isn't (the service never does), once a second
        at most, and never while the window is unfit in a way only the user sets right (framecheck.WAIT_KINDS)."""
        from ..perception.framecheck import WAIT_KINDS
        if not self.hello.acts or problem is None or problem.kind != "window" or not self.title:
            return
        now = time.monotonic()
        if now - self._fronted < 1.0 or problem.kind in WAIT_KINDS:
            return
        self._fronted = now
        try:
            window.bring_to_front(self.title)
        except RuntimeError:
            pass

    def frame_of(self, st: WorldState) -> np.ndarray:
        ring = self.ring
        frame = None if ring is None or st.slot is None or st.epoch != self.epoch else ring.get(st.slot, st.seq)
        if frame is not None:
            return frame
        w, h = st.geo.frame_size
        return np.zeros((h, w, 3), np.uint8)

    def full_frame(self, after: float, timeout: float = FULL_WAIT_S) -> np.ndarray:
        """The game window at full size (RGB) as shown at `after` or later: the pet's panels, the pick-up log. Raises
        BadLook if the service had none to give (the game not on screen, the service not reached). A command that sends
        keys has the game brought to the front first if it wasn't, and asks again once (as its own capture did)."""
        try:
            return self._full(after, timeout)
        except BadLook as e:
            if not self.hello.acts or e.kind != "window":
                raise
            self._fronted = -1e18
            self.bring_forward(e.problem)
            time.sleep(0.3)
            return self._full(time.monotonic(), timeout)

    def _full(self, after: float, timeout: float) -> np.ndarray:
        with self._cond:
            self._rid += 1
            rid = self._rid
        if not self._send("full", rid, after):
            raise BadLook(Problem("perception", self.why or "the perception service isn't reached"))
        with self._cond:
            self._cond.wait_for(lambda: rid in self._fulls or not self.alive, timeout)
            got = self._fulls.pop(rid, None)
            ring = self.full_ring
        if got is None:
            raise BadLook(Problem("perception", self.why or f"no frame from the perception service within {timeout} s"))
        if got["bad"] is not None:
            raise BadLook(got["bad"])
        frame = None if ring is None or got["epoch"] != self.epoch else ring.get(got["slot"], got["seq"])
        if frame is None:
            raise BadLook(Problem("perception", "the frame was gone before it was copied"))
        return frame

    def wait_frame(self, timeout: float) -> bool:
        with self._cond:
            newest = lambda: max((st.seq for st in self.states), default=0)
            seen = newest()
            return self._cond.wait_for(lambda: not self.alive or newest() > seen, timeout) and self.alive

    def drain_events(self, upto: float | None = None) -> list[Event]:
        with self._cond:
            return feed._drain(self.events, upto)

    def throw(self, t: float, side: str) -> None:
        if self.hello.throws:
            self._send("throw", t, side)

    def close(self) -> None:
        self._closing = True
        self._send("bye")
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        with self._cond:
            self.alive = False
            self.why = self.why or "closed"
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(2.0)
        for ring in (self.ring, self.full_ring):
            if ring is not None:
                ring.close()
        self.ring = self.full_ring = None
