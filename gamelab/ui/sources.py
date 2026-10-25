"""Where the mini-map's records come from: a run followed live, a recorded run replayed, or a route
planned offline (no game needed)."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from .. import telemetry


# What the perception service publishes (localization.service.Publisher: it looks at every frame, at all times) is what
# the page shows of what's on screen; the rest (her routes, chains, the hunt's rules, the log) is the command's. A
# hunting command writes scenes of its own too (respawn, her memory and coverage read them), which the page leaves out
# while the service's are coming.
PERCEIVED = ("scene", "tracks", "exp")
PERCEIVED_EVENTS = ("died", "exp_gain", "hp_drop")
PERCEIVE_LIVE_S = 3.0  # s since the service's file last grew within which its records are the page's


def perceive_file(name: str) -> bool:
    """Whether a telemetry file is the perception service's (<time>-perceive.jsonl, or a part of it)."""
    return "-perceive" in name


class _Followed:
    """The newest telemetry file of a kind (`match` its names), followed as it grows; a newer run's file replaces it,
    a run's next part continues it (see LiveSource)."""

    def __init__(self, folder: Path, match, rescan_s: float):
        self.folder, self.match, self.rescan_s = folder, match, rescan_s
        self.path: Path | None = None
        self._follower: telemetry.Follower | None = None
        self._scanned: float | None = None  # when the folder was last looked through

    def poll(self, now: float) -> tuple[bool, list[dict]]:
        if self._scanned is None or now - self._scanned >= self.rescan_s:
            self._scanned = now
            newest = telemetry.newest(self.folder, match=self.match)
            if newest is not None and newest != self.path:
                follower = telemetry.Follower(newest)
                records = follower.poll()
                if self._follower is None or (records and not self._continues(records[0])):
                    self.path, self._follower = newest, follower
                    return True, records
                if records:  # the run's next part: the rest of this one, then on into it
                    rest = self._follower.poll()
                    self.path, self._follower = newest, follower
                    return False, rest + records
                # nothing in it yet (a part is made a moment before its first line): the next scan sees
        return False, (self._follower.poll() if self._follower else [])

    def _continues(self, first: dict) -> bool:
        """Whether a file starting with record `first` is the next part of the one followed."""
        return first.get("topic") == "session" and self.path is not None and first.get("continues") == self.path.name


class LiveSource:
    """The newest run's telemetry file in a folder, followed as it grows; a newer run's file replaces it. The file is
    read on every poll, but the folder looked through at most once every `rescan_s`: the web page polls 40 times a
    second, and a new run is still picked up within a second. A run's next part (its first record the session record
    that `continues` the file followed, telemetry.Recorder) isn't a new run: the rest of the part before is read, then
    the new part, and the window and the page go on with what they had (the tour's and the hunt loop's grounds and safe
    platforms are sent once a run, her trail and the log lines go back further).

    What's on screen comes from the perception service's file (perceive_file: PERCEIVED, and its PERCEIVED_EVENTS) while
    it grows (PERCEIVE_LIVE_S), and the rest from the newest command's; with no command's file, the service's is the
    run."""

    def __init__(self, folder: Path = telemetry.FOLDER, rescan_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        self.folder, self.rescan_s, self._clock = folder, rescan_s, clock
        self._run = _Followed(folder, lambda name: not perceive_file(name), rescan_s)
        self._perceived = _Followed(folder, perceive_file, rescan_s)
        self._perceived_at: float | None = None  # when the service's file last gave records
        self._alone = False  # the service's file is the run (no command's yet)

    @property
    def path(self) -> Path | None:
        """The run's file followed (a command's; the service's while there's none)."""
        return self._run.path or self._perceived.path

    def poll(self) -> tuple[bool, list[dict]]:
        """(whether a new run started, the records since the last poll)."""
        now = self._clock()
        fresh, records = self._run.poll(now)
        new_seen, seen = self._perceived.poll(now)
        if seen:
            self._perceived_at = now
        if self._run.path is None:  # no command's file: the service's is the run
            self._alone = self._alone or self._perceived.path is not None
            return new_seen, seen
        if self._alone:  # a command's file has come: it's the run now, from its start
            self._alone, fresh = False, True
        if self._perceived_at is None or now - self._perceived_at > PERCEIVE_LIVE_S:
            return fresh, records
        own = [r for r in records if r.get("topic") not in PERCEIVED]
        theirs = [r for r in seen if r.get("topic") in PERCEIVED
                  or (r.get("topic") == "event" and r.get("kind") in PERCEIVED_EVENTS)]
        return fresh, sorted(own + theirs, key=lambda r: r.get("t", 0.0))


class ReplaySource:
    """Recorded records played back at their own pace (times `speed`)."""

    def __init__(self, records: list[dict], speed: float = 1.0, clock: Callable[[], float] | None = None):
        self.records = sorted(records, key=lambda r: r.get("t", 0.0))
        self.speed, self.paused = speed, False
        self._clock = clock or time.monotonic
        self.t0 = self.records[0].get("t", 0.0) if self.records else 0.0
        self.pos = 0.0          # seconds into the recording
        self._last = self._clock()
        self._next = 0

    def poll(self) -> tuple[bool, list[dict]]:
        now = self._clock()
        if not self.paused:
            self.pos += (now - self._last) * self.speed
        self._last = now
        out = []
        while self._next < len(self.records) and self.records[self._next].get("t", 0.0) - self.t0 <= self.pos:
            out.append(self.records[self._next])
            self._next += 1
        return False, out

    def seek(self, seconds: float) -> list[dict]:
        """Jump `seconds` (back or on); returns every record up to the new position (to rebuild the state)."""
        self.pos = max(0.0, self.pos + seconds)
        self._next = sum(1 for r in self.records if r.get("t", 0.0) - self.t0 <= self.pos)
        return self.records[:self._next]


def load(path: Path, map_id: int | None = None) -> list[dict]:
    """A recorded run's records: a telemetry file (.jsonl), or an older sweep track (JSON with "track":
    [[t, x, y, platform], ...], her positions only)."""
    if path.suffix == ".jsonl":
        return telemetry.read(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    track = data.get("track", data) if isinstance(data, dict) else data
    out = [{"topic": "session", "t": track[0][0] if track else 0.0, "map": map_id, "mode": "replay"}]
    for t, x, y, platform in track:
        out.append({"topic": "scene", "t": t, "her": {"x": x, "y": y, "platform": platform}})
    return out


def planned(model, src: int, x: float, dst: int) -> list[dict]:
    """The route from (platform src, x) to platform dst and its first chain, as the navigator would plan
    them (no game needed): records for a still picture."""
    from ..motion.flow import Body, Flow, chain_record
    from ..motion.route import Planner, route_record
    now = time.time()
    path = Planner(model).plan(src, x, dst) or []
    out = [{"topic": "session", "t": now, "map": model.id, "mode": "plan"},
           {"topic": "scene", "t": now, "her": {"x": x, "y": model.platform(src).y_at(x), "platform": src}},
           {"topic": "route", "t": now, **route_record(dst, path)}]
    if path:
        names = {k: k for k in ("left", "right", "jump", "up", "down")}  # no keys are sent: any names do
        legs, _, why = Flow(model, names).chain(path, Body(0.0, x, model.platform(src)))
        if legs:
            out.append({"topic": "chain", "t": now, **chain_record(legs, why)})
    out.append({"topic": "log", "t": now, "text": f"route {src} -> {dst}: " + (
        ", ".join(f"{m.kind}->{m.dst}" for m in path) if path else "none")})
    return out
