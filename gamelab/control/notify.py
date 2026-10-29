"""The Lie Detector pushed to the user's phone through ntfy (https://ntfy.sh), by the player the moment it sees one (the
user, 2026-10-29 ~01:40: "I need a better notification method, I didn't catch the notification in time to solve this";
asked which, "ntfy push to phone"). The PC's beeps and toast (control.alarm.page) stay; a phone push from whoever drove
gamelab came late, after a monitor saw the player's log.

What goes out (Pusher, kept by the player's Watchdog): at a cycle's first sight, an urgent push ("Lie Detector up",
its form); while a form she can answer (all but the Truth Booth's) is still seen, the same again every REPEAT_S; the
Truth Booth's timer, once ("missed: in the Truth Booth"); the cycle's end, once, at the default priority. The text is
only that: the form, the time and how long it has been up. Never a screenshot, her name or the map.

Where to (lab/ntfy.json, git-ignored like the rest of the lab: {"server": "https://ntfy.sh", "topic": "..."}): the
topic is the only secret (anyone who knows it can read and post), so `python -m gamelab notify --setup` makes a long
random one, and `notify --test` sends a push to check the phone gets it. No file, no push (the tests, a checkout
without it); the file is read at each push, so the player started before it was made pushes once it's there. Each push is sent on a thread of its own, SEND_TIMEOUT_S at most, and never raises: the player's looks
and locks never wait for the network."""
from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.request
from pathlib import Path

from ..paths import LAB

CONFIG = LAB / "ntfy.json"
SERVER = "https://ntfy.sh"
REPEAT_S = 30.0          # s between pushes while a form she can answer is still seen
SEND_TIMEOUT_S = 10.0    # s a push may take
MISSED = "truth booth"   # the Lie Detector's form that means it went unanswered (perception.lie_detector.FORMS)


def load(path: Path = CONFIG) -> dict | None:
    """The push's server and topic ({"server", "topic"}), or None (no file, or no topic in it)."""
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not cfg.get("topic"):
        return None
    return {"server": str(cfg.get("server") or SERVER).rstrip("/"), "topic": str(cfg["topic"])}


def setup(path: Path = CONFIG) -> dict:
    """lab/ntfy.json with a new random topic if there's none yet; returns what's in it."""
    cfg = load(path)
    if cfg is None:
        cfg = {"server": SERVER, "topic": "gamelab-" + secrets.token_urlsafe(18).replace("_", "x").replace("-", "y")}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def send(cfg: dict, title: str, text: str, priority: str = "urgent", tags: str = "rotating_light") -> None:
    """One push, now, on this thread (raises on failure: `push` is the one that never does)."""
    req = urllib.request.Request(f"{cfg['server']}/{cfg['topic']}", data=text.encode("utf-8"), method="POST",
                                 headers={"Title": title, "Priority": priority, "Tags": tags})
    with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S) as r:
        r.read()


def push(cfg: dict | None, title: str, text: str, priority: str = "urgent", log=None) -> threading.Thread | None:
    """`send` on a thread of its own, its failure said through `log` (if given), never raised; None without `cfg`."""
    if cfg is None:
        return None

    def go():
        try:
            send(cfg, title, text, priority)
        except Exception as e:  # (no network, the server down: the PC's beeps and toast still go)
            if log is not None:
                log(f"the phone push failed: {e}")
    t = threading.Thread(target=go, name="ntfy-push", daemon=True)
    t.start()
    return t


class Pusher:
    """A Lie Detector cycle's pushes (see the top), as the Watchdog's looks tell it: `seen(now, form)` at each look that
    sees one (the cycle's first included), `ended(now)` at its end. `push` stands in for the network in tests:
    push(title, text, priority)."""

    def __init__(self, push_fn=None, every_s: float = REPEAT_S, clock=time.strftime):
        self.push = push_fn
        self.every_s = every_s
        self.clock = clock
        self.t0: float | None = None      # the cycle's first sight (the Watchdog's monotonic s), None between cycles
        self.last: float | None = None    # the last push of the cycle
        self.missed = False               # the Truth Booth's push gone

    @classmethod
    def from_lab(cls, log=None, path: Path = CONFIG) -> "Pusher":
        """Pushes to lab/ntfy.json's topic, the file read at each push: none while there's none, and one set up or
        changed later is used without the player starting again."""
        return cls(lambda title, text, priority: push(load(path), title, text, priority, log=log))

    def seen(self, now: float, form: str | None) -> None:
        if self.t0 is None:
            self.t0, self.last, self.missed = now, now, form == MISSED
            self.push("Lie Detector up" if form != MISSED else "Lie Detector missed",
                      f"{form or 'the Lie Detector'} seen at {self.clock('%H:%M:%S')}: "
                      + ("answer it in the game now" if form != MISSED else "she's in the Truth Booth"), "urgent")
            return
        if form == MISSED:
            if not self.missed:
                self.missed, self.last = True, now
                self.push("Lie Detector missed", f"the Truth Booth at {self.clock('%H:%M:%S')}, "
                                                 f"{now - self.t0:.0f} s after it was first seen", "high")
            return
        if now - self.last >= self.every_s:
            self.last = now
            self.push("Lie Detector still up", f"{form or 'the Lie Detector'}, up {now - self.t0:.0f} s: answer it in "
                                               "the game now", "urgent")

    def ended(self, now: float) -> None:
        if self.t0 is None:
            return
        self.push("Lie Detector gone", f"off the screen at {self.clock('%H:%M:%S')} after {now - self.t0:.0f} s",
                  "default")
        self.t0 = self.last = None
        self.missed = False
