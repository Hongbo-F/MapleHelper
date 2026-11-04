"""The Lie Detector sent to the user's phone by the player the moment it sees one: a critical alert to their iPhone
through Bark, and a text mailed through their own account. The PC's beeps and toast (control.alarm.page) stay; a phone
push from whoever drove gamelab came late, after a monitor saw the player's log (the user, 2026-10-29 ~01:40: "I need
a better notification method, I didn't catch the notification in time to solve this").

What goes out (Pusher, kept by the player's Watchdog): at a cycle's first sight, an alert ("Lie Detector up", its
form); the Truth Booth's timer, once ("missed: in the Truth Booth"); the cycle's end, once, an ordinary one. It isn't
repeated while it's up (every 30 s until the user, 2026-11-04: "let's also remove the 'still up' reminder"). The text
is only that: the form, the time and how long it has been up. Never a screenshot, her name or the map.

The iPhone, through Bark (the user, 2026-10-29: "switch to something that can actually send urgent alert on iPhone";
asked which, "Bark"; then "Let's remove ntfy notification all along"): ntfy's iOS app, used before, can't break through
Focus or the mute switch (no Critical Alerts entitlement); Bark's `level=critical` is meant to
(https://github.com/Finb/Bark: "Critical alerts will ignore silent and do not disturb modes"). By lab/bark.json
({"server": "https://api.day.app", "key": "<the device key>"}, git-ignored with the lab: the key is the only secret;
`notify --setup-bark KEY`, `--test-bark`): the first sight and the Truth Booth critical, each its alarm sound
once (a few seconds; Bark's `call`, its sound again and again for 30 s, was too much: the user, "just 5s is enough"),
the cycle's end an ordinary one.

A text message too (the user, 2026-10-29 ~02:00: "Send lie detector alert to <their carrier's email-to-text address>
also"; asked how, "My email via SMTP"): the cycle's first sight and the Truth Booth's, once each, mailed through the user's own account to the addresses in lab/smtp.json (git-ignored with the
lab: {"host", "port", "starttls", "user", "password", "from", "to": [...]}), which `notify --setup-sms ADDRESS` makes
with the login left for the user to fill in themselves; `notify --test-sms` sends one. Short (a carrier's gateway
keeps about 160 characters).

No file, nothing sent (a checkout without them). Each file is read at each push, so one set up while a player runs is
used without a start again. Under NO_PUSH_ENV (the tests' conftest) the lab's own files are never read for one: the
suite run in the main checkout on 2026-10-29 ~05:35, the game closed, texted the user a Lie Detector from a test's fake
screen through the real lab files. Each push goes on a thread of its own, its timeout at most, and never raises: the
player's looks and locks never wait for the network."""
from __future__ import annotations

import json
import os
import smtplib
import threading
import time
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path

from ..paths import LAB

MAIL_CONFIG = LAB / "smtp.json"
BARK_CONFIG = LAB / "bark.json"
BARK_SERVER = "https://api.day.app"
BARK_VOLUME = 10         # a critical alert's volume, Bark's 0-10
BARK_SOUND = "alarm"     # one of Bark's sounds
MAIL_TIMEOUT_S = 20.0    # s a text's mail may take
SEND_TIMEOUT_S = 10.0    # s a push may take
MISSED = "truth booth"   # the Lie Detector's form that means it went unanswered (perception.lie_detector.FORMS)
NO_PUSH_ENV = "GAMELAB_NO_PUSH"  # set (tests/conftest.py): no push or text from the lab's own files, ever


def _lab_file(path: Path | None, default: Path) -> Path | None:
    """`path` if given, else the lab's file looked up now (BARK_CONFIG, MAIL_CONFIG as they are at the push), or None
    under NO_PUSH_ENV."""
    if path is not None:
        return Path(path)
    return None if os.environ.get(NO_PUSH_ENV) else default


def load_mail(path: Path | None = None) -> dict | None:
    """The text's mail account and addresses, or None (no file, or its host, login or addresses missing)."""
    path = _lab_file(path, MAIL_CONFIG)
    if path is None:
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    to = cfg.get("to") or []
    to = [to] if isinstance(to, str) else list(to)
    if not (cfg.get("host") and cfg.get("user") and cfg.get("password") and to):
        return None
    return {"host": str(cfg["host"]), "port": int(cfg.get("port") or 587), "starttls": bool(cfg.get("starttls", True)),
            "user": str(cfg["user"]), "password": str(cfg["password"]), "from": str(cfg.get("from") or cfg["user"]),
            "to": [str(a) for a in to]}


def setup_mail(to: str, path: Path | None = None) -> tuple[dict, bool]:
    """lab/smtp.json with `to` among its addresses, the login left as it is (empty in a new one, for the user to fill
    in: Gmail's by default, with an app password); returns (what's in it, whether it can send yet)."""
    path = Path(path) if path is not None else MAIL_CONFIG
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {"host": "smtp.gmail.com", "port": 587, "starttls": True, "user": "", "password": "", "from": "", "to": []}
    to_ = cfg.get("to") or []
    to_ = [to_] if isinstance(to_, str) else list(to_)
    if to not in to_:
        to_.append(to)
    cfg["to"] = to_
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg, load_mail(path) is not None


def send_mail(cfg: dict, subject: str, text: str) -> None:
    """One text's mail, now, on this thread (raises on failure: `mail` is the one that never does)."""
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = cfg["from"], ", ".join(cfg["to"]), subject
    msg.set_content(text)
    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=MAIL_TIMEOUT_S) as s:
        if cfg["starttls"]:
            s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)


def mail(cfg: dict | None, subject: str, text: str, log=None) -> threading.Thread | None:
    """`send_mail` on a thread of its own, its failure said through `log` (if given), never raised; None without `cfg`."""
    if cfg is None:
        return None

    def go():
        try:
            send_mail(cfg, subject, text)
        except Exception as e:  # (no network, a login refused: the alert and the PC's beeps still go)
            if log is not None:
                log(f"the text's mail failed: {type(e).__name__}: {e}")
    t = threading.Thread(target=go, name="text-mail", daemon=True)
    t.start()
    return t


def load_bark(path: Path | None = None) -> dict | None:
    """Bark's server, the device key and whether the first sight and the Truth Booth ring 30 s (`ring`, off unless the
    file says "ring": true) ({"server", "key", "ring"}), or None (no file, or no key in it)."""
    path = _lab_file(path, BARK_CONFIG)
    if path is None:
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not cfg.get("key"):
        return None
    return {"server": str(cfg.get("server") or BARK_SERVER).rstrip("/"), "key": str(cfg["key"]).strip(),
            "ring": bool(cfg.get("ring", False))}


def setup_bark(key: str, path: Path | None = None) -> dict:
    """lab/bark.json with this device key (Bark's own server unless the file names another); returns what's in it."""
    path = Path(path) if path is not None else BARK_CONFIG
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except (OSError, ValueError):
        cfg = {}
    cfg = {"server": cfg.get("server") or BARK_SERVER, "key": key.strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def bark_url(cfg: dict, title: str, text: str, level: str = "critical", call: bool = False,
             sound: str | None = BARK_SOUND) -> str:
    """The Bark push's URL (its GET form, /:key/:title/:body?...): `level` critical at BARK_VOLUME, `call` its sound
    again and again for 30 s, `sound` one of Bark's (None: the phone's own); grouped as gamelab's."""
    q = {"level": level, "group": "gamelab"}
    if sound is not None:
        q["sound"] = sound
    if level == "critical":
        q["volume"] = BARK_VOLUME
    if call:
        q["call"] = 1
    part = lambda s: urllib.parse.quote(s, safe="")
    return f"{cfg['server']}/{part(cfg['key'])}/{part(title)}/{part(text)}?{urllib.parse.urlencode(q)}"


def send_bark(cfg: dict, title: str, text: str, level: str = "critical", call: bool = False,
              sound: str | None = BARK_SOUND) -> None:
    """One Bark push, now, on this thread (raises on failure, or if Bark's answer isn't its code 200)."""
    with urllib.request.urlopen(urllib.request.Request(bark_url(cfg, title, text, level, call, sound), method="GET"),
                                timeout=SEND_TIMEOUT_S) as r:
        body = r.read()
    try:
        code = json.loads(body.decode("utf-8")).get("code")
    except (ValueError, AttributeError):
        code = None
    if code not in (None, 200):
        raise RuntimeError(f"Bark answered {body[:200]!r}")


def bark(cfg: dict | None, title: str, text: str, level: str = "critical", call: bool = False,
         log=None, sound: str | None = BARK_SOUND) -> threading.Thread | None:
    """`send_bark` on a thread of its own, its failure said through `log` (if given), never raised; None without
    `cfg`."""
    if cfg is None:
        return None

    def go():
        try:
            send_bark(cfg, title, text, level, call, **({} if sound == BARK_SOUND else {"sound": sound}))
        except Exception as e:  # (no network, a key Bark doesn't know: the text and the PC's beeps still go)
            if log is not None:  # (an exception may quote the URL: never the key)
                log(f"the Bark push failed: {type(e).__name__}: {str(e).replace(cfg['key'], '<key>')}")
    t = threading.Thread(target=go, name="bark-push", daemon=True)
    t.start()
    return t


class BackpackPush:
    """Her backpack full, on the phone as a regular push (the user, 2026-11-04: "if backpack is full, send a regular push
    notification"): Bark at level 'active', the phone's own sound, no ring and no text, the moment the game says a tab of hers
    is full (`full`: behavior.looting's, perception.loot.says_full), once a tab, and not again until it has been clear in
    between (`clear`: an item that didn't fit earned into it, or a kind she held none of picked up since: looting's evidence;
    its TAB_RETRY_S, a guess that it may have room, isn't). Never in the way of the hunt (`bark`: a thread of its own, its
    timeout at most, never raises), nothing under NO_PUSH_ENV or without lab/bark.json; the simulator's a stub (`push_fn`:
    sim.scenarios.Sim.hunter). The key is never logged."""

    def __init__(self, push_fn=None, clock=time.strftime):
        self.push = push_fn
        self.clock = clock
        self.sent: set[str] = set()       # the tabs told of, not seen clear since

    @classmethod
    def from_lab(cls, log=None, bark_path: Path | None = None) -> "BackpackPush":
        """Pushes to lab/bark.json's iPhone, the file read at each push (none while there's none)."""
        def go(title, text):
            bark(load_bark(bark_path), title, text, "active", log=log, sound=None)
        return cls(go)

    def full(self, tab: str) -> bool:
        """The game says tab `tab` is full: the push, if it hasn't gone since the tab was last clear; whether it went."""
        if tab in self.sent:
            return False
        self.sent.add(tab)
        if self.push is not None:
            self.push("Backpack full", f"her {tab} tab at {self.clock('%H:%M:%S')}: its items left, mesos and what still "
                                       f"fits taken; the hunt goes on")
        return True

    def clear(self, tab: str) -> None:
        """Tab `tab` has room again, as the game shows: the next time it's full is told again."""
        self.sent.discard(tab)


class Pusher:
    """A Lie Detector cycle's pushes (see the top), as the Watchdog's looks tell it: `seen(now, form)` at each look that
    sees one (the cycle's first included), `ended(now)` at its end. `push` stands in for the network in tests:
    push(title, text, priority, text_too): `priority` urgent, high or default; `text_too` the cycle's first sight and
    the Truth Booth's, a text as well."""

    def __init__(self, push_fn=None, clock=time.strftime):
        self.push = push_fn
        self.clock = clock
        self.t0: float | None = None      # the cycle's first sight (the Watchdog's monotonic s), None between cycles
        self.missed = False               # the Truth Booth's push gone

    @classmethod
    def from_lab(cls, log=None, mail_path: Path | None = None, bark_path: Path | None = None) -> "Pusher":
        """Pushes to lab/bark.json's iPhone (urgent and high critical: its sound once) and, `text_too`, a text
        through lab/smtp.json's account, each file read at each push: none while there's none (nor ever under
        NO_PUSH_ENV unless named), and one set up or changed later is used without the player starting again."""
        def go(title, text, priority, text_too=False):
            bcfg = load_bark(bark_path)
            bark(bcfg, title, text, "critical" if priority in ("urgent", "high") else "active",
                 call=bool(text_too and bcfg and bcfg["ring"]), log=log)
            if text_too:
                mail(load_mail(mail_path), title, text, log=log)
        return cls(go)

    def seen(self, now: float, form: str | None, cause: str | None = None) -> None:
        """`cause`: the watchdog's lie_recorder.UNKNOWN_CAUSE for a cycle first seen as the Truth Booth with no test
        seen before it, said in its first push."""
        if self.t0 is None:
            self.t0, self.missed = now, form == MISSED
            booth = "she's in the Truth Booth" + (" with no test seen before it: its cause unknown (a check missed?), "
                                                  "the 5 minutes before it saved" if cause else "")
            self.push("Lie Detector up" if form != MISSED else "Lie Detector missed",
                      f"{form or 'the Lie Detector'} seen at {self.clock('%H:%M:%S')}: "
                      + ("answer it in the game now" if form != MISSED else booth), "urgent", True)
            return
        if form == MISSED and not self.missed:
            self.missed = True
            self.push("Lie Detector missed", f"the Truth Booth at {self.clock('%H:%M:%S')}, "
                                             f"{now - self.t0:.0f} s after it was first seen", "high", True)

    def ended(self, now: float) -> None:
        if self.t0 is None:
            return
        self.push("Lie Detector gone", f"off the screen at {self.clock('%H:%M:%S')} after {now - self.t0:.0f} s",
                  "default", False)
        self.t0 = None
        self.missed = False
