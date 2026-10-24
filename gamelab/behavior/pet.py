"""Her pet kept fed: its fullness read once off her Character Info, then followed in time, and Pet Food given when
it's low.

Her pet drinks her potions for her (since 2026-10-23), and a hungry pet stops: on 2026-10-24 it was found at fullness
9 of 100 after a night and a morning summoned, fed only every 10 minutes while she hunted and never while she only
watched. Double-clicking her opens her Character Info, whose pet panel says its fullness; the stack's OCR
(maplehelper.ocr) reads it (FULLNESS 9, 39, 99 at 0.94-1.00 on the three saved panels, 0.25 s a read). Each Pet Food
gave 30 (9 -> 39 -> 69 -> 99) and a point of closeness.

A check (PetCare.check): a look for her name tag, a double-click on her body above it, the panel read, Pet Food one at
a time (each read again) until it's FEED_TO or MAX_FEEDS were given, then Esc to close the window, and it's read
again to be sure it closed. Esc is pressed only with a Character Info window seen open: with nothing open, Esc opens
the game menu (Change Channel, Logout, Exit Game). The window must be hers: another player standing on her took the
double-click on 2026-10-24 and opened theirs (a level 20 magician's). Hers is the one with her name under its portrait
(the name on her bottom bar, read the same way: "BugBaby"); any other is closed and the check tried again RETRY_S
later. Opened by a double-click on her, her window shows its pet panel collapsed ("SHOW PET INFO", read by the OCR as
"SHO U PET IHFO"): that button is clicked first. If the panel can't be read, Pet Food is given anyway once FALLBACK_S
have passed since the last feed, so a pet is never left to starve for a reading.

Read once, then followed (PetCare.tend): the game data gives each pet how much fullness it loses at a time (Item.wz
Pet/<id>.img info/hungry: 1 for both Snail pets, 5000054 and 5000033; 1-5 across the game's pets). How often isn't
in the client, and no Classic source says (MeowDB: "We do not know how often the hunger tick runs in Classic World
yet"; mapleclassic.wiki and planet-helper likewise). Two independent v95 server emulators, one ported from the
official routine (Rebirth95 CPet.cs:75-109, "CPet::Update"; Kinoko Pet.java:100-115), take 1 a time after 60 s plus a
random 0 to 36 - 6 x hungry s: for hungry 1, 60-90 s, 75 on average. That fits the three readings of 2026-10-24
(99 at 10:04:47, 85 at 10:21:14, 82 at 10:25:46: a fixed period fits only between 70.0 and 75.9 s, best 74.1; 60 s,
the OdinMS v55 timer remembered at first, and the v83 emulators' 180 s are ruled out). Where it goes home is
disputed: at 0 back at 5 with a closeness lost (the ported code, Nexon support), or at 5 or less back at 15 (the
OdinMS family, MeowDB "Unconfirmed"); PACKED_AT takes the higher. Notes and sources: D:/MapleHelper-scratch/pet.
No key opens the Character Info (the key bindings' unbound functions are MUTE and KEYS1-3; the equipment window shows
the pet's equipment, not its fullness), so the double-click is used once, to start from, or when asked (`pet --read`).
The fullness followed, when it was, and the last feed are kept in paths.PET. Time counts only while the game window is
there (the pet is out only with her in the game): `pet --every` looks for it every minute.

Keys: pet food (config.toml [keys] pet_food), Esc, two left clicks on her and one on her window's SHOW PET INFO.
Nothing else.
"""

import json
import time
from collections.abc import Callable

import numpy as np

from .. import paths, telemetry

HUNGRY = 1             # fullness the pet loses a tick (the game data's info/hungry of the Snail pets)
TICK_S = 75            # s a point on average: 60 + random 0..30 for hungry 1 (above); the readings' best fixed fit 74.1
FOOD = 30              # fullness a Pet Food gives (9 -> 39 -> 69 -> 99)
PACKED_AT = 5          # at or under it the pet may be sent home (5 by the OdinMS family and MeowDB; 0 by the ported code)
FOLLOW_FEED_BELOW = 60  # followed fullness under which she gives it a food: well above PACKED_AT, for a random tick
                        # (an hour is 48 points, give or take 1) and a period a few s off (74 s for 75: 1 an hour)
FEED_BELOW = 70        # read fullness under which she feeds it ...
FEED_TO = 80           # ... and fed until at least this (a food gives 30: 50 -> 80, 40 -> 70 -> 100)
MAX_FEEDS = 3          # Pet Food at most in one check (9 -> 99 took three)
CHECK_EVERY_S = 20 * 60  # s between checks; the pet lost about 90 fullness over a night and a morning summoned
FALLBACK_S = 2 * 60 * 60  # s since the last feed after which a check that can't read the panel feeds it anyway
BODY_ABOVE_TAG = 42    # game px from her name tag's middle up to her body's middle (tag at y 540, body 495-500 at 1366x768)
CLICK_GAP_MS = 90      # between the two clicks of a double-click
SETTLE_S = 0.6         # the window's (and a food's number's) time to show
RETRY_S = 90           # s after a check that found someone else's window, or none, before the next
LABEL = "FULLNESS"
TITLE_WORDS = ("CHARACTERINFO", "LEVEL", "JOB", "FAME")  # what shows a Character Info window is open (any one)
# Her name on the bottom bar, under her job, in game px at 1366x768 (x0, y0, x1, y1): "Rogue" above "BugBaby".
NAME_BOX = (355, 735, 500, 765)


def _field(rows, label: str) -> int | None:
    """The number in the field right of a row reading `label` (spaces and case aside), or None."""
    for lab in (r for r in rows if r.text.strip().upper().replace(" ", "") == label):
        values = [r for r in rows if abs(r.y - lab.y) <= max(6.0, lab.height / 2) and r.left > lab.right
                  and r.text.strip().isdigit()]
        if values:
            return int(min(values, key=lambda r: r.left - lab.right).text.strip())  # the nearest to its right
    return None


def _norm(text: str) -> str:
    return text.strip().upper().replace(" ", "")


def read_panel(full: np.ndarray, ocr) -> dict:
    """What a capture of the game window (RGB) shows of a Character Info window: {"open": a Character Info window is
    there, "fullness": its pet panel's FULLNESS (None without the panel), "level": its LEVEL, "names": the words in its
    left column (the name under its portrait among them), "show_pet": where its SHOW PET INFO button is (client px),
    if it shows one}. `ocr` is maplehelper.ocr's engine (read(image BGR) -> rows); the middle of the screen is read,
    where the window opens."""
    h, w = full.shape[:2]
    ox, oy = int(w * 0.2), int(h * 0.15)
    area = full[oy:int(h * 0.92), ox:int(w * 0.8)]
    rows = ocr.read(np.ascontiguousarray(area[..., ::-1])) or []
    words = {_norm(r.text) for r in rows}
    fullness = _field(rows, LABEL)
    level_label = next((r for r in rows if _norm(r.text) == "LEVEL"), None)
    title = next((r for r in rows if _norm(r.text) == "CHARACTERINFO"), None)
    names = []
    if level_label is not None:  # the portrait's column: left of LEVEL, from the title down the window's height
        top = title.y if title is not None else level_label.y - 60
        names = [r.text.strip() for r in rows if r.right < level_label.left and top < r.y < top + 160]
    show = next((r for r in rows if _norm(r.text).startswith("SHO") and "PET" in _norm(r.text)), None)
    return {"open": fullness is not None or any(word in words for word in TITLE_WORDS),
            "fullness": fullness if fullness is not None and 0 <= fullness <= 100 else None,
            "level": _field(rows, "LEVEL"), "names": names,
            "show_pet": None if show is None else (int(round(ox + show.x)), int(round(oy + show.y)))}


def read_fullness(full: np.ndarray, ocr) -> int | None:
    """Her pet's fullness on a Character Info's pet panel in a capture of the game window (RGB), or None."""
    return read_panel(full, ocr)["fullness"]


def follow(state: dict, now: float, present: bool = True) -> None:
    """Bring the followed fullness (state "level" as of "at") up to `now`: HUNGRY a TICK_S while the game is there
    (`present`); away, the time doesn't count."""
    if "level" in state and "at" in state and present:
        state["level"] = max(0.0, state["level"] - max(0.0, now - state["at"]) / TICK_S * HUNGRY)
    state["at"] = now


def _state() -> dict:
    try:
        return json.loads(paths.PET.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    paths.PET.parent.mkdir(parents=True, exist_ok=True)
    paths.PET.write_text(json.dumps(state), encoding="utf-8")


class PetCare:
    """Checks and feeds her pet through `actions` (control.actions.Actions: its keys and plays) and `eyes`
    (perception.eyes.Eyes: looks and captures); `ocr` reads the panel (by default maplehelper.ocr's shared engine)."""

    def __init__(self, actions, eyes, ocr=None, log: Callable[[str], None] = print, clock=time.time):
        self.actions, self.eyes, self.log, self.clock = actions, eyes, log, clock
        self._ocr = ocr

    @property
    def ocr(self):
        if self._ocr is None:
            from maplehelper.ocr import shared
            self._ocr = shared()
        return self._ocr

    def due(self) -> bool:
        """Whether CHECK_EVERY_S have passed since the last check (never yet: now). Never without a pet food key."""
        if self.actions.keys.pet_food is None:
            return False
        return self.clock() - _state().get("checked", 0.0) >= CHECK_EVERY_S

    def tend(self, present: bool = True, read_ok: bool = True) -> float | None:
        """Follow the pet's fullness in time (TICK_S, HUNGRY) and give it a food when that's under FOLLOW_FEED_BELOW.
        `present`: the game window is there (time away from the game doesn't count). Never read yet, it's read first
        (`check`, if `read_ok`). Returns the fullness followed (None if not known yet)."""
        if self.actions.keys.pet_food is None:
            return None
        now = self.clock()
        state = _state()
        follow(state, now, present)
        if "level" not in state:
            _save(state)
            return self.check() if present and read_ok else None
        if present and state["level"] < FOLLOW_FEED_BELOW:
            self._feed()
            before = state["level"]
            state["level"], state["fed"] = min(100.0, before + FOOD), now
            self.log(f"pet: fullness about {before:.0f}, fed it -> about {state['level']:.0f}")
            telemetry.emit("event", kind="pet", text=f"fed her pet at about {before:.0f}")
        _save(state)
        return state["level"]

    def check(self) -> int | None:
        """Open her Character Info, read the pet's fullness, feed it while low, close the window. Returns the
        fullness it was left at (None if it couldn't be read: then fed once if FALLBACK_S passed since a feed)."""
        if self.actions.keys.pet_food is None:
            return None
        state = _state()
        if self.clock() < state.get("retry_at", 0.0):
            return None
        state["checked"] = self.clock()
        state.pop("retry_at", None)
        fullness = self._open()
        if fullness is None:
            if self.clock() - state.get("fed", 0.0) >= FALLBACK_S:
                self._feed()
                state["fed"] = self.clock()
                self.log(f"pet: its fullness couldn't be read; fed it anyway ({FALLBACK_S // 3600} h since a feed)")
                telemetry.emit("event", kind="pet", text="fed her pet without reading its fullness")
            else:
                self.log("pet: its fullness couldn't be read")
            state["retry_at"] = self.clock() + RETRY_S
            _save(state)
            return None
        start, feeds = fullness, 0
        while fullness < FEED_TO and (feeds > 0 or fullness < FEED_BELOW) and feeds < MAX_FEEDS:
            self._feed()
            feeds += 1
            time.sleep(SETTLE_S)
            seen = read_fullness(self.eyes._capture(), self.ocr)
            fullness = seen if seen is not None else min(100, fullness + 30)
        self._close()
        if feeds:
            state["fed"] = self.clock()
        state["fullness"] = fullness
        state["level"], state["at"], state["read_at"] = float(fullness), self.clock(), self.clock()  # followed from here
        _save(state)
        self.log(f"pet: fullness {start}" + (f", fed {feeds} -> {fullness}" if feeds else ""))
        telemetry.emit("event", kind="pet", text=f"pet fullness {start}" + (f", fed {feeds} -> {fullness}" if feeds else ""))
        return fullness

    def _feed(self) -> None:
        self.actions.feed_pet()

    def _open(self) -> int | None:
        """Double-click her body (above her name tag in a look) and read the panel: its fullness if the window is hers
        (its pet panel showing, and its level hers if her EXP number says it); else None, any other Character Info
        closed."""
        s = self.eyes.look()
        if s.tag is None:
            return None
        geo = s.geo
        level = _her_level(s)
        name = self._her_name(geo)
        x = int(round(geo.look_scale * s.tag[0]))
        y = int(round(geo.look_scale * s.tag[1] - BODY_ABOVE_TAG * geo.stretch))
        self.actions.play([[0, "mouse", x, y], [0, "lbutton", "down"], [40, "lbutton", "up"],
                           [CLICK_GAP_MS, "lbutton", "down"], [CLICK_GAP_MS + 40, "lbutton", "up"]], "pet_info")
        time.sleep(SETTLE_S)
        panel = read_panel(self.eyes._capture(), self.ocr)
        if not panel["open"]:
            return None
        if name is not None:
            mine = name in panel["names"]
        else:  # her name not read off her bar: her level, and a pet panel or its button, at least
            mine = level is not None and panel["level"] == level
        if mine and panel["fullness"] is None and panel["show_pet"] is not None:
            x, y = panel["show_pet"]
            self.actions.play([[0, "mouse", x, y], [0, "lbutton", "down"], [40, "lbutton", "up"]], "pet_panel")
            time.sleep(SETTLE_S)
            panel = read_panel(self.eyes._capture(), self.ocr)
        if mine and panel["fullness"] is not None:
            return panel["fullness"]
        self.log(f"pet: the window opened isn't hers ({', '.join(panel['names']) or 'no name read'}, level "
                 f"{panel['level']})" if not mine else "pet: her window showed no pet panel")
        self._close()
        return None

    def _her_name(self, geo) -> str | None:
        """Her name off her bottom bar (NAME_BOX), read once and kept."""
        if getattr(self, "_name", None) is None:
            full = self.eyes._capture()
            x0, y0, x1, y1 = (int(round(v * geo.stretch)) for v in NAME_BOX)
            rows = self.ocr.read(np.ascontiguousarray(full[y0:y1, x0:x1, ::-1])) or []
            texts = [r.text.strip() for r in sorted(rows, key=lambda r: r.y)]
            self._name = texts[-1] if len(texts) >= 2 else None  # the lower line: the upper is her job
        return self._name

    def _close(self) -> None:
        """Esc, a Character Info window being open; once more if one is still there (never with none: Esc would open
        the game menu)."""
        for _ in range(2):
            self.actions.play([[0, "esc", "down"], [40, "esc", "up"]], "pet_close")
            time.sleep(SETTLE_S)
            if not read_panel(self.eyes._capture(), self.ocr)["open"]:
                return
        self.log("pet: the Character Info window didn't close with Esc")


def _her_level(s) -> int | None:
    """Her level as her EXP number says it in look `s` (perception.exp_number), or None."""
    box = getattr(s, "exp_box", None)
    if box is None:
        return None
    from ..perception import exp_number
    try:
        return exp_number.read(box, s.geo).level
    except Exception:
        return None
