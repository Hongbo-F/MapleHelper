"""Her pet kept fed: its fullness read off her Character Info, and Pet Food given while it's low.

Her pet drinks her potions for her (since 2026-10-23), and a hungry pet stops: on 2026-10-24 it was found at fullness
9 of 100 after a night and a morning summoned, fed only every 10 minutes while she hunted and never while she only
watched. Double-clicking her opens her Character Info, whose pet panel says its fullness; the stack's OCR
(maplehelper.ocr) reads it (FULLNESS 9, 39, 99 at 0.94-1.00 on the three saved panels, 0.25 s a read). Each Pet Food
gave 30 (9 -> 39 -> 69 -> 99) and a point of closeness.

A check (PetCare.check): a look for her name tag, a double-click on her body above it, the panel read, Pet Food one at
a time (each read again) until it's FEED_TO or MAX_FEEDS were given, then Esc to close the window, and it's read
again to be sure it closed. Esc is pressed only with the window seen open: with nothing open, Esc opens the game menu
(Change Channel, Logout, Exit Game). If the panel can't be read (her tag hidden, someone in the way), Pet Food is
given anyway once FALLBACK_S have passed since the last feed, so a pet is never left to starve for a reading.
When it was last checked and fed, and its fullness then, are kept in paths.PET.

Keys: pet food (config.toml [keys] pet_food), Esc, and two left clicks on her. Nothing else.
"""

import json
import time
from collections.abc import Callable

import numpy as np

from .. import paths, telemetry

FEED_BELOW = 70        # fullness under which she feeds it ...
FEED_TO = 85           # ... and fed until at least this (a food gives 30: from 60 it's 90)
MAX_FEEDS = 3          # Pet Food at most in one check (9 -> 99 took three)
CHECK_EVERY_S = 20 * 60  # s between checks; the pet lost about 90 fullness over a night and a morning summoned
FALLBACK_S = 2 * 60 * 60  # s since the last feed after which a check that can't read the panel feeds it anyway
BODY_ABOVE_TAG = 42    # game px from her name tag's middle up to her body's middle (tag at y 540, body 495-500 at 1366x768)
CLICK_GAP_MS = 90      # between the two clicks of a double-click
SETTLE_S = 0.6         # the window's (and a food's number's) time to show
LABEL = "FULLNESS"


def read_fullness(full: np.ndarray, ocr) -> int | None:
    """Her pet's fullness on her Character Info's pet panel in a capture of the game window (RGB), or None if the
    panel isn't there. `ocr` is maplehelper.ocr's engine (read(image BGR) -> rows); the middle of the screen is read,
    where the window opens."""
    h, w = full.shape[:2]
    area = full[int(h * 0.15):int(h * 0.92), int(w * 0.2):int(w * 0.8)]
    rows = ocr.read(np.ascontiguousarray(area[..., ::-1])) or []
    labels = [r for r in rows if r.text.strip().upper().replace(" ", "") == LABEL]
    for label in labels:
        values = [r for r in rows if abs(r.y - label.y) <= max(6.0, label.height / 2) and r.left > label.right
                  and r.text.strip().isdigit()]
        if values:
            value = min(values, key=lambda r: r.left - label.right)  # the field right of the label
            n = int(value.text.strip())
            if 0 <= n <= 100:
                return n
    return None


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

    def check(self) -> int | None:
        """Open her Character Info, read the pet's fullness, feed it while low, close the window. Returns the
        fullness it was left at (None if it couldn't be read: then fed once if FALLBACK_S passed since a feed)."""
        if self.actions.keys.pet_food is None:
            return None
        state = _state()
        state["checked"] = self.clock()
        fullness = self._open()
        if fullness is None:
            if self.clock() - state.get("fed", 0.0) >= FALLBACK_S:
                self._feed()
                state["fed"] = self.clock()
                self.log(f"pet: its fullness couldn't be read; fed it anyway ({FALLBACK_S // 3600} h since a feed)")
                telemetry.emit("event", kind="pet", text="fed her pet without reading its fullness")
            else:
                self.log("pet: its fullness couldn't be read")
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
        _save(state)
        self.log(f"pet: fullness {start}" + (f", fed {feeds} -> {fullness}" if feeds else ""))
        telemetry.emit("event", kind="pet", text=f"pet fullness {start}" + (f", fed {feeds} -> {fullness}" if feeds else ""))
        return fullness

    def _feed(self) -> None:
        self.actions.feed_pet()

    def _open(self) -> int | None:
        """Double-click her body (above her name tag in a look) and read the panel; None if it isn't there."""
        s = self.eyes.look()
        if s.tag is None:
            return None
        geo = s.geo
        x = int(round(geo.look_scale * s.tag[0]))
        y = int(round(geo.look_scale * s.tag[1] - BODY_ABOVE_TAG * geo.stretch))
        self.actions.play([[0, "mouse", x, y], [0, "lbutton", "down"], [40, "lbutton", "up"],
                           [CLICK_GAP_MS, "lbutton", "down"], [CLICK_GAP_MS + 40, "lbutton", "up"]], "pet_info")
        time.sleep(SETTLE_S)
        return read_fullness(self.eyes._capture(), self.ocr)

    def _close(self) -> None:
        """Esc, the window being open; once more if it's still there (never with it gone: Esc would open the menu)."""
        for _ in range(2):
            self.actions.play([[0, "esc", "down"], [40, "esc", "up"]], "pet_close")
            time.sleep(SETTLE_S)
            if read_fullness(self.eyes._capture(), self.ocr) is None:
                return
        self.log("pet: her Character Info didn't close with Esc")
