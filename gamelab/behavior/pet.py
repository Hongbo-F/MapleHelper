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

With the Lie Detector alarm up (since 2026-10-24: the user, after one kept the pet from being fed, "don't block pet
feeding"), the food goes on by the fullness followed: its plan (Actions.feed_pet: kind alarm.PET_FOOD, a tap of the
key) is the one the player takes then, a few a minute. Nothing is read until the alarm is cleared: the read
double-clicks her, and whoever is at the PC may be answering the Lie Detector with the mouse. Never read yet, she gets
a food once FALLBACK_S have passed since the last, as when the panel can't be read.

Keys: pet food (config.toml [keys] pet_food, or where a read of the game's KEY BINDINGS window followed it:
control.keymap), Esc, two left clicks on her and one on her window's SHOW PET INFO; and, while the game's client is
unfit, at most one Alt+Enter-twice try every TRY_EVERY_S (PetWait). Nothing else.

Her pet food doubted (since the review of 2026-10-28: a remembered pet food key the game no longer had fed nothing, and
the fullness followed rose by 30 a food all the same). At a check, FOOD_DOUBT taps in a row after which the panel's
fullness, read after each, didn't rise are the doubt (out of Pet Food, or the key another in the game): no more taps
then, said in capitals, paged, `on_doubt` told (the hunter marks a remembered key and, in the hunt loop, reads the
game's key bindings), and that key not tapped again for FOOD_DOUBT_WAIT_S. A key not `trusted` (neither config.toml's
nor one a read of the game's window had this run: Hunter._pet_key_trusted; nor one whose food a check saw take in the
last VERIFIED_S), or one doubted before, isn't fed by blind: a food due is a check (read, fed while low, each food read
again), waited for while a read can't be made now and the fullness followed is BLIND_BELOW or more; by blind only under
it, with the alarm up, or if the check can't read.

While the game's client isn't fit to read (`pet --every`, PetWait; since 2026-10-26): another size, not calibrated at
its size, or the game drawing its picture off it (perception.framecheck.FIX_KINDS). At 23:21 on 2026-10-25 the game was
full screen at 3840x2160 for about a minute and the pet's babysitter ended at once ("not calibrated at 3840x2160 yet"),
started again every 5 s. Now it waits: says so once, sends nothing (no food either: the game may be at its login
screen, where a key types into a field), keeps the fullness followed (`tend` counts the time when it's over, as time
the game window was there), and carries on once the client is right again. It doesn't send Alt+Enter in general (it
runs unattended beside the user's own play), with one bounded exception so the pet can't starve: once the fullness
followed says it's due a food (`due_to_feed`), the client's picture shows the game in play (its HUD's buttons:
framecheck.hud_seen, on the perception service's picture at whatever size, perception.frames.glimpse), the fault has
stood window_size.STEADY_S and no Lie Detector alarm is up, the one try the hunts make (control.window_size.refit), at
most every TRY_EVERY_S (kept in paths.PET: across the babysitter's restarts; counted as its first Alt+Enter goes, so
a try cut short is one too) and within the hunts' HOURLY, said in the log; a look before it that came to nothing (the
game not in play, no picture, no Alt+Enter sent) is made again window_size.AGAIN_S on, not at every poll.
"""

import json
import time
from collections.abc import Callable

import numpy as np

from .. import paths, telemetry
from ..control import client, window_size
from ..perception import framecheck
from ..perception.framecheck import BadLook, Problem
from ..screen import GAME_VIEW

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
FOOD_DOUBT = 2         # pet food taps in a row her pet's fullness, read after each, didn't rise for: the doubt (a food
                       # gives 30, live 9 -> 39 -> 69 -> 99; one not read isn't counted)
FOOD_DOUBT_WAIT_S = 30 * 60  # s that key isn't tapped again after its doubt (a food is due every 37 min at most)
BLIND_BELOW = 20       # followed fullness under which a food due on a key not trusted goes by blind when no read can
                       # be made now (over it, a moment to read is waited for: 40 points are 50 minutes or so)
VERIFIED_S = 2 * 60 * 60  # s a key not trusted goes by blind after a check saw its food take (then a check again)
FALLBACK_S = 2 * 60 * 60  # s since the last feed after which a check that can't read the panel feeds it anyway
BODY_ABOVE_TAG = 42    # game px from her name tag's middle up to her body's middle (tag at y 540, body 495-500 at 1366x768)
CLICK_GAP_MS = 90      # between the two clicks of a double-click
SETTLE_S = 0.6         # the window's (and a food's number's) time to show
RETRY_S = 90           # s after a check that found someone else's window, or none, before the next
LABEL = "FULLNESS"
TITLE_WORDS = ("CHARACTERINFO", "LEVEL", "JOB", "FAME")  # what shows a Character Info window is open (any one)
# Her name on the bottom bar, under her job, in game px at 1366x768 (x0, y0, x1, y1): "Rogue" above "BugBaby".
NAME_BOX = (355, 735, 500, 765)
# PetWait: while the client is unfit, its window looked at this often (no capture: its size, and the perception
# service's newest state) ...
WAIT_POLL_S = 5.0
# ... and an Alt+Enter-twice try at most this often (the task of 2026-10-26: "at most once every 30 min"; a food is due
# every 37 minutes or so at the most often: FOOD points at a point a TICK_S)
TRY_EVERY_S = 30 * 60


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
        return json.loads(paths.of().pet.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    path = paths.of().pet  # (hers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


class PetCare:
    """Checks and feeds her pet through `actions` (control.actions.Actions: its keys and plays) and `eyes`
    (perception.eyes.Eyes: looks and captures); `ocr` reads the panel (by default maplehelper.ocr's shared engine)."""

    def __init__(self, actions, eyes, ocr=None, log: Callable[[str], None] = print, clock=time.time):
        self.actions, self.eyes, self.log, self.clock = actions, eyes, log, clock
        self._ocr = ocr
        self.trusted: Callable[[], bool] = lambda: True  # whether her pet food key may be fed by blind (see the top)
        self._verified: tuple[str, float] | None = None  # a key not trusted whose food took at a check, and when
        self.on_doubt: Callable[[str], None] | None = None  # told of a doubt of her pet food (its words)

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
        (`check`, if `read_ok`), but not with the Lie Detector alarm up: then a food if FALLBACK_S have passed since the
        last. Returns the fullness followed (None if not known yet). A food the player refuses raises (client's
        PlayerError; the alarm's limits: LieDetectorAlarm), the fullness followed left as it was."""
        if self.actions.keys.pet_food is None:
            return None
        now = self.clock()
        state = _state()
        follow(state, now, present)
        if "level" not in state:
            if present and client.alarm_up():
                if now - state.get("fed", 0.0) >= FALLBACK_S:
                    self._feed()
                    state["fed"] = now
                    self.log(f"pet: its fullness isn't known, and isn't read with the Lie Detector alarm up; fed it "
                             f"anyway ({FALLBACK_S // 3600} h since a feed)")
                    telemetry.emit("event", kind="pet", text="fed her pet without reading its fullness (the alarm up)")
                _save(state)
                return None
            _save(state)
            return self.check() if present and read_ok else None
        if present and state["level"] < FOLLOW_FEED_BELOW:
            if (doubted := self._doubted(state, now)) is not None:  # (no tap on a key that took no effect lately)
                if not state.get("doubt_said"):
                    self.log(f"pet: fullness about {state['level']:.0f}, not fed: {doubted}")
                    state["doubt_said"] = True
                _save(state)
                return state["level"]
            if (not self._may_go_blind(now) or "food_doubt" in state) and not client.alarm_up():  # (a check)
                if not read_ok and state["level"] >= BLIND_BELOW:
                    _save(state)  # (a moment to read waited for)
                    return state["level"]
                if read_ok:
                    _save(state)
                    if (got := self.check()) is not None:
                        return got
                    fed = state.get("fed")
                    state = _state()  # (as the check left it: it couldn't read the panel; a food by blind, as ever)
                    follow(state, now, present)
                    if "level" not in state or state.get("fed") != fed or self._doubted(state, now) is not None:
                        _save(state)
                        return state.get("level")
            self._feed()
            before = state["level"]
            state["level"], state["fed"] = min(100.0, before + FOOD), now
            self.log(f"pet: fullness about {before:.0f}, fed it -> about {state['level']:.0f}")
            telemetry.emit("event", kind="pet", text=f"fed her pet at about {before:.0f}")
        _save(state)
        return state["level"]

    def check(self) -> int | None:
        """Open her Character Info, read the pet's fullness, feed it while low, close the window. Returns the
        fullness it was left at (None if it couldn't be read: then fed once if FALLBACK_S passed since a feed). Not with
        the Lie Detector alarm up (None): the read double-clicks her."""
        if self.actions.keys.pet_food is None:
            return None
        if client.alarm_up():
            self.log("pet: its fullness isn't read with the Lie Detector alarm up (the read double-clicks her): fed by "
                     "the fullness followed until it's cleared")
            return None
        state = _state()
        if self.clock() < state.get("retry_at", 0.0):
            return None
        state["checked"] = self.clock()
        state.pop("retry_at", None)
        try:
            fullness = self._open()
        except BadLook as e:
            # A look or capture not of the game as expected (perception.framecheck): nothing is done on it (no Esc,
            # no food), and it's tried again RETRY_S later. Before, it escaped the hunt's pause and stopped the hunt.
            self.log(f"pet: its fullness couldn't be read: {e.why}")
            state["retry_at"] = self.clock() + RETRY_S
            _save(state)
            return None
        if fullness is None:
            if self.clock() - state.get("fed", 0.0) >= FALLBACK_S and self._doubted(state, self.clock()) is None:
                self._feed()
                state["fed"] = self.clock()
                self.log(f"pet: its fullness couldn't be read; fed it anyway ({FALLBACK_S // 3600} h since a feed)")
                telemetry.emit("event", kind="pet", text="fed her pet without reading its fullness")
            else:
                self.log("pet: its fullness couldn't be read")
            state["retry_at"] = self.clock() + RETRY_S
            _save(state)
            return None
        start, feeds, took, none, reads = fullness, 0, 0, 0, [fullness]
        blocked = self._doubted(state, self.clock())  # (no tap on a key that took no effect lately: read only)
        while (blocked is None and fullness < FEED_TO and (feeds > 0 or fullness < FEED_BELOW) and feeds < MAX_FEEDS
               and none < FOOD_DOUBT):
            before = fullness
            self._feed()
            feeds += 1
            time.sleep(SETTLE_S)
            seen = self._read()
            if seen is not None and seen <= before:
                none += 1
            else:
                none, took = 0, took + 1
            fullness = seen if seen is not None else min(100, fullness + 30)
            reads.append(fullness)
        self._close()
        if took:
            state["fed"] = self.clock()
        if none < FOOD_DOUBT and took:
            state.pop("food_doubt", None)  # (a food took: its key works)
            self._verified = (getattr(self.actions.keys.pet_food, "name", None), self.clock())
            state.pop("doubt_said", None)
        state["fullness"] = fullness
        state["level"], state["at"], state["read_at"] = float(fullness), self.clock(), self.clock()  # followed from here
        if none >= FOOD_DOUBT:
            self._food_doubt(state, reads)
        _save(state)
        self.log(f"pet: fullness {start}" + (f", fed {feeds} -> {fullness}" if feeds else "")
                 + (f" (not fed: {blocked})" if blocked is not None and start < FEED_BELOW else ""))
        telemetry.emit("event", kind="pet", text=f"pet fullness {start}" + (f", fed {feeds} -> {fullness}" if feeds else ""))
        if none >= FOOD_DOUBT and self.on_doubt is not None:
            self.on_doubt(state["food_doubt"]["text"])
        return fullness

    def _may_go_blind(self, now: float) -> bool:
        """Whether a food due may go by blind (the fullness followed): her pet food key `trusted`, or one a check saw a
        food of take within VERIFIED_S (the babysitter beside the user's own play: a read every food would double-click
        her every 37 minutes or so)."""
        if self.trusted():
            return True
        key = getattr(self.actions.keys.pet_food, "name", None)
        return self._verified is not None and self._verified[0] == key and now - self._verified[1] < VERIFIED_S

    def _doubted(self, state: dict, now: float) -> str | None:
        """Why her pet food key isn't tapped now (its doubt FOOD_DOUBT_WAIT_S ago at the most, on this key), or None;
        a doubt of another key (followed since) forgotten."""
        doubt = state.get("food_doubt")
        if not isinstance(doubt, dict):
            return None
        key = getattr(self.actions.keys.pet_food, "name", None)
        if doubt.get("key") != key:
            state.pop("food_doubt", None)
            state.pop("doubt_said", None)
            return None
        if now - doubt.get("at", 0.0) >= FOOD_DOUBT_WAIT_S:
            return None
        return (f"her pet food on {key.upper()} took no effect {(now - doubt['at']) / 60:.0f} min ago (tried again "
                f"{FOOD_DOUBT_WAIT_S / 60:.0f} min after it)")

    def _food_doubt(self, state: dict, reads: list) -> None:
        """FOOD_DOUBT taps in a row that didn't raise her pet's fullness (`reads`: the panel's before and after each):
        said, sent as a telemetry event 'pet_doubt', paged (once in FOOD_DOUBT_WAIT_S a key at the most: no tap on it
        meanwhile); that key not tapped again for FOOD_DOUBT_WAIT_S. `on_doubt` is told once the window is closed."""
        key = getattr(self.actions.keys.pet_food, "name", "?")
        text = (f"HER PET FOOD TOOK NO EFFECT {FOOD_DOUBT} TIMES IN A ROW (its key {key.upper()}; her pet's fullness "
                f"{' -> '.join(str(r) for r in reads)}, read after each): out of Pet Food, or her pet food key another "
                "in the game's Key Bindings?")
        state["food_doubt"] = {"key": key, "at": self.clock(), "fullness": reads[-1], "text": text}
        state.pop("doubt_said", None)
        self.log(f"pet: {text}")
        telemetry.emit("event", kind="pet_doubt", text=text, key=key)
        from ..control import alarm
        alarm.page("MapleStory: her pet food", f"gamelab: {text[0] + text[1:].lower()} Her pet isn't fed by "
                   f"{key.upper()} for {FOOD_DOUBT_WAIT_S / 60:.0f} minutes. Check her Pet Food and the game's Key "
                   "Bindings (\\), or run `python -m gamelab keys --read`.")

    def _feed(self) -> None:
        self.actions.feed_pet()

    def _read(self) -> int | None:
        """The pet's fullness on the Character Info window open now (a capture read), or None."""
        return read_fullness(self.eyes._capture(), self.ocr)

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


def due_to_feed(now: float) -> tuple[bool, float | None]:
    """Whether the fullness followed (paths.PET) says the pet is due a food at `now` (time.time), as `tend` would give
    it one there (followed under FOLLOW_FEED_BELOW; never read, FALLBACK_S since the last food), the time since counted
    as the game window's (the pet out); and the fullness followed (None: not known). Nothing is saved."""
    state = _state()
    follow(state, now, True)
    if "level" in state:
        return state["level"] < FOLLOW_FEED_BELOW, state["level"]
    return now - state.get("fed", 0.0) >= FALLBACK_S, None


class PetWait:
    """`pet --every`'s wait while the game's client isn't fit to read (see the top): `unfit()` says what's wrong now
    (None: nothing), `picture()` is the client's picture at whatever size (the perception service's: raises BadLook if
    there's none), `spot()` the Lie Detector on it (window_size.refit's look before each Alt+Enter). Called
    (`__call__`) before each follow: whether to wait now, said once and when the why changes; and the one bounded
    Alt+Enter-twice try when the pet is due a food (`_try`). `clock`: time.time's stand-in (paths.PET's times)."""

    def __init__(self, title: str, unfit: Callable[[], Problem | None], picture: Callable[[], np.ndarray],
                 spot: Callable[[], object] | None = None, log: Callable[[str], None] = print, clock=time.time,
                 want: tuple[int, int] = GAME_VIEW):
        self.title, self.unfit, self.picture, self.spot, self.log, self.clock = title, unfit, picture, spot, log, clock
        self.want = want
        self._again = -1e18  # when a look before a try last came to nothing (the game not in play, no picture, no
                             # Alt+Enter sent): looked at again window_size.AGAIN_S on, not at every poll
        self.waiting: tuple[float, str] | None = None  # since when it has waited, and why (said)
        self._since: tuple[str, float] | None = None   # the fault, and since when it has been so (window_size.STEADY_S)
        self._said: set[str] = set()                   # why no try was made, said once while it waits
        self.tried = 0                                 # tries made

    def __call__(self) -> bool:
        problem = self.unfit()
        now = self.clock()
        if problem is None:
            if self.waiting is not None:
                self.log(f"pet: the game looks right again ({now - self.waiting[0]:.0f} s waited): its food goes by the "
                         f"fullness followed")
            self.waiting, self._since, self._said = None, None, set()
            return False
        if self.waiting is None or self.waiting[1] != problem.why:
            self.log(f"pet: waiting: {problem.why}; no food is sent until it's right again (its fullness is still "
                     f"followed)")
            telemetry.emit("event", kind="pet", text=f"waiting: {problem.why}")
            self.waiting = (now if self.waiting is None else self.waiting[0], problem.why)
        if self._since is None or self._since[0] != problem.why:
            self._since = (problem.why, now)
        self._try(problem, now)
        return True

    def _try(self, problem: Problem, now: float) -> None:
        """The one bounded try (see the top): made only for a fault Alt+Enter mends, stood window_size.STEADY_S, the pet
        due a food, no alarm, none in the last TRY_EVERY_S (nor HOURLY in the hour, the hunts' included), and the game
        in play on the client's picture."""
        if problem.kind not in framecheck.FIX_KINDS or now - self._since[1] < window_size.STEADY_S:
            return
        due, level = due_to_feed(now)
        if not due:
            return
        if client.alarm_up():
            return self._not_now("the Lie Detector alarm is up")
        if now - _state().get("window_try", -1e18) < TRY_EVERY_S or now - self._again < window_size.AGAIN_S:
            return
        if len(window_size.tries_within(window_size.WINDOW_TRIES, time.time())) >= window_size.HOURLY:
            return self._not_now(f"Alt+Enter was tried {window_size.HOURLY} times in the last hour")
        try:
            seen, _ = framecheck.hud_seen(self.picture())
        except BadLook as e:
            self._again = now
            return self._not_now(f"no picture of the client to look for the game in play on ({e.why})")
        if seen < framecheck.HUD_SEEN:
            self._again = now
            return self._not_now(f"the game isn't seen in play (its HUD's buttons match {seen:.2f}, "
                                 f"{framecheck.HUD_SEEN} needed: its login or character screen?)")
        food = "its fullness not known" if level is None else f"its fullness about {level:.0f}"
        self.log(f"pet: the pet is due a food ({food}) and the game is in play (its HUD seen, {seen:.2f}): one try at "
                 f"Alt+Enter twice, full screen and back")

        def counted() -> None:  # (refit's on_try, as its first Alt+Enter goes: a try cut short is one too)
            state = _state()
            state["window_try"] = now
            _save(state)
            window_size.book(window_size.WINDOW_TRIES, time.time())
            self.tried += 1
        try:
            out = window_size.refit(self.title, self.want, self.log, self.spot, on_try=counted)
        except RuntimeError as e:  # the game window gone meanwhile
            return self._not_now(str(e))
        if out.held:
            return self._not_now(out.why)
        if not out.counted:  # (brought forward, no Alt+Enter sent: no picture, say)
            self._again = now
            return self._not_now(out.why)
        telemetry.emit("event", kind="pet", text=f"Alt+Enter twice tried: {out.text(self.want)}")

    def _not_now(self, why: str) -> None:
        if why not in self._said:
            self._said.add(why)
            self.log(f"pet: no Alt+Enter: {why}")


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
