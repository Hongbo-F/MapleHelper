"""The game's KEY BINDINGS window (the backslash key opens it, Esc closes it unsaved) read off a frame of the client at
its own 1366x768: which key(s) the game has the Attack action on, the preset in use, and that the window is the one
read and wholly in view. What behavior.keycheck asks after her regular attack took no effect 5 times in a row.

The window is the newer one, not the classic v83 KeyConfig: live on 2026-10-28 at 10:56 (`gamelab act \\`, then
`act esc`: D:/MapleHelper-scratch/akey/live/keyconfig.png and keyconfig-closed.png) it opened at (177, 174)-(813, 619),
"KEY BINDINGS" on its title, tabs "Keyboard/Mouse" and "Controller", "Preset 1 2 3", a whole keyboard of key cells,
each bound one with its action's picture drawn on it under the key's name (both Ctrl cells "ATTACK", both Alt "JUMP",
Z "PICK UP", the item and skill pictures of her potions, pet food, chair and Lucky Seven on C, V, U, D and Shift), a
tray of the actions bound to no key (MUTE, KEYS1-3), and DEFAULT, CLEAR ALL, CONFIRM, CANCEL. Its pictures are the copy's
`UI/UI_000.wz` KeyBindings.img (not its orange KeyConfig icons, nor UIWindow's, which this copy hasn't), and they're
drawn as they are: on the live frame every one of the 43 function pictures on a key matches its own to the pixel
(rms 0.0-0.07), the best other at 57.6. They're in assets/keybindings, cut from the copy by `cut_assets`
(tests/test_keybindings.py: the same as the copy's, where the copy is).

How it's read (`Reader.read`, 20-25 ms for a frame; the whole map of what each key does, `Reader.functions`, about
0.5 s more, for the log):
1. the window: its title strip (backgrnd's top TITLE_ROWS rows, 636 wide) matched over the frame, TM_SQDIFF_NORMED
   TITLE_OK at most (0.0076 open live, 0.38 on the frame after Esc), and the whole of it in the frame;
2. that it's the Keyboard/Mouse tab (its picture selected at TAB_AT), and the keyboard's own pixels between the key cells
   as its picture has them (GAPS_OK of them: another window or the Controller tab over it, and nothing is read);
3. the preset, the page button drawn "checked" (the pointer over one draws the same picture as "mouseOver": the check
   moves it off the window first, behavior.keycheck.MOUSE_AT);
4. every key cell (CELLS, off the keyboard's picture: 77 the game binds, 74 keys by config.toml's names, both Ctrls one
   binding as the game binds by scan code, both Shifts and both Alts too; Tab, Caps Lock, Enter, Backspace, Print
   Screen, Pause, '/', the Windows and menu keys are drawn grey: it doesn't): ATTACK's picture (icon 52) drawn there,
   its bottom ICON_ROWS rows (the key's name is drawn over the top) matched within SLACK px, the alpha the mask, scored
   by the rms of the difference less its mean (a frame dimmed by 30% scores 20.4, an exact draw 0.0-2.4, the best other
   cell 57.6): ATTACK_OK at most is ATTACK, over ATTACK_UNSURE not, between unsure (`unsure`);
5. the tray: ATTACK's picture in one of its TRAY_COLS x TRAY_ROWS slots means the game has it on no key.

The frame is RGB, the client's at its own 1366x768, where the game draws its pictures as they are; a frame of
another size isn't read (the client stretched: the frame of 2026-10-18 at 1920x1080, lab/keyconfig.png, brought down
to 1366x768 has its title strip at 0.019-0.023 but no key's ATTACK picture under 33, the tab's own px 47% as drawn).
`compose` draws the window from the same pictures with any bindings, for the tests and the simulator."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..paths import ASSETS

FOLDER = ASSETS / "keybindings"
FRAME = (1366, 768)        # the client's size it's read at (w, h)
WINDOW = (636, 445)        # the window's size: backgrnd's
KEYBOARD_AT = (9, 53)      # backgrnd_keyboard's place in the window (its origin negated)
TAB_AT = (9, 24)           # tab:kind's lt: the Keyboard/Mouse tab's picture, selected, drawn there ...
TAB_STEP = 110             # ... and the Controller tab's that far right of it
PRESET_AT = {1: (74, 60), 2: (103, 60), 3: (132, 60)}  # button:presetPage<k>'s place (its origin negated)
TRAY_AT, TRAY_STEP, TRAY_COLS, TRAY_ROWS = (13, 301), 34, 18, 3  # the unbound actions' slots (MUTE drawn at (557, 335))
OPEN_AT = (177, 174)       # where it opened live (the window can be dragged: it's looked for anywhere)
TITLE_ROWS = 20            # the title strip ("KEY BINDINGS") matched to find it ...
TITLE_OK = 0.06            # ... TM_SQDIFF_NORMED at most (0.0076-0.0077 on the 13 live frames with it; 0.036 those 15%
                           # dimmer; 0.35-0.39 on the 13 frames without it of lab/shots at 1366x768)
GAPS_OK = 0.9              # share of the keyboard's own px between the cells as its picture has them (within GAP_LEVELS)
GAP_LEVELS = 16           # (the difference's mean taken out: a frame 15% dimmer reads)
DRAWN_LEVELS = 16          # a picture's px drawn as they are: within this of it, the difference's mean taken out
LEVELS_OFF = 40            # the levels' fit (Reader._levels) leaves out the title's px this far off it (its close box)
GAPS_FROM = 26             # ... from this row of the keyboard's picture down: its key rows (the first's pictures at 28)
ICON_ROWS = 16             # a function picture's bottom rows matched (its key's name is drawn over the top ones)
SLACK = 3                  # px it's looked for around its place
ATTACK_OK = 25.0           # mean-removed rms at most: ATTACK (exact 0.0-2.4, 30% dimmer 20.4, the best other 57.6) ...
ATTACK_UNSURE = 45.0       # ... over this, not; between, unsure (behavior.keycheck reads a second frame)
EMPTY_OK = 0.9             # a cell's px under its name as the keyboard's picture has them: nothing bound
ATTACK = 52
# The function pictures (KeyBindings.img/icon/<id>), as the window names them
FUNCTIONS = {0: "EQUIPMENT", 1: "ITEM", 2: "ABILITY", 3: "SKILL", 4: "BUDDY", 5: "WORLD MAP", 6: "MAPLE CHAT",
             7: "MINIMAP", 8: "QUEST", 9: "SET KEY", 10: "TO ALL", 11: "WHISPER", 12: "TO PARTY", 13: "TO BUDDY",
             14: "SHORTCUT", 15: "QUICK SLOT", 16: "CHAT+", 19: "PARTY", 22: "CASH SHOP", 45: "CHANNEL", 46: "MENU",
             47: "SCREEN SHOT", 48: "PICTURE MODE", 50: "PICK UP", 51: "SIT", 52: "ATTACK", 53: "JUMP", 54: "NPC CHAT",
             55: "CRAFTING", 58: "MAILBOX", 202: "MUTE", 204: "KEYS1", 205: "KEYS2", 206: "KEYS3", 300: "MOVE LEFT",
             301: "MOVE RIGHT", 302: "MOVE UP", 303: "MOVE DOWN", 400: "OK", 401: "CANCEL",
             **{i: f"FACE {i - 99}" for i in range(100, 107)}}
PICTURE = "picture"  # a cell with an item's or a skill's picture on it (not a function's)

# The keyboard (backgrnd_keyboard, 618x237): each row's top (px in the keyboard's picture) and its cells: x, width, the
# key (config.toml's name; None: drawn grey, the game doesn't bind it), its scan code (the game's key/<code> name
# picture), where its name is drawn (px right of the cell's x, one under its top) and how far off centre its function
# picture is (the right Ctrl's 2 px left). Measured on the live frame (D:/MapleHelper-scratch/akey/design/ui).
ROWS = [
    (30, [(10, 28, "esc", 1, 0), (78, 28, "f1", 59, 0), (112, 28, "f2", 60, 0), (146, 28, "f3", 61, 0),
          (180, 28, "f4", 62, 0), (222, 28, "f5", 63, 0), (256, 28, "f6", 64, 0), (290, 28, "f7", 65, 0),
          (324, 28, "f8", 66, 0), (366, 28, "f9", 67, 0), (400, 28, "f10", 68, 0), (434, 28, "f11", 87, 0),
          (468, 28, "f12", 88, 0), (510, 28, None, None, 0), (544, 28, "scrolllock", 70, 0), (578, 28, None, None, 0)]),
    (69, [(10, 28, "`", 41, 0), (44, 28, "1", 2, 0), (78, 28, "2", 3, 0), (112, 28, "3", 4, 0), (146, 28, "4", 5, 0),
          (180, 28, "5", 6, 0), (214, 28, "6", 7, 0), (248, 28, "7", 8, 0), (282, 28, "8", 9, 0), (316, 28, "9", 10, 0),
          (350, 28, "0", 11, 0), (384, 28, "-", 12, 0), (418, 28, "=", 13, 0), (452, 44, None, None, 0),
          (510, 28, "insert", 82, 0), (544, 28, "home", 71, 0), (578, 28, "pageup", 73, 0)]),
    (102, [(10, 44, None, None, 0), (60, 28, "q", 16, 0), (94, 28, "w", 17, 0), (128, 28, "e", 18, 0),
           (162, 28, "r", 19, 0), (196, 28, "t", 20, 0), (230, 28, "y", 21, 0), (264, 28, "u", 22, 0),
           (298, 28, "i", 23, 0), (332, 28, "o", 24, 0), (366, 28, "p", 25, 0), (400, 28, "[", 26, 0),
           (434, 28, "]", 27, 0), (468, 28, "\\", 43, 0), (510, 28, "delete", 83, 0), (544, 28, "end", 79, 0),
           (578, 28, "pagedown", 81, 0)]),
    (135, [(10, 61, None, None, 0), (77, 28, "a", 30, 0), (111, 28, "s", 31, 0), (145, 28, "d", 32, 0),
           (179, 28, "f", 33, 0), (213, 28, "g", 34, 0), (247, 28, "h", 35, 0), (281, 28, "j", 36, 0),
           (315, 28, "k", 37, 0), (349, 28, "l", 38, 0), (383, 28, ";", 39, 0), (417, 28, "'", 40, 0),
           (451, 45, None, None, 0)]),
    (168, [(10, 78, "shift", 42, 25), (94, 28, "z", 44, 0), (128, 28, "x", 45, 0), (162, 28, "c", 46, 0),
           (196, 28, "v", 47, 0), (230, 28, "b", 48, 0), (264, 28, "n", 49, 0), (298, 28, "m", 50, 0),
           (332, 28, ",", 51, 0), (366, 28, ".", 52, 0), (400, 28, None, None, 0), (434, 62, "shift", 42, 16),
           (544, 28, "up", 72, 0)]),
    (201, [(10, 43, "ctrl", 29, 7), (59, 44, None, None, 0), (109, 47, "alt", 56, 9), (162, 164, "space", 57, 66),
           (332, 50, "alt", 56, 11), (388, 50, None, None, 0), (444, 52, "ctrl", 29, 10), (510, 28, "left", 75, 0),
           (544, 28, "down", 80, 0), (578, 28, "right", 77, 0)]),
]
CELL_H = 25
ICON_DX = {(444, 201): -2}  # a function picture drawn off its cell's centre (the right Ctrl's), px
# config.toml's names that are one binding to the game (it binds by scan code): the right-hand twins
SAME_KEY = {"rctrl": "ctrl", "rshift": "shift", "ralt": "alt"}


@dataclass(frozen=True)
class Cell:
    """A key cell of the keyboard: its place in the keyboard's picture (x, y, width; CELL_H high), the key (None: the
    game doesn't bind it), its scan code, its name's place in it (label_dx) and where its function picture is drawn
    (icon: px in the keyboard's picture)."""
    x: int
    y: int
    w: int
    key: str | None
    code: int | None
    label_dx: int

    @property
    def icon(self) -> tuple[int, int]:
        return self.x + (self.w - 32) // 2 + ICON_DX.get((self.x, self.y), 0), self.y - 2


CELLS = [Cell(x, y, w, key, code, dx) for y, row in ROWS for x, w, key, code, dx in row]
BOUND = [c for c in CELLS if c.key is not None]
KEYS = tuple(dict.fromkeys(c.key for c in BOUND))  # the keys the game binds, by config.toml's names (74)


def same_key(a: str | None, b: str | None) -> bool:
    """Whether two of config.toml's key names are one binding to the game (the right Ctrl is the left's)."""
    return a is not None and b is not None and SAME_KEY.get(a, a) == SAME_KEY.get(b, b)


@dataclass
class Reading:
    """What a frame showed of the KEY BINDINGS window (Reader.read). `open`: its title found; `why`: "read", or why not
    (not open; partly off the frame; not the Keyboard/Mouse tab; covered); `at`: its top-left (client px); `title`: the
    title strip's score; `preset`: the page drawn checked (None: not told); `attack`: the keys carrying ATTACK (config.toml
    names, in the keyboard's order, a key once: both Ctrls are 'ctrl'); `in_tray`: ATTACK in the tray (on no key);
    `unsure`: keys whose cells scored between ATTACK_OK and ATTACK_UNSURE; `scores`: each bound key's best ATTACK score;
    `gaps`: the keyboard's px between its cells as its picture has them (share)."""
    open: bool
    why: str
    at: tuple[int, int] | None = None
    title: float = 1.0
    preset: int | None = None
    attack: tuple[str, ...] = ()
    in_tray: bool = False
    unsure: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    gaps: float = 0.0

    @property
    def read(self) -> bool:
        """Whether the window was read: found, wholly in view, the Keyboard/Mouse tab, not covered."""
        return self.why == "read"

    def text(self) -> str:
        if not self.read:
            return f"not read: {self.why}" + (f" (title {self.title:.3f})" if not self.open else "")
        where = ("on no key (in the tray)" if self.in_tray else "on no key" if not self.attack else
                 "on " + ", ".join(k.upper() for k in self.attack))
        return (f"ATTACK {where}" + (f" (unsure: {', '.join(k.upper() for k in self.unsure)})" if self.unsure else "")
                + f"; preset {self.preset or '?'}; the window at {self.at}")

    def record(self) -> dict:
        return {"open": self.open, "why": self.why, "at": None if self.at is None else list(self.at),
                "title": round(self.title, 4), "preset": self.preset, "attack": list(self.attack),
                "in_tray": self.in_tray, "unsure": list(self.unsure), "gaps": round(self.gaps, 3)}


def _rgba(path: Path) -> np.ndarray:
    """An asset as RGBA uint8 (cv2 reads BGRA)."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    return np.ascontiguousarray(im[:, :, [2, 1, 0, 3]])


class Reader:
    """Reads the window off frames with the pictures in `folder` (assets/keybindings by default)."""

    def __init__(self, folder: Path = FOLDER):
        self.folder = Path(folder)
        self.bg = _rgba(self.folder / "backgrnd.png")
        self.kb = _rgba(self.folder / "backgrnd_keyboard.png")
        self.strip = np.ascontiguousarray(self.bg[:TITLE_ROWS, :, :3])
        self._strip_px = self.bg[:TITLE_ROWS, :, 3] > 250  # (its opaque px: the levels' fit)
        self.tabs = {(state, k): _rgba(self.folder / f"tab_{state}_{k}.png") for state in ("selected", "normal")
                     for k in (0, 1)}
        self.presets = {k: {state: _rgba(self.folder / f"preset{k}_{state}.png") for state in ("checked", "normal")}
                        for k in PRESET_AT}
        self.icons: dict[int, np.ndarray] = {}
        for p in sorted((self.folder / "icon").glob("*.png"), key=lambda p: int(p.stem)):
            self.icons[int(p.stem)] = _rgba(p)
        self.labels = {int(p.stem): _rgba(p) for p in (self.folder / "key").glob("*.png")}
        a = self.icons[ATTACK]
        self._att = a[ICON_ROWS:, :, :3].astype(np.float32)
        self._att_mask = a[ICON_ROWS:, :, 3] > 200
        # The keyboard's own px between its cells: outside every bound cell's function picture (and SLACK about it),
        # opaque in its picture
        gaps = self.kb[:, :, 3] > 200
        gaps[:GAPS_FROM] = False  # (the preset pages and QUICK SLOTS KEYS are drawn over its top rows)
        for c in BOUND:
            ix, iy = c.icon
            gaps[max(0, iy - SLACK):iy + 32 + SLACK, max(0, ix - SLACK):ix + 32 + SLACK] = False
        self._gaps = gaps

    # --- the window ---------------------------------------------------------------------------------------------

    def find(self, frame: np.ndarray) -> tuple[tuple[int, int] | None, float]:
        """The window's top-left in `frame` (RGB) and its title strip's score, or (None, score)."""
        if frame.shape[0] < TITLE_ROWS or frame.shape[1] < WINDOW[0]:
            return None, 1.0
        r = cv2.matchTemplate(np.ascontiguousarray(frame[:, :, :3]), self.strip, cv2.TM_SQDIFF_NORMED)
        mn, _, loc, _ = cv2.minMaxLoc(r)
        return (loc if mn <= TITLE_OK else None), float(mn)

    def read(self, frame: np.ndarray) -> Reading:
        """What `frame` (the client, RGB) shows of the window: see the module's doc."""
        h, w = frame.shape[:2]
        if (w, h) != FRAME:
            return Reading(False, f"the frame is {w}x{h}, not {FRAME[0]}x{FRAME[1]}")
        at, score = self.find(frame)
        if at is None:
            return Reading(False, "not open", title=score)
        ox, oy = at
        if ox + WINDOW[0] > w or oy + WINDOW[1] > h:
            return Reading(True, "partly off the frame", at, score)
        win = self._levels(frame[oy:oy + WINDOW[1], ox:ox + WINDOW[0]])
        if self._drawn(win, self.tabs["selected", 0], TAB_AT) < 0.9:
            tab = self._drawn(win, self.tabs["selected", 1], (TAB_AT[0] + TAB_STEP, TAB_AT[1]))
            return Reading(True, "the Controller tab" if tab >= 0.9 else "not the Keyboard/Mouse tab", at, score)
        kx, ky = KEYBOARD_AT
        kbd = win[ky:ky + self.kb.shape[0], kx:kx + self.kb.shape[1]].astype(np.float32)
        d = (kbd - self.kb[:, :, :3].astype(np.float32))[self._gaps]
        d -= d.mean(axis=0)  # (a frame dimmed or brightened: its difference's mean taken out)
        gaps = float((np.abs(d).max(axis=1) <= GAP_LEVELS).mean())
        if gaps < GAPS_OK:
            return Reading(True, "covered (the keyboard's own px not as its picture has them)", at, score, gaps=gaps)
        checked = {k: self._drawn(win, p["checked"], PRESET_AT[k]) for k, p in self.presets.items()}
        best = max(checked, key=checked.get)
        preset = best if checked[best] >= 0.9 and sorted(checked.values())[-2] < 0.6 else None
        scores: dict[str, float] = {}
        for c in BOUND:
            s = self._attack_score(win, kx + c.icon[0], ky + c.icon[1])
            scores[c.key] = min(s, scores.get(c.key, float("inf")))
        attack = tuple(k for k in KEYS if scores[k] <= ATTACK_OK)
        unsure = tuple(k for k in KEYS if ATTACK_OK < scores[k] <= ATTACK_UNSURE)
        return Reading(True, "read", at, score, preset, attack, self._in_tray(win), unsure,
                       {k: round(v, 1) for k, v in scores.items()}, gaps)

    def _levels(self, win: np.ndarray) -> np.ndarray:
        """The window's crop with the frame's levels brought back to the pictures': its title strip fitted to the
        strip's picture as gain and offset (a frame the game dims, or a capture's levels), undone over the whole crop;
        as it is where they're 1 and 0 already."""
        x = win[:TITLE_ROWS][self._strip_px].astype(np.float32).reshape(-1)
        y = self.strip[self._strip_px].astype(np.float32).reshape(-1)
        a, b = 1.0, 0.0
        for _ in range(2):  # (a fit, then again without the px it leaves far off: the close box drawn over it)
            keep = np.abs(a * x + b - y) < LEVELS_OFF
            if keep.sum() < 100 or float(x[keep].var()) < 1.0:
                return win
            xs, ys = x[keep], y[keep]
            a = float(((xs - xs.mean()) * (ys - ys.mean())).mean() / xs.var())
            b = float(ys.mean() - a * xs.mean())
        if abs(a - 1.0) < 0.02 and abs(b) < 2.0:
            return win
        return np.clip(win.astype(np.float32) * a + b + 0.5, 0, 255).astype(np.uint8)

    def _drawn(self, win: np.ndarray, pic: np.ndarray, place: tuple[int, int], slack: int = 1) -> float:
        """The share of picture `pic`'s opaque px drawn as they are (within DRAWN_LEVELS, the difference's mean taken
        out: a frame dimmed or brightened) at `place` in the window (best within `slack` px)."""
        h, w = pic.shape[:2]
        a = pic[:, :, 3] > 200
        best = 0.0
        for dy in range(-slack, slack + 1):
            for dx in range(-slack, slack + 1):
                x, y = place[0] + dx, place[1] + dy
                if x < 0 or y < 0:
                    continue
                crop = win[y:y + h, x:x + w, :3]
                if crop.shape[:2] != (h, w):
                    continue
                d = (crop.astype(np.float32) - pic[:, :, :3].astype(np.float32))[a]
                d -= d.mean(axis=0)
                best = max(best, float((np.abs(d).max(axis=1) <= DRAWN_LEVELS).mean()))
        return best

    def _attack_score(self, frame: np.ndarray, x: int, y: int) -> float:
        """ATTACK's picture's bottom rows against `frame` with its top-left near client px (x, y): the least rms (per
        channel, over its opaque px) of the difference less its mean, within SLACK px."""
        tpl, m = self._att, self._att_mask
        th, tw = tpl.shape[:2]
        y0, x0 = y + ICON_ROWS - SLACK, x - SLACK
        region = frame[max(0, y0):y0 + th + 2 * SLACK, max(0, x0):x0 + tw + 2 * SLACK, :3].astype(np.float32)
        if region.shape[0] < th or region.shape[1] < tw:
            return float("inf")
        win = np.lib.stride_tricks.sliding_window_view(region, (th, tw, 3))[:, :, 0]  # (oy, ox, th, tw, 3)
        d = win[:, :, m, :] - tpl[m][None, None]                                      # (oy, ox, n, 3)
        d = d - d.mean(axis=2, keepdims=True)
        rms = np.sqrt((d ** 2).mean(axis=(2, 3)))
        return float(rms.min())

    def _in_tray(self, win: np.ndarray) -> bool:
        """Whether ATTACK's picture is drawn in one of the tray's slots (the actions bound to no key)."""
        a = self.icons[ATTACK]
        m = (a[:, :, 3] > 200).astype(np.uint8)
        x0, y0 = TRAY_AT
        area = np.ascontiguousarray(win[y0 - SLACK:y0 + TRAY_ROWS * TRAY_STEP + SLACK,
                                        x0 - SLACK:x0 + TRAY_COLS * TRAY_STEP + SLACK, :3])
        if area.shape[0] < 32 or area.shape[1] < 32:
            return False
        r = cv2.matchTemplate(area, np.ascontiguousarray(a[:, :, :3]), cv2.TM_SQDIFF, mask=m) / m.sum() / 3
        return float(np.sqrt(max(0.0, float(r.min())))) <= 6.0

    # --- the whole map, for the log --------------------------------------------------------------------------------

    def functions(self, frame: np.ndarray, reading: Reading) -> dict[str, str | None]:
        """What each key does as the window read shows it: a function's name (FUNCTIONS), PICTURE (an item's or a
        skill's picture: not a function), or None (nothing bound); {} for a window not read. About 0.5 s (every
        function picture at every cell, by the masked match's absolute rms: the log's, not the decision's)."""
        if not reading.read:
            return {}
        ox, oy = reading.at
        frame = self._levels(frame[oy:oy + WINDOW[1], ox:ox + WINDOW[0]])  # (the window's crop from here on)
        kx, ky = KEYBOARD_AT
        out: dict[str, str | None] = {}
        for c in BOUND:
            ix, iy = c.icon
            region = np.ascontiguousarray(frame[ky + iy + ICON_ROWS - SLACK:ky + iy + 32 + SLACK,
                                                kx + ix - SLACK:kx + ix + 32 + SLACK, :3])
            best, which = float("inf"), None
            for i, pic in self.icons.items():
                tpl = np.ascontiguousarray(pic[ICON_ROWS:, :, :3])
                m = (pic[ICON_ROWS:, :, 3] > 200).astype(np.uint8)
                r = cv2.matchTemplate(region, tpl, cv2.TM_SQDIFF, mask=m)
                s = float(np.sqrt(max(0.0, float(r.min())) / m.sum() / 3))
                if s < best:
                    best, which = s, i
            cell = frame[ky + c.y + 12:ky + c.y + CELL_H, kx + c.x:kx + c.x + c.w].astype(np.int16)
            empty = float((np.abs(cell - self.kb[c.y + 12:c.y + CELL_H, c.x:c.x + c.w, :3].astype(np.int16))
                           .max(axis=2) <= 8).mean())
            what = FUNCTIONS.get(which, str(which)) if best <= 8.0 else PICTURE if empty < EMPTY_OK else None
            if out.get(c.key) is None:
                out[c.key] = what
        return out


def functions_text(functions: dict[str, str | None], keys=None) -> str:
    """The keys' functions as one line ("CTRL ATTACK, ALT JUMP, ..."): every bound one, or those of `keys`."""
    items = [(k, v) for k, v in functions.items() if v is not None and (keys is None or k in keys)]
    return ", ".join(f"{k.upper()} {v}" for k, v in items) or "nothing bound"


# --- the window drawn from its pictures (the tests, the simulator) ------------------------------------------------

_DRAWN: dict[str, Reader] = {}


def _pictures(folder: Path = FOLDER) -> Reader:
    key = str(folder)
    if key not in _DRAWN:
        _DRAWN[key] = Reader(folder)
    return _DRAWN[key]


def _paste(dst: np.ndarray, pic: np.ndarray, x: int, y: int) -> None:
    """`pic` (RGBA) drawn over `dst` (RGB) at (x, y), its alpha blended, clipped to `dst`."""
    h, w = pic.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    p = pic[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = p[:, :, 3:4] / 255.0
    d = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(p[:, :, :3] * a + d * (1 - a) + 0.5, 0, 255).astype(np.uint8)


def picture_icon(seed: int = 0) -> np.ndarray:
    """A 32x32 RGBA picture that's no function's (an item's or a skill's stand-in), for `compose`."""
    rng = np.random.default_rng(seed)
    im = np.zeros((32, 32, 4), np.uint8)
    im[2:30, 2:30, :3] = rng.integers(0, 256, (28, 28, 3), dtype=np.uint8)
    im[2:30, 2:30, 3] = 255
    return im


def compose(bindings: dict[str, int | str | None], preset: int = 1, at: tuple[int, int] = OPEN_AT,
            frame: np.ndarray | None = None, tray: dict[tuple[int, int], int] | None = None,
            tab: int = 0, folder: Path = FOLDER) -> np.ndarray:
    """The KEY BINDINGS window drawn on `frame` (RGB, a copy; a plain one at FRAME's size by default) at `at`: its
    background, the Keyboard/Mouse tab selected (`tab` 1: the Controller's, its keyboard not drawn), preset `preset`
    checked, every key's name, and on each key of `bindings` its function's picture (an icon id), a picture that's no
    function's (PICTURE) or nothing (None, or not named); `tray`: the unbound actions' slots, (col, row) -> icon id
    (MUTE, KEYS1-3 where the live window has them, by default). As the game draws it: the function picture first, the
    key's name over it."""
    r = _pictures(folder)
    out = (np.full((FRAME[1], FRAME[0], 3), (90, 140, 70), np.uint8) if frame is None else frame.copy())
    ox, oy = at
    kx, ky = ox + KEYBOARD_AT[0], oy + KEYBOARD_AT[1]
    _paste(out, r.bg, ox, oy)
    if tab == 0:
        _paste(out, r.kb, kx, ky)
    _paste(out, r.tabs["selected" if tab == 0 else "normal", 0], ox + TAB_AT[0], oy + TAB_AT[1])
    _paste(out, r.tabs["selected" if tab == 1 else "normal", 1], ox + TAB_AT[0] + TAB_STEP, oy + TAB_AT[1])
    for k, (px, py) in PRESET_AT.items():
        _paste(out, r.presets[k]["checked" if k == preset else "normal"], ox + px, oy + py)
    tray = {(16, 1): 202, (17, 1): 204, (0, 2): 205, (1, 2): 206} if tray is None else tray
    for (col, row), icon in tray.items():
        _paste(out, r.icons[icon], ox + TRAY_AT[0] + col * TRAY_STEP, oy + TRAY_AT[1] + row * TRAY_STEP)
    if tab != 0:
        return out
    for c in BOUND:
        what = bindings.get(c.key)
        if what is not None:
            pic = picture_icon(c.code) if what == PICTURE else r.icons[int(what)]
            _paste(out, pic, kx + c.icon[0], ky + c.icon[1])
        _paste(out, r.labels[c.code], kx + c.x + c.label_dx, ky + c.y + 1)
    return out


# The window as it was live on 2026-10-28 at 10:56 (D:/MapleHelper-scratch/akey/live/keyconfig.png): her keys' functions,
# preset 2 (`compose`'s default bindings for the tests and the simulator)
LIVE = {"esc": 46, **{f"f{i}": 99 + i for i in range(1, 8)}, "f9": 22, "f10": 14, "f11": 48, "scrolllock": 47,
        "8": 13, "9": 12, "0": 10, "q": 8, "w": 5, "e": 0, "u": PICTURE, "i": 1, "o": 4, "p": 19, "[": 6, "]": 15,
        "\\": 9, "s": 2, "d": PICTURE, "h": 11, "k": 3, "l": 55, ";": 58, "'": 16, "shift": PICTURE, "z": 50, "x": 51,
        "c": PICTURE, "v": PICTURE, "m": 7, ",": 45, "up": 302, "ctrl": 52, "alt": 53, "space": 54, "left": 300,
        "down": 303, "right": 301}
LIVE_PRESET = 2


# --- the pictures, cut from the copy ----------------------------------------------------------------------------

def cut_assets(dest: Path = FOLDER, copy: Path | None = None) -> int:
    """Write the pictures the reader and `compose` use into `dest`, from the copy's UI/UI_000.wz KeyBindings.img (their
    pixels where its links point: gamedata.canvas.source): backgrnd, backgrnd_keyboard, both tabs selected and normal,
    the preset pages checked and normal, every function picture (icon/<id>) and every bound key's name (key/<code>).
    Returns the pictures written."""
    from PIL import Image

    from gamedata.canvas import pixels, source
    from gamedata.paths import COPY
    from gamedata.wz import Package
    copy = COPY if copy is None else Path(copy)
    tree = Package(copy / "UI" / "UI_000.wz").image("KeyBindings.img")
    dest = Path(dest)
    (dest / "icon").mkdir(parents=True, exist_ok=True)
    (dest / "key").mkdir(parents=True, exist_ok=True)

    def put(node, name: str) -> None:
        Image.fromarray(pixels(source(node, tree, copy)), "RGBA").save(dest / name)
    put(tree["backgrnd"], "backgrnd.png")
    put(tree["backgrnd_keyboard"], "backgrnd_keyboard.png")
    n = 2
    for state in ("selected", "normal"):
        for k in (0, 1):
            put(tree["tab:kind"][state][str(k)], f"tab_{state}_{k}.png")
            n += 1
    for k in PRESET_AT:
        for state in ("checked", "normal"):
            put(tree[f"button:presetPage{k}"][state]["0"], f"preset{k}_{state}.png")
            n += 1
    for i in FUNCTIONS:
        put(tree["icon"][str(i)], f"icon/{i}.png")
        n += 1
    for code in sorted({c.code for c in BOUND}):
        put(tree["key"][str(code)], f"key/{code}.png")
        n += 1
    return n
