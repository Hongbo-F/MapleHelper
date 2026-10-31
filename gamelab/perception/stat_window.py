"""Her stat window read off a frame: her W.DEF (what her touches are worked out at: behavior.bootstrap, her stat window's
DEF over a fit to her touches, over the prior), and what else it shows that the bot uses or keeps (her level, max HP and
MP, STR/DEX/INT/LUK, ATTACK, MAGIC, MAGIC DEF., ACCURACY, EVASION, SPEED, JUMP). Opened, read and closed by
behavior.statcheck (at a hunt's start, at a quiet moment), and by `gamelab stats --read [--record]`.

The user, 2026-10-28 ~18:20, on her DEF not known at a first run (the reserve at DEF 0, 453 HP of 840 on Tree Dungeon,
until 8 touches are kept): "You can open character stats panel to check the defense".

**The window, from the copy's pictures.** Her key bindings (lab/shots/keyconfig.png, 2026-10-28) have S on ABILITY (KEY,
FUNCTION; the key followed from her remembered bindings where they say another: behavior.statcheck.stats_key); the
bottom bar's stat button opens it too (StatusBar.img button:Stat). The client's windows are the newer UI the copy's
UI/UI_000.wz holds (its KEY BINDINGS window, KeyBindings.img, matched the live one to the pixel: perception.keybindings);
its stat window is CharacterStat.img: CHARACTER STAT (Main/backgrnd, 185 x 281: NAME, JOB, LEVEL, HP, MP, EXP., FAME,
STR, DEX, INT, LUK, ABILITY POINT) and, by its DETAIL button (Main/button:detail at (122, 256), 47 x 18), the panel
beside it (Detail/backgrnd, 187 x 223; backgrnd2 the same but for its left 4 columns: ATTACK, WEAPON DEF., MAGIC, MAGIC
DEF., ACCURACY, EVASION, CRIT. RATE, CRIT. DAMAGE, SPEED, JUMP). Each value is drawn from its vector (Main/vector:level
(60, 79), Detail/vector:pdd (78, 28), ...) in a light cell (225, 223, 213) of the picture (CELLS: measured off the
pictures, `cells_of`). The values are text in the game's font (UI/Font: NotoSansKRMedium), so a panel's value column is
read by the OCR in one go (maplehelper.ocr's shared RapidOCR, enlarged x2 as the message bar's is), each row taken for
the cell it falls in, and parsed (`parse`: "840 / 840", "60 ~ 75", "100%").

**Found** (`find`): a panel's picture matched over the frame, its value cells and the DETAIL button masked out (what's
drawn there changes), the left 4 columns of the Detail panel too (backgrnd and backgrnd2 differ there): at the place it
was last seen first (`near`, SLACK), else at half size over the whole frame and then to the px; fit in levels^2 a channel
over the picture's px, FIND_FIT at the most (0 for the picture pasted in, about 30 as a JPEG at quality 85; the best
place on a frame without it 20,000 and more).

**Not seen live yet**: no saved frame shows it (the 4,085 lossless frames under lab/ and every 20th frame of the rolling
recordings: D:/MapleHelper-scratch/safety3/stats/statwin). So a hunt opens it only once a live capture has calibrated the
reader (`Calibration`, lab/stat_window.json, written by `gamelab stats --read --record`): where each panel opened, whether
DETAIL opened with the window, and whose pictures it was found by ("copy": the copy's matched; "live": cut from the live
frame by `calibrate` where they didn't, the value cells found by their labels with the OCR, templates in lab/stat_window/).
`compose` draws the window from the copy's pictures with any values (the tests, the simulator)."""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import paths

KEY = "s"            # her stats key where nothing says another: ABILITY in her key bindings (lab/shots/keyconfig.png)
FUNCTION = "ABILITY"  # its function in the KEY BINDINGS window (perception.keybindings.FUNCTIONS[2])
FRAME = (1366, 768)  # the client's size it's read at (w, h): the game draws its pictures as they are there
FIND_FIT = 400.0     # levels^2 a channel over the picture's px at the most, for it to be there
SLACK = 4            # px around the place it was last seen that it's looked for first
SHOWN_EVERY = 4      # `shown`'s px: every this many of a panel's, each way
PART = {"main": ("Main", "backgrnd"), "detail": ("Detail", "backgrnd")}
LIGHT = (225, 223, 213)  # a value cell's colour in the pictures
# The value cells (x0, y0, x1, y1: panel px), measured off the pictures (`cells_of`: the light cell around each vector),
# HP, MP and the four stats cut short of their AP buttons (Main/button:hpup at 148, hpupall at 161: drawn with AP to give)
CELLS = {"main": {"name": (58, 33, 175, 48), "job": (58, 51, 175, 76), "level": (58, 79, 175, 94),
                  "hp": (58, 97, 147, 112), "mp": (58, 115, 147, 130), "exp": (58, 133, 175, 148),
                  "fame": (58, 151, 175, 166), "str": (58, 175, 147, 190), "dex": (58, 193, 147, 208),
                  "int": (58, 211, 147, 226), "luk": (58, 229, 147, 244)},
         "detail": {"attack": (76, 10, 177, 25), "wdef": (76, 28, 177, 43), "magic": (76, 46, 177, 61),
                    "mdef": (76, 64, 177, 79), "acc": (76, 82, 177, 97), "eva": (76, 100, 177, 115),
                    "crit": (76, 118, 177, 133), "crit_damage": (76, 136, 177, 151), "speed": (76, 154, 177, 169),
                    "jump": (76, 172, 177, 187)}}
VECTORS = {"main": {"name": "name", "job": "job", "level": "level", "hp": "hp", "mp": "mp", "exp": "exp",
                    "fame": "fame", "str": "strValue_lt", "dex": "dexValue_lt", "int": "intValue_lt",
                    "luk": "lukValue_lt"},
           "detail": {"attack": "attack", "wdef": "pdd", "magic": "mad", "mdef": "mdd", "acc": "acc", "eva": "eva",
                      "crit": "crt", "crit_damage": "crd", "speed": "spd", "jump": "jmp"}}
WDEF_AT = (78, 28)   # (Detail/vector:pdd: where W.DEF's value is drawn from, in the Detail panel)
CELL = CELLS["detail"]["wdef"]  # (... and its cell)
DETAIL_BUTTON = (122, 256, 47, 18)  # Main/button:detail: x, y (its origin negated), w, h (panel px)
DETAIL_LEFT = 4      # the Detail panel's left columns left out of its match (backgrnd2's differ there)
WHERE = {"wdef": "detail", "level": "main"}  # (which panel a field is on, for what's said when it isn't read)
# What each field is called on the window, for `calibrate` (OCR'd labels, letters only, upper case): the copy's first
LABELS = {"level": ("LEVEL", "LV"), "hp": ("HP",), "mp": ("MP",), "str": ("STR",), "dex": ("DEX",), "int": ("INT",),
          "luk": ("LUK",), "attack": ("ATTACK", "ATT", "ATTACKPOWER"),
          "wdef": ("WEAPONDEF", "WDEF", "WEAPONDEFENSE", "DEF", "DEFENSE", "PHYSICALDEF"),
          "magic": ("MAGIC", "MATT", "MAGICATTACK"), "mdef": ("MAGICDEF", "MDEF", "MAGICDEFENSE"),
          "acc": ("ACCURACY", "ACC"), "eva": ("EVASION", "AVOID", "AVOIDABILITY", "EVA"), "speed": ("SPEED",),
          "jump": ("JUMP",)}
NUMBER = re.compile(r"\d{1,6}")
PAIR = re.compile(r"(\d{1,6})\s*[/~\-]\s*(\d{1,6})")


# --- the pictures ---------------------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def picture(part: str) -> np.ndarray:
    """A picture of the stat window from the game data (UI CharacterStat.img: "Main/backgrnd", "Detail/backgrnd",
    "Main/button:detail"), RGBA."""
    from gamedata.canvas import pixels, source
    from gamedata.paths import COPY
    from gamedata.wz import Package
    tree = Package(COPY / "UI" / "UI_000.wz").image("CharacterStat.img")
    a, b = part.split("/", 1)
    node = tree[a][b]
    if isinstance(node, dict):  # (a button: its normal picture)
        node = node["normal"]["0"]
    out = pixels(source(node, tree, COPY))
    out.flags.writeable = False
    return out


def cells_of(panel: str) -> dict[str, tuple[int, int, int, int]]:
    """The light value cells around each vector of `panel`, measured off its picture (CELLS, checked against these by
    tests/test_stat_window.py): the run of LIGHT on the vector's row 5 px down through its x + 3, and down and up its
    column."""
    from gamedata.paths import COPY
    from gamedata.wz import Package
    tree = Package(COPY / "UI" / "UI_000.wz").image("CharacterStat.img")
    a = PART[panel][0]
    img = picture(f"{a}/backgrnd")[..., :3].astype(int)
    light = (np.abs(img - np.array(LIGHT)).max(axis=2) <= 2)
    out = {}
    for name, vec in VECTORS[panel].items():
        v = tree[a][f"vector:{vec}"]
        x, y = v.x + 3, v.y + 5
        x0 = x1 = x
        while x0 - 1 >= 0 and light[y, x0 - 1]:
            x0 -= 1
        while x1 + 1 < light.shape[1] and light[y, x1 + 1]:
            x1 += 1
        y0 = y1 = y
        while y0 - 1 >= 0 and light[y0 - 1, x]:
            y0 -= 1
        while y1 + 1 < light.shape[0] and light[y1 + 1, x]:
            y1 += 1
        out[name] = (x0, y0, x1 + 1, y1 + 1)
    return out


@dataclass
class Template:
    """A panel as it's found: its picture (RGB), the mask of its px that count (its value cells and buttons out), its
    value cells (panel px), and whose it is ("copy": the game data's; "live": cut from a live frame by `calibrate`)."""
    name: str
    rgb: np.ndarray
    mask: np.ndarray
    cells: dict[str, tuple[int, int, int, int]]
    source: str = "copy"

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


@lru_cache(maxsize=4)
def _copy_template(panel: str) -> Template:
    rgba = picture("/".join(PART[panel]))
    mask = (rgba[..., 3] > 250).astype(np.uint8)
    for x0, y0, x1, y1 in CELLS[panel].values():
        mask[y0:y1, x0:x1] = 0
    if panel == "main":
        x, y, w, h = DETAIL_BUTTON
        mask[y:y + h, x:x + w] = 0
        mask[256:275, 86:120] = 0  # (the ABILITY POINT box: her AP drawn in it)
        mask[96:246, 146:176] = 0  # (her AP buttons beside HP, MP and the four stats, drawn with AP to give)
    else:
        mask[:, :DETAIL_LEFT] = 0
    rgb = np.ascontiguousarray(rgba[..., :3])
    rgb.flags.writeable = False
    mask.flags.writeable = False
    return Template(panel, rgb, mask, dict(CELLS[panel]), "copy")


def templates(cal: "Calibration | None" = None) -> dict[str, Template]:
    """The panels as they're found: the live templates of a calibration that cut them (`cal.source` "live", its
    folder's PNGs), else the copy's."""
    if cal is not None and cal.source == "live":
        out = {}
        for name, panel in cal.panels.items():
            t = _live_template(name, str(cal.folder / f"{name}.png"), json.dumps(panel.get("cells", {})))
            if t is not None:
                out[name] = t
        if out:
            return out
    return {p: _copy_template(p) for p in PART}


@lru_cache(maxsize=8)
def _live_template(name: str, path: str, cells: str) -> Template | None:
    try:
        from PIL import Image
        rgba = np.asarray(Image.open(path).convert("RGBA"))
    except (OSError, ValueError):
        return None
    mask = (rgba[..., 3] > 250).astype(np.uint8)
    return Template(name, np.ascontiguousarray(rgba[..., :3]), mask,
                    {k: tuple(v) for k, v in json.loads(cells).items()}, "live")


# --- finding a panel -------------------------------------------------------------------------------------------------

def _local(frame: np.ndarray, t: Template, x: int, y: int, slack: int) -> tuple[tuple[int, int] | None, float]:
    """The best place for `t` within `slack` px of (x, y), by the masked match, and its fit."""
    import cv2
    w, h = t.size
    x0, y0 = max(0, x - slack), max(0, y - slack)
    x1, y1 = min(frame.shape[1], x + w + slack), min(frame.shape[0], y + h + slack)
    if x1 - x0 < w or y1 - y0 < h:
        return None, float("inf")
    region = np.ascontiguousarray(frame[y0:y1, x0:x1, :3])
    r = cv2.matchTemplate(region, t.rgb, cv2.TM_SQDIFF, mask=t.mask)
    v, _, loc, _ = cv2.minMaxLoc(r)
    return (x0 + int(loc[0]), y0 + int(loc[1])), float(v) / max(1, int(t.mask.sum())) / 3


def find(frame: np.ndarray, part: str | Template = "detail", near: tuple[int, int] | None = None,
         cal: "Calibration | None" = None) -> tuple[tuple[int, int] | None, float]:
    """Where panel `part` ("main", "detail"; or "Detail/backgrnd" as before, or a Template) is on `frame` (full size,
    RGB, the client at its own 1366x768), its top left, and how well it fits there (levels^2 a channel); None for the
    place if it fits worse than FIND_FIT. Looked for within SLACK of `near` first (where it was last seen), else over
    the whole frame at half size, then to the px around the best place there."""
    import cv2
    if isinstance(part, Template):
        t = part
    else:
        name = {"Detail/backgrnd": "detail", "Main/backgrnd": "main"}.get(part, part)
        t = templates(cal)[name]
    w, h = t.size
    if frame is None or frame.shape[0] < h or frame.shape[1] < w:
        return None, float("inf")
    if near is not None:
        at, fit = _local(frame, t, int(near[0]), int(near[1]), SLACK)
        if at is not None and fit <= FIND_FIT:
            return at, fit
    small = np.ascontiguousarray(frame[::2, ::2, :3])
    st = np.ascontiguousarray(t.rgb[::2, ::2])
    sm = np.ascontiguousarray(t.mask[::2, ::2])
    if small.shape[0] < st.shape[0] or small.shape[1] < st.shape[1] or not sm.any():
        return None, float("inf")
    r = cv2.matchTemplate(small, st, cv2.TM_SQDIFF, mask=sm)
    _, _, loc, _ = cv2.minMaxLoc(r)
    at, fit = _local(frame, t, 2 * int(loc[0]), 2 * int(loc[1]), 3)
    return (at if at is not None and fit <= FIND_FIT else None), fit


def find_all(frame: np.ndarray, cal: "Calibration | None" = None,
             near: dict | None = None) -> dict[str, tuple[tuple[int, int] | None, float]]:
    """Each panel's place on `frame` and its fit (`find`), near where `near` (panel -> (x, y)) or the calibration last
    had it."""
    near = dict(near or {})
    if cal is not None:
        for name, panel in cal.panels.items():
            if panel.get("at") is not None:
                near.setdefault(name, tuple(panel["at"]))
    return {name: find(frame, t, near.get(name)) for name, t in templates(cal).items()}


def is_open(frame: np.ndarray, cal: "Calibration | None" = None, near: dict | None = None) -> bool:
    """Whether any panel of the stat window is on `frame`."""
    return any(at is not None for at, _ in find_all(frame, cal, near).values())


def shown(frame: np.ndarray, cal: "Calibration | None", every: int = SHOWN_EVERY) -> str | None:
    """The first panel of the stat window on `frame` (full size, RGB, the client at its 1366x768) exactly where `cal`
    has it open, or None (None too without a calibration): its masked fit at that place alone, no search (the game
    draws it where it opens, to the px), every `every`-th px of it each way, FIND_FIT at the most. For a look on many
    frames (her place by the scene counted "covered" under it: localization.locate.Cover): 0.06-0.08 ms a panel, where
    `find`'s search within SLACK is 6-13 ms; on the live clip of 2026-10-30 17:54:33 (every 7th of its 4,503 frames)
    79 at the most under it, 14,000 and more on every other frame."""
    if cal is None or frame is None:
        return None
    for name, t in templates(cal).items():
        at = cal.place(name)
        if at is None:
            continue
        (x, y), (w, h) = at, t.size
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue
        m = t.mask[::every, ::every].astype(bool)
        if not m.any():
            continue
        want = t.rgb[::every, ::every][m].astype(np.int32)
        got = frame[y:y + h:every, x:x + w:every, :3][m].astype(np.int32)
        if float(((got - want) ** 2).sum()) / want.size <= FIND_FIT:
            return name
    return None


# --- reading it ------------------------------------------------------------------------------------------------------

@dataclass
class Row:
    """A line of text the OCR read (maplehelper.ocr.Row's fields this needs), in the px of the crop it was given."""
    text: str
    x: float
    y: float
    left: float = 0.0
    right: float = 0.0
    height: float = 0.0
    score: float = 1.0


def _shared_rows(crop_rgb: np.ndarray, box=None) -> list:
    """maplehelper.ocr's shared engine's rows off a crop (RGB), enlarged x2 as the message bar's is, in the crop's px;
    [] without the OCR packages."""
    try:
        from maplehelper.ocr import shared
    except ImportError:
        return []
    big = np.ascontiguousarray(np.repeat(np.repeat(crop_rgb[..., 2::-1], 2, axis=0), 2, axis=1))
    rows = shared().read(big, min_height=0) or []
    return [Row(r.text, r.x / 2, r.y / 2, r.left / 2, r.right / 2, r.height / 2, r.score) for r in rows]


def parse(name: str, text: str):
    """A field's value from its text: HP and MP (cur, max) ("840 / 840"; one number: (None, it)), ATTACK (lo, hi)
    ("60 ~ 75"), NAME and JOB the text, the rest the first whole number ("100%", "25 (20+5)", "111"); None if none."""
    text = (text or "").strip()
    if name in ("name", "job"):
        return text or None
    if name in ("hp", "mp", "attack"):
        m = PAIR.search(text)
        if m is not None:
            return int(m.group(1)), int(m.group(2))
        m = NUMBER.search(text)
        return None if m is None else (None, int(m.group(0))) if name != "attack" else (int(m.group(0)),) * 2
    m = NUMBER.search(text)
    return None if m is None else int(m.group(0))


@dataclass
class StatReading:
    """The stat window as read off a frame: where each panel is (`at`: name -> (x, y)), each field's text and value
    (`texts`, `values`: parse's), and `why` it isn't read (""; a panel not there, no number in W.DEF's cell)."""
    at: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    why: str = ""
    fits: dict = field(default_factory=dict)

    @property
    def open(self) -> bool:
        return any(v is not None for v in self.at.values())

    @property
    def detail(self) -> bool:
        """Whether the Detail panel (her W.DEF's) is on the frame."""
        return self.at.get("detail") is not None or "wdef" in self.values

    def get(self, name: str):
        return self.values.get(name)

    @property
    def wdef(self) -> int | None:
        return self.values.get("wdef")

    @property
    def level(self) -> int | None:
        return self.values.get("level")

    @property
    def max_hp(self) -> int | None:
        v = self.values.get("hp")
        return None if v is None else v[1]

    @property
    def max_mp(self) -> int | None:
        v = self.values.get("mp")
        return None if v is None else v[1]

    def text(self) -> str:
        """What was read, in words: "W.DEF 111, level 28, HP 840/840, MP 138/679, ... (the window at (560, 150))"."""
        if not self.open:
            return f"not open ({self.why})" if self.why else "not open"
        parts = []
        for name, label in (("wdef", "W.DEF"), ("level", "level"), ("hp", "HP"), ("mp", "MP"), ("attack", "ATTACK"),
                            ("acc", "ACC"), ("eva", "AVOID"), ("magic", "MAGIC"), ("mdef", "M.DEF"), ("str", "STR"),
                            ("dex", "DEX"), ("int", "INT"), ("luk", "LUK"), ("speed", "SPEED"), ("jump", "JUMP")):
            v = self.values.get(name)
            if v is None:
                continue
            if isinstance(v, tuple):
                v = ("~" if name == "attack" else "/").join("?" if x is None else str(x) for x in v)
            parts.append(f"{label} {v}")
        where = ", ".join(f"{n} at {tuple(a)}" for n, a in self.at.items() if a is not None)
        return (", ".join(parts) or "nothing read") + f" ({where})" + (f"; {self.why}" if self.why else "")

    def record(self) -> dict:
        return {"at": {k: (None if v is None else list(v)) for k, v in self.at.items()},
                "values": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.values.items()},
                "texts": dict(self.texts), "why": self.why}


def _columns(cells: dict) -> list[dict]:
    """A panel's value cells in columns (cells whose x spans overlap: the copy's panels have one each; a live window
    cut whole, both panels side by side, two), each read in one OCR call."""
    cols: list[list] = []
    for name, c in sorted(cells.items(), key=lambda kv: kv[1][0]):
        for col in cols:
            if c[0] < col[1] and col[0] < c[2]:
                col[0], col[1] = min(col[0], c[0]), max(col[1], c[2])
                col[2][name] = c
                break
        else:
            cols.append([c[0], c[2], {name: c}])
    return [col[2] for col in cols]


def _read_cells(frame: np.ndarray, at: tuple[int, int], t: Template, ocr) -> dict[str, str]:
    """A panel's value cells read: each value column (`_columns`: the cells' span) in one OCR call, each row taken for
    the cell its centre falls in (rows of one cell joined left to right). `ocr(crop RGB, box (x0, y0, x1, y1) on the
    frame)` -> rows in the crop's px (Row: text, x, y)."""
    out: dict[str, list] = {}
    for cells in _columns(t.cells):
        x0 = min(c[0] for c in cells.values())
        y0 = min(c[1] for c in cells.values())
        x1 = max(c[2] for c in cells.values())
        y1 = max(c[3] for c in cells.values())
        box = (at[0] + x0, at[1] + y0, at[0] + x1, at[1] + y1)
        crop = np.ascontiguousarray(frame[box[1]:box[3], box[0]:box[2], :3])
        for r in ocr(crop, box) or []:
            cx, cy = x0 + float(r.x), y0 + float(r.y)
            for name, (a, b, c, d) in cells.items():
                if b - 1 <= cy < d + 1 and a - 4 <= cx < c + 4:
                    out.setdefault(name, []).append(r)
                    break
    return {name: " ".join(r.text for r in sorted(rs, key=lambda r: r.x)).strip() for name, rs in out.items()}


def read(frame: np.ndarray, ocr=None, cal: "Calibration | None" = None, near: dict | None = None) -> StatReading:
    """The stat window off `frame` (full size, RGB): each panel found (`find_all`), its value cells read by `ocr`
    (crop RGB, its box on the frame -> rows: `_shared_rows`, maplehelper.ocr's, by default) and parsed. Not open: no
    panel there (`why` says how well the best place fitted)."""
    found = find_all(frame, cal, near)
    reading = StatReading(at={k: v[0] for k, v in found.items()}, fits={k: round(v[1], 1) for k, v in found.items()})
    if not reading.open:
        best = min((v[1] for v in found.values()), default=float("inf"))
        reading.why = f"no panel of the stat window on the frame (the best fits {best:.0f})"
        return reading
    for name, t in templates(cal).items():
        at = reading.at.get(name)
        if at is None:
            continue
        texts = _read_cells(frame, at, t, ocr or _shared_rows)
        reading.texts.update(texts)
        for k, text in texts.items():
            v = parse(k, text)
            if v is not None:
                reading.values[k] = v
    if reading.wdef is None:
        reading.why = (f"no number in W.DEF's cell ({reading.texts.get('wdef', '')!r})" if not wants_detail(
            reading.at, cal) else "the DETAIL panel isn't open (her W.DEF is on it)")
    return reading


def wants_detail(at: dict, cal: "Calibration | None" = None) -> bool:
    """Whether her W.DEF's cell is on none of the panels found (`at`: name -> place or None): the DETAIL panel is to be
    opened (a live window cut whole may hold it with the rest)."""
    ts = templates(cal)
    return not any(place is not None and "wdef" in ts[name].cells for name, place in at.items() if name in ts)


def can_click(cal: "Calibration | None" = None) -> bool:
    """Whether the DETAIL button's place is known: the game data's (its pictures found it), or one a live calibration
    read ("DETAIL" off the frame)."""
    return cal is None or cal.source == "copy" or cal.detail_button is not None


def detail_button(at: tuple[int, int], cal: "Calibration | None" = None) -> tuple[int, int]:
    """Where to click the DETAIL button (client px) with the Main panel at `at`: its centre (the calibration's place for
    it, if it has one)."""
    if cal is not None and cal.detail_button is not None:
        bx, by = cal.detail_button
        return at[0] + int(bx), at[1] + int(by)
    x, y, w, h = DETAIL_BUTTON
    return at[0] + x + w // 2, at[1] + y + h // 2


# --- what `gamelab stats --frame` reads, as before ---------------------------------------------------------------------

@dataclass(frozen=True)
class WdefReading:
    """W.DEF as read off the stat window: `wdef` (None: not read, `why` says why), `text` the OCR's, `at` the Detail
    panel's top left on the frame."""
    wdef: int | None
    text: str = ""
    at: tuple[int, int] | None = None
    why: str = ""


def cell(at: tuple[int, int]) -> tuple[int, int, int, int]:
    """W.DEF's cell on the frame (x0, y0, x1, y1), the Detail panel at `at`."""
    c = CELLS["detail"]["wdef"]
    return at[0] + c[0], at[1] + c[1], at[0] + c[2], at[1] + c[3]


def _ocr(bgr: np.ndarray) -> str:
    from .messages import _shared_ocr
    return _shared_ocr(bgr)


def read_wdef(frame: np.ndarray, ocr=None) -> WdefReading:
    """Her W.DEF off `frame` (full size, RGB) if the stat window's Detail panel is open on it: its cell read by `ocr`
    (a BGR crop -> text; None: maplehelper.ocr's shared engine), enlarged x2 as the message bar's is, the first whole
    number taken."""
    at, fit = find(frame, "detail")
    if at is None:
        return WdefReading(None, why=f"no stat window's Detail panel on the frame (its picture fits {fit:.0f} at best)")
    x0, y0, x1, y1 = cell(at)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1, 2::-1])
    big = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
    text = (ocr or _ocr)(big)
    m = NUMBER.search(text or "")
    if m is None:
        return WdefReading(None, text or "", at, why=f"no number in W.DEF's cell ({text!r})")
    return WdefReading(int(m.group(0)), text, at)


# --- the calibration: what a live capture found ------------------------------------------------------------------------

@dataclass
class Calibration:
    """What a live capture of the stat window found (`gamelab stats --read --record`, `stats --calibrate`), kept in
    lab/stat_window.json (paths.STAT_WINDOW): whose pictures found it (`source`: "copy", the game data's; "live", cut
    from the frame, PNGs in `folder`), each panel's place (and, for live ones, its value cells), whether the Detail
    panel opened with the window (no click needed), the DETAIL button's place in the Main panel (None: the copy's),
    the client's size, when, the stats key it was opened with and the frames it came from. A hunt opens the window only
    with one (behavior.statcheck)."""
    source: str = "copy"
    panels: dict = field(default_factory=dict)  # name -> {"at": [x, y], "size": [w, h], "cells": {field: box}}
    detail_with_main: bool = False
    detail_button: tuple[int, int] | None = None
    client: tuple[int, int] = FRAME
    at: float = 0.0
    key: str = KEY
    frames: str = ""
    why: str = ""

    @property
    def folder(self) -> Path:
        return paths.STAT_TEMPLATES

    def place(self, name: str) -> tuple[int, int] | None:
        at = (self.panels.get(name) or {}).get("at")
        return None if at is None else (int(at[0]), int(at[1]))

    def words(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.at)) if self.at else "?"
        where = ", ".join(f"{n} at {tuple(p['at'])}" for n, p in self.panels.items() if p.get("at") is not None)
        return (f"the stat window's reader calibrated {when} on a {self.client[0]}x{self.client[1]} client ("
                f"{'the game data' if self.source == 'copy' else 'cut from the live frame'}'s pictures; {where}; "
                f"DETAIL {'opens with it' if self.detail_with_main else 'by a click'})")

    def save(self, path: Path | None = None) -> Path:
        path = path or paths.STAT_WINDOW
        data = asdict(self)
        data["client"] = list(self.client)
        data["detail_button"] = None if self.detail_button is None else list(self.detail_button)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration | None":
        """The calibration kept (None: none, or a file not as it's written)."""
        path = path or paths.STAT_WINDOW
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("panels"), dict) or not data["panels"]:
                return None
            data["client"] = tuple(data.get("client") or FRAME)
            if data.get("detail_button") is not None:
                data["detail_button"] = tuple(data["detail_button"])
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            return None


def _letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", (text or "").upper())


def _label_of(text: str) -> tuple[str | None, str]:
    """Which field an OCR'd row names at its start (LABELS, the longest match first) and the rest of it (a value read
    on the same row), or (None, text)."""
    up = (text or "").upper()
    best = None
    for name, words in LABELS.items():
        for w in words:
            pattern = r"^\s*" + r"[\s.]*".join(map(re.escape, w)) + r"\.?"
            m = re.match(pattern, up)
            if m is not None and (best is None or len(w) > best[2]):
                best = (name, up[m.end():], len(w))
    if best is None:
        return None, text
    rest = (text or "")[len(text) - len(best[1]):] if best[1] else ""
    return best[0], rest


def changed_boxes(before: np.ndarray, after: np.ndarray, level: int = 48, least: tuple[int, int] = (100, 100),
                  share: float = 0.4) -> list[tuple[int, int, int, int]]:
    """The rectangles where `after` differs from `before` (a px by more than `level` in a channel), as a window opening
    would make them: at least `least` (w, h) and `share` of their px changed; the largest first (x0, y0, x1, y1)."""
    import cv2
    if before is None or after is None or before.shape != after.shape:
        return []
    d = (np.abs(before[..., :3].astype(np.int16) - after[..., :3].astype(np.int16)).max(axis=2) > level)
    m = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w >= least[0] and h >= least[1] and d[y:y + h, x:x + w].mean() >= share:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return sorted(out, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def layout(crop: np.ndarray, ocr=None, box=None) -> tuple[dict[str, tuple[int, int, int, int]], tuple[int, int] | None]:
    """A panel's value cells found by their labels (`calibrate`: a window the copy's pictures don't match): the OCR's
    rows of `crop` (RGB), each row that starts with a field's label (LABELS) taken as that field's row, its value cell
    from the label's right end (or, a value read on the same row, from where the value starts) to the crop's right edge,
    the row's height and 2 px each way; and the DETAIL button's centre if a row reads "DETAIL". Returns (cells, button);
    cells in the crop's px. `box`: the crop's place on its frame (for the OCR's call; None: its own px).
    A label's row and its value's to the right of it read apart (the copy's: tab and cell) or as one; a row of the
    other panel's labels beside it (a window cut whole) ends the value cell where it starts."""
    box = box or (0, 0, crop.shape[1], crop.shape[0])
    rows = (ocr or _shared_rows)(np.ascontiguousarray(crop[..., :3]), box) or []
    cells: dict[str, tuple[int, int, int, int]] = {}
    button = None
    w = crop.shape[1]
    labels = []  # (row, field, the rest of its text)
    for r in sorted(rows, key=lambda r: r.y):
        if "DETAIL" in _letters(r.text):
            button = (int(round(r.x)), int(round(r.y)))
            continue
        name, rest = _label_of(r.text)
        if name is not None:
            labels.append((r, name, rest))
    for r, name, rest in labels:
        if name in cells:
            continue
        h = max(8.0, float(r.height or 12.0))
        if rest.strip():
            span = max(1.0, float(r.right - r.left))
            start = float(r.left) + span * (len(r.text) - len(rest)) / max(1, len(r.text))
        else:
            start = float(r.right) + 2
        # (to the crop's right edge, or the next label right of it on its row: a window cut whole, the other panel's)
        end = min([float(o.left) - 2 for o, _, _ in labels if o is not r and o.left > r.right
                   and abs(float(o.y) - float(r.y)) < h], default=float(w - 2))
        cells[name] = (int(max(0, start - 2)), int(max(0, r.y - h / 2 - 2)), int(end),
                       int(min(crop.shape[0], r.y + h / 2 + 2)))
    return cells, button


def calibrate(frame: np.ndarray, before: np.ndarray | None = None, ocr=None,
              boxes: list | None = None) -> tuple[Calibration | None, str]:
    """The reader set up from a live frame with the stat window open (`frame`; `before`: one without it, from just
    before): the copy's pictures if they're found on it (source "copy": nothing cut), else each window that opened
    (`boxes`, client px (x0, y0, x1, y1) given; else `changed_boxes` against `before`) cut as a live template, its value
    cells found by their labels (`layout`, the OCR), masked out of it, saved under lab/stat_window/ (source "live").
    The panel with W.DEF is "detail" if another has her level, else "main". Returns the calibration (None: none could
    be made) and what was done, in words. Nothing is saved to lab/stat_window.json here (the caller does)."""
    found = find_all(frame)
    if any(at is not None for at, _ in found.values()):
        cal = Calibration("copy", {n: {"at": list(at), "size": list(_copy_template(n).size)}
                                   for n, (at, _) in found.items() if at is not None}, at=time.time())
        return cal, ("the game data's pictures match: " + ", ".join(
            f"{n} at {tuple(at)} (fit {fit:.1f})" for n, (at, fit) in found.items() if at is not None))
    if boxes is None:
        boxes = changed_boxes(before, frame) if before is not None else []
    if not boxes:
        return None, ("the game data's pictures aren't on the frame (best fit "
                      f"{min(f for _, f in found.values()):.0f}) and no window found to cut"
                      + ("" if before is not None else " (no frame from before it opened: give --before, or --box)"))
    panels, button, said = {}, None, []
    cut: dict[str, np.ndarray] = {}
    for x0, y0, x1, y1 in boxes[:3]:
        crop = np.ascontiguousarray(frame[y0:y1, x0:x1, :3])
        cells, b = layout(crop, ocr, (x0, y0, x1, y1))
        if not cells:
            said.append(f"the window at ({x0}, {y0})-({x1}, {y1}): no label read")
            continue
        name = "detail" if "wdef" in cells and not ({"level", "hp"} & set(cells)) else "main"
        if name in panels:
            name = f"panel{len(panels)}"
        if b is not None and name == "main":
            button = b
        panels[name] = {"at": [x0, y0], "size": [x1 - x0, y1 - y0], "cells": {k: list(v) for k, v in cells.items()}}
        rgba = np.dstack([crop, np.full(crop.shape[:2], 255, np.uint8)])
        for a, b0, c, d in cells.values():
            rgba[b0:d, a:c, 3] = 0
        cut[name] = rgba
        said.append(f"{name} at ({x0}, {y0}) {x1 - x0}x{y1 - y0}: {', '.join(sorted(cells))}")
    if not panels:
        return None, "; ".join(said) or "nothing cut"
    folder = paths.STAT_TEMPLATES
    try:
        from PIL import Image
        folder.mkdir(parents=True, exist_ok=True)
        for name, rgba in cut.items():
            Image.fromarray(rgba).save(folder / f"{name}.png")
    except OSError as e:
        return None, f"the templates couldn't be saved in {folder}: {e}"
    _live_template.cache_clear()
    cal = Calibration("live", panels, detail_with_main="detail" in panels, at=time.time(),
                      detail_button=None if button is None or "main" not in panels else button)
    return cal, "cut from the live frame: " + "; ".join(said)


# --- the window drawn from the pictures (the tests, the simulator) ---------------------------------------------------

DRAW_AT = {"main": (560, 150), "detail": (745, 150)}  # (where `compose` puts them by default: the simulator's)
INK = (68, 68, 68)   # the values' colour as `compose` draws them (the game's isn't seen yet)


def _font(size: int = 11):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def texts_of(values: dict) -> dict[str, str]:
    """The text each field shows, as `compose` draws it: HP and MP "cur / max", ATTACK "lo ~ hi", SPEED and JUMP with %,
    the rest as given."""
    out = {}
    for k, v in values.items():
        if v is None:
            continue
        if isinstance(v, (tuple, list)):
            v = (" ~ " if k == "attack" else " / ").join(str(x) for x in v)
        elif k in ("speed", "jump", "crit"):
            v = f"{v}%"
        out[k] = str(v)
    return out


@dataclass
class Drawn:
    """A frame `compose` drew, and what it wrote where (frame px boxes -> text): a stand-in OCR's answers."""
    frame: np.ndarray
    texts: list = field(default_factory=list)  # [(x0, y0, x1, y1), text]

    def ocr(self, crop: np.ndarray, box) -> list[Row]:
        """The OCR's rows off `crop` (taken from `box` of this frame): the texts drawn inside it, if the crop is this
        frame's px there (a crop of another frame reads nothing)."""
        x0, y0, x1, y1 = box
        if crop.shape[:2] != (y1 - y0, x1 - x0) or not np.array_equal(crop, self.frame[y0:y1, x0:x1, :3]):
            return []
        out = []
        for (a, b, c, d), text in self.texts:
            cx, cy = (a + c) / 2, (b + d) / 2
            if x0 <= cx < x1 and y0 <= cy < y1:
                out.append(Row(text, cx - x0, cy - y0, a - x0, c - x0, d - b))
        return out


def compose(values: dict | None = None, base: np.ndarray | None = None, at: dict | None = None,
            detail: bool = True, font_size: int = 11) -> Drawn:
    """The stat window drawn over `base` (a 1366x768 RGB frame; None: a plain one) from the copy's pictures: the Main
    panel at `at["main"]`, with `detail` the Detail panel at `at["detail"]` (DRAW_AT by default), each value of
    `values` (field -> value: texts_of) drawn from its cell's top left as the game draws from its vector, in a plain
    font (the game's own isn't to hand)."""
    from PIL import Image, ImageDraw
    frame = (np.full((FRAME[1], FRAME[0], 3), 90, np.uint8) if base is None else np.array(base[..., :3], np.uint8))
    places = {**DRAW_AT, **(at or {})}
    texts = texts_of(values or {})
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    drawn = []
    for panel in ("main", "detail") if detail else ("main",):
        x, y = places[panel]
        rgba = picture("/".join(PART[panel]))
        h, w = rgba.shape[:2]
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            continue
        a = rgba[..., 3:] / 255.0
        region = np.asarray(img)[y:y + h, x:x + w, :3].astype(float)
        img.paste(Image.fromarray((rgba[..., :3] * a + region * (1 - a)).astype(np.uint8)), (x, y))
        draw = ImageDraw.Draw(img)
        for name, (c0, r0, c1, r1) in CELLS[panel].items():
            if name not in texts:
                continue
            draw.text((x + c0 + 2, y + r0 + 1), texts[name], fill=INK, font=font)
            drawn.append(((x + c0, y + r0, x + c1, y + r1), texts[name]))
    return Drawn(np.asarray(img).copy(), drawn)
