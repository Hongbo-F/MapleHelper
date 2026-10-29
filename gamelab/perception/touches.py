"""Her touches, off the frame (TouchWatch): a monster that touches her makes the game draw a violet popup over her
head, "MISS" where her evasion took it (no HP lost, no knockback, no blink) or the damage she took (her HP bar drops,
the touch knocks her back and she blinks), and after either she can't be touched for a while (INVULNERABLE_S). As
the user put it on 2026-10-28: "The thief job has high evasion which means a good portion of hit were a MISS. ... A hit
should be indicated by touching the mob, usually the game UI will show a purple-ish MISS or a number with knock back
showing the damage taken. After a MISS or HIT the character is invincible for a short period." Until then a hit was
her HP bar dropping and nothing else (localization.worldstate's hp_drop): every MISS went unseen, half her touches.
The perception service's frame step reads it on every frame (worldstate.WorldStep: the 'touch' event); hud.touch_popup
was its first form (purple px counted in one box above her name tag), which nothing called during a hunt.

What it goes by (the frames of 2026-10-24..28, Tree Dungeon: 28.6 minutes of hunting, 265 touches by eye, and the walk
check's 15.5 + 4.6 minutes, 204; D:/MapleHelper-scratch/touch/meas, README: Touches):
- the colour: the game's NoViolet popups (Effect.wz BasicEff.img NoViolet0, NoViolet1) are filled blue-violet at the
  top (OpenCV hue about 128) to magenta at the bottom, a white rim inside a black one; her own damage on monsters
  (NoRed: orange) and criticals (NoCri: pink-red, no blue-violet part) don't pass (`lines`: CORE and ALL);
- where it's born: centred on her (BIRTH_DX across) with its bottom BIRTH_DY above her name tag's middle (her tag
  placed by the camera where it's hidden, as at a rope's top: BIRTH_DY_PLACED); one born a little off that (EDGE_DX,
  EDGE_DY) is hers once it has lasted EDGE_FRAMES frames. Another player's popup over them, anywhere else, isn't;
- it stays where it was born, rises about 30 px a second and fades from about 0.4 s: a line continuing one seen within
  LOST_S (MATCH_DX across, MATCH_DY and MATCH_VY a second up or down: the camera eases up and down) is the same popup,
  and a popup of hers born within REFRACT_S of one told, at its place, is that one re-found after her own orange
  numbers hid it (no touch comes that soon: INVULNERABLE_S); one born so near a number still waiting for her HP bar
  is that number re-found, unless the sprites read it MISS (the number may be another player's: NEAR_HELD);
- what it says: MISS or the digits, read against the game's own sprites (`Reader`, from the game data copy as the watch
  is made; without it, by width: MISS is one word MISS_W or more wide, a number two or three glyphs);
- a number with no HP change within HP_WAIT_S of it (none of 135 damage touches had none: a drop, or her pet's potion
  at once) is another player's damage, them standing on her; a line unread that lasts under EDGE_FRAMES frames is a
  skill's flash.
By eye on the hunting frames (the build's 272 and the review's one more, a MISS of hers 0.5 s after another player's 40
over her: 273), as ported here: 269 found, none false (the 4 lost: MISSes half under the Lie Detector's window or split
by her own numbers, read as digits), MISS or damage right on 268 of 269 (her 124 under her own orange numbers, MISS by
its width: localization.worldstate tells it again as damage when her bar drops, 'touch_fix'); on the walk check's 204,
202 found and 1 false (another player's 119 at the moment her HP changed; the measured detector's). 0.42 ms a frame
(median; p90 1.45, p99 3.3) on the 1366x768 frame in a box 340x285 around her tag, the reader 4.7 ms at the median and
8 at the most on a touch's first frame alone (D:/MapleHelper-scratch/touch/build/eval_cost1.json; with this rule
D:/MapleHelper-scratch/touch/fix/eval_fix.json); a MISS is told in the frame it shows, damage once her bar moves (a
median 0.04 s later, p90 0.13).

Px here are game px (the 1366x768 window's, a look frame's at that size); a look frame has Geometry.frame_scale of
them a game px (half size of 1920x1080: 0.703), and the watch converts them (`scale`).
"""

from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache

import cv2
import numpy as np

# --- the game's mechanics -------------------------------------------------------------------------------------

# s she can't be touched after a touch, a MISS and damage alike, counted from the frame its popup first shows. Measured
# on the frames (README, Touches): of the 106 gaps between her touches in the 15 fps hunting recordings (2026-10-26
# evening and 10-28's grind) the shortest 1.441 s, the 29 under 1.6 s 1.509 on average, none under 1.43; of the walk
# check's 212, 1.435 and 1.506 (53 under 1.6); after a MISS and after damage the same (1.441 and 1.467 hunting, 1.451
# and 1.435 walking). The sources say 1.5 s, the same after one that misses: meowdb's page on this game's damage taken
# ("Invincible after a hit: 1.5 s", meowdb.com/msclassic/formulas/damage-taken, read 2026-10-28) and the port of the
# decompiled v83 client (RustMS Char.cpp:957-961, invincible.set_for(1500) on every touch that resolves, hit or miss:
# Player.cpp:1256); the HP drops alone had none within 1.25 s of the one before (800 of them, 2026-10-25/26). The
# frames' figure: an onset is known to a frame (0.067 s at 15 fps), and the gaps pile up at 1.43-1.45. Nothing ends it
# early; every touch (a monster's body, a MISS, damage) starts it afresh. While it lasts, no touch is checked at all (no
# roll, no popup, no knock); one still on her touches her again the moment it ends.
INVULNERABLE_S = 1.45

MISS = "MISS"        # what a touch was: her evasion took it ...
DAMAGE = "damage"    # ... it took HP (and knocked her back) ...
UNREAD = "?"         # ... or its popup couldn't be read (it lasted, and her HP didn't say)
POPUP, HP = "popup", "hp"  # how it was seen: its popup, or her HP bar alone (worldstate's fallback: no popup of hers)

# --- the popup ------------------------------------------------------------------------------------------------

CORE = (120, 142, 100, 140)   # hue lo, hi, saturation and value at least (OpenCV HSV): the popup's blue-violet part
ALL = (118, 165, 80, 110)     # ... its whole violet-magenta fill
ROI = (170, 260, 25)          # game px looked at either side of her tag, above it, and below it (a popup that drifts
                              # with the camera stays in)
MERGE_GAP = 14                # game px between two pieces of one line (a number's glyphs)
MIN_CORE = 80                 # game px of blue-violet a line has at least (the fill of a 2-digit number: 120 and up)
MIN_SHARE = 0.25              # ... and of its violet
H_RANGE = (18, 48)            # its height (game px): the digits 32-39, MISS 38
W_MAX = 120                   # its width at most (MISS 98, 3 digits 56-69)
BIRTH_DX = 26                 # a popup of hers is born this near her tag's middle across (game px) ...
BIRTH_DY = (-165, -55)        # ... its bottom this far above it (her tag seen) ...
BIRTH_DY_PLACED = (-210, 90)  # ... or this (her tag hidden, placed by the camera: at a rope's top, under a window)
EDGE_DX = 34                  # one born a little off (this across, its bottom down to EDGE_DY above her tag) is hers
EDGE_DY = -36                 # once it has lasted EDGE_FRAMES frames (the 3 of 115 near her the first rule missed)
EDGE_FRAMES = 3
MATCH_DX = 12                 # a line is a popup seen before if within this across (and 40 px a second more) ...
MATCH_DY = 12                 # ... and this up or down, MATCH_VY a second more (the camera eases vertically)
MATCH_VY = 250
LOST_S = 0.45                 # a popup not seen for this long is gone
REFRACT_S = 1.0               # a popup of hers born this soon after one told, within REFRACT_DX of it: that one,
REFRACT_DX = 45               # re-found after her own numbers hid it
# ... and one born as near a number of hers still waiting for her HP bar (`held`): that number re-found, but for a line
# the sprites read MISS, which no number is (the number may yet be another player's). As first built the newest held was
# a base as one told is, and the review of 2026-10-28 found a MISS of hers lost so (S2 00:51:40.690: another player's 40
# born over her at dx +23, held, judged another's at 41.49; her MISS at 41.189, 92 px wide at dx -3, taken for the 40
# re-found; tests/fixtures/touches other_+5, other_+6). False: a held number is a base whatever the new line reads.
NEAR_HELD = True
MISS_W = 80                   # a line this wide (game px) is MISS, by width alone (its fill 91 wide; 3 digits 56-69)
HP_WAIT_S = 0.8               # s a number of hers waits for her HP bar to move; none, it's another player's
HP_EPS = 0.004                # her HP bar moved (a share of it: worldstate.HP_DROP)
HP_KEEP_S = 0.6               # s of her HP readings kept for the bar before a popup
HOLD_UNREAD = False           # a line the reader couldn't read, as wide as MISS, waits for her HP bar too (else MISS)
OTHERS_KEPT = 64              # popups judged another player's (or a flash) kept, for the record

_SPRITE_BACK = (40, 120, 40)  # the background the reader's sprites are drawn on (none of it passes ALL)


@dataclass
class Touch:
    """One touch of hers: `t` the frame its popup first showed in (the touch came in the frame interval before), what it
    was (`kind`: MISS, DAMAGE or UNREAD) and said (`text`: "MISS", the digits as read, ""), its popup's box (look-frame
    px) and its bottom's middle from her tag's (dx, dy), how long after `t` it was told (`late`: a number waits for her
    HP bar), and how it was seen (`how`: POPUP, or HP for worldstate's fallback, her HP bar alone)."""
    t: float
    kind: str
    text: str = ""
    box: tuple = ()
    dx: float = 0.0
    dy: float = 0.0
    late: float = 0.0
    how: str = POPUP

    def data(self) -> dict:
        """As an event's data (plain, for the pipe)."""
        return {"kind": self.kind, "text": self.text, "at": round(self.t, 3), "late": round(self.late, 3),
                "how": self.how, "box": list(self.box), "dx": self.dx, "dy": self.dy}


@dataclass
class _Pop:
    """A violet line followed from frame to frame (look-frame px)."""
    x: float
    y: float
    w: int
    first: float
    last: float
    hers: bool
    pending: bool = False
    frames: int = 1
    box: tuple = ()
    dx: float = 0.0
    dy: float = 0.0


@lru_cache(maxsize=8)
def _kernel(scale: float) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, round(9 * scale)), max(1, round(5 * scale))))


def violet(hsv: np.ndarray, lo: tuple) -> np.ndarray:
    """The px of an HSV picture within one of the violet ranges (CORE, ALL), 255 where they are."""
    return cv2.inRange(hsv, (lo[0], lo[2], lo[3]), (lo[1], 255, 255))


def lines(rgb: np.ndarray, x0: int = 0, y0: int = 0, scale: float = 1.0) -> list[list[int]]:
    """The violet text lines in an RGB crop whose top-left is (x0, y0) of the frame: [x0, y0, x1, y1, blue-violet px,
    violet px] each (frame px), the pieces of one line (a number's glyphs) put together. `scale`: frame px a game px."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    core = violet(hsv, CORE)  # (within ALL's ranges)
    area = scale * scale
    if cv2.countNonZero(core) < MIN_CORE * area:
        return []
    every = violet(hsv, ALL)
    m = cv2.morphologyEx(every, cv2.MORPH_CLOSE, _kernel(scale))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    nc = np.bincount(lab[core > 0], minlength=n)
    na = np.bincount(lab[every > 0], minlength=n)
    blobs = [[int(st[i][0]) + x0, int(st[i][1]) + y0, int(st[i][0] + st[i][2]) + x0, int(st[i][1] + st[i][3]) + y0,
              int(nc[i]), int(na[i])] for i in range(1, n) if nc[i] >= 15 * area]
    blobs.sort(key=lambda b: b[0])
    gap = MERGE_GAP * scale
    out: list[list[int]] = []
    for b in blobs:
        for o in out:
            ov = min(o[3], b[3]) - max(o[1], b[1])
            if ov >= 0.5 * min(o[3] - o[1], b[3] - b[1]) and b[0] - o[2] <= gap and o[0] - b[2] <= gap:
                o[0], o[1], o[2], o[3] = min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])
                o[4] += b[4]
                o[5] += b[5]
                break
        else:
            out.append(list(b))
    return [o for o in out if o[4] >= MIN_CORE * area and o[4] / max(1, o[5]) >= MIN_SHARE
            and H_RANGE[0] * scale <= o[3] - o[1] <= H_RANGE[1] * scale and o[2] - o[0] <= W_MAX * scale]


# --- what a popup says ----------------------------------------------------------------------------------------

class Reader:
    """What a violet popup says, MISS or the damage's digits, by the game's own sprites (`sprites`: Effect.wz
    BasicEff.img NoViolet0's small digits and Miss, NoViolet1's big ones, the first digit's), drawn as a look frame of
    `scale` px a game px shows them. Masks, not colours: a sprite's fill passing ALL against the frame's."""

    def __init__(self, sprites: dict[str, np.ndarray], scale: float = 1.0):
        self.scale = scale
        self.masks = {k: _mask_of(v, scale) for k, v in sprites.items()}

    @classmethod
    def of(cls, scale: float = 1.0) -> "Reader | None":
        """The reader with the sprites read from the game data copy (0.4 s); None without it."""
        try:
            return cls(sprites(), scale)
        except (ImportError, FileNotFoundError, OSError, KeyError, ValueError):
            return None

    def read(self, rgb: np.ndarray, box: tuple, pad: int = 6) -> tuple[str, str, float]:
        """(kind, text, score) for the popup in `box` (x0, y0, x1, y1) of an RGB picture: MISS, DAMAGE (its digits as
        read, '?' for one that wasn't) or UNREAD."""
        s = self.scale
        x0, y0, x1, y1 = box
        h, w = rgb.shape[:2]
        p = max(1, round(pad * s))
        a0, b0, a1, b1 = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
        m = (violet(cv2.cvtColor(np.ascontiguousarray(rgb[b0:b1, a0:a1]), cv2.COLOR_RGB2HSV), ALL) > 0)
        m = m.astype(np.float32)
        miss, _ = _score(m, self.masks["MISS"])
        # the digits: the mask's own pieces (their white rims part them), each read against both sizes
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        glyphs = sorted((st[i] for i in range(1, n) if st[i][4] >= 40 * s * s and st[i][3] >= 12 * s),
                        key=lambda g: g[0])
        digits, scores = [], []
        tall, wide, margin = 7 * s, 9 * s, max(1, round(3 * s))
        for gx, gy, gw, gh, _ in glyphs:
            g = m[max(0, gy - margin):gy + gh + margin, max(0, gx - margin):gx + gw + margin]
            best = (-1.0, "?")
            for k, t in self.masks.items():
                if k == "MISS" or abs(t.shape[0] - gh) > tall or t.shape[1] > gw + wide or t.shape[1] < gw - wide:
                    continue
                score, _ = _score(g, t)
                if score > best[0]:
                    best = (score, k)
            digits.append(best[1][1:] if best[1] != "?" else "?")
            scores.append(best[0])
        dscore = float(np.mean(scores)) if scores else -1.0
        if miss >= 0.55 and miss >= dscore - 0.05 and (x1 - x0) >= 70 * s:
            return MISS, MISS, round(float(miss), 3)
        if scores and dscore >= 0.45 and 1 <= len(digits) <= 3:
            return DAMAGE, "".join(digits), round(dscore, 3)
        return UNREAD, "".join(digits), round(max(miss, dscore), 3)


def sprites() -> dict[str, np.ndarray]:
    """The popups' sprites from the game data copy (RGBA): 'MISS' (NoViolet0/Miss), 's0'-'s9' (NoViolet0's digits: all
    but a number's first) and 'b0'-'b9' (NoViolet1's: its first)."""
    from gamedata.canvas import pixels, source
    from gamedata.paths import COPY
    from gamedata.wz import Package
    tree = Package(COPY / "Effect" / "Effect_000.wz").image("BasicEff.img")
    small, big = tree["NoViolet0"], tree["NoViolet1"]
    out = {"MISS": pixels(source(small["Miss"], tree))}
    for d in "0123456789":
        out[f"s{d}"] = pixels(source(small[d], tree))
        out[f"b{d}"] = pixels(source(big[d], tree))
    return out


def _mask_of(rgba: np.ndarray, scale: float) -> np.ndarray:
    """A sprite's fill passing ALL, drawn on _SPRITE_BACK (at `scale`: shrunk as a look frame shows it), cut to it."""
    a = rgba[..., 3:4] / 255.0
    rgb = (rgba[..., :3] * a + np.array(_SPRITE_BACK, np.float64) * (1 - a)).astype(np.uint8)
    if scale != 1.0:
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    m = violet(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), ALL) > 0
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)


def _score(m: np.ndarray, t: np.ndarray) -> tuple[float, tuple]:
    if m.shape[0] < t.shape[0] or m.shape[1] < t.shape[1]:
        m = np.pad(m, ((0, max(0, t.shape[0] - m.shape[0] + 2)), (0, max(0, t.shape[1] - m.shape[1] + 2))))
    r = cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return (mx if np.isfinite(mx) else -1.0), loc


# --- the watch --------------------------------------------------------------------------------------------------

@dataclass
class TouchWatch:
    """Her touches, frame after frame (`update`): the popups over her followed, a new one of hers told as a Touch, a
    number once her HP bar has moved. `reader`: what reads a new popup of hers (Reader; None: MISS by width alone);
    `scale`: look-frame px a game px (Geometry.frame_scale)."""
    reader: Reader | None = None
    scale: float = 1.0
    pops: list = field(default_factory=list)
    told: list = field(default_factory=list)      # her popups told within REFRACT_S (_Pop): the bases of re-finds
    held: list = field(default_factory=list)      # numbers of hers waiting for her HP bar: (Touch, HP before, how read,
                                                  # its _Pop)
    hp_hist: list = field(default_factory=list)   # (t, HP) of the last HP_KEEP_S
    others: deque = field(default_factory=lambda: deque(maxlen=OTHERS_KEPT))  # judged another player's, or a flash
    counts: dict = field(default_factory=lambda: {"touches": 0, "misses": 0, "others": 0, "read": 0})

    def update(self, frame: np.ndarray, tag: tuple | None, tag_seen: bool, hp: float | None, t: float) -> list[Touch]:
        """The touches this frame tells (shown at `t`): `frame` the look's (RGB), `tag` her name tag's middle (seen,
        or where the camera puts her: `tag_seen` False), `hp` her HP bar's share."""
        out: list[Touch] = []
        if hp is not None:
            self.hp_hist = [(a, b) for a, b in self.hp_hist if t - a <= HP_KEEP_S] + [(t, hp)]
        still = []
        for touch, before, how, p in self.held:  # numbers waiting for her HP bar (a fall, or a potion's rise at once)
            if before is not None and hp is not None and abs(hp - before) > HP_EPS:  # (one unread took HP: damage)
                out.append(replace(touch, kind=DAMAGE, late=round(t - touch.t, 3)))
                self.told.append(p)
            elif t - touch.t > HP_WAIT_S:
                if how == "reader" or touch.kind == DAMAGE:
                    self.others.append(touch)  # digits and her HP the same: another player's, or a flash
                    self.counts["others"] += 1
                else:  # unread but lasting, and her HP the same: hers, no damage (a MISS half hidden)
                    out.append(replace(touch, late=round(t - touch.t, 3)))
                    self.told.append(p)
            else:
                still.append((touch, before, how, p))
        self.held = still
        self.told = [p for p in self.told if t - p.first <= REFRACT_S]
        if tag is not None and frame is not None:
            self._look(frame, tag, tag_seen, t, out)
        for touch in out:
            self.counts["touches"] += 1
            self.counts["misses"] += touch.kind == MISS
        return out

    def _look(self, frame: np.ndarray, tag: tuple, tag_seen: bool, t: float, out: list[Touch]) -> None:
        s = self.scale
        tx, ty = int(tag[0]), int(tag[1])
        h, w = frame.shape[:2]
        x0, x1 = max(0, tx - round(ROI[0] * s)), min(w, tx + round(ROI[0] * s))
        y0, y1 = max(0, ty - round(ROI[1] * s)), min(h, ty + round(ROI[2] * s))
        if x1 <= x0 or y1 <= y0:
            return
        found = lines(np.ascontiguousarray(frame[y0:y1, x0:x1]), x0, y0, s)
        self.pops = [p for p in self.pops if t - p.last <= LOST_S]
        used: set[int] = set()
        win = BIRTH_DY_PLACED if not tag_seen else BIRTH_DY
        for ln in sorted(found, key=lambda ln: -ln[3]):
            cx, by, lw = (ln[0] + ln[2]) / 2, ln[3], ln[2] - ln[0]
            best = None
            for i, p in enumerate(self.pops):
                if i in used:
                    continue
                dt = t - p.last
                if abs(cx - p.x) > (MATCH_DX + 40 * dt) * s or abs(by - p.y) > (MATCH_DY + MATCH_VY * dt) * s:
                    continue
                d = abs(cx - p.x) + 0.3 * abs(by - p.y)
                if best is None or d < best[0]:
                    best = (d, i)
            if best is not None:  # a popup seen before
                p = self.pops[best[1]]
                used.add(best[1])
                p.x, p.y, p.last = cx, by, t
                if p.frames < 4:
                    p.w = max(p.w, lw)
                p.frames += 1
                if p.pending and p.frames >= EDGE_FRAMES:
                    p.pending = False
                    p.hers = self._born_hers(p, frame, ln, t, p.frames, out)
                continue
            dx, dy = cx - tx, by - ty
            hers = abs(dx) <= BIRTH_DX * s and win[0] * s <= dy <= win[1] * s
            edge = (not hers and abs(dx) <= EDGE_DX * s
                    and win[0] * s <= dy <= (max(win[1], EDGE_DY) if tag_seen else win[1]) * s)
            p = _Pop(cx, by, lw, t, t, False, pending=edge, box=tuple(ln[:4]), dx=round(dx, 1), dy=round(dy, 1))
            self.pops.append(p)
            used.add(len(self.pops) - 1)
            if hers:
                p.hers = self._born_hers(p, frame, ln, t, 1, out)

    @property
    def last_hers(self) -> _Pop | None:
        """The newest popup of hers told within REFRACT_S, or None."""
        return max(self.told, key=lambda p: p.first) if self.told else None

    def _born_hers(self, p: _Pop, frame: np.ndarray, ln: list, t: float, frames: int, out: list[Touch]) -> bool:
        """A new popup of hers (born where hers are, or lasted at the edge): told, or held for her HP bar; False if it's
        one told re-found (REFRACT_S), or a number held re-found (NEAR_HELD: not where it reads MISS)."""
        def near(q: _Pop) -> bool:
            return p.first - q.first <= REFRACT_S and abs(p.x - q.x) <= REFRACT_DX * self.scale
        if any(near(q) for q in self.told):
            return False
        kind, text, how = self._classify(frame, ln, p.w, frames)
        if any(near(q) and not (NEAR_HELD and kind == MISS and how == "reader") for _, _, _, q in self.held):
            return False
        touch = Touch(p.first, kind, text, tuple(ln[:4]), p.dx, p.dy, round(t - p.first, 3))
        if how == "width" and frames < EDGE_FRAMES:
            if kind == MISS and not (HOLD_UNREAD and self.reader is not None):
                out.append(touch)  # a MISS's width is unmistakable
                self.told.append(p)
            else:
                self.held.append((touch, self._hp_before(p.first), how, p))
        elif kind in (DAMAGE, UNREAD):
            self.held.append((touch, self._hp_before(p.first), how, p))
        else:
            out.append(touch)
            self.told.append(p)
        return True

    def _classify(self, frame: np.ndarray, ln: list, w: float, frames: int) -> tuple[str, str, str]:
        """(kind, text, how): by the reader (`how` 'reader'), or by the width (MISS_W) where it can't tell or there's
        none ('width'; an unread line that has lasted EDGE_FRAMES frames UNREAD)."""
        kind, text = (MISS, MISS) if w >= MISS_W * self.scale else (DAMAGE, "")
        if self.reader is None:
            return kind, text, "width"
        a0, b0 = max(0, ln[0] - 6), max(0, ln[1] - 6)
        crop = frame[b0:ln[3] + 6, a0:ln[2] + 6]
        read, said, _ = self.reader.read(crop, (ln[0] - a0, ln[1] - b0, ln[2] - a0, ln[3] - b0))
        self.counts["read"] += 1
        if read != UNREAD:
            return read, said, "reader"
        return (UNREAD if frames >= EDGE_FRAMES else kind), text, "width"

    def _hp_before(self, t0: float) -> float | None:
        """Her HP bar before the frame shown at `t0`: the most of her last 4 readings to then."""
        xs = [hp for t, hp in self.hp_hist if t < t0 + 1e-6]
        return max(xs[-4:]) if xs else None
