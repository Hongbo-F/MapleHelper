"""The continuous feed's step: one frame of the game in, the state of what's on screen out (WorldState), for
the feed (localization.feed) to hand to whoever asks and for `python -m gamelab watch` to publish. (Not
the simulator's world, gamelab.sim.world: this is what the stack sees of a world.)

Every frame: her place (the same look as Senses.look takes: her name tag, HP and MP, the ground, her minimap
dot, ropes), her EXP gains and HP drops (events, each at the frame it showed), and each monster tracked
re-found around where it should be (Eyes.refind: the template that matched it last, else its kind's).
Monsters not tracked yet are searched for on a schedule: her band every BAND_EVERY frames, the whole screen
every WIDE_EVERY_S; the drops only when asked (or, for `watch`, on a schedule of their own). A consumer can
ask for any of those on the next frame (`Asks`). The tracker (localization.tracking) keeps each monster's
id, velocity, facing, bar and state, and tells what happened to them.

The state holds plain data (no map objects, no frame unless kept in-process), so it can go to another
process; `to_scene` turns it back into the Scene a look of the stack's would have been.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..perception.eyes import Observation, band_rows
from ..perception.hud import exp_changed
from ..perception.minimap import MINIMAP_BOX
from ..screen import HUD_TOP, SCALE
from .mapdata import MapModel
from .monsters import Monster
from .pose import Where
from .scene import Loot, Scene, find_loot
from .senses import Senses
from .tracking import DYING, HIDDEN, LIVE, Event, MonsterTracker, Track, View, overlap, snapshot

# Ropes are searched every frame: placed by a rope in one frame and by her dot alone in the next, her x (and
# with it every monster's) would step by up to a minimap pixel (11.4 units) from frame to frame, which the
# tracker would take for walking. In the columns where the map's ropes can show it's 8.6 ms of a look's 16.
ROPES_EVERY = 1
BAND_EVERY = 6        # frames between searches of her band for monsters not tracked yet: 5 a second at 30 Hz
WIDE_EVERY_S = 1.0    # s between searches of the whole screen (122 ms each, bench-look)
LOOT_EVERY_S = 0.5    # `watch`: s between searches for the drops on her platform ...
LOOT_WIDE_EVERY_S = 2.0  # ... and on every platform in view
HP_DROP = 0.004       # HP lost from one frame to the next that counts as a hit (control.navigator.HIT)
HIT_MERGE_S = 0.2     # a drop this soon after the last is the same hit, if the bar takes a few frames to go down


@dataclass(frozen=True)
class Schedule:
    """How often the step searches without being asked."""
    ropes_every: int = ROPES_EVERY       # frames
    band_every: int = BAND_EVERY         # frames
    wide_s: float | None = WIDE_EVERY_S  # s (None: only when asked)
    loot_s: float | None = None          # s (None: only when asked)
    loot_wide_s: float | None = None


WATCH = Schedule(loot_s=LOOT_EVERY_S, loot_wide_s=LOOT_WIDE_EVERY_S)  # `watch`: the drops too, for the mini-map


@dataclass
class Asks:
    """What a consumer asks of the next frame shown at `after` or later."""
    after: float = 0.0
    band: bool = False        # her band searched for monsters
    wide: bool = False        # the whole screen
    loot: bool = False        # the drops on her platform ...
    loot_wide: bool = False   # ... or on every platform in view
    mesos_only: bool = False  # (only mesos: her inventory is full)

    def merge(self, other: "Asks | None") -> "Asks":
        if other is None:
            return self
        return Asks(min(self.after, other.after), self.band or other.band, self.wide or other.wide,
                    self.loot or other.loot, self.loot_wide or other.loot_wide, other.mesos_only)


@dataclass
class WorldState:
    """What one frame showed, placed on the map, and the tracks and events it brought. Plain data."""
    seq: int                       # the step's frame number
    t: float                       # when the screen showed it (time.monotonic's clock)
    visible: bool                  # False: the game wasn't wholly on screen; nothing was read (see `why`)
    why: str = ""
    tag: tuple[int, int] | None = None   # Observation's and Scene's fields
    tag_seen: bool = True
    hp: float = 0.0
    mp: float = 0.0
    ground: tuple[int, int] | None = None
    dot: tuple[float, float] | None = None
    ropes: list = field(default_factory=list)
    exp: bytes = b""               # her EXP number's text pixels, packed (np.packbits) ...
    exp_shape: tuple = ()          # ... and their shape
    where: tuple | None = None     # her map position: x, y, platform id (None: none), dot
    how: str = "dot"
    monsters: list[Monster] = field(default_factory=list)  # every monster seen in this frame, placed
    band_at: float | None = None   # the frame time of the last search of her band (or the whole screen) ...
    wide_at: float | None = None   # ... and of the whole screen
    loot: list[Loot] | None = None  # the drops seen, if this frame was searched for them (None: it wasn't) ...
    looked: list = field(default_factory=list)  # ... where (Scene.looked)
    loot_wide: bool = False        # ... on every platform in view (else on hers)
    mesos_only: bool = False
    tracks: list[Track] = field(default_factory=list)  # the monsters followed, alive (seen or not) or just dead
    events: list[Event] = field(default_factory=list)  # what happened, as of this frame
    work: dict = field(default_factory=dict)  # what this frame took: re-finds, searches, loot strips
    frame: np.ndarray | None = None  # the half-size frame (in-process; the feed's child puts it in its ring)
    slot: int | None = None          # ... the ring's slot it's in

    @property
    def placed(self) -> bool:
        return self.visible and self.tag is not None and self.where is not None

    def exp_pixels(self) -> np.ndarray | None:
        if not self.exp_shape:
            return None
        n = int(np.prod(self.exp_shape))
        return np.unpackbits(np.frombuffer(self.exp, np.uint8), count=n).reshape(self.exp_shape).astype(bool)


class WorldStep:
    """Frame after frame of the game, turned into WorldStates. `senses`: placing her and searching for monsters
    (its eyes look at the frames, its map places what they see); `throws`: whether her throws will be told
    (`throw`, for whose kills are whose: localization.tracking)."""

    def __init__(self, senses: Senses, throws: bool = True, schedule: Schedule = Schedule()):
        self.senses, self.eyes, self.model = senses, senses.eyes, senses.model
        self.tracker = MonsterTracker(self.model, throws)
        self.schedule = schedule
        self.seq = 0
        self.band_at: float | None = None
        self.wide_at: float | None = None
        self.loot_at: float | None = None
        self.loot_wide_at: float | None = None
        self.mesos_only = False
        self._exp: np.ndarray | None = None
        self._hp: float | None = None
        self._hit_at = -1e9
        self.counts = {"frames": 0, "not visible": 0, "placed": 0, "re-finds": 0, "re-found": 0, "fallbacks": 0,
                       "band": 0, "wide": 0, "loot": 0, "loot wide": 0, "exp gains": 0, "hits": 0}

    def throw(self, t: float, side: str) -> None:
        """She threw toward `side` at t."""
        self.tracker.throw(t, side)

    def step(self, look: Callable[[bool], Observation], t: float, visible: bool = True, asks: Asks | None = None,
             why: str = "") -> WorldState:
        """The frame shown at t. `look(ropes)` reads it (Eyes.look_at, or the simulator's FakeEyes.observe),
        searching for ropes or not; with `visible` False nothing is read. `asks`: what was asked of it."""
        self.seq += 1
        self.counts["frames"] += 1
        if not visible:
            self.counts["not visible"] += 1
            return WorldState(self.seq, t, False, why=why)
        asks = asks or Asks()
        if asks.loot or asks.loot_wide:
            self.mesos_only = asks.mesos_only
        sch = self.schedule
        obs = look(self.seq % sch.ropes_every == 0)
        s = self.senses.place(obs)
        events = self._vitals(s, t)
        work = {"re-finds": 0, "fallbacks": 0, "band": 0, "wide": 0, "loot strips": 0}
        fallbacks = getattr(self.eyes.detector, "fallbacks", 0)
        refound: dict[int, Monster] = {}
        found: list[Monster] = []
        placed = s.tag is not None and s.where is not None
        if placed:
            self.counts["placed"] += 1
            for tr in self.tracker.live():
                if tr.state not in (LIVE, HIDDEN, DYING):
                    continue
                box = self._box(s, tr, t)
                if box is None:
                    continue
                work["re-finds"] += 1
                m = self.eyes.refind(s, tr.mob, box, fallback=tr.state != HIDDEN)
                if m is not None:
                    refound[tr.id] = s.place(m, self.model, self.eyes.bar(s, m))
            wide = asks.wide or (sch.wide_s is not None and (self.wide_at is None or t - self.wide_at >= sch.wide_s))
            band = asks.band or self.seq % sch.band_every == 0
            if wide or band:
                found = self.senses.search(s, wide=wide)
                self.band_at = t
                work["band" if not wide else "wide"] = 1
                self.counts["wide" if wide else "band"] += 1
                if wide:
                    self.wide_at = t
            work["fallbacks"] = getattr(self.eyes.detector, "fallbacks", 0) - fallbacks
            self.counts["re-finds"] += work["re-finds"]
            self.counts["re-found"] += len(refound)
            self.counts["fallbacks"] += work["fallbacks"]
        her = (s.where.x, s.where.y) if placed else None
        events += self.tracker.update(t, refound, found, self._view(s) if placed else None, her)
        seen: list[Monster] = []
        for m in list(refound.values()) + found:
            if not any(m.kind == k.kind and overlap(m.mob, k.mob) for k in seen):
                seen.append(m)
        loot, wide_loot = None, False
        if placed:
            wide_loot = asks.loot_wide or (sch.loot_wide_s is not None and (
                self.loot_wide_at is None or t - self.loot_wide_at >= sch.loot_wide_s))
            if wide_loot or asks.loot or (sch.loot_s is not None and (self.loot_at is None
                                                                        or t - self.loot_at >= sch.loot_s)):
                loot = find_loot(s, self.model, self.eyes, wide_loot, self.mesos_only)
                work["loot strips"] = len(s.looked)
                self.loot_at = t
                self.counts["loot"] += 1
                if wide_loot:
                    self.loot_wide_at = t
                    self.counts["loot wide"] += 1
        exp = np.zeros((0,), bool) if s.exp is None else np.asarray(s.exp, bool)
        where = None if s.where is None else (
            s.where.x, s.where.y, None if s.where.platform is None else s.where.platform.id, s.where.dot)
        return WorldState(
            self.seq, t, True, tag=s.tag, tag_seen=s.tag_seen, hp=s.hp, mp=s.mp, ground=s.ground, dot=s.dot,
            ropes=list(s.ropes), exp=np.packbits(exp).tobytes() if s.exp is not None else b"",
            exp_shape=exp.shape if s.exp is not None else (), where=where, how=s.how, monsters=seen,
            band_at=self.band_at, wide_at=self.wide_at, loot=loot, looked=list(s.looked) if loot is not None else [],
            loot_wide=wide_loot, mesos_only=self.mesos_only, tracks=snapshot(list(self.tracker.tracks.values())),
            events=events, work=work, frame=s.frame)

    def _vitals(self, s: Scene, t: float) -> list[Event]:
        """Her EXP number changed since the last frame (a kill of hers: the tracker puts it on a monster), or
        her HP dropped (a hit)."""
        out = []
        if exp_changed(self._exp, s.exp):
            out.append(Event("exp_gain", t))
            self.tracker.gain(t)
            self.counts["exp gains"] += 1
        if s.exp is not None:
            self._exp = s.exp
        if self._hp is not None and s.hp < self._hp - HP_DROP:
            new = t - self._hit_at > HIT_MERGE_S
            out.append(Event("hp_drop", t, data={"lost": round(self._hp - s.hp, 4), "hp": round(s.hp, 3), "new": new}))
            self._hit_at = t
            self.counts["hits"] += new
        self._hp = s.hp
        return out

    def _box(self, s: Scene, tr: Track, t: float) -> tuple[int, int, int, int] | None:
        """Where track `tr`'s picture should be in this frame (half-size px), from where it should be on the
        map; None if that's off the play area."""
        m = tr.mob
        cx = s.screen_x(self.tracker.predicted(tr, t)) + (m.x - m.foot_x)
        cy = s.screen_y(tr.y) + (m.y - m.feet_y)
        h, w = s.frame.shape[:2]
        box = (round(cx - m.w / 2), round(cy - m.h / 2), round(cx + m.w / 2), round(cy + m.h / 2))
        if box[2] <= 0 or box[0] >= w or box[3] <= 0 or box[1] >= int(h * HUD_TOP):
            return None
        return box

    @staticmethod
    def _view(s: Scene) -> View:
        h, w = s.frame.shape[:2]
        bottom = int(h * HUD_TOP)
        mx, my = MINIMAP_BOX[2] / SCALE, MINIMAP_BOX[3] / SCALE
        return View(s.map_x(0), s.map_x(w), s.map_y(0), s.map_y(bottom), w, bottom,
                    (s.map_x(0), s.map_y(0), s.map_x(mx), s.map_y(my)))


def to_scene(st: WorldState, model: MapModel, band: tuple[float, float], mobs: bool = True, wide: bool = False,
             loot: bool = False, mesos_only: bool = False, frame: np.ndarray | None = None) -> Scene:
    """The Scene a look (Senses.look with these arguments) would have given in the frame of state `st`: her
    place; with `mobs`, the monsters seen in her band (or, if `wide`, all of them); with `loot` (and the
    state searched for drops), the drops on her platform (or, if `wide`, every one in view)."""
    p = None if st.where is None or st.where[2] is None else model.platform(st.where[2])
    where = None if st.where is None else Where(st.where[0], st.where[1], p, st.where[3])
    s = Scene(t=st.t, frame=st.frame if frame is None else frame, tag=st.tag, hp=st.hp, mp=st.mp, ground=st.ground,
              dot=st.dot, ropes=list(st.ropes), exp=st.exp_pixels(), where=where, how=st.how, tag_seen=st.tag_seen)
    if mobs and s.tag is not None and s.where is not None:
        if wide:
            s.monsters = list(st.monsters)
        else:
            top, bottom = band_rows(s, *band)
            s.monsters = [m for m in st.monsters if top <= m.mob.y - m.mob.h // 2 and m.mob.y + m.mob.h // 2 <= bottom]
        s.searched, s.wide = True, wide
    if loot and st.loot is not None:
        mine = None if p is None else p.id
        keep = (lambda pid: True) if wide else (lambda pid: pid == mine)
        s.loot = [d for d in st.loot if keep(d.platform) and (d.mesos or not mesos_only)]
        s.looked = [lk for lk in st.looked if keep(lk[0])]
    return s
