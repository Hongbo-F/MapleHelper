"""The perception's pass: one frame of the game in, the state of what's on screen out (WorldState), for the feed
(localization.feed) to hand to whoever asks and for `python -m gamelab watch` to publish. (Not the simulator's world,
gamelab.sim.world: this is what the stack sees of a world.)

Every frame the feed looks at (localization.rate.HZ, 15 a second: everything once a frame, the user, 2026-10-30), one
pass (`WorldStep.step`):
1. her look (the same as Senses.look takes: her place by the scene or her name tag and dot, HP and MP, the ground, ropes,
   the other players' dots) and the whole screen's search for monsters (MobDetector.detect_wide, the play area above the
   bottom bar) at once, on two threads (HALVES: the search needs only the frame, and both are mostly C that lets go of
   Python's lock);
2. her place, her EXP gains and HP drops (events, each at the frame it showed: a gain with her EXP number as read then,
   perception.exp_number, the kills its size was going to the tracker by kind, localization.gains) and her touches;
3. the search's detections placed on the map and matched to the tracks (localization.tracking: each one's id, place,
   heading and its kind's speed from the game data, predicted between frames along its platform), each detection with
   its track's id in the state (Monster.track), their HP bars read where due (`_bar_due`), and those lost lately or found
   weakly looked at for their death (DEATHS: perception.deaths, the game data's die1 frames where each was);
4. the drops: the strips asked for (a look's: her platform's, or every platform's in view) on the frame asked for, then
   DROPS_STRIPS strips in view by rotation (`drops`: every strip in view with when it was last searched and what it had).
A consumer asks for drops on the next frame (`Asks`); her band and the whole screen, which a look asked for until
2026-10-30, are in every state.

Until 2026-10-30 the step ran at 30 a second, each monster followed re-found around where it should be in every frame
(MobDetector.refind, its windows in one call in C, near-first and memos, the GPU ahead), and the whole screen searched
on a thread of its own every other frame, its answer merged into the tracker frames after its own and that frame's
state given again (`late`), the tracker's velocities an alpha-beta filter's. Replayed on the crowd's minute at 15 a
second (D:/MapleHelper-scratch/replan/p15: cpu15.py, 900 frames, Idle priority on CPUs 12-23, another workflow's replay
beside it): the whole screen alone took 0.503 cores at the live rate where today's code at 15 took 0.683 and at 30 at
least 0.752, the step 18.9 ms median (p90 28.9, p99 40.2, the halves one after the other); the tracker's outcomes about
the same (57 tracks against 61, died hers 15 against 14, died other 8 against 9, lost in a crowd 7 against 10, flickers
13 and 13, deaths seen 13 and 13, kills bound 6 and 6); of the tracks the re-finds found in a frame 3.5% had no
whole-screen detection over them in it (83 of 2,369; within 50 units of her 6.4%), which the tracks hidden between
sightings bridge (tracking.MISSING_S).

The state holds plain data (no map objects, no frame unless kept in-process), so it can go to another
process; `to_scene` turns it back into the Scene a look of the stack's would have been.
"""

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from ..character import THIEF_SHAPE
from ..perception import deaths, exp_number, hp_number, lv_number, touches
from ..perception.eyes import Observation, band_rows
from ..perception.exp_number import ExpReading
from ..perception.framecheck import Problem
from ..perception.hud import exp_changed
from ..perception.minimap import minimap_box
from ..perception.mobs import Mob, MobDetector, _one_opencv_thread
from ..screen import STRETCHED, Geometry
from .gains import ExpMeter, Gain
from .mapdata import MapModel
from .monsters import Monster
from .pose import Where
from .rate import HZ
from .scene import Loot, Scene, Strip, drop_strips, find_loot
from .senses import Senses
from .tracking import DYING, EDGE_PX, HIDDEN, LIVE, Event, MonsterTracker, Track, View, snapshot

# Ropes are searched every frame: placed by a rope in one frame and by her dot alone in the next, her x (and
# with it every monster's) would step by up to a minimap pixel (11.4 units) from frame to frame, which the
# tracker would take for walking. In the columns where the map's ropes can show it's 8.6 ms of a look's 16.
ROPES_EVERY = 1
# Her look and the whole screen's search at once, the search on a thread of the step's own (`_halves`, one, OpenCV on one
# thread in it) while the frame loop's thread looks at her: on 300 frames of the crowd at 1366x768, 2 rounds alternated
# (D:/MapleHelper-scratch/replan/p15/halves.py, Idle priority on CPUs 12-23), one after the other 15.95 ms (median; p90
# 22.7, p99 32.9), at once 10.30 (15.9, 23.6), for about the same CPU (26.9 ms a frame against 28.4); her look alone (the
# thief's, her dot's) 8.5, the search alone 7.2. Only where the look's frame is the capture itself (the game's own
# 1366x768: at 1920x1080 the look frame is the capture halved, made in the look) and with the real monster detector (the
# simulator's eyes search after placing her, on the fake clock). False: one after the other.
HALVES = True
HP_DROP = 0.004       # HP lost from one frame to the next that counts as a hit (control.navigator.HIT)
# A monster's HP bar (perception.hud.hp_bar, 1.1 ms each) is read in a frame while it shows one (it did within
# BAR_SHOWN_S: a frame or two), or a throw of hers went its way in the last tracking.STAR_WINDOW, or it's new (no track
# explains it); else BAR_EVERY_S after its last read (every third frame at 15 a second), its track's bar as last read
# meanwhile (a monster nobody hits shows none). Every 5th frame at 30 a second until 2026-10-30 (BAR_EVERY: 0.17 s, by
# frames); read in every frame, the bars were 74% of what the simulator's feed cost (loop_tree_lie_detector: 37.5 s of
# its 41 in SyncFeed.tick), and live pays for them as much. One seen again after BAR_UNSEEN_S or more unseen (hidden under
# her damage numbers, 1-3 s in game) has it read in the frame it's back in: its bar as last read is from before what hid
# it, and carried over, the probes read a star that hit as a miss (test_probe's hidden target: 2 of 30 so, with every
# star a hit). Times are compared half a frame short (a frame a little early still counts).
BAR_EVERY_S = 0.2
BAR_SHOWN_S = 0.25
BAR_UNSEEN_S = 0.15
HIT_MERGE_S = 0.2     # a drop this soon after the last is the same hit, if the bar takes a few frames to go down
# Her HP number's `before` (an `hp_read` event's: the HP it said at the change before, a drop from it a touch, exactly) is
# known only over frames all read: a frame not read (not visible, failing its checks) or a stretch over HP_GAP_S between
# two frames read has the number forgotten (hp_number.Follow.see(None)), the next read's `before` None. Two touches can't
# both come within it (her invulnerability after one, perception.touches.INVULNERABLE_S: the frames' shortest gap
# between two 1.435 s); a potion between makes a drop smaller, never larger. Without it a drop over 60 unread frames was
# kept as one touch of their two (the review of 2026-10-28: D:/MapleHelper-scratch/safety3/review-evidence/demo/gap_demo.py).
HP_GAP_S = 1.0
# Whether the frame step reads her touches off every frame (perception.touches.TouchWatch: the violet popup born over
# her name tag, MISS or the damage she took; 0.42 ms a frame at the median, p99 3.3, the sprites' reader 4.7 ms at the
# median and 8 at the most on a touch's first frame alone; README, Touches), each a 'touch' event, and the newest in
# every state (WorldState.touch): what the stack takes for her being touched. An HP drop no popup of hers was told for
# (TOUCH_SAME_S) is one too, its `how` perception.touches.HP: her name tag not found, a window over her head. With real
# eyes, the game's sprites read from the game data as the step is made (0.4 s; without it, MISS by width alone); the
# simulator's eyes have their own (`touches`, sim.fakes.FakeTouches), or none: then her HP drops alone. The 'hp_drop'
# event is told beside it as before. (Its rules were measured on recordings at 15 frames a second: touches.EDGE_FRAMES,
# INVULNERABLE_S known to a frame then.)
TOUCHES = True
# s: a touch told this soon after the last (onset to onset) is the same one: none comes within perception.touches.
# INVULNERABLE_S of the last (the frames' shortest gap 1.435 s), and a popup can show a frame or two after her bar
# moved.
TOUCH_SAME_S = 1.0
HP_FLOOR = 0.02       # an HP drop to this or less isn't a touch by itself (a bar read empty: behavior.hunter's HP_ZERO)
# A MISS told off its popup (in its first frame, by the sprites or its width) whose touch took HP after all: a new HP drop
# within perception.touches.HP_WAIT_S of its popup (none can come from another touch: INVULNERABLE_S) has it told again
# as damage, a 'touch_fix' event with the touch as it is now (`was` MISS; the same `at`, a new `told`), the state's
# newest touch that one: what knocks her (behavior.hunter's knock_at) and cuts a chain (control.navigator). On the
# hunting frames of 2026-10-24..28 one told MISS of the 269 touches had a drop so (her 124 under her own orange numbers,
# MISS by its width: the drop 0.169 s after it, its truth damage), no other (D:/MapleHelper-scratch/touch/fix,
# eval_fix.py). Before the review of 2026-10-28 it stayed a MISS: the drop was the same touch (TOUCH_SAME_S), so her
# knock was never waited out. False: a MISS stays one.
FIX_MISS = True
# A touch told off a popup alone (a MISS, or one unread: no HP of hers moved) while another player's minimap dot is within
# CROWD_DX across and CROWD_DY up or down of her (localization.players) is `crowded` (its data): their popup can be born
# over her, theirs (the review of 2026-10-28: CHANIKR's MISS 34.5 px from her placed tag, her own hidden under his, at
# 2026-10-26 23:10:19, 0.5 px out of perception.touches.EDGE_DX), so nothing goes by its invulnerability (Senses.
# invulnerable_until, the navigator's IMMUNE), while it's still a touch for the rest. Damage is hers by her HP.
CROWD_DX = 60.0  # map units (the edge window's 34 game px and a minimap px either side, 11.4)
CROWD_DY = 60.0
# Whether the frame step looks for the deaths of the monsters it follows (perception.deaths: at most every EVERY_S, a
# track found under REFOUND_UNDER in the frame or lost RECENT_S or less ago, its die1 frames where it was), and tells the
# tracker of each seen (tracking.seen_dying: her kill's monster put right, another player's dead at once). With real
# eyes, the die1 frames from the game data, read as the step is made (0.5 s); the simulator's eyes have their own
# (`deaths`, sim.fakes.FakeDeaths), or none. 0.72 ms a frame on one thread, replayed on the 23:10 recording
# (perception.deaths).
DEATHS = True
# The drops searched in every frame without being asked: this many strips in view (scene.drop_strips: one over each
# stretch of level ground of every platform), by rotation (the one searched longest ago first, her platform's taken as
# DROPS_HER times as old), each strip's last search kept with when it was (`WorldState.drops`): at 12 strips in view
# (Tree Dungeon's crowd: 12, 15 at most) every one is searched again every 0.4 s, 2 x 3.84 ms a frame (0.115 cores at
# 15 a second; every platform in view on every 15th frame, D:/MapleHelper-scratch/replan/p15/drops_cost.py: 46.5 ms a
# frame, p90 51.7, the most 62.9). What a look asks for (Asks: her platform's drops, or every platform's) is searched on
# the frame asked for, whole (a pile just picked up must not be seen again, and an item in two platforms' strips goes to
# the one it lies on: scene.find_loot), before the rotation. Until 2026-10-30 the drops were searched only when asked
# (and on `watch`'s schedule, her platform's every 0.5 s and every platform's every 2 s, while no command hunted).
DROPS_STRIPS = 2
DROPS_HER = 2.0


@dataclass(frozen=True)
class Schedule:
    """What the pass does without being asked."""
    ropes_every: int = ROPES_EVERY       # frames
    drops_strips: int = DROPS_STRIPS     # strips of drops in view a frame, by rotation (0: none)


WATCH = Schedule()  # (`watch`'s, the same as the hunt's since 2026-10-30: the drops' rotation for the mini-map too)


@dataclass
class Asks:
    """What a consumer asks of the next frame shown at `after` or later. Her band and the whole screen (`band`, `wide`)
    are in every frame's state since 2026-10-30: asking for them changes nothing (kept for a client of before)."""
    after: float = 0.0
    band: bool = False        # (her band searched for monsters: every frame's)
    wide: bool = False        # (the whole screen: every frame's)
    loot: bool = False        # the drops on her platform ...
    loot_wide: bool = False   # ... or on every platform in view
    mesos_only: bool = False  # (only mesos: her inventory is full)

    def merge(self, other: "Asks | None") -> "Asks":
        """This ask and a newer one (`other`) in one: everything either wants, from the sooner time, and the
        newer's mesos_only (her inventory may have filled or freed since)."""
        if other is None:
            return self
        return Asks(min(self.after, other.after), self.band or other.band, self.wide or other.wide,
                    self.loot or other.loot, self.loot_wide or other.loot_wide, other.mesos_only)


@dataclass(frozen=True)
class DropStrip:
    """A strip of drops in view (scene.Strip) as last searched: its platform, how much of it (map x from, to), when
    (the frame's time; None: not searched yet), whether for mesos alone, and what it had. Plain data."""
    platform: int
    x0: float
    x1: float
    at: float | None = None
    mesos_only: bool = False
    loot: tuple[Loot, ...] = ()


@dataclass
class WorldState:
    """What one frame showed, placed on the map, and the tracks and events it brought. Plain data."""
    seq: int                       # the step's frame number
    t: float                       # when the screen showed it (time.monotonic's clock)
    visible: bool                  # False: the game wasn't wholly on screen; nothing was read (see `why`)
    why: str = ""
    # Why nothing may be acted on from this frame (perception.framecheck): the game not wholly on screen (with `visible`
    # False: its window's problem), or a frame that failed its checks (the minimap window or the HUD not where they
    # should be, the capture offset, the map's minimap picture or her name tag not found), or her place off the map
    # (localization.placecheck). Nothing but the bars is read from it; None: it's of the game as expected.
    bad: Problem | None = None
    tag: tuple[int, int] | None = None   # Observation's and Scene's fields
    tag_seen: bool = True
    hp: float = 0.0
    mp: float = 0.0
    ground: tuple[int, int] | None = None
    dot: tuple[float, float] | None = None
    others: list = field(default_factory=list)   # other players' minimap dots (perception.minimap.Marker)
    players: list = field(default_factory=list)  # ... the players they are (localization.players.Player)
    ropes: list = field(default_factory=list)
    exp: bytes = b""               # her EXP number's text pixels, packed (np.packbits) ...
    exp_shape: tuple = ()          # ... and their shape
    where: tuple | None = None     # her map position: x, y, platform id (None: none), dot
    how: str = "dot"
    feet: tuple | None = None      # by the scene (localization.locate; Scene's): her feet on screen, the view (the map
    view: tuple | None = None      # point at the frame's top-left), whether it was registered on this frame, her mode
    view_sure: bool = False        # ('ground', 'air', 'rope') and how sure of her place; None: placed as before
    mode: str | None = None
    confidence: float | None = None
    monsters: list[Monster] = field(default_factory=list)  # every monster the whole screen found in this frame, placed,
                                                           # each with its track's id (Monster.track)
    band_at: float | None = None   # the frame time of the last search of her band and of the whole screen: this
    wide_at: float | None = None   # frame's own, every frame placed (kept for a client of before)
    loot: list[Loot] | None = None  # the drops asked for, if this frame was searched for them (None: it wasn't) ...
    looked: list = field(default_factory=list)  # ... where (Scene.looked)
    loot_wide: bool = False        # ... on every platform in view (else on hers)
    mesos_only: bool = False
    drops: list[DropStrip] = field(default_factory=list)  # every strip of drops in view, as last searched (DROPS_STRIPS)
    tracks: list[Track] = field(default_factory=list)  # the monsters followed, alive (seen or not) or just dead
    events: list[Event] = field(default_factory=list)  # what happened, as of this frame
    # what this frame's pass did and took (the real clock): `s` all of it, `look s` her look, `wide s` the whole
    # screen's search, `detections`, `matched` (to a track there was), `bars` read, `deaths` looked at (`deaths_s`),
    # `strips` of drops searched (`asked strips` of them for an ask)
    work: dict = field(default_factory=dict)
    frame: np.ndarray | None = None  # the look's frame (in-process; the feed's child puts it in its ring)
    slot: int | None = None          # ... the ring's slot it's in
    late: bool = False             # (always False since 2026-10-30: a frame's state was given again once its searches
                                   # came back; kept for a client of before)
    geo: Geometry = STRETCHED      # the frame's geometry (Observation.geo)

    epoch: int = 0                 # the perception service's run it's of (localization.service): its ring's frames
    hp_of: tuple | None = None     # her HP number as read at its last change, if it read: (HP, max HP); None: not
                                   # known (perception.hp_number: read at every change of its pixels, an `hp_read`
                                   # event each)
    level_of: int | None = None    # her level as the bottom bar's "LV." digits said at their last change, if they read
                                   # (perception.lv_number: read at every change of their pixels, an `lv_read` event
                                   # each); None: not known
    # The newest touch of hers told as of this frame (perception.touches.Touch.data(), and `told`: the frame it was told
    # in, which a number's is up to touches.HP_WAIT_S after its popup's `at`); None: none yet. Every state has it, so a
    # glance at any state after it knows of it (the 'touch' event is the frame's that told it alone).
    touch: dict | None = None
    # Her chat box as this frame showed it (perception.chatbox.ChatBox: expanded, collapsed or unknown; the look's,
    # Observation.chat); None: not read (the simulator's eyes, a frame not visible)
    chat: object | None = None

    @property
    def placed(self) -> bool:
        return self.visible and self.bad is None and self.tag is not None and self.where is not None

    @property
    def usable(self) -> bool:
        """Whether the stack may act on it: of the game as expected (placed or not: her dot may be hidden)."""
        return self.visible and self.bad is None

    def exp_pixels(self) -> np.ndarray | None:
        if not self.exp_shape:
            return None
        n = int(np.prod(self.exp_shape))
        return np.unpackbits(np.frombuffer(self.exp, np.uint8), count=n).reshape(self.exp_shape).astype(bool)


# --- the pass ----------------------------------------------------------------------------------------------

class WorldStep:
    """Frame after frame of the game, turned into WorldStates, one pass a frame (see the top). `senses`: placing her (its
    eyes look at the frames, its map places what they see); `throws`: whether her throws will be told (`throw`, for
    whose kills are whose: localization.tracking); `schedule`: what it does unasked (the drops' rotation)."""

    def __init__(self, senses: Senses, throws: bool = True, schedule: Schedule = Schedule()):
        self.senses, self.eyes, self.model = senses, senses.eyes, senses.model
        real = isinstance(getattr(self.eyes, "detector", None), MobDetector)
        self._real = real
        # the whole screen's search beside her look (HALVES): one thread of its own
        self._halves = ThreadPoolExecutor(1, "wide", _one_opencv_thread) if real and HALVES else None
        self.tracker = MonsterTracker(self.model, throws, getattr(senses, "shape", THIEF_SHAPE))  # (her Shape)
        self._watch_of: tuple[float, deaths.DeathWatch | None] | None = None  # (its look frame's scale, the watch)
        if real and DEATHS and (geo := getattr(self.eyes, "geo", None)) is not None:
            self._watch_of = (geo.frame_scale, self._death_watch(geo.frame_scale))  # (made now: 0.5 s, not mid-hunt)
        self.schedule = schedule
        self.seq = 0
        self.owed: Asks | None = None    # drops asked for and not searched yet (no frame shown at their time yet)
        self.mesos_only = False          # whether the drops are looked for as mesos alone (the last ask's)
        self._strips: dict[tuple[int, float], DropStrip] = {}  # each strip of drops' last search (Strip.key)
        self._bar_read: dict[int, float] = {}   # when each track's bar was last read (_bar_due)
        self._death_at: dict[int, float] = {}   # ... and when it was last looked at for its death (deaths.EVERY_S)
        self._exp: np.ndarray | None = None
        self.meter = ExpMeter(self.model.mob_exp)  # what her EXP gains were, for the tracker
        self.gains: deque[Gain] = deque(maxlen=64)  # ... the newest of them, as told (for watch's 'exp' records)
        self._hp: float | None = None
        self._hit_at = -1e9
        self.hp_number = hp_number.Follow()  # her HP number, read whenever its pixels change (_vitals) ...
        self._hp_number_t: float | None = None  # ... and the last frame it was seen on (HP_GAP_S)
        self.lv_number = lv_number.Follow()  # her level's digits, read whenever their pixels change (_vitals)
        if real and (geo := getattr(self.eyes, "geo", None)) is not None:
            hp_number.warm(geo)  # (its cells made now: 190 ms, not at the first frame)
            lv_number.warm(geo)  # (... and theirs: 600 ms)
        self.touch: dict | None = None  # the newest touch told (WorldState.touch) ...
        self._touch_at = -1e9           # ... and its popup's onset (its `at`)
        self._touch_of: tuple[float, touches.TouchWatch] | None = None  # (its look frame's scale, the watch)
        if real and TOUCHES and (geo := getattr(self.eyes, "geo", None)) is not None:
            self._touch_of = (geo.frame_scale, self._new_touch_watch(geo.frame_scale))  # (made now, not mid-hunt)
        self.counts = {"frames": 0, "not visible": 0, "placed": 0, "wide": 0, "detections": 0, "matched": 0,
                       "bars": 0, "loot": 0, "loot wide": 0, "strips": 0, "exp gains": 0, "exp read": 0, "hits": 0,
                       "touches": 0, "misses": 0, "touches by HP": 0}

    def throw(self, t: float, side: str, attack: str | None = None) -> None:
        """She threw toward `side` at t (with `attack`: its id, if told)."""
        self.tracker.throw(t, side, attack)

    def step(self, look: Callable[[bool], Observation], t: float, visible: bool = True, asks: Asks | None = None,
             why: str = "", bad: Problem | None = None) -> WorldState:
        """The frame shown at t, one pass (see the top). `look(ropes)` reads it (Eyes.look_at, or the simulator's
        FakeEyes.observe; the feed's Look has the frame's picture, `full`, for the search beside it), searching for ropes
        or not; with `visible` False nothing is read (`why`, and `bad`, the window's problem, say why). A frame that fails
        its checks has its Problem in the state (`bad`). `asks`: what was asked of it (the drops: searched on it)."""
        began = time.perf_counter()
        self.seq += 1
        self.counts["frames"] += 1
        if not visible:
            self.counts["not visible"] += 1
            self.owed = None  # whoever asked takes the look itself
            self.hp_number.see(None)  # (what her HP number did meanwhile isn't known: HP_GAP_S)
            return WorldState(self.seq, t, False, why=why, bad=bad or Problem("window", why or "not visible"),
                              touch=self.touch)
        if asks is not None and (asks.loot or asks.loot_wide):
            self.owed = asks if self.owed is None else self.owed.merge(asks)
        work: dict = {}
        ahead = self._search_ahead(look)  # (the whole screen on the frame alone, beside her look: HALVES)
        obs = look(self.seq % self.schedule.ropes_every == 0)
        work["look s"] = time.perf_counter() - began
        s = self.senses.place(obs)
        events = self._vitals(s, t) if s.bad is None else []  # (a frame that failed its checks: its EXP and HP aren't
                                                               # the game's, and the next good frame's are compared)
        if s.bad is not None:
            self.hp_number.see(None)  # (her HP number's isn't known across it: HP_GAP_S)
        placed = s.tag is not None and s.where is not None
        mobs = self._wide(s, ahead, placed, work)
        view = self._view(s) if placed else None
        found = [s.place(m, self.model, None) for m in mobs]
        her = (s.where.x, s.where.y) if placed else None
        unread: set[int] = set()  # the tracks matched whose bars weren't read (_bar_due)
        if placed:
            self.counts["placed"] += 1
        made = self.tracker.counts["tracks"]
        events += self.tracker.update(t, found, view, her, s.players,
                                      self._reader(s, t, unread, work) if placed else None,
                                      self._dying(s, t, work) if placed and DEATHS else None)
        seen: list[Monster] = []
        for tid, m in self.tracker.matched.items():
            self._bar_read.setdefault(tid, t)  # (a new one's bar was read)
            if tid in unread and (tr := self.tracker.tracks.get(tid)) is not None:
                m = replace(m, bar=tr.bar)  # (its bar as last read)
            seen.append(replace(m, track=tid))
        self._forget()
        work["detections"] = len(found)
        work["matched"] = len(self.tracker.matched) - (self.tracker.counts["tracks"] - made)
        self.counts["detections"] += len(found)
        self.counts["matched"] += work["matched"]
        loot, looked, loot_wide, drops = self._drops(s, t, placed, work)
        exp = np.zeros((0,), bool) if s.exp is None else np.asarray(s.exp, bool)
        where = None if s.where is None else (
            s.where.x, s.where.y, None if s.where.platform is None else s.where.platform.id, s.where.dot)
        work["s"] = time.perf_counter() - began  # (the real clock: what the pass took)
        return WorldState(
            self.seq, t, True, tag=s.tag, tag_seen=s.tag_seen, hp=s.hp, mp=s.mp, ground=s.ground, dot=s.dot,
            others=list(s.others), players=list(s.players), ropes=list(s.ropes),
            exp=np.packbits(exp).tobytes() if s.exp is not None else b"",
            exp_shape=exp.shape if s.exp is not None else (), where=where, how=s.how, feet=s.feet, view=s.view,
            view_sure=s.view_sure, mode=s.mode, confidence=s.confidence, monsters=seen,
            band_at=t if placed else None, wide_at=t if placed else None, loot=loot, looked=looked,
            loot_wide=loot_wide, mesos_only=self.mesos_only, drops=drops,
            tracks=snapshot(list(self.tracker.tracks.values())), events=events, work=work, frame=s.frame, geo=s.geo,
            bad=s.bad, hp_of=self.hp_number.of if s.bad is None else None,
            level_of=self.lv_number.level if s.bad is None else None, touch=self.touch, chat=getattr(s, "chat", None))

    def close(self) -> None:
        if self._halves is not None:
            self._halves.shutdown(wait=False)

    # --- the monsters --------------------------------------------------------------------------------------

    def _search_ahead(self, look) -> Future | None:
        """The whole screen searched on the frame alone on the halves' thread (HALVES), begun now, before her look:
        where the look has its frame's picture (`full`, the feed's Look) and that's the look frame itself (the game's own
        size), with the real monster detector. Its answer: (the monsters, s it took). None: searched after her look."""
        if self._halves is None:
            return None
        full, geo = getattr(look, "full", None), getattr(self.eyes, "geo", None)
        if full is None or geo is None or geo.look_scale != 1 or (full.shape[1], full.shape[0]) != geo.client:
            return None
        rows, detector = (0, int(full.shape[0] * geo.hud_top)), self.eyes.detector

        def search():
            began = time.perf_counter()
            return detector.detect_wide(full, rows), time.perf_counter() - began
        return self._halves.submit(search)

    def _wide(self, s: Scene, ahead: Future | None, placed: bool, work: dict) -> list[Mob]:
        """The whole screen's monsters in look `s`'s frame: the search begun beside her look (its answer waited for,
        whatever it is: it holds the frame's picture), or searched now; none where she isn't placed (nothing to place
        them by)."""
        if ahead is not None:
            mobs, took = ahead.result()  # (a search that failed raises here, in the frame loop)
        elif placed:
            began = time.perf_counter()
            mobs = self.eyes.monsters_wide(s)
            took = time.perf_counter() - began
        else:
            return []
        work["wide s"] = took
        if not placed:
            return []
        self.counts["wide"] += 1
        return mobs

    def _reader(self, s: Scene, t: float, unread: set[int], work: dict) -> Callable[[Monster, Track | None], Monster]:
        """How the tracker takes a detection of look `s` (its `read`): with its HP bar read if that's due for the track
        it is (`_bar_due`; a new one's always), else none (`unread` gets the track's id: its bar as last read goes in
        the state)."""
        work["bars"] = 0

        def read(m: Monster, tr: Track | None) -> Monster:
            if tr is not None and not self._bar_due(tr, t):
                unread.add(tr.id)
                return m
            work["bars"] += 1
            self.counts["bars"] += 1
            if tr is not None:
                self._bar_read[tr.id] = t
            return replace(m, bar=self.eyes.bar(s, m.mob))
        return read

    def _dying(self, s: Scene, t: float, work: dict) -> Callable[[dict[int, Monster]], list[int]]:
        """The tracker's `dying`: the tracks that may be dying looked at for it in the frame of look `s` (shown at t;
        perception.deaths: found under REFOUND_UNDER in it, `matched`, or lost RECENT_S or less ago, each at most every
        deaths.EVERY_S, its die1 frames where its feet are, or were when last seen); those seen dying. `work` gets how
        many were looked at and how long it took (real clock)."""
        def dying(matched: dict[int, Monster]) -> list[int]:
            watch = self._watch(s)
            if watch is None:
                return []
            began = time.perf_counter()
            asks = []
            for tr in self.tracker.live():
                if tr.state not in (LIVE, HIDDEN, DYING) or tr.died_seen is not None or tr.kind not in watch.pictures:
                    continue
                if t - self._death_at.get(tr.id, -1e18) < deaths.EVERY_S - 1e-9:
                    continue
                m = matched.get(tr.id)
                score = None if m is None else m.mob.score
                if not deaths.wants(score, t - tr.last_seen):
                    continue
                if m is not None:
                    fx, fy = m.mob.foot_x, m.mob.feet_y
                else:  # where its feet were, the camera's moves since taken out
                    fx, fy = s.screen_x(tr.seen_x if tr.seen_x is not None else tr.x), s.screen_y(tr.y)
                self._death_at[tr.id] = t
                asks.append((tr.id, tr.kind, fx, fy, score))
            if not asks:
                return []
            out = [a[0] for a in asks if watch.dying(s, a[1], a[2], a[3], a[4])]
            work["deaths"], work["deaths_s"] = len(asks), time.perf_counter() - began
            self.counts["death looks"] = self.counts.get("death looks", 0) + len(asks)
            return out
        return dying

    def _forget(self) -> None:
        """What's kept per track, for the tracks there are."""
        for kept in (self._bar_read, self._death_at):
            for tid in [tid for tid in kept if tid not in self.tracker.tracks]:
                del kept[tid]

    # --- the drops --------------------------------------------------------------------------------------------

    def _drops(self, s: Scene, t: float, placed: bool, work: dict):
        """The drops in look `s`'s frame (shown at t): what was asked for (`owed`, for a frame shown at its time or
        later: her platform's strips, or every platform's in view, searched whole: scene.find_loot), then the
        schedule's DROPS_STRIPS strips by rotation. (loot asked for or None, where it looked, on every platform,
        every strip in view as last searched)."""
        work["strips"] = work["asked strips"] = 0
        if not placed:
            return None, [], False, []
        inview = drop_strips(s, self.model, self.eyes, wide=True)
        loot, looked, wide, done = None, [], False, set()
        asked = self.owed if self.owed is not None and t >= self.owed.after - 1e-9 else None
        if asked is not None:
            self.owed, self.mesos_only, wide = None, asked.mesos_only, asked.loot_wide
            mine = s.platform.id if s.platform is not None else None
            strips = [st for st in inview if wide or st.platform.id == mine]
            before = len(s.looked)
            loot = find_loot(s, self.model, self.eyes, wide, self.mesos_only, strips)
            looked = list(s.looked[before:])
            self._searched(strips, loot, t)
            done = {st.key for st in strips}
            work["asked strips"] = len(strips)
            self.counts["loot"] += 1
            self.counts["loot wide"] += wide
        if self.schedule.drops_strips > 0:
            mine = s.platform.id if s.platform is not None else None

            def stale(st: Strip) -> float:  # (how long ago it was searched, her platform's counted DROPS_HER times)
                last = self._strips.get(st.key)
                age = 1e18 if last is None or last.at is None else t - last.at
                return -age * (DROPS_HER if st.platform.id == mine else 1.0)
            todo = sorted((st for st in inview if st.key not in done), key=lambda st: (stale(st), st.key))
            pick = todo[:self.schedule.drops_strips]
            if pick:
                self._searched(pick, find_loot(s, self.model, self.eyes, False, self.mesos_only, pick), t)
            work["strips"] = len(pick)
        work["strips"] += work["asked strips"]
        self.counts["strips"] += work["strips"]
        drops = [self._strips.get(st.key) or DropStrip(st.platform.id, st.looked[1], st.looked[2]) for st in inview]
        return loot, looked, wide and loot is not None, drops

    def _searched(self, strips: list[Strip], loot: list[Loot], t: float) -> None:
        """Strips searched in the frame shown at t, and what they had (`loot`: each on the strip of its platform that
        holds it), kept as each one's last search."""
        for st in strips:
            _, x0, x1 = st.looked
            mine = tuple(d for d in loot if d.platform == st.platform.id and x0 - 1.0 <= d.x <= x1 + 1.0)
            self._strips[st.key] = DropStrip(st.platform.id, x0, x1, t, self.mesos_only, mine)

    def _vitals(self, s: Scene, t: float) -> list[Event]:
        """Her EXP number changed since the last frame (`exp_gain`, with the number as read: the kills its size was go
        to the tracker, which puts each on a monster), or was read unchanged (`exp_read`: to start from, or to confirm a
        jump: ExpMeter.wants_reading); she was touched (`touch`: her popup, MISS or damage, or an HP drop no popup of
        hers was told for: _touched; `touch_fix`: a MISS told that took HP after all, FIX_MISS); her HP dropped
        (`hp_drop`); her HP number changed and read (`hp_read`: her HP and max HP, and the HP it said at the change
        before if that read too, `before`: a drop between the two is a touch, exactly; None: not known); her level's
        digits changed and read (`lv_read`: her level)."""
        out = []
        changed = exp_changed(self._exp, s.exp)
        if changed or self.meter.wants_reading(t):
            reading = exp_number.read(s.exp_box) if s.exp_box is not None else None
            if changed or (reading is not None and reading.exp is not None):
                out.append(Event("exp_gain" if changed else "exp_read", t, data=reading_data(reading)))
            self.counts["exp gains"] += changed
            self.counts["exp read"] += reading is not None and reading.exp is not None
            self._told(self.meter.update(t, reading, changed))
        self._told(self.meter.due(t))  # changes it couldn't read that no reading has explained: a kill each
        if s.exp is not None:
            self._exp = s.exp
        out += self._touched(s, t)  # (before the drop: a number of hers is told as her bar moves)
        if self._hp is not None and s.hp < self._hp - HP_DROP:
            new = t - self._hit_at > HIT_MERGE_S
            out.append(Event("hp_drop", t, data={"lost": round(self._hp - s.hp, 4), "hp": round(s.hp, 3), "new": new}))
            self._hit_at = t
            self.counts["hits"] += new
            if new and t - self._touch_at > TOUCH_SAME_S and s.hp > HP_FLOOR:  # no popup of hers told for it
                self.counts["touches by HP"] += 1
                out.append(self._tell(touches.Touch(t, touches.DAMAGE, how=touches.HP), t, s))
            elif new and (fix := self._fix_miss(t)) is not None:  # a MISS whose touch took HP after all
                out.append(fix)
        self._hp = s.hp
        if self._hp_number_t is not None and t - self._hp_number_t > HP_GAP_S:
            self.hp_number.see(None)  # (a stretch with no frame: what it did meanwhile isn't known)
        self._hp_number_t = t
        seen = self.hp_number.see(getattr(s, "hp_box", None), s.geo)
        if seen is not None and seen[0].cur is not None:
            out.append(Event("hp_read", t, data={"cur": seen[0].cur, "max": seen[0].max, "before": seen[1]}))
        level = self.lv_number.see(getattr(s, "lv_box", None), s.geo)
        if level is not None and level.level is not None:
            out.append(Event("lv_read", t, data={"level": level.level}))
        return out

    def _touched(self, s: Scene, t: float) -> list[Event]:
        """Her touches this frame tells (the touch watch's: `_touch_watch`), each a 'touch' event; one whose popup came
        within TOUCH_SAME_S of the last told is that one (an HP drop told first, its popup a frame later), not told."""
        watch = self._touch_watch(s)
        if watch is None:
            return []
        out = []
        for touch in watch.update(s.frame, s.tag, s.tag_seen, s.hp, t):
            if touch.t - self._touch_at > TOUCH_SAME_S:
                out.append(self._tell(touch, t, s))
        return out

    def _tell(self, touch: touches.Touch, t: float, s: Scene | None = None) -> Event:
        """A touch of hers told in the frame shown at t (look `s`): its event, and it's the state's newest (`touch`);
        `crowded` where it's a popup's alone with another player on her (CROWD_DX)."""
        data = touch.data()
        if touch.how == touches.POPUP and touch.kind != touches.DAMAGE and s is not None and self._crowded(s):
            data["crowded"] = True
            self.counts["crowded"] = self.counts.get("crowded", 0) + 1
        self.touch, self._touch_at = {**data, "told": round(t, 3)}, touch.t
        self.counts["touches"] += 1
        self.counts["misses"] += touch.kind == touches.MISS
        return Event("touch", t, data=data)

    @staticmethod
    def _crowded(s: Scene) -> bool:
        """Whether another player's dot is on her at look `s` (CROWD_DX, CROWD_DY of her place)."""
        if s.where is None:
            return False
        return any(abs(p.x - s.where.x) <= CROWD_DX and abs(p.y - s.where.y) <= CROWD_DY for p in s.players or ())

    def _fix_miss(self, t: float) -> Event | None:
        """A new HP drop at t: if the newest touch is a MISS told off its popup within perception.touches.HP_WAIT_S
        before (FIX_MISS), it took HP after all: it's damage now (`was` MISS, told at t), its 'touch_fix' event; else
        None."""
        last = self.touch
        if not FIX_MISS or last is None or last.get("kind") != touches.MISS or last.get("how") != touches.POPUP \
                or t - self._touch_at > touches.HP_WAIT_S:
            return None
        data = {**{k: v for k, v in last.items() if k not in ("told", "crowded")}, "kind": touches.DAMAGE, "text": "",
                "was": touches.MISS}
        self.touch = {**data, "told": round(t, 3)}
        self.counts["misses"] -= 1
        self.counts["MISS fixed"] = self.counts.get("MISS fixed", 0) + 1
        return Event("touch_fix", t, data=data)

    def _touch_watch(self, s: Scene):
        """The touch watch for look `s`'s frames: with real eyes, perception.touches.TouchWatch with the game's sprites
        drawn at its frame's scale (made once); else the eyes' own (`touches`), if they have one."""
        if not TOUCHES:
            return None
        if not self._real:
            return getattr(self.eyes, "touches", None)
        scale = s.geo.frame_scale
        if self._touch_of is None or self._touch_of[0] != scale:
            self._touch_of = (scale, self._new_touch_watch(scale))
        return self._touch_of[1]

    @staticmethod
    def _new_touch_watch(scale: float) -> touches.TouchWatch:
        """A touch watch for look frames of `scale`: the sprites' reader from the game data (0.4 s; none without it)."""
        return touches.TouchWatch(touches.Reader.of(scale), scale)

    def _told(self, gains: list[Gain]) -> None:
        """Her kills these gains were, told to the tracker by kind, each at its time (and the gains kept: `gains`)."""
        for g in gains:
            self.gains.append(g)
            for i, kind in enumerate(g.kinds):
                self.tracker.gain(g.when(i), kind)
            if g.odd and g.exp is not None and g.exp > 0:  # a share of a kill: another player's, a monster she hurt
                self.tracker.share(g.t)

    def _watch(self, s: Scene):
        """The death watch for look `s`'s frames: with real eyes, the map's monsters' die1 frames drawn at its frame's
        scale (made once, perception.deaths.DeathWatch.of); else the eyes' own (`deaths`), if they have one."""
        if not self._real:
            return getattr(self.eyes, "deaths", None)
        scale = s.geo.frame_scale
        if self._watch_of is None or self._watch_of[0] != scale:
            self._watch_of = (scale, self._death_watch(scale))
        return self._watch_of[1]

    def _death_watch(self, scale: float):
        """The map's monsters' die1 frames drawn at `scale` (perception.deaths.DeathWatch.of); None without the game
        data. About 0.5 s, once."""
        names = {sp.mob_id: sp.name for sp in getattr(self.model, "spawns", ()) if sp.name}
        try:
            return deaths.DeathWatch.of(names, scale)
        except (ImportError, FileNotFoundError, OSError):  # (no game data here)
            return None

    def _bar_due(self, tr: Track, t: float) -> bool:
        """Whether track `tr`'s HP bar is read in the frame shown at t (BAR_SHOWN_S, BAR_UNSEEN_S, a throw her way,
        BAR_EVERY_S since its last read, half a frame short), judged before the frame's detection is taken for it."""
        if tr.bar_at is not None and t - tr.bar_at <= BAR_SHOWN_S:
            return True
        if tr.state == HIDDEN or t - tr.last_seen >= BAR_UNSEEN_S:  # (seen again)
            return True
        if any(0 <= t - th.t <= self.tracker.shape.window_s and tr.id in th.aimed for th in self.tracker.throws):
            return True
        return t - self._bar_read.get(tr.id, -1e18) >= BAR_EVERY_S - 0.5 / HZ

    @staticmethod
    def _view(s: Scene) -> View:
        h, w = s.frame.shape[:2]
        bottom = int(h * s.geo.hud_top)
        box = minimap_box(s.geo)
        mx, my = box[2] / s.geo.look_scale, box[3] / s.geo.look_scale
        return View(s.map_x(0), s.map_x(w), s.map_y(0), s.map_y(bottom), w, bottom,
                    (s.map_x(0), s.map_y(0), s.map_x(mx), s.map_y(my)), s.geo.fpx(EDGE_PX))


def to_scene(st: WorldState, model: MapModel, band: tuple[float, float], mobs: bool = True, wide: bool = False,
             loot: bool = False, mesos_only: bool = False, frame: np.ndarray | None = None) -> Scene:
    """The Scene a look (Senses.look with these arguments) would have given in the frame of state `st`: her
    place; with `mobs`, the monsters seen in her band (or, if `wide`, all of them); with `loot` (and the
    state searched for drops), the drops on her platform (or, if `wide`, every one in view)."""
    p = None if st.where is None or st.where[2] is None else model.platform(st.where[2])
    where = None if st.where is None else Where(st.where[0], st.where[1], p, st.where[3])
    s = Scene(t=st.t, frame=st.frame if frame is None else frame, tag=st.tag, hp=st.hp, mp=st.mp, ground=st.ground,
              dot=st.dot, others=list(st.others), ropes=list(st.ropes), exp=st.exp_pixels(), where=where, how=st.how,
              tag_seen=st.tag_seen, players=list(st.players), geo=st.geo, tracks=list(st.tracks),
              hp_of=getattr(st, "hp_of", None), level_of=getattr(st, "level_of", None), touch=getattr(st, "touch", None),
              feet=getattr(st, "feet", None), view=getattr(st, "view", None), view_sure=getattr(st, "view_sure", False),
              mode=getattr(st, "mode", None), confidence=getattr(st, "confidence", None),
              chat=getattr(st, "chat", None))
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


def reading_data(reading: ExpReading | None) -> dict:
    """An EXP reading as an event's data (plain, for the pipe): its EXP, percent and the levels it fits; {} unread."""
    if reading is None or reading.exp is None:
        return {}
    return {"exp": reading.exp, "percent": reading.percent, "levels": list(reading.levels)}


def reading_of(data: dict) -> ExpReading | None:
    """The EXP reading an event's data holds (reading_data), or None."""
    return ExpReading(data["exp"], data.get("percent"), tuple(data.get("levels", ()))) if "exp" in data else None
