"""Monsters followed from frame to frame, for the continuous feed (localization.worldstate): each one keeps
an id while it walks and turns, flinches out of sight under damage numbers, or leaves the screen and comes
back, and what happens to it comes out as events.

A frame's sightings come two ways: monsters re-found where their track says they should be (Eyes.refind,
every frame), and monsters a search found (her band every few frames, the whole screen about once a second),
which are matched to tracks not re-found: same kind, same platform, and where its track would be by now
within 10 + 1.3 x its speed x the time since + 25 units (knockback). Greedy, nearest first: there are only a
few at a time. One no track explains is new: `entered` at the screen's edge (the camera moved, or it walked
in), `spawned` near a spawn point of its kind, else `appeared` (found where no search had looked yet).

A track not seen in a frame:
- where it should be is off screen: `offscreen` (event `left`), remembered OFFSCREEN_S to know it again
  when it comes back (`back`);
- on screen but unseen for MISSING_S: `hidden`. Hit, a monster flinches under its damage numbers and the
  detector loses it for 1-3 s; one back within HIDE_S is the same one;
- hidden for longer than HIDE_S: `dead`, killed by someone else (`died`, cause "other"): no EXP gain of hers
  explains it.

Her EXP gains (worldstate, from the EXP number each frame) are kills of hers. A killed monster is still seen
fading where it died for about a second, so a gain waits that long (FADE_S + MISSING_S) for the monster it was
to go out of sight: one she threw at (a throw toward it in the STAR_WINDOW before; with her throws unknown,
as in `watch`, any monster within a star's reach of her), out of sight by then, the one that went after the
gain first (else one already out of sight when it came: killed while it hid), the least HP left first, then
the nearest her: `dead`, cause "hers". Meanwhile the likeliest one still seen is `dying`. A monster's HP bar
going down is a `bar_hit`, by her when a throw of hers went its way ("other" otherwise; None with her throws
unknown).

A track seen in fewer than MIN_SEEN frames that goes is dropped without a death: a match on scenery for a
frame or two isn't a monster someone killed. Nothing grows without bound: dead tracks go DEAD_KEEP_S after
their death, offscreen ones after OFFSCREEN_S, gains and throws after their windows.
"""

from dataclasses import dataclass, field, replace

from maplehelper.engine import RIGHT

from ..perception.kinds import kind_of
from ..perception.mobs import Mob
from .mapdata import MapModel
from .monsters import DEFAULT_SPEED, MONSTER_HALF, MONSTER_HEIGHT, Monster, speeds

LIVE, HIDDEN, OFFSCREEN, DYING, DEAD = "live", "hidden", "offscreen", "dying", "dead"
STATES = (LIVE, HIDDEN, OFFSCREEN, DYING, DEAD)

MISSING_S = 0.3     # s unseen on screen before a track is `hidden`: the detector misses a monster in a look now and
                    # then (the simulator's game preset: 1 look in 10), which at 30 frames a second is one frame
HIDE_S = 3.5        # s a monster may stay out of sight on screen and still be the same one: hit, it hides under its
                    # damage numbers for 1-3 s (the README); longer, someone killed it
FADE_S = 1.5        # s a killed monster may still be seen fading where it died (about a second, the README; the
                    # simulator's FADE 0.8 s): a gain waits this long (and MISSING_S) for the one it was to go
EXTRAPOLATE_S = 0.5  # s an unseen monster is taken to walk on (a hit one flinches, knocked back, not walking)
MIN_SEEN = 5        # frames a track must have been seen in for its going to count as a death
OFFSCREEN_S = 10.0  # s a track off screen is remembered, to know it again when it comes back
DEAD_KEEP_S = 2.0   # s a dead track is kept (shown as dead) before it goes
MATCH_BASE, MATCH_SPEED, MATCH_KNOCKBACK = 10.0, 1.3, 25.0  # units: a sighting within 10 + 1.3 speed dt + 25
EDGE_PX = 24        # half-size px from the play area's side: a new monster this close came in from off screen
SPAWN_NEAR = 60.0   # units: a new monster this close to a spawn point of its kind (on its platform), away from the
                    # screen's edges, spawned there
STAR_WINDOW = 2.5   # s after a throw's key that a kill or a hit can be its star's (behavior.ledger.STAR_WINDOW)
THROW_REACH = 460.0  # units a star reaches ahead of her (attack.reach: the farthest hit counted on) ...
THROW_DY = 70.0     # ... and above or below her feet (60 measured, and a little slack)
BAR_STEP = 0.03     # a bar this much lower than its lowest seen before: a hit
# The velocity filter (alpha-beta on x, map units). Simulated at 30 Hz on a monster walking 125 units/s with
# 1.5 units of jitter and her dot stepping a minimap pixel (11.4 units) as she walks: 0.35/0.08 gave 117
# units/s after 0.5 s (106 at worst over 20 runs) and wavered 13 units/s rms after that; 0.5/0.3 gave 53
# (16 at worst) and 35 rms, the overshoot ringing against the clamp.
ALPHA, BETA = 0.35, 0.08
SPEED_CLAMP = 1.2   # x its kind's speed: the most the filter may say it walks (knockback is brief)
# A monster seen walking (a move frame) the other way than its velocity says has turned: its velocity turns
# with it, rather than taking the filter 0.5 s. Its facing was the way it moved in 192 of 199 stretches of
# 0.6 s where it walked (`bodies`, two recordings with the camera still; the misses at turns).
TURN_FLIP = True
LONG_GAP_S = 1.0    # a track seen again after longer than this starts its velocity afresh
FRAME_SLACK = 0.05  # s: a monster last seen this little before a gain went out of sight as it died


@dataclass
class Event:
    """Something that happened on screen, at the frame it showed."""
    kind: str               # entered, spawned, appeared, back, left, died, bar_hit; exp_gain, hp_drop (worldstate)
    t: float                # the frame's time (when the screen showed it)
    track: int | None = None
    data: dict = field(default_factory=dict)

    def text(self) -> str:
        bits = ", ".join(f"{k} {v}" for k, v in self.data.items() if v is not None)
        return self.kind + (f" #{self.track}" if self.track is not None else "") + (f": {bits}" if bits else "")


@dataclass
class Track:
    """One monster followed from frame to frame."""
    id: int
    kind: str
    platform: int | None
    x: float                 # map units, its feet (filtered)
    y: float
    mob: Mob                 # its last sighting: its picture's box on screen, the template that matched
    first_seen: float
    last_seen: float
    vx: float = 0.0          # units/s (filtered)
    facing: int = 0          # 1 right, -1 left, 0 not known (the template that matched)
    pose: str = ""
    seen: int = 1            # frames it was seen in
    bar: float | None = None  # its HP bar (fraction left), the lowest seen; None: never seen hurt
    bar_at: float | None = None
    hits: int = 0            # times its bar went down
    state: str = LIVE
    died_at: float | None = None
    cause: str | None = None  # "hers" or "other", once dead
    judged: float | None = None  # when it was found dead
    how: str = ""            # how it was first seen (entered, spawned, appeared)

    def record(self) -> dict:
        """As telemetry data ('tracks')."""
        return {"id": self.id, "kind": self.kind, "state": self.state, "x": round(self.x, 1), "y": round(self.y, 1),
                "vx": round(self.vx, 1), "facing": self.facing, "platform": self.platform,
                "bar": None if self.bar is None else round(self.bar, 2), "hits": self.hits, "seen": self.seen,
                **({"cause": self.cause} if self.cause else {})}


@dataclass(frozen=True)
class View:
    """What a frame shows of the map (map units: x from left to right, y from top to bottom), the play area
    above the bottom bar, and its size on screen (half-size px), to tell a monster out of sight on screen
    from one off it. A box of the map the minimap covers (`covered`) doesn't show either."""
    left: float
    right: float
    top: float
    bottom: float
    width: int                # half-size px
    height: int
    covered: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 (map units)

    def shows(self, kind: str, x: float, y: float) -> bool:
        """Whether a monster of that kind with its feet at (x, y) would show wholly on screen."""
        half, tall = MONSTER_HALF.get(kind, 34.0), MONSTER_HEIGHT.get(kind, 46.0)
        if not (self.left + half <= x <= self.right - half and self.top + tall <= y <= self.bottom):
            return False
        c = self.covered
        return c is None or not (c[0] < x + half and x - half < c[2] and c[1] < y and y - tall < c[3])

    def at_edge(self, mob: Mob) -> bool:
        """Whether a monster's picture is within EDGE_PX of the play area's sides."""
        return (mob.x - mob.w // 2 < EDGE_PX or mob.x + mob.w // 2 > self.width - EDGE_PX
                or mob.y - mob.h // 2 < EDGE_PX or mob.y + mob.h // 2 > self.height - EDGE_PX)


@dataclass
class _Throw:
    t: float
    side: str
    aimed: set[int]  # the tracks a star thrown then could meet


class MonsterTracker:
    """Tracks of the monsters on screen, kept up frame by frame (`update`), and what happened to them.
    `throws`: whether her throws are told (`throw`): then a kill or a hit is hers only after a throw her way;
    without (`watch`, played by hand) an EXP gain is hers anyway, put on a monster in a star's reach."""

    def __init__(self, model: MapModel | None = None, throws: bool = True):
        self.model = model
        self.walking = speeds(model)
        self.spawns = [] if model is None else [
            (kind_of(s.name), model.platform_under(s.x, s.y, reach=60), s.x) for s in model.spawns if s.name]
        self.throws_known = throws
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.gains: list[float] = []      # her EXP gains not yet put on a monster
        self.throws: list[_Throw] = []
        self.her: tuple[float, float] | None = None  # where she was in the last frame placed
        self.counts = {"tracks": 0, "died hers": 0, "died other": 0, "flickers": 0, "gains unexplained": 0}

    # --- told from outside -----------------------------------------------------------------------

    def gain(self, t: float) -> None:
        """Her EXP went up in the frame shown at t."""
        self.gains.append(t)

    def throw(self, t: float, side: str) -> None:
        """She threw toward `side` at t (Actions.throw_listeners): the monsters a star could meet from where
        she was in the last frame."""
        her = self.her
        aimed = set()
        if her is not None:
            sign = 1 if side == RIGHT else -1
            aimed = {tr.id for tr in self.tracks.values() if tr.state not in (DEAD, OFFSCREEN)
                     and -10 <= sign * (self.predicted(tr, t) - her[0]) <= THROW_REACH
                     and abs(tr.y - her[1]) <= THROW_DY}
        self.throws.append(_Throw(t, side, aimed))

    # --- each frame ---------------------------------------------------------------------------------

    def update(self, t: float, refound: dict[int, Monster], found: list[Monster], view: View | None,
               her: tuple[float, float] | None) -> list[Event]:
        """The frame shown at t: `refound`, the tracks re-found in it (id -> where), and `found`, what its
        searches found (re-found ones among them are told apart); `view`, what it shows of the map (None:
        she isn't placed, and nothing can be judged); `her`, where she is. Returns what happened."""
        events: list[Event] = []
        if view is None:
            return events
        self.her = her
        seen: set[int] = set()
        for tid, m in sorted(refound.items(), key=lambda kv: self._off(kv[0], kv[1], t)):
            tr = self.tracks.get(tid)
            if tr is None or tr.state == DEAD:
                continue
            if any(overlap(m.mob, self.tracks[i].mob) for i in seen if self.tracks[i].kind == m.kind):
                continue  # two tracks re-found one monster: it's the nearer one's
            self._seen(tr, m, t, events)
            seen.add(tid)
        fresh = [m for m in found if not any(overlap(m.mob, self.tracks[i].mob) and m.kind == self.tracks[i].kind
                                             for i in seen)]
        for m, tr in self._match(t, fresh, seen):
            if tr is None:
                tr = self._new(t, m, view)
                events.append(Event(tr.how, t, tr.id, {"kind": m.kind, "x": round(m.x), "platform": m.platform}))
            else:
                if tr.state == OFFSCREEN:
                    events.append(Event("back", t, tr.id, {"kind": m.kind, "after": round(t - tr.last_seen, 2)}))
                self._seen(tr, m, t, events)
            seen.add(tr.id)
        for tr in list(self.tracks.values()):
            if tr.id not in seen:
                self._unseen(tr, t, view, events)
        self._resolve(t, events)
        self.throws = [th for th in self.throws if t - th.t <= STAR_WINDOW + HIDE_S]
        return events

    def live(self) -> list[Track]:
        """The tracks of monsters alive (seen or not)."""
        return [tr for tr in self.tracks.values() if tr.state != DEAD]

    def predicted(self, tr: Track, t: float) -> float:
        """Where track `tr` should be at t (map x): walking on as it was, for EXTRAPOLATE_S at most, and on its
        platform."""
        x = tr.x + tr.vx * min(max(0.0, t - tr.last_seen), EXTRAPOLATE_S)
        if self.model is not None and tr.platform is not None:
            p = self.model.platform(tr.platform)
            half = MONSTER_HALF.get(tr.kind, 34.0)
            x = min(max(x, p.left + half), max(p.left + half, p.right - half))
        return x

    # --- inside ---------------------------------------------------------------------------------------

    def _off(self, tid: int, m: Monster, t: float) -> float:
        tr = self.tracks.get(tid)
        return 0.0 if tr is None else abs(m.x - self.predicted(tr, t))

    def _match(self, t: float, found: list[Monster], seen: set[int]) -> list[tuple[Monster, Track | None]]:
        """Each sighting and the track it is (None: a new monster): nearest pairs first, one each."""
        pairs = []
        for i, m in enumerate(found):
            for tr in self.tracks.values():
                if tr.id in seen or tr.state == DEAD or tr.kind != m.kind:
                    continue
                if tr.platform is not None and m.platform is not None and tr.platform != m.platform:
                    continue
                gap = abs(m.x - self.predicted(tr, t))
                speed = self.walking.get(tr.kind, DEFAULT_SPEED)
                if gap <= MATCH_BASE + MATCH_SPEED * speed * (t - tr.last_seen) + MATCH_KNOCKBACK:
                    pairs.append((gap, i, tr.id))
        out: dict[int, Track] = {}
        used: set[int] = set()
        for _, i, tid in sorted(pairs):
            if i not in out and tid not in used:
                out[i] = self.tracks[tid]
                used.add(tid)
        return [(m, out.get(i)) for i, m in enumerate(found)]

    def _new(self, t: float, m: Monster, view: View) -> Track:
        how = "entered" if view.at_edge(m.mob) else "spawned" if self._near_spawn(m) else "appeared"
        tr = Track(self.next_id, m.kind, m.platform, m.x, m.y, m.mob, t, t, facing=m.facing, pose=m.pose,
                   bar=m.bar, bar_at=t if m.bar is not None else None, how=how)
        self.next_id += 1
        self.counts["tracks"] += 1
        self.tracks[tr.id] = tr
        return tr

    def _near_spawn(self, m: Monster) -> bool:
        return any(kind == m.kind and p is not None and p.id == m.platform and abs(x - m.x) <= SPAWN_NEAR
                   for kind, p, x in self.spawns)

    def _seen(self, tr: Track, m: Monster, t: float, events: list[Event]) -> None:
        """Track `tr` seen as `m` in the frame at t: where it is, how fast it walks, its bar."""
        dt = t - tr.last_seen
        speed = self.walking.get(tr.kind, DEFAULT_SPEED)
        if dt <= 0:
            tr.x = m.x
        elif dt > LONG_GAP_S:
            tr.x, tr.vx = m.x, 0.0
        else:
            predicted = tr.x + tr.vx * dt
            r = m.x - predicted
            tr.x = predicted + ALPHA * r
            tr.vx = max(-SPEED_CLAMP * speed, min(SPEED_CLAMP * speed, tr.vx + BETA * r / dt))
        if TURN_FLIP and m.pose == "move" and m.facing and tr.vx * m.facing < 0:
            tr.vx = -tr.vx
        tr.y, tr.mob = m.y, m.mob
        if m.platform is not None:
            tr.platform = m.platform
        if m.facing:
            tr.facing = m.facing
        tr.pose = m.pose
        tr.last_seen, tr.seen = t, tr.seen + 1
        if tr.state in (HIDDEN, OFFSCREEN):
            tr.state = LIVE
        if m.bar is not None:
            low = tr.bar if tr.bar is not None else 1.0
            if m.bar <= low - BAR_STEP:
                tr.hits += 1
                events.append(Event("bar_hit", t, tr.id, {"bar": round(m.bar, 2), "by": self._by(tr, t)}))
            tr.bar, tr.bar_at = min(m.bar, low), t

    def _by(self, tr: Track, t: float) -> str | None:
        """Who hit it, as far as her throws tell."""
        if not self.throws_known:
            return None
        return "hers" if any(0 <= t - th.t <= STAR_WINDOW and tr.id in th.aimed for th in self.throws) else "other"

    def _unseen(self, tr: Track, t: float, view: View, events: list[Event]) -> None:
        dt = t - tr.last_seen
        if tr.state == DEAD:
            if t - tr.judged > DEAD_KEEP_S:
                del self.tracks[tr.id]
            return
        if not view.shows(tr.kind, self.predicted(tr, t), tr.y):
            if tr.state not in (OFFSCREEN, DYING):
                tr.state = OFFSCREEN
                events.append(Event("left", t, tr.id, {"kind": tr.kind}))
            if dt > OFFSCREEN_S:
                del self.tracks[tr.id]
            return
        if tr.state == OFFSCREEN:  # where it was going is in view again, but it's elsewhere by now: off still
            if dt > OFFSCREEN_S:
                del self.tracks[tr.id]
            return
        if dt >= MISSING_S and tr.state == LIVE:
            if tr.seen < MIN_SEEN:  # seen in a frame or two: a match on scenery, not a monster
                self.counts["flickers"] += 1
                del self.tracks[tr.id]
                return
            tr.state = HIDDEN
        if tr.state == HIDDEN and dt > HIDE_S:
            self._dies(tr, tr.last_seen, "other", events, t)

    def _resolve(self, t: float, events: list[Event]) -> None:
        """Her EXP gains put on the monsters they were (see the top); while the likeliest is still seen, fading,
        it's `dying`."""
        for tr in self.tracks.values():
            if tr.state == DYING:
                tr.state = LIVE if t - tr.last_seen < MISSING_S else HIDDEN
        left = []
        for g in self.gains:
            cands = [tr for tr in self.tracks.values() if tr.state in (LIVE, HIDDEN) and tr.last_seen >= g - HIDE_S
                     and self._aimed(tr, g)]
            gone = [tr for tr in cands if tr.state == HIDDEN]
            if t - g <= FADE_S + MISSING_S and not (len(cands) == 1 and gone):
                left.append(g)  # wait for the fade
                live = [tr for tr in cands if tr.state == LIVE]
                if live:
                    min(live, key=self._likely).state = DYING
            elif gone:
                after = [tr for tr in gone if tr.last_seen >= g - FRAME_SLACK]  # went out of sight as it died
                self._dies(min(after or gone, key=self._likely), g, "hers", events, t)
            else:
                self.counts["gains unexplained"] += 1
        self.gains = left

    def _aimed(self, tr: Track, g: float) -> bool:
        if self.throws_known:
            return any(0 <= g - th.t <= STAR_WINDOW and tr.id in th.aimed for th in self.throws)
        her = self.her
        return her is not None and abs(tr.x - her[0]) <= THROW_REACH and abs(tr.y - her[1]) <= THROW_DY

    def _likely(self, tr: Track) -> tuple[float, float]:
        """Which of the monsters a gain may be comes first: the least HP left (never hurt: all), the nearest her."""
        return (1.0 if tr.bar is None else tr.bar, abs(tr.x - self.her[0]) if self.her is not None else 0.0)

    def _dies(self, tr: Track, when: float, cause: str, events: list[Event], t: float) -> None:
        tr.state, tr.died_at, tr.cause, tr.judged = DEAD, when, cause, t
        self.counts[f"died {cause}"] += 1
        events.append(Event("died", t, tr.id, {"kind": tr.kind, "cause": cause, "x": round(tr.x),
                                                "platform": tr.platform, "hurt": tr.bar is not None,
                                                "at": round(when, 3)}))


def overlap(a: Mob, b: Mob) -> bool:
    """Two sightings of one monster: their pictures overlap as suppress_overlaps counts it."""
    return abs(a.x - b.x) <= max(a.w, b.w) // 2 and abs(a.y - b.y) <= max(a.h, b.h) // 2


def snapshot(tracks: list[Track]) -> list[Track]:
    """Copies of tracks as they are now (a state keeps them while the tracker goes on)."""
    return [replace(tr) for tr in tracks]
