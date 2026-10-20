"""Monsters on the map: what's seen on screen (perception's Mob) placed on the platform it walks on
(Monster), telling a monster seen again from a new one, and where a monster can be a moment later
(MonsterTrack, for planning moves around it).
"""

from collections import deque
from dataclasses import dataclass

from ..perception.mobs import Mob
from ..perception.kinds import kind_of
from .mapdata import MapModel, MobBody, Platform

TRACK_SLACK = 85   # how far a monster may move between two looks (knockback) and still be the same one
SEARCH_SLACK = 150  # ... and how far it's looked for once it has vanished
SAME_DY = 40       # ... and how far up or down (map units)

# Walking speeds (map units/s): the game's walkSpeed (125) times 1 + each monster's speed stat / 100 (pig and
# Orange Mushroom 0, slime -20, snail -65; mapdata reads them with the map, MapModel.mob_speeds, and
# tests/test_attack.py checks this table against it). Measured on recordings with the camera still: snails 42
# and 45 units/s, pigs 120-128. A base of 70 was used before, a guess 1.79 times too slow: 43% of the speeds
# the tracker measured in live runs were at that clamp. An upper bound for where they can get.
MONSTER_SPEED = {"pig": 125.0, "orange-mushroom": 125.0, "slime": 100.0, "snail": 43.75}
DEFAULT_SPEED = 125.0  # a monster of speed 0 (walkSpeed), for kinds the table and the map don't know
# Their bodies (map units), from the boxes (lt/rb) of their frames in the game data (Mob/<id>.img; mapdata
# reads them with the map, MapModel.mob_bodies, and tests/test_attack.py checks these tables against it).
# Half their width (half their widest stand or move frame), and how tall they are walking (the top of
# their first move frame): touches and hops go by these.
MONSTER_HALF = {"pig": 34.0, "orange-mushroom": 32.0, "slime": 28.0, "snail": 19.0}
MONSTER_HEIGHT = {"pig": 46.0, "orange-mushroom": 55.0, "slime": 41.0, "snail": 26.0}
# How tall they are at least while they stand or walk (the lowest top of those frames: pig stand/2, slime
# move/1, snail move/3), and while they flinch after a hit (hit1): what an attack can pick goes by these
# (attack.reach), and a snail is far shorter than the others.
MONSTER_TOP = {"pig": 38.0, "orange-mushroom": 55.0, "slime": 40.0, "snail": 24.0}
MONSTER_FLINCH_TOP = {"pig": 54.0, "orange-mushroom": 65.0, "slime": 46.0, "snail": 33.0}
FLINCH_S = 0.6       # s a monster flinches after a hit (its hit1 frame's delay, game data)
# How long a killed one is still seen, fading out where it died: its die1 frames' delays (game data: a
# snail 8 x 120 + 300 ms, a slime 3 x 180 + 300, a pig and an Orange Mushroom 2 x 180 + 300; the last frame
# fades it out).
MONSTER_DIE_S = {"pig": 0.66, "orange-mushroom": 0.66, "slime": 0.84, "snail": 1.26}
TURN_SLACK = 25      # units a monster seen walking one way may still come back (they turn now and then)

PLATFORM_SLACK = 25  # map units past a platform's ends a monster seen on screen may still be on it
# Monsters do leave the platforms they spawn on (after_rest.png: a pig on platform 26, which spawns nothing;
# live runs had pigs placed on the snails' 20 and 22, 9 times, as the nearest spawn platform). With its feet
# read from the template's origin while she stands on the ground (Mob.feet, Scene.grounded), a monster goes
# on whatever platform's surface is within FEET_SNAP of them: such feet were -1.4 to +2.5 units from the
# surface on 15 saved frames (`bodies`; +7.7 for a pig whose hop frame matched though it walked), against
# -8.5 to +16.4 by the template's bottom. Otherwise, or with none that close, the spawn rule
# (monster_platform). Not checked in game: None keeps to the spawn rule.
FEET_SNAP: float | None = 12

# Matching a look's monsters to the last look's (Tracker).
MATCH_WITHIN = 2.0   # s: an older last look is no help
MATCH_DX = 90        # map units a monster may have moved since, on the same platform
STILL_DX = 6         # ... and moved at most this: standing still (or turning)
MIN_SPEED_DT = 0.2   # s between looks before its speed is worth measuring
# True: a monster seen walking (a move frame matched it) heads the way it faces when looks can't tell yet
# (seen for the first time, or the last look too recent), so the move planner spreads its reach forward and
# only TURN_SLACK back. The facing itself is right: the template's was the way the monster moved in 192 of
# 199 stretches of 0.6 s where it walked (`bodies` on two recordings with the camera still; the misses at
# turns). But monsters turn, and at 125 units/s TURN_SLACK is 0.2 s of walking back: in the simulator, with
# its monsters showing which way they face, keeping clear of them went worse with it (nav_monsters_long_way
# on 16 seeds: 11 arrived against 13, in 114 s against 85 on average, touched 28.5 times against 19.8; rest
# trips with --rest-avoid 1.20 touches against 1.15). So off, until the planner's turn allowance is worked out
# per second; fights don't use it (no difference in the tour or the hunt loop).
FACING_HEADING = False
# True: a sighting is matched to a monster the last look saw walking (a move frame) by where it can have got
# to the way it faced (off_course), not by where it was, so two walking past each other aren't swapped. But
# the pose comes from the template that matched, and in game that was wrong for 18% of the sightings of
# monsters walking and 22% of those standing (`bodies` on 3 recordings with the camera still), for up to 1.4
# s on end: two pigs standing 100 units apart in move frames, facing each other, were each matched to the
# other and said to walk at 125 units/s. In the simulator, where poses are right, the tour and the hunt loop
# came out the same either way. False: the nearest within MATCH_DX of where it was.
WALK_MATCH = False
# Speeds kept per kind (Tracker.speeds, for `nav`'s summary), the latest: one a monster a look, so kept
# all, a look every frame would add about 110,000 an hour (4 MB); 400 is a few minutes' worth even at 1.4
# looks a second, and enough for the summary's median and 90th percentile.
SPEED_SAMPLES = 400


@dataclass(frozen=True)
class Monster:
    """A monster seen on screen, placed on the map."""
    mob: Mob
    x: float              # map units, its feet
    y: float
    platform: int | None  # the platform it walks on (None: none near its feet)
    bar: float | None = None  # its HP bar (fraction left), shown once it has been hit; None: unhurt
    facing: int = 0       # which way it faces: 1 right, -1 left, 0 not known (Mob.facing)
    pose: str = ""        # "stand", "move" (walking) or "hit1" (flinching); "": not known (Mob.pose)

    @property
    def kind(self) -> str:
        return self.mob.kind


def bodies(model: MapModel | None = None) -> dict[str, MobBody]:
    """Each kind's body (by kind): read from the game data with the map for the monsters it has
    (MapModel.mob_bodies), from the tables above for the rest."""
    out = {k: MobBody(MONSTER_TOP[k], MONSTER_FLINCH_TOP[k], MONSTER_HALF[k], MONSTER_DIE_S[k]) for k in MONSTER_TOP}
    if model is not None:
        out.update({kind_of(name): b for name, b in model.mob_bodies.items()})
    return out


def speeds(model: MapModel | None = None) -> dict[str, float]:
    """How fast each kind walks (units/s): from the game data with the map for the monsters it has
    (MapModel.mob_speeds), from MONSTER_SPEED for the rest."""
    out = dict(MONSTER_SPEED)
    if model is not None:
        out.update({kind_of(name): v for name, v in model.mob_speeds.items()})
    return out


def monster_platform(model: MapModel, kind: str, x: float, y: float, reach: float = 50,
                     exact: bool = False) -> Platform | None:
    """The platform a monster seen at map (x, y) walks on. With `exact` feet (FEET_SNAP), any platform
    whose surface is that close. Else only platforms where its kind spawns count (or any spawn platform,
    if none fits): feet read from the bottom of a template were ~20 units off at times, and a safe
    platform 42-60 units above a hunting ground once had pigs "on" it, so she backed away from them and
    off her platform."""
    if exact and FEET_SNAP is not None:
        near = [(abs(p.y_at(p.clamp(x)) - y), p.id) for p in model.platforms if p.covers(x, PLATFORM_SLACK)]
        near = [g for g in near if g[0] <= FEET_SNAP]
        if near:
            return model.platform(min(near)[1])
    best = None
    for own_kind in (True, False):
        for pid, names in model.monsters_by_platform.items():
            if own_kind and not any(kind_of(n) == kind for n in names):
                continue
            p = model.platform(pid)
            if not p.covers(x, PLATFORM_SLACK):
                continue
            gap = abs(p.y_at(p.clamp(x)) - y)
            if gap <= reach and (best is None or gap < best[0]):
                best = (gap, p)
        if best:
            return best[1]
    return None


def same(a: Monster, b: Monster) -> bool:
    """Whether two sightings can be one monster: same kind, near the same spot."""
    return a.kind == b.kind and abs(a.x - b.x) < TRACK_SLACK and abs(a.y - b.y) < SAME_DY


def off_course(x: float, start: float, go: float) -> tuple[float, float]:
    """How far map x is from where a monster seen at `start` can be after walking up to `go` units (+ right,
    - left; it may have stopped on the way), then how far from where it would be had it walked all the way:
    what the tracker matches sightings by, the least first. Two walking past each other are each matched to
    themselves, and one that stopped too. With `go` 0, the nearest to `start`."""
    lo, hi = min(start, start + go), max(start, start + go)
    return max(lo - x, 0.0, x - hi), abs(x - start - go)


def seen_again(m: Monster, now: list[Monster]) -> Monster | None:
    """The monster in `now` that is `m` seen again, if any (the nearest that can be). Not by the way it
    faced, as the tracker does: these are the monsters she threw at (fight.FightState), and a monster hit
    flinches and is knocked back rather than walking on. Matched to the way it faced, for up to the 1.8 s
    one stays hidden under its damage numbers, the hunt loop in the simulator lost 6.4% of its EXP a minute
    (20 seeds; better on 7), though the tour gained 4.0%."""
    near = [o for o in now if same(o, m)]
    return min(near, key=lambda o: abs(o.x - m.x)) if near else None


def dead_one(aimed: list[Monster], now: list[Monster], her_x: float) -> Monster | None:
    """Which of the monsters she was throwing at died: one no longer seen (nearest her), else the
    nearest (it may still be seen fading out)."""
    gone = [t for t in aimed if not any(same(o, t) for o in now)]
    pool = gone or aimed
    return min(pool, key=lambda t: abs(t.x - her_x)) if pool else None


@dataclass(frozen=True)
class MonsterTrack:
    """A monster as the move planner sees it: where it was (map x) at plan time `seen_at` (negative:
    before the plan starts), how fast it can walk, which way it was seen walking (+1/-1, or 0 if
    unknown), its measured velocity if two looks caught it, and its body's half width and height
    (by default a pig's)."""
    platform: int
    x: float
    speed: float = DEFAULT_SPEED
    heading: int = 0
    half: float = 34.0
    vx: float | None = None
    seen_at: float = 0.0
    height: float = 46.0
    facing: int = 0       # which way it faced, seen walking (1 right, -1 left; 0: not known, or standing)

    def at(self, t: float, p: Platform) -> float:
        """Where its centre most likely is at plan time t (walking on as it was, stopping at the ends)."""
        x = self.x + (self.vx or 0.0) * max(0.0, t - self.seen_at)
        return min(max(x, p.left + self.half), p.right - self.half)

    def reach(self, t: float, p: Platform) -> tuple[float, float]:
        """Where its centre can be at plan time t (it never leaves its platform)."""
        dt = max(0.0, t - self.seen_at)
        if self.vx is not None:  # seen walking: where it's heading, give or take a turn
            c, spread = self.at(t, p), TURN_SLACK + 20.0 * dt
            return max(p.left, c - spread), min(p.right, c + spread)
        go = self.speed * dt
        lo = self.x - (go if self.heading <= 0 else min(go, TURN_SLACK))
        hi = self.x + (go if self.heading >= 0 else min(go, TURN_SLACK))
        return max(p.left, lo), min(p.right, hi)


class Tracker:
    """Which way the monsters walk, and how fast: each look's monsters matched to the last look's (on
    the same platform, within MATCH_DX units, the last look under MATCH_WITHIN s ago): the nearest, or (with
    WALK_MATCH) for one seen walking, the nearest to the way it was going (off_course)."""

    def __init__(self, model: MapModel | None = None):
        """`model`: the map, for how fast its monsters walk (speeds)."""
        self.walking = speeds(model)  # kind -> how fast it walks (units/s; the game data's)
        self.last: tuple[float, list[MonsterTrack]] = (0.0, [])  # the last look's time, and its tracks
        # kind -> the latest walking speeds seen between looks (units/s), SPEED_SAMPLES at most
        self.speeds: dict[str, deque[float]] = {}

    def update(self, t: float, monsters: list[Monster], seen_at: float = 0.0) -> list[MonsterTrack]:
        """Tracks for the monsters of a look taken at `t` (time.monotonic). `seen_at`: when that look
        was, in the time of the plan they're for."""
        when, tracks = self.last
        # The last look's tracks, each with how far it can have walked since, if it was seen walking (WALK_MATCH).
        before = [(mo, mo.facing * mo.speed * (t - when) if WALK_MATCH else 0.0) for mo in tracks]             if t - when < MATCH_WITHIN else []
        out = []
        for m in monsters:
            if m.platform is None:
                continue
            speed = self.walking.get(m.kind, DEFAULT_SPEED)
            heading, vx = 0, None
            prev = [(mo, go) for mo, go in before if mo.platform == m.platform
                    and off_course(m.x, mo.x, go)[0] < MATCH_DX]  # within MATCH_DX of where it can be
            if prev:  # the nearest, or for one seen walking, the nearest to the way it was going
                mo = min(prev, key=lambda p: off_course(m.x, p[0].x, p[1]))[0]
                dx = m.x - mo.x
                heading = (1 if dx > 0 else -1) if abs(dx) > STILL_DX else 0
                if t - when > MIN_SPEED_DT:  # how fast monsters really walk (nav's summary)
                    self.speeds.setdefault(m.kind, deque(maxlen=SPEED_SAMPLES)).append(abs(dx) / (t - when))
                    vx = 0.0 if abs(dx) <= STILL_DX else max(-speed, min(speed, dx / (t - when)))
            walking = m.facing if m.pose == "move" else 0
            if FACING_HEADING and not heading and vx is None:  # looks can't tell yet: the way it faces walking
                heading = walking
            out.append(MonsterTrack(m.platform, m.x, speed, heading, MONSTER_HALF.get(m.kind, 34.0), vx, seen_at,
                                    MONSTER_HEIGHT.get(m.kind, 46.0), walking))
        self.last = (t, out)
        return out


def tracks_record(tracks: list[MonsterTrack], model: MapModel, horizon: float = 1.0) -> dict:
    """Monsters as the move planner predicts them, as telemetry data ('monsters'): where each can be
    `horizon` s after it was seen."""
    return {"list": [{"platform": t.platform, "x": round(t.x, 1), "vx": t.vx, "heading": t.heading,
                      "reach": [round(v, 1) for v in t.reach(t.seen_at + horizon, model.platform(t.platform))]}
                     for t in tracks]}
