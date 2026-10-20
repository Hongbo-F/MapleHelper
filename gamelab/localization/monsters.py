"""Monsters on the map: what's seen on screen (perception's Mob) placed on the platform it walks on
(Monster), telling a monster seen again from a new one, and where a monster can be a moment later
(MonsterTrack, for planning moves around it).
"""

from dataclasses import dataclass

from ..perception.mobs import Mob
from ..perception.kinds import kind_of
from .mapdata import MapModel, MobBody, Platform

TRACK_SLACK = 85   # how far a monster may move between two looks (knockback) and still be the same one
SEARCH_SLACK = 150  # ... and how far it's looked for once it has vanished
SAME_DY = 40       # ... and how far up or down (map units)

# Walking speeds (map units/s), from each monster's speed stat in the game data (pig and Orange
# Mushroom 0, slime -20, snail -65) applied to a base of 70; an upper bound for where they can get.
MONSTER_SPEED = {"pig": 70.0, "orange-mushroom": 70.0, "slime": 56.0, "snail": 25.0}
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

# Matching a look's monsters to the last look's (Tracker).
MATCH_WITHIN = 2.0   # s: an older last look is no help
MATCH_DX = 90        # map units a monster may have moved since, on the same platform
STILL_DX = 6         # ... and moved at most this: standing still (or turning)
MIN_SPEED_DT = 0.2   # s between looks before its speed is worth measuring


@dataclass(frozen=True)
class Monster:
    """A monster seen on screen, placed on the map."""
    mob: Mob
    x: float              # map units, its feet
    y: float
    platform: int | None  # the platform it walks on (monsters never leave it)
    bar: float | None = None  # its HP bar (fraction left), shown once it has been hit; None: unhurt

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


def monster_platform(model: MapModel, kind: str, x: float, y: float, reach: float = 50) -> Platform | None:
    """The platform a monster seen at map (x, y) walks on. Monsters never leave the platform they
    spawn on, so only platforms where its kind spawns count (or any spawn platform, if none fits):
    feet read from the screen are ~20 units off at times, and a safe platform 42-60 units above a
    hunting ground once had pigs "on" it, so she backed away from them and off her platform."""
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


def seen_again(m: Monster, now: list[Monster]) -> Monster | None:
    """The monster in `now` that is `m` seen again, if any (the nearest that can be)."""
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
    speed: float = 70.0
    heading: int = 0
    half: float = 34.0
    vx: float | None = None
    seen_at: float = 0.0
    height: float = 46.0

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
    the same platform, within MATCH_DX units, the last look under MATCH_WITHIN s ago)."""

    def __init__(self):
        self.last: tuple[float, list[MonsterTrack]] = (0.0, [])  # the last look's time, and its tracks
        self.speeds: dict[str, list[float]] = {}  # kind -> walking speeds seen between looks (units/s)

    def update(self, t: float, monsters: list[Monster], seen_at: float = 0.0) -> list[MonsterTrack]:
        """Tracks for the monsters of a look taken at `t` (time.monotonic). `seen_at`: when that look
        was, in the time of the plan they're for."""
        when, tracks = self.last
        before = tracks if t - when < MATCH_WITHIN else []
        out = []
        for m in monsters:
            if m.platform is None:
                continue
            speed = MONSTER_SPEED.get(m.kind, 70.0)
            heading, vx = 0, None
            prev = [mo for mo in before if mo.platform == m.platform and abs(mo.x - m.x) < MATCH_DX]
            if prev:
                mo = min(prev, key=lambda mo: abs(mo.x - m.x))
                dx = m.x - mo.x
                heading = (1 if dx > 0 else -1) if abs(dx) > STILL_DX else 0
                if t - when > MIN_SPEED_DT:  # how fast monsters really walk, for MONSTER_SPEED
                    self.speeds.setdefault(m.kind, []).append(abs(dx) / (t - when))
                    vx = 0.0 if abs(dx) <= STILL_DX else max(-speed, min(speed, dx / (t - when)))
            out.append(MonsterTrack(m.platform, m.x, speed, heading, MONSTER_HALF.get(m.kind, 34.0), vx, seen_at,
                                    MONSTER_HEIGHT.get(m.kind, 46.0)))
        self.last = (t, out)
        return out


def tracks_record(tracks: list[MonsterTrack], model: MapModel, horizon: float = 1.0) -> dict:
    """Monsters as the move planner predicts them, as telemetry data ('monsters'): where each can be
    `horizon` s after it was seen."""
    return {"list": [{"platform": t.platform, "x": round(t.x, 1), "vx": t.vx, "heading": t.heading,
                      "reach": [round(v, 1) for v in t.reach(t.seen_at + horizon, model.platform(t.platform))]}
                     for t in tracks]}
