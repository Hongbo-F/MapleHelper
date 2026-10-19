"""A deterministic stand-in for the game: her body, the monsters, stars, drops, HP and EXP on a real map
(mapdata.MapModel), in map units, driven by key events on a fake clock (see sim.fakes).

The physics are the ones gamelab plans with, so a plan that is right on paper works here too:

- on the ground an arrow runs her at physics.WALK_SPEED after a standing start of flow.RUN_LAG; let go,
  she slides flow.STOP_RUN on over flow.STOP_S (a short input lag, then braking); walking past a
  platform's end she falls, unless a wall rises there;
- the jump key lifts her off flow.JUMP_LAG later into physics.fly's arc (an exact copy of it, `trajectory`,
  without its cache); with down held it drops her through the platform flow.DROP_LAG later; a jump key
  in the air is ignored, and so is one inside the attack lock;
- up held at a rope's foot, or anywhere along it in the air, grabs it; up climbs at flow.ROPE_SPEED onto
  the platform at its top (a short climb-over, flow.CLIMB_OVER), down climbs down at physics.CLIMB_SPEED
  and lets go at its bottom; down on a rope's top grabs it from above;
- monsters walk their platform at monster_data.MONSTER_SPEED, turning and stopping now and then (seeded). A
  touch leaves her a second's invulnerability; she evades half of them (a thief's "MISS": nothing
  else happens), the others take 1.4% HP (the median of the lab's 118 logged hits) and knock her back
  ~58 units (the median logged push). At 0% HP she dies, and the run ends (SimDied);
- the attack key throws a star actions.STAR_DELAY_MS later that hits the first monster in its path on the
  side she faces (attack.reach): within reach.STAR_REACH, and in the band
  reach.SHOT_UP above to reach.SHOT_DOWN below her; on her level (reach.LEVEL) at any distance, off it
  only from reach.min_dx sideways (right under or over her isn't hit). Pressed on the ground with a
  monster on her level closer than reach.MIN_SHOT in front, the claw swipes it instead (a fifth of a
  star's damage). Kills give EXP (model.mob_exp) and, 2 s later, drops on its platform; dead monsters
  respawn 8 s later where they spawned;
- the pick-up key takes the nearest drop within reach (items only while her inventory has room), the
  HP potion key gives 30% HP, the chair key sits her down (+3% HP/s) until an arrow stands her up.

`Noise` (off by default) adds the errors of the real game, each one seeded and tied to the moment it
happens, not to how often anything was looked at: stars that miss, walks that stop a few units early or
late, her minimap dot read to whole pixels, and spells when her name tag or her dot can't be seen.

Time is kept in whole milliseconds on a 5 ms tick, so the state at any moment depends only on the key
events, never on when or how often it was looked at.
"""

import bisect
import math
import random
from dataclasses import dataclass

import numpy as np

from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT

from ..attack import reach
from ..control.actions import STAR_DELAY_MS
from ..localization import monsters as monster_data
from ..localization.mapdata import MapModel, Platform, Rope
from ..motion import flow
from ..motion.physics import CLIMB_SPEED, FALL_SPEED, GRAVITY, JUMP_HEIGHT, JUMP_SPEED, ROPE_REACH, WALK_SPEED
from ..perception.sprites import kind_of

TICK_MS = 5
FLY_DT = 0.004           # physics.fly's step: the arc is integrated exactly as the planner does
RUN_LAG = flow.RUN_LAG
TURN_EXTRA = flow.TURN_EXTRA
# Braking with no arrow held: she slides flow.LAND_SLIDE from full speed (as on landing). An arrow let go
# still counts for RELEASE_LAG, so she stops flow.STOP_RUN on in all (~0.27 s). With the run lag that is
# physics.py's measured walk: an arrow held t ms moves her ~8.5 + 0.149 t units, a tap under ~30 ms not at all.
DECEL = WALK_SPEED ** 2 / (2 * flow.LAND_SLIDE)
RELEASE_LAG = (flow.STOP_RUN - flow.LAND_SLIDE) / WALK_SPEED
JUMP_LAG = flow.JUMP_LAG
DROP_LAG = flow.DROP_LAG
CLIMB_UP = flow.ROPE_SPEED
CLIMB_DOWN = CLIMB_SPEED
CLIMB_OVER = flow.CLIMB_OVER
ARMS = ROPE_REACH - JUMP_HEIGHT  # she grabs a rope this far above her feet
GRAB_X = 10.0            # ... within this of its x
ROPE_TOP_REACH = 15.0    # down grabs a rope whose top is within this of her platform
REGRAB_AFTER = 0.3       # s after leaving a rope before up grabs one again
ROPE_JUMP_VY = 220.0     # jumping off a rope sideways: next to no lift
HER_HALF = flow.HER_HALF
HER_HEIGHT = 55.0
STAR_DELAY = STAR_DELAY_MS / 1000
ATTACK_LOCK = COMBAT.attack_lock_ms / 1000
# A touch: lab/hits.jsonl's 118 hits took 1-4 px of the 147 px HP bar (0.7-2.7%), 2 px (1.4%) the median.
TOUCH_DAMAGE = 0.014
# A thief's evasion: about half the touches show "MISS" and take nothing, without a knockback (lab/hop_log
# and hop_log2: 7 misses, 6 with damage). Seeded by the touch's tick, like Noise.
EVADE = 0.5
INVULNERABLE = 1.0
# Its knockback: ~40 units in the air (0.22 s) and a ~19-unit slide on landing, ~58 in all on flat ground
# (hits.jsonl's pushes on her own level: quartiles 21 / 58 / 87 units).
KNOCK_VX, KNOCK_VY = 180.0, 220.0
HITS = {"pig": 3, "orange-mushroom": 3, "slime": 2, "snail": 1}  # HP in stars
CLAW = 0.2               # a claw swipe's damage, in stars (9-11 against a star's 30-70)
DROP_DELAY = 2.0
RESPAWN = 8.0
FADE = 0.8               # s a killed monster is still seen fading out where it died
DROP_LIFE = 180.0        # s drops lie before they vanish
PICK_REACH = 15.0
POTION_HP = 0.30
CHAIR_HP_PER_S = 0.03
STAND_UP = 0.2           # s the arrow press that stands her up from the chair takes
PORTAL_REACH = (25.0, 40.0)
MESSAGE_FOR = 5.0        # s a pickup message stays in the log on screen
FULL_MESSAGE = "You can't get anymore items."
ITEM_OF = {"pig": "leather"}


class SimDied(Exception):
    """Her HP reached 0: she's dead, and a simulated run ends there (nothing she does counts any more)."""


@dataclass(frozen=True)
class Noise:
    """The real game's errors, all off by default. Each draw is seeded by the world's seed and the
    moment it's for (a star's tick, a release's tick), and the spells are fixed stretches of time, so a
    run is still the same every time, and a change that only adds or drops a look changes nothing here."""
    miss: float = 0.0            # chance a star that would hit misses (lab/shots.jsonl: 72.5% of 204 hit)
    walk_jitter: float = 0.0     # units: letting go of an arrow stops her up to this much early or late
    dot_px: bool = False         # her minimap dot read to whole pixels (11.4 map units each), not exact
    tag_hidden: tuple = ()       # (from, to) s since the start: her name tag can't be seen (behind a box)
    dot_hidden: tuple = ()       # (from, to) s: her minimap dot can't be seen (a tooltip over the minimap)

    @staticmethod
    def during(spells: tuple, t: float) -> bool:
        return any(a <= t < b for a, b in spells)


@dataclass
class Flight:
    """An arc through the air (physics.fly's integration, sampled): where it lands and when."""
    platform: Platform | None
    land_x: float
    seconds: float
    path: list[tuple[float, float]]

    def at(self, tau: float) -> tuple[float, float]:
        i = tau / FLY_DT
        k = min(int(i), len(self.path) - 1)
        if k + 1 >= len(self.path):
            return self.path[-1]
        f = i - k
        (x0, y0), (x1, y1) = self.path[k], self.path[k + 1]
        return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f


_WALLS: dict[int, tuple] = {}


def _walls(model: MapModel) -> tuple[list[float], list[tuple[float, float]]]:
    key = id(model)
    if key not in _WALLS:
        ws = sorted((w for w in model.walls if w.x1 == w.x2), key=lambda w: w.x1)
        _WALLS[key] = (model, [w.x1 for w in ws], [(min(w.y1, w.y2) - 1, max(w.y1, w.y2) + 1) for w in ws])
    return _WALLS[key][1], _WALLS[key][2]


def trajectory(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
               limit: float = 4.0, sides: tuple[float, float] | None = None) -> Flight:
    """physics.fly, step for step (the same integration, so she lands exactly where the planner says),
    keeping the path. Not cached. `sides`: the map's left and right limits, which stop her like walls
    (physics.fly has none: the planner never jumps at the map's edges)."""
    dt = FLY_DT
    wall_xs, wall_ys = _walls(model)
    reach = abs(vx) * limit + 1
    top = y - (vy * vy / (2 * GRAVITY) if vy < 0 else 0) - 1
    plats = [q for q in model.platforms if q is not skip and q.right >= x - reach and q.left <= x + reach
             and max(q.y_at(q.left), q.y_at(q.right)) >= top]
    t, path = 0.0, [(x, y)]
    while t < limit:
        nx, ny, nvy = x + vx * dt, y + vy * dt, min(vy + GRAVITY * dt, FALL_SPEED)
        if vx:
            lo, hi = (x, nx) if nx >= x else (nx, x)
            i = bisect.bisect_left(wall_xs, lo)
            while i < len(wall_xs) and wall_xs[i] <= hi:
                if wall_ys[i][0] <= ny <= wall_ys[i][1]:
                    nx, vx = x, 0.0
                    break
                i += 1
            if sides is not None and not sides[0] <= nx <= sides[1]:
                nx, vx = min(max(nx, sides[0]), sides[1]), 0.0
        if (vy > 0 or nvy > 0) and t > 0 and ny >= y:
            for q in plats:
                if q.covers(nx) and y <= q.y_at(nx) <= ny + 0.01:
                    path.append((nx, q.y_at(nx)))
                    return Flight(q, nx, t + dt, path)
        x, y, vy, t = nx, ny, nvy, t + dt
        path.append((x, y))
    return Flight(None, x, t, path)


# --- monsters ----------------------------------------------------------------------------------------

@dataclass
class Mob:
    """A monster walking its platform. Its motion is a list of segments drawn from its own rng (walk
    left, walk right or stand, for a while), evaluated lazily, so where it is at a moment doesn't depend
    on how often it was looked at."""
    idx: int
    name: str
    kind: str
    platform: Platform
    home: float               # spawn x (where it respawns)
    rng: random.Random
    speed: float = 70.0
    half: float = 34.0
    height: float = 46.0
    hp_max: float = 1.0        # in stars
    hp: float = 1.0
    alive: bool = True
    died_at: float | None = None
    died_x: float = 0.0
    # current motion segment: from t0 at x0, heading d (-1, 0, +1) until t1
    t0: float = 0.0
    x0: float = 0.0
    d: int = 0
    t1: float = 0.0

    @property
    def lo(self) -> float:
        return self.platform.left + self.half

    @property
    def hi(self) -> float:
        return max(self.lo, self.platform.right - self.half)

    def start(self, t: float, x: float) -> None:
        self.t0, self.x0 = t, min(max(x, self.lo), self.hi)
        self._draw()

    def _draw(self) -> None:
        r = self.rng.random()
        self.d = 0 if r < 0.35 else (-1 if r < 0.675 else 1)
        self.t1 = self.t0 + (self.rng.uniform(0.5, 2.5) if self.d == 0 else self.rng.uniform(1.0, 4.0))

    def _fold(self, t: float) -> tuple[float, int]:
        """Where it is at t within the current segment, and which way it's then going (bouncing at the ends)."""
        if not self.d:
            return self.x0, 0
        span = self.hi - self.lo
        if span <= 0:
            return self.lo, self.d
        u = (self.x0 - self.lo) + self.d * self.speed * (t - self.t0)
        m = u % (2 * span)
        return (self.lo + m, self.d) if m <= span else (self.lo + 2 * span - m, -self.d)

    def at(self, t: float) -> float:
        while t >= self.t1:
            self.t0, self.x0 = self.t1, self._fold(self.t1)[0]
            self._draw()
        return self._fold(t)[0]

    def y(self, x: float) -> float:
        return self.platform.y_at(x)


@dataclass
class Drop:
    kind: str           # "mesos" or an item kind
    x: float
    y: float
    platform: int
    since: float        # when it lands (shows) on the ground
    gone: bool = False

    @property
    def mesos(self) -> bool:
        return self.kind == "mesos"


# --- the world ---------------------------------------------------------------------------------------

@dataclass
class Event:
    ms: int
    seq: int
    kind: str           # 'down' / 'up' (keys), 'release' (all keys), or an internal one
    data: object = None
    tag: str = ""       # the plan a key event belongs to
    at: float = 0.0     # clock time it was sent for (before rounding up to a tick)


class World:
    def __init__(self, model: MapModel, keys: GameKeys, start: tuple[int, float], seed: int = 1,
                 monsters="all", hp: float = 1.0, inventory_full: bool = False, facing: str = RIGHT,
                 origin: float = 0.0, drops: list[tuple[str, int, float]] = (), hanging: float | None = None,
                 noise: Noise | None = None):
        """Her standing on platform start[0] at x start[1] (or, with `hanging`, on the rope nearest that x,
        her feet at y `hanging`). `monsters`: None, "all" (the map's spawns), a set of platform ids (the
        spawns on those), or [(name, platform, x)]. `drops`: [(kind, platform, x)] lying there already."""
        self.model = model
        self.origin = origin       # clock time of ms 0
        self.ms = 0
        self.seed = seed
        self.rng = random.Random(seed)
        self.noise = noise or Noise()
        self.actions = _actions(keys)
        self.held: dict[str, int] = {}  # action -> ms it went down
        self._events: list[tuple[int, int, Event]] = []
        self._seq = 0
        self.log: list[str] = []        # notable world events, for the trace
        # her
        p = model.platform(start[0])
        self.mode = "ground"            # ground | air | rope
        self.plat: Platform | None = p
        self.rope: Rope | None = None
        self.flight: Flight | None = None
        self.flight_ms = 0
        self.x, self.y = float(start[1]), p.y_at(start[1])
        self.vx = 0.0
        self.facing = facing
        self.go = 0                     # the arrow that starts her running (a standing start) ...
        self.go_ms = 0                  # ... once this tick comes (run lag), even if it's let go before
        self.release_ms = -10_000       # when the arrow was let go
        self.can_move_ms = 0            # standing start: not before this (landing, climb-over, chair)
        self.left_rope_ms = -10_000
        self.liftoff_due = False
        self.sitting = False
        self.hp = hp
        self.mp = 1.0
        self.exp = 0
        self.attack_lock_ms = -10_000
        self.invulnerable_ms = -10_000
        self.inventory_full = inventory_full
        self.kills: dict[str, int] = {}
        self.touches = 0                # monsters touching her (evaded ones too)
        self.evaded = 0
        self.picked: dict[str, int] = {}
        self.messages: list[tuple[float, str]] = []
        self.portals = [pt for pt in model.portals if pt.kind == 2]
        self.sides = (min(q.left for q in model.platforms), max(q.right for q in model.platforms))
        self.dead = False
        self.died_at: float | None = None
        self._blocked = {pid: (_wall_at(model, q, q.left), _wall_at(model, q, q.right))
                         for pid, q in enumerate(model.platforms)}
        self._near = {q.id: [o.id for o in model.platforms if o.right >= q.left - 60 and o.left <= q.right + 60
                             and -70 <= o.top - q.top <= 70] for q in model.platforms}
        if hanging is not None:
            self._grab(min(model.ropes, key=lambda r: abs(r.x - start[1])))
            self.y = min(max(hanging, self.rope.top), self.rope.bottom)
            self.log.clear()
        self.mobs: list[Mob] = []
        self.by_platform: dict[int, list[Mob]] = {}
        self._spawn(monsters)
        self.drops: list[Drop] = [Drop(kind, x, model.platform(pid).y_at(x), pid, 0.0) for kind, pid, x in drops]

    # --- set-up -------------------------------------------------------------------------------------

    def _spawn(self, monsters) -> None:
        if monsters is None or monsters == []:
            return
        if monsters == "all" or isinstance(monsters, (set, frozenset)):
            wanted = []
            for s in self.model.spawns:
                p = self.model.platform_under(s.x, s.y, reach=60)
                if p is not None and (monsters == "all" or p.id in monsters):
                    wanted.append((s.name or str(s.mob_id), p.id, s.x))
        else:
            wanted = list(monsters)
        for i, (name, pid, x) in enumerate(wanted):
            kind = kind_of(name)
            m = Mob(i, name, kind, self.model.platform(pid), float(x), random.Random(self.rng.random()),
                    monster_data.MONSTER_SPEED.get(kind, 70.0), monster_data.MONSTER_HALF.get(kind, 34.0),
                    monster_data.MONSTER_HEIGHT.get(kind, 46.0), float(HITS.get(kind, 2)), float(HITS.get(kind, 2)))
            m.start(0.0, float(x))
            self.mobs.append(m)
            self.by_platform.setdefault(pid, []).append(m)

    # --- time -----------------------------------------------------------------------------------------

    @property
    def t(self) -> float:
        """World seconds since its start."""
        return self.ms / 1000

    def ms_of(self, clock_t: float) -> int:
        """The tick a clock time falls on (events take effect on the tick at or after them)."""
        ms = (clock_t - self.origin) * 1000
        return int(math.ceil(ms / TICK_MS - 1e-6)) * TICK_MS

    def advance(self, clock_t: float) -> None:
        """Run the world up to clock time `clock_t` (whole ticks at or before it). Raises SimDied once
        she's dead: whatever looks next finds the run over."""
        end = int(math.floor(((clock_t - self.origin) * 1000) / TICK_MS + 1e-6)) * TICK_MS
        while self.ms < end and not self.dead:
            self._tick()
        if self.dead:
            raise SimDied(f"she died at {self.died_at:.3f} s ({self.touches} touches)")

    def chance(self, what: str, ms: int | None = None) -> float:
        """A noise draw in [0, 1) for `what` at tick `ms` (now by default): the same every run."""
        return random.Random(f"{what}:{self.seed}:{self.ms if ms is None else ms}").random()

    def _push(self, ms: int, kind: str, data=None, tag: str = "", at: float = 0.0) -> Event:
        self._seq += 1
        e = Event(max(ms, self.ms), self._seq, kind, data, tag, at)
        bisect.insort(self._events, (e.ms, e.seq, e), key=lambda v: (v[0], v[1]))
        return e

    def key(self, clock_t: float, name: str, down: bool, tag: str = "") -> None:
        """A key going down/up at clock time `clock_t` (in the future, or now)."""
        self._push(self.ms_of(clock_t), "down" if down else "up", name, tag, clock_t)

    def release_all(self, clock_t: float, tag: str = "") -> None:
        """The player lets go of every key at `clock_t` (a plan's end, or a cancel)."""
        self._push(self.ms_of(clock_t), "release", None, tag, clock_t)

    def withdraw(self, tag: str) -> None:
        """Forget the key events of plan `tag` the world hasn't taken yet."""
        self._events = [v for v in self._events if not (v[2].tag == tag and v[2].kind in ("down", "up", "release"))]

    # --- one tick ---------------------------------------------------------------------------------------

    def _tick(self) -> None:
        while self._events and self._events[0][0] <= self.ms:
            _, _, e = self._events.pop(0)
            self._handle(e)
        dt = TICK_MS / 1000
        if self.mode == "ground":
            self._ground(dt)
        elif self.mode == "air":
            self._air()
        else:
            self._climb(dt)
        self._touch()
        if self.sitting:
            self.hp = min(1.0, self.hp + CHAIR_HP_PER_S * dt)
        self.ms += TICK_MS

    def _handle(self, e: Event) -> None:
        if e.kind in ("down", "up"):
            action = self.actions.get(e.data)
            if action is None or self.dead:
                return
            if e.kind == "down":
                if action not in self.held:
                    self.held[action] = self.ms
                    self._pressed(action)
            elif self.held.pop(action, None) is not None:
                self._released(action)
        elif e.kind == "release":
            for action in list(self.held):
                del self.held[action]
                self._released(action)
        elif e.kind == "liftoff":
            self.liftoff_due = False
            if self.mode == "ground":
                self._launch(self.vx, -JUMP_SPEED)
        elif e.kind == "drop":
            if self.mode == "ground" and self.plat is not None and self.plat.id == e.data:
                f = self._trajectory(self.x, self.y + 1, 0.0, 0.0, skip=self.plat)
                if f.platform is not None:
                    self._fly(f, 0.0)
                    self.note(f"drops through platform {e.data}")
        elif e.kind == "star":
            self._attack(*e.data)
        elif e.kind == "loot":
            self.drops += e.data
        elif e.kind == "respawn":
            m = e.data
            m.alive, m.hp, m.died_at = True, m.hp_max, None
            m.start(self.t, m.home)

    def note(self, text: str) -> None:
        self.log.append(f"{self.t:8.3f} {text}")

    # --- keys -------------------------------------------------------------------------------------------

    def arrow(self) -> int:
        """The arrow that counts (the later pressed, if both are held), as -1/0/+1."""
        left, right = self.held.get("left"), self.held.get("right")
        if left is None and right is None:
            return 0
        if right is None or (left is not None and left > right):
            return -1
        return 1

    def _pressed(self, action: str) -> None:
        ground = self.mode == "ground"
        if action in ("left", "right"):
            self.facing = LEFT if action == "left" else RIGHT
            if self.sitting:
                self.sitting = False
                self.can_move_ms = self.ms + int(STAND_UP * 1000)
            self.go, self.go_ms = (-1 if action == "left" else 1), self.ms
        elif action == "jump":
            if self.ms < self.attack_lock_ms or self.liftoff_due:
                return
            if self.mode == "rope":
                d = self.arrow()
                if d:
                    self._leave_rope()
                    self._launch(d * WALK_SPEED, -ROPE_JUMP_VY)
                return
            if not ground:
                return  # in the air: ignored
            if self.sitting:
                self.sitting = False
                self.can_move_ms = self.ms + int(STAND_UP * 1000)
                return
            if "down" in self.held:
                self._push(self.ms + int(DROP_LAG * 1000), "drop", self.plat.id)
            else:
                self.liftoff_due = True
                self._push(self.ms + int(round(JUMP_LAG * 1000)), "liftoff")
        elif action == "attack":
            if self.ms < self.attack_lock_ms or self.mode == "rope":
                return
            self.attack_lock_ms = self.ms + int(ATTACK_LOCK * 1000)
            self.sitting = False
            # In the air (or about to lift off) it's always a star; on the ground the claw may swipe.
            air = self.mode == "air" or self.liftoff_due
            self._push(self.ms + int(STAR_DELAY * 1000), "star", (self.facing, air))
        elif action == "up":
            if ground:
                for pt in self.portals:
                    if abs(pt.x - self.x) < PORTAL_REACH[0] and abs(pt.y - self.y) < PORTAL_REACH[1]:
                        self.note(f"up at portal {pt.name} (to map {pt.to_map}): she'd leave the map")
                self._grab_from_ground()
        elif action == "down":
            if ground:
                self._grab_from_top()
        elif action == "pick":
            if ground:
                self._pick()
        elif action == "potion":
            self.hp = min(1.0, self.hp + POTION_HP)
            self.note("drinks a potion")
        elif action == "chair":
            if ground and abs(self.vx) < 1:
                self.sitting = not self.sitting
                self.note("sits on the chair" if self.sitting else "stands up from the chair")

    def _released(self, action: str) -> None:
        if action in ("left", "right") and self.arrow() == 0:
            self.release_ms = self.ms
            if self.noise.walk_jitter:  # she runs on a little longer or stops a little sooner
                off = (2 * self.chance("walk") - 1) * self.noise.walk_jitter
                self.release_ms += int(round(off / WALK_SPEED * 1000))

    # --- her motion -------------------------------------------------------------------------------------

    def _ground(self, dt: float) -> None:
        p = self.plat
        if "up" in self.held and self._grab_from_ground():
            return
        if "down" in self.held and self._grab_from_top():
            return
        lying = "down" in self.held or self.sitting
        d = 0 if lying else self.arrow()
        start = max(self.go_ms, self.can_move_ms) + int(RUN_LAG * 1000)  # a standing start
        if d:
            if self.vx * d > 0:
                self.vx = d * WALK_SPEED
            elif self.vx == 0:
                if self.ms >= start:
                    self.vx = d * WALK_SPEED
            else:
                self.vx = _brake(self.vx, DECEL * dt)
        elif not lying and self.vx == 0 and self.go and start <= self.ms < self.release_ms + RELEASE_LAG * 1000:
            self.vx = self.go * WALK_SPEED  # a short tap: let go before she started, she still moves
        elif self.vx and (lying or self.ms >= self.release_ms + RELEASE_LAG * 1000):
            self.vx = _brake(self.vx, DECEL * dt)
        if not self.vx:
            return
        nx = self.x + self.vx * dt
        if nx > p.right or nx < p.left:
            right = nx > p.right
            edge = p.right if right else p.left
            if self._blocked[p.id][1 if right else 0]:
                self.x, self.vx = edge, 0.0
                self.y = p.y_at(self.x)
                return
            self._fly(self._trajectory(nx, p.y_at(edge) - 1, self.vx, 0.0), self.vx)
            return
        self.x, self.y = nx, p.y_at(nx)

    def _trajectory(self, x: float, y: float, vx: float, vy: float, skip: Platform | None = None) -> Flight:
        return trajectory(self.model, x, y, vx, vy, skip, sides=self.sides)

    def _launch(self, vx: float, vy: float) -> None:
        y = self.y if self.mode != "ground" else self.plat.y_at(self.x)
        self._fly(self._trajectory(self.x, y, vx, vy), vx)

    def _fly(self, f: Flight, vx: float) -> None:
        self.mode, self.plat, self.rope, self.flight, self.flight_ms = "air", None, None, f, self.ms
        self.vx = vx if f.path and len(f.path) > 1 else 0.0
        self.sitting = False

    def _air(self) -> None:
        f = self.flight
        tau = (self.ms + TICK_MS - self.flight_ms) / 1000
        if f.platform is not None and tau >= f.seconds:
            self._land(f.platform, f.land_x)
            return
        self.x, self.y = f.at(tau)
        if "up" in self.held and self.ms - self.left_rope_ms >= REGRAB_AFTER * 1000:
            for r in self.model.ropes:
                if abs(self.x - r.x) <= GRAB_X and r.top <= self.y and self.y - ARMS <= r.bottom:
                    self._grab(r)
                    return

    def _land(self, q: Platform, x: float) -> None:
        f = self.flight
        # her sideways speed at landing (0 if a wall stopped her)
        (x0, _), (x1, _) = f.path[-2] if len(f.path) > 1 else f.path[-1], f.path[-1]
        vx = self.vx if abs(x1 - x0) > 1e-9 else 0.0
        self.mode, self.plat, self.flight = "ground", q, None
        self.x, self.y = x, q.y_at(x)
        d = self.arrow()
        if d and vx * d > 0:
            self.vx = d * WALK_SPEED
        else:
            self.vx = vx
            self.go, self.release_ms = d, -10_000  # no arrow: she slides to a stop at once
            self.can_move_ms = max(self.can_move_ms, self.ms)

    def _grab_from_ground(self) -> bool:
        """Up at a rope's foot: onto it."""
        if self.ms - self.left_rope_ms < REGRAB_AFTER * 1000:
            return False
        for r in self.model.ropes:
            if abs(self.x - r.x) <= GRAB_X and r.top < self.y - 5 and self.y - ARMS <= r.bottom:
                self._grab(r)
                return True
        return False

    def _grab_from_top(self) -> bool:
        """Down standing on a rope's top: onto it, climbing down."""
        if self.ms - self.left_rope_ms < REGRAB_AFTER * 1000:
            return False
        for r in self.model.ropes:
            if abs(self.x - r.x) <= GRAB_X and abs(r.top - self.y) <= ROPE_TOP_REACH:
                self._grab(r)
                self.y = r.top + 1
                return True
        return False

    def _grab(self, r: Rope) -> None:
        self.mode, self.rope, self.plat, self.flight = "rope", r, None, None
        self.x, self.y, self.vx = r.x, min(max(self.y, r.top), r.bottom), 0.0
        self.sitting = False
        self.liftoff_due = False
        self.note(f"grabs the rope at x {r.x:.0f}")

    def _leave_rope(self) -> None:
        self.left_rope_ms = self.ms
        self.rope = None

    def _climb(self, dt: float) -> None:
        r = self.rope
        if "up" in self.held and "down" not in self.held:
            self.y -= CLIMB_UP * dt
            if self.y <= r.top:
                self.y = r.top
                top = next((p for p in self.model.platforms
                            if p.covers(r.x, 4) and abs(p.y_at(r.x) - r.top) <= 20), None)
                if top is not None and r.from_platform:
                    self._leave_rope()
                    self.mode, self.plat = "ground", top
                    self.x = min(max(r.x, top.left), top.right)
                    self.y, self.vx = top.y_at(self.x), 0.0
                    self.can_move_ms = self.ms + int(CLIMB_OVER * 1000)
                    self.go = self.arrow()
                    self.note(f"climbs onto platform {top.id}")
        elif "down" in self.held and "up" not in self.held:
            self.y += CLIMB_DOWN * dt
            if self.y >= r.bottom:
                self.y = r.bottom
                self._leave_rope()
                self._fly(self._trajectory(r.x, r.bottom, 0.0, 0.0), 0.0)

    # --- monsters and her -------------------------------------------------------------------------------

    def _touch(self) -> None:
        if self.ms < self.invulnerable_ms or not self.mobs or self.dead:
            return
        t = self.t
        if self.mode == "ground":
            pids = self._near[self.plat.id]
        else:
            pids = [q.id for q in self.model.platforms if q.covers(self.x, 60) and -70 <= q.y_at(self.x) - self.y <= 70]
        for pid in pids:
            for m in self.by_platform.get(pid, ()):
                if not m.alive:
                    continue
                mx = m.at(t)
                if abs(mx - self.x) >= m.half + HER_HALF:
                    continue
                my = m.y(mx)
                if self.y > my - m.height and self.y - HER_HEIGHT < my:
                    self._hit_by(m, mx)
                    return

    def _hit_by(self, m: Mob, mx: float) -> None:
        self.touches += 1
        self.invulnerable_ms = self.ms + int(INVULNERABLE * 1000)
        if self.chance("evade") < EVADE:
            self.evaded += 1
            self.note(f"touched by {m.kind} #{m.idx} at x {mx:.0f}: MISS")
            return
        self.hp = max(0.0, self.hp - TOUCH_DAMAGE)
        away = 1 if self.x >= mx else -1
        self.note(f"touched by {m.kind} #{m.idx} at x {mx:.0f}: HP {self.hp:.1%}")
        if self.hp <= 1e-9:
            self.dead, self.died_at = True, self.t
            self.held.clear()
            self.note("HP 0: she died")
            return
        if self.mode == "rope":
            self._leave_rope()
        self.liftoff_due = False
        self._launch(away * KNOCK_VX, -KNOCK_VY)

    def in_path(self, facing: str) -> tuple[Mob, float] | None:
        """The first live monster a star thrown now that way meets, and where it is: on her side of
        `facing`, within STAR_REACH, in the band SHOT_UP above to SHOT_DOWN below her; on her level at
        any distance (point-blank too), off it only from reach.min_dx sideways."""
        t, d, best = self.t, (1 if facing == RIGHT else -1), None
        for m in self.mobs:
            if not m.alive:
                continue
            mx = m.at(t)
            dx, dy = (mx - self.x) * d, m.y(mx) - self.y
            if dx < 0 or dx > reach.STAR_REACH or not -reach.SHOT_UP <= dy <= reach.SHOT_DOWN:
                continue
            if abs(dy) > reach.LEVEL and dx < reach.min_dx(dy):
                continue  # right under or over her: the star passes it
            if best is None or dx < best[0]:
                best = (dx, m, mx)
        return None if best is None else (best[1], best[2])

    def _attack(self, facing: str, air: bool) -> None:
        """The attack key's effect, STAR_DELAY after it: a star at the first monster in its path, or,
        pressed on the ground with one on her level closer than MIN_SHOT in front, a claw swipe at it."""
        first = self.in_path(facing)
        if first is None:
            self.note(f"star thrown {facing}: no hit")
            return
        m, mx = first
        dx = mx - self.x
        swipe = not air and abs(dx) < reach.MIN_SHOT and abs(m.y(mx) - self.y) <= reach.LEVEL
        what = "claw swipe" if swipe else "star"
        if not swipe and self.noise.miss and self.chance("miss") < self.noise.miss:
            self.note(f"star misses {m.kind} #{m.idx} at {dx:+.0f}")
            return
        m.hp = round(m.hp - (CLAW if swipe else 1.0), 6)
        if m.hp > 1e-9:
            self.note(f"{what} hits {m.kind} #{m.idx} at {dx:+.0f} ({m.hp:g}/{m.hp_max:g} left)")
            return
        self._kill(m, mx, what)

    def _kill(self, m: Mob, mx: float, what: str) -> None:
        t = self.t
        m.alive, m.died_at, m.died_x = False, t, mx
        self.kills[m.kind] = self.kills.get(m.kind, 0) + 1
        self.exp += self.model.mob_exp.get(m.name, 1)
        self.note(f"{what} kills {m.kind} #{m.idx} at x {mx:.0f} (platform {m.platform.id})")
        p = m.platform
        drops = []
        for kind in ["mesos"] + ([ITEM_OF[m.kind]] if m.kind in ITEM_OF else []):
            x = min(max(mx + self.rng.uniform(-10, 10), p.left + 3), p.right - 3)
            drops.append(Drop(kind, x, p.y_at(x), p.id, t + DROP_DELAY))
        self._push(self.ms + int(DROP_DELAY * 1000), "loot", drops)
        self._push(self.ms + int(RESPAWN * 1000), "respawn", m)

    def _pick(self) -> None:
        t = self.t
        near = sorted((d for d in self.drops if not d.gone and d.since <= t < d.since + DROP_LIFE
                       and d.platform == self.plat.id and abs(d.x - self.x) <= PICK_REACH),
                      key=lambda d: (abs(d.x - self.x), d.x))
        for d in near:
            if d.mesos or not self.inventory_full:
                d.gone = True
                self.picked[d.kind] = self.picked.get(d.kind, 0) + 1
                self.messages.append((t, "You have gained mesos (+15)" if d.mesos else
                                      f"{d.kind.replace('-', ' ').title()} x1 earned. (Etc)"))
                self.note(f"picks up {d.kind} at x {d.x:.0f}")
                return
        if near:
            self.messages.append((t, FULL_MESSAGE))
            self.note("can't pick up: inventory full")

    # --- what can be seen -------------------------------------------------------------------------------

    def her(self) -> tuple[float, float]:
        return self.x, self.y

    def live_drops(self) -> list[Drop]:
        """The drops lying on the ground now (landed, not taken, not vanished yet)."""
        t = self.t
        return [d for d in self.drops if not d.gone and d.since <= t < d.since + DROP_LIFE]

    def visible_mobs(self) -> list[tuple[Mob, float, float, bool]]:
        """Every monster that shows: (monster, x, y, alive); killed ones fade out where they died."""
        t, out = self.t, []
        for m in self.mobs:
            if m.alive:
                x = m.at(t)
                out.append((m, x, m.y(x), True))
            elif m.died_at is not None and t - m.died_at < FADE:
                out.append((m, m.died_x, m.y(m.died_x), False))
        return out

    def recent_messages(self) -> list[str]:
        t = self.t
        return [text for when, text in self.messages if t - when < MESSAGE_FOR]

    def summary(self) -> dict:
        """Her end state, for the trace."""
        return {"x": round(self.x, 1), "y": round(self.y, 1), "mode": self.mode, "dead": self.dead,
                "platform": None if self.plat is None else self.plat.id, "hp": round(self.hp, 3),
                "exp": self.exp, "kills": dict(sorted(self.kills.items())), "touches": self.touches,
                "evaded": self.evaded,
                "picked": dict(sorted(self.picked.items())),
                "drops_left": sorted([d.kind, d.platform, round(d.x)] for d in self.live_drops())}


def _actions(keys: GameKeys) -> dict[str, str]:
    """Key name -> what it does."""
    out = {}
    for field_name, action in (("left", "left"), ("right", "right"), ("up", "up"), ("down", "down"),
                               ("jump", "jump"), ("attack", "attack"), ("lucky_seven", "attack"),
                               ("pick_up", "pick"), ("chair", "chair"), ("hp_potion", "potion")):
        k = getattr(keys, field_name, None)
        if k is not None:
            out[k.name] = action
    return out


def _wall_at(model: MapModel, p: Platform, edge: float) -> bool:
    """Whether a wall rises at this end of platform `p` (it stops her there)."""
    level = p.y_at(edge)
    return any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
               for w in model.walls)


def _brake(v: float, by: float) -> float:
    if abs(v) <= by:
        return 0.0
    return v - by if v > 0 else v + by


def exp_pixels(exp: int) -> np.ndarray:
    """Her EXP number's text pixels (hud.EXP_NUMBER): a pattern that changes whenever her EXP does."""
    return np.random.default_rng(exp).random((26, 140)) > 0.5
