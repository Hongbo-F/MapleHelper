"""A deterministic stand-in for the game: her body, the monsters, stars, drops, HP and EXP on a real map
(mapdata.MapModel), in map units, driven by key events on a fake clock (see sim.fakes).

Her body moves as the game moves it, by motion.body's engine (the one the planner's constants come from),
on the world's 5 ms tick:

- the game sees an arrow pressed body.Lags.on (60 ms) after its key and let go Lags.off (72 ms) after,
  so a tap is as long as the game sees it (12 ms longer), and she turns when it sees the press; of the
  arrows it sees held, the one pressed last counts;
- on the ground the arrow it sees speeds her up at 1400 units/s² to physics.WALK_SPEED (turning her round
  the same way), and none brakes her at 800 (body.ground_step); walking past a platform's end she falls,
  unless a wall rises there;
- the jump key lifts her off Lags.jump (67 ms) later, if she's on the ground then, with the kick: below
  0.8 × her speed toward the arrow it sees, she gets that much more (body.takeoff). With down held it
  drops her through the platform physics.DROP_LAG later. A jump key that comes inside the attack lock is
  ignored, and so is one pressed while she's in the air (JUMP_TAKEN_LATE: or one the game takes then);
- in the air she flies body.fly's arc with the arrow the game sees, flown again from where she is when
  that arrow changes: the opposite one brakes her at 200 units/s², the one she flies with does nothing,
  and none drags her at 1 (100 once she falls at the cap). Only the walls of her foothold group stop her
  (body.OWN_GROUP_WALLS: the group of the platform she last stood on; off a rope, none), and the map's
  sides. She lands with the speed she has and slides on, unless an attack is playing: that stops her
  (jump-backs and jump-throws slid 1.4-2.9 units in game). Below the map's lowest floor she has fallen out
  of the map, and the run ends there (SimFellOut);
- up held at a rope's foot, or anywhere along it in the air, grabs it; up climbs at flow.ROPE_SPEED onto
  the platform at its top (a short climb-over, flow.CLIMB_OVER), down climbs down at physics.CLIMB_SPEED
  and lets go at its bottom; down on a rope's top grabs it from above; the jump key with an arrow jumps
  her off it sideways (ROPE_JUMP); down and jump on a floor whose foothold forbids it (forbidFallDown)
  does nothing, as on the bottom floor;
- monsters walk their platform at monster_data.MONSTER_SPEED, turning and stopping now and then (seeded);
  a hit that doesn't kill pushes one KNOCK_MOB units back from her and stands it while it flinches
  (MOB_KNOCKBACK), never off its platform. A monster touches her when its body overlaps hers (HER_HALF
  either side of her feet, HER_HEIGHT tall): the box of the frame it shows then in the game data, walking,
  standing or flinching after a hit, turned the way it faces (SimMonster.body), so a hopping slime is off
  the ground and taller for 38% of its walk. A touch leaves her a second's invulnerability. She evades it
  (a thief's "MISS": nothing else happens) with a chance inversely proportional to the monster's accuracy
  (game data: half a pig's touches, four in five of a snail's); the others take a share of her HP by the
  monster's touch damage (game data: about 1.6% for a pig, 0.3% for a snail, give or take 70%) and knock
  her back ~58 units (the median logged push; the arrow she holds doesn't steer that flight, KNOCK_STEERS).
  At 0% HP she dies, and the run ends (SimDied);
- the attack picks its target when it starts, physics.JUMP_LAG after its key (the same input lag as a
  jump's lift-off), from where she is then, and its damage lands actions.STAR_DELAY_MS after the key, on
  that monster (attack.reach), whatever the distance (in game the star is then seen flying up to 0.6 s more;
  whether its damage shows when it gets there isn't measured, so the shot log allows for both:
  behavior.fight.LANDS_FROM_S). On the ground, the nearest monster on her level (reach.LEVEL) whose body
  reaches into the claw's swing (reach.CLAW_FAR ahead of her, reach.CLAW_DOWN below to reach.CLAW_UP
  above her feet) is swiped (a fifth of a star's damage); else the star's search picks (reach.search_band:
  of the monsters whose body overlaps one of its bands, the first band's, the nearest of those), so off
  her level a star needs room sideways, by the monster's body (right under or over her, none is hit).
  In the air, the nearest whose body reaches into the attack's area (its near side within
  reach.STAR_RANGE ahead, from reach.AREA_DOWN below to reach.AREA_UP above her feet, wherever she is
  then) takes a star. For all of these, bodies are their kind's half width either side and as tall as
  their kind walking (monster_data.MONSTER_HEIGHT), taller while they flinch after a hit
  (monster_data.FLINCH_S, flinch tops from the game data); not their frames (World.top says why). So a
  jump throw's attack, starting 46 units up, passes over a snail. Neither the claw nor a star gets past a
  wall of the foothold group of the platform she last stood on, where it crosses the level line
  reach.SHOT_H above her feet (reach.wall_between): from platform 4 a throw at a monster on 2 hits
  nothing, or a monster short of the wall (live, no star from 4 ever reached one on 2). Kills give EXP
  (model.mob_exp) and, 2 s later, drops on its platform; a killed monster is seen fading out where it
  died as long as its death animation runs (MobBody.die_s: a pig 0.66 s, a snail 1.26), and respawns 8 s
  later where it spawned;
- the pick-up key takes the nearest drop within reach (items only while her inventory has room), the
  HP potion key gives 30% HP, the chair key sits her down (+3% HP/s) until an arrow stands her up.

`Noise` (off by default) adds the errors of the real game, each one seeded and tied to the moment it
happens, not to how often anything was looked at: stars that miss, walks that stop a few units early or
late, lags spread as measured (so a jump keyed 80 ms after the arrow leaves with the kick or without it),
short arrow taps that turn her without walking her, her minimap dot read to whole pixels, and spells when
her name tag or her dot can't be seen.

Time is kept in whole milliseconds on a 5 ms tick, so the state at any moment depends only on the key
events, never on when or how often it was looked at.
"""

import bisect
import functools
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np

from gamedata.paths import OUT
from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT

from .. import directions
from ..attack import reach
from ..control.actions import STAR_DELAY_MS
from ..localization import monsters as monster_data
from ..localization.mapdata import MapModel, Platform, Rope
from ..motion import body, flow
from ..motion.physics import CLIMB_SPEED, DROP_LAG, JUMP_HEIGHT, JUMP_LAG, LAGS, ROPE_REACH, SHOE, WALK_SPEED
from ..perception.kinds import kind_of

TICK_MS = 5              # the world's tick, which is also the engine's step (body.TICK)
CLIMB_UP = flow.ROPE_SPEED
CLIMB_DOWN = CLIMB_SPEED
# A climb ends in an animation onto the platform at the rope's top: flow.CLIMB_OVER is how long after the
# top she makes a standing start, which the game then sees an arrow already held for Lags.on later. So she
# stands still for both (0.29 s), and with the arrow held from before the top is running at full speed 0.35
# s after it, as in the two climbs traced on the minimap (flow).
CLIMB_OVER = flow.CLIMB_OVER + LAGS.on
ARMS = ROPE_REACH - JUMP_HEIGHT  # she grabs a rope this far above her feet
GRAB_X = 10.0            # ... within this of its x
ROPE_TOP_REACH = 15.0    # down grabs a rope whose top is within this of her platform
REGRAB_AFTER = 0.3       # s after leaving a rope before up grabs one again
# Jumping off a rope sideways: the decompile's 1.3 × her walking speed out and half a jump's speed up
# (194 and 277 units/s: Shoe.rope_vx, rope_vy). The one rope jump traced on the minimap (`ropejump`) went
# 178 ± 18 units/s sideways and rose about 16-20 units, which fits these better than what the simulator
# used before (149 out, 220 up: a 12-unit rise), by about 2 sd: weak, so False puts those back.
ROPE_JUMP_SHOE = True
ROPE_JUMP = ((SHOE.rope_vx * SHOE.vmax, SHOE.rope_vy * SHOE.vjump) if ROPE_JUMP_SHOE else (WALK_SPEED, 220.0))
HER_HALF = flow.HER_HALF
HER_HEIGHT = 55.0
# A jump key the game takes in the air does nothing. When it takes one isn't measured: the 18 jumps recorded
# lifted off 45-95 ms after their key, and a key pressed while she was in the air (or held from before) never
# jumped her again on landing. Taken when it acts on it (Lags.jump after the key, as the world's clock
# counts), a key pressed in the last 67 ms of a fall jumps her once she lands; taken as the key goes out,
# it is lost. False takes it as the key goes out, as the simulator did before it ran on the engine: the
# harsher of the two for a plan that presses the next jump early, so a plan that copes here copes in game.
JUMP_TAKEN_LATE = False
STAR_DELAY = STAR_DELAY_MS / 1000
ATTACK_LOCK = COMBAT.attack_lock_ms / 1000
# Her HP: her HP bar read "HP[642/642]" in game (2026-10-18, at level 19; maplehelper.ocr.parse's test).
MAX_HP = 642.0
# A touch takes TOUCH_SHARE of the monster's touch damage (PADamage in the game data: snail 3, slime 13,
# pig 15, Orange Mushroom 17) from her HP, give or take TOUCH_SPREAD of that (uniform, seeded by the
# touch's tick, like Noise). Fitted to lab/hits.jsonl: its 100 hits of up to 5 px on the basin and platform
# 21, where only pigs and slimes spawn, took 1 px of the 147 px HP bar 30 times, 2 px 32, 3 px 28, 4 px 9
# and 5 px once (2.19 px on average: 9.6 HP, 0.68 of their PADamage); ±70% gives 26 / 33 / 30 / 9 / 0
# (and 2 too small to show). So a pig takes 1.6% of her HP on average, a slime 1.4%, an Orange Mushroom
# 1.8%, a snail 0.3%. How the game takes her defence off isn't known: a share of PADamage is the simplest
# form that fits, and no touch by a snail or an Orange Mushroom alone has been logged.
TOUCH_SHARE = 0.68
TOUCH_SPREAD = 0.7
TOUCH_DAMAGE = 0.014     # a touch by a monster without game data: the logged hits' median (2 px, 1.4%)
# A thief's evasion: a touch shows "MISS" and takes nothing, without a knockback, with a chance of
# EVADE_ACC over the monster's accuracy (acc in the game data: snail 33, slime 53, pig 54, Orange Mushroom
# 62). Inversely proportional is the form MapleStory's dodge rule is usually given in (not measured here).
# Fitted to the 13 touches logged hopping over monsters (lab/hop_log and hop_log2: 7 MISS, 6 with damage),
# pigs by their speed (most walked at the tracker's 70 units/s clamp, which only pigs and Orange Mushrooms
# reach): a pig's touches are evaded half the time, a slime's 51%, an Orange Mushroom's 44%, a snail's 82%.
# Seeded by the touch's tick, like Noise.
EVADE_ACC = 27.0
EVADE = 0.5              # ... a monster without game data
INVULNERABLE = 1.0
# Its knockback: ~40 units in the air (0.22 s) and a ~19-unit slide on landing, ~58 in all on flat ground
# (hits.jsonl's pushes on her own level: quartiles 21 / 58 / 87 units). Those are all hits with no key
# pressed since the look before (Hunter._log_hit), so what an arrow held does to the flight isn't measured.
# By the engine's air steering the opposite arrow would brake it at 200 units/s² (45 units on flat ground
# instead of 56, toward the monster) and the one held away would carry her on, landing at full speed (65).
# False: the arrow does nothing until she lands, as the fit had it (and the simulator before it ran on the
# engine); on the ground it walks her as ever.
KNOCK_STEERS = False
KNOCK_VX, KNOCK_VY = 180.0, 220.0
HITS = {"pig": 3, "orange-mushroom": 3, "slime": 2, "snail": 1}  # HP in stars
# A hit that doesn't kill (a star or a swipe) pushes the monster KNOCK_MOB units away from her over
# KNOCK_MOB_S, and it stands in its hit1 frame until monster_data.FLINCH_S after the hit (0.6 s, that
# frame's delay); then it walks or stands as its own draws say. In the game data `pushed` is 1 for all four
# kinds: any damage pushes them. Measured once: a slime walking at her, swiped on the basin floor
# (lab/recordings/do_attack2, static camera, 30 fps), went 13 half-px (18 units) back over about 0.25 s
# and stood about 0.6 s before it walked at her again; a pig hit in `fight1` lost about 80 units of its
# walk (moving camera, so no distance). One sample, so False puts the monsters back to walking through
# hits. Pushed from where it turns (its body at the platform's end), a monster stays on its platform:
# none is pushed off an end (pigs have been seen on platforms where none spawn; how they get there isn't
# known).
MOB_KNOCKBACK = True
KNOCK_MOB = 18.0
KNOCK_MOB_S = 0.25
CLAW = 0.2               # a claw swipe's damage, in stars (9-11 against a star's 30-70)
DROP_DELAY = 2.0
RESPAWN = 8.0
FADE = 0.8               # s a killed monster is still seen fading out where it died, for a kind whose death
                         # animation the game data doesn't have (else its own: MobBody.die_s, 0.66-1.26 s)
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


class SimFellOut(SimDied):
    """She fell below the map's lowest floor (a knock or a flight through a wall of another foothold group
    with nothing under it): she'd fall for good, and a simulated run ends there, named, not left standing
    still in the air for the rest of it. What the game does then isn't known (the map's bottom isn't a
    floor in its data)."""


TAP_LOST_MS = 50         # an arrow tap this short (as sent) can be lost (Noise.tap_lost)


@dataclass(frozen=True)
class Noise:
    """The real game's errors, all off by default. Each draw is seeded by the world's seed and the
    moment it's for (a star's tick, a key's tick), and the spells are fixed stretches of time, so a
    run is still the same every time, and a change that only adds or drops a look changes nothing here
    (detect_miss is drawn per look, so with it on, an extra look does change what later looks see)."""
    miss: float = 0.0            # chance a star that would hit misses (lab/shots.jsonl: 72.5% of 204 hit)
    walk_jitter: float = 0.0     # units: letting go of an arrow stops her up to this much early or late
    lag_sd: float = 0.0          # the lags' spread around their medians, times the measured one (body.Lags:
                                 # sd 10 ms for an arrow pressed, 11 let go, 12 for a jump), drawn for each
                                 # key; 1: as measured. 0: every lag is its median. (Spread alone, few taps
                                 # are lost, seen let go before they're seen pressed: of 200 taps, 36 of 20
                                 # ms moved her under half a unit, 5 of 39 ms, none of 50 ms)
    tap_lost: float = 0.0        # chance an arrow tap of TAP_LOST_MS or less turns her without walking her
                                 # (the game sees it pressed, then let go at once): in the recordings, of
                                 # five taps of 20-50 ms, three moved her 0 (0.6; research 4.7)
    dot_px: bool = False         # her minimap dot read to whole pixels (11.4 map units each), not exact
    tag_hidden: tuple = ()       # (from, to) s since the start: her name tag can't be seen (behind a box)
    dot_hidden: tuple = ()       # (from, to) s: her minimap dot can't be seen (a tooltip over the minimap)
    hit_hidden: tuple = ()       # (shortest, longest) s a monster hit by a star (and not killed) can't be
                                 # seen: in game it flinches under its damage numbers and the detector loses
                                 # it for 1-3 s (README); () off
    detect_miss: float = 0.0     # chance a look doesn't find a monster that shows (per monster and look)

    @staticmethod
    def during(spells: tuple, t: float) -> bool:
        return any(a <= t < b for a, b in spells)


@dataclass(frozen=True)
class Realism:
    """How busy and how hard the map is. The defaults are the simulator's own world (every spawn point
    filled, an 8 s respawn, nobody else hunting, endless potions), which is denser and easier than the
    game: tour_run makes about 143 EXP a minute where a sweep in game made about 76. The other values are
    knobs for comparing policies in a world closer to the game's; none of them is measured."""
    respawn: float = RESPAWN     # s from a kill until the monster is back at its spawn point
    spawn_share: float = 1.0     # share of the map's spawn points that have a monster (seeded)
    rival_every: float = 0.0     # s between another player's kills, on average (0: nobody else)
    rival_owns: float = 15.0     # s a rival's drops can't be picked up by her
    rival_takes: float = 20.0    # s after landing that a rival picks up its own drops
    potions: int | None = None   # potions she has (None: endless)
    lock_arrows: bool = False    # arrows don't move her during the attack lock (unmeasured in game;
                                 # Actions.walk waits the lock out, the default world lets her walk)


# --- monsters ----------------------------------------------------------------------------------------

MOBS = OUT / "mobs"      # what the simulator reads of each monster from the game data, cached as JSON (as
                         # mapdata caches maps): the simulator runs the same off the caches alone
FRAME_MS = 100           # how long a frame shows when the game data gives it no delay (the client's default)


@dataclass(frozen=True)
class Frame:
    """One frame of a monster's animation in the game data: how long it shows (delay), and its body then
    (its box, lt/rb, which is what touches her): from `lo` to `hi` sideways of its feet as drawn, facing
    left, and from `bottom` to `top` above them (a hop lifts its bottom off the ground)."""
    ms: int
    lo: float
    hi: float
    bottom: float
    top: float


@dataclass(frozen=True)
class MobInfo:
    """A monster as the game data has it: how much a touch by it takes (PADamage, before her defence), its
    accuracy (acc), which her evasion goes against, and its frames standing, walking and flinching after a
    hit (hit1). A walk cycle can lift it: a slime is off the ground for 400 of its 1,050 ms (its box 10-17
    above its feet, up to 66 tall where it stands 42-44), a pig 4 units for 300 of 500, an Orange Mushroom
    3 for 120 of 480 (67 tall where it stands 55-58)."""
    pad: float
    acc: float
    stand: tuple[Frame, ...] = ()
    move: tuple[Frame, ...] = ()
    hit: Frame | None = None

    @staticmethod
    def frame_at(frames: tuple[Frame, ...], ms: float) -> Frame:
        """The frame an animation of `frames` (looping) shows `ms` after it began."""
        k = ms % sum(f.ms for f in frames)
        for f in frames:
            if k < f.ms:
                return f
            k -= f.ms
        return frames[-1]


@functools.lru_cache(maxsize=None)
def mob_info(mob_id: int) -> MobInfo | None:
    """Monster `mob_id`'s MobInfo, from the game data (cached in MOBS), or None where there's none."""
    cache = MOBS / f"{mob_id:07d}.json"
    raw = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
    if raw is None or "move" not in raw:  # not cached yet, or cached before its frames were kept
        raw = _read_mob(mob_id)
        if raw is None:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_suffix(f".{os.getpid()}.part")  # whole, or not there: harness workers read it too
        part.write_text(json.dumps(raw), encoding="utf-8")
        os.replace(part, cache)
    stand, move = (tuple(Frame(*f) for f in raw[k]) for k in ("stand", "move"))
    return MobInfo(raw["pad"], raw["acc"], stand or move, move or stand, raw["hit"] and Frame(*raw["hit"]))


def _read_mob(mob_id: int) -> dict | None:
    """MobInfo's fields for monster `mob_id`, from its image in the game data copy (None without it). A
    monster that borrows another's frames (info/link) takes them from that one's image; an animation
    marked zigzag plays forward then back."""
    try:
        from gamedata.canvas import _mob_pack
        from gamedata.wz import Canvas
        pack = _mob_pack()
        tree = pack.image(f"Mob/{mob_id:07d}.img")
        info = tree.get("info") or {}
        if info.get("link") and not tree.get("stand"):
            tree = pack.image(f"Mob/{int(info['link']):07d}.img")
    except (OSError, KeyError, ValueError):  # no copy of the game data, or no such monster in it
        return None

    def frames(action: str) -> list[list]:
        node = tree.get(action) or {}
        out = []
        for index, f in sorted(node.items(), key=lambda kv: (len(kv[0]), kv[0])):
            lt, rb = (f.props.get("lt"), f.props.get("rb")) if isinstance(f, Canvas) else (None, None)
            if index.isdigit() and lt is not None and rb is not None:  # y is up negative in the data
                out.append([int(f.props.get("delay") or FRAME_MS), float(lt.x), float(rb.x), float(-rb.y),
                            float(-lt.y)])
        return out + out[-2:0:-1] if node.get("zigzag") else out

    hit = frames("hit1")
    return {"pad": float(info.get("PADamage") or 0), "acc": float(info.get("acc") or 0),
            "stand": frames("stand"), "move": frames("move"), "hit": hit[0] if hit else None}


def touch_damage(info: MobInfo | None, u: float) -> float:
    """The share of her HP a touch by a monster with `info` takes, `u` in [0, 1) placing it within the
    spread (TOUCH_SHARE of its touch damage, give or take TOUCH_SPREAD)."""
    if info is None or info.pad <= 0:
        return TOUCH_DAMAGE
    return info.pad * TOUCH_SHARE * (1 + TOUCH_SPREAD * (2 * u - 1)) / MAX_HP


def evade_chance(info: MobInfo | None) -> float:
    """The chance she evades a touch by a monster with `info`: EVADE_ACC over its accuracy."""
    if info is None or info.acc <= 0:
        return EVADE
    return min(1.0, EVADE_ACC / info.acc)


@dataclass
class SimMonster:
    """A monster walking its platform. Its motion is a list of segments drawn from its own rng (walk
    left, walk right or stand, for a while), evaluated lazily, so where it is at a moment doesn't depend
    on how often it was looked at. Its body at a moment is the box of the frame it shows then (body)."""
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
    hit_at: float | None = None  # when a star last hit it without killing it (Noise.hit_hidden; it flinches)
    flinch_top: float = 54.0   # how tall it is while it flinches
    fade: float = FADE         # s it's still seen once killed, fading out where it died
    info: MobInfo | None = None  # its touch damage, accuracy and frames (game data), if known
    # current motion segment: from t0 at x0, heading d (-1, 0, +1) until t1
    t0: float = 0.0
    x0: float = 0.0
    d: int = 0
    t1: float = 0.0
    facing: int = -1           # the way it faced when it last stopped walking (-1: left, as it's drawn) ...
    anim0: float = 0.0         # ... and when it began to stand or walk (its animation's start)
    pushed: tuple[float, float] | None = None  # a hit's (when, from x): pushed from there to x0 (knock)

    @property
    def lo(self) -> float:
        return self.platform.left + self.half

    @property
    def hi(self) -> float:
        return max(self.lo, self.platform.right - self.half)

    def start(self, t: float, x: float) -> None:
        self.t0, self.x0, self.anim0, self.pushed = t, min(max(x, self.lo), self.hi), t, None
        self._draw()

    def _draw(self) -> None:
        r = self.rng.random()
        self.d = 0 if r < 0.35 else (-1 if r < 0.675 else 1)
        self.t1 = self.t0 + (self.rng.uniform(0.5, 2.5) if self.d == 0 else self.rng.uniform(1.0, 4.0))

    def _fold(self, t: float) -> tuple[float, int]:
        """Where it is at t within the current segment, and which way it's then going (bouncing at the ends)."""
        if not self.d:
            if self.pushed is not None and self.pushed[0] == self.t0 and t < self.t0 + KNOCK_MOB_S:
                return self.pushed[1] + (self.x0 - self.pushed[1]) * (t - self.t0) / KNOCK_MOB_S, 0
            return self.x0, 0
        span = self.hi - self.lo
        if span <= 0:
            return self.lo, self.d
        u = (self.x0 - self.lo) + self.d * self.speed * (t - self.t0)
        m = u % (2 * span)
        return (self.lo + m, self.d) if m <= span else (self.lo + 2 * span - m, -self.d)

    def at(self, t: float) -> float:
        while t >= self.t1:
            (x, d), walked = self._fold(self.t1), self.d
            flinched = self.pushed is not None and self.pushed[0] == self.t0
            self.t0, self.x0, self.facing = self.t1, x, d or self.facing
            self._draw()
            if flinched or bool(self.d) != bool(walked):  # its stand or walk starts over (after a hit too)
                self.anim0 = self.t0
        return self._fold(t)[0]

    def y(self, x: float) -> float:
        return self.platform.y_at(x)

    def knock(self, t: float, away: int) -> None:
        """Hit at t, from her side: pushed KNOCK_MOB units `away` (+1/-1) over KNOCK_MOB_S, as far as its
        walk goes, standing until FLINCH_S after the hit (it faces the way it did). Its walk after is drawn
        then, as at the end of any of its segments."""
        self.at(t)
        x, d = self._fold(t)
        self.pushed, self.anim0, self.facing = (t, x), t, d or self.facing
        self.t0, self.x0, self.d = t, min(max(x + away * KNOCK_MOB, self.lo), self.hi), 0
        self.t1 = t + monster_data.FLINCH_S

    def body(self, t: float, flinching: bool = False) -> tuple[float, float, float, float]:
        """Its body at t: from `lo` to `hi` sideways of its feet and from `bottom` to `top` above them. With
        game data, the box of the frame it shows then (walking, standing, or its hit1 frame while it
        flinches after a hit), turned the way it faces; without, its half width either side, as tall as
        its kind (or its flinch top)."""
        self.at(t)
        if self.info is None or not self.info.move:
            return -self.half, self.half, 0.0, max(self.height, self.flinch_top) if flinching else self.height
        _, d = self._fold(t)
        if flinching and self.info.hit is not None:
            f = self.info.hit
        else:
            f = MobInfo.frame_at(self.info.move if self.d else self.info.stand, (t - self.anim0) * 1000)
        return (f.lo, f.hi, f.bottom, f.top) if (d or self.facing) < 0 else (-f.hi, -f.lo, f.bottom, f.top)


@dataclass
class Drop:
    kind: str           # "mesos" or an item kind
    x: float
    y: float
    platform: int
    since: float        # when it lands (shows) on the ground
    gone: bool = False
    owner: str = "her"  # "her", or "rival": another player's kill (Realism.rival_every)
    taken_at: float | None = None  # when it was picked up ...
    taken_by: str = ""             # ... and by whom ("her" or "rival")

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
                 noise: Noise | None = None, realism: Realism | None = None):
        """Her standing on platform start[0] at x start[1] (or, with `hanging`, on the rope nearest that x,
        her feet at y `hanging`). `monsters`: None, "all" (the map's spawns), a set of platform ids (the
        spawns on those), or [(name, platform, x)]. `drops`: [(kind, platform, x)] lying there already."""
        self.model = model
        self.origin = origin       # clock time of ms 0
        self.ms = 0
        self.seed = seed
        self.rng = random.Random(seed)
        self.noise = noise or Noise()
        self.realism = realism or Realism()
        # The knobs draw from rngs of their own, so with them off the world is the same as without them.
        self._rival_rng = random.Random(f"rival:{seed}")
        self.potions_left = self.realism.potions
        self.potions_used = 0
        self.hp_low = hp                # her lowest HP so far
        self.kill_log: list[tuple[float, str, int, str]] = []  # (t, kind, platform, "her" or "rival")
        self.touch_log: list[tuple[float, float]] = []  # (t, the share of her HP it took: 0 if evaded) for every touch
        self.actions = _actions(keys)
        self.held: dict[str, float] = {}  # action -> the clock time it went down (as sent)
        self._events: list[tuple[int, int, Event]] = []
        self._seq = 0
        self.log: list[str] = []        # notable world events, for the trace
        # her
        p = model.platform(start[0])
        self.mode = "ground"            # ground | air | rope
        self.plat: Platform | None = p
        self.last_plat: Platform = p    # the platform she last stood on: its foothold group's walls stop stars
        self.rope: Rope | None = None
        self.flight: body.Flight | None = None  # in the air: the arc she's on (body.fly, with its path) ...
        self.flight_ms = 0              # ... from this tick (its first point) ...
        self.flight_arrow = 0           # ... flown with this arrow held ...
        self.flight_skip: Platform | None = None  # ... through this platform (dropping)
        self.group = p.group            # her foothold group: the walls of it stop her in the air (body.fly)
        self.knocked = False            # in the air from a monster's touch (KNOCK_STEERS)
        self.fell_out_at: float | None = None  # when she fell below the map's lowest floor (SimFellOut)
        self.bottom = max(max(q.y_at(q.left), q.y_at(q.right)) for q in model.platforms)  # its lowest floor
        self._walls: dict = {}          # foothold group -> the walls that stop her with it (body.Walls)
        self.x, self.y = float(start[1]), p.y_at(start[1])
        self.vx = self.vy = 0.0         # her speed (units/s; y down)
        self.facing = facing
        self.seen: dict[str, int] = {}  # the arrows the game sees held -> the order it saw them pressed in
        self._seen_count = 0
        self._seen_due: dict[str, int] = {}  # each arrow's latest change the game is yet to see (its tick)
        self.still_ms = 0               # she can't move before this tick (climbing over a rope's top, standing up)
        self.left_rope_ms = -10_000
        self.attack_ms = (-10_000, -10_000)  # the last attack's animation: from its start to its lock's end
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
        self.monsters: list[SimMonster] = []
        self.by_platform: dict[int, list[SimMonster]] = {}
        self._spawn(monsters)
        self.drops: list[Drop] = [Drop(kind, x, model.platform(pid).y_at(x), pid, 0.0) for kind, pid, x in drops]
        if self.realism.rival_every > 0 and self.monsters:
            self._push(self._rival_due(), "rival")

    # --- set-up -------------------------------------------------------------------------------------

    def _spawn(self, monsters) -> None:
        if monsters is None or monsters == []:
            return
        if monsters == "all" or isinstance(monsters, (set, frozenset)):
            wanted = []
            share = random.Random(f"spawn:{self.seed}")
            for s in self.model.spawns:
                p = self.model.platform_under(s.x, s.y, reach=60)
                if p is not None and (monsters == "all" or p.id in monsters):
                    if self.realism.spawn_share < 1.0 and share.random() >= self.realism.spawn_share:
                        continue
                    wanted.append((s.name or str(s.mob_id), p.id, s.x))
        else:
            wanted = list(monsters)
        bodies = monster_data.bodies(self.model)
        ids = {kind_of(s.name): s.mob_id for s in self.model.spawns if s.name}  # by kind ...
        ids.update({s.name: s.mob_id for s in self.model.spawns if s.name})      # ... and by name
        for i, (name, pid, x) in enumerate(wanted):
            kind = kind_of(name)
            m = SimMonster(i, name, kind, self.model.platform(pid), float(x), random.Random(self.rng.random()),
                    monster_data.MONSTER_SPEED.get(kind, 70.0), monster_data.MONSTER_HALF.get(kind, 34.0),
                    monster_data.MONSTER_HEIGHT.get(kind, 46.0), float(HITS.get(kind, 2)), float(HITS.get(kind, 2)))
            if kind in bodies:
                m.flinch_top = bodies[kind].flinch_top
                m.fade = bodies[kind].die_s or FADE
            mob_id = ids.get(name, ids.get(kind))
            m.info = None if mob_id is None else mob_info(int(mob_id))
            m.start(0.0, float(x))
            self.monsters.append(m)
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
        while self.ms < end and not self.dead and self.fell_out_at is None:
            self._tick()
        if self.fell_out_at is not None:
            raise SimFellOut(f"she fell out of the map at {self.fell_out_at:.3f} s (x {self.x:.0f})")
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
                    self.held[action] = self._sent(e)
                    self._pressed(action, self.held[action])
            elif (since := self.held.pop(action, None)) is not None:
                self._released(action, self._sent(e), since)
        elif e.kind == "release":
            for action in list(self.held):
                self._released(action, self._sent(e), self.held.pop(action))
        elif e.kind == "seen":
            self._seen(*e.data)
        elif e.kind == "liftoff":
            self._liftoff(e.data)
        elif e.kind == "drop":
            if self.mode == "ground" and self.plat is not None and self.plat.id == e.data \
                    and self.plat.drop_ok(self.x):
                d = self.arrow()
                f = self._flight(self.x, self.y + 1, self.vx, 0.0, d, skip=self.plat)
                if f.platform is not None:
                    self._fly(f, self.ms, d, skip=self.plat)
                    self.note(f"drops through platform {e.data}")
        elif e.kind == "aim":
            self._aim(e.data)
        elif e.kind == "star":
            self._attack(*e.data)
        elif e.kind == "loot":
            self.drops += e.data
        elif e.kind == "respawn":
            m = e.data
            m.alive, m.hp, m.died_at, m.hit_at = True, m.hp_max, None, None
            m.start(self.t, m.home)
        elif e.kind == "rival":
            self._rival_kill()
            self._push(self._rival_due(), "rival")
        elif e.kind == "rival_takes":
            for d in e.data:
                if not d.gone:
                    d.gone, d.taken_at, d.taken_by = True, self.t, "rival"

    def _sent(self, e: Event) -> float:
        """The clock time key event `e` went out: when it was sent for (its lags count from then, not from
        the tick it falls on), or now, for one sent for a moment already past."""
        now = self.origin + self.ms / 1000
        return e.at if e.at > now - TICK_MS / 1000 else now

    def note(self, text: str) -> None:
        self.log.append(f"{self.t:8.3f} {text}")

    # --- keys -------------------------------------------------------------------------------------------

    def arrow(self) -> int:
        """The arrow the game sees held (the one it saw pressed later, if both are), as -1/0/+1."""
        if not self.seen:
            return 0
        return directions.sign(max(self.seen, key=self.seen.get))

    def _lag(self, what: str, median: float, sd: float) -> float:
        """s from a key going out (`what`, at this tick) to the game acting on it: the measured median
        (body.Lags), spread by Noise.lag_sd times the measured sd, drawn for that key and tick."""
        if not self.noise.lag_sd:
            return median
        z = random.Random(f"lag:{what}:{self.seed}:{self.ms}").gauss(0.0, 1.0)
        return max(0.0, median + max(-3.0, min(3.0, z)) * sd * self.noise.lag_sd)

    def _seen_later(self, action: str, down: bool, at: float, lag: float) -> None:
        """The game sees arrow `action` go down/up `lag` s after clock time `at`. One key's changes are
        seen in the order they happen: let go and pressed again at once, it stays held."""
        ms = max(self.ms_of(at + lag), self._seen_due.get(action, -10_000))
        self._seen_due[action] = ms
        self._push(ms, "seen", (action, down))

    def _seen(self, action: str, down: bool) -> None:
        """The game sees an arrow go down (she turns that way, and it stands her up from the chair) or up."""
        if self.dead:
            return
        if not down:
            self.seen.pop(action, None)
            return
        self._seen_count += 1
        self.seen[action] = self._seen_count
        self.facing = LEFT if action == "left" else RIGHT
        if self.sitting:
            self.sitting = False
            self.still_ms = self.ms + int(STAND_UP * 1000)

    def _pressed(self, action: str, at: float) -> None:
        ground = self.mode == "ground"
        if action in ("left", "right"):
            self._seen_later(action, True, at, self._lag(action, LAGS.on, LAGS.sd_on))
        elif action == "jump":
            if self.ms < self.attack_lock_ms:
                return
            if ground and self.sitting:
                self.sitting = False
                self.still_ms = self.ms + int(STAND_UP * 1000)
                return
            if "down" in self.held and self.mode != "rope":  # on the ground: a drop; in the air: nothing
                if ground:
                    self._push(self.ms + int(DROP_LAG * 1000), "drop", self.plat.id)
                return
            if self.mode == "air" and not JUMP_TAKEN_LATE:
                return  # pressed in the air: lost
            # It lifts her off Lags.jump later, if she's on the ground then (_liftoff); with JUMP_TAKEN_LATE,
            # one pressed just before she lands jumps her once she has.
            self._push(self.ms_of(at + self._lag("jump", LAGS.jump, LAGS.sd_jump)), "liftoff", self.mode)
        elif action == "attack":
            if self.ms < self.attack_lock_ms or self.mode == "rope":
                return
            self.attack_lock_ms = self.ms + int(ATTACK_LOCK * 1000)
            self.sitting = False
            self._push(self.ms_of(at + JUMP_LAG), "aim", self.ms)  # it starts then (_aim)
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
            if self.potions_left is not None:
                if self.potions_left <= 0:
                    self.note("no potion left")
                    return
                self.potions_left -= 1
            self.potions_used += 1
            self.hp = min(1.0, self.hp + POTION_HP)
            self.note("drinks a potion")
        elif action == "chair":
            if ground and abs(self.vx) < 1:
                self.sitting = not self.sitting
                self.note("sits on the chair" if self.sitting else "stands up from the chair")

    def _released(self, action: str, at: float, since: float) -> None:
        """Key `action`, down since clock time `since`, let go at clock time `at`."""
        if action in ("left", "right"):
            if (self.noise.tap_lost and (at - since) * 1000 <= TAP_LOST_MS + 1e-6 and action in self._seen_due
                    and self.chance("tap") < self.noise.tap_lost):
                # A short tap lost: the game sees it let go as it sees it pressed (it turns her), and it doesn't
                # walk her.
                self._push(self._seen_due[action], "seen", (action, False))
                return
            lag = self._lag(f"{action} up", LAGS.off, LAGS.sd_off)
            if self.noise.walk_jitter and not {"left", "right"} & self.held.keys():
                # letting go of the last arrow: she runs on a little longer or stops a little sooner
                lag += (2 * self.chance("walk") - 1) * self.noise.walk_jitter / WALK_SPEED
            self._seen_later(action, False, at, lag)

    # --- her motion -------------------------------------------------------------------------------------

    def _ground(self, dt: float) -> None:
        p = self.plat
        if "up" in self.held and self._grab_from_ground():
            return
        if "down" in self.held and self._grab_from_top():
            return
        lying = "down" in self.held or self.sitting
        locked = self.realism.lock_arrows and self.ms < self.attack_lock_ms
        d = 0 if lying or locked or self.ms < self.still_ms else self.arrow()
        if not d and not self.vx:
            return
        v, dx = body.ground_step(self.vx, d, SHOE, dt)
        nx = self.x + dx
        if nx > p.right or nx < p.left:
            right = nx > p.right
            edge = p.right if right else p.left
            if self._blocked[p.id][1 if right else 0]:
                # Against the wall, a hair inside the end: on her side of it, so a flight from here (even one
                # flown on in mid-air, with another arrow) goes into it only toward it (body.fly).
                self.x, self.vx = edge - (1e-6 if right else -1e-6), 0.0
                self.y = p.y_at(self.x)
                return
            # off the end: in the air from the end of this tick
            self._fly(self._flight(nx, p.y_at(edge) - 1, v, 0.0, d), self.ms + TICK_MS, d)
            return
        self.x, self.y, self.vx = nx, p.y_at(nx), v

    def _flight(self, x: float, y: float, vx: float, vy: float, arrow: int, skip: Platform | None = None,
                airborne: bool = False) -> body.Flight:
        """Her arc from (x, y) at (vx, vy) with `arrow` held: body.fly's, step by step on the world's tick,
        with the map's sides stopping her like walls (the planner never jumps at them), and the walls of
        her foothold group."""
        walls = self._walls.get(self.group)
        if walls is None:
            walls = self._walls[self.group] = body.vertical_walls(body.walls_for(self.model, self.group))
        return body.fly(self.model, x, y, vx, vy, shoe=SHOE, arrow=arrow, skip=skip, walls=walls,
                        sides=self.sides, dt=TICK_MS / 1000, path=True, airborne=airborne)

    def _steer(self) -> int:
        """The arrow that steers her in the air: the one the game sees, but none in a knock's flight (unless
        KNOCK_STEERS)."""
        return 0 if self.knocked and not KNOCK_STEERS else self.arrow()

    def _launch(self, vx: float, vy: float, from_ms: int) -> None:
        """Into the air from where she is, at (vx, vy), from tick `from_ms` (this one, or the next when it
        happens after this tick's motion)."""
        y = self.y if self.mode != "ground" else self.plat.y_at(self.x)
        d = self._steer()
        self._fly(self._flight(self.x, y, vx, vy, d), from_ms, d)

    def _fly(self, f: body.Flight, from_ms: int, arrow: int, skip: Platform | None = None) -> None:
        self.mode, self.plat, self.rope, self.flight, self.flight_ms = "air", None, None, f, from_ms
        self.flight_arrow, self.flight_skip = arrow, skip
        self.vx, self.vy = f.speeds[0]
        self.sitting = False

    def _liftoff(self, pressed_on: str) -> None:
        """The game takes a jump key pressed while she was `pressed_on` ground, in the air or on a rope:
        on the ground it jumps her, with the kick toward the arrow it sees (body.takeoff); on the rope it
        was pressed on, with an arrow, it jumps her off sideways. Anywhere else it does nothing."""
        d = self.arrow()
        if self.mode == "ground" and pressed_on != "rope":
            self.knocked = False
            self._launch(body.takeoff(self.vx, d, SHOE), -SHOE.vjump, self.ms)
        elif self.mode == "rope" and pressed_on == "rope" and d:
            self._leave_rope()
            self._launch(d * ROPE_JUMP[0], -ROPE_JUMP[1], self.ms)

    def _air(self) -> None:
        f = self.flight
        d = self._steer()
        if d != self.flight_arrow:  # the game sees another arrow: she flies on from here with it
            k = min(max(0, (self.ms - self.flight_ms) // TICK_MS), len(f.path) - 1)
            (x, y), (vx, vy) = f.path[k], f.speeds[k]
            f = self._flight(x, y, vx, vy, d, skip=self.flight_skip, airborne=k > 0)
            self._fly(f, self.ms, d, self.flight_skip)
        k = (self.ms + TICK_MS - self.flight_ms) // TICK_MS
        if k >= len(f.path) - 1:
            if f.platform is not None:
                self._land(f.platform, f.land_x, f.vx_land)
                return
            # No floor under her within the flight's 4 s (a long fall): she flies on from where it ends.
            (x, y), (vx, vy) = f.path[-1], f.speeds[-1]
            f = self._flight(x, y, vx, vy, d, skip=self.flight_skip, airborne=True)
            self._fly(f, self.ms, d, self.flight_skip)
            k = 1
        k = min(k, len(f.path) - 1)
        (self.x, self.y), (self.vx, self.vy) = f.path[k], f.speeds[k]
        if f.platform is None and self.y > self.bottom:  # below the map's lowest floor, with none to land on
            self.fell_out_at = self.t
            self.held.clear()
            self.seen.clear()
            self.note(f"falls out of the map at x {self.x:.0f} (below its lowest floor)")
            return
        if "up" in self.held and self.ms - self.left_rope_ms >= REGRAB_AFTER * 1000:
            for r in self.model.ropes:
                if abs(self.x - r.x) <= GRAB_X and r.top <= self.y and self.y - ARMS <= r.bottom:
                    self._grab(r)
                    return

    def _land(self, q: Platform, x: float, vx: float) -> None:
        """Down on platform q at x, going sideways at vx (0 if a wall stopped her): she slides on as the
        ground brakes her (or runs on, with the arrow), except while an attack plays: it stops her. Five
        jump-backs and jump-throws landing during their attack slid 1.4-2.9 units, where braking alone
        has 7.6-11.7 (research: 60 fps recordings)."""
        self.mode, self.plat, self.flight, self.last_plat, self.group, self.knocked = "ground", q, None, q, q.group, False
        self.x, self.y = x, q.y_at(x)
        start, end = self.attack_ms
        self.vx, self.vy = (0.0 if start <= self.ms < end else vx), 0.0

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
        self.mode, self.rope, self.plat, self.flight, self.group = "rope", r, None, None, body.ROPE_GROUP
        self.knocked = False
        self.x, self.y, self.vx, self.vy = r.x, min(max(self.y, r.top), r.bottom), 0.0, 0.0
        self.sitting = False
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
                    self.mode, self.plat, self.last_plat, self.group = "ground", top, top, top.group
                    self.x = min(max(r.x, top.left), top.right)
                    self.y, self.vx = top.y_at(self.x), 0.0
                    self.still_ms = self.ms + int(round(CLIMB_OVER * 1000))
                    self.note(f"climbs onto platform {top.id}")
        elif "down" in self.held and "up" not in self.held:
            self.y += CLIMB_DOWN * dt
            if self.y >= r.bottom:
                self.y = r.bottom
                self._leave_rope()
                self._launch(0.0, 0.0, self.ms + TICK_MS)  # falling from the end of this tick

    # --- monsters and her -------------------------------------------------------------------------------

    def _touch(self) -> None:
        if self.ms < self.invulnerable_ms or not self.monsters or self.dead:
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
                lo, hi, bottom, top = self.body(m)
                if self.x + HER_HALF <= mx + lo or self.x - HER_HALF >= mx + hi:
                    continue
                my = m.y(mx)
                if self.y > my - top and self.y - HER_HEIGHT < my - bottom:
                    self._hit_by(m, mx)
                    return

    def _hit_by(self, m: SimMonster, mx: float) -> None:
        self.touches += 1
        self.invulnerable_ms = self.ms + int(INVULNERABLE * 1000)
        evaded = self.chance("evade") < evade_chance(m.info)
        damage = 0.0 if evaded else touch_damage(m.info, self.chance("damage"))
        self.touch_log.append((self.t, damage))
        if evaded:
            self.evaded += 1
            self.note(f"touched by {m.kind} #{m.idx} at x {mx:.0f}: MISS")
            return
        self.hp = max(0.0, self.hp - damage)
        self.hp_low = min(self.hp_low, self.hp)
        away = 1 if self.x >= mx else -1
        self.note(f"touched by {m.kind} #{m.idx} at x {mx:.0f}: HP {self.hp:.1%}")
        if self.hp <= 1e-9:
            self.dead, self.died_at = True, self.t
            self.held.clear()
            self.seen.clear()
            self.note("HP 0: she died")
            return
        if self.mode == "rope":
            self._leave_rope()
        self.knocked = True
        self._launch(away * KNOCK_VX, -KNOCK_VY, self.ms + TICK_MS)  # after this tick's motion

    def flinching(self, m: SimMonster) -> bool:
        """Whether monster `m` flinches now: for FLINCH_S after a star or a swipe hit it."""
        return m.hit_at is not None and self.t - m.hit_at < monster_data.FLINCH_S

    def body(self, m: SimMonster) -> tuple[float, float, float, float]:
        """Monster `m`'s body now (SimMonster.body): its frame's box, which is what touches her."""
        return m.body(self.t, self.flinching(m))

    def top(self, m: SimMonster) -> float:
        """How tall monster `m` is for an attack to pick it: as tall as its kind walking (MONSTER_HEIGHT, as
        the decisions' reach.body_of has it), taller while it flinches. Not its frame's top (body): the star's
        search was fitted by these heights, and the decisions work out its reach by them. By its frames a pig
        standing 60 below her (38-40 tall) would be picked by the search only from 171 units sideways, not
        151, and by a jump's area (its floor 19 below her feet, reach.AREA_DOWN) not at all, where in game
        pigs 60 below were hit. The attack and the frames don't agree there yet (README: Found offline)."""
        return max(m.height, m.flinch_top) if self.flinching(m) else m.height

    def nearest_in(self, facing: str, far: float, down: float, up: float, level: bool = False) -> SimMonster | None:
        """The live monster nearest her whose body reaches into a box on her `facing` side, from her to
        `far` ahead and `down` below to `up` above her feet (reach.reaches_into); with `level`, only one
        on her level (reach.LEVEL: the claw's swing, as reach.claw_takes). Not one behind a wall of the
        foothold group she last stood on (reach.wall_between, from her height now)."""
        t, d, best = self.t, directions.sign(facing), None
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead, dy = (mx - self.x) * d, m.y(mx) - self.y
            if not reach.reaches_into(ahead, dy, self.top(m), m.half, far, down, up):
                continue
            if level and abs(dy) > reach.LEVEL:
                continue
            if reach.wall_between(self.last_plat, self.x, self.y, mx, target=m.platform) is not None:
                continue
            if best is None or abs(ahead) < best[0]:
                best = (abs(ahead), m)
        return None if best is None else best[1]

    def nearest_star(self, facing: str) -> SimMonster | None:
        """The live monster the star's search from the ground picks on her `facing` side: of those whose
        body overlaps one of its bands (reach.search_band), the first band's, the nearest of those. Not
        one behind a wall of the foothold group she last stood on (reach.wall_between)."""
        t, d, best = self.t, directions.sign(facing), None
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead = (mx - self.x) * d
            k = reach.search_band(*reach.body_box(ahead, m.y(mx) - self.y, self.top(m), m.half))
            if k is None or reach.wall_between(self.last_plat, self.x, self.y, mx, target=m.platform) is not None:
                continue
            if best is None or (k, abs(ahead)) < best[0]:
                best = ((k, abs(ahead)), m)
        return None if best is None else best[1]

    def _aim(self, key_ms: int) -> None:
        """The attack keyed at `key_ms` starts: it picks its target from where she is now, and its damage
        lands STAR_DELAY after the key (_attack). On the ground, the claw swipes the nearest monster on her
        level its swing reaches; else the star's search picks (nearest_star). In the air, the nearest in
        the attack's area takes the star."""
        self.attack_ms = (self.ms, key_ms + int(ATTACK_LOCK * 1000))  # its animation plays until the lock ends
        if self.mode == "ground":
            target = self.nearest_in(self.facing, reach.CLAW_FAR, reach.CLAW_DOWN, reach.CLAW_UP, level=True)
            swipe = target is not None
            if target is None:
                target = self.nearest_star(self.facing)
        else:
            target = self.nearest_in(self.facing, reach.STAR_RANGE, reach.AREA_DOWN, reach.AREA_UP)
            swipe = False
        self._push(key_ms + int(STAR_DELAY * 1000), "star", (self.facing, target, swipe))

    def _attack(self, facing: str, m: SimMonster | None, swipe: bool) -> None:
        """The attack's damage, STAR_DELAY after its key: on the monster it picked when it started (_aim),
        a claw swipe or a star."""
        if m is None or not m.alive:
            self.note(f"star thrown {facing}: no hit")
            return
        mx = m.at(self.t)
        dx = mx - self.x
        what = "claw swipe" if swipe else "star"
        if not swipe and self.noise.miss and self.chance("miss") < self.noise.miss:
            self.note(f"star misses {m.kind} #{m.idx} at {dx:+.0f}")
            return
        m.hp = round(m.hp - (CLAW if swipe else 1.0), 6)
        if m.hp > 1e-9:
            m.hit_at = self.t
            if MOB_KNOCKBACK:
                m.knock(self.t, 1 if mx >= self.x else -1)
            self.note(f"{what} hits {m.kind} #{m.idx} at {dx:+.0f} ({m.hp:g}/{m.hp_max:g} left)")
            return
        self._kill(m, mx, what)

    def _kill(self, m: SimMonster, mx: float, what: str) -> None:
        t = self.t
        m.alive, m.died_at, m.died_x = False, t, mx
        self.kills[m.kind] = self.kills.get(m.kind, 0) + 1
        self.kill_log.append((t, m.kind, m.platform.id, "her"))
        self.exp += self.model.mob_exp.get(m.name, 1)
        self.note(f"{what} kills {m.kind} #{m.idx} at x {mx:.0f} (platform {m.platform.id})")
        p = m.platform
        drops = []
        for kind in ["mesos"] + ([ITEM_OF[m.kind]] if m.kind in ITEM_OF else []):
            x = min(max(mx + self.rng.uniform(-10, 10), p.left + 3), p.right - 3)
            drops.append(Drop(kind, x, p.y_at(x), p.id, t + DROP_DELAY))
        self._push(self.ms + int(DROP_DELAY * 1000), "loot", drops)
        self._push(self.ms + int(self.realism.respawn * 1000), "respawn", m)

    def _rival_due(self) -> int:
        """The tick of another player's next kill: every Realism.rival_every s on average."""
        return self.ms + int(self.realism.rival_every * self._rival_rng.uniform(0.5, 1.5) * 1000)

    def _rival_kill(self) -> None:
        """Another player kills a monster somewhere on the map: no EXP for her, and drops she can't pick
        up for Realism.rival_owns s, which the rival takes Realism.rival_takes s after they land."""
        live = [m for m in self.monsters if m.alive]
        if not live:
            return
        m = live[self._rival_rng.randrange(len(live))]
        t, mx = self.t, m.at(self.t)
        m.alive, m.died_at, m.died_x = False, t, mx
        self.kill_log.append((t, m.kind, m.platform.id, "rival"))
        self.note(f"another player kills {m.kind} #{m.idx} at x {mx:.0f} (platform {m.platform.id})")
        p, drops = m.platform, []
        for kind in ["mesos"] + ([ITEM_OF[m.kind]] if m.kind in ITEM_OF else []):
            x = min(max(mx + self._rival_rng.uniform(-10, 10), p.left + 3), p.right - 3)
            drops.append(Drop(kind, x, p.y_at(x), p.id, t + DROP_DELAY, owner="rival"))
        self._push(self.ms + int(DROP_DELAY * 1000), "loot", drops)
        self._push(self.ms + int((DROP_DELAY + self.realism.rival_takes) * 1000), "rival_takes", drops)
        self._push(self.ms + int(self.realism.respawn * 1000), "respawn", m)

    def _pick(self) -> None:
        t = self.t
        near = sorted((d for d in self.drops if not d.gone and d.since <= t < d.since + DROP_LIFE
                       and d.platform == self.plat.id and abs(d.x - self.x) <= PICK_REACH
                       and (d.owner == "her" or t >= d.since + self.realism.rival_owns)),
                      key=lambda d: (abs(d.x - self.x), d.x))
        for d in near:
            if d.mesos or not self.inventory_full:
                d.gone, d.taken_at, d.taken_by = True, t, "her"
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

    def visible_monsters(self) -> list[tuple[SimMonster, float, float, bool]]:
        """Every monster that shows: (monster, x, y, alive); killed ones fade out where they died."""
        t, out = self.t, []
        for m in self.monsters:
            if m.alive:
                x = m.at(t)
                out.append((m, x, m.y(x), True))
            elif m.died_at is not None and t - m.died_at < m.fade:
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
                "evaded": self.evaded, **({"fell_out_at": round(self.fell_out_at, 3)} if self.fell_out_at else {}),
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
    """Whether a wall of its foothold group rises at this end of platform `p` (it stops her there)."""
    level = p.y_at(edge)
    return any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
               for w in body.walls_for(model, p.group))


def exp_pixels(exp: int) -> np.ndarray:
    """Her EXP number's text pixels (hud.EXP_NUMBER): a pattern that changes whenever her EXP does."""
    return np.random.default_rng(exp).random((26, 140)) > 0.5
