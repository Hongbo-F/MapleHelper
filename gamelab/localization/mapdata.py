"""A map's layout, read from the game's own data (gamedata): platforms, walls, ropes, portals, monster
spawns (and each monster's EXP, body, speed, touch damage and whether it hops) and the minimap's scale, in the
game's map units ("world" units: x to the right, y downward).
Also which platforms monsters can reach (`MapModel.safe`), and a picture of the map (`render`).

The game lays out a 1366x768 view, one game px to a world unit, and shows it stretched to its window: 1080/768 =
1.40625x larger at 1920x1080, pixel for pixel at its own 1366x768. So world units are converted with the look's
screen.Geometry wherever screen pixels are involved.

Game data stays on this computer (gamedata/data, git-ignored); nothing here is committed.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from gamedata.paths import COPY, OUT
from gamedata.wz import Package, jsonable


MONSTER_REACH = 55          # map units a monster's body reaches above its platform (pigs, mushrooms)
CACHE = OUT / "maps"         # parsed maps, as JSON (git-ignored, like all extracted game data)

FEET_BELOW = 8               # map units her feet may sink below a platform's surface and still stand on it
# A monster walks at the game's walkSpeed (Physics.img: 125 units/s) times 1 + its speed stat / 100 (Mob/<id>.img
# info/speed: pigs and Orange Mushrooms 0, slimes -20, snails -65): 125, 125, 100 and 43.75 units/s. Measured on
# recordings with the camera still: snails 42 and 45 units/s, pigs 120-128 (camera motion taken out), slimes
# 77-106 (they hop as they go).
MOB_WALK_SPEED = 125.0       # walkSpeed, when the map's physics don't say
MIN_FALL = 2                 # a platform must be this far below her to be one she falls onto
# A twin: a platform lying on another's footholds, within its ends and TWIN_DY of its surface all along (less than
# a fall: MIN_FALL), is one place with it (MapModel.twins). On West Rocky Mountain II two do, each in the same foothold
# group as the one it lies on: 11 (-1426..-1171) on 14's foothold from -1484 to -1171, 21 (-1404..-1149) on 22's
# from -1462 to -1149, 0.00-0.15 units above them. Each is a chain of its own whose last foothold joins the long
# one's next (11's 119 -> 14's 88, 21's 76 -> 22's 73), with a free end on the left: she lands on the twin (it's the
# higher), walks off its right end onto the long one without a step, and off its left end drops 0.15 units onto it;
# from the long one's right part walking left she's on the twin (88's prev is 119). For everything the stack does she
# is on the long one: a landing on either is arriving, and moves between them are walking. None on the first map.
TWIN_DY = MIN_FALL
SPAWN_REACH = 60             # how far above its platform a spawn point may be placed
# The most monsters the server keeps on a map (MapModel.mob_cap), as OrionAlpha (github.com/EricSoftTM/OrionAlpha,
# KMS-1.23, a replica of Nexon's early Korean server) has it: LifePool.init, one monster in CAP_AREA square units of the
# box about the map's footholds as WvsPhysicalSpace2D makes it (each foothold's x CAP_BOX_X inside, its top CAP_BOX_UP
# over it, its bottom CAP_BOX_DOWN under it; within the VR bounds with VRLimit; CAP_BOX_GROW more each side), its height
# less CAP_SCREEN_DY and both at least the 800 x 600 screen (CAP_SCREEN), times the map's mobRate, 1 to CAP_MOST;
# tryCreateMob: with more than half that many users on the map, more, up to twice (OrionAlpha's `1 << cap` read as
# `cap << 1`: RustMS's "up to twice as many"). Tree Dungeon II 22, Thicket Around the Beach II 26, West Rocky Mountain
# II 23.
# The 11-15 counted with her alone on Tree Dungeon (sim.world's top) were of the parts of the map she saw.
CAP_AREA = 128000.0          # square map units a monster (OrionAlpha's 0.0000078125 a square px)
CAP_MOST = 40
CAP_BOX_X, CAP_BOX_UP, CAP_BOX_DOWN, CAP_BOX_GROW = 30, 300, 10, 10
CAP_SCREEN = (800, 600)
CAP_SCREEN_DY = 450


@dataclass(frozen=True)
class Segment:
    """A straight foothold (or wall) from (x1, y1) to (x2, y2). `group`: the foothold's (layer, group) in
    the map image (foothold/<layer>/<group>/<id>). Each floating block is one group: its floor and the
    walls hanging from its ends; steps joined by a wall share one. The client's collisions go by it
    (walls of another group are passed through in the air), and so, by the live logs, does what stops a
    star (attack.reach). `no_drop`: the game won't drop her through this foothold (forbidFallDown, on
    14,197 footholds in 51 of the 306 map images; none on this map). `prev`, `next`: the footholds joined
    to its start and its end, as (x1, y1, x2, y2) (the map image's prev and next; None at a free end):
    where a star's line runs exactly through a foothold's end, the client asks which side of them it ends
    on (attack.reach.sight_stop)."""
    x1: float
    y1: float
    x2: float
    y2: float
    group: tuple[int, int] | None = field(default=None, compare=False, repr=False)
    no_drop: bool = field(default=False, compare=False, repr=False)
    prev: tuple[float, float, float, float] | None = field(default=None, compare=False, repr=False)
    next: tuple[float, float, float, float] | None = field(default=None, compare=False, repr=False)

    def y_at(self, x: float) -> float:
        if self.x2 == self.x1:
            return self.y1
        t = (x - self.x1) / (self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)


@dataclass
class Platform:
    """A walkable stretch of ground: footholds joined end to end, left to right, with no wall between.
    `walls`: the walls of its foothold group (MapModel.walls_of; parse attaches them): the walls that stop
    a star thrown from it (attack.reach). `floors`: every floor of the map (all its platforms' footholds,
    one tuple for every platform; parse attaches it): what can stop a star's line going down, by the v83
    client's rule (attack.reach.LINE). `peers`: every platform of the map, by id (parse attaches it): a
    monster's own platform, by its id (Monster.platform), where a star at it is checked (attack.reach.wall_to)."""
    id: int
    segments: list[Segment]
    walls: tuple[Segment, ...] = field(default=(), repr=False, compare=False)
    floors: tuple[Segment, ...] = field(default=(), repr=False, compare=False)
    peers: tuple["Platform", ...] = field(default=(), repr=False, compare=False)

    def peer(self, pid: int | None) -> "Platform | None":
        """Platform `pid` of its map (peers), or None if that isn't known (no id, or no map attached)."""
        return self.peers[pid] if pid is not None and 0 <= pid < len(self.peers) else None

    @property
    def left(self) -> float:
        return self.segments[0].x1

    @property
    def right(self) -> float:
        return self.segments[-1].x2

    @property
    def top(self) -> float:
        return min(min(s.y1, s.y2) for s in self.segments)

    @property
    def bottom(self) -> float:
        """The y of its lowest point (y down: the most of its footholds' ends), where a floor dipping between its ends
        is lower than either (Tree Dungeon's 13: -1182 and -1200 at its ends, -1177 at x -86)."""
        return max(max(s.y1, s.y2) for s in self.segments)

    @property
    def group(self) -> tuple[int, int] | None:
        """Its footholds' (layer, group) (Segment.group): platforms joined by a step wall share one."""
        return self.segments[0].group

    @property
    def no_drop(self) -> bool:
        """None of it can be dropped through (every foothold forbids it: Segment.no_drop). In the game
        data a platform's footholds almost always agree: of 2,100 platforms on the 51 maps that have the
        flag, 1,340 forbid it all along, 757 nowhere and 3 in part."""
        return all(s.no_drop for s in self.segments)

    def drop_ok(self, x: float, room: float = 0.0) -> bool:
        """Whether she can drop through it at x (the foothold there doesn't forbid it), and, with `room`,
        anywhere within that of x (no foothold that forbids it reaches nearer)."""
        if room > 0 and any(s.no_drop for s in self.segments if s.x1 < x + room and s.x2 > x - room):
            return False
        for s in self.segments:
            if s.x1 <= x <= s.x2:
                return not s.no_drop
        return not (self.segments[0] if x < self.left else self.segments[-1]).no_drop

    def y_at(self, x: float) -> float:
        for s in self.segments:
            if s.x1 <= x <= s.x2:
                return s.y_at(x)
        return self.segments[0].y1 if x < self.left else self.segments[-1].y2

    def covers(self, x: float, margin: float = 0) -> bool:
        return self.left - margin <= x <= self.right + margin

    def clamp(self, x: float) -> float:
        """x, moved onto the platform if it's past either end."""
        return min(max(x, self.left), self.right)


def _lies_on(a: Platform, b: Platform) -> bool:
    """Whether platform a's surface is within TWIN_DY of b's all along a (at a's ends and every foothold end of
    either between them)."""
    xs = {a.left, a.right} | {v for s in a.segments + b.segments for v in (s.x1, s.x2) if a.left < v < a.right}
    return all(abs(a.y_at(x) - b.y_at(x)) <= TWIN_DY for x in xs)


def runs_under(other: Platform, p: Platform, reach: float = MONSTER_REACH) -> bool:
    """Whether platform `other` runs under `p` within `reach`, judged at the middle of where they overlap:
    monsters walking on `other` can touch her standing on `p` (their bodies stick up that far)."""
    lo, hi = max(p.left, other.left), min(p.right, other.right)  # where they overlap
    mid = (lo + hi) / 2
    return lo < hi and 0 < other.y_at(mid) - p.y_at(mid) <= reach


@dataclass(frozen=True)
class Rope:
    """A rope or ladder."""
    x: float
    top: float       # y1: the upper end (where it hangs from)
    bottom: float    # y2
    ladder: bool     # "l": a ladder, not a rope
    from_platform: bool  # "uf": she can climb onto the platform at its top


@dataclass(frozen=True)
class Portal:
    name: str
    kind: int
    x: float
    y: float
    to_map: int | None
    image: str = ""   # the map's `image`: which of MapHelper.img's portal pictures it's drawn with ("": PORTAL_IMAGE)


# Where a portal takes her (HeavenClient, Gameplay/MapleMap/Portal.cpp and Stage.cpp, read 2026-10-24): while up is held
# and she isn't climbing, sitting or attacking (Stage::update; no ground check, so in the air too), with her feet within
# PORTAL_BOX[0] of its x and from PORTAL_BOX[1] above its point to PORTAL_BOX[2] below (Portal::bounds: (-25, -100) to
# (25, 25)). Whether the game itself takes one in the air isn't known; the simulator does (sim.world). A portal of
# PORTAL_EXIT's kind leads to another map (the spawn points, kind 0, lead nowhere).
PORTAL_BOX = (25.0, 100.0, 25.0)
PORTAL_EXIT = 2
# A portal's light (`portal_light`, `portal_lights`): the client draws a visible portal (PORTAL_SHOWN's kinds: an exit,
# a changeable one and a script one, RustMS's MapPortals.cpp, its port of the decompiled v83 client, `visible = type ==
# Portal::REGULAR || type == Portal::TYPE4 || type == Portal::SCRIPTED`, read 2026-10-31) with Map/MapHelper.img
# portal/game/pv/<its image>'s frames, one after another, each picture with its origin on the portal's point, over her:
# its pulsing pillar of light (PORTAL_IMAGE's where the map names none: 464 of the game data's 466 exits; 2 name "2").
# A hidden portal (PORTAL_HIDDEN's kinds) draws ph/<image>'s (a script one psh/<image>'s) portalStart, then its
# portalContinue, while her feet are within its reveal box (Portal::reveal_bounds: 100 either side, 50 up and down; of
# two holding her, the one of the higher id: not told apart here, none of the game data's 9 within another's); her body
# in its light as locate.Cover has it (her feet then within 16.5 across, 43 up and 1 down of its point) has her in that
# box. The spawn points and the invisible portals draw nothing.
# Its light where it's over her (locate.Cover: her body's box all inside its lit px' box, her feet in its dense
# rows'): a px of a frame lit with its alpha over LIGHT_ALPHA (a quarter opaque), the dense rows those of the frames'
# union lit across LIGHT_DENSE of its lit width or more. A glow fainter than that (alpha 51 at the most) reaches the
# pictures' tops, the default's 248 px over its point (lit to 200), and over 116 up only streaks are lit (27% of its
# width at the most in some frame, 6% in one on average; its dense rows, 110 up to 7 under the point, 52-100% and
# 15-94%): her feet there, her lower body is under its dense light (at Tree Dungeon's out01, her floor 6.7 over its
# point, its foot over her plate's top rows too). Its pictures' union had had her feet counted in it up to 175 px over
# its point, on a platform above (the review of 2026-10-31: Tree Dungeon, Forest Up North IV's platform 8, 156 over
# its out01; 51 platforms' stretches in 26 maps, most 120-160 over their portal's point; now 29 in 14 maps, 32-97 over
# it).
PORTAL_SHOWN = (PORTAL_EXIT, 4, 7)
PORTAL_HIDDEN = {10: "ph", 11: "psh"}
PORTAL_IMAGE = "default"
LIGHT_ALPHA, LIGHT_DENSE = 64, 0.5


@dataclass(frozen=True)
class MobBody:
    """A monster's body, from the boxes (lt/rb, map units from its feet) of its frames in the game data
    (Mob/<id>.img): how tall it is at least while it stands or walks (the lowest top of its stand and move
    frames), how tall while it flinches after a hit (hit1), and half its width (its widest stand or move
    frame). And how long it's seen dying, fading out where it died: its die1 frames' delays, in s (0 if
    the data has none)."""
    top: float
    flinch_top: float
    half: float
    die_s: float = 0.0


@dataclass(frozen=True)
class Spawn:
    mob_id: int
    x: float
    y: float
    left: float      # "rx0", "rx1": the x range the game data gives the spawn
    right: float
    name: str = ""


@dataclass
class MapModel:
    """One map, as the game data describes it. Platform ids are their index in `platforms`."""
    id: int
    name: str
    bounds: tuple[float, float, float, float]      # VR left, top, right, bottom
    minimap: dict                                    # width, height, centerX, centerY, mag
    platforms: list[Platform]
    walls: list[Segment]
    ropes: list[Rope]
    portals: list[Portal]
    spawns: list[Spawn] = field(default_factory=list)
    physics: dict = field(default_factory=dict)      # the game's Physics.img (motion.body.PHYSICS_IMG is a
                                                     # copy, which a test checks against this)
    mob_exp: dict[str, int] = field(default_factory=dict)  # monster name -> EXP per kill (game data)
    mob_bodies: dict[str, MobBody] = field(default_factory=dict)  # monster name -> its body (game data)
    mob_speeds: dict[str, float] = field(default_factory=dict)    # monster name -> how fast it walks (units/s;
                                                                  # game data: mob_walk_speed)
    mob_damage: dict[str, float] = field(default_factory=dict)    # monster name -> its touch damage (PADamage, before
                                                                  # her defence: attack.danger)
    mob_hops: dict[str, bool] = field(default_factory=dict)       # monster name -> whether it hops (a jump action in
                                                                  # its Mob data): it can leave its platform
    mob_flies: dict[str, bool] = field(default_factory=dict)      # monster name -> whether it flies (a fly action):
                                                                  # no floor holds it (none on her three maps)
    mob_tall: dict[str, float] = field(default_factory=dict)      # monster name -> the highest its body reaches
                                                                  # above its feet, a hop's frames too (a slime 66,
                                                                  # 42 standing): what a rope above it must clear
    mob_hp: dict[str, float] = field(default_factory=dict)        # monster name -> its HP (maxHP) ...
    mob_pdd: dict[str, float] = field(default_factory=dict)       # ... and its defence against her stars (PDDamage):
                                                                  # what her throws must take (attack.choice)
    mob_mdd: dict[str, float] = field(default_factory=dict)       # ... against magic (MDDamage: a magician's Magic Claw;
                                                                  # a Green Mushroom 30, a Horny Mushroom 0) ...
    mob_eva: dict[str, float] = field(default_factory=dict)       # ... its avoid (eva) and level: what a miss is
    mob_level: dict[str, float] = field(default_factory=dict)     # reckoned from (nothing reads these three yet)
    mob_rate: float = 1.0                                         # info's mobRate, and whether its VR bounds limit
    vr_limit: bool = False                                        # the server's box of it (VRLimit): mob_cap

    # --- one place: twins -----------------------------------------------------------------------

    @cached_property
    def twins(self) -> dict[int, int]:
        """Platform id -> the platform it lies on (TWIN_DY): a twin is that platform for everything the stack does
        (where she is, landings, routes, spawns). The longer of two is the one stood for (if as long, the one of
        lower id); one lying on a twin is the place that twin stands for."""
        def size(p: Platform) -> tuple[float, int]:
            return p.right - p.left, -p.id

        on: dict[int, int] = {}
        for a in self.platforms:
            hosts = [b for b in self.platforms if b is not a and b.left <= a.left and a.right <= b.right
                     and size(b) > size(a) and _lies_on(a, b)]
            if hosts:
                on[a.id] = max(hosts, key=size).id
        for a in list(on):  # one lying on a twin: the place that twin stands for
            while on[a] in on:
                on[a] = on[on[a]]
        return on

    def place_of(self, pid: int) -> int:
        """The platform `pid` is, for everything the stack does: the one it lies on if it's a twin, else itself."""
        return self.twins.get(pid, pid)

    @cached_property
    def places(self) -> list[Platform]:
        """The platforms that are places of their own (all but twins), in id order: where she can be found to be,
        land, stand and go. Every platform on a map without twins."""
        return [p for p in self.platforms if p.id not in self.twins] if self.twins else self.platforms

    # --- where things are -------------------------------------------------------------------

    def platform_under(self, x: float, y: float, reach: float = 40, slack: float = 15) -> Platform | None:
        """The platform her feet are on (or just above): the closest one at or below y, within reach.
        `slack` allows for x being a little off (the minimap dot is good to ~11 map units). A twin is never
        the answer (MapModel.twins): the platform it lies on is."""
        best = None
        for p in self.places:
            if not p.covers(x, slack):
                continue
            gap = p.y_at(x) - y
            if -FEET_BELOW <= gap <= reach and (best is None or gap < best[0]):
                best = (gap, p)
        return best[1] if best else None

    def platform_below(self, x: float, y: float) -> Platform | None:
        """The first platform she'd land on falling from (x, y) (never a twin: the one it lies on)."""
        best = None
        for p in self.places:
            if p.covers(x):
                gap = p.y_at(x) - y
                if gap > MIN_FALL and (best is None or gap < best[0]):
                    best = (gap, p)
        return best[1] if best else None

    def platform(self, pid: int) -> Platform:
        return self.platforms[pid]

    def walls_of(self, group: tuple[int, int] | None) -> list[Segment]:
        """The walls of a foothold group (Segment.group): those hanging from a block's ends, and the steps
        joining platforms of one group."""
        return self._walls_by_group.get(group, []) if group is not None else []

    @cached_property
    def _walls_by_group(self) -> dict[tuple[int, int], list[Segment]]:
        out: dict[tuple[int, int], list[Segment]] = {}
        for w in self.walls:
            if w.group is not None:
                out.setdefault(w.group, []).append(w)
        return out

    # --- monsters -----------------------------------------------------------------------------

    def mob_cap(self, users: int = 1) -> int:
        """The most monsters the server keeps on the map with `users` players there (see CAP_AREA): its respawn fills
        it to this (sim.world, behavior.patrol)."""
        segs = [s for p in self.platforms for s in p.segments] + list(self.walls)
        if not segs:
            return 1
        left = min(min(s.x1, s.x2) + CAP_BOX_X for s in segs)
        right = max(max(s.x1, s.x2) - CAP_BOX_X for s in segs)
        top = min(min(s.y1, s.y2) - CAP_BOX_UP for s in segs)
        bottom = max(max(s.y1, s.y2) + CAP_BOX_DOWN for s in segs)
        vl, vt, vr, vb = (b or 0 for b in self.bounds)
        if self.vr_limit:
            left = vl + 20 if vl > 0 and left < vl + 20 else left
            right = vr if vr > 0 and right > vr else right
            top = vt + 65 if vt > 0 and top < vt + 65 else top
            bottom = vb if vb > 0 and bottom > vb else bottom
        w = right - left + 2 * CAP_BOX_GROW
        h = bottom - top + 2 * CAP_BOX_GROW
        least = min(CAP_MOST, max(1, int(max(h - CAP_SCREEN_DY, CAP_SCREEN[1]) * max(w, CAP_SCREEN[0])
                                         * self.mob_rate / CAP_AREA)))
        if users <= least // 2:
            return least
        if users >= 2 * least:
            return 2 * least
        return least + least * (2 * users - least) // (3 * least)

    def safe(self, pid: int, reach: float = MONSTER_REACH) -> bool:
        """No monster can touch her standing on this platform: none spawn on it (monsters never leave
        their platform), and no platform they spawn on runs underneath within `reach` (their bodies
        stick up that far: pigs on platform 1 hit her on platform 0, 42 units above them). A twin is as
        safe as the platform it lies on."""
        pid = self.place_of(pid)
        if pid in self.spawn_platforms:
            return False
        p = self.platform(pid)
        return not any(runs_under(self.platform(q), p, reach) for q in self.spawn_platforms)

    def portal_near(self, p: Platform, x: float, dx: float = 60, dy: float = 40, below: float = 125) -> bool:
        """Whether holding up on platform `p` near map x, or jumping from there, could take a portal: one within `dx`
        along it and `dy` of its surface (the spawn points too), or an exit up to `below` under it (PORTAL_BOX: an
        exit takes her from 100 above it). On Tree Dungeon, Forest Up North II no platform is within 125 above
        either exit."""
        return any(abs(pt.x - x) < dx and -dy < pt.y - p.y_at(pt.x) < (below if pt.kind == PORTAL_EXIT else dy)
                   for pt in self.portals)

    @cached_property
    def spawn_platforms(self) -> dict[int, int]:
        """Platform id -> number of monster spawn points on it."""
        return {pid: len(names) for pid, names in self.monsters_by_platform.items()}

    @cached_property
    def monsters_by_platform(self) -> dict[int, list[str]]:
        """Platform id -> the names of the monsters spawning on it (one per spawn point)."""
        out: dict[int, list[str]] = {}
        for pid, s in self.spawn_points:
            out.setdefault(pid, []).append(s.name or str(s.mob_id))
        return out

    @cached_property
    def spawn_points(self) -> list[tuple[int, Spawn]]:
        """Each spawn point on a platform, with that platform's id: where a monster of its kind appears (at its x: on
        Tree Dungeon the live telemetry's first sightings of monsters back came at the points' x again and again, not
        anywhere in their rx0..rx1, which there is most of the map)."""
        out = []
        for s in self.spawns:
            p = self.platform_under(s.x, s.y, reach=SPAWN_REACH)
            if p:
                out.append((p.id, s))
        return out


# --- reading the game data ------------------------------------------------------------------------

def _footholds(tree: dict) -> list[dict]:
    """Every foothold in a map image, flattened: its layer, group and id, its ends, its neighbours' ids
    ("0": none), and whether it forbids falling through it."""
    out = []
    for layer, groups in (tree.get("foothold") or {}).items():
        for group, segs in groups.items():
            for sid, s in segs.items():
                out.append({"layer": layer, "group": group, "id": sid, **{k: s[k] for k in ("x1", "y1", "x2", "y2")},
                            "prev": str(s.get("prev", 0)), "next": str(s.get("next", 0)),
                            "no_drop": bool(s.get("forbidFallDown"))})
    return out


def _platforms(footholds: list[dict]) -> tuple[list[Platform], list[Segment]]:
    """Join walkable footholds (going right) into platforms; vertical ones are walls. Platforms are
    numbered by the height of their left end (top first), then left to right."""
    by_key = {(f["layer"], f["group"], f["id"]): f for f in footholds}

    def line(f: dict, joined: str) -> tuple[float, float, float, float] | None:
        g = by_key.get((f["layer"], f["group"], f.get(joined, "0")))
        return None if g is None else (g["x1"], g["y1"], g["x2"], g["y2"])

    def segment(f: dict) -> Segment:
        return Segment(f["x1"], f["y1"], f["x2"], f["y2"], (int(f["layer"]), int(f["group"])),
                       f.get("no_drop", False), line(f, "prev"), line(f, "next"))

    walls = [segment(f) for f in footholds if f["x1"] == f["x2"]]
    walkable = {k for k, f in by_key.items() if f["x2"] > f["x1"]}
    seen, chains = set(), []
    # Taken left to right, a foothold always comes after the one joined before it (which ends where
    # it starts), so each chain is met at its first foothold and walked forward to its end.
    for key in sorted(walkable, key=lambda k: (by_key[k]["x1"], by_key[k]["y1"])):
        if key in seen:
            continue
        chain, cur = [], key
        while cur in walkable and cur not in seen:
            seen.add(cur)
            f = by_key[cur]
            chain.append(segment(f))
            nxt = (f["layer"], f["group"], f["next"])
            if nxt in walkable and by_key[nxt]["x1"] == f["x2"]:
                cur = nxt
            else:
                break
        chains.append(chain)
    chains.sort(key=lambda c: (c[0].y1, c[0].x1))
    return [Platform(i, c) for i, c in enumerate(chains)], walls


def _monster_lives(tree: dict) -> list[dict]:
    """A map image's monster spawn points (its 'life' entries of type 'm')."""
    return [s for s in (tree.get("life") or {}).values() if isinstance(s, dict) and s.get("type") == "m"]


def _mob_id(life: dict) -> str:
    """A spawn point's monster id, as the name and EXP tables key it (no leading zeros)."""
    return str(int(life["id"]))


def parse(map_id: int, tree: dict, name: str, physics: dict, mob_names: dict | None = None,
          mob_exp: dict | None = None, mob_bodies: dict | None = None, mob_die_s: dict | None = None,
          mob_speeds: dict | None = None, mob_damage: dict | None = None, mob_hops: dict | None = None,
          mob_flies: dict | None = None, mob_tall: dict | None = None, mob_hp: dict | None = None,
          mob_pdd: dict | None = None, mob_mdd: dict | None = None, mob_eva: dict | None = None,
          mob_level: dict | None = None) -> MapModel:
    """A map's model from its image in the game data (`tree`, as JSON). `mob_names`, `mob_exp`,
    `mob_bodies` (as _mob_bodies gives them), `mob_die_s` (_mob_die_s), `mob_speeds` (speed stats, as
    _mob_speeds gives them), `mob_damage` (touch damage), `mob_hops` (whether it hops), `mob_flies` (whether it
    flies) and `mob_tall` (how high its body reaches: _mob_touch gives these four), `mob_hp`, `mob_pdd` and `mob_mdd`
    (its HP and defences), `mob_eva` and `mob_level` (_mob_hp) are keyed by monster id (_mob_id); the model keys them by
    name."""
    info = tree.get("info", {})
    platforms, walls = _platforms(_footholds(tree))
    ropes = [Rope(r["x"], r["y1"], r["y2"], bool(r.get("l")), bool(r.get("uf")))
             for r in (tree.get("ladderRope") or {}).values()]
    portals = [Portal(p.get("pn", ""), p.get("pt", 0), p["x"], p["y"],
                      None if p.get("tm") in (None, 999999999) else p["tm"],  # 999999999: leads nowhere
                      str(p.get("image") or ""))
               for p in (tree.get("portal") or {}).values()]
    mob_names = mob_names or {}
    spawns = [Spawn(int(s["id"]), s["x"], s.get("cy", s["y"]), s.get("rx0", s["x"]), s.get("rx1", s["x"]),
                    mob_names.get(_mob_id(s), ""))
              for s in _monster_lives(tree)]
    mm = tree.get("miniMap") or {}
    model = MapModel(
        id=map_id, name=name,
        bounds=(info.get("VRLeft"), info.get("VRTop"), info.get("VRRight"), info.get("VRBottom")),
        minimap={k: mm.get(k) for k in ("width", "height", "centerX", "centerY", "mag")},
        platforms=platforms, walls=walls, ropes=ropes, portals=portals, spawns=spawns, physics=physics,
        mob_exp={mob_names.get(i, i): e for i, e in (mob_exp or {}).items()},
        mob_bodies={mob_names.get(i, i): MobBody(**b, die_s=(mob_die_s or {}).get(i, 0.0))
                    for i, b in (mob_bodies or {}).items()},
        mob_speeds={mob_names.get(i, i): mob_walk_speed(v, physics) for i, v in (mob_speeds or {}).items()},
        mob_damage={mob_names.get(i, i): float(v) for i, v in (mob_damage or {}).items()},
        mob_hops={mob_names.get(i, i): bool(v) for i, v in (mob_hops or {}).items()},
        mob_flies={mob_names.get(i, i): bool(v) for i, v in (mob_flies or {}).items()},
        mob_tall={mob_names.get(i, i): float(v) for i, v in (mob_tall or {}).items()},
        mob_hp={mob_names.get(i, i): float(v) for i, v in (mob_hp or {}).items()},
        mob_pdd={mob_names.get(i, i): float(v) for i, v in (mob_pdd or {}).items()},
        mob_mdd={mob_names.get(i, i): float(v) for i, v in (mob_mdd or {}).items()},
        mob_eva={mob_names.get(i, i): float(v) for i, v in (mob_eva or {}).items()},
        mob_level={mob_names.get(i, i): float(v) for i, v in (mob_level or {}).items()},
        mob_rate=float(info.get("mobRate", 1.0)), vr_limit=bool(info.get("VRLimit", 0)))
    floors, peers = tuple(s for p in model.platforms for s in p.segments), tuple(model.platforms)
    for p in model.platforms:
        p.walls, p.floors, p.peers = tuple(model.walls_of(p.group)), floors, peers
    return model


def mob_walk_speed(stat: float, physics: dict) -> float:
    """How fast a monster walks (units/s), from its speed stat and the map's physics (MOB_WALK_SPEED)."""
    return float(physics.get("walkSpeed", MOB_WALK_SPEED)) * (1 + stat / 100)


def _image(map_id: int) -> dict:
    name = f"{map_id:09d}.img"
    for part in sorted((COPY / "Map" / "Map").glob("Map*/Map*_000.wz")):
        pkg = Package(part)
        if name in pkg.images():
            return pkg.image(name)
    raise FileNotFoundError(f"map {map_id} isn't in {COPY}; run `python -m gamedata copy --install <game folder>`")


def _map_name(map_id: int) -> str:
    string = Package(COPY / "String" / "String_000.wz").image("Map.img")
    for region in string.values():
        entry = region.get(f"{map_id:09d}") if isinstance(region, dict) else None
        if isinstance(entry, dict):
            return f"{entry.get('streetName')}: {entry.get('mapName')}"
    return str(map_id)


_PICTURES: dict[int, np.ndarray] = {}  # minimap pictures loaded (only those: one that failed is tried again)


def load_minimap_picture(map_id: int) -> np.ndarray:
    """The map's own minimap picture (its miniMap canvas: the map at 1/2^mag, RGB, game px), from the game data
    copy (perception.minimap.MinimapPicture finds it in the window). Raises KeyError, OSError or ValueError if the
    copy or the picture isn't there."""
    if map_id not in _PICTURES:
        from gamedata.canvas import outlinked, pixels
        canvas, _ = outlinked(f"Map/Map/Map{map_id // 100000000}/_Canvas/{map_id:09d}.img/miniMap/canvas")
        _PICTURES[map_id] = np.ascontiguousarray(pixels(canvas)[..., :3])
    return _PICTURES[map_id]


def minimap_picture(map_id: int) -> np.ndarray | None:
    """load_minimap_picture, or None if it isn't there. Without it her dot is read with the first map's origin, which
    on Tree Dungeon put her 640 units off (2026-10-24 16:46:50, a process that had none): the commands that act load
    it with load_minimap_picture, and stop saying why if the map has one and it won't load (__main__._eyes)."""
    try:
        return load_minimap_picture(map_id)
    except (KeyError, OSError, ValueError):
        return None


def load(map_id: int) -> MapModel:
    """The map's model, parsed from the game data (cached as JSON next to the other extracts)."""
    cache = CACHE / f"{map_id}.json"
    raw = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    # Not cached yet, or cached before EXP, bodies or their death animations were kept (each a key of its
    # own, so code from before reads the cache as it did).
    if raw is None or "mob_die_s" not in raw:
        physics = jsonable(Package(COPY / "Map" / "Map_000.wz").image("Physics.img"))
        tree = jsonable(_image(map_id))
        ids = {_mob_id(s) for s in _monster_lives(tree)}
        names = _mob_names(ids)
        raw = {"id": map_id, "name": _map_name(map_id), "tree": tree, "physics": physics, "mob_names": names,
               "mob_exp": _mob_exp(ids), "mob_bodies": _mob_bodies(ids), "mob_die_s": _mob_die_s(ids),
               "mob_speeds": _mob_speeds(ids), **_mob_touch(ids), **_mob_hp(ids)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw), encoding="utf-8")
    elif "mob_speeds" not in raw:  # cached before speeds were kept: added to it
        _add_speeds(cache, raw)
    touch = {k: raw[k] for k in TOUCH_KEYS if k in raw}
    if len(touch) < len(TOUCH_KEYS):
        # Cached before touch damage, hops, flight and height were kept: read with the model from the game data copy
        # (none without it), and not written to the cache: one cache can serve several checkouts of the code at once.
        lives = {_mob_id(s) for s in _monster_lives(raw["tree"])}
        try:
            touch = _mob_touch(lives)
        except (OSError, KeyError, ValueError) as e:  # no copy of the game data, or no such monster in it
            touch = {}
            if lives:  # every kind then counts as no danger (attack.danger): the positioning is off, said so
                print(f"map {map_id}: no touch damage for its monsters ({type(e).__name__}: {e}): every place counts "
                      f"as safe (attack.danger)", file=sys.stderr)
    hp = {k: raw[k] for k in HP_KEYS if k in raw}
    if len(hp) < len(HP_KEYS):  # cached before the monsters' HP and defences were kept (or their magic defence, avoid
        # and level, from 2026-10-27): read as the touch keys are, what the cache has kept as it has it
        try:
            hp = {**_mob_hp({_mob_id(s) for s in _monster_lives(raw["tree"])}), **hp}
        except (OSError, KeyError, ValueError):  # no copy of the game data: attack.choice goes by her throws alone
            pass
    return parse(raw["id"], raw["tree"], raw["name"], raw["physics"], raw["mob_names"], raw["mob_exp"],
                 raw["mob_bodies"], raw["mob_die_s"], raw.get("mob_speeds"), touch.get("mob_damage"),
                 touch.get("mob_hops"), touch.get("mob_flies"), touch.get("mob_tall"), hp.get("mob_hp"),
                 hp.get("mob_pdd"), hp.get("mob_mdd"), hp.get("mob_eva"), hp.get("mob_level"))


def _add_speeds(cache: Path, raw: dict) -> None:
    """The monsters' speed stats added to a map cached without them. Without the game data copy they stay
    out (localization.monsters' table stands in), and the cache as it was. Written whole to a file of its
    own first, so another process reading the cache meanwhile (the harness's) never sees half of it."""
    try:
        raw["mob_speeds"] = _mob_speeds({_mob_id(s) for s in _monster_lives(raw["tree"])})
    except OSError:
        return
    part = cache.with_name(f"{cache.stem}.{os.getpid()}.part")
    try:
        part.write_text(json.dumps(raw), encoding="utf-8")
        os.replace(part, cache)
    except OSError:  # another process has it open: it's written next time
        part.unlink(missing_ok=True)


def _mob_hp(ids: set[str]) -> dict[str, dict]:
    """Monster id -> its HP (info/maxHP) as "mob_hp" and its defence against her stars (info/PDDamage) as "mob_pdd",
    from the game data (its own info, or the one it links to where it has none): a Horny Mushroom 403 and 10, a
    Green Mushroom 233 and 0. And its defence against magic (MDDamage: 0 and 30) as "mob_mdd", its avoid (eva: 13 and
    8) as "mob_eva" and its level (22 and 14) as "mob_level": a magician's Magic Claw goes by these."""
    from gamedata.canvas import _mob_pack
    pack = _mob_pack()
    hp, pdd, mdd, eva, level = {}, {}, {}, {}, {}
    for i in ids:
        info = pack.image(f"Mob/{int(i):07d}.img").get("info") or {}
        if not info.get("maxHP") and info.get("link"):
            info = pack.image(f"Mob/{int(info['link']):07d}.img").get("info") or {}
        hp[i], pdd[i] = float(info.get("maxHP") or 0), float(info.get("PDDamage") or 0)
        mdd[i], eva[i], level[i] = (float(info.get(k) or 0) for k in ("MDDamage", "eva", "level"))
    return {"mob_hp": hp, "mob_pdd": pdd, "mob_mdd": mdd, "mob_eva": eva, "mob_level": level}


def _mob_names(ids: set[str]) -> dict[str, str]:
    """Monster id -> name, from the game's strings."""
    strings = Package(COPY / "String" / "String_000.wz").image("Mob.img")
    return {i: (strings.get(i.zfill(7)) or strings.get(i) or {}).get("name", "") for i in ids}


def _mob_exp(ids: set[str]) -> dict[str, int]:
    """Monster id -> EXP per kill, from each monster's info in the game data."""
    from gamedata.canvas import _mob_pack
    pack = _mob_pack()
    return {i: int((pack.image(f"Mob/{int(i):07d}.img").get("info") or {}).get("exp") or 0) for i in ids}


def _mob_speeds(ids: set[str]) -> dict[str, float]:
    """Monster id -> its speed stat (info/speed: 0 for the game's walkSpeed, -65 for a snail), from the game data."""
    from gamedata.canvas import _mob_pack
    pack = _mob_pack()
    return {i: float((pack.image(f"Mob/{int(i):07d}.img").get("info") or {}).get("speed") or 0) for i in ids}


def _mob_touch(ids: set[str]) -> dict[str, dict]:
    """Monster id -> its touch damage (info/PADamage) as "mob_damage"; whether it hops (a "jump" action in its
    Mob data, its own or the one it links to: on the first map pigs, slimes and Orange Mushrooms do, on Tree Dungeon
    the slimes and not the mushrooms) as "mob_hops"; whether it flies (a "fly" action: no monster of her three maps
    has one) as "mob_flies"; and as "mob_tall" the highest its body reaches above its feet, by the boxes of its
    stand, move and jump frames (lt: a slime's hop 66, standing 42-44; the mushrooms 52), from the game data. A
    monster that hops can leave the platform it walks (pigs were seen on platforms where none spawn:
    localization.monsters); one that doesn't never does."""
    from gamedata.canvas import _mob_pack
    from gamedata.wz import Canvas
    pack = _mob_pack()
    damage, hops, flies, tall = {}, {}, {}, {}
    for i in ids:
        tree = pack.image(f"Mob/{int(i):07d}.img")
        info = tree.get("info") or {}
        damage[i] = float(info.get("PADamage") or 0)
        if info.get("link") and not tree.get("stand"):  # its frames are another monster's
            tree = pack.image(f"Mob/{int(info['link']):07d}.img")
        hops[i] = bool(tree.get(JUMP_ACTION))
        flies[i] = bool(tree.get(FLY_ACTION))
        tops = [-f.props["lt"].y for a in (*BODY_ACTIONS, JUMP_ACTION, FLY_ACTION) for f in (tree.get(a) or {}).values()
                if isinstance(f, Canvas) and f.props.get("lt") is not None]
        if tops:
            tall[i] = float(max(tops))
    return {"mob_damage": damage, "mob_hops": hops, "mob_flies": flies, "mob_tall": tall}


BODY_ACTIONS = ("stand", "move")  # a monster's frames while it stands or walks ...
FLINCH_ACTION = "hit1"            # ... while it flinches after a hit ...
DIE_ACTION = "die1"               # ... and while it dies (no boxes: only their delays are read)
JUMP_ACTION = "jump"              # a monster with frames for this hops (_mob_touch) ...
FLY_ACTION = "fly"                # ... and one with frames for this flies
TOUCH_KEYS = ("mob_damage", "mob_hops", "mob_flies", "mob_tall")  # what _mob_touch gives, kept with a map's cache
HP_KEYS = ("mob_hp", "mob_pdd", "mob_mdd", "mob_eva", "mob_level")   # ... and what _mob_hp gives (the last three
                                  # from 2026-10-27: a map cached before is read for them, its cache left as it is)
FRAME_DELAY = 100                 # ms a frame lasts when the data gives it no delay (the client's default)


def _mob_bodies(ids: set[str]) -> dict[str, dict]:
    """Monster id -> its body (MobBody's fields), from the boxes (lt/rb: from its feet, y up negative) of
    its frames in the game data. A monster whose frames have no boxes is left out (localization.monsters'
    tables stand in for it)."""
    from gamedata.canvas import _mob_pack
    from gamedata.wz import Canvas
    pack = _mob_pack()
    out = {}
    for i in ids:
        tree = pack.image(f"Mob/{int(i):07d}.img")
        link = (tree.get("info") or {}).get("link")
        if link and not tree.get("stand"):  # its frames are another monster's
            tree = pack.image(f"Mob/{int(link):07d}.img")

        def boxes(actions: tuple[str, ...]) -> list:
            return [(f.props["lt"], f.props["rb"]) for a in actions for f in (tree.get(a) or {}).values()
                    if isinstance(f, Canvas) and f.props.get("lt") is not None and f.props.get("rb") is not None]

        body, flinch = boxes(BODY_ACTIONS), boxes((FLINCH_ACTION,))
        if not body:
            continue
        top = float(min(-lt.y for lt, _ in body))
        out[i] = {"top": top, "flinch_top": float(max((-lt.y for lt, _ in flinch), default=top)),
                  "half": max(rb.x - lt.x for lt, rb in body) / 2}
    return out


def _mob_die_s(ids: set[str]) -> dict[str, float]:
    """Monster id -> how long it's seen dying, in s: the delays of its die1 frames in the game data (the
    last one fades it out: a snail 8 x 120 + 300 ms, a pig 2 x 180 + 300). A monster without them is left
    out (localization.monsters' table stands in for it)."""
    from gamedata.canvas import _mob_pack
    from gamedata.wz import Canvas
    pack = _mob_pack()
    out = {}
    for i in ids:
        tree = pack.image(f"Mob/{int(i):07d}.img")
        link = (tree.get("info") or {}).get("link")
        if link and not tree.get(DIE_ACTION):  # its frames are another monster's
            tree = pack.image(f"Mob/{int(link):07d}.img")
        delays = [f.props.get("delay") or FRAME_DELAY for f in (tree.get(DIE_ACTION) or {}).values()
                  if isinstance(f, Canvas)]
        if delays:
            out[i] = sum(delays) / 1000
    return out


Box = tuple[int, int, int, int]


@lru_cache(maxsize=None)
def portal_light(image: str = "", kind: int = PORTAL_EXIT, copy: Path = COPY) -> tuple[Box, Box]:
    """A portal's light about its point (boxes x0, y0, x1, y1 in map units, a game px each; the ends exclusive): of its
    animation's frames as the client draws a portal of that `kind` (MapHelper.img portal/game/pv/<image> for a visible
    one, PORTAL_HIDDEN's ph or psh /<image>'s portalStart and portalContinue for a hidden one; PORTAL_IMAGE's for ""),
    each picture with its origin on the point, the px lit (alpha over LIGHT_ALPHA) in any of them: (its lit px' box,
    its dense rows' box: the rows lit across LIGHT_DENSE of its lit width or more, from the first such row to the last).
    The default visible one's 8 frames (84-88 x 182-256 px, origins (41-44, 173-248); their pictures' union (-44, -248,
    45, 9)): (-41, -200, 42, 7) and (-41, -110, 42, 7); a hidden one's (ph/default, 11 frames): (-45, -116, 46, 2) and
    (-45, -105, 46, 1). The data image holds 1x1 placeholders with the origins; the pictures are where their `_outlink`s
    point, in Map/_Canvas. Raises KeyError or OSError if the copy hasn't got them, ValueError if nothing of them is
    lit."""
    from gamedata.canvas import pixels, source
    from gamedata.wz import Canvas
    tree = Package(copy / "Map" / "Map_000.wz").image("MapHelper.img")
    name = image or PORTAL_IMAGE
    if kind in PORTAL_HIDDEN:
        node = tree["portal"]["game"][PORTAL_HIDDEN[kind]][name]
        nodes, where = [node["portalStart"], node["portalContinue"]], f"{PORTAL_HIDDEN[kind]}/{name}"
    else:
        nodes, where = [tree["portal"]["game"]["pv"][name]], f"pv/{name}"
    frames = []
    for node in nodes:
        for k, c in node.items():
            if k.isdigit() and isinstance(c, Canvas):
                o = c.props.get("origin")
                rgba = pixels(source(c, tree, copy))
                frames.append(((o.x, o.y) if o is not None else (0, 0), rgba[..., 3] > LIGHT_ALPHA))
    if not frames:
        raise KeyError(f"MapHelper.img portal/game/{where} has no frames")
    x0, y0 = min(-ox for (ox, _), _m in frames), min(-oy for (_, oy), _m in frames)
    x1 = max(m.shape[1] - ox for (ox, _), m in frames)
    y1 = max(m.shape[0] - oy for (_, oy), m in frames)
    lit = np.zeros((y1 - y0, x1 - x0), bool)          # the frames' union, lit
    for (ox, oy), m in frames:
        lit[-oy - y0:-oy - y0 + m.shape[0], -ox - x0:-ox - x0 + m.shape[1]] |= m
    cols, rows = np.flatnonzero(lit.any(axis=0)), np.flatnonzero(lit.any(axis=1))
    if not len(cols):
        raise ValueError(f"MapHelper.img portal/game/{where}: nothing lit")
    share = lit[:, cols[0]:cols[-1] + 1].mean(axis=1)   # each row's share of the lit width that's lit
    dense = np.flatnonzero(share >= LIGHT_DENSE)
    dcols = np.flatnonzero(lit[dense[0]:dense[-1] + 1].any(axis=0))
    return ((int(x0 + cols[0]), int(y0 + rows[0]), int(x0 + cols[-1] + 1), int(y0 + rows[-1] + 1)),
            (int(x0 + dcols[0]), int(y0 + dense[0]), int(x0 + dcols[-1] + 1), int(y0 + dense[-1] + 1)))


def portal_lights(model: MapModel, copy: Path = COPY) -> list[tuple[Portal, Box, Box]]:
    """Every visible or hidden portal of the map (PORTAL_SHOWN, PORTAL_HIDDEN) and its light's boxes on the map (its
    lit px', its dense rows': `portal_light` placed at its point): Tree Dungeon's out00 (282, -49) x 241..324,
    y -249..-42 and -159..-42, out01 (281, -1183) x 240..323, y -1383..-1176 and -1293..-1176. Raises KeyError or
    OSError if the game data copy hasn't got its pictures."""
    out = []
    for p in model.portals:
        if p.kind in PORTAL_SHOWN or p.kind in PORTAL_HIDDEN:
            lit, dense = portal_light(p.image, PORTAL_EXIT if p.kind in PORTAL_SHOWN else p.kind, copy)
            out.append((p, *(tuple(v + d for v, d in zip(box, (p.x, p.y, p.x, p.y))) for box in (lit, dense))))
    return out


# --- a picture of the map -------------------------------------------------------------------------

def render(model: MapModel, path: Path, scale: float = 0.5, marks: list[tuple[float, float, str]] = ()) -> None:
    """Draw the map's platforms (green), walls (grey), ropes (orange), portals (blue) and spawns (red),
    and `marks` (map x, y, label) in yellow."""
    from PIL import Image, ImageDraw
    left, top, right, bottom = model.bounds
    pad = 10  # px around the map
    W, H = int((right - left) * scale) + 2 * pad, int((bottom - top) * scale) + 2 * pad

    def pt(x, y):
        return (x - left) * scale + pad, (y - top) * scale + pad

    img = Image.new("RGB", (W, H), (25, 30, 45))
    d = ImageDraw.Draw(img)
    for w in model.walls:
        d.line([pt(w.x1, w.y1), pt(w.x2, w.y2)], fill=(110, 110, 110), width=2)
    for p in model.platforms:
        for s in p.segments:
            d.line([pt(s.x1, s.y1), pt(s.x2, s.y2)], fill=(90, 220, 90), width=3)
        d.text(pt(p.left, p.top - 22), str(p.id), fill=(200, 255, 200))
    for r in model.ropes:
        d.line([pt(r.x, r.top), pt(r.x, r.bottom)], fill=(240, 170, 60), width=3)
    for p in model.portals:
        x, y = pt(p.x, p.y)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(80, 160, 255), width=2)
    for s in model.spawns:
        x, y = pt(s.x, s.y)
        d.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 80, 80))
    for x, y, label in marks:
        px, py = pt(x, y)
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 230, 0))
        d.text((px + 8, py - 6), label, fill=(255, 230, 0))
    img.save(path)
