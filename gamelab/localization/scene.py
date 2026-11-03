"""One look placed on the map (Scene): where she is, the monsters and the drops, and conversions
between screen and map. `find_loot` looks for the drops lying on the platforms in view.
"""

from dataclasses import dataclass, field

from maplehelper.engine import RIGHT

from ..perception.eyes import Eyes, Observation
from ..perception.loot import ABOVE, BELOW, Item
from ..perception.minimap import minimap_box
from ..perception.mobs import Mob
from .mapdata import MapModel, Platform
from .monsters import Monster, monster_platform
from .players import Player
from .pose import Where

# Loot strips (half-size px of the 1920x1080 window; the look's geometry converts them): drops are looked for in a
# strip just above each stretch of ground.
STRIP_PAD = 4    # a strip reaches this far past each end of its ground
MIN_STRIP = 12   # narrower ones (cut short by the screen's side or the minimap) aren't searched
SAME_ITEM = 4    # items this close (across and down) are one item, found in two platforms' strips
GROUNDED_PX = 6  # half-size px of ground under her feet that show she's standing on it (in the air: none;
                 # navigator.MIN_GROUND_PX)


@dataclass(frozen=True)
class Loot:
    """An item or mesos seen on the ground, placed on the map."""
    item: Item
    x: float         # map units
    y: float
    platform: int    # the platform it lies on

    @property
    def mesos(self) -> bool:
        return self.item.mesos


@dataclass(kw_only=True)
class Scene(Observation):
    """A look (Observation) placed on the map."""
    where: Where | None = None    # her map position (minimap dot, snapped to the platform she's on; by the scene, her
                                  # feet: the view plus her place on screen)
    how: str = "dot"              # how `where.x` was worked out: 'camera', 'rope', 'edge' (on screen) or 'dot'; by the
                                  # scene (localization.locate) the cue that placed her: 'tag', 'plate', 'body', or
                                  # 'carried' (none did: the filter's prediction, `tag_seen` False), 'covered' (so,
                                  # under the Lie Detector, her stat window or a portal's light: locate.Cover)
    mode: str | None = None       # by the scene: 'ground', 'air' or 'rope' (None: placed as before)
    confidence: float | None = None  # ... and how sure of her place (the cue's score; carried, less each frame)
    tag_seen: bool = True         # `tag` is her name tag as seen (False: where the camera puts her, the tag
                                  # being hidden; Senses fills `tag` in then, so `tag` is always her place
                                  # on screen once she's placed)
    monsters: list[Monster] = field(default_factory=list)  # monsters seen, placed on the map
    searched: bool = False        # monsters were looked for (an empty `monsters` then means none was seen)
    wide: bool = False            # ... on the whole screen (else only in the band around her level)
    swept: list[Monster] = field(default_factory=list)  # monsters perception follows off her band, re-found in the
                                                         # look's frame (feed.Feed.look; not `monsters`: the fights go
                                                         # by the band's)
    rows: list[tuple[float, float]] = field(default_factory=list)  # where monsters were looked for: map y from, to
                                                                    # (the band, the screen)
    loot: list[Loot] = field(default_factory=list)         # drops seen on the ground (find_loot)
    looked: list[tuple[int, float, float]] = field(default_factory=list)  # where loot was looked for:
                                                                          # (platform, map x from, to)
    events: list | None = None    # with a continuous feed, what it saw happen since the last look, frame by
                                  # frame (tracking.Event: her EXP gains, hits, monsters' deaths); None: no feed
    players: list[Player] = field(default_factory=list)  # other players on the map, by their minimap dots
                                                          # (localization.players: confirmed over two looks)
    touch: dict | None = None     # with a continuous feed, the newest touch of hers told as of this look's frame
                                  # (worldstate.WorldState.touch: MISS or damage, when); None: none yet, or no feed
    misses: list = field(default_factory=list)  # her attack's MISS popups in this look's frame over her regular presses'
                                                # targets, as map (x, y) of their bottom centre (behavior.hunter._misses)
    tracks: list = field(default_factory=list)  # with a continuous feed, the monsters it follows as of this look's frame
                                                # (tracking.Track: live, hidden on screen up to HIDE_S, or just dead):
                                                # those it didn't find are about still (localization.unseen); [] without
    strips: list = field(default_factory=list)  # with a continuous feed, every strip of drops in view as perception last
                                                # searched it, each at its own time (worldstate.DropStrip: the rotation's
                                                # and the asked-for; the drop store takes them, behavior.ledger); []
                                                # without
    what_if: bool = False         # a look made up from another (her put where a chain ends: the navigator's _stop_calls,
                                  # _look_ahead) for what a trip's `until` would say there: nothing it decides is kept
    seq: int | None = None        # with a continuous feed, the frame number of the state it's of (worldstate.WorldState:
                                  # the perception service's, its recording's `seq`); None without

    @classmethod
    def of(cls, obs: Observation) -> "Scene":
        return cls(**vars(obs))

    @property
    def platform(self) -> Platform | None:
        return None if self.where is None else self.where.platform

    @property
    def grounded(self) -> bool:
        """Whether she stands on the ground in this look: her name tag seen, on a platform, with ground under
        her feet (by the scene: placed by a cue, standing by the filter). In the air every monster's y is off by her
        height above where her dot puts her (up to her 77-unit jump), but where the view is registered."""
        if self.mode is not None:
            return self.tag_seen and self.mode == "ground" and self.platform is not None
        g = self.ground
        return (self.tag_seen and self.platform is not None and g is not None
                and g[1] - g[0] >= self.geo.fpx(GROUNDED_PX))

    def place(self, m: Mob, model: MapModel, bar: float | None) -> Monster:
        """Put a monster seen on screen onto the map, by its offset from her: where it stands (its origin,
        when the template that matched knows it; else the template's centre and bottom). With its origin
        known and her on the ground, on whatever platform is right under its feet; else on the platform it
        walks on (monster_platform). Never past that platform's ends (read from the screen, one can seem up to
        PLATFORM_SLACK past one; past the step at the end of platform 2, it would seem in front of the wall
        it's behind)."""
        x, y = self.map_x(m.foot_x), self.map_y(m.feet_y)
        p = monster_platform(model, m.kind, x, y, exact=m.feet is not None and (self.grounded or self.view_sure))
        if p is None:
            return Monster(m, x, y, None, bar, m.facing, m.pose)
        x = p.clamp(x)
        return Monster(m, x, p.y_at(x), p.id, bar, m.facing, m.pose)

    # Look-frame px <-> map units, by the view where the scene registered it (`view`: the map point at the frame's
    # top-left), else by the offset from her (the camera doesn't scale, so offsets from her hold anywhere on screen).

    def screen_x(self, map_x: float) -> float:
        """Look-frame x of a map x."""
        if self.view is not None:
            return (map_x - self.view[0]) / self.geo.units_per_px
        return self.tag[0] + (map_x - self.where.x) / self.geo.units_per_px

    def screen_y(self, map_y: float) -> float:
        """Look-frame y of a map y."""
        if self.view is not None:
            return (map_y - self.view[1]) / self.geo.units_per_px
        return self.floor_y + (map_y - self.where.y) / self.geo.units_per_px

    def map_x(self, screen_x: float) -> float:
        if self.view is not None:
            return self.view[0] + screen_x * self.geo.units_per_px
        return self.where.x + (screen_x - self.tag[0]) * self.geo.units_per_px

    def map_y(self, screen_y: float) -> float:
        if self.view is not None:
            return self.view[1] + screen_y * self.geo.units_per_px
        return self.where.y + (screen_y - self.floor_y) * self.geo.units_per_px

    def in_view(self, platform: int, x: float) -> bool:
        """Whether loot was looked for at map x on that platform in this look."""
        return any(p == platform and lo <= x <= hi for p, lo, hi in self.looked)

    def searched_x(self, y: float) -> tuple[float, float] | None:
        """The map x this look searched for monsters at map level `y`: the frame's breadth where its rows (`rows`: the
        band, the screen) take in `y`; None where they don't, or it searched none (where loot was looked for,
        `looked`, is a strip of it at a time: worldstate.DROPS_EVERY_S)."""
        if not self.searched or self.tag is None or self.where is None                 or not any(min(a, b) <= y <= max(a, b) for a, b in self.rows):
            return None
        shape = getattr(self.frame, "shape", None)
        return self.map_x(0), self.map_x(shape[1] if shape is not None else self.geo.frame_size[0])

    def record(self) -> dict:
        """This look as telemetry data ('scene'): where she is (and whether she stands: `grounded`), HP and MP, the
        monsters and the drops; and what it searched (from 2026-10-21's evening, for analysis.respawn: an empty list
        of monsters is otherwise a look that saw none or one that didn't look): `searched` and `wide` (Scene's),
        `looked` (where drops were looked for: platform, map x from, to) and `view` (the map the screen showed above
        the bottom bar: x from, y from, x to, y to, whole units); and the other players (`players`, left out when there
        are none); and where the map's minimap picture was found and how well it matched (`minimap`: its corner x, y in
        full-size px, the score; left out without one), which a dot read with the wrong origin is told by; from
        2026-10-24's evening, where it searched for monsters (`rows`: map y from, to) and the monsters a slice of the
        sweep found (`swept`, left out when it found none); from 2026-11-02 the frame number of the perception
        service's state it's of (`seq`, left out without a feed: matched to a recording's frames by its index's `seq`)."""
        her = None if self.where is None else {
            "x": round(self.where.x, 1), "y": round(self.where.y, 1),
            "platform": None if self.platform is None else self.platform.id, "how": self.how, "seen": self.tag_seen,
            "grounded": self.grounded,
            **({"mode": self.mode, "conf": round(self.confidence, 2), "view_sure": self.view_sure}
               if self.mode is not None and self.confidence is not None else {})}  # (by the scene)
        view = None
        if self.tag is not None and self.where is not None and getattr(self.frame, "shape", None) is not None:
            h, w = self.frame.shape[:2]
            view = [round(self.map_x(0)), round(self.map_y(0)), round(self.map_x(w)),
                    round(self.map_y(h * self.geo.hud_top))]
        return {**({"seq": self.seq} if self.seq is not None else {}), "her": her, "hp": round(self.hp, 3),
                "mp": round(self.mp, 3),
                **({"hp_of": list(self.hp_of)} if self.hp_of is not None else {}),
                **({"level_of": self.level_of} if getattr(self, "level_of", None) is not None else {}),
                "monsters": [_monster_record(m) for m in self.monsters],
                **({"swept": [_monster_record(m) for m in self.swept]} if self.swept else {}),
                "rows": [[round(lo), round(hi)] for lo, hi in self.rows],
                "loot": [{"kind": d.item.kind, "x": round(d.x, 1), "y": round(d.y, 1), "platform": d.platform}
                         for d in self.loot],
                "searched": self.searched, "wide": self.wide,
                "looked": [[pid, round(lo), round(hi)] for pid, lo, hi in self.looked], "view": view,
                **({"players": [p.record() for p in self.players]} if self.players else {}),
                **({"minimap": [round(v, 3) for v in self.picture]} if self.picture is not None else {})}

    def room(self, direction: str, margin: float) -> float:
        """Map units she can walk that way before the end of her platform (less a margin)."""
        p = self.platform
        if p is None:
            return 0.0
        return max(0.0, (p.right - margin - self.where.x) if direction == RIGHT else (self.where.x - p.left - margin))


def _monster_record(m: Monster) -> dict:
    return {"kind": m.kind, "x": round(m.x, 1), "y": round(m.y, 1), "platform": m.platform,
            "bar": None if m.bar is None else round(m.bar, 2),
            **({"facing": m.facing} if m.facing else {}),  # left out when not known
            **({"fading": m.fading} if m.fading is not None else {})}  # (found fading in: perception.spawns)


@dataclass(frozen=True)
class Strip:
    """Where drops are looked for in a look's frame: a strip just above a stretch of a platform's level ground
    (level_runs), as much of it as shows. `run`: the stretch's left end (map x), what it is from frame to frame;
    `looked`: (platform, map x from, to), as Scene.looked notes it; `box`: look-frame px; `whole`: none of its height
    cut off by the bottom bar or the screen's top (a drop lies on its ground: in a strip cut short under the bar the
    drop may be under the bar, and its search finding none doesn't say it's gone: the review of 2026-10-30, strips cut
    by the bar found none where the simulator's world had drops 10 times in 26, whole ones 74 in 962)."""
    platform: Platform
    run: float
    looked: tuple[int, float, float]
    box: tuple[int, int, int, int]
    whole: bool = True

    @property
    def key(self) -> tuple[int, float]:
        return self.platform.id, self.run


def drop_strips(s: Scene, model: MapModel, eyes: Eyes, wide: bool = False) -> list[Strip]:
    """The strips a search for drops looks in (find_loot): just above each stretch of level ground of her platform (or,
    if `wide`, of every platform in view), as much as shows above the bottom bar and right of the minimap; none too
    narrow or too short for an item."""
    if s.tag is None or s.where is None:
        return []
    h, w = s.frame.shape[:2]
    geo = s.geo
    bottom = int(h * geo.hud_top)
    box = minimap_box(geo)
    mini_x, mini_y = box[2] // geo.look_scale, box[3] // geo.look_scale  # the minimap's bottom-right corner
    pad, above, below, narrow = geo.fpx(STRIP_PAD), geo.fpx(ABOVE), geo.fpx(BELOW), geo.fpx(MIN_STRIP)
    platforms = model.places if wide else [p for p in (s.platform,) if p is not None]
    out: list[Strip] = []
    for p in platforms:
        for x0, x1, high, low in level_runs(p):
            left, right = max(0.0, s.screen_x(x0) - pad), min(float(w), s.screen_x(x1) + pad)
            high_y, low_y = s.screen_y(high) - above, s.screen_y(low) + below
            top, under = max(0.0, high_y), min(float(bottom), low_y)
            if top < mini_y and left < mini_x:  # the minimap covers the strip's left end
                left = max(left, float(mini_x))
            if right - left < narrow or under - top < eyes.item_height:
                continue
            out.append(Strip(p, x0, (p.id, s.map_x(left), s.map_x(right)),
                             (int(left), int(top), int(right), int(under)), whole=high_y >= 0.0 and low_y <= bottom))
    return out


def find_loot(s: Scene, model: MapModel, eyes: Eyes, wide: bool = False, mesos_only: bool = False,
              strips: list[Strip] | None = None) -> list[Loot]:
    """Drops lying on her platform (or, if `wide`, on every platform in view), placed on the map. Only
    a strip just above each platform's top is searched: drops lie on the ground (drop_strips; `strips`: those alone).
    Where it looked is noted in `s.looked`."""
    if s.tag is None or s.where is None:
        return []
    geo = s.geo
    strips = drop_strips(s, model, eyes, wide) if strips is None else strips
    s.looked.extend(st.looked for st in strips)
    found: list[tuple[float, Loot]] = []  # (px between its bottom and the ground under it, loot)
    for (p, _), items in zip(((st.platform, st.box) for st in strips),
                             eyes.items_in(s, [st.box for st in strips], mesos_only=mesos_only)):
        for it in items:
            x = p.clamp(s.map_x(it.x))
            ground = p.y_at(x)
            found.append((abs(it.bottom - s.screen_y(ground)), Loot(it, x, ground, p.id)))
    return _on_nearest_platform(found, geo.fpx(SAME_ITEM))


def _on_nearest_platform(found: list[tuple[float, Loot]], same: float = SAME_ITEM) -> list[Loot]:
    """Platforms close above one another share strips, so an item may be found for each: it goes to
    the one it lies on (whose ground is nearest its bottom). Items within `same` look-frame px (SAME_ITEM) are one."""
    kept: list[Loot] = []
    for _, loot in sorted(found, key=lambda f: f[0]):
        if not any(abs(loot.item.x - k.item.x) <= same and abs(loot.item.y - k.item.y) <= same
                   for k in kept):
            kept.append(loot)
    return kept


def level_runs(p: Platform, flat: float = 20) -> list[tuple[float, float, float, float]]:
    """A platform as stretches of nearly level ground (rising or falling at most `flat` map units):
    (x from, x to, highest y, lowest y), map units."""
    runs: list[tuple[float, float, float, float]] = []
    for seg in p.segments:
        hi, lo = min(seg.y1, seg.y2), max(seg.y1, seg.y2)
        if runs and max(runs[-1][3], lo) - min(runs[-1][2], hi) <= flat:
            x0, _, top, low = runs[-1]
            runs[-1] = (x0, seg.x2, min(top, hi), max(low, lo))
        else:
            runs.append((seg.x1, seg.x2, hi, lo))
    return runs
