"""One look placed on the map (Scene): where she is, the monsters and the drops, and conversions
between screen and map. `find_loot` looks for the drops lying on the platforms in view.
"""

from dataclasses import dataclass, field

from maplehelper.engine import RIGHT

from ..perception.eyes import Eyes, Observation
from ..perception.loot import ABOVE, BELOW, Item
from ..perception.minimap import MINIMAP_BOX
from ..perception.mobs import Mob
from ..screen import HUD_TOP, SCALE, UNITS_PER_HALF_PX
from .mapdata import MapModel, Platform
from .monsters import Monster, monster_platform
from .pose import Where

# Loot strips (half-size px): drops are looked for in a strip just above each stretch of ground.
STRIP_PAD = 4    # a strip reaches this far past each end of its ground
MIN_STRIP = 12   # narrower ones (cut short by the screen's side or the minimap) aren't searched
SAME_ITEM = 4    # items this close (across and down) are one item, found in two platforms' strips


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
    where: Where | None = None    # her map position (minimap dot, snapped to the platform she's on)
    how: str = "dot"              # how `where.x` was worked out: 'rope', 'edge' (on screen) or 'dot'
    tag_seen: bool = True         # `tag` is her name tag as seen (False: where the camera puts her, the tag
                                  # being hidden; Senses fills `tag` in then, so `tag` is always her place
                                  # on screen once she's placed)
    monsters: list[Monster] = field(default_factory=list)  # monsters seen, placed on the map
    loot: list[Loot] = field(default_factory=list)         # drops seen on the ground (find_loot)
    looked: list[tuple[int, float, float]] = field(default_factory=list)  # where loot was looked for:
                                                                          # (platform, map x from, to)

    @classmethod
    def of(cls, obs: Observation) -> "Scene":
        return cls(**vars(obs))

    @property
    def platform(self) -> Platform | None:
        return None if self.where is None else self.where.platform

    def place(self, m: Mob, model: MapModel, bar: float | None) -> Monster:
        """Put a monster seen on screen onto the map, by its offset from her."""
        x, y = self.map_x(m.x), self.map_y(m.feet_y)
        p = monster_platform(model, m.kind, x, y)
        return Monster(m, x, y if p is None else p.y_at(x), None if p is None else p.id, bar)

    # Half-size screen px <-> map units, by the offset from her (the camera doesn't scale, so offsets
    # from her hold anywhere on screen).

    def screen_x(self, map_x: float) -> float:
        """Half-size screen x of a map x."""
        return self.tag[0] + (map_x - self.where.x) / UNITS_PER_HALF_PX

    def screen_y(self, map_y: float) -> float:
        """Half-size screen y of a map y."""
        return self.floor_y + (map_y - self.where.y) / UNITS_PER_HALF_PX

    def map_x(self, screen_x: float) -> float:
        return self.where.x + (screen_x - self.tag[0]) * UNITS_PER_HALF_PX

    def map_y(self, screen_y: float) -> float:
        return self.where.y + (screen_y - self.floor_y) * UNITS_PER_HALF_PX

    def in_view(self, platform: int, x: float) -> bool:
        """Whether loot was looked for at map x on that platform in this look."""
        return any(p == platform and lo <= x <= hi for p, lo, hi in self.looked)

    def record(self) -> dict:
        """This look as telemetry data ('scene'): where she is, HP and MP, the monsters and the drops."""
        her = None if self.where is None else {
            "x": round(self.where.x, 1), "y": round(self.where.y, 1),
            "platform": None if self.platform is None else self.platform.id, "how": self.how, "seen": self.tag_seen}
        return {"her": her, "hp": round(self.hp, 3), "mp": round(self.mp, 3),
                "monsters": [{"kind": m.kind, "x": round(m.x, 1), "y": round(m.y, 1), "platform": m.platform,
                              "bar": None if m.bar is None else round(m.bar, 2)} for m in self.monsters],
                "loot": [{"kind": d.item.kind, "x": round(d.x, 1), "y": round(d.y, 1), "platform": d.platform}
                         for d in self.loot]}

    def room(self, direction: str, margin: float) -> float:
        """Map units she can walk that way before the end of her platform (less a margin)."""
        p = self.platform
        if p is None:
            return 0.0
        return max(0.0, (p.right - margin - self.where.x) if direction == RIGHT else (self.where.x - p.left - margin))


def find_loot(s: Scene, model: MapModel, eyes: Eyes, wide: bool = False, mesos_only: bool = False) -> list[Loot]:
    """Drops lying on her platform (or, if `wide`, on every platform in view), placed on the map. Only
    a strip just above each platform's top is searched: drops lie on the ground. Where it looked is
    noted in `s.looked`."""
    if s.tag is None or s.where is None:
        return []
    h, w = s.frame.shape[:2]
    bottom = int(h * HUD_TOP)
    mini_x, mini_y = MINIMAP_BOX[2] // SCALE, MINIMAP_BOX[3] // SCALE  # the minimap's bottom-right corner
    platforms = model.platforms if wide else [p for p in (s.platform,) if p is not None]
    found: list[tuple[float, Loot]] = []  # (px between its bottom and the ground under it, loot)
    for p in platforms:
        for x0, x1, high, low in level_runs(p):
            left, right = max(0.0, s.screen_x(x0) - STRIP_PAD), min(float(w), s.screen_x(x1) + STRIP_PAD)
            top, under = max(0.0, s.screen_y(high) - ABOVE), min(float(bottom), s.screen_y(low) + BELOW)
            if top < mini_y and left < mini_x:  # the minimap covers the strip's left end
                left = max(left, float(mini_x))
            if right - left < MIN_STRIP or under - top < eyes.item_height:
                continue
            s.looked.append((p.id, s.map_x(left), s.map_x(right)))
            box = (int(left), int(top), int(right), int(under))
            for it in eyes.items(s, box, mesos_only=mesos_only):
                x = p.clamp(s.map_x(it.x))
                ground = p.y_at(x)
                found.append((abs(it.bottom - s.screen_y(ground)), Loot(it, x, ground, p.id)))
    return _on_nearest_platform(found)


def _on_nearest_platform(found: list[tuple[float, Loot]]) -> list[Loot]:
    """Platforms close above one another share strips, so an item may be found for each: it goes to
    the one it lies on (whose ground is nearest its bottom)."""
    kept: list[Loot] = []
    for _, loot in sorted(found, key=lambda f: f[0]):
        if not any(abs(loot.item.x - k.item.x) <= SAME_ITEM and abs(loot.item.y - k.item.y) <= SAME_ITEM
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
