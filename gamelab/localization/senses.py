"""Senses: every look at the game, placed on the map. Eyes (perception) says what's on screen; Senses
works out where she is (her minimap dot, refined by a rope or her platform's end on screen), where she
is on screen when her name tag is hidden (the camera), and where the monsters and drops are.

Every look anyone takes goes through `Senses.look`, and the listeners run after each one, in order:
the hunter keeps count of her EXP and HP there, and the mini-map's telemetry is sent from there.
"""

from collections.abc import Callable

from ..perception.eyes import TAG_BELOW_FEET, Eyes
from ..perception.terrain import MAX_ROPE_WIDTH
from ..screen import SCALE, SCREEN_SCALE
from .mapdata import MapModel
from .pose import ROPE_MATCH_PX, Camera, Localizer, refine
from .scene import Scene, find_loot

# Full-size px either side of where her minimap dot puts each of the map's ropes on screen that a look
# searches for ropes: all `pose.rope_x` can use (a rope within ROPE_MATCH_PX of there, up to MAX_ROPE_WIDTH
# wide). 8.6 ms a look (median, bench-look), against 19 for the whole screen. On 17 saved frames the
# ropes seen were 0-20 px from there (her dot 0-14.5 map units off), the same ropes as on the whole
# screen, and she was placed the same. The mover's line-up on a rope looks 70 px around where the map
# puts it from her refined position: the same ropes while her dot is within about 30 units. None
# searches the whole screen.
ROPE_PAD: int | None = ROPE_MATCH_PX + MAX_ROPE_WIDTH


class Senses:
    def __init__(self, eyes: Eyes, model: MapModel, band: tuple[float, float]):
        """`band`: how far above and below her feet (map units) a look searches for monsters."""
        self.eyes, self.model, self.band = eyes, model, band
        self.localizer = Localizer(model)
        self.camera = Camera(model.bounds)
        self.listeners: list[Callable[[Scene], None]] = []
        eyes.want_ropes = True  # ropes in view pin down her x
        eyes.rope_columns = self._rope_columns

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False, mesos_only: bool = False) -> Scene:
        """Her map position from the minimap dot; her screen position from her name tag, or from the
        dot and the camera when the tag is hidden; (if `mobs`) the monsters she could shoot, or (if
        `wide`) every monster on screen; and (if `loot`) the drops on her platform, or (if `wide`) on
        every platform in view (only mesos if `mesos_only`)."""
        s = Scene.of(self.eyes.look())
        self._place_her(s)
        if mobs and s.tag is not None and s.where is not None:
            seen = self.eyes.monsters_wide(s) if wide else self.eyes.monsters(s, *self.band)
            s.monsters = [s.place(m, self.model, self.eyes.bar(s, m)) for m in seen]
            s.searched, s.wide = True, wide
        if loot:
            s.loot = find_loot(s, self.model, self.eyes, wide, mesos_only)
        for listener in self.listeners:
            listener(s)
        return s

    def _rope_columns(self, dot: tuple[float, float] | None,
                      tag: tuple[int, int] | None) -> list[tuple[int, int]] | None:
        """Where a look searches for ropes (ranges of x, full-size px): around where her dot puts the map's
        ropes on screen (ROPE_PAD). With no dot or no name tag, ropes can't place her: nowhere."""
        if ROPE_PAD is None:
            return None
        if dot is None or tag is None:
            return []
        where, her_x = self.localizer.where_dot(dot), SCALE * tag[0]
        xs = (her_x + (rope.x - where.x) * SCREEN_SCALE for rope in self.model.ropes)
        return [(int(x) - ROPE_PAD, int(x) + ROPE_PAD + 1) for x in xs]

    def _place_her(self, s: Scene) -> None:
        """Her map position (`s.where`, `s.how`) from her minimap dot, refined on screen. Then, with her
        name tag seen, the camera learns where it keeps her; with it hidden, the camera places her
        on screen instead (`s.tag`, and `s.tag_seen` False)."""
        if s.dot is not None:
            s.where = self.localizer.where_dot(s.dot)
            if s.tag is not None:
                s.where, s.how = refine(s.where, s, self.model)
        if s.where is None:
            return
        if s.tag is not None:
            self.camera.learn((s.where.x, s.where.y), (SCALE * s.tag[0], SCALE * s.floor_y))
        else:
            sx, sy = self.camera.to_screen(s.where.x, s.where.y, (s.where.x, s.where.y))
            s.tag, s.tag_seen = (round(sx / SCALE), round(sy / SCALE) + TAG_BELOW_FEET), False
