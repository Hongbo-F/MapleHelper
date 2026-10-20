"""Senses: every look at the game, placed on the map. Eyes (perception) says what's on screen; Senses
works out where she is (her minimap dot, refined by a rope or her platform's end on screen), where she
is on screen when her name tag is hidden (the camera), and where the monsters and drops are.

Every look anyone takes goes through `Senses.look`, and the listeners run after each one, in order:
the hunter keeps count of her EXP and HP there, and the mini-map's telemetry is sent from there.
"""

from collections.abc import Callable

from ..perception.eyes import TAG_BELOW_FEET, Eyes
from ..screen import SCALE
from .mapdata import MapModel
from .pose import Camera, Localizer, refine
from .scene import Scene, find_loot


class Senses:
    def __init__(self, eyes: Eyes, model: MapModel, band: tuple[float, float]):
        """`band`: how far above and below her feet (map units) a look searches for monsters."""
        self.eyes, self.model, self.band = eyes, model, band
        self.localizer = Localizer(model)
        self.camera = Camera(model.bounds)
        self.listeners: list[Callable[[Scene], None]] = []
        eyes.want_ropes = True  # ropes in view pin down her x (+~25 ms per look)

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
