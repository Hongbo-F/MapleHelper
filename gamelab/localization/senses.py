"""Senses: every look at the game, placed on the map. Eyes (perception) says what's on screen; Senses
works out where she is (her minimap dot, refined by a rope or her platform's end on screen), where she
is on screen when her name tag is hidden (the camera), and where the monsters and drops are.

Every look anyone takes goes through `Senses.look`, and the listeners run after each one, in order:
the hunter keeps count of her EXP and HP there, and the mini-map's telemetry is sent from there.

With a continuous feed (localization.feed: the game looked at every frame in a process of its own), a look
is the feed's state of the first frame shown after it was asked for, with what it needs fresh (see
feed.fresh), and the events since the last look come with it (Scene.events); glances come from the feed's
newest frames. When the feed can't give one (the game not in front, the feed gone or too slow), the look or
glance is taken here as without it. Without a feed (the default) nothing changes.
"""

import time
from collections.abc import Callable

from ..perception.eyes import GLANCE_EVERY_S, TAG_BELOW_FEET, Eyes, Observation
from ..perception.terrain import MAX_ROPE_WIDTH
from ..screen import SCALE, SCREEN_SCALE
from .mapdata import MapModel
from .monsters import Monster
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
    def __init__(self, eyes: Eyes, model: MapModel, band: tuple[float, float], feed=None):
        """`band`: how far above and below her feet (map units) a look searches for monsters. `feed`: a
        continuous feed to take looks and glances from (localization.feed.Feed), or None."""
        self.eyes, self.model, self.band = eyes, model, band
        self.feed = feed
        self.localizer = Localizer(model)
        self.camera = Camera(model.bounds)
        self.listeners: list[Callable[[Scene], None]] = []
        eyes.want_ropes = True  # ropes in view pin down her x
        eyes.rope_columns = self._rope_columns
        self._glanced = -1e18  # the frame time of the last glance from the feed
        self.counts = {"from the feed": 0, "taken here": 0, "glances from the feed": 0}

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False, mesos_only: bool = False) -> Scene:
        """Her map position from the minimap dot; her screen position from her name tag, or from the
        dot and the camera when the tag is hidden; (if `mobs`) the monsters she could shoot, or (if
        `wide`) every monster on screen; and (if `loot`) the drops on her platform, or (if `wide`) on
        every platform in view (only mesos if `mesos_only`)."""
        s = None
        if self.feed is not None and self.feed.alive:
            s = self.feed.look(self, mobs, wide, loot, mesos_only)
        served = s is not None
        if s is None:
            s = self.place(self.eyes.look())
            if mobs and s.tag is not None and s.where is not None:
                s.monsters = self.search(s, wide)
                s.searched, s.wide = True, wide
            if loot:
                s.loot = find_loot(s, self.model, self.eyes, wide, mesos_only)
            if self.feed is not None:
                self.counts["taken here"] += 1
        else:
            self.counts["from the feed"] += 1
        if self.feed is not None:
            # What the feed saw up to the frame this look shows (all of it for a look taken here, now): an EXP
            # gain of a later frame goes with the next look, whose monsters no longer show the one it killed.
            s.events = self.feed.drain_events(s.t if served else None) if self.feed.alive else None
        for listener in self.listeners:
            listener(s)
        return s

    def glance(self) -> tuple[tuple[float, float] | None, float]:
        """Her minimap dot and HP (Eyes.glance): with a feed, from its newest frame shown GLANCE_EVERY_S after
        the last glance's at the soonest, so what the navigator makes of two glances in a row (her dot
        standing still) holds as without it."""
        if self.feed is not None and self.feed.alive:
            st = self.feed.glance(self._glanced + GLANCE_EVERY_S)
            if st is not None:
                self._glanced = st.t
                self.counts["glances from the feed"] += 1
                return st.dot, st.hp
        return self.eyes.glance()

    def wait_frame(self, timeout: float) -> None:
        """Until the feed's next frame comes, `timeout` s at most; without a feed, `timeout` s."""
        if self.feed is not None and self.feed.alive:
            self.feed.wait_frame(timeout)
        else:
            time.sleep(timeout)

    def place(self, obs: Observation) -> Scene:
        """A look placed on the map: where she is (see _place_her); no monsters or drops yet."""
        s = Scene.of(obs)
        self._place_her(s)
        return s

    def search(self, s: Scene, wide: bool = False) -> list[Monster]:
        """The monsters in a placed look's band around her (or, if `wide`, on the whole screen), placed on the
        map with their HP bars (her screen position must be known)."""
        seen = self.eyes.monsters_wide(s) if wide else self.eyes.monsters(s, *self.band)
        return [s.place(m, self.model, self.eyes.bar(s, m)) for m in seen]

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
