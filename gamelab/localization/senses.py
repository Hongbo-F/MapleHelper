"""Senses: every look at the game, placed on the map. Eyes (perception) says what's on screen; Senses
works out where she is (her minimap dot, refined by a rope or her platform's end on screen), where she
is on screen when her name tag is hidden (the camera), where the monsters and drops are, and where other
players are (their minimap dots, followed from look to look: localization.players).

Every look anyone takes goes through `Senses.look`, and the listeners run after each one, in order:
the hunter keeps count of her EXP and HP there, and the mini-map's telemetry is sent from there.

With a continuous feed (localization.feed: the game looked at every frame in a process of its own), a look
is the feed's state of the first frame shown after it was asked for, with what it needs fresh (see
feed.fresh), and the events since the last look come with it (Scene.events); glances come from the feed's
newest frames. When the feed can't give one (the game not in front, the feed gone or too slow), the look or
glance is taken here as without it. Without a feed (the default) nothing changes.

A look taken here is never handed on unless it passed its checks: the eyes' (perception.framecheck: the game window,
the frame lined up with it, the minimap's picture found) and her place on the map (localization.placecheck). One that
fails is taken again, RETRY_GAP_S later, after the eyes have done what they can (Eyes.recover); after LOST_S of
failing, `look` raises LookLost, and nothing acts on the game until a look passes again (a hunt stands still and
looks every PAUSE_POLL_S: `regain`). What failed, and when it's right again, is logged and sent as telemetry events.
"""

import time
from collections.abc import Callable

from .. import telemetry
from ..perception.drops import items_for
from ..perception.eyes import GLANCE_EVERY_S, Eyes, Observation, tag_below_feet
from ..perception.framecheck import BadLook, LookLost, Problem
from ..perception.terrain import MAX_ROPE_WIDTH
from ..screen import STRETCHED
from .mapdata import MapModel
from .monsters import Monster
from .placecheck import off_map
from .players import LOCAL_IDS, Players
from .pose import ROPE_MATCH_PX, Camera, Localizer, refine
from .scene import Scene, find_loot

# Full-size px (1920x1080; the look's geometry converts them) either side of where her minimap dot puts each of the
# map's ropes on screen that a look searches for ropes: all `pose.rope_x` can use (a rope within ROPE_MATCH_PX of
# there, up to MAX_ROPE_WIDTH wide). 8.6 ms a look (median, bench-look), against 19 for the whole screen. On 17 saved
# frames the ropes seen were 0-20 px from there (her dot 0-14.5 map units off), the same ropes as on the whole
# screen, and she was placed the same. The mover's line-up on a rope looks 70 px around where the map
# puts it from her refined position: the same ropes while her dot is within about 30 units. None
# searches the whole screen.
ROPE_PAD: int | None = ROPE_MATCH_PX + MAX_ROPE_WIDTH
# Glances from a feed come this far apart at the soonest (their frames' times): GLANCE_EVERY_S, as glances taken here
# (63 ms: about 15 a second); 0, every frame the feed looks at (30 a second). The navigator judges glances in time
# (navigator.STILL_S, OFF_PATH_S), so what it makes of them means the same either way, but every frame didn't pay in
# the simulator (2026-10-21; README, Continuous perception): the nav_* trips on 10 seeds of the game's noise were 2.9%
# quicker but missed 16 landings of 290 against 11 (11 falls against 6), `recover` taking her for stopped once her
# dot had stood 46 ms twice as often while she ran (at 149 units/s it can sit 76 ms in one of this map's 11.4-unit
# minimap pixels), and over 20 seeds of 20 minutes the tour made 5.0% less EXP a minute (standard error 2.6%), the
# hunt loop 2.4% more (1.6%). Not tried in game.
FEED_GLANCE_EVERY_S = GLANCE_EVERY_S
# Looks that fail their checks are taken again this long apart (each a capture: 5-30 ms), for up to LOST_S; then
# LookLost. The window dragged at 16:48:44 on 2026-10-24 gave four bad looks in 4.1 s (16:48:44.0-48.1); a look that
# fails for longer than LOST_S makes a hunt stand still (`regain`), looking again every PAUSE_POLL_S.
RETRY_GAP_S = 0.2
LOST_S = 3.0
PAUSE_POLL_S = 0.5


class Senses:
    def __init__(self, eyes: Eyes, model: MapModel, band: tuple[float, float], feed=None):
        """`band`: how far above and below her feet (map units) a look searches for monsters. `feed`: a
        continuous feed to take looks and glances from (localization.feed.Feed), or None."""
        self.eyes, self.model, self.band = eyes, model, band
        self.feed = feed
        self.localizer = Localizer(model, geo=getattr(eyes, "geo", STRETCHED))  # (the simulator's: 1920x1080)
        self.camera = Camera(model.bounds)
        # Other players, from the looks placed here, and from the feed's (Players.adopt): ids apart from the feed's
        self.players = Players(self.localizer, 1 if feed is None else LOCAL_IDS)
        self.listeners: list[Callable[[Scene], None]] = []
        eyes.want_ropes = True  # ropes in view pin down her x
        eyes.rope_columns = self._rope_columns
        eyes.drop_names = items_for(model)  # the drops looked for: the map's monsters' (perception.drops)
        self._glanced = -1e18  # the frame time of the last glance from the feed
        self.glanced_at: float | None = None  # when the frame the last glance saw was shown (time.monotonic)
        self.counts = {"from the feed": 0, "taken here": 0, "glances from the feed": 0}
        self.failed = {"looks": 0, "lost": 0}  # looks that failed their checks, and times that made LookLost
        self.log: Callable[[str], None] = print  # where what failed is said (the hunter gives its own)
        self._failing: tuple[float, str] | None = None  # since when looks have failed in a row, and the last why said
        self._attempts = 0  # looks failed in a row (Eyes.recover goes by it)

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
            s = self._take()
            if mobs and s.tag is not None and s.where is not None:
                s.monsters = self.search(s, wide)
                s.searched, s.wide = True, wide
            if loot:
                s.loot = find_loot(s, self.model, self.eyes, wide, mesos_only)
            if self.feed is not None:
                self.counts["taken here"] += 1
        else:
            self.counts["from the feed"] += 1
            self._learn(s)  # the feed placed her with its own camera; this one learns too, for looks taken here
            self.players.adopt(s.t, s.players)  # ... and the other players it follows, for them too
        if self.feed is not None:
            # What the feed saw up to the frame this look shows (all of it for a look taken here, now): an EXP
            # gain of a later frame goes with the next look, whose monsters no longer show the one it killed.
            s.events = self.feed.drain_events(s.t if served else None) if self.feed.alive else None
        for listener in self.listeners:
            listener(s)
        return s

    def _take(self) -> Scene:
        """A look taken here and placed that passed its checks (see the module's docstring), taken again until one
        does; raises LookLost once looks have failed for LOST_S (at once while they still are, after that)."""
        while True:
            try:
                s = self.place(self.eyes.look())
                problem = s.bad
            except BadLook as e:
                problem = e.problem
            if problem is None:
                self._passed()
                return s
            now = time.monotonic()
            self.failed["looks"] += 1
            if self._failing is None or self._failing[1] != problem.why:
                if self._failing is None:
                    self.log(f"  a look failed its checks: {problem.why}")
                else:
                    self.log(f"  ... then: {problem.why}")
                telemetry.emit("event", kind="look", text=problem.why)
                self._failing = (now if self._failing is None else self._failing[0], problem.why)
            self.eyes.recover(problem, self._attempts)  # (before the next look, which may be a while: `regain`)
            self._attempts += 1
            if now - self._failing[0] >= LOST_S:
                self.failed["lost"] += 1
                raise LookLost(problem)
            time.sleep(RETRY_GAP_S)

    def _passed(self) -> None:
        """A look passed its checks: if looks had failed before it, that's over (said so)."""
        if self._failing is not None:
            text = f"the game looks as expected again ({time.monotonic() - self._failing[0]:.1f} s after a look failed)"
            self.log(f"  {text}")
            telemetry.emit("event", kind="look", text=text)
        self._failing, self._attempts = None, 0

    def regain(self, lost: LookLost, until: float) -> Scene | None:
        """After LookLost: nothing is sent while a look is taken every PAUSE_POLL_S, until one passes its checks (it's
        returned) or it's `until` (time.monotonic; None then). What's wrong is logged when it changes."""
        self.log(f"  standing still: {lost.why} (nothing is sent until the game looks as expected again)")
        telemetry.emit("event", kind="lost", text=lost.why)
        why = lost.why
        while time.monotonic() < until:
            time.sleep(min(PAUSE_POLL_S, max(0.0, until - time.monotonic())))
            try:
                return self.look(mobs=False)
            except LookLost as e:
                if e.why != why:
                    self.log(f"  still standing: {e.why}")
                    why = e.why
        self.log(f"  still lost when the time was up: {why}")
        return None

    def glance(self) -> tuple[tuple[float, float] | None, float]:
        """Her minimap dot and HP (Eyes.glance), and when its frame was shown (`glanced_at`): with a feed, from its
        newest frame, one shown FEED_GLANCE_EVERY_S after the last glance's at the soonest (and after it)."""
        if self.feed is not None and self.feed.alive:
            st = self.feed.glance(self._glanced + max(FEED_GLANCE_EVERY_S, 1e-3))
            if st is not None:
                self._glanced = self.glanced_at = st.t
                self.counts["glances from the feed"] += 1
                return st.dot, st.hp
        out = self.eyes.glance()
        self.glanced_at = self.eyes.seen_at if self.eyes.seen_at is not None else time.monotonic()
        return out

    def wait_frame(self, timeout: float) -> None:
        """Until the feed's next frame comes, `timeout` s at most; without a feed, `timeout` s."""
        if self.feed is not None and self.feed.alive:
            self.feed.wait_frame(timeout)
        else:
            time.sleep(timeout)

    def place(self, obs: Observation) -> Scene:
        """A look placed on the map: where she is (see _place_her), and the other players (Players: a look that
        didn't see her own dot, the minimap covered, keeps the players as they were); no monsters or drops yet. A look
        that failed its checks (`bad`: the eyes', or her place off the map) isn't placed, and keeps the players."""
        s = Scene.of(obs)
        if s.bad is None:
            self._place_her(s)
        s.players = self.players.update(s.t, obs.others) if s.dot is not None and s.bad is None else self.players.current
        return s

    def search(self, s: Scene, wide: bool = False) -> list[Monster]:
        """The monsters in a placed look's band around her (or, if `wide`, on the whole screen), placed on the
        map with their HP bars (her screen position must be known)."""
        seen = self.eyes.monsters_wide(s) if wide else self.eyes.monsters(s, *self.band)
        return [s.place(m, self.model, self.eyes.bar(s, m)) for m in seen]

    def _learn(self, s: Scene) -> None:
        """With her name tag seen and her placed, the camera learns where it keeps her."""
        if s.tag is not None and s.tag_seen and s.where is not None:
            k = s.geo.look_scale
            self.camera.learn((s.where.x, s.where.y), (k * s.tag[0], k * s.floor_y), geo=s.geo)

    def _rope_columns(self, dot: tuple[float, float] | None,
                      tag: tuple[int, int] | None) -> list[tuple[int, int]] | None:
        """Where a look searches for ropes (ranges of x, full-size px): around where her dot puts the map's
        ropes on screen (ROPE_PAD). With no dot or no name tag, ropes can't place her: nowhere."""
        if ROPE_PAD is None:
            return None
        if dot is None or tag is None:
            return []
        geo = getattr(self.eyes, "geo", STRETCHED)  # (asked by the eyes, in the middle of a look of their frames)
        self.localizer.use(geo)
        where, her_x, pad = self.localizer.where_dot(dot), geo.look_scale * tag[0], geo.ipx(ROPE_PAD)
        xs = (her_x + (rope.x - where.x) * geo.stretch for rope in self.model.ropes)
        return [(int(x) - pad, int(x) + pad + 1) for x in xs]

    def _place_her(self, s: Scene) -> None:
        """Her map position (`s.where`, `s.how`) from her minimap dot, refined on screen. Then, with her
        name tag seen, the camera learns where it keeps her; with it hidden, the camera places her
        on screen instead (`s.tag`, and `s.tag_seen` False)."""
        if s.dot is not None:
            self.localizer.use(s.geo)
            s.where = self.localizer.where_dot(s.dot)
            if (off := off_map(self.model, s.where)) is not None:  # not placed, and the camera learns nothing from it
                s.where, s.bad = None, Problem("place", off)
                return
            if s.tag is not None:
                s.where, s.how = refine(s.where, s, self.model, self.camera)
        if s.where is None:
            return
        if s.tag is not None:
            self._learn(s)
        else:
            k = s.geo.look_scale
            sx, sy = self.camera.to_screen(s.where.x, s.where.y, (s.where.x, s.where.y), s.geo)
            s.tag, s.tag_seen = (round(sx / k), round(sy / k) + tag_below_feet(s.geo)), False
