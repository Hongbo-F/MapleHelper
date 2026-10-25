"""Senses: every look at the game, placed on the map. Eyes (perception) says what's on screen; Senses
works out where she is (her minimap dot, refined by a rope or her platform's end on screen), where she
is on screen when her name tag is hidden (the camera), where the monsters and drops are, and where other
players are (their minimap dots, followed from look to look: localization.players).

Every look anyone takes goes through `Senses.look`, and the listeners run after each one, in order:
the hunter keeps count of her EXP and HP there, and the mini-map's telemetry is sent from there.

With the sweep on (`sweep`) and no feed serving her looks, a look at the band around her also searches one slice of the
rest of the screen, the next in turn from the top down (SWEEP_SLICE map units tall, overlapping by the tallest body): the whole view every few slices
(4 at the game's own 1366x768, standing on a platform), for a slice's search more. What a slice finds goes into
`Scene.swept`, not `Scene.monsters`: the fights go by the band's monsters as before. `sweep_hold`, asked after the
band's search, holds the slice back where it would cost her (behavior.hunter.Hunter.hold_sweep: a monster in her band
and nothing to wait for), so the look comes back sooner. The memory of monsters (localization.memory) takes every
look's band and slice (`Scene.rows` says where a look searched).

With a continuous feed (localization.feed: the game looked at every frame in a process of its own), a look
is the feed's state of the first frame shown after it was asked for, with what it needs fresh (see
feed.fresh), and the events since the last look come with it (Scene.events); a look at her band has the monsters the
feed follows off it too (`Scene.swept`, for her memory: Feed.look), so there's no sweep while the feed serves her
looks. Glances come from the feed's newest frames. When the feed can't give one (the game not in front, the feed gone
or too slow), the look or glance is taken here as without it, and once the feed is gone her looks sweep again; a feed
whose looks keep coming slowly (starved of the CPU) is rested a while, her looks and glances taken here meanwhile
(FEED_SLOW_S, FEED_REST_S). Without a feed nothing changes.

A look taken here is never handed on unless it passed its checks: the eyes' (perception.framecheck: the game window,
the frame lined up with it, the minimap's picture found) and her place on the map (localization.placecheck). One that
fails is taken again, RETRY_GAP_S later, after the eyes have done what they can (Eyes.recover); after LOST_S of
failing, `look` raises LookLost, and nothing acts on the game until a look passes again (a hunt stands still and
looks every PAUSE_POLL_S: `regain`; while the window is unfit in a way only the user sets right, another size, say, a
look asks the window alone: Eyes.window_wait). What failed, and when it's right again, is logged and sent as telemetry
events.
"""

import math
import time
from collections import deque
from collections.abc import Callable

from .. import telemetry
from ..perception.drops import items_for
from ..perception.eyes import GLANCE_EVERY_S, Eyes, Observation, band_rows, tag_below_feet
from ..perception.framecheck import BadLook, LookLost, Problem
from ..perception.terrain import MAX_ROPE_WIDTH
from ..screen import STRETCHED
from .mapdata import MapModel
from .monsters import Monster, bodies
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
# The sweep (Senses.sweep): each slice of the screen it searches is this many map units tall (193 rows at the game's own
# 1366x768, as tall as each of a whole-screen search's bands there: perception.mobs.detect_wide), and slices overlap by
# the tallest monster body of the map and SWEEP_BODY_SLACK, so a body cut by one slice's edge is whole in the next (and
# one cut by the band's edge, in the slice beside it). On the 20 saved 1366x768 frames of Tree Dungeon, Forest Up North
# II, one such slice took 16.7 ms (median; 25.4 at p90), against 21.9 for her band and 72.9 for the whole screen
# (bench-look), and the slices together found the 53 monsters the whole-screen search did, at the same places.
SWEEP_SLICE = 193.0
SWEEP_BODY_SLACK = 8.0
# A feed whose looks come slowly (its child starved of the CPU: it runs at below-normal priority) is rested: once
# FEED_SLOW_LOOKS looks in a row at her band or her place (not the whole screen, whose search takes 0.1-0.25 s however it's
# taken) have waited FEED_SLOW_S or more for it, or one wasn't given at all (it waited FEED_WAIT_S), her looks and glances
# are taken here for FEED_REST_S; then the next look tries the feed again, and rests it again at once if that one is slow
# too. On the 20 saved 1366x768 frames of Tree Dungeon with the machine 93% busy (2026-10-24; other work at normal and idle
# priority), the child looked at 12 frames a second, a look at her band waited 149 ms for it (median; p90 655) against 61
# taken here, and 30 of 226 looks waited out FEED_WAIT_S; resting it so (the machine 74-94% busy), the feed was rested 17
# times in a minute, 217 of 329 looks were taken here, and a look at her band waited 62 ms (p90 214) against 61 (104).
# FEED_SLOW_S is a guess: a look at her band taken here costs about 40 ms at the game's own size, and the feed's as much
# with its frames coming (README, Continuous perception). In the simulator the feed is never rested.
FEED_SLOW_S = 0.15
FEED_SLOW_LOOKS = 3
FEED_REST_S = 2.0


class Senses:
    def __init__(self, eyes: Eyes, model: MapModel, band: tuple[float, float], feed=None, sweep: bool = False):
        """`band`: how far above and below her feet (map units) a look searches for monsters. `feed`: a
        continuous feed to take looks and glances from (localization.feed.Feed), or None. `sweep`: a look at the band
        also searches a slice of the rest of the screen (see the top; only while no feed is up: the feed searches the
        whole screen every second, and follows what it found)."""
        self.eyes, self.model, self.band = eyes, model, band
        self.feed = feed
        self.sweep = sweep
        self.sweep_hold: Callable[[Scene], bool] | None = None  # asked after the band's search: hold the slice back
        self._sweep_row = 0.0  # map units below the view's top where the next slice starts
        self._tallest = max((b.top for b in bodies(model).values()), default=60.0)  # the map's tallest monster
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
        self.glanced = None  # the feed's state (worldstate.WorldState) the last glance came from; None: taken here
        self.counts = {"from the feed": 0, "taken here": 0, "glances from the feed": 0, "feed rested": 0}
        self._feed_waits: deque[float] = deque(maxlen=FEED_SLOW_LOOKS)  # the latest looks' waits for the feed
        self._feed_rest = -1e18  # no looks or glances from the feed before this (FEED_REST_S)
        self.failed = {"looks": 0, "lost": 0}  # looks that failed their checks, and times that made LookLost
        self.log: Callable[[str], None] = print  # where what failed is said (the hunter gives its own)
        self._failing: tuple[float, str] | None = None  # since when looks have failed in a row, and the last why said
        self.sweep_counts = {"slices": 0, "held": 0}  # slices the sweep searched, and held back (sweep_hold)

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False, mesos_only: bool = False,
             after: float | None = None) -> Scene:
        """Her map position from the minimap dot; her screen position from her name tag, or from the
        dot and the camera when the tag is hidden; (if `mobs`) the monsters she could shoot, or (if
        `wide`) every monster on screen; and (if `loot`) the drops on her platform, or (if `wide`) on
        every platform in view (only mesos if `mesos_only`). `after`: from a feed, a frame shown then or later
        will do (None: now, as it's asked for): one a glance from it saw her standing still in once the keys were
        let go of, say (Navigator.recover)."""
        s = None
        if self._feed_up():
            asked = time.monotonic()
            s = self.feed.look(self, mobs, wide, loot, mesos_only, after)
            if not wide:
                self._feed_paced(s is None, time.monotonic() - asked)
        served = s is not None
        if s is None:
            s = self._take()
            if mobs and s.tag is not None and s.where is not None:
                s.monsters = self.search(s, wide)
                s.searched, s.wide = True, wide
                if not wide and self.sweep and (self.feed is None or not self.feed.alive):
                    if self.sweep_hold is not None and self.sweep_hold(s):
                        self.sweep_counts["held"] += 1
                    else:
                        self._sweep_slice(s)
            if loot:
                s.loot = find_loot(s, self.model, self.eyes, wide, mesos_only)
            if self.feed is not None:
                self.counts["taken here"] += 1
        else:
            self.counts["from the feed"] += 1
            self._learn(s)  # the feed placed her with its own camera; this one learns too, for looks taken here
            self.players.adopt(s.t, s.players)  # ... and the other players it follows, for them too
            if s.searched and not s.rows and s.tag is not None and s.where is not None:  # where it searched
                s.rows = [self._play_rows(s) if s.wide else self._map_rows(s, band_rows(s, *self.band))]
        if self.feed is not None:
            # What the feed saw up to the frame this look shows (all of it for a look taken here, now): an EXP
            # gain of a later frame goes with the next look, whose monsters no longer show the one it killed.
            s.events = self.feed.drain_events(s.t if served else None) if self.feed.alive else None
        for listener in self.listeners:
            listener(s)
        return s

    def _feed_up(self) -> bool:
        """Whether looks and glances come from the feed now: there's one, it's alive, and it isn't resting
        (FEED_REST_S)."""
        return self.feed is not None and self.feed.alive and time.monotonic() >= self._feed_rest

    def _feed_paced(self, refused: bool, waited: float) -> None:
        """A look at her band or place asked of the feed waited `waited` s (`refused`: and wasn't given): the feed is
        rested if it's slow (FEED_SLOW_S, FEED_SLOW_LOOKS), said in the log and as telemetry."""
        self._feed_waits.append(math.inf if refused else waited)
        if refused or (len(self._feed_waits) == FEED_SLOW_LOOKS and min(self._feed_waits) >= FEED_SLOW_S):
            self._feed_rest = time.monotonic() + FEED_REST_S
            self.counts["feed rested"] += 1
            text = (f"the feed is slow (its last looks waited "
                    f"{', '.join('out' if w == math.inf else f'{1000 * w:.0f} ms' for w in self._feed_waits)}): "
                    f"looks taken here for {FEED_REST_S:.0f} s")
            if self.counts["feed rested"] <= 3 or self.counts["feed rested"] % 50 == 0:  # (not every time)
                self.log(f"  {text} ({self.counts['feed rested']} times)")
            telemetry.emit("event", kind="feed", text=text)

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
            self.eyes.recover(problem)  # (before the next look, which may be a while: `regain`)
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
        self._failing = None

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
        newest frame, one shown FEED_GLANCE_EVERY_S after the last glance's at the soonest (and after it), its state
        kept (`glanced`: the monsters it follows, for a plan running to react to: control.navigator)."""
        if self._feed_up():
            st = self.feed.glance(self._glanced + max(FEED_GLANCE_EVERY_S, 1e-3))
            if st is not None:
                self._glanced = self.glanced_at = st.t
                self.glanced = st
                self.counts["glances from the feed"] += 1
                return st.dot, st.hp
        self.glanced = None
        out = self.eyes.glance()
        self.glanced_at = self.eyes.seen_at if self.eyes.seen_at is not None else time.monotonic()
        return out

    def wait_frame(self, timeout: float) -> None:
        """Until the feed's next frame comes, `timeout` s at most; without a feed (or while it rests), `timeout` s."""
        if self._feed_up():
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
        map with their HP bars (her screen position must be known). Where it searched goes into `s.rows`."""
        seen = self.eyes.monsters_wide(s) if wide else self.eyes.monsters(s, *self.band)
        s.rows.append(self._play_rows(s) if wide else self._map_rows(s, band_rows(s, *self.band)))
        return [s.place(m, self.model, self.eyes.bar(s, m)) for m in seen]

    # --- the sweep ------------------------------------------------------------------------------------

    def _sweep_slice(self, s: Scene) -> None:
        """One slice of the screen outside the band (the next in turn, from the top down), searched for monsters into
        `s.swept` (those the band found too left out) and `s.rows`."""
        rows = self._next_slice(s)
        if rows is None:
            return
        self.sweep_counts["slices"] += 1
        found = [s.place(m, self.model, self.eyes.bar(s, m)) for m in self.eyes.monsters_rows(s, rows)]
        s.swept = [m for m in found if not any(o.kind == m.kind and abs(o.x - m.x) < 25 and abs(o.y - m.y) < 25
                                               for o in s.monsters)]
        s.rows.append(self._map_rows(s, rows))

    def _next_slice(self, s: Scene) -> tuple[int, int] | None:
        """The rows of the frame the sweep searches next: SWEEP_SLICE tall, from where the last one ended (less the
        overlap), within what the band leaves out: above its top (and a body's height into it, for one cut by its
        edge) and below its bottom (likewise). Past the bottom it starts again at the top."""
        geo = s.geo
        bottom = int(s.frame.shape[0] * geo.hud_top)
        b0, b1 = band_rows(s, *self.band)
        body = int(math.ceil((self._tallest + SWEEP_BODY_SLACK) / geo.units_per_px))
        size = max(body + 1, int(round(SWEEP_SLICE / geo.units_per_px)))
        regions = [(0, min(bottom, b0 + body)), (max(0, b1 - body), bottom)]
        if regions[0][1] >= regions[1][0]:  # the band is no taller than two bodies: the whole screen
            regions = [(0, bottom)]
        regions = [(a, b) for a, b in regions if b - a > body]
        if not regions:
            return None
        start = int(round(self._sweep_row / geo.units_per_px))
        region = next(((a, b) for a, b in regions if b - max(a, start) > body), None)
        if region is None:  # past the last: from the top again
            region, start = regions[0], regions[0][0]
        top = max(region[0], start)
        end = min(region[1], top + size)
        if region[1] - end <= body:  # not worth a slice of its own: this one goes to the region's end
            end = region[1]
        last = region is regions[-1] and end >= region[1]
        self._sweep_row = 0.0 if last else (end - body) * geo.units_per_px
        return top, end

    @staticmethod
    def _map_rows(s: Scene, rows: tuple[int, int]) -> tuple[float, float]:
        """Rows of a look's frame as map y (from, to)."""
        return s.map_y(rows[0]), s.map_y(rows[1])

    def _play_rows(self, s: Scene) -> tuple[float, float]:
        """The whole screen above the bottom bar, as map y (from, to)."""
        return self._map_rows(s, (0, int(s.frame.shape[0] * s.geo.hud_top)))

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
