"""Senses: every look at the game, placed on the map. Eyes (perception) says what's on screen; Senses
works out where she is (her minimap dot, refined by a rope or her platform's end on screen), where she
is on screen when her name tag is hidden (the camera), where the monsters and drops are, and where other
players are (their minimap dots, followed from look to look: localization.players).

Every look anyone takes goes through `Senses.look`, and the listeners run after each one, in order:
the hunter keeps count of her EXP and HP there, and the mini-map's telemetry is sent from there. Then its checks, which
may raise out of the look, wherever it was taken (the hunter's safety stop: Hunter._stop_now); and a glance's state
goes to its glance checks the same way (Hunter._glanced: her HP in a chain's watch, where no look is taken).

Every look and glance is the latest of what perception, looking at the game all the time, has seen (a
localization.feed.Feed: for a command the perception service, localization.service, whose world step runs on every
frame; in the simulator SyncFeed; on saved frames and the tests' made-up ones InlineFeed, the step run on the eyes' frame
when asked). A look is the newest state of a frame newer than the last look's, shown KEY_EFFECT_S after her last key
and MAX_AGE_S before the look at the most (`fresh_after`, KEYS_WAIT), with what it needs fresh (feed.fresh: her band
searched within 0.3 s, the whole screen within 0.5 s, the drops on that very frame), the events since the last look
with it (Scene.events); a look at her band has the monsters the feed follows off it too (`Scene.swept`, for her memory:
Feed.look). Glances are the newest frames, GLANCE_EVERY_S apart at the soonest. (Until 2026-10-25 a command could look
for itself, `--no-feed`, and such looks swept the screen a slice at a time for her memory; the user: "We should just
remove the --no-feed mode, perception should just always run and the other modules just work based on latest
perception result".)

No look is handed on that isn't of the game as expected: one whose state failed its checks (WorldState.bad: the eyes'
own, perception.framecheck: the game window, the frame lined up with it, the minimap's picture found; her place on the
map, localization.placecheck), or the game not wholly on screen, or no state at all (perception down), is asked again,
RETRY_GAP_S later; after LOST_S of that `look` raises LookLost, and nothing acts on the game until a look passes again
(a hunt stands still and looks every PAUSE_POLL_S: `regain`). A command that sends keys has the game brought to the
front meanwhile when it isn't (Feed.bring_forward). A glance with no state to give has no dot and the last HP; after
GLANCE_LOST_S of none, it raises LookLost too (a chain running is cut: control.navigator). What failed, and when it's
right again, is logged and sent as telemetry events.

A fault Alt+Enter twice mends (framecheck.FIX_KINDS: the client another size, the game drawing off it) is handed to
`window_fix` where the command gave one (control.window_size.SizeFix: the commands that act), at every look that has
nothing to act on for it: it makes its tries once the fault has stood a while, the hunt loop's while it stands still
(`regains`, Hunt.run); any other command's looks wait for its first try before LookLost. Once it has set the client
right, the looks after have LOST_S afresh, and no look or glance is of a frame shown before the window settled.
"""

import time
from collections.abc import Callable

from .. import telemetry
from ..character import THIEF_SHAPE, Shape
from ..perception.drops import items_for
from ..perception.eyes import GLANCE_EVERY_S, Eyes, Observation, band_rows, tag_below_feet
from ..perception.framecheck import FIX_KINDS, LookLost, Problem
from ..perception.terrain import MAX_ROPE_WIDTH
from ..screen import STRETCHED
from .mapdata import MapModel
from .monsters import Monster
from .placecheck import off_map
from .players import LOCAL_IDS, Players
from .pose import ROPE_MATCH_PX, Camera, Localizer, refine
from .scene import Scene

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
# What a look from the feed is by default (2026-10-25; the user: "perception should just always run and the other
# modules just work based on latest perception result"): the newest state of a frame
# - newer than the last look's (a look never decides on a frame twice),
# - shown KEY_EFFECT_S after the last key her plans sent at the soonest (`keyed`: Actions.last_key): the game acts on a
#   key 66-68 ms after it's sent (README, Motion planning), so the look sees what came of the keys before it,
# - and shown MAX_AGE_S before the look at the most;
# with its searches as fresh as a look needs (feed.fresh). Before, a look was the first frame shown after it was asked
# for, whatever came before. KEYS_WAIT says which looks wait for the keys' effect: every look ("all"), the one placing
# her after a move alone (Mover.locate's first: "moves"), or none. In the simulator (master 1c9d1e3, her jump attacks
# in; the hunt loop on Tree Dungeon, crowd, zero-walls, the attack choice, the costs measured, 16 seeds of 10 minutes;
# README, The perception service) every look so had her touched on her trips 16.4 times a run (their HP 1.45 bars)
# against 20.1 (1.66) for the mover's alone and 22.9 (1.85) for none, falling 33 times against 44 and 55, 75.4
# touches in all against 85.6 and 90.0, 0.73 bars of HP lost a minute against 0.83 and 0.88, her plans cut by a hit
# 30.2 times a run against 40.4 and 39.1, her EXP a minute level (225.3, 228.0, 235.4: the others +5.5% and +9.0% seed
# by seed, standard errors 7.1% and 7.9%), her decisions later (a plan's end to the next, 135 ms at the median against
# 65), and the Lie Detector's retreat on a safe place 10 s after the sighting on 27 of 31 runs with her moves alone on
# every one (the mover's alone 28 of 31, a run with more than her moves; none 29 of 30, one so). Measured first with
# what reads a look right after her keys taking a frame from before them for one after (feed.to_ask, Actions.heading,
# Attacks.on_throw), the mover's alone had looked safest (13.2 touches on her trips against 21.8, 8 seeds).
KEY_EFFECT_S = 0.068
MAX_AGE_S = 0.1
# A look for monsters (the next throw is decided on it) is of a frame shown when her last throw at a monster's kill has
# shown, if it killed (`shows`: behavior.attacks.KILL_SHOWN_S after its key), where that's at most this far off: she's
# locked until 0.79 s after the key (0.89 after a jump throw), arrows swallowed until 0.8, and a throw's decision took
# 13-21 ms after its look live (p99 79-95 ms, 2,737 throws of 2026-10-26), so a look 0.4 s after the key costs her no
# throwing time, and has the kill: the throw after it goes at the next monster at once. The hunt loop's look decides her
# trips too (Hunt._look: its pass throws from it), so it waits as well: live a trip chain after a throw would go out a
# median 0.15 s later where her lock didn't hold it anyway (145 of 230 in 74 minutes: 0.31 s a minute at most, against
# the drops' re-decisions' 0.46; D:/MapleHelper-scratch/akill/review-behaviour/trip_delay.py). Live (the grinds of 2026-10-26 21:04
# and 21:18) the look came a median 0.27 s after the key (its record), where the perception service had a kill's gain a
# median 0.22 s after the killing key, 97.4% of 237 by 0.43 s (D:/MapleHelper-scratch/akill/diag): 86 throws were
# dropped at their key for a kill come in since (4.4 a minute, Hunter._still) and decided again after the lock, and 36
# went at a monster dead by then. 0: none waits (looks as before).
SHOWS_WAIT_S = 0.45
# ... and such a look is asked this long before that frame at the soonest: asked at once, 0.33 s ahead, a look with a
# search of the whole screen outwaited the feed's FEED_WAIT_S now and then (the harness, the hunt loop: 5 looks "no state
# from perception in time" in 4 runs of 5 minutes, master none), each 0.2 s lost (RETRY_GAP_S)
SHOWS_LEAD_S = 0.05
KEYS_WAIT = "all"  # which looks wait for her keys' effect: "all", "moves" (the mover's), "none"
# Looks that fail their checks are taken again this long apart (each a capture: 5-30 ms), for up to LOST_S; then
# LookLost. The window dragged at 16:48:44 on 2026-10-24 gave four bad looks in 4.1 s (16:48:44.0-48.1); a look that
# fails for longer than LOST_S makes a hunt stand still (`regain`), looking again every PAUSE_POLL_S.
RETRY_GAP_S = 0.2
LOST_S = 3.0
PAUSE_POLL_S = 0.5
# A glance gets no state to act on this long (perception down, the game not on screen or not as expected): it raises
# LookLost, and a chain of keys running is cut, every key let go (control.navigator). A glance waits feed.GLANCE_WAIT_S
# (0.25 s) for a frame, so the second glance without one cuts it.
GLANCE_LOST_S = 0.3


class Senses:
    def __init__(self, eyes: Eyes, model: MapModel, band: tuple[float, float], feed=None, shape: Shape = THIEF_SHAPE):
        """`band`: how far above and below her feet (map units) a look searches for monsters. `feed`: what its looks
        and glances come from (localization.feed.Feed: a command's the perception service's, localization.service;
        the simulator's SyncFeed); None: the world step run on the eyes' own frame when a look asks (feed.InlineFeed:
        saved frames and tests; made on the first look, so a world step's own Senses, which only places and searches,
        makes none). `shape`: what her attacks are to localization (gamelab.character.Shape: its trackers', and how long
        a look for monsters waits for a kill to show, SHOWS_WAIT_S the thief's)."""
        self.eyes, self.model, self.band = eyes, model, band
        self.shape = shape
        self._feed = feed
        self.localizer = Localizer(model, geo=getattr(eyes, "geo", STRETCHED))  # (the simulator's: 1920x1080)
        self.camera = Camera(model.bounds)
        # Other players, from the looks placed here (a world step's), and from the feed's (Players.adopt): ids apart from
        # the feed's
        self.players = Players(self.localizer, 1 if feed is None else LOCAL_IDS)
        self.listeners: list[Callable[[Scene], None]] = []
        self.checks: list[Callable[[Scene], None]] = []  # after the listeners, every look: each may raise (the safety
                                                          # stop, behavior.safety: Hunter._stop_now)
        self.glance_checks: list[Callable] = []  # ... and every glance with a state (worldstate.WorldState): the same
        eyes.want_ropes = True  # ropes in view pin down her x
        eyes.rope_columns = self._rope_columns
        eyes.drop_names = items_for(model)  # the drops looked for: the map's monsters' (perception.drops)
        self._glanced = -1e18  # the frame time of the last glance
        self.glanced_at: float | None = None  # when the frame the last glance saw was shown (time.monotonic)
        self.glanced = None  # the state (worldstate.WorldState) the last glance came from; None: it had none
        self.counts = {"looks": 0, "glances": 0, "glances without a state": 0}
        self.failed = {"looks": 0, "lost": 0}  # looks that failed their checks, and times that made LookLost
        self.log: Callable[[str], None] = print  # where what failed is said (the hunter gives its own)
        self._failing: tuple[float, str] | None = None  # since when looks have failed in a row, and the last why said
        self.keyed: Callable[[], float] = lambda: -1e18  # when her last key went (the hunter's: Actions.last_key)
        # When a frame shows what her last throw at a monster did to it, if it killed it (the hunter's:
        # behavior.attacks.Attacks.shown_by, KILL_SHOWN_S after its key): a look for monsters waits for it (SHOWS_WAIT)
        self.shows: Callable[[], float] = lambda: -1e18
        self._looked = -1e18  # the frame time of the last look
        self._usable = -1e18  # when the last state to act on came (a look's or a glance's: GLANCE_LOST_S)
        self._hp = 1.0        # the last HP a state read (a glance without one keeps it)
        # What sets the game's client right with Alt+Enter twice (control.window_size.SizeFix, which the commands that
        # act give; None: the user sets it back): called with each look that has nothing to act on for a fault it
        # mends (FIX_KINDS), True if it did; `passed()` once a look passes again.
        self.window_fix = None
        self.settled = -1e18  # when the window last settled after it set it right: no frame shown before counts
        self._set_right = -1e18  # ... and when it said so: LOST_S counts from then
        # Whether LookLost is met by standing still and looking again (`regain`: the hunt loop, Hunt.run), where the
        # fix makes its tries; if not, a look with nothing to act on for its fault waits for its first try first.
        self.regains = False

    @property
    def feed(self):
        """What its looks and glances come from (see __init__)."""
        if self._feed is None:
            from .feed import InlineFeed
            self._feed = InlineFeed(self.eyes, self.model, self.band)
        return self._feed

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False, mesos_only: bool = False,
             after: float | None = None) -> Scene:
        """Her map position from the minimap dot; her screen position from her name tag, or from the
        dot and the camera when the tag is hidden; (if `mobs`) the monsters she could shoot, or (if
        `wide`) every monster on screen; and (if `loot`) the drops on her platform, or (if `wide`) on
        every platform in view (only mesos if `mesos_only`): the newest state of a frame as fresh as `fresh_after`
        says (a look for monsters, of the frame `shows` gives too, where that's SHOWS_WAIT_S off at most: her last
        throw at one's kill shown, if it killed), or with `after`, of a frame shown then or later (one a glance saw her standing still in once the keys were
        let go of, say: Navigator.recover). Raises LookLost once there has been none to act on for LOST_S (at once
        while there still is none, after that)."""
        while True:
            at = self.fresh_after() if after is None else max(after, self.settled)
            if after is None and mobs and at < (shows := self.shows()) <= time.monotonic() + self.shape.shows_wait_s:
                at = shows  # (a look for monsters, the next throw decided on it: her last one's kill shown, if any)
                if (wait := at - time.monotonic() - SHOWS_LEAD_S) > 0:
                    time.sleep(wait)  # (asked then: the feed's FEED_WAIT_S from about its frame, a search too)
            s = self.feed.look(self, mobs, wide, loot, mesos_only, at)
            if s is not None:
                break
            problem = self.feed.problem() or Problem("perception", "no state from perception in time")
            # The game brought forward first (once a second at most): after LOST_S every look raises at once, and a hunt
            # standing still on a game not in front (`regain`) would never bring it forward again otherwise.
            self.feed.bring_forward(problem)
            if self._mend(problem):
                continue  # (set right: the next look is of a frame shown after the window settled)
            self._failed(problem, hold=self._fixing(problem))  # (raises LookLost after LOST_S)
            time.sleep(RETRY_GAP_S)
        self._passed()
        self.counts["looks"] += 1
        self._looked, self._usable, self._hp = max(self._looked, s.t), time.monotonic(), s.hp
        self._learn(s)  # the feed placed her with its own camera; this one learns too (her name tag hidden)
        self.players.adopt(s.t, s.players)  # ... and the other players it follows
        if s.searched and not s.rows and s.tag is not None and s.where is not None:  # where it searched
            s.rows = [self._play_rows(s) if s.wide else self._map_rows(s, band_rows(s, *self.band))]
        # What the feed saw up to the frame this look shows: an EXP gain of a later frame goes with the next look, whose
        # monsters no longer show the one it killed.
        s.events = self.feed.drain_events(s.t)
        for listener in self.listeners:
            listener(s)
        for check in self.checks:
            check(s)
        return s

    def fresh_after(self, keys: bool | None = None) -> float:
        """When a look's frame may have been shown at the soonest, by default (KEY_EFFECT_S, MAX_AGE_S): after the last
        look's frame, KEY_EFFECT_S after her last key (with `keys`; by default as KEYS_WAIT says), and MAX_AGE_S before
        now."""
        keys = KEYS_WAIT == "all" if keys is None else keys
        return max(time.monotonic() - MAX_AGE_S, self._looked + 1e-6, self.keyed() + KEY_EFFECT_S if keys else -1e18,
                   self.settled)

    def _failed(self, problem: Problem, hold: bool = False) -> None:
        """A look found nothing to act on (`problem`): said once (and when the why changes); LookLost once that has
        gone on for LOST_S (since the window fix last set the client right, if that's later), unless `hold`."""
        now = time.monotonic()
        self.failed["looks"] += 1
        if self._failing is None or self._failing[1] != problem.why:
            if self._failing is None:
                self.log(f"  a look failed its checks: {problem.why}")
            else:
                self.log(f"  ... then: {problem.why}")
            telemetry.emit("event", kind="look", text=problem.why)
            self._failing = (now if self._failing is None else self._failing[0], problem.why)
        if now - max(self._failing[0], self._set_right) >= LOST_S and not hold:
            self.failed["lost"] += 1
            raise LookLost(problem)

    def _mend(self, problem: Problem) -> bool:
        """A look with nothing to act on for `problem`, handed to the window fix if it's one it mends: whether it set the
        client right now (the looks after then have LOST_S afresh, of frames shown after the window settled)."""
        fix = self.window_fix
        if fix is None or problem.kind not in FIX_KINDS or not fix(problem):
            return False
        self._set_right = time.monotonic()
        self.settled = max(self.settled, getattr(fix, "settled", None) or self._set_right)
        return True

    def _fixing(self, problem: Problem) -> bool:
        """Whether a look with nothing to act on for a fault the window fix mends is taken again past LOST_S: in a
        command that stops on LookLost (not `regains`), while the fix is still to make its first try at it, which it
        makes once the fault has stood window_size.STEADY_S (a plan holding it off is waited for a while:
        SizeFix.pending). Without this, LookLost (3 s) came before it, and those commands stopped untried."""
        fix = self.window_fix
        return problem.kind in FIX_KINDS and not self.regains and fix is not None and fix.pending()

    def _passed(self) -> None:
        """A look passed its checks: if looks had failed before it, that's over (said so)."""
        if self._failing is not None:
            text = f"the game looks as expected again ({time.monotonic() - self._failing[0]:.1f} s after a look failed)"
            self.log(f"  {text}")
            telemetry.emit("event", kind="look", text=text)
            if self.window_fix is not None:
                self.window_fix.passed()  # (the client right again: its going wrong again starts afresh)
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
        """Her minimap dot and HP from the feed's newest frame, one shown FEED_GLANCE_EVERY_S after the last glance's at
        the soonest (and after it), when its frame was shown (`glanced_at`), and its state (`glanced`: the monsters it
        follows, for a plan running to react to: control.navigator). With no state to act on: no dot and the last HP,
        and LookLost once there has been none for GLANCE_LOST_S."""
        st = self.feed.glance(max(self._glanced + max(FEED_GLANCE_EVERY_S, 1e-3), self.settled))
        if st is not None:
            self._glanced = self.glanced_at = st.t
            self.glanced, self._usable, self._hp = st, time.monotonic(), st.hp
            self.counts["glances"] += 1
            for check in self.glance_checks:
                check(st)
            return st.dot, st.hp
        self.glanced, self.glanced_at = None, time.monotonic()
        self.counts["glances without a state"] += 1
        if time.monotonic() - self._usable >= GLANCE_LOST_S:
            raise LookLost(self.feed.problem() or Problem("perception", "no state from perception in time"))
        return None, self._hp

    def wait_frame(self, timeout: float) -> None:
        """Until the feed's next frame comes, `timeout` s at most."""
        self.feed.wait_frame(timeout)

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
