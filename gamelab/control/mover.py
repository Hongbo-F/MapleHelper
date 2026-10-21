"""Carrying out route moves in the game: one look, one key plan per move, one look to check.

Each move is a single plan from wherever she stands, the way a player does it: run towards the end
and jump on the way (no stopping at a run-up spot first), run at a rope and jump onto it holding up
(no lining up underneath first), walk and drop in one go. Only climbing down a rope still lines up
first, since she has to stand on its top to grab it. A jump off a rope sideways (rope_jump) looks once
she's on the rope, and puts her at the height it jumps from before it does.

Her position comes from Senses.look (localization.pose.refine): a rope in view or the visible end of
her platform when her name tag shows, else the minimap dot.
"""

import math
import time

from maplehelper.engine import LEFT, RIGHT

from .. import directions
from ..localization.mapdata import MapModel, Platform, Rope
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..motion.body import ROPE_GROUP
from ..motion.flow import ARROW_LAG
from ..motion.physics import CLIMB_SPEED, GRAVITY, JUMP_LAG, JUMP_SPEED, ROPE_REACH, WALK_SPEED, fly, hold_ms, run_ms
from ..motion.route import JUMP_FROM, ROPE_TOP_DY, Move, drop_clear, drop_spots, grab_height, hung_from, side_jump
from ..screen import SCALE, SCREEN_SCALE
from .actions import WALK_PICK_AFTER_MS, Actions, tap


SLIDE_MS = 220            # after releasing the arrow she slides to a stop within this
X_TOLERANCE = 8           # map units: close enough when lining up
SPOT_TOLERANCE = 6        # map units: close enough to a run-up spot, or under a rope
ROPE_TOLERANCE = 6        # screen px: close enough to a rope's top to climb down it
ROPE_SEARCH_PX = 70       # screen px: a rope this close to where the map puts it is the one
GRAB_MARGIN = 8           # grab with at least this much rope to spare
# The shortest run-up a jump is timed for: 15 units is keyed 155 ms after the arrow (physics.run_ms), past
# the blind spot (physics.BLIND_ENDS_MS, 150), and leaves at full speed but for 0.01% of jumps by the median
# lags (body.short_takeoff_chance). A shorter one would be put off to 150 ms and lift her off 14 units along,
# not where it was planned for. It was 12 (keyed 127 ms in), before the jump sweeps of 2026-10-21: there 3 of
# the 4 jumps keyed 120 ms after the arrow left at 126 units/s, three of the game's 30 ms steps into the run.
MIN_RUN = 15              # map units ...
RUN_UP_SPARE = 10         # ... and backing off for one goes this much further
CLIMB_EXTRA_MS = 500      # up (or down) is held this much longer than the climb takes
# A jump off a rope (rope_jump): once on the rope, a look; off by more than HANG_TOLERANCE from the height it jumps
# from (her dot's raw height: it's good to ~11 units, and snaps to a platform within 25 of her), a climb up or down
# to it and another look. On the rope means her dot within ON_ROPE_X of it and between its ends, not at the floor she
# jumped from. The jump key goes ROPE_JUMP_KEY_MS after the arrow: the game sees the arrow 60 ms after its key, and
# acts on the jump 67 ms after its own (off_rope's, 30).
HANG_TOLERANCE = 15
ON_ROPE_X = 25
ROPE_JUMP_KEY_MS = 40


def grab_lead(gap: float) -> float | None:
    """How far before a rope to jump (running at her walking speed) to grab it at the top of the jump,
    when its bottom is `gap` above her feet; None if a jump can't reach it."""
    if gap > ROPE_REACH - GRAB_MARGIN:
        return None
    return WALK_SPEED * JUMP_SPEED / GRAVITY  # ~41 units: she's at the top of the jump over the rope


def climb_ms(height: float) -> int:
    """How long to hold up (or down) to climb `height` map units of rope."""
    return int(1000 * height / CLIMB_SPEED) + CLIMB_EXTRA_MS


class Mover:
    """Carries out route moves one at a time, each from a look at where she is."""

    def __init__(self, senses: Senses, actions: Actions, model: MapModel, log=print):
        self.senses, self.actions, self.model, self.log = senses, actions, model, log

    # --- looking ----------------------------------------------------------------------------

    def locate(self, tries: int = 40, mobs: bool = False, wide: bool = False) -> Scene:
        """A look that places her (its `where` is set). Waits up to ~10 s for her minimap dot (a window
        or notification popping up over the minimap hides it)."""
        for i in range(tries):
            s = self.senses.look(mobs=mobs, wide=wide)
            if s.where is not None:
                return s
            if i == 3:
                self.actions.nudge_mouse()  # a tooltip from another program once hid the minimap
            time.sleep(0.25)
        raise RuntimeError("can't find her minimap dot")

    def settle(self, seconds: float = 2.0) -> Scene:
        """Look until she's standing still on a platform (or give up and return the last look)."""
        end = time.monotonic() + seconds
        last = self.locate()
        while time.monotonic() < end:
            time.sleep(0.08)
            s = self.locate()
            if s.where.platform is not None and last.where.platform is s.where.platform \
                    and math.dist(last.where.dot, s.where.dot) < 0.6:
                return s
            last = s
        return last

    # --- walking ----------------------------------------------------------------------------

    def _send(self, events, name):
        """Run a move's key plan (in any order: it's sorted by time here)."""
        self.actions.play(sorted(events, key=lambda e: e[0]), name)
        self.actions.facing = None  # moves turn her; the next throw turns her to its target

    def walk_to(self, x: float, tolerance: float = X_TOLERANCE, tries: int = 4, s: Scene | None = None) -> Scene:
        """Walk along her platform until her feet are at map x."""
        s = s or self.locate()
        for _ in range(tries):
            dx = x - s.where.x
            if abs(dx) <= tolerance:
                return s
            events, _ = self.walk_events(dx, tolerance)
            self._send(events, "walk")
            time.sleep(0.25)  # let her slide to a stop before looking
            s = self.locate()
        return s

    def walk_events(self, dx: float, tolerance: float) -> tuple[list, int]:
        """Key events to walk dx map units (nothing if within tolerance), and when she's stopped."""
        if abs(dx) <= tolerance:
            return [], 0
        arrow = self.actions.arrow(directions.toward(dx))
        ms = hold_ms(abs(dx))
        # Pick-up pressed all the way, as on every walk (Actions.walk): anything she passes over is looted.
        events = [[0, arrow, "down"], [ms, arrow, "up"]] + self.actions.pickups(ms + WALK_PICK_AFTER_MS)
        return events, ms + SLIDE_MS

    def line_up_rope(self, rope_x: float) -> Scene:
        """Stand right on top of the rope at map x: walk there, then fine-tune on screen."""
        s = self.walk_to(rope_x, tolerance=12)
        for _ in range(3):
            if not (s.tag and s.tag_seen):
                return s
            her = SCALE * s.tag[0]
            expected = her + (rope_x - s.where.x) * SCREEN_SCALE
            near = [r for r in s.ropes if abs(r.x - expected) <= ROPE_SEARCH_PX]
            if not near:
                return s
            target = min(near, key=lambda r: abs(r.x - expected)).x
            dx = target - her
            if abs(dx) <= ROPE_TOLERANCE:
                return s
            arrow = self.actions.arrow(directions.toward(dx))
            # As long as the engine says a nudge that far takes (physics.hold_ms: 13 px, 9 units, is 57 ms).
            # Timed by her walking speed alone, 13 px was 87 ms, which by the engine walks her 19 units.
            ms = hold_ms(abs(dx) / SCREEN_SCALE)
            self._send([[0, arrow, "down"], [ms, arrow, "up"]], "nudge")
            time.sleep(0.2)
            s = self.locate()
        return s

    # --- route moves ------------------------------------------------------------------------

    def do(self, m: Move, s: Scene | None = None) -> Scene:
        """Carry out move `m` from where she is (`s`, or a fresh look) and return a look once she's landed."""
        s = s or self.locate()
        by_kind = {"climb": self._climb, "climb_down": self._climb_down, "jump_up": self._jump_up,
                   "jump": self._run_and_jump, "walk_off": self._walk_off, "drop": self._drop,
                   "rope_jump": self._rope_jump}
        if m.kind not in by_kind:
            raise ValueError(f"unknown move {m.kind}")
        by_kind[m.kind](m, s)
        return self.settle()

    def _climb_down(self, m: Move, s: Scene) -> None:
        """Stand on top of the rope (lining up looks again), then hold down all the way."""
        self.line_up_rope(m.rope_x)
        down = self.actions.keys.down.name
        self._send([[0, down, "down"], [climb_ms(m.rope_bottom - m.rope_top), down, "up"]], "climb_down")

    def _jump_up(self, m: Move, s: Scene) -> None:
        """Walk to the takeoff spot (unless she's there) and jump straight up."""
        events, t = self.walk_events(m.x - s.where.x, 15)
        self._send(events + tap(t, self.actions.keys.jump.name), "jump_up")
        time.sleep(0.45)  # then settle looks for her standing still

    def _walk_off(self, m: Move, s: Scene) -> None:
        """Walk off the end of her platform, on to where she lands: from MIN_RUN in from the end at least, so she
        leaves it at full speed (7.9 units after the game sees the arrow), as the route has her. From 0.4 units
        in she left 31's left end at 35 units/s, came down past 34's end and through the basin's cliff below it,
        out of the map (harness, tour seed 3)."""
        sign = directions.sign(m.direction)
        p = self.model.platform(m.src)
        edge = p.right if sign > 0 else p.left
        if (edge - s.where.x) * sign < MIN_RUN:  # too close to the end for a run-up: back off first
            s = self.walk_to(edge - sign * (MIN_RUN + RUN_UP_SPARE), tolerance=SPOT_TOLERANCE, s=s)
        arrow = self.actions.arrow(m.direction)
        ms = hold_ms(abs(m.land_x - s.where.x) + 20)
        self._send([[0, arrow, "down"], [ms, arrow, "up"]], "walk_off")
        time.sleep(0.3)  # then settle looks for her standing still

    def _drop(self, m: Move, s: Scene) -> None:
        """Walk to the drop spot (unless she's there) and drop through her platform: down + jump. Within 20
        units of the move's spot is close enough, unless a drop there wouldn't go through (by a rope's top,
        or on a foothold that forbids it) or would come down by a cliff of the platform below (route.KNOCK_CARRY):
        then to the nearest spot that would do (route.drop_spots)."""
        down, jump = self.actions.keys.down.name, self.actions.keys.jump.name
        a, here, below = self.model.platform(m.src), s.where.x, self.model.platform(m.dst)
        dx, tolerance = m.x - here, 20
        if abs(dx) <= tolerance and not drop_clear(self.model, a, here, below):
            spots = drop_spots(self.model, a, below)
            if spots:
                dx, tolerance = min(spots, key=lambda x: abs(x - here)) - here, 0
        events, t = self.walk_events(dx, tolerance)
        self._send(events + [[t, down, "down"], *tap(t + 120, jump), [t + 320, down, "up"]], "drop")
        time.sleep(0.3)  # then settle looks for her standing still

    def _run_and_jump(self, m: Move, s: Scene) -> None:
        """Run towards the end and jump on the way: jump is pressed for her to leave the ground at the move's takeoff
        (JUMP_FROM from the end, for a move without one)."""
        sign = directions.sign(m.direction)
        p = self.model.platform(m.src)
        edge = p.right if sign > 0 else p.left
        takeoff = m.takeoff if m.takeoff is not None else edge - sign * JUMP_FROM
        run = (takeoff - s.where.x) * sign  # how far she runs before jumping
        if run < MIN_RUN:  # too close to the end (or past the takeoff): back up to the run-up start first
            s = self.walk_to(m.x, tolerance=SPOT_TOLERANCE, s=s)
            run = (takeoff - s.where.x) * sign
        arrow, jump = self.actions.arrow(m.direction), self.actions.keys.jump.name
        t = run_ms(max(run, MIN_RUN))
        # The arrow is let go just before she lands (flow.ARROW_LAG, as a chain's stop on landing), and she slides
        # 14 units on. Held through a level jump's air time and 150 ms more, it ran her 59 units on past a landing
        # 60 up (0.41 s in the air, not 0.56), and off the far end of 19 from 20 (move_jump_short_landing).
        _, _, air = fly(self.model, takeoff, p.y_at(takeoff), sign * WALK_SPEED, -JUMP_SPEED, group=p.group)
        hold = int(t + 1000 * (JUMP_LAG + air - ARROW_LAG))
        self._send([[0, arrow, "down"], *tap(t, jump), [hold, arrow, "up"]], "jump")

    def _climb(self, m: Move, s: Scene, to: float | None = None) -> None:
        """Grab the rope on the way: run at it and jump holding up (or just walk into it holding up
        when it comes down to her feet). Standing under it already: jump straight up holding up. Up is
        held to the rope's top, or (`to`) about as long as it takes from where she grabs it to there."""
        k = self.actions.keys
        p = self.model.platform(m.src)
        gap = p.y_at(m.rope_x) - m.rope_bottom  # how far above her feet the rope ends
        if to is None:
            hold_up = climb_ms(p.y_at(m.rope_x) - m.rope_top)
        else:
            on = grab_height(p.y_at(m.rope_x), _rope(self.model, m))
            hold_up = max(0, int(1000 * (on - to) / CLIMB_SPEED))
        dx = m.rope_x - s.where.x
        if abs(dx) <= SPOT_TOLERANCE:
            self._climb_from_under(hold_up)
            return
        sign = 1 if dx > 0 else -1
        walk_in = gap <= 10 and not self.model.portal_near(p, m.rope_x)  # up at a portal would take it
        lead = 0.0 if walk_in else grab_lead(gap)
        if lead is None:
            raise ValueError(f"rope at {m.rope_x:.0f} ends {gap:.0f} above her platform: out of reach")
        run = abs(dx) - lead
        if run < MIN_RUN:  # too close to jump at it: back off to a run-up first (on her platform)
            back = s.where.x - sign * (MIN_RUN - run + RUN_UP_SPARE)
            if not p.covers(back, -10):
                # No room to back off: stand under it instead (a rope past her platform's end: as far back as it goes).
                back = m.rope_x if p.covers(m.rope_x) else p.clamp(back)
            s = self.walk_to(back, tolerance=SPOT_TOLERANCE, s=s)
            if back == m.rope_x:
                # Under it now (within the walk's SPOT_TOLERANCE). Calling _climb again here recursed
                # forever when the walk stopped 5-6 units off, past the 4 it took as "under".
                self._climb_from_under(hold_up)
                return
            dx = m.rope_x - s.where.x
            sign, run = (1 if dx > 0 else -1), abs(dx) - lead
        arrow = self.actions.arrow(directions.toward(sign))
        t = run_ms(max(run, MIN_RUN))
        at_rope = t + int(1000 * lead / WALK_SPEED)
        # Up is held from the jump on (in the air, so a portal on the way can't take her), or when
        # walking into a rope, from shortly before she reaches it.
        up_from = t if lead else max(0, t - 250)
        events = [[0, arrow, "down"], [up_from, k.up.name, "down"], [at_rope + 120, arrow, "up"],
                  [at_rope + hold_up, k.up.name, "up"]]
        if lead:
            events += tap(t, k.jump.name)
        self._send(events, "climb")

    def _climb_from_under(self, hold_up: int) -> None:
        """Standing under the rope: jump straight up holding up."""
        k = self.actions.keys
        self._send([[0, k.up.name, "down"], *tap(40, k.jump.name), [hold_up + 350, k.up.name, "up"]], "climb")

    def _rope_jump(self, m: Move, s: Scene) -> None:
        """Onto the rope (as a climb does, or from its top as a climb down does) and to about the height the jump goes
        from (m.hang); a look there, and a climb up or down to it if she's off by more than HANG_TOLERANCE; then off
        it sideways: the arrow, then jump, holding up for the rope she jumps onto (m.onto) and on up it onto its top,
        or letting go of the arrow just before she lands on m.dst. Not on the rope at the look, nothing more: the
        navigator replans from where she is."""
        k = self.actions.keys
        p, rope = self.model.platform(m.src), _rope(self.model, m)
        if abs(p.y_at(m.rope_x) - m.rope_top) <= ROPE_TOP_DY:  # from its top: down onto it
            self.line_up_rope(m.rope_x)
            down = k.down.name
            self._send([[0, down, "down"], [int(1000 * (m.hang - m.rope_top) / CLIMB_SPEED), down, "up"]], "climb_down")
        else:
            self._climb(m, s, to=m.hang)
        x, y = self._on_rope()
        if not self._hanging(x, y, m, p):
            self.log(f"    not on the rope at x {m.rope_x:.0f} (her dot at {x:.0f}, {y:.0f}): no jump off it")
            return
        if abs(y - m.hang) > HANG_TOLERANCE:
            key = (k.up if y > m.hang else k.down).name
            self._send([[0, key, "down"], [int(1000 * abs(y - m.hang) / CLIMB_SPEED), key, "up"]], "climb")
            x, y = self._on_rope()
            if not self._hanging(x, y, m, p):
                return
        sign = directions.sign(m.direction)
        arrow, jump = self.actions.arrow(m.direction), k.jump.name
        off = ROPE_JUMP_KEY_MS + 1000 * JUMP_LAG  # when she leaves the rope (ms)
        events = [[0, arrow, "down"], *tap(ROPE_JUMP_KEY_MS, jump)]
        got = side_jump(self.model, rope, min(max(y, rope.top), rope.bottom), sign, grab=m.onto is not None)
        if m.onto is not None:
            # Up from just after she leaves the rope (on it, up would climb), held up the rope she grabs and over its top.
            grab_y, t = (got[2], got[3]) if got is not None and got[0] == "rope" else (m.onto.bottom, 0.5)
            grab = int(off + 1000 * t)
            events += [[int(off) + 10, k.up.name, "down"], [grab + climb_ms(grab_y - m.onto.top), k.up.name, "up"],
                       [grab + 100, arrow, "up"]]
        else:
            t = got[3] if got is not None else 0.5
            events += [[int(off + 1000 * (t - ARROW_LAG)), arrow, "up"]]
        self._send(events, "rope_jump")
        time.sleep(0.2)  # then settle looks for her standing still

    def _on_rope(self, seconds: float = 1.5) -> tuple[float, float]:
        """Her dot's map position (raw: not snapped to a platform) once it has stopped moving, or after `seconds`."""
        end, last = time.monotonic() + seconds, None
        while True:
            s = self.locate()
            dot = s.where.dot
            if last is not None and math.dist(dot, last) < 0.6 or time.monotonic() >= end:
                return self.senses.localizer.to_map(dot)
            last = dot
            time.sleep(0.08)

    @staticmethod
    def _hanging(x: float, y: float, m: Move, p: Platform) -> bool:
        """Whether her dot at (x, y) is on move m's rope: by it, between its ends, and not at the floor she left."""
        return abs(x - m.rope_x) <= ON_ROPE_X and m.rope_top - HANG_TOLERANCE <= y <= m.rope_bottom + HANG_TOLERANCE \
            and abs(y - p.y_at(m.rope_x)) > HANG_TOLERANCE

    def off_rope(self, s: Scene) -> Scene:
        """She's on a rope (no platform under her dot): climb up onto the platform if she's near its
        top, else jump off it sideways (towards the side a fall lands on a platform). Off a rope that
        takes her onto no platform at its top (959's, under 17's block) she climbs down and lets go
        at its bottom, onto what's under it (up there did nothing: the trip ended lost)."""
        k = self.actions.keys
        x, y = s.where.x, s.where.y
        # The rope she's on: the nearest one reaching her height.
        rope = min(self.model.ropes, key=lambda r: abs(r.x - x) + (0 if r.top - 10 <= y <= r.bottom + 10 else 1000))
        topped = rope.from_platform and hung_from(self.model, rope) is not None
        if y - rope.top < 70 and topped:
            self._send([[0, k.up.name, "down"], [900, k.up.name, "up"]], "off_rope")
        elif not topped and self.model.platform_below(rope.x, rope.bottom) is not None:
            self._send([[0, k.down.name, "down"], [climb_ms(max(0.0, rope.bottom - y)), k.down.name, "up"]], "off_rope")
        else:
            # Right if a jump off to the right (up, at walking speed) comes down on a platform.
            # (Off a rope she's in no foothold group: no wall stops her.)
            side = RIGHT if fly(self.model, x + 15, y, WALK_SPEED, -300, group=ROPE_GROUP)[0] is not None else LEFT
            arrow = self.actions.arrow(side)
            self._send([[0, arrow, "down"], [30, k.jump.name, "down"], [80, k.jump.name, "up"], [400, arrow, "up"]],
                       "off_rope")
        return self.settle(2.5)


def _rope(model: MapModel, m: Move) -> Rope:
    """Move m's rope (a climb's, or the one a rope jump jumps off)."""
    return next(r for r in model.ropes if (r.x, r.top, r.bottom) == (m.rope_x, m.rope_top, m.rope_bottom))


def platform_text(p: Platform | None) -> str:
    """Platform `p` for the log (None: she's in the air or on a rope)."""
    return "air/rope" if p is None else f"platform {p.id} ({p.left:.0f}..{p.right:.0f}, y {p.top:.0f})"
