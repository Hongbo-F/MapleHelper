"""On the Lie Detector, away to a safe place within 10 s. The user, 2026-10-24, after one failed over Chrome Remote
Desktop's lag while a hunt went on around her: "just stop any in-progress hunt when this happens", then "On a lie
detector you should move to a safe place, within 10s". And at the stops that end a hunt for whoever is at the PC to put
right (behavior.safety.HuntStop: the safety stop, her HP staying low, her pet's potions not keeping her up; out of
darts, behavior.darts), the same way, from the stop; then whoever is at the PC is paged, and out of darts the rolling
recording's last minutes are saved (`on_alarm`).

Who does it: a running hunt (`hunt kill`, `hunt clear`, `nav`, `tour`, `tour --loop`, `probe run`: `on_alarm` around
it), which has the map, the danger model and the navigator. From the sighting the player refuses its plans, so its next
key plan, or its next look (Hunter.look, a navigator's pass: client.check_alarm), raises client.LieDetectorAlarm; the
retreat runs there, and the alarm goes on: the command stops (exit code 3) as before. With no hunt running nothing moves
her: the player only raises the alarm and records, as before (it knows neither the map nor where she is).

Where to (`choose`): of the safe places she gets to within RETREAT_S (alarm.RETREAT_S, 10 s) of the sighting, by the
route planner's time (motion.route.Planner) stretched for her looks and stops on the way (`trip_s`): the nearest safe
platform (MapModel.safe: no monster can touch her there) or place the danger model rates safe enough (attack.danger:
SAFE_DANGER of her HP at most, with every spawn point's monster at its point and with the look's monsters; only on a map
where a kind counts at all, Danger.any: on the first map none does, and anywhere would be "safe"), if it's near
(NEAR_S); else the quickest to get to of those and the ropes as safe (positioning's rope model, attack.ropes: from any
platform she gets to, at its safest height; she regains no HP there, so only when no safe platform is near). None in
time: the safest place she gets to in that time, by the same danger, or where she is if that's it. Hanging on a rope
as safe already (the hunt waits on ropes: its rule 9), she stays on it (`safe_rope`).

How: her moves alone, as the navigator plans them keeping clear of monsters (navigator.AVOID: never a fight; the
quickest way; a monster in the way of a move waited out where she can stand, or gone round, as on any trip), its plans of
kind alarm.RETREAT, which the player takes for alarm.RETREAT_WINDOW_S (12 s) after the sighting and cuts there: her arrows,
jump, up and down (Actions.moves_only: no attack, pick-up or pet food), every key let go by a plan's end, and no mouse
(whoever is at the PC may be answering the Lie Detector with it: the mover's nudge of the mouse sends nothing). A go
knocked back, or a rope missed, is followed by another from where she is (TRIES) while there's time. No potion or pet
food on the way: her pet's care stops with the hunt (the pet babysitter, `pet --every`, goes on). The log and the
telemetry ('retreat' events) say where, why and how long it took.

In the simulator (Tree Dungeon, the hunt loop, the Lie Detector at 40, 75, 110 or 150 s): shooting 23 from 24, as live,
47 of 48 runs (seeds 1-12) are on a safe place 10 s after the sighting (onto 25, 2.3 s at the median); hunting the
whole map, mostly its floor (33) among its Green Mushrooms, with her looks from the feed as `tour --loop` takes them by
default, 24 of 29 (seeds 1-8, 2026-10-25), the rest short of one in the time (the safest place within reach) or off the
rope at its foot. Without the feed (`--no-feed`) the hunt loop's sweep (hunt.SWEEP) has her worse placed when it comes,
and it's 15 of 32 (28 of 29 with the sweep off; 39 of 47 over seeds 1-12 at master's head before the sweep came in, the
rest knocked back on the way up to the rope at 146 or to 28): tests/test_retreat.py;
D:/MapleHelper-scratch/ldpet/sweep.py, D:/MapleHelper-scratch/integ3/rebase/ldsweep.py. Since her jumps' attacks are
weighed (2026-10-25): with the feed 28 of 32, without it 26 of 32, shooting 23 from 24 32 of 32
(D:/MapleHelper-scratch/ov-jump/sweep_count.py). Since every look is the perception service's (2026-10-25: no
`--no-feed`), 27 of 31, and 32 of 32 shooting 23 from 24 (D:/MapleHelper-scratch/perceive/build/ld_halves.py).
"""

import math
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from .. import telemetry
from ..attack import ropes
from ..attack.ropes import Refuge
from ..control import alarm, client
from ..control.alarm import RETREAT_S, RETREAT_WINDOW_S
from ..control.mover import HANG_TOLERANCE, ON_ROPE_X
from ..control.navigator import AVOID, Navigator
from ..localization.mapdata import Rope
from ..localization.scene import Scene
from ..motion.physics import WALK_SPEED
from ..perception.framecheck import LookLost
from .safety import HuntStop, SafetyStop

SAFE_DANGER = 0.01  # a place whose danger (attack.danger: the share of her HP a touch takes, by how likely and soon) is
                    # at most this is safe enough: 1% of her HP, a Horny Mushroom's walk of 4.7 s off (16% a touch)
SLACK = 1.5         # her trips take about this much longer than the planner's time: the looks and stops between
                    # chains, the waits for monsters (the simulator's retreats from Tree Dungeon's floor: 1.0-2.4 times)
TRIP_S = 0.6        # ... and this long more: the look on arrival, a chain sent a moment after its look
NEAR_S = 6.0        # a safe platform or place she gets to this quickly is near: past it, a rope as safe and quicker
                    # comes first (from the floor of Tree Dungeon, 28 is 4.3 s by route, which the crowd there made 5-12 s)
PLACE_STEP = 20.0   # map units between the places along a platform weighed
PLACE_IN = 15.0     # ... kept this far in from its ends (her half width: she stands on it)
LOCATE_TRIES = 8    # looks at most for her dot before she sets off (a look every 0.25 s: 2 s)
TRIES = 3           # goes at a safe place at most: from wherever a go left her (knocked back, the rope missed) ...
GO_SLACK_S = 1.5    # ... a go given this long past its trip's time (trip_s), then the next from where she is
ROPE_STILL_LOOKS = 3  # looks in a row with her dot put (on no platform) that have her hanging on a rope, not in the
                      # air: a jump's top keeps it put over one look's gap at most (0.08 s) ...
ROPE_STILL_S = 1.0    # ... looked for this long at most


@dataclass(frozen=True)
class Place:
    """Where a retreat goes: platform `platform`, at map x `x` (None: anywhere on it, a safe platform), or on `rope`
    (got onto from `platform`), or (platform None) the rope at x she hangs on already; why there, the planner's seconds
    to get there, and its danger."""
    platform: int | None
    x: float | None
    why: str
    seconds: float
    danger: float = 0.0
    rope: Refuge | None = None

    def text(self) -> str:
        if self.rope is not None:
            return f"the rope at x {self.rope.x:.0f} from platform {self.platform}, feet at y {self.rope.y:.0f}"
        if self.platform is None:
            return f"the rope at x {self.x:.0f}"
        return f"platform {self.platform}" + ("" if self.x is None else f" at x {self.x:.0f}")


@dataclass
class Retreat:
    """How a retreat went: s after the sighting when it began, where it went and why (`place`; None: it didn't
    choose), the trip's result, s after the sighting at her last look, and where that put her (platform, x; platform
    None: on a rope or in the air)."""
    began: float
    place: Place | None = None
    result: str = ""
    took: float | None = None
    where: tuple[int | None, float] | None = None

    def record(self) -> dict:
        p = self.place
        return {"began": round(self.began, 3), "to": None if p is None else p.text(),
                "why": None if p is None else p.why, "planned_s": None if p is None else round(p.seconds, 3),
                "danger": None if p is None else round(p.danger, 4), "result": self.result,
                "took": None if self.took is None else round(self.took, 3),
                "where": None if self.where is None else [self.where[0], round(self.where[1], 1)]}


class Watch:
    """What `on_alarm` saw: the retreat, if the alarm or a stop came (None: neither did, or the retreat failed), and
    what came of saving the recording after a stop that keeps it (`kept`: what the log said of it; None: none)."""
    kept: str | None = None
    retreat: Retreat | None = None


@dataclass(frozen=True)
class Why:
    """What a retreat is for, as its log lines and telemetry events say it: the Lie Detector's, or the safety stop's."""
    label: str  # the log's
    event: str  # the telemetry's
    after: str  # "N s <after>"


LIE_DETECTOR = Why("LIE DETECTOR", "Lie Detector", "after it was seen")
SAFETY = Why(SafetyStop.label, SafetyStop.title, "after the stop")
# Before the key check that follows a doubt of her regular attack's key (behavior.keycheck: a monster near her, the game's
# KEY BINDINGS window is opened from a safe place)
KEY_CHECK = Why("KEY CHECK", "Key check", "after the doubt")


def why_of(stop: HuntStop) -> Why:
    """A stop's retreat's words (SAFETY for the safety stop's)."""
    return Why(stop.label, stop.title, "after the stop")


@contextmanager
def on_alarm(h, log: Callable[[str], None] | None = None, rng: random.Random | None = None):
    """Run a hunt (the `with` block); on the Lie Detector alarm (client.LieDetectorAlarm), her retreat, then the alarm
    goes on (the command stops as before). Nothing that goes wrong in the retreat hides the alarm: it's logged. On a
    stop (behavior.safety.HuntStop: the safety stop, her HP staying low; out of darts, behavior.darts), the same retreat
    from the stop, said in capitals, and whoever is at the PC paged (control.alarm.page, once her retreat is over: her
    moves come first); a stop that keeps the recording (`keep_s`) has its last minutes saved then
    (control.alarm.keep_recent), and the log says where; then the stop goes on (the command stops)."""
    watch = Watch()
    try:
        yield watch
    except client.LieDetectorAlarm:
        try:
            watch.retreat = retreat(h, log, rng)
        except Exception as e:  # the alarm is what the command stops on
            (log or h.log)(f"retreat: failed ({type(e).__name__}: {e})")
        raise
    except HuntStop as stop:
        say = log or h.log
        say(f"{stop.label}: {stop}. The hunt ends: {stop.ends}")
        telemetry.emit("event", kind=stop.event, text=str(stop))
        try:
            watch.retreat = away(h, time.monotonic(), why_of(stop), log, rng)
        except Exception as e:  # the stop is what the command stops on
            say(f"retreat: failed ({type(e).__name__}: {e})")
        finally:
            alarm.page(*stop.page())
            if stop.keep_s:
                watch.kept = alarm.keep_recent(stop.keep_s, stop.event.replace("_", "-"))
                say(f"{stop.label}: {watch.kept}")
        raise


def retreat(h, log: Callable[[str], None] | None = None, rng: random.Random | None = None) -> Retreat:
    """Her retreat from the Lie Detector, from its sighting (the alarm's `at`): `away`."""
    age = client.alarm_age()
    seen = time.monotonic() - (age or 0.0)  # the sighting on this clock (the alarm's file not written yet: now)
    return away(h, seen, LIE_DETECTOR, log, rng)


def away(h, seen: float, why: Why, log: Callable[[str], None] | None = None,
         rng: random.Random | None = None) -> Retreat:
    """Her retreat, hunter `h`'s (its senses, map and danger; its actions' keys and locks), from `seen` (time.monotonic:
    the Lie Detector's sighting, or the safety stop): where to (`choose`), then there; and again from wherever she ends
    up if that wasn't it (a touch knocked her back, the rope missed), TRIES goes at most while RETREAT_S has time for
    one, all within the player's window (the Lie Detector's RETREAT_WINDOW_S: the safety stop's trips are cut there
    too). `why`: what it's for, in its log lines and events. Returns how it went (also logged). Meanwhile the hunter's
    safety stop isn't raised at her looks (Hunter.retreating): the Lie Detector's way isn't cut short by it."""
    was = getattr(h, "retreating", False)
    h.retreating = True
    try:
        return _away(h, seen, why, log, rng)
    finally:
        h.retreating = was


def _away(h, seen: float, why: Why, log: Callable[[str], None] | None = None,
          rng: random.Random | None = None) -> Retreat:
    """away's retreat."""
    log = log or h.log
    out = Retreat(time.monotonic() - seen)
    if out.began >= RETREAT_S:
        out.result = "too late"
        log(f"{why.label}: seen {out.began:.1f} s ago, past the {RETREAT_S:g} s a retreat has: she stays where she is")
        return _end(out, seen, log, None, why=why)
    actions = h.actions.moves_only()
    nav = Navigator(h.senses, actions, h.model, log=log, variety=0, rng=rng)  # no escort: she keeps clear of monsters
    s = None
    try:
        for go in range(TRIES):
            s = nav.mover.locate(tries=LOCATE_TRIES, mobs=True, wide=True)
            # On a rope or in the air: a rope as safe as a safe place is one, where she stays; off any other at once
            # (_onto_platform's look until she's still is over), and down from the air.
            if s.where.platform is None and (hung := hanging(nav, s)) is not None:
                still, rope = hung
                if (p := safe_rope(h, rope, still.where.y, s.monsters)) is not None:
                    out.place, out.result = (p, "stayed") if go == 0 else (out.place, "on a safe rope")
                    if go == 0:
                        log(f"{why.label}: retreat: she stays on {p.text()} ({p.why}; danger {p.danger:.3f}), "
                            f"{time.monotonic() - seen:.1f} s {why.after}")
                    return _end(out, seen, log, nav, still, why)
                log(f"  on the rope at x {rope.x:.0f} (danger {rope_risk(h, rope, still.where.y, s.monsters):.3f} "
                    "there): getting off")
                s = nav.mover.off_rope(still)
            if s.where.platform is None:
                s = nav._onto_platform(s)
            if s.where.platform is None:
                out.result = "lost: not on any platform"
                return _end(out, seen, log, nav, why=why)
            left = RETREAT_S - (time.monotonic() - seen)
            p = choose(h, nav, s, left)
            if p.seconds <= 0:  # there (or nowhere better to be had in the time left)
                out.place, out.result = (p, "stayed") if go == 0 else (out.place, out.result)
                if go == 0:
                    log(f"{why.label}: retreat: she stays ({p.why}; danger {p.danger:.3f}), "
                        f"{time.monotonic() - seen:.1f} s {why.after}")
                return _end(out, seen, log, nav, s, why)
            if go and trip_s(p.seconds) > left:
                break
            out.place = p
            log(f"{why.label}: retreat to {p.text()} ({p.why}; {p.seconds:.1f} s by route, danger {p.danger:.3f}), "
                f"{time.monotonic() - seen:.1f} s {why.after}" + ("; her moves only, no mouse" if go == 0 else
                                                                  f" (go {go + 1})"))
            telemetry.emit("event", kind="retreat", text=f"{why.event}: retreat to {p.text()} ({p.why})")
            deadline = seen + RETREAT_WINDOW_S
            seconds = min(deadline - time.monotonic(), trip_s(p.seconds) + GO_SLACK_S)  # longer: from where she is
            trip = nav.go(p.platform, x=p.rope.x if p.rope else p.x, seconds=max(0.1, seconds), mode=AVOID, scene=s,
                          vary=False, stand_until=deadline)
            out.result = trip.result
            if p.rope is not None and trip.result == "arrived":
                s = nav.mover.locate(tries=LOCATE_TRIES)
                if s.where.platform is not None and s.where.platform.id == p.platform:
                    s = nav.mover.hang(p.rope.rope, p.rope.y, p.rope.from_top, s)
                if s.where is not None and s.where.platform is None:
                    out.result = "on the rope"
                    return _end(out, seen, log, nav, s, why)
                out.result = f"not on the rope (on {platform_text(s)})"
            if RETREAT_S - (time.monotonic() - seen) <= TRIP_S:
                break
    except client.LieDetectorAlarm as e:  # a plan past the player's window, or one it refused
        out.result = f"stopped: {e}"
    except (client.PlayerError, RuntimeError, ValueError, LookLost) as e:  # no dot, a move the mover can't make from
        out.result = f"stopped: {type(e).__name__}: {e}"                  # where she is, a look failing, no game
    return _end(out, seen, log, nav, why=why)


def platform_text(s: Scene) -> str:
    return "nothing" if s.where is None or s.where.platform is None else f"platform {s.where.platform.id}"


def hanging(nav: Navigator, s: Scene) -> tuple[Scene, Rope] | None:
    """Look `s` has her on no platform: a look of her hanging on a rope, and the rope; None if she's in the air
    (landed meanwhile, or still moving after ROPE_STILL_S). Hanging: her dot still over ROPE_STILL_LOOKS looks on no
    platform, by the nearest rope reaching her height (Mover.off_rope's), between its ends."""
    model = nav.model
    if not model.ropes:
        return None
    last, still, end = s, 1, time.monotonic() + ROPE_STILL_S
    while still < ROPE_STILL_LOOKS:
        if time.monotonic() > end:
            return None
        time.sleep(0.08)
        now = nav.mover.locate()
        if now.where is None or now.where.platform is not None:  # (landed)
            return None
        still = still + 1 if math.dist(now.where.dot, last.where.dot) < 0.6 else 1
        last = now
    x, y = last.where.x, last.where.y
    rope = min(model.ropes, key=lambda r: abs(r.x - x) + (0 if r.top - 10 <= y <= r.bottom + 10 else 1000))
    if abs(x - rope.x) > ON_ROPE_X or not rope.top - HANG_TOLERANCE <= y <= rope.bottom + HANG_TOLERANCE:
        return None
    return last, rope


def rope_risk(h, rope: Rope, y: float, seen) -> float:
    """The danger to her hanging on `rope` with her feet at map y, as choose has a rope's: the worst within HANG_SLACK
    of y, with every spawn point's monster at its point or with the monsters `seen` (a look's)."""
    d = h.danger
    return max(d.on_rope(rope, y + dy, look) for dy in (-ropes.HANG_SLACK, 0.0, ropes.HANG_SLACK)
               for look in (None, seen))


def safe_rope(h, rope: Rope, y: float, seen) -> Place | None:
    """Hanging on `rope`, her feet at map y (`hanging`): staying there (a Place of 0 s, platform None) if it's as safe as
    a safe place (`rope_risk` SAFE_DANGER at most), else None; None too on a map where no kind counts (Danger.any), as
    choose has it: only a safe platform is a safe place there. Got off first, a rope she waited on safely (the hunt's
    rule 9) had sent her over 9 s across the map (review, 2026-10-24: the simulator, from the rope at -8, its danger
    0)."""
    if not h.danger.any:
        return None
    risk = rope_risk(h, rope, y, seen)
    return Place(None, rope.x, "here: a safe rope", 0.0, risk) if risk <= SAFE_DANGER else None


def choose(h, nav: Navigator, s: Scene, left: float) -> Place:
    """Where her retreat goes from look `s`, with `left` s of RETREAT_S to get there (see the module's doc). Of the safe
    places she gets to in time (`trip_s` within `left`): the nearest safe platform or place safe enough, if it's near
    (NEAR_S); else the quickest to get to of those and the ropes as safe (from any platform she gets to: attack.ropes,
    each at its safest height); else the safest place she gets to in time, or where she is."""
    model, d = nav.model, h.danger
    here, x = s.where.platform, s.where.x
    counts = d.any
    seen = s.monsters

    def risk(pid: int, gx: float) -> float:
        """The danger there with every spawn point's monster at its point, or with the look's, the worse."""
        return max(d.at(pid, gx, None), d.at(pid, gx, seen)) if counts else 0.0

    def in_time(t: float | None) -> bool:
        return t is not None and trip_s(t) <= left

    if model.safe(here.id):
        return Place(here.id, x, "here: a safe platform", 0.0)
    now = risk(here.id, x)
    if counts and now <= SAFE_DANGER:
        return Place(here.id, x, "here: safe enough", 0.0, now)
    times = nav.planner.times(here.id, x)
    safe, hangs, safest = [], [], [Place(here.id, x, "here: the safest within reach", 0.0, now)]
    for p in model.places:
        t = 0.0 if p.id == here.id else times.get(p.id)
        if t is None or not in_time(t):
            continue
        if p.id != here.id and model.safe(p.id):
            safe.append(Place(p.id, None, "a safe platform", t))
            continue
        if not counts:
            continue
        lo, hi = p.left + PLACE_IN, p.right - PLACE_IN
        for gx in [lo + i * PLACE_STEP for i in range(int(max(0.0, hi - lo) // PLACE_STEP) + 1)] or [(lo + hi) / 2]:
            tt = nav.planner.arrive(here.id, x, p.id, gx)
            if not in_time(tt):
                continue
            r = risk(p.id, gx)
            (safe if r <= SAFE_DANGER else safest).append(
                Place(p.id, gx, "safe enough" if r <= SAFE_DANGER else "the safest within reach", tt, r))
        for r in ropes.refuges(model, d, p.id, None):  # at its safest height with every spawn filled; the look too
            tt = nav.planner.arrive(here.id, x, p.id, r.x)
            if tt is None or not in_time(tt + r.on_s):
                continue
            worse = max([r.danger] + [d.on_rope(r.rope, r.y + dy, seen) for dy in (-ropes.HANG_SLACK, 0.0,
                                                                                   ropes.HANG_SLACK)])
            if worse <= SAFE_DANGER:
                hangs.append(Place(p.id, r.x, "a rope as safe: no safe platform near", tt + r.on_s, worse, r))
    first = (lambda c: (round(c.seconds, 3), c.x is not None))  # the quickest; a safe platform before a place on one
    if near := [c for c in safe if trip_s(c.seconds) <= NEAR_S]:
        return min(near, key=first)
    if safe or hangs:
        return min(safe + hangs, key=lambda c: (round(c.seconds, 3), c.rope is not None, c.x is not None))
    return min(safest, key=lambda c: (round(c.danger, 4), c.seconds))


def trip_s(route_s: float) -> float:
    """How long a trip the planner times at `route_s` takes her, by SLACK and TRIP_S."""
    return SLACK * route_s + TRIP_S


def _end(out: Retreat, seen: float, log, nav: Navigator | None, s: Scene | None = None,
         why: Why = LIE_DETECTOR) -> Retreat:
    """The retreat over: her last look (`s`, or one taken now), from the sighting (or the stop) to it (`took`), said."""
    try:
        if nav is not None and s is None:
            s = nav.mover.locate(tries=LOCATE_TRIES)
    except Exception:  # (no dot: where she is isn't known)
        s = None
    out.took = (s.t if s is not None else time.monotonic()) - seen
    if s is not None and s.where is not None:
        out.where = (None if s.where.platform is None else s.where.platform.id, s.where.x)
    where = ("where she is isn't known" if out.where is None else
             f"on platform {out.where[0]} at x {out.where[1]:.0f}" if out.where[0] is not None else
             f"off any platform at x {out.where[1]:.0f}")
    log(f"{why.label}: retreat over ({out.result}): {where}, {out.took:.1f} s {why.after}"
        + ("" if out.took <= RETREAT_S else f" (past the {RETREAT_S:g} s)"))
    telemetry.emit("event", kind="retreat", text=f"{why.event}: retreat over ({out.result}), {where}, "
                                                 f"{out.took:.1f} s {why.after}")
    return out
