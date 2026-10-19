"""A hunting tour of a map: sweep the platforms monsters spawn on (nearest next, varied a little, the
order then tidied to cut travel), shoot each one clear, collect the drops, rest on the chair at a safe
platform when HP runs low, and finish standing by on a safe platform.

A platform is safe when no monster can touch her there (MapModel.safe): none spawn on it (monsters
stay on the platform they spawn on), and none walk a platform just underneath. Her stars reach other
platforms (attack.reach), so each hunting platform is shot from a spot she can stand on when one has
enough of it in reach (attack.spots: e.g. platform 17 from platform 14, 60 units above it); only then
does she go down to pick up what dropped. Otherwise she hunts on the platform itself. On the way
there, she starts shooting as soon as a target is in reach.

A platform's worth is the EXP a visit is expected to give: at first a share of its spawns' EXP (game
data), then blended with what visits really gave (lab/grounds.json).
"""

import json
import random
import time
from dataclasses import dataclass, field

from .. import paths, telemetry
from ..attack import reach, spots
from ..attack.spots import Spot
from ..control.navigator import Navigator
from ..localization.monsters import Monster
from ..localization.pose import Where
from ..localization.scene import Scene
from ..motion.physics import WALK_SPEED
from ..motion.route import Move
from ..perception.kinds import kind_of
from .fight import Report, add_kills, kills_text
from .hunter import Hunter
from .looting import DROPS_LAST, DropSpot

WORTH = 0.25         # monsters giving under this share of the map's best EXP aren't hunted: snails, 2 vs 15
SEEN_SHARE = 0.3     # share of a platform's spawn EXP a visit gives (platform 36: 13-52 of 100), the prior
PRIOR = 2            # visits the spawn data counts for, against what visits really gave
MIN_VALUE = 0.15     # grounds expected to give under this share of the best one's EXP per visit are skipped
MIN_REST_WIDTH = 120  # narrower safe platforms aren't rested on
REST_SECONDS = 120   # longest she sits on the chair waiting for her HP to come back
GOES_PER_STOP = 5    # trips to a spot and fights there, at most, per stop
DROP_FALL = 120      # a kill placed on no platform: its drops are fetched from a platform at most this far below
LOOT_SECONDS = 180   # no loot trip starts this long or longer into a collection (the longest one in the
                     # simulator's scenarios started its last trip 151 s in)
UNREACHABLE = 1e6    # s of travel to a hunting ground no route reaches (grounds unreachable from where
                     # she starts are left out of the sweep)
BLIND_SAMPLES = 9    # random orders of the stops whose median travel is logged, to compare the sweep with


@dataclass
class TourReport:
    """What a tour did: the platforms visited, the kills there and on the way, rests and loot trips."""
    visited: list[int] = field(default_factory=list)
    kills: dict[str, int] = field(default_factory=dict)
    throws: int = 0
    exp_gains: int = 0    # times her EXP went up while hunting at the stops (kills of hers)
    on_the_way: int = 0   # monsters shot dead while travelling
    rests: int = 0
    loot_trips: int = 0
    stop: str = ""

    def __str__(self):
        return (f"visited platforms {self.visited}; kills: {kills_text(self.kills)} (EXP gains: {self.exp_gains}), "
                f"plus {self.on_the_way} on the way; throws: {self.throws}; "
                f"loot trips: {self.loot_trips}; rests: {self.rests}; ended: {self.stop}")

    def add_stop(self, pid: int, r: Report) -> None:
        """Count a stop at platform `pid` in: its kills, throws and EXP gains."""
        self.visited.append(pid)
        add_kills(self.kills, r.kills)
        self.throws += r.throws
        self.exp_gains += r.exp_kills


@dataclass
class _Stop:
    """What a stop (Tour.hunt) keeps from one go at it to the next."""
    pid: int       # the platform hunted
    kinds: object  # the kinds hunted (None: all)
    kills: int     # kills wanted (after those, only what's still in reach is finished)
    end: float     # when its time is up (time.monotonic)
    spot: Spot     # where she shoots it from, this go
    total: Report = field(default_factory=Report)  # what the stop did so far
    arrived: bool = False  # she got to a spot (the visit is recorded)


class Tour:
    """A hunting tour with a hunter and her (fighting) navigator; `run` sweeps the map."""

    def __init__(self, hunter: Hunter, nav: Navigator, log=print, rng: random.Random | None = None,
                 rest_below: float = 0.5, rest_to: float = 0.9):
        self.h, self.nav, self.log = hunter, nav, log
        self.rng = rng or random.Random()
        self.rest_below, self.rest_to = rest_below, rest_to
        self.report = TourReport()
        model = nav.model
        self.safe = [p.id for p in model.platforms if model.safe(p.id) and p.right - p.left >= MIN_REST_WIDTH]
        self.exp = {kind_of(name): e for name, e in model.mob_exp.items()}  # kind -> EXP per kill
        best = max(self.exp.values(), default=0)
        self.worth = {k for k, e in self.exp.items() if e >= WORTH * best} or None  # kinds worth hunting
        self.stats = _load_grounds(model.id)

    # --- where to hunt ---------------------------------------------------------------------------

    def ground_values(self) -> dict[int, float]:
        """Expected EXP from a visit, per platform that monsters worth hunting spawn on (all of them
        kinds her eyes can recognize: she can't see the others coming). The spawn data's guess
        (SEEN_SHARE of its spawns' EXP) counts as PRIOR visits, blended with what visits really gave.
        By the data, platforms 34 and 35 spawn slimes, but 11 visits found only snails there."""
        known = self.h.senses.eyes.kinds
        out = {}
        for pid, names in self.nav.model.monsters_by_platform.items():
            kinds = [kind_of(n) for n in names]
            if not all(k in known for k in kinds):
                continue
            prior = SEEN_SHARE * sum(self.exp.get(k, 10) for k in kinds if self.worth is None or k in self.worth)
            if prior <= 0:
                continue
            st = self.stats.get(str(pid), {})
            out[pid] = (PRIOR * prior + st.get("exp", 0)) / (PRIOR + st.get("visits", 0))
        return out

    def hunting_grounds(self) -> list[int]:
        """The platforms worth a visit: expected to give at least MIN_VALUE of the best one's EXP."""
        values = self.ground_values()
        best = max(values.values(), default=0)
        return [pid for pid, v in values.items() if v >= MIN_VALUE * best]

    def _record(self, pid: int, r: Report) -> None:
        """Remember what a visit to `pid` gave, so the sweep learns where the monsters really are. EXP
        gains no kill was placed on count as a kill of the average kind worth hunting."""
        exp = sum(self.exp.get(k, 0) * n for k, n in r.kills.items())
        worth = [self.exp[k] for k in (self.worth or ())]
        exp += max(0, r.exp_kills - sum(r.kills.values())) * (sum(worth) / len(worth) if worth else 0)
        st = self.stats.setdefault(str(pid), {"visits": 0, "exp": 0})
        st["visits"] += 1
        st["exp"] = round(st["exp"] + exp, 1)
        _save_grounds(self.nav.model.id, self.stats)

    def spot(self, pid: int, on_it: bool = False) -> Spot:
        """Where to shoot platform `pid` from (attack.spots.spot), from where she is now."""
        here = self.nav.mover.locate().where
        return spots.spot(self.nav.model, pid, here, self.nav.planner.travel, self.worth, on_it)

    def spot_for(self, pid: int, left: list[Monster], current: Spot) -> Spot | None:
        """A spot that reaches more of the monsters `left` on `pid` (attack.spots.spot_for)."""
        here = self.nav.mover.locate().where
        return spots.spot_for(self.nav.model, pid, left, current, here, self.nav.planner.travel, self.worth)

    # --- resting -------------------------------------------------------------------------------

    def nearest_safe(self) -> int:
        """The safe platform quickest to get to by route: the one she's on, if it's safe."""
        s = self.nav.mover.settle(3)  # she may still be falling (knocked off a platform)
        here = s.where.platform
        if here is not None and here.id in self.safe:
            return here.id
        best = None
        for pid in self.safe:
            path = self.nav.planner.plan(here.id, s.where.x, pid) if here else None
            if path is not None:
                cost = _path_seconds(path)
                if best is None or cost < best[0]:
                    best = (cost, pid)
        if best is None:
            raise RuntimeError("no safe platform reachable")
        return best[1]

    def rest(self, max_seconds: float = REST_SECONDS) -> None:
        """Go to the nearest safe platform, sit on the chair until HP is back (or for `max_seconds` at
        most), and stand up."""
        pid = self.nearest_safe()
        self.log(f"resting on platform {pid}")
        escort = self.nav.escort
        if escort is not None:
            escort.hunting = False  # low on HP: straight there
        try:
            # Low HP is why she's going: no reason to stop on the way (potions are still drunk). It used
            # to end the trip at its first look, and she sat down among the monsters.
            trip = self.nav.go(pid, health=self.h.drink_if_low)
        finally:
            if escort is not None:
                escort.hunting = True
        if trip.result != "arrived":
            self.log(f"  couldn't reach a safe platform ({trip.result}); resting here")
        self.report.rests += 1
        telemetry.emit("event", kind="rest", text=f"resting on platform {pid}")
        self.h.actions.sit()
        end = time.monotonic() + max_seconds
        while time.monotonic() < end:
            time.sleep(5)
            hp = self.h.look(mobs=False).hp
            self.log(f"  HP {hp:.0%}")
            if hp >= self.rest_to:
                break
        self.h.actions.stand()

    def rest_and_stand_by(self) -> None:
        """The tour's end: over to the nearest safe platform, resting there first (and staying seated)
        if her HP is under `rest_to`."""
        if self.h.look(mobs=False).hp < self.rest_to:
            self.rest()
            self.h.actions.sit()  # stay seated while standing by
        else:
            pid = self.nearest_safe()
            self.nav.go(pid)
        self.report.stop = "standing by on a safe platform"

    # --- a stop ----------------------------------------------------------------------------------

    def hunt(self, pid: int, kinds=None, kills: int = 5, seconds: float = 90) -> Report:
        """Shoot platform `pid` clear like a player keeping the high (or low) ground: from the spot
        chosen, every monster in reach is killed before anything else (the kill count only decides
        whether to go after more); monsters left out of reach are shot from another spot that
        reaches them, if there is one. Only then are the drops collected; she goes onto the platform
        itself only when no spot reaches what's left. Killing some, then walking down into the rest
        to pick up the drops, gave up the safe spot."""
        end = time.monotonic() + seconds
        st = _Stop(pid, kinds, kills, end, self.spot(pid))
        for _ in range(GOES_PER_STOP):
            if not self._go_to_spot(st):
                break
            r = self._fight(st)
            if r.stop.startswith("HP low") or time.monotonic() >= st.end:
                break
            if r.stop.startswith("left platform"):
                continue  # knocked off: go back to the spot
            left = self._left_on(st)
            enough = sum(st.total.kills.values()) >= kills
            if left and not enough and self._over_to_another_spot(st, left):
                continue
            if self._collect_then_done(st, r, left, enough):
                break
            self._next_spot(st, left)
        self.collect_loot()
        if st.arrived:  # what the visit gave, so later sweeps know where the monsters really are
            self._record(pid, st.total)
        return st.total

    # --- the steps of a stop (hunt) --------------------------------------------------------------

    def _go_to_spot(self, st: _Stop) -> bool:
        """Over to the stop's spot, stopping on the way as soon as a target is in reach where she can
        stand (_in_reach). False if she didn't get there (the stop's total.stop says why)."""
        spot = st.spot
        telemetry.emit("tour", stop=st.pid, spot={"platform": spot.platform, "x": round(spot.x, 1),
                                                  "coverage": spot.coverage})
        how = "from platform" if spot.platform != st.pid else "on the platform"
        self.log(f"  shooting platform {st.pid} {how} {spot.platform} at x {spot.x:.0f} "
                 f"({spot.coverage:.0%} in reach)")
        trip = self.nav.go(spot.platform, x=spot.x, until=lambda s: self._in_reach(st, s),
                           seconds=max(20, st.end - time.monotonic()))
        self.report.on_the_way += trip.kills
        if trip.result not in ("arrived", "in position"):
            st.total.stop = f"didn't get there: {trip.result}"
            return False
        st.arrived = True
        if trip.result == "in position":
            self.log("  targets in reach on the way: shooting from here")
        return True

    def _in_reach(self, st: _Stop, s: Scene) -> bool:
        """Whether to stop on the way to the stop's (current) spot: a target is in reach, and she can stand
        here. On the way to platform 2 her route crosses platform 4, and stopping there put her in its
        crowd for the whole stop."""
        here = s.platform
        if here is None or (here.id != st.spot.platform and not spots.spot_ok(self.nav.model, here.id, self.worth)):
            return False
        return any(m.platform == st.pid and (st.kinds is None or m.kind in st.kinds)
                   and reach.can_shoot(m.x - s.where.x, m.y - s.where.y) for m in s.monsters)

    def _fight(self, st: _Stop) -> Report:
        """Kill all she can reach from here (fight.clear: those in reach are finished after the kill count,
        far ones too); counted into the stop's total."""
        self.h.report = Report()
        # Waiting for respawns looked idle and slowed the sweep: move on soon when nothing's there, sooner
        # on the platform itself than from a spot off it.
        off_it = st.spot.platform != st.pid
        r = self.h.clear(st.kinds, ground={st.pid}, max_kills=max(1, st.kills - sum(st.total.kills.values())),
                         seconds=max(5, st.end - time.monotonic()), patience=2.0 if off_it else 1.0,
                         finish=True, far=True)
        st.total.add(r)
        st.total.stop = r.stop
        return r

    def _left_on(self, st: _Stop) -> list[Monster]:
        """A whole-screen look: the monsters (of the kinds hunted) still on the platform."""
        s = self.h.look(wide=True)
        return [m for m in s.monsters if m.platform == st.pid and (st.kinds is None or m.kind in st.kinds)]

    def _over_to_another_spot(self, st: _Stop, left: list[Monster]) -> bool:
        """A spot that reaches more of the monsters `left`, out of reach from here: if there's one, it's
        the next go's. True if so."""
        nxt = self.spot_for(st.pid, left, st.spot)
        if nxt is None:
            return False
        self.log(f"  {len(left)} left on platform {st.pid}, out of reach here: over to platform "
                 f"{nxt.platform} at x {nxt.x:.0f}")
        st.spot = nxt
        return True

    def _collect_then_done(self, st: _Stop, r: Report, left: list[Monster], enough: bool) -> bool:
        """Collect the drops (collect_loot); then whether the stop is over: nothing `left` on the platform,
        `enough` kills, or nothing came into reach in fight `r` and she hasn't moved since."""
        trips = self.report.loot_trips
        self.collect_loot()
        if not left:
            self.log(f"  nothing left on platform {st.pid}: moving on")
            return True
        if enough:
            return True
        if not r.throws and self.report.loot_trips == trips:
            # Nothing thrown and she hasn't moved: going back would only repeat that. In the basin
            # a whole-screen look saw pigs that the fight's own looks never did, 4 times running.
            self.log(f"  {len(left)} seen on platform {st.pid} but none came into reach: moving on")
            return True
        return False

    def _next_spot(self, st: _Stop, left: list[Monster]) -> None:
        """Monsters are `left` that no spot reaches: from off the platform, onto it; on it, back at them."""
        if st.spot.platform != st.pid:
            self.log(f"  {len(left)} left on platform {st.pid} that no spot reaches: onto the platform")
            st.spot = self.spot(st.pid, on_it=True)
        else:
            self.log(f"  {len(left)} still on platform {st.pid}: back at it")

    # --- loot --------------------------------------------------------------------------------

    def collect_loot(self) -> None:
        """Go to the drops left on other platforms and pick them up: her kills' (far kills' can lie across
        the map) and any seen on the ground. A whole-screen look first, and again after each trip, notes
        the drops in view and forgets remembered ones seen gone (Looter.remember). The nearest by route
        goes first. With her inventory full, only to mesos seen.

        Drops that fights on the way note during the trips are remembered for the next collection (or
        a fight's sweep), not gone after in this one: each kill on a trip there would make another
        trip, and in the sim a whole tour then went on collecting. And no trip starts LOOT_SECONDS or
        more into a collection: where respawns guard the drops, going back for them never ended."""
        looter = self.h.looter
        start = time.monotonic()
        self._look_for_loot()
        for d in looter.pending:  # a kill placed on no platform: its drops fell onto the one below
            if d.platform is None and (p := self.nav.model.platform_under(d.x, d.y, reach=DROP_FALL)) is not None:
                d.platform, d.y = p.id, p.y_at(d.x)
        later: list[DropSpot] = []  # noted by fights on the way during the trips
        while looter.pending:
            if time.monotonic() - start >= LOOT_SECONDS:
                self.log(f"  {LOOT_SECONDS} s collecting drops: the rest is left for later")
                break
            fresh = [d for d in looter.pending if time.monotonic() - d.since < DROPS_LAST and d.platform is not None
                     and (d.mesos or not looter.inventory_full)]
            looter.pending = []
            if not fresh:
                break
            s = self.nav.mover.locate()
            fresh.sort(key=lambda d: self._travel_to(s.where, d))
            d, rest = fresh[0], fresh[1:]
            self.log(f"  collecting drops on platform {d.platform} at x {d.x:.0f}")
            self.report.loot_trips += 1
            trip = self.nav.go(d.platform, x=d.x, seconds=60)
            self.report.on_the_way += trip.kills
            later += looter.pending  # these used to be forgotten here (unless a later look saw them again)
            looter.pending = rest
            if trip.result != "arrived":
                self.log(f"    couldn't get there ({trip.result})")
                continue
            if not looter.collect(d):  # guarded: shoot the guard, then try again
                self.h.clear(ground={d.platform}, max_kills=2, seconds=20)
                looter.loot_pending()
            self._look_for_loot()
        looter.pending += later

    def _look_for_loot(self) -> None:
        """A look at every platform in view for drops (about 0.2 s), noted for collect_loot."""
        self.h.looter.remember(self.h.look(mobs=False, wide=True, loot=True))

    def _travel_to(self, where: Where, d: DropSpot) -> float:
        """Seconds to drops `d` from `where`, by route (straight-line walking if there's no route known)."""
        here = where.platform
        if here is not None and here.id != d.platform:
            t = self.nav.planner.travel(here.id, where.x, d.platform)
            if t is not None:
                return t + abs(d.x - where.x) / WALK_SPEED * 0.2
        return (abs(d.x - where.x) + abs(d.y - where.y)) / WALK_SPEED

    # --- the sweep -------------------------------------------------------------------------------

    def sweep(self, grounds: list[int], stops: int, here: tuple[int, float]) -> tuple[list[int], float]:
        """An order to visit `stops` of the hunting grounds that keeps travelling short, like a player
        sweeping the map: nearest next (by route time from where she'll be, judged with a random
        +-25% so it isn't the same sweep every time), then improved by swapping stretches of it
        (2-opt) on the real route times. Returns the order and its travel time (s, by the model)."""
        at_spot = {pid: self._spot_at(pid) for pid in grounds}
        cache: dict[tuple, float] = {}

        def travel(a: tuple[int, float], pid: int) -> float:
            key = (a[0], round(a[1]), pid)
            if key not in cache:
                t = self.nav.planner.travel(a[0], a[1], at_spot[pid][0])
                cache[key] = UNREACHABLE if t is None else t
            return cache[key]

        order, at, left = [], here, [g for g in grounds if travel(here, g) < UNREACHABLE]
        while left and len(order) < stops:
            nxt = min(left, key=lambda g: travel(at, g) * self.rng.uniform(0.75, 1.25))
            order.append(nxt)
            left.remove(nxt)
            at = at_spot[nxt]

        def total(seq: list[int]) -> float:
            t, a = 0.0, here
            for g in seq:
                t += travel(a, g)
                a = at_spot[g]
            return t

        better = True
        while better:
            better = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    trial = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    if total(trial) < total(order) - 0.5:
                        order, better = trial, True
        return order, total(order)

    def run(self, stops: int | None = None, kinds=None, kills_per_stop: int = 5,
            seconds_per_stop: float = 90) -> TourReport:
        """Sweep the hunting grounds (all of them worth a visit, or the `stops` nearest), hunting the
        monsters worth it (`kinds`, by default all but those giving little EXP, like snails: they're
        only shot when in her way)."""
        grounds = self.hunting_grounds()
        kinds = self.worth if kinds is None else kinds
        if self.nav.escort is not None:
            self.nav.escort.kinds = kinds
        s = self.nav.mover.locate()
        here = (s.where.platform.id, s.where.x) if s.where.platform else (grounds[0], 0.0)
        n = len(grounds) if stops is None else min(stops, len(grounds))
        order, travel = self.sweep(grounds, n, here)
        blind = sorted(self._travel_of(self.rng.sample(grounds, n), here)
                       for _ in range(BLIND_SAMPLES))[BLIND_SAMPLES // 2]
        self.log(f"tour: sweep {order}, about {travel:.0f} s of travel ({n} hunting grounds picked at random: "
                 f"{blind:.0f} s); safe platforms: {self.safe}")
        telemetry.emit("tour", order=order, safe=self.safe, grounds=grounds)
        remaining = list(order)
        while remaining:
            pid = remaining.pop(0)
            if self.h.look(mobs=False).hp < self.rest_below:
                self.rest()
            self.log(f"heading to platform {pid}")
            result = self.hunt(pid, kinds, kills_per_stop, seconds_per_stop)
            self.report.add_stop(pid, result)
            self.log(f"  platform {pid}: {result}")
            if result.stop.startswith("HP low"):
                self.rest()
            # Far kills' drops can take her across the map: the rest of the sweep is planned again from
            # wherever she ended up.
            s = self.nav.mover.locate()
            if remaining and s.where.platform is not None:
                remaining = self.sweep(remaining, len(remaining), (s.where.platform.id, s.where.x))[0]
                telemetry.emit("tour", order=self.report.visited + remaining)
        self.rest_and_stand_by()
        return self.report

    def _spot_at(self, pid: int) -> tuple[int, float]:
        """Where she'll stand to hunt platform `pid` (attack.spots.spot_at)."""
        return spots.spot_at(self.nav.model, pid, self.worth)

    def _travel_of(self, seq: list[int], here: tuple[int, float]) -> float:
        """Seconds of travel visiting `seq` in that order (an unreachable stop counts as none)."""
        t, a = 0.0, here
        for g in seq:
            at = self._spot_at(g)
            t += self.nav.planner.travel(a[0], a[1], at[0]) or 0.0
            a = at
        return t


def _path_seconds(path: list[Move]) -> float:
    """A route's seconds: its moves, and the walks between them (from where one lands to where the next starts)."""
    return sum(m.seconds for m in path) + sum(abs(a.land_x - b.x) / WALK_SPEED for a, b in zip(path, path[1:]))


def _load_grounds(map_id: int) -> dict:
    """What visits to this map's hunting grounds gave (lab/grounds.json): {platform id: {visits, exp}}."""
    try:
        return json.loads(paths.GROUNDS.read_text(encoding="utf-8")).get(str(map_id), {})
    except (OSError, ValueError):
        return {}


def _save_grounds(map_id: int, stats: dict) -> None:
    """Keep this map's hunting-ground stats in lab/grounds.json, next to other maps'."""
    try:
        everything = json.loads(paths.GROUNDS.read_text(encoding="utf-8")) if paths.GROUNDS.exists() else {}
    except (OSError, ValueError):
        everything = {}
    everything[str(map_id)] = stats
    try:
        paths.GROUNDS.parent.mkdir(parents=True, exist_ok=True)
        paths.GROUNDS.write_text(json.dumps(everything, indent=1), encoding="utf-8")
    except OSError:
        pass
