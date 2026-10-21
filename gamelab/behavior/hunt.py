"""The hunt loop: hunt mode of gamelab/docs/hunt-modes.md. One decision after every look, instead of a
tour's stops, loot trips and fights on the way: she shoots what's in reach from a good place, picks up
loot, moves to a better spot, waits a moment for respawns, or heads for the next hunting ground, and a
monster coming into reach or a pile of drops changes the next step at once.

After each look, the first rule that applies decides (the design's numbering; at a ground they're tried
in the order listed, 6 before 3):

   1  close quarters   a target on her platform the claw would swipe (stance.close_to, by its size: a pig
                       within 111, a snail within 96): stance's answer (jump back, jump-throw, step back)
   2  high ground      on a platform she can't safely stand on, with a spot that reaches her ground: back up;
                       also back to the spot from any other platform (after a pile there), unless the
                       ground is in reach from it or she's leaving
   6  urgent pile      drops whose remaining life is under the time to get there plus URGENT_MARGIN, within
                       URGENT_TRAVEL; above shooting, or drops lay until they vanished where respawns never stop
   3  shoot            a target in reach from her spot (or the ground she hunts on): throw, step out from
                       under, close in along her platform
   4  in passing       travelling: a target in reach, thrown at from where she is, PASSING_S at most per
                       platform per objective (the navigator shoots what stands in the way of a move)
   5  pile here        drops on her platform within PILE_NEAR, nothing by them or between her and them (also
                       piles she passes while travelling)
   7  better spot      wanted monsters seen out of reach, and a spot that reaches clearly more of them
   8  lull loot        nothing wanted on this ground and a kill of hers due back: piles she can pick up first
   9  wait             monsters about or due back: stand still, for the patience at most
  10  loot elsewhere   leaving: the piles left when she decided, most per second of travel first
  11  advance          leaving: one trip to the next ground's spot

She leaves a ground by the rate rule (`_leaving`): her EXP rate there falls clearly below what the best
other ground would give, travel included, or nothing is left there and nothing is due back. Objectives
are suspended, not ended, when a higher rule fires: they keep their route weights and their clock
(OBJECTIVE_S, suspensions included).

The numbers marked "guess" are starting values for the simulator's A/B runs (gamelab.sim.harness); none
is measured in game (respawns, how long drops last).
"""

import random
import time
from collections import Counter
from dataclasses import dataclass, field

from .. import telemetry
from ..attack import reach, spots
from ..attack.spots import Spot
from ..attack.stance import close_to, open_above, stance
from ..control.navigator import TRANSIT, TripState
from ..localization.monsters import Monster, Tracker, tracks_record
from ..localization.scene import Scene
from ..motion.physics import WALK_SPEED
from . import fight
from .fight import THROWS, FightState, Report, kills_text
from .ledger import Pile
from .looting import DROPS_LAST, DropSpot
from .tour import PRIOR

IN_HER_WAY = fight.IN_HER_WAY  # any monster this close on her platform is a target, whatever its kind
PASSING_S = 15.0       # rule 4: s of shooting per platform per objective while passing through
PILE_NEAR = 150        # rule 5: drops on her platform this close are picked up from where she is
PILE_GUARD = 110       # ... unless a monster is this close to them (looting.TOO_CLOSE, today's guard)
PILE_TRIES = 2         # a pile is gone after at most this many times (a suspension counts)
URGENT_MARGIN = 30.0   # rule 6: s of slack before a pile's expected end (time for a guard to be shot first) ...
URGENT_TRAVEL = 12.0   # ... for piles at most this far (s of travel): farther ones wait until she leaves a
                       # ground (rule 10); one 84 s round trip for a pile from the basin took a quarter of a run
DROP_LIFE = DROPS_LAST  # guess: s a drop is still worth going for (the code's own assumption; sim: 180)
OTHERS_AFTER = 30.0    # guess: s before drops no kill of hers explains are worth a trip (not hers before)
SPOT_MORE, SPOT_TIMES = 2, 1.5  # rule 7: another spot must reach 2 more monsters, or 1.5 times as many ...
SPOT_TRAVEL = 8.0      # ... within this many s of travel ...
SPOT_REST = 10.0       # ... not sooner than this after taking a spot ...
SPOT_CHANGES = 4       # ... and at most this many times per visit to a ground
RESPAWN = 8.0          # guess: s from a kill until the monster is back (the simulator's; unmeasured)
PATIENCE = {"ground": 1.0, "spot": 2.0, "none": 0.4}  # rule 9: today's patience values
PRIOR_S = 20.0         # the rate rule: a ground's expected rate counts as this many s of hunting there
MIN_STAY, MIN_THROWS = 15.0, 6  # ... judged only after this long there, or this many throws
LEAVE_MARGIN = 0.8     # ... leave when her rate is under this share of the best other ground's
VISIT_S = 60.0         # guess: a visit's length, until measured visits replace it
SKIP_S = 120.0         # a ground left with nothing on it is skipped this long
EMPTY_CONFIRM_S = 3.0  # ... once looks covering it (GROUND_SEEN of it, at least) have seen it empty this long:
GROUND_SEEN = 0.75     # a hit monster hides for 1-3 s, and one look missing it skipped a busy ground
WIDE_EVERY = 2.0       # s between whole-screen looks while nothing is going on
OBJECTIVE_S = 60.0     # backstop: s on one objective, suspensions included
GROUND_S = 600.0       # backstop: s at one ground
STUCK_S = 120.0        # backstop: s without a kill, a pick-up or a new ground before she stands by
PICK_S = 0.6           # s standing on a pile (4 presses)


@dataclass
class Objective:
    """Where she's heading: a ground's spot, or a pile."""
    kind: str                  # "spot" or "pile"
    platform: int
    x: float | None
    ground: int | None = None  # the ground a spot is for
    pile: Pile | None = None
    spot: Spot | None = None
    weights: dict = field(default_factory=dict)  # its route weights, kept through its suspensions
    # Its clock (OBJECTIVE_S) runs while it's suspended and while rules 1-4 shoot: that bounds a
    # shoot-respawns-on-the-way loop. (Counting only its own trips, in the harness's seed-7 world she tried
    # for platform 4's spot for 270 s and never got there.)
    started: float = field(default_factory=lambda: time.monotonic())

    def age(self) -> float:
        return time.monotonic() - self.started
    trip: TripState | None = None
    passing: dict[int, float] = field(default_factory=dict)  # platform -> s shot there while passing (rule 4)

    def text(self) -> str:
        what = f"spot for {self.ground}" if self.kind == "spot" else "pile"
        return f"{what} on {self.platform}" + ("" if self.x is None else f" at x {self.x:.0f}")


@dataclass
class Visit:
    """A stay at a hunting ground."""
    ground: int
    spot: Spot
    arrived: float | None = None   # when she got to a spot that reaches it (None: on her way)
    exp: float = 0.0               # EXP her kills there gave
    throws: int = 0
    changes: int = 0               # spots taken after the first (rule 7)
    spot_since: float = 0.0
    kills: list[float] = field(default_factory=list)  # when her kills there were (respawns due back)
    empty_since: float | None = None  # since when looks covering the ground have seen nothing wanted on it


@dataclass
class HuntReport:
    """What a hunt did."""
    seconds: float = 0.0
    grounds: list[int] = field(default_factory=list)  # visited, in order
    kills: dict[str, int] = field(default_factory=dict)
    exp_gains: int = 0
    exp: float = 0.0
    throws: int = 0
    piles: int = 0             # piles gone after (picked up, or tried)
    on_the_way: int = 0        # EXP gains while travelling to an objective
    rests: int = 0
    rules: Counter = field(default_factory=Counter)  # passes each rule decided
    stop: str = ""

    def __str__(self):
        return (f"{self.seconds:.0f} s; grounds {self.grounds}; kills: {kills_text(self.kills)} "
                f"(EXP gains {self.exp_gains}, EXP {self.exp:.0f}); throws {self.throws}; piles {self.piles}; "
                f"rests {self.rests}; ended: {self.stop}")


class Hunt:
    """Hunt mode over a Tour's knowledge of the map (its value map, kinds worth hunting, safe platforms,
    spots and rest) and its hunter and navigator."""

    def __init__(self, tour, log=print, rng: random.Random | None = None):
        self.tour, self.h, self.nav, self.model = tour, tour.h, tour.nav, tour.nav.model
        self.log = log
        self.rng = rng or tour.rng
        self.kinds = tour.worth
        self.report = HuntReport()
        self.visit: Visit | None = None
        self.obj: Objective | None = None
        self.stack: list[Objective] = []    # suspended objectives, the latest last
        self.fs = FightState()
        self._tops, self._halves = fight.body_sizes(self.model)  # each kind's size (game data), for stance
        self._fs_key = None
        self._owe_move = False               # cut short: move on (one chain) before shooting
        self._skip: dict[int, float] = {}    # ground -> until when it's skipped
        self._kills_seen = 0
        self._gone = 0                       # piles she went for that looks have seen gone since (progress)
        self._progress = 0.0
        self._wide_at = -1e9
        self._idle_since: float | None = None
        self._rule = ""
        self._potions_out = False
        self._potion_misses = 0              # potions in a row that brought no HP back
        # Leaving a ground: the piles to collect first (None: not leaving), the ones left when she decided
        # to go. Only those: going after the drops of monsters shot on the way to them too, every kill on
        # a trip made another trip, and she never left. The decision itself is made again at every pass
        # (kept, a weak reason such as "nothing came into reach" lost respawns that came after it).
        self._exit_piles: list[Pile] | None = None
        self._exit_reason: str | None = None  # why, kept while she's off the ground and its spot (see `_at_ground`)
        self._no_high_until = 0.0            # after a failed trip back up to the spot, not again for a while
        self._tracker = Tracker(self.model)  # the monsters for the mini-map (the navigator sends none with an escort)
        self._s: Scene | None = None         # the last look, for the 'hunt' record
        self._seen: Scene | None = None      # the latest look of all (a trip starts from it if it's still fresh)
        self._published = None

    # --- the loop ------------------------------------------------------------------------------------

    def run(self, minutes: float) -> HuntReport:
        """Hunt for `minutes`, then go to a safe platform and stand by (as a tour ends)."""
        start = time.monotonic()
        end = start + minutes * 60
        self.h.report = Report()  # one report for the whole hunt
        if self.nav.escort is not None:
            self.nav.escort.kinds = self.kinds
        self._kills_seen = len(self.h.ledger.kills)
        self._gone = self.h.drops.gone_tried
        self._progress = start
        telemetry.emit("tour", order=[], safe=self.tour.safe, grounds=self.tour.hunting_grounds(self._visit_s))
        self._next_ground(self.nav.mover.settle())
        while time.monotonic() < end:
            if self._pass() == "stop":
                break
        self._leave_ground()
        self.report.seconds = time.monotonic() - start
        self.report.stop = self.report.stop or "time up"
        self.tour.rest_and_stand_by(mode=TRANSIT)
        return self.report

    def _pass(self) -> str | None:
        """One look and one step."""
        s = self._look()
        if s.where is None or s.platform is None:
            if s.where is not None:  # in the air, or on a rope
                self.nav._onto_platform(s)
            else:
                time.sleep(0.2)
            return None
        self._book(s)
        self._s = s
        self._publish()
        if self._stuck():
            return "stop"
        if self._safety(s):
            return None
        o = self.obj
        if o is not None and o.age() >= OBJECTIVE_S:
            telemetry.emit("event", kind="backstop", text=f"OBJECTIVE_S: {OBJECTIVE_S:.0f} s on {o.text()}")
            self.log(f"  {o.text()}: {OBJECTIVE_S:.0f} s on it, suspensions included: dropped")
            return self._drop_objective(o, "backstop")
        if o is not None and o.kind == "pile" and o.pile is not None and self._worthless(o.pile):
            # Seen gone, or the inventory filled up on the way: not worth the rest of the trip.
            self.obj = self.stack.pop() if self.stack else None
            return None
        if o is not None and o.kind == "spot" and o.platform == o.ground == s.platform.id:
            # A ground no standable spot reaches is hunted on itself: anywhere on it she's there (stance
            # closes in). Waiting to reach the spot's x, she was interrupted by close monsters for good.
            self._arrived(o)
        now = time.monotonic()
        targets, visible, hidden = self._targets(s, now)
        if self._close_quarters(s, targets):
            return self._shoot(s, targets, visible, hidden, now, "close quarters", approach=False)
        if self._owe_move and self.obj is not None:
            return self._travel("move on first")
        if self.obj is not None:  # travelling: shoot in passing, an urgent pile, else on its way
            return self._while_travelling(s, targets, visible, hidden, now)
        return self._at_ground(s, targets, visible, hidden, now)

    def _while_travelling(self, s: Scene, targets, visible, hidden, now: float) -> str | None:
        if self._can_pass_shoot(s, targets):
            return self._passing_shot(s, targets, visible, hidden, now)
        if (pile := self._pile_here(s)) is not None and pile is not self.obj.pile:  # one she passes
            return self._pick_here(s, pile)
        if self.obj.kind != "pile" and self._exit_piles is None and (pile := self._urgent_pile(s)) is not None:
            return self._go_for(pile, "urgent pile", s)
        return self._travel(self._rule if self._rule in ("advance", "loot elsewhere", "urgent pile",
                                                         "high ground", "back to spot", "better spot", "lull loot")
                            else "travel")

    def _at_ground(self, s: Scene, targets, visible, hidden, now: float) -> str | None:
        v, here = self.visit, s.platform.id
        leaving = self._exit_piles is not None  # the piles left when she decided to go (`_leave`)
        if not leaving and not self._standable(here) and v.spot.platform != v.ground \
                and time.monotonic() >= self._no_high_until:
            return self._to_spot(v.spot, "high ground")
        # About to expire, a pile outranks shooting: where respawns never stop, something is always in
        # reach, and below shooting the drops lay there until they vanished (a quarter of them). Not while
        # she's leaving: then only the piles left when she decided; her kills on the trips to them made
        # new urgent piles, and she never left.
        if not leaving and (pile := self._urgent_pile(s)) is not None:
            return self._go_for(pile, "urgent pile", s)
        # Off the ground and its spot (after a pile elsewhere): back, unless she can stand here with the
        # ground in reach, or she's leaving it. Without this she hunted the pile's platform under the old
        # visit. Shooting (rule 3) is only from where she can stand, or on the ground or its spot.
        safe_here = here in (v.ground, v.spot.platform) or self._standable(here)
        if not leaving and here not in (v.ground, v.spot.platform) and time.monotonic() >= self._no_high_until \
                and not (safe_here and any(m.platform == v.ground and self._hits(s, m) for m in targets)):
            return self._to_spot(v.spot, "back to spot")
        if safe_here and any(self._hits(s, m) or (m.platform == here) for m in targets):
            act = self._stance(s, targets, visible, hidden, approach=True, log=False)
            if act[0] != "wait":
                return self._shoot(s, targets, visible, hidden, now, "shoot", approach=True)
        if (pile := self._pile_here(s)) is not None:
            return self._pick_here(s, pile)
        wide = self._wide(s)
        if not leaving and (spot := self._better_spot(wide)) is not None:
            v.changes += 1
            return self._to_spot(spot, "better spot")
        if self._nothing_here(wide) and (pile := self._lull_pile(wide)) is not None:
            return self._go_for(pile, "lull loot", wide)
        # Away from the ground and its spot (an exit pile), the decision stands: from there she can't
        # judge the ground (out of view, it looked empty, and was skipped for nothing).
        away = here not in (v.ground, v.spot.platform)
        reason = self._exit_reason if leaving and away else self._leaving(wide)
        if reason is None:
            self._exit_piles = self._exit_reason = None  # staying after all
            return self._wait(wide, targets)
        if self._exit_piles is None:  # decided: the piles left now, and only those (see __init__)
            self._exit_piles = [p for p in self.h.drops.known if self._fresh(p)]
        self._exit_reason = reason
        return self._leave(wide, reason)

    def _leave(self, s: Scene, reason: str) -> None:
        """Rules 10 and 11: the piles left when she decided to go, then the next ground."""
        if (pile := self._best_pile(s, near_only=False, among=self._exit_piles)) is not None:
            return self._go_for(pile, "loot elsewhere", s)
        self.log(f"  leaving platform {self.visit.ground}: {reason}")
        self._next_ground(s)
        return self._travel("advance")

    # --- looking and bookkeeping ------------------------------------------------------------------

    def _look(self) -> Scene:
        """A look at the band around her, with her platform's drops."""
        s = self._seen = self.h.look(mobs=True, wide=False, loot=True)
        if telemetry.recording() and s.where is not None:
            telemetry.emit("monsters", **tracks_record(self._tracker.update(s.t, s.monsters), self.model))
        return s

    def _wide(self, s: Scene) -> Scene:
        """A whole-screen look (monsters and drops on every platform in view), at most every WIDE_EVERY s."""
        if time.monotonic() - self._wide_at < WIDE_EVERY:
            return s
        wide = self._seen = self.h.look(mobs=True, wide=True, loot=True)
        if wide.where is None or wide.platform is None:  # dot hidden, or in the air: it searched no monsters
            return s
        self._wide_at = time.monotonic()
        return wide

    def _book(self, s: Scene) -> None:
        """New kills in the ledger: the visit's EXP and kill times, the report, and progress."""
        kills = self.h.ledger.kills[self._kills_seen:]
        self._kills_seen = len(self.h.ledger.kills)
        avg = self._avg_exp()
        for k in kills:
            exp = self.tour.exp.get(k.kind, avg) if k.placed else avg
            self.report.exp_gains += 1
            self.report.exp += exp
            if k.placed:
                self.report.kills[k.kind] = self.report.kills.get(k.kind, 0) + 1
            v = self.visit
            mine = v is not None and v.arrived is not None and k.t >= v.arrived and (  # not on her way there
                (k.placed and (k.platform == v.ground  # on the ground, or a far kill from it or its spot
                               or (self.obj is None and s.platform.id in (v.ground, v.spot.platform))))
                or (not k.placed and k.ground in (v.ground, v.spot.platform)))  # unplaced: where she was
            if mine:
                v.exp += exp
                if k.placed and k.platform == v.ground:
                    v.kills.append(time.monotonic())
            else:
                self.report.on_the_way += 1
            self._progress = time.monotonic()
        # A pick-up is progress once a look sees the pile she went for gone (standing on one proves
        # nothing: someone else's, or a full inventory). Counted so, the stuck guard can fire.
        gone = self.h.drops.gone_tried
        if gone > self._gone:
            self._gone, self._progress = gone, time.monotonic()
        key = (s.platform.id, id(self.obj))
        if key != self._fs_key:  # a new place or objective: what she was throwing at there is forgotten
            self.fs, self._fs_key = FightState(exp_counted=self.h.report.exp_kills), key

    def _avg_exp(self) -> float:
        worth = [self.tour.exp[k] for k in (self.kinds or ())]
        return sum(worth) / len(worth) if worth else 10.0

    def _targets(self, s: Scene, now: float):
        """What's there to shoot: wanted monsters on the ground she hunts, anywhere in reach (far), on her
        platform, and any monster in her way there (fight._targets, with its hidden and fading ones)."""
        ground = {self.visit.ground} if self.visit is not None else None
        targets, visible, hidden = fight._targets(self.h, self.fs, s, self.kinds, ground, True, now)
        self.h.looter.pending.clear()  # drops are the drop store's here (behavior.ledger)
        return targets, visible, hidden

    @staticmethod
    def _hits(s: Scene, m: Monster) -> bool:
        """Whether a star from where she stands in look `s` hits `m` (reach.can_hit: in reach, and no wall of
        her foothold group in the way)."""
        return reach.can_hit(s.platform, s.where.x, s.where.y, m)

    def _standable(self, pid: int) -> bool:
        v = self.visit
        return spots.spot_ok(self.model, pid, self.kinds) or (v is not None and pid == v.ground == v.spot.platform)

    # --- safety -------------------------------------------------------------------------------------

    def _safety(self, s: Scene) -> bool:
        """Potions, and a rest below --low-hp (at once) or --rest-below (at a calm moment, or at once once
        potions have run out). True if it did something."""
        potions = self.h.report.potions
        stop = self.h.health_check(s)
        if self.h.report.potions > potions:
            self._check_potion(s.hp)
        if stop or (s.hp < self.tour.rest_below and (self._potions_out or not s.monsters)):
            self._rest(stop or f"HP {s.hp:.0%}")
            return True
        return False

    def _check_potion(self, before: float) -> None:
        """Two potions in a row that bring no HP back mean none are left: stop counting on them. (One isn't
        enough: a hit in the same moment takes back what it gave.)"""
        time.sleep(0.5)
        if self.h.look(mobs=False).hp > before + 0.01:
            self._potion_misses = 0
            self._potions_out = False  # a false alarm after all
            return
        self._potion_misses += 1
        if self._potion_misses >= 2 and not self._potions_out:
            # She still presses the potion key when HP is low (it does nothing without potions): switched
            # off for good, a false alarm would leave her without potions on the trips to rest.
            self._potions_out = True
            self.log("  two potions brought no HP back: none left; resting as soon as HP is low")

    def _rest(self, why: str) -> None:
        self.log(f"  rest: {why}")
        self.report.rests += 1
        self.stack.clear()
        self.obj = None
        self._leave_ground()  # the visit so far goes into the value map: a rest isn't hunting time
        self.tour.rest()
        self._progress = time.monotonic()
        self._exit_piles = self._exit_reason = None  # the new visit starts undecided
        self._potion_misses = 0
        if self.visit is not None:  # back to the same ground: a new visit, from its spot
            v = self.visit
            self.visit = Visit(v.ground, v.spot, changes=v.changes)
            self._to_spot_obj(v.spot)

    def _stuck(self) -> bool:
        if time.monotonic() - self._progress < STUCK_S:
            return False
        self.log(f"  no kill, pick-up or new ground for {STUCK_S:.0f} s: standing by")
        telemetry.emit("event", kind="backstop", text=f"STUCK_S: {STUCK_S:.0f} s without progress")
        self.report.stop = "stuck"
        return True

    # --- shooting -----------------------------------------------------------------------------------

    def _close_quarters(self, s: Scene, monsters) -> bool:
        """Rule 1: one of `monsters` on her platform is as close as stance's own threshold (close_to: the
        claw would swipe it, by its size)."""
        return any(m.platform == s.platform.id and close_to(m, s.where.x, self._halves) for m in monsters)

    def _stance(self, s: Scene, targets, visible, hidden, approach: bool, log: bool = True) -> tuple:
        if log:
            return fight._take_stance(self.h, self.fs, s, targets, visible, hidden, approach)
        here = s.platform
        return stance(s.where.x, s.where.y, here, targets, approach, self.h.actions.facing,
                      can_jump=lambda a, b: open_above(self.model, here, a, b), tops=self._tops,
                      halves=self._halves)

    def _shoot(self, s: Scene, targets, visible, hidden, now: float, rule: str, approach: bool) -> None:
        """One step of stance's answer: a throw (fight's bookkeeping kept), or a walk."""
        self._note(rule)
        act = self._stance(s, targets, visible, hidden, approach)
        if fight.knocked_jump(self.h, s, act):
            return None
        if act[0] in THROWS:
            self.fs.note_throw(now)
            {"throw": self.h.actions.throw, "jump_throw": self.h.actions.jump_throw,
             "jump_back": self.h.actions.jump_back}[act[0]](act[1])
            self.h.report.throws += 1
            self.report.throws += 1
            if self.visit is not None:
                self.visit.throws += 1
            self._idle_since = None
        elif act[0] in ("back", "approach"):
            self.h.actions.walk(act[1], min(act[2], 300.0))
            self._idle_since = None
        return None

    def _can_pass_shoot(self, s: Scene, targets) -> bool:
        """Rule 4: a wanted target in reach while travelling, with this platform's budget not spent."""
        if self.obj is None or self.obj.passing.get(s.platform.id, 0.0) >= PASSING_S:
            return False
        return any(self._hits(s, m) for m in targets)

    def _passing_shot(self, s: Scene, targets, visible, hidden, now: float) -> None:
        t0 = time.monotonic()
        self._note("in passing")
        act = self._stance(s, targets, visible, hidden, approach=False)
        if fight.knocked_jump(self.h, s, act):
            return None
        if act[0] in THROWS:
            self.fs.note_throw(now)
            {"throw": self.h.actions.throw, "jump_throw": self.h.actions.jump_throw,
             "jump_back": self.h.actions.jump_back}[act[0]](act[1])
            self.h.report.throws += 1
            self.report.throws += 1
        else:  # nothing to throw from here without walking after it: on her way
            self.obj.passing[s.platform.id] = PASSING_S
        pid = s.platform.id
        self.obj.passing[pid] = self.obj.passing.get(pid, 0.0) + (time.monotonic() - t0)
        if self.obj.passing[pid] >= PASSING_S:
            telemetry.emit("event", kind="backstop", text=f"PASSING_S: {PASSING_S:.0f} s shooting on platform {pid}")
        return None

    # --- loot ---------------------------------------------------------------------------------------

    def _worthless(self, p: Pile) -> bool:
        """A pile she's going for that isn't worth the rest of the trip: seen gone, or items with her
        inventory full. Not its age: how long drops last is a guess (DROP_LIFE; the simulator's last
        twice as long), and dropping trips at that age lost 6% of her drops there. The objective's
        clock bounds the trip."""
        return p.status == "gone" or (not p.mesos and self.h.looter.inventory_full)

    def _fresh(self, p: Pile) -> bool:
        now = time.monotonic()
        return (p.status != "gone" and p.platform is not None and now - p.since < DROP_LIFE
                and p.tries < PILE_TRIES and (p.hers or now - p.since >= OTHERS_AFTER)
                and (p.mesos or not self.h.looter.inventory_full))

    def _pile_here(self, s: Scene) -> Pile | None:
        """Rule 5: drops seen on her platform within PILE_NEAR, nothing by them, nothing in between."""
        here, x = s.platform.id, s.where.x
        mine = [m for m in s.monsters if m.platform == here]
        if mine and not self._standable(here):
            return None
        for d in sorted((d for d in s.loot if d.platform == here and abs(d.x - x) <= PILE_NEAR),
                        key=lambda d: abs(d.x - x)):
            if any(abs(m.x - d.x) < PILE_GUARD for m in mine):
                continue
            pile = next((p for p in self.h.drops.known if p.platform == here and abs(p.x - d.x) <= 60), None)
            if pile is not None and self._fresh(pile):
                return pile
        return None

    def _pick_here(self, s: Scene, pile: Pile) -> None:
        self._note("pile here")
        self._tried(pile)
        self.h.looter.walk_piles(s, [DropSpot(pile.x, pile.y, pile.platform, pile.since, pile.mesos, True)], margin=0)
        self._idle_since = None
        return None

    def _tried(self, pile: Pile) -> None:
        self.h.drops.tried(pile)
        self.report.piles += 1

    def _travel_to_pile(self, s: Scene, p: Pile) -> float:
        here = s.platform
        if here is not None and here.id == p.platform:
            return abs(p.x - s.where.x) / WALK_SPEED
        t = self.nav.planner.travel(here.id, s.where.x, p.platform) if here is not None else None
        return 1e6 if t is None else t + 1.0

    def _urgent_pile(self, s: Scene) -> Pile | None:
        """Rule 6: a pile whose remaining life is under the time to get there, pick it up, and the margin."""
        now, best = time.monotonic(), None
        for p in self.h.drops.known:
            if not self._fresh(p):
                continue
            left = DROP_LIFE - (now - p.since)
            travel = self._travel_to_pile(s, p)
            need = travel + PICK_S + URGENT_MARGIN
            if travel <= URGENT_TRAVEL and need >= left and need - URGENT_MARGIN < left \
                    and (best is None or left < best[0]):
                best = (left, p)
        return None if best is None else best[1]

    def _lull_pile(self, s: Scene) -> Pile | None:
        """Rule 8: nothing wanted here; a pile she can get to and pick up before the next respawn of her kills
        here is due. With none due there's no lull: the ground is empty and she leaves (rules 10 and 11 take
        the piles). Going after piles on an empty ground, she never left it."""
        due = min((t + RESPAWN for t in self.visit.kills if t + RESPAWN > time.monotonic()), default=None)
        if due is None:
            return None
        return self._best_pile(s, near_only=True, budget=due - time.monotonic())

    def _best_pile(self, s: Scene, near_only: bool, budget: float = 1e6, among: list[Pile] | None = None) -> Pile | None:
        """The pile with the most drops per second of travel (1 a pile: no prices yet), within `budget` s
        (of `among`, if given)."""
        best = None
        for p in self.h.drops.known if among is None else among:
            if not self._fresh(p):
                continue
            t = self._travel_to_pile(s, p) + PICK_S
            if t <= budget and (not near_only or t < 20) and (best is None or t < best[0]):
                best = (t, p)
        return None if best is None else best[1]

    def _go_for(self, pile: Pile, rule: str, s: Scene) -> None:
        if self.obj is not None:
            self.stack.append(self.obj)  # suspended
        self.obj = Objective("pile", pile.platform, pile.x, pile=pile)
        return self._travel(rule)

    # --- spots and grounds --------------------------------------------------------------------------

    def _to_spot(self, spot: Spot, rule: str) -> None:
        self._to_spot_obj(spot)
        return self._travel(rule)

    def _to_spot_obj(self, spot: Spot) -> None:
        if self.obj is not None:
            self.stack.append(self.obj)
        self.obj = Objective("spot", spot.platform, spot.x, ground=self.visit.ground, spot=spot)

    def _better_spot(self, s: Scene) -> Spot | None:
        """Rule 7: wanted monsters on the ground seen out of reach, and a spot reaching clearly more of them."""
        v = self.visit
        if v.changes >= SPOT_CHANGES or time.monotonic() - v.spot_since < SPOT_REST:
            return None
        left = [m for m in s.monsters if m.platform == v.ground and (self.kinds is None or m.kind in self.kinds)]
        if not left:
            return None
        now_n = sum(self._hits(s, m) for m in left)
        nxt = spots.spot_for(self.model, v.ground, left, v.spot, s.where, self.nav.planner.travel, self.kinds)
        if nxt is None:
            return None
        p = self.model.platform(nxt.platform)
        n = sum(reach.can_hit(p, nxt.x, p.y_at(nxt.x), m) for m in left)
        travel = (abs(nxt.x - s.where.x) / WALK_SPEED if nxt.platform == s.platform.id
                  else self.nav.planner.travel(s.platform.id, s.where.x, nxt.platform))
        if travel is None or travel > SPOT_TRAVEL or not (n >= now_n + SPOT_MORE or n >= SPOT_TIMES * max(now_n, 1)):
            return None
        return nxt

    def _nothing_here(self, s: Scene) -> bool:
        return not any(m.platform == self.visit.ground and (self.kinds is None or m.kind in self.kinds)
                       for m in s.monsters)

    def _due_back(self) -> float | None:
        """When the next of her kills on this ground is due back (s from now), if one is."""
        now = time.monotonic()
        due = [t + RESPAWN - now for t in self.visit.kills if t + RESPAWN + 2 > now]
        return min(due) if due else None

    def _leaving(self, s: Scene) -> str | None:
        """The rate rule: why she leaves this ground now, or None to stay."""
        v, now = self.visit, time.monotonic()
        if v.arrived is None:
            return None
        here_s = now - v.arrived
        if here_s >= GROUND_S:
            telemetry.emit("event", kind="backstop", text=f"GROUND_S: {GROUND_S:.0f} s at platform {v.ground}")
            return f"{GROUND_S:.0f} s here"
        due = self._due_back()
        covered = self._covers_ground(s)
        if covered and self._nothing_here(s) and due is None:
            if v.empty_since is None:
                v.empty_since = now
            if now - v.empty_since < EMPTY_CONFIRM_S:
                return None  # to be confirmed (not left on patience meanwhile: then it wouldn't be skipped)
            self._skip[v.ground] = now + SKIP_S
            return "nothing left here, and nothing due back"
        if covered or due is not None:
            v.empty_since = None
        judged = covered or s.platform.id in (v.ground, v.spot.platform)
        if judged and self._idle_since is not None and now - self._idle_since >= self._patience(s) and due is None:
            return "nothing came into reach"
        if here_s < MIN_STAY and v.throws < MIN_THROWS:
            return None
        rate, best = self._rate_here(), self._best_rate(s)
        if best is not None and rate < LEAVE_MARGIN * best[0]:
            return f"{rate:.2f} EXP/s here, {best[0]:.2f} to be had at platform {best[1]}"
        return None

    def _covers_ground(self, s: Scene) -> bool:
        """Whether look `s` searched GROUND_SEEN of the ground at least (a sliver at the screen's edge says
        nothing about the monsters on the rest of it)."""
        p = self.model.platform(self.visit.ground)
        seen = sum(max(0.0, min(hi, p.right) - max(lo, p.left)) for pid, lo, hi in s.looked if pid == p.id)
        return seen >= GROUND_SEEN * (p.right - p.left)

    def _patience(self, s: Scene) -> float:
        if not s.monsters:
            return PATIENCE["none"]
        return PATIENCE["ground"] if s.platform.id == self.visit.ground else PATIENCE["spot"]

    def _rate_here(self) -> float:
        v = self.visit
        prior = self.tour.ground_values().get(v.ground, 0.0) / self._visit_s(v.ground)
        return (v.exp + prior * PRIOR_S) / (time.monotonic() - v.arrived + PRIOR_S)

    def _visit_s(self, pid: int) -> float:
        """A visit's expected length: VISIT_S counted as PRIOR visits (as the value map's EXP per visit is),
        blended with the timed visits. Unblended, one empty ground left in 0.6 s looked like the best one:
        its EXP per second jumped several-fold."""
        st = self.tour.stats.get(str(pid), {})
        return (PRIOR * VISIT_S + st.get("seconds", 0.0)) / (PRIOR + st.get("timed", 0))

    def _best_rate(self, s: Scene) -> tuple[float, int] | None:
        """The best other ground's EXP per second, the trip there included."""
        best, now = None, time.monotonic()
        values = self.tour.ground_values()
        for g in self.tour.hunting_grounds(self._visit_s):
            if g == self.visit.ground or self._skip.get(g, 0) > now:
                continue
            at = spots.spot_at(self.model, g, self.kinds)
            t = self.nav.planner.travel(s.platform.id, s.where.x, at[0]) if s.platform.id != at[0] else 0.0
            if t is None:
                continue
            rate = values.get(g, 0.0) / (t + self._visit_s(g))
            if best is None or rate > best[0]:
                best = (rate, g)
        return best

    def _next_ground(self, s: Scene) -> None:
        """The ground to hunt next: the best EXP per second, travel included (varied a little), and its spot."""
        self._leave_ground()
        self._exit_piles = self._exit_reason = None
        now, values, best = time.monotonic(), self.tour.ground_values(), None
        here = s.where.platform if s.where is not None else None
        for g in self.tour.hunting_grounds(self._visit_s):
            if self._skip.get(g, 0) > now or (self.visit is not None and g == self.visit.ground):
                continue
            at = spots.spot_at(self.model, g, self.kinds)
            if here is None:  # on a rope or in the air: no route times, so by value alone
                t = 0.0
            else:
                t = 0.0 if here.id == at[0] else self.nav.planner.travel(here.id, s.where.x, at[0])
            if t is None:
                continue
            score = values.get(g, 0.0) / (t + self._visit_s(g)) * self.rng.uniform(0.75, 1.25)
            if best is None or score > best[0]:
                best = (score, g)
        if best is None:  # every ground skipped: the soonest back
            grounds = self.tour.hunting_grounds(self._visit_s)
            if not self._skip and not grounds:
                raise RuntimeError("no hunting ground on this map")
            failed = self.visit.ground if self.visit is not None else None
            back = [k for k in self._skip if k != failed] or list(self._skip)
            g = min(back, key=self._skip.get) if back else grounds[0]
            self._skip.pop(g, None)
        else:
            g = best[1]
        spot = spots.spot(self.model, g, s.where, self.nav.planner.travel, self.kinds)
        self.visit = Visit(g, spot)
        self.report.grounds.append(g)
        self.stack.clear()
        self.obj = Objective("spot", spot.platform, spot.x, ground=g, spot=spot)
        telemetry.emit("tour", order=list(self.report.grounds), stop=g,
                       spot={"platform": spot.platform, "x": round(spot.x, 1), "coverage": spot.coverage})
        self.log(f"  next: platform {g}, from platform {spot.platform} at x {spot.x:.0f} ({spot.coverage:.0%} in reach)")

    def _leave_ground(self) -> None:
        """What the visit gave goes into the value map (with how long it lasted)."""
        v = self.visit
        if v is None or v.arrived is None:
            return
        self.tour.record_visit(v.ground, v.exp, time.monotonic() - v.arrived)

    # --- travelling ---------------------------------------------------------------------------------

    def _travel(self, rule: str) -> None:
        """Carry out the objective: one trip, until she gets there, a rule interrupts it (the navigator
        asks `_interrupt` before every chain), or it fails."""
        self._note(rule)
        o = self.obj
        o.trip = self.nav.begin(o.platform, o.x, seconds=max(5.0, OBJECTIVE_S - o.age()),
                                until=self._interrupt, mode=TRANSIT, scene=self._seen)
        o.trip.weights = o.weights
        chains = o.trip.trip.chains
        while not o.trip.over:
            self.nav.step(o.trip)
            self._owe_move = o.trip.quick
            if o.trip.trip.chains > chains:
                chains = o.trip.trip.chains
        result = o.trip.trip.result
        if result == "arrived" or (result == "in position" and o.kind == "spot" and self._on_spot(o)):
            return self._arrived(o)
        if result == "in position":
            return None  # a rule fired: the next pass decides (OBJECTIVE_S is checked there)
        self.log(f"  {o.text()}: {result}")
        return self._drop_objective(o, result)

    def _on_spot(self, o: Objective) -> bool:
        """Stopped 'in position' where she can stand with targets of the ground in reach: as good as there.
        Judged from the look the trip stopped on (`_interrupt` said so to it a moment ago)."""
        s = o.trip.last if o.trip is not None and o.trip.last is not None else self.h.look()
        return (s.platform is not None and self._standable(s.platform.id)
                and any(m.platform == o.ground and self._hits(s, m) for m in s.monsters))

    def _interrupt(self, s: Scene) -> bool:
        """Asked before every chain of a trip: whether a rule calls for her here (close quarters, a target in
        reach she may shoot, an urgent pile, low HP)."""
        if s.hp < self.h.low_hp:
            return True
        here = s.platform.id
        if self._close_quarters(s, s.monsters):
            return True
        o = self.obj
        if o is not None and o.kind == "pile" and o.pile is not None and self._worthless(o.pile):
            return True  # not worth it any more (`_pass` drops it)
        wanted = [m for m in s.monsters if self.kinds is None or m.kind in self.kinds]
        in_reach = [m for m in wanted if self._hits(s, m)]
        if in_reach and o is not None and o.passing.get(here, 0.0) < PASSING_S:
            return True
        if o is not None and o.kind == "spot" and self._standable(here) \
                and any(m.platform == o.ground for m in in_reach):
            return True
        return o is not None and o.kind != "pile" and self._exit_piles is None and self._urgent_pile(s) is not None

    def _arrived(self, o: Objective) -> None:
        if o.kind == "spot":
            v = self.visit
            if v is not None and o.ground == v.ground:
                v.spot, v.spot_since = o.spot or v.spot, time.monotonic()
                if v.arrived is None:
                    v.arrived = time.monotonic()
                    self._progress = time.monotonic()
                    self.log(f"  at platform {v.ground}'s spot (platform {o.platform})")
        elif o.kind == "pile" and o.pile is not None:
            p = o.pile
            # Guarded (a monster by it): not a try. The guard is a target now (it's on her platform), so the
            # next looks shoot it, and the pile is gone for again while it's still worth it.
            if self.h.looter.collect(DropSpot(p.x, p.y, p.platform, p.since, p.mesos, p.status == "seen")):
                self._tried(p)
            self.h.looter.pending.clear()
        self.obj = self.stack.pop() if self.stack else None
        self._idle_since = None
        return None

    def _drop_objective(self, o: Objective, why: str) -> None:
        if o.kind == "pile" and o.pile is not None:
            self._tried(o.pile)
        if o.kind == "spot" and self.visit is not None and o.ground == self.visit.ground and self.visit.arrived is None:
            self._skip[o.ground] = time.monotonic() + SKIP_S  # couldn't get there: another ground
            self._next_ground(self.nav.mover.settle())
            return None
        if o.kind == "spot":  # the trip back up failed or ran out its clock: not again for a while, so the
            # rate rule gets a look in (without it, high ground was re-created at once, again and again)
            self._no_high_until = time.monotonic() + SPOT_REST
        self.obj = self.stack.pop() if self.stack else None
        return None

    # --- waiting ------------------------------------------------------------------------------------

    def _wait(self, s: Scene, targets) -> None:
        self._note("wait")
        if self._idle_since is None:
            self._idle_since = time.monotonic()
        time.sleep(0.1)
        return None

    def _note(self, rule: str) -> None:
        self.report.rules[rule] += 1
        if rule != self._rule:
            self._rule = rule
            self._publish()

    def _publish(self) -> None:
        """The 'hunt' record for the mini-map (gamelab.ui): why she does what she does. Sent when the rule,
        the objective or the piles change."""
        s = self._s
        if not telemetry.recording() or s is None or s.platform is None:
            return
        urgent = self._urgent_pile(s)
        piles = [{"x": round(p.x, 1), "y": round(p.y, 1), "platform": p.platform, "status": p.status,
                  "hers": p.hers, "urgent": p is urgent} for p in self.h.drops.known if self._fresh(p)]
        v, o = self.visit, self.obj
        key = (self._rule, None if o is None else o.text(), None if v is None else v.ground,
               tuple((d["x"], d["platform"], d["status"], d["urgent"]) for d in piles))
        if key == self._published:
            return
        self._published = key
        best = None if v is None else self._best_rate(s)
        telemetry.emit("hunt", mode="hunt", rule=self._rule, objective=key[1], ground=key[2],
                       rate_here=None if v is None or v.arrived is None else round(self._rate_here(), 3),
                       rate_to_beat=None if best is None else [round(best[0], 3), best[1]], piles=piles)
