"""Grinding (`tour --loop --grind`): the hunt loop (behavior.hunt) going round the map as an experienced player grinds,
rather than hunting one ground and waiting there for its respawns. The user, playing 4.5 minutes on Tree Dungeon,
Forest Up North II on 2026-10-24 (lab/telemetry/20261024-170558-watch.jsonl) made 484 EXP a minute where her hunt loop
made 283 on one ground: never camping (a median 1 s on a platform, 11 s at the longest, a fifth of the time in the air
or on ropes), a circuit up the tower and back down, killing what came into reach on the way.

What changes from the hunt loop (the rules not named here are its own, unchanged):
- Where she goes (`_next_ground`): the stops of a loop over the map's spawn points (behavior.patrol: the loop chosen by
  its EXP a minute, the next stop along it the first worth going to, a pile she has seen elsewhere pulling her off it),
  not the grounds of the value map. Each stop is a visit of the hunt loop to that platform, at the place there that
  reaches the most of the spawn points around it.
- At a stop (`_at_ground`), she stays while what's there gives as much EXP a second as going on would (the marginal
  value theorem: a patch is left when its rate falls under the habitat's): going on gives the loop's long-run rate
  (Plan.rate) or, if more, the next stop's (Patrol.choose, looked at every RECHOOSE_S). What's there: a monster in a
  star's reach now is shot at once (her kill time, 0.085 s a point of EXP live, 11.8 EXP/s, beats any loop: the best on
  Tree Dungeon is 7.7); the monsters her memory has, seen within HOLD_S, that she can get at from her platform (on it,
  stance closing in; or in reach from somewhere along it, which she walks to: `_reach_step`, the rule 'reach'), by their
  EXP over their kill time and the walk (`_stay_gain`); with none, the respawns expected in reach within patrol.DUE_S
  (Patrol.wait_gain: 1-2 EXP/s at a point watched empty, so she doesn't wait for respawns). Under it, she leaves
  LEAVE_CONFIRM_S after; STOP_MAX_S at the most with nothing in reach. No better spot (rule 7), no lull loot (rule 8),
  no waiting out respawns (rule 9 only for those confirmations), no rate rule. Got to the stop, a knock (or a pile) that
  leaves her on another platform where it's more dangerous than at the stop's place has her go back to it first (from
  2026-10-31, `_back_due`: the hunt loop's 'back to spot', by the danger), and rule 1 yields to her way back unless a
  monster is on her.
- On the legs: the hunt loop's rule 4, shooting in passing (anything a star reaches, any platform), its finishing (4b)
  and the loot sweep's piles on her way to the stop (behavior.sweep: those worth their time, within its budget): kills
  and pick-ups between moves, without stopping to clear a ground. Every RECHOOSE_S of a trip to a stop, a pile of
  monsters seen elsewhere (Patrol.choose with the stop she's heading to as the default) takes her there instead, and on
  a trip to one that no longer pays for its detour back to the loop's next stop (`_interrupt`, `_while_travelling`).
- Her drops (from 2026-10-30, the loot sweep): at a stop, with nothing in reach, those worth a round trip from its place
  at what staying there gives (`_stay_gain`); leaving it, before she goes, those worth their time on her way to the next
  stop (`_leave`: Patrol.choose's, peeked), the drops of her last kills there waited for where that pays (live on
  2026-10-30 half her departures from a stop, 24 of 50, came within 3 s of a kill of hers, before its drops landed: 21 of
  those kills' drops were never seen, 10 picked up, 3 lost). Until then only the hunt loop's piles here (5: on her
  platform within 150) and urgent ones (6).
- What she knows: every look updates the belief of each spawn point (patrol.Belief) from her memory of the monsters.
- What a kill is credited to: the platform it died on (`credit`), not the ground she hunted. The value map
  (lab/grounds.json) isn't written: its EXP a visit credits a ground with its neighbours' kills (live, 48% of 23's on
  2026-10-24 died on 22 and 24), and a stop's visit is a few seconds.

Every loop taken up and every next stop is logged with its numbers ("loop: ...", "next: ..."), and sent as the
telemetry topic 'grind' (what "loop", "next", and "end" with the run's stays, pulls, skips and credit by platform).
"""

import math
import time
from statistics import median

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from ..attack.spots import Spot
from ..attack.stance import CONTACT_SLACK, beside, half_of
from ..localization.memory import SURE
from ..localization.monsters import Monster
from ..localization.scene import Scene
from ..motion.bodies import HER_HALF
from ..motion.physics import WALK_SPEED
from . import fight
from .hunt import PASSING_S, Hunt, Objective, Visit
from . import patrol as patrol_mod
from .patrol import TICK_S, Patrol, choice_text, reaching

STOP_MAX_S = 40.0      # guess: backstop, s at one stop with nothing in reach (there's a round to go)
# s with nothing left in reach before she leaves a stop: one look's miss isn't enough (a hit monster hides under its
# damage numbers for 1-3 s: fight's hidden targets keep it a target meanwhile, and HOLD_S keeps it held)
LEAVE_CONFIRM_S = 0.3  # (a guess; the drops of her last kills there are the sweep's when she leaves: `_leave`)
HOLD_S = 3.5           # a monster her memory has, seen within this long (tracking.HIDE_S), holds her at a stop
REACH_STEP = 200.0     # guess: units she walks along her platform at once at most to bring held monsters into reach
HP_GAP_S = 2.0         # guess: s between two looks at most for the HP gone between them to be charged to a platform
RECHOOSE_S = 1.0       # guess: s between two looks at where she'd go instead (Patrol.choose takes about 1 ms on Tree
                       # Dungeon, its routes kept by the planner)
HP_LOG_S = 10.0        # s on a platform at least for the log to say what HP she lost there (the log's choice)
STATE_EVERY_S = 60.0   # s between two 'grind' records 'state' (what it did so far: a run cut short has them; the
                       # log's choice)
# Back to the stop (`_back_due`, from 2026-10-31): got to a stop, on another platform where the danger (attack.danger,
# with the look's monsters) is BACK_MARGIN over the stop's place's, she goes back to it before she shoots from there: a
# third of a Horny Mushroom's touch on Tree Dungeon (0.156 of her HP). In the simulator's grind there (D:/MapleHelper-
# scratch/replan/pos) a knock from 10 lifted her onto 9 (between 8 and 10, every place on it 0.32-0.57 where 10's stop
# is 0.22), and she stayed 36.7 s a run there shooting what was in reach, 16.5 touches, where the hunt loop has its
# 'back to spot'; the hunt loop's rule 1 yields to her way back unless a monster is on her (`_close_quarters`), and she
# doesn't shoot in passing on the platform she leaves. BACK_TRIES a visit at the most, BACK_REST_S apart (guesses:
# unbounded, a prototype sent her back 124 times in a run, each trip cut short by what came into reach). False: she
# shoots from where she is, as before. With the sweep's guards (behavior.hunt.BESIDE_GUARDS), in the harness's standard
# grind (12 seeds of 5 minutes, paired against the branch before: pos/impl/out/cmp_E.txt) touches 86.8 -> 81.2 a run
# (se 4.0), those at piles 16.1 -> 12.2 (se 2.0), EXP a minute 385 -> 373 (se 16).
BACK = True
BACK_MARGIN = 0.05
BACK_TRIES = 3
BACK_REST_S = 5.0
# What she learns of the HP a stop costs (Patrol.learn_hp): her HP bar between two looks at most HP_GAP_S apart,
# charged to the platform she's on (False, until 2026-10-31), where her pet's potions net out touches and her trips
# aren't looked at (in the simulator's grind on Tree Dungeon it saw 3.27 of the 8.68 bars she lost on seed 1, and none
# of what she lost on 9 and 8 counted against the loop's stops, 10 and 7); or her touches (True: `_learn_touches`),
# charged to the stop she's at or going to, her trips' time and touches too.
HP_BY_TOUCHES = True
TOUCH_GAP_S = 30.0     # s between two passes at most for the touches between them to be charged (a trip's)
# Whether what staying at a stop gives (`_stay_gain`) is net of what touches there cost meanwhile (Patrol.hp_rate at
# HP_PRICE), as the loop's rate is (Patrol.rate), from 2026-10-31; False: its EXP alone, as before. With HP_BY_TOUCHES,
# her way back and the sweep's guards, in the harness's standard grind (as above: cmp_G.txt) touches 86.8 -> 80.8 (se
# 3.0), those that hurt 57.3 -> 53.1 (se 2.4), HP potions 24.5 -> 22.8, EXP a minute 385 -> 365 (se 16); with the
# stops weighing danger and the HP priced as the hunt loop's too (behavior.patrol: STOP_DANGER, HP_PRICE, HP_PRIOR_S,
# cmp_J.txt) 77.6 touches, 50.0 hurt, 6.6 bars of HP (8.0), 20.3 potions, 359 EXP a minute.
STAY_NET = True


class Grind(Hunt):
    """The hunt loop grinding round the map (see the top). `patrol`: its loop and belief (None: one for the map, from
    the tour's kinds worth hunting and their EXP, the navigator's route times, and for the danger hook the hunter's
    map danger, the positioning model's: Hunter.danger, attack.danger; patrol's default for a hunter without one)."""

    def __init__(self, tour, log=print, rng=None, patrol: Patrol | None = None):
        super().__init__(tour, log=log, rng=rng)
        h = self.h  # the positioning model's Danger (its `at` is in the hook's unit), as of each call: her stats as
        # perception reads them (Hunter.danger: behavior.bootstrap)
        self.patrol = patrol or Patrol(self.model, self.kinds, tour.exp,
                                       travel=lambda a, x, b, bx: self.nav.planner.arrive(a, x, b, bx),
                                       danger=None if getattr(h, "danger", None) is None
                                       else (lambda pid, x: h.danger.at(pid, x)))
        self.stop: int | None = None              # the stop she's at or going to (patrol.stops)
        self.credit: dict[int, list] = {}         # platform -> [kills, EXP] of her kills that died there
        self.stays: list[float] = []              # s at each stop she got to
        self.pulls = self.skips = 0
        self._alt: tuple | None = None            # where she'd go instead: (when, heading, stop, choose's info)
        self._reroute = False                     # a pile seen on a trip (or hers gone): to it at the next pass
        self._start = time.monotonic()
        self._hp: tuple[float, float, int] | None = None  # the last look's (time, HP, platform), for learn_hp
        self._stated = time.monotonic()

    def run(self, minutes: float):
        report = super().run(minutes)
        self._log_grind()
        return report

    def _paused(self, seconds: float) -> None:
        """She stood still `seconds` while looks failed their checks (the hunt loop's pause: Senses.regain): the hunt
        loop's clocks move on (a stop's stay and STOP_MAX_S, LEAVE_CONFIRM_S go by the visit's), and so do grinding's
        own: her EXP a second over the run (`_rate_here`), and where she'd go instead, looked at afresh (`_alt`); the
        HP gone over the pause isn't charged to a platform (`_hp`). The spawn points' belief and the stops' `left_at`
        are the game's clocks, which ran on, and aren't moved."""
        super()._paused(seconds)
        self._start += seconds
        self._alt = self._hp = None

    # --- what she knows ------------------------------------------------------------------------------------

    def _look(self) -> Scene:
        s = super()._look()
        if s.where is not None:
            self.patrol.belief.update(s.t, self.h.memory, s.players if s.dot is not None else None)
        self._learn_hp(s)
        return s

    def _learn_hp(self, s: Scene) -> None:
        """Her HP bar between two looks (HP_GAP_S apart at most), what went charged to the platform she was on
        (Patrol.learn_hp): a rise (a potion) counts as none lost; a reading under 2% isn't taken (the bar
        flashes). With HP_BY_TOUCHES, her touches instead (`_learn_touches`)."""
        if HP_BY_TOUCHES:
            return self._learn_touches(s)
        if s.platform is None or s.hp is None or s.hp < 0.02:
            return
        last, self._hp = self._hp, (s.t, s.hp, s.platform.id)
        if last is not None and 0 < s.t - last[0] <= HP_GAP_S:
            self.patrol.learn_hp(last[2], max(0.0, last[1] - s.hp), s.t - last[0])

    def _learn_touches(self, s: Scene) -> None:
        """HP_BY_TOUCHES: the touches that took HP she was told of since the last pass (the hunter's: Hunter.report, a
        MISS not), each the share of her HP the map's hardest-hitting kind's touch takes (attack.danger), with the time
        since (TOUCH_GAP_S at the most: her trips' too), charged to the stop she's at or going to (Patrol.learn_hp)."""
        rep, d = self.h.report, self.danger
        told = getattr(rep, "touches", 0) - getattr(rep, "misses", 0)
        pid = self.visit.ground if self.visit is not None else None
        last, self._hp = self._hp, (s.t, told, pid)
        if last is None or last[2] is None or d is None or not 0 < s.t - last[0] <= TOUCH_GAP_S:
            return
        share = max((t.share for t in d.threats.values()), default=0.0)
        self.patrol.learn_hp(last[2], max(0, told - last[1]) * share, s.t - last[0])

    def _book(self, s: Scene) -> None:
        """The hunt loop's bookkeeping, and each new kill credited to the platform it died on (`credit`)."""
        new = self.h.ledger.kills[self._kills_seen:]
        super()._book(s)
        avg = self._avg_exp()
        for k in new:
            kind = k.kind or k.named
            exp = self.tour.exp.get(kind, avg) if kind is not None else avg
            c = self.credit.setdefault(k.platform if k.placed else k.ground, [0, 0.0])
            c[0] += 1
            c[1] += exp

    def _covered(self) -> set[int]:
        """The platforms the stop gets at: its own, and those of the spawn points it reaches."""
        stop = self.patrol.stops[self.stop]
        return {stop.platform} | {self.patrol.points[q].platform for q, c in stop.cover if c > 0}

    def _targets(self, s: Scene, now: float):
        """The hunt loop's targets with the stop's platforms for its ground: wanted monsters on them, on hers, and any
        in reach."""
        ground = self._covered() if self.stop is not None else None
        targets, visible, hidden = fight._targets(self.h, self.fs, s, self.kinds, ground, True, now)
        self.h.looter.pending.clear()
        return self._with_unseen(s, targets, visible, hidden)

    # --- at a stop ---------------------------------------------------------------------------------------------

    def _at_ground(self, s: Scene, targets, visible, hidden, now: float) -> None:
        here = s.platform.id
        if self._back_due(s):  # a knock (or a pile) left her somewhere worse than the stop: back first
            return self._back(s)
        if any(self._hits(s, m) for m in targets):  # in reach now: shot (beats going on: see the top)
            act = self._stance(s, targets, visible, hidden, approach=True, log=False)
            if act[0] != "wait":
                return self._shoot(s, targets, visible, hidden, now, "shoot", approach=True)
        wide = self._wide(s)
        held = self._held(wide)
        reason = self._leaving(wide, held)
        if reason is not None:
            return self._leave(wide, reason, visible, hidden, now)
        if any(m.platform == here for m in targets):  # on her platform: stance closes in
            act = self._stance(s, targets, visible, hidden, approach=True, log=False)
            if act[0] != "wait":
                return self._shoot(s, targets, visible, hidden, now, "shoot", approach=True)
        if (stop := self._sweep(wide, self._home())) is not None:  # piles worth a round trip from the stop's place
            return self._sweep_act(wide, stop, visible, hidden, now)
        if held and (step := self._reach_step(wide, held)) is not None:
            self._note("reach")
            self.h.actions.walk(*step)
            self._idle_since = None
            return None
        return self._wait(wide, targets)

    def _back_due(self, s: Scene) -> bool:
        """Whether she goes back to the stop she's at (see the top): she got to it, she's on another platform (a
        touch's knock left her there, or a pile), and the danger where she stands (attack.danger, with the look's
        monsters) is BACK_MARGIN over the stop's place's; BACK_TRIES times a visit at the most, BACK_REST_S apart, and
        not within the hunt loop's SPOT_REST of a trip there that failed. Never where nothing counts as a danger."""
        v, d = self.visit, self.danger
        if not BACK or d is None or not d.any or v is None or v.arrived is None or s.platform is None \
                or s.where is None or s.platform.id == v.ground:
            return False
        now = time.monotonic()
        if v.backs >= BACK_TRIES or now < self._no_high_until or now < getattr(self, "_back_rest", 0.0):
            return False
        return d.at(s.platform.id, s.where.x, s.monsters) > d.at(v.ground, v.spot.x, s.monsters) + BACK_MARGIN

    def _back(self, s: Scene) -> None:
        """Back to the stop's place (`_back_due`), 'back to spot': a trip as any (the navigator's: planned and
        replanned at every state against every monster), with no shooting in passing on the platform she leaves."""
        v, d = self.visit, self.danger
        v.backs += 1
        self._back_rest = time.monotonic() + BACK_REST_S
        self.log(f"  back to stop {v.ground} at x {v.spot.x:.0f} from platform {s.platform.id} (danger "
                 f"{d.at(s.platform.id, s.where.x, s.monsters):.2f} here, {d.at(v.ground, v.spot.x, s.monsters):.2f} "
                 f"there)")
        self._to_spot_obj(v.spot)
        self.obj.why = "back to spot"
        self.obj.passing[s.platform.id] = PASSING_S
        return self._travel("back to spot")

    def _backing(self, s: Scene) -> bool:
        """Whether she's on her way back to the stop (`_back`), or due to go (`_back_due`)."""
        o = self.obj
        if o is not None:
            return o.why == "back to spot"
        return self._back_due(s)

    def _close_quarters(self, s: Scene, monsters) -> bool:
        """The hunt loop's rule 1, which yields to her way back to the stop (`_backing`: the navigator's trip keeps
        her body off theirs) unless one of `monsters` is on her (its body on hers, CONTACT_SLACK more: on her platform
        or beside her, stance's `beside`)."""
        if not super()._close_quarters(s, monsters):
            return False
        if not self._backing(s):
            return True
        x, here = s.where.x, s.platform
        return any(beside(m, here, x, self.danger)
                   and abs(m.x - x) <= half_of(m.kind, self._halves) + HER_HALF + CONTACT_SLACK for m in monsters)

    def _held(self, s: Scene) -> list[Monster]:
        """The wanted monsters her memory has (memory.SURE), seen within HOLD_S, that she can get at from her platform:
        on it, or in a star's reach from somewhere along it."""
        here, out = s.platform, []
        for e in self.h.memory.alive(s.t, SURE):
            if (self.kinds is not None and e.kind not in self.kinds) or s.t - e.seen > HOLD_S:
                continue
            if e.platform == here.id or any(True for _ in reaching(here, [e.monster])):
                out.append(e.monster)
        return out

    def _reach_step(self, s: Scene, held: list[Monster]) -> tuple[str, float] | None:
        """A walk along her platform (which way, how far: REACH_STEP at most) to where a star reaches more of the `held`
        monsters on other platforms than from where she stands (the most, then the nearest); None if none does."""
        here, x = s.platform, s.where.x
        off = [m for m in held if m.platform != here.id]
        now = sum(self._hits(s, m) for m in off)
        best = None
        for g, n in reaching(here, off):
            if n > now and (best is None or (n, -abs(g - x)) > (best[0], -abs(best[1] - x))):
                best = (n, g)
        if best is None or abs(best[1] - x) < 10:
            return None
        return (RIGHT if best[1] > x else LEFT), min(abs(best[1] - x), REACH_STEP)

    def _stay_gain(self, s: Scene, held: list[Monster]) -> float:
        """EXP a second of staying at the stop (see the top): the `held` monsters' EXP over their kill time and the
        walk to where she gets at the nearest of them, less what touches at the stop cost her meanwhile (STAY_NET:
        Patrol.hp_rate at HP_PRICE); with none, the respawns expected in reach (Patrol.wait_gain)."""
        if not held:
            return self.patrol.wait_gain(self.stop, s.t)
        here, x, avg = s.platform, s.where.x, self._avg_exp()
        exp = sum(self.tour.exp.get(m.kind, avg) for m in held)
        walk = 0.0
        if not any(self._hits(s, m) for m in held):
            near = [abs(m.x - x) for m in held if m.platform == here.id]
            near += [abs(g - x) for g, _ in reaching(here, [m for m in held if m.platform != here.id])]
            walk = min(near, default=0.0) / WALK_SPEED
        gain = exp / (exp * self.patrol.kill_s + walk)
        if STAY_NET and self.stop is not None:  # (less what touches there cost meanwhile, as the loop's rate)
            st = self.patrol.stops[self.stop]
            gain -= patrol_mod.HP_PRICE * self.patrol.hp_rate(st.platform, st.x)
        return gain

    def _next_dest(self, s: Scene) -> tuple[int, float] | None:
        """Leaving the stop: the place of the stop she'd go to next (Patrol.choose, peeked: `_instead`), the sweep's way
        on; None: none."""
        if s.platform is None or s.where is None:
            return None
        j, _ = self._instead(s)
        if j is None:
            return None
        stop = self.patrol.stops[j]
        return stop.platform, stop.x

    def _sweep_terms(self, s: Scene, dest: tuple[int, float] | None) -> tuple[float, float | None]:
        """The sweep's terms: her time's worth (`_time_worth`), sweep.budget_for's budget (no lulls: she doesn't wait for
        respawns at a stop)."""
        return self._time_worth(s, dest), None

    def _time_worth(self, s: Scene, dest: tuple[int, float] | None = None) -> float:
        """What her time is worth now, EXP a second, on her way to `dest` (the sweep's price of it): at the stop, on a
        round trip back to it, what staying there gives (`_stay_gain`, by her memory's monsters she gets at from her
        platform: `_held`); else the loop's rate (Plan.rate: going on); or, more, what's in reach now (EXP before
        loot)."""
        v, plan = self.visit, self.patrol.plan
        if v is not None and v.arrived is not None and dest is not None and dest == self._home() \
                and s.platform is not None and s.platform.id == v.ground:
            rate = self._stay_gain(s, self._held(s))
        else:
            rate = 0.0 if plan is None else plan.rate
        return max(rate, self._shoot_rate(s, list(s.monsters))[0])

    def _instead(self, s: Scene, heading: int | None = None) -> tuple[int | None, dict]:
        """Where she'd go instead as of look `s` (Patrol.choose, peeked: with the stop she's `heading` to as the
        default on a trip, else from the stop she's at), looked at again every RECHOOSE_S."""
        a = self._alt
        if a is None or a[1] != heading or not 0 <= s.t - a[0] < RECHOOSE_S:
            skip = [g for g, until in self._skip.items() if until > time.monotonic()]
            j, info = self.patrol.choose(s.t, s.platform.id, s.where.x, current=self.stop if heading is None else None,
                                         skip=skip, heading=heading, peek=True)
            self._alt = a = (s.t, heading, j, info)
        return a[2], a[3]

    def _leaving(self, s: Scene, held: list[Monster]) -> str | None:
        """Why she leaves the stop now, or None to stay (see the top)."""
        v, now = self.visit, time.monotonic()
        if v.arrived is None:
            return "not at the stop"
        if s.platform.id != v.ground:
            return None if held else "off the stop, nothing in reach"
        if now - v.arrived >= STOP_MAX_S and not held:
            telemetry.emit("event", kind="backstop", text=f"STOP_MAX_S: {STOP_MAX_S:.0f} s at stop {v.ground}")
            return f"{STOP_MAX_S:.0f} s here"
        stay = self._stay_gain(s, held)
        j, info = self._instead(s)
        plan = self.patrol.plan
        bar = max(0.0 if plan is None else plan.rate, 0.0 if j is None else info.get("gain", 0.0))
        if stay > 0 and stay >= bar:
            v.empty_since = None
            return None
        if v.empty_since is None:
            v.empty_since = now
        if now - v.empty_since < LEAVE_CONFIRM_S:
            return None
        if not held:
            return "nothing left in reach"
        return f"{stay:.1f} EXP/s here, going on gives {bar:.1f}"

    def _leave(self, s: Scene, reason: str, visible=(), hidden=(), now: float | None = None) -> None:
        """Leaving the stop: the sweep's piles on her way to the next one first (`_next_dest`), its drops' landing waited
        for where that pays; then the next stop."""
        if (stop := self._sweep(s, self._next_dest(s))) is not None:
            return self._sweep_act(s, stop, list(visible), list(hidden), time.monotonic() if now is None else now)
        v = self.visit
        there = 0.0 if v.arrived is None else time.monotonic() - v.arrived
        self.log(f"  leaving stop {v.ground}: {reason} ({there:.1f} s there, {v.exp:.0f} EXP)")
        if self.stop is not None and v.arrived is not None:
            self.patrol.left(self.stop, time.monotonic(), reason == "nothing left in reach", v.exp)
        self._next_ground(s)
        return self._travel("advance")

    # --- on the way ----------------------------------------------------------------------------------------------

    def _interrupt(self, s: Scene) -> bool:
        """The hunt loop's reasons to stop a trip, and on a trip to a stop, every RECHOOSE_S, a pile seen elsewhere
        worth going to instead (Patrol.choose, the stop she's heading to as the default), or on a trip to a pile the
        loop's next stop when the pile no longer pays (Patrol.choose's "back to the loop"): taken at the next pass."""
        if super()._interrupt(s):
            return True
        o, v = self.obj, self.visit
        if s.where is None or s.platform is None or o is None or o.kind != "spot" or self.stop is None \
                or v is None or v.arrived is not None or o.ground != v.ground:
            return False
        self.patrol.belief.update(s.t, self.h.memory, s.players if s.dot is not None else None)
        j, info = self._instead(s, heading=self.stop)
        self._reroute = j is not None and j != self.stop and info["why"] in ("pulled", "back to the loop")
        return self._reroute

    def _while_travelling(self, s: Scene, targets, visible, hidden, now: float) -> str | None:
        """The hunt loop's, on its way to the pile `_interrupt` saw (or back to the loop) if it did."""
        if self._reroute:
            self._reroute = False
            o, v = self.obj, self.visit
            if o is not None and o.kind == "spot" and v is not None and v.arrived is None and self.stop is not None:
                self._next_ground(s, heading=self.stop)
        return super()._while_travelling(s, targets, visible, hidden, now)

    # --- the next stop -----------------------------------------------------------------------------------------

    def _next_ground(self, s: Scene, heading: int | None = None) -> None:
        """The next stop (Patrol.choose; on a trip, `heading`: the stop she's on her way to, which only a pile seen
        elsewhere replaces), the loop looked at again first (Patrol.replan), and a trip to it."""
        self._leave_ground()
        self._exit_reason = None
        now, p = time.monotonic(), self.patrol
        p.learn_kills(self.report.throws, self.report.exp)
        why = p.replan(now)
        if why is not None:
            camp = p.camping(now)
            self.log(f"  loop: {p.plan.text(p.platform_of)} ({why}; camping at {p.platform_of(camp.stops[0])} would "
                     f"give {camp.rate * 60:.0f})")
            telemetry.emit("grind", what="loop", why=why, **p.plan.record(p.platform_of),
                           camping=[p.platform_of(camp.stops[0]), round(camp.rate * 60, 1)])
        if now - self._stated >= STATE_EVERY_S:
            self._stated = now
            telemetry.emit("grind", what="state", **self._state())
        where = s.where
        here = where.platform if where is not None else None
        skip = [g for g, until in self._skip.items() if until > now]
        j, info = p.choose(now, None if here is None else here.id, None if where is None else where.x,
                           current=self.stop if heading is None else None, skip=skip, heading=heading)
        self._alt = None
        if j is None:
            if not p.stops:
                raise RuntimeError("no stop on this map")
            j, info = (self.stop or 0), {**info, "stop": p.stops[self.stop or 0].platform}
        else:
            self.pulls += info["why"] == "pulled"
            self.skips += len(info["skipped"])
            self.log(f"  next: {choice_text(info)}")
        stop = p.stops[j]
        spot = Spot(stop.platform, stop.x, 1.0, self.model.safe(stop.platform))
        self.stop = j
        self.visit = Visit(stop.platform, spot)
        self.report.grounds.append(stop.platform)
        self.stack.clear()
        self.obj = Objective("spot", stop.platform, stop.x, ground=stop.platform, spot=spot)
        telemetry.emit("grind", what="next", **info)
        telemetry.emit("tour", order=list(self.report.grounds), stop=stop.platform,
                       spot={"platform": stop.platform, "x": round(stop.x, 1), "coverage": 1.0})

    def _leave_ground(self) -> None:
        """A stop's stay is kept (`stays`); the value map isn't written (see the top)."""
        v = self.visit
        if v is None or v.arrived is None:
            return
        self._left[v.ground] = time.monotonic()
        self.stays.append(time.monotonic() - v.arrived)

    def _rate_here(self) -> float:
        """Her EXP a second over the hunt so far (the mini-map's 'hunt' record)."""
        return self.report.exp / max(1.0, time.monotonic() - self._start)

    def _best_rate(self, s: Scene) -> tuple[float, int] | None:
        """The loop's EXP a second, and the stop she's going to (the mini-map's rate to beat)."""
        p = self.patrol.plan
        return None if p is None or self.stop is None else (p.rate, self.patrol.platform_of(self.stop))

    # --- the end of the run ------------------------------------------------------------------------------------

    def _state(self) -> dict:
        """What the grinding did so far, and what she learned (the 'grind' records 'state' and 'end')."""
        p, st, b = self.patrol, sorted(self.stays), self.patrol.belief
        credit = sorted(self.credit.items(), key=lambda kv: (-kv[1][1], kv[0] if kv[0] is not None else -1))
        return {"loops": p.loops, "stops": len(st), "stay_median": round(median(st), 2) if st else None,
                "stay_longest": round(st[-1], 2) if st else None, "pulls": self.pulls, "skips": self.skips,
                "refill_scale": round(b.scale, 3), "refill_tick": round(1.0 - math.exp(-b.rate * TICK_S), 3),
                "refill_others_tick": round(1.0 - math.exp(-b.others * TICK_S), 3), "own": round(b.own(b.t), 3),
                "refills": b.refills, "watched_s": round(b.watched_s), "kill_s": round(p.kill_s, 4),
                "takes": {k: round(b.take(k), 2) for k in sorted({q.kind for q in p.points})},
                "hp": {str(pid): round(p.hp_seen[pid] / p.hp_time[pid] * 60, 2) for pid in sorted(p.hp_time)
                       if p.hp_time[pid] >= HP_LOG_S},
                "credit": {str(pid): [n, round(e, 1)] for pid, (n, e) in credit}}

    def _log_grind(self) -> None:
        """What the grinding did: loops, stops and stays, pulls and skips, what she learned, and her kills by the
        platform they died on (the log's line, and the 'grind' record 'end')."""
        d = self._state()
        stays = (f" (a median {d['stay_median']:.1f} s at each, {d['stay_longest']:.1f} s at the longest)"
                 if d["stops"] else "")
        self.log(f"  grinding: {d['loops']} loops taken up, {d['stops']} stops{stays}, {d['pulls']} pulled off the "
                 f"loop, {d['skips']} stops skipped; a free point refilling at {d['refill_tick']:.0%} a tick lately "
                 f"(x{d['refill_scale']:.2f} the prior's pace; {d['refills']} refills seen, {d['watched_s']} s of "
                 f"points seen free), a kill {d['kill_s']:.3f} s a point of EXP, others take {d['takes']}; HP bars a "
                 f"minute by platform ({HP_LOG_S:.0f} s there at least) {d['hp']}; kills by platform: "
                 + ", ".join(f"{pid}: {n} ({e:.0f} EXP)" for pid, (n, e) in d["credit"].items()))
        telemetry.emit("grind", what="end", **d)
