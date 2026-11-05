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
  and the loot sweep's piles on her way to the stop (behavior.sweep: each within what it may add to her way): kills
  and pick-ups between moves, without stopping to clear a ground. Every RECHOOSE_S of a trip to a stop, a pile of
  monsters seen elsewhere (Patrol.choose with the stop she's heading to as the default) takes her there instead, and on
  a trip to one that no longer pays for its detour back to the loop's next stop (`_interrupt`, `_while_travelling`).
- Her drops (from 2026-10-30, the loot sweep): at a stop, with nothing in reach, those within what each may add to a
  round trip from its place (Hunt._allow: until 2026-11-01 those worth it at what staying there gives, `_stay_gain`, at
  MESO_EXP an meso); leaving it, before she goes, those within it of her way to the next
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
from ..attack.danger import level
from ..attack.stance import CONTACT_SLACK, beside, half_of
from ..localization.memory import SURE
from ..localization.monsters import Monster, bodies
from ..localization.scene import Scene
from ..motion.bodies import HER_HALF
from ..motion.physics import WALK_SPEED
from . import fight, lootchain
from ..perception.loot import MESOS
from .hunt import PASSING_S, Hunt, Objective, Visit, bar_exp
from .ledger import DROP_LANDS, EXPIRE_S, FOLLOW_S, LANDED_S, REVIVE_S, dying
from .looting import STOOD_ON
from .patrol import STOP_S, TICK_S, Patrol, choice_text, filled, reaching
from .targeting import ASKS, IDLE_S, NO_WAY_S, REACT_AHEAD_S, ROUTE_STEP, Targeting
from .targeting import on as target_on

SEARCH_S = 2.5         # s past a kill's drops' landing (ledger.DROP_LANDS, LANDED_S) she waits by its pile for a search of it that can say
                       # it's empty (`_none_landed`; the rotation is back on a strip in 1.2-2.4 s: ledger.REVIVE_S)
STOP_MAX_S = 40.0      # guess: backstop, s at one stop with nothing in reach (there's a round to go); and from the final
                       # review of 2026-11-02, s waiting there on end (Hunt._wait, nothing shot, no step) with monsters
                       # held: on the thief's seed 19 of the standard grind she stood 123 s at 16 for a Horny on 15 the
                       # fight took for dead (a kill of hers bound to its track) and memory had alive (`_held`)
# s with nothing left in reach before she leaves a stop: one look's miss isn't enough (a hit monster hides under its
# damage numbers for 1-3 s: fight's hidden targets keep it a target meanwhile, and HOLD_S keeps it held)
TO_TARGET = "to target"  # (Objective.why) a trip to the place her target can be hit from (`_to_place`)
LEAVE_CONFIRM_S = 0.3  # (a guess; the drops of her last kills there are the sweep's when she leaves: `_leave`)
HOLD_S = 3.5           # a monster her memory has, seen within this long (tracking.HIDE_S), holds her at a stop
REACH_STEP = 200.0     # guess: units she walks along her platform at once at most to bring held monsters into reach
HP_GAP_S = 2.0         # guess: s between two looks at most for the HP gone between them to be charged to a platform
RECHOOSE_S = 1.0       # guess: s between two looks at where she'd go instead (Patrol.choose takes about 1 ms on Tree
                       # Dungeon, its routes kept by the planner)
HP_LOG_S = 10.0        # s at a stop at least for the log to say what HP she lost there (the log's choice; by the
                       # stop's platform, what knocks took her to from it in; by the platform she was on without
                       # HP_BY_TOUCHES)
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
# grind (12 seeds of 5 minutes, paired against the branch before: pos/impl/out/cmp_E.txt) touches 86.8 -> 81.2 a run (se
# 4.0), those at piles 16.1 -> 12.2 (se 2.0), EXP a minute 385 -> 373 (se 16). Off from 2026-10-31 ~16:00, the user's
# choice of "more exp and loot" over as safe as master: at their prices, the rest as first built (12 seeds,
# D:/MapleHelper-scratch/replan/loot/price), off made 18.6 EXP a minute more (se 11.9) and 15.0 more loot, 2.8 touches
# and 0.33 bars of HP fewer (the simulator's crowd takes her about 6 bars a run whatever she does).
BACK = False
BACK_MARGIN = 0.05
BACK_TRIES = 3
BACK_REST_S = 5.0
# What she learns of the HP a stop costs (Patrol.learn_hp): her HP bar between two looks at most HP_GAP_S apart,
# charged to the platform she's on (False, until 2026-10-31), where her pet's potions net out touches and her trips
# aren't looked at (in the simulator's grind on Tree Dungeon it saw 3.27 of the 8.68 bars she lost on seed 1, and none
# of what she lost on 9 and 8 counted against the loop's stops, 10 and 7); or her touches (True: `_learn_touches`),
# charged to the stop she's at or going to, her trips' time and touches too, each at the share of her HP a touch by the
# kind nearest her takes (`_touch_share`: until the review of 2026-10-31 every touch as the map's hardest-hitting
# kind's, on Tree Dungeon a Horny Mushroom's 0.156 where a Green Mushroom takes 0.07 and a Slime 0.012). From that
# review to the user's choice that evening ("more exp and loot") a stop was charged only what she lost having got to
# it (on its platform or wherever a knock took her from it), her trips not: at their prices, the rest as first built
# (12 seeds, D:/MapleHelper-scratch/replan/loot/price), charging the trips again made 22.4 EXP a minute more (se 10.5)
# and 19.5 more loot for 0.16 bars of HP (se 0.61).
HP_BY_TOUCHES = True
TOUCH_GAP_S = 30.0     # s between two looks at most for the touches between them to be charged (the grind's looks
                       # come between her moves)
TOUCH_NEAR = 100.0     # units between her body and a monster's at most for its kind to be a touch's (`_touch_share`)
# Whether what staying at a stop gives (`_stay_gain`) is net of what touches there cost meanwhile (Patrol.hp_rate at
# hp_price), as the loop's rate is (Patrol.rate), from 2026-10-31; False: its EXP alone, as before. With HP_BY_TOUCHES,
# her way back and the sweep's guards, in the harness's standard grind (as above: cmp_G.txt) touches 86.8 -> 80.8 (se
# 3.0), those that hurt 57.3 -> 53.1 (se 2.4), HP potions 24.5 -> 22.8, EXP a minute 385 -> 365 (se 16); with the
# stops weighing danger and the HP priced as the hunt loop's too (behavior.patrol: STOP_DANGER, HP_PRICE, HP_PRIOR_S,
# cmp_J.txt) 77.6 touches, 50.0 hurt, 6.6 bars of HP (8.0), 20.3 potions, 359 EXP a minute.
STAY_NET = True
# (Until 2026-11-01, two switches priced her drops in EXP at MESO_EXP an meso, which the sweep no longer does: its
# piles each within what it may add to her way, Hunt._allow. DROPS_IN_CHOICE, off: the next stop's choice counted what
# a visit to each would pick up, the sweep's round trip from its place; in 15-minute runs of the standard grind EXP a
# minute -4.4 (se 7.1), loot -4.6 (7.0), in the live drops' worlds EXP -8.9 (3.5) for loot +0.8 (1.2). PILE_HP, on from
# the review of 2026-10-31: the sweep's seconds dearer by the HP a second her touches took while she was after a pile;
# in the live drops' worlds her time on her drops -7.3 s, touches that hurt there -0.85 (0.59), EXP -0.7 (3.8); its
# place is the allowance's SAFER_MARGIN, a pile more dangerous than where she stands on her way only.)


class Grind(Hunt):
    """The hunt loop grinding round the map (see the top). `patrol`: its loop and belief (None: one for the map, from
    the tour's kinds worth hunting and their EXP, the navigator's route times, and for the danger hook the hunter's
    map danger, the positioning model's: Hunter.danger, attack.danger; patrol's default for a hunter without one)."""
    tp: Targeting | None = None  # (none: tests make a grind without __init__)
    _target_place: tuple | None = None

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
        self._credited: dict[int, tuple] = {}     # ... each recent kill's (by id): (it, the platform credited, its EXP)
        self._moves_seen = 0                      # the ledger's moved kills taken (`_book_kills`)
        self.stays: list[float] = []              # s at each stop she got to
        self.pulls = self.skips = 0
        self._alt: tuple | None = None            # where she'd go instead: (when, heading, stop, choose's info)
        self._reroute = False                     # a pile seen on a trip (or hers gone): to it at the next pass
        self._start = time.monotonic()
        self._hp: tuple | None = None  # the last look's (time, HP or touches told, platform), for learn_hp
        self._stated = time.monotonic()
        # The loop's promise against what it delivered (the 'grind' records 'state' and 'end': `_promise`): when the loop
        # she's on was taken up and her EXP then; and the loop's rate (Plan.rate, the bar her stays and pulls are held to)
        # over her looks' time, and that time (the last look's, None after a pause)
        self._loop_at, self._loop_exp = time.monotonic(), 0.0
        self._promised = self._promised_s = 0.0
        self._promised_at: float | None = None
        # The target-first planner (behavior.targeting; GAMELAB_TARGET): the fights at a stop, arrivals included
        self.tp: Targeting | None = Targeting(self) if target_on() else None
        self._chain_off = -1e18  # until when her drops go the old way: a monster stood in a chain's first walk (lootchain.BLOCKED_S)
        self._target_place: tuple | None = None  # where her last trip to a target went (`_to_place`): a chain goes on to it
        self._planned = False  # (the stance being asked is the planner's, for the primary: `_among`)
        self._asked = (None, None, None, None)  # (the look, target, approach and act of the stance last asked unsaid: `_stance`)
        if self.tp is not None and self.tp.chain:
            self.nav.keep_avoid = True
            self.h.looter.store_has_them = True
        if self.tp is not None and self.tp.screen:
            self.nav.react_ahead = REACT_AHEAD_S
            self.h.attacks.whole_screen = True
            if self.nav.escort is not None:
                self.nav.escort.not_ignored = True  # (no shot first at one her attacks ignore: it isn't a target anywhere)
        if self.tp is not None:
            self.h.actions.lock_watch = self.tp.watch_lock
            self.h.drops.revive_s = EXPIRE_S  # (a pile of hers seen gone and then seen again is hers, however late: 100%)

    def run(self, minutes: float):
        try:
            report = super().run(minutes)
        finally:
            if self.tp is not None:
                self.h.actions.lock_watch = None
                self.h.drops.revive_s = REVIVE_S
                self.tp.end()
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
        if getattr(self, "_loop_at", None) is not None:  # (and the loop's clock: its promise against what it gave)
            self._loop_at += seconds
        self._alt = self._hp = self._promised_at = None

    # --- what she knows ------------------------------------------------------------------------------------

    def _look(self) -> Scene:
        s = super()._look()
        if s.where is not None:
            self.patrol.belief.update(s.t, self.h.memory, s.players if s.dot is not None else None)
        self._learn_hp(s)
        self._price_hp()
        now, plan = time.monotonic(), self.patrol.plan
        self.h.exp_made = (self.report.exp, now - self._start)  # (what a throw's lock costs: fight.made_rate)
        if self._promised_at is not None and plan is not None:  # (the loop's rate over the time since the last look)
            self._promised += plan.rate * (now - self._promised_at)
            self._promised_s += now - self._promised_at
        self._promised_at = now
        return s

    def _price_hp(self) -> None:
        """What a bar of her HP costs (Patrol.hp_price, the loop's and her stays' price of her touches): its potions at
        the user's reference prices (hunt.bar_exp) from her max HP as perception reads it (Hunter.max_hp: her HUD's
        number), patrol.HP_PRICE until it's read. And her EXP a second (Hunter.exp_rate: the loop's, Plan.rate), what
        her landings' prices weigh her time by (behavior.fight.landing_prices), and on each platform (Hunter.ground_rate:
        `_ground_rate`, her hang under a rope's top)."""
        most = getattr(self.h, "max_hp", None)
        if most is not None:
            self.patrol.hp_price = bar_exp(most)
        plan = self.patrol.plan
        if plan is not None and plan.rate > 0:
            self.h.exp_rate = plan.rate
        self.h.ground_rate = self._ground_rate

    def _ground_rate(self, pid: int) -> float | None:
        """EXP a second she'd make on platform `pid` as the patrol expects it now: at its stop there (Patrol.expect: the
        monsters her memory has in reach of it and the belief's stock at its other points, her share), net of what
        touches cost her meanwhile (Patrol.net), over its shooting at her kill time and STOP_S (as Patrol.choose rates
        a stop with no travel); None where the platform has no stop (control.navigator.HANG_TAKE, through
        Hunter.top_price: what the wait under the top of a rope up to it costs her)."""
        i = self.patrol.stop_on(pid)
        if i is None:
            return None
        exp, _, shoot, _ = self.patrol.expect(i, time.monotonic())
        return self.patrol.net(i, exp) / (shoot + STOP_S) if exp > 0 else 0.0

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
        """HP_BY_TOUCHES: the touches that took HP she was told of since the last look (the hunter's: Hunter.report, a
        MISS not), each the share of her HP a touch by the kind nearest her takes (`_touch_share`), with the time since
        (TOUCH_GAP_S at the most: her trips' too), charged to the stop she's at or going to (Patrol.learn_hp: the
        visit's ground as of the last look)."""
        rep, d = self.h.report, self.danger
        told = getattr(rep, "touches", 0) - getattr(rep, "misses", 0)
        pid = self.visit.ground if self.visit is not None else None
        last, self._hp = self._hp, (s.t, told, pid)
        if last is None or last[2] is None or d is None or not 0 < s.t - last[0] <= TOUCH_GAP_S:
            return
        new = max(0, told - last[1])
        self.patrol.learn_hp(last[2], new * self._touch_share(s) if new else 0.0, s.t - last[0])

    def _touch_share(self, s: Scene) -> float:
        """The share of her HP a touch she was told of in look `s` took: a touch by the kind of the monster nearest her
        body that could have touched her (on her platform, or on one just under or over hers level enough for their
        bodies to meet: attack.danger's `level`), within TOUCH_NEAR of it (Danger.taken: what its touch takes, under
        DANGER_MIN too); with none, the map's hardest-hitting kind's (as until the review of 2026-10-31 every touch
        was)."""
        d, here = self.danger, s.platform
        best = None
        if here is not None and s.where is not None:
            x = s.where.x
            for m in s.monsters:
                t = d.threats.get(m.kind)
                if t is None or m.platform is None:
                    continue
                if m.platform != here.id:
                    q = here.peer(m.platform)
                    if q is None or not level(t, q, here, x):
                        continue
                gap = abs(m.x - x) - t.half - HER_HALF
                if gap <= TOUCH_NEAR and (best is None or gap < best[0]):
                    best = (gap, m.kind)
        if best is not None:
            return d.taken(best[1])
        return max((d.taken(k) for k in d.threats), default=0.0)

    def _book_kills(self, s: Scene, kills: list) -> None:
        """The hunt loop's bookkeeping, and each kill of the hunt's credited to the platform it died on (`credit`); one
        the tracker put on a monster of another platform since (the ledger's `moved`: the magician's, whose kills go by
        her search), its credit moved there (at every look: Hunt._book passes its new kills, none too)."""
        super()._book_kills(s, kills)
        avg = self._avg_exp()
        for k in kills:  # (the stand-by trip's aren't: the report has them, `credit` is where she hunted)
            _, exp = self._kill_exp(k, avg)
            where = k.platform if k.placed else k.ground
            c = self.credit.setdefault(where, [0, 0.0])
            c[0] += 1
            c[1] += exp
            self._credited[id(k)] = (k, where, exp)
        moved = getattr(self.h.ledger, "moved", ())
        for _, k, _, _, _ in moved[self._moves_seen:]:
            got = self._credited.get(id(k))
            if got is None or not k.placed or k.platform is None or k.platform == got[1]:
                continue
            _, where, exp = got
            if where in self.credit:
                self.credit[where][0] -= 1
                self.credit[where][1] -= exp
            c = self.credit.setdefault(k.platform, [0, 0.0])
            c[0] += 1
            c[1] += exp
            self._credited[id(k)] = (k, k.platform, exp)
        self._moves_seen = len(moved)
        self._credited = {i: v for i, v in self._credited.items() if s.t - v[0].t <= FOLLOW_S + 1.0}

    def _covered(self) -> set[int]:
        """The platforms the stop gets at: its own, and those of the spawn points it reaches."""
        stop = self.patrol.stops[self.stop]
        return {stop.platform} | {self.patrol.points[q].platform for q, c in stop.cover if c > 0}

    def _targets(self, s: Scene, now: float):
        """The hunt loop's targets with the stop's platforms for its ground: wanted monsters on them, on hers, and any
        in reach."""
        screen = self.tp is not None and self.tp.screen
        ground = self._covered() if self.stop is not None and not screen else None
        targets, visible, hidden = fight._targets(self.h, self.fs, s, self.kinds, ground, True, now)
        self.h.looter.pending.clear()
        targets, visible, hidden = self._with_unseen(s, targets, visible, hidden)
        if self.tp is not None:  # (one monster to hit: behavior.targeting)
            wanted = targets
            free = self._free_trip(self.obj)
            targets = self.tp.decide(s, wanted, visible, hidden, self.obj is not None and not free, free)
            if self.obj is None or free:  # (none she has nothing against for now: set aside at once)
                for _ in range(ASKS):
                    if not targets or self._stance(s, targets, visible, hidden, True, log=False)[0] != "wait":
                        break
                    targets = self.tp.decide(s, wanted, visible, hidden, False, free)
        return targets, visible, hidden

    def _alive(self, s: Scene, monsters: list[Monster], now: float) -> list[Monster]:
        """Of `monsters` of look `s`, those not seen fading out where a kill died, nor taken for dying where one the ledger
        placed died, nor ignored by her attacks; of the kinds she hunts (fight._targets' for the rest of the screen)."""
        fading, led = self.fs.fading(monsters), self.h.ledger
        die = {k: b.die_s for k, b in bodies(self.model).items()}
        ignored = getattr(self.h.attacks, "ignored", None) or ()
        return [m for m in monsters if m not in fading and getattr(m, "track", None) not in ignored
                and (self.kinds is None or m.kind in self.kinds)
                and not dying(led.kills, m, now, monsters, die, led.dead)]

    def _rest_of_screen(self, s: Scene, now: float) -> list[Monster]:
        """The monsters on the screen that aren't in her band's look (Scene.swept: the feed's), as `_alive`."""
        swept = getattr(s, "swept", None)
        return self._alive(s, swept, now) if swept else []

    # --- the target-first planner (behavior.targeting; none of this with it off) ---------------------------------

    def _among(self, visible, hidden) -> list[Monster] | None:
        return [*visible, *hidden] if self._planned else None

    def _ends_on_arrival(self, o) -> bool:
        return self.tp is not None and o.kind == "spot"

    def _varies(self, o) -> bool:
        return o.why != TO_TARGET and super()._varies(o)

    def _instead_of_waiting(self, o):
        if self.tp is None or not self.tp.screen or o.why not in (TO_TARGET, "sweep"):
            return None  # (a stop's trip: she waits, and any monster on the screen ends it)
        return self.tp.rather

    def _may_stop_her(self, s: Scene) -> list[Monster]:
        if self.tp is None or not self.tp.screen:
            return super()._may_stop_her(s)
        return self.tp.targets_of(s.monsters)  # (else she stops for one the pass won't have, look after look, no move made)

    def _free_trip(self, o) -> bool:
        """Whether a trip's next pass may take any target on the screen (the whole screen: `_while_travelling`): any but one to a
        pile, unless what's blocked the way to that pile (Targeting.rather)."""
        return (self.tp is not None and self.tp.screen and o is not None
                and (o.kind == "spot" or (o.why == "sweep" and self.tp.piles_wait())))

    def _home(self) -> tuple[int, float] | None:
        """(With the planner no place at a stop she comes back to: her piles' way goes on to nowhere.)"""
        return None if self.tp is not None else super()._home()

    def _whole(self, p, s: Scene, now: float) -> float | None:
        return None if self.tp is None else self.tp.loot_allow(p, s, now)

    def _sweep_most(self) -> float | None:
        return None if self.tp is not None else super()._sweep_most()

    def _landing_wait(self, lands: float) -> None:
        if self.tp is None or not self.tp.screen:
            return super()._landing_wait(lands)
        self.tp.gave_way_at = time.monotonic()  # (her target now if she has one: the next pass)
        return super()._landing_wait(lands)

    def _none_landed(self, s: Scene, pile) -> bool:
        """With the planner: a kill's drops have landed by now and a search there that could say so has none: it dropped
        nothing (0.64 drops a kill; on the dev seeds 2 of 3 of her stands were on nothing, a walk of 60 units and 3 presses
        each, live 43% on the last hours), seen gone as two sightings would have it; drops a monster hid from it have it back as
        soon as a look sees them (DropStore.revive_s). No stand either before they've landed, or (SEARCH_S) while the next look
        may have that search; past that, her walk over its spread."""
        if self.tp is None:
            return False
        now = time.monotonic()
        landed = pile.since + DROP_LANDS + LANDED_S
        if now < landed or (not self.h.drops.searched_empty(s, pile) and now < landed + SEARCH_S):
            if self.tp.screen:  # (nothing to press yet: her target now if she has one, and no wait for it)
                self.tp.gave_way_at = now
            return True
        if not self.h.drops.searched_empty(s, pile):
            return False
        pile.status, pile.gone_at, pile.none_landed = "gone", s.t, True
        self.tp.counts["none landed"] += 1
        self.log(f"  none landed: {pile.kind}'s kill {s.t - pile.since:.1f} s ago on platform {pile.platform} at x {pile.x:.0f}")
        if telemetry.recording():
            telemetry.emit("event", kind="none_landed", killed=round(pile.since, 2), platform=pile.platform, x=round(pile.x),
                           monster=pile.kind)
        return True

    def _guards_it(self, s: Scene, p, m: Monster, on_screen: bool) -> bool:
        """With the planner a monster guards a pile only as her target can: seen, on the platform she's on with it (it's among
        her targets there: "a monster guarding a pile is simply the next target"). She goes to one elsewhere whoever is by it
        now, and who's there then is her target then."""
        if self.tp is None:
            return True
        return on_screen and s.platform is not None and p.platform == s.platform.id

    def _here(self, s: Scene, say: bool = True):
        """The sweep's next stop among her drops on her platform and those at the end of their life, nowhere after them
        (those elsewhere wait for her way on: `_leave`, with where she goes next as its way)."""
        self.tp.here_only = True
        try:
            return self._sweep(s, None, say=say)
        finally:
            self.tp.here_only = False

    def _pile_first(self, s: Scene):
        """With the planner: the sweep's next stop where it goes before the primary (behavior.targeting.pile_first); None:
        the primary first, or no piles."""
        c = self.tp.choice
        if c is None or c.primary is None:
            return None
        stop = self._here(s, say=False)
        if stop is None:
            return None
        pile = self._pile_of.get(stop.spot.key)
        if pile is None or not self.tp.pile_first(pile, stop, time.monotonic()) or self.tp.piles_wait():
            return None
        if self.tp.screen and self._at_pile(s, pile) and (stop.spot.lands > 0.0 or (
                pile.status == "expected" and not any(d.platform == pile.platform for d in s.loot) and self._none_landed(s, pile))):
            return None  # (by it with nothing to press yet: her target now, not a wait)
        return stop

    def _stance(self, s: Scene, targets, visible, hidden, approach: bool, log: bool = True,
                goal: float | None = None) -> tuple:
        """The stance's answer, for the primary alone with the planner (no closing in: its own walk, `Targeting.plan`); asked
        of a look for its answer alone a second time, the first's."""
        if self.tp is None or not self.tp.mine(targets):  # (her wounds on a trip's way, as master's)
            return super()._stance(s, targets, visible, hidden, approach, log, goal)
        if self.tp.screen and self.tp.choice.primary.way.how == "trip":  # (nothing from here: the trip's the plan)
            return self.tp.trip_act(s) if approach else ("wait",)
        if not log and self._asked[0] is s and self._asked[1] is targets[0] and self._asked[2] == approach:
            return self._asked[3]
        self._planned = True
        try:
            act = super()._stance(s, targets, visible, hidden, False, log, goal)
        finally:
            self._planned = False
        act = self.tp.plan(s, act, approach, log)
        if not log:
            self._asked = (s, targets[0], approach, act)
        return act

    # --- at a stop ---------------------------------------------------------------------------------------------

    def _at_ground(self, s: Scene, targets, visible, hidden, now: float) -> None:
        here = s.platform.id
        if self.tp is not None:
            return self._planned_step(s, targets, visible, hidden, now)
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

    def _planned_step(self, s: Scene, targets, visible, hidden, now: float) -> None:
        """A step of the target-first planner at a stop: her drops where they go first (`_pile_first`), the plan against the
        primary at once (Targeting.plan: her attack where it lands, a step or a jump, the walk to its place); with no monster
        to hit her drops within what each may add to her way, then today's choice of where to go next."""
        if (stop := self._pile_first(s)) is not None:
            return self._sweep_act(s, stop, visible, hidden, now)
        if targets:
            act = self._stance(s, targets, visible, hidden, approach=True, log=False)
            if act[0] == "trip":
                return self._to_place(s, act, visible, hidden, now)
            if act[0] != "wait":
                return self._shoot(s, targets, visible, hidden, now, "shoot", approach=True)
        wide = self._wide(s)
        self.tp.quiet = self.tp.screen and not targets  # (with no monster on the screen, her drops whatever they take)
        try:
            stop = self._here(wide)
        finally:
            self.tp.quiet = False
        if stop is not None:
            return self._sweep_act(wide, stop, visible, hidden, now)
        held = [m for m in self._held(wide) if m.track not in self.tp.no_way]
        reason = self._leaving(wide, held)
        if reason is not None:
            return self._leave(wide, reason, visible, hidden, now)
        if self.tp.screen and not targets and not held:  # (nothing on the screen and no drops: another area, no wait)
            return self._leave(wide, "nothing left in reach", visible, hidden, now)
        if self._idle_since is not None and now - self._idle_since >= IDLE_S:
            self._idle_since = None
            return self._leave(wide, "nothing she can get at here", visible, hidden, now)
        return self._wait(wide, targets)

    # --- the loot chain (behavior.lootchain) ------------------------------------------------------------------------

    def _candidate(self, p, s: Scene | None = None) -> bool:
        """The hunt loop's, and with the chain not a pile a look has seen empty: no stop for it while the store's second sighting comes
        (a look that sees drops there has it back at once)."""
        if self.tp is not None and self.tp.chain and p.empty_since is not None:
            return False
        return super()._candidate(p, s)

    def _stand_s(self, p) -> float:
        """With the chain, seen drops: the walk over their stretch and only the presses a run through them won't give."""
        if self.tp is None or not self.tp.chain or not p.xs:
            return super()._stand_s(p)
        lo, hi = self._extent(p)
        return sum(lootchain.stand_s(n, p.tries > 0) for _, n in lootchain.bunches(p.xs)) + (hi - lo) / WALK_SPEED

    def _sweep_act(self, s: Scene, stop, visible, hidden, now: float) -> None:
        """The sweep's next stop carried out: as a chain through it and the ones after (`_loot_plan`); else as the hunt loop's: its guards
        first, and by a kill's drops not seen or not landed yet their wait, or the walk over their spread (`_stand`)."""
        sp = stop.spot
        if self.tp is None or not self.tp.chain or sp.guards or time.monotonic() < self._chain_off:
            return super()._sweep_act(s, stop, visible, hidden, now)
        pile = self._pile_of.get(sp.key)
        if pile is None or ((pile.status != "seen" or sp.lands > 0.0) and self._at_pile(s, pile)):
            return super()._sweep_act(s, stop, visible, hidden, now)
        chain = self._loot_plan(s)
        if chain is None:
            return None  # (nothing to take where it was: said so, the next look's sweep goes on)
        self.log(f"  loot: {chain.text()}")
        return self._go_for(chain.stretches[0].pile, "sweep", s, after=self._sweeper.dest, chain=chain)

    def _loot_plan(self, s: Scene) -> "lootchain.LootChain | None":
        """The chain through the first MOST_PILES of the sweep's itinerary as look `s` has them, up to one with a monster to shoot first;
        a pile not seen yet or not landed its last: to its middle, where the next look says; before her target only those that may go
        before it (Targeting.pile_first); on to her target's place. None: nothing to take."""
        it, c, now = self._sweeper.current, self.tp.choice, time.monotonic()
        before = self._sweeper.dest is None and c is not None and c.primary is not None
        stretches, est = [], 0.0
        for stop in it.stops:
            p = self._pile_of.get(stop.spot.key)
            if len(stretches) >= lootchain.MOST_PILES or p is None or stop.spot.guards \
                    or (before and stretches and not self.tp.pile_first(p, stop, now)):
                break
            if p.status != "seen" or stop.spot.lands > 0.0:  # (nothing to run over for sure: the look there has it)
                return lootchain.LootChain([*stretches, lootchain.Stretch(p, [(p.x, 1)], seen=False)], None, s, stop.leave, now)
            st = self._stretch(s, p)
            if st is not None:
                stretches.append(st)
                est = stop.leave
        if not stretches:
            return None
        dest = self._sweeper.dest if self._sweeper.dest == self._target_place else None  # (a next stop's: the next look's to say)
        return lootchain.LootChain(stretches, dest, s, est, now)

    def _stretch(self, s: Scene, p) -> "lootchain.Stretch | None":
        """Pile `p` as a stretch of a chain: where its drops she can take lie as the store last saw them, each in one pile (Pile.xs, kept to
        date by every look that sees them: DropStore._sighting; Looter.wanted's; a drop she has been at twice that's still there:
        looting.cant_pick, as a stand's). None, with its try spent, if it has nothing to take."""
        looter = self.h.looter
        can = getattr(looter, "can_take", None)
        items = p.items if len(p.items) == len(p.xs) else (None,) * len(p.xs)
        drops = [(x, k) for x, k in zip(p.xs, items) if k is None or k == MESOS or can is None or can(k)]
        drops = [(x, k) for x, k in drops if not any(q == p.platform and abs(x - ux) <= STOOD_ON for ux, q, _ in looter.unpickable)]
        kept = self._stood.get(id(p))
        if kept is not None and kept[0] is p:
            stuck = [d for d in s.loot if d.platform == p.platform and any(abs(d.x - x) < 1.0 for x, _ in drops)
                     and sum(abs(sx - d.x) <= STOOD_ON for sx in kept[1]) >= 2]
            if stuck:
                looter.cant_pick(stuck)
                drops = [(x, k) for x, k in drops if not any(abs(d.x - x) < 1.0 for d in stuck)]
        xs, kinds = [x for x, _ in drops], tuple(k for _, k in drops if k is not None and k != MESOS)
        if not xs:
            self._tried(p)
            return None
        q = self.model.platform(p.platform)
        bunches = [(min(max(x, q.left + lootchain.END_ROOM), q.right - lootchain.END_ROOM), n) for x, n in lootchain.bunches(xs)]
        return lootchain.Stretch(p, bunches, kinds, p.tries > 0)

    def _through(self, o):
        return None if o.chain is None else self._loot_way

    def _loot_way(self, s: Scene) -> list:
        """The chain's way on as of look `s` (TripState.through). Its first look's what it was planned on."""
        o = self.obj
        if o is None or o.chain is None:
            return []
        planner = self.nav.planner

        def enter(a, ax, b, lo, hi):
            got = planner.arrive_within(a, round(ax / ROUTE_STEP) * ROUTE_STEP, b, lo, hi)
            return None if got is None else got[1]
        return o.chain.way(s.platform.id, s.where.x, enter)

    def _chain_over(self, o, result: str) -> bool:
        """Her trip through her drops over, one plan: each stretch she ran over taken as stood on (its try, what a stand tells: `_stand`'s
        end; what's left the next look says), its 'loot_chain' record, and the objective done with: the next pass's step is the
        planner's, with what the look has then (`_planned_step`: a monster in reach, a new chain through what's left). A failed trip
        (`_travel`'s) is as any pile's."""
        chain, st = o.chain, o.trip
        chain.book(st.ran)
        looter = self.h.looter
        for part in chain.stretches:
            if part.ran and not part.booked and part.seen:
                part.booked = True
                kept = self._stood.get(id(part.pile))
                if kept is None or kept[0] is not part.pile:
                    kept = self._stood[id(part.pile)] = (part.pile, [])
                kept[1].extend(part.bunches[i][0] for i in sorted(part.ran))
                if (seen := getattr(looter, "stood_on", None)) is not None:
                    seen(list(part.kinds), part.pile)
                if part.done:  # (a try's the whole of it: cut short halfway, it's as if she'd been stopped on her way there)
                    self._tried(part.pile)
                self.h.report.loot_runs += 1
                if telemetry.recording():
                    telemetry.emit("event", kind="pick", platform=part.platform, x=round(part.pile.x), drops=sum(n for _, n in part.bunches),
                                   stood=[round(part.bunches[i][0]) for i in sorted(part.ran)], tries=part.pile.tries)
        looter.pending.clear()
        self._idle_since = None
        counts = self.tp.counts
        counts["loot chains"] += 1
        counts["loot stops planned"] += len(chain.stretches)
        counts["loot stops done"] += sum(p.done for p in chain.stretches)
        why = st.chain_why or ("a monster in its first walk" if st.walk_blocked else result)
        if telemetry.recording():
            telemetry.emit("loot_chain", why=why, result=result, s=round(time.monotonic() - chain.started, 2), **chain.record())
        if st.walk_blocked:
            self._chain_off = time.monotonic() + lootchain.BLOCKED_S
        if result not in ("arrived", "in position"):
            return False  # (`_travel`: said, and dropped as any trip that failed)
        self.obj = self.stack.pop() if self.stack else None
        return True

    def _chain_stops(self, s: Scene, o) -> bool:
        """Whether look `s` ends a chain that's about to go on (asked before it sets off and at its look ahead): a monster on her,
        its first pile gone, a valuable pile that goes first, one she has hurt to finish, or the choice has a monster in reach now. Not the
        look it was planned on: that's said its say. Asked of the choice rather than of a star's reach as a trip's are (`Hunt._interrupt`): a
        monster the ledger takes for dead, in a star's reach, has no shot from the pass that follows, and stopped her look after look
        (the harness's seed 3 at 62 s)."""
        if s is o.chain.planned_on:
            return False
        wanted = self._may_stop_her(s)
        if self._close_quarters(s, wanted) or self._worthless(o.pile) or self._owed(s, s.monsters) is not None:
            return True
        if (pile := self._valuable_pile(s)) is not None and self._loot_first(s, pile, wanted, say=False):
            return True
        if getattr(s, "what_if", False):
            return False
        c = self.tp.peek(s)
        return c.primary is not None and c.primary.way.how not in ("walk", "trip")

    def _to_place(self, s: Scene, act: tuple, visible, hidden, now: float) -> None:
        """To where her target can be hit from: a trip of the navigator's (its legs against every monster, as any), her drops on
        the way first (the sweep's, within what each may add to her way). Decided again as `_interrupt` says."""
        _, platform, x = act
        self._target_place = (platform, round(x))
        if not self.tp.piles_wait() and (stop := self._sweep(s, (platform, round(x)))) is not None:
            return self._sweep_act(s, stop, visible, hidden, now)
        spot = Spot(platform, x, 1.0, self.model.safe(platform))
        self.obj = Objective("spot", platform, x, ground=platform, spot=spot, why=TO_TARGET)
        return self._travel(TO_TARGET)

    def _drop_objective(self, o, why: str) -> None:
        if o.why != TO_TARGET:
            return super()._drop_objective(o, why)
        m = self.tp.chooser.primary  # (no way there as things are: another target at the next look)
        if m is not None and m.track is not None:
            self.tp.no_way[m.track] = time.monotonic() + NO_WAY_S
        self.obj = self.stack.pop() if self.stack else None
        return None

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
        replanned at every state against every monster), with no shooting in passing on the platform she leaves, her
        finishing on the way of what she has hurt (rule 4b) and the sweep of piles on the way as on any trip ("kill all
        the mobs that are in the way", "pick as much as possible": the user, 2026-10-30), but never cut for one
        (hunt's `_sweep_calls`). From the review of 2026-10-31 to the user's choice that evening the way back neither
        finished nor swept (3 of the 8 trips back in Tree Dungeon's goldens had gone straight to a sweep of the platform
        she was sent back from): with BACK on, at their prices, the rest as first built (12 seeds,
        D:/MapleHelper-scratch/replan/loot/price), finishing and sweeping on it made 14.5 EXP a minute more (se 6.4) and
        15.0 more loot, 4.2 touches fewer."""
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
        on it, or in a star's reach from somewhere along it; not one the fight takes for dead or dying (ledger.dying, as
        fight._targets' dying_seen: no target, so nothing she'd stay for)."""
        here, out = s.platform, []
        led, now = getattr(self.h, "ledger", None), time.monotonic()
        for e in self.h.memory.alive(s.t, SURE):
            if (self.kinds is not None and e.kind not in self.kinds) or s.t - e.seen > HOLD_S:
                continue
            if led is not None and dying(led.kills, e.monster, now, s.monsters, led.die, led.dead):
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

    def _fill(self, period: float) -> float:
        """A free spawn point's chance of a newborn at the map's next pass (the spawn guard's: Hunt._respawn_guard), as
        the grind has learned the refill (patrol.Belief.rate, a second a free point: what her own kills and the other
        players' bring back) over the pass's `period` (patrol.filled)."""
        return filled(self.patrol.belief.rate * period)

    def _stay_gain(self, s: Scene, held: list[Monster]) -> float:
        """EXP a second of staying at the stop (see the top): the `held` monsters' EXP over their kill time and the
        walk to where she gets at the nearest of them, less what touches at the stop cost her meanwhile (STAY_NET:
        Patrol.hp_rate at Patrol.hp_price); with none, the respawns expected in reach (Patrol.wait_gain)."""
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
            gain -= self.patrol.hp_price * self.patrol.hp_rate(st.platform, st.x)
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

    def _sweep_to(self, dest: tuple[int, float] | None) -> dict | None:
        """Why the sweep's way goes on to `dest`: where she'd go next as last peeked (`_instead`: Patrol.choose's why,
        the stop, its travel, the EXP expected and seen there, its gain, the stops skipped: `peek`), when that's the
        place; else the hunt loop's (the objective, the spot)."""
        a = getattr(self, "_alt", None)
        if dest is not None and a is not None and a[2] is not None:
            stop = self.patrol.stops[a[2]]
            if (stop.platform, stop.x) == dest:
                return {"peek": {k: v for k, v in a[3].items() if k != "x"}, "heading": a[1]}
        return super()._sweep_to(dest)

    def _sweep_terms(self, s: Scene, dest: tuple[int, float] | None) -> tuple[float | None, float]:
        """The sweep's terms: its piles' allowances alone (Hunt._allow), no budget, no floor (no lulls: she doesn't wait
        for respawns at a stop). Until 2026-11-01 her time's worth (`_time_worth`: what staying at the stop gives on a
        round trip back to it, else the loop's rate, or what's in reach) with PILE_HP's price of the HP her detours
        cost, at MESO_EXP an meso."""
        return None, 0.0

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
        idle = getattr(self, "_idle_since", None)
        if held and now - v.arrived >= STOP_MAX_S and idle is not None and now - idle >= STOP_MAX_S:
            names = ", ".join(f"a {m.kind or 'monster'} on {m.platform} at x {m.x:.0f}" for m in held[:3])
            telemetry.emit("event", kind="backstop", text=f"STOP_MAX_S: {STOP_MAX_S:.0f} s waiting at stop {v.ground} "
                                                          f"for {names}")
            return f"{STOP_MAX_S:.0f} s waiting here for {names}, neither shot nor stepped to"
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
        loop's next stop when the pile no longer pays (Patrol.choose's "back to the loop"): taken at the next pass. With
        the whole screen (Targeting.screen): a trip to her target's place stops when the choice has something else for her,
        or for a pile that goes first; a trip to a stop, for any monster on the screen."""
        o = self.obj
        if getattr(o, "chain", None) is not None:
            return self._chain_stops(s, o)
        if self.tp is not None and self.tp.screen and o is not None and o.kind == "spot":
            if o.why == TO_TARGET:
                wanted = [m for m in s.monsters if self.kinds is None or m.kind in self.kinds]
                return s is not self.tp.chose_on and (
                    self.tp.changed(s) or (not self.tp.piles_wait() and self._sweep_calls(s))
                    or ((pile := self._valuable_pile(s)) is not None and self._loot_first(s, pile, wanted, say=False)))
            if self.tp.on_screen(s):
                return True
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
        """The hunt loop's, on its way to the pile `_interrupt` saw (or back to the loop) if it did. With the whole screen, a
        trip to her target's place is decided again here, and a trip to a stop ends for any monster on the screen."""
        o = self.obj
        if self._free_trip(o) and (o.why == TO_TARGET or targets):
            self.obj = None
            self.stack.clear()
            return self._planned_step(s, targets, visible, hidden, now)
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
        if self.tp is not None:
            self.tp.forget()
        now, p = time.monotonic(), self.patrol
        p.learn_kills(self.report.throws, self.report.exp)
        why = p.replan(now)
        if why is not None:
            self._loop_at, self._loop_exp = now, self.report.exp
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
        """A stop's stay is kept (`stays`), and with kills there her handling of them (Patrol.handled: from getting there
        to her last kill); the value map isn't written (see the top)."""
        v = self.visit
        if v is None or v.arrived is None:
            return
        if v.kills:
            self.patrol.handled(max(0.0, max(v.kills) - v.arrived), len(v.kills))
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
                "credit": {str(pid): [n, round(e, 1)] for pid, (n, e) in credit}, **self._census(), **self._promise()}

    def _census(self) -> dict:
        """The census and its trips (Patrol.choose; the 'grind' records 'state' and 'end'): trips taken, her view's
        occupancy against the map's average (the gate: Belief.ratio), her kills a minute as the trips' bar (None: the
        gate shut), her handling time a kill, the newborns she saw fading in against her kills' passes, and what the
        census believes by platform (in view and out)."""
        p, b = self.patrol, self.patrol.belief
        r, bar, h = b.ratio(b.t), p.bar(b.t), p.handle_s()
        by: dict[str, float] = {}
        for q in list(b.points) + b.extra:
            by[str(q.platform)] = by.get(str(q.platform), 0.0) + b.census.get(q.id, 0.0)
        return {"trips": p.trips, "occupancy": None if r is None else round(r, 3),
                "trip_bar": None if bar is None else round(bar * 60, 2), "handle_s": None if h is None else round(h, 2),
                "births_seen": b.births_seen, "births_due": b.due_n,
                "census": {pid: round(v, 2) for pid, v in by.items() if v >= 0.01}}

    def _promise(self) -> dict:
        """The loop's promise against what it delivered so far (the 'grind' records 'state' and 'end', from 2026-11-02:
        P30 of the recordings' analysis, where the bar in use, 556 EXP a minute over the hour, was 1.27 times the 437
        delivered, and nothing recorded either): the EXP a minute delivered since the loop she's on was taken up
        (`loop_exp`, over `loop_s`) against its rate (`loop_rate`), and over the run (`delivered`, her kills' EXP
        by their kinds) against the loop's rate over her looks' time (`promised`, the bar her stays and pulls are held
        to, time-weighted); the s by hunt rule (`rule_s`: Hunt._note)."""
        now, plan = time.monotonic(), self.patrol.plan
        a_min = (lambda e, s: None if s < 1.0 else round(e * 60.0 / s, 1))
        loop_s = now - self._loop_at
        return {"loop_s": round(loop_s, 1), "loop_exp": round(self.report.exp - self._loop_exp, 1),
                "loop_rate": None if plan is None else round(plan.rate * 60.0, 1),
                "promised": a_min(self._promised, self._promised_s),
                "delivered": a_min(self.report.exp, now - self._start),
                "rule_s": {k: round(v, 1) for k, v in sorted(self.report.rule_s.items())}}

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
                 f"minute by {'stop' if HP_BY_TOUCHES else 'platform'} ({HP_LOG_S:.0f} s there at least) {d['hp']}; "
                 f"kills by platform: "
                 + ", ".join(f"{pid}: {n} ({e:.0f} EXP)" for pid, (n, e) in d["credit"].items()))
        telemetry.emit("grind", what="end", **d)
