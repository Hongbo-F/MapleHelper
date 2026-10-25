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
- At a stop (`_at_ground`): she shoots what's in reach on any platform, and what's on hers (stance closes in); she
  walks along her platform to bring monsters she remembers into reach (`_reach_step`, the rule 'reach'); and she leaves
  the moment nothing is left in reach (`_leaving`: no monster seen within HOLD_S that she can get at from her platform,
  no respawn in reach due within patrol.DUE_S), LEAVE_CONFIRM_S after, STOP_MAX_S at the most. No better spot (rule 7),
  no lull loot (rule 8), no waiting out respawns (rule 9 only for those confirmations), no rate rule.
- On the legs: the hunt loop's rule 4, shooting in passing (anything a star reaches, any platform), its finishing (4b)
  and the piles on the way (5b) and on her platform (5): kills and pick-ups between moves, without stopping to clear a
  ground. No trip back for the piles of a stop she leaves (rule 10): those on the way are picked up.
- What she knows: every look updates the belief of each spawn point (patrol.Belief) from her memory of the monsters.
- What a kill is credited to: the platform it died on (`credit`), not the ground she hunted. The value map
  (lab/grounds.json) isn't written: its EXP a visit credits a ground with its neighbours' kills (live, 48% of 23's on
  2026-10-24 died on 22 and 24), and a stop's visit is a few seconds.

Every loop taken up and every next stop is logged with its numbers ("loop: ...", "next: ..."), and sent as the
telemetry topic 'grind' (what "loop", "next", and "end" with the run's stays, pulls, skips and credit by platform).
"""

import time
from statistics import median

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from ..attack import reach
from ..attack.reach import EDGE_MARGIN, SHOT_DOWN, SHOT_UP
from ..attack.spots import Spot
from ..localization.memory import SURE
from ..localization.monsters import Monster
from ..localization.scene import Scene
from . import fight
from .hunt import Hunt, Objective, Visit
from .patrol import DUE_S, STAND_STEP, Patrol, choice_text

STOP_MAX_S = 40.0      # backstop: s at one stop, however many respawns keep coming (there's a round to go)
# s with nothing left in reach before she leaves a stop: one look's miss isn't enough (a hit monster hides under its
# damage numbers for 1-3 s: fight's hidden targets keep it a target meanwhile, and HOLD_S keeps it held)
LEAVE_CONFIRM_S = 0.3
HOLD_S = 3.5           # a monster her memory has, seen within this long (tracking.HIDE_S), holds her at a stop
REACH_STEP = 200.0     # units she walks along her platform at once at most to bring held monsters into reach
HP_GAP_S = 2.0         # s between two looks at most for the HP gone between them to be charged to a platform
HP_LOG_S = 10.0        # s on a platform at least for the log to say what HP she lost there
STATE_EVERY_S = 60.0   # s between two 'grind' records 'state' (what it did so far: a run cut short has them)


class Grind(Hunt):
    """The hunt loop grinding round the map (see the top). `patrol`: its loop and belief (None: one for the map, from
    the tour's kinds worth hunting and their EXP, and the navigator's route times)."""

    def __init__(self, tour, log=print, rng=None, sweep: bool | None = None, patrol: Patrol | None = None):
        super().__init__(tour, log=log, rng=rng, sweep=sweep)
        self.patrol = patrol or Patrol(self.model, self.kinds, tour.exp,
                                       travel=lambda a, x, b, bx: self.nav.planner.arrive(a, x, b, bx))
        self.stop: int | None = None              # the stop she's at or going to (patrol.stops)
        self.credit: dict[int, list] = {}         # platform -> [kills, EXP] of her kills that died there
        self.stays: list[float] = []              # s at each stop she got to
        self.pulls = self.skips = 0
        self._start = time.monotonic()
        self._hp: tuple[float, float, int] | None = None  # the last look's (time, HP, platform), for learn_hp
        self._stated = time.monotonic()

    def run(self, minutes: float):
        report = super().run(minutes)
        self._log_grind()
        return report

    # --- what she knows ------------------------------------------------------------------------------------

    def _look(self) -> Scene:
        s = super()._look()
        if s.where is not None:
            self.patrol.belief.update(s.t, self.h.memory, s.players if s.dot is not None else None)
        self._learn_hp(s)
        return s

    def _learn_hp(self, s: Scene) -> None:
        """Her HP bar between two looks (HP_GAP_S apart at most), what went charged to the platform she was on
        (Patrol.learn_hp): a rise (a potion, the chair) counts as none lost; a reading under 2% isn't taken (the bar
        flashes)."""
        if s.platform is None or s.hp is None or s.hp < 0.02:
            return
        last, self._hp = self._hp, (s.t, s.hp, s.platform.id)
        if last is not None and 0 < s.t - last[0] <= HP_GAP_S:
            self.patrol.learn_hp(last[2], max(0.0, last[1] - s.hp), s.t - last[0])

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
        return targets, visible, hidden

    # --- at a stop ---------------------------------------------------------------------------------------------

    def _at_ground(self, s: Scene, targets, visible, hidden, now: float) -> None:
        here = s.platform.id
        if (pile := self._urgent_pile(s)) is not None:
            return self._go_for(pile, "urgent pile", s)
        if any(self._hits(s, m) or m.platform == here for m in targets):
            act = self._stance(s, targets, visible, hidden, approach=True, log=False)
            if act[0] != "wait":
                return self._shoot(s, targets, visible, hidden, now, "shoot", approach=True)
        if (pile := self._pile_here(s)) is not None:
            return self._pick_here(s, pile)
        wide = self._wide(s)
        held = self._held(wide)
        if held and (step := self._reach_step(wide, held)) is not None:
            self._note("reach")
            self.h.actions.walk(*step)
            self._idle_since = None
            return None
        reason = self._leaving(wide)
        if reason is None:
            return self._wait(wide, targets)
        return self._leave(wide, reason)

    def _held(self, s: Scene) -> list[Monster]:
        """The wanted monsters her memory has (memory.SURE), seen within HOLD_S, that she can get at from her platform:
        on it, or in a star's reach from somewhere along it."""
        here, out = s.platform, []
        for e in self.h.memory.alive(s.t, SURE):
            if (self.kinds is not None and e.kind not in self.kinds) or s.t - e.seen > HOLD_S:
                continue
            if e.platform == here.id or any(True for _ in self._reaching(here, [e.monster])):
                out.append(e.monster)
        return out

    def _reaching(self, here, monsters: list[Monster]):
        """The places along platform `here` (every STAND_STEP, EDGE_MARGIN from its ends) from which a star reaches one
        of `monsters` at least, as (x, how many)."""
        near = [m for m in monsters if -SHOT_UP - 30 <= m.y - here.y_at(min(max(m.x, here.left), here.right))
                <= SHOT_DOWN + 30]
        if not near:
            return
        lo, hi = here.left + EDGE_MARGIN, here.right - EDGE_MARGIN
        xs = [(lo + hi) / 2] if hi < lo else [lo + i * STAND_STEP for i in range(int((hi - lo) // STAND_STEP) + 1)]
        for x in xs:
            n = sum(reach.can_hit(here, x, here.y_at(x), m) for m in near)
            if n:
                yield x, n

    def _reach_step(self, s: Scene, held: list[Monster]) -> tuple[str, float] | None:
        """A walk along her platform (which way, how far: REACH_STEP at most) to where a star reaches more of the `held`
        monsters on other platforms than from where she stands (the most, then the nearest); None if none does."""
        here, x = s.platform, s.where.x
        off = [m for m in held if m.platform != here.id]
        now = sum(self._hits(s, m) for m in off)
        best = None
        for g, n in self._reaching(here, off):
            if n > now and (best is None or (n, -abs(g - x)) > (best[0], -abs(best[1] - x))):
                best = (n, g)
        if best is None or abs(best[1] - x) < 10:
            return None
        return (RIGHT if best[1] > x else LEFT), min(abs(best[1] - x), REACH_STEP)

    def _leaving(self, s: Scene) -> str | None:
        """Why she leaves the stop now, or None to stay (see the top)."""
        v, now = self.visit, time.monotonic()
        if v.arrived is None:
            return "not at the stop"
        if s.platform.id != v.ground:
            return None if self._held(s) else "off the stop, nothing in reach"
        if now - v.arrived >= STOP_MAX_S:
            telemetry.emit("event", kind="backstop", text=f"STOP_MAX_S: {STOP_MAX_S:.0f} s at stop {v.ground}")
            return f"{STOP_MAX_S:.0f} s here"
        due = self.patrol.due(self.stop, now)
        if self._held(s) or (due is not None and due <= DUE_S):
            v.empty_since = None
            return None
        if v.empty_since is None:
            v.empty_since = now
        if now - v.empty_since < LEAVE_CONFIRM_S:
            return None
        return "nothing left in reach"

    def _leave(self, s: Scene, reason: str) -> None:
        v = self.visit
        there = 0.0 if v.arrived is None else time.monotonic() - v.arrived
        self.log(f"  leaving stop {v.ground}: {reason} ({there:.1f} s there, {v.exp:.0f} EXP)")
        self._next_ground(s)
        return self._travel("advance")

    # --- the next stop -----------------------------------------------------------------------------------------

    def _next_ground(self, s: Scene) -> None:
        """The next stop (Patrol.choose), the loop looked at again first (Patrol.replan), and a trip to it."""
        self._leave_ground()
        self._exit_piles = self._exit_reason = None
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
                           current=self.stop, skip=skip)
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

    def _rest(self, why: str) -> None:
        """A rest (the hunt loop's), then the next stop from where she rested, not back to the one she left."""
        super()._rest(why)
        if self.visit is not None:
            self.stop = None
            self._next_ground(self.nav.mover.settle())

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
                "refill_scale": round(b.scale, 3), "refills": len(b.refills), "kill_s": round(p.kill_s, 4),
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
                 f"loop, {d['skips']} stops skipped; refills x{d['refill_scale']:.2f} of the model ({d['refills']} "
                 f"watched), a kill {d['kill_s']:.3f} s a point of EXP, others take {d['takes']}; HP bars a minute by "
                 f"platform ({HP_LOG_S:.0f} s there at least) {d['hp']}; kills by platform: "
                 + ", ".join(f"{pid}: {n} ({e:.0f} EXP)" for pid, (n, e) in d["credit"].items()))
        telemetry.emit("grind", what="end", **d)
