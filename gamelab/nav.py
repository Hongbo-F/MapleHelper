"""Navigation: plan a route over the map (route.Planner), carry it out like a player (flow.Flow:
several moves in one continuous key plan), and replan from wherever she actually ends up.

  look (dot + screen + monsters)  ->  plan (route.Planner)  ->  chain the moves (flow.Flow)
        ^                                                              |
        +--------------- check where she landed  <-  one key plan for the whole chain

One look per chain instead of several per move: she keeps running through landings, jumps on the way
and grabs ropes on the run. Moves the chain can't predict (climbing down a rope, a run-up she'd have
to back off for) are done on their own, closed loop (moves.Mover).

Monsters, without fighting (the default): the look before each chain covers the whole screen. A
chain stops before a leg a monster could walk into (flow.danger); that move then costs extra in the
route planner so a way around is taken if there is one, else she waits for the monster to move off
(a few looks), and then goes anyway. A hit during a move shows as lost HP: she replans from wherever
she was pushed to, without counting it as a missed move.

With `fight` on (hunting tours), monsters in reach are shot dead before each move instead, and any
in the way of a move along her platform are shot first.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from . import shots
from .flow import MONSTER_HALF, MONSTER_HEIGHT, MONSTER_SPEED, Body, Flow, Leg, Monster, Span
from .hunt import TOO_CLOSE, Hunter, Sight, pick_target
from .mapdata import MapModel
from .moves import Fix, Mover, describe
from .route import Move, Planner

GUARD_SECONDS = 15  # longest spent shooting a monster out of the way
ON_WAY_TRIES = 5    # monsters engaged per platform while passing through (respawns could keep her forever)
AVOID_SECONDS = 6.0  # extra route cost of a move a monster is in the way of (so a way around wins)
WAIT_LOOKS = 6       # looks spent waiting for a monster to move off before going anyway
LOOK_AHEAD = 0.9     # s before a chain ends that the look for the next one is taken (it takes ~0.35 s)


@dataclass
class Trip:
    result: str = ""
    moves: list[str] = field(default_factory=list)
    misses: int = 0       # moves that didn't land where planned (not counting hits)
    hits: int = 0         # moves during which a monster hit her
    kills: int = 0        # monsters shot dead on the way
    chains: int = 0       # key plans that carried out several moves in one go
    waits: int = 0        # looks spent waiting for a monster to move off
    time: dict = field(default_factory=lambda: {"look": 0.0, "think": 0.0, "move": 0.0})  # s spent


class Navigator:
    def __init__(self, hunter: Hunter, model: MapModel, log=print, fight: bool = False):
        self.h, self.model, self.log = hunter, model, log
        self.loc = hunter.localizer
        self.planner = Planner(model)
        self.mover = Mover(hunter, model, self.loc, log)
        self.flow = Flow(model, {k: getattr(hunter.keys, k).name for k in ("left", "right", "jump", "up", "down")})
        self.fight = fight          # shoot monsters on the way (hunting tours) instead of avoiding them
        self.hunt_on_way = True     # with `fight`: shoot monsters in reach between moves (off when going to rest)
        self._seen: list[tuple[float, Monster]] = []  # last look's monsters, to tell which way they walk
        self.speeds: dict[str, list[float]] = {}  # kind -> walking speeds seen between looks (units/s)
        self._off = 0  # glances in a row on another platform than planned
        self._last_look: tuple[float, list[Monster]] = (0.0, [])  # when, and the monsters seen

    def go(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6,
           until: Callable[[Sight], bool] | None = None) -> Trip:
        """Get onto platform `dst` (and to map x on it, if given). `until` is asked before every chain,
        with a look that includes monsters: True ends the trip early ('in position')."""
        trip, deadline = Trip(), time.monotonic() + seconds
        fights = 0  # in a row, without a move between: after a few she moves on regardless
        tries: dict[int, int] = {}  # platform -> monsters engaged there on the way
        avoid: dict[tuple, float] = {}  # (kind, src, dst, x) of moves a monster was in the way of -> when
        waited = 0
        moved_at = 0.0  # when the last plan ended (she may still be sliding)
        quick = False   # just cut short: replan from a fast look with the monsters already seen
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            f = self.mover.fix() if quick else self.mover.fix(mobs=True, wide=not self.fight)
            t1 = time.monotonic()
            trip.time["look"] += t1 - t0
            if stop := self.h.health_check(f.sight):
                trip.result = stop
                return trip
            if until is not None and f.where.platform is not None and until(f.sight):
                trip.result = "in position"
                return trip
            here = f.where.platform
            if here is None:
                f = self.mover.settle(2)
                if f.where.platform is None:  # hanging on a rope
                    self.log(f"  on a rope at x {f.where.x:.0f}: getting off")
                    f = self.mover.off_rope(f)
                here = f.where.platform
                if here is None:
                    trip.result = "lost: not on any platform"
                    return trip
            if self.fight and self.hunt_on_way and tries.get(here.id, 0) < ON_WAY_TRIES and (t := self.in_reach(f)):
                tries[here.id] = tries.get(here.id, 0) + 1
                self.log(f"  {t.kind} in reach on platform {t.platform}: shooting it on the way")
                if self.h.kill(t, seconds=GUARD_SECONDS) == "killed":
                    trip.kills += 1
                continue
            if here.id == dst:
                if x is not None:
                    self.mover.walk_to(x, tolerance=20, f=f)
                trip.result = "arrived"
                return trip
            now = time.monotonic()
            avoid = {key: t for key, t in avoid.items() if now - t < 4}
            path = self.planner.plan(here.id, f.where.x, dst,
                                     penalty=lambda m: AVOID_SECONDS if _key(m) in avoid else 0.0)
            if not path:
                trip.result = f"no route from platform {here.id} to {dst}"
                return trip
            if self.fight:
                if fights < 3 and self.clear_way(f, path[0]):
                    fights += 1
                    continue  # look again after the fight
                fights = 0
            if self.fight:
                monsters = []
            elif quick:  # the last look's monsters, moved on by the time since
                looked, seen_then = self._last_look
                monsters = [replace(mo, seen=looked - time.monotonic()) for mo in seen_then]
            else:
                monsters = self.monsters(f.sight, seen=f.sight.t - time.monotonic())
            quick = False
            # Right after a plan she may still be sliding or landing: the first jump/drop waits.
            settle = max(0.0, 0.35 - (time.monotonic() - moved_at))
            legs, keys, why = self.flow.chain(path, Body(0.0, f.where.x, here, 0, settle > 0, settle), monsters,
                                              keep_going=not self.fight, hops=not self.fight)
            trip.time["think"] += time.monotonic() - t1
            # Waiting only makes sense where no monster can reach her meanwhile.
            safe_here = not self.flow.danger(
                Leg(path[0], Body(0.0, f.where.x, here), Body(1.0, f.where.x, here),
                    [Span(here.id, 0.0, 1.0, f.where.x, f.where.x)]), monsters)
            if not legs and why == "monster" and waited < WAIT_LOOKS and safe_here:
                if _key(path[0]) not in avoid:
                    avoid[_key(path[0])] = now
                    self.log(f"  a monster could get in the way of {path[0].kind} -> {path[0].dst}: looking for a way around")
                    continue
                waited += 1
                trip.waits += 1
                self.log(f"  waiting for the monster to move off ({waited})")
                time.sleep(0.3)
                continue
            waited = 0
            hp = f.sight.hp
            if legs:
                self.log(f"  {describe(here)} at x {f.where.x:.0f} ({f.how}): " + _chain_text(legs, len(path), why))
                t2 = time.monotonic()
                cut, legs = self.follow(legs, keys, hp, dst, why, trip)
                trip.time["move"] += time.monotonic() - t2
                moved_at = time.monotonic()
                self.h.facing = None
                if cut:
                    after = self.recover()
                    quick = True
                    self.log(f"    {cut}: plan cut short; on {describe(after.where.platform)} at x "
                             f"{after.where.x:.0f} after {time.monotonic() - moved_at:.2f} s: replanning from there")
                else:
                    after = self.mover.fix()
                    if after.where.platform is None:
                        after = self.mover.settle(1.5)
                planned = legs[-1].end.platform
                moved = [leg.move for leg in legs]
            else:
                move = path[0]
                self.log(f"  {describe(here)} at x {f.where.x:.0f} ({f.how}): {move.kind} {move.direction} -> "
                         f"platform {move.dst}  [{len(path)} moves to go]")
                t2 = time.monotonic()
                after = self.mover.do(move, f)
                trip.time["move"] += time.monotonic() - t2
                moved_at = time.monotonic()
                planned = self.model.platform(move.dst)
                moved = [move]
            landed = after.where.platform
            hit = after.sight.hp < hp - 0.004
            trip.moves += [f"{m.kind} {m.src}->{m.dst}" for m in moved]
            trip.moves[-1] += f": landed on {landed.id if landed else None}" + (" (hit)" if hit else "")
            if hit:
                trip.hits += 1
            if landed is None or landed.id != planned.id:
                if hit:
                    self.log(f"    hit on the way ({hp:.0%} -> {after.sight.hp:.0%}): on {describe(landed)}; replanning")
                    continue
                trip.misses += 1
                self.log(f"    ended on {describe(landed)} instead of platform {planned.id}; replanning")
                if trip.misses >= max_misses:
                    trip.result = "too many misses"
                    return trip
        trip.result = "timeout"
        return trip

    def follow(self, legs, keys, hp: float, dst: int, why: str, trip: Trip) -> tuple[str, list]:
        """Run a chain's plan and keep going like a player: while it runs, look around once and plan
        the next chain from where this one will end, queued so the player starts it the moment this
        one finishes (no stop between). Meanwhile her dot and HP are watched ~15 times a second: a
        hit, or her being on another platform than planned for a few glances, cuts the running plan
        and the queued one short (the player lets go of every key). Returns why it was cut ('' if
        not) and all the legs sent."""
        from . import client
        from .flow import _ms
        keep = why == "long" and not self.fight

        def send(chain_legs, chain_keys, more: bool) -> str:
            trip.chains += 1
            # The plan lasts until she's predicted to land (or the next chain starts), so the look
            # after it doesn't catch her still falling.
            last = max(e[0] for e in chain_keys.sorted())
            return client.send(chain_keys.sorted(), lead_ms=0, focus=True, name="flow",
                               end_ms=max(last, _ms(chain_legs[-1].end.t)))

        current = (send(legs, keys, keep), legs, time.monotonic() + 0.015, keep)
        queued, sent, cut = None, list(legs), ""
        last_dot = None
        self.h._pressed = True
        while True:
            plan_id, run_legs, start, more = current
            # Look for the next chain late in this one: from where she starts, where she's going can
            # be off screen (from platform 34 the basin's slimes were under the status bar).
            ends = start + run_legs[-1].end.t
            if more and queued is None and ends - time.monotonic() <= LOOK_AHEAD:
                queued = self.plan_ahead(run_legs[-1].end, start + run_legs[-1].end.t, dst, send)
                if queued is None:
                    current = (plan_id, run_legs, start, False)  # nothing to follow on with
                else:
                    sent += queued[1]
            time.sleep(0.03)
            dot, now_hp = self.h.eyes.glance()
            if now_hp < hp - 0.004:
                cut = f"hit ({hp:.0%} -> {now_hp:.0%})"
            elif dot is not None:
                t = time.monotonic() - start
                want = self.flow.expected(run_legs, t)
                still = last_dot is not None and abs(dot[0] - last_dot[0]) < 0.6 and abs(dot[1] - last_dot[1]) < 0.6
                # On a platform only when the dot sits at its height and has stopped: in the air the
                # dot passes 40-60 units under the platforms above, which read as "on" them before.
                mx, my = self.loc.to_map(dot)
                on = self.model.platform_under(mx, my - 6, reach=16) if still else None
                leg = next((lg for lg in run_legs if lg.start.t <= t <= lg.end.t + 0.3), run_legs[-1])
                if on is not None and on.id not in (leg.start.platform.id, leg.end.platform.id):
                    # Standing where this leg doesn't go (e.g. a drop that fell one platform too far):
                    # stop before the plan's next keys fire there.
                    self._off += 2
                elif on is not None and want is not None and on.id != want:
                    self._off += 1
                else:
                    self._off = 0
                if self._off >= 3:
                    cut = f"off the path (on {on.id if on else 'nothing'}, planned {want or leg.end.platform.id})"
            last_dot = dot
            if cut:
                for pid in [plan_id] + ([queued[0]] if queued else []):
                    if not client.cancel(pid):
                        client.wait(pid)
                if queued:
                    sent = sent[:len(sent) - len(queued[1])]
                break
            if client.done(plan_id):
                client.wait(plan_id)
                if queued is None:
                    break
                current, queued = queued, None
        self._off = 0
        return cut, sent

    def recover(self, longest: float = 0.8) -> Fix:
        """Right after a plan was cut short: wait until she's stopped moving (a hit knocks her back
        and she can't be steered for a moment), glancing at her dot, then one look to place her."""
        end, last = time.monotonic() + longest, None
        while time.monotonic() < end:
            dot, _ = self.h.eyes.glance()
            if dot is not None and last is not None and abs(dot[0] - last[0]) < 0.6 and abs(dot[1] - last[1]) < 0.6:
                w = self.loc.where_dot(dot)
                if w.platform is not None:
                    break
            last = dot
            time.sleep(0.03)
        return self.mover.fix()

    def plan_ahead(self, end, starts_at: float, dst: int, send):
        """Look around (she's on the move) and send the chain that follows on from body `end`, which
        she'll be in when the running plan finishes at `starts_at`. None if there's nothing to add
        (she'll be there, or the next move can't be chained)."""
        if end.platform.id == dst:
            return None
        s = self.h.look(mobs=True, wide=True)
        monsters = self.monsters(s, seen=s.t - starts_at)
        path = self.planner.plan(end.platform.id, end.x, dst)
        if not path:
            return None
        start = Body(0.0, end.x, end.platform, end.run, end.landed, end.slack)
        legs, keys, why = self.flow.chain(path, start, monsters, keep_going=True)
        if not legs:
            return None
        more = why == "long"
        self.log(f"    then from platform {end.platform.id} at x {end.x:.0f} (planned on the move): "
                 + _chain_text(legs, len(path), why))
        return send(legs, keys, more), legs, starts_at, more

    def monsters(self, s: Sight, seen: float = 0.0) -> list[Monster]:
        """The monsters in this look, with which way (and how fast) each was walking since the last
        look. `seen`: when the look was taken, in the time of the plan they're for."""
        now, out = s.t, []
        before = [(t, mo) for t, mo in self._seen if now - t < 2.0]
        for t in s.targets:
            if t.platform is None:
                continue
            heading, vx = 0, None
            prev = [(when, mo) for when, mo in before if mo.platform == t.platform and abs(mo.x - t.x) < 90]
            if prev:
                when, mo = min(prev, key=lambda wm: abs(wm[1].x - t.x))
                dx = t.x - mo.x
                heading = (1 if dx > 0 else -1) if abs(dx) > 6 else 0
                if now - when > 0.2:  # how fast monsters really walk, for MONSTER_SPEED
                    self.speeds.setdefault(t.kind, []).append(abs(dx) / (now - when))
                    top = MONSTER_SPEED.get(t.kind, 70.0)
                    vx = 0.0 if abs(dx) <= 6 else max(-top, min(top, dx / (now - when)))
            out.append(Monster(t.platform, t.x, MONSTER_SPEED.get(t.kind, 70.0), heading,
                               MONSTER_HALF.get(t.kind, 34.0), vx, seen, MONSTER_HEIGHT.get(t.kind, 46.0)))
        self._seen = [(now, mo) for mo in out]
        self._last_look = (now, out)
        return out

    def in_reach(self, f: Fix):
        """The monster to shoot from where she stands, if any is in reach (pick_target's choice)."""
        s, here = f.sight, f.where.platform
        reach = [t for t in s.targets if shots.can_shoot(t.x - f.where.x, t.y - f.where.y)]
        return pick_target(f.where.x, f.where.y, here, reach) if reach else None

    def clear_way(self, f: Fix, move: Move) -> bool:
        """Shoot a monster standing on her platform between her and where the move starts (or too
        close to her). Returns True if she fought."""
        s = f.sight
        lo, hi = sorted((f.where.x, move.x))
        here = f.where.platform
        in_way = [t for t in s.targets if t.platform == here.id
                  and lo - TOO_CLOSE <= t.x <= hi + TOO_CLOSE]
        if not in_way:
            return False
        t = min(in_way, key=lambda t: abs(t.x - f.where.x))
        self.log(f"  {t.kind} in the way at x {t.x:.0f}: shooting it first")
        self.h.kill(t, seconds=GUARD_SECONDS)
        return True


def _key(m: Move) -> tuple:
    return (m.kind, m.src, m.dst, round(m.x))


def _chain_text(legs, to_go: int, why: str) -> str:
    return (", ".join(f"{leg.move.kind}->{leg.move.dst}" if leg.move.kind != "hop" else f"hop@{leg.move.x:.0f}"
                      for leg in legs) + f"  [{to_go} moves to go; chain ended: {why}]")
