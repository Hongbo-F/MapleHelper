"""The target-first planner at a grind's stop (`Targeting`): what perception says at each look turned into attack.target's
Her and Foes, its choice narrowing the hunt's targets to the one monster to hit, and the plan against it: the stance's
own answer for that monster (attack.stance: up close, a jump attack, a step to range; a jump's shot), or the walk to the
nearest place along her platform that works (attack.target.places: its way checked once, flow.danger, by the navigator). On
by default: the user's call of 2026-11-04, its second gate not passed (GAMELAB_TARGET=0 turns it off: `on`). Off, none of this
runs, and the grind is the one before it to the digit. README: "The target-first planner".

Everything else about a stop stays the grind's: where she goes next (Grind._leaving, Patrol.choose), her potions and the
safety stop, the Lie Detector's stop. Her drops come in as they're due (`pile_first`, `loot_allow`: LOOT_ALL ...). The choice
is made at each pass's look, and at each frame while her lock is waited out her plan is read against that frame's tracks and
newborns (`watch_lock`, a glance at the feed's newest state: nothing can be keyed meanwhile, and a look there would be the
wrong one to decide on: her own spell's effect is over the body just then, and "no effect" judged at 0.6 s on the dot holds a
whole side: arms e3, e4). A newborn is in the choice from the first look that has it (Mob.fading, its track's solid_at:
`foes_of`), and in a glance from the feed's events not drained yet (`pending_events`).

The whole screen (`screen_on`, GAMELAB_SCREEN: '0' off, the planner before it to the digit; the user, 2026-11-04): every monster on
the screen is a candidate (`foes_of`: Scene.swept, the look's missed by presence's chance), its way across platforms by the
navigator's moves and its score its place's (attack.target: see its top), the trip itself the navigator's (`Grind._to_place`). With
none on the screen her drops whatever they take (`quiet`), then another area. README: "The target-first planner".
"""

import contextlib
import os
import time
from collections import Counter
from dataclasses import replace
from time import perf_counter_ns

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from .attacks import DEAD_LIKELY
from .ledger import EXPIRE_S, SETTLE
from ..attack import target as tg
from ..attack.stance import MIN_MOVE, _sizes, _stand_limits, open_above
from ..localization.monsters import Monster, speeds
from ..localization.rate import HZ
from ..perception.spawns import FADE_PER_S
from . import fight

FRAME_S = 1.0 / HZ    # s between two frames perception looks at: `watch_lock`'s step
HELD_P = 0.25         # (her casts) a monster her cast won't go at for now (a hold: rule 3): this much less likely to be hit
# Her drops (the user, 2026-11-04: "you should aim for 100% pick up rate"): every pile she may take is gone for, whether a kill
# of hers explains it or the store only found it (Hunt._candidate: a kill it couldn't place, a drop past LOOT_REACH of its
# kill: of her lost drops the 15-minute runs had most were those): on her platform at once whatever it adds to her way (none is left behind as she leaves it), on another LOOT_FRESH s on her way, whatever it
# takes once LOOT_DUE_S are left of its life (ledger.EXPIRE_S, the game's). A far pile waits for her way to come near it, and
# goes before its life ends.
LOOT_ALL = 1e6        # s: no bound
LOOT_FRESH = 3.0      # s (hunt.ALLOW_FRESH's): what far piles may cost her, the user's to choose (README: what it costs her)
LOOT_DUE_S = 30.0     # s of a pile's life left, its way there and its stand aside, from which it comes before the primary
ASKS = 4              # targets the plan is asked for at most at a pass (the rest wait for the next look)
NO_WAY_S = 0.5        # s a target the plan has nothing against for now (the stance's answer a wait, no place to walk to) is
                      # set aside: another's taken at the next look, and she never idles on one
IDLE_S = 1.5          # s with nothing she can act on (a wait, no pile) after which she goes on to the next stop
ROUTE_STEP = 8.0      # units her x is rounded to for a route's time: 53 ms of her walk at most, and the searches kept (Planner.times)
HIST_MS = 0.05        # ms a bin of the choice's time (`Targeting.ms`: for the first live runs) ...
HIST_N = 800          # ... up to 40 ms, the last bin all above

_FORCED: list[bool | None] = [None]    # (`forced_on`: a set-piece's)


def on() -> bool:
    """Whether the grind's fights go by the target-first planner (GAMELAB_TARGET: '0' off, the grind before it to the digit;
    on by default, the user's call of 2026-11-04, its second gate not passed: README)."""
    return _FORCED[0] if _FORCED[0] is not None else os.environ.get("GAMELAB_TARGET", "1") != "0"


def screen_on() -> bool:
    """Whether the planner takes every monster on the screen (GAMELAB_SCREEN: '0' off, the planner before it to the digit; on by
    default on this branch: README)."""
    return os.environ.get("GAMELAB_SCREEN", "1") != "0"


@contextlib.contextmanager
def forced_on():
    """On within the block, whatever GAMELAB_TARGET says (a set-piece's: sim.scenarios.loop_tree_rope_top_chaser), off with
    GAMELAB_TARGET_SETPIECE=0 (master's arm, measured)."""
    was, _FORCED[0] = _FORCED[0], os.environ.get("GAMELAB_TARGET_SETPIECE", "1") == "1"
    try:
        yield
    finally:
        _FORCED[0] = was


def _hist() -> list[int]:
    return [0] * (HIST_N + 1)


def _bin(hist: list[int], ns: int) -> None:
    hist[min(HIST_N, int(ns / 1e6 / HIST_MS))] += 1


def _act_text(act: tuple) -> str:
    if act[0] in ("back", "approach"):
        return f"{'step' if act[0] == 'back' else 'walk'} {act[1]} {act[2]:.0f}"
    if act[0] == "jump_throw":
        return f"jump attack {act[1]}"
    if act[0] == "jump_back":
        return f"jump back, attack {act[1]}"
    if act[0] == "trip":
        return f"trip to {act[1]} at x {act[2]:.0f}"
    return f"{act[0]} {act[1]}" if len(act) > 1 else act[0]


def _pile_text(t: tg.Terms) -> str:
    """A place's pile for the log: 'pile of 3 on 23 at x -60: 3.1 s away'; '' with none."""
    w = t.way
    if not w.pile:
        return ""
    where = "here" if w.place is None else (f"at x {w.place:.0f}" if w.platform is None else f"on {w.platform} at x {w.place:.0f}")
    return f", pile of {len(w.pile) + 1} {where}: {w.s:.1f} s away"


def _next_text(n: tg.Terms | None, m: Monster) -> str:
    """The next for the log: 'next horny-mushroom #5', and where she goes for it unless she can hit it from where she'll be:
    'next: walk to 18 x 40 as #12 dies'."""
    if n is None:
        return ""
    w = n.way
    if w.how == "trip":
        return f", next: {n.m.kind} #{n.m.track}, trip to {w.platform} x {w.place:.0f} as #{m.track} dies"
    if w.how == "walk" and w.place is not None:
        return f", next: {n.m.kind} #{n.m.track}, walk to x {w.place:.0f} as #{m.track} dies"
    return f", next {n.m.kind} #{n.m.track}"


class Targeting:
    """The planner's state in a grind (`g`: Grind, its hunter `g.h`): the chooser, the last look's choice, what it's said."""

    def __init__(self, g):
        self.g, self.h = g, g.h
        self.chooser = tg.Chooser()
        self.choice: tg.Choice | None = None
        self.her: tg.Her | None = None
        self.foes: list[tg.Foe] = []
        self._said = None
        self._walking = speeds(g.model)
        self.no_way: dict[int, float] = {}  # a target's track -> until when it's set aside (`plan`)
        self.glanced: set[int] = set()      # newborns the glances have taken in (`_changes`)
        self.asked: set[int] = set()        # those it called her attack off for, since her last key (`_changes`) ...
        self._asked_at = 0.0                # ... that key's time
        self.here_only = False              # (`Grind._here`: the sweep asked for what's on her platform or due alone)
        self._chose_at = 0.0                # when the last choice was made
        self.counts: Counter = Counter()
        self.ms = {"choice": _hist(), "glance": _hist(), "peek": _hist()}
        self.screen = screen_on()
        self.quiet = False                  # (`Grind._planned_step`: no monster on the screen, her drops whatever they take)
        self._stands = tg.in_height(g.model.places) if self.screen else {}

    # --- while her lock lasts --------------------------------------------------------------------------------

    def watch_lock(self) -> bool:
        """Her lock waited out before an attack's key (Actions.lock_watch), each frame the feed has looked at in that time
        read for what changes her plan (`_changes`, a glance as Hunter._still's: nothing asked of the feed, no look: a look's
        the first after a key, as in the hunt loop): False, the attack called off, the next pass deciding at once."""
        h = self.h
        actions, senses, feed = h.actions, h.senses, h.senses.feed
        seen = -1e18
        while (left := actions.lock_until - time.monotonic()) > 0.0:
            st = feed.newest(max(senses.fresh_after(), seen + 1e-6))
            if st is not None:
                seen = st.t
                t0 = perf_counter_ns()
                off = self._changes(st, feed.pending_events(st.t), left)
                _bin(self.ms["glance"], perf_counter_ns() - t0)
                if off:
                    return False
            time.sleep(min(left, FRAME_S))
        return True

    def _changes(self, st, events, left: float) -> bool:
        """Whether frame state `st` (its tracks, the feed's `events` not drained yet) has a newborn that beats her target or
        a monster coming at her that isn't her target: the choice's own rules (`Chooser`), asked of just those."""
        c, her = self.choice, self.her
        if c is None or c.primary is None or her is None:
            return False
        fresh = []
        for e in events:
            if e.kind == "spawned" and e.track is not None and e.track not in self.glanced:
                self.glanced.add(e.track)
                fresh.append(e.track)
        foes = []
        primary = c.primary.m
        now = time.monotonic()
        her2 = replace(her, lead=max(0.0, left), invulnerable=max(0.0, self._inv_until() - now))
        for tr in st.tracks:
            if tr.state != "live" or tr.id == primary.track or tr.id in self.no_way or tr.platform is None:
                continue
            m = Monster(tr.mob, tr.x, tr.y, tr.platform, bar=tr.bar, facing=tr.facing, pose=tr.pose, track=tr.id)
            if tr.id in fresh or (tr.platform == her.platform.id or (her.danger is not None and tg.beside(m, her.platform, her.x, her.danger))):
                foes.append(self._glance_foe(m, tr, st.t, tr.id in fresh))
        if not foes:
            return False
        key = getattr(self.h.actions, "attack_at", 0.0)
        if key != self._asked_at:  # (a monster the choice stayed away from isn't asked about again until her next key)
            self._asked_at = key
            self.asked.clear()
        finish = max(0.0, c.primary.way.s + c.primary.foe.kill_s - (now - self._chose_at))  # (way.s holds her lock)
        come = tg.coming(her2, foes, finish)
        if come is not None and come.foe.m.track not in self.asked and tg.terms_of(her2, come.foe) is not None:
            self.asked.add(come.foe.m.track)
            self.counts["lock: coming"] += 1
            self._called_off(come.foe.m, f"coming at her: {come.arrives:.1f} s")
            return True
        for f in foes:
            if f.born and f.m.track not in self.asked and (t := tg.terms_of(her2, f, far=False)) is not None \
                    and t.score > c.primary.score * self.chooser.margin:
                self.asked.add(f.m.track)
                self.counts["lock: respawn"] += 1
                self._called_off(f.m, f"a respawn, {t.score:.1f} over {c.primary.score:.1f}")
                return True
        return False

    def _called_off(self, m: Monster, why: str) -> None:
        self.h.log(f"  target: attack at #{self.choice.primary.m.track} called off: {m.kind} #{m.track} {why}")

    def _glance_foe(self, m: Monster, tr, t: float, born: bool) -> tg.Foe:
        a = tr.fading
        solid = max(0.0, tr.solid_at - t) if tr.solid_at is not None else 0.0
        hit = 0.0 if a is None else max(0.0, (tg.HIT_ALPHA - a) / FADE_PER_S)
        g = self.g
        return tg.Foe(m, g.tour.exp.get(m.kind, g._avg_exp()), self.h.attacks.kill_s(m), 1.0, tr.vx or None, False, solid, hit,
                      born or solid > 0.0)

    def _inv_until(self) -> float:
        until = getattr(self.h.senses, "invulnerable_until", None)
        return 0.0 if until is None else until()

    # --- her drops ------------------------------------------------------------------------------------------

    def loot_allow(self, p, s, now: float) -> float:
        """The s a pile may add to her way (see LOOT_ALL ...): whatever it takes near its end or on her platform, one
        no look has seen yet only while its drops may still be landing (a kill drops nothing as often as not: a look tells
        from nearer by, or the wide look past its landing); else LOOT_FRESH."""
        age = now - p.born
        if EXPIRE_S - age <= LOOT_DUE_S or self.quiet:
            return LOOT_ALL
        if s.platform is not None and p.platform == s.platform.id and (p.status == "seen" or age < SETTLE):
            return LOOT_ALL
        return 0.0 if self.here_only else LOOT_FRESH

    def pile_first(self, p, stop, now: float) -> bool:
        """Whether the sweep's first pile `p` (its `stop`) goes before the primary: never against a monster coming at her;
        when what's left of its life wouldn't see the primary killed first (its stand and LOOT_DUE_S more); and when it's
        nearer than the primary's place, the primary not in reach from where she stands. Else the primary: the drops wait."""
        c = self.choice
        t = c.primary
        if c.how == "coming" or (c.coming is not None and c.coming.foe is t.foe):
            return False
        by = stop.leave
        if EXPIRE_S - (now - p.born) <= by + t.way.s + t.foe.kill_s + LOOT_DUE_S:
            return True
        return t.way.how in ("walk", "trip") and by <= t.way.s

    # --- the choice state -------------------------------------------------------------------------------------

    def forget(self) -> None:
        """She goes off on a trip: the target isn't hers to keep on the next platform."""
        self.chooser.forget()
        self.choice = None

    # --- the choice ----------------------------------------------------------------------------------------

    def her_at(self, s, pace, visible, hidden) -> tg.Her:
        g, h = self.g, self.h
        here, x, y = s.platform, s.where.x, s.where.y
        around = visible + hidden
        lo, hi = _stand_limits(here, crowded=True)
        attacks = getattr(h, "attacks", None)
        picks = getattr(attacks, "picks", False)
        return tg.Her(here, x, y, pace.lead, pace.window, h.actions.facing, fight._air() and fight.JUMP_SHOTS, around,
                      _sizes(g._tops, g._halves) if picks else None, g._halves, g.danger, pace, lo, hi,
                      jump_hits=lambda m: g._jump_hits(s, m), can_jump=open_above(g.model, here, x, x),
                      routes=self._route if self.screen else None, stands=self._stands)

    def _route(self, src: int, x: float, dst: int, lo: float, hi: float) -> tuple[float, float] | None:
        """Her.routes: the navigator's route times (Planner.arrive_within), her x rounded (ROUTE_STEP) so its searches are kept."""
        return self.g.nav.planner.arrive_within(src, round(x / ROUTE_STEP) * ROUTE_STEP, dst, lo, hi)

    def foes_of(self, s, targets, visible, pace, more=()) -> list[tg.Foe]:
        """This look's Foes: her targets, those about her of any kind on or beside her platform (they can touch her:
        attack.target.coming), and `more` of the kinds she hunts (the rest of the screen, those the look missed: `screen_on`).
        Each with its kind's EXP, its kill's time and the chance it's there as Attacks has them, its
        walk (its track's heading and speed, else the chase her aggro belief has it on, at her at its kind's speed), its fade
        (Mob.fading, else its track's solid_at)."""
        h, g = self.h, self.g
        attacks = h.attacks
        avg = g._avg_exp()
        now = s.t
        self.no_way.update(getattr(attacks, "ignored", None) or {})  # (her attacks hit nothing: Attacks._watch_hits)
        if self.no_way:
            self.no_way = {k: v for k, v in self.no_way.items() if v > now}
        chasers = {c.track for c in h.aggro.chasers.values() if c.track is not None and not c.dying}
        tracks = {tr.id: tr for tr in (getattr(s, "tracks", None) or ())}
        picks = getattr(attacks, "picks", False)
        x, out = s.where.x, []
        here = s.platform
        wanted = list(targets)
        kinds = g.kinds
        wanted += [m for m in visible if not any(m is t for t in wanted) and (kinds is None or m.kind in kinds or (
            m.platform == here.id or (g.danger is not None and tg.beside(m, here, x, g.danger))))]
        wanted += [m for m in more if (kinds is None or m.kind in kinds) and not any(m is t for t in wanted)]
        for m in wanted:
            if m.track in self.no_way:
                continue
            p = 1.0
            if picks:
                why = attacks.refused(m)
                if why in ("unseen", "dead", "likely dead"):
                    continue
                if why is not None:
                    p *= HELD_P
            else:
                c = attacks.chance(m)
                if c is not None:
                    p *= c
            left = attacks.reckon(m)
            spent = False
            if left is not None:
                p *= max(0.0, 1.0 - left.dead)
                spent = self.screen and left.dead >= DEAD_LIKELY
            v = pace.walk(m) if pace.walk is not None else None
            chasing = m.track is not None and m.track in chasers
            if chasing and not v:
                v = (1.0 if x > m.x else -1.0) * self._walking.get(m.kind, 125.0)
            a = m.fading
            tr = tracks.get(m.track)
            solid = 0.0
            if a is not None:
                solid = (1.0 - a) / FADE_PER_S
            elif tr is not None and getattr(tr, "solid_at", None) is not None:
                solid = max(0.0, tr.solid_at - s.t)
                a = 1.0 - solid * FADE_PER_S
            hit = 0.0 if a is None else max(0.0, (tg.HIT_ALPHA - a) / FADE_PER_S)
            out.append(tg.Foe(m, g.tour.exp.get(m.kind, avg), attacks.kill_s(m), p, v, chasing, solid, hit, solid > 0.0, spent))
        return out

    @staticmethod
    def _close_first(her, foes: list) -> list:
        """Of `foes` those too close (as master's rule 1 has them first), or all with none."""
        near = [f for f in foes if tg.beside(f.m, her.platform, her.x, her.danger)
                and tg.too_close(f.m, her.x, her.halves, her.danger, her.pace, her.platform,
                                 tg._side(f.m, her.x) != her.facing)]
        return near or foes

    def decide(self, s, targets, visible, hidden, trip: bool, free: bool = False) -> list:
        """The primary at look `s` as the hunt's targets (none: nothing to hit); the choice kept for the plan. On a `trip`
        (she has an objective, and shoots in passing or keeps off what's on her, master's rules) no walk goes for a target,
        and one too close is the target, as master's rule 1 has it first. `free`: on a trip the choice can end (`screen_on`:
        any but a pile's), one too close first as on a trip, else any on the screen."""
        t0 = perf_counter_ns()
        pace = fight.pace_of(self.h, s)
        her = self.her = self.her_at(s, pace, visible, hidden)
        self.foes = self.foes_of(s, targets, visible, pace, [*self.g._rest_of_screen(s, time.monotonic()), *hidden]
                                 if self.screen else ())
        foes = self.foes
        if trip or free:
            foes = self._close_first(her, foes)
        c = self.choice = self.chooser.choose(her, foes, walks=not trip)
        self._chose_at = time.monotonic()
        self.counts["decisions"] += 1
        self.counts["decisions on a trip"] += trip
        _bin(self.ms["choice"], perf_counter_ns() - t0)
        if c.why and c.primary is not None:
            self.counts["changes"] += 1
            self.counts[c.how] += 1
        return [] if c.primary is None else [c.primary.m]

    def peek(self, s) -> tg.Choice:
        """What the choice would be at look `s`, a trip's own (Grind._interrupt), nothing kept of it."""
        t0 = perf_counter_ns()
        g, chooser = self.g, self.chooser
        pace = fight.pace_of(self.h, s)
        her = self.her_at(s, pace, s.monsters, [])
        now = time.monotonic()
        foes = self.foes_of(s, g._alive(s, s.monsters, now), s.monsters, pace, g._rest_of_screen(s, now))
        kept = chooser.primary, chooser.next
        try:
            return chooser.choose(her, self._close_first(her, foes))
        finally:
            chooser.primary, chooser.next = kept
            _bin(self.ms["peek"], perf_counter_ns() - t0)

    def changed(self, s) -> bool:
        """On a trip to her target's place: whether look `s` has something else for her than going on with it: another
        target, or this one no more a trip. A look made up from another (Scene.what_if) has nothing to say."""
        if getattr(s, "what_if", False):
            return False
        c, p = self.peek(s), self.chooser.primary
        return c.primary is None or p is None or not tg.same(p, c.primary.m) or c.primary.way.how != "trip"

    def on_screen(self, s) -> bool:
        """On her way somewhere else: whether look `s` has a monster on the screen she can get at."""
        return not getattr(s, "what_if", False) and self.peek(s).primary is not None

    # --- the plan ------------------------------------------------------------------------------------------

    def walk_step(self, s) -> tuple[str, float, float] | None:
        """The walk to the place that works for the primary (attack.target.places: the nearer whose way is clear, flow.danger,
        by the navigator; with neither clear the nearer all the same: EXP over touches, the user's, and a look after WALK_STEP
        at most): (the way, how far, the place). None: none (or she's there)."""
        c, her = self.choice, self.her
        if c is None or c.primary is None or her is None or her.platform is not s.platform:
            return None
        far = [p for p in tg.places(her, c.primary.m) if abs(p - her.x) >= MIN_MOVE]
        if not far:
            return None
        w = c.primary.way
        if self.screen and w.place is not None and w.platform is None and abs(w.place - her.x) >= MIN_MOVE:
            far = [w.place, *far]  # (the one that also reaches the next: tg.also_next)
        clear = self.g.nav.first_clear(s, her.x, far)
        pick = far[0] if clear is None else clear
        return (RIGHT if pick > her.x else LEFT), min(abs(pick - her.x), tg.WALK_STEP), pick

    def mine(self, targets) -> bool:
        """Whether `targets` are the primary alone (the stance asked is for it)."""
        c = self.choice
        return c is not None and c.primary is not None and len(targets) == 1 and targets[0] is c.primary.m

    def trip_act(self, s) -> tuple:
        """(`screen_on`) The primary out of reach of her platform: ('trip', the platform, the place), said when it changes."""
        w = self.choice.primary.way
        act = ("trip", w.platform, w.place)
        self._say(s, act)
        return act

    def plan(self, s, act: tuple, approach: bool, acting: bool = True) -> tuple:
        """The act for the primary: the stance's for it (`act`, asked with no closing in), or with nothing from where she
        stands and `approach` the walk to its place; said when it changes. Not `acting` and no `approach` (a rule only asking
        what the stance would do: Hunt._pass's close quarters): its answer alone, nothing set aside."""
        c = self.choice
        if act[0] == "wait" and approach and (step := self.walk_step(s)) is not None:
            act = ("approach", step[0], step[1])
        if act[0] == "wait" and not approach and not acting:
            return act
        if act[0] == "wait":  # (nothing against it for now: set aside, another taken at once: Grind._targets)
            m = c.primary.m
            if m.track is not None:
                self.no_way[m.track] = s.t + NO_WAY_S
            self.counts["set aside"] += 1
            self.h.log(f"  target: {m.kind} #{m.track} at {m.x - s.where.x:+.0f} set aside for {NO_WAY_S} s: nothing to do "
                       f"against it from here")
            return act
        self._say(s, act)
        return act

    def _say(self, s, act: tuple) -> None:
        """One log line and one 'target' record when the target or the plan changes."""
        c = self.choice
        p, n = c.primary, c.next
        key = (p.m.track, act[0], act[1] if len(act) > 1 else None)
        if key == self._said:
            return
        self._said = key
        x = s.where.x
        m = p.m
        why = self._why(c)
        text = (f"{m.kind} #{m.track} at {m.x - x:+.0f}, {('%d%% HP' % round(m.bar * 100)) if m.bar is not None else 'no HP bar'}"
                f"{', chasing' if p.foe.chasing else ''}{', fading in' if p.foe.born else ''}, score {p.score:.1f} "
                f"({p.foe.exp:.0f} EXP in {p.way.s + p.foe.kill_s:.1f} s)" + _pile_text(p) + _next_text(n, m))
        what = _act_text(act)
        self.h.log(f"  target: {text}: {what}" + (f" ({why})" if why else ""))
        self.counts["said"] += 1
        if telemetry.recording():
            telemetry.emit("target", why=why, act=act[0], side=act[1] if len(act) > 1 else None, x=round(x, 1),
                           **({"to": [act[1], round(act[2], 1)]} if act[0] == "trip" else {}),
                           platform=s.platform.id, lead=round(self.her.lead, 3), inv=round(self.her.invulnerable, 3),
                           primary=p.record(x), next=None if n is None else n.record(x),
                           top=[t.record(x) for t in c.top], coming=None if c.coming is None else
                           {"track": c.coming.foe.m.track, "kind": c.coming.foe.m.kind, "s": round(c.coming.arrives, 2)})

    def _why(self, c: tg.Choice) -> str:
        """Why the choice changed, said as far as it's known: the one it left may be dead, set aside by the plan, or only not
        in this look (a look misses a monster now and then: about a quarter of those left so were back within 5 s)."""
        if c.how not in ("next", "gone") or c.left is None:
            return c.why
        tr = c.left.track
        was = ("set aside" if tr in self.no_way else "killed" if tr in self.h.ledger.dead else "reckoned dead"
               if any(f.spent and f.m.track == tr for f in self.foes) else "not in this look")
        return f"#{tr} {was}" + (", the next" if c.how == "next" else "")

    def end(self) -> None:
        """The run's numbers: the 'target_end' record (`revived`: piles a sighting brought back, DropStore.revived)."""
        telemetry.emit("target_end", counts=dict(self.counts), revived=self.h.drops.revived,
                       ms={k: {str(i): v for i, v in enumerate(h) if v} for k, h in self.ms.items()}, bin_ms=HIST_MS)
