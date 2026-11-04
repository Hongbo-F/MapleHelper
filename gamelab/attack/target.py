"""The one monster she hits next, and the one after it: the target-first planner's choice (the user, 2026-11-04: "I think on
a short term, planner should just decide which mob to hit and plan against that. This is simple, less costly, and easy to
follow."). Pure: what perception says at this look comes in as `Her` and `Foe`s (behavior.targeting builds them), no clock,
no keys. No simulation either: one closed-form score a monster, read again at every look (README: "The target-first
planner").

**Which** (`score`, `Chooser.choose`): the kind's EXP over the time to get an attack on it (`soonest`: her lock, then at once
where her attack's real reach lands on it from where she stands; up close a two-key jump attack, or a step to range where
her attack doesn't go from a jump; else a jump's shot, else the walk to the nearest place from which it lands) and the time
to kill it from its HP bar or reckoned HP, weighted by the chance it's there (`Foe.p`: behavior.presence's for one the look
missed, her own throws' reckoned chance it's dead by now). A hurt or chasing one has no term of its own: it's close and
quick to kill, the numbers favour it (the dev arms say so: README). The target is kept until it's dead or gone, or another
beats it by MARGIN, so the plan doesn't flip.

**A newborn** (the user: "you should watch for respawns at each frame") is a Foe like any other from the first look that has
it, its `solid_s` and `hit_s` from its fade (perception.spawns: Mob.fading), and its `born` says so in the log.

**A monster coming at her** (`coming`; the user: "change target if a mob is moving towards you and can knock you back, if you
are not already attacking that mob"): the first of those on or beside her platform moving at her (its track's heading and
speed, or her aggro belief's chase) that reaches her after her invulnerability ends and before she has finished what she's
doing (`finish`: her attack on the primary and the kill's time). Its switch says why ("coming at her: 0.6 s").

**The next** (`after`; the user: "think about what's your next target so that you can immediately pick another target to
attack after kill"): the same score as if the primary were dead, from where she'll stand and with the time that kill takes
(the others walked on that long). Gone from the look, the primary's next is the primary at once, not chosen again, unless a
newborn beats it by MARGIN or `coming` says otherwise.

**Where** (`places`): where her attack doesn't land from where she stands, the nearest place along her platform from which it
does, each side of the monster, the nearer first.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from maplehelper.engine import LEFT, RIGHT

from ..localization.mapdata import Platform
from ..localization.monsters import Monster
from ..motion import physics
from ..motion.bodies import HER_HALF
from . import reach
from .danger import Danger
from .stance import (GROUND_PICK_S, NO_PACE, STEP_SETTLE_S, TAP_S, Pace, beside, half_of, jump_pick_s, too_close,
                     walks_on)

MARGIN = 1.3         # guess: another must beat the target's score by this share to take over (the plan doesn't flip)
MATCH_DX = 45.0      # units: the last look's monster is a look's of its kind and platform within this, with no track to go
                     # by (stance.HOLD_MATCH's: the tracks' match and her hit's push)
FAR_SLACK = 20.0     # units inside her attack's farthest a place to walk to is (a band of the search: a monster walks on)
NEAR_SLACK = 12.0    # ... and past its nearest (reach.CLAW_MARGIN's: the same reason)
WALK_STEP = 100.0    # units she walks at once at most, a look after each (hunt.WAIT_STEP's: 0.7 s at her 149 units/s)
JUMP_S = jump_pick_s(physics.JUMP_ATTACK_MS)  # s from her keys to an attack from a jump picking
HIT_ALPHA = 0.36     # a newborn can be hit from about this alpha (perception.spawns: its HP bar shows from then)


@dataclass(frozen=True)
class Foe:
    """A monster as the choice takes it at this look: the monster, its kind's EXP, s to kill it from what it has left, the
    chance it's there and alive (1: a look has it, none of her throws reckoned to have killed it), its speed along its
    platform as known at this look (units/s, + right; None: not known), whether her aggro belief has it chasing her, s
    until it can touch her and until it can be hit (a newborn's fade), and whether it's a newborn."""
    m: Monster
    exp: float
    kill_s: float
    p: float = 1.0
    vx: float | None = None
    chasing: bool = False
    solid_s: float = 0.0
    hit_s: float = 0.0
    born: bool = False


@dataclass(frozen=True)
class Her:
    """Her at this look, as the choice needs her: her platform, feet, s until her next keys can go (`lead`), her
    invulnerability left, the way she faces, whether her attack goes from a jump (`jumps`, the thief's), every monster about
    her (`around`), bodies (`sizes`: the magician's search by the game's; `halves`, half each kind's width), the map's danger,
    the stance's pace, where she keeps to (`lo`, `hi`), and what says a jump's attack surely takes one that no attack from the
    ground does (`jump_hits`: stance.jump_hits, asked only where it decides)."""
    platform: Platform
    x: float
    y: float
    lead: float = 0.0
    invulnerable: float = 0.0
    facing: str | None = None
    jumps: bool = False
    around: list[Monster] = field(default_factory=list)
    sizes: dict | None = None
    halves: dict[str, float] | None = None
    danger: Danger | None = None
    pace: Pace = NO_PACE
    lo: float = -math.inf
    hi: float = math.inf
    jump_hits: Callable[[Monster], bool] | None = None
    can_jump: bool = True


@dataclass(frozen=True)
class Way:
    """How she gets an attack on it: s until an attack can pick it, how ('now', 'close': a jump attack or a step to range,
    'jump': a jump's shot, 'walk'), and where she walks to (None: from where she stands)."""
    s: float
    how: str
    place: float | None = None


@dataclass(frozen=True)
class Terms:
    """A monster's score and its terms: what the log and the 'target' record show."""
    foe: Foe
    way: Way
    score: float

    @property
    def m(self) -> Monster:
        return self.foe.m

    def record(self, x: float) -> dict:
        m, f = self.foe.m, self.foe
        return {"kind": m.kind, "track": m.track, "dx": round(m.x - x, 1), "platform": m.platform,
                "bar": None if m.bar is None else round(m.bar, 2), "exp": round(f.exp, 1), "reach_s": round(self.way.s, 2),
                "kill_s": round(f.kill_s, 2), "p": round(f.p, 2), "how": self.way.how, "score": round(self.score, 2),
                **({"place": round(self.way.place, 1)} if self.way.place is not None else {}),
                **({"chasing": True} if f.chasing else {}), **({"born": True} if f.born else {})}


@dataclass(frozen=True)
class Coming:
    """A monster coming at her (`coming`): it, and s until it reaches her."""
    foe: Foe
    arrives: float


@dataclass(frozen=True)
class Choice:
    """One look's choice: the primary and the next (their terms; None: none), why it changed (''; it didn't), the top
    monsters by score (up to 3)."""
    primary: Terms | None = None
    next: Terms | None = None
    why: str = ""
    top: tuple = ()
    coming: Coming | None = None
    how: str = ""   # what kind of a change: 'first', 'next', 'gone', 'beaten', 'respawn', 'coming', 'no way'
    left: Monster | None = None  # with 'next' and 'gone': the one it left, not among this look's foes


def _side(m: Monster, x: float) -> str:
    return RIGHT if m.x > x else LEFT


def near_far(her: Her, m: Monster, y: float | None = None) -> tuple[float, float]:
    """How near and how far off (middle to middle) her attack from the ground lands on monster `m`, from her platform's
    height `y` (where she stands by default): reach.shot_dx by its body (the claw's edge or her search's first band, on her
    level; a star's search off it), and no nearer than its danger warrants on her platform (Danger.keep_off); to her attack's
    farthest (reach.SHOT_RANGE, which magic.apply sets for hers). `near` inf: no attack from her height takes it."""
    y = her.y if y is None else y
    top_half = reach.body_of(m.kind, her.sizes)
    near = reach.shot_dx(m.y - y, m.kind, her.sizes)
    if m.platform == her.platform.id:
        near = max(near, reach.ground_sure(half_of(m.kind, her.halves))[0])
        if her.danger is not None:
            near = max(near, her.danger.keep_off(m.kind))
    return near, min(float(reach.SHOT_RANGE), reach.max_dx(top_half[1]))


def places(her: Her, m: Monster) -> list[float]:
    """The places along her platform from which her attack from the ground lands on monster `m`: on each side of it the
    nearest of its stretch to her, checked (reach.can_hit, at the platform's height there: walls, what the attack takes
    first), the nearer first. None where there's none."""
    p, x, out = her.platform, her.x, []
    y = p.y_at(min(max(m.x, p.left), p.right))
    near, far = near_far(her, m, y)
    if not math.isfinite(near):
        return out
    a, b = near + NEAR_SLACK, far - FAR_SLACK
    if a > b:
        a = b = (near + far) / 2
    lo, hi = max(p.left, her.lo), min(p.right, her.hi)
    for d in (-1, 1):
        u, v = sorted((m.x + d * a, m.x + d * b))
        u, v = max(u, lo), min(v, hi)
        if u > v:
            continue
        g = min(max(x, u), v)
        if reach.can_hit(p, g, p.y_at(g), m, others=her.around, sizes=her.sizes):
            out.append(g)
    return sorted(out, key=lambda g: abs(g - x))


def soonest(her: Her, f: Foe, x: float | None = None, y: float | None = None, lead: float | None = None,
            far: bool = True) -> Way | None:
    """How she gets an attack on it soonest from map (x, y) (where she stands by default), her next keys `lead` s on (her
    lock's by default): at once if her attack from the ground lands on it there; up close (stance.too_close: it can't, by
    where it will be then) a two-key jump attack, or a step to range where her attack doesn't go from a jump; a jump's shot
    (`her.jump_hits`, asked only from where she stands); else the walk to the nearest place from which it lands (`places`).
    None where none does, or (`far` False: `Chooser.rank`, the search for a jump's or a walk's left out) where it would take
    one."""
    m, p = f.m, her.platform
    here = x is None
    x, y, lead = her.x if x is None else x, her.y if y is None else y, her.lead if lead is None else lead
    side = _side(m, x)
    turn = side != her.facing
    if beside(m, p, x, her.danger) and too_close(m, x, her.halves, her.danger, her.pace, p, turn):
        if her.jumps and her.can_jump:
            return Way(lead + JUMP_S, "close")
        if not her.jumps and reach.can_hit(p, x, y, m, others=her.around, sizes=her.sizes):
            return Way(lead + (TAP_S if turn else 0.0), "now")  # (her search picks it from here: nearer than she'd like)
        gap = abs(her.pace.at(m, x, lead + GROUND_PICK_S, walks_on(m, p)) - x)
        step = max(0.0, near_far(her, m)[0] + NEAR_SLACK - gap)
        return Way(lead + step / physics.WALK_SPEED + STEP_SETTLE_S, "close")
    if reach.can_hit(p, x, y, m, others=her.around, sizes=her.sizes):
        return Way(lead + (TAP_S if turn else 0.0), "now")
    if not far:
        return None
    if here and her.jumps and her.jump_hits is not None and her.jump_hits(m):
        return Way(lead + JUMP_S, "jump")
    near, far = near_far(her, m, y)
    if not math.isfinite(near):
        return None
    dx = abs(m.x - x)
    goal = None
    if here:
        options = places(her, m)
        if not options:
            return None
        goal = options[0]
        dist = abs(goal - x)
    else:  # (the next's: from where she'll stand, by the distance alone)
        dist = max(0.0, near + NEAR_SLACK - dx, dx - (far - FAR_SLACK))
    return Way(lead + dist / physics.WALK_SPEED + STEP_SETTLE_S, "walk", goal)


def score(f: Foe, way: Way) -> float:
    """What hitting it next makes her: the kind's EXP over the time to get an attack on it (and for it to be hittable) and
    to kill it, weighted by the chance it's there."""
    return f.p * f.exp / max(1e-3, max(way.s, f.hit_s) + f.kill_s)


def terms_of(her: Her, f: Foe, far: bool = True) -> Terms | None:
    """Monster `f`'s score at this look (`soonest`, `score`); None: her attack can't get at it."""
    way = soonest(her, f, far=far)
    return None if way is None else Terms(f, way, score(f, way))


def rank(her: Her, foes: list[Foe], walks: bool = True, keep: tuple = ()) -> list[Terms]:
    """The monsters her attack can get at by score, best first (`walks` False: none she'd have to walk for): those in reach
    first; a jump's or a walk's search (`soonest`: the dear part of a look's choice) with one in reach only for those it
    decides anyway (`keep`: her target and her next), with none for one that could still beat the best so far if its attack
    went at once."""
    near, rest = [], []
    for f in foes:
        t = terms_of(her, f, far=False)
        if t is not None:
            near.append(t)
        else:
            rest.append(f)
    best = max((t.score for t in near), default=0.0)
    out = near
    at_once = her.lead + JUMP_S if her.jumps else her.lead
    for f in sorted(rest, key=lambda f: -score(f, Way(at_once, ""))):
        if f not in keep and (near or score(f, Way(at_once, "")) <= best):
            continue
        t = terms_of(her, f)
        if t is not None and (walks or t.way.how != "walk"):
            out = out + [t]
            best = max(best, t.score)
    return sorted(out, key=lambda t: -t.score)


def coming(her: Her, foes: list[Foe], finish: float) -> Coming | None:
    """The first monster on or beside her platform moving at her (Foe.vx) that reaches her (its body hers) after her
    invulnerability ends and before she has finished what she's doing (`finish` s on), one already on her counting from now,
    a newborn from when it's solid (Foe.solid_s). A monster that gets there while she's invulnerable can't knock her."""
    best = None
    x = her.x
    for f in foes:
        m = f.m
        if not beside(m, her.platform, x, her.danger):
            continue
        gap = abs(m.x - x) - half_of(m.kind, her.halves) - HER_HALF
        if gap <= 0.0:
            s = 0.0
        else:
            v = f.vx
            if not v or (x - m.x) * v <= 0.0:
                continue
            s = gap / abs(v)
        s = max(s, f.solid_s)
        if s < her.invulnerable or s > finish:
            continue
        if best is None or s < best.arrives:
            best = Coming(f, s)
    return best


def after(her: Her, t: Terms, foes: list[Foe]) -> Terms | None:
    """The next: of `foes`, the one with the best score as if `t`'s were dead, from where she'll stand and with the time
    that kill takes (each walked on that long by its Foe.vx, within its platform); None with no other."""
    m = t.m
    x = her.x if t.way.place is None else t.way.place
    p = her.platform
    y = p.y_at(min(max(x, p.left), p.right))
    done = t.way.s + t.foe.kill_s
    best = None
    for f in foes:
        o = f.m
        if o is m or o.track is not None and o.track == m.track:
            continue
        moved = o
        if f.vx:
            mx = o.x + f.vx * done
            own = walks_on(o, p)
            moved = replace(o, x=min(max(mx, own.left), own.right) if own is not None else mx)
        way = soonest(her, replace(f, m=moved), x, y, 0.0)
        if way is None:
            continue
        s = score(f, way)
        if best is None or s > best.score:
            best = Terms(f, Way(done + way.s, way.how, way.place), s)
    return best


def same(a: Monster, b: Monster) -> bool:
    """Whether a look's monster `b` is the last look's `a`: by the feed's track where both have one, else of its kind and
    platform within MATCH_DX."""
    if a.track is not None and b.track is not None:
        return a.track == b.track
    return a.kind == b.kind and a.platform == b.platform and abs(a.x - b.x) <= MATCH_DX


def _find(m: Monster | None, foes: list[Foe]) -> Foe | None:
    """`m` among this look's `foes` (`same`; its track's first, then the nearest), or None."""
    if m is None:
        return None
    if m.track is not None:
        for f in foes:
            if f.m.track == m.track:
                return f
    near = [f for f in foes if (m.track is None or f.m.track is None) and same(m, f.m)]
    return min(near, key=lambda f: abs(f.m.x - m.x), default=None)


def _text(t: Terms) -> str:
    return f"{t.m.kind} {t.score:.1f}"


class Chooser:
    """The target and the next, kept from look to look (see the top): `choose` at every look, `forget` when she goes off on a
    trip."""

    def __init__(self, margin: float = MARGIN):
        self.margin = margin
        self.primary: Monster | None = None
        self.next: Monster | None = None

    def forget(self) -> None:
        self.primary = self.next = None

    def choose(self, her: Her, foes: list[Foe], walks: bool = True) -> Choice:
        """The primary and the next at this look (see the top). `walks` False: none that she'd have to walk for (on her way
        somewhere: what's in reach from where she stands)."""
        had, had_next = self.primary, self.next
        cur = _find(had, foes)
        kept = _find(had_next, foes)
        ranked = rank(her, foes, walks, tuple(f for f in (cur, kept) if f is not None))
        pick, why, how = None, "", ""
        if not ranked:
            self.primary = self.next = None
            return Choice(None, None, "" if had is None else "none", (), None, "" if had is None else "none")
        top = ranked[0]

        def beats(t: Terms) -> bool:
            return top.foe is not t.foe and top.score > t.score * self.margin

        if cur is not None:
            pick = next((t for t in ranked if t.foe is cur), None)
            if pick is None:
                why, how = f"no way at it: {_text(top)}", "no way"
            elif beats(pick):
                why, how = f"{_text(top)} over {_text(pick)}", "beaten"
                if top.foe.born:
                    why, how = f"respawn at {top.m.x:.0f}: {why}", "respawn"
                pick = None
        elif had is not None and (n := kept) is not None:
            pick = next((t for t in ranked if t.foe is n), None)
            if pick is not None:
                why, how = "next", "next"
                if top.foe.born and beats(pick):  # (a newborn beats it: the user's rule, each frame)
                    why, how = f"respawn at {top.m.x:.0f}: {_text(top)} over {_text(pick)}", "respawn"
                    pick = None
        if pick is None:
            pick = top
            if not why:
                why, how = ("gone", "gone") if had is not None else ("first", "first")
        come = coming(her, foes, pick.way.s + pick.foe.kill_s)
        if come is not None and come.foe is not pick.foe:
            t = next((t for t in ranked if t.foe is come.foe), None) or terms_of(her, come.foe)
            if t is not None:
                pick, why, how = t, f"coming at her: {come.arrives:.1f} s" + (" (a respawn)" if come.foe.born else ""), "coming"
        nxt = after(her, pick, foes)
        self.primary, self.next = pick.m, None if nxt is None else nxt.m
        return Choice(pick, nxt, why, tuple(ranked[:3]), come, how, had if how in ("next", "gone") else None)


