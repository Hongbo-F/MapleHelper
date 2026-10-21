"""What to do about the monsters around her: throw, jump-throw, jump back and throw, step away, or
close in. Pure decisions from map positions; the fight loop (behavior.fight) carries them out.

`stance` is the policy for a group (fight.clear, what tours use) and `decide` the one for a single
target (fight.kill, `hunt kill`). Both take a monster on her platform as too close while a throw from the
ground at it would be a claw swipe, by its size (`close_to`, reach.claw_edge: a pig within 111 units, a
snail within 96), decide at least within TOO_CLOSE; stance closes in to SHOT_RANGE - 40 of a target where
decide walks to the nearest of SHOT_RANGE - 30 and 20 past the claw's edge from it (off her level, a band
past reach.shot_dx), either side.
`pick_target` says which monster to shoot first, `open_above` whether a jump keeps her on her
platform, and `act_text` and `pct` word it for the log. Thresholds built on reach's constants are
read from reach at call time, and so are the attacks' timings (physics.JUMP_ATTACK_MS and
JUMP_BACK_ATTACK_MS): whether a jump throw or a jump back can hit a kind follows from them.
A monster whose star terrain stops (reach.wall_to: a wall of her own foothold group, as for one on
platform 2 for her on 4, from anywhere on her platform; by the v83 line a floor too, from where she stands)
is out of reach from there: neither policy throws at it, or walks along her platform for it but to where
its line is clear. So is one behind a monster the attack takes first whose star is stopped (reach.shielded:
the client checks the line to its pick alone), so all of them go by every monster around her (`around`).
Off her level, how much room a star needs sideways goes by the monster's kind (reach.shot_dx: the star's
search picks a small or a short one from farther off).
"""

from collections.abc import Callable

from maplehelper.engine import LEFT, RIGHT

from .. import directions
from ..localization.mapdata import MapModel, Platform
from ..localization.monsters import MONSTER_TOP, Monster
from ..motion import physics
from ..motion.physics import JUMP_HEIGHT
from . import reach


# Distances in map units.
TOO_CLOSE = 110    # a monster on her platform nearer than this is about to touch her: step away first
STEP_AWAY = 120    # how far she steps away from a monster too close
STEP_ROOM = 40     # the least room worth stepping away into (with less she fights from where she is)
KNOCKBACK_ROOM = 25  # extra room kept from her platform's ends while monsters share it: standing ~100
                     # units from an end of platform 12, pigs walking into her knocked her off it
MAX_WALK = 300     # longest walk between two looks
MIN_MOVE = 15      # stance: a shorter step or approach isn't worth it
CLEAR_STEP = 10    # where a star's line is stopped from where she'd go, the places along her platform tried
                   # instead, this far apart (by the v83 line a floor stops it from part of her platform only)
JUMP_BACK_ROOM = 110  # room she needs behind her (from her platform's end, less the margin) to jump back
JUMP_BACK_CLEAR = 160  # ... with no monster on her platform this close on that side
JUMP_BACK_SPAN = 95  # how far a jump back carries her (by the engine 86-94 units, control.actions.JB_RUN_MS)

_THROW_VERBS = {"throw": "throwing", "jump_throw": "jump-throwing", "jump_back": "jumping back, throwing"}


def _side_of(t: Monster, x: float) -> str:
    """Which side of map x monster `t` is on (LEFT when right at x)."""
    return RIGHT if t.x > x else LEFT


def _nearest(monsters: list[Monster], x: float) -> Monster:
    """The monster nearest to map x along the platforms (the first of equally near ones)."""
    return min(monsters, key=lambda t: abs(t.x - x))


def _goal_reach() -> float:
    """How far from its target a spot she walks to may be: a little past SHOT_RANGE."""
    return reach.SHOT_RANGE + 15


def _clear_from(platform: Platform, x: float, y: float, t: Monster, lo: float, hi: float) -> float | None:
    """The nearest place to map x along `platform`, from `lo` to `hi` (every CLEAR_STEP units), from where a
    star reaches monster `t` (reach.can_shoot, out to _goal_reach) and terrain doesn't stop it (reach.wall_to),
    or None. A wall of her foothold group stops it from anywhere on her platform; by the v83 line (reach.LINE)
    her own floor stops a star at a monster below unless she stands near enough its end."""
    xs = [lo + i * CLEAR_STEP for i in range(int((hi - lo) // CLEAR_STEP) + 1)] + [hi]
    ok = [g for g in xs if reach.can_shoot(t.x - g, t.y - y, reach=_goal_reach(), kind=t.kind)
          and reach.wall_to(platform, g, y, t) is None]
    return min(ok, key=lambda g: abs(g - x), default=None)


def _stand_limits(platform: Platform, crowded: bool) -> tuple[float, float]:
    """The stretch of `platform` she keeps to: EDGE_MARGIN from its ends, and KNOCKBACK_ROOM more while
    monsters share it (`crowded`): a hit knocks her back."""
    margin = reach.EDGE_MARGIN + (KNOCKBACK_ROOM if crowded else 0)
    return platform.left + margin, platform.right - margin


def open_above(model: MapModel, platform: Platform, x0: float, x1: float) -> bool:
    """Whether she can jump anywhere between x0 and x1 on `platform` and land back on it: no other
    platform overhead within her jump height (she'd go up through it and land on top)."""
    lo, hi = sorted((x0, x1))
    for q in model.platforms:
        if q.id == platform.id or q.right < lo or q.left > hi:
            continue
        mid = min(max((lo + hi) / 2, q.left), q.right)
        if 0 < platform.y_at(min(max(mid, platform.left), platform.right)) - q.y_at(mid) <= JUMP_HEIGHT + 5:
            return False
    return True


def stance(x: float, y: float, platform: Platform, wanted: list[Monster], approach: bool = True,
           facing: str | None = None, can_jump: Callable[[float, float], bool] | None = None,
           tops: dict[str, float] | None = None, halves: dict[str, float] | None = None,
           around: list[Monster] | None = None) -> tuple:
    """What to do from map (x, y) on `platform` with these monsters around: ('throw', side),
    ('jump_back', side) at one too close for a throw from the ground (jump away, throw back from the
    air), ('jump_throw', side) the same when cornered (straight up), ('back', side, units) to step away,
    ('approach', side, units) to bring the nearest into reach, or ('wait',). She keeps working the side
    she faces while it still has monsters in reach (switching sides every throw spread the damage over
    both), and never walks past a monster. `can_jump(x0, x1)` says whether jumping over that stretch
    keeps her on her platform: under a platform within her jump height, a jump lands her up on it.
    A monster on her platform is too close for a throw from the ground while the claw would swipe it
    (close_to). `tops`: how tall each kind is at least, `halves`: half its width (the map's game data,
    localization.monsters.bodies; MONSTER_TOP and MONSTER_HALF by default). `around`: every monster around
    her, of any kind (`wanted` by default): one the attack takes first whose star is stopped shields those
    behind it (reach.shielded)."""
    can_jump = can_jump or (lambda a, b: True)
    around = wanted if around is None else around
    mine = [t for t in wanted if t.platform == platform.id]
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    close = [t for t in mine if close_to(t, x, halves)]  # a throw from the ground would be a claw swipe
    in_reach = [t for t in wanted if reach.can_hit(platform, x, y, t, others=around)]
    if close:
        return _close_quarters(x, close, mine, facing, lo, hi, can_jump, tops or MONSTER_TOP, halves)
    if in_reach:
        return ("throw", _throw_side(x, in_reach, facing))
    act = _step_from_under(x, y, platform, wanted, lo, hi, around)
    if act is None and approach:
        act = _close_in(x, y, platform, wanted, lo, hi, around)
    return act or ("wait",)


def air_gap(act: str) -> float | None:
    """How long after its jump key a stance's attack key comes ('jump_throw', 'jump_back'; physics), or
    None for a throw from the ground: how high she is when the attack picks its target follows from it
    (reach.air_rise)."""
    return {"jump_throw": physics.JUMP_ATTACK_MS, "jump_back": physics.JUMP_BACK_ATTACK_MS}.get(act)


def half_of(kind: str, halves: dict[str, float] | None = None) -> float:
    """Half a monster's width by its kind: from `halves` (the map's game data) if it has the kind, else
    MONSTER_HALF, else a pig's (reach.body_of)."""
    return (halves or {}).get(kind, reach.body_of(kind)[1])


def close_to(t: Monster, x: float, halves: dict[str, float] | None = None) -> bool:
    """Whether monster `t` (on her level) is too close to map x for a throw from the ground at it to be a
    star: the claw would swipe it (reach.claw_edge, by half its width: `halves`, as half_of). By its middle
    alone (a star from 100 units, whatever the kind), the claw swiped pigs, slimes and Orange Mushrooms she
    counted on a star for (in the simulator, out to 118-124 units off with the claw's old reach, 90 ahead;
    by the client's, 65 ahead, out to 93-99)."""
    return abs(t.x - x) < reach.claw_edge(half_of(t.kind, halves))


def _close_quarters(x: float, close: list[Monster], mine: list[Monster], facing: str | None, lo: float,
                    hi: float, can_jump: Callable[[float, float], bool], tops: dict[str, float],
                    halves: dict[str, float] | None = None) -> tuple:
    """A monster on her platform too close for a throw from the ground (`close`; `mine`: all on her
    platform), the nearest one deciding (on the side she faces, if any is): jump away from it and throw
    back from the air, if there's room behind her and no monster there; cornered, straight up. Each only
    if the attack starts low enough in the jump to pick that kind (reach.air_hits, by how tall it is at
    least: `tops`). A jump throw's starts 42-49 units up, over a snail: at most one of 45 jump-throws at
    snails hurt one, and monsters on the platform above took the stars. One too short for either is shot
    from the ground: from reach.star_sure away if she can step back that far (a star; by half its width,
    `halves`), else from where she is (the claw swipes it: weak, but every swipe hits)."""
    on_facing = [t for t in close if _side_of(t, x) == facing]
    nearest = _nearest(on_facing or close, x)
    side = _side_of(nearest, x)
    away = directions.opposite(side)
    back = directions.sign(away)
    room = (hi - x) if away == RIGHT else (x - lo)
    behind = [t for t in mine if 0 < (t.x - x) * back < JUMP_BACK_CLEAR]
    top = tops.get(nearest.kind, MONSTER_TOP["pig"])  # a kind of unknown size: a pig's
    if (reach.air_hits(top, physics.JUMP_BACK_ATTACK_MS) and room >= JUMP_BACK_ROOM and not behind
            and can_jump(x, x + back * JUMP_BACK_SPAN)):
        return ("jump_back", side)
    if not reach.air_hits(top, physics.JUMP_ATTACK_MS):
        step = reach.star_sure(half_of(nearest.kind, halves)) - abs(nearest.x - x)
        if room >= step and not behind:
            return ("back", away, step)
        return ("throw", side)
    if can_jump(x, x):
        return ("jump_throw", side)
    # A platform just overhead: a jump lands her on it (on 20, five jump-throws at a snail put
    # her up on 19 five times). Step away, else throw from the ground.
    if room >= STEP_ROOM and not behind:
        return ("back", away, min(STEP_AWAY, room))
    return ("throw", side)


def _throw_side(x: float, in_reach: list[Monster], facing: str | None) -> str:
    """Which way to throw at the monsters in reach: the way she faces while that side has any, else
    the side with more of them, then the one with the nearer."""
    if facing is not None and any(_side_of(t, x) == facing for t in in_reach):
        return facing
    right = [t for t in in_reach if t.x > x]
    left = [t for t in in_reach if t.x <= x]

    def score(group: list[Monster]) -> tuple:
        return len(group), -min((abs(t.x - x) for t in group), default=1e9)

    return RIGHT if score(right) > score(left) else LEFT


def _step_from_under(x: float, y: float, platform: Platform, wanted: list[Monster], lo: float,
                     hi: float, around: list[Monster] = ()) -> tuple | None:
    """Off her level and too close sideways (right under or over her), a star misses: step away from
    the nearest such monster, along her platform, to where a star reaches it (reach.shot_dx, by its
    kind; throwing at it from right over it used to go on and on), or where its star isn't stopped if it is
    from there (_clear_from). None if there's none (one whose star terrain stops from all along her platform
    doesn't count), or no such step. `around`: as stance's."""
    def goal_of(t: Monster) -> tuple[str, float] | None:
        away = LEFT if t.x > x else RIGHT
        goal = min(max(t.x + directions.sign(away) * (reach.shot_dx(t.y - y, t.kind) + 40), lo), hi)
        if reach.wall_to(platform, goal, y, t) is None:
            return away, goal
        goal = _clear_from(platform, x, y, t, lo, hi)
        return None if goal is None else (RIGHT if goal > x else LEFT, goal)

    under = [(t, g) for t in wanted if t.platform != platform.id and reach.in_band(t.y - y)
             and abs(t.x - x) < reach.shot_dx(t.y - y, t.kind) and (g := goal_of(t)) is not None]
    if not under:
        return None
    t, (side, goal) = min(under, key=lambda tg: abs(tg[0].x - x))
    if abs(goal - x) >= MIN_MOVE and reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around):
        return ("back", side, abs(goal - x))
    return None


def _close_in(x: float, y: float, platform: Platform, wanted: list[Monster], lo: float,
              hi: float, around: list[Monster] = ()) -> tuple | None:
    """Walk toward the nearest monster level enough to shoot, to SHOT_RANGE - 40 from it (never past it), or
    where terrain doesn't stop its star if it does from there (reach.wall_to, _clear_from: a wall of her
    foothold group does from anywhere on her platform, and then she doesn't walk for it). None if there's
    none, or it can't be brought into reach from her platform. `around`: as stance's."""
    def goal_of(t: Monster) -> float | None:
        goal = min(max(t.x - directions.sign(_side_of(t, x)) * (reach.SHOT_RANGE - 40), lo), hi)
        return goal if reach.wall_to(platform, goal, y, t) is None else _clear_from(platform, x, y, t, lo, hi)

    level = [(t, g) for t in wanted if reach.in_band(t.y - y) and (g := goal_of(t)) is not None]
    if not level:
        return None
    t, goal = min(level, key=lambda tg: abs(tg[0].x - x))
    side = _side_of(t, x)
    if ((goal - x) * directions.sign(side) > MIN_MOVE
            and reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around)):
        return ("approach", side, abs(goal - x))
    return None


def act_text(act: tuple, wanted: list[Monster], x: float, hidden: list[Monster]) -> str:
    """A stance as a log line: "throwing right (pig +120, slime +300 hidden)"."""
    if act[0] in _THROW_VERBS:
        side = [t for t in wanted if (t.x > x) == (act[1] == RIGHT)]

        def name(t: Monster) -> str:
            return f"{t.kind} {t.x - x:+.0f}" + (" hidden" if any(t is h for h in hidden) else "")

        return f"{_THROW_VERBS[act[0]]} {act[1]} ({', '.join(name(t) for t in side[:3])})"
    if act[0] in ("back", "approach"):
        return f"{'stepping away' if act[0] == 'back' else 'closing in'} {act[1]} {act[2]:.0f}"
    return "nothing in reach"


def pct(bar: float | None) -> str:
    """An HP bar for the log: "40%", or "-" when none is shown (unhurt)."""
    return "-" if bar is None else f"{bar:.0%}"


def pick_target(x: float, y: float, platform: Platform, targets: list[Monster],
                around: list[Monster] | None = None) -> Monster:
    """Monsters on her platform that are close first (they can reach her), then the most hurt one a star
    reaches from where she stands (finish what was started), then the nearest in reach, then the nearest.
    `around`: as stance's (`targets` by default)."""
    near = [t for t in targets if t.platform == platform.id and abs(t.x - x) < reach.SHOT_RANGE]
    if near:
        return _nearest(near, x)
    around = targets if around is None else around
    in_reach = [t for t in targets if reach.can_hit(platform, x, y, t, others=around)]
    hurt = [t for t in in_reach if t.bar is not None]  # finish the ones already hit first
    if hurt:
        return min(hurt, key=lambda t: (t.bar, abs(t.x - x)))
    return min(in_reach or targets, key=lambda t: abs(t.x - x) + 2 * abs(t.y - y))


def decide(x: float, y: float, platform: Platform, target: Monster, threats: list[Monster],
           halves: dict[str, float] | None = None, around: list[Monster] | None = None) -> tuple:
    """What to do about `target` from map (x, y) on `platform`: ('throw', direction),
    ('walk', direction, units), or ('skip', reason) when no spot on her platform is in reach (or terrain
    stops the target's star from all of them). `threats` are the other monsters on her platform; `around`:
    every monster around her (as stance's; `threats` by default). Never walks off her platform. One on her
    platform is too close within TOO_CLOSE, or while the claw would swipe it (close_to: by half its width,
    `halves`, as half_of)."""
    around = threats if around is None else around
    dx, dy = target.x - x, target.y - y
    mine = threats + ([target] if target.platform == platform.id else [])
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    close = [t for t in mine if abs(t.x - x) < TOO_CLOSE or close_to(t, x, halves)]
    if close:
        # Step away from whatever is closest, unless that walks into another monster or off the end.
        nearest = _nearest(close, x)
        away = LEFT if nearest.x > x else RIGHT
        room = (hi - x) if away == RIGHT else (x - lo)
        blocked = any(0 < (t.x - x) * directions.sign(away) < STEP_AWAY + TOO_CLOSE for t in mine)
        if not blocked and room >= STEP_ROOM:
            return ("walk", away, min(STEP_AWAY, room))
        # Cornered: fight from here. Jumping over it put the rest of the group on her other side.
        return ("throw", _side_of(nearest, x))
    if reach.can_hit(platform, x, y, target, others=around):
        return ("throw", directions.toward(dx))
    if not reach.in_band(dy):
        return ("skip", "not level enough with her platform")
    # Walk to the nearest spot on her platform that has the target in reach (either side of it): near it,
    # 20 past where the claw would swipe it (reach.claw_edge, by its size; it was 150 for every kind), or off
    # her level one band of the star's search past where it first picks the target (reach.shot_dx, by its
    # kind: a pig standing needs that band more than one walking).
    far = reach.SHOT_RANGE - 30
    near = max(reach.claw_edge(half_of(target.kind, halves)) + 20,
               reach.shot_dx(dy, target.kind) + reach.SEARCH_STEP)
    goals = [min(max(target.x + d, lo), hi) for d in (-far, -near, near, far)]
    goals = [g for g in goals if reach.can_shoot(target.x - g, dy, reach=_goal_reach(), kind=target.kind)]
    if not goals:
        return ("skip", "out of reach from this platform")
    goals = ([g for g in goals if reach.wall_to(platform, g, y, target) is None]
             or [g for g in [_clear_from(platform, x, y, target, lo, hi)] if g is not None])
    if not goals:
        return ("skip", "behind a wall of her platform's foothold group")
    # Of those, where no monster the attack takes first has its star stopped, if there's one: monsters move.
    goals = [g for g in goals if reach.shielded(platform, g, y, target, around) is None] or goals
    goal = min(goals, key=lambda g: abs(g - x))
    direction = RIGHT if goal > x else LEFT
    # A monster on her platform in the way gets shot first.
    ahead = [t for t in threats if 0 < (t.x - x) * directions.sign(direction) < abs(goal - x) + TOO_CLOSE]
    if ahead:
        return ("throw", _side_of(_nearest(ahead, x), x))
    return ("walk", direction, min(abs(goal - x), MAX_WALK))
