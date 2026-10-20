"""What to do about the monsters around her: throw, jump-throw, jump back and throw, step away, or
close in. Pure decisions from map positions; the fight loop (behavior.fight) carries them out.

`stance` is the policy for a group (fight.clear, what tours use) and `decide` the one for a single
target (fight.kill, `hunt kill`). Their thresholds differ: a monster on her platform is too close
within MIN_SHOT + 10 for stance and TOO_CLOSE for decide, and stance closes in to SHOT_RANGE - 40 of
a target where decide walks to the nearest of SHOT_RANGE - 30 and MIN_SHOT + 60 from it, either side.
`pick_target` says which monster to shoot first, `open_above` whether a jump keeps her on her
platform, and `act_text` and `pct` word it for the log. Thresholds built on reach's constants are
read from reach at call time, and so are the attacks' timings (physics.JUMP_ATTACK_MS and
JUMP_BACK_ATTACK_MS): whether a jump throw or a jump back can hit a kind follows from them.
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
JUMP_BACK_ROOM = 110  # room she needs behind her (from her platform's end, less the margin) to jump back
JUMP_BACK_CLEAR = 160  # ... with no monster on her platform this close on that side
JUMP_BACK_SPAN = 95  # how far a jump back carries her (85 measured)

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
           tops: dict[str, float] | None = None) -> tuple:
    """What to do from map (x, y) on `platform` with these monsters around: ('throw', side),
    ('jump_back', side) at one too close for a throw from the ground (jump away, throw back from the
    air), ('jump_throw', side) the same when cornered (straight up), ('back', side, units) to step away,
    ('approach', side, units) to bring the nearest into reach, or ('wait',). She keeps working the side
    she faces while it still has monsters in reach (switching sides every throw spread the damage over
    both), and never walks past a monster. `can_jump(x0, x1)` says whether jumping over that stretch
    keeps her on her platform: under a platform within her jump height, a jump lands her up on it.
    `tops`: how tall each kind is at least (the map's game data; MONSTER_TOP by default)."""
    can_jump = can_jump or (lambda a, b: True)
    mine = [t for t in wanted if t.platform == platform.id]
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    close = [t for t in mine if abs(t.x - x) < reach.MIN_SHOT + 10]  # too close to throw at
    in_reach = [t for t in wanted if reach.can_shoot(t.x - x, t.y - y)]
    if close:
        return _close_quarters(x, close, mine, facing, lo, hi, can_jump, tops or MONSTER_TOP)
    if in_reach:
        return ("throw", _throw_side(x, in_reach, facing))
    act = _step_from_under(x, y, platform, wanted, lo, hi)
    if act is None and approach:
        act = _close_in(x, y, wanted, lo, hi)
    return act or ("wait",)


def air_gap(act: str) -> float | None:
    """How long after its jump key a stance's attack key comes ('jump_throw', 'jump_back'; physics), or
    None for a throw from the ground: how high she is when the attack picks its target follows from it
    (reach.air_rise)."""
    return {"jump_throw": physics.JUMP_ATTACK_MS, "jump_back": physics.JUMP_BACK_ATTACK_MS}.get(act)


def _close_quarters(x: float, close: list[Monster], mine: list[Monster], facing: str | None, lo: float,
                    hi: float, can_jump: Callable[[float, float], bool], tops: dict[str, float]) -> tuple:
    """A monster on her platform too close for a throw from the ground (`close`; `mine`: all on her
    platform), the nearest one deciding (on the side she faces, if any is): jump away from it and throw
    back from the air, if there's room behind her and no monster there; cornered, straight up. Each only
    if the attack starts low enough in the jump to pick that kind (reach.air_hits, by how tall it is at
    least: `tops`). A jump throw's starts ~49 units up, over a snail: at most one of 45 jump-throws at
    snails hurt one, and monsters on the platform above took the stars. One too short for either is shot
    from the ground: from STAR_SURE away if she can step back that far (a star), else from where she is
    (the claw swipes it: weak, but every swipe hits)."""
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
        step = reach.STAR_SURE - abs(nearest.x - x)
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
                     hi: float) -> tuple | None:
    """Off her level and too close sideways (right under or over her), a star misses: step away from
    the nearest such monster, along her platform, to where a star reaches it (throwing at it from
    right over it used to go on and on). None if there's none, or no such step."""
    under = [t for t in wanted if t.platform != platform.id and reach.in_band(t.y - y)
             and abs(t.x - x) < reach.min_dx(t.y - y)]
    if not under:
        return None
    t = _nearest(under, x)
    away = LEFT if t.x > x else RIGHT
    goal = t.x + directions.sign(away) * (reach.min_dx(t.y - y) + 40)
    goal = min(max(goal, lo), hi)
    if abs(goal - x) >= MIN_MOVE and reach.can_shoot(t.x - goal, t.y - y, reach=_goal_reach()):
        return ("back", away, abs(goal - x))
    return None


def _close_in(x: float, y: float, wanted: list[Monster], lo: float, hi: float) -> tuple | None:
    """Walk toward the nearest monster level enough to shoot, to SHOT_RANGE - 40 from it (never past
    it). None if there's none, or it can't be brought into reach from her platform."""
    level = [t for t in wanted if reach.in_band(t.y - y)]
    if not level:
        return None
    t = _nearest(level, x)
    side = _side_of(t, x)
    d = directions.sign(side)
    goal = min(max(t.x - d * (reach.SHOT_RANGE - 40), lo), hi)
    if (goal - x) * d > MIN_MOVE and reach.can_shoot(t.x - goal, t.y - y, reach=_goal_reach()):
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


def pick_target(x: float, y: float, platform: Platform, targets: list[Monster]) -> Monster:
    """Monsters on her platform that are close first (they can reach her), then the most hurt one a star
    reaches from where she stands (finish what was started), then the nearest in reach, then the nearest."""
    near = [t for t in targets if t.platform == platform.id and abs(t.x - x) < reach.SHOT_RANGE]
    if near:
        return _nearest(near, x)
    in_reach = [t for t in targets if reach.can_shoot(t.x - x, t.y - y)]
    hurt = [t for t in in_reach if t.bar is not None]  # finish the ones already hit first
    if hurt:
        return min(hurt, key=lambda t: (t.bar, abs(t.x - x)))
    return min(in_reach or targets, key=lambda t: abs(t.x - x) + 2 * abs(t.y - y))


def decide(x: float, y: float, platform: Platform, target: Monster, threats: list[Monster]) -> tuple:
    """What to do about `target` from map (x, y) on `platform`: ('throw', direction),
    ('walk', direction, units), or ('skip', reason) when no spot on her platform is in reach.
    `threats` are the other monsters on her platform. Never walks off her platform."""
    dx, dy = target.x - x, target.y - y
    mine = threats + ([target] if target.platform == platform.id else [])
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    close = [t for t in mine if abs(t.x - x) < TOO_CLOSE]
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
    if reach.can_shoot(dx, dy):
        return ("throw", directions.toward(dx))
    if not reach.in_band(dy):
        return ("skip", "not level enough with her platform")
    # Walk to the nearest spot on her platform that has the target in reach (either side of it).
    far, near = reach.SHOT_RANGE - 30, reach.MIN_SHOT + 60
    goals = [min(max(target.x + d, lo), hi) for d in (-far, -near, near, far)]
    goals = [g for g in goals if reach.can_shoot(target.x - g, dy, reach=_goal_reach())]
    if not goals:
        return ("skip", "out of reach from this platform")
    goal = min(goals, key=lambda g: abs(g - x))
    direction = RIGHT if goal > x else LEFT
    # A monster on her platform in the way gets shot first.
    ahead = [t for t in threats if 0 < (t.x - x) * directions.sign(direction) < abs(goal - x) + TOO_CLOSE]
    if ahead:
        return ("throw", _side_of(_nearest(ahead, x), x))
    return ("walk", direction, min(abs(goal - x), MAX_WALK))
