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
platform 2 for her on 4, from anywhere on her platform; or a floor, her own for one below her unless she
stands near enough its end, as the user saw in game on 2026-10-21) is out of reach from there: neither
policy throws at it, or walks along her platform for it but to where its line is clear (_clear_from). So
is one behind a monster the attack takes first whose star is stopped (reach.shielded: the client checks the
line to its pick alone), so all of them go by every monster around her (`around`).
Off her level, how much room a star needs sideways goes by the monster's kind (reach.shot_dx: the star's
search picks a small or a short one from farther off).
A snail too close for a star from the ground is never jump-thrown at (JUMP_THROW_SNAILS): she steps back to
where a throw from the ground is a star if her platform has the room, else stabs it (not to a short platform's
far end: SNAIL_FAR_END).
With the map's danger (attack.danger), the group policy lets a monster of a kind that hits hard come only as near
as its danger warrants (Danger.keep_off: a Horny Mushroom 169 units on Tree Dungeon; no kind on the first map, and
on West Rocky Mountain II an Axe Stump's 87 is inside the claw's reach, 110), and then does whichever of a jump back, a
step away and a throw from where she is leaves her the least danger (_escape: the monsters around her and the spawn
points they don't account for; a jump back toward another one, or onto a spawn point, isn't an escape, nor a move
into or through a monster's body), a step only if it's clearly safer (ESCAPE_THROW: it doesn't throw). From
where she is it's a throw from the ground, a jump throw only where that would be a claw swipe: a jump in place by one
walks it into her as she lands.
With what her jumps' attacks reach (`jumps`: attack.reach.coverage), nothing in reach from the ground has her
jump-throw at a monster on a platform above where an attack from a jump surely picks it (_jump_shot), in place,
forward or back, its attack key as soon in the jump as that pick is sure (JumpPlan); and on her way somewhere
(`toward`), a jump forward along it that hits one goes before a throw from the ground (PASSING_JUMPS).
"""

from collections.abc import Callable
from dataclasses import dataclass

from maplehelper.engine import LEFT, RIGHT

from .. import directions
from ..localization.mapdata import MapModel, Platform
from ..localization.monsters import DEFAULT_SPEED, MONSTER_SPEED, MONSTER_TOP, Monster, touch_box
from ..motion import physics
from ..motion import bodies as her_body
from ..motion.bodies import HER_HALF
from ..motion.physics import JUMP_HEIGHT, WALK_SPEED
from . import reach
from .danger import Danger


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
JUMP_BACK_SPAN = 95  # how far a jump back carries her (by the engine 87-97 units with the lags spread as measured,
                     # 94 at the median: control.actions.JB_RUN_MS)
# Escaping a monster that hits hard (_escape), a step away (STEP_AWAY) is taken over a jump back or a throw from where
# she is only if it leaves her this much less danger (attack.danger: the share of her HP touches would take): they
# throw, the step doesn't (0.8 s without one live). A guess, at what the throw buys: two in three of a Horny
# Mushroom's stars push it back (sim.world.PUSH_BY_DAMAGE), and 40 units more between them is 0.01-0.02 less danger
# 150-200 off. By danger alone the step always won (it goes 120 to the jump's 95: a lone Horny 150 off, 0.042 against
# 0.049), and she never threw.
ESCAPE_THROW = 0.03
# Whether she jump-throws at a monster too close for a star from the ground that's too short for a jump back's
# attack to pick (a snail: reach.air_hits). By the attack's area keyed 40 ms after the jump (physics.JUMP_ATTACK_MS)
# it's picked, and in game on 2026-10-20 one such star killed a snail. But the user, watching her hunt on
# 2026-10-21, saw jump throws mostly miss snails, "especially on short platform[s]", and the day's telemetry says
# so: of 26 jump throws at a snail under 46 units off on her platform (in reach of a stab from the ground, out of
# the star's search), 5 hurt or killed it; at 46-84 off 18 of 33, at 85-130 6 of 10. RustMS's port of the v83
# client, from as high as she is then (21-26 up), picks no snail 35-85 off. False: never; she steps back to where
# a throw from the ground is a star, if her platform has the room, else stabs it from the ground (the claw, weak
# but always hitting: 5 swipes kill one). `python -m gamelab probe run air` weighs the two pictures of the attack
# from a jump.
JUMP_THROW_SNAILS = False
SMALL_STEP_REACT_S = 0.45  # s from the end of a step back to her next attack's pick (the slide, a look, the key's
                           # lag): a monster walking at her closes in that much more (a snail 43.75 units/s: 20)
# Whether that step back from a snail may go past the 50 she keeps from her platform's ends with monsters about
# (stand limits), to SMALL_STEP_END from an end: on the 168-unit platforms 3, 20, 22, 31 and 33, to the far end. In
# the simulator it lost to stabbing the snail where she stands. With preset game's noise, one snail too close on
# her platform and a fight to its kill (every 20 units along those five platforms, the snail 20-80 units off either
# side, seeds 1-8: 1,120 starts), of the 187 starts where it took her to the far end she was knocked off her platform
# 27 times against 9, and killed the snail in 6.5 s on average (20 for none in 20 s) against 5.7 (within 4 s 25%
# against 16%, within 10 s 94% against 97%): on a short platform the snail is soon back at her, and there a touch
# knocks her off the end. Over all 1,120: 4.6 s against 4.5, knocked off 75 times against 57. On the long platforms
# 1 and 34 (432 starts, seeds 1-3) a step back within her room killed it in 3.0 s against 3.3 for stabbing first,
# with half the touches (183 against 331), so she still steps back where the room is. Not measured in game. False:
# on a short platform she stabs it.
SNAIL_FAR_END = False
SMALL_STEP_END = 40  # ... and how near her platform's end that step may take her then: not to EDGE_MARGIN's 25, the
                     # step's walk is planned before her last attack's lock runs out, and a touch meanwhile knocked
                     # her 40 units toward the end and a step of 110 took her off 3 (nav_fight_up, seed 1). Of that
                     # scenario's seeds 1-48, 36 arrived with 25, 38 with 35, 41 with 40 and with 50, with the step
                     # to the far end on; with it off (SNAIL_FAR_END False) this isn't used: 37 arrive with any

_THROW_VERBS = {"throw": "throwing", "jump_throw": "jump-throwing", "jump_back": "jumping back, throwing"}
# A jump throw at what no throw from the ground reaches from where she stands (_jump_shot: a monster on a platform above
# her, up close only from high in the jump, farther off from lower: attack.reach.shots, the windows widening with the
# distance): she takes one if it lands her back on her platform, where she keeps to, her body meeting no monster's on
# the way (motion.bodies, each where it can be by then: MonsterTrack.reach's spread, at its speed), and, forward or
# back, where the danger is at most JUMP_DANGER more than where she stands (attack.danger: a guess, a tenth of a Horny
# Mushroom's touch, 16% of her HP). The choice among them (the choice model): every one of these throws once, and a
# throw's lock (0.73 s, the game's: COMBAT.attack_lock_ms) is most of what it costs, the jump's flight (0.62 s) inside
# it; so what decides is what it gains: a throw from the ground when anything's in reach from there (no jump's lock
# extra, JUMP_THROW_LOCK_EXTRA_MS), else the jump whose attack comes soonest at the target pick_target's order takes
# first (hurt first, then the nearest), in place before forward before back (the least ground covered: the least
# danger), forward along the way she's going first; travelling, a jump forward along her way that hits one over a
# throw from the ground (PASSING_JUMPS: the throw and 84 units of her way, 0.56 s of her walk, in one lock).
JUMP_DANGER = 0.016
ARC_STEP = 0.02  # s between the places along a jump her body is checked at (as motion.flow's danger, DANGER_STEP)
# In passing (stance's `toward`: travelling that way), a jump forward that way whose attack hits a monster takes her
# on as she shoots (a jump's 83 units, 0.56 s of her walk, for an attack 40 ms later and a lock 0.1 s longer than a
# throw's from the ground): taken over a throw from the ground at a monster that way, on the same terms as above.
PASSING_JUMPS = True


@dataclass(frozen=True)
class JumpPlan:
    """How a jump throw or a jump back goes (control.actions.Actions.jump_throw, jump_back): the attack key `key_ms`
    after the jump key; `way` +1/-1 a jump forward that way (0: in place); `air_turn` False: turning, the tap goes
    before the jump (an attack too soon in the jump to turn first: a short monster); why it was chosen (`why`: the
    telemetry 'jump' record's kind) and the shot it makes (attack.reach.Shot), if worked out."""
    key_ms: int = physics.JUMP_ATTACK_MS
    way: int = 0
    air_turn: bool = True
    why: str = ""
    shot: reach.Shot | None = None


def plan_of(act: tuple) -> JumpPlan | None:
    """A jump throw's or jump back's plan (JumpPlan), if the stance gave one (else its defaults)."""
    return act[2] if len(act) > 2 and isinstance(act[2], JumpPlan) else None


def _side_of(t: Monster, x: float) -> str:
    """Which side of map x monster `t` is on (LEFT when right at x)."""
    return RIGHT if t.x > x else LEFT


def _nearest(monsters: list[Monster], x: float) -> Monster:
    """The monster nearest to map x along the platforms (the first of equally near ones)."""
    return min(monsters, key=lambda t: abs(t.x - x))


def _goal_reach() -> float:
    """How far from its target a spot she walks to may be: a little past SHOT_RANGE."""
    return reach.SHOT_RANGE + 15


def _clear_from(platform: Platform, x: float, y: float, t: Monster, lo: float, hi: float,
                around: list[Monster] | None = None) -> float | None:
    """The nearest place to map x along `platform`, from `lo` to `hi` (every CLEAR_STEP units), from where a
    star reaches monster `t` (reach.can_shoot, out to _goal_reach) and terrain doesn't stop it (reach.wall_to),
    or None. A wall of her foothold group stops it from anywhere on her platform; by the v83 line (reach.LINE)
    her own floor stops a star at a monster below unless she stands near enough its end. With `around` (the
    monsters around her), nor does a monster the attack takes first have its star stopped there (reach.can_hit's
    shield): from 14 at x 690 a pig on 17 at 880, in her floor's shadow, takes the stars at one at 1000, and
    from x 729 on the search doesn't pick it (under 151 off), so the one at 1000 gets them."""
    xs = [lo + i * CLEAR_STEP for i in range(int((hi - lo) // CLEAR_STEP) + 1)] + [hi]
    if around is None:
        ok = [g for g in xs if reach.can_shoot(t.x - g, t.y - y, reach=_goal_reach(), kind=t.kind)
              and reach.wall_to(platform, g, y, t) is None]
    else:
        ok = [g for g in xs if reach.can_hit(platform, g, y, t, reach=_goal_reach(), others=around)]
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
    for q in model.places:  # (not a twin of hers, 0.15 units above her: mapdata.MapModel.twins)
        if q.id == platform.id or q.right < lo or q.left > hi:
            continue
        mid = min(max((lo + hi) / 2, q.left), q.right)
        if 0 < platform.y_at(min(max(mid, platform.left), platform.right)) - q.y_at(mid) <= JUMP_HEIGHT + 5:
            return False
    return True


def stance(x: float, y: float, platform: Platform, wanted: list[Monster], approach: bool = True,
           facing: str | None = None, can_jump: Callable[[float, float], bool] | None = None,
           tops: dict[str, float] | None = None, halves: dict[str, float] | None = None,
           around: list[Monster] | None = None, danger: Danger | None = None,
           jumps: Callable[[], reach.Coverage] | None = None, toward: int = 0) -> tuple:
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
    behind it (reach.shielded). `danger`: the map's (attack.danger), for how near a monster that hits hard may come
    (Danger.keep_off) and where she goes from it (_escape). `jumps`: what her attacks reach from here, from the ground
    and from her jumps (attack.reach.coverage, worked out when first asked): with nothing in reach from the ground, a
    jump throw whose attack surely hits a wanted monster from the air comes before a step or a walk (_jump_shot: a
    monster on a platform above her, out of a star's reach from the ground), ('jump_throw', side, JumpPlan)
    or ('jump_back', side, JumpPlan); `toward` (+1/-1: she's travelling that way), a jump forward that way that hits
    one comes before a throw from the ground (PASSING_JUMPS)."""
    can_jump = can_jump or (lambda a, b: True)
    around = wanted if around is None else around
    mine = [t for t in wanted if t.platform == platform.id]
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    close = [t for t in mine if close_to(t, x, halves)  # a throw from the ground would be a claw swipe ...
             or (danger is not None and abs(t.x - x) < danger.keep_off(t.kind))]  # ... or it hits too hard
    in_reach = [t for t in wanted if reach.can_hit(platform, x, y, t, others=around)]
    if close:
        return _close_quarters(x, close, mine, facing, lo, hi, can_jump, tops or MONSTER_TOP, halves,
                               (platform.left + SMALL_STEP_END, platform.right - SMALL_STEP_END) if SNAIL_FAR_END
                               else None, danger, platform, around)
    if in_reach:
        if toward and PASSING_JUMPS and jumps is not None:
            jump = _jump_shot(x, y, platform, wanted, jumps(), lo, hi, facing, around, danger, halves, toward,
                              kinds=("forward",), why="passing")
            if jump is not None:
                return jump
        return ("throw", _throw_side(x, in_reach, facing))
    if jumps is not None:
        jump = _jump_shot(x, y, platform, wanted, jumps(), lo, hi, facing, around, danger, halves, toward)
        if jump is not None:
            return jump
    act = _step_from_under(x, y, platform, wanted, lo, hi, around)
    if act is None and approach:
        act = _close_in(x, y, platform, wanted, lo, hi, around)
    return act or ("wait",)


def air_gap(act: str | tuple) -> float | None:
    """How long after its jump key a stance's attack key comes ('jump_throw', 'jump_back': its plan's, plan_of, else
    physics'), or None for a throw from the ground: how high she is when the attack picks its target follows from it
    (reach.air_rise)."""
    plan = plan_of(act) if isinstance(act, tuple) else None
    name = act[0] if isinstance(act, tuple) else act
    if plan is not None and name in ("jump_throw", "jump_back"):
        return plan.key_ms
    return {"jump_throw": physics.JUMP_ATTACK_MS, "jump_back": physics.JUMP_BACK_ATTACK_MS}.get(name)


def _jump_shot(x: float, y: float, platform: Platform, wanted: list[Monster], cover: reach.Coverage, lo: float,
               hi: float, facing: str | None, around: list[Monster], danger: Danger | None,
               halves: dict[str, float] | None, toward: int = 0, kinds: tuple[str, ...] = reach.JUMP_KINDS,
               why: str = "") -> tuple | None:
    """A jump throw at one of `wanted` from map (x, y) on `platform` (stance's): of the shots `cover` has at them
    (attack.reach.shots) of `kinds`, those whose jump lands her back on her platform between `lo` and `hi` (where
    she keeps to), her body meeting no monster's on the way (_arc_meets), a jump forward or back neither landing nor
    passing into one (_into) nor, with `danger`, landing where the danger is more than JUMP_DANGER over where she
    stands; an 'up' as she'd make it (turning in the air if she doesn't face that way by her keys, `facing`). Of
    those, at the target pick_target's order would take first (hurt first, then the nearest), the one soonest done:
    in place, then forward (`toward` first: she's going that way; in passing, `why` 'passing', only that way), then
    back, the soonest attack key first. None if there's none. `why`: the plan's (else 'above' at a monster off her
    level above her, 'reach' at one that isn't)."""
    here_danger = None
    best = None
    for t in wanted:
        for sh in cover.of(t):
            if sh.jump is None or sh.kind not in kinds:
                continue
            side = RIGHT if sh.side > 0 else LEFT
            if sh.kind == "up" and sh.turn != (facing != side):
                continue
            if why == "passing" and sh.jump.way != toward:
                continue
            if sh.land is None or sh.land.id != platform.id or not lo <= sh.land_x <= hi:
                continue
            if sh.kind != "up":
                if _into(x, sh.land_x, platform, around, halves):
                    continue
                if danger is not None:
                    if here_danger is None:
                        here_danger = danger.at(platform.id, x, around)
                    if danger.at(platform.id, sh.land_x, around) > here_danger + JUMP_DANGER:
                        continue
            if _arc_meets(x, y, platform, sh, around, halves):
                continue
            hurt = t.bar is not None
            order = {"up": 0, "forward": 1 if not toward or sh.jump.way == toward else 2, "back": 3}[sh.kind]
            rank = (not hurt, t.bar if hurt else 0.0, abs(t.x - x) + 2 * abs(t.y - y), order, sh.key_ms)
            if best is None or rank < best[0]:
                best = (rank, sh, t)
    if best is None:
        return None
    _, sh, t = best
    side = RIGHT if sh.side > 0 else LEFT
    reason = why or ("above" if t.y < y - reach.LEVEL else "reach")
    if sh.kind == "back":
        return ("jump_back", side, JumpPlan(sh.key_ms, -sh.side, True, reason, sh))
    return ("jump_throw", side, JumpPlan(sh.key_ms, sh.jump.way, True, reason, sh))


def _arc_meets(x: float, y: float, platform: Platform, sh: reach.Shot, around: list[Monster],
               halves: dict[str, float] | None) -> bool:
    """Whether her body (motion.bodies) over shot `sh`'s jump from map (x, y), from the plan's first key until she
    lands, meets a monster's of `around`, each where it can be by then (from where it was seen, either way at its
    kind's speed, MONSTER_SPEED: DEFAULT_SPEED for a kind the table lacks; its box for touches, touch_box), every
    ARC_STEP. A jump in place lands where she stood: what walks into her there is the fight's (her keeping off)."""
    if sh.kind == "up":
        return False
    j = sh.jump
    landing = (sh.land, sh.land_x, sh.land_s)
    boxes = []
    for m in around:
        q = platform if m.platform == platform.id else platform.peer(m.platform)
        if q is None:
            continue
        half, tall = touch_box(m.kind)
        if halves and m.kind in halves:
            half = halves[m.kind]
        boxes.append((m, q, half, tall, MONSTER_SPEED.get(m.kind, DEFAULT_SPEED)))
    if not boxes:
        return False
    t, end = -j.lead_s, physics.JUMP_LAG + sh.land_s
    while t <= end + 1e-9:
        fx, fy, _ = _pose(platform, x, j, max(t, -j.lead_s), landing)
        since = t + j.lead_s
        for m, q, half, tall, speed in boxes:
            lo, hi = max(q.left, m.x - speed * since), min(q.right, m.x + speed * since)
            feet = q.y_at(min(max(fx, lo), hi))
            if her_body.touches(fx, fy, lo, hi, half, feet, tall):
                return True
        t += ARC_STEP
    return False


def _pose(platform: Platform, x: float, j: physics.Jump, t: float, landing: tuple) -> tuple[float, float, bool]:
    """Her feet t s after jump `j`'s key (from -j.lead_s, the plan's first key) from map x on `platform`:
    physics.pose without the walls (the shot's own landing, worked out with them, ends it)."""
    land, lx, s = landing
    if t <= physics.JUMP_LAG:
        dx, _ = j.arc(t)
        x1 = min(max(x + dx, platform.left), platform.right)
        return x1, platform.y_at(x1), False
    tau = t - physics.JUMP_LAG
    if land is not None and tau >= s:
        return lx, land.y_at(lx), False
    x0 = min(max(x + j.way * j.run, platform.left), platform.right)
    return x0 + j.vx * tau, platform.y_at(x0) - physics.rise_after(tau), True


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
                    halves: dict[str, float] | None = None, ends: tuple[float, float] | None = None,
                    danger: Danger | None = None, platform: Platform | None = None,
                    around: list[Monster] | None = None) -> tuple:
    """A monster on her platform too close for a throw from the ground (`close`; `mine`: all on her
    platform), the nearest one deciding (on the side she faces, if any is): jump away from it and throw
    back from the air, if there's room behind her and no monster there; cornered, straight up. Each only
    if the attack starts low enough in the jump to pick that kind (reach.air_hits, by how tall it is at
    least: `tops`). A jump throw's starts 42-49 units up, over a snail: at most one of 45 jump-throws at
    snails hurt one, and monsters on the platform above took the stars. One too short for either is shot
    from the ground: from reach.star_sure away if she can step back that far (a star; by half its width,
    `halves`), else from where she is (the claw swipes it: weak, but every swipe hits). So is a snail
    (too short for a jump back's attack) unless JUMP_THROW_SNAILS: she steps back as far as it walks at her
    meanwhile past star_sure, within `lo` and `hi` (or `ends`, with SNAIL_FAR_END: on a short platform, its far
    end), else stabs it. `lo`, `hi`: where she keeps to with monsters about. With `danger` (on `platform`, the
    monsters `around` her), one of a kind she keeps further off than the claw's reach (Danger.keep_off) is escaped
    as _escape has it."""
    on_facing = [t for t in close if _side_of(t, x) == facing]
    nearest = _nearest(on_facing or close, x)
    side = _side_of(nearest, x)
    away = directions.opposite(side)
    back = directions.sign(away)
    room = (hi - x) if away == RIGHT else (x - lo)
    behind = [t for t in mine if 0 < (t.x - x) * back < JUMP_BACK_CLEAR]
    top = tops.get(nearest.kind, MONSTER_TOP["pig"])  # a kind of unknown size: a pig's
    small = not reach.air_hits(top, physics.JUMP_BACK_ATTACK_MS)  # a snail
    if (danger is not None and platform is not None and not small
            and danger.keep_off(nearest.kind) > reach.claw_edge(half_of(nearest.kind, halves))):
        return _escape(x, side, room, can_jump, danger, platform, mine if around is None else around,
                       close_to(nearest, x, halves), halves)
    if not small and room >= JUMP_BACK_ROOM and not behind and can_jump(x, x + back * JUMP_BACK_SPAN):
        return ("jump_back", side)
    if small and not JUMP_THROW_SNAILS:
        need = reach.star_sure(half_of(nearest.kind, halves)) - abs(nearest.x - x)
        lo_far, hi_far = ends or (lo, hi)
        far = max(room, (hi_far - x) if away == RIGHT else (x - lo_far))
        if far >= need and not behind:
            walk = MONSTER_SPEED.get(nearest.kind, DEFAULT_SPEED) * (need / WALK_SPEED + SMALL_STEP_REACT_S)
            return ("back", away, min(far, need + walk))
        return ("throw", side)
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


def _escape(x: float, side: str, room: float, can_jump: Callable[[float, float], bool], danger: Danger,
            platform: Platform, around: list[Monster], swipe: bool, halves: dict[str, float] | None = None) -> tuple:
    """From a monster on her platform that hits hard, on `side` of her, too near (Danger.keep_off): of a jump back
    (JUMP_BACK_SPAN away, throwing back; with JUMP_BACK_ROOM of `room` behind her), a step away (STEP_AWAY, or the
    room there is, STEP_ROOM at least) and a throw from where she is, the one leaving her the least danger (Danger.at,
    with the monsters `around` her and the spawn points they don't account for), the first of those of equal danger;
    the step, which doesn't throw, only if it's ESCAPE_THROW safer than the others. From where she is it's a throw
    from the ground, or a jump throw where that would be a claw swipe (`swipe`) and a jump doesn't land her on a
    platform overhead. A monster behind her makes the way back worse by its own danger (with a Horny Mushroom on each
    side of her on Tree Dungeon's 23 the old rule, no monster within JUMP_BACK_CLEAR behind, let a jump back 1 time in
    332), and a move whose way or landing would take her into a monster's body (any kind's, on her platform: _into, by
    half its width, `halves`) is no escape."""
    away = directions.opposite(side)
    back = directions.sign(away)
    options = []  # (danger there, order, act)
    jump = x + back * JUMP_BACK_SPAN
    if room >= JUMP_BACK_ROOM and can_jump(x, jump) and not _into(x, jump, platform, around, halves):
        options.append((danger.at(platform.id, jump, around), 0, ("jump_back", side)))
    step = min(STEP_AWAY, room)
    if room >= STEP_ROOM and not _into(x, x + back * step, platform, around, halves):
        options.append((danger.at(platform.id, x + back * step, around) + ESCAPE_THROW, 1, ("back", away, step)))
    options.append((danger.at(platform.id, x, around), 2,
                    ("jump_throw", side) if swipe and can_jump(x, x) else ("throw", side)))
    return min(options, key=lambda o: (round(o[0], 6), o[1]))[2]


def _into(x: float, goal: float, platform: Platform, around: list[Monster], halves: dict[str, float] | None) -> bool:
    """Whether a move along `platform` from map x to `goal` would take her into a monster's body: one of `around` on it
    ahead of her, up to where its body (half its width: half_of, and her HER_HALF) meets hers at `goal` (hunt's
    _in_the_way, for any kind: a touch, and its knock, whatever it takes)."""
    for m in around:
        if m.platform != platform.id:
            continue
        r = half_of(m.kind, halves) + HER_HALF
        if (x < m.x <= goal + r) if goal > x else (goal - r <= m.x < x):
            return True
    return False


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
        if reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around):
            return away, goal
        goal = _clear_from(platform, x, y, t, lo, hi, around)
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
        if reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around):
            return goal
        return _clear_from(platform, x, y, t, lo, hi, around)

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

        plan = plan_of(act)
        how = "" if plan is None else (f" {'forward ' if plan.way and act[0] == 'jump_throw' else ''}"
                                       f"at {plan.key_ms} ms{f', {plan.why}' if plan.why else ''}")
        return f"{_THROW_VERBS[act[0]]} {act[1]}{how} ({', '.join(name(t) for t in side[:3])})"
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
    stops the target's star from all of them: a wall of her group, or a floor, her own for one below her
    unless she can stand near enough its end; or from every one a monster the attack takes first has its star
    stopped and none is MIN_MOVE or more away). `threats` are the other monsters on her platform; `around`:
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
    stops = [reach.wall_to(platform, g, y, target) for g in goals]
    goals = ([g for g, stop in zip(goals, stops) if stop is None]
             or [g for g in [_clear_from(platform, x, y, target, lo, hi)] if g is not None])
    if not goals:
        wall = stops[0].x1 == stops[0].x2
        return ("skip", "behind a wall of her platform's foothold group" if wall
                else "a floor in the way from all along her platform")
    # Of those, where no monster the attack takes first has its star stopped, if there's one; else the nearest
    # place on her platform where none has (_clear_from, with the monsters around her: one in her floor's shadow,
    # picked first from farther back, isn't picked from nearer her platform's end); else one at least MIN_MOVE
    # off, as monsters move. Already at one so shielded (she'd tap back and forth on the spot until the kill's
    # patience ran out), the target is out of reach from here for now.
    clear = [g for g in goals if reach.shielded(platform, g, y, target, around) is None]
    if not clear and (g := _clear_from(platform, x, y, target, lo, hi, around)) is not None:
        clear = [g]
    goals = clear or [g for g in goals if abs(g - x) >= MIN_MOVE]
    if not goals:
        return ("skip", "behind a monster whose star is stopped")
    goal = min(goals, key=lambda g: abs(g - x))
    if abs(goal - x) < 1:
        # At the spot already (her platform's end keeps her from a nearer one): the star gets there from here.
        return ("throw", directions.toward(dx))
    direction = RIGHT if goal > x else LEFT
    # A monster on her platform in the way gets shot first.
    ahead = [t for t in threats if 0 < (t.x - x) * directions.sign(direction) < abs(goal - x) + TOO_CLOSE]
    if ahead:
        return ("throw", _side_of(_nearest(ahead, x), x))
    return ("walk", direction, min(abs(goal - x), MAX_WALK))
