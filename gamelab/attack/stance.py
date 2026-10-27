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
Toward an end of her platform with a long drop under it (drop_limits: the next platform LONG_DROP or more down, or
none, and no wall of her foothold group there), her steps away and jumps back keep DROP_ROOM from it (`edges`), and
under a platform overhead an escape may be the jump up onto it (`up`: overhead), by the danger there.
Of the monsters in reach from the ground, one on her platform whose kind counts as a danger is thrown at first.
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
From 2026-10-27 (clearance kept ahead of time: the block of constants before JumpPlan) who's too close, and what each
throw would take, go by where each monster will be when the attack starts (`Pace`: her lock, the look's age, its walk),
from a jump by the star's search alone (reach.AIR_MODEL), on the slope as it is (reach.air_sure): up close the throw
from where she stands is the one whose attack surely takes the monster (`_from_here`: from the ground past the claw's
reach; a jump in place, keyed 40 ms into the jump if she faces it, else turning in the air where that picks it, else
tapping the arrow on the ground first), else a step to where the next one will (`_clear_step`), else a jump back, else
the claw from the ground within CLAW_ONLY, the jump in place beyond it.
With what her jumps' attacks reach (`jumps`: attack.reach.coverage), nothing in reach from the ground has her
jump-throw at a monster on a platform above where an attack from a jump surely picks it (_jump_shot), in place,
forward or back, its attack key as soon in the jump as that pick is sure (JumpPlan); and on her way somewhere
(`toward`), a jump forward along it that hits one goes before a throw from the ground (PASSING_JUMPS).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import MOVEMENT

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
# distance): she takes one if it lands her back on her platform, where she keeps to, however a jump forward or back goes
# (lifting off 12 units early or late, 10% slower, the kick short at 126 or 119: physics.jump_variants, as a route
# jump's landing is checked), her body meeting no monster's on the way (motion.bodies, each where it can be by then:
# MonsterTrack.reach's spread, at its speed; in place, the monsters off her platform: right under one above, her box
# meets it), and, forward or back, where the danger is at most JUMP_DANGER more than where she stands (attack.danger: a
# guess, a tenth of a Horny Mushroom's touch, 16% of her HP). The choice among them (the choice model): every one of
# these throws once, and a throw's lock (0.73 s, the game's: COMBAT.attack_lock_ms) is most of what it costs, the jump's
# flight (0.62 s) inside it; so what decides is what it gains: a throw from the ground when anything's in reach from
# there (no jump's lock extra, JUMP_THROW_LOCK_EXTRA_MS), else the jump whose attack comes soonest at the target
# pick_target's order takes first (hurt first, then the nearest), in place before forward before back (the least ground
# covered: the least danger), forward along the way she's going first; travelling, a jump forward along her way that
# hits one over a throw from the ground (PASSING_JUMPS: the throw and 84 units of her way, 0.56 s of her walk, in one
# lock).
JUMP_DANGER = 0.016
ARC_STEP = 0.02  # s between the places along a jump her body is checked at (as motion.flow's danger, DANGER_STEP)
# In passing (stance's `toward`: travelling that way), a jump forward that way whose attack hits a monster takes her
# on as she shoots (a jump's 83 units, 0.56 s of her walk, for an attack 40 ms later and a lock 0.1 s longer than a
# throw's from the ground): taken over a throw from the ground at a monster that way, on the same terms as above, and
# only if it lands her no more than PASS_SLACK past where she's going on her platform (the objective's place there, or
# where her route leaves it: behavior.hunt's _toward): a rope's foot or a jump's takeoff 15-80 ahead, overshot, has her
# walk back (review, 2026-10-25), which a throw from the ground doesn't.
PASSING_JUMPS = True
PASS_SLACK = MIN_MOVE  # units: a shorter walk back isn't worth a step (stance's own MIN_MOVE)
# Toward an end of her platform with a long drop under it, her steps away and jumps back from monsters on it keep this
# far from it (`edges`: drop_limits). A touch's knock takes her 53-79 units: live on 2026-10-25 on Tree Dungeon (the
# perception service's 10 fps records, D:/MapleHelper-scratch/analysis-1025/damage-skeptic knock.py) 53, 55, 58, 63, 66
# and 67 on flat ground over 0.8 s, and three knocks took her off 23's left end from 52, 53 and 79 units in, 389 units
# down to 29 and 43 s back up (with 2 touches on 29), each after she had stepped away left (53, 120, then 120 and 55
# units: the step's limit was EDGE_MARGIN and KNOCKBACK_ROOM, 50, from any end). An end's drop is long when the next
# platform under it is LONG_DROP or more down, or there's none, and no wall of her foothold group rises from it (it
# stops her: Platform.walls; 24's right end, the tower's wall): 23's left end (389 down to 29), not its right (37 to 24)
# nor 24's left (74 to 25: back up in 2-3.6 s live). Guesses past the measured knocks: 100 units, the farthest knock and
# a little.
DROP_ROOM = 100.0
LONG_DROP = 100.0
END_PAST = 5.0       # units past a platform's end where what's under it is looked for (end_drop)
# Under a platform overhead, the jump up onto it is an escape (_escape's `up`); off, as before. With DROP_ROOM, in the
# simulator (Tree Dungeon, crowd, zero-walls, near-misses, 8 seeds of 5 minutes): 0.92 bars of HP lost a minute against
# 0.97 without both, her knocks off a platform 7.5 a run against 8.1, 982 units down against 997.
ESCAPE_UP = True
# One in reach on her platform whose kind counts as a danger is thrown at first (_throw_side); off, as before. With
# spots.SPAWN_CLEAR, in the simulator as above: touches off her trips 7.85 a minute against 8.90 without both, 0.92 bars
# of HP lost a minute against 1.02.
TOUCHER_FIRST = True

# --- clearance, kept ahead of time (from 2026-10-27) -------------------------------------------------------------------
# The user, 2026-10-27, watching her hunt Tree Dungeon: "I frequently find the player standing at the same X pos as the
# mob, doing jump hit but couldn't hit the mob because no minimal clearance is maintained", and "jump attack is
# extremely hard to hit a mob on a down slop, even slightly". The live hunts of 2026-10-25 23:43 to 10-26 23:47 (14 on
# Tree Dungeon, 189 minutes and 6,209 attacks out of darts' spells: D:/MapleHelper-scratch/clear/diag-clearance) bear it
# out: from a jump the attack picks by the star's search alone (reach.AIR_MODEL), so a jump throw in place hurt a monster
# under 40 ahead 40% of the time (8% of the casts at an unhurt Green killed it under 40 ahead, 53% at 60-90), one turning
# her in the air (the attack 120 ms after the jump key, her feet 52-56 up) a third of the time at one level with her or
# lower, and from the ground the claw swipes one within its reach (a Lucky Seven press took MP 27% of the time under 90
# ahead, the rest swipes of 15-24). And she got that close: a monster on her platform within 60 of her 8.7 s a minute,
# 27% of her attacks, 30% of her touches; 41% of those episodes began with a landing, 11% with a monster walking in
# during her lock (Greens and Hornies walk 67-70% of the time at a median 95 units/s, and from the look her ground throw
# was decided on to its attack's start was 0.66 s: 86 of 632 ground throws decided as stars had the target under 93 at
# the pick, 51 of their 80 casts swipes). So her decisions go by where each monster will be when the attack starts
# (`Pace`: her lock and the look's age, the attack's own key, its walk by its track), a throw counts only where its
# attack surely picks the monster then (reach.air_sure from a jump, on the slope as it is; past the claw's edge from the
# ground), and up close she steps to where the next one will, or with no such step the claw swipes (CLAW_ONLY).
GROUND_PICK_S = physics.JUMP_LAG + physics.ATTACK_DECIDE_LAG_MS / 2000  # s from a throw's key to its attack's start
TAP_S = (MOVEMENT.tap_turn_ms + 30) / 1000  # a throw from the ground that taps the arrow first has its key this much
                                            # later (control.actions: the tap, and TURN_SETTLE_MS)
CLEAR_MARGIN = 10.0  # units: a jump throw in place counts only at a monster this much farther than where its attack
                     # surely picks it (reach.air_sure, AIR_MARGIN into the search's window): the tracks' error. Live the
                     # in-place throws without a turn hurt 40% of the time at 0-40 ahead, 79% at 40-60 (a mushroom is
                     # surely picked from 44), 81% at 60-130: 54 for a level mushroom
CLAW_ONLY = 15.0     # units, middle to middle: nearer than this (on her), with no step that clears it, she swipes it
                     # with the claw from the ground (15-24 a swipe, no MP), not a jump throw in place: by eye on the
                     # recordings of 2026-10-26 those hurt one on or under her 0 times in 9 (within 25), and by the
                     # service's tracks at the pick her casts in place killed one 0-10 ahead 1 time in 31 (3%), 10-20 2
                     # in 25, 20-30 5 in 32 (16%), 30-40 16 in 43 (37%, as at 40-130): a cast 20-30 ahead does about
                     # twice a swipe's damage on average, 0-10 about as much for its 14 MP. In the harness (Tree, crowd,
                     # 8 seeds of 5 minutes, the world as first fitted) 30 made 361.1 EXP a minute on the hunt loop
                     # where 15 made 370.6, 358.3 grinding where 347.8: the same within the seeds' noise
STEP_SETTLE_S = 0.25  # s after a step's walk before her next keys: the slide (STOP_RUN) and the look after it
# A monster walking at her stops where its body meets hers (its half and HER_HALF from her x), on the side of her it was
# seen on: it touches her there, and the touch's knock (about 58 units) leaves it on that side. Until the reviews of
# 2026-10-27 it walked on through her, and right after a throw (her lock still 0.7-0.8 s) a walker within about 85 in
# front had her tap round and throw at her empty back (86 such throws in its Tree goldens, none in master's). Replayed
# on the 839 live close-up decisions whose target's track at the pick is known (D:/MapleHelper-scratch/clear/fin/live/
# pace_stop.py, walks matched to the service's tracks record nearest the look as pace_of does), the 217 walking at her
# that reach her by the pick were a median 49 from her middle then, 207 on the side the look saw them: stopped so, 19.8
# off on average (walked through her 32.6, stopped at her x 29.5, where they were seen 27.1; all 839: 15.5, 18.8, 18.0,
# 20.1). False: as before.
PACE_STOP = True
# A monster whose walk nothing tells (no track, or one seen too briefly) stays where it was seen; True walks it at her at
# its kind's speed (Pace.speeds: the game data's, 100 units/s for the mushrooms; live a median 95), the worst case, as
# until the reviews of 2026-10-27: of the same decisions those 46 were 20.8 off on average where they were seen, 42.8
# walked at her, 28.2 walked at her and stopped where they meet her (fin/live/walk_at.py).
WALK_AT = False
TRACK_MATCH = 25.0   # units: a look's monster is its track's (Scene.tracks) within this of the track's x, its kind and
                     # platform the same (behavior.fight.pace_of)
TRACK_SEEN = 3       # ... and that track's speed counts once it was seen in this many frames (its speed is filtered)


@dataclass(frozen=True)
class Pace:
    """How soon her next attack picks its target, and how the monsters walk meanwhile (clearance kept ahead of time):
    `lead`, s from the look's frame to when her next keys can go (what's left of her attack lock, and the look's age);
    `walk`, a monster's speed along its platform (units/s, + right) as its track has it, or None where nothing tells
    (it stays where it was seen; with WALK_AT it walks at her, at `speeds` of its kind, DEFAULT_SPEED for one it
    lacks); `halves`, half each kind's width (where one walking at her meets her: PACE_STOP). Without `walk` the
    monsters stay where they were seen, as before."""
    lead: float = 0.0
    walk: Callable[[Monster], float | None] | None = None
    speeds: dict[str, float] | None = None
    halves: dict[str, float] | None = None

    def at(self, m: Monster, x: float, t: float, platform: Platform | None = None) -> float:
        """Where monster `m` will be (map x) `t` s after the look, her at map x `x`: as its walk has it, within
        `platform`'s ends if it's on it; one walking at her stops where its body meets hers (PACE_STOP), on the side
        of her it was seen on."""
        if self.walk is None:
            return m.x
        v = self.walk(m)
        if v is None:
            if not WALK_AT:
                return m.x
            v = (self.speeds or {}).get(m.kind, DEFAULT_SPEED) * (1.0 if m.x < x else -1.0)
        px = m.x + v * t
        if PACE_STOP and v and (m.x - x) * v < 0:  # walking at her: no nearer than where it meets her, if not yet
            s = 1.0 if m.x > x else -1.0
            px = x + s * max((px - x) * s, min((m.x - x) * s, half_of(m.kind, self.halves) + HER_HALF))
        if platform is not None and m.platform == platform.id:
            px = min(max(px, platform.left), platform.right)
        return px

    def meets(self, m: Monster, x: float, t: float, platform: Platform | None = None, half: float | None = None) -> bool:
        """Whether monster `m`'s body (`half` its width either side; by default its kind's, `halves`) meets hers at map
        x `x` `t` s after the look (where `at` has it): a touch, unless she's still invulnerable from the last."""
        half = half_of(m.kind, self.halves) if half is None else half
        return abs(self.at(m, x, t, platform) - x) <= half + HER_HALF + 1e-6


NO_PACE = Pace()


def jump_pick_s(key_ms: float) -> float:
    """s from a jump's key to its attack's start, the attack keyed `key_ms` after it (JUMP_LAG after the key, half
    ATTACK_DECIDE_LAG_MS on average)."""
    return key_ms / 1000 + GROUND_PICK_S


def too_close(t: Monster, x: float, halves: dict[str, float] | None = None, danger: Danger | None = None,
              pace: Pace = NO_PACE, platform: Platform | None = None, turn: bool = False) -> bool:
    """Whether monster `t` (on her platform) is too close to map x for her next throw from the ground to be a star, as
    it will be when that attack starts (`pace`: her lock and the look's age, and its walk; a tap first, `turn`, TAP_S
    later): the claw would swipe it (close_to, by half its width: `halves`), or with `danger` nearer than its kind's
    danger warrants (Danger.keep_off). The hunt loop's rule 1 and the stance go by it."""
    at = pace.lead + GROUND_PICK_S + (TAP_S if turn else 0.0)
    gap = abs(pace.at(t, x, at, platform) - x)
    return gap < reach.claw_edge(half_of(t.kind, halves)) or (danger is not None and gap < danger.keep_off(t.kind))


@dataclass(frozen=True)
class Here:
    """The throw she'd make from where she stands at a monster too close (_from_here): the stance's act, and whether its
    attack surely takes it (`sure`); `gap`, how far ahead of her it will be when the attack starts (middle to middle)."""
    act: tuple
    sure: bool
    gap: float


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


def end_drop(model: MapModel, platform: Platform, side: int) -> float:
    """How far she'd fall knocked off `platform`'s end on `side` (-1 its left, +1 its right): down to the highest
    platform under the map x just past it (inf with none), 0 where a wall of its foothold group rises from that end
    (Platform.walls: it stops her)."""
    e = platform.left if side < 0 else platform.right
    y = platform.y_at(e)
    for w in platform.walls:
        if w.x1 == w.x2 and abs(w.x1 - e) <= 2 and min(w.y1, w.y2) < y - 1 and max(w.y1, w.y2) > y - her_body.HER_TALL:
            return 0.0
    x = e + side * END_PAST  # where her feet leave it, knocked past it
    under = [q.y_at(x) - y for q in model.places if q.id != platform.id and q.covers(x) and q.y_at(x) > y + 1]
    return min(under, default=float("inf"))


def drop_limits(model: MapModel, platform: Platform) -> tuple[float, float]:
    """The stretch of `platform` her steps away and jumps back keep to (stance's `edges`): DROP_ROOM in from an end with
    a long drop under it (end_drop LONG_DROP or more), the end itself at another; worked out once a platform. On one
    too narrow for that, as far from the long drops as it goes: its middle between two (a step only toward it; review
    of 2026-10-26: Tree Dungeon's 18, 27 and 30, 187-188 units wide, drop 125-254 units at both ends, and the stretch
    was the whole platform), the other end with one."""
    key = (id(model), platform.id)
    got = _DROPS.get(key)
    if got is None or got[0] is not model:
        left, right = end_drop(model, platform, -1) >= LONG_DROP, end_drop(model, platform, 1) >= LONG_DROP
        lo = platform.left + (DROP_ROOM if left else 0.0)
        hi = platform.right - (DROP_ROOM if right else 0.0)
        if lo > hi:
            lo = hi = (platform.left + platform.right) / 2 if left and right else platform.right if left \
                else platform.left
        if len(_DROPS) > 4096:
            _DROPS.clear()
        got = _DROPS[key] = (model, (lo, hi))
    return got[1]


_DROPS: dict[tuple[int, int], tuple] = {}  # (id(model), platform) -> (the model, drop_limits' answer)


def overhead(model: MapModel, platform: Platform, x: float) -> Platform | None:
    """The platform a jump from map x on `platform` lands her on: the nearest one overhead within her jump height (she
    goes up through it and lands on top: open_above's); None where a jump lands her back on hers."""
    best = None
    for q in model.places:  # (not a twin of hers: mapdata.MapModel.twins)
        if q.id == platform.id or not q.covers(x):
            continue
        dy = platform.y_at(min(max(x, platform.left), platform.right)) - q.y_at(x)
        if 0 < dy <= JUMP_HEIGHT + 5 and (best is None or dy < best[0]):
            best = (dy, q)
    return None if best is None else best[1]


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
           jumps: Callable[[], reach.Coverage] | None = None, toward: int = 0, goal: float | None = None,
           edges: tuple[float, float] | None = None, up: Callable[[float], Platform | None] | None = None,
           pace: Pace = NO_PACE) -> tuple:
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
    or ('jump_back', side, JumpPlan); `toward` (+1/-1: she's travelling that way, to map x `goal` on her platform if
    given), a jump forward that way that hits one, landing short of `goal` (PASS_SLACK), comes before a throw from the
    ground (PASSING_JUMPS). `edges`: the stretch of her platform her steps away and jumps back keep to (drop_limits:
    DROP_ROOM from an end with a long drop under it); `up(x)`: the platform a jump from x lands her on (overhead), for
    an escape (_escape). `pace`: how soon her next attack picks and how the monsters walk meanwhile: who's too close
    goes by where each will be then (too_close), and up close what each throw would take then (_from_here)."""
    can_jump = can_jump or (lambda a, b: True)
    around = wanted if around is None else around
    mine = [t for t in wanted if t.platform == platform.id]
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    # a throw from the ground would be a claw swipe, or it hits too hard, as it will be when the attack starts
    close = [t for t in mine if too_close(t, x, halves, danger, pace, platform, _side_of(t, x) != facing)]
    in_reach = [t for t in wanted if reach.can_hit(platform, x, y, t, others=around)]
    if close:
        slo, shi = (lo, hi) if edges is None else (max(lo, edges[0]), min(hi, edges[1]))
        return _close_quarters(x, close, mine, facing, slo, shi, can_jump, tops or MONSTER_TOP, halves,
                               (platform.left + SMALL_STEP_END, platform.right - SMALL_STEP_END) if SNAIL_FAR_END
                               else None, danger, platform, around, up, y, pace)
    if in_reach:
        if toward and PASSING_JUMPS and jumps is not None:
            jump = _jump_shot(x, y, platform, wanted, jumps(), lo, hi, facing, around, danger, halves, toward,
                              kinds=("forward",), why="passing", goal=goal)
            if jump is not None:
                return jump
        return ("throw", _throw_side(x, in_reach, facing, platform, danger))
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
               why: str = "", goal: float | None = None) -> tuple | None:
    """A jump throw at one of `wanted` from map (x, y) on `platform` (stance's): of the shots `cover` has at them
    (attack.reach.shots) of `kinds`, those whose jump lands her back on her platform between `lo` and `hi` (where
    she keeps to), a jump forward or back however it goes (reach.Shot.lands_sure, land_span: lifting off early or
    late, slower, the kick short, as a route jump's landing is checked), her body meeting no monster's on the way
    (_arc_meets), a jump forward or back neither landing nor passing into one (_into) nor, with `danger`, landing
    where the danger is more than JUMP_DANGER over where she stands; an 'up' as she'd make it (turning in the air if
    she doesn't face that way by her keys, `facing`). Of those, at the target pick_target's order would take first
    (hurt first, then the nearest), the one soonest done: in place, then forward (`toward` first: she's going that
    way; in passing, `why` 'passing', only that way, and landing no more than PASS_SLACK past `goal`, where she's
    going on her platform, if given), then back, the soonest attack key first. None if there's none. `why`: the
    plan's (else 'above' at a monster off her level above her, 'reach' at one that isn't)."""
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
            if sh.land is None or sh.land.id != platform.id or not sh.lands_sure:
                continue
            near, far = sh.land_span
            if not (lo <= near and far <= hi):
                continue
            if sh.kind != "up":
                end = far if sh.jump.way > 0 else near  # the farthest along her way it may take her
                if why == "passing" and goal is not None and (end - goal) * sh.jump.way > PASS_SLACK:
                    continue
                if _into(x, end, platform, around, halves):
                    continue
                if danger is not None:
                    if here_danger is None:
                        here_danger = danger.at(platform.id, x, around)
                    if any(danger.at(platform.id, lx, around) > here_danger + JUMP_DANGER
                           for lx in {sh.land_x, near, far}):
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


def jump_hits(x: float, y: float, platform: Platform, m: Monster, cover: reach.Coverage, facing: str | None = None,
              around: list[Monster] | None = None, danger: Danger | None = None,
              halves: dict[str, float] | None = None) -> bool:
    """Whether the stance would jump-throw at `m` from map (x, y) on `platform` with nothing in reach from the ground
    (_jump_shot's terms: a shot `cover` has at it landing her back where she keeps to however it goes, her body clear
    of every monster `around` her, the danger kept; as she'd make it, facing `facing`): what the hunt loop and grinding
    count as in reach from a jump (behavior.hunt's _jump_hits), so they don't stay to shoot what the stance won't.
    Where she keeps to as with monsters on her platform if any of `around` is on it."""
    around = [m] if around is None else around
    lo, hi = _stand_limits(platform, crowded=any(t.platform == platform.id for t in around))
    return _jump_shot(x, y, platform, [m], cover, lo, hi, facing, around, danger, halves) is not None


def _arc_meets(x: float, y: float, platform: Platform, sh: reach.Shot, around: list[Monster],
               halves: dict[str, float] | None) -> bool:
    """Whether her body (motion.bodies) over shot `sh`'s jump from map (x, y), from the plan's first key until she
    lands, meets a monster's of `around`, each where it can be by then (from where it was seen, either way at its
    kind's speed, MONSTER_SPEED: DEFAULT_SPEED for a kind the table lacks; its box for touches, touch_box), every
    ARC_STEP; a jump forward or back so too as its slowest variant goes (the one landing nearest: Shot.alts), longer
    at each place. A jump in place, against the monsters off her platform: up into the one she throws at on a
    platform above (review, 2026-10-25: right under it, her box met a pig 100 above her from 0.14 s to 0.54 s after
    the jump key, before the star left); where she stands, what walks into her is the fight's (her keeping off),
    jumping or not."""
    j = sh.jump
    flights = [(j, (sh.land, sh.land_x, sh.land_s))]
    if sh.alts:
        flights.append(min(sh.alts, key=lambda a: abs(a[1][1] - x)))
    boxes = []
    for m in around:
        if sh.kind == "up" and m.platform == platform.id:
            continue
        q = platform if m.platform == platform.id else platform.peer(m.platform)
        if q is None:
            continue
        half, tall = touch_box(m.kind)
        if halves and m.kind in halves:
            half = halves[m.kind]
        boxes.append((m, q, half, tall, MONSTER_SPEED.get(m.kind, DEFAULT_SPEED)))
    if not boxes:
        return False
    for jj, landing in flights:
        if landing[0] is None:
            continue
        t, end = -jj.lead_s, physics.JUMP_LAG + landing[2]
        while t <= end + 1e-9:
            fx, fy, _ = _pose(platform, x, jj, max(t, -jj.lead_s), landing)
            since = t + jj.lead_s
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
                    around: list[Monster] | None = None, up: Callable[[float], Platform | None] | None = None,
                    y: float | None = None, pace: Pace = NO_PACE) -> tuple:
    """A monster on her platform too close for a throw from the ground (`close`, as it will be when her next attack
    starts: too_close; `mine`: all on her platform), the nearest one then deciding (on the side she faces, if any will
    be there: one walking at her stops where it meets her, on its side, Pace's PACE_STOP). The throw from where she
    stands (_from_here, `pace`: where each will be when its attack starts) if its attack surely
    takes it: from the ground past the claw's reach, else a jump in place keyed so that it does (reach.air_sure: her feet
    then, its body on the slope as it stands then), at once if she faces it, else turning in the air where that picks
    it, or tapping the arrow on the ground first and jumping keyed early (JumpPlan air_turn False). Else a step away to
    where her next throw would surely take it, its walk meanwhile allowed for (_clear_step: against one walking at her
    no step does on these platforms, 54 units gained for each 149 walked); else a jump back, if there's room behind her
    and no monster there (its own attack, turning in the air, picks one on her level only from about 145 off: it buys
    room); else the throw from where she is all the same: the claw's swipe from the ground within CLAW_ONLY of her or
    under a platform overhead, else the jump in place. `lo`, `hi`: where she keeps to with monsters about (`ends`, with
    SNAIL_FAR_END, for a kind no jump in place picks on her level: a snail, on a short platform its far end). With
    `danger` (on `platform`, the monsters `around` her), one of a kind she keeps further off than the claw's reach
    (Danger.keep_off) is escaped as _escape has it (`up`: the platform overhead, stance's). `y`: her feet (map y; her
    platform's y_at at x by default)."""
    at = pace.lead + GROUND_PICK_S  # where each will be when a throw from the ground would start (its side, how near)
    where = {id(t): pace.at(t, x, at, platform) for t in close}

    def side_of(t: Monster) -> str:
        return RIGHT if where[id(t)] > x or (where[id(t)] == x and t.x > x) else LEFT
    on_facing = [t for t in close if side_of(t) == facing]
    nearest = min(on_facing or close, key=lambda t: abs(where[id(t)] - x))
    side = side_of(nearest)
    away = directions.opposite(side)
    back = directions.sign(away)
    room = (hi - x) if away == RIGHT else (x - lo)
    behind = [t for t in mine if 0 < (t.x - x) * back < JUMP_BACK_CLEAR]
    top = tops.get(nearest.kind, MONSTER_TOP["pig"])  # a kind of unknown size: a pig's
    half = half_of(nearest.kind, halves)
    if y is None:
        y = nearest.y if platform is None else platform.y_at(min(max(x, platform.left), platform.right))
    others = mine if around is None else around
    here = _from_here(x, y, side, nearest, facing, can_jump, top, half, mine, halves, platform, pace)
    if danger is not None and platform is not None and danger.keep_off(nearest.kind) > reach.claw_edge(half):
        return _escape(x, side, room, can_jump, danger, platform, others, here, halves, up)
    if here.sure:
        return here.act
    jumpable = jump_reach(top, half) < reach.claw_edge(half)
    if JUMP_THROW_SNAILS and not jumpable and jump_reach(top, half) < 2 * reach.claw_edge(half) and can_jump(x, x):
        # (cornered or not, a kind the jump's attack picks at all near her: a snail from 143, not one 5 tall)
        return (("jump_throw", side) if facing == side
                else ("jump_throw", side, JumpPlan(physics.JUMP_ATTACK_MS, 0, False, "tap first")))
    far = room
    if ends is not None and not jumpable:
        far = max(room, (ends[1] - x) if away == RIGHT else (x - ends[0]))
    step = _clear_step(x, y, nearest, away, far, top, half, platform, others, halves, pace, can_jump)
    if step is not None and not behind:
        return ("back", away, step)
    if room >= JUMP_BACK_ROOM and not behind and can_jump(x, x + back * JUMP_BACK_SPAN) and jumpable:
        return ("jump_back", side)
    if not can_jump(x, x) and room >= STEP_ROOM and not behind:
        # A platform just overhead: a jump lands her on it (on 20, five jump-throws at a snail put her up on 19 five
        # times). Step away, else throw from the ground.
        return ("back", away, min(STEP_AWAY, room))
    if not jumpable:
        return ("throw", side)
    return here.act


def jump_reach(top: float, half: float) -> float:
    """How far ahead of her (middle to middle) a jump in place keyed 40 ms after the jump key surely picks a monster on
    her level `top` tall and `half` wide either side, CLEAR_MARGIN farther (reach.air_sure; inf if within STAR_RANGE
    none is): a Green or Horny Mushroom 54, a slime 73, a pig 88; a snail 143, past the claw's edge (96): a jump in place
    never takes one the claw would swipe, so she doesn't jump at one (JUMP_THROW_SNAILS) nor back from one."""
    key = (top, half, physics.JUMP_ATTACK_MS, reach.AIR_MODEL, CLEAR_MARGIN)
    if key not in _JUMP_REACH:
        a = next((float(a) for a in range(0, int(reach.STAR_RANGE))
                  if reach.air_sure(a, 0.0, top, half, physics.JUMP_ATTACK_MS)), math.inf)
        _JUMP_REACH[key] = a + CLEAR_MARGIN
    return _JUMP_REACH[key]


_JUMP_REACH: dict[tuple, float] = {}  # (top, half, the jump throw's key, reach's air model, CLEAR_MARGIN) -> jump_reach


def _from_here(x: float, y: float, side: str, m: Monster, facing: str | None, can_jump: Callable[[float, float], bool],
               top: float, half: float, mine: list[Monster], halves: dict[str, float] | None,
               platform: Platform | None, pace: Pace) -> Here:
    """The throw from map (x, y) on `platform` at monster `m`, on `side` of her, too close (_close_quarters), and whether
    its attack surely takes it where `pace` has it when the attack starts: from the ground (after a tap, TAP_S later,
    unless she faces `side`) if every monster of `mine` that way is then past the claw's edge (reach.claw_edge, by half
    its width: `halves`); else a jump in place (`can_jump`: none under a platform overhead) whose attack surely picks it
    then (reach.air_sure, CLEAR_MARGIN farther: `top` tall, `half` wide, its feet on the slope where it will be), keyed
    40 ms after the jump if she faces it, else turning in the air (physics.AIR_TURN_ATTACK_MS: her feet 52-56 up then,
    a mushroom on her level picked from 144 ahead) or tapping the arrow first and jumping once the tap's walk has
    stopped (physics.JUMP_THROW_TURN_MS, then 40 ms: JumpPlan air_turn False), whichever picks it, the air turn first.
    With none of those sure (`sure` False): within CLAW_ONLY of her, or with no jump, the claw from the ground; else the
    jump in place keyed 40 ms after the jump (tapping first if she must turn)."""
    d = directions.sign(side)
    turn = facing != side

    def gap(t: float) -> float:
        return (pace.at(m, x, t, platform) - x) * d

    def drop(t: float) -> float:
        if platform is None or m.platform != platform.id:
            return m.y - y
        mx = pace.at(m, x, t, platform)
        return platform.y_at(min(max(mx, platform.left), platform.right)) - y

    tg = pace.lead + (TAP_S if turn else 0.0) + GROUND_PICK_S
    ground = ("throw", side)
    g0 = gap(tg)
    swiped = [o for o in mine if -half_of(o.kind, halves) < (pace.at(o, x, tg, platform) - x) * d
              < reach.claw_edge(half_of(o.kind, halves))]
    if g0 >= reach.claw_edge(half) and not swiped:
        return Here(ground, True, g0)
    jump = ("jump_throw", side)
    tap = ("jump_throw", side, JumpPlan(physics.JUMP_ATTACK_MS, 0, False, "tap first"))
    if not can_jump(x, x):
        return Here(ground, False, g0)
    if not turn:
        tries = ((jump, physics.JUMP_ATTACK_MS, 0.0),)
    else:
        tries = ((jump, physics.AIR_TURN_ATTACK_MS, 0.0),
                 (tap, physics.JUMP_ATTACK_MS, physics.JUMP_THROW_TURN_MS / 1000))
    for act, key, wait in tries:
        t = pace.lead + wait + jump_pick_s(key)
        if reach.air_sure(gap(t) - CLEAR_MARGIN, drop(t), top, half, key):
            return Here(act, True, gap(t))
    if g0 < CLAW_ONLY:
        return Here(ground, False, g0)
    act, key, wait = tries[-1]
    return Here(act, False, gap(pace.lead + wait + jump_pick_s(key)))


def _clear_step(x: float, y: float, m: Monster, away: str, room: float, top: float, half: float,
                platform: Platform | None, around: list[Monster], halves: dict[str, float] | None,
                pace: Pace, can_jump: Callable[[float, float], bool] | None = None) -> float | None:
    """The shortest step away from monster `m` (on `away`'s side of her; `room` there, MIN_MOVE to STEP_AWAY) after which
    her next throw at it surely takes it, as `pace` has it walk meanwhile: the step's walk (physics.hold_ms) and
    STEP_SETTLE_S, then (facing away from it) a tap and a throw from the ground past its claw's edge (reach.star_sure),
    or the tap, physics.JUMP_THROW_TURN_MS and a jump in place keyed 40 ms after it (reach.air_sure, CLEAR_MARGIN
    farther; only where `can_jump` there). Never into a monster's body on her platform (_into). None if no step within
    `room` does: against one walking at her (95 units/s) she gains 54 units for each 149 walked, less what it walks
    while she starts and stops."""
    d = directions.sign(away)
    most = min(room, STEP_AWAY)
    tried = [float(s) for s in range(int(MIN_MOVE), int(most) + 1, int(CLEAR_STEP))]
    if most >= MIN_MOVE and (not tried or tried[-1] < most):
        tried.append(float(most))  # (all the room there is, as _clear_from tries its far end)
    for step in tried:
        gx = x + d * step
        if platform is not None and _into(x, gx, platform, around, halves):
            return None
        done = pace.lead + physics.hold_ms(step) / 1000 + STEP_SETTLE_S
        star_t = done + TAP_S + GROUND_PICK_S
        if pace.meets(m, gx, star_t, platform, half):
            continue  # (on her there when her throw starts: a touch, no clearance)
        if (gx - pace.at(m, gx, star_t, platform)) * d >= reach.star_sure(half):
            return step
        if can_jump is not None and not can_jump(gx, gx):
            continue
        jump_t = done + physics.JUMP_THROW_TURN_MS / 1000 + jump_pick_s(physics.JUMP_ATTACK_MS)
        if pace.meets(m, gx, jump_t, platform, half):
            continue
        mx = pace.at(m, gx, jump_t, platform)
        dy = (m.y - y if platform is None or m.platform != platform.id
              else platform.y_at(min(max(mx, platform.left), platform.right))
              - platform.y_at(min(max(gx, platform.left), platform.right)))
        if reach.air_sure((gx - mx) * d - CLEAR_MARGIN, dy, top, half, physics.JUMP_ATTACK_MS):
            return step
    return None


def _escape(x: float, side: str, room: float, can_jump: Callable[[float, float], bool], danger: Danger,
            platform: Platform, around: list[Monster], here: Here, halves: dict[str, float] | None = None,
            up: Callable[[float], Platform | None] | None = None) -> tuple:
    """From a monster on her platform that hits hard, on `side` of her, too near (Danger.keep_off): of a jump back
    (JUMP_BACK_SPAN away, throwing back; with JUMP_BACK_ROOM of `room` behind her), a step away (STEP_AWAY, or the
    room there is, STEP_ROOM at least) and the throw from where she is (`here`: _from_here's), the one leaving her the
    least danger (Danger.at, with the monsters `around` her and the spawn points they don't account for), the first of
    those of equal danger; the step, which doesn't throw, only if it's ESCAPE_THROW safer than the others, and so a
    throw from where she is whose attack wouldn't surely take it (`here.sure` False: a swipe, or a jump in place at one
    too near for the star's search). A monster behind her makes the way back worse by its own danger (with a Horny
    Mushroom on each side of her on Tree Dungeon's 23 the old rule, no monster within JUMP_BACK_CLEAR behind, let a jump
    back 1 time in 332), and a move whose way or landing would take her into a monster's body (any kind's, on her platform: _into, by
    half its width, `halves`) is no escape. Under a platform overhead (`up(x)`: where a jump from here lands her),
    the jump up onto it, throwing back at the monster as she rises, is one too, by the danger where she lands there
    (the monsters on hers that can't reach it count for nothing there: Danger.walk): on Tree Dungeon's 23 left of x 10,
    under 22, a jump back or a jump throw in place lands her on 22, so the old rule left her a step toward the drop at
    23's left end or a throw (all three of the live knocks off it, 2026-10-25, began there: DROP_ROOM). `room`: the
    room behind her within where her steps keep to (stance's `edges`)."""
    away = directions.opposite(side)
    back = directions.sign(away)
    options = []  # (danger there, order, act)
    jump = x + back * JUMP_BACK_SPAN
    if room >= JUMP_BACK_ROOM and can_jump(x, jump) and not _into(x, jump, platform, around, halves):
        options.append((danger.at(platform.id, jump, around), 0, ("jump_back", side)))
    step = min(STEP_AWAY, room)
    if room >= STEP_ROOM and not _into(x, x + back * step, platform, around, halves):
        options.append((danger.at(platform.id, x + back * step, around) + ESCAPE_THROW, 1, ("back", away, step)))
    options.append((danger.at(platform.id, x, around) + (0.0 if here.sure else ESCAPE_THROW), 2, here.act))
    if ESCAPE_UP and up is not None and (q := up(x)) is not None:
        options.append((danger.at(q.id, x, around), 3, ("jump_throw", side, JumpPlan(why=f"up onto {q.id}"))))
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


def _throw_side(x: float, in_reach: list[Monster], facing: str | None, platform: Platform | None = None,
                danger: Danger | None = None) -> str:
    """Which way to throw at the monsters in reach: with `danger`, toward the nearest of them on her `platform` whose
    kind counts as a danger (it can touch her: live on 2026-10-25 her spot's own Green Mushroom on Tree Dungeon's 24
    made 14 of her 34 touches, 3 while she shot 23's Horny Mushrooms the other way); else the way she faces while that
    side has any, else the side with more of them, then the one with the nearer."""
    if TOUCHER_FIRST and danger is not None and platform is not None:
        mine = [t for t in in_reach if t.platform == platform.id and danger.share(t.kind) > 0]
        if mine:
            return _side_of(_nearest(mine, x), x)
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
