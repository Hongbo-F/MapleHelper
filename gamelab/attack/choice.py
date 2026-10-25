"""Which of her attacks a throw is: Lucky Seven or the regular attack (her claw's star), by what it would do to the
monster it takes and what she has of MP. Pure: the attacks' damage, a monster's HP left and her MP in, the attack
out (`pick`); the hunter keeps what it goes by up from her looks and throws (behavior.attacks).

The user, on 2026-10-24, out of MP and potions: "please try to use a mixture of regular attack and lucky seven. The
lucky seven simply consumes MP too quick and I'm broke, especially with your inefficient use of skills", then "use
lucky seven first but for finishing up a low hp monster, use regular attack. Not defaulting to regular attack for
pretty much everything", and "if MP amount is not enough to cast a single lucky seven, then use regular attack". So a
throw is
- the regular attack while her MP is under one cast (`can_cast`): a cast with too little MP throws nothing in game,
  and no other reserve is kept;
- the regular attack at a monster whose HP left it takes about as fast as Lucky Seven would (`finishes`: its expected
  throws within FINISH_SLACK of Lucky Seven's), where Lucky Seven would be overkill: its MP and its stars spent for
  nothing;
- Lucky Seven otherwise: three times the regular attack's damage a throw (SKILL), so a Horny Mushroom dies in about 4
  throws, not 12, and she wins a bigger share of a monster others hit too (EXP is split by damage).

`Attack` is one attack's damage: its stars a throw, each a whole number of damage from `lo` to `hi` as likely each,
less the monster's defence (x 100 / (PDDamage + 100), meowdb's formula, as the simulator's stars: sim.world), and its
MP a throw. How many throws take a monster's HP (`throws`) is worked out exactly from that (misses aside: they cost
both attacks alike).

What a monster has left is reckoned as a chance of each amount (`Left`), not one number: each of her throws at it takes
off its attack's damage as drawn, or nothing with a miss's chance (MISS), and a look seeing it alive after a throw
could have landed rules out every amount that throw would have killed it with (`Left.alive`). By her throws' average
damage alone, a Green Mushroom with two casts thrown at it and still walking had 16 HP left, and the regular attack
went to finish it; seen alive then, one of the two most likely missed, and it has about 83 (four times in five), a
throw of Lucky Seven's, not a regular star's: in the simulator 17 of 25 of those finishing throws didn't kill.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np

# Her regular attack's star: what it does before the monster's defence. What hers did on the first map at level 19
# (sim.world.STAR_DAMAGE, 30-70 as read off the monsters' bars there); at level 24 on Tree Dungeon not measured: the HP
# bars over the monsters read empty there until 2026-10-25 (perception.hud.hp_bar: of the 87 bars seen in the 20:57 run
# of 2026-10-24, 83 read 0, walking on after, 4 a share), and her damage hasn't been read off them since. A guess, the
# simulator's.
STAR = (30, 70)
# Lucky Seven's damage, in regular stars (two stars a cast, each SKILL / 2 of the regular one's): sim.world.LUCKY_SEVEN,
# fitted to her live hunts with it on Tree Dungeon (tour --loop --skill, 2026-10-24: 3.85 throws a Horny Mushroom's
# kill, 2.95 a Green's; the simulator's 3 stars a cast 4.1 and 2.8 with the game preset's misses).
SKILL = 3.0
SKILL_STARS = 2
# Of her throws, the share that do nothing (a star that misses, or a monster gone from where it was thrown at): the
# simulator's game preset (Noise.miss), whose Lucky Seven fits her live throws a kill (above). A guess: the chance a
# throw of hers took nothing off a monster as she reckons it (Left.after), never what an attack is chosen by.
MISS = 0.275
# Throws (expected, misses aside) the regular attack may take more than Lucky Seven at a monster for it to be the
# regular attack (`finishes`): a quarter of a throw, about 0.2 s of hers (THROW_S), against a cast's MP. A guess. At 0
# it would be the regular attack only where it's as sure a kill as Lucky Seven; at it, a Green Mushroom (233 HP, no
# defence) is finished with it from 40 HP left (76% one throw), a Horny Mushroom (403, DEF 10) from 36.
FINISH_SLACK = 0.25
THROW_S = 0.84   # s between her throws: her attack's lock and its margin (0.79-0.85 s live; behavior.patrol.THROW_S)
# MP a Lucky Seven cast costs at each of its skill levels, 1-20 (the game data: Skill 4001003, mpCon; a test checks it
# against the copy). Hers is config.toml [character] lucky_seven.
LUCKY_SEVEN_MP = (8, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16)
# The steps her MP bar's share is read in: its px at the game's own 1366x768 (perception.hud: blue from x 611 to 715), so
# to 1/104 of her MP (5.8 of 601), as the game draws it. She casts only with a step of MP more than a cast reads (the
# bar may be drawn rounded up), so a cast is never pressed with too little MP: at most a cast's worth is left unused.
MP_STEPS = 104


@dataclass(frozen=True)
class Attack:
    """One of her attacks: its key (GameKeys' field: 'attack' or 'lucky_seven'), its stars a throw and each star's
    damage before the monster's defence (whole numbers from `lo` to `hi`, as likely each), and its MP a throw."""
    key: str
    stars: int
    lo: int
    hi: int
    mp: float = 0.0

    @property
    def name(self) -> str:
        return "Lucky Seven" if self.key == "lucky_seven" else "the regular attack"

    def mean(self, pdd: float = 0.0) -> float:
        """Its damage a throw on average, after a defence of `pdd`."""
        return self.stars * (self.lo + self.hi) / 2 * 100 / (pdd + 100)

    def throws(self, hp: float, pdd: float = 0.0) -> float:
        """How many throws take `hp` HP of a monster of defence `pdd` on average (misses aside)."""
        if hp <= 0:
            return 0.0
        k = need(hp, pdd)
        return float(self.table(k)[k])

    def table(self, top: int) -> np.ndarray:
        """Expected throws (misses aside) to deal each whole damage 0 .. `top` at least, before the defence."""
        return _throws(self.stars, self.lo, self.hi, max(256, 1 << (top + 1).bit_length()))

    def pmf(self) -> np.ndarray:
        """The chance of each whole damage a throw does before the defence (index = damage)."""
        return _pmf(self.stars, self.lo, self.hi)

    def one_throw(self, hp: float, pdd: float = 0.0) -> float:
        """The chance one throw takes `hp` HP (misses aside)."""
        if hp <= 0:
            return 1.0
        return float(self.pmf()[need(hp, pdd):].sum())


REGULAR = Attack("attack", 1, *STAR)


def need(hp: float, pdd: float = 0.0) -> int:
    """The damage before the defence that takes `hp` HP of a monster of defence `pdd` (x 100 / (pdd + 100))."""
    return max(0, math.ceil(hp * (pdd + 100) / 100 - 1e-9))


@dataclass(frozen=True, eq=False)
class Left:
    """What a monster has left as she reckons it: `p[k]` the chance it takes k more damage before its defence `pdd`
    to kill it, k from 0 (dead already) to its whole HP's."""
    p: np.ndarray
    pdd: float = 0.0

    @classmethod
    def of(cls, hp: float, pdd: float = 0.0) -> "Left":
        """`hp` HP left, for sure."""
        p = np.zeros(need(hp, pdd) + 1)
        p[-1] = 1.0
        return cls(p, pdd)

    def after(self, attack: Attack, miss: float = MISS) -> "Left":
        """As it is after a throw of `attack` at it: its damage as drawn taken off, or nothing (a miss, `miss`)."""
        dealt = np.convolve(self.p[::-1], attack.pmf())  # the damage dealt so far, from her first throw on
        top = len(self.p) - 1
        folded = dealt[:top + 1].copy()
        folded[top] += dealt[top + 1:].sum()  # all of its HP or more: dead
        return Left((1 - miss) * folded[::-1] + miss * self.p, self.pdd)

    def hit(self) -> "Left | None":
        """As it is, known hit by at least one of the throws reckoned in it (its HP bar up: the game shows a monster's
        bar to whoever damages it): the chance they all missed gone, the rest in proportion; None if nothing else is
        left. That chance is all on its first amount (`p[-1]`), the one it was reckoned from (Left.of): a throw that
        lands takes at least its least damage off (a regular star's 30), so nothing else stays there."""
        p = self.p.copy()
        p[-1] = 0.0
        total = p.sum()
        return None if total < 1e-12 else Left(p / total, self.pdd)

    def alive(self) -> "Left | None":
        """As it is, seen alive: none of it dead, the rest in proportion; None if her throws as reckoned can't have
        left it alive (they did less than reckoned)."""
        p = self.p.copy()
        p[0] = 0.0
        total = p.sum()
        return None if total < 1e-12 else Left(p / total, self.pdd)

    @property
    def dead(self) -> float:
        """The chance it's dead already."""
        return float(self.p[0])

    @property
    def hp(self) -> float:
        """The HP it has left on average."""
        return float(self.p @ np.arange(len(self.p))) * 100 / (self.pdd + 100)

    def throws(self, attack: Attack) -> float:
        """How many throws of `attack` take what it has left, on average (misses aside)."""
        top = len(self.p) - 1
        return float(self.p @ attack.table(top)[:top + 1])


def lucky_seven(level: float, skill: float = SKILL) -> Attack:
    """Lucky Seven at skill `level` (its MP a cast: LUCKY_SEVEN_MP), `skill` regular stars of damage a cast."""
    lo, hi = STAR
    each = skill / SKILL_STARS
    return Attack("lucky_seven", SKILL_STARS, round(lo * each), round(hi * each), cast_mp(level))


def cast_mp(level: float) -> float:
    """MP a Lucky Seven cast costs at skill `level` (LUCKY_SEVEN_MP; past its ends, its nearest)."""
    return float(LUCKY_SEVEN_MP[min(max(int(level), 1), len(LUCKY_SEVEN_MP)) - 1])


def mp_points(share: float, max_mp: float) -> float:
    """Her MP as her bar's share reads it, less a step of it (MP_STEPS): what she surely has."""
    return share * max_mp - max_mp / MP_STEPS


def can_cast(share: float, max_mp: float, cost: float) -> bool:
    """Whether her MP (her bar's `share` of `max_mp`) surely pays for a cast of `cost` (mp_points)."""
    return mp_points(share, max_mp) >= cost


def finishes(hp: "float | Left", pdd: float = 0.0, regular: Attack = REGULAR, skill: Attack | None = None) -> bool:
    """Whether the regular attack takes what a monster of defence `pdd` has left (`hp` HP, or as reckoned: Left)
    about as fast as `skill` would: its expected throws within FINISH_SLACK of the skill's."""
    skill = skill or lucky_seven(1)
    left = hp if isinstance(hp, Left) else Left.of(hp, pdd)
    return left.throws(regular) <= left.throws(skill) + FINISH_SLACK


def pick(hp: "float | Left | None", pdd: float, mp_share: float | None, max_mp: float, skill: Attack | None,
         mode: str = "choice", regular: Attack = REGULAR) -> tuple[Attack, str]:
    """The attack a throw is, and why: `hp` what its monster has left as she reckons it (HP, or a Left; None: no
    monster seen in its way), `pdd` its defence, `mp_share` her MP bar's share (None: not read) of `max_mp`, `skill`
    Lucky Seven (None: she has none bound), `mode` 'regular' (the regular attack always), 'skill' (Lucky Seven whenever
    she has the MP) or 'choice' (the module's rules)."""
    if skill is None or mode == "regular":
        return regular, "regular"
    if mp_share is not None and not can_cast(mp_share, max_mp, skill.mp):
        return regular, "MP"
    if mode == "skill":
        return skill, "skill"
    if hp is None:
        return regular, "nothing in its way"
    if not isinstance(hp, Left) and math.isinf(hp):  # a kind whose HP isn't known
        return skill, "damage"
    if finishes(hp, pdd, regular, skill):
        return regular, "finish"
    return skill, "damage"


@functools.lru_cache(maxsize=None)
def _pmf(stars: int, lo: int, hi: int) -> np.ndarray:
    """The chance of each whole damage a throw of `stars` stars does (before the defence): index = damage."""
    one = np.zeros(hi + 1)
    one[lo:hi + 1] = 1.0 / (hi - lo + 1)
    out = np.array([1.0])
    for _ in range(stars):
        out = np.convolve(out, one)
    return out


@functools.lru_cache(maxsize=None)
def _throws(stars: int, lo: int, hi: int, top: int) -> np.ndarray:
    """Expected throws to deal each whole damage 0 .. top - 1 (before the defence): E(0) = 0, E(k) = 1 + the mean of
    E(k - d) over a throw's damage d (E of anything at or under 0 being 0)."""
    pmf = _pmf(stars, lo, hi)
    support = np.nonzero(pmf)[0]
    weights = pmf[support]
    e = np.zeros(top)
    for k in range(1, top):
        e[k] = 1.0 + float(weights @ e[np.maximum(k - support, 0)])
    return e
