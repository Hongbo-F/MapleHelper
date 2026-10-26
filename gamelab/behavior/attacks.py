"""Her attack a throw (attack.choice: Lucky Seven or the regular attack), chosen as each throw goes out from what she
knows then: the monster it takes, the HP she reckons it has left, and her MP as her bar reads it. Kept up from every
look (Senses.listeners), whoever takes it, and from the throws themselves; every throw goes through Actions, whose
attack key asks `key` (the fights, the hunt loop, grinding and the navigator's fights on the way alike).

- The monster a throw takes: the likeliest in its way from where she stands in the last look that searched for
  monsters (ledger.in_path: the claw's stance, else the star's search's first band) that isn't likelier dead than
  alive by her throws (DEAD_LIKELY: one is, and the star takes the next), else the one she last threw at that way
  within fight.HIDDEN_FOR (hit, it hides under its damage numbers), one not likely dead first, else none: then
  nothing seen is in its way, and it's the regular attack (no MP thrown at nothing). Never one dying where a kill of
  hers was placed (ledger.dying: of its kind on its platform, the nearest it within DYING_DX, while its death animation
  plays): it's still drawn, and no throw at it makes a mark. Live on 2026-10-25 (the hunt of 23:43) the kill was
  placed and its mark dropped, but the dying monster stayed drawn 0.6-1.3 s, was taken from the path again with its
  bar's HP or its whole (a dying monster's bar reads None), got a new mark, and once it was gone the hidden fallback
  followed that mark for 1.8 s with no DEAD_LIKELY check: 18 casts of Lucky Seven and 4 regular throws went at the
  dead in 11 minutes, a tenth of her MP potions (D:/MapleHelper-scratch/analysis-1025/mp-attack-check/atdead.py).
- What it has left (`reckon`: attack.choice.Left, a chance of each amount): its HP in the game data (MapModel.mob_hp),
  less her throws at it. Her throws at a monster are kept with it (`Mark`, followed look to look: monsters.seen_again),
  each taking off its attack's damage as drawn, or nothing (choice.MISS); a look that sees it alive LANDS_S or more
  after a throw rules out what that throw would have killed it with (Left.alive), and one that reads its HP bar
  between empty and full sets it from the bar (the throws that can't have landed by then still taken off). A bar up
  but read empty (at 1366x768 every hurt monster's bar read so until 2026-10-25, perception.hud.hp_bar taking the
  bar's black frame for its fill; the harness's `--world bars-empty`) still says
  her damage shows: not all her throws at it missed (Left.hit), once none is out that could have shown since the
  last look and not landed by LANDS_S (SHOWS_S). Before (review, 2026-10-25) such a bar counted for nothing, and in
  the simulator with the bars drawn empty she reckoned a target's HP 86-98 off (median), throwing no regular attack
  to finish one in 518 throws. A throw is chosen by what the monster has left if it's still alive when the throw
  lands. A monster she hasn't thrown at, whose bar no look reads, has its whole HP. Others' damage to it isn't
  known: a monster's bar shows its damage to whoever dealt it, and the bars on monsters she hadn't thrown at in the
  live hunts of 2026-10-24 didn't go with other players by them (7.2% of sightings with a player on its platform,
  9.7% without: her stars that hit another in their path than the one she aimed at, and misreads).
- Her MP: the latest look's bar share (Scene.mp) of her MP when full (config.toml [character] mp), less her casts
  that look's frame can't show yet. A cast's MP is taken off as its throw goes out (`on_throw`, told by Actions after
  its key is chosen, at the key's own time); live a cast's MP first showed 0.21-0.53 s after its key, so a bar read
  off a frame shown at t is less the casts keyed after t - MP_FIRST_S (not shown yet), and less those keyed between
  t - MP_SHOWS_S and then (shown or not) where her MP as reckoned says so: between the two her reckoning stands, past
  them the bar is believed so (her pet's drink, her regen, a press that took no MP: `_believed`). Until 2026-10-26 the
  bar was believed from senses.KEY_EFFECT_S, 68 ms, on, and Actions.throw told of a ground throw's cast only at the
  next throw, with that one's key time: harmless while the look before the next key came before the cast showed, 8
  of 10 pairs live, one cast low in the other 2. (Taking the bar as it read, with KEYS_WAIT "moves" in
  loop_tree_mp_runs_out she cast Lucky Seven with 5 MP, nothing thrown: the look after her last cast, shown before
  its key, read the 12 MP it took.) For a day (review, 2026-10-26) a read within MP_SHOWS_S of a cast was believed
  only where it was lower: with a look 0.2-0.34 s after each key, nothing that raised her MP showed while she kept
  casting, and replayed through the live hunts of 23:43 to 01:03 she threw the regular attack for 'MP' 91 times, 45
  of them with a cast's MP surely hers (D:/MapleHelper-scratch/hunt-ab/review-a-correct/mp_replay_code.py); the
  harness 11.75 a run, every one with the MP. Under her pet's MP warning the bar reads 0 for up to 0.6 s after the
  cast that took her there, while she still has the MP for a cast (WARNING_STEPS): such a 0 isn't believed, her MP
  is the last read less her casts since.

Each choice is logged with its numbers, and sent as a telemetry 'attack' record; `counts` keeps them for the run's
report.
"""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from maplehelper.engine import RIGHT

from .. import telemetry
from ..attack import choice
from ..attack.choice import Attack, Left
from ..localization.mapdata import MapModel
from ..localization.monsters import TRACK_SLACK, Monster, bodies, seen_again
from ..localization.scene import Scene
from ..perception.kinds import kind_of
from .ledger import dying, in_path

HIDDEN_FOR = 1.8   # s a monster she threw at is still taken to be in a throw's way unseen (fight.HIDDEN_FOR)
LANDS_S = 0.6      # s after a throw's attack key by which its damage has shown on its target's bar, if it hit: the
                   # server's (actions.DAMAGE_SHOWS_MS), live a Horny Mushroom's bar first lower 0.21-0.55 s after a
                   # cast's key (p10-p90), a kill's gain by 0.46 s. 0.85 until 2026-10-26, when the star's flight was
                   # taken for when it showed (its release 0.52 s after the key and up to 0.6 s in the air)
SHOWS_S = 0.15     # s after a throw's attack key that its damage may first show: a kill's gain live from 0.15 s on, a
                   # bar's drop from 0.21 (0.47 until 2026-10-26, about when the star leaves)
LOST_S = 3.0       # s unseen before a monster she threw at is forgotten (ledger.WOUND_LOST_S) ...
FORGET_S = 15.0    # ... and s since her last throw at it (ledger.WOUND_S)
MODES = ("choice", "skill", "regular")  # attack.choice.pick's modes: `tour --attack` (choice by default)
DEAD_LIKELY = 0.5  # a monster in a throw's way this likely dead by her throws (Left.dead) isn't its target: the next is
# s after a cast's attack key that its MP may first show on her bar (MP_FIRST_S), and by when it has (MP_SHOWS_S): live
# on 2026-10-25 (the hunt of 23:43, 1366x768) a cast that went out first showed 0.21-0.53 s after its key, the bar
# falling over 0.1-0.2 s more (isolated casts: D:/MapleHelper-scratch/analysis-1025/mp-attack-check/nodrop.py,
# lowcheck2.py). A read off a frame shown at t is less the casts keyed after t - MP_FIRST_S, and those keyed from
# t - MP_SHOWS_S to then are taken off it as far as her reckoning (on_throw took them off) says (`_believed`).
MP_FIRST_S = 0.21
MP_SHOWS_S = 0.55
# Her pet's MP warning (System Options, 5% steps: sim.world.PET_MP) makes the bar read 0: live on 2026-10-25 every one
# of 26 runs of 0 reads began 0.20-0.46 s after a cast's key and lasted 0.6 s at most, ending in the pet's drink (24)
# or, the potions out, in the true 3 px and 1 px; no look ever read 4 or 5 px, and the bar read 6-9 px before each, so
# the cast took her under 5% (about 20-35 MP: a cast's 14 more), and 11 regular throws went out for 'MP 0' with the MP
# for a cast (mp-attack-check/zeros.py, series.py). Which frame the game draws then isn't known (perception.hud.mp_fill
# finds no blue on its row: a warning flash?). So a 0 read within WARNING_S of a read above 0 isn't believed where that
# read, or her MP as reckoned since (the read less her casts), is WARNING_STEPS or less (of choice.MP_STEPS): her MP is
# the reckoning (and a 0 read longer is believed). By the reckoning too since the review of 2026-10-26: replayed, 2 of
# the live runs of 0 came after a read of 10 px and two casts (5.3 px, 33 MP) and were believed.
WARNING_STEPS = 9
WARNING_S = 2.0


@dataclass(frozen=True)
class _Read:
    """A frame's MP bar as `Attacks._mp_read` takes it (a look's Scene has the same)."""
    t: float
    mp: float | None


@dataclass
class Mark:
    """A monster she has thrown at: as a look last saw it, when, what it has left as of the throws looks have seen
    land (`left`), and her throws at it since (when, which attack)."""
    m: Monster
    seen: float
    left: Left
    pending: list[tuple[float, Attack]] = field(default_factory=list)
    last: float | None = None  # her last throw at it
    whole: bool = True  # `left` is reckoned from its HP before any throw of hers (False: from a bar read since)

    @property
    def thrown(self) -> float:
        return self.seen if self.last is None else self.last


class Attacks:
    """Her attack a throw on map `model` with keys `keys`: in `mode` (attack.choice.pick: 'choice', 'skill' or
    'regular'), with MP `max_mp` when full and Lucky Seven at skill `level` (None if she has no Lucky Seven key).
    `kills`: her kills so far (the ledger's), which end the marks they're placed on."""

    def __init__(self, model: MapModel, keys, mode: str = "choice", max_mp: float = 623.0, level: float = 16.0,
                 log: Callable[[str], None] = print, kills: Callable[[], list] | None = None):
        if mode not in MODES:
            raise ValueError(f"no such attack mode: {mode!r} (one of {', '.join(MODES)})")
        self.keys, self.mode, self.max_mp, self.log = keys, mode, max_mp, log
        self.skill: Attack | None = choice.lucky_seven(level) if keys.lucky_seven is not None else None
        self.regular = choice.REGULAR
        self.stats = {kind_of(name): (hp, model.mob_pdd.get(name, 0.0)) for name, hp in model.mob_hp.items() if hp > 0}
        # each kind's death animation (ledger.dying), the map's game data
        self._die = {k: b.die_s for k, b in bodies(model).items()} if hasattr(model, "mob_bodies") else {}
        self.marks: list[Mark] = []
        self.mp: float | None = None    # her MP bar's share in the latest look, less her casts it can't show (on_look)
        self._cast: float | None = None  # the MP share of the cast `key` chose, until its throw says when it's keyed
        self._casts: list[tuple[float, float]] = []  # her casts' attack keys (on_throw) and their MP shares, from
                                                      # MP_SHOWS_S before the latest look on
        self._read: tuple[float, float] | None = None  # the last bar share read above 0, and when (WARNING_STEPS)
        self._warned = False  # the last bar read was a 0 under her pet's warning (counted once a run of them)
        self.counts: Counter = Counter()  # throws by attack (its key) and by why
        self._last: Scene | None = None  # the latest look that searched for monsters, placed
        self._kills = kills or (lambda: [])
        self._killed = 0
        # The monster the next throw is at, if the stack knows better than a throw from the ground would (a jump's
        # attack at one on a platform above: behavior.fight.throw_act), taken by that throw's key alone
        self.aimed: Monster | None = None

    # --- after every look ------------------------------------------------------------------------

    def on_look(self, s: Scene) -> None:
        """Her MP; the marks followed, their bars read, forgotten when unseen LOST_S, untouched FORGET_S or killed."""
        if (read := self._mp_read(s)) is not None:
            self.mp = self._believed(read, s.t)
        kills = self._kills()
        for k in kills[self._killed:]:
            if getattr(k, "placed", False):
                self.marks = [mk for mk in self.marks if not (mk.m.kind == k.kind and mk.m.platform == k.platform
                                                              and abs(mk.m.x - k.x) < TRACK_SLACK)]
        self._killed = len(kills)
        if not s.searched or s.where is None or s.platform is None:
            return
        self._last = s
        for mk in self.marks:
            if (again := seen_again(mk.m, s.monsters)) is None:
                continue
            mk.m, mk.seen = again, s.t
            landed = [a for t, a in mk.pending if t <= s.t - LANDS_S]
            mk.pending = [(t, a) for t, a in mk.pending if t > s.t - LANDS_S]
            if _believable(again.bar) and (stats := self.stats.get(again.kind)) is not None:
                mk.left, mk.whole = Left.of(again.bar * stats[0], stats[1]), False
                continue
            if landed:  # alive after them: what would have killed it ruled out
                left = mk.left
                for a in landed:
                    left = left.after(a, choice.MISS)
                mk.left = left.alive() or mk.left  # (none alive: her throws did less than reckoned, taken as nothing)
            if again.bar is not None and mk.whole and not any(t <= s.t - SHOWS_S for t, _ in mk.pending):
                mk.left = mk.left.hit() or mk.left  # her damage shows: not all of her throws reckoned in it missed
        self.marks = [mk for mk in self.marks if s.t - mk.seen < LOST_S and s.t - mk.thrown < FORGET_S]

    def glance(self, t: float, mp: float | None) -> None:
        """Her MP bar's share `mp` off a frame shown at `t` newer than the latest look (None: not read), as on_look
        reads it: the glance before a throw's key (behavior.hunter.Hunter._still). Her pet's drink shows about 0.3 s
        after the cast that brings it, and the look a throw is decided on comes 0.2-0.34 s after the key before (live,
        2026-10-25): without it, a drink shown between the look and the key went unseen, and the throw went the regular
        attack for 'MP' (the harness: 1.25 a run with her MP back)."""
        if (read := self._mp_read(_Read(t, mp))) is not None:
            self.mp = self._believed(read, t)

    # --- at every throw --------------------------------------------------------------------------

    def key(self, side: str) -> str:
        """The attack key a throw that way goes out with now (Actions' attack key asks this): its choice logged. Its
        target: `aimed`, if set (and then cleared), else what a throw that way takes (target)."""
        now = time.monotonic()
        aimed, self.aimed = self.aimed, None
        seen = [] if self._last is None else self._last.monsters
        target = aimed if aimed is not None and not self._dying(aimed, now, seen) else self.target(side)
        hp, pdd = self.stats.get(target.kind, (0.0, 0.0)) if target is not None else (0.0, 0.0)
        left = None if target is None else self.reckon(target)
        alive = None if left is None else left.alive() or left  # as it is if it's alive when the throw lands
        what = None if target is None else float("inf") if alive is None else alive
        attack, why = choice.pick(what, pdd, self.mp, self.max_mp, self.skill, self.mode, self.regular)
        self._cast = attack.mp / self.max_mp if attack.mp > 0 and self.max_mp > 0 else None
        if target is not None and target.kind in self.stats and not self._dying(target, now, seen):
            mk = self._mark_of(target)
            if mk is None:
                mk = Mark(target, now, left, whole=not _believable(target.bar))
                self.marks.append(mk)
            mk.pending.append((now, attack))
            mk.last = now
        self.counts[attack.key] += 1
        self.counts[why] += 1
        self._say(attack, why, target, left, alive, hp)
        return getattr(self.keys, attack.key).name

    def on_throw(self, at: float, side: str) -> None:
        """A throw (Actions.throw_listeners), its attack key going down at `at`: a cast `key` chose takes its MP off
        now (looks shown before it could show still read it: on_look)."""
        if self._cast is None:
            return
        if self.mp is not None:
            self.mp = max(0.0, self.mp - self._cast)
        self._casts.append((at, self._cast))
        self._cast = None

    def target(self, side: str) -> Monster | None:
        """The monster a throw that way takes, as she knows it: the likeliest in its way in the last look that isn't
        likely dead by her throws (DEAD_LIKELY; the likeliest if all are), else the one she last threw at that way
        within HIDDEN_FOR (unseen: under its damage numbers), one not likely dead first, else None; never one dying
        where a kill of hers was placed (ledger.dying)."""
        s = self._last
        if s is not None and s.where is not None and s.platform is not None:
            now = time.monotonic()
            path = in_path(s, side, [m for m in s.monsters if not self._dying(m, now, s.monsters)])  # (no star: dying)
            if path:
                return next((m for m in path if (left := self.reckon(m)) is None or left.dead < DEAD_LIKELY), path[0])
            x = s.where.x
            recent = [mk for mk in self.marks if now - mk.thrown < HIDDEN_FOR and (mk.m.x > x) == (side == RIGHT)
                      and not self._dying(mk.m, now)]
            alive = [mk for mk in recent if (left := self.reckon(mk.m)) is None or left.dead < DEAD_LIKELY]
            if recent:  # (as the path: one not likely dead first, the likeliest if all are)
                return max(alive or recent, key=lambda mk: mk.thrown).m
        return None

    def in_way(self, side: str, kinds: set) -> str | None:
        """The kind of a monster a throw that way would take (`aimed`, its target, its path in the last look) that
        `kinds` names (None in it: any kind), or None: what drops a throw at its last look before the key (a kill of
        that kind come in since: Hunter._still)."""
        if not kinds:
            return None
        s = self._last
        ways = [m for m in (self.aimed, self.target(side)) if m is not None]
        if s is not None and s.where is not None and s.platform is not None:
            ways += in_path(s, side, s.monsters)
        return next((m.kind for m in ways if None in kinds or m.kind in kinds), None)

    def reckon(self, m: Monster) -> Left | None:
        """What monster `m` has left as she reckons it (see the module's docstring), her throws at it that no look
        has seen land yet taken off; None for a kind whose HP isn't known."""
        stats = self.stats.get(m.kind)
        if stats is None:
            return None
        mk = self._mark_of(m)
        if mk is None:
            return Left.of(m.bar * stats[0] if _believable(m.bar) else stats[0], stats[1])
        left = mk.left
        for _, a in mk.pending:
            left = left.after(a, choice.MISS)
        return left

    def left(self, m: Monster) -> float:
        """The HP monster `m` has left on average as she reckons it if it's alive (`reckon`); inf for a kind whose HP
        isn't known."""
        left = self.reckon(m)
        if left is None:
            return float("inf")
        return (left.alive() or left).hp

    def kill_s(self, m: Monster) -> float:
        """s she takes to kill monster `m` from what it has left if alive (`reckon`), with the attack she'd throw
        (Lucky Seven while she has the MP for a cast in a mode that throws it), her misses (choice.MISS), her throws
        THROW_S apart."""
        left = self.reckon(m)
        attack = self.skill if (self.skill is not None and self.mode != "regular" and
                                (self.mp is None or choice.can_cast(self.mp, self.max_mp, self.skill.mp))) \
            else self.regular
        if left is None:  # an unknown kind: three regular stars
            return 3 * choice.REGULAR.mean() / attack.mean() / (1 - choice.MISS) * choice.THROW_S
        return (left.alive() or left).throws(attack) / (1 - choice.MISS) * choice.THROW_S

    def summary(self) -> str:
        """Her throws by attack and why, for the end of a run's log ("" without a choice asked: mode 'regular')."""
        c = self.counts
        if not c:
            return ""
        whys = ", ".join(f"{k} {c[k]}" for k in ("finish", "MP", "nothing in its way") if c[k])
        warned = c["MP 0 under the warning"]
        return (f"Lucky Seven {c['lucky_seven']}, the regular attack {c['attack']}" + (f" ({whys})" if whys else "")
                + (f"; her MP bar's 0 under her pet's warning not believed {warned} times" if warned else ""))

    def _dying(self, m: Monster, t: float, seen: list[Monster] = ()) -> bool:
        """Whether `m` is taken for one dying where a kill of hers was placed (ledger.dying), at t, seen with `seen`."""
        return dying(self._kills(), m, t, seen, self._die)

    def _mp_read(self, s: Scene) -> float | None:
        """Her MP bar's share as look `s` read it, or None: not read, or a 0 under her pet's warning (within WARNING_S
        of a read above 0, it or her MP as reckoned WARNING_STEPS px or less), counted once a run of them ('MP 0 under
        the warning': live the bar read 0 on up to 6 looks in a row, 10 a second)."""
        if s.mp is None:
            return None
        if s.mp > 0.0:
            self._read, self._warned = (s.t, s.mp), False
            return s.mp
        if self._read is not None and s.t - self._read[0] <= WARNING_S \
                and min(self._read[1], 1.0 if self.mp is None else self.mp) <= WARNING_STEPS / choice.MP_STEPS + 1e-9:
            if not self._warned:
                self.counts["MP 0 under the warning"] += 1
            self._warned = True
            return None
        self._warned = False
        return s.mp

    def _believed(self, read: float, t: float) -> float:
        """Her MP from bar share `read` off a frame shown at `t`: less her casts keyed after t - MP_FIRST_S (it can't
        show them yet), and her MP as reckoned where that's within what the casts keyed from t - MP_SHOWS_S to then
        (shown or not) leave of it; past those bounds, the bound (a drink, regen or a press that took no MP shows,
        less a cast at most; a cast her reckoning missed, at once)."""
        self._casts = [(at, mp) for at, mp in self._casts if at > t - MP_SHOWS_S]
        hi = max(0.0, read - sum(mp for at, mp in self._casts if at > t - MP_FIRST_S))
        lo = max(0.0, hi - sum(mp for at, mp in self._casts if at <= t - MP_FIRST_S))
        return hi if self.mp is None else min(max(self.mp, lo), hi)

    def _mark_of(self, m: Monster) -> Mark | None:
        return next((mk for mk in self.marks if mk.m is m or seen_again(mk.m, [m]) is m), None)

    def _say(self, attack: Attack, why: str, target: Monster | None, left: Left | None, alive: Left | None,
             hp: float) -> None:
        mp = None if self.mp is None else max(0.0, choice.mp_points(self.mp, self.max_mp))
        if target is None:
            what = "nothing seen in its way"
        elif left is None or alive is None:
            what = target.kind
        else:
            what = f"{target.kind} ~{alive.hp:.0f}/{hp:.0f} HP" + (f" if alive ({left.dead:.0%} dead)"
                                                                   if left.dead >= 0.05 else "")
        reason = {"MP": f"MP {mp:.0f} < a cast's {self.skill.mp:.0f}" if self.skill is not None and mp is not None
                  else "MP", "finish": "finishing it", "damage": "", "skill": "", "regular": "",
                  "nothing in its way": ""}.get(why, why)
        self.log(f"    {attack.name}: {what}" + (f"; {reason}" if reason else "")
                 + ("" if mp is None or why == "MP" else f" (MP {mp:.0f})"))
        if telemetry.recording():  # (where its target was seen too: sim.harness counts the throws at the dead)
            telemetry.emit("attack", key=attack.key, why=why, kind=None if target is None else target.kind,
                           left=None if alive is None else round(alive.hp),
                           dead=None if left is None else round(left.dead, 3), hp=hp,
                           mp=None if mp is None else round(mp),
                           platform=None if target is None else target.platform,
                           x=None if target is None else round(target.x, 1))


def _believable(bar: float | None) -> bool:
    """Whether an HP bar read is a share to go by: between empty and full (a bar is up only once she has hurt it, and
    one with no green, as a monster dies, reads None: perception.hud.hp_bar)."""
    return bar is not None and 0.0 < bar < 1.0
