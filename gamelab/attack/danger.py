"""How badly she can be touched at a place: the danger of standing on a platform at an x over the next moments, from
the monsters she sees and from the map's spawn points. The hunt loop chooses where to shoot a ground from and where to
wait for respawns by it (attack.spots, behavior.hunt), and a fight how near it lets a monster on her platform come
(attack.stance, `keep_off`). Pure, from the map's game data (localization.mapdata) and her HP and level (`Her`).

A monster counts by
- what a touch by its kind takes (`Threat.share`): its touch damage (PADamage, game data) as meowdb's formula has it
  (`touch_taken`, at her level and DEFENCE), a share of her HP. A kind under DANGER_MIN counts for nothing: at level 19
  a pig takes 1.7% of her 642 HP, so on the first map nothing does, and every decision there is as it was;
- whether it can get to her (`walk`): along its own platform (a monster never leaves it: localization.monsters), and
  onto another only if its kind hops (MapModel.mob_hops), off its platform within HOP_ACROSS across and HOP_UP up of
  hers (from one over hers by an end: it can't drop through its own), HOP_CHANCE as likely. One walking a platform
  just under or over hers touches her where their bodies overlap (hers HER_HALF either side of her feet and HER_TALL
  tall, its own half its width and its top: game data), as one on hers does;
- how soon (`soon`): the walk to where it would touch her, at its speed (game data), weighed exp(-t / THREAT_S): 1 for
  one touching her now, as a respawn on top of her does, about a third for one THREAT_S away.
A spawn point counts as its monster at the point (`at` with no look: the picture she plans by), or, with the monsters a
look sees (`seen`), as its monster to come: of each kind's spawn points on a platform, as many as the monsters of
that kind seen there don't account for are empty, and each may fill at any moment (on Tree Dungeon the live respawns
on 23 came 1.7-57 s after her last kill there, a median of about 17 s: not on behavior.hunt's RESPAWN).

The danger at a place is the sum over them: the share of her HP each one's touch takes times how likely it comes
soon. Places are compared by it, never a kind or a map by name.

A rope or ladder is a place too (`on_rope`: her feet at a height on it): a monster that walks its floor touches her
there only where its body, up to the highest its frames reach (MapModel.mob_tall: a slime's hop), meets hers; a
flier (MapModel.mob_flies: none on her three maps) anywhere, straight at her. Where on a rope she hangs and what it
takes to get on and off is attack.ropes'.
"""

import math
from collections import Counter
from dataclasses import dataclass

from ..localization.mapdata import MapModel, Platform, Rope
from ..localization.monsters import DEFAULT_SPEED, Monster, bodies, speeds
from ..motion.flow import HER_HALF
from ..perception.kinds import kind_of

# Her DEF in meowdb's touch formula (touch_taken): not known; this is what makes the formula give the first map's 100
# logged touches their 9.56 HP on average at level 19 (sim.world.DEFENCE, which the simulator's touches use). Taken as
# hers since: on Tree Dungeon it gives a Horny Mushroom 111-131 HP of her 752, and the live hunts of 2026-10-24 took
# 113-134 a touch (15-18% of her HP bar).
DEFENCE = 75.0
# A kind whose touch takes less than this share of her HP is no danger worth giving up reach, a throw or a step for:
# her pet drinks a potion for her under 45%, and the first map's golden traces were made with every kind so (by the
# formula a pig takes 1.7% of her 642 HP at level 19, an Orange Mushroom 2.0%, a slime 1.2%). Of the other maps' kinds,
# a Horny Mushroom takes 16% of her 752 HP at level 24, a Green Mushroom 7.2%, an Axe Stump 10.8% of her 730 at 23;
# a Dark Stump 4.3% and a Slime 1.2% don't count.
DANGER_MIN = 0.05
# s: how soon a monster gets to her counts by exp(-t / THREAT_S). About two of her throws (her attack lock 0.73 s and
# its margin: a throw every 0.79-0.85 s): a monster further off than that takes them first, with Lucky Seven two
# thirds of a Horny Mushroom's kill (3.85 casts live on 2026-10-24) and most of a Green's (2.95), and a star pushes it
# back (two in three of a Horny's, every one a Green's: sim.world.PUSH_BY_DAMAGE). At 100 units/s a Horny Mushroom
# 100 units off counts 0.71, 200 off 0.39, 400 off 0.12.
THREAT_S = 1.7
HER_TALL = 55.0     # her body's height (the simulator's: sim.world.HER_HEIGHT): a monster on a platform this far
                    # above hers at most, or as far below as its top, touches her where they overlap
# A kind that hops (MapModel.mob_hops) can leave its platform onto one whose end is within HOP_ACROSS across and
# HOP_UP up of where it is (or any way down); how often it does isn't measured (pigs were seen on platforms where
# none spawn: localization.monsters): HOP_CHANCE, a guess. No kind that hops passes DANGER_MIN on the three maps.
HOP_ACROSS = 60.0
HOP_UP = 60.0
HOP_CHANCE = 0.25
# A fight lets a monster on her platform come as near as where its danger reaches this (keep_off): then she jumps back
# or steps away (attack.stance). Half a Horny Mushroom's touch (8% of her HP): it walks at her 100 units/s, and this
# keeps it 162 units off, where a fix in game on 2026-10-24 had 170 for that kind alone (stance.KEEP_OFF); a Green
# Mushroom (7.2%) or an Axe Stump (10.8% at 62.5 units/s: 80 units) is let as near as the claw would swipe it, as
# before.
KEEP_DANGER = 0.08


@dataclass(frozen=True)
class Her:
    """What a touch is measured against: her HP and level (config.toml [character], hp and level), and her DEF."""
    hp: float = 752.0
    level: float = 24.0
    defence: float = DEFENCE


def touch_taken(pad: float, defence: float, level: float, u: float = 0.5) -> int:
    """The HP a touch by a monster of touch damage `pad` (PADamage) takes from her at `defence` and `level`, as
    meowdb's page on this game's formulas has it (meowdb.com/msclassic/formulas/damage-taken, read 2026-10-24): raw =
    PADamage x (1.2 + 0.2 u), taken = raw x (1 - DEF / (DEF + 300 + 2 raw)) - (DEF / 25 + level / 10), cut to whole
    HP, at least 1; `u` in [0, 1) places it in the roll (0.5: its middle). The simulator's touches are these
    (sim.world.touch_damage)."""
    raw = pad * (1.2 + 0.2 * u)
    taken = raw * (1 - defence / (defence + 300 + 2 * raw)) - (defence / 25 + level / 10)
    return max(1, math.trunc(taken))


@dataclass(frozen=True)
class Threat:
    """A kind as a danger to her: the share of her HP a touch by it takes (0: under DANGER_MIN), how fast it walks,
    half its width, how tall it is walking, whether it hops off its platform, whether it flies, and the highest its
    body reaches (a hop's frames too: MapModel.mob_tall)."""
    kind: str
    share: float
    speed: float
    half: float
    top: float
    hops: bool
    flies: bool = False
    tall: float = 0.0


def soon(t: float) -> float:
    """How much a monster `t` s from touching her counts (THREAT_S)."""
    return math.exp(-max(0.0, t) / THREAT_S)


class Danger:
    """The danger of places on one map, for her (`Her`): `at` a place, `keep_off` a kind on her platform."""

    def __init__(self, model: MapModel, her: Her | None = None):
        self.model, self.her = model, her or Her()
        body, speed = bodies(model), speeds(model)
        self.threats: dict[str, Threat] = {}
        for name, pad in model.mob_damage.items():
            kind = kind_of(name)
            share = touch_taken(pad, self.her.defence, self.her.level) / self.her.hp if pad > 0 else 0.0
            b = body.get(kind)
            top = b.top if b else 46.0
            self.threats[kind] = Threat(kind, share if share >= DANGER_MIN else 0.0, speed.get(kind, DEFAULT_SPEED),
                                        b.half if b else 34.0, top, model.mob_hops.get(name, False),
                                        model.mob_flies.get(name, False), max(top, model.mob_tall.get(name, top)))
        # The spawn points of kinds that count, and how many of each kind each platform has.
        self.points = [(pid, s.x, kind) for pid, s in model.spawn_points
                       if self.share(kind := kind_of(s.name or str(s.mob_id))) > 0]
        self.counts = Counter((pid, kind) for pid, _, kind in self.points)

    @property
    def any(self) -> bool:
        """Whether any kind of the map counts (DANGER_MIN): if none does, every place is as safe as any other."""
        return any(t.share > 0 for t in self.threats.values())

    def share(self, kind: str) -> float:
        """The share of her HP a touch by `kind` takes, 0 if it doesn't count (or isn't known)."""
        t = self.threats.get(kind)
        return t.share if t is not None else 0.0

    def keep_off(self, kind: str) -> float:
        """How near a monster of `kind` on her platform she lets it come (middle to middle) before she moves away: where
        its danger reaches KEEP_DANGER; 0 for a kind that never does (she lets it come to the claw's reach)."""
        t = self.threats.get(kind)
        if t is None or t.share <= KEEP_DANGER:
            return 0.0
        return t.half + HER_HALF + t.speed * THREAT_S * math.log(t.share / KEEP_DANGER)

    def walk(self, t: Threat, q: Platform, mx: float, p: Platform, x: float) -> tuple[float, float] | None:
        """How far a monster of threat `t` at map x `mx` on platform `q` walks before it touches her standing at `x` on
        `p` (0: it does already), and how likely it goes there (1 along its platform, HOP_CHANCE off it); None if it
        can't touch her there. Its middle keeps half its width inside its platform."""
        reach = t.half + HER_HALF
        if t.flies:  # no floor holds it: straight at her
            return _flight(t, mx, q.y_at(q.clamp(mx)), x, p.y_at(x)), 1.0
        lo, hi = q.left + t.half, q.right - t.half
        if hi < lo:
            lo = hi = (q.left + q.right) / 2
        if q.id == p.id or -HER_TALL < q.y_at(q.clamp(x)) - p.y_at(x) < t.top:
            a, b = max(lo, x - reach), min(hi, x + reach)
            if a <= b:
                return (0.0 if a <= mx <= b else min(abs(mx - a), abs(mx - b))), 1.0
        if not t.hops or q.id == p.id:
            return None
        # Off its platform: from where it is nearest hers, onto hers, then along it. Over hers it can't drop through its
        # own platform: it leaves by an end, the one of the shorter way that gets it onto hers.
        gap = max(p.left - q.right, q.left - p.right, 0.0)
        if gap > HOP_ACROSS:
            return None
        eq = q.clamp(p.clamp(mx))
        if gap == 0 and q.y_at(eq) < p.y_at(p.clamp(eq)):
            ends = [e for e in (q.left, q.right) if abs(p.clamp(e) - e) <= HOP_ACROSS]
            if not ends:
                return None
            eq = min(ends, key=lambda e: abs(mx - e) + abs(p.clamp(e) - e) + abs(x - p.clamp(e)))
        ep = p.clamp(eq)
        if q.y_at(eq) - p.y_at(ep) > HOP_UP:
            return None
        return max(0.0, abs(mx - eq) + abs(ep - eq) + abs(x - ep) - reach), HOP_CHANCE

    def of(self, kind: str, q: Platform, mx: float, p: Platform, x: float) -> float:
        """The danger to her at `x` on `p` from one monster of `kind` at `mx` on `q`."""
        t = self.threats.get(kind)
        if t is None or t.share <= 0:
            return 0.0
        way = self.walk(t, q, mx, p, x)
        if way is None:
            return 0.0
        d, chance = way
        return t.share * chance * soon(d / max(t.speed, 1.0))

    def at(self, pid: int, x: float, seen: list[Monster] | None = None) -> float:
        """The danger to her standing at map x on platform `pid`: from the monsters `seen` (a look's) and the spawn
        points they don't account for; with no look (None), from every spawn point's monster at its point."""
        p = self.model.platform(pid)
        return sum(w * self.of(kind, q, mx, p, x) for kind, q, mx, w in self._sources(seen))

    def on_rope(self, rope: Rope, y: float, seen: list[Monster] | None = None) -> float:
        """The danger to her hanging on `rope` with her feet at map y, from the same monsters and spawn points as `at`:
        a monster that walks its floor touches her there only if its body reaches hers (walk_rope)."""
        total = 0.0
        for kind, q, mx, w in self._sources(seen):
            t = self.threats.get(kind)
            if t is None or t.share <= 0:
                continue
            way = self.walk_rope(t, q, mx, rope, y)
            if way is not None:
                total += w * t.share * way[1] * soon(way[0] / max(t.speed, 1.0))
        return total

    def walk_rope(self, t: Threat, q: Platform, mx: float, rope: Rope, y: float) -> tuple[float, float] | None:
        """How far a monster of threat `t` at map x `mx` on platform `q` walks before it touches her hanging on `rope`
        with her feet at y, and how likely (1); None if it can't. Her body is HER_HALF either side of the rope and
        HER_TALL up from her feet; its, half its width either side of its middle (kept on its platform) and from its
        feet up to the highest its frames reach (Threat.tall: a slime's hop, 66, where it stands 42-44). A flier goes
        straight at her."""
        reach = t.half + HER_HALF
        if t.flies:
            return _flight(t, mx, q.y_at(q.clamp(mx)), rope.x, y), 1.0
        lo, hi = q.left + t.half, q.right - t.half
        if hi < lo:
            lo = hi = (q.left + q.right) / 2
        a, b = max(lo, rope.x - reach), min(hi, rope.x + reach)
        if a > b:
            return None
        floor = q.y_at(q.clamp(rope.x))
        if not (floor - t.tall < y and floor > y - HER_TALL):  # its top under her feet, or its feet over her head
            return None
        return (0.0 if a <= mx <= b else min(abs(mx - a), abs(mx - b))), 1.0

    def _sources(self, seen: list[Monster] | None):
        """The monsters a place's danger comes from, as (kind, platform, x, weight): the ones `seen` (a look's, of
        kinds that count, each 1) and the spawn points they don't account for (of each kind's on a platform, as many as
        the look's monsters of that kind there don't account for, each as likely); with no look (None), every spawn
        point's monster at its point."""
        model, fill = self.model, None
        if seen is not None:
            here = Counter()
            for m in seen:
                if m.platform is None or self.share(m.kind) <= 0:
                    continue
                here[(m.platform, m.kind)] += 1
                yield m.kind, model.platform(m.platform), m.x, 1.0
            fill = {key: max(0, n - here[key]) / n for key, n in self.counts.items()}
        for q, sx, kind in self.points:
            share = 1.0 if fill is None else fill[(q, kind)]
            if share > 0:
                yield kind, model.platform(q), sx, share


def _flight(t: Threat, mx: float, floor: float, x: float, feet: float) -> float:
    """How far a flier of threat `t` at map x `mx` over a floor at y `floor` goes straight at her (her feet at (x,
    feet)) before their bodies meet: middle to middle, less half its width and hers."""
    return max(0.0, math.hypot(mx - x, (floor - t.tall / 2) - (feet - HER_TALL / 2)) - t.half - HER_HALF)


# Danger by id(model) and her, with the model (kept, so its id can't go to another one while it's here).
_KEPT: dict[tuple[int, Her], tuple[MapModel, Danger]] = {}
MODELS_KEPT = 8


def of_map(model: MapModel, her: Her | None = None) -> Danger:
    """The map's Danger for her (made once per map and her)."""
    her = her or Her()
    kept = _KEPT.get((id(model), her))
    if kept is None or kept[0] is not model:
        if len(_KEPT) >= MODELS_KEPT:
            _KEPT.clear()
        kept = _KEPT[(id(model), her)] = (model, Danger(model, her))
    return kept[1]
