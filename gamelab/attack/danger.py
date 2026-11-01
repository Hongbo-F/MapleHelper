"""How badly she can be touched at a place: the danger of standing on a platform at an x over the next moments, from
the monsters she sees and from the map's spawn points. The hunt loop chooses where to shoot a ground from and where to
wait for respawns by it (attack.spots, behavior.hunt), and a fight how near it lets a monster on her platform come
(attack.stance, `keep_off`). Pure, from the map's game data (localization.mapdata) and her max HP, level and DEF
(`Her`: perception's, behavior.bootstrap).

A monster counts by
- what a touch by its kind takes (`Threat.share`): its touch damage (PADamage, game data) as meowdb's formula has it
  (`touch_taken`, at her level and DEFENCE), a share of her HP. A kind under DANGER_MIN counts for nothing: at level 19
  a pig takes 1.7% of her 642 HP, so on the first map nothing does, and every decision there is as it was;
- whether it can get to her (`walk`): along its own platform (a monster never leaves it: localization.monsters), and
  onto another only if its kind hops (MapModel.mob_hops), off its platform within HOP_ACROSS across and HOP_UP up of
  hers (from one over hers by an end: it can't drop through its own), HOP_CHANCE as likely. One walking a platform
  just under or over hers touches her where their bodies overlap (hers the client's box, HER_HALF either side of her
  feet and HER_TALL tall: motion.bodies, which her trips are planned by too; its own half its width and its top: game
  data), as one on hers does (`level`; her fights and the sweep's guards count such a one as on her platform:
  `beside`);
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
from ..localization.monsters import DEFAULT_SPEED, Monster, bodies, speeds, walk_edges
from ..motion.bodies import HER_HALF, HER_TALL
from ..perception.kinds import kind_of

# Her DEF in meowdb's touch formula (touch_taken): not known; this is what makes the formula give the first map's 100
# logged touches their 9.56 HP on average at level 19 (sim.world.DEFENCE, which the simulator's touches use). Taken as
# hers since: on Tree Dungeon it gives a Horny Mushroom 111-131 HP of her 752, and the live hunts of 2026-10-24 took
# 113-134 a touch (15-18% of her HP bar). Her HP number read off the rolling recordings exactly (2026-10-28: 178
# touches of 17 recordings, 10-24 23:50 to 10-28 12:16; D:/MapleHelper-scratch/safety3/design, formula.py) fits the
# formula with DEF about 64 at levels 24-26 (a Horny Mushroom 113-134, a Green 51-61: as seen) and about 111 at level 28
# (Horny 103-123, Green 44-53; seen 103-122 over 38 and 44-52 over 33): her DEF went up between 10-26 23:47 and 10-28
# 00:30, new gear presumably. Her DEF is perception's since (behavior.bootstrap: her stat window's W.DEF when it's
# read, else fitted to her touches); 75 is the positioning model's prior until one says otherwise, kept while her
# touches don't contradict it (bootstrap.PLAN_DEF), and the simulator's DEF (sim.world: its touches). The safety stop
# takes none of it on trust: until her DEF is read or fitted it takes a touch at DEF 0, the most one can take
# (bootstrap.PRIOR_DEF).
DEFENCE = 75.0
TOP_U = 1 - 1e-9  # the roll's top (touch_taken's u is in [0, 1))
# A kind whose touch takes less than this share of her HP is no danger worth giving up reach, a throw or a step for:
# her pet drinks a potion for her under 70% (live on 2026-10-25: 6 Orange Potions in 10.95 minutes, each 230-250 HP of
# her 774, only under 0.70; sim.world.PET_HP), and the first map's golden traces were made with every kind so (by the
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
# A kind that hops (MapModel.mob_hops) can leave its platform onto one whose end is within HOP_ACROSS across and
# HOP_UP up of where it is (or any way down); how often it does isn't measured (pigs were seen on platforms where
# none spawn: localization.monsters): HOP_CHANCE, a guess. No kind that hops passes DANGER_MIN on the three maps.
HOP_ACROSS = 60.0
HOP_UP = 60.0
HOP_CHANCE = 0.25
# A fight lets a monster on her platform come as near as where its danger reaches this (keep_off): then she jumps back
# or steps away (attack.stance). Half a Horny Mushroom's touch (8% of her HP): it walks at her 100 units/s, and this
# keeps it 169 units off (162 with the simulator's 15-wide body for hers), where a fix in game on 2026-10-24 had 170
# for that kind alone (bc036b9's stance.KEEP_OFF, which this replaced); a Green Mushroom (7.2%) or an Axe Stump
# (10.8% at 62.5 units/s: 87 units) is let as near as the claw would swipe it, as before.
KEEP_DANGER = 0.08


@dataclass(frozen=True)
class Her:
    """What a touch is measured against: her max HP and level as perception read them (the HUD's HP number and "LV."
    digits: behavior.bootstrap, Hunter.her), and her DEF as estimated (her stat window's, else fitted to her touches,
    else DEFENCE). No defaults: none is hers but what was read (tests give theirs)."""
    hp: float
    level: float
    defence: float = DEFENCE


def touch_taken(pad: float, defence: float, level: float, u: float = 0.5) -> int:
    """The HP a touch by a monster of touch damage `pad` (PADamage, the game data's: a touch is a regular attack at its
    AttackRatio 100) takes from her at `defence` (her W.DEF) and `level`, as this game's client works it out: raw =
    PADamage x (1.2 + 0.2 u), taken = raw x (1 - DEF / (DEF + 300 + 2 raw)) - (DEF / 25 + level / 10), cut to whole
    HP, at least 1; `u` in [0, 1) places it in the roll (0.5: its middle). Two readings of the current client agree on
    it to the term: meowdb's damage-taken page (meowdb.com/msclassic/formulas/damage-taken, read 2026-10-24; saved
    2026-10-28 in D:/MapleHelper-scratch/touch/mech/web: "EffectiveMobAttack = trunc(MobAttack x AttackRatio / 100)",
    "FinalTaken = trunc(clamp(Taken, 1, 50,000,000))") and the osms datamine's (D:/MapleHelper-scratch/touch/mech/
    src_keep, ohmi69_tabs_formulas.js: "Re-derived from all three regular damage-taken paths in the current version";
    the older 1.1 + 0.4 u roll and DEF / (5 level + 200 + 1.2 raw) scale were the client before). Her 178 exact touches
    of 2026-10-24 to 28 fit it at one DEF a gear (behavior.bootstrap: 64 at levels 24-26, 111 at 28). The v83 client's
    (RustMS MobDamage.h: W.DEF against a job's standard) isn't this game's. No miss or guard here: what a touch takes
    when it lands. The simulator's touches are these (sim.world.touch_damage)."""
    raw = pad * (1.2 + 0.2 * u)
    taken = raw * (1 - defence / (defence + 300 + 2 * raw)) - (defence / 25 + level / 10)
    return max(1, math.trunc(taken))


def touch_range(pad: float, defence: float, level: float) -> tuple[int, int]:
    """The least and the most HP a touch by a monster of touch damage `pad` takes from her at `defence` and `level`:
    touch_taken at the roll's bottom and its top (TOP_U). A Horny Mushroom (PADamage 110) 110-131 at DEF 75 and level
    28, a Green Mushroom (54) 49-59."""
    return touch_taken(pad, defence, level, 0.0), touch_taken(pad, defence, level, TOP_U)


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


def level(t: Threat, q: Platform, p: Platform, x: float) -> bool:
    """Whether a monster of threat `t` walking platform `q` is level enough with her standing at map x on `p` for their
    bodies to meet where it comes by her: `q` is `p`, or its floor there (at x, or its end nearest x) is less than its
    top under her feet, or less than HER_TALL over them (hers the client's box, its from its feet to its top: game
    data). Where it comes within reach along its platform is Danger.walk's."""
    return q.id == p.id or -HER_TALL < q.y_at(q.clamp(x)) - p.y_at(x) < t.top


class Danger:
    """The danger of places on one map, for her (`Her`): `at` a place, `keep_off` a kind on her platform."""

    def __init__(self, model: MapModel, her: Her):
        self.model, self.her = model, her
        body, speed = bodies(model), speeds(model)
        self.threats: dict[str, Threat] = {}
        self.takes: dict[str, float] = {}  # kind -> the share of her HP its touch takes, under DANGER_MIN too (`taken`)
        for name, pad in model.mob_damage.items():
            kind = kind_of(name)
            share = touch_taken(pad, self.her.defence, self.her.level) / self.her.hp if pad > 0 else 0.0
            self.takes[kind] = share
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

    def taken(self, kind: str) -> float:
        """The share of her HP a touch by `kind` takes, whether it counts as a danger or not (a Slime's 1.2% on Tree
        Dungeon, under DANGER_MIN): what she loses, for the HP her stops cost (behavior.grind) and her landings' prices
        (behavior.fight.landing_prices). A kind the map's game data doesn't have (a track of no kind, '', which her
        landings' judge takes for a 'pig') the map's hardest-touching kind's, as the navigator's stand-in for the
        monster that touched her is (Navigator._toucher); 0 with none known. Until the review of 2026-11-01 0, so a
        meeting with one was priced at nothing (no shot first, no wait at a trip's last landing), where stance's flat
        MEET_PRICE_S had priced it."""
        share = self.takes.get(kind)
        return share if share is not None else max(self.takes.values(), default=0.0)

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
        can't touch her there. Its middle turns where the game turns it (walk_edges: WALK_EDGE from its platform's ends,
        a kind that hops at the ends themselves; until 2026-10-31 half its width inside them)."""
        reach = t.half + HER_HALF
        if t.flies:  # no floor holds it: straight at her
            return _flight(t, mx, q.y_at(q.clamp(mx)), x, p.y_at(x)), 1.0
        lo, hi = walk_edges(q, t.hops)
        if level(t, q, p, x):
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

    def beside(self, m: Monster, p: Platform, x: float) -> bool:
        """Whether monster `m`, on another platform than `p`, touches her standing at map x on `p` as one on `p` does
        (`walk`'s rule: `level`, its platform's stretch coming within its body and hers of x): a kind that counts and
        doesn't fly, walking a platform just under or over hers. Her fights count such a one as on her platform (her
        full box against its: attack.stance's `mine`, the hunt loop's rule 1, the sweep's guards: behavior.hunt), as
        `at` counts its danger. Live-like in the simulator's grind on Tree Dungeon (2026-10-31: D:/MapleHelper-scratch/
        replan/pos) the touches from a monster on the platform under or over hers went from 11 to 23 a run of 5 minutes
        with the loot sweep, on 9 between 8 and 10 above all (40-50 apart: every place on it 0.32-0.57), where her fights
        went by her own platform's alone. False for one on `p`, one not placed, or a platform not known."""
        if m.platform is None or m.platform == p.id:
            return False
        t = self.threats.get(m.kind)
        if t is None or t.share <= 0 or t.flies:
            return False
        q = p.peer(m.platform)
        if q is None:
            try:
                q = self.model.platform(m.platform)
            except (KeyError, IndexError):
                return False
        return q.left - HER_HALF <= x <= q.right + HER_HALF and level(t, q, p, x)

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
        straight at her. Its middle turns as in `walk`."""
        reach = t.half + HER_HALF
        if t.flies:
            return _flight(t, mx, q.y_at(q.clamp(mx)), rope.x, y), 1.0
        lo, hi = walk_edges(q, t.hops)
        a, b = max(lo, rope.x - reach), min(hi, rope.x + reach)
        if a > b:
            return None
        floor = q.y_at(q.clamp(rope.x))
        if not (floor - t.tall < y and floor > y - HER_TALL):  # its top under her feet, or its feet over her head
            return None
        return (0.0 if a <= mx <= b else min(abs(mx - a), abs(mx - b))), 1.0

    def free_points(self, pid: int, seen: list[Monster]) -> list[tuple[float, str]]:
        """The spawn points of platform `pid` (of kinds that count) the monsters `seen` don't account for, as (map x,
        kind): of each kind's there, as many as the look's monsters of that kind on it don't, the nearest to none of them
        first (_sources' share, here whole points: where a respawn can land on her)."""
        out = []
        for kind in {k for q, _, k in self.points if q == pid}:
            pts = [x for q, x, k in self.points if q == pid and k == kind]
            mons = sorted(m.x for m in seen if m.platform == pid and m.kind == kind)
            free = list(pts)
            for mx in mons:  # each seen monster accounts for its nearest point
                if not free:
                    break
                free.remove(min(free, key=lambda x: abs(x - mx)))
            out += [(x, kind) for x in free]
        return sorted(out)

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


def of_map(model: MapModel, her: Her) -> Danger:
    """The map's Danger for her (made once per map and her)."""
    kept = _KEPT.get((id(model), her))
    if kept is None or kept[0] is not model:
        if len(_KEPT) >= MODELS_KEPT:
            _KEPT.clear()
        kept = _KEPT[(id(model), her)] = (model, Danger(model, her))
    return kept[1]
