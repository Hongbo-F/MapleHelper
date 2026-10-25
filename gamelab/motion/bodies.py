"""Bodies for touches: hers and the monsters', as boxes about their feet, and where hers goes over a planned move.

A monster hurts her when its body and hers overlap: none of the maps' monsters has an attack of its own, so the
client hurts by the body. So the planner (motion.flow's danger) checks a move by her box all along the way her feet
go, on the ground, in the air and on a rope, against each monster's box wherever it can be by then
(localization.monsters.MonsterTrack.reach), on any platform her box passes, not only along her own. Her box here is
the one the whole stack goes by: the danger of a place (attack.danger: who can touch her where she stands or hangs,
how far a fight keeps a monster off), a fight's escape (attack.stance) and a wait's walk (behavior.hunt) too.

Her box (HER_HALF, HER_TALL) is the v83 client's for her standing, Box(-22, -65, 22, 0) from her feet (RustMS's port of
the decompiled client, Gameplay/MapleMap/MobAttackRules.h, `user_body`, from the client's FUN_0045183b; prone it's
(-46, -31, 0, 0), not used here). Her drawn sprites agree, without hair or equipment: standing -22 to 13 from her feet
facing left and 64 tall, 62 jumping, 60-65 on a rope (Character.wz's body and head frames have no boxes of their own:
D:/MapleHelper-scratch/safe-travel/char_extent.py). HeavenClient's touch test (MapMobs::find_colliding) takes her as a
line 50 tall, from where she was a frame ago to where she is, with no width; the simulator's world has her 15 either
side and 55 tall (sim.world.HER_HALF, HER_HEIGHT; the hop was timed in game with the 15: flow.HOP_HALF), the client's
with `--world client-body`. In game on
2026-10-24 the one touch with her on a rope (15:17:49, climbing 11 -> 10 on Tree Dungeon, Forest Up North II) came with
her feet 41-58 under platform 10's surface by her minimap dot (16 units a pixel), the monster unseen: her body
reaches 41 above her feet at least. Nothing logged tells 50, 55 and 65 apart (a MISS leaves no trace, and few hits had
a look just before them); `gamelab watch` during play would. Taller is the safe side for keeping clear.

A monster's box is the widest and tallest of its kind's frames (localization.monsters.touch_box, kept on each
MonsterTrack: `half`, `tall`): half its widest stand or move frame (MobBody.half, or the first map's tables) and the
tallest top of its walking and its flinch after a hit (MobBody.flinch_top, the tables' MONSTER_HEIGHT): a Horny or
Green Mushroom 28 either side and 56 tall (52 walking), a pig 34 and 54. A slime's hop (66 tall, 10-17 off the ground)
isn't in it.
"""

from dataclasses import dataclass

from ..localization.mapdata import Platform

HER_HALF = 22.0   # half her body's width (see the top: the client's box)
HER_TALL = 65.0   # how tall she is, from her feet up


def touches(x: float, y: float, lo: float, hi: float, half: float, feet: float, top: float) -> bool:
    """Whether her box, her feet at map (x, y), overlaps a monster's whose middle may be anywhere from map x `lo` to `hi`
    with its feet at map y `feet`, `half` wide either side and `top` tall (map y down; overlap strict, as the client's
    rectangles and the simulator's world have it)."""
    return x - HER_HALF < hi + half and x + HER_HALF > lo - half and y > feet - top and y - HER_TALL < feet


def low_end(p: Platform) -> float:
    """The lowest map y (the largest) of platform `p`'s surface (Platform.top is the highest)."""
    return max(max(s.y1, s.y2) for s in p.segments)


@dataclass(frozen=True)
class Arc:
    """Her feet in the air between two moments of a plan (s): from (x0, y0) to (x1, y1), leaving at `vy` (units/s, y
    down: a jump's -555, a fall's 0), drawn by gravity and capped at the fall speed (the game's, physics). Sideways she
    goes in a straight line from x0 to x1 (the planner's flight lands her at x1 at t1: physics.flight, with the arrow
    held her speed stays within 9 units/s of what she left with)."""
    t0: float
    t1: float
    x0: float
    y0: float
    x1: float
    y1: float
    vy: float

    def at(self, t: float, gravity: float, fall: float) -> tuple[float, float]:
        """Where her feet are at t (clamped to the arc's time): sideways in line, down by `gravity` (units/s²) up to
        `fall` (units/s), and never past the landing's height on the way down."""
        tau = min(max(t, self.t0), self.t1) - self.t0
        span = self.t1 - self.t0
        x = self.x0 + (self.x1 - self.x0) * (tau / span if span > 0 else 1.0)
        cap = (fall - self.vy) / gravity  # s until she falls at the cap
        if tau <= cap:
            y = self.y0 + self.vy * tau + gravity * tau * tau / 2
        else:
            y = self.y0 + self.vy * cap + gravity * cap * cap / 2 + fall * (tau - cap)
        going_down = self.vy + gravity * tau > 0
        return x, (min(y, self.y1) if going_down else y)


@dataclass(frozen=True)
class Climb:
    """Her feet on a rope at map x `x` between two moments of a plan: from y0 to y1, steadily (flow.ROPE_SPEED)."""
    t0: float
    t1: float
    x: float
    y0: float
    y1: float

    def at(self, t: float) -> tuple[float, float]:
        span = self.t1 - self.t0
        k = min(1.0, max(0.0, (t - self.t0) / span)) if span > 0 else 1.0
        return self.x, self.y0 + (self.y1 - self.y0) * k
