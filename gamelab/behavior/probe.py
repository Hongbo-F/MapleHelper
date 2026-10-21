"""In-game probes of what a star hits (`python -m gamelab probe`): each stands her where one open question
of attack.reach shows, throws only when a single monster can take the star, and logs every throw so the
question can be settled from the log alone (`score`).

The questions (attack.reach, gamelab/README.md): the target search's window, its top (SEARCH_TOP0: the
client's 44, as a port of the v83 client and meowdb have it; 21 fitted to the logs) and its floor by kind,
and the search between her level and 60 below her (against the old ramp her decisions keep to there,
RAMP_BELOW); whether an attack from a jump picks by the area or by the search; what stops a star (the walls
of her own foothold group where they cross a level line SHOT_H above her feet, against a line from her to
the target's feet, or every wall); and how far the claw swipes, on her level (CLAW_FAR, CLAW_MARGIN) and
off it (claw_takes: its stab boxes). A `Probe` is a place on Thicket Around the Beach II (the platform she
stands on, her x, the way she throws), the monsters it throws at (their kinds, their platforms, how far off
sideways by their middle), and how she throws: from the ground, a jump throw or a jump back, always with
the regular attack.

A run (`run`) sends her to her place (the navigator, keeping clear of monsters) and keeps her there. After
every look she throws only when exactly one monster is in front of her (`in_front`: within FRONT by its
middle, from FRONT_UP above her feet to FRONT_DOWN below, all an attack could pick) and it's one the
probe throws at, its middle in the probe's range sideways (`crowd`: or with that many others in front
too). After a throw she neither moves nor throws until its star's evidence is in: every look up to
EVIDENCE_S after the star was due (longer while the target is hidden, up to HIDDEN_MAX_S: a hit monster
flinches under its damage numbers for 1-3 s). So one throw's evidence is never another's, and a monster
hit by one has stopped flinching (0.6 s, taller) when the next attack picks. A monster of a kind the
probe doesn't throw at that comes close on her platform is fought off first (behavior.fight.clear, her
platform only); knocked off her place, she walks or goes back to it, the last walk the way she throws, so
that no throw turns her (a turn's tap walks her on). Nor does she walk or throw from a look a touch's knock
may still be moving her in (Hunter.knocked). What her decisions take as stopping
a star (reach.WALLS_BLOCK: the counting of who's in front, and the fights) is switched for the run: off
for the probes that measure it, so it throws at monsters behind such a wall too.

Each throw is a line of lab/probe_geometry.jsonl: the probe, how she threw (and the attack key's delay
after the jump key, `gap_ms`), whether the wall rule was on, where she stood and her foothold group, the
target, every monster in front at the throw (`front`, walls or not) and at the first look after it, about
when the attack picked its target (`pick`: that look's frame is taken once the throw's key plan is done,
about when the attack starts, ~67 ms after its key), the rule's verdict on the target (`clear`), her EXP
gains from the key to the end (`exp`), and for every monster in front then its `evidence`: its HP bar
before the star could land, every bar seen after (with when, from the star's due time), the looks it
wasn't seen in, and whether it was still there at the end.

The scorer (`score`) reads the log with the map's game data. What each monster showed (`shown`): a hit (its
HP bar new or lower after the star was due, or killed: her EXP up and it gone), none (seen with its bar as
before), or can't tell (hidden). Then:
- the window's top and floor (SEARCH_TOP0, SEARCH_FLOOR0), fitted to the throws from the ground: for each
  value, which monster each attack takes (`taken`: the claw first where a stance's box reaches, then the
  search's first band, the nearest in it, by the positions at the pick and each kind's walking body (a
  probe throws at a lone monster, so the claw's stance and the order within a band rarely matter); none
  behind the rule's walls), scored by likelihood with its two hit rates fitted (a monster taken shows a hit
  with chance a, one not taken with chance f), as the research fitted the shot log and the telemetry;
- what stops a star: the rule against no walls, every wall at the same line, and a line from her to the
  target's feet through her group's walls, over every throw, with what each attack picks (the search
  from the ground, the area from a jump); and the line's height (SHOT_H), raised in a jump (reach.air_rise);
- between her level and 60 below her: the search alone against the search with the old ramp (none closer by
  its middle than reach.ramp_dx);
- from a jump: the area (reach.in_area) against the search, from as high as she is then and from the ground
  she jumped from, and the hits by how far off the target was;
- the claw: the claw probes' hits by how far off the target was, each the claw's or a star's by how much of
  its bar it took (under CLAW_DROP: the claw's).
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT

from .. import directions, paths, telemetry
from ..attack import reach
from ..attack.stance import JUMP_BACK_SPAN, open_above, pct
from ..control import client
from ..control.actions import JB_RUN_MS, JB_THROW_MS, JB_TURN_MS, STAR_DELAY_MS, TAP_MS
from ..localization.mapdata import MapModel, MobBody
from ..localization.monsters import Monster, bodies, same
from ..localization.scene import Scene
from ..motion import body, physics
from .fight import GUARD_PATIENCE, GUARD_SECONDS, IN_HER_WAY, UNPLACED_LOOKS

if TYPE_CHECKING:
    from ..control.navigator import Navigator
    from .hunter import Hunter

FRONT = 420          # map units: a monster this far in front of her by its middle could take the star (a pig's
                     # body reaches into the search out to 434) ...
FRONT_UP = reach.LOOK_UP  # ... from this far above her feet (140: the search's highest window reaches 124, 101
                     # with the logs' lower top, and a jump's area 65 above her as high as 49 up in the jump) ...
FRONT_DOWN = 140     # ... to this far below (the search's lowest floor is 68 below her feet, and the tallest body
                     # here, an Orange Mushroom, 67). A look searches only LOOK_DOWN (75) below her, unless wide.
STAND_SLACK = 20     # she walks back to her place when she's farther off it than this (a jump back carries her 85)
APPROACH = 12        # ... or faces the other way: then she comes at it walking the way she throws, from at least
                     # this far behind it (a 67 ms hold: no tap short enough to be lost, under 50 ms)
EVIDENCE_S = 0.9     # s after a star was due that a throw's evidence is gathered (as fight._score_shots: a miss) ...
HIDDEN_MAX_S = 2.5   # ... and up to this long while its target is hidden (under its damage numbers: 1-3 s)
EARLY_S = 0.15       # a target seen this long before its star was due, or later, has been seen since (`over`)
BEFORE_HIT_S = (STAR_DELAY_MS - 50) / 1000  # s after the attack key a look still sees every bar as before the
                     # throw: its damage shows STAR_DELAY_MS (520) after the key at the earliest
SETTLED_S = 0.3      # a monster seen this long after the star was due with its bar as before wasn't hit
HURT = 0.02          # a bar this much lower than before is a hit
WAIT_S = 0.1         # s between looks while there's nothing to throw at
STYLES = ("throw", "jump_throw", "jump_back")  # from the ground, a jump throw, a jump back (control.Actions)
# The scorer's grids: the first window's top and floor (above her feet), the line's height.
TOP0_GRID = range(0, 81)
FLOOR0_GRID = range(-30, 41)
SHOT_H_GRID = range(0, 101)
LOGS_TOP0 = 21       # the window's top the shot log and the telemetry fit (reach has the client's, 44: 28 + 16)
FITS = 1.92          # values within this much log-likelihood of the best fit (95%, one parameter)
# The two hit rates the likelihood fits are kept to what the logs have measured, or a model saying "not
# taken" would explain a run of hits as well as one saying "taken" (most probe throws have one monster):
A_MIN = 0.3          # a monster the attack takes shows a hit at least this often (72.5% of the shot log's 204
                     # throws hit; the telemetry missed about half the bars)
F_MAX = 0.2          # one it doesn't take shows one at most this often (the telemetry's monsters behind her,
                     # which no star can hit: 7.6% of 224)
RATE_FLOOR = 0.02    # and neither rate quite 0 or 1
CLAW_DROP = 0.15     # a hit that takes less of its target's HP bar than this was the claw's: 9-11 damage against a
                     # star's 30-70, of a pig's 128 HP (game data) 7-9% against 23-55%, of a slime's 115 8-10%
                     # against 26-61%


@dataclass(frozen=True)
class Probe:
    """Where she stands, what she throws at and how: one open question of attack.reach."""
    name: str
    asks: str               # the question it settles (`probe list`, the log)
    platform: int           # she stands on this platform ...
    x: float                # ... at this map x ...
    side: str               # ... and throws this way
    at: tuple[int, ...]     # at monsters on these platforms ...
    kinds: tuple[str, ...]  # ... of these kinds ...
    dx: tuple[float, float]  # ... their middle this far off sideways (map units, from her middle)
    by_kind: tuple = ()     # (kind, (lo, hi)): a kind whose range differs from dx
    style: str = "throw"    # how she throws (STYLES)
    throws: int = 20
    wide: bool = False      # looks over the whole screen: the targets stand above what a look's band searches
    walls: bool = True      # whether her decisions keep to the wall rule during the run (off: it's measured)
    expect: str = ""        # what each model predicts
    source: str = ""        # the research's name for it

    def range_of(self, kind: str) -> tuple[float, float]:
        """How far off sideways (by its middle, either way) a monster of `kind` is thrown at."""
        return dict(self.by_kind).get(kind, self.dx)


PROBES = {p.name: p for p in (
    Probe("top-60", "the search window's top (reach.SEARCH_TOP0)", 13, -14.0, LEFT, (12,),
          ("pig", "orange-mushroom"), (110, 200), throws=30, source="geometry P1",
          expect="60 above her, a pig is hit from 111 units with the client's top (44, reach's), from 191 with the "
                 "logs' fitted one (21); an Orange Mushroom from 113 or 193"),
    Probe("top-120", "the window's top, two platforms up", 15, 76.0, LEFT, (12,), ("pig", "orange-mushroom"),
          (340, 420), wide=True, source="geometry P2",
          expect="120 above her: hits from about 350 units with the client's top (44), none with the logs' (21)"),
    Probe("floor", "the window's floor by kind (reach.SEARCH_FLOOR0)", 14, 824.0, RIGHT, (17,),
          ("orange-mushroom", "pig", "slime"), (90, 200),
          by_kind=(("orange-mushroom", (90, 150)), ("pig", (120, 190)), ("slime", (130, 200))), throws=40,
          source="geometry P3",
          expect="60 below her, by either top: an Orange Mushroom hit from 113 units, a pig from 151 (171 standing), "
                 "a slime from 177 (closer while it hops)"),
    # Facing right from 22 nothing else is in front (24, 60 below, has no spawns; facing left, 21's pigs and
    # slimes 60 above took the stars). A snail closer than 33 touches her, and a touch's knockback (21-87 units)
    # from the right can take her off 22's left end into the basin, so she stands 69 from it and the range
    # starts past touching (--dx 15-30 asks for the rest).
    Probe("air", "what an attack from a jump picks: the area or the search", 22, 30.0, RIGHT, (22,), ("snail",),
          (35, 85), style="jump_throw", throws=40, source="geometry P4",
          expect="by the area (reach.in_area), hits at every distance; by the search, which starts 65 ahead, none "
                 "closer than 47 (a snail's far side short of 65)"),
    Probe("other-wall", "whether the side wall of another block stops a star", 19, 104.0, RIGHT, (18,),
          ("pig", "orange-mushroom"), (200, 260), source="geometry P5",
          expect="hits as on open ground: the wall of 18 that the line crosses is another foothold group's"),
    Probe("step-up", "whether her own group's step wall stops a star, and how high its line runs (reach.SHOT_H)",
          4, 500.0, LEFT, (2,), ("pig", "orange-mushroom"), (200, 400), throws=10, walls=False,
          source="terrain 1",
          expect="past the 60-unit step wall up to 2: from the ground (the line 28 up) and from jump throws (54 up) "
                 "no hits, from jump backs (72 up) hits; again with --x 260, and --style jump_throw, jump_back"),
    Probe("step-down", "a level line or a line to the target's feet, down past the step", 2, 50.0, RIGHT, (4,),
          ("pig", "orange-mushroom"), (200, 400), walls=False, source="terrain 2, geometry P5's second",
          expect="by the level line hits; by a line to the feet none from x 50 (it crosses the step wall about "
                 "30 down); again with --x 195 --dx 160-250: hits by either"),
    # 42 below her, between her level and the 60 the search was fitted to: 0 is 1's only such neighbour. Facing
    # left from 25 inside 0's left end, 1 runs on 336 units ahead of her (monsters on 1 can touch her on 0: their
    # bodies reach 38-55 up).
    Probe("below-42", "the search between her level and 60 below her, against the old ramp (reach.RAMP_BELOW)",
          0, 1222.0, LEFT, (1,), ("pig", "orange-mushroom", "snail"), (70, 160), by_kind=(("snail", (130, 220)),),
          throws=30, source="review of the search",
          expect="42 below her, by the search a pig is hit from 90 units (110 standing), an Orange Mushroom at "
                 "any (33 by its body), a snail from 166; by the old ramp every kind from 126. The shot log missed "
                 "pigs 124-125 off 3 times of 4 and an Orange Mushroom 91 off there"),
    # The claw's reach (reach.CLAW_FAR, CLAW_MARGIN), told by how much of the bar a hit takes (CLAW_DROP). On her
    # level, in the middle of 21 (two slimes and a pig spawn there; 174 units to either end, past a knockback).
    Probe("claw-edge", "how far off the claw still swipes, by kind (reach.CLAW_FAR, CLAW_MARGIN, claw_edge)", 21,
          -225.0, RIGHT, (21,), ("pig", "slime"), (95, 160), throws=40, source="bodies D",
          expect="by the client's stab boxes (cut to 65 ahead, reach's) and the bodies, swipes (a small bar drop: "
                 "9-11 damage of a pig's 128 HP) up to about 99 off a pig and 93 off a slime, stars from there (her "
                 "decisions count on a star from 111 and 105: claw_edge); by the swings' boxes (73-102 ahead, taken "
                 "as 90, as reach had it before), swipes out to 124 and 118"),
    # The claw off her level: the swings' boxes reach 17 below her feet, into a walking pig's or an Orange
    # Mushroom's body 60 below her (46 and 55 tall), not a slime's (41); the stabs' boxes, which the client
    # tests, reach down to 15 above her feet, into neither. So it tells the swings from the stabs, not the stabs
    # from a claw kept to her level. From 35 inside 3's left end, at 4.
    Probe("claw-down", "whether the claw swipes a monster off her level (reach.claw_takes: its stab boxes)", 3,
          716.0, LEFT, (4,), ("pig", "orange-mushroom"), (60, 125), source="bodies D",
          expect="by the stab boxes (reach's), no hit at a pig (the search picks one 60 below from 151) and stars "
                 "at Orange Mushrooms from 113; by the swings' boxes, small bar drops at both within about 124"),
    Probe("hanging-wall", "a level line or a line to the feet, down past her own block's hanging wall", 14, 730.0,
          RIGHT, (17,), ("pig", "orange-mushroom", "slime"), (180, 360), walls=False, source="terrain 3",
          expect="by the level line hits at every distance; by a line to the feet none from 198 units on (from x 730 "
                 "it crosses 14's own wall, 25 deep, at 198-374)"),
)}


def _jump_back_out() -> float:
    """How far a jump back has carried her when its attack picks its target (JUMP_LAG after the attack key), by the
    engine with the median lags: control.actions' plan (the arrow away, the jump JB_RUN_MS after it, the turn at
    JB_TURN_MS), replayed on flat ground."""
    events = [(0.0, "left", True), (JB_RUN_MS / 1000, "jump", True), (JB_TURN_MS / 1000, "left", False),
              (JB_TURN_MS / 1000, "right", True), ((JB_TURN_MS + TAP_MS) / 1000, "right", False)]
    picks = (JB_THROW_MS + physics.JUMP_LAG * 1000) / 1000
    return -body.replay(events, picks, physics.SHOE, physics.LAGS).points[-1][1]


# A jump back's target, as far off as a probe's range reaches at the look, is this much farther off when the attack
# picks (25 units, with the jump keyed 130 ms after the arrow), and it may walk on meanwhile: from 4 at 2 (step-up,
# 200-400 off) jump backs at Orange Mushrooms 387-394 off at the look were 434-445 off at the pick, past a star's
# reach (an Orange Mushroom's to 432 by its middle), and missed for that, not for the wall. So a jump back's range
# ends this much nearer.
JUMP_BACK_OUT = _jump_back_out()


def adjust(probe: Probe, *, x: float | None = None, dx: tuple[float, float] | None = None, style: str | None = None,
           throws: int | None = None, walls: bool | None = None) -> Probe:
    """`probe` with what's given changed (a range sideways given applies to every kind)."""
    changes = {k: v for k, v in (("x", x), ("style", style), ("throws", throws), ("walls", walls)) if v is not None}
    if dx is not None:
        changes.update(dx=tuple(dx), by_kind=())
    return replace(probe, **changes)


def problems(model: MapModel, probe: Probe) -> list[str]:
    """Why `probe` can't run on `model` as it is: her place off her platform, a jump under a platform (she'd
    land on it), no room behind her for a jump back."""
    out = []
    if probe.style not in STYLES:
        out.append(f"no such way to throw: {probe.style} (one of {', '.join(STYLES)})")
    if not 0 <= probe.platform < len(model.platforms):
        return out + [f"no platform {probe.platform} on map {model.id}"]
    p = model.platform(probe.platform)
    if not p.left <= probe.x <= p.right:
        out.append(f"x {probe.x:.0f} isn't on platform {p.id} ({p.left:.0f} to {p.right:.0f})")
    if probe.style != "throw" and not open_above(model, p, probe.x - STAND_SLACK, probe.x + STAND_SLACK):
        out.append(f"a platform overhead at x {probe.x:.0f}: a jump there lands on it")
    behind = (p.right - probe.x) if probe.side == LEFT else (probe.x - p.left)
    if probe.style == "jump_back" and behind < JUMP_BACK_SPAN + reach.EDGE_MARGIN:
        out.append(f"only {behind:.0f} units behind her at x {probe.x:.0f}: a jump back needs "
                   f"{JUMP_BACK_SPAN + reach.EDGE_MARGIN:.0f}")
    lo, hi = probe.dx
    if not 0 <= lo <= hi:
        out.append(f"the range sideways {lo:.0f}-{hi:.0f} isn't one")
    return out


def gap_ms(style: str) -> float | None:
    """How long after the jump key an attack of `style` is keyed (None: from the ground)."""
    return {"jump_throw": physics.JUMP_ATTACK_MS, "jump_back": physics.JUMP_BACK_ATTACK_MS}.get(style)


def rise_of(style: str) -> float:
    """How high she is when an attack of `style` starts (reach.air_rise; 0 from the ground)."""
    gap = gap_ms(style)
    return 0.0 if gap is None else reach.air_rise(gap)


@dataclass
class ProbeReport:
    """How a probe run went (its own reading of each throw's evidence; `score` reads the log)."""
    probe: str
    throws: int = 0
    hits: int = 0       # throws whose target showed a hit
    misses: int = 0     # ... showed none
    unknown: int = 0    # ... couldn't be told (hidden throughout)
    stop: str = ""

    def __str__(self):
        return (f"{self.probe}: {self.throws} throws, the target hit by {self.hits}, not by {self.misses}, "
                f"{self.unknown} can't tell; stopped: {self.stop}")


# --- a run ---------------------------------------------------------------------------------------------

def run(h: "Hunter", nav: "Navigator", probe: Probe, seconds: float = 900, crowd: int = 0) -> ProbeReport:
    """Run `probe` (see the module's docstring) until it has thrown probe.throws times, for at most
    `seconds`, or until her HP is too low (h.health_check) or she can't be placed or get back to her place.
    Every throw is logged to lab/probe_geometry.jsonl. The wall switch (reach.WALLS_BLOCK) is the probe's for
    the run, and put back after it, however it ends."""
    if trouble := problems(h.model, probe):
        raise ValueError(f"probe {probe.name}: {'; '.join(trouble)}")
    before, reach.WALLS_BLOCK = reach.WALLS_BLOCK, probe.walls
    h.actions.skill = False  # the regular attack: what every log measured, and no MP
    try:
        return _run(h, nav, probe, seconds, crowd)
    finally:
        reach.WALLS_BLOCK = before


def _run(h: "Hunter", nav: "Navigator", probe: Probe, seconds: float, crowd: int) -> ProbeReport:
    report = ProbeReport(probe.name)
    deadline = time.monotonic() + seconds
    flying: _Flying | None = None
    unplaced = 0
    out = JUMP_BACK_OUT if probe.style == "jump_back" else 0.0
    ranges = ", ".join(f"{k} {lo:.0f}-{hi - out:.0f}" for k in probe.kinds for lo, hi in (probe.range_of(k),))
    h.log(f"probe {probe.name} ({probe.asks}): on {probe.platform} at x {probe.x:.0f}, {probe.style} {probe.side} "
          f"at monsters on {', '.join(map(str, probe.at))} ({ranges} units off); the wall rule "
          f"{'on' if reach.WALLS_BLOCK else 'off'}")
    while flying is not None or (report.throws < probe.throws and time.monotonic() < deadline):
        # Waiting for a lone monster, she can look for minutes without a key, and a key plan is where the alarm
        # is noticed (client.send). A look brings the game to the front: none once the player has seen the
        # Lie Detector.
        client.check_alarm()
        s = h.look(wide=probe.wide)
        if flying is not None and s.where is not None:
            flying.see(s, h.exp_gains)  # in mid-jump too (a jump back's pick, and while she's in the air)
        if s.where is None or s.platform is None:
            unplaced += 1
            if unplaced >= UNPLACED_LOOKS:
                report.stop = "lost track of her"
                break
            if flying is None:
                time.sleep(0.2)
            continue
        unplaced = 0
        if stop := h.health_check(s):
            report.stop = stop
            break
        if flying is not None:
            if not flying.over(s.t):
                continue
            _done(h, report, flying)
            flying = None
            if report.throws >= probe.throws or time.monotonic() >= deadline:
                break
        if h.knocked(s):  # a touch's knock may still be moving her: neither a walk nor a throw from this look
            time.sleep(WAIT_S)
            continue
        if (stop := _to_place(h, nav, probe, s)) is not None:
            if stop:
                report.stop = stop
                break
            continue
        if _fight_off(h, probe, s):
            continue
        target = choose(s, probe, crowd)
        if target is None:
            time.sleep(WAIT_S)
            continue
        flying = _throw(h, s, probe, target)
        report.throws += 1
    if flying is not None:  # stopped while its evidence came in: kept as far as it got
        _done(h, report, flying)
    report.stop = report.stop or ("throws done" if report.throws >= probe.throws else "time limit")
    h.log(f"probe {report}")
    return report


def _to_place(h: "Hunter", nav: "Navigator", probe: Probe, s: Scene) -> str | None:
    """Get her back to her place, facing the way she throws: go there (the navigator) when she's on another
    platform; walk when she's more than STAND_SLACK off her x or faces the other way, the last walk toward the
    way she throws, from APPROACH behind her place at least. None: she's there; "": she moved (look again); else
    why she can't get there."""
    if s.platform.id != probe.platform:
        h.log(f"  on platform {s.platform.id}: going back to {probe.platform} at x {probe.x:.0f}")
        trip = nav.go(probe.platform, x=probe.x, scene=s)
        return "" if trip.result in ("arrived", "in position") else f"couldn't get to her place: {trip.result}"
    off = probe.x - s.where.x
    if abs(off) <= STAND_SLACK and h.actions.facing == probe.side:
        return None
    # Facing the way she throws, no throw turns her: the turn's arrow tap walks her on (7.4 units for 50 ms), and
    # within STAND_SLACK nothing walked her back. In the simulator from x 30 on 22 (`probe run air`), the jump
    # throws after the first came from x 38, 32 units from where the snail came back: inside the 33 it touches her
    # from, and a touch that wasn't evaded knocked her 45 back.
    ahead = off * directions.sign(probe.side)  # how far her place is ahead of her, the way she throws
    if ahead >= APPROACH:
        h.actions.walk(probe.side, ahead)
    else:  # past her place, at it or just short of it: back off first, to come at it the way she throws
        h.actions.walk(directions.opposite(probe.side), APPROACH - ahead)
    return ""


def _fight_off(h: "Hunter", probe: Probe, s: Scene) -> bool:
    """A monster the probe doesn't throw at close to her on her platform (within IN_HER_WAY): fought (a fight
    on her platform only, up to one kill). Whether there was one."""
    here, x = s.platform.id, s.where.x
    close = [m for m in s.monsters if m.platform == here and abs(m.x - x) < IN_HER_WAY
             and not (here in probe.at and m.kind in probe.kinds)]
    if not close:
        return False
    m = min(close, key=lambda m: abs(m.x - x))
    h.log(f"  {m.kind} close on her platform at {m.x - x:+.0f}: fighting it off first")
    h.clear(None, ground={here}, max_kills=1, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE, approach=False)
    return True


def in_front(s: Scene, side: str, rise: float = 0.0, walls: bool = True) -> list[Monster]:
    """The monsters of look `s` that could take a star she throws `side`: in front of her (a body reaching in
    front of her counts) within FRONT by their middle, from FRONT_UP above her feet to FRONT_DOWN below; with
    `walls`, none her decisions take as behind a wall of her foothold group (reach.wall_to, `rise` up in a
    jump; with reach.WALLS_BLOCK off, none is)."""
    x, y, d = s.where.x, s.where.y, directions.sign(side)
    out = []
    for m in s.monsters:
        ahead, half = (m.x - x) * d, reach.body_of(m.kind)[1]
        if not (-half < ahead <= FRONT and -FRONT_UP <= m.y - y <= FRONT_DOWN):
            continue
        if walls and reach.wall_to(s.platform, x, y, m, rise) is not None:
            continue
        out.append(m)
    return out


def choose(s: Scene, probe: Probe, crowd: int = 0) -> Monster | None:
    """What to throw at in look `s`, if anything: the nearest monster in front of her (in_front) that the probe
    throws at (its kind, its platform, its middle in range sideways: for a jump back, its far end JUMP_BACK_OUT
    nearer), when it's the only one in front (or with at most `crowd` others)."""
    front = in_front(s, probe.side, rise_of(probe.style))
    x = s.where.x
    out = JUMP_BACK_OUT if probe.style == "jump_back" else 0.0  # she's that much farther off when it picks
    fits = [m for m in front if m.kind in probe.kinds and m.platform in probe.at
            and probe.range_of(m.kind)[0] <= abs(m.x - x) <= probe.range_of(m.kind)[1] - out]
    if not fits or len(front) > 1 + crowd:
        return None
    return min(fits, key=lambda m: abs(m.x - x))


def _throw(h: "Hunter", s: Scene, probe: Probe, target: Monster) -> "_Flying":
    """Throw at `target` as the probe does, and start gathering the evidence."""
    row = _row(h, s, probe, target)
    exp0, keyed = h.exp_gains, []

    def on_throw(when: float, side: str) -> None:
        keyed.append(when)
    if telemetry.recording():
        telemetry.emit("stance", act=probe.style, text=f"probe {probe.name}: {probe.style} {probe.side}",
                       aimed=[{"kind": target.kind, "x": round(target.x, 1), "y": round(target.y, 1)}])
    h.actions.throw_listeners.append(on_throw)
    try:
        getattr(h.actions, probe.style)(probe.side)
    finally:
        h.actions.throw_listeners.remove(on_throw)
    h.report.throws += 1
    key = keyed[0] if keyed else time.monotonic()
    flight = abs(target.x - s.where.x) + (JUMP_BACK_SPAN if probe.style == "jump_back" else 0)
    due = key + COMBAT.attack_release_ms / 1000 + flight / reach.STAR_SPEED
    h.log(f"  {probe.style} {probe.side} at {target.kind} on {target.platform}: {target.x - s.where.x:+.0f},"
          f"{target.y - s.where.y:+.0f} (bar {pct(target.bar)}), {len(row['front'])} in front")
    return _Flying(row, s, target, key, due, exp0)


def _monster(m: Monster, target: bool = False) -> dict:
    """A monster as the probe log keeps it."""
    return {"kind": m.kind, "x": round(m.x, 1), "y": round(m.y, 1), "platform": m.platform, "bar": m.bar,
            **({"target": True} if target else {})}


def _row(h: "Hunter", s: Scene, probe: Probe, target: Monster) -> dict:
    """A throw's line in the probe log, as far as the look before it tells (the rest comes with _Flying)."""
    here, x, y = s.platform, s.where.x, s.where.y
    on = h.model.platform(target.platform) if target.platform is not None else None
    front = in_front(s, probe.side, walls=False)
    return {"probe": probe.name, "style": probe.style, "gap_ms": gap_ms(probe.style), "walls_block": reach.WALLS_BLOCK,
            "map": h.model.id, "side": probe.side, "x": round(x, 1), "y": round(y, 1), "her": here.id,
            "her_group": list(here.group) if here.group is not None else None,
            "target": _monster(target, True), "dx": round(target.x - x, 1), "dy": round(target.y - y, 1),
            "clear": reach.wall_between(here, x, y, target.x, rise_of(probe.style), target=on) is None,
            "front": [_monster(m, m is target) for m in front], "time": time.strftime("%Y-%m-%d %H:%M:%S")}


class _Flying:
    """A probe throw whose evidence is coming in: every monster in front at the throw, and at the first look
    after it (the pick), followed look by look until EVIDENCE_S after its star was due (HIDDEN_MAX_S while
    the target is hidden)."""

    def __init__(self, row: dict, s: Scene, target: Monster, key: float, due: float, exp0: int):
        self.row, self.side, self.key, self.due, self.exp0 = row, row["side"], key, due, exp0
        self.exp = 0
        self.looks = 0  # looks from when the star could have landed on (BEFORE_HIT_S after the key)
        self.picked = False
        self.entries = [self._entry(m, m is target) for m in in_front(s, self.side, walls=False)]
        if not any(e["target"] for e in self.entries):
            self.entries.append(self._entry(target, True))

    @staticmethod
    def _entry(m: Monster, target: bool) -> dict:
        return {"m": m, "kind": m.kind, "target": target, "at_pick": False, "x": m.x, "y": m.y,
                "platform": m.platform, "bar_before": m.bar, "after": [], "unseen": [], "seen_last": True}

    def see(self, s: Scene, exp_gains: int) -> None:
        """A look after the throw that placed her (in the air too; `exp_gains`: the hunter's EXP gains so far).
        The first is the pick: where each monster was then. A look before the star can have landed (BEFORE_HIT_S
        after the key) has each one's bar as before it; no other star of hers is in flight, as the next throw
        waits for this one's evidence. Every look after that is evidence, timed from when the star was due."""
        self.exp = exp_gains - self.exp0
        dt = round(s.t - self.due, 2)
        early = s.t - self.key < BEFORE_HIT_S
        claimed: set[int] = set()
        for e in self.entries:
            near = [o for o in s.monsters if id(o) not in claimed and same(o, e["m"])]
            again = min(near, key=lambda o: abs(o.x - e["m"].x)) if near else None
            e["seen_last"] = again is not None
            if again is not None:
                claimed.add(id(again))
                e["m"] = again
                if not self.picked:
                    e.update(at_pick=True, x=again.x, y=again.y, platform=again.platform)
                if early:
                    e["bar_before"] = again.bar
                else:
                    e["after"].append([dt, again.bar])
            elif not early:
                e["unseen"].append(dt)
        if not early:
            self.looks += 1
        if self.picked:
            return
        self.picked = True  # the first look after the throw: about when the attack picked its target
        front = in_front(s, self.side, walls=False)
        for m in front:
            if id(m) not in claimed:  # in front now, not at the throw: it may have taken the star
                e = self._entry(m, False)
                e["at_pick"] = True
                if not early:
                    e["after"].append([dt, m.bar])
                self.entries.append(e)
        self.row["pick"] = {"x": round(s.where.x, 1), "y": round(s.where.y, 1),
                            "platform": None if s.platform is None else s.platform.id,
                            "t": round(s.t - self.key, 3), "front": [_monster(m) for m in front]}

    def over(self, t: float) -> bool:
        """Whether the evidence is all in by `t`: EVIDENCE_S after the star was due, and by then the target
        seen since it was due, or (her EXP up) one of them gone: a kill fades out where it died for as long as
        its death animation runs, a snail 1.26 s. Else up to HIDDEN_MAX_S."""
        dt = t - self.due
        if dt < EVIDENCE_S:
            return False
        if dt >= HIDDEN_MAX_S:
            return True
        if self.exp > 0:
            return any(not e["seen_last"] for e in self.entries)
        return any(when >= -EARLY_S for when, _ in next(e for e in self.entries if e["target"])["after"])

    def finish(self) -> dict:
        """The throw's line for the log, its evidence in."""
        row = self.row
        row["due_s"] = round(self.due - self.key, 3)
        row["exp"], row["looks"] = self.exp, self.looks
        row["evidence"] = [{"kind": e["kind"], "x": round(e["x"], 1), "y": round(e["y"], 1), "platform": e["platform"],
                            "target": e["target"], "at_pick": e["at_pick"], "bar_before": e["bar_before"],
                            "after": e["after"], "unseen": e["unseen"], "gone": not e["seen_last"]}
                           for e in self.entries]
        return row


def _done(h: "Hunter", report: ProbeReport, flying: _Flying) -> None:
    """A throw's evidence is in: logged, read, counted."""
    row = flying.finish()
    _log_row(row)
    told = dict((id(e), v) for e, v in outcomes(row))
    target = next(e for e in row["evidence"] if e["target"])
    seen = told[id(target)]
    others = sum(1 for e in row["evidence"] if not e["target"] and told[id(e)])
    if seen is None:
        report.unknown += 1
    elif seen:
        report.hits += 1
    else:
        report.misses += 1
    word = {True: "hit", False: "no hit", None: "can't tell (hidden)"}[seen]
    h.log(f"  -> {word}" + (f"; {others} other(s) hit" if others else "")
          + (f"; EXP +{row['exp']}" if row["exp"] else ""))
    telemetry.emit("event", kind="probe", text=f"probe {row['probe']}: {word}")


def _log_row(row: dict) -> None:
    try:
        with paths.PROBE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def listing() -> str:
    """The probes, for `probe list`."""
    lines = []
    for p in PROBES.values():
        ranges = ", ".join(f"{k} {lo:.0f}-{hi:.0f}" for k in p.kinds for lo, hi in (p.range_of(k),))
        lines += [f"{p.name}: {p.asks} ({p.source})",
                  f"    on {p.platform} at x {p.x:.0f}, {p.style} {p.side} at monsters on "
                  f"{', '.join(map(str, p.at))} ({ranges} units off), {p.throws} throws"
                  + ("" if p.walls else ", the wall rule off"),
                  f"    expect: {p.expect}"]
    return "\n".join(lines)


# --- the scorer ----------------------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    """The probe log's throws (none if there's no log)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shown(e: dict, exp: int, only_gone: bool) -> bool | None:
    """What a monster of a throw's evidence showed after the star was due: True a hit (its HP bar new or
    lower than before, or killed: her EXP up, and it the only one of them gone (`only_gone`)), False none
    (seen SETTLED_S or more after the star was due with its bar as before), None: can't tell (hidden)."""
    before = e["bar_before"]
    if any(b is not None and (before is None or b < before - HURT) for _, b in e["after"]):
        return True
    if exp and e["gone"] and only_gone:
        return True
    if any(dt >= SETTLED_S for dt, _ in e["after"]):
        return False
    return None


def outcomes(row: dict) -> list[tuple[dict, bool | None]]:
    """What each monster of a throw's evidence showed (`shown`)."""
    gone = [e for e in row.get("evidence", ()) if e["gone"]]
    return [(e, shown(e, row.get("exp", 0), len(gone) == 1)) for e in row.get("evidence", ())]


def _her(row: dict) -> tuple[float, float]:
    """Her map x when the attack picked its target (the first look after the throw; for a jump back she's
    on her way by then), and her feet's y on the ground she threw from."""
    pick = row.get("pick")
    return (pick["x"] if pick else row["x"]), row["y"]


def _rise(row: dict) -> float:
    return 0.0 if row["gap_ms"] is None else reach.air_rise(row["gap_ms"])


def _key(row: dict, e: dict, body: dict[str, MobBody], top0: float | None = None, floor0: float | None = None,
         air: str = "area", ramp: bool = False) -> tuple | None:
    """Where monster `e` comes in the order the throw's attack takes them (the least first), or None if it
    can't take it, terrain aside: from the ground the claw (if a stance's box reaches it: reach.claw_chance)
    then the search's bands, the nearest first within one (as reach.ground_pick; `top0`, `floor0`: the first
    window's edges; with `ramp`, none below her closer by its middle than the old ramp,
    reach.ramp_dx); from a jump (`air`) the area ("area", reach.in_area), or the search from as high as she is
    then ("search"), or from the ground she jumped from ("ground"). Bodies as the fight goes by them
    (fight._in_path)."""
    x, y = _her(row)
    ahead, dy = (e["x"] - x) * directions.sign(row["side"]), e["y"] - y
    top, half = reach.body_of(e["kind"], body)
    if row["gap_ms"] is None:
        if reach.claw_chance(ahead, dy, top, half) > 0:
            return 0, math.hypot(ahead, dy)
        if ramp and abs(ahead) < reach.ramp_dx(dy):
            return None
        k = reach.search_band(*reach.body_box(ahead, dy, top, half), top0=top0, floor0=floor0)
        return None if k is None else (1 + k, abs(ahead))
    rise = _rise(row)
    if air in ("search", "ground"):
        up = rise if air == "search" else 0.0
        k = reach.search_band(*reach.body_box(ahead, dy + up, top, half), top0=top0, floor0=floor0)
        return None if k is None else (1 + k, abs(ahead))
    least = body[e["kind"]].top if e["kind"] in body else top
    return (0, abs(ahead)) if reach.in_area(ahead, dy, least, rise, half) else None


Terrain = Callable[[dict, dict, MapModel], bool]  # (throw, monster, map) -> whether a star gets to it


def by_rule(shot_h: float | None = None) -> Terrain:
    """The wall rule (reach.wall_between): her own group's walls across the level line `shot_h` (SHOT_H) above
    her feet, raised in a jump."""
    def ok(row: dict, e: dict, model: MapModel) -> bool:
        x, y = _her(row)
        on = model.platform(e["platform"]) if e["platform"] is not None else None
        return reach.wall_between(model.platform(row["her"]), x, y, e["x"], _rise(row), target=on,
                                  shot_h=shot_h) is None
    return ok


def no_walls(row: dict, e: dict, model: MapModel) -> bool:
    """No wall stops a star."""
    return True


def every_wall(row: dict, e: dict, model: MapModel) -> bool:
    """Every wall stops a star, any foothold group's, across the same level line as the rule's."""
    x, y = _her(row)
    line = y - _rise(row) - reach.SHOT_H
    lo, hi = sorted((x, e["x"]))
    return not any(w.x1 == w.x2 and lo < w.x1 < hi and min(w.y1, w.y2) <= line < max(w.y1, w.y2) for w in model.walls)


def line_to_feet(row: dict, e: dict, model: MapModel) -> bool:
    """Her own group's walls stop a star on a straight line from SHOT_H above her feet (raised in a jump) to the
    target's feet."""
    x, y = _her(row)
    y0, x1, y1 = y - _rise(row) - reach.SHOT_H, e["x"], e["y"]
    for w in model.platform(row["her"]).walls:
        if min(x, x1) < w.x1 < max(x, x1):
            at = y0 + (y1 - y0) * (w.x1 - x) / (x1 - x)
            if min(w.y1, w.y2) <= at < max(w.y1, w.y2):
                return False
    return True


def taken(row: dict, model: MapModel, body: dict[str, MobBody], terrain: Terrain | None = None, **kw) -> dict | None:
    """The monster of a throw's evidence its attack takes (`_key`'s order; `kw` its window and how a jump's
    attack picks), of those `terrain` lets a star get to (by default the rule); None if none."""
    terrain = terrain or by_rule()
    best = None
    for e in row.get("evidence", ()):
        k = _key(row, e, body, **kw)
        if k is not None and terrain(row, e, model) and (best is None or k < best[0]):
            best = (k, e)
    return None if best is None else best[1]


def likelihood(rows: list[dict], pick: Callable[[dict], dict | None]) -> tuple[float, dict]:
    """How well a model of which monster each throw's attack takes (`pick`) fits what they showed: the log of
    the chance of every monster's hit or none, a monster taken showing a hit with chance a and one not taken
    with chance f, both fitted (a at least A_MIN, f at most F_MAX, neither within RATE_FLOOR of 0 or 1).
    Returns it and the counts."""
    n = {(t, s): 0 for t in (True, False) for s in (True, False)}  # (taken, showed a hit) -> monsters
    for row in rows:
        took = pick(row)
        for e, s in outcomes(row):
            if s is not None:
                n[(e is took, s)] += 1
    ll = 0.0
    for t, lo, hi in ((True, A_MIN, 1 - RATE_FLOOR), (False, RATE_FLOOR, F_MAX)):
        total = n[(t, True)] + n[(t, False)]
        if total:
            rate = min(hi, max(lo, n[(t, True)] / total))
            ll += n[(t, True)] * math.log(rate) + n[(t, False)] * math.log(1 - rate)
    return ll, n


@dataclass
class Fit:
    """A parameter fitted on a grid by likelihood: the best value, the values within FITS of it, and the
    log-likelihood at every value."""
    name: str
    lls: dict[float, float] = field(default_factory=dict)
    throws: int = 0
    decided: int = 0     # monsters whose hit or none was told
    hits: int = 0

    @property
    def best(self) -> float:
        top = max(self.lls.values())
        at = [v for v, ll in self.lls.items() if ll >= top - 1e-9]
        return at[len(at) // 2]  # the middle of a level stretch

    @property
    def fits(self) -> tuple[float, float]:
        top = max(self.lls.values())
        ok = [v for v, ll in self.lls.items() if ll >= top - FITS]
        return min(ok), max(ok)

    def holds(self, value: float) -> bool:
        """Whether `value` fits (within FITS of the best)."""
        return self.lls[value] >= max(self.lls.values()) - FITS


def _fit(name: str, rows: list[dict], grid, pick_at: Callable[[float], Callable[[dict], dict | None]]) -> Fit:
    """`name` fitted over `grid` (and the values reach has now, so they can be compared)."""
    fit = Fit(name, throws=len(rows))
    n: dict = {}
    for v in grid:
        fit.lls[v], n = likelihood(rows, pick_at(v))
    fit.decided = sum(n.values())
    fit.hits = n.get((True, True), 0) + n.get((False, True), 0)
    return fit


def ground(rows: list[dict]) -> list[dict]:
    """The throws from the ground."""
    return [r for r in rows if r["gap_ms"] is None and r["style"] == "throw"]


def fit_top0(rows: list[dict], model: MapModel) -> Fit:
    """The search window's top (SEARCH_TOP0) fitted to the throws from the ground (the rule's walls kept)."""
    body, grid = bodies(model), sorted(set(TOP0_GRID) | {reach.SEARCH_TOP0, LOGS_TOP0})
    return _fit("SEARCH_TOP0", ground(rows), grid, lambda v: lambda r: taken(r, model, body, top0=v))


def fit_floor0(rows: list[dict], model: MapModel) -> Fit:
    """The window's floor (SEARCH_FLOOR0) fitted to the throws from the ground (the top as reach has it)."""
    body, grid = bodies(model), sorted(set(FLOOR0_GRID) | {reach.SEARCH_FLOOR0})
    return _fit("SEARCH_FLOOR0", ground(rows), grid, lambda v: lambda r: taken(r, model, body, floor0=v))


def fit_shot_h(rows: list[dict], model: MapModel) -> Fit:
    """The rule's line height (SHOT_H) fitted to every throw (from the ground and from jumps)."""
    body, grid = bodies(model), sorted(set(SHOT_H_GRID) | {reach.SHOT_H})
    return _fit("SHOT_H", rows, grid, lambda v: lambda r: taken(r, model, body, by_rule(v)))


TERRAIN_MODELS: dict[str, Terrain] = {
    "her group's walls, a level line (the rule)": by_rule(),
    "no wall stops a star": no_walls,
    "every wall, a level line": every_wall,
    "her group's walls, a line to the feet": line_to_feet,
}


def terrain_lls(rows: list[dict], model: MapModel) -> dict[str, float]:
    """Each model of what stops a star (TERRAIN_MODELS), scored over every throw."""
    body = bodies(model)
    return {name: likelihood(rows, lambda r, t=t: taken(r, model, body, t))[0] for name, t in TERRAIN_MODELS.items()}


AIR_MODELS = {"area": "the area (reach.in_area)", "search": "the search from as high as she is",
              "ground": "the search from the ground she jumped from"}


def air_lls(rows: list[dict], model: MapModel) -> dict[str, float]:
    """How an attack from a jump picks (AIR_MODELS), scored over the throws from a jump."""
    body, use = bodies(model), [r for r in rows if r["gap_ms"] is not None]
    return {air: likelihood(use, lambda r, a=air: taken(r, model, body, air=a))[0] for air in AIR_MODELS}


def below(rows: list[dict]) -> list[dict]:
    """The throws from the ground at a target between her level and 60 below her (where reach.RAMP_BELOW holds)."""
    return [r for r in ground(rows) if reach.ramp_dx(r["target"]["y"] - _her(r)[1]) > 0]


def ramp_lls(rows: list[dict], model: MapModel) -> dict[str, float]:
    """The search alone against the search with the old ramp below her (reach.RAMP_BELOW), over those throws."""
    body, use = bodies(model), below(rows)
    return {"the search": likelihood(use, lambda r: taken(r, model, body))[0],
            "the search, the old ramp below her": likelihood(use, lambda r: taken(r, model, body, ramp=True))[0]}


def drop(e: dict, exp: int, only_gone: bool) -> float | None:
    """How much of its HP bar a monster of a throw's evidence lost, if it showed a hit (`shown`): from its bar before
    (full if none) to the lowest after; the whole of what was left if it was killed. None: no hit, or can't tell."""
    if not shown(e, exp, only_gone):
        return None
    before = 1.0 if e["bar_before"] is None else e["bar_before"]
    seen = [b for _, b in e["after"] if b is not None]
    if exp and e["gone"] and only_gone:
        return before
    return before - min(seen) if seen else None


CLAW_COUNTS = ("throws", "claw", "star", "killed", "no hit", "can't tell")


def claw_table(rows: list[dict]) -> dict[tuple, dict[str, int]]:
    """The claw probes' throws (named claw-*) by (probe, the target's kind, its height below her (+), how far off it
    was at the pick in 10-unit bins): how many (CLAW_COUNTS), and what the target showed: a hit the claw's (under
    CLAW_DROP of its bar) or a star's, killed (either), no hit, or can't tell."""
    table: dict[tuple, dict[str, int]] = {}
    for row in rows:
        if not row["probe"].startswith("claw"):
            continue
        target, s = _target(row)
        if target is None:
            continue
        gone = [e for e in row["evidence"] if e["gone"]]
        d = drop(target, row.get("exp", 0), len(gone) == 1)
        killed = bool(row.get("exp")) and target["gone"] and len(gone) == 1
        x, y = _her(row)
        key = (row["probe"], target["kind"], round(target["y"] - y), int(abs(target["x"] - x) // 10) * 10)
        g = table.setdefault(key, dict.fromkeys(CLAW_COUNTS, 0))
        g["throws"] += 1
        g["can't tell" if s is None else "no hit" if s is False else "killed" if killed
          else "claw" if d < CLAW_DROP else "star"] += 1
    return table


def claw_lines(rows: list[dict]) -> list[str]:
    """claw_table, as text."""
    return [f"  {name} {kind} {dy:+d} {lo}-{lo + 10}: " + ", ".join(f"{k} {g[k]}" for k in CLAW_COUNTS)
            for (name, kind, dy, lo), g in sorted(claw_table(rows).items())]


def _target(row: dict) -> tuple[dict | None, bool | None]:
    found = [(e, s) for e, s in outcomes(row) if e["target"]]
    return found[0] if found else (None, None)


def _table(rows: list[dict], key: Callable[[dict], tuple]) -> list[str]:
    """Throws grouped by `key`: how many, and what their targets showed."""
    groups: dict[tuple, list[int]] = {}
    for row in rows:
        _, s = _target(row)
        g = groups.setdefault(key(row), [0, 0, 0, 0])
        g[0] += 1
        g[{True: 1, False: 2, None: 3}[s]] += 1
    return [f"  {' '.join(map(str, k))}: {n} throws, the target hit by {hit}, not by {miss}, {unknown} can't tell"
            for k, (n, hit, miss, unknown) in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))]


def _fit_text(fit: Fit, now: float, other: tuple[str, float] | None = None) -> str:
    if not fit.throws or not fit.decided:
        return f"{fit.name}: no throws to fit it to"
    lo, hi = fit.fits
    ends = [w for w, at_end in (("below", lo == min(fit.lls)), ("above", hi == max(fit.lls))) if at_end]
    text = (f"{fit.name}: fits {lo:g} to {hi:g}"
            + (f" (not bounded {' or '.join(ends)} by these throws)" if ends else "")
            + f", best {fit.best:g} (log-likelihood {max(fit.lls.values()):.2f}); "
            f"reach has {now:g} ({fit.lls[now]:.2f}, {'fits' if fit.holds(now) else 'does not fit'})")
    if other is not None:
        name, v = other
        text += f"; {name} {v:g} ({fit.lls[v]:.2f}, {'fits' if fit.holds(v) else 'does not fit'})"
    return text + f". {fit.throws} throws, {fit.decided} monsters told, {fit.hits} hit"


def score(rows: list[dict], model: MapModel) -> str:
    """The probe log scored (see the module's docstring), as text."""
    lines = [f"{len(rows)} probe throws; by probe, how she threw, and where she stood (map x):"]
    lines += _table(rows, lambda r: (r["probe"], r["style"], f"x {round(r['x'] / 10) * 10:g}"))
    lines.append("By the target's kind and how far off it was at the pick (20-unit bins, middle to middle):")
    lines += _table(rows, lambda r: (r["probe"], r["style"], r["target"]["kind"], _bin(r)))
    lines.append("The search's window, from the throws from the ground:")
    lines.append("  " + _fit_text(fit_top0(rows, model), reach.SEARCH_TOP0, ("the logs' fit", LOGS_TOP0)))
    lines.append("  " + _fit_text(fit_floor0(rows, model), reach.SEARCH_FLOOR0))
    if under := below(rows):
        lines.append(f"Between her level and 60 below her, from {len(under)} throws (log-likelihood; reach.RAMP_BELOW "
                     f"is {'on' if reach.RAMP_BELOW else 'off'}):")
        for name, ll in sorted(ramp_lls(rows, model).items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name}: {ll:.2f}")
    if claws := claw_lines(rows):
        lines.append(f"The claw, by the target's kind, its height (+ below her) and how far off it was at the pick "
                     f"(a hit under {CLAW_DROP:.0%} of its bar is the claw's):")
        lines += claws
    lines.append("What stops a star, from every throw (log-likelihood, higher fits better):")
    for name, ll in sorted(terrain_lls(rows, model).items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name}: {ll:.2f}")
    blocked = [r for r in rows if _target(r)[0] is not None and not by_rule()(r, _target(r)[0], model)]
    told = [s for r in blocked for _, s in (_target(r),) if s is not None]
    lines.append(f"  behind the rule's walls: {len(blocked)} throws, the target hit by {sum(told)} of {len(told)} told")
    lines.append("  " + _fit_text(fit_shot_h(rows, model), reach.SHOT_H))
    jumps = [r for r in rows if r["gap_ms"] is not None]
    if jumps:
        lines.append(f"What an attack from a jump picks, from {len(jumps)} throws (log-likelihood):")
        for air, ll in sorted(air_lls(rows, model).items(), key=lambda kv: -kv[1]):
            lines.append(f"  {AIR_MODELS[air]}: {ll:.2f}")
    return "\n".join(lines)


def _bin(row: dict) -> str:
    """How far off the target was at the pick, in 20-unit bins."""
    target, _ = _target(row)
    x, _ = _her(row)
    off = abs((target or row["target"])["x"] - x)
    lo = int(off // 20) * 20
    return f"{lo}-{lo + 20}"
