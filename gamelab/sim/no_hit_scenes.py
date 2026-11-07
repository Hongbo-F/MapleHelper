"""The no-hit scenes: moments of the failsafe's recordings of 2026-11-04 and 11-05 (D:/MapleHelper-scratch/nohit: raw
frames, packs, verified ledgers) put in the world, then the full stack for a short while, to see the fixes solve what
those recordings showed before they go live. The whole-run tables can't tell: a small prize is under their seeds' noise.

Each scene starts about 0.15 s before the look of a clip's first attack of its cause (a scene's first decision comes that
far in): her place and facing, her HP and MP, the monsters on the screen with their bars. A monster by her walks as the
frames have it (`Mob.rails`, read off the pack's timeline: D:/MapleHelper-scratch/nofix/moment.py) until her first hit on
it or the end of what was recorded, then by the world's own AI: the world's walkers go on past her up to AGGRO_PASS where
these stopped, turned and came back. The world is the harness's standard thief's but what's named on `scenario_of`.

Not in scenarios.SCENARIOS: no golden, no effect on the others. `measures` is what D:/MapleHelper-scratch/nofix/scenes.py
prints; `fault_shows` says whether runs show what the recording showed, `Scene.solved` what solved means in the user's
words (tests/test_no_hit_scenes.py). A verdict holds at the recorded phase: `SHIFTS` are the small changes the frames can't rule
out, `over_shifts` how many of them keep it (most don't: README). A scene tests an attack or two, not a stuck run."""

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

from ..perception.kinds import kind_of
from .world import Camp, SimJumper, SimMonster

# Her as she played these clips, level 35: her stat window of 2026-11-05 07:48 (lab/her_stats.json: HP 1244, MP 1048,
# W.DEF 117, EVA 80, ATTACK 60-182), a cast 16 MP and an MP potion +150 on her HUD, Lucky Seven's stars 174-285 each and
# the claw's 19-25 by her numbers (the verified ledgers). Every other world's thief is level 25, where a Horny Mushroom's
# touch is 16% of her HP and its keep-off 164, not 72.
HER_35 = {"max_hp": 1244.0, "level": 35, "defence": 117.0, "avoid": 80.0, "mp": 1048.0, "star": (60, 182),
          "skill_star": (174, 285), "cast_mp": 16.0, "mp_potion": 150.0, "claw": (19, 25)}
WORLD = ("zero-walls", "near-misses", "walkers", "spawn-fade", "map-drops", "drop-takers", "near-misses-exact")  # (the
#         harness's standard thief)
# A clear pick that shows nothing at a Green live (42 of 1,058: nohit/causes.md), where the standard world's 0.275 is the
# first map's at level 19; of the ledgers' 61 attacks none shows a MISS over a monster.
MISS = 0.04
NOTHING = ("nothing", "stopped", "miss")  # (World.attack_log's outcomes that hit nothing)
FAULT_SHARE = 0.6                         # of the recording's attacks at nothing, for a scene's runs to show its fault
# What the frames can't rule out, as (s the rails go later, units her place along): a monster's walk by the world's 0.1 s look offset,
# her place by the 6 units it's off in a tenth of keys.
SHIFTS = {"late": (0.1, 0.0), "early": (-0.1, 0.0), "her+6": (0.0, 6.0), "her-6": (0.0, -6.0)}


class _RailWalker(SimMonster):
    """A walker on `rails`: (s, way) stretches, until her first hit on it (World._struck), then its own AI."""
    rails: tuple = ()

    def _draw(self) -> None:
        if self.rails and self.hit_at is None:
            (s, self.d), self.rails = self.rails[0], self.rails[1:]
            self.t1 = self.t0 + s
        else:
            super()._draw()


class _RailJumper(SimJumper):
    """A jumper on `rails`, as _RailWalker, its AI's steps at their ends: no hop at a platform's end, no limit of its reach
    (a slime's hop in the recording isn't in them)."""
    rails: tuple = ()
    rail_until: float = 0.0

    def _think(self, t: float) -> None:
        if not (self.rails or t < self.rail_until) or self.hit_at is not None:
            super()._think(t)
        elif t >= self.rail_until:
            (s, way), self.rails = self.rails[0], self.rails[1:]
            self.rail_until = self.ai_t = t + s
            self._walk(t, way)

    def _walk_end(self, s) -> tuple[float, str]:
        return (math.inf, "end") if self.rails or s[1] < self.rail_until else super()._walk_end(s)


@dataclass(frozen=True)
class Mob:
    """A monster on the frames at the scene's start."""
    name: str
    platform: int
    x: float
    rails: tuple = ()       # its walk as recorded: (s, way) stretches (way +1 right, -1 left, 0 standing), then its AI
    hp: float = 1.0         # share of its HP left: its bar
    chasing: bool = False   # she has hit it: it chases her off its rails (World's aggro), and her belief has it so


@dataclass(frozen=True)
class Scene:
    name: str
    clip: str               # the recording, its frames and what they show
    saw: tuple[float, int, int]  # in the recording, from where the scene starts: over so many s, her attacks, at nothing
    start: tuple[int, float]
    facing: str
    mobs: tuple[Mob, ...]
    target: int = 0         # which of `mobs` she goes at
    hp: float = 1.0
    mp: float = 1.0
    seconds: float = 15.0
    pet_mp: bool = True     # her pet's MP potions (out in 081701)
    inv: float = 0.0        # s of her invulnerability left, from the touch before
    locked: float = 0.0     # s of her last attack's lock left
    misses: int = 0         # attacks of the failsafe's count before the scene's start, as in the recording (Attacks._misses)
    others: tuple = ()      # other players' places (platform, x): dots on her minimap
    script: Callable | None = None  # (sim, hunter): what else happens
    fault: Callable | None = None   # (runs' measures) -> bool: the recording's fault shows, where `saw` doesn't say it
    solved: tuple = ()      # (what, runs' measures -> bool): what solved means, in the user's words


SCENES: dict[str, Scene] = {}


def _scene(*args, **kw) -> None:
    sc = Scene(*args, **kw)
    SCENES[sc.name] = sc


def _place(m, mob: Mob, w) -> None:
    """Monster `m` as `mob` says at the scene's start."""
    m.hp = mob.hp * m.hp_max
    if mob.chasing:
        m.aggro(w.t)
    if isinstance(m, SimJumper):
        m.__class__, m.rails, m.ai_t = _RailJumper, mob.rails, w.t
    else:
        m.__class__, m.rails = _RailWalker, mob.rails
        if mob.rails:
            m.t0, m.x0, m.d, m.t1 = w.t, mob.x, 0, w.t  # (at its first `at`: the first stretch)


def _she_hit(h, mobs) -> None:
    """Her belief as the recording had it: the monsters of `mobs` that chase her do from the look that first has them
    (localization.aggro), a bar over them."""
    aggro, left, own = h.aggro, [m for m in mobs if m.chasing], h.aggro.update

    def update(t, her, tracks, events):
        own(t, her, tracks, events)
        for mob in list(left):
            tr = next((tr for tr in tracks if tr.kind == kind_of(mob.name) and tr.platform == mob.platform
                       and tr.bar is not None), None)
            if tr is not None:
                aggro._new(tr, t, "hit")
                left.remove(mob)
    aggro.update = update


def _counted(h, n: int, now: float) -> None:
    """The failsafe's count as the recording had it: `n` attacks with no hit (Attacks._misses), and the bars the first
    looks have are bars it knew (Attacks._watch_hits: a new one is a hit)."""
    attacks, own = h.attacks, h.attacks._watch_hits
    attacks._misses = [(now - 1.0, None)] * n

    def watch(s):
        if s.t - now < 0.5:
            attacks._bars_seen.update({m.track: m.bar for m in s.monsters if m.bar is not None and m.track is not None})
        own(s)
    attacks._watch_hits = watch


def shifted(sc: Scene, shift: str) -> Scene:
    """Scene `sc` by SHIFTS[`shift`]: the rails standing that long first, or going in that much earlier (its monster that far on, at
    100 a second); her place."""
    late, her = SHIFTS[shift]

    def mob(m: Mob) -> Mob:
        if not m.rails or not late:
            return m
        if late > 0:
            return dataclasses.replace(m, rails=((late, 0), *m.rails))
        (s, way), rest = m.rails[0], m.rails[1:]
        return dataclasses.replace(m, x=m.x + way * 100.0 * min(-late, s), rails=((s + late, way), *rest) if s > -late else rest)
    return dataclasses.replace(sc, start=(sc.start[0], sc.start[1] + her), mobs=tuple(mob(m) for m in sc.mobs))


def scenario_of(name: str, seed: int, shift: str = ""):
    """Scene `name` (`shifted` by `shift`) in world `seed`: the harness's standard thief's world with her at level 35 (HER_35), MISS,
    nobody else but `others` and nothing back within the scene. Her stack as it was then: her stats known, her facing, her target."""
    from ..attack import danger
    from ..behavior.grind import Grind
    from ..localization.monsters import Monster
    from . import harness, scenarios
    sc = shifted(SCENES[name], shift) if shift else SCENES[name]
    base = harness.scenario("loop-grind", seed, "crowd", sc.seconds / 60, 0.0, scenarios.TREE, WORLD, (), "choice")

    def run(sim):
        w = sim.world
        for m, mob in zip(w.monsters, sc.mobs):
            _place(m, mob, w)
        w.mp, w.invulnerable_ms, w.pet_full = sc.mp, round(sc.inv * 1000), 100.0  # (fed: no stop for it at the start)
        h = sim.hunter(potion_hp=0.45, her=danger.Her(HER_35["max_hp"], HER_35["level"], HER_35["defence"]))
        h.actions.facing = sc.facing
        _she_hit(h, sc.mobs)
        if sc.locked:
            w.attack_lock_ms, h.actions.lock_until = round(sc.locked * 1000), sim.clock.now + sc.locked
        _counted(h, sc.misses, sim.clock.now)
        if sc.script is not None:
            sc.script(sim, h)
        g = Grind(sim.tour(h, sim.navigator(h, fight=True)), log=sim.log)
        t = sc.mobs[sc.target]
        y = sim.model.platform(t.platform).y_at(t.x)
        g.tp.chooser.primary = Monster(SimpleNamespace(kind=kind_of(t.name), fading=None), t.x, y, t.platform)
        return str(g.run(minutes=sc.seconds / 60))
    return dataclasses.replace(
        base, name=f"{name}:{seed}", fn=run, start=sc.start, facing=sc.facing, hp=sc.hp, limit=sc.seconds, doc=sc.clip,
        monsters=[(m.name, m.platform, m.x) for m in sc.mobs], noise=dataclasses.replace(base.noise, miss=MISS),
        her=dataclasses.replace(base.her, **HER_35, **({} if sc.pet_mp else {"pet_mp": None})),
        realism=dataclasses.replace(base.realism, camps=tuple(Camp(p, x, (), ()) for p, x in sc.others),
                                    respawn_tick=0.0, respawn=600.0))


def measures(w, sc: Scene) -> dict:
    """What a run of scene `sc` came to, from its world: her attacks, those hitting nothing (the world's own outcome: none
    picked or dead by then, stopped, missed) and both in the recording's span, when the first hit went in, a claw's too, and the
    first that wasn't a claw's swipe, her claw's swipes (`stabs`), when her target died (s of the scene), EXP, touches, knock-offs,
    the longest she stood with no key while a monster was within 60 (`_watch_stand`), MP spent, and her attacks as (key s, ground
    or air, outcome)."""
    hits = [a[0] for a in w.attack_log if a[5] in ("hit", "kill", "swipe")]
    stars = [a[0] for a in w.attack_log if a[5] in ("hit", "kill")]
    target, early = w.monsters[sc.target], [a for a in w.attack_log if a[0] <= sc.saw[0]]
    return {"attacks": len(w.attack_log), "nothing": sum(a[5] in NOTHING for a in w.attack_log),
            "span_attacks": len(early), "span_nothing": sum(a[5] in NOTHING for a in early),
            "first_hit": round(hits[0], 2) if hits else None, "first_star": round(stars[0], 2) if stars else None,
            "stabs": sum(a[5] == "swipe" for a in w.attack_log),
            "death": round(target.died_at, 2) if not target.alive and target.killed_by == "her" else None,
            "exp": w.exp, "touches": w.touches, "knocks": len(w.knock_log), "stand": round(w.stand_most, 2),
            "mp": round(w.mp_used), "outcomes": [(a[0], a[2], a[5]) for a in w.attack_log]}


def _watch_stand(w) -> None:
    """World.stand_most: the longest she stands on the ground with no key held and no attack playing, a live monster
    within 60 of her on her platform (sampled as World._idleness)."""
    own, w.stand_most, since = w._idleness, 0.0, [None]

    def idleness():
        own()
        close = (w.mode == "ground" and not w.vx and not w.glide and not w.held and w.ms >= w.attack_lock_ms
                 and any(m.alive and abs(m.at(w.t) - w.x) < 60.0 for m in w._on(w.last_plat.id)))
        since[0] = (w.t if since[0] is None else since[0]) if close else None
        if close:
            w.stand_most = max(w.stand_most, w.t - since[0])
    w._idleness = idleness


def run_scene(name: str, seed: int, shift: str = "") -> tuple[dict, object, dict]:
    """Run scene `name` (`shifted` by `shift`) in world `seed` with the measured costs of her looks, as harness.run_one a hunt: its
    trace, its Sim, its `measures` and the failsafe's firings."""
    from .. import telemetry
    from . import fakes, harness, scenarios
    costs = harness.COSTS["measured"]
    before = {k: getattr(fakes, k) for k in costs}
    scenario = scenario_of(name, seed, shift)

    def run(sim):
        _watch_stand(sim.world)
        return scenario.fn(sim)
    try:
        for k, v in costs.items():
            setattr(fakes, k, v)
        with telemetry.captured() as records:
            trace, sim = scenarios.run_sim(dataclasses.replace(scenario, fn=run))
        sim.records = records  # (what her stack recorded)
    finally:
        for k, v in before.items():
            setattr(fakes, k, v)
    return trace, sim, measures(sim.world, SCENES[name]) | {"firings": sum("attacks over" in x for x in trace["log"])}


def _mean(runs: list[dict], key: str, never: float = 15.0) -> float:
    return sum(never if r[key] is None else r[key] for r in runs) / len(runs)


def fault_shows(name: str, runs: list[dict]) -> bool:
    """Whether runs of scene `name` show the recording's fault (`Scene.fault`, else by `Scene.saw`: her attacks in its
    span hit nothing about as they did there, at least FAULT_SHARE as many, and as much of what she keyed)."""
    sc = SCENES[name]
    if sc.fault is not None:
        return sc.fault(runs)
    _, attacks, nothing = sc.saw
    none = _mean(runs, "span_nothing")
    return none >= FAULT_SHARE * nothing and none >= FAULT_SHARE * _mean(runs, "span_attacks") * nothing / attacks


def unmet(name: str, runs: list[dict]) -> list[str]:
    """What solved means for scene `name` (`Scene.solved`) that runs of it don't show."""
    return [what for what, ok in SCENES[name].solved if not ok(runs)]


def over_shifts(name: str, runs_of: Callable[[str], list[dict]]) -> dict[str, int]:
    """Of the SHIFTS (`runs_of`: a shift's runs of scene `name`), on how many the fault shows and on how many it's solved."""
    runs = {shift: runs_of(shift) for shift in SHIFTS}
    return {"fault": sum(fault_shows(name, r) for r in runs.values()), "solved": sum(not unmet(name, r) for r in runs.values())}


# --- what solved means: the user's words of 2026-10-27 and 11-05, each against master's table halved or so ------------

def _fewer(n: float) -> tuple:
    """"Fewer attacks at nothing"."""
    return f"at most {n:g} attacks at nothing in the recording's span", lambda runs: _mean(runs, "span_nothing") <= n


def _sooner(s: float) -> tuple:
    """"First hit sooner" (a run with none counts as the scene's length)."""
    return f"the first hit by {s:g} s", lambda runs: _mean(runs, "first_hit") <= s


# "No standing still" (the user on the hold: "no, jump aside and attack from a distance").
_NO_STAND = ("no standing 0.5 s with a monster within 60", lambda runs: max(r["stand"] for r in runs) < 0.5)


def _exp(n: float) -> tuple:
    """"More EXP and loot over fewer touches": whatever she does about it, no less than master's EXP."""
    return f"{n:g} EXP or more", lambda runs: _mean(runs, "exp") >= n


def _stabs(n: float) -> tuple:
    """"First hit sooner" isn't a claw's swipe under her pile: master's about none."""
    return f"at most {n:g} of the claw's swipes", lambda runs: _mean(runs, "stabs") <= n


def _kills_target(n: int) -> tuple:
    return f"her target dead in {n} of 8 runs or more", lambda runs: sum(r["death"] is not None for r in runs) >= n


# --- the scenes -------------------------------------------------------------------------------------------------------

G, H, S = "Green Mushroom", "Horny Mushroom", "Slime"


def _strangers_kill(*kills):
    """Another player kills monster `i` of the scene `s` seconds in (World._others_kill)."""
    def script(sim, h):
        w = sim.world
        for i, s in kills:
            w.call_at(w.origin + s, lambda _, m=w.monsters[i]: m.alive and w._others_kill(m, w._camp_rng, 1.0))
    return script


_scene("too_close_081701",
       "20261105-081701, frames 00077-00125 (a03-a05): a slime, bar 59%, turns at +24 and at P33's end at +56, over and "
       "over; her regular attack (MP out), a jump in place every second: no number, no bar", (3.0, 3, 3),
       (33, 255.5), "right",
       (Mob(S, 33, 300.0, ((0.19, -1), (0.07, 0), (0.31, 1), (0.36, 0), (0.31, -1), (0.09, 0), (0.31, 1), (0.36, 0),
                           (0.31, -1), (0.09, 0), (0.31, 1), (0.36, 0)), hp=0.59, chasing=True),
        Mob(S, 33, -54.0, ((0.84, 1), (0.09, 0), (0.81, -1), (0.12, 0), (1.2, -1)))),
       hp=0.779, mp=25 / 1048, pet_mp=False, misses=2, solved=(_fewer(1.0), _sooner(2.0), _NO_STAND))
_scene("too_close_111448",
       "20261105-111448, frames 00039-00072 (a03-a04): a green, bar 89%, turning at +20 and at +60; a tap, a jump in "
       "place, the star through it, a touch of 49 knocks her off P8", (2.0, 2, 2), (8, 128.4), "left",
       (Mob(G, 8, 189.0, ((0.11, 0), (0.4, -1), (0.42, 1), (0.12, 0), (0.46, -1), (0.13, 0), (0.44, 1), (0.09, 0),
                          (0.27, -1), (0.2, 1), (0.13, 0), (1.45, -1), (0.93, 1), (0.23, 0)), hp=0.89, chasing=True),
        Mob(H, 10, -125.0, ((0.51, 0), (0.17, -1), (0.1, 0), (0.25, 1), (0.16, 0), (2.45, 1))),
        Mob(H, 10, -27.0, ((1.13, -1), (0.07, 0), (0.86, 1), (3.12, 0)))),
       hp=0.731, mp=0.24, inv=0.75, solved=(_fewer(0.6), _sooner(2.0), _NO_STAND))
_scene("wall_foot_231613",
       "20261104-231613, frames 00058-00121 (a02-a05): a horny standing at the trunk's foot, x -220 to -212, 4.9 s; she "
       "jumps in place at it from 26-41 off, four times, nothing shows", (4.0, 4, 4), (7, -171.5), "right",
       (Mob(H, 7, -220.0, ((0.08, 1), (4.77, 0), (0.13, 1))), Mob(H, 10, -138.0), Mob(H, 10, -4.0), Mob(H, 6, 238.0)),
       hp=0.683, mp=0.096, inv=1.3, misses=1, solved=(_fewer(1.5), _sooner(3.0), _NO_STAND))
_scene("walked_past_080032",
       "20261105-080032, frames 00063-00128 (a04-a06): greens #447 and #431 walk back and forth through her on P29; she "
       "taps toward where #447 was, it has left", (4.2, 3, 3), (29, -63.5), "right",
       (Mob(G, 29, -94.0, ((0.84, 1), (0.13, 0), (1.25, -1), (0.07, 0), (0.4, -1), (0.19, 1), (0.21, 0), (0.84, 1), (0.25, 0))),
        Mob(G, 29, -28.0, ((0.17, 1), (0.13, 0), (1.53, -1), (0.6, 0), (0.24, -1), (0.2, 0), (1.74, 1), (0.13, 0), (0.4, -1))),
        Mob(G, 30, 6.0, ((0.07, -1), (0.06, 0), (0.15, 1), (0.39, 0), (0.82, 1), (0.1, 0), (0.8, -1), (0.27, 0), (0.67, 1),
                         (0.13, 0), (0.6, -1), (0.2, 0), (0.91, 1)), hp=0.59, chasing=True),
        Mob(H, 31, -79.0), Mob(G, 22, -199.0)),
       hp=0.692, mp=0.067, inv=0.77, misses=1, solved=(_fewer(1.2), _sooner(2.0), _NO_STAND))
_scene("walked_past_080939",
       "20261105-080939, frames 00166-00207 (a05-a06): green #769 walks through her on P16, right, then left; a tap and "
       "a jump in place each way, the stars fly the side it left", (2.2, 2, 2), (16, 108.5), "right",
       (Mob(G, 16, 59.0, ((1.11, 1), (0.07, 0), (0.1, -1), (0.05, 0), (1.1, -1), (0.09, 0))),),
       hp=0.731, mp=0.173, misses=3, solved=(_fewer(0.9), _sooner(1.5), _NO_STAND, _kills_target(7)))
_scene("from_below_233732",
       "20261104-233732, frames 00055-00090 (a01-a03, a04-a05 after): on P14 under P13's right end, hornies on P13 70 "
       "above; ground throws on looks up to 0.4 s old, the window shut by the pick: level stars under P13 (a02 rose to "
       "its left end and stopped)", (2.4, 3, 3), (14, 18.5), "right",
       (Mob(H, 13, -106.0, ((1.05, -1), (0.16, 0), (0.04, 1), (0.09, 0), (2.34, 1), (0.06, 0), (0.51, -1))),
        Mob(H, 13, -29.0, ((0.37, 1), (0.16, 0), (0.05, -1), (2.77, 0), (0.05, -1), (0.22, 0), (0.65, -1))),
        Mob(H, 13, -16.0, ((0.44, 1), (0.08, 0), (2.41, -1), (0.07, 1), (0.2, 0), (1.05, 1))),
        Mob(G, 11, -84.0), Mob(G, 18, 67.0), Mob(G, 12, 176.0, ((2.19, 0), (0.71, 1), (0.93, 0), (0.37, -1)))),
       hp=0.769, mp=0.135, solved=(_fewer(1.3), _sooner(2.0), _exp(160), _stabs(1.0)))
_scene("from_below_234441",
       "20261104-234441, frames 00102-00147 (a03-a05): on P9 under a pile on P8; ground throws under its lip, then a jump "
       "up onto 8 keyed at take-off: nothing, 42 on landing", (2.6, 3, 3), (9, 47.5), "right",
       (Mob(H, 8, 94.0, ((0.04, -1), (1.43, 0), (0.2, 1), (0.2, 0), (0.77, 1), (0.16, 0), (0.27, -1))),
        Mob(G, 8, 177.0, ((0.18, -1), (0.1, 0), (1.11, -1), (0.07, 0), (0.16, 1), (0.11, 0), (0.08, -1), (0.45, 0))),
        Mob(H, 7, -177.0), Mob(H, 7, -44.0), Mob(H, 6, 239.0)),
       hp=0.769, mp=0.154, misses=1, solved=(_fewer(1.2), _sooner(1.5), _exp(187), _stabs(1.0)))
_scene("stranger_dead_080929",
       "20261105-080929, frames 00063-00098 (a03-a04): xXNephewXx's kills: horny #763 dead 0.23 s before her key, on a "
       "look 0.4 s old, at her lock's end; green #764's number 90 ms after her next key", (2.0, 2, 2), (14, 38.5), "left",
       (Mob(H, 13, -134.0, ((0.13, 1),)), Mob(G, 12, 231.0, ((1.47, 0),)), Mob(G, 7, -207.0)),
       hp=0.731, mp=0.115, locked=0.56, misses=1, others=((13, -120.0),), script=_strangers_kill((0, 0.13), (1, 1.5)),
       solved=(_fewer(1.0), ("most runs a cast less than master's 3 (on 2 of 8 perception has it hidden at her key, not dead)",
                             lambda runs: _mean(runs, "mp") <= 42.0)))
_scene("stranger_dying_080929",
       "20261105-080929, frames 00171-00191 (a05): up the rope onto P10, horny #766 walking left at +270, the stranger "
       "standing by it, his number 66 ms after her MP step, her stars fly on (here his kill just before her key: which "
       "came first can't be told on the frames)", (1.0, 1, 1), (10, -80.5), "left",
       (Mob(H, 10, 198.0, ((0.13, 0), (0.15, -1), (1.0, 0))), Mob(H, 7, -91.0), Mob(G, 8, 162.0)),
       hp=0.731, mp=0.087, misses=3, others=((10, 85.6),), script=_strangers_kill((0, 0.15)), solved=(_fewer(0.5),))
_scene("own_kill_235621",
       "20261104-235621, frames 00033-00062 (a02-a03): her throw kills green #6996 at +292; its EXP goes to the nearer "
       "green #7008 on P14 below (tracking._picks, as live), so #6996 stays a target: the next throw goes at its corpse",
       (1.5, 2, 1), (13, -53.5), "right",
       (Mob(G, 12, 255.0, ((0.78, -1),)),
        Mob(G, 14, 24.0, ((0.34, -1), (0.18, 0), (0.27, 1), (0.13, 0), (0.11, -1), (0.82, 0))),
        Mob(G, 11, -163.0, ((1.32, 0), (1.2, 1))), Mob(G, 8, 44.0), Mob(H, 15, -148.0), Mob(H, 10, -63.0),
        Mob(H, 10, -7.0), Mob(H, 10, 74.0), Mob(H, 10, 190.0), Mob(H, 7, -152.0), Mob(H, 6, 222.0)),
       hp=0.683, mp=0.087, solved=(_fewer(0.25),))
_scene("ignored_235621",
       "20261104-235621, frames 00114-00182 (a06 and after): the 4th no-hit attack, at green #6992 walking through her, "
       "fires the failsafe; ignored, it walks back and touches her, MISS, as she stands", (6.0, 1, 1), (11, -152.5), "right",
       (Mob(G, 11, -184.0, ((0.13, 0), (1.44, 1), (0.16, 0), (0.27, -1), (0.13, 0), (0.27, 1), (0.13, 0), (1.02, -1), (0.8, 0))),
        Mob(G, 14, -3.0), Mob(G, 8, 90.0), Mob(H, 15, -182.0), Mob(H, 10, -143.0), Mob(H, 10, -131.0), Mob(H, 10, 19.0),
        Mob(H, 7, -116.0), Mob(H, 6, 304.0)),
       hp=0.683, mp=0.173, inv=1.37, misses=3,
       fault=lambda runs: sum(r["death"] is not None for r in runs) <= 2 and _mean(runs, "touches") >= 1.0,
       solved=(_kills_target(6), _sooner(3.0), _NO_STAND))
