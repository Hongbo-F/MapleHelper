"""Results of simulated hunts, and A/B comparisons of hunting policies on the same seeds.

    python -m gamelab.sim.harness run POLICY [--seeds 10] [--minutes 20] [--preset game] [--jobs 4] [--save NAME]
                                             [--costs coded|measured] [--ways N] [--map 10004021]
                                             [--world zero-walls ...] [--grounds 23,...]
                                             [--attack choice|skill|regular]
    python -m gamelab.sim.harness compare BASELINE CANDIDATE [--bar rest]   two saved runs, seed by seed, against the bar
    python -m gamelab.sim.harness check                         the bookkeeping agrees with every scenario's world
    python -m gamelab.sim.harness rests [--trips 40]            trips to rest at low HP, transit vs --rest-avoid

A run hunts for a fixed time (the horizon) from one of its map's START_AT, on a world seed, with the policy's own
random choices fixed (POLICY_SEED), so two policies meet the same monsters until their kills differ. On the first
map by default; `--map 10004021` on West Rocky Mountain II, `--map 10002073` on Tree Dungeon, Forest Up North II, with
her as she plays each (sim.scenarios.HER: level 23 and 24, her pet drinking her potions), saved as
POLICY-PRESET-MAP; the presets are the same worlds on every map, but `crowd`, `crowd-17` and `crowd-slow`, Tree
Dungeon's other players (TREE_CAMPS), which are that map's alone. `--world` switches on what isn't known of the game
there (WORLDS: group 0's walls, the client's sides, the client's rope reach, a touch as a share of PADamage, no
evasion, the monsters' HP bars read empty), each saved as a suffix of the name. `--attack` has the stack choose her
attack a throw as live (behavior.attacks: Lucky Seven but the regular attack to finish a monster and while her MP is
under a cast; Lucky Seven whenever she has the MP; the regular attack always) in a world where her Lucky Seven key
casts it at its MP (ATTACK_HER). Her looks and glances are the feed's (SyncFeed), as live takes them from the
perception service; its costs are localization.feed.COSTS.
What a run gave is read from the simulated world, not from the bot's own reports (those miss kills):

  exp_per_min      EXP she got, per minute of the horizon (a death stops the count)
  kills, rival     her kills, and another player's (Realism.rival_every)
  hp_lost, hp_low  HP lost to touches (each as the world took it: by the monster's touch damage), and
                   her lowest HP; potions; died_at
  casts, mp_used   her Lucky Seven casts and the MP they took (world.CAST_MP each), mp_per_min a minute alive;
                   mp_potions, the MP potions her pet drank for her (world.Her.pet_mp); no_mp, casts pressed with too
                   little MP (nothing thrown); attacks, her attacks with either key; finishes, the regular attack
                   thrown to finish a monster her throws or its bar make low (`--attack choice`: its 'attack'
                   records)
  fell_out_at      when she fell below the map's lowest floor (world.SimFellOut): the run ends there, and
                   counts against the bar as a death does; left_map_at, the same for up held at a portal
                   (world.SimLeftMap)
  touches, hurt    monsters touching her, and those of the touches she didn't evade
  trip_s           the seconds she was on the move (from a chain's or a lone move's start, 'chain' or 'move' records,
                   to its 'moved' record), and trip_hurt, trip_hp_lost: the touches that hurt her then, and the HP
                   they took (what her trips cost, against her hunting where she stands)
  drops_share      of her drops whose fate is known by the end (picked up, or lain their whole life),
                   the share she picked up; value_share, the same weighed by what each is worth on the world's
                   map (behavior.looting.worth_for: the game data's prices, mesos by what its monsters drop), and
                   value_per_min, the mesos' worth she picked up a minute
  longest_idle     the longest stretch without a kill or a pick-up, and stuck_s: the time beyond 60 s
                   in such stretches
  rest_hp_lost     HP lost to touches on the trips to rest (from 'rest_trip' to 'rest' events), and
                   rest_trips: how many (the bar for `--rest-avoid`: compare --bar rest)
  alive_s          the seconds of the horizon she was hunting (to her death, fall or portal): rates a minute alive
  fresh_hurt       touches that hurt her by a monster within world.FRESH_S of its respawn (from its spawn point),
                   and fresh_hp_lost, the HP they took
  wait_s           seconds the hunt loop spent waiting for respawns (its rules 'wait' and 'lull loot', from its 'hunt'
                   records), and wait_hp_lost, the HP touches took then
  ropes            how often the hunt loop went to wait on a rope (its rule 9: telemetry events kind 'rope')
  backstops        how often each backstop cap fired (telemetry events kind 'backstop')
  nav              her trips, from the stack's log (`navigation`): plans cut short by a hit and off the planned
                   path, replans, misses (landed elsewhere than planned) and falls among them, and her glances
  coverage         her moves, platforms and plans, for the coverage report (analysis.coverage.tally: what it
                   keeps of the run's telemetry, 3-7 KB a run as saved)
  turning          her throws and jump throws, those that tapped an arrow to turn her first (control.actions: 80 ms
                   more a throw, 205 a jump throw, and a few units walked), her attacks that went the other way from
                   where the stack threw them (world.World.wrong_way), and the frames that told the other way from
                   her last arrow (the hunter's 'heading' events): `--world knock-turns` has a knock turn her,
                   `heading-wrong` and `heading-wrong-sure` her frames read the wrong way

Saved runs go to gamelab/lab/harness/<name>.json (git-ignored, like all game data). Runs are made with
PYTHONHASHSEED=0 and her stats as measured (physics.FIXED_STATS), as golden traces are.
"""

import argparse
import dataclasses
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from ..analysis import coverage
from ..behavior.looting import worth_for
from ..paths import LAB, REPO
from .world import DROP_LIFE, HER_BOX_CLIENT, LUCKY_SEVEN, MAP_DROPS, ROPE_REACH_CLIENT, Camp, Noise, Realism

FOLDER = LAB / "harness"
POLICY_SEED = 7   # the policy's own random choices (route weights, sweep order): the same in every run
FIRST_MAP = 10000012  # (sim.scenarios.MAP_ID)
START_AT = [(36, 300.0), (32, 600.0), (16, 585.0), (14, 760.0), (29, 420.0), (12, -300.0), (21, -300.0),
            (24, 120.0), (4, 450.0), (1, 1200.0)]  # where seed k starts: START_AT[k % len]
# ... on West Rocky Mountain II: its hunting grounds (the bridge 22, the high bridge 14, 17, 19, 10, the ledges 7, 6
# and 9), the floating step 24 under the bridge and stair 30 in the pit.
ROCKY_START_AT = [(22, -700.0), (14, -930.0), (17, -277.5), (19, -700.0), (10, -200.0), (7, -500.0), (30, -1009.0),
                  (24, -946.0), (6, -300.0), (9, -800.0)]
# ... on Tree Dungeon, Forest Up North II: her spot by the portal on 28, the study's best stands (23 at 73, 8 at 137,
# 20 at -95, 19 at 130, 15 at -110, 10 at -103), the mid portal's shelf 12, the floor 33 and 30.
TREE = 10002073
TREE_START_AT = [(28, 247.0), (23, 73.0), (8, 137.0), (20, -95.0), (19, 130.0), (12, 160.0), (33, 100.0), (15, -110.0),
                 (10, -103.0), (30, 60.0)]
STARTS = {FIRST_MAP: START_AT, 10004021: ROCKY_START_AT, TREE: TREE_START_AT}
# Worlds to compare in. "plain" is the simulator's own (every spawn filled, 8 s respawns, nobody else,
# perfect eyes). "game" adds the game's errors and a less crowded, contested map; its values are guesses
# (nothing about respawns, other players or potions is measured), chosen to be harder than "plain"; and arrows
# pressed right after an attack's key swallowed, as measured in game (world.ARROW_LOCK).
_GAME_NOISE = Noise(miss=0.275, walk_jitter=6.0, lag_sd=1.0, tap_lost=0.6, dot_px=True, hit_hidden=(1.0, 3.0),
                    detect_miss=0.1, heading_unsure=0.014)
PRESETS = {
    "plain": (Noise(), Realism()),
    "game": (_GAME_NOISE, Realism(spawn_share=0.6, rival_every=25.0, potions=60, lock_arrows=True)),
    # "game" with the drops seen in game (world.GAME_DROPS: a slime's Slime Bubble one kill in 10, and its other
    # monsters' items; on another map its monsters', world.game_drops) instead of mesos and a pig's leather every time:
    # the hunt loop's valuable piles (hunt.VALUABLE)
    "game-drops": (_GAME_NOISE, Realism(spawn_share=0.6, rival_every=25.0, potions=60, lock_arrows=True,
                                        drops=MAP_DROPS)),
    # "game" with jump throws missing a snail close by as often as in game (world.Noise.air_snail): what
    # stance.JUMP_THROW_SNAILS saves, which "game"'s jump throws, taking every snail their area picks, can't show
    "game-air": (dataclasses.replace(_GAME_NOISE, air_snail=(46.0, 0.35)),
                 Realism(spawn_share=0.6, rival_every=25.0, potions=60, lock_arrows=True)),
    # "game" with 15% of the spawns filled, not 60%: few monsters in the basin, so the trips aren't spent fighting
    # there (in "game" a quarter of the tour's trips time out, 28 of 246 before she set off: analysis.coverage), and
    # what's left to decide where she goes is the routes and the grounds chosen. The coverage test's world
    # (tests/test_coverage.py: every platform monsters spawn on reached).
    "game-sparse": (_GAME_NOISE, Realism(spawn_share=0.15, rival_every=25.0, potions=60, lock_arrows=True)),
}
# The other players on Tree Dungeon, Forest Up North II as the crowding study saw them from 28 (world.Camp; D:/
# MapleHelper-scratch/map3/study/crowding.md, 2026-10-24): Fuiyoohhhh by the portal on 28 killing 29's and 30's Greens
# (one every 7.1 s), BaileyBlake on 21 and siwooda by 22 killing 21's and 22's, TopAce on 23 (whose Hornies lived on), a
# player on 19 and Grasje on 32/33 (Greens and, unseen, Slimes). Five at once (3-6 were in view, mean 3.9); the upper
# tower (6-15), out of the minimap's view from 28, has nobody: unknown.
GREEN, SLIME = "green-mushroom", "slime"
TREE_CAMPS = (Camp(28, 250.0, (29, 30), (GREEN,)), Camp(21, 90.0, (21, 22), (GREEN,)), Camp(23, 0.0, (22,), (GREEN,)),
              Camp(19, 200.0, (19,), (GREEN,)), Camp(33, 120.0, (32, 33), (GREEN, SLIME)))
# Tree Dungeon's crowd: "game"'s errors, other players' dots under hers unseen (Noise.dot_under: 56 units, the study's
# frames), every spawn point filled (MeowDB lifts its cap from 6 players on), the camps above instead of "game"'s one
# roaming rival, and respawns at MeowDB's mean, 8.8 s ("crowd"), at 17 s ("crowd-17": the live median on 23 after her
# last kill there, 1.7-57 s, telemetry of 2026-10-24: README, Positioning), or 30 s ("crowd-slow": the first map's slow
# refill, README: Respawns and drop life). Only on that map (PRESET_MAPS).
_CROWD_NOISE = dataclasses.replace(_GAME_NOISE, dot_under=56.0)
PRESETS["crowd"] = (_CROWD_NOISE, Realism(respawn=8.8, camps=TREE_CAMPS, potions=60, lock_arrows=True))
PRESETS["crowd-17"] = (_CROWD_NOISE, Realism(respawn=17.0, camps=TREE_CAMPS, potions=60, lock_arrows=True))
PRESETS["crowd-slow"] = (_CROWD_NOISE, Realism(respawn=30.0, camps=TREE_CAMPS, potions=60, lock_arrows=True))
PRESET_MAPS = {"crowd": (TREE,), "crowd-17": (TREE,), "crowd-slow": (TREE,)}
# What isn't known of the game, switched on with `--world` (each a world.Realism field, or her own: world.Her):
# group 0's walls stopping her whatever her group, the map's sides as HeavenClient has them, a rope grabbed as it does
# (82 up at most), a touch's HP as a share of PADamage (as fitted on the first map) rather than meowdb's formula, and
# no evasion at all (the study: at EVA 42 she'd evade no Horny Mushroom), and her potions Red Potions (100 HP: which
# she carries isn't known; world.POTION_HP's 30% otherwise, 226 HP on Tree Dungeon). And how she plays: `lucky-seven`,
# her attacks as Lucky Seven's live (`tour --loop --skill`: world.LUCKY_SEVEN, a Horny Mushroom's kill in about 4
# throws, not 12), the stack unchanged (the simulator acts on its key as on the attack's).
WORLDS = {"zero-walls": {"zero_walls": True}, "client-sides": {"sides": "client"},
          "client-reach": {"rope_reach": ROPE_REACH_CLIENT}, "touch-share": {"defence": None},
          "no-evade": {"evade": 1e-9}, "red-potion": {"potion": 100.0}, "lucky-seven": {"stars": LUCKY_SEVEN},
          "knock-turns": {"knock_turns": True}}
HER_FIELDS = ("defence", "evade", "potion", "stars")
IDLE_GRACE = 60.0  # s without a kill or a pick-up that still counts as hunting
# ... and her body as the client has it (`client-body`: 22 either side of her feet and 65 tall, motion.bodies, which the
# planner keeps her clear of monsters by), not the simulator's 15 and 55 (world.HER_HALF, HER_HEIGHT).
WORLDS["client-body"] = {"her_box": HER_BOX_CLIENT}
# ... and the monsters' HP bars read empty once they're hurt (`bars-empty`: Realism.bars_empty, sim.fakes.FakeEyes.bar),
# as the stack read them live at 1366x768 until 2026-10-25 (perception.hud.hp_bar took the bar's black frame for its
# fill: 488 of 567 bars read 0), so what a monster has left was her throws' reckoning alone (behavior.attacks).
WORLDS["bars-empty"] = {"bars_empty": True}
# ... and her facing read the wrong way off 1% of her frames, in runs of one pose (world.Noise.heading_wrong): as weakly
# as every frame read the wrong way on 2026-10-24 (0.31, 0.24 ahead: never strong, which her throws' taps go by), and,
# the worst case, as surely as she's read right (0.6, 0.4 ahead: strong).
WORLDS["heading-wrong"] = {"heading_wrong": (0.01, 0.31, 0.24, True)}
WORLDS["heading-wrong-sure"] = {"heading_wrong": (0.01, 0.6, 0.4, True)}
NOISE_FIELDS = ("heading_wrong",)  # WORLDS' fields that are world.Noise's
# Her with `--attack` (any mode, so the three compare in one world): her Lucky Seven key's attack Lucky Seven as thrown
# live (world.LUCKY_SEVEN), CAST_MP a cast of her MP (world.Her.mp: LEVEL_24's 601) and nothing with less, her attack
# key's the regular star; each star's damage drawn against the monster's HP in the game data (world.Her.spread), so a
# monster's HP left is as the stack reckons it (attack.choice: its finishing throws).
ATTACK_HER = {"skill": LUCKY_SEVEN, "spread": True}
# What the simulator's looks take (sim.fakes), to compare against: "coded", the fakes' own (a look 100 ms, as
# the README had them before the speedups of 2026-10-20); "measured", today's, as bench-look and bench-capture
# measured them offline (17 saved frames x 5, medians, after the profiling pass): a look placing her 12.0 ms of
# computing plus the wait for a frame (5.6 ms median with the screen changing) and its copy off the card and
# conversion (2.0), her band 23.8 ms more, the whole screen 83.8, a strip of drops about 8 (her platform's
# 15.6 ms in 1.9 strips). Before the profiling pass (as step 5 measured them): 0.024, 0.006, 0.032, 0.122 and
# 0.008. Glances keep their pace (Eyes.GLANCE_EVERY_S). Her looks and glances come from the feed (SyncFeed), whose
# own costs are localization.feed.COSTS: these are what a look taken by the fakes costs (Hunter's on FakeEyes
# without a feed, the tests'), and the pick-up log's OCR (fakes.MESSAGES_S).
COSTS = {
    "coded": {},
    "measured": {"LOOK_S": 0.020, "LOOK_LAG": 0.006, "DETECT_S": 0.024, "DETECT_WIDE_S": 0.084, "LOOT_S": 0.008},
}


# --- policies ------------------------------------------------------------------------------------------

def _tour(sim) -> None:
    """Today's tour, as `python -m gamelab tour` runs it, started again whenever a sweep ends."""
    h = sim.hunter(low_hp=0.3, potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)
    while True:
        tour.run()


def _loop(sim, rest_avoid: bool = False, low_hp: float = 0.3) -> None:
    """The hunt loop (`python -m gamelab tour --loop`), on the same hunter, navigator and map knowledge."""
    from ..behavior.hunt import Hunt
    h = sim.hunter(low_hp=low_hp, potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)
    tour.rest_avoid = rest_avoid
    while True:
        Hunt(tour, log=tour.log).run(minutes=600)


def _loop_avoid(sim) -> None:
    """The hunt loop with `--rest-avoid`: the trips to rest keep clear of monsters (navigator.TRANSIT_AVOID)."""
    _loop(sim, rest_avoid=True)


def _loop_hp50(sim) -> None:
    """The hunt loop with `--low-hp 0.5`: it stops to rest under half her HP, not 30% (the Tree Dungeon study's advice:
    at 30% two Horny Mushroom touches by meowdb's formula kill her)."""
    _loop(sim, low_hp=0.5)


def _loop_grind(sim) -> None:
    """The hunt loop grinding round the map (`tour --loop --grind`: behavior.grind), on the same hunter, navigator and
    map knowledge as `loop`."""
    from ..behavior.grind import Grind
    h = sim.hunter(low_hp=0.3, potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)
    while True:
        Grind(tour, log=tour.log).run(minutes=600)


POLICIES = {"tour": _tour, "loop": _loop, "loop-avoid": _loop_avoid, "loop-hp50": _loop_hp50,
            "loop-grind": _loop_grind}


def _only(policy, grounds: tuple[int, ...]):
    """`policy`, hunting only `grounds` (`--grounds`): its tours' hunting grounds cut to those (all of them if none is
    one by the value map). One ground, as live on Tree Dungeon on 2026-10-24 (23 alone: emptied, the hunt loop left it
    for itself again, and waited there)."""
    def run(sim):
        make = sim.tour

        def tour(*a, **kw):
            t = make(*a, **kw)
            every = t.hunting_grounds
            t.hunting_grounds = lambda visit_s=None: [g for g in every(visit_s) if g in grounds] or list(grounds)
            return t
        sim.tour = tour
        return policy(sim)
    return run


# --- one run -------------------------------------------------------------------------------------------

def scenario(policy: str, seed: int, preset: str = "game", minutes: float = 20.0, think: float = 0.0,
             map_id: int = FIRST_MAP, world: tuple = (), grounds: tuple = (), attack: str | None = None):
    """The Scenario for one run: the policy hunting on map `map_id` from its START_AT[seed] (STARTS) in world `seed`
    until the horizon. `think`: charge the stack's own computing to the fake clock (fakes.Clock; the runs then vary a
    little); `world`: the switches of WORLDS on; `grounds`: the hunting grounds the policy may hunt (_only; all of them by default);
    `attack`: the stack's attack a throw (behavior.attacks.MODES), with her as ATTACK_HER (None: the regular attack,
    her as the map has her)."""
    from . import scenarios
    if map_id not in PRESET_MAPS.get(preset, (map_id,)):
        raise ValueError(f"preset {preset} is map {', '.join(map(str, PRESET_MAPS[preset]))}'s alone")
    noise, realism = PRESETS[preset]
    her = scenarios.her_on(map_id)[1]
    for name in world:
        fields = WORLDS[name]
        realism = dataclasses.replace(realism, **{k: v for k, v in fields.items()
                                                  if k not in HER_FIELDS and k not in NOISE_FIELDS})
        her = dataclasses.replace(her, **{k: v for k, v in fields.items() if k in HER_FIELDS})
        noise = dataclasses.replace(noise, **{k: v for k, v in fields.items() if k in NOISE_FIELDS})
    if attack is not None:
        her = dataclasses.replace(her, **ATTACK_HER)
    fn = POLICIES[policy] if not grounds else _only(POLICIES[policy], tuple(grounds))
    starts = STARTS[map_id]
    return scenarios.Scenario(f"{policy}:{preset}:{seed}" + (f":{map_id}" if map_id != FIRST_MAP else "")
                              + "".join(f":{w}" for w in world) + (f":g{'-'.join(map(str, grounds))}" if grounds else "")
                              + (f":attack-{attack}" if attack else ""),
                              fn,
                              starts[seed % len(starts)], monsters="all", seed=seed, limit=minutes * 60, noise=noise,
                              realism=realism, may_die=True, think=think, map_id=map_id,
                              her=her if world or attack else None, attack=attack)


def run_one(policy: str, seed: int, preset: str = "game", minutes: float = 20.0, think: float = 0.0,
            costs: str = "coded", ways: int | None = None, map_id: int = FIRST_MAP,
            world: tuple = (), grounds: tuple = (), attack: str | None = None) -> dict:
    """Run one hunt and return what it gave (outcomes). `costs`: what the simulator's looks take (COSTS); `ways`:
    the ways her trips choose among (route.ROUTES; None: as it is); `map_id`: the map (STARTS); `world`: WORLDS on;
    `attack`: the stack's attack a throw (scenario)."""
    from .. import telemetry
    from ..motion import route
    from . import fakes, scenarios
    before = {name: getattr(fakes, name) for name in COSTS[costs]}
    routes = route.ROUTES
    try:
        for name, value in COSTS[costs].items():
            setattr(fakes, name, value)
        if ways is not None:
            route.ROUTES = ways
        with telemetry.captured() as records:
            trace, sim = scenarios.run_sim(scenario(policy, seed, preset, minutes, think, map_id, world, grounds,
                                                    attack))
    finally:
        for name, value in before.items():
            setattr(fakes, name, value)
        route.ROUTES = routes
    out = outcomes(sim.world, records, minutes * 60)
    out.update(policy=policy, seed=seed, preset=preset, minutes=minutes, think=think, feed=True, costs=costs,
               ways=routes if ways is None else ways, map=map_id, result=str(trace["result"])[:120])
    if world:
        out["world"] = list(world)
    if grounds:
        out["grounds"] = list(grounds)
    if attack:
        out["attack"] = attack
    out["coverage"] = coverage.tally(records, coverage.graph_for(sim.model))  # her moves, for the coverage report
    f = trace["feed"]
    out["feed_record"] = {"frames": f["frames"], "asked": f["asked"], "busy_s": f["busy_s"],
                          "searches": f.get("searches"), "searching_s": f.get("searching_s"),
                          "senses": f["senses"], "tracker": f["tracker"]}
    out["looks"] = f["senses"].get("looks", 0)
    out["turning"] = turning(trace["plans"], records, sim.world)
    out["attack_kinds"] = attack_kinds(sim.world, records)
    out["nav"] = navigation(trace["log"], sim.model)
    out["nav"]["glances"] = f["senses"].get("glances", 0)
    return out


def turning(plans: list, records: list[dict], world=None) -> dict:
    """Her throws and jump throws (the plans the player ran), those of them that tapped an arrow to turn her first
    (control.actions.Actions._turn), her attacks that went the other way from where the stack threw them (the
    `world`'s count), and the frames that told the other way from her last arrow (telemetry events 'heading':
    behavior.hunter)."""
    throws = [p for p in plans if p[1] in ("throw", "jump_throw")]
    turned = [p for p in throws if any(e[1] in ("left", "right") for e in p[2])
              and not (p[1] == "jump_throw" and _forward(p[2]))]  # (a forward jump's arrow carries her: no tap)
    return {"throws": len(throws), "turned": len(turned),
            "jump_turned": sum(1 for p in turned if p[1] == "jump_throw"),
            "attacks": getattr(world, "attacks_meant", None), "wrong_way": getattr(world, "wrong_way", None),
            "against": sum(1 for r in records if r.get("topic") == "event" and r.get("kind") == "heading")}


def _forward(events: list) -> bool:
    """Whether a jump throw's key plan is a forward one (control.actions.Actions.jump_throw with `way`): its arrow
    held from before its jump key until after it, not a turning tap let go before the jump nor one in the air."""
    jump = min((e[0] for e in events if e[1] in ("alt", "jump") and e[2] == "down"), default=None)
    downs = [e[0] for e in events if e[1] in ("left", "right") and e[2] == "down"]
    ups = [e[0] for e in events if e[1] in ("left", "right") and e[2] == "up"]
    return jump is not None and any(d < jump for d in downs) and any(u > jump for u in ups)


def attack_kinds(world, records: list[dict]) -> dict:
    """What each kind of throw hit, from the world's own log of her attacks (world.World.attack_log): by the key plan
    that pressed it (control.actions: "throw", "jump_throw", "jump_back"; a chain's, "flow"), how many and what came of
    each ("kill", "hit", "swipe", "miss", "stopped", "nothing": `hit_nothing` the last two), those started in the air,
    and of those the ones that picked a monster whose feet were LEVEL or more above hers then (`up`); and the jump
    throws the stack chose, by why (its telemetry 'jump' records' `kind`: attack.stance), with what came of each
    (`chosen_by`: why and the jump's kind -> outcomes, the record matched to the first jump's attack keyed within
    CHOSEN_KEY_S after it; 'unmatched' if none was)."""
    out: dict = {}
    log = getattr(world, "attack_log", [])
    for _, plan, mode, _, _, outcome, up in log:
        k = out.setdefault(plan, Counter())
        k["n"] += 1
        k[outcome] += 1
        k["air"] += mode == "air"
        k["up"] += up is not None and up >= _LEVEL
        k["hit_nothing"] += outcome in ("stopped", "nothing")
    jumps = [r for r in records if r.get("topic") == "jump"]
    chosen = Counter(r.get("kind") for r in jumps)
    from .fakes import EPOCH
    keyed = sorted((t, outcome) for t, plan, *_rest, outcome, _ in log if plan in ("jump_throw", "jump_back"))
    by: dict[str, Counter] = {}
    i = 0
    for r in sorted(jumps, key=lambda r: r["t"]):
        at = r["t"] - EPOCH
        while i < len(keyed) and keyed[i][0] < at - 1e-6:
            i += 1
        hit = keyed[i][1] if i < len(keyed) and keyed[i][0] <= at + CHOSEN_KEY_S else "unmatched"
        if hit != "unmatched":
            i += 1
        by.setdefault(f"{r.get('kind')}: {r.get('jump')}", Counter())[hit] += 1
    return {"plans": {k: dict(v) for k, v in out.items()}, "chosen": dict(chosen),
            "chosen_by": {k: dict(v) for k, v in by.items()}}


CHOSEN_KEY_S = 1.5  # (attack_kinds) a chosen jump's attack key comes this soon after its record at the latest: its key
                    # at most 0.48 s into the jump (attack.reach.KEY_LAST_MS), after a wait for her lock


_LEVEL = 15.0  # (attack.reach.LEVEL) a pick this far above her feet or more is off her level, above her


def navigation(log: list[str], model) -> dict:
    """How her trips went, from the stack's log (control.navigator): plans cut short by a hit ('cut_hit') and off
    the planned path ('cut_off_path'); replans from where she ended up rather than where planned (a cut, a hit on
    the way, a landing elsewhere: once each, a cut and the landing after it once); misses, the landings elsewhere
    than planned (the trip's own count); and falls, those of them onto a platform more than 10 units lower than
    the one planned, or onto none."""
    out = {"cut_hit": 0, "cut_off_path": 0, "replans": 0, "misses": 0, "falls": 0}
    cut = False  # the line before was a cut: the landing it logs next is the same replan
    for line in log:
        if "plan cut short" in line:
            out["cut_off_path" if "off the path" in line else "cut_hit"] += 1
            out["replans"] += 1
            cut = True
            continue
        if "replanning" in line:
            out["replans"] += not cut
            if (m := _MISSED.search(line)) is not None:
                out["misses"] += 1
                planned = model.platform(int(m["planned"]))
                landed = None if m["landed"] is None else model.platform(int(m["landed"]))
                out["falls"] += landed is None or landed.top > planned.top + 10
        cut = False
    return out


_MISSED = re.compile(r"ended on (?:platform (?P<landed>\d+) \(.*?\)|air/rope) instead of platform (?P<planned>\d+)")


def outcomes(world, records: list[dict], horizon: float) -> dict:
    """What a run gave, from the simulated world (and the telemetry it recorded)."""
    end = min(horizon, world.died_at) if world.died_at is not None else horizon
    if world.fell_out_at is not None:
        end = min(end, world.fell_out_at)
    if world.left_map_at is not None:
        end = min(end, world.left_map_at)
    mine = [k for k in world.kill_log if k[3] == "her"]
    drops = [d for d in world.drops if d.owner == "her" and d.since <= horizon]
    known = [d for d in drops if d.taken_by == "her" or d.since + DROP_LIFE <= horizon]
    picked = [d for d in known if d.taken_by == "her"]
    worth = worth_for(world.model)  # what drops are worth on the world's map (looting.Worth)
    worth_known, worth_picked = sum(worth(d.kind) for d in known), sum(worth(d.kind) for d in picked)
    times = sorted([0.0] + [k[0] for k in mine] + [d.taken_at for d in world.drops if d.taken_by == "her"] + [end])
    gaps = [b - a for a, b in zip(times, times[1:])]
    events = [r for r in records if r.get("topic") == "event"]
    trips, waits = _rest_trips(records), _waits(records)
    return {
        "exp": world.exp,
        "exp_per_min": round(world.exp / (horizon / 60), 2),
        "kills": len(mine),
        "rival": sum(1 for k in world.kill_log if k[3] == "rival"),
        "hp_lost": round(sum(hp for _, hp in world.touch_log), 3),
        "touches": world.touches,
        "hurt": world.touches - world.evaded,
        **_travelling(world, records, end),
        "hp_low": round(world.hp_low, 3),
        "potions": world.potions_used,
        "casts": world.casts,
        "mp_used": round(world.mp_used, 1),
        "mp_per_min": round(world.mp_used / (end / 60), 2) if end else None,
        "mp_potions": world.mp_potions,
        "no_mp": world.no_mp,
        "attacks": world.attacks,
        "finishes": sum(1 for r in records if r.get("topic") == "attack" and r.get("why") == "finish"),
        "died_at": world.died_at,
        "fell_out_at": world.fell_out_at,
        "left_map_at": world.left_map_at,
        "drops": len(drops),
        "drops_share": round(len(picked) / len(known), 3) if known else None,
        "value_share": round(worth_picked / worth_known, 3) if worth_known else None,
        "value_per_min": round(sum(worth(d.kind) for d in drops if d.taken_by == "her") / (horizon / 60), 2),
        "longest_idle": round(max(gaps, default=0.0), 1),
        "stuck_s": round(sum(max(0.0, g - IDLE_GRACE) for g in gaps), 1),
        "rests": sum(1 for r in events if r.get("kind") == "rest"),
        "rest_trips": len(trips),
        "rest_hp_lost": round(sum(hp for t, hp in world.touch_log
                                  if t <= end and any(a <= t <= b for a, b in trips)), 3),
        "alive_s": round(end, 1),
        "fresh_hurt": sum(1 for t, hp in world.fresh_log if hp > 0 and t <= end),
        "fresh_hp_lost": round(sum(hp for t, hp in world.fresh_log if t <= end), 3),
        "wait_s": round(sum(max(0.0, min(b, end) - a) for a, b in waits), 1),
        "wait_hp_lost": round(sum(hp for t, hp in world.touch_log if t <= end and any(a <= t <= b for a, b in waits)),
                              3),
        "ropes": sum(1 for r in events if r.get("kind") == "rope"),
        "backstops": dict(Counter(r.get("text", "").split(":")[0] for r in events if r.get("kind") == "backstop")),
    }


def _travelling(world, records: list[dict], end: float) -> dict:
    """trip_s, trip_hurt and trip_hp_lost (see the top): her moves, from each 'chain' or 'move' record (a chain sent, a
    move made alone) to the next 'moved' record (chains that follow on are in their first one's), or a move the mover
    couldn't do (a 'cannot' event: nothing made), in world time, up to `end`; and the touches that hurt her then."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return {"trip_s": 0.0, "trip_hurt": 0, "trip_hp_lost": 0.0}
    spans, began = [], None
    for r in records:
        if r.get("topic") in ("chain", "move") and began is None:
            began = r["t"] - start
        elif (r.get("topic") == "moved" or r.get("topic") == "event" and r.get("kind") == "cannot") \
                and began is not None:
            spans.append((began, min(r["t"] - start, end)))
            began = None
    if began is not None:
        spans.append((began, end))
    on = [hp for t, hp in world.touch_log if t <= end and any(a <= t <= b for a, b in spans)]
    return {"trip_s": round(sum(max(0.0, b - a) for a, b in spans if a < end), 1),
            "trip_hurt": sum(1 for hp in on if hp > 0), "trip_hp_lost": round(sum(on), 3)}


def _rest_trips(records: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of each trip to rest, in world time (s from the run's start): from a 'rest_trip' event
    to the next 'rest' event (or the end)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return []
    out, began = [], None
    for r in records:
        if r.get("topic") != "event":
            continue
        if r.get("kind") == "rest_trip":
            began = r["t"] - start
        elif r.get("kind") == "rest" and began is not None:
            out.append((began, r["t"] - start))
            began = None
    if began is not None:
        out.append((began, float("inf")))
    return out


WAITING = ("wait", "lull loot")  # the hunt loop's rules for waiting out respawns (behavior.hunt: 9 and 8)


def _waits(records: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of each stretch the hunt loop's rule was one of WAITING, in world time (s from the run's start):
    from the 'hunt' record that names one to the next that names another rule (or the end)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return []
    out, began = [], None
    for r in records:
        if r.get("topic") != "hunt":
            continue
        waiting = r.get("rule") in WAITING
        if waiting and began is None:
            began = r["t"] - start
        elif not waiting and began is not None:
            out.append((began, r["t"] - start))
            began = None
    if began is not None:
        out.append((began, float("inf")))
    return out


# --- many runs -----------------------------------------------------------------------------------------

def run_many(policy: str, seeds: int, preset: str, minutes: float, jobs: int = 1, think: float = 0.0,
             costs: str = "coded", ways: int | None = None, map_id: int = FIRST_MAP,
             world: tuple = (), grounds: tuple = (), attack: str | None = None) -> list[dict]:
    args = [(policy, k, preset, minutes, think, costs, ways, map_id, tuple(world), tuple(grounds), attack)
            for k in range(1, seeds + 1)]
    if jobs <= 1:
        return [run_one(*a) for a in args]
    from gamelab.sim import harness  # by its module name, so worker processes can find it (not __main__)
    with ProcessPoolExecutor(max_workers=jobs, initializer=harness._worker) as pool:
        return list(pool.map(harness._run_one_args, args))


def _worker() -> None:
    """A worker process at below-normal priority (feed.quiet): the perception service, at normal, keeps its pace on a
    machine the runs keep busy (it looked at 20.8 frames a second beside 22 busy workers so, 14 at their priority)."""
    from ..localization.feed import quiet
    quiet()


def _run_one_args(a: tuple) -> dict:
    return run_one(*a)


COLUMNS = ("seed", "exp_per_min", "kills", "rival", "hp_lost", "hp_low", "potions", "mp_per_min", "mp_potions",
           "finishes", "died_at", "fell_out_at",
           "left_map_at",
           "drops_share", "value_share", "value_per_min", "longest_idle", "stuck_s", "rests", "rest_hp_lost",
           "alive_s", "hurt", "fresh_hurt", "wait_s", "wait_hp_lost")


def table(results: list[dict]) -> str:
    rows = [" ".join(f"{c:>12}" for c in COLUMNS)]
    for r in results:
        rows.append(" ".join(f"{_fmt(r.get(c)):>12}" for c in COLUMNS))
    rows.append(" ".join(f"{_fmt(v):>12}" for v in ["mean"] + [_mean(results, c) for c in COLUMNS[1:]]))
    caps = Counter()
    for r in results:
        caps.update(r["backstops"])
    rows.append(f"backstops fired: {dict(caps) or 'none'}")
    return "\n".join(rows)


def _mean(results: list[dict], key: str):
    vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    if key in ("died_at", "fell_out_at", "left_map_at"):
        return sum(1 for r in results if r.get(key) is not None)  # deaths, falls out of the map, portals taken
    return round(statistics.fmean(vals), 3) if vals else None


def _fmt(v) -> str:
    return "-" if v is None else str(v)


# --- the bar -------------------------------------------------------------------------------------------

MIN_PAIRS = 10  # the bars are for 10 seeds or more, in both runs


def _paired(base: list[dict], cand: list[dict]) -> tuple[list[dict], list[dict]]:
    """The runs of the seeds both have, in the same order (means over different worlds compare nothing)."""
    by_seed = {r["seed"]: r for r in base}
    pairs = [(by_seed[r["seed"]], r) for r in cand if r["seed"] in by_seed]
    return [a for a, _ in pairs], [b for _, b in pairs]


def compare(base: list[dict], cand: list[dict]) -> tuple[bool, list[str]]:
    """Whether `cand` passes the bar against `base` (seed by seed, over the seeds both have: MIN_PAIRS at
    least), and why: better EXP a minute on at least 8 of 10 seeds and no seed more than 10% worse; at
    least the drops share on average; no more HP lost on average; no death, and no fall out of the map."""
    every = cand  # deaths count on every seed, compared or not: "no death" isn't a comparison
    if (maps := _maps(base, cand)) is not None:
        return False, [maps, "does NOT pass the bar"]
    base, cand = _paired(base, cand)
    if len(cand) < MIN_PAIRS:
        return False, [f"{len(cand)} seed(s) in both runs: the bar needs {MIN_PAIRS}", "does NOT pass the bar"]
    pairs = list(zip(base, cand))
    lines, ok = [], True
    if len(every) > len(cand):
        lines.append(f"{len(every) - len(cand)} candidate seed(s) without a baseline: not compared")
    better = sum(1 for a, b in pairs if b["exp_per_min"] > a["exp_per_min"])
    worst = min((b["exp_per_min"] / a["exp_per_min"] - 1 for a, b in pairs if a["exp_per_min"]), default=0.0)
    need = -(-len(pairs) * 8 // 10)
    lines.append(f"EXP a minute better on {better} of {len(pairs)} seeds (need {need}); worst seed {worst:+.0%}")
    ok &= better >= need and worst >= -0.10
    for key, higher in (("drops_share", True), ("hp_lost", False)):
        a, b = _mean(base, key), _mean(cand, key)
        good = a is not None and b is not None and (b >= a if higher else b <= a)
        lines.append(f"{key}: {a} -> {b} ({'ok' if good else 'worse' if a is not None and b is not None else 'missing'})")
        ok &= good
    deaths = sum(1 for r in every if r.get("died_at") is not None)
    lines.append(f"deaths: {deaths}")
    ok &= deaths == 0
    fell = [r["seed"] for r in every if r.get("fell_out_at") is not None]
    if fell:
        lines.append(f"fell out of the map: seed(s) {', '.join(map(str, fell))}")
    ok &= not fell
    away = [r["seed"] for r in every if r.get("left_map_at") is not None]
    if away:
        lines.append(f"went through a portal: seed(s) {', '.join(map(str, away))}")
    ok &= not away
    if all("nav" in r for r in base + cand):  # (runs saved before it was counted have none)
        lines.append("her trips, in all (shown only): " + ", ".join(
            f"{key} {sum(r['nav'][key] for r in base)} -> {sum(r['nav'][key] for r in cand)}" for key in NAV))
    lines.append("PASSES the bar" if ok else "does NOT pass the bar")
    return ok, lines


NAV = ("cut_hit", "cut_off_path", "replans", "misses", "falls", "glances")  # navigation()'s counts, and the glances


def _maps(base: list[dict], cand: list[dict]) -> str | None:
    """Why two saved runs can't be compared: they're on different maps (runs saved before the map was kept are the
    first map's), or None."""
    a, b = ({r.get("map", FIRST_MAP) for r in runs} for runs in (base, cand))
    return None if a == b and len(a) == 1 else f"the runs are on different maps ({sorted(a)} and {sorted(b)})"


def compare_rest(base: list[dict], cand: list[dict]) -> tuple[bool, list[str]]:
    """Step 7's bar (`--rest-avoid`) on 20-minute runs: no more HP lost on the trips to rest on average
    (over the seeds both have, and with trips to rest on both sides), and no death. EXP a minute and the
    drops share are shown, not judged. (`rests` measures the trips themselves, with far more of them.)"""
    every = cand
    if (maps := _maps(base, cand)) is not None:
        return False, [maps, "does NOT pass the rest bar"]
    base, cand = _paired(base, cand)
    if len(cand) < MIN_PAIRS:
        return False, [f"{len(cand)} seed(s) in both runs: the bar needs {MIN_PAIRS}", "does NOT pass the rest bar"]
    lines, ok = [], True
    a, b = _mean(base, "rest_hp_lost"), _mean(cand, "rest_hp_lost")
    trips = (sum(r.get("rest_trips") or 0 for r in base), sum(r.get("rest_trips") or 0 for r in cand))
    a = a if trips[0] else None  # no trip to rest measured on a side: nothing to judge
    b = b if trips[1] else None
    good = a is not None and b is not None and b <= a  # runs saved before rest_hp_lost existed don't count
    lines.append(f"rest_hp_lost: {a} -> {b} ({'ok' if good else 'worse' if a is not None and b is not None else 'missing'}); rest trips "
                 f"{_mean(base, 'rest_trips')} -> {_mean(cand, 'rest_trips')}")
    ok &= good
    deaths = sum(1 for r in every if r.get("died_at") is not None)
    lines.append(f"deaths: {deaths}")
    ok &= deaths == 0
    for key in ("exp_per_min", "drops_share", "hp_lost"):
        lines.append(f"{key}: {_mean(base, key)} -> {_mean(cand, key)} (shown only)")
    lines.append("PASSES the rest bar" if ok else "does NOT pass the rest bar")
    return ok, lines


def save(name: str, results: list[dict]) -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / f"{name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def load(name: str) -> list[dict]:
    return json.loads((FOLDER / f"{name}.json").read_text(encoding="utf-8"))


# --- trips to rest (step 7) ------------------------------------------------------------------------------

REST_FROM = [(36, 300.0), (36, 800.0), (17, 1100.0), (4, 450.0), (12, -300.0), (2, 0.0), (18, 300.0),
             (21, -300.0), (1, 1200.0), (34, -300.0)]  # hunting grounds, where a stop on "HP low" leaves her
REST_HP = 0.28  # under the hunter's 30%: the trip a stop on low HP starts


def rest_trip(avoid: bool, k: int) -> dict:
    """One trip to rest from REST_FROM[k], at REST_HP, in world 100 + k of the game preset: the touches that
    hurt her on the way (from the 'rest_trip' event to the 'rest' one: not while she sits) and the HP they
    took, how long the trip took (None if it never ended), and whether she died."""
    from .. import telemetry
    from . import scenarios
    noise, realism = PRESETS["game"]

    def fn(sim):
        tour = sim.tour_stack()
        tour.rest_avoid = avoid
        tour.rest(max_seconds=0.1)

    sc = scenarios.Scenario("rest", fn, REST_FROM[k % len(REST_FROM)], monsters="all", seed=100 + k, hp=REST_HP,
                            limit=300, noise=noise, realism=realism, may_die=True)
    with telemetry.captured() as records:
        _, sim = scenarios.run_sim(sc)
    w = sim.world
    a, b = (_rest_trips(records) or [(0.0, float("inf"))])[0]
    return {"avoid": avoid, "k": k, "hurt": sum(1 for t, hp in w.touch_log if hp and a <= t <= b),
            "hp_lost": sum(hp for t, hp in w.touch_log if a <= t <= b),
            "dead": w.dead, "seconds": round(b - a, 1) if b != float("inf") else None,
            "kills": sum(1 for t, _, _, who in w.kill_log if who == "her" and a <= t <= b)}


def _rest_trip_args(a: tuple) -> dict:
    return rest_trip(*a)


def rest_trials(trips: int, jobs: int = 1) -> tuple[bool, list[str]]:
    """`trips` trips to rest each way (transit, and transit with avoidance: `--rest-avoid`), paired by start
    and world. Step 7's bar: less HP lost on the way on average, and no death."""
    args = [(a, k) for a in (False, True) for k in range(trips)]
    if jobs <= 1:
        res = [rest_trip(*a) for a in args]
    else:
        from gamelab.sim import harness
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            res = list(pool.map(harness._rest_trip_args, args))
    lines, means = [], {}
    for avoid in (False, True):
        rs = [r for r in res if r["avoid"] == avoid]
        done = [r["seconds"] for r in rs if r["seconds"] is not None]
        means[avoid] = statistics.fmean(r["hp_lost"] for r in rs)
        lines.append(f"{'avoid' if avoid else 'transit'}: HP lost {means[avoid]:.1%} a trip, "
                     f"{statistics.fmean(done) if done else float('nan'):.1f} s, deaths {sum(r['dead'] for r in rs)}, "
                     f"kills on the way {sum(r['kills'] for r in rs)}, not there in time {len(rs) - len(done)}")
    ok = means[True] <= means[False] and not any(r["dead"] for r in res if r["avoid"])
    lines.append("--rest-avoid PASSES the rest bar" if ok else "--rest-avoid does NOT pass the rest bar")
    return ok, lines


# --- checking the bookkeeping ----------------------------------------------------------------------------

def check() -> int:
    """Run every scenario and check that the world's own logs, which outcomes() reads, agree with its
    summary: her kills, pick-ups and drops."""
    from . import scenarios
    bad = 0
    for name, sc in scenarios.SCENARIOS.items():
        trace, sim = scenarios.run_sim(sc)
        w, summary = sim.world, trace["world"]
        kills = sum(1 for k in w.kill_log if k[3] == "her")
        picked = sum(1 for d in w.drops if d.taken_by == "her")
        problems = []
        if kills != sum(summary["kills"].values()):
            problems.append(f"kills {kills} vs {summary['kills']}")
        if picked != sum(summary["picked"].values()):
            problems.append(f"picked {picked} vs {summary['picked']}")
        print(f"{'ok' if not problems else 'DIFFERS'} {name}" + (f": {'; '.join(problems)}" if problems else ""))
        bad += bool(problems)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from ..motion.physics import FIXED_STATS
    # The same set order in every run, and her stats as measured, not as config.toml has them.
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get(FIXED_STATS) != "1":
        env = dict(os.environ, PYTHONHASHSEED="0", **{FIXED_STATS: "1"})
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.harness", *argv], env=env,
                               cwd=str(REPO))
    p = argparse.ArgumentParser(prog="python -m gamelab.sim.harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("policy", choices=sorted(POLICIES))
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--minutes", type=float, default=20.0)
    r.add_argument("--preset", choices=sorted(PRESETS), default="game")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--think", type=float, default=0.0,
                   help="charge the stack's computing to the fake clock, times this (1: this computer's speed)")
    r.add_argument("--save",
                   help="name to save the results under (default: POLICY-PRESET, then -COSTS and -waysN)")
    r.add_argument("--ways", type=int, help="the ways her trips choose among (route.ROUTES; default: route's own)")
    r.add_argument("--costs", choices=sorted(COSTS), default="coded",
                   help="what the simulator's looks take: coded (the fakes' own), measured (today's)")
    r.add_argument("--map", type=int, choices=sorted(STARTS), default=FIRST_MAP,
                   help="the map (default: the first, Thicket Around the Beach II; 10004021: West Rocky Mountain II; "
                        "10002073: Tree Dungeon, Forest Up North II)")
    r.add_argument("--world", action="append", choices=sorted(WORLDS), default=[],
                   help="what isn't known of the game, switched on (WORLDS; again for another)")
    r.add_argument("--grounds", type=lambda v: tuple(int(g) for g in v.split(",")), default=(),
                   help="hunt only these grounds, e.g. 23 (default: all of them)")
    r.add_argument("--attack", choices=("choice", "skill", "regular"),
                   help="the stack's attack a throw (behavior.attacks), her Lucky Seven key casting it at its MP "
                        "(ATTACK_HER): choice, Lucky Seven but the regular attack to finish a monster and while her MP "
                        "is under a cast (live's default); skill, Lucky Seven while she has the MP; regular, the "
                        "regular attack (default: the regular attack, her as the map has her)")
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--bar", choices=("hunt", "rest"), default="hunt",
                   help="hunt: the loop's bar (default); rest: --rest-avoid's (HP lost on the trips to rest)")
    sub.add_parser("check")
    t = sub.add_parser("rests")
    t.add_argument("--trips", type=int, default=40)
    t.add_argument("--jobs", type=int, default=1)
    args = p.parse_args(argv)
    from . import scenarios
    map_id = getattr(args, "map", FIRST_MAP)
    if not scenarios.available(map_id):
        print(f"map {map_id} isn't available on this computer (gamedata/data): nothing to run")
        return 2
    if args.cmd == "check":
        return check()
    if args.cmd == "rests":
        ok, lines = rest_trials(args.trips, args.jobs)
        print("\n".join(lines))
        return 0 if ok else 1
    if args.cmd == "compare":
        ok, lines = (compare_rest if args.bar == "rest" else compare)(load(args.baseline), load(args.candidate))
        print("\n".join(lines))
        return 0 if ok else 1
    if args.map not in PRESET_MAPS.get(args.preset, (args.map,)):
        print(f"preset {args.preset} is map {', '.join(map(str, PRESET_MAPS[args.preset]))}'s alone")
        return 2
    results = run_many(args.policy, args.seeds, args.preset, args.minutes, args.jobs, args.think, args.costs,
                       args.ways, args.map, tuple(args.world), args.grounds, args.attack)
    print(table(results))
    name = args.save or (f"{args.policy}-{args.preset}"
                         + (f"-{args.costs}" if args.costs != "coded" else "")
                         + (f"-ways{args.ways}" if args.ways is not None else "")
                         + (f"-{args.map}" if args.map != FIRST_MAP else "")
                         + "".join(f"-{w}" for w in args.world)
                         + (f"-g{'-'.join(map(str, args.grounds))}" if args.grounds else "")
                         + (f"-attack-{args.attack}" if args.attack else ""))
    save(name, results)
    print(f"saved {FOLDER / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

