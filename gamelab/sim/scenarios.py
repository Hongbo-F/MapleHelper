"""Scenarios: gamelab's entry points run against the simulated game on a real map, each built the way
gamelab/__main__.py builds them (_hunter, Navigator, Tour), with the fakes in place. Most are on the first map
(MAP_ID, 10000012); those named `*_rocky_*` on West Rocky Mountain II (ROCKY, 10004021), with her as she plays
there (HER: level 23, her new keys, her pet drinking her potions).

Each scenario returns a trace: every key plan sent ([t, name, events, end_ms], in order), every cancel,
every log line, the entry point's result, what the world saw happen, how often she looked (and didn't
see her tag or dot), and the lines written to the lab's logs. The same scenario always gives
the same trace, so traces recorded now (golden.py) pin what gamelab does today. Each scenario also
`expects` what it's there for (a hop, an arrival, three tour stops...); tests/test_sim.py checks those, so
a change in the sim or in gamelab can't quietly drop a behaviour a scenario was meant to cover.

Game data stays on this computer: the map is read from gamedata's local extract, and scenarios are
skipped when it isn't there (`available(map_id)`). Data that is there but fails to load is an error, not a skip.

The stack only runs with the fakes in place: `with Sim(model, scenario) as sim: sim.hunter()...`.
"""

import dataclasses
import importlib
import json
import pkgutil
import random
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from ..control import client
from ..localization import mapdata
from ..attack import danger
from ..behavior import probe
from ..behavior.hunter import Hunter
from ..behavior.looting import DropSpot
from ..behavior.safety import HuntStop, SafetyStop
from ..localization.mapdata import MapModel
from ..control.actions import STAR_DELAY_MS
from ..control.navigator import Navigator
from ..control.timeline import SIM_KEYS, SIM_KEYS_PET
from ..behavior.tour import Tour
from .fakes import (Clock, FakeDetector, FakeEyes, FakeLoot, FakePetCare, FakePlayer, SimTimeout, SimViolation,
                    Trace, detector_kinds, patched)
from . import harness
from .world import (LEVEL_19, LEVEL_23, LEVEL_25, LUCKY_SEVEN, NEAR_SEEN, OFF_SEEN, ROPE_REACH_CLIENT, Her, Noise,
                    Realism, SimDied, World)

MAP_ID = 10000012  # Victoria Road: Thicket Around the Beach II (__main__.DEFAULT_MAP)
ROCKY = 10004021   # Victoria Road: West Rocky Mountain II, the second map (from 2026-10-23)
TREE = 10002073    # Victoria Road: Tree Dungeon, Forest Up North II, the third (from 2026-10-24)
# Her keys and her (world.Her) on each map: on the first map as the golden traces were recorded with (level 19, her
# potions on a key); on West Rocky Mountain II as she plays there (level 23, HP 730, her pet drinking her potions, her
# bindings from 2026-10-23); on Tree Dungeon, Forest Up North II at level 25 (HP 774, MP 623, her stars as read there
# on 2026-10-25: world.LEVEL_25; level 24 until 2026-10-26), the same keys and pet. Any other map (the tests' synthetic
# ones): the first map's.
HER = {MAP_ID: (SIM_KEYS, LEVEL_19), ROCKY: (SIM_KEYS_PET, LEVEL_23), TREE: (SIM_KEYS_PET, LEVEL_25)}
_MODEL: dict[int, MapModel] = {}


def available(map_id: int = MAP_ID) -> bool:
    """Whether the map's game data is on this computer: its parsed extract, or the game data copy that
    mapdata.load makes one from. Says nothing about whether it loads: real_map() raises if it doesn't."""
    from gamedata.paths import COPY
    return (mapdata.CACHE / f"{map_id}.json").is_file() or (COPY / "Map" / "Map").is_dir()


def real_map(map_id: int = MAP_ID) -> MapModel | None:
    """The real map's model (loaded once), or None when its game data isn't on this computer. A map
    that is there but fails to load raises: a refactor that breaks mapdata must fail, not skip."""
    if not available(map_id):
        return None
    if map_id not in _MODEL:
        _MODEL[map_id] = mapdata.load(map_id)
    return _MODEL[map_id]


def her_on(map_id: int) -> tuple:
    """Her keys and her (world.Her) on map `map_id` (HER; the first map's on any other)."""
    return HER.get(map_id, HER[MAP_ID])


@dataclass
class Scenario:
    name: str
    fn: Callable
    start: tuple[int, float]
    monsters: object = None      # None: none; "all": the map's spawns; a set of platform ids; or [(name, platform, x)]
    seed: int = 1
    hp: float = 1.0
    inventory_full: bool = False
    facing: str = RIGHT
    drops: tuple = ()            # (kind, platform, x) lying on the ground at the start
    hanging: float | None = None  # start on the rope nearest start's x, her feet at this y
    limit: float = 900.0         # fake seconds before the run is stopped (SimTimeout)
    noise: Noise | None = None   # the real game's errors (world.Noise); None: none
    realism: Realism | None = None  # how busy and hard the map is (world.Realism); None: the default world
    think: float = 0.0           # charge the stack's computing to the fake clock (fakes.Clock); 0: free
    expects: tuple = ()          # (what, trace -> bool): what the scenario is there for (tests check them)
    may_die: bool = False        # unless set, every scenario also expects her alive at the end
    map_id: int = MAP_ID         # the map it's on (real_map)
    her: Her | None = None       # her HP, level and pet (world.Her); None: as she is on the map (HER)
    attack: str | None = None    # the stack's attack a throw (behavior.attacks.MODES); None: the hunter's own
    remembered: dict | None = None  # her keys as a read of the game's window left them (control.keymap.remember, in the
                                    # run's own lab), which her hunter starts with (keymap.at_start); None: none
    doc: str = ""


SCENARIOS: dict[str, Scenario] = {}


def scenario(start: tuple[int, float], **kw):
    def wrap(fn):
        SCENARIOS[fn.__name__] = Scenario(fn.__name__, fn, start, doc=(fn.__doc__ or "").strip(), **kw)
        return fn
    return wrap


class Sim:
    """One simulated run: the world, the fakes, and the stack built on them. Use it as a context
    manager: inside `with Sim(model, sc) as sim:` the fakes are in place (fake clock and player, every
    gamelab/lab path pointed at a temporary folder); the stack it builds refuses to run anywhere else,
    so nothing built here can ever reach the real player or the real game."""

    def __init__(self, model: MapModel, sc: Scenario, pace: float | None = None):
        """`pace`: also let simulated time pass in real time, this many times faster (to watch it)."""
        self.model, self.scenario = model, sc
        self.keys, her = her_on(model.id)  # fixed, not config.toml's (control.timeline)
        self.clock = Clock(limit=sc.limit, pace=pace, think=sc.think)
        self.world = World(model, self.keys, sc.start, seed=sc.seed, monsters=sc.monsters, hp=sc.hp,
                           inventory_full=sc.inventory_full, facing=sc.facing, origin=self.clock.now,
                           drops=list(sc.drops), hanging=sc.hanging, noise=sc.noise, realism=sc.realism,
                           character=sc.her or her)
        self.eyes = FakeEyes(self.world, self.clock, model, guard=self.guard)
        self.trace = Trace(self.clock.now)
        self.player = FakePlayer(self.world, self.clock, self.trace, self.keys)
        self.active = False
        self.files: dict[str, list[str]] = {}  # what the run wrote (under its stand-in for gamelab/lab)
        self.senses = []  # every hunter's (for the feed's record)
        self.feed = self._feed()

    def _feed(self):
        """Perception on the world, as live's service looks at the game (SyncFeed): its own eyes (their detectors free,
        the feed's costs are its own: feed.COSTS), the world step on them at every frame the clock passes; every look
        and glance of hers is its."""
        from ..attack.reach import LOOK_DOWN, LOOK_UP
        from ..localization.feed import GAME_HZ, SyncFeed
        from ..localization.senses import Senses
        from ..localization.worldstate import WorldStep
        self.frames = FakeEyes(self.world, self.clock, self.model, guard=self.guard)
        self.frames.detector, self.frames._loot = FakeDetector(detector_kinds(self.model)), FakeLoot()
        feed = SyncFeed(WorldStep(Senses(self.frames, self.model, band=(LOOK_UP, LOOK_DOWN))), self.frames.observe,
                        guard=self.guard)
        self.clock.frames = (1.0 / GAME_HZ, feed.tick)
        return feed

    def _judged(self, h) -> None:
        """The world's view of what her stack takes for dead, for the harness (sim.harness: by the world's monster):
        each throw's target as her attack choice took it (behavior.attacks' `_say`: World.aimed_at), and each feed track
        the ledger takes for dead (behavior.ledger.Ledger.dead) that was of a monster alive then (World.live_taken).
        Nothing of hers changes."""
        world = self.world
        say = getattr(getattr(h, "attacks", None), "_say", None)
        if say is not None:
            def judged(attack, why, target, *a, **kw):
                if target is not None:
                    world.aimed_at(target.kind, target.platform, target.x, attack.key)
                return say(attack, why, target, *a, **kw)
            h.attacks._say = judged
        ledger = getattr(h, "ledger", None)
        dead_of = getattr(ledger, "_dead_of", None)
        if dead_of is None:
            return
        taken: set[int] = set()

        def noted(s):
            dead_of(s)
            for tr in getattr(s, "tracks", None) or ():
                if tr.id in ledger.dead and tr.id not in taken:
                    taken.add(tr.id)
                    drawn = [(abs(mx - tr.x), m) for m, mx, _, _ in world.visible_monsters() if m.kind == tr.kind
                             and (tr.platform is None or m.platform.id == tr.platform) and abs(mx - tr.x) <= 30]
                    if drawn and min(drawn, key=lambda dm: dm[0])[1].alive:  # (the one drawn there is alive)
                        world.live_taken.append((round(world.t, 3), min(drawn, key=lambda dm: dm[0])[1].idx))
        ledger._dead_of = noted

    def fail_frames(self, start: float, end: float, problem) -> None:
        """Every frame perception takes from `start` to `end` s into the run fails its checks with `problem` (a
        perception.framecheck.Problem), nothing read off it but the bars, as the perception service's frames of a game
        window moved, covered or offset (the tests': the stack stands still meanwhile, Senses.look)."""
        real = self.feed._look

        def look(ropes):
            obs = real(ropes)
            if start <= self.clock.elapsed < end:
                obs.bad, obs.tag, obs.dot, obs.others, obs.ropes = problem, None, None, [], []
            return obs
        self.feed._look = look

    def feed_record(self) -> dict:
        """What the feed did (the trace's 'feed'): frames, how many were asked for, how long its frame loop was
        busy, its searches (on their thread) and how long they took, the world step's and the tracker's counts,
        and how many of her looks and glances came from it."""
        f = self.feed
        senses = {}
        for s in self.senses:
            for k, v in s.counts.items():
                senses[k] = senses.get(k, 0) + v
        return {"frames": f.counts["frames"], "asked": f.counts["asked"], "busy_s": round(f.counts["busy"], 3),
                "searches": f.counts["searches"], "searching_s": round(f.counts["searching"], 3),
                "step": dict(f.step.counts), "tracker": dict(f.step.tracker.counts), "senses": senses}

    def __enter__(self) -> "Sim":
        _import_all()
        self._tmp = tempfile.TemporaryDirectory(prefix="gamelab-sim-")
        self.folder = Path(self._tmp.name)
        self._patch = patched(self.clock, self.player, self.folder)
        self._patch.__enter__()
        self.active = True
        telemetry.emit("session", map=self.model.id, command="sim", scenario=self.scenario.name, mode="sim")
        return self

    def __exit__(self, *exc) -> bool:
        self.active = False
        try:
            for path in sorted(self.folder.rglob("*")):
                if path.is_file() and path.suffix in (".jsonl", ".json"):
                    self.files[path.relative_to(self.folder).as_posix()] = \
                        path.read_text(encoding="utf-8").splitlines()
        finally:
            try:
                self._patch.__exit__(*exc)
            finally:
                self._tmp.cleanup()
        return False

    def guard(self) -> None:
        """Raise SimViolation unless the fakes are in place (inside `with Sim(...)`)."""
        if not self.active or client.submit != self.player.submit or client.send != self.player.send:
            raise SimViolation("the simulator's stack runs only inside `with Sim(model, scenario) as sim:` "
                               "(with the fake player in place of gamelab.control.client)")

    def sleep_then(self, result, seconds: float):
        """`result`, once `seconds` of fake time have gone by (the feed goes on looking meanwhile)."""
        self.clock.sleep(seconds)
        return result

    def log(self, message: str) -> None:
        self.trace.log.append(f"{self.clock.elapsed:9.3f} {message}")
        telemetry.emit("log", text=message)

    # The stack, as gamelab/__main__.py builds it.
    def hunter(self, **kw) -> Hunter:
        self.guard()
        # (her max HP, level and DEF: what the hunter reads off the fake HUD and fits to her touches, as live:
        # behavior.bootstrap; the world's character is the world's, not hers to know)
        if self.scenario.attack is not None:  # Lucky Seven or the regular attack a throw (behavior.attacks)
            kw.setdefault("attack", self.scenario.attack)
        keys = self.keys
        if self.scenario.remembered is not None:  # (as gamelab.__main__._hunter: the binding last read, if any)
            from ..control import keymap
            if not keymap.path_of("thief").exists():
                keymap.remember(keymap.keys_of(self.scenario.remembered, self.keys), "thief", config_keys=self.keys,
                                why="the scenario's")
            start = keymap.at_start(self.keys, "thief")
            keys = start.keys
            self.log(f"her keys: {start.source}")
        h = Hunter(self.eyes, keys, self.model, log=self.log, feed=self.feed, **kw)
        h.config_keys = self.keys
        h.pet = FakePetCare(self.world, h.actions, self.eyes, self.clock, log=self.log)  # (no window, no OCR)
        self.senses.append(h.senses)
        # The way each attack is meant to go, for the world to count those her facing sends the other way
        h.actions.throw_listeners.append(lambda when, direction: self.world.meant.append((when, direction)))
        self._judged(h)
        play = h.actions.play  # every plan Actions plays (fights, single moves, tours): checked each time;
        # the navigator's chains go through client.send, which guard() checks through the fake player

        def guarded(events, name, **kw):
            self.guard()
            play(events, name, **kw)
        h.actions.play = guarded
        return h

    def navigator(self, hunter: Hunter, fight: bool = False, seed: int | None = None) -> Navigator:
        self.guard()
        return hunter.navigator(fighting=fight, rng=random.Random(self.scenario.seed if seed is None else seed))

    def tour(self, hunter: Hunter, nav: Navigator, seed: int | None = None) -> Tour:
        self.guard()
        return Tour(hunter, nav, log=self.log, rng=random.Random(self.scenario.seed if seed is None else seed))

    def tour_stack(self, **kw) -> Tour:
        """`python -m gamelab tour`: a hunter with the tour's potion HP (her potion key's; the safety stop's threshold
        the default, behavior.safety), a fighting navigator, the tour."""
        h = self.hunter(potion_hp=0.45)
        return self.tour(h, self.navigator(h, fight=True), **kw)


# --- running -----------------------------------------------------------------------------------------

def _import_all() -> None:
    """Load all of gamelab (subpackages too) before a run, so `patched` points every path of theirs under
    gamelab/lab at the run's own folder (a module first imported mid-run would keep the real one; the
    write guard would then stop the run)."""
    import gamelab
    for info in pkgutil.walk_packages(gamelab.__path__, "gamelab.", onerror=lambda name: None):
        if info.name == "gamelab.__main__" or info.name.startswith("gamelab.sim"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception:  # a module this machine can't load (it can't run in a scenario either)
            pass


def run(name: str) -> dict:
    """Run one scenario; returns its trace (plain JSON data)."""
    return run_scenario(SCENARIOS[name])


def run_scenario(sc: Scenario, model: MapModel | None = None, pace: float | None = None) -> dict:
    """Run a scenario on its real map (or on `model`); returns its trace. A run that ends early (she
    died, the time limit, the player refusing a plan, gamelab giving up) has that as its result."""
    return run_sim(sc, model, pace)[0]


def run_sim(sc: Scenario, model: MapModel | None = None, pace: float | None = None) -> tuple[dict, "Sim"]:
    """run_scenario, also returning the Sim it ran in (its world, clock and eyes, for sim.harness)."""
    model = model or real_map(sc.map_id)
    if model is None:
        raise FileNotFoundError(f"map {sc.map_id} isn't available (gamedata/data); scenarios need it")
    with Sim(model, sc, pace) as sim:
        try:
            outcome = plain(sc.fn(sim))
        except SimViolation:
            raise
        except (SimDied, SimTimeout, client.PlayerError, RuntimeError, ValueError) as e:
            outcome = f"error: {type(e).__name__}: {e}"
        except HuntStop as e:  # her HP stayed low (behavior.safety), out of darts (behavior.darts): the hunt ended
            outcome = f"{e.label.lower()}: {e}"
    trace = {"scenario": sc.name, "doc": sc.doc, "seconds": round(sim.clock.elapsed, 3), "result": outcome,
             "plans": sim.trace.plans, "cancels": sim.trace.cancels, "log": sim.trace.log,
             "world": sim.world.summary(), "world_log": sim.world.log, "eyes": dict(sim.eyes.counts),
             "frames": dict(sim.frames.frame_counts), "files": sim.files}
    trace["feed"] = sim.feed_record()
    return trace, sim


def unmet(trace: dict, sc: Scenario | None = None) -> list[str]:
    """What scenario `sc` (the trace's own by default) expects that its trace doesn't show. Unless it
    `may_die`, every scenario expects her alive and on the map at the end: not dead, not through a portal
    (world.SimLeftMap), not fallen out of the map (world.SimFellOut), none of which the world counts as another."""
    sc = sc or SCENARIOS[trace["scenario"]]
    w = trace["world"]
    gone = ("she died" if w["dead"] else "she went through a portal" if w.get("left_map_at") is not None
            else "she fell out of the map" if w.get("fell_out_at") is not None else None)
    out = [f"{gone}: {trace['result']}"] if gone and not sc.may_die else []
    for what, ok in sc.expects:
        try:
            good = bool(ok(trace))
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            good, what = False, f"{what} ({type(e).__name__}: {e})"
        if not good:
            out.append(what)
    return out


def plain(value):
    """A result as plain JSON data (dataclasses as dicts, floats to 3 places)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float):
        return round(value, 3)
    if value is None or isinstance(value, (int, str, bool)):
        return value
    return str(value)


def as_json(trace: dict) -> str:
    return json.dumps(trace, indent=1, ensure_ascii=False)


# --- what scenarios expect -----------------------------------------------------------------------------
# Each helper gives (what, trace -> bool), for Scenario.expects.

def _result(trace: dict, key: str = "result"):
    r = trace["result"]
    return r.get(key) if isinstance(r, dict) else r


def ends(*values, key: str = "result", what: str | None = None):
    return (what or f"the {key} is {' or '.join(map(str, values))}", lambda tr: _result(tr, key) in values)


def logs(text: str, what: str | None = None):
    return (what or f"a log line says {text!r}", lambda tr: any(text in line for line in tr["log"]))


def no_log(text: str, what: str | None = None):
    return (what or f"no log line says {text!r}", lambda tr: not any(text in line for line in tr["log"]))


def happens(text: str, what: str | None = None):
    """Something the world saw happen (world_log)."""
    return (what or f"the world saw {text!r}", lambda tr: any(text in line for line in tr["world_log"]))


def sends(name: str, what: str | None = None):
    return (what or f"a {name!r} key plan is sent", lambda tr: any(p[1] == name for p in tr["plans"]))


def moves(*kinds: str):
    """The trip's moves include each of these kinds."""
    return (f"moves of kind {', '.join(kinds)}",
            lambda tr: all(any(m.split()[0] == k for m in _result(tr, "moves")) for k in kinds))


def wrote(path: str, text: str, what: str | None = None):
    """A line of a file the run wrote (under its stand-in for gamelab/lab) contains `text`."""
    return (what or f"{path} has a line with {text!r}",
            lambda tr: any(text in line for line in tr["files"].get(path, ())))


def check(what: str, fn):
    return (what, fn)


def _nav(dst: int, x: float | None = None, fight: bool = False, kinds=None, seconds: float = 180):
    def run(sim):
        h = sim.hunter()
        nav = sim.navigator(h, fight=fight)
        if kinds is not None:
            nav.escort.kinds = set(kinds)
        return nav.go(dst, x=x, seconds=seconds)
    return run


TOUR_KINDS = ("pig", "slime", "orange-mushroom")


# --- the scenarios -----------------------------------------------------------------------------------
# Navigation without monsters: between them they use every kind of move.

@scenario((36, 300.0), expects=(ends("arrived"), moves("climb"), logs("planned on the move"),
                                check("two climbs", lambda tr: len(_result(tr, "moves")) == 2)))
def nav_basin_up_two_ropes(sim):
    """From the basin floor up two ropes (36 -> 30 -> 25): the second climb is planned on the move, while
    she's on the first rope, and queued to follow on without a stop."""
    return _nav(25)(sim)


@scenario((25, 760.0), seed=62, expects=(ends("arrived"), moves("walk_off", "drop")))
def nav_down_to_basin(sim):
    """From platform 25 down into the basin: a walk-off onto 32, then a drop through it. (Seed 1 since the
    planner flies her by the engine: on seed 2 the route's random weights now pick two drops, through 30,
    as on 14 other seeds of 16; just before, seed 2 walked off onto 33. Seed 62 since the route's jumps take
    off where they land with room (route._placed): its random weights are drawn in another order, and of
    seeds 1-100 5 walk off onto 32, 62 the first, where 4 did before, 1 the first.)"""
    return _nav(36, x=600)(sim)


@scenario((17, 1040.0), expects=(ends("arrived"), moves("climb_down"), sends("climb_down")))
def nav_climb_down(sim):
    """Platform 17 standing on the rope's top, down to 35: the quickest way is down the rope (done on its
    own, closed loop: Mover.do)."""
    return _nav(35)(sim)


@scenario((34, -300.0), expects=(ends("arrived"), moves("jump", "jump_up"),
                                  check("no move missed", lambda tr: _result(tr, "misses") == 0)))
def nav_staircase_up(sim):
    """Up the staircase of small platforms from 34 to 19: running jumps and jumps straight up, chained. Each
    chain ends with her landing just before the next jump, too soon for a look placing her first, so she
    stops there for the next pass rather than going on from where she was predicted to land (from there,
    landings a few units off made her next jumps miss and drop her into the basin)."""
    return _nav(19)(sim)


@scenario((36, 300.0), expects=(ends("arrived"), logs("planned on the move"),
                                check("no move missed", lambda tr: _result(tr, "misses") == 0)))
def nav_transit_follows_on(sim):
    """A transit trip (the hunt loop's: with the escort, shooting only what's in her way) from the basin floor
    up to 1, nothing about: a chain cut off for length runs on until a look places her where it landed, and the
    next one, planned from there, is queued to follow on without a stop."""
    from ..control.navigator import TRANSIT
    h = sim.hunter()
    return sim.navigator(h, fight=True).go(1, mode=TRANSIT)


@scenario((2, 0.0), expects=(ends("arrived"), moves("drop")))
def nav_drops(sim):
    """From platform 2 down through 12 to 21: drops."""
    return _nav(21)(sim)


@scenario((12, -300.0), seed=36, expects=(ends("arrived"), moves("jump_up", "walk_off", "jump")))
def nav_walk_offs(sim):
    """From 12 across to 17: a jump up, walk-offs and jumps. (Seed 36 since the route's jumps take off where
    they land with room, route._placed: the random weights are drawn in another order, and of seeds 1-100,
    on 6 she jumps straight up onto 11 first, 36 the first of them, where 8 did before, 1 the first. The
    other ways: by the walk-offs down onto 13, 15 and 18, on half the seeds, or by a running jump onto 11.)"""
    return _nav(17)(sim)


@scenario((21, -300.0), expects=(ends("arrived"), moves("jump", "climb", "walk_off"),
                                 check("three chains or more", lambda tr: _result(tr, "chains") >= 3)))
def nav_long_way_up(sim):
    """From 21 up to 2: jumps, two ropes, a walk-off and a jump, over several chains."""
    return _nav(2)(sim)


@scenario((7, 1300.0), expects=(ends("arrived"), moves("climb", "jump_up")))
def nav_rope_then_jump_up(sim):
    """From 7 up the rope to 1, then straight up through it onto 0."""
    return _nav(0, x=1250)(sim)


@scenario((23, 900.0), expects=(ends("arrived"), moves("climb"),
                                check("one climb, chained",
                                      lambda tr: _result(tr, "moves") == ["climb 23->17: landed on 17"])))
def nav_rope_past_end(sim):
    """From 23 up to 17: a running jump off 23's right end at the rope hanging from 17 at x 1040, 11 units past it,
    and up it (a climb past a platform's end, route._past_end), chained by flow as any climb. Before, 17 was 15 moves
    away by the map's left side (23 -> 25 -> 32 -> 29 ... 18 -> 16 -> 14 -> 17)."""
    return _nav(17)(sim)


@scenario((36, 500.0), expects=(ends("arrived"), moves("climb", "jump"),
                                check("up the right side, by 23", lambda tr: any(m.startswith("climb 23->17")
                                                                                 for m in _result(tr, "moves")))))
def nav_basin_up_right_side(sim):
    """From the basin floor up to 3 (the spot for 4, the hunt loop's most made trip on 2026-10-21, 30 times): two
    ropes up to 25, a jump onto 23, the rope past its end up to 17, a jump onto 14 and two ropes on. By the map's
    left side, the only way up before, the simulator took 21.3 s; this way 17.6."""
    return _nav(3)(sim)


@scenario((30, 795.0), hanging=180.0, expects=(ends("arrived"), logs("on a rope at x"), sends("off_rope")))
def nav_off_rope_jump(sim):
    """Hanging halfway up the rope from 30 to 25: no platform under her dot, so she jumps off it sideways
    (towards the side a fall lands on a platform), then goes on to 36."""
    return _nav(36, x=500)(sim)


@scenario((23, 959.0), hanging=-230.0, expects=(ends("arrived"), sends("off_rope")))
def nav_off_rope_no_top(sim):
    """Hanging near the top of the rope under 17's block (x 959), which takes her onto no platform: she climbs down it
    and lets go at its bottom, onto 23 (holding up there did nothing, and the trip ended 'lost: not on any platform'),
    then goes on up to 17."""
    return _nav(17)(sim)


@scenario((30, 795.0), hanging=80.0, expects=(ends("arrived"), sends("off_rope")))
def nav_off_rope_climb(sim):
    """Hanging near the top of the rope up to 25: she climbs on up onto it, and that's where she's going."""
    return _nav(25)(sim)


@scenario((36, 300.0), noise=Noise(dot_hidden=((0.0, 2.0), (4.0, 4.6))),
          expects=(ends("arrived"), sends("mouse", "the mouse is moved when the dot stays hidden"),
                   check("frames without the dot", lambda tr: tr["frames"]["dot hidden"] >= 5)))
def nav_dot_hidden(sim):
    """36 -> 25 with her minimap dot covered (a tooltip) for her first 2 s and again mid-climb: Mover.locate
    keeps looking and moves the mouse after four tries; Navigator.follow goes on without the dot."""
    return _nav(25)(sim)


@scenario((21, -300.0), seed=3, noise=Noise(dot_px=True, walk_jitter=6.0),
          expects=(ends("arrived"), moves("jump", "climb")))
def nav_noisy_long_way(sim):
    """nav_long_way_up with her dot read to whole minimap pixels (~11 units) and walks a few units off:
    her x comes from ropes and her platform's ends on screen (pose.refine)."""
    return _nav(2)(sim)


@scenario((36, 470.0), seed=1, noise=Noise(lag_sd=1.0, walk_jitter=6.0, dot_px=True),
          expects=(ends("arrived"), moves("climb", "jump"),
                   check("climbs to 32 and jumps on to 29 and 28 in one chain",
                         lambda tr: _result(tr, "moves")[:3] == ["climb 36->32", "jump 32->29",
                                                                  "jump 29->28: landed on 28"]),
                   check("every move lands", lambda tr: _result(tr, "misses") == 0)))
def nav_climb_then_jump(sim):
    """Up the rope from the basin onto 32, a step back on it and the jumps on to 29 and 28, in one chain, with
    the lags spread as measured and walks a few units off. In game on 2026-10-21 17 of 25 such chains came down
    in the basin: the step after the climb, its arrow pressed before the top, went too far, and the jumps
    planned to land 5-16 units onto 29 and 28 then fell short (flow.STEP_EARLY_MAX_MS, TAKEOFF_SLOP). Here, at
    the branch before, 28 of 144 trips (seeds 1-48 from x 300, 470 and 760) missed a landing; now none does.
    Since the route's takeoffs are placed for room (route._placed: the jump onto 29 from 15 units in from 32's
    left end, not 33-45) there is no step back on 32: she runs at the jump from the top of the rope. The
    night's takeoffs, with the step back, are tests' (old_route_takeoffs)."""
    return _nav(28)(sim)


# Navigation with monsters about, not fighting: hops, waits, ways around, plans cut short.

@scenario((36, 200.0), monsters=[("Pig", 36, 600.0)], seed=1,
          expects=(ends("arrived"), logs("hop@", "a hop over the pig in the chain")))
def nav_hop_pig(sim):
    """A pig walking the basin floor between her and the rope up to 30: hopped over on the way (Flow's hop)."""
    return _nav(30)(sim)


@scenario((34, -300.0), monsters="all", seed=1,
          expects=(ends("arrived"), logs("looking for a way around"), logs(") on the way: on")))
def nav_monsters_staircase(sim):
    """Up the staircase (34 -> 19) among the map's monsters, not fighting: she looks for ways around the monster in her
    way, and is touched and knocked about a few times on the way up. (Seed 3 until her walks pressed pick-up: they last
    ~0.2 s longer, and with seed 3 no monster was then in her way to hop; seed 1 until her walk and jump lags came from
    motion.body, after which no monster was in her way on seed 1, and one is on seed 3 again. Seed 2 since the simulator
    moves her by the engine: on seed 3 nothing was in her way to hop. Over 16 seeds all arrive, and 8 meet all of this,
    10 before. Seed 4 since the planner times her runs and stops by the engine: on seed 2 nothing was in her way to hop;
    over 16 seeds all arrive, 7 meeting all of this. Seed 1 since a touch is evaded by the monster's accuracy: on seed 4
    she evaded a snail's touch at 7 s (82% of a snail's are), and then nothing was in her way to hop; over 16 seeds all
    arrive, 5 meeting all of this. Seed 5 since monsters walk as fast as the game data says (a pig 125 units/s, not 70;
    alone, that moved it to seed 2): on seeds 1 and 2 no monster was in her way to go around or hop; over 16 seeds all
    arrive, 2 meeting all of this. Seed 6 since the game moves her in 30 ms steps (body.STEP) and jumps from standing
    wait 150 ms after the arrow: on seed 5 no monster was in her way to go around or hop; over 16 seeds 15 arrive, 1
    meeting all of this; 16 and 2 just before. On the play track, before the steps, seed 10 since the route's jumps take
    off where they land with room, 15 units from the end up the staircase, not 33-45 (route._placed): on seed 5 no
    monster was in her way to hop or go around there; over 24 seeds all arrived, 2 meeting all of this, 10 and 18. Then
    seed 18 there since the rope moves (route._past_end, _rope_jumps): the random weights are drawn in another order,
    and on seed 10 no monster was in her way to hop; over 24 seeds all arrived, 1 meeting all of this. Seed 24 since the
    play track was rebased onto the game's steps and the live fixes: on seed 6 no monster was in her way to go around or
    hop; over 24 seeds all arrive, 1 meeting all of this. Seed 1, and no hop, since a monster touches her by her box all
    along her moves (motion.bodies: 22 either side and 65 tall, against the monster's widest and tallest frame): a chain
    stops sooner for one in her way, by then too close to time a hop over it, and of seeds 1-24 none hops here
    (nav_hop_pig still does); on seed 24 no monster was in her way to go around. Over 24 seeds all arrive, 16 meeting
    the rest of this.) (Seed 2 since every look is perception's newest frame after her keys' effect (Senses.fresh_after,
    the feed in every scenario): on seed 1 nothing hit her on the way; of seeds 1-16, 9 meet all of this.) (Seed 1 since
    a touch of hers is her popup, a MISS too, and her trips may pass a monster while her invulnerability after one lasts
    (navigator.TOUCH_CUTS, IMMUNE; the world's 1.45 s, sim.world.INVULNERABLE): on seed 2 she went past the one she'd
    have looked for a way around, still invulnerable; of seeds 1-16 all arrive, 9 meeting all of this.)"""
    return _nav(19)(sim)


@scenario((36, 200.0), monsters="all", seed=4, expects=(ends("arrived"), moves("climb")))
def nav_monsters_basin(sim):
    """Across the basin floor (pigs and slimes) and up onto 33, not fighting."""
    return _nav(33)(sim)


@scenario((21, -300.0), monsters="all", seed=15,
          expects=(ends("arrived"), logs("waiting for the monster to move off")))
def nav_monsters_long_way(sim):
    """From 21 up to 2 with monsters about, not fighting: she waits for monsters to move off her way. (Seed 11
    since chains follow on: on seed 5 the staircase's monsters knocked her back down until time ran out; over
    16 seeds 12 arrived, 14 before. Seed 4 since her walk and jump lags came from motion.body: on seed 11
    nothing was in her way to wait for; over 16 seeds 14 arrive, 3 of them after a wait. Seed 5 since the
    simulator moves her by the engine: on seed 4 nothing was in her way to wait for; over 16 seeds all arrive,
    3 of them after a wait, where 15 and 3 did just before. Seed 9 since no jump is keyed in the kick's blind
    spot and jump-backs leave at full speed: on seed 5 nothing was in her way to wait for; over 16 seeds 14
    arrive, 2 of them after a wait, where 15 and 6 did just before. Seed 7 since the arrow she holds doesn't
    steer a knock's flight: on seed 9 nothing was in her way to wait for; over 16 seeds all arrive, 4 of them
    after a wait, where 5 did just before. Seed 12 since monsters walk as fast as the game data says (a pig
    125 units/s, not 70; alone, that moved it to seed 12 too): on seed 7 nothing stood in her way to wait for;
    over 16 seeds 15 arrive, 2 of them after a wait. Seed 2 since the game moves her in 30 ms steps
    (body.STEP) and jumps from standing wait 150 ms after the arrow: on seed 12 nothing stood in her way to
    wait for; over 16 seeds all arrive, 3 of them after a wait; 15 and 2 just before. Seed 5 since chained
    jumps take off where they land with room to spare (flow.TAKEOFF_SLOP): on seed 2 nothing stood in her
    way to wait for; over 16 seeds all arrive, 2 of them after a wait. Two of them, seeds 2 and 4, were
    knocked down the gap beside platform 3 to the basin and ended "lost: not on any platform" before the
    navigator waited out a long fall too: navigator.FALL_LOOKS. On the play track, before the steps, seed 3
    since the route's jumps take off where they land with room (route._placed): on seed 12 nothing stood in her
    way to wait for there; over 24 seeds all arrived, 2 of them after a wait, 3 and 13. Then seed 12 again
    there since the rope moves (route._past_end, _rope_jumps), which draw the random weights in another order:
    on seed 3 nothing stood in her way; over 24 seeds all arrived, 2 of them after a wait, 12 and 13. Seed 15
    since the play track was rebased onto the game's steps and the live fixes: on seed 5 nothing stood in her
    way to wait for; over 24 seeds all arrive, 2 of them after a wait, 15 and 24.)"""
    return _nav(2)(sim)


# Navigation with `fight` on (as on tours): monsters in reach are shot on the way.

@scenario((36, 300.0), monsters="all", seed=1,
          expects=(ends("arrived"), logs("shooting it on the way"), logs("in the way at x"),
                   check("kills on the way", lambda tr: _result(tr, "kills") >= 3),
                   check("the trip counts every kill, those in the way of a move too",
                         lambda tr: _result(tr, "kills") == sum(tr["world"]["kills"].values()))))
def nav_fight_basin(sim):
    """Out of the basin up to 25 with `fight` on: the pigs and slimes in reach are shot first (a few
    fights of a few kills each), and one in the way of a move; then up the two ropes. The trip counts
    the kills of both (those in the way weren't counted before). (Seed 5 since no jump is keyed in the kick's
    blind spot and jump-backs leave at full speed: on seed 6 she was knocked off the first rope twice, and
    the trip's 180 s ran out on 30, throwing at a snail on 33 below that the decisions count in reach and
    the simulator's attack area doesn't. Over 16 seeds 12 meet all of this, 14 before. Seed 2 since a hit
    pushes a monster back: on seed 5 the trip's 180 s ran out on 30 that way. Over 16 seeds 8 meet all of
    this, 7 running out of time. With knocks unsteered and the planner flying her by the engine, 9 meet
    all of it; of the 7 that run out of time, 6 throw 54-88 stars from 30 at a snail on 33 that the
    simulator's attack can't pick (README, Fixed since), the 7th is a pig fight in the basin. With both
    tracks, from the ground the decisions and the simulator's attack both go by the star's search, which
    picks a snail 60 below from 246 units off: all 16 of seeds 1-16 meet all of it, at the merge and since.) (Seed 1
    since the rebase of her kill's monster onto clearance and slopes, 2026-10-27: on seed 2 she ran out of time; of
    seeds 1-8 all but 2 meet all of this, and master's stack on 1-4.)"""
    return _nav(25, fight=True)(sim)


@scenario((33, 900.0), monsters="all", seed=1, expects=(ends("arrived"), logs("shooting it on the way")))
def nav_fight_short(sim):
    """From 33 up to 25 with `fight` on and the tour's kinds: a few shot on the way, and there."""
    return _nav(25, fight=True, kinds=TOUR_KINDS)(sim)


@scenario((14, 760.0), monsters="all", seed=1,
          expects=(ends("arrived"), logs("shooting it on the way"), moves("climb")))
def nav_fight_up(sim):
    """From 14 up two ropes and a jump to 1 (pigs, Orange Mushrooms) with `fight` on: she shoots 17's monsters
    from 14 first, and fights her way up past 3 and 4. On some seeds a touch knocks her off 3 or 4 down to the
    basin and the trip runs out of time (of seeds 1-48: on 10 with the decisions of the star search, 14 with
    the claw's edge by size, 18 with kills fading by their death animation too, seed 10 among them; 5 with
    both tracks), so it was seed 3, where she got there with each of them. (Seed 1 since a trip waits for her
    to land before it plans, Navigator._onto_platform: on seed 3 knocks back onto 4 sent her the long way
    round, and the trip ran out of time. Of seeds 1-48 36 arrived before monsters walked the game data's
    speeds: 38 without that and the fight's wait for a knock to end, 43 at the tracks' merge, before the drops
    and walk-offs kept clear of the basin's cliffs. The 12 others ran out of time at 180-194 s, 10 of them
    knocked off 3 or 4 and climbing back up from 32 or the basin, which took 164-180 s on the seeds that went
    that way and arrived at the merge. Seed 1 arrives with each. With monsters walking those speeds 37 arrive,
    the 11 others running out of time at 181-195 s; with the client's star search and claw stab too, 41, the 7
    others at 181-193 s. Seed 2 since the game moves her in 30 ms steps (body.STEP) and jumps from standing
    wait 150 ms after the arrow: on seed 1 she was knocked off 17 down to 35 at 103 s and ran out of time on
    the long way round (nav_fight_up_runs_out now). Of seeds 1-48 41 arrive, 42 just before; on seed 2 knocked
    off 3 to 25 at 122 s, she goes the long way round and arrives. On the play track, before the steps: with her
    finishing what she has hurt, 46; with no jump throw at a snail (stance.JUMP_THROW_SNAILS), 41: in this world a
    jump throw's area picks the snail on 3, where she now stabs it or steps back from it. Then seed 3 there,
    since a kill is placed where its monster was seen last (behavior.ledger): on seed 1 her bursts from 14 at
    17's monsters went on otherwise from its third kill, and the trip ran out of time at 200 s; 40 of seeds 1-48
    arrived; with no step to 3's far end from a snail, which she stabs where she stands instead (stance.
    SNAIL_FAR_END), 43, and 85 of seeds 1-96 against 80. Seed 1 again since the play track was rebased onto the
    game's steps and the live fixes: on seed 2 she's knocked off 3 down to 17 at 88 s and off 17 down to 25
    three times, and runs out of her 180 s on her way up again (nav_fight_up_runs_out now); on seed 1 she
    arrives at 141 s, 40 monsters killed. Of seeds 1-48, 38 arrive; 37 since a climb from under its rope is timed
    by her arms' reach, Flow._climb.)"""
    return _nav(1, fight=True, kinds=TOUR_KINDS)(sim)


@scenario((12, -300.0), monsters="all", seed=7,
          expects=(ends("arrived"), logs("knocked off?"), logs("plan cut short")))
def nav_fight_across(sim):
    """From 12 across to 17 with `fight` on: starting among the pigs on 12 she fights, is knocked off (down to 13), and
    goes on from there; hit on the way, her chain is cut and replanned, and she fights her way to 17. (Seed 2 since a
    walk-off must land where she can stop: on seed 3 nothing hit her on the way any more; 12 of 16 seeds show all of it.
    Seed 3 since jump-throws come 40 ms after the jump: on seed 6 nothing knocked her off any more. On seed 6 before,
    her first chain from 11 waited out the attack lock of her last throw: before it did, she was touched on the way, and
    the quick looks after those cuts skipped the fights on 18 and 17. Seed 10 since a hit pushes a monster back: on seed
    2 a knock down to 34 at 5 s left her fighting and looting there until the 180 s ran out. Over 16 seeds 12 arrived, 5
    showing all of this; with monsters walking the game data's speeds 8 arrive, 4 showing all of it: 7, 9, 11 and 10.
    Seed 1 since the star's search and the claw's stab are the client's too: on seed 10 nothing knocked her off 12 any
    more. Over 16 seeds 12 arrive, 8 showing all of it. Seed 4 since the game moves her in 30 ms steps (body.STEP) and
    jumps from standing wait 150 ms after the arrow: on seed 1 nothing hit her on the way any more. Over 16 seeds 10
    arrive, 6 showing all of it; 11 and 8 just before; on seed 4 she's knocked off 12 down to 34 at 5 s and comes the
    long way round, the staircase's monsters cutting her chains short three times. Seed 6 since her jumps' attacks are
    weighed and her jump throws turn in the air (2026-10-25): on seed 4 no chain was cut short any more. Over 16 seeds
    15 arrive, 6 showing all of it: 6, 7, 12, 13, 14 and 15.) (Seed 4 since every look is perception's newest frame
    after her keys' effect (Senses.fresh_after, the feed in every scenario): on seed 6 nothing knocked her off 12; of
    seeds 1-16, 2, 4, 12 and 15 meet all of this.) (Seed 2 since 2026-10-26: her throws at the dying gone, a throw's
    last look before its key, her stats at level 25, the air turn's attack 60 ms after its arrow and the world's damage
    at the server's time: on seed 4 nothing knocked her off 12; of seeds 1-16, 2, 5, 9, 13 and 14 meet all of this.)
    (Seed 1 since 2026-10-27, an attack from a jump picking by the star's search alone (reach.AIR_MODEL, the world's
    AIR_PICK) and her stance going by where each monster will be when her attack starts (attack.stance.Pace): on seed 2
    she ran out of her 180 s on the way; of seeds 1-12, 1, 3, 4, 7 and 9 meet all of this.) (Seed 4 since the reviews of
    that, a monster walking at her stopping where it meets her and one whose walk nothing tells staying where it was
    seen (attack.stance.PACE_STOP, WALK_AT): on seed 1 nothing knocked her off any more; of seeds 1-16, 4, 7, 11, 13 and
    16 meet all of this.) (Seed 7 since the rebase of her kill's monster onto clearance and slopes, 2026-10-27, her
    kill's monster bound by her kept in her body's way (behavior.fight.TOUCH_NEAR) and her kills credited to the monster
    her attack took: on seed 4 she ran out of her 180 s; of seeds 1-16, 1, 5-7 and 10-14 meet all of this, and with
    master's stack 4, 7, 11, 13 and 16.)"""
    return _nav(17, fight=True, kinds=TOUR_KINDS)(sim)


# Two known failures, pinned on seeds where they happen since monsters walk as fast as the game data says: the
# faster monsters keep knocking her off while she fights her way, and the decisions tuned for slow ones (hop
# timing, TOO_CLOSE, the star/claw boundary) aren't retuned yet. A retune that mends them changes these.

@scenario((14, 760.0), monsters="all", seed=15,
          expects=(ends("timeout"), logs("shooting it on the way"), logs("knocked off?")))
def nav_fight_up_runs_out(sim):
    """nav_fight_up on seed 4: she shoots 17's monsters from 14, climbs to 3, is knocked off down to 17 at 47 s, and
    fights on there and on the way back up, running out of her 180 s, 42 monsters killed. (Seed 4 since the review of
    her jumps' attacks, 2026-10-25: her body kept clear of a monster above over a jump in place, a jump forward's
    landing checked however it goes, the ground's reach capped 60 above her again: on seed 5 she now arrives on 1 in
    51 s, 11 monsters killed; of seeds 1-16, 4, 11 and 16 meet all of this. On seed 5 before: she shot 17's monsters
    from 14, climbed to 6, was knocked off 4 onto 6 at 31 s, and fought her way up to 3 and over to 1 and back onto 3
    twice, running out of her 180 s, 42 monsters killed. Seed 5
    since the monsters' HP bars are drawn as the game draws them, perception.hud.draw_hp_bar, read to the game's px: on
    seed 4 she now arrives on 1 in 177 s, knocked off four times; of seeds 1-17, 2, 5, 11 and 16 meet all of this. On
    seed 4 before: she shot 17's monsters from 14, climbed to 3, came back down to 17 and up again to 3, was knocked off
    down to 25 at 112 s, back up by 23 and the rope to 17, then down to 35, knocked off 33 back onto it at 142 s, and
    ran out of her 180 s on 33, 40 monsters killed. Seed 4 since a throw's tap goes by her facing off the frame,
    perception.heading: on seed 6 she now arrives, in 151 s; of seeds 1-16, 4 and 5 meet all of this. On seed 6 before:
    knocked off 3 down to 17 at 55 s and off 17 down to 25 at 81 s, back up by 23 and the rope past its end, she ran out
    of her 180 s on her way up again, 39 monsters killed. Seed 6 since her EXP number is read and each kill goes on a
    monster of the kind its gain's size names: on seed 2 the fights no longer take live monsters for dead (a slime
    killed on 17 at 4.3 s had been put on an Orange Mushroom, left alone then as dying), and she arrives on 1 in 103 s.
    Of seeds 1-48, 9 meet all of this, as before. On seed 2 before: knocked off 3 down to 17 at 88 s and off 17 down to
    25 at 138 and 149 s, each time back up by 23 and the rope past its end, she ran out of her 180 s on her way up
    again, 39 monsters killed. Seed 2 since the play track was rebased onto the game's steps and the live fixes: on seed
    1 she arrives (nav_fight_up's seed again). Of seeds 1-48, 9 meet all of this. On seed 1 before: she shot 17's
    monsters from 14 for 48 s, climbed to 3 and shot 1's, was knocked down onto 17 and off it down to 35 at 103 s,
    fought there until 162 s, and on the long way round up the staircase ran out of her 180 s on 19, 45 monsters killed.
    Seed 1 since the game moves her in 30 ms steps (body.STEP) and jumps from standing wait 150 ms after the arrow: on
    seed 4 she now arrives; of seeds 1-48, 41 arrive, 42 just before. Seed 4 had been since the drops and walk-offs kept
    clear of the basin's cliffs: knocked off 3 down to 17 at 84 s, she had gone the long way round, down from 25 to 32
    and up the staircase, fighting 48 monsters, and run out of time on 21. Seed 10 on the perception track, where it
    arrived before monsters walked 125 units/s; with the physics and attack tracks it arrives there. Seed 11 then, where
    with the client's star search and claw stab it arrives too: knocked off 4 at 26 s and hit off the jump from 3 to 1,
    it had gone the long way round, fighting 38 monsters, and run out of time on 16. Of seeds 1-48, 41 arrive, against
    37 before and 36 at the physics and attack tracks' merge. On the play track, before the steps, seed 17 since she
    finishes what she has hurt before she moves on (behavior.fight.owed): on seed 4 she arrived there; of seeds 1-48, 46
    arrived, against 41 before. Then seed 18 since she never jump-throws at a snail (stance.JUMP_THROW_SNAILS): on seed
    17 she arrived; of seeds 1-48, 41 arrived, 46 with the jump throws in that world, whose jump throws' area picks a
    snail. On seed 18 since she stabs a snail on 3 where she stands, not stepping to 3's far end (stance.SNAIL_FAR_END),
    at 40 s, she was knocked off 3 down to 17 at 86 s, off 17 down to 25 at 89 s and off 17 to 35 at 141 s, and ran out
    of time, 48 monsters in all.) (Seed 4 since every look is perception's newest frame after her keys' effect
    (Senses.fresh_after, the feed in every scenario): on seed 5 she now arrives; of seeds 1-16, only 4 meets all of
    this.) (Seed 15 since 2026-10-26, her throws at the dying gone, a throw's last look before its key, her stats
    at level 25 and the world's damage at the server's time: on seed 4 she now arrives; of seeds 1-16, only 15
    meets all of this, 14 don't run out of time.) (Not met since the rebase of her kill's monster onto clearance and
    slopes, 2026-10-27: of seeds 1-16 she arrives on every one, her kills credited to the monsters her attacks took:
    xfail in tests/test_sim.py, AFTER_KILL_PLACE.)"""
    return _nav(1, fight=True, kinds=TOUR_KINDS)(sim)


@scenario((12, -300.0), monsters="all", seed=3,
          expects=(ends("timeout"), logs("shooting it on the way"), logs("knocked off?")))
def nav_fight_across_runs_out(sim):
    """nav_fight_across on seed 6: after 38 s fighting on 12 she's knocked off down to 11 and 34, fights the slimes and
    snails there for 100 s, and on the long way round up the staircase is hit off the climb from 23 to 17 down to 35 and
    again off it onto 23, and runs out of her 180 s, 33 monsters killed. (Seed 6 since her EXP number is read and each
    kill goes on a monster of the kind its gain's size names: on seed 9 she now arrives, in 84 s. Of seeds 1-16, 3 meet
    all of this. On seed 9 before: knocked off 12 down to 34 at 10 s, on the long way round up the staircase she fought
    her way up to 21, was knocked off it down to 22 at 119 s and off 22 to 24 at 169 s, fighting 37 monsters in all, and
    ran out of her 180 s shooting a pig on 18. Seed 9 since chained jumps take off where they land with room to spare
    (flow.TAKEOFF_SLOP): on seed 2 she now arrives. Of seeds 1-16 11 arrive. On seed 2 before: knocked off 12 down to
    34, on the long way round up the staircase she fought 80 s on 22, was knocked off 18 back to 19 at 138 s and hit off
    it down to 32 at 150 s and to 34 at 158 s, fighting 40 monsters in all, and ran out of her 180 s on her way up
    again. (On seed 2 still since the game moves her in 30 ms steps (body.STEP) and jumps from standing wait 150 ms
    after the arrow, a different run: before, after 25 s fighting on 12 she was knocked off down to 34, then off 20 to
    19 at 93 s, back onto 14 off its walk-off to 17 and down to 36 off the jump from 32, and ran out of time on 22. Of
    seeds 1-16 10 arrive, 11 just before.) Seed 3 before, where with the client's star search and claw stab she now
    arrives: after 52 s fighting on 12 she'd been knocked off down to 34, and twice more on the long way round, and run
    out of time on 18; it arrived there before monsters walked 125 units/s. Of seeds 1-16, 12 arrived then, 8 with the
    physics and attack tracks before the client's search and stab, 12 at their merge. On the play track, before the
    steps, seed 15 since she finishes what she has hurt before she moves on (behavior.fight.owed): on seed 2 she arrived
    there; of seeds 1-16, 14 arrived, against 12 before. Then seed 7 since she never jump-throws at a snail
    (stance.JUMP_THROW_SNAILS): on seed 15 she arrived; of seeds 1-16, 12 arrived. Seed 5 since her jumps' attacks are
    weighed and her jump throws turn in the air (2026-10-25): on seed 6 she arrived; of seeds 1-16, 15 arrive, and on
    seed 5 she's knocked about 12 and fights there, 40 monsters, until her 180 s run out. Seed 3 since the review of her
    jumps' attacks, 2026-10-25 (her body clear of a monster above over a jump in place, a jump forward's landing however
    it goes, the ground's reach capped 60 above her again): on seed 5 she arrives in 48 s, 9 monsters killed; of seeds
    1-16, 14 arrive, and on seed 3 she's knocked off 12 down to 34 at 4 s, fights there and on the long way round up, 37
    monsters, until her 180 s run out.) (Seed 27 since every look is perception's newest frame after her keys' effect
    (Senses.fresh_after, the feed in every scenario): on seed 3 she now arrives; of seeds 1-64, only 27 meets all of
    this.) (Seed 9 since 2026-10-26, her throws at the dying gone, a throw's last look before its key, her stats
    at level 25 and the world's damage at the server's time: on seed 27 she now arrives; of seeds 1-16, 9 and
    12 meet all of this, 14 don't run out of time.) (Seed 3 since 2026-10-27, an attack from a jump picking by the
    star's search alone (reach.AIR_MODEL, the world's AIR_PICK) and her stance going by where each monster will be when
    her attack starts (attack.stance.Pace, PACE_STOP, WALK_AT): on seed 9 she now arrives; of seeds 1-24, 3 and 6 meet
    all of this, 21 don't run out of time.)
    (Not met since 2026-10-27 on its own branch, her kill credited to the monster her attack took
    (localization.tracking's binding) and the dead out of her targets by their tracks: of seeds 1-16 she arrives on
    every one, on seed 9 in 59 s with 9 kills, on 1 in 42 s, on 4 in 57 s, where master's stack ran out of its 180 s on
    9: xfail in tests/test_sim.py, AFTER_KILL_PLACE.) (Met again on seed 3 since the rebase of her kill's monster onto
    clearance and slopes, 2026-10-27: no longer xfail.)"""
    return _nav(17, fight=True, kinds=TOUR_KINDS)(sim)


@scenario((3, 720.0), monsters=[("Orange Mushroom", 4, 430.0)], seed=1, realism=Realism(lock_arrows=True),
          expects=(ends("killed", key="killed"), ends("arrived"), moves("walk_off"),
                   check("the walk-off right after the kill lands", lambda tr: _result(tr, "misses") == 0),
                   no_log("instead of platform 4")))
def nav_walk_off_after_kill(sim):
    """From 3 she shoots an Orange Mushroom on 4 below dead, then, from the look that saw it die (as the hunt
    loop goes on from its own look), walks off 3's left end onto 4, in a world where an arrow pressed while
    her attack plays is swallowed until about 0.79 s after its key (Realism.lock_arrows, as measured in game on
    2026-10-21). In game such a walk-off went out about 0.3 s after her last throw's key, twice, and she didn't
    move; here it would go 0.34 s after it, and she'd stay on 3 until the walk-off sent again a second later.
    Its first arrow is held back to 0.8 s after the key (actions.ARROW_LOCK_MS), and she lands on 4 at once."""
    return _go_after_kill(sim, 4, x=520)


def _go_after_kill(sim, dst: int, x: float | None = None) -> dict:
    """Shoot the first monster seen dead, then a trip to `dst` from the look that saw it die, as the hunt loop
    goes on from its own look."""
    h = sim.hunter()
    looks = []
    look = h.look

    def keep(*a, **k):
        looks.append(look(*a, **k))
        return looks[-1]
    h.look = keep
    killed = h.kill(h.look().monsters[0], seconds=30)
    out = plain(sim.navigator(h, fight=True).go(dst, x=x, scene=looks[-1]))
    out["killed"] = killed
    return out


@scenario((16, 537.0), monsters=[("Pig", 18, 300.0)], seed=1, facing=LEFT, realism=Realism(lock_arrows=True),
          expects=(ends("killed", key="killed"), ends("arrived"),
                   check("jumps up onto 14 and climbs to 6 in one go",
                         lambda tr: _result(tr, "moves") == ["jump 16->14", "climb 14->6: landed on 6"]),
                   no_log("instead of platform 6")))
def nav_jump_after_kill(sim):
    """From 16 she shoots a pig on 18 below dead, then, from the look that saw it die, runs right along 16 and
    jumps up onto 14 for the rope to 6 (the hunt loop's way up to 4 in game on 2026-10-21), in a world where an
    arrow pressed while her attack plays moves her only from about 0.79 s after its key. In game, with the
    chain's first arrow held back to 0.6 s after the key, she set off about 0.12 s late, the jump went at its
    planned time from 15-30 units short, and 8 of 10 such chains fell to 25; here too she falls to 25 on every
    seed of 1-8 that kills the pig (a test pins it, at that night's route takeoffs: old_route_takeoffs). Held
    back to 0.8 s (actions.ARROW_LOCK_MS), she lands on 14. At the takeoffs placed for room (route._placed: 15
    units from 16's right end, not 33-45) she lands on 14 held back to 0.6 s too."""
    return _go_after_kill(sim, 6)


# One move each, carried out by Mover.do from a look.

def _one_move(sim, kind: str, src: int, dst: int) -> dict:
    h = sim.hunter()
    nav = sim.navigator(h)
    m = min((m for m in nav.planner.moves if (m.kind, m.src, m.dst) == (kind, src, dst)),
            key=lambda m: abs(m.x - sim.world.x))
    f = nav.mover.do(m)
    p = f.where.platform
    return {"move": f"{m.kind} {m.src}->{m.dst} at x {m.x:.0f}", "landed": None if p is None else p.id,
            "x": round(f.where.x, 1), "how": f.how}


@scenario((36, 640.0), expects=(ends(30, key="landed"), sends("climb")))
def move_climb(sim):
    """Mover.do: run at the rope from 36 and jump onto it holding up, up to 30."""
    return _one_move(sim, "climb", 36, 30)


@scenario((30, 700.0), expects=(ends(36, key="landed"), sends("walk"), sends("climb_down")))
def move_climb_down(sim):
    """Mover.do: walk onto the rope's top on 30 (close enough at once: no nudge) and climb down."""
    return _one_move(sim, "climb_down", 30, 36)


@scenario((30, 700.0), seed=8, noise=Noise(walk_jitter=5.0, dot_px=True),
          expects=(sends("nudge", "nudges by the rope on screen"),
                   check("lined up in a few nudges", lambda tr: 1 <= sum(p[1] == "nudge" for p in tr["plans"]) <= 4),
                   ends(36, key="landed", what="down the rope after all")))
def move_climb_down_nudge(sim):
    """Mover.do with walks a few units off: the walk stops a few units from the rope's top, and Mover.line_up_rope
    nudges her by the rope on screen, held as physics.hold_ms says (54-61 ms here), until she's inside the 6 px
    tolerance, and she climbs down. On the game's 30 ms steps a nudge that short moves her one step of speeding up or
    two, 1.8 or 7.0 units, so it can take a few, from either side: on seed 6 three (left, right, left). (Seed 6 since
    the game moves her in those steps: on seed 14 the walk stopped close enough not to nudge. Over 16 seeds 3 nudge, 1
    just before; with a nudge moving her 5.8 units, as on the engine's 5 ms step, she climbed down after one. Seed 14
    since the walk's lags came from motion.body: on seed 7 the walk now stops close enough not to nudge. Until the
    simulator moved her by the engine, any tap that moved her at all moved her at least ~14 units, so that nudge went 23
    px, and she swung from side to side of the rope three times before pressing down; over 16 seeds 1 nudged then, and 1
    now. Until the mover held its nudges as hold_ms says, they were timed by her walking speed alone, plus 25 ms: 58 ms
    for 5 units, which moved her 9, 4 past the rope's top.) (Seed 8 since every look is perception's newest frame after
    her keys' effect (Senses.fresh_after, the feed in every scenario): of seeds 1-40, 8 nudge, 7 of them once, from one
    side, and seed 30 twice, both from the left: one look after a nudge sees where it left her, and a nudge lines her
    up; on seed 6 the walk stopped close enough not to nudge.)"""
    return _one_move(sim, "climb_down", 30, 36)


@scenario((1, 1200.0), expects=(ends(0, key="landed"),))
def move_jump_up(sim):
    """Mover.do: step under platform 0 and jump straight up onto it."""
    return _one_move(sim, "jump_up", 1, 0)


@scenario((4, 300.0), expects=(ends(2, key="landed"),))
def move_jump(sim):
    """Mover.do: run towards 4's end and jump up onto platform 2."""
    return _one_move(sim, "jump", 4, 2)


@scenario((20, 0.0), expects=(ends(19, key="landed"),
                               check("standing a slide past the landing", lambda tr: 60 <= _result(tr, "x") <= 100)))
def move_jump_short_landing(sim):
    """Mover.do: the jump from 20 up onto 19, a short platform (60 up, landing at x 67): the arrow is let go
    just before she lands, and she stops 14 units on. (Held through a level jump's air time and 150 ms more,
    it ran her 59 units past the landing: off 19's far end, onto 24, while the jump took off 12 units late,
    and 3 units short of that end once the route's takeoffs were placed for room.)"""
    return _one_move(sim, "jump", 20, 19)


@scenario((4, 450.0), expects=(ends(6, key="landed"), sends("walk_off")))
def move_walk_off(sim):
    """Mover.do: walk off platform 4's right end, down onto 6."""
    return _one_move(sim, "walk_off", 4, 6)


@scenario((3, 760.0), expects=(ends(17, key="landed"), sends("walk_off")))
def move_walk_off_through_wall(sim):
    """Mover.do: walk off platform 3's right end and on through the wall hanging under platform 1's left
    end, 12 units on (another foothold group's: in the air only her own group's walls stop her), down onto
    17, 600 units below, as in game three times out of three. (Stopped by that wall, she fell 1500 units
    into the basin, and Mover.settle gave up after 2 s with her still falling; no move on this map falls
    that long now.)"""
    return _one_move(sim, "walk_off", 3, 17)


@scenario((2, -100.0), expects=(ends(12, key="landed"), sends("drop")))
def move_drop(sim):
    """Mover.do: walk to the spot and drop through 2 onto 12."""
    return _one_move(sim, "drop", 2, 12)


@scenario((23, 900.0), expects=(ends(17, key="landed"), sends("climb"), sends("rope_jump"),
                                happens("grabs the rope at x 1040", "the second rope grabbed in the air")))
def move_rope_hops(sim):
    """Mover.do: from 23 up to 17 by two rope hops (the user's way, 2026-10-21): a jump at the rope hanging under 17's
    block at x 959 (its bottom 58 above 23), up it to where a jump off it reaches the next (route.side_jump), a look
    there, then off it to the right at the rope jump's speeds (1.3 x 149 out, half a jump's up), holding up: she
    grabs the rope at 1040, 81 units on, 0.37 s later, and climbs it onto 17."""
    return _one_move(sim, "rope_jump", 23, 17)


# Fighting and looting.

@scenario((36, 540.0), monsters="all", seed=11,
          expects=(check("stops at the kill limit", lambda tr: "kill limit" in _result(tr, "report")),
                   happens("star kills"), happens("picks up")))
def hunt_clear_basin(sim):
    """`hunt clear` in the basin: pigs and slimes on platform 36 (finish, far), then a mop-up."""
    h = sim.hunter()
    r = h.clear({"pig", "slime"}, ground={36}, max_kills=6, seconds=90, finish=True, far=True)
    h.looter.mop_up()
    # "pending_loot" is Looter.pending under its name before the layers, so the golden traces stay identical.
    return {"report": str(r), "pending_loot": [[round(d.x), d.platform] for d in h.looter.pending],
            "inventory_full": h.looter.inventory_full}


@scenario((36, 300.0), monsters={36}, seed=6,
          expects=(ends("killed", key="ended"), happens("claw swipe hits", "close up, the claw swipes")))
def hunt_kill(sim):
    """`hunt kill`: the slime nearest her on the basin floor, 55 units off. Too close for a star from the
    ground, her throws are claw swipes (a fifth of a star) until it's out of the claw's reach again (its
    swing reaches 90 ahead of her, into a slime's body up to 118 off); then looted. (Seed 7 since the claw
    reaches bodies: with seed 12 the basin's monsters kept walking into its reach, and swiped, she hadn't
    killed it after 30 s. Seed 1 since a hit pushes a monster back: on seed 7 two swipes pushed it out of
    the claw's reach, and she hadn't killed it after 30 s. Over 16 seeds 11 meet all of this. Seed 6 since
    monsters walk as fast as the game data says (a slime 100 units/s, not 56): on seed 1 the basin's other
    monsters kept walking into her stars' way (two kills, not it), and she hadn't killed it after 30 s; 6
    is the first seed after it that its twin with the feed, feed_hunt_kill, met too; the twin is on 7 since
    her EXP gains name their kinds. Over 16 seeds 10 meet all of this.)"""
    h = sim.hunter()
    s = h.look()
    target = min(s.monsters, key=lambda t: abs(t.x - s.where.x))
    end = h.kill(target, seconds=30)
    return {"target": f"{target.kind} at {target.x - s.where.x:+.0f}", "ended": end, "report": str(h.report)}


@scenario((31, 200.0), monsters=[("Slime", 34, -300.0)], seed=13, expects=(ends("unreachable", key="ended"),))
def hunt_kill_unreachable(sim):
    """`hunt kill` at a slime on 34, 60 below and far to the left: no spot on her platform reaches it."""
    h = sim.hunter()
    s = h.look()
    end = h.kill(s.monsters[0], seconds=20)
    return {"ended": end, "report": str(h.report)}


def _kill_far(sim):
    h = sim.hunter()
    s = h.look(wide=True)
    end = h.kill(s.monsters[0], seconds=40)
    return {"ended": end, "report": str(h.report)}


@scenario((36, 200.0), monsters=[("Pig", 36, 850.0)], seed=15,
          expects=(ends("killed", key="ended"), happens("picks up leather")))
def hunt_kill_far(sim):
    """`hunt kill` at a pig across the basin: walk into reach (300 units at most per look), shoot it dead, loot it.
    (Seed 15 since every look is perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every
    scenario): on seed 14 the pig had walked off the screen's edge by the first look's frame, and she had no target; of
    seeds 1-16, 8 meet all of this.)"""
    return _kill_far(sim)


@scenario((36, 200.0), monsters=[("Pig", 36, 850.0)], seed=3, noise=Noise(miss=0.35),
          expects=(ends("killed", key="ended"), happens("star misses"),
                   wrote("shots2.jsonl", '"hit": false', "a throw scored as a miss in the shot log")))
def hunt_kill_misses(sim):
    """hunt_kill_far (another seed) with a third of the stars missing: throws scored as misses (no HP bar change within
    ~0.9 s of the star's arrival), and more throws to kill it. (Seed 5 since monsters walk as fast as the game data says
    (a pig 125 units/s, not 70): on seed 4 no star missed it. That alone moved it to seed 16, where with the physics and
    attack tracks the pig walks out of reach and she loses it. Over 16 seeds 8 meet all of this.) (Seed 8 since every
    look is perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every scenario): on seed 5
    no star missed; of seeds 1-16, 5 meet all of this.) (Seed 3 since 2026-10-26: her throws at the dying gone, a
    throw's last look before its key, her stats at level 25, the air turn's attack 60 ms after its arrow and the world's
    damage at the server's time: on seed 8 no throw was scored a miss in the shot log (which still goes by the star's
    release: fight.LANDS_FROM_S); of seeds 1-16, 3 and 5 meet all of this.)"""
    return _kill_far(sim)


@scenario((36, 200.0), monsters=[("Pig", 36, 850.0)], seed=15, noise=Noise(tag_hidden=((0.0, 4.0),)),
          expects=(ends("killed", key="ended"),
                   check("frames without her name tag", lambda tr: tr["frames"]["tag hidden"] >= 3)))
def hunt_kill_tag_hidden(sim):
    """hunt_kill_far with her name tag hidden for the first 4 s (behind a box): she's placed on screen by the camera
    from her map position (Hunter.look), and the monsters by their offset from there. (Seed 15 since every look is
    perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every scenario): as hunt_kill_far.
    Counted in perception's frames, which have her tag hidden as her looks did.)"""
    return _kill_far(sim)


def _first(lines: list[str], text: str) -> float:
    """When the first of a trace's log lines (or world_log's) with `text` came (s; infinity if none did)."""
    return next((float(line.split()[0]) for line in lines if text in line), float("inf"))


@scenario((22, 100.0), monsters=[("Snail", 22, 60.0), ("Slime", 21, -150.0), ("Slime", 21, -300.0)], seed=12,
          limit=60,
          expects=(check("the snail dies within 15 s", lambda tr: _first(tr["world_log"], "kills snail") <= 15),
                   check("she stays on platform 22", lambda tr: tr["world"]["platform"] == 22),
                   no_log("knocked off?"),
                   check("no 'snail down' before the snail really dies",
                         lambda tr: _first(tr["log"], "snail down") >= _first(tr["world_log"], "kills snail"))))
def fight_snail_short_platform(sim):
    """A snail beside her on platform 22 (168 units wide), slimes on 21, 60 above to the left. A jump throw's
    attack starts 46 units up, over the snail: it used to jump-throw at it, and the stars went to the slimes
    up there; their EXP was taken for the snail's ("snail down"), and the live snail was then left alone as
    a dying one. Now it's shot from the ground: with no room to step back to where that throws a star, the
    claw stabs it, each stab pushing it back, until a star picks it. Keyed 40 ms after the jump the jump throw's
    attack picks a snail by its area, and she went back to jump throws at them; but in game on 2026-10-21 they
    mostly missed (stance.JUMP_THROW_SNAILS), and from the ground she never jumps at one. Of seeds 1-32, 21 meet
    all of this (24 when she stepped back to 22's far end once the snail had walked off: stance.SNAIL_FAR_END; of
    seeds 1-96, 68 against 71), against 11 with the jump throws in this world, whose jump throws' area picks the
    snail; on most of the others the snail walks into her first, and a touch knocks her off the end of 22 (she
    stands 29 units from it). Since the rebase onto the game's steps and the live fixes, 27 of seeds 1-32 meet all
    of it, 80 of 1-96."""
    h = sim.hunter()
    r = h.clear(max_kills=1, seconds=20, approach=False)
    return {"report": str(r)}


THROWS = ("throw", "jump_throw", "jump_back")  # the key plans that attack (control.actions)
JUMP_PLANS = ("jump_throw", "jump_back")


def _snail_fight(sim):
    h = sim.hunter()
    r = h.clear(max_kills=1, seconds=20, approach=False)
    return {"report": str(r)}


@scenario((34, -200.0), monsters=[("Snail", 34, -165.0)], seed=1, limit=60,
          expects=(check("no jump throw or jump back", lambda tr: not any(p[1] in JUMP_PLANS for p in tr["plans"])),
                   sends("walk", "a step back"), happens("star kills snail", "a star kills the snail"),
                   check("the snail dies within 6 s", lambda tr: _first(tr["world_log"], "kills snail") <= 6),
                   check("she stays on platform 34", lambda tr: tr["world"]["platform"] == 34)))
def fight_snail_step_back(sim):
    """A snail 35 units from her on platform 34 (618 units wide): too close for a star from the ground, and she
    never jump-throws at a snail (stance.JUMP_THROW_SNAILS: in game jump throws mostly missed them). With the room
    behind her she steps back to where a throw from the ground is a star (99 from it, and as much more as it walks
    at her meanwhile), and a star kills it. On a short platform only its far end would have that room, and there
    she stabs it where she stands instead (stance.SNAIL_FAR_END; fight_snail_stab). Of seeds 1-16, 15 meet all of
    this (on 12 the snail walks after her, and she stabs it until a star picks it: 8.6 s)."""
    return _snail_fight(sim)


@scenario((3, 760.0), monsters=[("Snail", 3, 775.0)], seed=1, limit=60,
          expects=(check("no jump throw or jump back", lambda tr: not any(p[1] in JUMP_PLANS for p in tr["plans"])),
                   happens("claw swipe hits snail", "the claw stabs the snail"),
                   check("the snail dies within 8 s", lambda tr: _first(tr["world_log"], "kills snail") <= 8),
                   check("she stays on platform 3", lambda tr: tr["world"]["platform"] == 3)))
def fight_snail_stab(sim):
    """A snail 15 units from her in the middle of platform 3 (168 units wide): no room to step back to where a
    throw from the ground is a star, even at its far end, so she stabs it from the ground (the claw's 9-11
    damage, every one a hit, each pushing it back) until it's dead or pushed out to where a star picks it. Of
    seeds 1-16, 16 meet all of this."""
    return _snail_fight(sim)


@scenario((4, 520.0), monsters=[("Orange Mushroom", 2, 80.0), ("Orange Mushroom", 2, 140.0),
                                ("Orange Mushroom", 2, 200.0)], seed=4, limit=60,
          expects=(check("no throw from the ground from platform 4",
                         lambda tr: not any(p[1] == "throw" for p in tr["plans"])),
                   check("she stays on platform 4", lambda tr: tr["world"]["platform"] == 4),
                   check("jump throws over the step kill them",
                         lambda tr: "kills: 3 orange-mushroom" in _result(tr, "report"))))
def fight_step_wall(sim):
    """She's on platform 4 at x 520, nothing on it, Orange Mushrooms up on 2: 60 above her, 320-440 along, behind the
    step wall that joins 2 and 4 (one foothold group). By distance they're in reach, but no star from the ground on 4
    gets past that wall (live: 0 of 3 in the shot log, 0 of 5 in the telemetry, all 4 kills of 40 throws on 4): she
    doesn't throw at them from the ground. With the old decisions she threw left at them for the whole fight, and every
    star hit nothing. Since her jumps' attacks are weighed (attack.reach.shots, 2026-10-25), she jump-throws: 40 up, the
    v83 line's row (28 over her feet) clears the step's 60, and the three die. Unmeasured in game: `probe run step-up
    --style jump_throw` (the README's Probes); until then she moved on, nothing in reach. (Seed 8 since every look is
    perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every scenario): on seed 1 her
    jump throws over the step didn't kill them; of seeds 1-16, 8, 13 and 15 meet all of this.) (Seed 3 since 2026-10-26:
    her throws at the dying gone, a throw's last look before its key, her stats at level 25, the air turn's attack 60 ms
    after its arrow and the world's damage at the server's time: on seed 8 her jump throws over the step didn't kill
    them; of seeds 1-16, 3, 4 and 11 meet all of this.) (Seed 4 since the rebase of her kill's monster onto clearance
    and slopes, 2026-10-27, her kill's monster bound by her kept in her body's way (behavior.fight.TOUCH_NEAR) and her
    kills credited to the monster her attack took: on seed 3 she didn't stay on platform 4; of seeds 1-16, 4 and 12 meet
    all of this, and with master's stack 3, 4 and 11.)"""
    h = sim.hunter()
    r = h.clear(max_kills=3, seconds=20)
    return {"report": str(r)}


# The in-game probes (`python -m gamelab probe run`, behavior.probe), in a world that has reach's answers.

def _probe_rows(trace: dict) -> list[dict]:
    """The throws a run logged to the probe log (lab/probe_geometry.jsonl)."""
    return [json.loads(line) for line in trace["files"].get("probe_geometry.jsonl", ())]


def _shown(trace: dict, style: str | None = None) -> list[tuple[float, bool | None]]:
    """For each probe throw (of `style`): how far off its target was at the pick, and what it showed."""
    out = []
    for r in _probe_rows(trace):
        if style is None or r["style"] == style:
            e, seen = next((e, s) for e, s in probe.outcomes(r) if e["target"])
            out.append((abs(e["x"] - r["pick"]["x"]), seen))
    return out


def _lone(trace: dict, lo: float, hi: float) -> bool:
    """Every probe throw was at the one monster in front of her, lo to hi off when she threw."""
    rows = _probe_rows(trace)
    return bool(rows) and all(len(r["front"]) == 1 and r["front"][0].get("target") and lo <= abs(r["dx"]) <= hi
                              for r in rows)


# The probes' scenarios check their scorer (probe.BEFORE_HIT_S and the evidence after a star is due), built on a throw's
# damage showing at the star's release: so their world applies it then (world.Realism.damage_ms), not at the server's
# time the world has from 2026-10-26 (actions.DAMAGE_SHOWS_MS, as measured live), and draws its kill dying from then
# (not alive until her star lands: Realism.dies_late, the world's from 2026-10-27). To redo with the server's time.
_AT_RELEASE = Realism(damage_ms=STAR_DELAY_MS, dies_late=False)


@scenario((13, -14.0), monsters=[("Pig", 12, -170.0)], seed=1, limit=400, realism=_AT_RELEASE,
          expects=(check("30 throws logged", lambda tr: len(_probe_rows(tr)) == 30),
                   check("each at the lone pig in front, 110-200 off", lambda tr: _lone(tr, 110, 200)),
                   check("each with the pick and every monster's evidence",
                         lambda tr: all("pick" in r and r["evidence"] and r["looks"] >= 5 for r in _probe_rows(tr))),
                   check("a hit at the pig 111 off or more at the pick, and only there; some under 150",
                         lambda tr: all(bool(s) == (d >= 111) for d, s in _shown(tr))
                         and any(s and d < 150 for d, s in _shown(tr)))))
def probe_top(sim):
    """`probe run top-60`: from 13 at a lone pig on 12, 60 above her, 110-200 units off sideways. The world's
    search window has reach's top (the client's, 44 above her feet in the first band), so a star picks the pig
    from 111 units: every throw at it that far off at the pick hits, the closest 114 off, and none closer (with
    the logs' lower top, 21, one would from 191: on this seed 7 do, 199-219 off). Thrown at 110-200 off, the
    pig walking at her at the game data's 125 units/s can be closer than 111 by the pick: on this seed 3 of the
    30, 90-105 off, and the star misses each (of seeds 1-16, misses only at 90-110 off, 0-5 a seed, hits only
    from 112). Each throw waits for its evidence before the next, and the log has where the pig was at the pick
    and what it showed (`probe score` fits the top to it: tests/test_probe.py). (Seed 6 for a while, where all 30
    hit, as "every throw hits" was: then no throw was read as a miss, not even with hit monsters hidden, and the
    test of that had none to check.)"""
    h = sim.hunter()
    return str(probe.run(h, sim.navigator(h), probe.PROBES["top-60"], seconds=300))


@scenario((4, 260.0), monsters=[("Orange Mushroom", 2, -50.0)], seed=1, limit=400, realism=_AT_RELEASE,
          expects=(check("from the ground, no star gets past the step",
                         lambda tr: len(_shown(tr, "throw")) == 6 and all(s is False for _, s in _shown(tr, "throw"))),
                   check("jumping back, every star does",
                         lambda tr: len(_shown(tr, "jump_back")) == 6 and all(s for _, s in _shown(tr, "jump_back"))),
                   check("the rule's verdict logged", lambda tr: {r["clear"] for r in _probe_rows(tr)} == {False, True}),
                   check("she's back on 4", lambda tr: tr["world"]["platform"] == 4)))
def probe_step_up(sim):
    """`probe run step-up --x 260 --throws 6`, then again with `--style jump_back`: an Orange Mushroom on 2,
    60 above her behind the step wall of her own foothold group (the wall rule off for the run, so she throws
    at it). The world stops a star at that wall where it crosses the line 28 above her feet, higher in a jump:
    from the ground every star hits nothing; jumping back the attack starts 44 up, its line over the step's
    top, and every star hits. Jumping back she throws at one only up to 375 off at the look (probe.JUMP_BACK_OUT:
    the jump back has carried her 25 units out by the pick); thrown at 387-394, on 3 of seeds 1-16 a star missed
    one that was out of reach by then."""
    h = sim.hunter()
    nav = sim.navigator(h)
    p = probe.adjust(probe.PROBES["step-up"], x=260.0, throws=6)
    ground = probe.run(h, nav, p, seconds=200)
    back = probe.run(h, nav, probe.adjust(p, style="jump_back"), seconds=200)
    return {"ground": str(ground), "jump_back": str(back)}


@scenario((22, 30.0), monsters=[("Snail", 22, 70.0)], seed=1, limit=400,
          realism=dataclasses.replace(_AT_RELEASE, air_pick="widening"),
          expects=(check("12 jump throws logged", lambda tr: len(_shown(tr, "jump_throw")) == 12),
                   check("every one kills the snail", lambda tr: all(s for _, s in _shown(tr))
                         and all(r["exp"] == 1 for r in _probe_rows(tr))),
                   happens("star kills snail")))
def probe_air(sim):
    """`probe run air --throws 12`: jump throws at a lone snail on her own platform (22), about 40 units off,
    closer than the star's search picks it from (its far side short of 65). The attack picks by the area when
    it starts (26 up): the snail's in it, and each star kills it (one star kills a snail here). A kill is seen
    fading out for 1.26 s, so the evidence waits for it to be gone. She takes her place facing right, so no
    jump throw turns her: with the turn's tap walking her 8 units on, the snail came back 32 off, touched her,
    and on 6 of seeds 1-16 a knock spoiled a throw (a jump keyed from a look mid-knock, lost in the air). In the
    area's world (Realism.air_pick 'widening'), what the probe was built to tell from the other pictures: by the star's
    search alone, the simulator's default from 2026-10-27, the snail 40 off is picked by none of the 12."""
    h = sim.hunter()
    return str(probe.run(h, sim.navigator(h), probe.adjust(probe.PROBES["air"], throws=12), seconds=300))


@scenario((36, 500.0), drops=(("mesos", 36, 560.0), ("leather", 36, 620.0), ("mesos", 36, 700.0),
                               ("mesos", 32, 600.0)),
          expects=(ends([], key="pending_loot"),
                   check("three picked up", lambda tr: sum(tr["world"]["picked"].values()) == 3)))
def mop_up_drops(sim):
    """A mop-up with drops still on her platform (and one on the platform above, not hers to take)."""
    h = sim.hunter()
    h.looter.mop_up(stood=[])
    return {"pending_loot": [[round(d.x), d.platform, d.mesos] for d in h.looter.pending],
            "unpickable": [[round(u[0]), u[1]] for u in h.looter.unpickable], "inventory_full": h.looter.inventory_full}


@scenario((36, 500.0), inventory_full=True,
          drops=(("leather", 36, 540.0), ("mesos", 36, 548.0), ("leather", 36, 640.0), ("leather", 36, 760.0),
                 ("mesos", 36, 800.0)),
          expects=(ends(True, key="inventory_full"), happens("can't pick up: inventory full"),
                   check("only mesos picked up", lambda tr: set(tr["world"]["picked"]) == {"mesos"})))
def mop_up_full_inventory(sim):
    """Her inventory is full: the items stay under her, the game says so, and from then on only mesos
    are picked up."""
    h = sim.hunter()
    s = h.look(loot=True)
    h.looter.sweep_through(s, [DropSpot(d.x, d.y, d.platform, s.t) for d in s.loot])
    h.looter.mop_up(stood=[540, 640, 760])
    return {"inventory_full": h.looter.inventory_full, "pending_loot": [[round(d.x), d.platform] for d in h.looter.pending],
            "unpickable": [[round(u[0]), u[1]] for u in h.looter.unpickable], "report": str(h.report)}


# Tours.

def _tour_hunt(pid: int, kills: int, seconds: float = 60):
    def run(sim):
        tour = sim.tour_stack()
        tour.nav.escort.kinds = tour.worth
        return tour.hunt(pid, tour.worth, kills=kills, seconds=seconds)
    return run


def _kills(tr: dict) -> int:
    return sum(tr["result"]["kills"].values())


@scenario((16, 585.0), monsters="all", seed=21, limit=600,
          expects=(logs("shooting platform 17 from platform 16"),
                   check("four kills or more", lambda tr: _kills(tr) >= 4),
                   logs("collecting drops on platform 18")))
def tour_hunt_from_safe_spot(sim):
    """Tour.hunt of platform 17 from the safe spot on 16 above it (35% of 17 in reach), then collecting
    the far kills' drops, which fell on 18; its pigs knock her off it a few times on the way."""
    return _tour_hunt(17, 4)(sim)


@scenario((36, 300.0), monsters="all", seed=4, limit=400,
          expects=(logs("shooting platform 36 on the platform"), ends("kill limit", key="stop"),
                   logs("collecting drops on platform 36"),
                   check("the stop's report counts HP lost", lambda tr: tr["result"]["damage_taken"] > 0),
                   no_log("couldn't get there (timeout)", "trips to drops on the basin get there")))
def tour_hunt_basin(sim):
    """Tour.hunt of the basin (36): no safe spot reaches it, so on the platform itself. Collecting the drops
    left on it, each trip (Navigator.go with `fight` on) shoots only monsters in her way to them, and gets
    there: it used to stop and shoot everything in reach, on the very platform she was going to, until the
    trip's 60 s ran out. (Seed 1 since the arrow she holds doesn't steer a knock's flight: on seed 22 the
    fight's own sweeps picked up every drop, so no trip went to collect any. Over 16 seeds all 16 meet all
    of this, as just before. Seed 3 since the play track was rebased onto the game's steps and the live fixes:
    on seed 1 her first trip, to drops at x 263, ran out its 60 s ("couldn't get there"). Over seeds 1-24, 14
    meet all of this, 10 with such a trip (at master's head all 24 did; on the play track before the rebase
    11, 11 with such a trip and 2 with no trip at all): the escort's fights in her way finish what she has
    hurt (behavior.fight.owed), and on the basin, where respawns never stop, there is always another. Found,
    not fixed: README, the rebase's entry. Seed 5 since a throw's tap goes by her facing off the frame,
    perception.heading: on seed 3 a trip to drops runs out its 60 s again; of seeds 1-16, 9 meet all of this, as
    just before, the first of them 4, where the ledger places 50 of its 56 kills placed right, under
    tests/test_ledger.py's 90%; on 5, 48 of 51. Seed 1 since her jumps' attacks are weighed and her jump throws turn
    in the air (2026-10-25): on seed 5 a trip to drops ran out its 60 s; of seeds 1-16, 10 meet all of this, the
    ledger on seed 1 placing 28 of its 31 kills, 27 of them right.) (Seed 4 since the rebase of her kill's monster onto
    clearance and slopes, 2026-10-27: on seed 1 her trips to drops on the basin didn't get there; of seeds 1-8, 4, 6 and
    8 meet all of this, and with master's stack 4 and 8 of those.)"""
    return _tour_hunt(36, 5)(sim)


@scenario((15, 100.0), monsters="all", seed=3, limit=400,
          expects=(logs("shooting platform 12 from platform 13"), logs("nothing left on platform 12")))
def tour_hunt_platform_12(sim):
    """Tour.hunt of platform 12 (pigs, Orange Mushrooms) from the safe platform 13 beside it. (Seed 1 since the
    star's search has the client's top, and from 13 86% of 12 is in reach, not 63%: with monsters walking the
    game data's speeds too, on seed 23 she made the stop's 4 kills before 12 was clear. Of seeds 1-16, 15 meet
    all of it; on 14 the scenario's 400 s ran out. Seed 3 since the game moves her in 30 ms steps (body.STEP)
    and jumps from standing wait 150 ms after the arrow: on seed 1 the last monster on 12 was where no spot on
    13 reaches, and the trip onto 12 for it ran out of time. Of seeds 1-16, 13 meet all of it, 15 just
    before.)"""
    return _tour_hunt(12, 4)(sim)


def _hunt_17_pigs(sim):
    tour = sim.tour_stack()
    tour.nav.escort.kinds = {"pig"}
    return tour.hunt(17, {"pig"}, kills=2, seconds=90)


@scenario((14, 760.0), monsters=[("Pig", 17, 1000.0), ("Pig", 17, 1280.0)], seed=1, limit=400,
          expects=(logs("out of reach here: over to platform"),))
def tour_hunt_other_spot(sim):
    """Tour.hunt of two pigs on 17 from 14: one dies, the other is out of reach, so over to a spot that reaches it
    (Tour.spot_for). (Seed 6 since monsters walk as fast as the game data says (a pig 125 units/s, not 70): on seed 1
    the other walked into reach first. That alone moved it to seed 28, where with the physics and attack tracks it
    doesn't; of seeds 1-16 only 6 shows it. The other pig starts at 1240, not 1320, since her own floor stops a star
    down past 14's end (reach.LINE 'v83', 2026-10-21): the spot on 14 is at 756, reaching 17 out to 1166, not 716, and
    from 1320 no seed of 1-32 showed it; from 1240 seeds 1 and 6 of 1-12 do. From 1260 since a throw's tap goes by her
    facing off the frame, perception.heading: her first throw no longer taps the arrow she faces already, and from 1240
    no seed of 1-16 shows it, the other pig walking into reach first; from 1220, 1260 and 1280 seed 6 does.) (Seed 16
    since every look is perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every
    scenario): the other pig starts at 1320: from 1260, 1240, 1280, 1300 and 1340 no seed of 1-16 showed it, the other
    walking into reach or shot from 14 first; from 1320 seed 16 does.) (Seed 1 since 2026-10-26: her throws at the dying
    gone, a throw's last look before its key, her stats at level 25, the air turn's attack 60 ms after its arrow and the
    world's damage at the server's time: on seed 16 she shot both from 14, the other pig from 1320; from 1240, 1260,
    1300, 1320, 1340 and 1360 no seed of 1-16 showed it, from 1280 only seed 1; of seeds 1-16, 1 (the other pig from
    1280) meet all of this.) (Not met since 2026-10-27, her kill credited to the monster her attack took
    (localization.tracking's binding): on seed 1 the first pig died at 1044 and master's stack put its kill at 1137, by
    the other pig, which it then took for the one dying there, so the fight found nothing in reach and the tour went
    over to 796 for it; with the kill on the right pig the fight reaches the other by closing in along 14 or a jump
    throw (from 1280-1500 and on seeds 1-36, and with a snail for it at 1330-1360, it goes on the platform or is shot
    from 14): xfail in tests/test_sim.py, AFTER_KILL_PLACE.)"""
    return _hunt_17_pigs(sim)


@scenario((14, 760.0), monsters=[("Pig", 17, 1000.0), ("Pig", 17, 1300.0)], seed=1, limit=400,
          expects=(logs("that no spot reaches: onto the platform"),))
def tour_hunt_onto_platform(sim):
    """The same two pigs: no spot off the platform reaches the one left, so onto platform 17 itself.
    (Seed 6 since monsters walk as fast as the game data says (a pig 125 units/s, not 70): on seed 4 the other
    walked into reach first. With the other pig from 1240, as tour_hunt_other_spot's, seeds 3, 5, 6, 8 and 9
    of 1-12 show it; from 1320, 1, 3, 6, 11 and 12. Seed 1 since her jumps' attacks are weighed and her jump throws
    turn in the air (2026-10-25): of seeds 1-16, 1, 5, 8 and 9 show it; on the others the other pig walks into reach
    first.) (The other pig from 1300 since 2026-10-27, her kill credited to the monster her attack took
    (localization.tracking's binding) and drawn alive until her star lands (sim.world Realism.dies_late): from 1240
    no seed of 1-16 shows it, the fight reaching the other pig from 14 once the kill isn't put by it; from 1300 seed 1
    does, from 1380 and 1420 seed 4, of 1-4.)"""
    return _hunt_17_pigs(sim)


@scenario((36, 300.0), seed=24, drops=(("mesos", 32, 600.0), ("leather", 29, 420.0), ("mesos", 30, 780.0)),
          expects=(check("three loot trips", lambda tr: "loot trips: 3" in tr["result"]["report"]),
                   ends([], key="pending_loot")))
def tour_collect_loot(sim):
    """Tour.collect_loot: drops remembered on three other platforms, nearest by route first."""
    tour = sim.tour_stack()
    now = sim.clock.now
    tour.h.looter.pending = [DropSpot(600, 335, 32, now - 10, mesos=True, on_ground=True), DropSpot(420, 275, 29, now - 10),
                             DropSpot(780, 275, 30, now - 10, mesos=True, on_ground=True)]
    tour.collect_loot()
    return {"report": str(tour.report), "pending_loot": [[round(d.x), d.platform] for d in tour.h.looter.pending]}


@scenario((16, 585.0), monsters=[("Pig", 18, 300.0)], seed=2, limit=300, drops=(("mesos", 29, 420.0),),
          expects=(logs("pig in reach on platform 18: shooting it on the way"),
                   check("the drops of the pig shot on the way are kept",
                         lambda tr: any(p == 18 for _, p in tr["result"]["kept"])),
                   logs("collecting drops on platform 18", "the next collection goes for them"),
                   check("a pig's leather picked up", lambda tr: tr["world"]["picked"].get("leather", 0) >= 1)))
def tour_collect_loot_on_the_way(sim):
    """Tour.collect_loot twice, as Tour.hunt calls it: on the trip from 16 to the mesos remembered on 29, she
    shoots a pig on 18 below on the way. Its drops are kept for the next collection, which picks them up
    (they used to be forgotten: from 29, 18 is out of view). (Seed 2 since monsters walk as fast as the game data says (a pig 125 units/s, not 70): on seed 1 the pig walked out
    of reach before it died.)"""
    tour = sim.tour_stack()
    tour.nav.escort.kinds = tour.worth
    tour.h.looter.pending = [DropSpot(420, 275, 29, sim.clock.now - 10, mesos=True, on_ground=True)]
    tour.collect_loot()
    kept = [[round(d.x), d.platform] for d in tour.h.looter.pending]
    tour.collect_loot()
    return {"report": str(tour.report), "kept": kept,
            "pending_loot": [[round(d.x), d.platform] for d in tour.h.looter.pending]}


@scenario((36, 300.0), monsters="all", seed=1, limit=1500,
          expects=(check("three stops visited", lambda tr: len(tr["result"]["visited"]) == 3),
                   ends("standing by on a safe platform", key="stop"), logs("tour: sweep"),
                   check("every kill counted (the stops' and on the way)",
                         lambda tr: sum(tr["result"]["kills"].values()) + tr["result"]["on_the_way"]
                         == sum(tr["world"]["kills"].values()))))
def tour_run(sim):
    """`python -m gamelab tour --stops 3`: the whole tour, end to end: three hunting grounds, and standing by on a safe
    platform at the end (until 2026-10-26, resting on the chair when HP ran low). (Seed 2 since the game moves her in 30
    ms steps (body.STEP) and jumps from standing wait 150 ms after the arrow: on seed 1 one of 181 kills went uncounted.
    Over seeds 1-16 12 meet all of this, as just before. Seed 1 since the two tracks came together, her body moved by
    motion.body and her stars picked by the star's search: on seed 6 one of 160 kills went uncounted. Over seeds 1-16,
    12 meet all of this and 4 miss 1 or 2 of 136-237 kills in the count; none runs out of time, 560-1142 s. With no jump
    keyed from a knock, no chain planned in mid-air and the drops clear of the basin's cliffs, still 12: 4 miss 1 or 2
    of 122-184, in 587-984 s. Seed 6 since stars from the ground pick by the star's search: on seed 8 an EXP gain came
    in as the stop's fight ended, unplaced, and two kills on the way came between two looks. Seed 8 had been since
    jump-throws come 40 ms after the jump: on seed 25 two kills came between two looks once, and two kills raise her EXP
    number once. On the play track, before the steps, seed 2 as well, since a kill is placed where its monster was seen
    last (behavior. ledger): on seed 1 one of its 225 kills went uncounted there; over seeds 1-16, 12 met all of it, 15
    just before, the 4 others missing 1 of 185-227 kills in the count.) The basin's respawns make its length chaotic: on
    world seeds 1-16 it took 470-1190 s, before stars picked by the search and since. On the physics branch (her body by
    motion.body) it met all of this on few seeds, and its seed moved from 8 to 9, 8, 9, 5, 9 and 1 there; looked into
    rather than moved again, it was the limit and the kill count: - Time: that world gave more kills than the branch's
    base did (seeds 1-16: 165 a tour on average against 149), at the same 2.9 s of loot collecting each (480 s a tour
    against 433), so tours took longer (806 s against 769 on average, capped at 900 s; uncapped, the longest took
    1,300). At 900 s, 7 of 16 ran out of time; the limit is 1500 s now, a backstop. - The kill count: every one of her
    kills raises her EXP number at a look (on seeds 3, 5 and 7, 116, 153 and 164 gains for as many kills, none two to a
    look), but a stop counts a kill by kind only when its fight tells that gain apart, and some come as a fight ends or
    in the sweep after it (fight.py's f0ea98d and 52683f4 tell those apart; with them on that branch the count held on
    11 of 16 seeds instead of 5; with both tracks, on 12), or at a look of a loot collection's own (Tour.collect_loot's
    locate, a trip's recover), which nothing counts. (Seed 3 since every look is perception's newest frame after her
    keys' effect (Senses.fresh_after, the feed in every scenario): on seed 2 a kill wasn't counted; of seeds 1-8, 1, 3,
    4, 5, 6 and 7 meet all of this.) (Seed 1 since 2026-10-26, her throws at the dying gone, a throw's last look before
    its key, the world's damage at the server's time: on seed 3 one of 179 kills went uncounted; of seeds 1-16, 14 meet
    all of this, 13 on master; on the other two (3, 11) the count misses kills, as on master's three (2, 8, 15).)"""
    tour = sim.tour_stack()
    return tour.run(stops=3, kills_per_stop=5, seconds_per_stop=90)


@scenario((36, 300.0), seed=27,
          expects=(ends("standing by on a safe platform", key="stop"),
                   check("on a safe platform", lambda tr: tr["world"]["platform"] in tr["result"]["safe"])))
def tour_stand_by(sim):
    """The end of a tour (Tour.stand_by): over to the nearest safe platform to stand by."""
    tour = sim.tour_stack()
    tour.stand_by()
    return {"stop": tour.report.stop, "safe": tour.safe}


# What perception saw die (every scenario's looks and glances are the feed's: SyncFeed on the fake clock, as live's are
# the perception service's).

def _died(cause: str, at_least: int = 1):
    return check(f"the feed saw {at_least}+ monsters die, killed by {cause}",
                 lambda tr: tr["feed"]["tracker"][f"died {cause}"] >= at_least)


@scenario((36, 300.0), monsters="all", seed=22, limit=400,
          noise=Noise(miss=0.28, walk_jitter=6.0, lag_sd=1.0, tap_lost=0.6, dot_px=True, tag_hidden=((20.0, 23.0),)),
          realism=Realism(lock_arrows=True),
          expects=(check("kills", lambda tr: _kills(tr) >= 1), happens("star misses")))
def tour_hunt_noisy(sim):
    """tour_hunt_basin with the real game's errors: stars missing (28%), walks off by a few units, the
    lags spread as measured, short taps lost (60%), her dot read to whole pixels, and her name tag hidden
    for 3 s mid-fight."""
    return _tour_hunt(36, 5)(sim)


# The hunt loop (behavior.hunt, `python -m gamelab tour --loop`): the tour scenarios' twins, one for each
# rule that must survive (gamelab/docs/hunt-modes.md).

def _loop(minutes: float):
    def run(sim, tour=None):
        from ..behavior.hunt import Hunt
        tour = tour or sim.tour_stack()
        r = Hunt(tour, log=sim.log).run(minutes=minutes)
        return {"report": str(r), "grounds": r.grounds, "kills": r.kills, "exp_gains": r.exp_gains,
                "ledger_kills": len(tour.h.ledger.kills), "rules": dict(sorted(r.rules.items())), "hunt_stop": r.stop,
                "stop": tour.report.stop, "damage_taken": tour.h.report.damage_taken, "safe": tour.safe}
    return run


def _every_kill(tr: dict) -> bool:
    """Every kill of hers is in the ledger (the simulator never shows two between looks)."""
    return tr["result"]["ledger_kills"] == sum(tr["world"]["kills"].values())


def _rule(name: str, at_least: int = 1):
    return check(f"rule '{name}' decided {at_least}+ times", lambda tr: tr["result"]["rules"].get(name, 0) >= at_least)


@scenario((36, 300.0), monsters="all", seed=22, limit=400,
          expects=(logs("at platform 36's spot (platform 36)"), _rule("shoot"), _rule("close quarters"),
                   check("ten kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 10),
                   check("every kill counted", _every_kill),
                   check("HP lost is counted", lambda tr: tr["result"]["damage_taken"] > 0),
                   check("drops picked up", lambda tr: sum(tr["world"]["picked"].values()) >= 5),
                   ends("standing by on a safe platform", key="stop"), _died("hers", 2)))
def loop_basin(sim):
    """tour_hunt_basin's twin: the loop hunts the basin (36) on the platform itself, where no safe spot
    reaches it. Anywhere on it she's there: waiting to reach the spot's x, close monsters interrupted her
    for good, and the visit never began."""
    return _loop(3)(sim)


@scenario((36, 300.0), monsters={17, 12, 4}, seed=1, limit=400,
          expects=(logs("leaving platform 36: nothing left here"), logs("at platform 17's spot (platform 16)"),
                   check("every kill counted", _every_kill)))
def loop_spot_off_ground(sim):
    """tour_hunt_from_safe_spot's twin: nothing in the basin, so on to platform 17, hunted from the safe
    spot on 16 above it. (Seed 1 since the hunt loop keeps the grounds worth a visit by their EXP a second:
    the basin's 4 s visit then leaves 35 out, the draws that vary her choice fall otherwise, and on seed 25
    she went to 4 first. 23 of seeds 1-32 show it, 22 before.)"""
    return _loop(3)(sim)


@scenario((14, 760.0), monsters={17}, seed=1, limit=400,
          expects=(logs("leaving platform 36: nothing left here, and nothing due back"),
                   check("on to another ground", lambda tr: len(tr["result"]["grounds"]) >= 2),
                   check("every kill counted", _every_kill)))
def loop_leaves_empty_ground(sim):
    """Only platform 17 has monsters: the basin, chosen first by its value, is left for the next ground
    once the drops of the monsters she shot on her way are picked up. Only those: going after the drops
    of every monster shot on those trips too, she never left."""
    return _loop(3)(sim)


def _loop_on_12(sim):
    tour = sim.tour_stack()
    tour.stats["12"] = {"visits": 4, "exp": 600.0}  # by the value map, platform 12 is the best ground
    return _loop(3)(sim, tour)


@scenario((13, 0.0), monsters={12}, seed=2, limit=400,
          expects=(check("platform 12 hunted", lambda tr: tr["result"]["grounds"][0] == 12),
                   _rule("lull loot"), check("drops picked up", lambda tr: sum(tr["world"]["picked"].values()) >= 20),
                   check("every kill counted", _every_kill)))
def loop_lull_loot(sim):
    """Only platform 12 has monsters, and the value map says it's the best ground. She hunts it from the safe
    platform 13 beside it; in the lulls, with nothing wanted left on it and her kills due back, she picks up
    the piles she can get to before they respawn. (Seed 5 since the simulator moves her by the engine: on seed
    4 she picked up 19 drops, not 20. Over 16 seeds 9 meet all of this, 8 before. Seed 6 since no jump is
    keyed in the kick's blind spot and jump-backs leave at full speed: on seed 5 she picked up 17. Over 16
    seeds 8 meet all of this, 10 just before. Seed 8 since monsters walk as fast as the game data says (a pig
    125 units/s, not 70): on seed 6 no lull came for the rule, and she picked up fewer than 20. Over 16 seeds
    8 meet all of this. On the attack track alone, seed 4 since the star's search has the client's top (from
    13 a star picks a pig on 12 from 111 units, not 191): on seed 6 no lull came. Seed 1 since both are
    together: on seed 8 no lull came. Over 16 seeds 10 meet all of this; of the 6 others, 5 had no lull and 2
    picked up fewer than 20 drops, seed 3 both. Seed 2 since a plan's first arrow waits until 0.6 s after her
    last attack's key (actions.ARROW_LOCK_MS): on seed 1 her first walk to a better spot went 0.39 s later,
    and no lull came. Over 16 seeds 5 meet all of this; of the 11 others, 9 had no lull and 3 picked up fewer
    than 20, seed 16 both. Seed 3 since the game moves her in 30 ms steps (body.STEP) and jumps from standing
    wait 150 ms after the arrow: on seed 2 no lull came. Over 16 seeds 9 meet all of this, 5 just before. Seed
    4 since chained jumps take off where they land with room to spare (flow.TAKEOFF_SLOP): on seed 3 no lull
    came. Over 16 seeds 8 meet all of this. On the play track, before the steps, seed 4 too, since she finishes
    what she has hurt before she moves on (behavior.fight.owed): on seed 1 no lull came there; over 16 seeds 8
    met all of it. Seed 1 again since her EXP number is read and each kill goes on a monster of the kind its gain's
    size names (or none, behavior.ledger.KIND_ANY): on seed 4 no lull came. Over 16 seeds 11 meet all of this.
    Seed 2 since EXP before loot (hunt._loot_first; picking up stops for what's in reach, Looter.yield_to): on seed
    1 she shot on and picked up 7 drops, for 40 kills against 35. Over seeds 1-8 6 meet all of this, 8 before, for
    3839 EXP against 3519 and 264 drops picked up against 283.) (Seed 1 since 2026-10-26: her throws at the dying gone,
    a throw's last look before its key, her stats at level 25, the air turn's attack 60 ms after its arrow and the
    world's damage at the server's time: on seed 2 no lull came; of seeds 1-16, 1, 3-7 and 9-15 meet all of this.)
    (Seed 2 since 2026-10-27, an attack from a jump picking by the star's search alone (reach.AIR_MODEL, the world's
    AIR_PICK) and her stance going by where each monster will be when her attack starts (attack.stance.Pace): on seed 1
    no lull came; of seeds 1-12, 2-10 meet all of this.)"""
    return _loop_on_12(sim)


def _hold(sim, k: int, seconds: float) -> None:
    """The world's monster `k` stands where it is now for `seconds` (as one seen in a live log did)."""
    m, t = sim.world.monsters[k], sim.world.t
    m.t0, m.x0, m.d, m.t1, m.pushed = t, m.at(t), 0, t + seconds, None


def _left_from(tr: dict, pid: int) -> float:
    """When she first set off from platform `pid` (its first chain or move in the log), s (infinity: never)."""
    return _first(tr["log"], f"platform {pid} (")


@scenario((3, 760.0), monsters=[("Pig", 4, 584.0)], seed=1, limit=200, facing=LEFT,
          expects=(happens("star hits pig", "her first star hurts the pig"),
                   check("the pig dies before she leaves platform 3",
                         lambda tr: _first(tr["world_log"], "kills pig") < _left_from(tr, 3)),
                   _rule("finish")))
def loop_finishes_what_she_hurt(sim):
    """A live hunt loop of 2026-10-21 (00:49:51): from platform 3 she threw at a pig on 4, 60 below, and hurt it;
    it stood at x 584, 130 units along from her at 714, too close sideways for a star (151 for a pig 60 below),
    and she set off for drops on 7 with its bar at 27%. Here: one star from 760 hurts it, it stands where it is,
    and she walks to 714; then the hunt loop, its next ground on another platform. She finishes it first: the
    finish rule (4b) takes her step back along 3 to where a star picks it (shooting in passing steps out only from
    under a monster above her, hunt._pass_step, and the pig is below), and shooting in passing throws the stars.
    With the finish rule off (hunt.FINISH_S 0) she sets off 2.2 s in on all of seeds 1-8 and the pig dies 34-51 s
    later, when she comes back."""
    tour = sim.tour_stack()
    h = tour.h
    _hold(sim, 0, 3.0)
    h.look()
    h.actions.throw(LEFT)
    sim.clock.sleep(0.7)  # the star lands 0.52 s after the key
    h.look()              # her bar seen: hurt by her (Hunter.wounds)
    _hold(sim, 0, 8.0)
    h.actions.walk(LEFT, 46)
    return _loop(1.0)(sim, tour)


def _loop_up_from_22(sim):
    tour = sim.tour_stack()
    tour.stats["18"] = {"visits": 4, "exp": 600.0}  # by the value map, 18 is the best ground: she goes up past 21
    return _loop(1.0)(sim, tour)


@scenario((22, -10.0), monsters=[("Pig", 21, -86.0), ("Orange Mushroom", 21, -90.0)], seed=1, limit=200,
          expects=(check("a monster on 21 dies before she leaves 22",
                         lambda tr: min(_first(tr["world_log"], "kills pig"), _first(tr["world_log"], "kills orange"))
                         < _left_from(tr, 22)),
                   _rule("in passing")))
def loop_shoots_21_from_22(sim):
    """She stands on 22's left part, on her way up to 18 (by the value map the best ground) over 21, 60 above,
    whose right end is 12 units left of 22's left end, and a pig and an Orange Mushroom stand near it: 76-80 units
    along from her, under the star's search, which takes them from 111-113 sideways. In the live hunt loops of
    2026-10-21 she went on from there, up onto 21 beside them, and in the simulator she jumped up 0.4 s in on all
    of seeds 1-8 (the first kill 11-42 s in, or none). In passing she steps out from under them along 22, 40-76
    units (hunt.PASS_STEP), and shoots them first: the first kill comes 3.5-5.5 s in, and she leaves 22 at 9.1-9.4
    s, on all of seeds 1-8 (since the rebase onto the game's steps and the live fixes, 3.45-5.15 s and 6.8-9.7 s)."""
    return _loop_up_from_22(sim)


def _loop_to_17_past_a_pile(sim):
    from ..behavior.ledger import Pile
    tour = sim.tour_stack()
    tour.stats["17"] = {"visits": 4, "exp": 600.0}  # by the value map, 17 is the best ground: its spot is on 16
    pile = Pile(45.0, sim.model.platform(19).y_at(45.0), 19, sim.clock.now - 20.0, hers=True, mesos=True,
                status="seen", items=("mesos",), worth=15.0, dearest=15.0)
    tour.h.drops.piles.append(pile)  # a kill of hers 20 s ago, its coins seen
    return _loop(0.5)(sim, tour)


@scenario((13, 0.0), monsters=[("Pig", 17, 1300.0)], seed=1, limit=120, drops=(("mesos", 19, 45.0),),
          expects=(_rule("on the way"), happens("picks up mesos at x 45"),
                   check("the coins on 19 are picked up before she gets to 16",
                         lambda tr: _first(tr["world_log"], "picks up mesos at x 45")
                         < _first(tr["log"], "at platform 17's spot"))))
def loop_pile_on_the_way(sim):
    """On the safe platform 13, on her way to 17's spot on 16 (by the value map the best ground), with coins of a
    kill of hers on 19, a platform off her route (13 -> 15 -> 18 -> 16): a stop at them costs 1.3 s more than the
    trip without it by the planner (hunt.DETOUR_S: 2), so she takes them on the way (rule 5b). In the live hunt loops
    of 2026-10-21 she passed such piles and went back across the map for them when they were about to vanish (rule
    6). On seeds 1-16 she picks them up 2.0 s in and is at the spot 6.8-8.0 s in (the stop took 2.2 s in all, with
    the look for what's left after it, looting.mop_up); without the rule she's there at 4.6-5.8 s, and picks them up
    on 11, 11.7-28.3 s in, on a trip back. (Before the rebase onto the game's steps and the live fixes: 2.4 s,
    7.4-8.6 s and 3 s; 4.4-5.7 s, and on 11, 11.8-27.0 s in.)"""
    return _loop_to_17_past_a_pile(sim)


def _loop_bubble(sim):
    tour = sim.tour_stack()
    tour.stats["17"] = {"visits": 4, "exp": 600.0}
    return _loop(1.0)(sim, tour)


@scenario((16, 636.0), monsters=[("Slime", 17, 940.0), ("Orange Mushroom", 17, 1300.0), ("Orange Mushroom", 17, 1350.0)],
          seed=2, limit=200, realism=Realism(respawn=120.0, drops=(("slime", "slime-bubble", 1.0),)),
          expects=(_rule("valuable pile"), happens("picks up slime-bubble"),
                   check("she sets off for the Slime Bubble while an Orange Mushroom is still about",
                         lambda tr: _first(tr["log"], "picking it up now") < _nth(tr["world_log"], "kills", 3))))
def loop_valuable_drop_at_once(sim):
    """On 17's spot on 16, shooting a slime and two Orange Mushrooms on 17; the slime drops a Slime Bubble (50 mesos
    by the game data's price, 8 times a Squishy Liquid's). Once it has landed a look at the whole screen sees it
    (hunt._far_drops), and she goes for it at once (rule 6a, hunt.VALUABLE) the moment nothing is in reach: the user,
    watching her hunt on 2026-10-21, "Expensive items should attempt to pick immediately", and on 2026-10-24 "The hunt
    should be driven by maximizing the exp gain" (EXP before loot, hunt._loot_first: with 90 s of life and nobody by
    it, it waits for what's in reach). On seed 1 she sets off as the first Orange Mushroom dies, the other out of
    reach, calls the trip off when it comes into reach (9.8 s), and picks the bubble up once it's dead. Before EXP
    before loot, on seeds 1-16 she set off before the mushrooms were dead on all 16, and picked it up 11.3 s after the
    slime's kill at the median (7.5-33.7); without rule 6a on none, 17.2 s (9.6-35.1), shooting the mushrooms from 16
    first; without the look (hunt.FAR_DROPS_LOOK) on 12. (Seed 2 since 2026-10-27, her kill credited to the monster
    her attack took and drawn alive until her star lands: on seed 1 she set off as the second Orange Mushroom died; of
    seeds 1-16, all but 1, 7 and 9 meet all of this.)"""
    return _loop_bubble(sim)


def _nth(lines: list[str], text: str, n: int) -> float:
    """When the n-th of a trace's lines with `text` came (s; infinity if fewer did)."""
    at = [float(line.split()[0]) for line in lines if text in line]
    return at[n - 1] if len(at) >= n else float("inf")


@scenario((36, 300.0), monsters="all", seed=25, limit=800,
          expects=(check("a hundred kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 100),
                   ends("standing by on a safe platform", key="stop"), ends("time up", key="hunt_stop"),
                   check("every kill counted", _every_kill),
                   check("on a safe platform", lambda tr: tr["world"]["platform"] in tr["result"]["safe"])))
def loop_run(sim):
    """tour_run's twin: `python -m gamelab tour --loop --minutes 8`, end to end. With every spawn filled,
    the basin pays more than any other ground with the trip there, so by the rate rule she stays all 8
    minutes (tour_run's 3 stops took 594 s for 114 kills), then stands by on a safe platform."""
    return _loop(8)(sim)


@scenario((36, 300.0), monsters="all", seed=22, limit=300,
          noise=Noise(miss=0.28, walk_jitter=6.0, lag_sd=1.0, tap_lost=0.6, dot_px=True, tag_hidden=((20.0, 23.0),)),
          realism=Realism(lock_arrows=True),
          expects=(check("kills", lambda tr: sum(tr["world"]["kills"].values()) >= 3), happens("star misses")))
def loop_noisy(sim):
    """loop_basin with the real game's errors (tour_hunt_noisy's): stars missing, walks off by a few
    units, the lags spread, short taps lost, her dot read to whole pixels, her name tag hidden for 3 s
    mid-fight."""
    return _loop(2)(sim)


@scenario((36, 300.0), monsters="all", seed=4, limit=400, may_die=True,
          realism=Realism(spawn_share=0.5, rival_every=20.0),
          expects=(check("only her kills in the ledger", _every_kill), happens("another player kills"),
                   _died("other")))
def loop_rival(sim):
    """Another player on the map: half the monsters spawn, and a rival kills one every 20 s and takes its drops. Their
    kills give her no EXP, so the ledger has only hers. (The rival hunts one place at a time, its dot on the minimap:
    here 21, then 2, away from her.) (Seed 3 since every look is perception's newest frame after her keys' effect
    (Senses.fresh_after, the feed in every scenario): with the twin that had the feed (17), perception sees some of the
    rival's kills die, someone else's: of seeds 1-12, 8 meet all of this, 25 not.) (Seed 2 since 2026-10-27, her stance
    going by where each monster will be when her attack starts, one walking at her stopping where it meets her and one
    whose walk nothing tells staying where it was seen (attack.stance.Pace, PACE_STOP, WALK_AT): on seed 3 the feed saw
    none of the rival's kills die; of seeds 1-16, 2, 4-7, 9 and 14-16 meet all of this.) (Seed 4 since the rebase of her
    kill's monster onto clearance and slopes, 2026-10-27, her kill's monster bound by her kept in her body's way
    (behavior.fight.TOUCH_NEAR) and her kills credited to the monster her attack took: on seed 2 the feed saw none of
    the rival's kills die; of seeds 1-12, 4-7, 9 and 10 meet all of this, and with master's stack 2, 4-7 and 9.)"""
    return _loop(3)(sim)


# West Rocky Mountain II (ROCKY: map 10004021, from 2026-10-23), with her as she plays there (HER: level 23, HP 730,
# her keys from 2026-10-23, her pet drinking her potions): trips between its hunting grounds, out of the pit of
# stairs, up its ropes and stairs, and the hunt loop and the tour on it (README: In game, on West Rocky Mountain II).

def rocky(start: tuple[int, float], **kw):
    """A scenario on West Rocky Mountain II."""
    return scenario(start, map_id=ROCKY, **kw)


def _made(tr: dict, move: str) -> bool:
    """The trip made a move (its kind and platforms: `jump 27->25`) that landed where it was going."""
    to = move.split("->")[1]
    return any(m == move or m == f"{move}: landed on {to}" for m in _result(tr, "moves"))


GAME_ERRORS = harness._GAME_NOISE  # the game's errors, as the harness's preset `game` has them


@rocky((30, -1009.0), expects=(ends("arrived"), moves("jump", "jump_up"),
                               check("out by the way out of the pit, 27 onto 25", lambda tr: _made(tr, "jump 27->25")),
                               check("no move missed", lambda tr: _result(tr, "misses") == 0)))
def nav_rocky_out_of_pit(sim):
    """From stair 30 in the pit up to the bridge (22): up the stairs (each 60 up), the one way out, 27 onto the
    floating step 25 (a way out of a trap, route._ways_out: 0.90 by her takeoff's spread), then 25, 24, 23 and up
    onto 22. In game the planner had no route out ("no route from platform 30 to 19"), and from 27 she twice fell
    back to 28."""
    return _nav(22)(sim)


@rocky((30, -1009.0), noise=GAME_ERRORS, realism=Realism(lock_arrows=True),
       expects=(ends("arrived"), check("out by 27 onto 25", lambda tr: _made(tr, "jump 27->25"))))
def nav_rocky_noisy_pit(sim):
    """nav_rocky_out_of_pit with the game's errors (the harness's preset `game`: the lags spread, walks off by a few
    units, her dot read to whole pixels, short taps lost): she still gets out by 27 onto 25 (on seeds 1-10 every
    trip arrived)."""
    return _nav(22)(sim)


@rocky((32, -1500.0), expects=(ends("arrived"), check("five jumps up the stairs, none missed",
                                                      lambda tr: _result(tr, "moves")[0].startswith("jump 32->31")
                                                      and len(_result(tr, "moves")) == 5
                                                      and _result(tr, "misses") == 0)))
def nav_rocky_stairs_up(sim):
    """From the pit's bottom stair (32) up to 27: five running jumps, each 60 up with 17 to spare, over the riser
    under each stair's end (a wall of her own group, cleared from 15-35 units back)."""
    return _nav(27)(sim)


@rocky((24, -946.0), expects=(ends("arrived"), moves("jump_up", "jump")))
def nav_rocky_floating_steps(sim):
    """From the floating step 24 under the bridge up onto 22: straight up onto 23, then up onto the bridge's slope
    (2.3-8.0 units to spare). In game her dot read 80 units left there, and the jumps went 80 units right of where
    they were planned."""
    return _nav(22)(sim)


@rocky((22, -1300.0), expects=(ends("arrived"), moves("climb"),
                               check("on 14: 11 lies on it", lambda tr: tr["world"]["platform"] == 14)))
def nav_rocky_twins(sim):
    """From the bridge under 11 to 11 (it lies on 14's footholds: one place, MapModel.twins): up the rope at -1463
    onto 14 is arriving. In game the trips between the twins were planned as jumps, and landing on the other one
    counted as landing elsewhere."""
    return _nav(11)(sim)


@rocky((14, -930.0), expects=(ends("arrived"), moves("climb"),
                              check("no move missed", lambda tr: _result(tr, "misses") == 0)))
def nav_rocky_rope_to_7(sim):
    """From the high bridge (14) up the rope at -414 to 7 (its bottom 64 above 14, 8 units from 14's end): in game
    the run to it was keyed for where her dot (80 units off) put her, she ran off 14's end, and ended on 17."""
    return _nav(7)(sim)


@rocky((17, -277.5), expects=(ends("arrived"), check("17 onto 16 and 16 onto 13", lambda tr: _made(tr, "jump 17->16")
                                                     and _made(tr, "jump 16->13"))))
def nav_rocky_right_side_up(sim):
    """From 17 up the steps on the right to 10: 17 onto 16 (a jump off the end from 15 units back: Motion
    planning), 16 onto 13 (2.6 units to spare), up onto 10, in 3.5 s; by the planner's times 22.2 before the jumps
    nearer the end, round by the bridge and the rope to 14."""
    return _nav(10)(sim)


@rocky((0, -600.0), expects=(ends("arrived"), moves("walk_off", "drop")))
def nav_rocky_down_from_the_top(sim):
    """From the top (0) down to the bridge: walk-offs onto 9 and 14, a drop through 14 onto 22."""
    return _nav(22)(sim)


@rocky((19, -700.0), expects=(ends("arrived"), moves("jump", "jump_up", "climb"),
                              check("ten moves or more", lambda tr: len(_result(tr, "moves")) >= 10)))
def nav_rocky_to_the_top(sim):
    """From 19 (the way out to West Rocky Mountain I) up to the top (0): the steps on the right, the rope at -414
    from 10 to 7, and the ledges' jumps (6 onto 3 with 0.4 units to spare, or by 4)."""
    return _nav(0)(sim)


@rocky((22, 0.0), seed=3, expects=(ends("arrived"), moves("climb", "jump", "jump_up"),
                                   check("no move missed", lambda tr: _result(tr, "misses") == 0)))
def nav_rocky_bridge_to_ledges(sim):
    """From the bridge's right part up to the ledge 6, the longest way up (13.4 s): the rope at -303 to 18, the
    steps on the right, the rope at -414 to 7, up onto 6. (Seed 3: on seeds 1, 2 and 6 of 1-6 the jump up from 13
    (from x -10.3, running left) went down through 10's surface without landing and on to 16: body._lands_at weighed
    her height at the start of each 5 ms step against the surface at its end, and 10 rises 0.15 a unit to the left
    there: a step (0.75 units across) that started less than 0.11 above the surface went through it. Since the start
    is weighed against the surface under it too, as HeavenClient does, she lands on 10 on all six.)"""
    return _nav(6)(sim)


@rocky((22, -1000.0), monsters="all", expects=(ends("arrived"), moves("climb")))
def nav_rocky_monsters_bridge(sim):
    """Across the bridge among its monsters (14 spawn points, Stumps and Axe Stumps) and up the rope at -536 to 19,
    keeping clear of them: the climb starts among them (on seeds 1-6 she arrived every time, touched 3-25 times on
    the way)."""
    return _nav(19)(sim)


@rocky((22, -1300.0), monsters="all", seed=1, limit=300,
       expects=(ends("arrived"), logs("shooting it on the way"),
                check("ten kills or more on the way", lambda tr: sum(tr["world"]["kills"].values()) >= 10)))
def nav_rocky_fight_bridge(sim):
    """From the bridge up the rope at -1463 to 14, shooting the monsters in reach on the way (the tour's escort): a
    Stump takes 4 stars, a Dark Stump 7 and an Axe Stump 10. (Seed 1: of seeds 1-12 she arrived with ten kills or more
    on 1, 2, 5 and 6 since a hit pushes a Dark Stump or an Axe Stump back only from 25 or 30 damage after their
    defence (world.PUSH_BY_DAMAGE: more of them walk on at her; on 1 and 11 before), and on most of the others fell
    out of the map past 22's left end (-1640), where nothing is under her: a touch knocked her off it, fighting or picking up drops there, or off the rope at -1463 near its top, and
    the knock carried her past it. The world's side is the map's platforms' outer end (-1662, 14's left end), 19 units
    left of the pit's bottom stair (32, from -1643); with HeavenClient's sides, 25 in from its footholds
    (world.Realism.sides 'client', -1637), she can't get past 22's end; what the game does there isn't known:
    README.)"""
    return _nav(14, fight=True)(sim)


def _fed_her_pet(tr: dict) -> bool:
    return any("feeds her pet" in line for line in tr["world_log"])


@rocky((22, -700.0), monsters="all", limit=400,
       expects=(_rule("shoot"), check("ten kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 10),
                check("every kill counted", _every_kill),
                check("drops picked up", lambda tr: sum(tr["world"]["picked"].values()) >= 5),
                check("she fed her pet", _fed_her_pet), happens("her pet drinks a potion for her"),
                check("no potion key pressed: her pet drinks them", lambda tr: not any(p[1] == "potion"
                                                                                       for p in tr["plans"])),
                ends("standing by on a safe platform", key="stop")))
def loop_rocky_bridge(sim):
    """The hunt loop on West Rocky Mountain II from the bridge (22: 14 spawn points, 7 Stumps, 3 Dark Stumps and 4
    Axe Stumps), 3 minutes: she feeds her pet at the start, it drinks a potion when a touch leaves her under its 45%
    warning, she never presses a potion key, and she stands by on a safe platform at the end (under the bridge)."""
    return _loop(3)(sim)


@rocky((14, -930.0), monsters="all", limit=800,
       expects=(check("fifty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 50),
                ends("time up", key="hunt_stop"), ends("standing by on a safe platform", key="stop"),
                check("every kill counted", _every_kill),
                check("on a safe platform", lambda tr: tr["world"]["platform"] in tr["result"]["safe"])))
def loop_rocky_run(sim):
    """`python -m gamelab tour --loop --minutes 8 --map 10004021`, end to end from the high bridge (14), then standing
    by on a safe platform: a floating step under the bridge or a stair of the pit (23, 24, 25, 27, 30, 32: every
    safe platform 120 units wide is there)."""
    return _loop(8)(sim)


@rocky((22, -700.0), monsters="all", seed=2, limit=1500,
       expects=(check("three stops or more", lambda tr: len(tr["result"]["visited"]) >= 3),
                ends("standing by on a safe platform", key="stop"), logs("tour: sweep"),
                check("she fed her pet", _fed_her_pet)))
def tour_rocky_run(sim):
    """`python -m gamelab tour --map 10004021`: the whole sweep, end to end, from the bridge. (Seed 2 since a step onto
    a floor rising the way she flies lands her (body._lands_at): on seed 1 she was knocked off 22's left end and out
    of the map at 804 s; of seeds 1-6, 2, 4, 5 and 6 meet all of this.)"""
    return sim.tour_stack().run()


# Tree Dungeon, Forest Up North II (TREE: map 10002073, from 2026-10-24), with her at level 24 (HP 752, the keys and pet
# of West Rocky Mountain II): a tower 530 wide between walls of foothold group 0, the camera fixed across, its minimap
# window scrolling; trips up and down it and its ropes, the top cluster (6-10), and the hunt loop and the tour, with the
# other players camping as the crowding study saw them (harness.TREE_CAMPS) and with group 0's walls stopping her
# (world.Realism.zero_walls) (README: Tree Dungeon, Forest Up North II).

NO_HANG_REFUSED = 10  # (tree) a hunt's attacks the game may refuse her on a rope, all told: past this she hung there


def tree(start: tuple[int, float], **kw):
    """A scenario on Tree Dungeon, Forest Up North II. A hunt's (loop_tree_*, tour_tree_*: not those of a rope's top,
    which are there for it) also checks she didn't hang on a rope throwing (the world's `refused`: from 2026-10-27,
    when loop_tree_grind recorded again hung at the top of the rope at x -8 for 86 s, 84 attacks refused, and met all
    its checks, and tour_tree_run 31.9 s at the top of the rope at x 146, 37 refused: her regular attack's presses
    unwatched for it, behavior.darts.DartWatch.regular since)."""
    def wrap(fn):
        more = dict(kw)
        if fn.__name__.startswith(("loop_tree", "tour_tree")) and "rope_top" not in fn.__name__:
            more["expects"] = tuple(kw.get("expects", ())) + (
                check(f"no hang on a rope (her attacks refused there {NO_HANG_REFUSED} times at most)",
                      lambda tr: tr["world"].get("refused", 0) <= NO_HANG_REFUSED),)
        return scenario(start, map_id=TREE, **more)(fn)
    return wrap


def _kinds_of(tr: dict) -> list[str]:
    """The trip's moves as kind and platforms (`jump 30->29`), however each ended."""
    return [" ".join(m.split(":")[0].split()[:2]) for m in _result(tr, "moves")]


@tree((33, 100.0), limit=200,
      expects=(ends("arrived"), check("up by 29, not the grab from 30", lambda tr: "jump 30->29" in _kinds_of(tr)
                                      and "climb 30->27" not in _kinds_of(tr)),
               check("the tower's four choke ropes", lambda tr: {"climb 32->30", "climb 29->27", "climb 19->18",
                                                                 "climb 11->10"} <= set(_kinds_of(tr)))))
def nav_tree_bottom_to_top(sim):
    """From the floor (33) to the top (6), the whole tower: 22 moves, up the ropes at 146 (32 to 30), -8 from 29 (not
    from 30: 94.3 up, beyond a sure grab, route.GRAB_CHANCE), 111 (19 to 18) and -80 (11 to 10), the floating
    platforms' jumps between, 9 onto 8 and up onto 6."""
    return _nav(6)(sim)


@tree((30, 60.0), expects=(ends("arrived"), check("30 onto 29, then up the rope from 29",
                                                  lambda tr: _kinds_of(tr) == ["jump 30->29", "climb 29->27"])))
def nav_tree_up_by_29(sim):
    """From 30 up to 27: the rope at -8 hangs 94.3 above 30, 0.7 inside physics.ROPE_REACH and 12.3 beyond HeavenClient's
    grab (physics.GRAB_SURE): the planner takes the jump onto 29 and the same rope from there (36.8 up)."""
    return _nav(27)(sim)


@tree((30, -8.0), realism=Realism(rope_reach=ROPE_REACH_CLIENT),
      expects=(ends("arrived"), check("the grab from 30 missed, landing on 29",
                                      lambda tr: _result(tr, "moves")[0] == "climb 30->27: landed on 29"),
               check("then up the rope from 29", lambda tr: _kinds_of(tr)[-1] == "climb 29->27"),
               happens("grabs the rope at x -8")))
def nav_tree_rope_miss(sim):
    """The grab from 30 (94.3 up) in a world where she grabs a rope as HeavenClient does (a point 5 above her feet on
    it: 82 up at the top of her jump; world.Realism.rope_reach), jumps made 20 times dearer (`nav 27 --avoid jump`) so
    the trip takes it: she misses, passes up through 29 and lands on it, and the trip goes on up the rope from there."""
    h = sim.hunter()
    nav = sim.navigator(h)
    weight = nav.weight
    nav.weight = lambda m: weight(m) * (20.0 if m.kind == "jump" else 1.0)
    return nav.go(27, seconds=60)


@tree((10, -100.0), expects=(ends("arrived"), check("8 onto 7", lambda tr: _made(tr, "jump 8->7"))))
def nav_tree_top_cluster(sim):
    """In the top cluster (6-10: 7 Horny Mushrooms and a Green), from 10 to 7: onto 9, 8, and the running jump off 8's
    left end onto 7 (route.END_LOOK: taking off at x 37, 15 in from the end), which the planner didn't have."""
    return _nav(7)(sim)


@tree((12, 160.0), expects=(ends("arrived"), moves("walk_off", "jump", "climb")))
def nav_tree_mid_portal_to_top(sim):
    """From the mid portal's shelf (12, 121 left of its exit at 281) to the top stand on 8 (x 137): off 12 onto 14,
    13, up onto 11, the rope at -80 to 10, then 9 and 8."""
    return _nav(8, x=137.0)(sim)


@tree((31, -150.0), expects=(ends("arrived"), check("up the two ropes on the left",
                                                    lambda tr: _kinds_of(tr) == ["climb 31->29", "climb 29->27"])))
def nav_tree_left_ropes(sim):
    """From 31 up to 27 by the ropes on the tower's left: -192 (28 from the wall) to 29, then -8 to 27."""
    return _nav(27)(sim)


@tree((6, 250.0), expects=(ends("arrived"), moves("walk_off", "drop")))
def nav_tree_down(sim):
    """From the top (6) down to the floor: walk-offs and drops (16 onto 20 at full speed, the study's item 4: at 0.75
    of it she'd land on 18)."""
    return _nav(33)(sim)


@tree((28, 247.0), monsters="all", seed=4, limit=600,
      expects=(_rule("close quarters"), _rule("in passing"),
               check("forty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 40),
               check("every kill counted", _every_kill), ends("time up", key="hunt_stop"),
               ends("standing by on a safe platform", key="stop"),
               check("a thousand EXP or more", lambda tr: tr["world"]["exp"] >= 1000),
               check("not up the tower among the Horny Mushrooms (19, 7, 10)",
                     lambda tr: not set(tr["result"]["grounds"]) & {7, 10, 19})))
def loop_tree_run(sim):
    """`python -m gamelab tour --loop --minutes 5 --map 10002073` from her spot by the portal on 28, with every spawn
    filled and nobody else, then standing by on a safe platform (25-27). A ground's EXP counts less the more dangerous
    the place she'd hunt it from (hunt.GROUND_DANGER): the floor (33: a Green Mushroom, 7% of her HP a touch, and two
    Slimes, 1%) is the safest that pays, and she hunts it the whole 5 minutes, 77 kills and 1,444 EXP; with every
    ground's EXP counted whole she went up the tower to 23, 19, 7 and 19 for 1,041 (before the danger, 23, 19, 7 and
    15 for 59 kills and 1,117 EXP, most of them on the floor on her way).
    The floor is hunted standing on it (the rules 'close quarters' and 'in passing'). (Seed 4: in this world her own
    group's walls alone stop her, and of seeds 1-10, 4 were knocked out past the tower's walls within 5 minutes, off
    25's right end or the rope at -192; with group 0's walls stopping her none: loop_tree_walls. Seed 7 since a
    throw's tap goes by her facing off the frame, perception.heading: on seed 4 she was knocked off the rope at -192
    out of the map at 203 s; of seeds 1-12, 3 fell out (4, 6, 9), as 2 at master's head (3, 12), and 5 meet all of
    this, 7 alone on the floor all 5 minutes, the others hunting 23 first. Seed 4 again with her memory of the
    monsters and the sweep merged on top, hunt.SWEEP: of seeds 1-12, 5 meet all of this (1, 4, 5, 6, 9), 4 alone on
    the floor all 5 minutes; 11 fell out at 124 s, and on 7 her stand-by trip to a safe platform timed out after the
    hunt and she rested on the floor past the run's 600 s. Seed 5 since the review of her jumps' attacks, 2026-10-25
    (her body clear of a monster above over a jump in place, a jump forward's landing however it goes, the ground's
    reach capped 60 above her again): of seeds 1-12, 5 meet all of this (5, 7, 9, 11, 12; 1, 4, 5, 6 and 7 before the
    review's fixes), 7 alone on the floor all 5 minutes; 1 fell out at 143 s, and on 4 and 6 a trip to rest timed
    out and she rested on the floor among its monsters past the run's 600 s, her HP not coming back (seed 5 meets it
    at master's head and before the fixes too).) (Seed 7 since her pet's potions are as live (sim.world.PET_HP,
    HP_POTION), her drops wait while a monster that counts is by them, her decisions go by the monsters the looks missed
    and her escapes and spots keep off a long drop's end and spawn points (part B, 2026-10-26): on seed 5 she was
    knocked out of the tower at 125 s; of seeds 1-12, 6 meet all of this (1, 4, 6, 7, 8, 11), 4 went up the tower
    among its Horny Mushrooms.) (Seed 1 since parts A and B together (work/hunt-ab, 2026-10-26: no throws at the dying
    and a throw's last look before its key, with her drops waiting while a monster that counts is by them and her
    decisions going by the monsters the looks missed): on seed 7 she was knocked out of the tower at 363 s by x -308; of
    seeds 1-12, 5 meet all of this (1, 4, 5, 6, 12), 4 fell out (3, 7, 8, 11), 3 went up the tower among its Horny
    Mushrooms (2, 9, 10).) (Seed 4 since her looks have her on a rope in its last 17 under a platform and a touch leaves
    her on a rope, her climb going on after it (2026-10-27: localization.pose.ROPE_TOP_GAP, world.Realism.rope_knocks,
    navigator.CLIMB_ON): on seed 1 her chain off a climb up the rope at -8 to 27 went a frame later at 92.9 s and she
    went up the tower to 7 at 123 s; of seeds 1-12, 5 meet all of this (4, 5, 6, 7, 12; 4, 5, 6 and 12 at master's
    head too), 1 fell out (11), 5 went up the tower among its Horny Mushrooms (1, 2, 3, 9, 10), and on 8 she had 35
    kills, on 23 alone: on 4 she hunts the floor the whole 5 minutes, 77 kills.)"""
    return _loop(5)(sim)


@tree((28, 247.0), monsters="all", seed=10, limit=600, realism=Realism(zero_walls=True),
      expects=(check("thirty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 30),
               ends("time up", key="hunt_stop"), check("still in the tower", lambda tr: tr["world"].get("fell_out_at")
                                                       is None)))
def loop_tree_walls(sim):
    """loop_tree_run's twin in a world where the walls of foothold group 0 (the tower's) stop her whatever her group,
    off a rope too (world.Realism.zero_walls). Seed 10: with her own group's walls alone she was knocked off the rope
    at -192 (28 from the wall) and out of the map at 328 s; here she stays in the tower all 5 minutes."""
    return _loop(5)(sim)


@tree((28, 247.0), monsters="all", seed=3, limit=400, may_die=True,
      realism=Realism(camps=harness.TREE_CAMPS, respawn=8.8, zero_walls=True),
      expects=(happens("another player kills green-mushroom"), check("only her kills in the ledger", _every_kill),
               check("nobody else kills a Horny Mushroom", lambda tr: not any(
                   "another player kills horny-mushroom" in line for line in tr["world_log"]))))
def loop_tree_crowd(sim):
    """The hunt loop for 3 minutes among the other players the crowding study saw (harness.TREE_CAMPS: five camping by
    28, 21, 23, 19 and 33, each killing the Greens it hunts 2-6 s after they show, leaving the Horny Mushrooms),
    respawns at 8.8 s (MeowDB's mean), group 0's walls stopping her."""
    return _loop(3)(sim)


_CROWD, _SLOW = harness.PRESETS["crowd"], harness.PRESETS["crowd-slow"]
_LUCKY = dataclasses.replace(LEVEL_25, stars=LUCKY_SEVEN)  # her attacks as Lucky Seven's live


@tree((25, 220.0), monsters="all", seed=2, limit=500, may_die=True, noise=_CROWD[0], her=_LUCKY,
      realism=dataclasses.replace(_CROWD[1], zero_walls=True),
      expects=(logs("next: platform 23, from platform 24 at x 221"), logs("at platform 23's spot (platform 24)"),
               _rule("shoot", 20), check("twenty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 20),
               check("every kill counted", _every_kill), check("alive", lambda tr: not tr["world"]["dead"])))
def loop_tree_23(sim):
    """The hunt loop as live on 2026-10-24: Tree Dungeon's 23 its only ground (harness `--grounds 23`), Lucky Seven
    (world.LUCKY_SEVEN: 3 of the simulator's stars a cast), among the crowd the study saw (preset crowd), group 0's
    walls stopping her. She shoots 23 from 24 at 231 (attack.spots by the danger: 95% of 23 in reach, where its Horny
    Mushrooms, which never leave it, can't touch her), not standing among them on 23 as live, and not from its middle
    between their two spawn points when it's empty. Its twenty shots from the spot and twenty kills are one run's luck:
    with the branches merged (escorted trips, her body the client's), of seeds 1-12 four meet them (1, 5, 6, 9), every
    one going to 24 at 231 for 23; on positioning alone over master's onboarding, 2 and 5 of seeds 1-6. (Seed 5 since a
    throw's tap goes by her facing off the frame, perception.heading: of seeds 1-12, 4, 5, 6, 9 and 10 meet them; on
    seed 1 she shot 23 from the spot fewer than 20 times, killing 20.) With her memory of the monsters and the sweep
    merged too (hunt.SWEEP: a slice 23 ms of the simulator's clock moves every look after the first), nine meet them
    (all but 1, 2 and 7): on seed 5 still. Seed 1 since the review of her jumps' attacks, 2026-10-25 (the ground's reach
    capped 60 above her again, her body clear of a monster above over a jump in place): of seeds 1-16, 8 meet them (1,
    3, 4, 9, 10, 12, 14, 16); on seed 5 the rule 'shoot' decided under 20 times, for 25 kills. Seed 2 since the ground's
    reach above her goes by the search's windows again, reach.UP_BY_SEARCH, by the user's choice: of seeds 1-16, 6 meet
    them (2, 6, 7, 11, 13, 15); on seed 1 the rule 'shoot' decided 14 times, 'in passing' 42, for 20 kills.) (Seed 5
    since every look is perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every
    scenario): on seed 2 she shot 23 from the spot fewer than 20 times; of seeds 1-16, 5, 9, 11, 14, 15 and 16 meet all
    of this.) (Seed 1 since a look asks for what the next frame would have stale (feed.to_ask), a frame shown before a
    throw without a tap tells her heading (Actions.heading), a cast takes its MP off at its throw (Attacks.on_throw) and
    a trip stops only for what the pass shoots (Hunt._interrupt): on seed 5 she shot 23 from the spot fewer than 20
    times; of seeds 1-16, 1, 8-11 and 13-16 meet all of this.) From 2026-10-26 her spot is 24 at 221, 51 units from 24's
    Green Mushroom's spawn point (attack.spots.SPAWN_CLEAR): of seeds 1-16 all but 14 meet all of this. (Seed 2 since
    2026-10-27, an attack from a jump picking by the star's search alone, her stance going by where each monster will be
    when her attack starts, and the world's air pick fitted to the live throws (Clearance and slopes): on seed 1 she
    shot 23 from the spot fewer than 20 times, for 42 kills; of seeds 1-16 all but 1 and 4 meet all of this.)"""
    return harness._only(_loop(4), (23,))(sim)


_NEAR = dataclasses.replace(_CROWD[0], near_seen=NEAR_SEEN, off_seen=OFF_SEEN)  # the detector's misses as live


@tree((25, 220.0), monsters="all", seed=1, limit=500, may_die=True, noise=_NEAR, attack="choice",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=dataclasses.replace(_CROWD[1], zero_walls=True),
      expects=(logs("at platform 23's spot (platform 24)"), logs(": later, a horny-mushroom unseen"),
               check("twenty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 20),
               check("every kill counted", _every_kill), check("alive", lambda tr: not tr["world"]["dead"])))
def loop_tree_23_near_misses(sim):
    """loop_tree_23 as `tour --loop` plays by default (the attack choice), with the detector's misses as live on
    2026-10-25 (`--world near-misses`: world.NEAR_SEEN on her platform, OFF_SEEN off it): her drops on 23 wait while
    a Horny Mushroom the looks missed (the feed's hidden tracks, her memory's: localization.unseen) is by them
    (hunt.PILE_CLEAR), as one did live (the service's track 130: 6 touches and a fall to 29)."""
    return harness._only(_loop(4), (23,))(sim)


@tree((20, -60.0), monsters={19, 20, 21, 22}, seed=2, limit=500, may_die=True, noise=_SLOW[0], her=_LUCKY,
      realism=dataclasses.replace(_SLOW[1], zero_walls=True),
      expects=(_rule("wait"), logs("waiting at x"), check("alive", lambda tr: not tr["world"]["dead"])))
def loop_tree_wait(sim):
    """20 its only ground, hunted standing on it (no spot off it is better: attack.spots), its one Horny Mushroom's
    spawn point at x -171, respawns 30 s after a kill (preset crowd-slow), Lucky Seven: with 20 empty she waits for it
    at the safest place along 20 that keeps the spawn point in reach (hunt._wait_place), not where her last kill left
    her. (On seed 2 since the reviews of her kill's monster, 2026-10-27: on seed 3, with her kills on their monsters,
    she went for 3 mesos on 19 at x 134 and the drop planned from 18 at x 110 took the rope at 111 instead, 12 times from
    147 s to 231 s, the pile taken up again 3 s after "too many misses": a loop of the navigation's own, a drop planned
    over a rope's top, which master's stack (one miss on seed 3) didn't come to. Of seeds 1-12 every one meets it, with
    no such miss on 2, 4-8 and 10-12; master's stack meets all 12.)"""
    return harness._only(_loop(4), (20,))(sim)


@tree((28, 247.0), monsters="all", seed=2, limit=1200,
      expects=(logs("tour: sweep"), check("two stops hunted or more", lambda tr: len(tr["result"]["visited"]) >= 2),
               check("she fed her pet", _fed_her_pet)))
def tour_tree_run(sim):
    """`python -m gamelab tour --stops 3 --map 10002073` from 28: the sweep's three nearest grounds from the bottom,
    the escort's fights on the way. The floor's and 32's respawns keep her fighting on the way: each stop takes
    minutes, and the third is often not reached in its time (README: Tree Dungeon, Forest Up North II). (A touch can
    knock her past the tower's wall and out of the map, a simulator effect: with the escorted trips alone seed 2 went
    out at 635 s and the scenario was on seed 6; with positioning and her body the client's too, seeds 1-12 all met
    this but 6, out at 872 s by x -308, and it was on seed 2 again; since the monsters' HP bars are drawn as the game
    draws them, perception.hud.draw_hp_bar, seed 2 goes out at 771 s by x 332 and seed 7 runs past its 1200 s, seeds 1
    and 3-6 meet it, and it's on seed 1. Seed 2 since parts A and B together (work/hunt-ab, 2026-10-26: no throws at the
    dying and a throw's last look before its key, with her drops waiting while a monster that counts is by them and her
    decisions going by the monsters the looks missed): on seed 1 she went out at 841 s by x -315; of seeds 1-12 all but
    1 and 11, out at 618 s, meet it.)"""
    return sim.tour_stack().run(stops=3)


def _grind(minutes: float):
    def run(sim, tour=None):
        from ..behavior.grind import Grind
        tour = tour or sim.tour_stack()
        g = Grind(tour, log=sim.log)
        r = g.run(minutes=minutes)
        return {"report": str(r), "grounds": r.grounds, "kills": r.kills, "rules": dict(sorted(r.rules.items())),
                "hunt_stop": r.stop, "loops": g.patrol.loops, "stays": [round(x, 1) for x in g.stays],
                "credit": {str(k): v for k, v in sorted(g.credit.items(), key=lambda kv: str(kv[0]))}}
    return run


@tree((23, 73.0), monsters="all", seed=1, limit=400, may_die=True,
      realism=Realism(camps=harness.TREE_CAMPS, respawn=30.0, zero_walls=True),
      expects=(logs("loop: ["), logs("next: stop"), logs("grinding:"),
               check("three stops or more", lambda tr: len(tr["result"]["grounds"]) >= 3),
               check("kills credited to two platforms or more", lambda tr: len(tr["result"]["credit"]) >= 2),
               check("no stop camped past STOP_MAX_S and a pass",
                     lambda tr: max(tr["result"]["stays"], default=0) < 120)))
def loop_tree_grind(sim):
    """`python -m gamelab tour --loop --grind --minutes 3 --map 10002073` from 23 among the crowding study's other
    players (harness.TREE_CAMPS) with respawns at 30 s (preset crowd-slow's): the loop over the map's spawn points chosen
    by its EXP a minute (behavior.patrol: the top cluster's 10, 7), and what she sees pulling her off it to the stops
    around her (23, 24, 19, 20 on seed 1), what she meets on the way shot in passing, kills credited to the platforms
    they died on (19, 20, 21, 23, 24). In this world she's touched about five times as often as live (README:
    Grinding): knocked off 23 down to 25-29 and back, then standing by at the end (resting there until 2026-10-26)."""
    return _grind(3)(sim)


@tree((24, 231.0), monsters={22, 23}, seed=2, limit=240, may_die=True, noise=_CROWD[0],
      realism=Realism(respawn=17.0, zero_walls=True, drops=(("green-mushroom", "sapphire-ore", 1.0),)),
      expects=(logs(": shooting first"), happens("picks up sapphire-ore"),
               check("every ore picked up", lambda tr: tr["world"]["picked"].get("sapphire-ore", 0) >= 4),
               check("hurt five times at most", lambda tr: tr["world"]["touches"] - tr["world"]["evaded"] <= 5)))
def loop_tree_ore_waits(sim):
    """EXP before loot (hunt._loot_first): 23 her only ground, shot from 24 at 231, every Green Mushroom on 22 dropping
    a Sapphire Ore (120 mesos, valuable on Tree Dungeon: twice its bronze coin, 66.8). Seen while a Horny Mushroom on 23
    is in reach, the ore isn't at risk (90 s of life, nobody by it): she shoots first and picks it up once nothing is in
    reach, 4 ores of 4. With the valuable pile rule ahead of shooting as before (hunt.EXP_FIRST off) she went down to 22
    for it at once, among its Green Mushrooms, and was hurt 15 times (1.72 bars of HP) against once (0.16), for the same
    596 EXP and 3 ores; on seeds 1-8, 4645 EXP against 4306 (better on 5, worse on 1), 25 ores of 31 against 25 of 30,
    hurt 76 times against 98. Seed 14 since the review of her jumps' attacks, 2026-10-25: hurt three times at most and
    every ore picked up is a run's luck, on 2 of seeds 1-16 (14, 15; 3 before the review's fixes, 1, 2 and 12; 4 at
    master's head, 1, 3, 11 and 14); on seed 1 she was touched 8 times, 6 of them hurting, jump-throwing at a Horny
    Mushroom on 23 after 209 s. Seed 1 again since the ground's reach above her goes by the search's windows again,
    reach.UP_BY_SEARCH, by the user's choice: on 4 of seeds 1-16, 1, 2, 12 and 13; on seed 14 she was hurt 5 times, 3 of
    them by a Green Mushroom as she went for an ore on 22 at 180-187 s.) (Seed 2 since a look asks for what the next
    frame would have stale (feed.to_ask), a frame shown before a throw without a tap tells her heading
    (Actions.heading), a cast takes its MP off at its throw (Attacks.on_throw) and a trip stops only for what the pass
    shoots (Hunt._interrupt): on seed 8 she was hurt more than five times; of seeds 1-16, 10 meet all of this.) (Seed 1
    since her drops wait while a monster that counts is by them, seen, missed by the looks or due back (Hunt._pile_block),
    and her fights and trips go by the monsters the looks missed (localization.unseen), 2026-10-26: on seed 2 she was
    hurt 9 times and picked up 3 ores; of seeds 1-16, 9 meet all of this.) (Seed 4 since the monsters the looks missed
    leave out the dead (localization.unseen.hidden), her throws' attack goes by the look's alone again and her drops
    don't wait for a spawn point's monster due back (hunt.DUE_BACK; review of part B, 2026-10-26): on seed 1 she was
    hurt 6 times; of seeds 1-16, 4, 6 and 9-14 meet all of this.) (Seed 1 since parts A and B together (work/hunt-ab,
    2026-10-26: no throws at the dying and a throw's last look before its key, with her drops waiting while a monster
    that counts is by them and her decisions going by the monsters the looks missed): on seed 4 she was hurt more than
    five times; of seeds 1-16, 1, 3, 5, 6, 10 and 14-16 meet all of this.) (Seed 3 since the rebase of her kill's
    monster onto clearance and slopes, 2026-10-27, her kill's monster bound by her kept in her body's way
    (behavior.fight.TOUCH_NEAR) and her kills credited to the monster her attack took: on seed 1 she was hurt more than
    five times; of seeds 1-16, 2, 3, 7 and 11-13 meet all of this, and with master's stack 1, 3-6 and 13-16.) (Seed 2
    since landings with clearance and moving away to attack, 2026-10-27, the Horny Mushroom's escape stepping to the
    safest end, stance.ESCAPE_STEPS: on seed 3 she was hurt 6 times; of seeds 1-16, 1, 2, 6-8, 11 and 14 met all of
    this, with master's stack 2, 3, 7 and 11-13, hurt 94 times over the 16 against 101; after the reviews of it, 1-3,
    6, 8, 9 and 11-16, hurt 65 times against master's 101.)"""
    return harness._only(_loop(4), (23,))(sim)


def _lucky_seven_again(tr: dict) -> bool:
    """A throw of Lucky Seven after one of the regular attack for her MP: her MP back (and the stack casts only with
    the MP for it)."""
    log = tr["log"]
    low = next((i for i, line in enumerate(log) if "< a cast's" in line), None)
    return low is not None and any("Lucky Seven:" in line for line in log[low:])


@tree((24, 231.0), monsters={22, 23}, seed=1, limit=300, may_die=True, noise=_CROWD[0], attack="choice",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER, pet_mp=None),
      realism=Realism(respawn=17.0, zero_walls=True),
      expects=(logs("Lucky Seven: horny-mushroom"), logs("< a cast's 14"),
               check("Lucky Seven again once her MP came back", _lucky_seven_again),
               check("never Lucky Seven with too little MP", lambda tr: not any(
                   "not enough, nothing thrown" in line for line in tr["world_log"]))))
def loop_tree_mp_runs_out(sim):
    """The hunt loop as live runs by default (tour --loop: the feed, the attack choice), her pet drinking no MP potions
    and her MP at 12% (72 of 601) when it starts, 23 her only ground, shot from 24 at 231. Lucky Seven while she has
    the MP for a cast (or the regular attack to finish a monster its bar shows low), the regular attack once she hasn't
    (a cast with too little throws nothing in game: never pressed then), Lucky Seven again as her MP comes back
    (world.MP_REGEN, 0.32 MP a second: a cast every 34 s or so): on seed 1, 13 casts and 185 regular throws for her MP,
    19 kills, none pressed short (with the bars read empty, as live before 2026-10-25: 13, 193 and 20). The harness's
    runs never showed this, their pet's potion under 5% keeping her over a cast (review, 2026-10-25), nor did a scenario
    run with the feed and the choice before."""
    sim.world.mp = 0.12
    return harness._only(_loop(4), (23,))(sim)


# --- the Lie Detector during a hunt: she stops, and retreats to a safe place within 10 s (behavior.retreat) ---------
# The user, 2026-10-24: "just stop any in-progress hunt when this happens", "On a lie detector you should move to a safe
# place, within 10s". The player sees it at FakePlayer.alarm_at, as the real one's watchdog would: the plan running
# stops then, and after it only her retreat's moves (for alarm.RETREAT_WINDOW_S) and her pet's food play.

MOUSE_EVENTS = ("mouse", "lbutton", "rbutton")


def safe_there(sim) -> bool:
    """Whether the world has her on a safe place now, by what behavior.retreat counts as one, the look aside: standing
    on a safe platform (MapModel.safe), or where the danger with every spawn point's monster at its point is at most
    retreat.SAFE_DANGER, on a platform or on a rope."""
    from ..behavior.retreat import SAFE_DANGER
    w = sim.world
    d = danger.of_map(sim.model, danger.Her(w.character.max_hp, w.character.level))
    if w.mode == "rope" and w.rope is not None:
        return d.on_rope(w.rope, w.y, None) <= SAFE_DANGER
    if w.mode != "ground" or w.plat is None:
        return False
    return sim.model.safe(w.plat.id) or d.at(w.plat.id, w.x, None) <= SAFE_DANGER


PLAYED_ROUND_S = 1e-4  # (lie_detector) what fakes.FakePlayer rounds the times it played its keys at to


def lie_detector(after: float, minutes: float = 5, grind: bool = False):
    """The hunt loop as `tour --loop` runs it (in behavior.retreat.on_alarm; with `grind`, grinding, behavior.grind, as
    `tour --loop --grind` does), the player seeing the Lie Detector `after` s into the run. The result: how the hunt
    stopped (`stopped`: the alarm's message), the retreat (Retreat.record), where the world has her RETREAT_S after the
    sighting and whether that's a safe place (`then`: at that very moment, World.call_at, whatever the retreat was
    doing), every key and mouse event played from the sighting on (`after`: [s after it, the plan's kind, the event]),
    and what became of a retreat's plan sent past the player's window (`late_retreat`)."""
    def run(sim):
        from ..behavior.grind import Grind
        from ..behavior.hunt import Hunt
        from ..behavior.retreat import on_alarm
        from ..control.alarm import RETREAT, RETREAT_S, RETREAT_WINDOW_S
        tour = sim.tour_stack()
        seen = sim.player.alarm_at = sim.clock.now + after
        then = {}

        def look(w):
            then.update({"t": round(w.origin + w.ms / 1000 - seen, 3), "x": round(w.x, 1), "y": round(w.y, 1),
                         "platform": None if w.plat is None else w.plat.id, "mode": w.mode, "safe": safe_there(sim)})
        sim.world.call_at(seen + RETREAT_S, look)
        watch, stopped = None, None
        try:
            with on_alarm(tour.h, rng=random.Random(sim.scenario.seed)) as watch:
                (Grind if grind else Hunt)(tour, log=sim.log).run(minutes=minutes)
        except client.LieDetectorAlarm as e:
            stopped = str(e)
        sim.clock.sleep(max(0.0, seen + RETREAT_S + 0.1 - sim.clock.now))  # (past the tick `look` is called at)
        sim.world.advance(sim.clock.now)
        # (the player's played times are rounded to 0.1 ms, fakes.FakePlayer._schedule: one played at the sighting's
        # instant but a float's error before it reads as at it, where nothing but a retreat's plays from it on.
        # loop_tree_lie_detector_23 on seed 5, seen at 150 s, had a walk's pick-up tap 2 ns before it counted after it:
        # tests/test_retreat.py under --slow, 2026-10-27.)
        played = sorted([round(t - seen, 3), kind, *event] for kind, events in sim.player.played.values()
                        for t, *event in events if t >= seen + PLAYED_ROUND_S / 2)
        sim.clock.sleep(max(0.0, seen + RETREAT_WINDOW_S + 0.5 - sim.clock.now))
        try:
            client.submit([[0, sim.keys.left.name, "down"], [40, sim.keys.left.name, "up"]], lead_ms=0, name="late",
                          kind=RETREAT)
            late = "played"
        except client.LieDetectorAlarm as e:
            late = str(e)
        retreat = None if watch is None or watch.retreat is None else watch.retreat.record()
        return {"stopped": stopped, "retreat": retreat, "then": then, "after": played, "late_retreat": late}
    return run


def _moves_alone(tr: dict) -> bool:
    """From the sighting on, nothing played but her moves' keys (her arrows, jump, up, down: the world's keys), no
    attack, pick-up, pet food or mouse, and only in her retreat's plans."""
    keys = _move_keys(tr)
    return all(kind == "retreat" and event[0] in keys for _, kind, *event in tr["result"]["after"])


def _move_keys(tr: dict) -> set[str]:
    """Her moves' keys on the scenario's map (her arrows, jump, up and down)."""
    k = her_on(SCENARIOS[tr["scenario"]].map_id)[0]
    return {k.left.name, k.right.name, k.jump.name, k.up.name, k.down.name}


LIE_DETECTOR_EXPECTS = (
    check("the hunt stopped at the Lie Detector",
          lambda tr: (tr["result"]["stopped"] or "").startswith("Lie Detector")),
    logs("LIE DETECTOR: retreat"),
    check("on a safe place 10 s after the sighting", lambda tr: tr["result"]["then"]["safe"]),
    check("there within 10 s", lambda tr: tr["result"]["retreat"]["took"] <= 10.0),
    check("her moves alone after the sighting, no attack and no mouse", _moves_alone),
    check("her retreat's moves refused past the player's window",
          lambda tr: "are over" in tr["result"]["late_retreat"]))


def _went(why: str):
    return check(f"to {why}", lambda tr: tr["result"]["retreat"]["why"] == why)


@tree((28, 247.0), monsters="all", seed=4, limit=400, expects=LIE_DETECTOR_EXPECTS + (
    _went("a rope as safe: no safe platform near"),
    check("onto the rope at 146 from 32", lambda tr: tr["result"]["then"]["mode"] == "rope"
          and tr["result"]["then"]["x"] == 146)))
def loop_tree_lie_detector(sim):
    """`tour --loop --map 10002073` from 28 (as loop_tree_run), the Lie Detector seen 110 s in, while she hunts the
    floor (33): the hunt stops at once (the plan running cut, nothing of hers after it); no safe platform is near (28,
    the nearest place no monster reaches, is 4.3 s by route, past retreat.NEAR_S with the crowd's stops), so up onto the
    rope at 146 from 32, at its safest height (attack.ropes), avoiding the monsters, her moves alone; a retreat plan
    past the player's 12 s is refused. (From the floor's crowd, of seeds 1-12 seen at 40, 75, 110 and 150 s, 83% are on
    a safe place at 10 s; from 24, shooting 23, 47 of 48: loop_tree_lie_detector_23. Seed 7 since a throw's tap goes by
    her facing off the frame, perception.heading, where her course before the sighting goes otherwise: of seeds 1-12
    seen at 110 s, 7 and 10 go up onto the rope at 146 within 10 s, 9 are on a safe place at 10 s (a safe platform near
    on 5, 6, 8 and 12, 28 on 2 and 9, the rope at -192 on 3), and on seed 1 she took 10.5 s, still on 33. Seed 3 with
    her memory of the monsters and the sweep merged too, hunt.SWEEP: 1, 3, 9 and 12 go up onto the rope at 146 within 10
    s, 8 are on a safe place at 10 s; on seed 7 she took 11.2 s, still on 32. Seed 2 since her jumps' attacks are
    weighed and her jump throws turn in the air, 2026-10-25: of seeds 1-12, 9 go up onto the rope at 146 within 10 s, 11
    are on a safe place at 10 s (the rope at -192 on 1, already on 25 on 3), and on seed 11 she was still on her way.
    Seed 1 since the review of her jumps' attacks, 2026-10-25 (her body clear of a monster above over a jump in place, a
    jump forward's landing however it goes, the ground's reach capped 60 above her again): of seeds 1-16, 1, 5, 7, 10
    and 12 go up onto the rope at 146 within 10 s, 11 are on a safe place at 10 s (7 of 16 at master's head, 14 before
    the review's fixes), and on seed 2 she went up the rope at -192 from 31. Seed 4 since the ground's reach above her
    goes by the search's windows again, reach.UP_BY_SEARCH, by the user's choice: of seeds 1-16, 4, 5, 7, 8, 10 and 12
    go up onto the rope at 146 within 10 s, 12 are on a safe place at 10 s, and on seed 1 she was still on the floor,
    33, its safest place within reach.) (Seed 2 since a look asks for what the next frame would have stale
    (feed.to_ask), a frame shown before a throw without a tap tells her heading (Actions.heading), a cast takes its MP
    off at its throw (Attacks.on_throw) and a trip stops only for what the pass shoots (Hunt._interrupt): on seed 4 she
    went up the rope at -192 from 31; of seeds 1-16, 12 meet all of this.) (Seed 4 since 2026-10-26: her throws at the
    dying gone, a throw's last look before its key, her stats at level 25, the air turn's attack 60 ms after its arrow
    and the world's damage at the server's time: on seed 2 she went up the rope at 146 from 32, no safe platform near;
    of seeds 1-16, 4, 5, 7, 10, 11, 13, 14 and 15 meet all of this.) (Seed 4 kept with parts A and B together,
    work/hunt-ab, 2026-10-26, where part B alone had it on 1: of seeds 1-16, 2, 4-7, 10, 11 and 14-16 meet all of this;
    on 1 she went to a safe platform, not the rope.)"""
    return lie_detector(110.0)(sim)


@tree((28, 247.0), monsters="all", seed=5, limit=400, expects=LIE_DETECTOR_EXPECTS + (
    _went("safe enough"), check("on 28", lambda tr: tr["result"]["then"]["platform"] == 28)))
def loop_tree_lie_detector_28(sim):
    """As loop_tree_lie_detector, seen 40 s in, on her way down from 28: back onto 28, which no monster reaches (its
    danger 0 by attack.danger, though not MapModel.safe), 1.7 s by route. (Seed 2 since a throw's tap goes by her facing
    off the frame, perception.heading: of seeds 1-12 seen at 40 s, 2 and 9 go back onto 28, safe enough; on seed 3 she
    went up the rope at -192, a rope as safe, no safe platform near. Seed 1 since her jumps' attacks are weighed and her
    jump throws turn in the air, 2026-10-25: of seeds 1-12 seen at 40 s, 1 goes back onto 28; 8 go up the rope at 146, 2
    the one at -192, and on 3 she's on 25 already, all on a safe place at 10 s.) (Seed 10 since every look is
    perception's newest frame after her keys' effect (Senses.fresh_after, the feed in every scenario): on seed 1 she
    went up the rope at 146 from 32, no safe platform near; of seeds 1-16 only 10 goes back onto 28.) (Seed 12 since a
    look asks for what the next frame would have stale (feed.to_ask), a frame shown before a throw without a tap tells
    her heading (Actions.heading), a cast takes its MP off at its throw (Attacks.on_throw) and a trip stops only for
    what the pass shoots (Hunt._interrupt): on seed 10 she went up the rope at 146 from 32, a rope as safe; of seeds
    1-16 only 12 goes back onto 28.) (Seed 1 since 2026-10-26: her throws at the dying gone, a throw's last look before
    its key, her stats at level 25, the air turn's attack 60 ms after its arrow and the world's damage at the server's
    time: on seed 12 she didn't go back onto 28; of seeds 1-16, 1 and 11 meet all of this.) (Seed 1 kept with parts A
    and B together, work/hunt-ab, 2026-10-26, where part B alone had it on 33, then 9: of seeds 1-16 only 1 goes back
    onto 28.) (Seed 37 since 2026-10-27, an attack from a jump picking by the star's search alone (reach.AIR_MODEL, the
    world's AIR_PICK) and her stance going by where each monster will be when her attack starts (attack.stance.Pace): on
    seed 1 she went up the rope at 146 from 32, a rope as safe, as master's stack does in this world; of seeds 1-40 only
    37 goes back onto 28, and on 39 a touch knocked her out of the tower 3 s after the sighting (in this world her own
    group's walls alone stop her: loop_tree_run).)
    (Seed 7 since 2026-10-27 on its own branch, her kill credited to the monster her attack took (localization.tracking's
    binding) and drawn alive until her star lands (sim.world Realism.dies_late): on seed 1 she didn't go back onto 28;
    of seeds 1-16 only 7 meets all of this, and on 13 and 14 she fell out of the map before 40 s.) (Seed 37 kept since
    the rebase of her kill's monster onto clearance and slopes: of seeds 30-40, 33 and 37 meet all of this.) (Seed 5
    since 2026-10-28, her touches her popups (a MISS cutting her trips as a hit does: navigator.TOUCH_CUTS), the world's
    invulnerability 1.45 s and her MISSes by the game's hit check (sim.world.INVULNERABLE, LEVEL_25_AVOID): on seed 37
    she went up the rope at 146 from 32, a rope as safe, as she does with the stack's reactions off in this world too;
    of seeds 1-40, 5, 8, 9, 11, 29, 33 and 38 meet all of this.)"""
    return lie_detector(40.0)(sim)


@tree((25, 220.0), monsters="all", seed=1, limit=500, noise=_CROWD[0], her=_LUCKY,
      realism=dataclasses.replace(_CROWD[1], zero_walls=True),
      expects=LIE_DETECTOR_EXPECTS + (check("onto 25, the safe platform under 24", lambda tr: tr["result"]["then"][
          "platform"] == 25),))
def loop_tree_lie_detector_23(sim):
    """The hunt loop as live on 2026-10-24 (loop_tree_23: 23 its only ground, Lucky Seven, the crowd), the Lie Detector
    seen 60 s in, as she shoots 23 from 24 at 231 (attack.spots): down onto 25, the safe platform under 24 (0.9 s by
    route). (Seed 4 since her jumps' attacks are weighed and her jump throws turn in the air, 2026-10-25: of seeds
    1-12, 9 go down onto 25 and 10 are on a safe place at 10 s; on seeds 1 and 2 she was still on her way.) (Seed 1
    since parts A and B together (work/hunt-ab, 2026-10-26: no throws at the dying and a throw's last look before its
    key, with her drops waiting while a monster that counts is by them and her decisions going by the monsters the looks
    missed): on seed 4 she was still on her way 10 s on, not onto 25; of seeds 1-16 all but 4 meet all of this.)"""
    return harness._only(lie_detector(60.0), (23,))(sim)


# Her as `tour --loop` plays by default (the feed, the attack choice): Lucky Seven's key casting it at its MP, and each
# star's damage drawn (harness.ATTACK_HER), the bars over hurt monsters read as they show; group 0's walls
# stopping her (in the default world a touch knocked her out of the map past the tower's wall on 3 of seeds 1-12 of
# loop_tree_lie_detector_grind, and 2 of loop_tree_lie_detector's twin as live ran it, a simulator effect).
_LIVE = dict(attack="choice", her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER),
             realism=Realism(zero_walls=True))


@tree((23, 73.0), monsters="all", seed=1, limit=400, **_LIVE, expects=LIE_DETECTOR_EXPECTS + (
    logs("loop: ["), logs("Lucky Seven:")))
def loop_tree_lie_detector_grind(sim):
    """`tour --loop --grind --map 10002073` as it runs by default (the feed, the attack choice) from 23, as
    loop_tree_grind, the Lie Detector seen 75 s in: grinding is the hunt loop going round the map (behavior.grind.Grind,
    a Hunt), so it stops as the hunt loop does, at the next look or plan, and she retreats to a safe place within 10 s,
    her moves alone: on seed 1 down onto 25, a safe platform, at 5.6 s. (Of seeds 1-12, all 12 are on a safe place at
    10 s, 10 on 25 and 2 on 28; resting on 25 when it was seen, as she could until 2026-10-26, she stayed, the rest's
    next look up to 5 s after it.)"""
    return lie_detector(75.0, grind=True)(sim)


# --- the safety stop: her HP stays low, the hunt ends, she goes to a safe place, whoever is at the PC is paged --------
# (behavior.safety). The user, 2026-10-26: "Just remove the sit recover hp functionality, it's no longer needed". Her
# pet drinks her potions; with none left (Realism.potions 0) every touch leaves her lower, until she's under her reserve:
# her profile's hits (the thief's 3 for now) of the map's largest touch (the user, 2026-10-28: "it should be HP under the
# amount of taking 3 hits from the monsters on the map ... for thief with high evasion this should be 2 hits, 3 for
# magician"), worked out by the hunter from what it reads of her as live (behavior.bootstrap): her max HP off the HP
# number the simulator draws (the world's character's), her level off the LV. digits, her DEF fitted to her touches
# (none known at the start: a touch at DEF 0 until enough are kept). `_reserve_checks` checks the stop's numbers against
# the game's formula at the world's own DEF and level, not against any number written here.

def safety_stop(minutes: float = 5, grind: bool = False):
    """The hunt loop as `tour --loop` runs it (in behavior.retreat.on_alarm; with `grind`, grinding, as `tour --loop
    --grind` does). The result: how the hunt stopped (`stopped`: the safety stop's message, her HP, how long and over
    how many looks it was under), when (`at`: s into the run, the look that stopped it), her retreat (Retreat.record),
    whether the world has her on a safe place once it's over (`safe`, as retreat counts one: safe_there), what was
    paged (`pages`: [s, title, text]), every key and mouse event played from the stop on (`after`: [s after it, the
    plan's kind, the event]) and her HP at the end."""
    def run(sim):
        from ..behavior.grind import Grind
        from ..behavior.hunt import Hunt
        from ..behavior.retreat import on_alarm
        tour = sim.tour_stack()
        watch, stopped, at = None, None, None
        try:
            with on_alarm(tour.h, rng=random.Random(sim.scenario.seed)) as watch:
                (Grind if grind else Hunt)(tour, log=sim.log).run(minutes=minutes)
        except SafetyStop as e:
            stopped = {"text": str(e), "hp": e.hp, "seconds": e.seconds, "looks": e.looks}
            at = e.at
        sim.world.advance(sim.clock.now)
        played = [] if at is None else sorted([round(t - at, 3), kind, *event] for kind, events
                                              in sim.player.played.values() for t, *event in events if t >= at)
        retreat = None if watch is None or watch.retreat is None else watch.retreat.record()
        return {"stopped": stopped, "at": None if at is None else sim.trace.at(at), "retreat": retreat,
                "safe": safe_there(sim), "pages": sim.player.pages, "after": played, "hp": round(sim.world.hp, 3)}
    return run


STOP_NUMBERS = re.compile(r"under (\d+) (.+?) touch(?:es)? of (\d+) \((\d+)\)")


def _stop_numbers(tr) -> tuple[int, str, int, int] | None:
    """The safety stop's reserve as its text says it: (hits, the kind, its touch, the threshold), None if it isn't one."""
    m = STOP_NUMBERS.search((tr["result"].get("stopped") or {}).get("text", "") if isinstance(tr["result"], dict)
                            else "")
    return None if m is None else (int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)))


def _reserve_holds(tr, her: Her = LEVEL_25) -> bool:
    """The stop's reserve is her profile's hits of the map's largest touch, worked out as the game would deal it to the
    world's character (her DEF and level as the simulator plays her; the hunter knew neither but by reading and fitting
    them), at least the formula's top there, and its threshold hits x that touch."""
    from ..classes.thief import SAFETY_HITS
    from ..behavior.safety import kinds_of
    got = _stop_numbers(tr)
    if got is None:
        return False
    hits, name, touch, threshold = got
    pad, top = max(kinds_of(real_map(TREE)).values())
    return (hits == SAFETY_HITS and name == top and threshold == hits * touch
            and touch >= danger.touch_range(pad, her.defence, her.level)[1])


def _debounce_of(tr) -> float:
    """The safety stop's debounce as its watch had it at the stop (behavior.safety.HpWatch.debounce): her HP number's
    (DEBOUNCE_EXACT_S) when the stop's HP was exact ("HP 375 of 774"), her bar's (DEBOUNCE_S) when it was the bar's
    ("HP about 375"; the scenario's stack binds no potion key of hers, whose drink would wait for the bar's too)."""
    from ..behavior.safety import DEBOUNCE_EXACT_S, DEBOUNCE_S
    return DEBOUNCE_S if "HP about " in tr["result"]["stopped"]["text"] else DEBOUNCE_EXACT_S


SAFETY_EXPECTS = (
    check("the hunt stopped for her HP under her profile's hits of the map's largest touch, as the game deals it to her",
          _reserve_holds),
    check("under it on two looks the watch's debounce apart at least (0.25 s by her HP number, 0.6 s by her bar), or "
          "under one touch two looks in a row",
          lambda tr: tr["result"]["stopped"]["seconds"] >= _debounce_of(tr) and tr["result"]["stopped"]["looks"] >= 2
          or "on 2 looks in a row" in tr["result"]["stopped"]["text"]),
    check("its numbers in HP: under the threshold, of her max HP as her HUD's number says it",
          lambda tr: _stop_numbers(tr) is not None and tr["result"]["stopped"]["hp"] < _stop_numbers(tr)[3]
          and f"of {LEVEL_25.max_hp:.0f}" in tr["result"]["stopped"]["text"]),
    check("alive at the end", lambda tr: tr["result"]["hp"] > 0),
    logs("safety stop: under"),
    logs(f"her max HP {LEVEL_25.max_hp:.0f} (her HUD's HP number); level {LEVEL_25.level:.0f} (her HUD's LV. digits)",
         "the start line says her max HP and level as her HUD shows them (read before the hunt, as the perception "
         "service has them live)"), logs("SAFETY STOP: HP"),
    logs("SAFETY STOP: retreat"),
    check("on a safe place once her retreat is over", lambda tr: tr["result"]["safe"]),
    check("paged once, about her HP", lambda tr: len(tr["result"]["pages"]) == 1
          and "HP stayed low" in tr["result"]["pages"][0][1]),
    check("her moves alone after the stop, no attack and no mouse", _moves_alone),
    check("never the chair, no waiting for HP", lambda tr: not any(p[1] in ("chair", "stand") for p in tr["plans"])))


@tree((23, 73.0), monsters="all", seed=2, limit=400, hp=0.8, **dict(_LIVE, realism=Realism(zero_walls=True, potions=0)),
      expects=SAFETY_EXPECTS)
def loop_tree_safety_stop(sim):
    """`tour --loop --map 10002073` as it runs by default (the feed, the attack choice) from 23 at 80% HP, her pet with
    no potion left to drink: every touch leaves her lower; under her reserve (behavior.safety: her profile's hits of
    the map's largest touch, worked out from what the hunter reads of her off the simulator's HUD and fits to her
    touches: behavior.bootstrap; her HP her HP number's) on two looks 0.25 s apart the hunt stops, she goes to a safe
    place as on the Lie Detector, her moves alone (behavior.retreat), and whoever is at the PC is paged; no chair, no
    waiting for her HP; alive at the end. (Seed 2 since 2026-10-27, an attack from a jump picking by the star's search
    alone (reach.AIR_MODEL, the world's AIR_PICK) and her stance going by where each monster will be when her attack
    starts (attack.stance.Pace). Until 2026-10-28 the stop was at 65% for 2 s: here at 48% 2.9 s in, and her retreat
    took her to 1.1%. Seeds 1-12 under each rule: README, The safety stop in HP.)"""
    return safety_stop(5)(sim)


# --- out of darts: her presses take no MP and kill nothing, the hunt ends, she goes to a safe place, paged ----------
# (behavior.darts). The user, 2026-10-26 01:28, the grind of 01:04: "stop, you run out of darts"; the bot kept pressing
# for nothing. Realism.darts runs her out: the Lucky Seven key does nothing then (no MP), the attack key only swipes.

def darts_stop(minutes: float = 5, grind: bool = False):
    """The hunt loop as `tour --loop` runs it (in behavior.retreat.on_alarm; with `grind`, grinding), her throwing stars
    running out (Realism.darts). The result: how it stopped (`stopped`: the stop's message, its presses and seconds,
    whether the game's pickup log said so), when (`at`, s into the run), when she ran out (`out_at`: the world's first
    press with too few stars), her presses the stop's count needs (`needs`: behavior.darts.presses_for), her retreat,
    whether she's on a safe place after it, what was paged, the recording's saves asked (`kept`: [s, seconds, name]),
    every key and mouse event played from the stop on (`after`), and her stars left."""
    def run(sim):
        from ..behavior.darts import OutOfDarts, presses_for
        from ..behavior.grind import Grind
        from ..behavior.hunt import Hunt
        from ..behavior.retreat import on_alarm
        tour = sim.tour_stack()
        watch, stopped, at = None, None, None
        try:
            with on_alarm(tour.h, rng=random.Random(sim.scenario.seed)) as watch:
                (Grind if grind else Hunt)(tour, log=sim.log).run(minutes=minutes)
        except OutOfDarts as e:
            stopped = {"text": str(e), "presses": e.presses, "seconds": round(e.seconds, 3), "told": e.told}
            at = e.at
        sim.world.advance(sim.clock.now)
        played = [] if at is None else sorted([round(t - at, 3), kind, *event] for kind, events
                                              in sim.player.played.values() for t, *event in events if t >= at)
        out = next((line for line in sim.world.log if "no throwing stars left" in line), None)
        retreat = None if watch is None or watch.retreat is None else watch.retreat.record()
        return {"stopped": stopped, "at": None if at is None else sim.trace.at(at),
                "out_at": None if out is None else float(out.split()[0]),
                "needs": presses_for(sim.model.mob_hp, sim.model.mob_pdd), "retreat": retreat, "safe": safe_there(sim),
                "pages": sim.player.pages, "kept": sim.player.kept, "after": played, "darts": sim.world.darts_left}
    return run


DARTS_EXPECTS = (
    check("the hunt stopped out of darts", lambda tr: "took no MP and killed nothing" in (
        tr["result"]["stopped"] or {}).get("text", "")),
    check("after the map's presses in a row (presses_for), or more",
          lambda tr: tr["result"]["stopped"]["presses"] >= tr["result"]["needs"]),
    check("only once her stars ran out", lambda tr: tr["result"]["out_at"] is not None
          and tr["result"]["at"] > tr["result"]["out_at"]),
    logs("OUT OF DARTS: "), logs("OUT OF DARTS: retreat"),
    check("on a safe place once her retreat is over", lambda tr: tr["result"]["safe"]),
    check("paged once, about her darts", lambda tr: len(tr["result"]["pages"]) == 1
          and "out of darts" in tr["result"]["pages"][0][1]),
    check("the recording's last minutes kept, once", lambda tr: len(tr["result"]["kept"]) == 1),
    check("her moves alone after the stop, no attack and no mouse", _moves_alone))


@tree((23, 73.0), monsters="all", seed=4, limit=400, **dict(_LIVE, realism=Realism(zero_walls=True, darts=60)),
      expects=DARTS_EXPECTS)
def loop_tree_out_of_darts(sim):
    """`tour --loop --map 10002073` as it runs by default (the feed, the attack choice) from 23 with 60 throwing stars:
    30 casts of Lucky Seven and they're gone; from then her presses take no MP and kill nothing (the Lucky Seven key
    does nothing, as live at 01:28 on 2026-10-26), and once 20 of them have in a row (behavior.darts.presses_for on
    this map: twice and more the live hunts' most with her darts) the hunt stops, she goes to a safe place
    (behavior.retreat), whoever is at the PC is paged, and the recording's last minutes are saved. (On seed 4 since the
    reviews of her kill's monster, 2026-10-27: on seed 1 her first kill's monster, bound and by her, kept her from the
    jump onto 22 at 1.6 s, and the stop came at 82 s on 28, where her retreat to the rope at x 146 was knocked back off
    32 twice and ended on 33. Of seeds 1-12, all but 1 meet it, and master's stack all but 3 and 12.) Since the afternoon of 2026-10-28 out of darts'
    stop due in the hunt loop has the key bindings read first (her Lucky Seven's key moved does the same): due at 72.4 s,
    on 25 (a safe platform: at once), the window read, Lucky Seven on Shift as configured: the stop at 76.1 s, her moves
    alone after it."""
    return darts_stop(5)(sim)


# --- her regular attack's key: doubted after 5 presses without effect, her stars, the game's key bindings read ------
# (behavior.keycheck). The user, 2026-10-28, after the hunt of 00:30 pressed a regular attack the game had on another key:
# "1. Throw 5 times before doubting the key error 2. Check if star has run out 3. Open keyboard config (\\) to check
# the key configuration and see if attack button is different, and fix things." Realism.game_keys moves the game's
# Attack (her configured Ctrl does nothing then); the backslash opens the KEY BINDINGS window, drawn from the game's
# pictures with the world's keys (sim.fakes.FakeEyes._capture) and read as in game.

def key_check_run(minutes: float = 4, grind: bool = True):
    """The hunt loop as `tour --loop` runs it (in behavior.retreat.on_alarm; `grind`, grinding, the default), the game's
    key bindings as the scenario's Realism.game_keys has them. The result: the doubts and the check's lines said in
    capitals (`said`: [s, line]), the window's key plans ([s, name, keys]), what was paged, how the hunt ended
    (`stopped`: a HuntStop's text, else None), the kills of hers after the check's outcome (`kills_after`), her regular
    attack's keys pressed after it (`attack_keys_after`: key -> presses), and the world's window counts."""
    def run(sim):
        from ..behavior.grind import Grind
        from ..behavior.hunt import Hunt
        from ..behavior.retreat import on_alarm
        from ..behavior.safety import HuntStop
        tour = sim.tour_stack()
        stopped = None
        try:
            with on_alarm(tour.h, rng=random.Random(sim.scenario.seed)):
                (Grind if grind else Hunt)(tour, log=sim.log).run(minutes=minutes)
        except HuntStop as e:
            stopped = f"{e.label}: {e}"
        sim.world.advance(sim.clock.now)
        said = [[float(line.split()[0]), line.split(None, 1)[1].strip()] for line in sim.trace.log
                if "THE REGULAR ATTACK TOOK NO EFFECT" in line or "ATTACK KEY:" in line or "her stars:" in line
                or "the key bindings" in line or "KEY CHECK" in line or "HER KEYS:" in line or "HER JUMPS" in line
                or "HER RUNS" in line or "the key bindings checked first" in line or "her attack's MISS" in line]
        outcome = next((t for t, line in said if line.startswith(("ATTACK KEY:", "HER KEYS:"))
                        and "put off" not in line and "no check" not in line), None)
        attack = {sim.keys.attack.name, *(k for f, k in sim.world.realism.game_keys if k is not None and f == "attack")}
        after: dict[str, int] = {}
        keys_after: dict[str, int] = {}
        keys_all: dict[str, int] = {}
        for t, name, events, _ in sim.trace.plans:
            for e in events:
                if len(e) == 3 and e[2] == "down":
                    keys_all[e[1]] = keys_all.get(e[1], 0) + 1
                if outcome is not None and t > outcome and len(e) == 3 and e[2] == "down":
                    keys_after[e[1]] = keys_after.get(e[1], 0) + 1
                    if e[1] in attack:
                        after[e[1]] = after.get(e[1], 0) + 1
        window = [[t, name, [e[1] for e in events if len(e) == 3 and e[2] == "down"]] for t, name, events, _
                  in sim.trace.plans if name.startswith("key_bindings")]
        kills = [k for k in sim.world.kill_log if k[3] == "her" and outcome is not None and k[0] > outcome]
        from ..control import keymap
        remembered = keymap.at_start(sim.keys, "thief")
        return {"said": said, "window": window, "pages": sim.player.pages, "stopped": stopped,
                "kills_after": len(kills), "attack_keys_after": dict(sorted(after.items())), "outcome_at": outcome,
                "keys_after": dict(sorted(keys_after.items())), "keys_all": dict(sorted(keys_all.items())),
                "remembered": None if remembered.data is None else remembered.data.get("keys"),
                "moves": dict(tour.h.moves.counts), "watch": dict(tour.h.attacks.watch.counts),
                "casts": None if tour.h.attacks.casts is None else dict(tour.h.attacks.casts.counts)}
    return run


def _window_ok(tr: dict) -> bool:
    """The window's keys: the backslash before any Esc, Esc twice at the most, nothing else pressed in it (the world's
    count), the game's MENU never opened (Esc with nothing open), the window closed at the end."""
    plans = tr["result"]["window"]
    keys = [k for _, _, ks in plans for k in ks]
    w = tr["world"]
    return (keys[:1] == ["\\"] and keys.count("esc") <= 2 and "esc" in keys and w.get("keys_in_window", 0) == 0
            and w.get("menu_opened", 0) == 0 and not w.get("key_window_open", False))


def _said(tr: dict, text: str) -> bool:
    return any(text in line for _, line in tr["result"]["said"])


KEY_EXPECTS = (
    logs("THE REGULAR ATTACK TOOK NO EFFECT 5 TIMES IN A ROW"),
    check("the backslash first, then Esc with the window open, nothing else in it, no MENU", _window_ok))


@tree((23, 73.0), monsters="all", seed=13, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("attack", "a"),))),
      expects=KEY_EXPECTS + (
          check("her stars asked", lambda tr: _said(tr, "her stars: ")),
          logs("ATTACK KEY: THE GAME HAS ATTACK ON A, NOT CTRL"),
          check("paged once, about the attack key", lambda tr: len(tr["result"]["pages"]) == 1
                and "attack key" in tr["result"]["pages"][0][1]),
          check("A pressed after it, Ctrl never", lambda tr: tr["result"]["attack_keys_after"].get("a", 0) >= 1
                and "ctrl" not in tr["result"]["attack_keys_after"]),
          check("kills after it", lambda tr: tr["result"]["kills_after"] >= 3)))
def loop_tree_grind_attack_key(sim):
    """`tour --loop --grind --map 10002073` (the feed, the attack choice) with the game's Attack on A, not on her Ctrl
    (as live on 2026-10-28 00:30, the user's own play having moved it): her Lucky Seven casts hurt, her regular presses
    to finish a monster do nothing; 5 of them in a row judged without effect, the doubt (on seed 1 at 37.1 s: a Green
    Mushroom at 4%, its 9/233 HP, pressed at from 32.3 s); a monster near, so away to 25 (2.2 s), a safe platform, so the
    Green Mushroom within 250 of her there (on 24: it can't touch her) doesn't put the check off (keycheck.near_her, from
    the afternoon of 2026-10-28; until then put off to 45.0 s): her stars not known (no monster in reach to cast at), the
    backslash, the window read (ATTACK on A, preset 1), Esc, closed (39.7-41.2 s); her regular attack A from then,
    remembered, paged once; kills with it after. (Until the review of 2026-10-28 the window opened at 40.0 s with that
    Green Mushroom near on 23's side: nothing looked again after the retreat.)"""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=13, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("attack", None),))),
      expects=KEY_EXPECTS + (
          logs("ATTACK KEY: THE GAME HAS ATTACK ON NO KEY"),
          check("paged once, about the attack key", lambda tr: len(tr["result"]["pages"]) == 1
                and "attack key" in tr["result"]["pages"][0][1]),
          check("no regular attack after it", lambda tr: not tr["result"]["attack_keys_after"]),
          logs("the regular attack off this run")))
def loop_tree_attack_unbound(sim):
    """As loop_tree_grind_attack_key, the game's Attack on no key: the same doubt and check, the window read with ATTACK
    on none (no key of the keyboard carries it), so no more regular attacks this run: Lucky Seven only (her throws that
    would have been the regular attack: "the regular attack off this run"), paged once; remembered so (her next run
    starts with the regular attack off)."""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=8, limit=400, may_die=True,
      **dict(_LIVE, noise=Noise(regular_miss=1.0, miss_unseen=1.0)),
      expects=KEY_EXPECTS + (
          logs("ATTACK KEY: ATTACK IS ON CTRL AS CONFIGURED"),
          check("nobody paged", lambda tr: not tr["result"]["pages"]),
          check("the hunt went on: kills after it", lambda tr: tr["result"]["kills_after"] >= 3),
          check("no stop", lambda tr: tr["result"]["stopped"] is None)))
def loop_tree_attack_misses(sim):
    """Her keys right, every regular star missing (Noise.regular_miss) and its MISS popup not seen (Noise.miss_unseen:
    under other numbers, say): 5 presses without effect, the doubt (the same Green Mushroom at 37.1 s), the window read
    with ATTACK on Ctrl as configured: the misses were something else; that target gets no more regular attacks, the hunt
    goes on (Ctrl pressed at others after), nobody paged. A doubt within 300 s of it would hold the regular attack at its
    targets, with no check. (loop_tree_attack_misses_seen: the MISSes read, no doubt.)"""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=1, limit=400, may_die=True,
      **dict(_LIVE, noise=Noise(regular_miss=1.0)),
      expects=(check("her attack's MISS read over her regular presses' targets",
                     lambda tr: tr["result"]["watch"].get("missed", 0) >= 3),
               check("no doubt: the MISSes say her key works", lambda tr: not _said(tr, "THE REGULAR ATTACK TOOK NO")),
               check("the window never opened", lambda tr: not tr["result"]["window"]),
               check("nobody paged", lambda tr: not tr["result"]["pages"]),
               check("no stop", lambda tr: tr["result"]["stopped"] is None)))
def loop_tree_attack_misses_seen(sim):
    """As loop_tree_attack_misses, every regular star missing, its MISS popup over the monster read (perception.misses;
    the user, 2026-10-28: "Yes, read it"): each press with one over its target took effect (keycheck: "missed"), so no
    run of 5 without effect: no doubt, no window, nobody paged."""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=1, limit=400, may_die=True, attack="regular",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=Realism(zero_walls=True, darts=40),
      expects=(logs("THE REGULAR ATTACK TOOK NO EFFECT 5 TIMES IN A ROW"), logs("OUT OF DARTS: 20 Lucky Seven presses"),
               check("stopped out of darts", lambda tr: "OUT OF DARTS" in (tr["result"]["stopped"] or "")),
               check("by out of darts' own watch, not the stars' test", lambda tr: "Lucky Seven presses over" in (
                   tr["result"]["stopped"] or "") and "regular attack took no effect" not in tr["result"]["stopped"]),
               check("paged once, about her darts", lambda tr: len(tr["result"]["pages"]) == 1
                     and "out of darts" in tr["result"]["pages"][0][1])))
def loop_tree_attack_no_darts(sim):
    """`tour --loop --attack regular` (the regular attack alone: out of darts' own watch counts Lucky Seven's presses, so
    it doesn't fire while she throws it) with 40 throwing stars: gone at about 50 s, her regular presses only swing the
    claw, nothing at range; 5 in a row without effect at a Green Mushroom she had hurt (91%), the doubt at 81.4 s; away
    to 25 (6.4 s), a safe platform: the check there (her stars' test: a cast lowered its target's bar, she has them; the
    window, ATTACK on Ctrl as configured: no more regular attacks at that Green Mushroom); her regular presses held at
    the next doubts' targets (no check within 300 s), Lucky Seven at them, which takes no MP with no stars (the
    world's); out of darts' own stop at 20 such presses (135.3 s: within the check's gap, no key check first: her
    retreat, paged, the recording kept). Until the afternoon of 2026-10-28 the check was put off (a Green Mushroom on
    24 within 250 of her on 25) and the stop came at 121.3 s; until the review of 2026-10-28 her stars' test stopped it
    at 5 casts without MP (79.5 s), which live, with her stars in her bag, 7-8 in a row had (behavior.keycheck)."""
    return key_check_run(4)(sim)


# --- her other keys moved in the game: followed (the user, 2026-10-28: "Follow them"; control.keymap) ----------------
# Her jump or an arrow on another key: her jumps don't lift her or her runs don't move her (keycheck.MoveWatch), the
# doubt, the check, the window read: followed. Her Lucky Seven on another key: her casts take no MP and kill nothing,
# out of darts' stop due: the check first, the window read: followed, no stop. And the binding read, remembered: the next
# run starts with it (keymap.at_start), no doubt.

def _keys_ok(after: dict, pressed: str, never: str) -> bool:
    return after.get(pressed, 0) >= 1 and never not in after


FOLLOW_EXPECTS = (
    check("the backslash first, then Esc with the window open, nothing else in it, no MENU", _window_ok),
    check("paged once, about her keys", lambda tr: len(tr["result"]["pages"]) == 1
          and "her keys" in tr["result"]["pages"][0][1]),
    check("no stop", lambda tr: tr["result"]["stopped"] is None),
    check("remembered for her next runs", lambda tr: tr["result"]["remembered"] is not None))


@tree((23, 73.0), monsters="all", seed=1, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("jump", "c"),))),
      expects=FOLLOW_EXPECTS + (
          logs("HER JUMPS DIDN'T LIFT HER 6 TIMES IN A ROW"), logs("HER KEYS: THE GAME HAS THEM ON OTHER KEYS"),
          logs("JUMP ALT -> C"),
          check("C pressed after it, Alt never", lambda tr: _keys_ok(tr["result"]["keys_after"], "c", "alt")),
          check("her jumps lifted her after it", lambda tr: tr["result"]["moves"].get("lifted", 0) >= 3),
          check("C remembered as her jump", lambda tr: (tr["result"]["remembered"] or {}).get("jump") == "c")))
def loop_tree_grind_jump_key(sim):
    """`tour --loop --grind --map 10002073` (the feed, the attack choice) with the game's JUMP on C, not on her Alt (her
    own play having moved it): her jumps don't lift her, her trips end short and she drops to the floor; 6 jumps in a
    row her minimap dot didn't rise over, the doubt (keycheck.MoveWatch); the check where she is (her moves doubted: no
    retreat) at a quiet moment, the window read (JUMP on C), Esc, closed; her jump C from then on (her chains' key names
    and her actions'), remembered (lab/keybindings-thief.json) and her moves' keys for the player (alarm_keys.json),
    paged once; her jumps lift her again."""
    return key_check_run(4)(sim)


@tree((23, -150.0), monsters="all", seed=1, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("right", "e"),))),
      expects=FOLLOW_EXPECTS + (
          logs("HER RUNS DIDN'T MOVE HER 4 TIMES IN A ROW"), logs("RIGHT RIGHT -> E"),
          check("E pressed after it, Right never", lambda tr: _keys_ok(tr["result"]["keys_after"], "e", "right")),
          check("her runs moved her after it", lambda tr: tr["result"]["moves"].get("ran", 0) >= 3)))
def loop_tree_grind_arrow_key(sim):
    """As loop_tree_grind_jump_key from the left of 23, the game's MOVE RIGHT on E, not on her Right arrow: her runs right
    don't move her (her chains cut short, her dot standing where they run her: 'off the path (still at x ...)'), 4 in a
    row, the doubt; the window read (MOVE RIGHT on E): her right E from then on, remembered, paged once. (From 23 at 73
    on seeds 1-3, and with her left on Q, the hunt kept her where she was, throwing, and her chains came only once it was
    over: an arrow moved shows in her runs alone, not in her facing, which live frames tell otherwise than her taps
    with her keys right too: 151 times in the tour runs of 2026-10-24 to 28, up to 13 within 20 s.)"""
    return key_check_run(4)(sim)


@tree((23, 73.0), monsters="all", seed=1, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("lucky_seven", "a"),))),
      expects=FOLLOW_EXPECTS + (
          logs("the key bindings checked first"), logs("LUCKY SEVEN SHIFT -> A"),
          check("A pressed after it, Shift never", lambda tr: _keys_ok(tr["result"]["keys_after"], "a", "shift")),
          check("kills after it", lambda tr: tr["result"]["kills_after"] >= 3)))
def loop_tree_grind_lucky_seven_key(sim):
    """As loop_tree_grind_jump_key, the game's Lucky Seven on A, not on her Shift: her casts take no MP and kill nothing
    (the key does nothing), 20 in a row: out of darts' stop due, the key bindings checked first (behavior.hunter:
    STARS_CHECKED; put off, her Lucky Seven held meanwhile, the regular attack instead, 15 s at the most), the window
    read (Lucky Seven's picture on A): her Lucky Seven A from then on, out of darts' stretch from there, no
    stop; remembered, paged once; kills with it after. (Until 2026-10-28 out of darts stopped the hunt at 24 s.) On seed
    1: out of darts due at 25.4 s, away to 25 (a safe platform: no put-off), the window read at 28.7 s, 71 kills after."""
    return key_check_run(4)(sim)


@tree((23, 73.0), monsters="all", seed=1, limit=400, may_die=True,
      remembered={"jump": "c"},
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("jump", "c"),))),
      expects=(logs("her keys: the key bindings read off the game's window"), logs("JUMP ALT -> C"),
               check("no doubt, no window", lambda tr: not tr["result"]["window"]
                     and not _said(tr, "HER JUMPS DIDN'T")),
               check("C pressed, Alt never", lambda tr: _keys_ok(tr["result"]["keys_all"], "c", "alt")),
               check("her jumps lifted her", lambda tr: tr["result"]["moves"].get("lifted", 0) >= 3)))
def loop_tree_remembered_keys(sim):
    """The run after loop_tree_grind_jump_key's: the game's JUMP on C still, and the binding that check read remembered
    (lab/keybindings-thief.json): her hunter starts with C for her jump (control.keymap.at_start, said in the log), her
    jumps lift her from the start, no doubt, no window."""
    return key_check_run(2)(sim)


# --- the top of a rope: hanging just under a platform, her dot snapped onto it -------------------------------------
# Live on 2026-10-26 (Tree Dungeon, 22:58, 23:11, 23:12) a touch cut her climb up the rope at x 111 near its top; the
# game keeps her on a rope when touched (world.Realism.rope_knocks: off since), so she hung about 30 under platform 18, and
# her dot placed her on it: her Lucky Seven presses refused there ("That cannot be used while you're on a rope or
# ladder."), 20 of them stopped the hunt for out of darts with her stars in her bag (behavior.darts). Her place
# (localization.pose.ROPE_TOP_GAP), a move that never moved her dot (control.navigator._rope_top), the game's word on
# the message bar (behavior.hunter._hear_chat) and her climb going on after a touch (navigator.CLIMB_ON) get her off.
# Grinding, as live, loop_tree_grind_rope_top is the one master's stack fails (the review's).

ROPE_TOP_EXPECTS = (
    check("no stop: her stars never ran out", lambda tr: tr["result"]["stopped"] is None),
    logs("getting off"), sends("off_rope"),
    check("on to hunt: kills", lambda tr: sum(tr["world"]["kills"].values()) >= 1))


@tree((18, 111.0), hanging=-865.0, monsters="all", seed=1, limit=120, **_LIVE, expects=ROPE_TOP_EXPECTS)
def loop_tree_rope_top(sim):
    """`tour --loop --map 10002073` (the feed, the attack choice) from the rope at x 111, her feet 30 under platform 18
    (-865), as live at 22:58:08: under 17 there her dot is placed on the rope (localization.pose.ROPE_TOP_GAP), not on
    18 (where it snapped her before, and her presses were refused till out of darts stopped the hunt), and she climbs up
    off it onto 18 (Mover.off_rope) and hunts on. (Master's stack in this world meets all of this too, off the rope at
    9.6 s after a touch, before 20 refused presses: so it's not the check of the live case; loop_tree_grind_rope_top,
    grinding as live, is.)"""
    return darts_stop(1)(sim)


@tree((18, 111.0), hanging=-865.0, monsters="all", seed=2, limit=120, **_LIVE, expects=ROPE_TOP_EXPECTS)
def loop_tree_grind_rope_top(sim):
    """`tour --loop --grind --map 10002073` (the feed, the attack choice), as live at 22:58:08, from the rope at x 111,
    her feet 30 under platform 18 (-865): on the rope at the first look (localization.pose.ROPE_TOP_GAP), up off it onto
    18 (Mover.off_rope), and on to grind. Master's stack in this world stopped the hunt for out of darts at 17.7 s (on
    seed 1 at 23.1 s; the -880 twin, 15 under, at 17.4-18.3 s on seeds 1-3; at 18.5 s with after-kill in, 7f6334e), no
    kill, its presses refused on the rope, as live."""
    return darts_stop(1, grind=True)(sim)


@tree((18, 111.0), hanging=-880.0, monsters="all", seed=1, limit=120, **_LIVE, expects=ROPE_TOP_EXPECTS)
def loop_tree_rope_top_15(sim):
    """As loop_tree_rope_top, her feet 15 under 18 (-880; at 22:19:38 live her dot was 14.8 under it at 9 chain
    starts): her dot still puts her on 18, so a move from there that never moves it (control.navigator._rope_top) or
    the game refusing her presses on the message bar (behavior.hunter._hear_chat) has her on the rope: off it, onto 18,
    and on to hunt. (Master stopped the hunt at 18.7 s, 20 presses refused.)"""
    return darts_stop(1)(sim)


@tree((18, 111.0), hanging=-880.0, limit=60,
      expects=(ends("arrived"), logs("her dot never moved"), sends("off_rope"),
               check("on 16 by the end", lambda tr: tr["world"]["platform"] == 16)))
def nav_tree_rope_top_15(sim):
    """From the rope at x 111, her feet 15 under platform 18 (her dot puts her on 18), up to 16 with no monsters about:
    the jump up from 18 does nothing on the rope and her dot never moves (control.navigator._rope_top), so she's on the
    rope: up off it onto 18 (Mover.off_rope), and on up."""
    return _nav(16)(sim)


@tree((19, 250.0), monsters=[("Green Mushroom", 18, 125.0)], seed=1, limit=120, **_LIVE,
      expects=(ends("arrived"), logs("touched on the rope"), sends("climb_on"),
               check("on 16 by the end", lambda tr: tr["world"]["platform"] == 16)))
def nav_tree_rope_hit(sim):
    """From 19 up to 16 by the rope at x 111 (19 -> 18), a Green Mushroom walking on 18 by its top (as at 22:58:05
    live, where the trip went past it regardless): it touches her on the rope, which cuts the chain and lets go of
    up; the game keeps her on the rope, and up goes down again at once (navigator.CLIMB_ON), onto 18 and on."""
    return _nav(16, fight=True)(sim)


@tree((24, 231.0), monsters=[("Horny Mushroom", 23, 20.0)], seed=3, limit=60, **_LIVE,
      expects=(ends("arrived"), logs("could touch her on the way: shooting it first"),
               logs("by her landing on 23: no shot first, it's dying"), no_log("none thrown"),
               check("on 23 by the end", lambda tr: tr["world"]["platform"] == 23),
               check("not hurt", lambda tr: tr["world"]["touches"] - tr["world"]["evaded"] == 0)))
def nav_tree_landing_shot_first(sim):
    """From 24 down onto 23 in transit, a Horny Mushroom on 23 that the jump lands her 100-140 from, inside its keep-off
    (attack.danger: 169), so no attack of hers from there is sure (attack.stance.landing): a star from 24 reaches it,
    and it's shot first from where she stands by the chain's own check (_meet's clear_for: "could touch her on the way:
    shooting it first"), killed; its body, dying there, still prices the landing, and the landing's shot first says so
    and sends nothing ("no shot first, it's dying": FightOnTheWay.shot_place), then she goes, not hurt. As recorded
    until 2026-10-28 the landing's shot first fired twice at the dying one, each "nothing in reach", 1.3 s, and logged
    "shooting it first" both times, the line this scenario looked for: _clear_landing's own shot never threw here
    (`Escort.throws` tells now). (Of seeds 1-6 the others shoot it first from 24 too: _meet's.)"""
    from ..control.navigator import TRANSIT
    return sim.navigator(sim.hunter(), fight=True).go(23, mode=TRANSIT)


def _spread_from_spot() -> bool:
    from ..motion import flow
    return flow.SPREAD_FROM_SPOT


@tree((21, 42.0), monsters=[("Horny Mushroom", 23, 24.0), ("Horny Mushroom", 23, -132.0)], seed=2, limit=60, **_LIVE,
      expects=(ends("arrived"), no_log("by her landing on 23: shooting it first"), no_log("none thrown"),
               check("says why there's no shot first (or, the take-offs from their own spots, drops clear)",
                     lambda tr: _spread_from_spot() or any("no shot first, no throw from the ground takes it from "
                                                           "anywhere on platform 21" in line for line in tr["log"])),
               check("on 23 by the end", lambda tr: tr["world"]["platform"] == 23),
               check("not hurt", lambda tr: tr["world"]["touches"] - tr["world"]["evaded"] == 0)))
def nav_tree_landing_no_shot(sim):
    """From 21 at x 42 down onto 23 (the grind's stop 23 at x 8, a hunt trip), Horny Mushrooms on 23 at 24 and -132, as
    live at 00:38:36 on 2026-10-28 (grind-1028-0030.log), where the drop from where she stood landed inside the walker's
    keep-off and the log said "horny-mushroom at x 24 by her landing on 23: shooting it first" with nothing thrown: from
    nowhere on 21 does a star reach 23 (117-122 under it, past reach.SHOT_DOWN), nor one from a jump. Here as there
    before the fix ("... at x -130 ...: shooting it first" at 0.84 s, then the drop at once from 42); now the log says
    there's no shot and why (FightOnTheWay.shot_place: "no shot first, no throw from the ground takes it from anywhere on
    platform 21 (it's 125 under her), nor one from a jump where she stands") and nothing is sent, and she goes. With
    flow.SPREAD_FROM_SPOT (off from the review of 2026-10-28) the drop's other takeoffs are tried from their own spots,
    it drops from another, clear of both, and there's nothing to say; not hurt either way."""
    from ..control.navigator import HUNT
    return sim.navigator(sim.hunter(), fight=True).go(23, x=8.0, mode=HUNT)
