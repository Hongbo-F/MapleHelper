"""Scenarios: gamelab's entry points run against the simulated game on the real map (10000012), each
built the way gamelab/__main__.py builds them (_hunter, Navigator, Tour), with the fakes in place.

Each scenario returns a trace: every key plan sent ([t, name, events, end_ms], in order), every cancel,
every log line, the entry point's result, what the world saw happen, how often she looked (and didn't
see her tag or dot), and the lines written to the lab's logs. The same scenario always gives
the same trace, so traces recorded now (golden.py) pin what gamelab does today. Each scenario also
`expects` what it's there for (a hop, an arrival, three tour stops...); tests/test_sim.py checks those, so
a change in the sim or in gamelab can't quietly drop a behaviour a scenario was meant to cover.

Game data stays on this computer: the map is read from gamedata's local extract, and scenarios are
skipped when it isn't there (`available()`). Data that is there but fails to load is an error, not a skip.

The stack only runs with the fakes in place: `with Sim(model, scenario) as sim: sim.hunter()...`.
"""

import dataclasses
import importlib
import json
import pkgutil
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from maplehelper.engine import RIGHT

from .. import telemetry
from ..control import client
from ..localization import mapdata
from ..behavior.hunter import Hunter
from ..behavior.looting import DropSpot
from ..localization.mapdata import MapModel
from ..control.navigator import Navigator
from ..control.timeline import her_setup
from ..behavior.tour import Tour
from .fakes import Clock, FakeEyes, FakePlayer, SimTimeout, SimViolation, Trace, patched
from .world import Noise, Realism, SimDied, World

MAP_ID = 10000012  # Victoria Road: Thicket Around the Beach II (__main__.DEFAULT_MAP)
_MODEL: list = []


def available() -> bool:
    """Whether the map's game data is on this computer: its parsed extract, or the game data copy that
    mapdata.load makes one from. Says nothing about whether it loads: real_map() raises if it doesn't."""
    from gamedata.paths import COPY
    return (mapdata.CACHE / f"{MAP_ID}.json").is_file() or (COPY / "Map" / "Map").is_dir()


def real_map() -> MapModel | None:
    """The real map's model (loaded once), or None when its game data isn't on this computer. A map
    that is there but fails to load raises: a refactor that breaks mapdata must fail, not skip."""
    if not available():
        return None
    if not _MODEL:
        _MODEL.append(mapdata.load(MAP_ID))
    return _MODEL[0]


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
        _, self.keys, _ = her_setup()
        self.clock = Clock(limit=sc.limit, pace=pace, think=sc.think)
        self.world = World(model, self.keys, sc.start, seed=sc.seed, monsters=sc.monsters, hp=sc.hp,
                           inventory_full=sc.inventory_full, facing=sc.facing, origin=self.clock.now,
                           drops=list(sc.drops), hanging=sc.hanging, noise=sc.noise, realism=sc.realism)
        self.eyes = FakeEyes(self.world, self.clock, model, guard=self.guard)
        self.trace = Trace(self.clock.now)
        self.player = FakePlayer(self.world, self.clock, self.trace, self.keys)
        self.active = False
        self.files: dict[str, list[str]] = {}  # what the run wrote (under its stand-in for gamelab/lab)

    def __enter__(self) -> "Sim":
        _import_all()
        _fresh_caches(self.model)
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

    def log(self, message: str) -> None:
        self.trace.log.append(f"{self.clock.elapsed:9.3f} {message}")
        telemetry.emit("log", text=message)

    # The stack, as gamelab/__main__.py builds it.
    def hunter(self, **kw) -> Hunter:
        self.guard()
        h = Hunter(self.eyes, self.keys, self.model, log=self.log, **kw)
        play = h.actions.play  # every plan Actions plays (fights, single moves, tours): checked each time;
        # the navigator's chains go through client.send, which guard() checks through the fake player

        def guarded(events, name):
            self.guard()
            play(events, name)
        h.actions.play = guarded
        return h

    def navigator(self, hunter: Hunter, fight: bool = False, seed: int | None = None) -> Navigator:
        self.guard()
        return hunter.navigator(fighting=fight, rng=random.Random(self.scenario.seed if seed is None else seed))

    def tour(self, hunter: Hunter, nav: Navigator, seed: int | None = None, rest_below: float = 0.5) -> Tour:
        self.guard()
        return Tour(hunter, nav, log=self.log, rng=random.Random(self.scenario.seed if seed is None else seed),
                    rest_below=rest_below)

    def tour_stack(self, **kw) -> Tour:
        """`python -m gamelab tour`: a hunter with the tour's HP limits, a fighting navigator, the tour."""
        h = self.hunter(low_hp=0.3, potion_hp=0.45)
        return self.tour(h, self.navigator(h, fight=True), **kw)


# --- running -----------------------------------------------------------------------------------------

_BASE_CACHE: dict[int, tuple] = {}


def _fresh_caches(model: MapModel) -> None:
    """physics.fly caches landings by rounded start: a run must not reuse one an earlier run left (it was
    computed from a slightly different start), so each starts from the same cache: the planner's own."""
    from ..motion import physics, route
    cache = physics._FLY_CACHE
    key = id(model)
    if key not in _BASE_CACHE or _BASE_CACHE[key][0] is not model:
        for k in [k for k in cache if k[0] == key]:
            del cache[k]
        route.Planner(model)
        _BASE_CACHE[key] = (model, {k: v for k, v in cache.items() if k[0] == key})  # the model kept: ids get reused
    for k in [k for k in cache if k[0] == key]:
        del cache[k]
    cache.update(_BASE_CACHE[key][1])


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
    """Run a scenario on the real map (or on `model`); returns its trace. A run that ends early (she
    died, the time limit, the player refusing a plan, gamelab giving up) has that as its result."""
    return run_sim(sc, model, pace)[0]


def run_sim(sc: Scenario, model: MapModel | None = None, pace: float | None = None) -> tuple[dict, "Sim"]:
    """run_scenario, also returning the Sim it ran in (its world, clock and eyes, for sim.harness)."""
    model = model or real_map()
    if model is None:
        raise FileNotFoundError(f"map {MAP_ID} isn't available (gamedata/data); scenarios need it")
    with Sim(model, sc, pace) as sim:
        try:
            outcome = plain(sc.fn(sim))
        except SimViolation:
            raise
        except (SimDied, SimTimeout, client.PlayerError, RuntimeError, ValueError) as e:
            outcome = f"error: {type(e).__name__}: {e}"
    trace = {"scenario": sc.name, "doc": sc.doc, "seconds": round(sim.clock.elapsed, 3), "result": outcome,
             "plans": sim.trace.plans, "cancels": sim.trace.cancels, "log": sim.trace.log,
             "world": sim.world.summary(), "world_log": sim.world.log, "eyes": dict(sim.eyes.counts),
             "files": sim.files}
    return trace, sim


def unmet(trace: dict, sc: Scenario | None = None) -> list[str]:
    """What scenario `sc` (the trace's own by default) expects that its trace doesn't show. Unless it
    `may_die`, every scenario expects her alive at the end."""
    sc = sc or SCENARIOS[trace["scenario"]]
    out = [f"she died: {trace['result']}"] if trace["world"]["dead"] and not sc.may_die else []
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


@scenario((25, 760.0), seed=2, expects=(ends("arrived"), moves("walk_off", "drop")))
def nav_down_to_basin(sim):
    """From platform 25 down into the basin: a walk-off onto 33, then a drop through it."""
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


@scenario((12, -300.0), expects=(ends("arrived"), moves("jump_up", "walk_off", "jump")))
def nav_walk_offs(sim):
    """From 12 across to 17: a jump up, walk-offs and jumps."""
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


@scenario((30, 795.0), hanging=180.0, expects=(ends("arrived"), logs("on a rope at x"), sends("off_rope")))
def nav_off_rope_jump(sim):
    """Hanging halfway up the rope from 30 to 25: no platform under her dot, so she jumps off it sideways
    (towards the side a fall lands on a platform), then goes on to 36."""
    return _nav(36, x=500)(sim)


@scenario((30, 795.0), hanging=80.0, expects=(ends("arrived"), sends("off_rope")))
def nav_off_rope_climb(sim):
    """Hanging near the top of the rope up to 25: she climbs on up onto it, and that's where she's going."""
    return _nav(25)(sim)


@scenario((36, 300.0), noise=Noise(dot_hidden=((0.0, 2.0), (4.0, 4.6))),
          expects=(ends("arrived"), sends("mouse", "the mouse is moved when the dot stays hidden"),
                   check("looks and glances without the dot", lambda tr: tr["eyes"]["dot hidden"] >= 5)))
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


# Navigation with monsters about, not fighting: hops, waits, ways around, plans cut short.

@scenario((36, 200.0), monsters=[("Pig", 36, 600.0)], seed=1,
          expects=(ends("arrived"), logs("hop@", "a hop over the pig in the chain")))
def nav_hop_pig(sim):
    """A pig walking the basin floor between her and the rope up to 30: hopped over on the way (Flow's hop)."""
    return _nav(30)(sim)


@scenario((34, -300.0), monsters="all", seed=1,
          expects=(ends("arrived"), logs("looking for a way around"), logs("hop@"), logs("hit on the way")))
def nav_monsters_staircase(sim):
    """Up the staircase (34 -> 19) among the map's monsters, not fighting: she looks for ways around the
    monster in her way, hops over one, and is touched and knocked about a few times on the way up.
    (Seed 3 until her walks pressed pick-up: they last ~0.2 s longer, and with seed 3 no monster was then
    in her way to hop.)"""
    return _nav(19)(sim)


@scenario((36, 200.0), monsters="all", seed=4, expects=(ends("arrived"), moves("climb")))
def nav_monsters_basin(sim):
    """Across the basin floor (pigs and slimes) and up onto 33, not fighting."""
    return _nav(33)(sim)


@scenario((21, -300.0), monsters="all", seed=11,
          expects=(ends("arrived"), logs("waiting for the monster to move off")))
def nav_monsters_long_way(sim):
    """From 21 up to 2 with monsters about, not fighting: she waits for monsters to move off her way. (Seed
    11 since chains follow on: on seed 5 the staircase's monsters knocked her back down until time ran out;
    over 16 seeds 12 arrive, 14 before.)"""
    return _nav(2)(sim)


# Navigation with `fight` on (as on tours): monsters in reach are shot on the way.

@scenario((36, 300.0), monsters="all", seed=6,
          expects=(ends("arrived"), logs("shooting it on the way"), logs("in the way at x"),
                   check("kills on the way", lambda tr: _result(tr, "kills") >= 3),
                   check("the trip counts every kill, those in the way of a move too",
                         lambda tr: _result(tr, "kills") == sum(tr["world"]["kills"].values()))))
def nav_fight_basin(sim):
    """Out of the basin up to 25 with `fight` on: the pigs and slimes in reach are shot first (a few
    fights of a few kills each), and one in the way of a move; then up the two ropes. The trip counts
    the kills of both (those in the way weren't counted before)."""
    return _nav(25, fight=True)(sim)


@scenario((33, 900.0), monsters="all", seed=1, expects=(ends("arrived"), logs("shooting it on the way")))
def nav_fight_short(sim):
    """From 33 up to 25 with `fight` on and the tour's kinds: a few shot on the way, and there."""
    return _nav(25, fight=True, kinds=TOUR_KINDS)(sim)


@scenario((14, 760.0), monsters="all", seed=10,
          expects=(ends("arrived"), logs("shooting it on the way"), moves("climb")))
def nav_fight_up(sim):
    """From 14 up two ropes and a jump to 1 (pigs, Orange Mushrooms) with `fight` on: she shoots 17's
    monsters from 14 first, and fights her way up past 3 and 4."""
    return _nav(1, fight=True, kinds=TOUR_KINDS)(sim)


@scenario((12, -300.0), monsters="all", seed=3,
          expects=(ends("arrived"), logs("knocked off?"), logs("plan cut short")))
def nav_fight_across(sim):
    """From 12 across to 17 with `fight` on: starting among the pigs on 12 she fights, is knocked off (down
    to 34), and goes on from there; hit on the way, her chains are cut and replanned, and she fights her way
    to 17. (Seed 3 since jump-throws come 40 ms after the jump: on seed 6 nothing knocked her off any more.
    On seed 6 before, her first chain from 11 waited out the attack lock of her last throw: before it did,
    she was touched on the way, and the quick looks after those cuts skipped the fights on 18 and 17.)"""
    return _nav(17, fight=True, kinds=TOUR_KINDS)(sim)


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


@scenario((30, 700.0), seed=7, noise=Noise(walk_jitter=5.0, dot_px=True),
          expects=(sends("nudge", "nudges by the rope on screen"),
                   ends(30, key="landed", what="she's left beside the rope, on 30")))
def move_climb_down_nudge(sim):
    """Mover.do with walks a few units off: the walk stops 5 units (7 px) from the rope's top, and
    Mover.line_up_rope nudges her by the rope on screen. Each nudge is timed by walking speed alone, but
    any tap moves her at least ~12 px (physics.py's measured walk: 8.5 + 0.149 t units), so a 7 px nudge
    goes 22 px: she swings from side to side of the rope, and after three nudges she presses down beside
    it and stays on 30. A gamelab bug, pinned here until it's fixed."""
    return _one_move(sim, "climb_down", 30, 36)


@scenario((1, 1200.0), expects=(ends(0, key="landed"),))
def move_jump_up(sim):
    """Mover.do: step under platform 0 and jump straight up onto it."""
    return _one_move(sim, "jump_up", 1, 0)


@scenario((4, 300.0), expects=(ends(2, key="landed"),))
def move_jump(sim):
    """Mover.do: run towards 4's end and jump up onto platform 2."""
    return _one_move(sim, "jump", 4, 2)


@scenario((20, 0.0), expects=(check("she doesn't stay on 19", lambda tr: _result(tr, "landed") != 19),))
def move_jump_short_landing(sim):
    """Mover.do: the jump from 20 up onto 19 lands, but the arrow is held on 150 ms past the landing (and
    she slides on): on a platform this short she runs off its far end."""
    return _one_move(sim, "jump", 20, 19)


@scenario((4, 450.0), expects=(ends(6, key="landed"), sends("walk_off")))
def move_walk_off(sim):
    """Mover.do: walk off platform 4's right end, down onto 6."""
    return _one_move(sim, "walk_off", 4, 6)


@scenario((3, 760.0), expects=(ends(None, key="landed"), sends("walk_off")))
def move_walk_off_long_fall(sim):
    """Mover.do: walk off platform 3's right end, 1500 units above the basin: Mover.settle gives up after
    2 s with her still falling, and the move ends on no platform."""
    return _one_move(sim, "walk_off", 3, 36)


@scenario((2, -100.0), expects=(ends(12, key="landed"), sends("drop")))
def move_drop(sim):
    """Mover.do: walk to the spot and drop through 2 onto 12."""
    return _one_move(sim, "drop", 2, 12)


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


@scenario((36, 300.0), monsters={36}, seed=7,
          expects=(ends("killed", key="ended"), happens("claw swipe hits", "close up, the claw swipes")))
def hunt_kill(sim):
    """`hunt kill`: the slime nearest her on the basin floor, 55 units off. Too close for a star from the
    ground, her throws are claw swipes (a fifth of a star) until it's out of the claw's reach again (its
    swing reaches 90 ahead of her, into a slime's body up to 118 off); then looted. (Seed 7 since the claw
    reaches bodies: with seed 12 the basin's monsters kept walking into its reach, and swiped, she hadn't
    killed it after 30 s.)"""
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


@scenario((36, 200.0), monsters=[("Pig", 36, 850.0)], seed=14,
          expects=(ends("killed", key="ended"), happens("picks up leather")))
def hunt_kill_far(sim):
    """`hunt kill` at a pig across the basin: walk into reach (300 units at most per look), shoot it dead,
    loot it."""
    return _kill_far(sim)


@scenario((36, 200.0), monsters=[("Pig", 36, 850.0)], seed=4, noise=Noise(miss=0.35),
          expects=(ends("killed", key="ended"), happens("star misses"),
                   wrote("shots2.jsonl", '"hit": false', "a throw scored as a miss in the shot log")))
def hunt_kill_misses(sim):
    """hunt_kill_far (another seed) with a third of the stars missing: throws scored as misses (no HP bar
    change within ~0.9 s of the star's arrival), and more throws to kill it."""
    return _kill_far(sim)


@scenario((36, 200.0), monsters=[("Pig", 36, 850.0)], seed=14, noise=Noise(tag_hidden=((0.0, 4.0),)),
          expects=(ends("killed", key="ended"),
                   check("looks without her name tag", lambda tr: tr["eyes"]["tag hidden"] >= 3)))
def hunt_kill_tag_hidden(sim):
    """hunt_kill_far with her name tag hidden for the first 4 s (behind a box): she's placed on screen
    by the camera from her map position (Hunter.look), and the monsters by their offset from there."""
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
    claw swipes it, five times. (In about 2 seeds of 5 the snail walks into her first, and a touch knocks
    her off the end of 22: she stands 29 units from it.)"""
    h = sim.hunter()
    r = h.clear(max_kills=1, seconds=20, approach=False)
    return {"report": str(r)}


THROWS = ("throw", "jump_throw", "jump_back")  # the key plans that attack (control.actions)


@scenario((4, 520.0), monsters=[("Orange Mushroom", 2, 80.0), ("Orange Mushroom", 2, 140.0),
                                ("Orange Mushroom", 2, 200.0)], seed=1, limit=60,
          expects=(check("no throw from platform 4", lambda tr: not any(p[1] in THROWS for p in tr["plans"])),
                   check("she stays on platform 4", lambda tr: tr["world"]["platform"] == 4),
                   check("she moves on: nothing in reach",
                         lambda tr: "nothing left in reach" in _result(tr, "report"))))
def fight_step_wall(sim):
    """She's on platform 4 at x 520, nothing on it, Orange Mushrooms up on 2: 60 above her, 320-440 along,
    behind the step wall that joins 2 and 4 (one foothold group). By distance they're in reach, but no star
    from 4 gets past that wall (live: 0 of 3 in the shot log, 0 of 5 in the telemetry, all 4 kills of 40
    throws on 4): she doesn't throw at them, and moves on. With the old decisions she threw left at them
    for the whole fight, and every star hit nothing."""
    h = sim.hunter()
    r = h.clear(max_kills=3, seconds=20)
    return {"report": str(r)}


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


@scenario((36, 300.0), monsters="all", seed=22, limit=400,
          expects=(logs("shooting platform 36 on the platform"), ends("kill limit", key="stop"),
                   logs("collecting drops on platform 36"),
                   check("the stop's report counts HP lost", lambda tr: tr["result"]["damage_taken"] > 0),
                   no_log("couldn't get there (timeout)", "trips to drops on the basin get there")))
def tour_hunt_basin(sim):
    """Tour.hunt of the basin (36): no safe spot reaches it, so on the platform itself. Collecting the drops
    left on it, each trip (Navigator.go with `fight` on) shoots only monsters in her way to them, and gets
    there: it used to stop and shoot everything in reach, on the very platform she was going to, until the
    trip's 60 s ran out."""
    return _tour_hunt(36, 5)(sim)


@scenario((15, 100.0), monsters="all", seed=23, limit=400,
          expects=(logs("shooting platform 12 from platform 13"), logs("nothing left on platform 12")))
def tour_hunt_platform_12(sim):
    """Tour.hunt of platform 12 (pigs, Orange Mushrooms) from the safe platform 13 beside it."""
    return _tour_hunt(12, 4)(sim)


def _hunt_17_pigs(sim):
    tour = sim.tour_stack()
    tour.nav.escort.kinds = {"pig"}
    return tour.hunt(17, {"pig"}, kills=2, seconds=90)


@scenario((14, 760.0), monsters=[("Pig", 17, 1000.0), ("Pig", 17, 1320.0)], seed=1, limit=400,
          expects=(logs("out of reach here: over to platform"),))
def tour_hunt_other_spot(sim):
    """Tour.hunt of two pigs on 17 from 14: one dies, the other is out of reach, so over to a spot that
    reaches it (Tour.spot_for)."""
    return _hunt_17_pigs(sim)


@scenario((14, 760.0), monsters=[("Pig", 17, 1000.0), ("Pig", 17, 1320.0)], seed=4, limit=400,
          expects=(logs("that no spot reaches: onto the platform"),))
def tour_hunt_onto_platform(sim):
    """The same two pigs: no spot off the platform reaches the one left, so onto platform 17 itself."""
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


@scenario((16, 585.0), monsters=[("Pig", 18, 300.0)], seed=1, limit=300, drops=(("mesos", 29, 420.0),),
          expects=(logs("pig in reach on platform 18: shooting it on the way"),
                   check("the drops of the pig shot on the way are kept",
                         lambda tr: any(p == 18 for _, p in tr["result"]["kept"])),
                   logs("collecting drops on platform 18", "the next collection goes for them"),
                   check("a pig's leather picked up", lambda tr: tr["world"]["picked"].get("leather", 0) >= 1)))
def tour_collect_loot_on_the_way(sim):
    """Tour.collect_loot twice, as Tour.hunt calls it: on the trip from 16 to the mesos remembered on 29, she
    shoots a pig on 18 below on the way. Its drops are kept for the next collection, which picks them up
    (they used to be forgotten: from 29, 18 is out of view)."""
    tour = sim.tour_stack()
    tour.nav.escort.kinds = tour.worth
    tour.h.looter.pending = [DropSpot(420, 275, 29, sim.clock.now - 10, mesos=True, on_ground=True)]
    tour.collect_loot()
    kept = [[round(d.x), d.platform] for d in tour.h.looter.pending]
    tour.collect_loot()
    return {"report": str(tour.report), "kept": kept,
            "pending_loot": [[round(d.x), d.platform] for d in tour.h.looter.pending]}


@scenario((36, 300.0), monsters="all", seed=6, limit=1200,
          expects=(check("three stops visited", lambda tr: len(tr["result"]["visited"]) == 3),
                   ends("standing by on a safe platform", key="stop"), logs("tour: sweep"),
                   check("every kill counted (the stops' and on the way, resting included)",
                         lambda tr: sum(tr["result"]["kills"].values()) + tr["result"]["on_the_way"]
                         == sum(tr["world"]["kills"].values()))))
def tour_run(sim):
    """`python -m gamelab tour --stops 3`: the whole tour, end to end: three hunting grounds, rests when
    HP runs low, and standing by on a safe platform at the end. (Seed 6 since stars from the ground pick by
    the star's search: on seed 8 an EXP gain came in as the stop's fight ended, unplaced, and two kills on
    the way came between two looks. Seed 8 had been since jump-throws come 40 ms after the jump: on seed 25
    two kills came between two looks once, and two kills raise her EXP number once.) The basin's respawns
    make its length chaotic: on world seeds 1-16 it took 470-1190 s, before stars picked by the search and
    since, so the limit is 1200 s."""
    tour = sim.tour_stack()
    return tour.run(stops=3, kills_per_stop=5, seconds_per_stop=90)


@scenario((36, 300.0), seed=27,
          expects=(ends("standing by on a safe platform", key="stop"), no_log("resting on platform"),
                   check("on a safe platform", lambda tr: tr["world"]["platform"] in tr["result"]["safe"])))
def tour_stand_by(sim):
    """The end of a tour with HP still high (Tour.rest_and_stand_by): no rest, just over to the nearest safe
    platform to stand by."""
    tour = sim.tour_stack()
    tour.rest_and_stand_by()
    return {"stop": tour.report.stop, "safe": tour.safe}


def _rest(sim):
    tour = sim.tour_stack()
    tour.rest()
    return {"report": str(tour.report), "hp": round(tour.h.look(mobs=False).hp, 3)}


@scenario((36, 300.0), monsters="all", seed=26, hp=0.32, limit=400,
          expects=(logs("resting on platform 32"), no_log("couldn't reach a safe platform"),
                   check("HP back", lambda tr: tr["result"]["hp"] >= 0.9),
                   check("the kills on the way to rest are counted",
                         lambda tr: f"plus {sum(tr['world']['kills'].values())} on the way" in tr["result"]["report"])))
def tour_rest(sim):
    """Low on HP in the basin (32%, above the tour's 30% stop): to the nearest safe platform (32), sit on
    the chair until HP is back, stand up."""
    return _rest(sim)


@scenario((32, 600.0), monsters="all", seed=26, hp=0.32, limit=400,
          expects=(logs("resting on platform 32"), sends("chair"),
                   check("no trip: on a safe platform already",
                         lambda tr: not any(p[1] == "flow" for p in tr["plans"]))))
def tour_rest_on_safe(sim):
    """Low on HP on a safe platform already (32): she sits down where she is."""
    return _rest(sim)


@scenario((36, 300.0), monsters="all", seed=26, hp=0.25, limit=400,
          expects=(logs("resting on platform 32"), logs("drinking a potion", "a potion on the way"),
                   no_log("couldn't reach a safe platform", "low HP doesn't end the trip to rest"),
                   check("rests on 32", lambda tr: tr["world"]["platform"] == 32),
                   check("HP back", lambda tr: tr["result"]["hp"] >= 0.9)))
def tour_rest_hp_low(sim):
    """Tour.rest as Tour.run calls it, after a hunt stopped for 'HP low' (under the tour's 30%): low HP is
    why she's going, so it doesn't end the trip (it used to, at its first look, and she sat down in the
    basin among the monsters and died there). She drinks a potion on the way, shoots what's in her way,
    and rests on the safe platform 32."""
    return _rest(sim)


@scenario((36, 300.0), monsters="all", seed=26, hp=0.25, limit=400,
          expects=(logs("resting on platform 32"), logs("drinking a potion", "a potion on the way"),
                   no_log("couldn't reach a safe platform", "low HP doesn't end the trip to rest"),
                   check("rests on 32", lambda tr: tr["world"]["platform"] == 32),
                   check("no monster shot: she keeps clear of them", lambda tr: not tr["world"]["kills"]),
                   check("HP back", lambda tr: tr["result"]["hp"] >= 0.9)))
def tour_rest_avoid(sim):
    """tour_rest_hp_low with `--rest-avoid` (navigator.TRANSIT_AVOID): the same trip to platform 32, keeping
    clear of the monsters instead of shooting the ones in her way; low HP still doesn't end it."""
    tour = sim.tour_stack()
    tour.rest_avoid = True
    tour.rest()
    return {"report": str(tour.report), "hp": round(tour.h.look(mobs=False).hp, 3)}


@scenario((36, 300.0), monsters="all", seed=26, hp=0.25, limit=400,
          expects=(logs("resting on platform 32"), no_log("couldn't reach a safe platform", "low HP doesn't end the trip"),
                   check("rests on 32", lambda tr: tr["world"]["platform"] == 32),
                   check("HP back", lambda tr: tr["result"]["hp"] >= 0.9)))
def tour_rest_no_escort(sim):
    """tour_rest_hp_low without the fighting escort (a tour built on a navigator that avoids monsters): the
    trip to rest avoids them and only drinks potions. It used to stop at its first look on low HP, and she
    sat down in the basin."""
    h = sim.hunter(low_hp=0.3, potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=False))
    tour.rest()
    return {"report": str(tour.report), "hp": round(tour.h.look(mobs=False).hp, 3)}


@scenario((36, 300.0), monsters="all", seed=22, limit=400,
          noise=Noise(miss=0.28, walk_jitter=6.0, dot_px=True, tag_hidden=((20.0, 23.0),)),
          expects=(check("kills", lambda tr: _kills(tr) >= 1), happens("star misses")))
def tour_hunt_noisy(sim):
    """tour_hunt_basin with the real game's errors: stars missing (28%), walks off by a few units, her
    dot read to whole pixels, and her name tag hidden for 3 s mid-fight."""
    return _tour_hunt(36, 5)(sim)


# The hunt loop (behavior.hunt, `python -m gamelab tour --loop`): the tour scenarios' twins, one for each
# rule that must survive (gamelab/docs/hunt-modes.md).

def _loop(minutes: float):
    def run(sim, tour=None):
        from ..behavior.hunt import Hunt
        tour = tour or sim.tour_stack()
        r = Hunt(tour, log=sim.log).run(minutes=minutes)
        return {"report": str(r), "grounds": r.grounds, "kills": r.kills, "exp_gains": r.exp_gains,
                "ledger_kills": len(tour.h.ledger.kills), "rules": dict(sorted(r.rules.items())), "rests": r.rests,
                "hunt_stop": r.stop, "stop": tour.report.stop, "damage_taken": tour.h.report.damage_taken,
                "safe": tour.safe}
    return run


def _every_kill(tr: dict) -> bool:
    """Every kill of hers is in the ledger (the simulator never shows two between looks)."""
    return tr["result"]["ledger_kills"] == sum(tr["world"]["kills"].values())


def _back_after_rest(tr: dict) -> bool:
    log = tr["log"]
    rested = next((i for i, line in enumerate(log) if "resting on platform" in line), None)
    return rested is not None and any("at platform 36's spot" in line for line in log[rested:])


def _rule(name: str, at_least: int = 1):
    return check(f"rule '{name}' decided {at_least}+ times", lambda tr: tr["result"]["rules"].get(name, 0) >= at_least)


@scenario((36, 300.0), monsters="all", seed=22, limit=400,
          expects=(logs("at platform 36's spot (platform 36)"), _rule("shoot"), _rule("close quarters"),
                   check("ten kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 10),
                   check("every kill counted", _every_kill),
                   check("HP lost is counted", lambda tr: tr["result"]["damage_taken"] > 0),
                   check("drops picked up", lambda tr: sum(tr["world"]["picked"].values()) >= 5),
                   ends("standing by on a safe platform", key="stop")))
def loop_basin(sim):
    """tour_hunt_basin's twin: the loop hunts the basin (36) on the platform itself, where no safe spot
    reaches it. Anywhere on it she's there: waiting to reach the spot's x, close monsters interrupted her
    for good, and the visit never began."""
    return _loop(3)(sim)


@scenario((36, 300.0), monsters={17, 12, 4}, seed=25, limit=400,
          expects=(logs("leaving platform 36: nothing left here"), logs("at platform 17's spot (platform 16)"),
                   check("every kill counted", _every_kill)))
def loop_spot_off_ground(sim):
    """tour_hunt_from_safe_spot's twin: nothing in the basin, so on to platform 17, hunted from the safe
    spot on 16 above it."""
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


@scenario((13, 0.0), monsters={12}, seed=4, limit=400,
          expects=(check("platform 12 hunted", lambda tr: tr["result"]["grounds"][0] == 12),
                   _rule("lull loot"), check("drops picked up", lambda tr: sum(tr["world"]["picked"].values()) >= 20),
                   check("every kill counted", _every_kill)))
def loop_lull_loot(sim):
    """Only platform 12 has monsters, and the value map says it's the best ground. She hunts it from the
    safe platform 13 beside it; in the lulls, with nothing wanted left on it and her kills due back, she
    picks up the piles she can get to before they respawn."""
    return _loop_on_12(sim)


@scenario((36, 300.0), monsters="all", seed=26, hp=0.25, limit=400,
          expects=(logs("rest: HP"), logs("resting on platform 32"),
                   no_log("couldn't reach a safe platform", "low HP doesn't end the trip to rest"),
                   check("back to the ground after resting", _back_after_rest)))
def loop_rest_hp_low(sim):
    """tour_rest_hp_low's twin: low on HP in the basin, the loop rests on the safe platform 32 (low HP
    doesn't end the trip there) and goes back to the ground she was hunting."""
    return _loop(2)(sim)


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
          noise=Noise(miss=0.28, walk_jitter=6.0, dot_px=True, tag_hidden=((20.0, 23.0),)),
          expects=(check("kills", lambda tr: sum(tr["world"]["kills"].values()) >= 3), happens("star misses")))
def loop_noisy(sim):
    """loop_basin with the real game's errors (tour_hunt_noisy's): stars missing, walks off by a few
    units, her dot read to whole pixels, her name tag hidden for 3 s mid-fight."""
    return _loop(2)(sim)


@scenario((36, 300.0), monsters="all", seed=25, limit=400, may_die=True,
          realism=Realism(spawn_share=0.5, rival_every=20.0),
          expects=(check("only her kills in the ledger", _every_kill), happens("another player kills")))
def loop_rival(sim):
    """Another player on the map: half the monsters spawn, and a rival kills one every 20 s and takes its
    drops. Their kills give her no EXP, so the ledger has only hers."""
    return _loop(3)(sim)
