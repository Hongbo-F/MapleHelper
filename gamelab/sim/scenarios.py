"""Scenarios: gamelab's entry points run against the simulated game on a real map, each built the way
gamelab/__main__.py builds them (_hunter, Navigator, Tour), with the fakes in place. All are on Tree Dungeon, Forest
Up North II (TREE, 10002073), with her as she plays there (HER); the first map's (MAP_ID, 10000012) and West Rocky
Mountain II's (ROCKY, 10004021) were removed on 2026-10-31, when she played only Tree Dungeon.

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
import time
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from maplehelper.engine import RIGHT

from .. import telemetry
from ..control import client
from ..localization import mapdata
from ..attack import danger
from ..behavior.hunter import Hunter
from ..behavior.safety import HuntStop, SafetyStop
from ..localization.mapdata import MapModel
from ..control.navigator import Navigator
from ..control.timeline import SIM_KEYS, SIM_KEYS_MAGE, SIM_KEYS_PET
from ..behavior.tour import Tour
from .fakes import (EPOCH, Clock, FakeDetector, FakeEyes, FakeLoot, FakePetCare, FakePlayer, SimTimeout,
                    SimViolation, Trace, detector_kinds, patched)
from . import harness
from .world import (LEVEL_19, LEVEL_23, LEVEL_25, LUCKY_SEVEN, MAGE_32, NEAR_SEEN, NEAR_SPELL, OFF_SEEN, OFF_SPELL,
                    ROPE_REACH_CLIENT, TREE_DROP_MIX, Her, Noise, Realism, SimDied, World, her_climb, her_shoe)

MAP_ID = 10000012  # Victoria Road: Thicket Around the Beach II (__main__.DEFAULT_MAP)
ROCKY = 10004021   # Victoria Road: West Rocky Mountain II, the second map (from 2026-10-23)
TREE = 10002073    # Victoria Road: Tree Dungeon, Forest Up North II, the third (from 2026-10-24)
# Her keys and her (world.Her) on each map: on the first map as its golden traces were recorded with (level 19, her
# potions on a key; the harness's runs there); on West Rocky Mountain II as she plays there (level 23, HP 730, her pet drinking her potions, her
# bindings from 2026-10-23); on Tree Dungeon, Forest Up North II at level 25 (HP 774, MP 623, her stars as read there
# on 2026-10-25: world.LEVEL_25; level 24 until 2026-10-26), the same keys and pet. Any other map (the tests' synthetic
# ones): the first map's.
HER = {MAP_ID: (SIM_KEYS, LEVEL_19), ROCKY: (SIM_KEYS_PET, LEVEL_23), TREE: (SIM_KEYS_PET, LEVEL_25)}
# ... and another character than the thief, by (map, her character's id), where the process plays her (gamelab.character:
# GAMELAB_CHARACTER; the harness's `--character`): the magician on Tree Dungeon as the recordings of 2026-11-02 measured
# her (world.MAGE_32: Magic Claw as the game casts it, her stat window, her walk), with Magic Claw on Shift (SIM_KEYS_MAGE).
HERS = {(TREE, "magician"): (SIM_KEYS_MAGE, MAGE_32)}
# Her damage book as her stack had learned it when the hour of 2026-11-02 began (00:17:48: lab/characters/magician/
# damage.json, its reads up to then, of her level 31 and 32 and the one under, LEVELS_BACK), which a simulated run of hers
# starts from as live's did (behavior.damage.Book: what she reckons a monster has left); without it her stack would go by
# her profile's prior (2 x 76-267 a cast) until 15 exact reads of the run's own.
HER_BOOK = Path(__file__).with_name("her_damage_book.json")
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


def her_on(map_id: int, who: str | None = None) -> tuple:
    """Her keys and her (world.Her) on map `map_id`: character `who`'s (None: the one this process plays,
    gamelab.character) where HERS has her there, else the thief's (HER; the first map's on any other)."""
    if who is None:
        from .. import character
        who = character.active_id()
    return HERS.get((map_id, who)) or HER.get(map_id, HER[MAP_ID])


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
    think_reads: bool = False    # ... and at every read of the clock too, as it goes (fakes.Clock.reads)
    expects: tuple = ()          # (what, trace -> bool): what the scenario is there for (tests check them)
    may_die: bool = False        # unless set, every scenario also expects her alive at the end
    map_id: int = MAP_ID         # the map it's on (real_map)
    her: Her | None = None       # her HP, level and pet (world.Her); None: as she is on the map (HER)
    attack: str | None = None    # the stack's attack a throw (behavior.attacks.MODES); None: the hunter's own
    remembered: dict | None = None  # her keys as a read of the game's window left them (control.keymap.remember, in the
                                    # run's own lab), which her hunter starts with (keymap.at_start); None: none
    book: Path | None = None     # her damage book to start from (behavior.damage: her own file in the run's lab, HER_BOOK
                                 # for the magician in the harness); None: none, her profile's prior
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
        self.clock = Clock(limit=sc.limit, pace=pace, think=sc.think, reads=sc.think_reads)
        self.world = World(model, self.keys, sc.start, seed=sc.seed, monsters=sc.monsters, hp=sc.hp,
                           inventory_full=sc.inventory_full, facing=sc.facing, origin=self.clock.now,
                           drops=list(sc.drops), hanging=sc.hanging, noise=sc.noise, realism=sc.realism,
                           character=sc.her or her)
        self.world.on_attack = self._attack_told
        self.eyes = FakeEyes(self.world, self.clock, model, guard=self.guard)
        self.trace = Trace(self.clock.now)
        self.player = FakePlayer(self.world, self.clock, self.trace, self.keys)
        self.active = False
        self.files: dict[str, list[str]] = {}  # what the run wrote (under its stand-in for gamelab/lab)
        self.senses = []  # every hunter's (for the feed's record)
        self.books = []   # ... and their damage books (behavior.damage: none saved past the run, below)
        self.feed = self._feed()

    def _attack_told(self, row: tuple) -> None:
        """One of her attacks as the world resolved it (a World.attack_log row) as a telemetry 'sim_attack' record:
        the simulator's own outcome of it (kill, hit, swipe, miss, stopped, nothing), its plan's name, ground or air,
        her rise, the kind it picked, for what's worked out from a simulated run's telemetry alone (analysis.jumps: her
        jump attacks' outcomes by their jitter, `python -m gamelab jitter`); `key`, its key's time as the run's records
        have theirs (time.time(): fakes.EPOCH on)."""
        t, plan, mode, rise, kind, outcome, up = row
        telemetry.emit("sim_attack", key=round(EPOCH + t, 3), plan=plan, mode=mode, rise=rise, kind=kind,
                       outcome=outcome, up=up)

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
        from .. import character
        shape = character.active().shape  # (her Shape, as the live service's: localization.service.screen_world)
        feed = SyncFeed(WorldStep(Senses(self.frames, self.model, band=(LOOK_UP, LOOK_DOWN), shape=shape)),
                        self.frames.observe,
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
        busy, the world step's and the tracker's counts, and how many of her looks and glances came from it. (Its
        searches on their thread and their time until 2026-10-30, `searches` and `searching_s`: in every frame's pass
        since.)"""
        f = self.feed
        senses = {}
        for s in self.senses:
            for k, v in s.counts.items():
                senses[k] = senses.get(k, 0) + v
        return {"frames": f.counts["frames"], "asked": f.counts["asked"], "busy_s": round(f.counts["busy"], 3),
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
        if getattr(self, "held", False):  # (her planner held apart, Her.planned: back to her profile's for what follows)
            from ..motion import physics
            physics.reset()
        for book in self.books:  # (its save at the process's end would write into the run's lab, gone by then)
            book.path = None
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
        if self.scenario.book is not None:  # (her damage book as her stack had learned it: Scenario.book)
            from .. import paths
            book = paths.of().damage
            book.parent.mkdir(parents=True, exist_ok=True)
            book.write_text(self.scenario.book.read_text(encoding="utf-8"), encoding="utf-8")
        h = Hunter(self.eyes, keys, self.model, log=self.log, feed=self.feed, **kw)
        if (planned := getattr(self.world.character, "planned", None)) is not None:  # (her planner held apart from her
            from ..motion import physics                                            # stat window's SPEED: Her.planned)
            physics.set_planned(*planned)
            self.held = True
            self.log(f"her planner held at walk {physics.WALK_SPEED:g} and climb {physics.CLIMB_SPEED:g} units/s, the "
                     f"world's own {her_shoe(self.world.character).vmax:g} and {her_climb(self.world.character):g} "
                     f"(the world's Her.planned)")
        if getattr(h.attacks, "book", None) is not None:  # (its fits at once, not on a thread: the same run every time)
            h.attacks.book.threaded = False
            self.books.append(h.attacks.book)
        h.config_keys = self.keys
        h.jitter = random.Random(f"jitter:{self.scenario.seed}")  # (her jump attacks' jitter: the same run again)
        h.pet = FakePetCare(self.world, h.actions, self.eyes, self.clock, log=self.log)  # (no window, no OCR)
        h.pet.trusted, h.pet.on_doubt = h._pet_key_trusted, h._pet_doubted  # (as Hunter's own: her pet food doubted)
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
             "plans": sim.trace.plans, "cancels": sim.trace.cancels, "splices": sim.trace.splices, "log": sim.trace.log,
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


# --- the scenarios -----------------------------------------------------------------------------------

THROWS = ("throw", "jump_throw", "jump_back")  # the key plans that attack (control.actions)


# Set-pieces on the monsters they start with and when each is back: on their own clock (respawn_tick 0, the default
# until 2026-10-31), where the server's respawn, every scenario's since, starts the map at its cap with random points
# empty and has a kill back at the next pass: of seeds 1-13 none met theirs with it.
_CLOCK = Realism(respawn_tick=0.0)


# The hunt loop (behavior.hunt, `python -m gamelab tour --loop`), and what its runs are checked for.

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


def _made(tr: dict, move: str) -> bool:
    """The trip made a move (its kind and platforms: `jump 27->25`) that landed where it was going."""
    to = move.split("->")[1]
    return any(m == move or m == f"{move}: landed on {to}" for m in _result(tr, "moves"))


def _fed_her_pet(tr: dict) -> bool:
    return any("feeds her pet" in line for line in tr["world_log"])


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


@tree((28, 247.0), monsters="all", seed=7, limit=600,
      expects=(_rule("close quarters"), _rule("in passing"),
               check("twenty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 20),
               check("every kill counted", _every_kill), ends("time up", key="hunt_stop"),
               ends("standing by on a safe platform", key="stop"),
               check("650 EXP or more", lambda tr: tr["world"]["exp"] >= 650),
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
    kills, on 23 alone: on 4 she hunts the floor the whole 5 minutes, 77 kills.) (Seed 5 since the rebase of the
    whole screen 15 times a second onto her touches (master 73394ba, 2026-10-28: worldstate.WIDE_EVERY_S with
    navigator.TOUCH_CUTS and the world's invulnerability and MISSes): on seed 4 she went up the tower among its
    Horny Mushrooms; of seeds 1-12, 5, 6, 7, 8 and 11 meet all of this, 5 went up the tower (1, 3, 4, 9, 10), on 2
    no safe platform was reachable at 305 s and on 12 she fell out at 181 s; on 5 hurt 32 times, master's stack
    meeting all of it too (hurt 38 times). Kept on the rebase onto her keys followed and her stat window, master
    d9774e5: on 5 57 kills, hurt 32 times, 1,236 EXP; master's stack on 5 as before.) (Seed 1 since the evade work's
    review (2026-10-29: her evade step and R7's dodge, the rope fixes; rebased onto master 945f201): on seed 5 under
    forty kills; of seeds 1-13, 1, 4, 6-8 and 11 meet all of this.) (The same at the evade work's integration with
    master 8dcc2e6, the pull's detour rule and the magician's casts (2026-10-29, work/evade-int; swept again after its
    review): of seeds 1-13, 1, 4, 6-8 and 11, every seed's touches and what it misses as on work/evade.) (Twenty kills
    and 650 EXP since the server's respawn became every scenario's, 2026-10-31: the map at its cap, 22 of its 31 points,
    a kill back at a clear point at the next pass: 22 kills and 752 EXP in its 5 minutes, where every point filled and
    each back 8 s after its death gave forty and a thousand.) (With the server's respawn and the chase together
    (respawn_tick and Realism.chase every world's, 2026-10-31) of seeds 1-13 none met all of this, as the goldens run
    them: 8 went up the tower among its Horny Mushrooms, 5 were knocked out past the tower's walls (off 11's left end on
    2); with the server's respawn alone 5 of 13 met it (6 up the tower, 2 out), with the chase alone and the old respawn
    7 of 13 (its forty kills). Marked expected to fail, then looked into.) (Its falls and its back and
    forth found and fixed the same day: group 0's walls stop her as live (Realism.zero_walls, the default since), and
    the hunt loop's check that a look covers a ground goes by where monsters were searched (Scene.searched_x), where it
    had gone by where drops were, a strip at a time since the drops' rotation, so an empty ground was left on her
    patience alone and the next picked was the one just left, every 4-5 s. Of seeds 1-13 none falls out now, every run
    23-31 kills and 726-878 EXP; 3 meet all of this (1, 2, 7; on 1 under the goldens' settings and two other hash seeds)
    and 10 hunt a tower ground a while (7, 10 or 19, a minute or so), by its EXP for its danger, standing on 10 or
    shooting 19 from 20: on 5 and 6 1-2 hurting touches there, 10-13 on 23.) (Seed 7 since walls stop her in the air by
    the game's rule (body.wall_table, 2026-10-31: the planner's flights from a floating platform count the tower's walls
    now): on seed 1 a tower ground hunted; of seeds 1-13, 7 alone meets all of this, 1, 2 and 7 before.)"""
    return _loop(5)(sim)


@tree((28, 247.0), monsters="all", seed=10, limit=600, realism=Realism(zero_walls=True),
      expects=(check("fifteen kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 15),
               ends("time up", key="hunt_stop"), check("still in the tower", lambda tr: tr["world"].get("fell_out_at")
                                                       is None)))
def loop_tree_walls(sim):
    """loop_tree_run's twin in a world where the walls of foothold group 0 (the tower's) stop her whatever her group,
    off a rope too (world.Realism.zero_walls). Seed 10: with her own group's walls alone she was knocked off the rope at
    -192 (28 from the wall) and out of the map at 328 s; here she stays in the tower all 5 minutes. (Fifteen kills since
    the server's respawn became every scenario's, 2026-10-31: 19 in its 5 minutes, where the denser world before gave
    thirty.)"""
    return _loop(5)(sim)


@tree((28, 247.0), monsters="all", seed=3, limit=400, may_die=True,
      realism=Realism(camps=harness.TREE_CAMPS, zero_walls=True),
      expects=(happens("another player kills green-mushroom"), check("only her kills in the ledger", _every_kill),
               check("nobody else kills a Horny Mushroom", lambda tr: not any(
                   "another player kills horny-mushroom" in line for line in tr["world_log"]))))
def loop_tree_crowd(sim):
    """The hunt loop for 3 minutes among the other players the crowding study saw (harness.TREE_CAMPS: five camping by
    28, 21, 23, 19 and 33, each killing the Greens it hunts 2-6 s after they show, leaving the Horny Mushrooms),
    respawns at 8.8 s (MeowDB's mean), group 0's walls stopping her."""
    return _loop(3)(sim)


_CROWD = harness.PRESETS["crowd"]
_LUCKY = dataclasses.replace(LEVEL_25, stars=LUCKY_SEVEN)  # her attacks as Lucky Seven's live


@tree((25, 220.0), monsters="all", seed=3, limit=500, may_die=True, noise=_CROWD[0], her=_LUCKY,
      realism=dataclasses.replace(_CROWD[1], zero_walls=True),
      expects=(logs("next: platform 23, from platform 24 at x 221"), logs("at platform 23's spot (platform 24)"),
               _rule("shoot", 20), check("fifteen kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 15),
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
    shot 23 from the spot fewer than 20 times, for 42 kills; of seeds 1-16 all but 1 and 4 meet all of this.) (Fifteen
    kills since the server's respawn became every scenario's, 2026-10-31: 18 in its 4 minutes, where the denser world
    before gave twenty.) (Seed 3 since chains are replanned at every state as they run (2026-10-30, work/replan:
    navigator.REPLAN, KNOCK_REPLAN, MEET_WALK): on seed 2 the rule 'shoot' decided fewer than 20 times; of seeds 1-3
    only 3 meets all of this.) (Seed 1 since work/replan's rebase onto the server's respawn and the chase with where she
    stands to fight after its review (2026-10-31: D:/MapleHelper-scratch/replan/pos/fix/out/scan_A.txt, scan_B.txt): on
    seed 3 fewer than fifteen kills, the rule 'shoot' fewer than 20 times (the map at its cap, nothing came back onto 23
    after 64 s); of seeds 1-10, 1, 4, 5 and 9 meet all of this.) (Seed 1 since the monsters move as the v83 client's AI
    moves them, 2026-10-31: on seed 2 under fifteen kills; of seeds 1-13 1 and 5-12 show all of this.) (Seed 3 since the
    stack's monsters turn where the game turns them, 20 from a platform's end (monsters.walk_edges), 2026-10-31: on seed
    1 under fifteen kills; of seeds 1-13 3, 7, 8 and 10 show all of this.) (Seed 1 since her drops are swept by the
    seconds each may add to her way and a fresh pile by her is left to her pets a moment (2026-11-01, work/replan:
    hunt._allow, PET_LOOT; D:/MapleHelper-scratch/replan/loot2/eval/seeds/s16.txt): on seed 3 seven kills, the rule
    'shoot' fewer than 20 times; of seeds 1-16 1, 4, 5, 6, 9 and 12 show all of this.) (Seed 3 since his walk and climb
    by his SPEED (2026-11-02: the thief at his stat window's 151.25 and 121, world and planner, where 149 and the world's
    130 up; D:/MapleHelper-scratch/recfix/calib/seeds/slow23*.json, v23_*.json): on seed 1 ten kills; of seeds 1-16 3,
    5, 7 and 8 show all of this, at work/rec-all 1, 3, 6, 8, 9, 11-13 and 16; her kills over 1-16 12.2 a run against
    16.8 there, the walk's change (at his walk 149 and climb 119.2 16.4; at 151.25 with the climb 130 12.4), but over
    17-32 14.8 against 11.4 at 149: over 1-32 13.5 against 13.9, all of this met on 12 of 32 either way, the seeds'
    luck.)"""
    return harness._only(_loop(4), (23,))(sim)


_NEAR = dataclasses.replace(_CROWD[0], near_seen=NEAR_SEEN, off_seen=OFF_SEEN, near_spell=NEAR_SPELL,  # the
                            off_spell=OFF_SPELL)                                 # detector's misses as live


@tree((25, 220.0), monsters="all", seed=1, limit=500, may_die=True, noise=_NEAR, attack="choice",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=dataclasses.replace(_CROWD[1], zero_walls=True),
      expects=(logs("at platform 23's spot (platform 24)"), logs(": later, a horny-mushroom unseen"),
               check("twenty kills or more", lambda tr: sum(tr["world"]["kills"].values()) >= 20),
               check("every kill counted", _every_kill), check("alive", lambda tr: not tr["world"]["dead"])))
def loop_tree_23_near_misses(sim):
    """loop_tree_23 as `tour --loop` plays by default (the attack choice), with the detector's misses as live
    (`--world near-misses`: world.NEAR_SEEN on her platform, OFF_SEEN off it, each miss in a run, NEAR_SPELL and
    OFF_SPELL: the perception of 2026-11-02's from 2026-11-03, 2026-10-25's before, NEAR_SEEN_1025): her drops on 23
    wait while a Horny Mushroom the looks missed (the feed's hidden tracks, her memory's: localization.unseen) is by them
    (hunt.PILE_CLEAR), as one did live (the service's track 130: 6 touches and a fall to 29). (Seed 2 since the evade
    work's review (2026-10-29: her evade step and R7's dodge, the rope fixes; rebased onto master 945f201): on seed 1 no
    'later, a horny-mushroom unseen' line; of seeds 1-13 all but 1 and 11 meet all of this.) (The same at the evade
    work's integration with master 8dcc2e6 (2026-10-29, work/evade-int; swept again after its review): of seeds 1-13
    all but 1 and 11, every seed's touches as on work/evade.) (Seed 4 since the server's respawn and the chase together
    (respawn_tick and Realism.chase every world's, 2026-10-31): on seed 2 not all of this; of seeds 1-13, 4, 9 and 10
    meet all of it.) (Seed 3 since walls stop her in the air by the game's rule (body.wall_table, 2026-10-31: the
    planner's flights from a floating platform count the tower's walls now): on seed 4 no Horny Mushroom missed by the
    looks later; of seeds 1-13, 3 and 13 meet all of this, 4, 9 and 10 before.) (Seed 2 since the simulator's looks have
    her place as the world does (sim.fakes.TRUE_PLACE, 2026-10-31: live's localizer taken as right): on seed 3 not all
    of this; of seeds 1-13, 9 meet all of it, 4 with her dot.) (Seed 8 since the monsters move as the v83 client's AI
    moves them, 2026-10-31: on seed 2 no monster later unseen; of seeds 1-13 8, 11 and 12 show all of this.) (Seed 6
    since the stack's monsters turn where the game turns them, 20 from a platform's end (monsters.walk_edges),
    2026-10-31: on seed 8 under twenty kills; of seeds 1-13 6, 11 and 12 show all of this.) (Seed 1 since work/replan's
    loot work on master 072e1ec (2026-10-31: her HP at the user's prices, the positioning switches as first built, her
    drops' life by the live evidence: eval/seeds/near.txt): on seed 6 under twenty kills, no Horny Mushroom the looks
    missed by a pile of hers; of seeds 1-13 1, 3, 5, 7, 10 and 11 show all of this; on 1, 25 kills.) (Seed 10 since
    her drops are swept by the seconds each may add to her way and a fresh pile by her is left to her pets a moment
    (2026-11-01, work/replan: hunt._allow, PET_LOOT; D:/MapleHelper-scratch/replan/loot2/eval/seeds/s16.txt): on seed 1
    13 kills; of seeds 1-16 10, 12 and 16 show all of this, most of the others under twenty kills; on 10, 22.) (Seed 11
    since the world's touches as live (2026-11-02: a newborn touches nothing while it fades in, world.NEWBORN_SOLID_S,
    and her body for touches is the client's line, world.HER_TOUCH; D:/MapleHelper-scratch/recfix/sim/seeds/near.txt):
    on seed 10 7 kills, the rule 'wait' most of the run; of seeds 1-16 1, 11 and 16 show all of this, the others under
    twenty kills (a median 14); on 11, 26.) (Seed 1 since her knock is the client's (2026-11-02: 270 out and up, its push
    sideways added to the speed she has, landing at her walk's speed at most, world.KNOCK_VX, KNOCK_ADDS,
    KNOCK_LANDS_WALKING: ~87 units on flat ground from standing where it was ~58;
    D:/MapleHelper-scratch/recfix/sim-fix/scan4.txt): on seed 11 14 kills; of seeds 1-16 1, 5 and 16 show all of this,
    the others under twenty kills (a median 12); on 1, 24.) (Seed 2 since his walk and climb by his SPEED (2026-11-02: the
    thief at his stat window's 151.25 and 121, where 149 and the world's 130 up; D:/MapleHelper-scratch/recfix/calib/
    seeds/slow23*.json): on seed 1 ten kills; of seeds 1-16 2, 12 and 16 show all of this, at work/rec-all 1, 5 and 16
    (13.2 kills a run over 1-16 against 14.0); on 2, 21.) (Seed 1 since the touch study's two changes became the
    defaults (2026-11-03, work/sim-defaults: the detector's misses as the perception of 2026-11-02 has them, in runs,
    world.NEAR_SEEN_1102 and NEAR_SPELL_1102, where 2026-10-25's never found a monster within 25 of her; her
    invulnerability 1.5 s, world.INVULNERABLE; D:/MapleHelper-scratch/sim-defaults/seeds/scan1.txt): on seed 2, 22 kills
    but no Horny Mushroom the looks missed by a pile of hers (the new eyes miss fewer near her: the run parts from the
    old within its first second); of seeds 1-16 1, 3, 7, 8 and 15 show all of this; on 1, 26 kills, a Horny Mushroom
    unseen by her drops on 20 at 135.4 s and on 23 at 165.7 s.)"""
    return harness._only(_loop(4), (23,))(sim)


# The harness's standard thief world (`run loop-grind --map 10002073 --preset crowd --attack choice --world zero-walls
# --world near-misses --world walkers`: harness.scenario), for a set-piece in it.
_STANDARD = dict(noise=_NEAR, attack="choice", her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER),
                 realism=dataclasses.replace(_CROWD[1], **harness.WORLDS["zero-walls"], **harness.WORLDS["walkers"]))
_DECISION = re.compile(r"^\s*([\d.]+) {3}(jump-throwing|throwing|jumping back|stepping away|closing in|nothing in reach)"
                       r"(.*)$")
_PRESSED = re.compile(r"^\s*([\d.]+) {5}(?:Lucky Seven|the regular attack): ")
_TOUCHED_BY = re.compile(r"^\s*([\d.]+) touched by ")
_CLAW_HITS = re.compile(r"^\s*([\d.]+) claw swipe hits ")
CLAW_EDGE = 105.0  # units: a mushroom past this is out of the claw's box (reach.ground_sure(28): its middle's)
TOLD_S = 0.1       # s: a touch this long before her look at least is one it has (its popup shown and read)


def far_claws(tr: dict, within: float | None = None) -> list[tuple[float, str]]:
    """Her presses from the ground at a target past the claw's edge that the world's claw took, decided within `within`
    s of a touch of hers (control.navigator.TOUCHER_S by default): each claw swipe the world saw (World.log, 'claw swipe
    hits'), its press's look (her log's last attack line before it: 'Lucky Seven: ...') TOLD_S to `within` s after a
    touch of hers (World.log, 'touched by'), the decision in force then (her log's last stance line before that look) a
    throw from the ground whose nearest target it names is CLAW_EDGE or more off: [(s, the decision's line)]. A press
    the claw takes on purpose (CLAW_ONLY: one on her) names one nearer."""
    from ..control.navigator import TOUCHER_S
    within = TOUCHER_S if within is None else within
    decided = [(float(m[1]), m[2], m[3]) for line in tr["log"] if (m := _DECISION.match(line))]
    pressed = [float(m[1]) for line in tr["log"] if (m := _PRESSED.match(line))]
    touched = [float(m[1]) for line in tr["world_log"] if (m := _TOUCHED_BY.match(line))]
    out = []
    for line in tr["world_log"]:
        if (m := _CLAW_HITS.match(line)) is None:
            continue
        t = float(m[1])
        look = max((p for p in pressed if p <= t), default=None)
        before = [d for d in decided if look is not None and d[0] <= look]
        if not before or before[-1][1] != "throwing":
            continue
        gaps = [abs(int(v)) for v in re.findall(r" ([+-]\d+)", before[-1][2])]
        if gaps and min(gaps) >= CLAW_EDGE and any(TOLD_S <= look - u <= within for u in touched):
            out.append((t, f"{before[-1][0]:.3f} throwing{before[-1][2]}"))
    return out


@tree((7, -163.5), monsters=[("Horny Mushroom", 7, -150.0), ("Horny Mushroom", 7, -40.0)], seed=3, limit=30,
      may_die=True, **_STANDARD,
      expects=(check("no press from the ground at a target past the claw's edge went to the claw within TOUCHER_S of a "
                     "touch (far_claws)", lambda tr: not far_claws(tr)),
               logs("jump (touched", "a jump throw went from the air for a touch with no monster seen by her"),
               check("a kill", lambda tr: sum(tr["world"]["kills"].values()) >= 1)))
def loop_tree_touched_unseen(sim):
    """R1 of the cornered study (the user's clip of 2026-11-02 22:51, "cornered character making bad attack move":
    D:/MapleHelper-scratch/cornered): the looks are blind at her body, and a press from the ground goes to the claw on a
    monster there whatever she aimed at. The hunt loop on 7 alone in the harness's standard thief world (near-misses:
    a monster on her platform within 25 of her never found, 7% within 50, 34% to 100; walkers: one she hits chases her),
    from x -163.5 between 7's Horny Mushroom points as in episode B (22:49:39-22:49:51), one Horny Mushroom on her at
    -150, the other 123 off at -40. On 690e103 (seed 3, 10 s) the first touches her at once, and every one of her six
    presses from the ground, at ones 179 and 125 off as hidden and 183 seen, went to the claw on it 8-80 in front of her
    (swipes, no star), four of them within 2 s of a touch the looks had no monster for: 10 touches, no kill, as the
    clip's corner at x -220 (22:49:56-22:49:58). Now a touch with none about her that could have made it (behavior.fight
    .clawed: the navigator's R7 test, TOUCHER_S, TOUCHER_NEAR) has those go from a jump in place (attack.stance
    .CLAWED_AIR: 'jump (touched)'), whose star takes the one on her if its body reaches the search's window, else the
    one aimed at. (Of seeds 1-6 of this set-piece, 10 s each: off, 46 touches, 33 that hurt, 6 kills, 23 swipes; on, 36,
    31, 11, 2: D:/MapleHelper-scratch/cornered/proto.)"""
    return harness._only(_loop(1 / 6), (7,))(sim)


@tree((20, -60.0), monsters={19, 20, 21, 22}, seed=7, limit=500, may_die=True, noise=_CROWD[0], her=_LUCKY,
      realism=dataclasses.replace(_CROWD[1], zero_walls=True),
      expects=(_rule("wait"), logs("waiting at x"), check("alive", lambda tr: not tr["world"]["dead"])))
def loop_tree_wait(sim):
    """20 its only ground, hunted standing on it (no spot off it is better: attack.spots), its one Horny Mushroom's
    spawn point at x -171, respawns 30 s after a kill (preset crowd-slow), Lucky Seven: with 20 empty she waits for it
    at the safest place along 20 that keeps the spawn point in reach (hunt._wait_place), not where her last kill left
    her. (On seed 2 since the reviews of her kill's monster, 2026-10-27: on seed 3, with her kills on their monsters,
    she went for 3 mesos on 19 at x 134 and the drop planned from 18 at x 110 took the rope at 111 instead, 12 times from
    147 s to 231 s, the pile taken up again 3 s after "too many misses": a loop of the navigation's own, a drop planned
    over a rope's top, which master's stack (one miss on seed 3) didn't come to. Of seeds 1-12 every one meets it, with
    no such miss on 2, 4-8 and 10-12; master's stack meets all 12.) (Seed 1 since walls stop her in the air by the
    game's rule (body.wall_table, 2026-10-31: the planner's flights from a floating platform count the tower's walls
    now): on seed 2 not all of this; of seeds 1-13, 1, 5, 7, 10 and 13 meet all of it, five before too.) (Seed 7 since
    the monsters move as the v83 client's AI moves them, 2026-10-31: on seed 1 no wait; of seeds 1-13 7 and 12 show all
    of this.) (Seed 1 since every jump attack is jittered, those the stance keys unsure too (fight.JITTER_ALL, the
    review of 2026-11-01; D:/MapleHelper-scratch/replan/speed/fix/seeds/scan3.txt): on 7 no wait at x; of seeds 1-24,
    1, 4-6, 10-12, 17, 19 and 21-23 show all of this, with JITTER_ALL off 3, 7, 10, 16, 19, 20 and 22.) (Seed 7 since
    her knock is the client's (2026-11-02: 270 out and up, its push sideways added to the speed she has, landing at her
    walk's speed at most, world.KNOCK_VX, KNOCK_ADDS, KNOCK_LANDS_WALKING: ~87 units on flat ground from standing where
    it was ~58; D:/MapleHelper-scratch/recfix/sim-fix/scan2.txt, scanned with the push set outright, not added): on seed
    1 no wait; of seeds 1-16 7 and 11 show all of this, six others the rule 'wait' but no 'waiting at x'.)"""
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
      realism=Realism(camps=harness.TREE_CAMPS, zero_walls=True),
      expects=(logs("loop: ["), logs("next: stop"), logs("grinding:"),
               check("three stops or more", lambda tr: len(tr["result"]["grounds"]) >= 3),
               check("kills credited to two platforms or more", lambda tr: len(tr["result"]["credit"]) >= 2),
               check("no stop camped past STOP_MAX_S and a pass",
                     lambda tr: max(tr["result"]["stays"], default=0) < 120)))
def loop_tree_grind(sim):
    """`python -m gamelab tour --loop --grind --minutes 3 --map 10002073` from 23 among the crowding study's other
    players (harness.TREE_CAMPS) with respawns at 30 s (preset crowd-slow's): the loop over the map's spawn points chosen
    by its EXP a minute (behavior.patrol: the top cluster's 10, 7), what she sees pulling her off it when it pays for
    its detour against the loop, what she meets on the way shot in passing, kills credited to the platforms they died
    on. In this world she's touched about five times as often as live (README: Grinding). Until the pull's detour rule
    (2026-10-29: behavior.patrol, a pile against the loop's rate) piles pulled her to 23, 24, 19 and 20 on seed 1,
    knocked off 23 down to 25-29 and back, then standing by at the end (resting there until 2026-10-26); since, none
    pays for its detour: on her way up to 7 she's knocked about between 16 and 24 shooting in passing, then round by
    the loop to 10 and to 19, the one stop she reaches (499 EXP against 488; 473 until a pile's touches counted against
    it, the review of 2026-10-29, when a trip's look at 31.6 s had a pile pay and the choice after it, her kill time
    learned afresh, not). With the evade work (its integration, 2026-10-29: work/evade-int) the same loop and no pile
    pays either: up to 7 (stop 7 again at 41.8 s), round to 10 (102.1 s) and to 16 (168.9 s, 14 skipped as empty), the
    one stop she reaches; 469 EXP by her EXP number (510 by the world's), touched 33 times (20 on master). With the
    refill learned from what she sees (2026-10-29: work/respawn, behavior.patrol: one rate for every free point) the
    same run, but for the loop's numbers: 10, 7 at 483 EXP a minute, camping at 8 470."""
    return _grind(3)(sim)


@tree((23, 73.0), monsters="all", seed=3, limit=400, may_die=True,
      realism=Realism(camps=harness.TREE_CAMPS, zero_walls=True, drop_mix=TREE_DROP_MIX,
                      drop_takers=True),
      expects=(logs("sweep: "), happens("picks up mesos"), logs("next: stop"),
               check("four drops picked up or more", lambda tr: sum(tr["world"]["picked"].values()) >= 4),
               check("an item picked up", lambda tr: any(k != "mesos" for k in tr["world"]["picked"]))))
def loop_tree_grind_drops(sim):
    """loop_tree_grind with her drops as the perception service saw them live (the loot design of 2026-10-30:
    world.TREE_DROP_MIX, `--world map-drops`, 0.64 drops a kill, 57% mesos by its kind's amounts, piles of 1-4 drops 20
    units apart; world.HER_DROPS_LAST, `--world drop-takers`, hers taken by others as long as they lay live, until
    2026-10-31, the live evidence's and the server's expiry passes since: world.others_life, expiry_life): the loot
    sweep (behavior.sweep) on her legs between stops and before she leaves one, said in her log ("sweep: ..."), each
    pile's stand made only where no monster would meet her there. On seed 1, 12 kills of hers, 4 drops picked up (2
    mesos, a Horny and a Green Mushroom's cap), 3 lying on 10 at the end, 11 sweeps said. (Seed 2 since where she stands
    to fight (2026-10-31: the sweep's guards by her full box against theirs, back to the stop, her HP priced as the hunt
    loop's, the stops by danger: D:/MapleHelper-scratch/replan/pos/impl/out/scan_drops.txt): on seed 1 she picked up
    fewer than four drops; of seeds 1-10, 2, 3, 6 and 8 meet all of this.) (Seed 3 since work/replan's rebase onto the
    server's respawn and the chase with where she stands to fight after its review (2026-10-31:
    D:/MapleHelper-scratch/replan/pos/fix/out/scan_A.txt, scan_B.txt): on seed 2 she picked up fewer than four drops, no
    mesos; of seeds 1-10, 3, 5, 6 and 8-10 meet all of this.) (Since 2026-10-31 `--world drop-takers` has hers taken by
    others only once past her ownership, at world.TAKE_RATE a minute (others_life: the live curve counted her pets'
    pick-ups and the detector's misses): met on seed 3 still.)"""
    return _grind(3)(sim)


@tree((24, 231.0), monsters={22, 23}, seed=7, limit=240, may_die=True, noise=_CROWD[0],
      realism=Realism(zero_walls=True, drops=(("green-mushroom", "sapphire-ore", 1.0),)),
      expects=(check("no pile ahead of what's in reach (no ': the pile first')",
                     lambda tr: not any(": the pile first" in line for line in tr["log"])),
               happens("picks up sapphire-ore"),
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
    6, 8, 9 and 11-16, hurt 65 times against master's 101.) (Seed 1 since the whole screen 10 times a second with her
    band's own search beside it where a whole screen takes longer than 0.05 s (worldstate.BAND_STAND_IN_S, 2026-10-28:
    at the simulator's costs her band about 5 times a second and the whole screen 5): on seed 2 she was hurt 10 times
    and picked up 5 ores; of seeds 1-16, 1, 4-8, 11, 15 and 16 meet all of this.) (Seed 5 since the rebase onto the
    regular attack's key check, master 4558f61, with the whole screen 15 times a second (worldstate.WIDE_EVERY_S,
    2026-10-28; at the simulator's costs the same timeline as at 10, WIDE_SHARE holding a 100 ms whole screen to 5 a
    second): on seed 1 she was hurt more than five times; of seeds 1-16, 4-6, 11-13, 15 and 16 meet all of this, with
    master's stack 2, 6, 8, 9, 11, 12 and 14-16 (on 5 hurt more than five times). Her regular attack was doubted with
    her keys right on 7 of the 16, the key check finding ATTACK on Ctrl after (master's stack: 6), on 5 never.) (Seed 5
    kept at the rebase of the whole screen 15 times a second onto her touches (master 73394ba, 2026-10-28): on 5 hurt
    none of its touches, 6 ores picked up.) (Seed 13 since the server's respawn became every scenario's, 2026-10-31: the
    map at its cap from the start, a kill back at the next pass; on 5 not all of this; of seeds 1-13, 13 alone (of 1-26)
    meet all of it.) (Seed 1 since the server's respawn and the chase together (respawn_tick and Realism.chase every
    world's, 2026-10-31): on seed 13 she never picked up the ore; of seeds 1-13, 1 and 6 meet all of this, with the
    chase alone 1, 5, 8 and 12 (the old respawn).) (Seed 1 since a trip's chains are planned by the whole state's
    monsters, each walking as perception has it head at its kind's speed from the game data, with no quick look after a
    touch (2026-10-30, work/replan: navigator.trip_monsters, POST_HIT): on seed 5 she was hurt more than five times; of
    seeds 1-3, 1 and 3 meet all of this.) (Since the loot sweep, 2026-10-30, behavior.sweep: a valuable pile goes at
    once only where nothing would meet her standing on it, hunt._unsafe, so an ore among 22's Green Mushrooms isn't one
    EXP before loot weighs against what's in reach, whose word for it, "shooting first", was what this checked; that no
    pile goes ahead of what's in reach, ": the pile first", is: on seed 1 she picked up every ore, 6 of 6, each when
    nothing was in reach, and none of her touches hurt.) (Seed 2 since the review's fixes of 2026-10-30 (the frames on
    their grid at 15 a second whatever is asked, feed.run; a monster standing planned either way a while,
    navigator.STAND_UNSURE_S; her piles seen gone only by strips sure of them, worldstate.DropStrip.sure;
    D:/MapleHelper-scratch/replan/fix/out/scan_A.txt, scan_B.txt): on seed 1 she was hurt more than five times; of seeds
    1-8 met on 2-8.) (Seed 7 since walls stop her in the air by the game's rule (body.wall_table, 2026-10-31: the
    planner's flights from a floating platform count the tower's walls now): on seed 1 the ore never picked up in time;
    of seeds 1-13, 7 alone meets all of this, 1 and 6 before.) (Seed 10 since work/replan's rebase onto the server's
    respawn and the chase with where she stands to fight after its review (2026-10-31:
    D:/MapleHelper-scratch/replan/pos/fix/out/scan_A.txt, scan_B.txt): on seed 2 not every ore picked up; of seeds 1-10,
    10 alone meet all of this.) (Seed 6 since the simulator's looks have her place as the world does
    (sim.fakes.TRUE_PLACE, 2026-10-31: live's localizer taken as right): on seed 7 the ore never picked up in time; of
    seeds 1-13, 6 alone meets all of this, 7 alone with her dot.) (Seed 24 since the monsters move as the v83 client's
    AI moves them, 2026-10-31: on seed 6 the ore not picked up in the time, and hurt more than five times; none of seeds
    1-23 shows all of this, of 1-40 24, 29 and 35.) (Seed 14 since the stack's monsters turn where the game turns them,
    20 from a platform's end (monsters.walk_edges), 2026-10-31: on seed 24 the ore not picked up in the time; of seeds
    1-40 14, 33, 35, 36 and 40 show all of this.) (Seed 22 since work/replan's loot work on master 072e1ec (2026-10-31:
    her HP at the user's prices, the positioning switches as first built, her drops' life by the live evidence:
    D:/MapleHelper-scratch/replan/loot/eval/seeds/ore.txt): on seed 14 no ore picked up in the time, each shot first
    against what was in reach and from 152 s a Green Mushroom by it, hurt 6 times; of seeds 1-40 22, 23, 25 and 37 show
    all of this, the others hurt more than five times or an ore left. On 22 the 4 ores picked up at 112-124 s, hurt 4
    times of 8 touches.) (Seed 6 since the review's fixes of 2026-10-31 (a chain after a touch planned once the knock's
    slide is over, a splice's start by the chain's own physics, her landings at her own prices, her drops' life past 180
    s, her detours priced by their HP: D:/MapleHelper-scratch/replan/loot/fix2/seeds): on 22 not every ore picked up and
    hurt more than five times; of seeds 1-40, 6 and 14 meet all of this, where 4 of 40 before.) (Seed 26 since every
    jump attack is jittered, those the stance keys unsure too (fight.JITTER_ALL, the review of 2026-11-01;
    D:/MapleHelper-scratch/replan/speed/fix/seeds/scan3.txt, scan_ore2.txt): on 6 not every ore picked up; of seeds
    1-24 none meets all of this (with JITTER_ALL off 6 and 21), of 25-64, 26, 32, 36, 48, 54, 56 and 58. On 26 the six
    ores picked up at 193-196 s, touched 3 times, 2 of them evaded.) (Seed 7 since her knock is the client's (2026-11-02:
    270 out and up, its push sideways added to the speed she has, landing at her walk's speed at most, world.KNOCK_VX,
    KNOCK_ADDS, KNOCK_LANDS_WALKING: ~87 units on flat ground from standing where it was ~58;
    D:/MapleHelper-scratch/recfix/sim-fix/scan4.txt): on 26 she picked up no ore and was hurt 17 times; of seeds 1-32 7,
    11, 13 and 21 meet all of this.)"""
    return harness._only(_loop(4), (23,))(sim)


def _lucky_seven_again(tr: dict) -> bool:
    """A throw of Lucky Seven after one of the regular attack for her MP: her MP back (and the stack casts only with
    the MP for it)."""
    log = tr["log"]
    low = next((i for i, line in enumerate(log) if "< a cast's" in line), None)
    return low is not None and any("Lucky Seven:" in line for line in log[low:])


@tree((24, 231.0), monsters={22, 23}, seed=1, limit=300, may_die=True, noise=_CROWD[0], attack="choice",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER, pet_mp=None),
      realism=Realism(zero_walls=True),
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
    on a safe platform (retreat.wholly_safe: MapModel.safe, no monster's platform over it within her height), or where
    the danger with every spawn point's monster at its point is at most retreat.SAFE_DANGER, on a platform or on a
    rope."""
    from ..behavior.retreat import SAFE_DANGER, wholly_safe
    w = sim.world
    d = danger.of_map(sim.model, danger.Her(w.character.max_hp, w.character.level))
    if w.mode == "rope" and w.rope is not None:
        return d.on_rope(w.rope, w.y, None) <= SAFE_DANGER
    if w.mode != "ground" or w.plat is None:
        return False
    return wholly_safe(sim.model, w.plat.id) or d.at(w.plat.id, w.x, None) <= SAFE_DANGER


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


@tree((33, 100.0), monsters="all", seed=1, limit=400, expects=LIE_DETECTOR_EXPECTS + (
    _went("a rope as safe: no safe platform near"),
    check("onto the rope at 146 from 32", lambda tr: tr["result"]["then"]["mode"] == "rope"
          and tr["result"]["then"]["x"] == 146)))
def loop_tree_lie_detector(sim):
    """`tour --loop --map 10002073`, the Lie Detector seen while she hunts the floor (33): the hunt stops at once (the
    plan running cut, nothing of hers after it); no safe platform is near (28, the nearest place no monster reaches, is
    past retreat.NEAR_S by route with the floor's crowd), so up onto the rope at 146 from 32, at its safest height
    (attack.ropes), avoiding the monsters, her moves alone; a retreat plan past the player's 12 s is refused. (From the
    floor's crowd, of seeds 1-12 seen at 40, 75, 110 and 150 s on 2026-10-24, 83% were on a safe place at 10 s; from
    24, shooting 23, 47 of 48: loop_tree_lie_detector_23.) The moment is set up (2026-10-31): she starts on the floor at
    x 100 and the Lie Detector is seen 2 s into the hunt loop. Until then the hunt loop started from 28 at x 247 (as
    loop_tree_run) and it was seen 110 s in, wherever she was by then: at master 35a1196 (on its monsters' own clock,
    _CLOCK) 6 of seeds 1-13 met all of this, and the seed had moved 7 times since 2026-10-24. On seed 4 up onto the rope
    at 146, on it 10 s after the sighting; in the default world all of seeds 1-13 but 7 meet all of this (on 7 back onto
    28, safe enough), and seed 4 with the chase off and with the old respawn clock too. (Seed 1 since the review's fixes
    of 2026-10-31 (a chain after a touch planned once the knock's slide is over: D:/MapleHelper-scratch/replan/loot/fix2/
    seeds): on 4 knocked back off 32 twice on her way to the rope, on it 10.9 s after the sighting; of seeds 1-16 all but
    4 and 16 meet all of this, 12 of 1-13 as before.)"""
    return lie_detector(2.0)(sim)


@tree((30, 100.0), monsters="all", seed=1, limit=400, expects=LIE_DETECTOR_EXPECTS + (
    _went("safe enough"), check("on 28", lambda tr: tr["result"]["then"]["platform"] == 28)))
def loop_tree_lie_detector_28(sim):
    """As loop_tree_lie_detector, on her way down from 28: back onto 28, which no monster reaches (its danger 0 by
    attack.danger, though not MapModel.safe), close by route, safe enough. The moment is set up (2026-10-31): she starts
    on 30 at x 100, just under 28 (where her way down from 28 takes her: the drop onto 30, then the rope at 146 down to
    32), and the Lie Detector is seen 2 s into the hunt loop. Until then it was seen 40 s into the hunt loop from 28 at
    x 247, where she was by then by her run's luck: on her way down from 28 on 2 of seeds 1-13 at master 35a1196 (on its
    monsters' own clock, _CLOCK), and the seed had moved 13 times since 2026-10-24. On seed 1 back onto 28 at x 227, on
    it 10 s after the sighting; in the default world all of seeds 1-13 but 10 meet all of this (on 10 she goes to 27, a
    safe platform), on the old respawn clock all 13, and seed 1 with the chase off too."""
    return lie_detector(2.0)(sim)


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
    missed): on seed 4 she was still on her way 10 s on, not onto 25; of seeds 1-16 all but 4 meet all of this.)
    (Seed 2 since chains are replanned at every state as they run (2026-10-30, work/replan: navigator.REPLAN,
    KNOCK_REPLAN, MEET_WALK): on seed 1 she went to the rope at x 146 from 32, a rope as safe, not onto 25; of seeds
    1-3, 2 and 3 meet all of this.)
    (Seed 1 since the review's fixes of 2026-10-30 (the frames on their grid at 15 a second whatever is asked,
    feed.run; a monster standing planned either way a while, navigator.STAND_UNSURE_S; her piles seen gone only by
    strips sure of them, worldstate.DropStrip.sure; D:/MapleHelper-scratch/replan/fix/out/scan_A.txt, scan_B.txt):
    on seed 2 she retreated to 28, safe enough, not onto 25; of seeds 1-8 met on 1 and 3-8.)"""
    return harness._only(lie_detector(60.0), (23,))(sim)


# Her as `tour --loop` plays by default (the feed, the attack choice): Lucky Seven's key casting it at its MP, and each
# star's damage drawn (harness.ATTACK_HER), the bars over hurt monsters read as they show; group 0's walls
# stopping her (in the default world a touch knocked her out of the map past the tower's wall on 3 of seeds 1-12 of
# loop_tree_lie_detector_grind, and 2 of loop_tree_lie_detector's twin as live ran it, a simulator effect).
_LIVE = dict(attack="choice", her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER),
             realism=Realism(zero_walls=True))


@tree((23, 73.0), monsters="all", seed=2, limit=400, **_LIVE, expects=LIE_DETECTOR_EXPECTS + (
    logs("loop: ["), logs("Lucky Seven:")))
def loop_tree_lie_detector_grind(sim):
    """`tour --loop --grind --map 10002073` as it runs by default (the feed, the attack choice) from 23, as
    loop_tree_grind, the Lie Detector seen 75 s in: grinding is the hunt loop going round the map (behavior.grind.Grind,
    a Hunt), so it stops as the hunt loop does, at the next look or plan, and she retreats to a safe place within 10 s,
    her moves alone: on seed 1 down onto 25, a safe platform, at 5.6 s. (Of seeds 1-12, all 12 are on a safe place at
    10 s, 10 on 25 and 2 on 28; resting on 25 when it was seen, as she could until 2026-10-26, she stayed, the rest's
    next look up to 5 s after it.) (With the refill learned from what she sees, 2026-10-29: work/respawn, the grind
    before it goes otherwise: on seed 1 onto 25 at x 119, 4.9 s after the sighting, safe at 10 s.) (Seed 2 since
    perception at 15 frames a second, one pass a frame (2026-10-30): on seed 1 a Horny Mushroom touched her on her
    walk-off from 20 1.2 s into her retreat to 25, which ended there: on 20 at 10 s, not safe; on 2 to the rope at x 111
    by 18 (no safe platform near), safe at 10 s.)"""
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
    took her to 1.1%. Seeds 1-12 under each rule: README, The safety stop in HP.) (Seed 1 since the stack's monsters
    turn where the game turns them, 20 from a platform's end (monsters.walk_edges), 2026-10-31: on seed 2 an attack or
    the mouse after the stop; of seeds 1-13 all but 2-4 and 12 show all of this.) (Seed 2 since his walk and climb
    by his SPEED (2026-11-02: the thief walks his stat window's 151.25 and climbs 121, the world as his planner, where 149
    and the world's 130 up; D:/MapleHelper-scratch/recfix/calib/seeds/scenarios.json): on seed 1 an attack or the mouse
    after the stop; of seeds 1-16 all but 1, 3, 14 and 15 show all of this.)"""
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


@tree((23, 73.0), monsters="all", seed=3, limit=400, **dict(_LIVE, realism=Realism(zero_walls=True, darts=60)),
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
    stop due in the hunt loop has the key bindings read first (her Lucky Seven's key moved does the same): due at 72.4
    s, on 25 (a safe platform: at once), the window read, Lucky Seven on Shift as configured: the stop at 76.1 s, her
    moves alone after it. (Seed 1 since the server's respawn became every scenario's, 2026-10-31: the map at its cap
    from the start, a kill back at the next pass; on 4 not all of this; of seeds 1-13, 1, 2, 5-7 and 9-13 meet all of
    it.) (Seed 2 since the simulator's looks have her place as the world does (sim.fakes.TRUE_PLACE, 2026-10-31: live's
    localizer taken as right): on seed 1 not on a safe place once her retreat was over; of seeds 1-13, 11 meet all of
    this, 10 with her dot.) (Seed 1 since the monsters move as the v83 client's AI moves them, 2026-10-31: on seed 2 not
    on a safe place once her retreat was over; of seeds 1-13 all but 2 and 10 show all of this.) (Seed 2 since her knock
    is the client's (2026-11-02: 270 out and up, its push sideways added to the speed she has, landing at her walk's
    speed at most, world.KNOCK_VX, KNOCK_ADDS, KNOCK_LANDS_WALKING: ~87 units on flat ground from standing where it was
    ~58; D:/MapleHelper-scratch/recfix/sim-fix/scan3.txt, scanned with the push set outright, not added): on seed 1 not
    on a safe place once her retreat was over; of seeds 1-16 all but 1, 7 and 10 show all of this.) (Seed 1 since his
    walk and climb by his SPEED (2026-11-02: the thief at his stat window's 151.25 and 121, where 149 and the world's 130
    up; D:/MapleHelper-scratch/recfix/calib/seeds/scenarios.json): on seed 2 not on a safe place once her retreat was
    over; of seeds 1-16 all but 2 and 12 show all of this.) (Seed 3 since her invulnerability after a touch is the
    sources' 1.5 s (2026-11-03, work/sim-defaults: world.INVULNERABLE, where 1.45; the run parts from the old at 1.45 s,
    a Horny Mushroom's touch again now at 1.50; D:/MapleHelper-scratch/sim-defaults/seeds/scan1.txt): on seed 1 her
    retreat to the rope at x 111 timed out on 17 (not a safe place); of seeds 1-16 all but 1, 2, 13 and 14 show all of
    this. On 3: out of darts' stop at 137.4 s, her retreat to 25 at x 199 arrived in 7.0 s, paged.)"""
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

# The regular attack's doubt set up, not left to a run's luck (2026-10-31): the map's Green Mushrooms there at the start
# have a tenth of their HP left, their bars up (world.Realism.hurt, as the crowd leaves some live), so her attack choice
# finishes the first she meets with the regular attack (attack.choice.finishes: Lucky Seven would be overkill), and her
# presses are judged from her first fight on. Until then a press to finish came only of a cast leaving a monster low, a
# few times a run: each of the four below met all of its checks on 5 of seeds 1-13 at master 35a1196, and their seeds
# had moved at every change of the world since 2026-10-28 (loop_tree_grind_attack_key's 4 times).
_HURT = (("green-mushroom", 0.1),)
_HURT_LIVE = dict(_LIVE, realism=Realism(zero_walls=True, hurt=_HURT))


@tree((23, 73.0), monsters="all", seed=3, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("attack", "a"),), hurt=_HURT)),
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
    to finish a monster do nothing; 5 of them in a row judged without effect, the doubt; with a monster near, away to a
    safe place first, where a monster within 250 of her on another platform (it can't touch her) doesn't put the check
    off (keycheck.near_her, from the afternoon of 2026-10-28; on master's stack then put off); her stars asked, the
    backslash, the window read (ATTACK on A, preset 1), Esc, closed; her regular attack A from then on, remembered,
    paged once; kills with it after. (Until the review of 2026-10-28 the window opened with a Green Mushroom near on
    23's side: nothing looked again after the retreat.) The monster to finish is set up (_HURT: the map's Green
    Mushrooms start at a tenth of their HP): on seed 1 the doubt at 15.1 s, ATTACK followed to A at 19.2 s, 18 kills
    after; of seeds 1-13 all but 11 meet all of this (on 11 the Green she presses at stays within her claw's reach,
    where no press is judged: the doubt never comes in its 169 s), and seed 1 with the chase off and with the old
    respawn clock too. (Seed 2 since work/replan's loot work on master 072e1ec (2026-10-31: her HP at the user's prices,
    the positioning switches as first built, her drops' life by the live evidence: eval/seeds/keys.txt): on seed 1 the
    doubt came at 131.2 s, too late for the check in its 2 minutes; of seeds 1-13 all but 1, 5 and 10 (no doubt in the
    time) meet all of this; on 2 the doubt at 24.2 s, ATTACK followed to A at 28.9 s, 20 kills after.) (Seed 3 since
    her two jump attacks, jittered (2026-11-01: reach.TOP_KEY_MS, fight.jittered): on 2 no doubt in its 2 minutes; of
    seeds 1-13 all but 1, 2 and 5 meet all of this; on 3 the doubt at 9.8 s, the window read from 11.5 s,
    ATTACK followed to A, 24 kills after.)"""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=3, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("attack", None),), hurt=_HURT)),
      expects=KEY_EXPECTS + (
          logs("ATTACK KEY: THE GAME HAS ATTACK ON NO KEY"),
          check("paged once, about the attack key", lambda tr: len(tr["result"]["pages"]) == 1
                and "attack key" in tr["result"]["pages"][0][1]),
          check("no regular attack after it", lambda tr: not tr["result"]["attack_keys_after"]),
          logs("the regular attack off this run")))
def loop_tree_attack_unbound(sim):
    """As loop_tree_grind_attack_key, the game's Attack on no key: the same doubt and check, the window read with ATTACK
    on none (no key of the keyboard carries it), so no more regular attacks this run: Lucky Seven only (her throws that
    would have been the regular attack: "the regular attack off this run"), paged once; remembered so: her next run's
    start says it and tries the regular attack again (control.keymap.at_start; until the review of 2026-10-28 every
    later run started with it off, nothing to doubt it again). The monster to finish is set up as there (_HURT): on
    seed 1 the doubt at 15.1 s, ATTACK on none at 19.2 s; of seeds 1-13 all but 11 meet all of this (as there), and
    seed 1 with the chase off and with the old respawn clock too. (Seed 2 since work/replan's loot work on master
    072e1ec (2026-10-31: her HP at the user's prices, the positioning switches as first built, her drops' life by the
    live evidence), as there: on seed 1 the doubt at 131.2 s, too late; of seeds 1-13 the same ten meet all of this; on
    2 the doubt at 24.2 s, ATTACK on none at 28.9 s.) (Seed 3 since her two jump attacks, jittered (2026-11-01), as
    there: on 2 no doubt in its 2 minutes; of seeds 1-13 the same ten meet all of this; on 3 the doubt at 9.8 s, the window
    read from 11.5 s, ATTACK on none, 23 kills after.)"""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=5, limit=400, may_die=True,
      **dict(_HURT_LIVE, noise=Noise(regular_miss=1.0, miss_unseen=1.0)),
      expects=KEY_EXPECTS + (
          logs("ATTACK KEY: ATTACK IS ON CTRL AS CONFIGURED"),
          check("nobody paged", lambda tr: not tr["result"]["pages"]),
          check("the hunt went on: kills after it", lambda tr: tr["result"]["kills_after"] >= 3),
          check("no stop", lambda tr: tr["result"]["stopped"] is None)))
def loop_tree_attack_misses(sim):
    """Her keys right, every regular star missing (Noise.regular_miss) and its MISS popup not seen (Noise.miss_unseen:
    under other numbers, say): 5 presses without effect, the doubt, the window read with ATTACK on Ctrl as configured:
    the misses were something else; that target gets no more regular attacks, the hunt goes on, nobody paged. A doubt
    within 300 s of it would hold the regular attack at its targets, with no check. (loop_tree_attack_misses_seen: the
    MISSes read, no doubt.) The monster to finish is set up as in loop_tree_grind_attack_key (_HURT): on seed 3 the
    doubt at 60.1 s, ATTACK on Ctrl at 69.9 s, 7 kills after; of seeds 1-13 all but 1 (no doubt) and 12 (2 kills
    after) meet all of this, and seed 3 with the chase off and with the old respawn clock too (on that clock seed 2
    has under 3 kills after the check, seed 4 no doubt). (Seed 1 since walls stop her in the air by the game's rule
    (body.wall_table, 2026-10-31: the planner's flights from a floating platform count the tower's walls now): on seed 3
    no regular attack's doubt; of seeds 1-13, 10 meet all of this (1, 2, 4-6, 9-13), and on 1 with the chase off and on
    the old respawn clock too.) (Seed 2 since the monsters move as the v83 client's AI moves them, 2026-10-31: on seed 1
    not the key bindings' check; of seeds 1-13 all but 1 show all of this.) (Seed 4 since the stack's monsters turn
    where the game turns them, 20 from a platform's end (monsters.walk_edges), 2026-10-31: on seed 2 not the key
    bindings' check; of seeds 1-13 4-11 and 13 show all of this.) (Seed 5 since the rope hang beside the world's touches
    and knock as live (work/rec-all, the integration of 2026-11-02; D:/MapleHelper-scratch/recfix/integrate/diag/
    scan1.txt): on seed 4 she was paged; of seeds 1-16, 2, 3, 5-7, 9, 11, 12 and 14-16 show all of this. On 5 the doubt
    at 14.0 s.)"""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=2, limit=400, may_die=True,
      **dict(_HURT_LIVE, noise=Noise(regular_miss=1.0)),
      expects=(check("her attack's MISS read over her regular presses' targets",
                     lambda tr: tr["result"]["watch"].get("missed", 0) >= 3),
               check("no doubt: the MISSes say her key works", lambda tr: not _said(tr, "THE REGULAR ATTACK TOOK NO")),
               check("the window never opened", lambda tr: not tr["result"]["window"]),
               check("nobody paged", lambda tr: not tr["result"]["pages"]),
               check("no stop", lambda tr: tr["result"]["stopped"] is None)))
def loop_tree_attack_misses_seen(sim):
    """As loop_tree_attack_misses, every regular star missing, its MISS popup over the monster read (perception.misses;
    the user, 2026-10-28: "Yes, read it"): each press with one over its target first seen 0.5-1.6 s after its key
    took effect (keycheck: "missed"), so no run of 5 without effect: no doubt, no window, nobody paged. The presses are
    set up as in loop_tree_grind_attack_key (_HURT): on seed 2, 64 presses judged "missed", no doubt; of seeds 1-13 all
    but 3 and 8 meet all of this (on those, 19 and 31 read missed, and 5 in a row unread all the same: the doubt), and
    seed 2 with the chase off and with the old respawn clock too. (Seed 1 since the simulator's looks have her place as
    the world does (sim.fakes.TRUE_PLACE, 2026-10-31: live's localizer taken as right): on seed 2 the MISSes read said
    her key works; of seeds 1-13, 11 meet all of this, 12 with her dot.) (Seed 2 since the review's fixes of 2026-10-31
    (D:/MapleHelper-scratch/replan/loot/fix2/seeds): on 1 her MISSes read over fewer than 3 of her regular presses'
    targets; of seeds 1-16 all but 1, 6 and 16 meet all of this.)"""
    return key_check_run(2)(sim)


@tree((23, 73.0), monsters="all", seed=11, limit=400, may_die=True, attack="regular",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=Realism(zero_walls=True, darts=40, chase=False),
      expects=(logs("THE REGULAR ATTACK TOOK NO EFFECT 5 TIMES IN A ROW"), logs("OUT OF DARTS: 20 Lucky Seven presses"),
               check("stopped out of darts", lambda tr: "OUT OF DARTS" in (tr["result"]["stopped"] or "")),
               check("by out of darts' own watch, not the stars' test", lambda tr: "Lucky Seven presses over" in (
                   tr["result"]["stopped"] or "") and "regular attack took no effect" not in tr["result"]["stopped"]),
               logs("her stars: MAYBE OUT"),
               check("paged twice: her stars maybe out (the regular attack off), then out of darts",
                     lambda tr: len(tr["result"]["pages"]) == 2 and "STARS MAY BE OUT" in tr["result"]["pages"][0][2]
                     and "out of darts" in tr["result"]["pages"][1][1])))
def loop_tree_attack_no_darts(sim):
    """`tour --loop --attack regular` (the regular attack alone: out of darts' own watch counts Lucky Seven's presses, so
    it doesn't fire while she throws it) with 40 throwing stars: gone at about 50 s, her regular presses only swing the
    claw, nothing at range; 5 in a row without effect at a Green Mushroom she had hurt, the doubt at 107.4 s; away to 25
    (its retreat timed out, 8.2 s: later checks wait where she is), a safe platform: the check there, her stars' test,
    5 casts that took no MP and hurt nothing: "MAYBE OUT" (124.9 s); the window, ATTACK on Ctrl as configured: no more
    regular attacks this run, paged; Lucky Seven only, which takes no MP with no stars (the world's); out of darts' own
    stop at 20 such presses (146.3 s: her retreat, paged, the recording kept). Found after the rebase onto her touches
    (2026-10-28): her stars' test's first cast at a Horny Mushroom at 35% did nothing, but another at 3.5% walked over it
    and the looks' tracks swapped them, so its bar "lowered": "she has them", the regular attack held at one target only,
    and no stop for the rest of the run; a target another of its kind crossed tells nothing now (keycheck._crossed).
    Until the afternoon of 2026-10-28 the check was put off (a Green Mushroom on 24 within 250 of her on 25); until the
    review of 2026-10-28 her stars' test stopped it at 5 casts without MP, which live, with her stars in her bag,
    7-8 in a row had (behavior.keycheck). (Seed 5 since the rebase of the whole screen 15 times a second
    (worldstate.WIDE_EVERY_S) onto her keys followed and her stat window (master d9774e5, 2026-10-28): on seed 1 no
    5 regular presses in a row went without effect in its 4 minutes, so neither the doubt nor out of darts' stop; of
    seeds 1-12, 5 and 7 meet all of this, master's stack on 5 no doubt. On 5: the doubt at 108.8 s, away to 25 (7.6
    s), MAYBE OUT at 125.8 s, the window read and closed at 126.9-127.6 s, ATTACK on Ctrl: Lucky Seven only, paged;
    out of darts' stop after 20 such presses, 36 s, paged.) (Seed 24 since the pull's detour rule (2026-10-29:
    behavior.patrol, a pile against the loop's rate): on seed 5 no 5 regular presses in a row went without effect in its
    4 minutes, so neither the doubt nor out of darts' stop; of seeds 1-24, 24 alone meets all of this, 3, 4, 6, 8, 10
    and 17 doubt and stop with her stars' test not MAYBE OUT. On 24: the doubt at 113.7 s, a Horny Mushroom at 4%; MAYBE
    OUT at 121.2 s, the window read and closed at 121.6-122.3 s, ATTACK on Ctrl: Lucky Seven only, paged; out of darts'
    stop at 144.5 s, paged.) (Seed 5 again since the evade work's integration (2026-10-29: work/evade-int, her evade step
    and R7's dodge with the pull's detour rule): on 24, and on 3 (the evade work's pick without the pull's rule), out of
    darts' stop never came in its 4 minutes; of seeds 1-24, 5, 8 and 12 meet all of this. On 5: the doubt at 75.1 s (a
    Green and a Horny Mushroom), MAYBE OUT at 82.5 s, the window read and closed at 83.7-84.4 s, ATTACK on Ctrl: Lucky
    Seven only, paged; out of darts' stop at 100.9 s, paged.) (Seed 8 since perception at 15 frames a second, one pass a
    frame, with the drops' rotation a strip in view every 0.2 s (worldstate.DROPS_EVERY_S, 2026-10-30): on seed 5 her
    stars' test had no monster in reach to cast at (her stars: not known), so neither MAYBE OUT nor out of darts' stop
    in its 4 minutes; of seeds 1-8, 8 alone meets all of this (on 1, 2 and 4 out of darts' stop without MAYBE OUT). On
    8: the doubt at 79.8 s, a Green Mushroom at 63%, the check put off (it stayed near her), MAYBE OUT at 100.1 s, the
    window read and closed at 102.3-103.7 s, ATTACK on Ctrl: Lucky Seven only, paged; out of darts' stop at 105.3 s,
    paged.) (Its monsters walk as with nobody there, hit or not, since the world's monsters chase her once she hits them
    (Realism.chase off, 2026-10-31): with their chase none of seeds 1-40 shows it; on 8 the first doubt at 100.7 s found
    ATTACK on Ctrl as configured with her stars' test not MAYBE OUT, so no Lucky Seven-only run nor out of darts' stop,
    175 touches in its 344 s. To redo with the chase.) (Seed 3 since a trip's chains are planned by the whole state's
    monsters, each walking as perception has it head at its kind's speed from the game data, with no quick look after a
    touch (2026-10-30, work/replan: navigator.trip_monsters, POST_HIT): on seed 8 her stars' test didn't say MAYBE OUT;
    of seeds 1-16 only 3 meets all of this.) (Seed 6 since the loot sweep (2026-10-30, behavior.sweep: the hunt loop's
    drops one rule, priced in her time): on seed 3 no 20 Lucky Seven presses in a row came to out of darts' stop; of
    seeds 1-6, 6 alone meet all of this.) (Seed 3 since the review's fixes of 2026-10-30 (the frames on their grid at 15
    a second whatever is asked, feed.run; a monster standing planned either way a while, navigator.STAND_UNSURE_S; her
    piles seen gone only by strips sure of them, worldstate.DropStrip.sure;
    D:/MapleHelper-scratch/replan/fix/out/scan_A.txt, scan_B.txt): on seed 6 out of darts wasn't told by its own watch;
    of seeds 1-8 met on 3 alone.) (Seed 16 since where she stands to fight (2026-10-31: the sweep's guards by her full
    box against theirs, back to the stop, her HP priced as the hunt loop's, the stops by danger;
    D:/MapleHelper-scratch/replan/pos/impl/out/scan_darts*.txt): on seed 3 out of darts' own stop never came in its 4
    minutes; of seeds 1-30 met on 16 and 20, her stars' test not MAYBE OUT on 27 of the 28 others.) (Seed 3 since
    work/replan's rebase onto the server's respawn and the chase with where she stands to fight after its review
    (2026-10-31: D:/MapleHelper-scratch/replan/pos/fix/out/scan_A.txt, scan_B.txt): on seed 16 her stars' test never
    said MAYBE OUT, so she was paged once; of seeds 1-10, 3 and 7 (and of 1-30 before this, 16 and 20: fragile, the key
    check's chain of events a run's luck) meet all of this.) (Seed 6 since the simulator's looks have her place as the
    world does (sim.fakes.TRUE_PLACE, 2026-10-31: live's localizer taken as right): on seed 8 her stars' test not MAYBE
    OUT; of seeds 1-13, 6, 7, 9 and 11 meet all of this, four with her dot too.) (Seed 2 since the monsters move as the
    v83 client's AI moves them, 2026-10-31: on seed 6 her stars never maybe out; of seeds 1-13 2 alone shows all of
    this.) (Seed 4 since the stack's monsters turn where the game turns them, 20 from a platform's end
    (monsters.walk_edges), 2026-10-31: on seed 2 her stars never maybe out; of seeds 1-13 4, 6, 7, 10 and 12 show all of
    this.) (Seed 1 since the review's fixes of 2026-10-31 (a chain after a touch planned once the knock's slide is over,
    a splice's start by the chain's own physics, her landings at her own prices, her drops' life past 180 s, her detours
    priced by their HP: D:/MapleHelper-scratch/replan/loot/fix2/seeds): on 4 neither her stars' test's MAYBE OUT nor the
    page; of seeds 1-16, 1 and 8 meet all of this.) (Seed 2 since her drops are swept by the seconds each may add to
    her way and a fresh pile by her is left to her pets a moment (2026-11-01, work/replan: hunt._allow, PET_LOOT;
    D:/MapleHelper-scratch/replan/loot2/eval/seeds/s16.txt): on 1 her stars' test cast nothing in its 10 s (not on a
    platform: "her stars: not known"), so no MAYBE OUT and one page, out of darts'; of seeds 1-16, 2 and 12 meet all of
    this.) (Seed 6 since the loot review's fixes (2026-11-01: PET_LOOT off, the sweep's allowances bounded; failing on
    666c54b) and her two jump attacks, jittered (reach.TOP_KEY_MS, fight.jittered;
    D:/MapleHelper-scratch/replan/speed/rec/seeds/no_darts.txt): on 2 her stars' test had no monster in reach to cast
    at ("her stars: not known"), so no MAYBE OUT and one page, out of darts'; of seeds 1-24, 6 and 13 meet all of this.
    On 6: the doubt at 67.2 s, a Horny Mushroom at 26%; MAYBE OUT at 74.9 s, ATTACK on Ctrl as configured at 76.8 s:
    Lucky Seven only, paged; out of darts' stop at 93.7 s, paged.) (Seed 12 since every jump attack is jittered, those
    the stance keys unsure too (fight.JITTER_ALL, the review of 2026-11-01;
    D:/MapleHelper-scratch/replan/speed/fix/seeds/scan3.txt): on 6 no doubt of the regular attack before a key check's
    retreat at 174 s; of seeds 1-24, 12 alone meets all of this (with JITTER_ALL off 6 and 13). On 12: the doubt at
    61.1 s, a Horny Mushroom at 31%; MAYBE OUT at 68.7 s, ATTACK on Ctrl as configured at 69.8 s, paged; out of darts'
    stop at 83.9 s, paged.) (Seed 3 since the world's touches as live (2026-11-02: a newborn touches nothing while it
    fades in, world.NEWBORN_SOLID_S, and her body for touches is the client's line, world.HER_TOUCH;
    D:/MapleHelper-scratch/recfix/sim/seeds/no_darts.txt): on 12 her stars' test never said MAYBE OUT; of seeds 1-24 3,
    5, 7, 13 and 14 meet all of this. On 3: the doubt at 76.2 s, a Horny Mushroom at 78%; MAYBE OUT at 83.9 s, ATTACK on
    Ctrl as configured at 88.6 s, paged at 89.3 s; out of darts' stop at 116.4 s, paged.) (Seed 6 since her knock is the
    client's (2026-11-02: 270 out and up, its push sideways added to the speed she has, landing at her walk's speed at
    most, world.KNOCK_VX, KNOCK_ADDS, KNOCK_LANDS_WALKING: ~87 units on flat ground from standing where it was ~58;
    D:/MapleHelper-scratch/recfix/sim-fix/scan4.txt): on 3 her stars' test never said MAYBE OUT; of seeds 1-16 6, 8 and
    14 meet all of this. On 6: the doubt at 110.2 s, a Green Mushroom at 47%.) (Seed 11 since the rope hang beside the
    world's touches and knock as live (work/rec-all, the integration of 2026-11-02; D:/MapleHelper-scratch/recfix/
    integrate/diag/scan1.txt): on 6 her stars' test never said MAYBE OUT; of seeds 1-16, 11 and 14 meet all of this. On
    11 the doubt at 129.4 s, a Green Mushroom at 84-90%.) (Seed 9 since his walk and climb by his SPEED (2026-11-02: the
    thief at his stat window's 151.25 and 121, where 149 and the world's 130 up; D:/MapleHelper-scratch/recfix/calib/
    seeds/scenarios.json): on 11 her stars' test never said MAYBE OUT; of seeds 1-16, 9, 13 and 16 meet all of this. On
    9 the doubt at 106.7 s, a Green Mushroom.) (Seed 14 since her hang under a rope's top waits longer for a walker and
    takes a touch worth less than the wait (work/rec-rope2, 2026-11-02: navigator.HANG_LONGER, HANG_TAKE;
    D:/MapleHelper-scratch/recfix/rope2/scan_no_darts.txt): on 11 her stars' test never said MAYBE OUT; of seeds 1-16, 8
    and 14 meet all of this. On 14: the doubt at 57.4 s, a Green Mushroom at 25%; MAYBE OUT at 95.2 s, ATTACK on Ctrl as
    configured at 99.8 s, paged; out of darts' stop at 112.3 s, paged.) (Seed 4 since the two together (work/rec-all,
    the merge of work/rec-panel and work/rec-rope2, 2026-11-02; D:/MapleHelper-scratch/recfix/measure/
    scan_no_darts.txt): on 9 and on 14 her stars' test never said MAYBE OUT; of seeds 1-16, 4, 6, 13 and 16 meet all of
    this. On 4: the doubt at 62.7 s, a Horny Mushroom; MAYBE OUT at 70.6 s, ATTACK on Ctrl as configured at 72.4 s,
    paged; out of darts' stop at 88.5 s, paged.) (Seed 13 since the final review of 2026-11-02: her hang under a rope's
    top the 2 s hang alone again, navigator.HANG_LONGER and HANG_TAKE off, and a monster the fight takes for dead
    holding her at a stop no more, grind's _held; D:/MapleHelper-scratch/recfix/final-fix/scan_no_darts.txt): on 4 she
    was never paged twice, on 9 (the 2 s hang's seed of work/rec-panel) her stars' test never said MAYBE OUT; of seeds
    1-16, 13 and 16 meet all of this. On 13: the doubt at 54.8 s, a Horny Mushroom at 81%; MAYBE OUT at 62.4 s, ATTACK
    on Ctrl as configured at 64.3 s, paged; out of darts' stop at 77.7 s, paged at 80.3 s.) (Seed 11 since her
    invulnerability after a touch is the sources' 1.5 s (2026-11-03, work/sim-defaults: world.INVULNERABLE, where 1.45;
    the run parts from the old at 1.45 s, a Green Mushroom's touch again now at 1.50;
    D:/MapleHelper-scratch/sim-defaults/seeds/scan1.txt): on 13 her stars' test cast nothing (her retreat to the rope
    at -80, not on a platform: "her stars: not known"), so no MAYBE OUT and one page, out of darts'; of seeds 1-16, 11
    alone meets all of this. On 11: the doubt at 87.4 s, a Horny Mushroom at 41%; MAYBE OUT at 103.7 s, ATTACK on Ctrl as
    configured at 105.5 s, paged; out of darts' stop at 117.3 s, paged.) It is re-seeded at most changes of the grind or
    the world (more than ten times): placing its monsters would make it hold, not a seed.)"""
    return key_check_run(13)(sim)


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
    paged once; her jumps lift her again. On seed 3: the doubt at 30.6 s, a Green Mushroom near (put off), the window
    read at 36.8-37.5 s, JUMP followed to C at 38.2 s, 66 kills after. (Seed 3 since the rebase onto her touches: on
    seed 1 her chains were cut short by touches, MISSes too, before their jumps' lift-offs were seen through, and the
    doubt came only after the hunt; of seeds 1-13, 2, 3, 4, 5, 9, 11 and 13 meet all of this, 6, 7 and 12 follow C
    with too few jumps after, 1, 8 and 10 doubt only after the hunt.) (Seed 1 since the rebase of the whole screen
    15 times a second (worldstate.WIDE_EVERY_S) onto her keys followed and her stat window (master d9774e5,
    2026-10-28): on seed 3 JUMP was followed at 201.7 s without the doubt said (no 6 jumps in a row without a lift);
    of seeds 1-12, 1, 4-8, 11 and 12 meet all of this, master's stack on 1 past its 400 s. On 1: the doubt at 100.7
    s, a slime near (put off), the window read at 138.3-139.0 s, JUMP followed to C, 35 kills after.) (Since the pull's
    detour rule (2026-10-29: behavior.patrol, a pile against the loop's rate) on 1: the doubt at 78.5 s, the window read
    at 79.0-79.7 s, JUMP followed to C at 80.4 s, 53 kills after.) (Seed 2 since the evade work's integration
    (2026-10-29: work/evade-int, her evade step and R7's dodge with the pull's detour rule): on 1, and on 4 (the evade
    work's pick without the pull's rule), fewer than 3 of her jumps lifted her after the check; of seeds 1-13, 2, 3, 5-11
    and 13 meet all of this. On 2: the doubt at 138.3 s, put off (a Horny Mushroom and a slime near), the window read at
    150.0-150.7 s, JUMP followed to C, 42 kills after.) (Seed 1 since the respawn as the map's fill to its cap,
    2026-10-30: work/fill-to-cap, grinding's loop counting what its own kills bring back: on 2 not all of this; of seeds
    1-13, 1, 3-8 and 10-13 meet all of it. On 1: the doubt at 122.1 s, the window read at 122.6-123.3 s, JUMP followed
    to C at 124.0 s, 38 kills after.) (Seed 1 since the review's fixes of 2026-10-30 (the frames on their grid at 15 a
    second whatever is asked, feed.run; a monster standing planned either way a while, navigator.STAND_UNSURE_S; her
    piles seen gone only by strips sure of them, worldstate.DropStrip.sure;
    D:/MapleHelper-scratch/replan/fix/out/scan_A.txt, scan_B.txt): on seed 2 her jumps' six in a row without a lift
    weren't told; of seeds 1-8 met on 1, 5 and 7.)"""
    return key_check_run(4)(sim)


@tree((23, -150.0), monsters=set(range(0, 17)), seed=10, limit=400, may_die=True,
      **dict(_LIVE, realism=Realism(zero_walls=True, game_keys=(("right", "e"),))),
      expects=FOLLOW_EXPECTS + (
          logs("HER RUNS DIDN'T MOVE HER 4 TIMES IN A ROW"), logs("RIGHT RIGHT -> E"),
          check("E pressed after it, Right never", lambda tr: _keys_ok(tr["result"]["keys_after"], "e", "right")),
          check("her runs moved her after it", lambda tr: tr["result"]["moves"].get("ran", 0) >= 3)))
def loop_tree_grind_arrow_key(sim):
    """As loop_tree_grind_jump_key from the left of 23, the game's MOVE RIGHT on E, not on her Right arrow: her runs right
    don't move her (her chains cut short, her dot standing where they run her: 'off the path (still at x ...)'), 4 in a
    row, the doubt; the check where she is (her moves doubted: no retreat), the window read (MOVE RIGHT on E): her right
    E from then on, remembered, paged once. (From 23 at 73, and with her left on Q, the hunt kept her where she was,
    throwing, and her chains came only once it was over: an arrow moved shows in her runs alone, not in her facing,
    which live frames tell otherwise than her taps with her keys right too: 151 times in the tour runs of 2026-10-24 to
    28, up to 13 within 20 s.) The doubt and the check are set up (2026-10-31): the map's monsters only up the tower
    (0-16, as loop_tree_stat_window's), so her first trip from 23's left runs right with nothing near her, and the check
    has its quiet moment at once. With every spawn filled her runs right came or not as her grinding went, and a Horny
    Mushroom near put the check off three times, ending the hunt (at master 35a1196 2 of seeds 1-13 met all of this;
    the seed moved 3 times). On seed 10 the doubt at 6.2 s, MOVE RIGHT followed to E at 8.0 s, 56 kills after; all of
    seeds 1-13 meet all of this (the doubt at 5.5-6.6 s), and seed 10 with the chase off and with the old respawn clock
    too."""
    return key_check_run(4)(sim)


@tree((23, 73.0), monsters="all", seed=2, limit=400, may_die=True,
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
    1: out of darts due at 26.5 s, away to 25 (1.6 s; a safe platform: no put-off), the window read at 29.5-30.2 s,
    followed at 30.9 s, 63 kills after (27.2, 1.7, 29.3-30.0, 30.7 and 67 before the pull's detour rule (2026-10-29:
    behavior.patrol, a pile against the loop's rate)). (Seed 2 since the server's respawn and the chase together
    (respawn_tick and Realism.chase every world's, 2026-10-31): on seed 1 not all of this; of seeds 1-13, 2-13 meet all
    of it.) (Seed 2 since chains are replanned at every state as they run (2026-10-30, work/replan: navigator.REPLAN,
    KNOCK_REPLAN, MEET_WALK): on seed 1 the key check was put off on the rope at -8 and out of darts' stop came first;
    of seeds 1-2, 2 meets all of this.)"""
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


# --- her stat window: her W.DEF read at a hunt's start (behavior.statcheck) -------------------------------------------
# The user, 2026-10-28 ~18:20, on her DEF not known at a first run (the safety stop's reserve at DEF 0 until 8 touches
# are kept): "You can open character stats panel to check the defense". The world draws her stat window from the game
# data's pictures with her stats (Realism.stat_key: S toggles it; its DETAIL panel by a click on its button; Esc closes
# it), found as in game, its text read by the simulator's stand-in for the OCR (FakeEyes.stat_ocr). A hunt opens it
# only with the reader calibrated by a live capture (`gamelab stats --read --record`): seeded here as one would leave it
# on a client drawing the game data's pictures (`_stat_calibrated`).

def _stat_calibrated(sim) -> None:
    """The stat window's reader calibrated as `gamelab stats --read --record` would leave it on a client that draws the
    game data's pictures where the simulator does (lab/stat_window.json, the run's own)."""
    from ..perception import stat_window as sw
    sw.Calibration("copy", {k: {"at": list(v), "size": list(sw.templates()[k].size)} for k, v in sw.DRAW_AT.items()},
                   at=time.time()).save()


def stat_run(minutes: float = 1.5, grind: bool = True, calibrated: bool = True):
    """A hunt's start as gamelab.__main__'s _hunter has it (her stat window read if it's due: Hunter.read_stats), then
    the hunt loop as `tour --loop --grind` runs it (in behavior.retreat.on_alarm; `grind` False, the hunt loop alone).
    The result: the start's outcome (`start`: what, and what was sent), the lines that speak of her stat window
    (`said`: [s, line]), its plans ([s, name, keys and buttons down]), how the hunt ended (`stopped`: a HuntStop's text,
    else None), what was paged, her DEF at the end (`defence`: where from, its low end) and her reserve then."""
    def run(sim):
        from ..behavior.grind import Grind
        from ..behavior.hunt import Hunt
        from ..behavior.retreat import on_alarm
        from ..behavior.safety import HuntStop
        if calibrated:
            _stat_calibrated(sim)
        tour = sim.tour_stack()
        stopped, start = None, None
        try:
            start = tour.h.read_stats(at_start=True)
            if minutes > 0:
                with on_alarm(tour.h, rng=random.Random(sim.scenario.seed)):
                    (Grind if grind else Hunt)(tour, log=sim.log).run(minutes=minutes)
        except HuntStop as e:
            stopped = f"{e.label}: {e}"
        sim.world.advance(sim.clock.now)
        said = [[float(line.split()[0]), line.split(None, 1)[1].strip()] for line in sim.trace.log
                if "stat window" in line or "STAT WINDOW" in line]
        window = [[t, name, [e[1] for e in events if len(e) == 3 and e[2] == "down"]] for t, name, events, _
                  in sim.trace.plans if name.startswith("stat_window")]
        d = tour.h.stats.defence
        return {"start": None if start is None else [start.what, list(start.sent)], "said": said, "window": window,
                "stopped": stopped, "pages": sim.player.pages, "defence": [d.source, d.low],
                "reserve": tour.h.hp_watch.reserve.limit(tour.h.max_hp)}
    return run


def _stat_keys_ok(tr: dict) -> bool:
    """Her stat window's keys: S to open it, a click on DETAIL, S to close it, no Esc (nothing open after it, so none
    needed), nothing else; the game's MENU never opened, the window closed at the end, opened once."""
    keys = [k for _, _, ks in tr["result"]["window"] for k in ks]
    w = tr["world"]
    return (keys == ["s", "lbutton", "s"] and w.get("menu_opened", 0) == 0 and not w.get("stat_window_open", True)
            and w.get("stat_windows") == 1)


STAT_EXPECTS = (
    logs("her stat window: W.DEF 75, level 25, HP "),
    logs("her DEF 75 (her stat window's W.DEF, read "),
    check("her DEF her stat window's at the end (75)", lambda tr: tr["result"]["defence"] == ["stat window", 75.0]),
    check("S, a click on DETAIL, S; no Esc, no MENU, closed", _stat_keys_ok),
    check("nobody paged, no stop", lambda tr: not tr["result"]["pages"] and tr["result"]["stopped"] is None))


@tree((23, 73.0), monsters=set(range(0, 17)), seed=1, limit=400, attack="choice",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=Realism(zero_walls=True),
      expects=STAT_EXPECTS + (
          check("read at the start", lambda tr: tr["result"]["start"][0] == "read"),
          logs("the safety stop now under 393 HP of 774 (51%: 3 Horny Mushroom touches of 131) (was 453 HP of 774")))
def loop_tree_stat_window(sim):
    """`tour --loop --grind --map 10002073` (the feed, the attack choice) from 23 with nothing near her (the map's
    monsters on 0-16 only), the stat window's reader calibrated: at the start, a quiet moment (two looks 0.3 s apart),
    S, the window seen (0.6 s), a click on DETAIL, its panel seen, read (W.DEF 75: the world's DEF, level 25, HP 774/774,
    MP 623/623), S, closed (about 2.5 s in all); her DEF 75 from then on, the reserve 393 HP of 774 (3 Horny Mushroom
    touches of 131) where it was 453 (DEF 0) until 8 touches were kept; then the hunt."""
    return stat_run(1.5)(sim)


@tree((23, 73.0), monsters="all", seed=1, limit=400, attack="choice",
      her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=Realism(zero_walls=True),
      expects=STAT_EXPECTS + (
          check("put off at the start", lambda tr: tr["result"]["start"][0] == "put off"),
          logs("her stat window put off (green-mushroom, horny-mushroom within 250 of her)"),
          logs("her stat window put off: a quiet moment now")))
def loop_tree_stat_window_put_off(sim):
    """As loop_tree_stat_window with every spawn point filled: a Green and a Horny Mushroom within 250 of her at the
    start, so the read is put off at once (nothing moves her for it), the hunt goes on, and the hunt loop reads it at
    its first quiet look 5 s or more later (StatsDue from health_check), as at the start: S, DETAIL, S."""
    return stat_run(1.5)(sim)


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


@tree((18, 111.0), hanging=-865.0, monsters="all", seed=1, limit=120, **_LIVE, expects=ROPE_TOP_EXPECTS)
def loop_tree_grind_rope_top(sim):
    """`tour --loop --grind --map 10002073` (the feed, the attack choice), as live at 22:58:08, from the rope at x 111,
    her feet 30 under platform 18 (-865): on the rope at the first look (localization.pose.ROPE_TOP_GAP), up off it onto
    18 (Mover.off_rope), and on to grind. Master's stack in this world stopped the hunt for out of darts at 17.7 s (on
    seed 1 at 23.1 s; the -880 twin, 15 under, at 17.4-18.3 s on seeds 1-3; at 18.5 s with after-kill in, 7f6334e), no
    kill, its presses refused on the rope, as live. (Seed 1 since the loot sweep (2026-10-30, behavior.sweep: the hunt
    loop's drops one rule, priced in her time): on seed 2 the run passed its 120 s limit (her stand-by's fights after
    the hunt, the tour's drops guarded again and again); of seeds 1-4, 1 and 4 meet all of this.)"""
    return darts_stop(1, grind=True)(sim)


@tree((18, 111.0), hanging=-880.0, monsters="all", seed=1, limit=120, **_LIVE, expects=ROPE_TOP_EXPECTS)
def loop_tree_rope_top_15(sim):
    """As loop_tree_rope_top, her feet 15 under 18 (-880; at 22:19:38 live her dot was 14.8 under it at 9 chain
    starts): her dot still puts her on 18, so a move from there that never moves it (control.navigator._rope_top) or
    the game refusing her presses on the message bar (behavior.hunter._hear_chat) has her on the rope: off it, onto 18,
    and on to hunt. (Master stopped the hunt at 18.7 s, 20 presses refused.)"""
    return darts_stop(1)(sim)


@tree((18, 111.0), hanging=-880.0, limit=60, noise=Noise(unfound=((0.0, 3.0),)),
      expects=(ends("arrived"), logs("her dot never moved"), sends("off_rope"),
               check("on 16 by the end", lambda tr: tr["world"]["platform"] == 16)))
def nav_tree_rope_top_15(sim):
    """From the rope at x 111, her feet 15 under platform 18, up to 16 with no monsters about, the localizer not
    finding her for the first 3 s (Noise.unfound: her dot alone places her, on 18): the jump up from 18 does nothing on
    the rope and her dot never moves (control.navigator._rope_top), so she's on the rope: up off it onto 18
    (Mover.off_rope), and on up. (Found, the localizer has her on the rope: tests/test_locate.py.)"""
    return _nav(16)(sim)


@tree((19, 250.0), monsters=[("Green Mushroom", 18, 125.0)], seed=1, limit=120, **_LIVE,
      expects=(ends("arrived"), logs("hanging on the rope"), sends("hang_up"),
               check("touched once at most", lambda tr: tr["world"]["touches"] <= 1),
               check("on 16 by the end", lambda tr: tr["world"]["platform"] == 16)))
def nav_tree_rope_hit(sim):
    """From 19 up to 16 by the rope at x 111 (19 -> 18), a Green Mushroom walking on 18 by its top (as at 22:58:05
    live, where the trip went past it regardless): it touches her on the rope, which cuts the chain and lets go of
    up; the game keeps her on the rope, and up goes down again at once (navigator.CLIMB_ON), onto 18 and on. (Seed 2
    since the review's fixes of 2026-10-30 (the frames on their grid at 15 a second whatever is asked, feed.run; a
    monster standing planned either way a while, navigator.STAND_UNSURE_S; her piles seen gone only by strips sure
    of them, worldstate.DropStrip.sure; D:/MapleHelper-scratch/replan/fix/out/scan_A.txt, scan_B.txt): on seed 1 the
    Green Mushroom never touched her on the rope; of seeds 1-8 met on 2, 4 and 6-8.) (Seed 1 again, master's, since
    work/replan's loot work on master 072e1ec (2026-10-31: her HP at the user's prices, the positioning switches as
    first built, her drops' life by the live evidence: eval/seeds/rope.txt): on seed 2 the Green Mushroom never touched
    her on the rope; of seeds 1-24, 1, 3, 5-7, 9-11, 13 and 15-23 meet all of this and tests/test_touches.py's MISS on
    the trip.)
    (From 2026-11-02, navigator.ROPE_HANG: the Green Mushroom by the top makes her stop on the rope 88 under it and
    hang there; it stays within reach of the top the whole HANG_S, so she goes up regardless after 2.1 s, touched on the
    rope then (a MISS, no knock), and her invulnerability after it takes her past it (the touch is the cut's: POST_HIT),
    on 16 at 7.1 s with 1 touch where 5.1 s with 1 (climbed on after a touch 55 under the top). Her climb on after a
    touch, CLIMB_ON, isn't shown here any more: tests/test_rope_top.py, test_evade.py and test_touches.py (without
    ROPE_HANG).)"""
    return _nav(16, fight=True)(sim)


# Hanging under a rope's top until the way up clears (navigator.ROPE_HANG, from 2026-11-02): live that hour (RiceBaby
# grinding, three recordings and the hour's telemetry) 19-23 of the 170 touches came at the top of the rope at x -80 from
# 11 to 10, the camp's only way in, by a Horny Mushroom on 10 beside it (10's Horny point is at -109), 16-48 under the
# top or standing up there.

@tree((11, -150.0), monsters=[("Horny Mushroom", 10, -60.0)], seed=8, limit=60,
      **_LIVE, expects=(ends("arrived"), logs("hanging on the rope"), logs("the way up clear"),
                        no_log("touched on the rope"), check("not touched", lambda tr: tr["world"]["touches"] == 0),
                        check("on 10 by the end", lambda tr: tr["world"]["platform"] == 10)))
def nav_tree_rope_top_10(sim):
    """From 11 at x -150 up the rope at x -80 onto 10 (the grind's way back into the camp), a Horny Mushroom walking 10
    by the rope's top: the climb stops on the rope 80-90 under the top while the way up isn't clear (navigator.ROPE_HANG:
    "hanging on the rope"), out of its reach, and goes on up once it is (or after HANG_S), never touched on the rope. A
    trip in transit, as the grind's (only what's in her way is shot), to anywhere on 10: it ends at the rope's top. On
    seed 8 she hangs 0.5 s 83 under the top and goes up untouched, on 10 0.5 s later than without the hang (touched on
    the rope 50 under its top then, damage 118, and up again). On seeds 1-12 (D:/MapleHelper-scratch/recfix/rope/
    seeds_hang.py): touched on 2 of the 12 trips (1 on the rope) where without the hang on 5 (4 on the rope); 4 hangs, 2
    of them up regardless after HANG_S (one of those touched); the 12 trips 54.2 s in all against 48.7."""
    from ..control.navigator import TRANSIT
    return sim.navigator(sim.hunter(), fight=True).go(10, mode=TRANSIT)


@tree((24, 231.0), monsters=[("Horny Mushroom", 23, 20.0)], seed=8, limit=60, **_LIVE,
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
    (`Escort.throws` tells now). (Of seeds 1-6 the others shoot it first from 24 too: _meet's.) (Seed 5 since the whole
    screen 10 times a second with her band's own search beside it where a whole screen takes longer than 0.05 s
    (worldstate.BAND_STAND_IN_S, 2026-10-28): on seed 3 her throw went 2 ms later, the star pushed the mushroom back
    and it died at x 2, not by her landing, so nothing priced it; of seeds 1-16, 5, 8, 9, 15 and 16 meet all of this,
    and on every seed she shoots it first from 24.) (Seed 3 since a trip's chains are planned by the whole state's
    monsters, each walking as perception has it head at its kind's speed from the game data, with no quick look after a
    touch (2026-10-30, work/replan: navigator.trip_monsters, POST_HIT): on seed 5 the jump's landing had no price once
    the mushroom was shot, nothing to say; of seeds 1-9, 3 and 9 meet all of this.) (Not shown since work/replan's
    rebase onto the server's respawn and the chase (2026-10-31: D:/MapleHelper-scratch/replan/pos/fix/out/scan_A.txt,
    scan_C.txt, one_landing5.txt): on none of seeds 1-40 does she land by its dying body. Hit, the mushroom walks at her
    (the chase) and dies about 44 units from where her kill is placed (seed 5: at x 65, the kill at x 21), so its dying
    body is taken for a live one for 1.4 s ('could touch her on the way: shooting it first', 'nothing in reach' twice),
    and she jumps once it's gone; tests/test_sim.py's DYING_AWAY.) (Seed 8 since the monsters move as the v83 client's
    AI moves them, 2026-10-31: on seed 5 the Horny Mushroom not dying by her landing; of seeds 1-13 8 alone shows all of
    this.)"""
    from ..control.navigator import TRANSIT
    return sim.navigator(sim.hunter(), fight=True).go(23, mode=TRANSIT)


def _spread_from_spot() -> bool:
    from ..motion import flow
    return flow.SPREAD_FROM_SPOT


@tree((21, 42.0), monsters=[("Horny Mushroom", 23, 24.0), ("Horny Mushroom", 23, -132.0)], seed=2, limit=60,
      **dict(_LIVE, realism=dataclasses.replace(_LIVE["realism"], respawn_tick=0.0)),
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
    it drops from another, clear of both, and there's nothing to say; not hurt either way. (On its monsters' own clock
    since the server's respawn became every scenario's, 2026-10-31: _CLOCK's note.)"""
    from ..control.navigator import HUNT
    return sim.navigator(sim.hunter(), fight=True).go(23, x=8.0, mode=HUNT)


def _no_mover(tr: dict) -> bool:
    """No move made by the mover alone (its log line ends "[N moves to go]"; a chain's says why it ended)."""
    return not any(line.rstrip().endswith("moves to go]") for line in tr["log"])


@tree((21, 60.0), monsters=[("Green Mushroom", 21, 130.0)], seed=3, limit=70,
      **dict(_LIVE, realism=Realism(zero_walls=True, chase=True, invulnerable=harness.INVULNERABLE_LIVE)),
      expects=(ends("arrived"), logs("contact:"),
               check("no move a monster blocks made by the mover", _no_mover),
               check("on 19 by the end", lambda tr: tr["world"]["platform"] == 19),
               check("touched once at most", lambda tr: tr["world"]["touches"] <= 1)))
def nav_tree_evade(sim):
    """A movement-only trip (no escort: navigator.AVOID) from 21 at x 60 up to 19, a Green Mushroom 70 off on 21 chasing
    her (hit before the trip: the walkers world's aggro, world.AGGRO_S; her invulnerability as live): the evade step of
    2026-10-28 (navigator.EVADE). It can touch her within CONTACT_S whatever she does, so the touch is taken as given
    and the jump planned through it in the window after (the contact rule, "contact:"); knocked back, past it ("pass:
    NO_REFUGE"); on 22, the next jump blocked and her spot unsafe, a step along 22 to the nearest spot clear of it
    ("spot:"), and from there past it once EVADE_S has gone by ("pass: EVADE_S"). No move of the mover's (the move a
    monster was in the way of went to control.mover, unwatched, before). Since a trip's chains are planned by the whole
    state's monsters, each walking as perception has it head at its kind's speed, with no quick look after a touch
    (2026-10-30, work/replan: navigator.trip_monsters, POST_HIT), the chain after that first touch is planned from the
    newest state in the window it leaves her and goes up past the mushroom: touched once, on 19 at 2.1 s where it took
    8.2 s, three touches, the spot and the passes before; on none of seeds 1-16 is a spot or a pass needed (the evade
    step's own tests: tests/test_evade.py). (Seed 3 since her knock is the client's (2026-11-02: 270 out and up, its push
    sideways added to the speed she has, landing at her walk's speed at most, world.KNOCK_VX, KNOCK_ADDS,
    KNOCK_LANDS_WALKING: ~87 units on flat ground from standing where it was ~58;
    D:/MapleHelper-scratch/recfix/sim-fix/scan3.txt, scanned with the push set outright, not added): on seed 2 the first
    touch's knock landed her on 22 and the
    mushroom touched her again back on 21, twice in all; of seeds 1-16, 3, 5-7, 10, 13 and 16 meet all of this. On 3: the
    jump touched and knocked back onto 21, the next up past it onto 19, touched once.)"""
    sim.world.monsters[0].aggro(sim.world.t)
    return _nav(19, seconds=60)(sim)
