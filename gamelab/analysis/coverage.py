"""Route coverage: how often each move of the map's route graph (motion.route) is planned, taken and missed, where
she spends her time, and the moves she never takes: their cost gap to the routes she took instead, and why they lose.
From the harness's runs (sim.harness keeps each run's tally, `tally`) and the live telemetry (lab/telemetry).

    python -m gamelab coverage [RUN ...] [--live] [--picture PATH]

A run (a harness seed, or a live telemetry file) is tallied once, from its telemetry records:
- planned: the route records a move is in (every plan, replans too);
- taken: the moves she set off on, in the chains sent and the moves the mover made on their own (from 2026-10-21
  the navigator's 'moved' record after each; before, its chains and log lines); missed: of those, the ones she
  ended elsewhere than on their platform after, not from a hit. Which move of a chain it was is read from where she
  was seen standing in between (`stood`: her looks and the glances at her dot, on a platform's surface twice in a
  row), against the platforms the chain goes through; a hit's move (hit on the way, and ended elsewhere than the
  chain was to: one that still got her there made every move) is counted apart ('hit');
- per platform: visits (runs of looks on it) and seconds (from each look to the next, GAP_S at most), and
  'arrived': the moves that got her onto it (made, or missed or knocked onto it on the way: a hit's move that left her
  elsewhere than where it set off), which the regression test reads;
- the plans: where each was planned from and to, and what its route costs by the planner's own times (no random
  weights, no penalties), against the quickest; and what it kept off and avoided (for why a cheaper move lost);
- the trips' ends, and the grounds the tour stopped at (the sweep's stops) and the hunt loop hunted.

`report` then gives, for every move of the graph, what the runs did with it, and for those never taken, the gap from
the best route through it to the route chosen on the plans that could have used it, and why it lost (`why`):
replanned (planned, but replanned before she got to it), the weights (each trip weighs every move by a random 1 to
1 + VARIETY, and with route.ROUTES over 1 chooses among the planner's ways: it was cheaper and still not drawn), a
penalty (a trip's keep-off or a monster in the way made a
cheaper route through it lose), the ground choice (it is the quickest way between two places on the map, but no
trip went between them) or its cost (it is that nowhere); 'then', on runs planned on an older graph that may not
have had it. Moves the chains can't take from their own start (no takeoff passing flow's Flow.sure_takeoff) are
named: the mover then makes them alone. The map is passed in (this layer doesn't load it: gamelab's `coverage`
command does).
"""

import math
import random
import re
from collections import Counter, defaultdict

from ..motion import route
from ..motion.flow import Body, Flow, Keys
from ..motion.physics import WALK_SPEED
from ..motion.route import NOISE, RISK, Move, Planner

GAP_S = 5.0        # s: a look's time counts toward its platform up to this long (a longer gap is a pause)
STAND_DX = 6       # units: a look or a glance this near a platform's span ...
STAND_DY = 6       # ... and its surface reads as on it (her dot is good to a minimap pixel: the play track's
STAND_N = 2        # ... jump margins read it so), this many in a row as standing there
MATCH_DX = 3       # units: a move in a record is the graph's move of its kind and platforms that starts this near
PLAN_DX = 5        # units: plans from within this of each other (and to the same platform, at the same cost) are one
NEAR_TIE_S = 0.05  # s: a route through a move this near the quickest is a quickest one
VARIETY = NOISE    # the navigator's variety (route.NOISE, 0.35): a trip weighs each move 1 to 1.35 times, so a
                   # route up to 35% dearer than the one chosen can be drawn instead, and no dearer one ever is
START_STEP = 60    # units between the starts tried along each platform, for where each move is the quickest
DRAWS = 40         # draws of the navigator's weights a plan, for how often they'd send her through a move (drawn):
                   # a total of about 1 over 200 plans is then some 13 hits of 8,000 draws
WAY_DRAWS = 8      # ... with route.ROUTES over 1: each draw gives the ways' own chances (route.chances), not one of
                   # them, and a draw is the planner's ways (8 ms on this map) where the weights' was one search (1 ms)
NAMES = {k: k for k in ("left", "right", "jump", "up", "down")}  # flow's key names (no keys are sent: any do)

# The navigator's log lines before the 'moved' record (2026-10-21): a move made alone, a miss, a hit.
_ALONE = re.compile(r"^\s+platform (\d+) \(.*\) at x (-?\d+) \(\w*\): (\w+) \w* -> platform (\d+)"
                    r"\s+\[\d+ moves to go\]$")
_ELSEWHERE = re.compile(r"^\s+ended on (?:platform (\d+)|nothing|None).* instead of platform (\d+)")
_CUT = re.compile(r"^\s+(?:hit on the way|hit \(.*plan cut short)")


def key_of(m: Move) -> str:
    """A move's name in the report: its kind, platforms and where it starts (unique on this map)."""
    return f"{m.kind} {m.src}->{m.dst} @{round(m.x)}"


class Graph:
    """The map's moves (route.Planner), told apart by key_of, and the planner's own costs."""

    def __init__(self, model, planner: Planner | None = None):
        self.model = model
        self.planner = planner or Planner(model)
        self.moves = self.planner.moves
        self.by_key = {key_of(m): m for m in self.moves}
        self._by = defaultdict(list)
        for m in self.moves:
            self._by[(m.kind, m.src, m.dst)].append(m)
        self._to: dict[tuple[int, int], dict[int, float]] = {}

    def match(self, kind: str, src: int, dst: int, x: float | None, loose: bool = False) -> Move | None:
        """The graph's move of that kind between those platforms starting within MATCH_DX of x (with x None, the
        only one there is; `loose`, the nearest one whatever its x: a move of an older graph, whose jumps took off
        elsewhere). None if there's none."""
        options = self._by.get((kind, src, dst), [])
        if x is None:
            return options[0] if len(options) == 1 else None
        best = min(options, key=lambda m: abs(m.x - x), default=None)
        return best if best is not None and (loose or abs(best.x - x) <= MATCH_DX) else None

    @staticmethod
    def step(m: Move) -> float:
        """A move's cost to the planner, the walk to where it starts left out."""
        return m.seconds + RISK.get(m.kind, 0.0)

    def cost(self, moves: list[Move], x: float) -> float:
        """What a route costs the planner from x on its first move's platform: walks and moves, unweighed."""
        t, at = 0.0, x
        for m in moves:
            t += abs(m.x - at) / WALK_SPEED + self.step(m)
            at = m.land_x
        return t

    def onward(self, m: Move) -> dict[int, float]:
        """Seconds from where move m lands to each platform."""
        k = (m.dst, round(m.land_x))
        if k not in self._to:
            self._to[k] = self.planner.times(m.dst, m.land_x)
        return self._to[k]

    def via(self, m: Move, src: int, x: float, dst: int) -> float | None:
        """The quickest route from (src, x) to dst that makes move m: to where m starts, m, and on from where it
        lands (None: there's none)."""
        to = self.planner.arrive(src, x, m.src, m.x)
        on = self.onward(m).get(dst)
        return None if to is None or on is None else to + self.step(m) + on

    def chains(self, m: Move) -> bool | None:
        """Whether flow chains move m from standing where it starts, any way (variant): no for a jump none of
        whose takeoffs lands with Flow.sure_takeoff's room; None for the kinds only the mover makes (a rope's
        climb down, a jump off a rope)."""
        if m.kind in ("climb_down", "rope_jump"):
            return None
        flow = Flow(self.model, NAMES)
        start = Body(0.0, m.x, self.model.platform(m.src))
        return any(flow.leg(m, start, Keys(NAMES), v) is not None for v in Flow.VARIANTS)


_GRAPH: list = []  # (model, its Graph): the last one asked for


def graph_for(model) -> Graph:
    """The Graph of a map model, built once a process (0.25 s on this map): each harness run tallies with it."""
    if not _GRAPH or _GRAPH[0][0] is not model:
        _GRAPH[:] = [(model, Graph(model))]
    return _GRAPH[0][1]


# --- one run ---------------------------------------------------------------------------------------------

def tally(records: list[dict], graph: Graph) -> dict:
    """One run's counts from its telemetry records (see the top): plain JSON data, merged by `merge`."""
    t = _Tally(graph)
    for r in records:
        t.add(r)
    return t.done()


class _Tally:
    """`tally`'s state, record by record."""

    def __init__(self, graph: Graph):
        self.g = graph
        self.moves: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # key -> planned, taken, missed, hit
        self.unknown: Counter = Counter()        # moves in records that aren't in today's graph (by their own key)
        self.elsewhere: dict[str, Counter] = defaultdict(Counter)  # key -> where she ended instead
        self.platforms: dict[int, list[float]] = defaultdict(lambda: [0, 0.0])  # visits, seconds
        self.arrived: Counter = Counter()
        self.plans: dict[tuple, list] = {}
        self.trips: Counter = Counter()
        self.idle: Counter = Counter()  # trips that ended with no route planned
        self.stops: Counter = Counter()  # the tour's stops, by ground
        self.grounds: Counter = Counter()  # the hunt loop's grounds, each time it took one up
        self.hops = 0
        self.loose = 0      # moves of older routes put on today's graph's of the same kind and platforms
        self.older = 0      # routes recorded before they said where they were planned from (an older graph's)
        self.modern = False
        self.scene: tuple | None = None  # (t, platform, x) of the last look
        self.on: int | None = None       # the platform of the last look that put her on one
        self.last_route: list[Move | None] = []
        self.routes_in_trip = 0
        self.points: list[tuple[float, float]] = []  # where she was seen since the pass began
        self.sent = False   # a chain sent since the points began (follow-on routes don't start a pass)
        self.open: dict | None = None  # before the 'moved' record: the chain or move whose outcome is awaited
        self.ground = None

    # records, in order
    def add(self, r: dict) -> None:
        topic = r.get("topic")
        if topic == "scene":
            self._scene(r)
        elif topic == "glance":
            if r.get("x") is not None:
                self.points.append((r["x"], r["y"]))
        elif topic == "route":
            self._route(r)
        elif topic == "chain":
            self.sent = True
            if not self.modern:
                self._legacy_chain(r)
        elif topic == "moved":
            if not self.modern:  # the run has the navigator's own record: its chains' and log lines' guesses go
                self.modern, self.open = True, None
            self._moved(r)
        elif topic == "log":
            if not self.modern:
                self._legacy_log(r.get("text", ""))
        elif topic == "trip":
            self._close()
            self.trips[f"{r.get('dst')} {_result(r.get('result', ''))}"] += 1
            if not self.routes_in_trip:
                self.idle[f"{r.get('dst')} {_result(r.get('result', ''))}"] += 1
            self.routes_in_trip = 0
            self.points, self.sent = [], False
        elif topic == "tour" and r.get("stop") is not None and "order" not in r:
            # (the hunt loop's 'tour' records name its next ground, with the grounds so far: its grounds, below)
            self.stops[str(r["stop"])] += 1
        elif topic == "hunt" and r.get("ground") is not None and r.get("ground") != self.ground:
            self.ground = r["ground"]
            self.grounds[str(self.ground)] += 1

    def _scene(self, r: dict) -> None:
        her = r.get("her") or {}
        pid, t = her.get("platform"), r.get("t", 0.0)
        if self.scene is not None and self.scene[1] is not None:  # the time since the last look: its platform's
            self.platforms[self.scene[1]][1] += min(max(0.0, t - self.scene[0]), GAP_S)
        if pid is not None and pid != self.on:
            self.platforms[pid][0] += 1
            self.on = pid
        self.scene = (t, pid, her.get("x"))
        if her.get("x") is not None:
            self.points.append((her["x"], her["y"]))

    def _route(self, r: dict) -> None:
        moves = r.get("moves") or []
        # A route recorded before it said where it was planned from (2026-10-21) was planned on an older graph: its
        # moves are today's of the same kind and platforms, the nearest where they start.
        loose = "from" not in r
        self.older += loose
        found = [self.g.match(m["kind"], m["src"], m["dst"], m.get("x"), loose) for m in moves]
        self.loose += sum(1 for m, got in zip(moves, found)
                          if loose and got is not None and abs(got.x - (m.get("x") or got.x)) > MATCH_DX)
        for m, got in zip(moves, found):
            if got is not None:
                self.moves[key_of(got)][0] += 1
            else:
                self.unknown[f"{m['kind']} {m['src']}->{m['dst']} @{round(m.get('x') or 0)}"] += 1
        self.last_route = found
        self.routes_in_trip += 1
        if not self.sent:  # a new pass: where she was seen before it is no part of it
            self.points = []
        if moves and all(found):
            self._plan(r, found)

    def _plan(self, r: dict, path: list[Move]) -> None:
        start = r.get("from")
        if start is not None:
            src, x = start["platform"], start["x"]
        elif self.scene is not None and self.scene[1] == path[0].src and self.scene[2] is not None:
            src, x = path[0].src, self.scene[2]
        else:
            src, x = path[0].src, path[0].x
        if src != path[0].src:
            return
        dst, x = r.get("dst", path[-1].dst), PLAN_DX * round(x / PLAN_DX)
        chosen = round(self.g.cost(path, x), 2)
        k = (src, x, dst, chosen, tuple(r.get("keep_off") or ()), tuple(tuple(a) for a in r.get("avoid") or ()))
        if k in self.plans:
            self.plans[k][-1] += 1
        else:
            best = self.g.planner.travel(src, k[1], dst)
            self.plans[k] = [src, k[1], dst, chosen, round(best, 2) if best is not None else None,
                             list(k[4]), [list(a) for a in k[5]], 1]

    def _moved(self, r: dict) -> None:
        legs = [m for m in r.get("moves") or [] if m["kind"] != "hop"]
        self.hops += len(r.get("moves") or []) - len(legs)
        landed, hit = r.get("landed"), bool(r.get("hit"))
        off = landed != r.get("planned")  # hit on the way and still where it was to end: every move made
        self._outcome([(m["kind"], m["src"], m["dst"], m.get("x")) for m in legs], landed, hit and off,
                      off and not hit)
        self.points, self.sent = [], False

    # before the 'moved' record: chains, moves made alone and the verdicts from the log
    def _legacy_chain(self, r: dict) -> None:
        self._close()
        legs = r.get("legs") or []
        self.hops += sum(1 for leg in legs if leg["kind"] == "hop")
        legs = [leg for leg in legs if leg["kind"] != "hop"]
        route = self.last_route
        out = []
        for i, leg in enumerate(legs):
            m = route[i] if i < len(route) else None
            same = m is not None and (m.kind, m.src, m.dst) == (leg["kind"], leg["src"], leg["dst"])
            out.append((leg["kind"], leg["src"], leg["dst"], m.x if same else None))
        self.open = {"legs": out, "landed": None, "hit": False, "miss": False}
        self.points = self.points[-1:]  # her last place before it: on its first move's platform

    def _legacy_log(self, text: str) -> None:
        if (m := _ALONE.match(text)) is not None:
            self._close()
            kind, src, dst = m.group(3), int(m.group(1)), int(m.group(4))
            first = self.last_route[0] if self.last_route else None
            x = first.x if first is not None and (first.kind, first.src, first.dst) == (kind, src, dst) else None
            self.open = {"legs": [(kind, src, dst, x)], "landed": None, "hit": False, "miss": False}
            self.points = self.points[-1:]
            self.sent = True
        elif self.open is not None and (m := _ELSEWHERE.match(text)) is not None:
            self.open["landed"] = int(m.group(1)) if m.group(1) else None
            self.open["miss"] = True
            self._close()
        elif self.open is not None and _CUT.match(text):
            self.open["hit"] = True

    def _close(self) -> None:
        """The chain or move awaiting its outcome (before the 'moved' record): a miss only if the log said so."""
        if self.open is None:
            return
        o, self.open = self.open, None
        self._outcome(o["legs"], o["landed"], o["hit"], o["miss"])

    def _outcome(self, legs: list[tuple], landed: int | None, hit: bool, miss: bool) -> None:
        """Count a chain's (or a move's) legs, (kind, src, dst, x) each: made, missed (and where she ended), a hit's,
        or not reached. The navigator says whether it ended where planned (`miss`) and whether she was hit on the
        way, ending elsewhere (`hit`); where she was seen standing in between (and `landed`, where she ended) says which
        leg it was. The platform a missed or hit leg left her on, if not the one it set off from, is one she got onto."""
        if not legs:
            return
        failed = where = None
        if hit or miss:
            plan = [legs[0][1]] + [leg[2] for leg in legs]
            seq = stood(self.g.model, self.points)
            if landed is not None:
                seq.append(landed)
            reached, where = follow(plan, seq)
            if where is not None:
                failed = reached
            elif reached < len(legs) or miss:  # seen on its platforms only: the leg after the last she got to
                failed = min(reached, len(legs) - 1)
                where = landed if landed is not None else plan[failed]
            # (else hit, but seen standing where it was to end, before the 'moved' record: every move made)
        for i, (kind, src, dst, x) in enumerate(legs):
            if failed is not None and i > failed:
                break  # sent, but not reached
            got = self.g.match(kind, src, dst, x)
            key = key_of(got) if got is not None else f"{kind} {src}->{dst} @{round(x or 0)}"
            if got is None:
                self.unknown[key] += 1
            counts = self.moves[key] if got is not None else [0, 0, 0, 0]
            counts[1] += 1
            if i != failed:
                self.arrived[dst] += 1
                continue
            if hit:
                counts[3] += 1
            else:
                counts[2] += 1
                self.elsewhere[key][str(where)] += 1
            if where is not None and where != src:
                self.arrived[where] += 1

    def done(self) -> dict:
        self._close()
        return {"moves": {k: v for k, v in sorted(self.moves.items())},
                "unknown": dict(self.unknown), "elsewhere": {k: dict(v) for k, v in self.elsewhere.items()},
                "platforms": {str(p): [v[0], round(v[1], 1)] for p, v in sorted(self.platforms.items())},
                "arrived": {str(p): n for p, n in sorted(self.arrived.items())},
                "plans": list(self.plans.values()), "trips": dict(self.trips), "idle": dict(self.idle),
                "stops": dict(self.stops), "grounds": dict(self.grounds), "hops": self.hops, "loose": self.loose,
                "older": self.older}


def _result(text: str) -> str:
    """A trip's result, without its numbers ('no route from platform 3 to 9': 'no route')."""
    return text.split(" from ")[0].split(":")[0].split(" (")[0]


def stood(model, points: list[tuple[float, float]]) -> list[int]:
    """The platforms she was seen standing on, in order, from where she was seen (looks and glances): on one's
    surface STAND_N times in a row (in the air she's at one's height only a moment)."""
    seq, run, n = [], None, 0
    for x, y in points:
        p = on_surface(model, x, y)
        n = n + 1 if p == run else 1
        run = p
        if p is not None and n == STAND_N and (not seq or seq[-1] != p):
            seq.append(p)
    return seq


def on_surface(model, x: float, y: float) -> int | None:
    """The platform whose surface (x, y) is on, within STAND_DX along and STAND_DY up or down (the nearest)."""
    best = None
    for p in model.platforms:
        if p.covers(x, STAND_DX):
            dy = abs(p.y_at(x) - y)
            if dy <= STAND_DY and (best is None or dy < best[0]):
                best = (dy, p.id)
    return best[1] if best else None


def follow(plan: list[int], seq: list[int]) -> tuple[int, int | None]:
    """How far along `plan` (the platforms a chain goes through: its start, then each move's) the platforms she
    stood on (`seq`) took her: (the moves made, and where she was instead if one took her off it, else None)."""
    j = 0
    for p in seq:
        if j == len(plan) - 1:
            break  # at the end of it: what comes after is no part of it
        if p in plan[j + 1:]:
            j = plan.index(p, j + 1)
        elif p != plan[j]:
            return j, p
    return j, None


# --- many runs -------------------------------------------------------------------------------------------

_KEY = re.compile(r"^(\w+) (\d+)->(\d+) @(-?\d+)$")


def rekey(graph: Graph, key: str) -> str:
    """A tally's move key as `graph` names it: the move of its kind and platforms nearest where it starts (a
    harness run keyed with her stats as measured, physics.FIXED_STATS, matches a graph from config.toml's)."""
    m = _KEY.match(key)
    got = None if m is None else graph.match(m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)))
    return key if got is None else key_of(got)


def merge(tallies: list[dict], graph: Graph | None = None) -> dict:
    """Runs' tallies added up (plans concatenated), their moves named as `graph` names them (rekey), if given."""
    out = {"runs": len(tallies), "moves": defaultdict(lambda: [0, 0, 0, 0]), "unknown": Counter(),
           "elsewhere": defaultdict(Counter), "platforms": defaultdict(lambda: [0, 0.0]), "arrived": Counter(),
           "plans": [], "trips": Counter(), "idle": Counter(), "stops": Counter(), "grounds": Counter(), "hops": 0,
           "loose": 0, "older": 0}
    named = (lambda k: rekey(graph, k)) if graph is not None else (lambda k: k)
    for t in tallies:
        for k, v in t["moves"].items():
            k = named(k)
            out["moves"][k] = [a + b for a, b in zip(out["moves"][k], v)]
        for k, v in t.get("elsewhere", {}).items():
            out["elsewhere"][named(k)].update(v)
        for k, v in t["platforms"].items():
            out["platforms"][int(k)] = [out["platforms"][int(k)][0] + v[0], out["platforms"][int(k)][1] + v[1]]
        out["arrived"].update({int(k): v for k, v in t.get("arrived", {}).items()})
        for name in ("unknown", "trips", "idle", "stops", "grounds"):
            out[name].update(t.get(name, {}))
        out["plans"] += t.get("plans", [])
        out["hops"] += t.get("hops", 0)
        out["loose"] += t.get("loose", 0)
        out["older"] += t.get("older", 0)
    return out


# --- why moves lose ----------------------------------------------------------------------------------------

def starts(model) -> list[tuple[int, float]]:
    """Places to start from, for where each move is the quickest: every START_STEP units along each platform
    (the middle of each stretch)."""
    out = []
    for p in model.platforms:
        n = max(1, int((p.right - p.left) // START_STEP))
        out += [(p.id, p.left + (p.right - p.left) * (i + 0.5) / n) for i in range(n)]
    return out


def map_wide(graph: Graph) -> dict[str, tuple]:
    """For each move of the graph, the least it adds to the quickest route between two places on the map (starts()
    to each other platform) by going through it, and where: key -> (gap s, share of the quickest, start platform,
    x, dst). A gap under NEAR_TIE_S is a quickest route through it."""
    best: dict[str, tuple] = {}
    for src, x in starts(graph.model):
        times = graph.planner.times(src, x)
        for m in graph.moves:
            to = graph.planner.arrive(src, x, m.src, m.x)
            if to is None:
                continue
            base = to + graph.step(m)
            k = key_of(m)
            for dst, on in graph.onward(m).items():
                quickest = times.get(dst)
                if quickest is None or dst == src or dst == m.src:
                    continue
                gap = base + on - quickest
                if k not in best or gap < best[k][0] - 1e-9:
                    best[k] = (gap, gap / quickest if quickest else math.inf, src, x, dst)
    return best


def plan_gaps(graph: Graph, plans: list[list], moves: list[Move]) -> dict[str, dict]:
    """Each of `moves` against the routes chosen on `plans` ([src, x, dst, chosen, quickest, keep_off, avoid, n]
    each), by key: on the plans that kept nothing off and avoided no monster, and whose trip could have made it (a
    route through it exists; a trip to where it starts doesn't count), how many ('could'), the least gap from the
    best route through it to the route chosen (s, share, src, x, dst), how many had it within VARIETY of the chosen
    (the weights can draw it: 'close') and cheaper ('cheaper': the weights drew the other). And the plans that kept
    something off or avoided a monster on which it was cheaper and its route went onto what they kept off or through
    a move they avoided ('penalty'). Plans first, then moves: the planner keeps its searches from 256 starts only."""
    out = {key_of(m): {"could": 0, "close": 0, "cheaper": 0, "least": None, "penalty": Counter()} for m in moves}
    for src, x, dst, chosen, _, keep_off, avoid, n in plans:
        if chosen is None:
            continue
        for m in moves:
            if dst == m.src and src != m.src:
                continue
            through = graph.via(m, src, x, dst)
            if through is None:
                continue
            g = out[key_of(m)]
            if keep_off or avoid:
                if through < chosen - NEAR_TIE_S:
                    path = _path_via(graph, m, src, x, dst)
                    if keep_off and any(q.dst in keep_off and q.dst != dst for q in path):
                        g["penalty"]["a ground kept off"] += n
                    elif avoid and any([q.kind, q.src, q.dst, round(q.x)] in avoid for q in path):
                        g["penalty"]["a move a monster was in the way of"] += n
                continue
            g["could"] += n
            gap = through - chosen
            if g["least"] is None or gap < g["least"][0]:
                g["least"] = (gap, gap / chosen if chosen else math.inf, src, x, dst)
            g["close"] += n if through <= (1 + VARIETY) * chosen else 0
            g["cheaper"] += n if gap < -NEAR_TIE_S else 0
    return out


def _path_via(graph: Graph, m: Move, src: int, x: float, dst: int) -> list[Move]:
    """The moves of the best route from (src, x) through m to dst."""
    head = (graph.planner.plan(src, x, m.src) or []) if src != m.src else []
    tail = (graph.planner.plan(m.dst, m.land_x, dst) or []) if m.dst != dst else []
    return head + [m] + tail


def drawn(graph: Graph, plans: list[list], moves: list[Move], draws: int | None = None,
          seed: int = 0) -> tuple[float, int]:
    """How many of `plans` (those keeping nothing off, avoiding nothing, within VARIETY, or route.STRETCH with
    route.ROUTES over 1, of a route through one of `moves`) the navigator would be expected to send through one of
    them: each trip weighs every move by its own draw of 1 to 1 + VARIETY (Navigator.weight), and its route is the
    quickest so weighed; with route.ROUTES over 1 it chooses among the planner's ways so weighed (by route.chances,
    without its floor: which moves were stale then isn't known here). `draws` draws a plan (DRAWS, or
    WAY_DRAWS with route.ROUTES over 1). Returns (that expectation, the plans tried)."""
    rng = random.Random(seed)
    keys = {key_of(m) for m in moves}
    ways = route.ROUTES > 1
    draws = draws if draws is not None else WAY_DRAWS if ways else DRAWS
    window = max(VARIETY, route.STRETCH) if ways else VARIETY
    expected, tried = 0.0, 0
    for src, x, dst, chosen, _, keep_off, avoid, n in plans:
        if keep_off or avoid or chosen is None:
            continue
        through = [v for m in moves if (v := graph.via(m, src, x, dst)) is not None]
        if not through or min(through) > (1 + window) * chosen:
            continue
        tried += n
        hits = 0.0
        for _ in range(draws):
            w: dict[str, float] = {}

            def weight(q: Move) -> float:
                k = key_of(q)
                if k not in w:
                    w[k] = 1.0 + VARIETY * rng.random()
                return w[k]
            if ways:
                offered = graph.planner.alternatives(src, x, dst, weight=weight)
                chances = route.chances(offered, floor=0.0) if offered else []
                hits += sum(p for way, p in zip(offered, chances) if any(key_of(q) in keys for q in way.moves))
            else:
                path = graph.planner.plan(src, x, dst, weight=weight) or []
                hits += any(key_of(q) in keys for q in path)
        expected += n * hits / draws
    return expected, tried


def why(m: Move, counts: list[int], gaps: dict, wide: tuple | None, older: bool = False) -> tuple[str, str]:
    """Why move m was never taken: (a short reason, in words). See the top. `older`: the runs planned on an older
    graph, which may not have had it (their routes are put on today's graph's moves: Graph.match's loose)."""
    if counts[0]:
        return "replanned", f"planned {_times(counts[0])}, replanned (or the trip over) before she got to it"
    if gaps["cheaper"] and older:
        return "then", (f"cheaper on today's graph than the route chosen then on {_plans(gaps['cheaper'])}: not in "
                        "the graph then, or the random weights drew the other")
    if gaps["cheaper"]:
        return "weights", (f"cheaper than the route chosen on {_plans(gaps['cheaper'])}: the random weights (or the "
                           "trip's choice of way) drew the other")
    if gaps["penalty"]:
        return "penalty", (f"cheaper on {_plans(sum(gaps['penalty'].values()))}, but its route paid what they kept "
                           "it from: " + ", ".join(f"{why} {n}" for why, n in gaps["penalty"].most_common()))
    if wide is None:
        return "nowhere", "on no route anywhere"
    where = f"{wide[0]:+.2f} s ({wide[1]:+.0%}) from {wide[2]} at x {wide[3]:.0f} to {wide[4]}"
    if wide[0] >= NEAR_TIE_S:
        return "cost", f"never the quickest between two places on the map: at best {where}"
    return "ground", f"the quickest from {wide[2]} at x {wide[3]:.0f} to {wide[4]} (among others), but no trip went so"


# --- the report ----------------------------------------------------------------------------------------------

def report(graph: Graph, sets: list[tuple[str, dict]], focus: tuple = (12,)) -> list[str]:
    """The report, as lines: for each set of runs (name, merged tallies) what it did with the map's moves and
    platforms, the moves it never took and why; then the platforms in `focus` and 23 -> 17 by the two rope hops."""
    wide = map_wide(graph)
    chains = {key_of(m): graph.chains(m) for m in graph.moves}
    stuck = [k for k, v in chains.items() if v is False]
    alone = sorted({k.split()[0] for k, v in chains.items() if v is None})
    dear = [k for k in chains if k not in wide or wide[k][0] >= NEAR_TIE_S]
    kinds = Counter(m.kind for m in graph.moves).most_common()
    lines = [f"Route graph: {len(graph.moves)} moves on map {graph.model.id} "
             f"({', '.join(f'{n} {k}' for k, n in kinds)}).",
             f"Never the quickest between two places on the map (from every {START_STEP} units along each platform "
             f"to each other platform): {len(dear)} moves.",
             f"The chains can't take {len(stuck)} moves from standing where they start (no takeoff passes "
             f"Flow.sure_takeoff's room){': ' + ', '.join(stuck) if stuck else ''}; the mover alone makes the "
             f"{sum(1 for v in chains.values() if v is None)} {' and '.join(alone)}.", ""]
    never = None
    for name, t in sets:
        lines += _set_lines(graph, name, t, wide)
        gone = {key_of(m) for m in graph.moves if not t["moves"].get(key_of(m), [0, 0])[1]}
        never = gone if never is None else never & gone
    if len(sets) > 1:
        lines += [f"Never taken in any of the {len(sets)} sets ({len(never)}): " + ", ".join(sorted(never)), ""]
    for pid in focus:
        lines += _platform_lines(graph, pid, sets)
    lines += _hop_lines(graph, sets)
    return lines


def _set_lines(graph: Graph, name: str, t: dict, wide: dict) -> list[str]:
    moves = t["moves"]
    total = [sum(v[i] for v in moves.values()) for i in range(4)]
    used = sorted((m for m in graph.moves if moves.get(key_of(m), [0, 0])[1] > 0), key=lambda m: -moves[key_of(m)][1])
    planned = sum(1 for m in graph.moves if moves.get(key_of(m), [0])[0] > 0)
    timeouts = {k: n for k, n in t["idle"].items() if k.endswith(" timeout")}
    trips = sum(t["trips"].values())
    out = [f"== {name}: {t['runs']} runs ==",
           f"Plans {sum(p[-1] for p in t['plans'])}; moves planned {total[0]}, taken {total[1]} (missed {total[2]}, a "
           f"hit's {total[3]}); hops over monsters {t['hops']}. Of the graph's {len(graph.moves)} moves, {planned} "
           f"planned and {len(used)} taken.",
           f"Trips {trips}: " + ", ".join(f"{r} {n}" for r, n in _by_result(t["trips"]).most_common())
           + (f"; timed out with no route planned (she never set off) {sum(timeouts.values())}: "
              + ", ".join(f"to {k.split()[0]} {n}" for k, n in sorted(timeouts.items(), key=lambda kv: -kv[1]))
              if timeouts else "")]
    if t["stops"]:
        out.append("Tour stops by ground: " + ", ".join(f"{k} {n}" for k, n in t["stops"].most_common()))
    if t["grounds"]:
        out.append("Hunt loop grounds taken up: " + ", ".join(f"{k} {n}" for k, n in t["grounds"].most_common()))
    secs = sum(v[1] for v in t["platforms"].values()) or 1.0
    out.append("Platforms (visits / share of the time): " + ", ".join(
        f"{p} {v[0]}/{v[1] / secs:.1%}" for p, v in sorted(t["platforms"].items(), key=lambda kv: -kv[1][1])))
    spawn = sorted(graph.model.spawn_platforms)
    missing = [p for p in spawn if not t["arrived"].get(p)]
    out.append(f"Platforms where monsters spawn that a move got her onto: {len(spawn) - len(missing)} of {len(spawn)}"
               + (f"; never: {', '.join(map(str, missing))}" if missing else ""))
    out.append("Taken (planned, taken, missed, a hit's), most first:")
    for m in used:
        p, tk, mi, hi = moves[key_of(m)]
        where = t["elsewhere"].get(key_of(m))
        out.append(f"  {key_of(m):26s} {p:5d} {tk:5d} {mi:4d} {hi:4d}"
                   + (f"  ended on {', '.join(f'{k} x{n}' for k, n in where.most_common())}" if where else ""))
    if t.get("older"):
        out.append(f"Routes planned on an older graph (recorded before they said where they were planned from): "
                   f"{t['older']}; their moves are today's of the same kind and platforms that start nearest ("
                   f"{t.get('loose', 0)} more than {MATCH_DX} units off), and the gaps are today's graph's against "
                   "the routes chosen then")
    if t["unknown"]:
        out.append(f"In the records but not in today's graph ({sum(t['unknown'].values())}): " + ", ".join(
            f"{k} x{n}" for k, n in t["unknown"].most_common(12)))
    unused = [m for m in graph.moves if not moves.get(key_of(m), [0, 0])[1]]
    all_gaps = plan_gaps(graph, t["plans"], unused)
    rows = []
    for m in unused:
        gaps = all_gaps[key_of(m)]
        code, text = why(m, moves.get(key_of(m), [0, 0, 0, 0]), gaps, wide.get(key_of(m)), bool(t.get("older")))
        rows.append((m, gaps, code, text))
    out.append(f"Never taken ({len(rows)}), by why: " + ", ".join(
        f"{r} {n}" for r, n in Counter(row[2] for row in rows).most_common()) + ". Each with the least gap from the "
        f"best route through it to the route chosen, on the plans that could have made it (but those that kept a "
        f"ground off or avoided a monster), and on how many it was within {VARIETY:.0%}:")
    order = ("replanned", "then", "weights", "penalty", "ground", "cost", "nowhere")
    for m, gaps, code, text in sorted(rows, key=lambda r: (order.index(r[2]), r[1]["least"][0] if r[1]["least"]
                                                           else math.inf)):
        least = gaps["least"]
        gap = "no plan could" if least is None else (
            f"{least[0]:+6.2f} s {least[1]:+5.0%} ({least[2]} x {least[3]:.0f} -> {least[4]}), "
            f"{gaps['close']}/{gaps['could']}")
        out.append(f"  {key_of(m):26s} {gap:44s} {code}: {text}")
    out.append("")
    return out


def _drawer() -> str:
    """Who sends her along a route, for the report (`drawn`): the weights, or with route.ROUTES over 1 the trip's
    choice of way."""
    return (f"the choice among {route.ROUTES} ways (temperature {route.TEMPERATURE:g})" if route.ROUTES > 1
            else "the weights")


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def _plans(n: int) -> str:
    return f"{n} plan" + ("" if n == 1 else "s")


def _by_result(trips: Counter) -> Counter:
    """Trips by how they ended, whatever their platform."""
    out = Counter()
    for k, n in trips.items():
        out[k.split(" ", 1)[1]] += n
    return out


def _platform_lines(graph: Graph, pid: int, sets: list[tuple[str, dict]]) -> list[str]:
    """What the runs did on and around platform pid, and going through it on their plans."""
    model = graph.model
    p = model.platform(pid)
    into = [m for m in graph.moves if m.dst == pid]
    spawns = Counter(model.monsters_by_platform.get(pid, []))
    out = [f"== Platform {pid} ({p.left:.0f}..{p.right:.0f}, y {p.y_at((p.left + p.right) / 2):.0f}; spawns "
           + (", ".join(f"{n} {k}" for k, n in spawns.items()) if spawns else "nothing") + ") ==",
           "Onto it: " + ", ".join(key_of(m) for m in into),
           "Off it: " + ", ".join(key_of(m) for m in graph.moves if m.src == pid)]
    for name, t in sets:
        v = t["platforms"].get(pid, [0, 0.0])
        secs = sum(w[1] for w in t["platforms"].values()) or 1.0
        planned = sum(t["moves"].get(key_of(m), [0])[0] for m in into)
        taken = sum(t["moves"].get(key_of(m), [0, 0])[1] for m in into)
        could, close, least = 0, 0, None
        for src, x, dst, chosen, _, keep_off, avoid, n in t["plans"]:
            if dst == pid or src == pid or chosen is None or keep_off or avoid:
                continue
            through = [g for m in into if (g := graph.via(m, src, x, dst)) is not None]
            if not through:
                continue
            gap = min(through) - chosen
            could += n
            close += n if min(through) <= (1 + VARIETY) * chosen else 0
            if least is None or gap < least[0]:
                least = (gap, gap / chosen if chosen else math.inf, src, x, dst)
        expected, _ = drawn(graph, [q for q in t["plans"] if q[0] != pid and q[2] != pid], into)
        out.append(f"{name}: visits {v[0]}, {v[1]:.0f} s ({v[1] / secs:.1%} of the time); moves onto it planned "
                   f"{planned}, taken {taken}; tour stops at it {t['stops'].get(str(pid), 0)}, hunt loop grounds "
                   f"{t['grounds'].get(str(pid), 0)}. Through it, on the plans going elsewhere: "
                   + ("none could" if least is None else
                      f"at best {least[0]:+.2f} s ({least[1]:+.0%}) more than the route chosen (from {least[2]} at x "
                      f"{least[3]:.0f} to {least[4]}), within {VARIETY:.0%} on {close} of {could}; {_drawer()} would "
                      f"send her through it on {expected:.1f} of those"))
    out.append("")
    return out


def _hop_lines(graph: Graph, sets: list[tuple[str, dict]]) -> list[str]:
    """23 -> 17 by the two rope hops (959's rope, then 1040's), against the climb past 23's end at 1040's rope."""
    hops = [m for m in graph.moves if m.kind == "rope_jump" and m.src == 23 and m.dst == 17]
    climb = [m for m in graph.moves if m.kind == "climb" and m.src == 23 and m.dst == 17]
    out = ["== 23 -> 17 by two rope hops =="]
    if not hops:
        return out + ["Not in the graph (route.ROPE_HOPS off, or no hop found).", ""]
    h = hops[0]
    out.append(f"In the graph: {key_of(h)}, off {h.rope_x}'s rope at y {h.hang} onto {h.onto.x if h.onto else None}'s: "
               f"{graph.step(h):.2f} s with its risk, against the climb past 23's end "
               + ", ".join(f"{key_of(c)}: {graph.step(c):.2f} s" for c in climb) + ".")
    for x in (900, 959, 1000):
        a = graph.via(h, 23, x, 17)
        b = min((v for c in climb if (v := graph.via(c, 23, x, 17)) is not None), default=None)
        if a is not None and b is not None:
            out.append(f"  from 23 at x {x}: by the hops {a:.2f} s, by the climb {b:.2f} s ({(a - b) / b:+.0%})")
    for name, t in sets:
        counts = t["moves"].get(key_of(h), [0, 0, 0, 0])
        g = plan_gaps(graph, t["plans"], [h])[key_of(h)]
        least = g["least"]
        expected, _ = drawn(graph, t["plans"], [h])
        out.append(f"{name}: planned {counts[0]}, taken {counts[1]} (missed {counts[2]}); the climb taken "
                   f"{sum(t['moves'].get(key_of(c), [0, 0])[1] for c in climb)}. Plans that could have made it "
                   f"{g['could']}, within {VARIETY:.0%} {g['close']}"
                   + ("" if least is None else
                      f", at best {least[0]:+.2f} s ({least[1]:+.0%}; from {least[2]} at x {least[3]:.0f} to "
                      f"{least[4]})") + f"; {_drawer()} would send her by the hops on {expected:.1f} of them")
    out.append("")
    return out


def picture_rows(graph: Graph, t: dict) -> list[tuple[Move, int, int]]:
    """Each move of the graph with how often it was planned and taken (ui.render.coverage_picture's rows)."""
    return [(m, *t["moves"].get(key_of(m), [0, 0])[:2]) for m in graph.moves]
