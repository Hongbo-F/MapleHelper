"""Respawns and drop life, from the telemetry (lab/telemetry): how long after a monster dies on a platform one is seen
there again, how long a drop lies before it's gone without her picking it up, and how full the platforms are when in
view. Each time is a survival estimate over censored observations (Kaplan-Meier: n, the median and its 95% interval,
and the share left at set times), to set behavior.hunt's and sim.world's RESPAWN and DROP_LIFE by.

    python -m gamelab respawn [FILES ...] [--text FILE]

**Which looks count.** A look sees a platform empty only if it searched for monsters with her standing (from mid-jump
the band and the screen are higher than her platform puts them) and all of the platform in the area searched: her
band (LOOK_UP above her feet to LOOK_DOWN below, a monster's body BODY tall held in it) or, for a whole-screen look,
the screen above the bottom bar and clear of the minimap; across, the screen less EDGE either side. A monster seen on
it in any look that searched, with her standing, is one there. From 2026-10-21's evening a scene record says whether
it searched (`searched`), how (`wide`), where it looked for drops (`looked`), what the screen showed (`view`) and
whether she stood (her `grounded`). Before, those are inferred (`looks`): a look searched if it saw a monster, if a
record of what its monsters were followed it (the hunt loop's own looks, and the navigator's for the monsters it keeps
clear of: 'monsters'; a fight's decision: 'stance'), in a probe's run off a trip (a probe looks for minutes with every
look searching), or in `watch`; it was wide if it saw a monster off her band or drops off her platform, in a probe that
looks wide, or in `watch`; she stood if her name tag was seen on her platform's surface, not within AIR_S of a fight's
jump; the hunt loop's own looks that saw a drop on her platform looked for drops all along it. The rest (the
navigator's looks for her landing, the looter's, the rests') say nothing. Where the screen was is put by the camera's
model (localization.pose.Camera: her at its offset, stopped at the map's edges), which the caller passes in.

**(a) Respawn: an episode** starts at a kill on a platform (her 'kill' events; with the continuous feed, its 'died',
others' too) whose monster was seen there in the KILL_SEEN_S before (a kill placed on the wrong monster would start
one on a platform empty long since), if the looks covering the platform after it see it empty: EMPTY_N of them,
EMPTY_S apart at least, from FADE_S after the kill (its monster is seen dying where it died for up to 1.26 s:
MobBody.die_s). A monster seen there then (or a later kill there) ends it at the midpoint between that look and the
last that saw the platform empty, if they're MAX_GAP apart at most; a longer gap (she was away) censors it at the last
empty look, as does the end of the run or HORIZON. A monster seen first with an HP bar is one she had hurt, hidden for a
while under its damage numbers (1-3 s): it never left, and the episode is dropped ('a hurt one back'); so is one never
seen emptied ('not emptied': another monster was there). 'lone' episodes have no other kill on the platform in the
LONE_S before: under a respawn per spawn point (each point's monster back a fixed time after its own death), an
earlier kill's monster coming back first cuts the time short, and in the simulator (8 s, at the spawn point) the
episodes on the basin came out at 3-4 s. 'near' ones had her within NEAR of one of the platform's spawn points at the
kill (whether a spawn point by her spawns at all is what that split could tell). The kill's event comes at the look
that saw her EXP go up, so the times are a little short (in the simulator 0.53 s after the death at the median, 1.8 s
at the 90th percentile).

**(b) Drop life: each drop** is told apart from the others by its kind and where it lies (DROP_DX: drops lie still).
It's born at the kill of hers it lies by (a kill on its platform within LOOT_REACH, at most BIRTH_S before it was first
seen: 'hers'), else when it was first seen ('found': its age then isn't known, so its life comes out short). It's gone
when two looks that looked for drops where it lies, with her standing, her inventory not full (then only mesos are
looked for) and no other drop within OVERLAP_DX (drops drawn over one another are found as one), EMPTY_LOOKS apart see
it empty: at the midpoint between the last look that saw it and the first that didn't. Gone with her by it (within
PICK_DX, or her walk from one look to the next over it) since it was last seen, it's taken to be hers, picked up:
censored at the last look that saw it; never seen gone, censored there too. A kill by it lands drops where it lies,
and one of its kind there would be taken for it: it's followed only until then ('a kill by it'), and what's seen where
it lay is neither it nor a new drop until the spot is seen empty. Other players take drops as well (once they're no
longer the killer's), and nothing in the telemetry tells that from a drop vanishing, so the estimate is of how long a
drop lies until it vanishes or someone else takes it: a lower bound of the game's drop life. Where she hunts, most
drops are picked up or have a kill by them within seconds: in the simulator, whose drops lie 180 s, 21 of the 481 of 8
runs of 10 minutes were seen gone, 316 had a kill by them first.

**(c) How full**: for each platform monsters spawn on, over the looks that covered it (each weighed by the time to
the next look, OCCUPANCY_GAP_S at most), the monsters seen on it on average against its spawn points, and how much
of the time none was seen. Detection misses one now and then, and her own kills keep down the ones she hunts.

The map comes in (this layer doesn't load it: gamelab's `respawn` command does), and so does the camera.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..perception.minimap import MINIMAP_BOX
from ..screen import GAME_SIZE, HUD_TOP, SCREEN_SCALE

VIEW = (GAME_SIZE[0] / SCREEN_SCALE, GAME_SIZE[1] / SCREEN_SCALE)  # map units on screen (localization.pose.VIEW)
MINIMAP = tuple(v / SCREEN_SCALE for v in MINIMAP_BOX)  # the minimap over the screen's top left, in map units
EDGE = 40.0        # map units: a platform is in view only this far inside the screen's sides (a monster half off
                   # screen isn't found, and the camera trails her a little as she moves)
BODY = 60.0        # map units above a platform a monster's body reaches standing (an Orange Mushroom's 55; its walk's
                   # 67 for 120 ms): a band look sees a platform's monsters only if their bodies fit in the band
STAND_DY = 6.0     # map units: her feet this near her platform's surface: standing (analysis.coverage.STAND_DY) ...
AIR_S = 1.3        # ... unless a fight jumped her (a jump throw or a jump back: 'stance') this long before: the record
                   # puts her on the platform her dot is over, but from mid-jump a look's band is up to 77 units higher
                   # than that (her own platform's monsters below it), and its loot strips miss the drops on the ground.
                   # In the live hunt loops, of the looks after a jump stance whose look before saw 2 drops or more on
                   # her platform, 104 of 130 saw none 0.9-1.1 s after it, 13 of 23 at 1.2 s (the simulator's: 353 of
                   # 492, and 89 of 150)
JUMPS = ("jump_throw", "jump_back")  # the stances that jump (behavior.fight.JUMPS)
FADE_S = 1.5       # s after a kill that its monster may still be seen dying (MobBody.die_s: 0.66-1.26 s)
KILL_SEEN_S = 3.0  # s: a kill counts only if a monster of its kind was seen where it died this long before its event,
KILL_DX = 60.0     # within this: the live kills' events came 0.9 s after their monster was last seen at the median,
                   # 1.9 s at the 90th percentile, 2.5 s at the 95th (a hit monster hides under its damage numbers)
EMPTY_N = 2        # looks covering a platform that must see it empty, ...
EMPTY_S = 0.5      # ... this far apart at least, before it counts as emptied: a look misses a monster now and then
MAX_GAP = 2.0      # s: a monster seen this soon after the last look that saw its platform empty came in between
HORIZON = 180.0    # s an episode is followed at most
LONE_S = 15.0      # s with no other kill on the platform before an episode's: 'lone' (MeowDB's slowest respawn per
                   # spawn point is 12.56 s: a 5 s cooldown and the 7.56 s tick)
NEAR = 150.0       # map units from her to one of the platform's spawn points at the kill: 'near'
SPAWN_REACH = 60   # how far above its platform a spawn point may be placed (localization.mapdata.SPAWN_REACH)
LOOT_REACH = 60.0  # map units: a kill's drops spread this far around where it died (behavior.looting.LOOT_REACH)
DROP_DX = 8.0      # map units: a drop seen this near one of its kind seen before is that one (from one look to the next
                   # within a second, 76% of the live drops seen again were within a unit of where they'd been, 89%
                   # within two)
OVERLAP_DX = 15.0  # map units: a drop with another this near isn't seen gone (overlapping drops are found as one)
PICK_DX = 25.0     # map units: she picks up a drop standing this near it (sim.world.PICK_REACH 15, and her place's error)
BIRTH_S = 20.0     # s: a kill of hers this long at most before a drop was first seen, where it lies, is its kill
EMPTY_LOOKS = 1.5  # s between the two looks that must both see a drop gone (behavior.looting.EMPTY_LOOKS: monsters
                   # are drawn over drops)
OCCUPANCY_GAP_S = 1.0  # s: a look counts for how full a platform is until the next look, this long at most
AT = (5.0, 8.0, 12.56, 20.0, 30.0, 60.0, 90.0, 180.0)  # s the share left is reported at (12.56: MeowDB's slowest)
MIN_FOLLOWED = 5   # a time (a quantile, or the share left) is given only where this many were still followed
Z = 1.959964       # the normal quantile of a two-sided 95% interval

_KILL = re.compile(r"^(\S+) down on platform (-?\d+)")
_DIED = re.compile(r"^died(?: #\d+)?: kind (\S+), cause (\w+), x (-?\d+), platform (-?\d+)")


@dataclass
class Look:
    """One scene record, as this report reads it."""
    t: float
    x: float
    y: float
    platform: int | None
    standing: bool                 # on her platform's surface, not in the air (as recorded, or inferred: see the top)
    monsters: list[dict]
    loot: list[dict]
    searched: bool                 # monsters were looked for ...
    wide: bool                     # ... on the whole screen
    looked: list[tuple[int, float, float]]  # where drops were looked for (platform, map x from, to)
    view: tuple[float, float, float, float]  # the map the screen showed above the bottom bar: x0, y0, x1, y1
    full: bool = False             # her inventory was full: only mesos were looked for
    recorded: bool = False         # the record said what it searched (else inferred)


@dataclass
class Kill:
    t: float
    kind: str
    platform: int
    x: float | None
    hers: bool = True


@dataclass
class Episode:
    """A platform emptied by a kill, until a monster was seen on it again (event) or no longer followed."""
    platform: int
    kind: str
    t0: float          # the kill
    time: float        # s from the kill: the respawn's (event) or the last look that saw the platform empty
    event: bool
    width: float       # s between the last empty look and the look that saw one back (0 when censored)
    lone: bool         # no other kill on the platform in the LONE_S before
    near: bool         # her within NEAR of one of its spawn points at the kill
    why: str = ""      # how a censored one ended: 'away' (a gap of over MAX_GAP), 'end' (the run or HORIZON)


@dataclass
class DropLife:
    platform: int
    kind: str
    x: float
    born: float
    hers: bool         # born at a kill of hers (else first seen: its age then unknown)
    time: float        # s from birth: gone (event) or last seen (censored)
    event: bool
    why: str           # 'gone' | 'picked' (gone with her by it) | 'unseen' (never seen gone) | 'a kill by it'
                       # (another kill's drops landed where it lay)


# --- reading the records ---------------------------------------------------------------------------------

def looks(records: list[dict], model, corner, band: tuple[float, float], probe_wide: bool = False) -> list[Look]:
    """The run's looks that placed her, in order (see the top: what each searched, as recorded or inferred).
    `corner(x, y)`: the map point at the screen's top-left with her feet at (x, y) (localization.pose.Camera);
    `band`: how far above and below her feet a look searches (attack.reach.LOOK_UP, LOOK_DOWN); `probe_wide`:
    the run is a probe's, which looks at the whole screen."""
    commands = {str(r.get("command", "")) for r in records if r.get("topic") == "session"}
    probe = any(c.startswith("probe") for c in commands)
    watching = "watch" in commands  # `watch`: every look the feed's, at the whole screen
    up, down = band
    out: list[Look] = []
    tripping = False    # between a route and the trip's end (probe runs: the navigator's looks)
    jumped = -math.inf  # when a fight last jumped her
    for i, r in enumerate(records):
        topic = r.get("topic")
        if topic == "route":
            tripping = True
        elif topic == "trip":
            tripping = False
        elif topic == "stance" and r.get("act") in JUMPS:
            jumped = r.get("t", jumped)
        elif topic == "drops" and out and i and records[i - 1] is out[-1].record:
            out[-1].look.full = bool(r.get("inventory_full"))  # the drops she remembers, after each look
        if topic != "scene" or not r.get("her") or r["her"].get("x") is None:
            continue
        her = r["her"]
        x, y, pid = her["x"], her["y"], her.get("platform")
        p = model.platform(pid) if pid is not None and 0 <= pid < len(model.platforms) else None
        monsters, loot = r.get("monsters") or [], r.get("loot") or []
        view = tuple(r["view"]) if r.get("view") else _view(corner, x, y)
        if "searched" in r:
            searched, wide, recorded = bool(r["searched"]), bool(r.get("wide")), True
            looked = [tuple(lk) for lk in r.get("looked") or []]
            standing = bool(her.get("grounded"))
        else:
            recorded = False
            after = _after(records, i)
            searched = (bool(monsters) or bool(after & {"monsters", "stance"}) or (probe and not tripping)
                        or watching)
            wide = (any(m["y"] < y - up - 5 or m["y"] > y + down + 5 for m in monsters)
                    or any(d.get("platform") != pid for d in loot) or (probe and not tripping and probe_wide)
                    or watching)
            standing = (her.get("seen", True) and p is not None and abs(p.y_at(x) - y) <= STAND_DY
                        and not 0 <= r.get("t", 0.0) - jumped < AIR_S)
            # The hunt loop's own looks (a 'monsters' record after) look for drops: on her platform at least. Only those
            # that saw one there show it was looked at (from mid-jump, or while a jump lasts longer than AIR_S, a look
            # sees none): a drop's absence counts only from a look that saw others on its platform.
            looked = ([(pid, view[0], view[2])] if "monsters" in after and pid is not None and standing
                      and any(d.get("platform") == pid for d in loot) else [])
        out.append(_Read(r, Look(r.get("t", 0.0), x, y, pid, standing, monsters, loot, searched, wide, looked, view,
                                 recorded=recorded)))
    return [o.look for o in out]


@dataclass
class _Read:
    record: dict
    look: Look


def _after(records: list[dict], i: int) -> set[str]:
    """The topics of the records after record i, up to the next look (a few at most)."""
    seen = set()
    for r in records[i + 1:i + 8]:
        if r.get("topic") == "scene":
            break
        seen.add(r.get("topic"))
    return seen


def _view(corner, x: float, y: float) -> tuple[float, float, float, float]:
    """The map the screen shows above the bottom bar, with her feet at (x, y), by the camera's model."""
    cx, cy = corner(x, y)
    return cx, cy, cx + VIEW[0], cy + VIEW[1] * HUD_TOP


def kills(records: list[dict]) -> list[Kill]:
    """The run's kills: hers ('kill' events: the look that saw her EXP go up), and with the continuous feed, the
    deaths it saw ('died': hers and others')."""
    out = []
    for r in records:
        if r.get("topic") != "event":
            continue
        text = r.get("text", "")
        if r.get("kind") == "kill" and (m := _KILL.match(text)):
            out.append(Kill(r["t"], m.group(1), int(m.group(2)), r.get("x")))
        elif r.get("kind") == "died" and (m := _DIED.match(text)):
            out.append(Kill(r["t"], m.group(1), int(m.group(4)), float(m.group(3)), m.group(2) == "hers"))
    return out


# --- what a look covered --------------------------------------------------------------------------------

def covers(look: Look, p, band: tuple[float, float]) -> bool:
    """Whether look `look` searched all of platform `p` for monsters, with her standing (see the top)."""
    if not look.searched or not look.standing:
        return False
    x0, y0, x1, y1 = look.view
    if not look.wide:
        y0, y1 = max(y0, look.y - band[0]), min(y1, look.y + band[1])
    lo = min(min(s.y1, s.y2) for s in p.segments)
    hi = max(max(s.y1, s.y2) for s in p.segments)
    if not (x0 + EDGE <= p.left and p.right <= x1 - EDGE and lo - BODY >= y0 and hi <= y1):
        return False
    if look.wide:  # the minimap hides what's under it
        top = look.view[1]
        if p.left < x0 + MINIMAP[2] and p.right > x0 + MINIMAP[0] and lo - BODY < top + MINIMAP[3] \
                and hi > top + MINIMAP[1]:
            return False
    return True


def _on(look: Look, pid: int) -> list[dict]:
    return [m for m in look.monsters if m.get("platform") == pid]


# --- (a) respawns -----------------------------------------------------------------------------------------

def respawns(seq: list[Look], dead: list[Kill], model, band: tuple[float, float]) -> tuple[list[Episode], Counter]:
    """The episodes of one run (see the top), and what became of the kills that made none."""
    episodes, dropped = [], Counter()
    by_platform = defaultdict(list)
    for k in dead:
        by_platform[k.platform].append(k.t)
    points = defaultdict(list)  # platform -> its spawn points (x, y)
    for sp in model.spawns:
        if (q := model.platform_under(sp.x, sp.y, reach=SPAWN_REACH)) is not None:
            points[q.id].append((sp.x, q.y_at(sp.x)))
    times = [lk.t for lk in seq]
    for k in dead:
        if not 0 <= k.platform < len(model.platforms):
            dropped["off the map"] += 1
            continue
        p = model.platform(k.platform)
        lone = not any(k.t - LONE_S <= t < k.t for t in by_platform[k.platform])
        after = min((t for t in by_platform[k.platform] if t > k.t), default=math.inf)
        at = seq[max(0, _bisect(times, k.t + 1e-6) - 1)] if seq else None  # where she was at the kill
        near = at is not None and any(math.hypot(at.x - sx, at.y - sy) <= NEAR for sx, sy in points[k.platform])
        got = _episode(seq, times, k, p, band, lone, near, after)
        if isinstance(got, str):
            dropped[got] += 1
        else:
            episodes.append(got)
    return episodes, dropped


def _episode(seq: list[Look], times: list[float], k: Kill, p, band, lone: bool, near: bool,
             after: float) -> "Episode | str":
    """Kill k's episode on platform p (see the top), or why it makes none. `after`: the next kill there (its monster
    was there before it, seen or not)."""
    i = _bisect(times, k.t)
    if not any(m.get("platform") == p.id and m.get("kind") == k.kind and (k.x is None or abs(m["x"] - k.x) <= KILL_DX)
               for lk in seq[_bisect(times, k.t - KILL_SEEN_S):i] for m in lk.monsters):
        return "its monster not seen there"
    empties: list[float] = []

    def emptied() -> bool:
        return len(empties) >= EMPTY_N and empties[-1] - empties[0] >= EMPTY_S

    def back(at: float, hurt: bool) -> "Episode | str":
        """A monster on the platform at `at`: the episode's end, if it had been seen emptied."""
        if not emptied():
            return "not emptied"
        if hurt:
            return "a hurt one back"
        last = empties[-1]
        if at - last > MAX_GAP:
            return Episode(p.id, k.kind, k.t, last - k.t, False, 0.0, lone, near, "away")
        return Episode(p.id, k.kind, k.t, (last + at) / 2 - k.t, True, at - last, lone, near)

    for lk in seq[_bisect(times, k.t + FADE_S):]:
        if lk.t - k.t > HORIZON:
            break
        if lk.t >= after:
            return back(after, False)
        # A monster seen there in any look that searched with her standing is one back, the whole platform in view or
        # not; only a look that covered all of it sees it empty.
        if seen := (_on(lk, p.id) if lk.searched and lk.standing else []):
            return back(lk.t, any(m.get("bar") is not None for m in seen))
        if covers(lk, p, band):
            if empties and lk.t - empties[-1] > MAX_GAP and not emptied():
                empties = []  # a gap before it was seen emptied: start again (its emptiness then isn't known)
            empties.append(lk.t)
    if not emptied():
        return "not covered after" if not empties else "not emptied"
    return Episode(p.id, k.kind, k.t, empties[-1] - k.t, False, 0.0, lone, near, "end")


def _bisect(times: list[float], t: float) -> int:
    """The index of the first time at or after t."""
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --- (b) drop life ----------------------------------------------------------------------------------------

@dataclass
class _Drop:
    platform: int
    kind: str
    x: float
    last: float
    born: float
    hers: bool
    empty_since: float | None = None
    near: bool = False  # she has been by it since it was last seen (or in the look that saw it last)


def drops(seq: list[Look], dead: list[Kill]) -> list[DropLife]:
    """The drops of one run and how each ended (see the top). Told apart one by one, not by pile: where a basin's
    kills keep dropping, a pile of drops near one another lives on long after its first drops are gone."""
    open_: list[_Drop] = []
    blocked: list[_Drop] = []  # drops no longer followed (a kill by them) but maybe still there: what's seen where
    out: list[DropLife] = []   # one lies is neither it nor a new one, until its place is seen empty
    mine = [k for k in dead if k.hers]
    landing = sorted((k for k in dead if k.x is not None), key=lambda k: k.t)
    j = 0
    before: Look | None = None
    for lk in seq:
        # A kill by a drop lands drops where it lies, and one of its kind there would be taken for it: each drop is
        # followed until the next kill by it (censored at the look that saw it last).
        while j < len(landing) and landing[j].t <= lk.t:
            k = landing[j]
            j += 1
            for q in [q for q in open_ if q.platform == k.platform and abs(q.x - k.x) <= LOOT_REACH + DROP_DX
                      and q.born < k.t - 1e-6]:
                open_.remove(q)
                blocked.append(q)
                out.append(DropLife(q.platform, q.kind, q.x, q.born, q.hers, q.last - q.born, False, "a kill by it"))
        # Where she went since the look before, on one platform: the looter walks over a row of piles pressing pick-up
        # with no look in between (looting.walk_piles).
        walked = ((min(before.x, lk.x), max(before.x, lk.x)) if before is not None and before.platform == lk.platform
                  else (lk.x, lk.x))
        before = lk
        seen = set()
        for d in lk.loot:
            pid, x, kind = d.get("platform"), d.get("x"), d.get("kind")
            if pid is None or x is None:
                continue
            q = min((q for q in open_ + blocked if q.platform == pid and q.kind == kind and abs(q.x - x) <= DROP_DX
                     and id(q) not in seen), key=lambda q: abs(q.x - x), default=None)
            if q is None:
                kill = max((k for k in mine if k.platform == pid and k.x is not None and abs(k.x - x) <= LOOT_REACH
                            and lk.t - BIRTH_S <= k.t <= lk.t), key=lambda k: k.t, default=None)
                q = _Drop(pid, kind, x, lk.t, kill.t if kill else lk.t, kill is not None)
                open_.append(q)
            q.last, q.empty_since, q.near = lk.t, None, lk.platform == pid and abs(lk.x - x) <= PICK_DX
            seen.add(id(q))
        for q in open_ + blocked:
            if id(q) in seen:
                continue
            if lk.platform == q.platform and walked[0] - PICK_DX <= q.x <= walked[1] + PICK_DX:
                q.near = True
            if not _gone(q, lk):
                continue
            if q in blocked:
                blocked.remove(q)
            else:
                open_.remove(q)
                if q.near:
                    out.append(DropLife(q.platform, q.kind, q.x, q.born, q.hers, q.last - q.born, False, "picked"))
                else:
                    out.append(DropLife(q.platform, q.kind, q.x, q.born, q.hers,
                                        (q.last + q.empty_since) / 2 - q.born, True, "gone"))
    out += [DropLife(q.platform, q.kind, q.x, q.born, q.hers, q.last - q.born, False, "unseen") for q in open_]
    return out


def _gone(q: _Drop, lk: Look) -> bool:
    """Whether look lk, not seeing drop q, makes it gone: it looked where q lies, with her standing (and her inventory
    not full, or q mesos), no other drop over it, EMPTY_LOOKS after the first such look."""
    if not (lk.standing and (not lk.full or q.kind == "mesos")
            and any(pid == q.platform and lo <= q.x <= hi for pid, lo, hi in lk.looked)):
        return False
    if any(d.get("platform") == q.platform and abs(d.get("x", math.inf) - q.x) <= OVERLAP_DX for d in lk.loot):
        return False  # another drop lies over it: one drawn over another is found as one (and not always the same)
    if q.empty_since is None:
        q.empty_since = lk.t
        return False
    return lk.t - q.empty_since >= EMPTY_LOOKS


# --- (c) how full -----------------------------------------------------------------------------------------

def occupancy(seq: list[Look], model, band: tuple[float, float]) -> dict[int, list[float]]:
    """Platform -> [seconds covered, monster-seconds seen on it, seconds seen empty] (see the top)."""
    out: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    spawn = model.spawn_platforms
    for lk, nxt in zip(seq, seq[1:]):
        w = min(max(0.0, nxt.t - lk.t), OCCUPANCY_GAP_S)
        if w <= 0:
            continue
        for pid in spawn:
            if covers(lk, model.platform(pid), band):
                n = len(_on(lk, pid))
                row = out[pid]
                row[0] += w
                row[1] += w * n
                row[2] += w * (n == 0)
    return out


# --- the estimator --------------------------------------------------------------------------------------

def km(obs: list[tuple[float, bool]]) -> list[tuple[float, float, float]]:
    """The Kaplan-Meier estimate of the survival function from (time, event) pairs (event False: censored then; at a
    time with both, the censored are still at risk): at each event time, (t, S(t), Greenwood's sum of d/(n(n-d)))."""
    out, s, g = [], 1.0, 0.0
    obs = sorted(obs)
    at_risk, i = len(obs), 0
    while i < len(obs):
        t, d, c = obs[i][0], 0, 0
        while i < len(obs) and obs[i][0] == t:
            d, c, i = d + obs[i][1], c + (not obs[i][1]), i + 1
        if d:
            s *= 1 - d / at_risk
            g += d / (at_risk * (at_risk - d)) if at_risk > d else math.inf
            out.append((t, s, g))
        at_risk -= d + c
    return out


def _band(s: float, g: float) -> tuple[float, float]:
    """The 95% interval of S = s with Greenwood's sum g, on the log-log scale (it stays within 0 to 1)."""
    if not 0 < s < 1 or not math.isfinite(g):
        return s, s
    se = math.sqrt(g) / abs(math.log(s))
    return s ** math.exp(Z * se), s ** math.exp(-Z * se)


def survival(curve: list[tuple[float, float, float]], t: float) -> tuple[float, float, float]:
    """S(t) from km's curve, with its 95% interval."""
    s, g = 1.0, 0.0
    for ti, si, gi in curve:
        if ti > t:
            break
        s, g = si, gi
    return (s, *_band(s, g))


def quantile(curve: list[tuple[float, float, float]], q: float = 0.5) -> tuple[float | None, float | None,
                                                                              float | None]:
    """The time by which a share q of them had happened (S(t) <= 1 - q), with its 95% interval (Brookmeyer and
    Crowley's: where the log-log interval of S crosses 1 - q, as R's survfit gives it). None: not reached (the
    estimate, or the interval's end, is past the last event)."""
    level = 1 - q + 1e-12
    est = lo = hi = None
    for t, s, g in curve:
        lower, upper = _band(s, g)
        if est is None and s <= level:
            est = t
        if lo is None and lower <= level:
            lo = t
        if hi is None and upper <= level:
            hi = t
    return est, lo, hi


def summary(obs: list[tuple[float, bool]]) -> dict:
    """n, events, the median with its 95% interval, the quartiles, the share left at each of AT (with its interval
    and how many were still followed then), and the longest followed. A time is given only where MIN_FOLLOWED were
    still followed (past that the curve rests on one or two, and an estimate there says nothing)."""
    curve = km(obs)

    def followed(t: float | None) -> float | None:
        return t if t is not None and sum(1 for o in obs if o[0] >= t) >= MIN_FOLLOWED else None

    med, lo, hi = (followed(t) for t in quantile(curve, 0.5))
    return {"n": len(obs), "events": sum(e for _, e in obs), "median": med, "lo": lo, "hi": hi,
            "q1": followed(quantile(curve, 0.25)[0]), "q3": followed(quantile(curve, 0.75)[0]),
            "left": [(t, *survival(curve, t), n) for t in AT if (n := sum(1 for o in obs if o[0] >= t)) >= MIN_FOLLOWED],
            "longest": max((t for t, _ in obs), default=None)}


# --- the report -------------------------------------------------------------------------------------------

@dataclass
class Runs:
    """What the runs gave, together."""
    files: int = 0
    looks: int = 0
    recorded: int = 0              # looks whose record said what they searched
    searched: int = 0
    kills: int = 0
    others: int = 0                # deaths the feed saw that weren't hers
    episodes: list[Episode] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    drops: list[DropLife] = field(default_factory=list)
    full: dict = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))

    def add(self, records: list[dict], model, corner, band: tuple[float, float], probe_wide: bool = False) -> None:
        seq = looks(records, model, corner, band, probe_wide)
        dead = kills(records)
        self.files += 1
        self.looks += len(seq)
        self.recorded += sum(lk.recorded for lk in seq)
        self.searched += sum(lk.searched for lk in seq)
        self.kills += sum(k.hers for k in dead)
        self.others += sum(not k.hers for k in dead)
        eps, dropped = respawns(seq, dead, model, band)
        self.episodes += eps
        self.dropped.update(dropped)
        self.drops += drops(seq, dead)
        for pid, row in occupancy(seq, model, band).items():
            for j in range(3):
                self.full[pid][j] += row[j]


def _s(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _lines(name: str, obs: list[tuple[float, bool]], table: bool = True) -> list[str]:
    s = summary(obs)
    if not s["n"]:
        return [f"  {name}: none"]
    ci = (f"{_s(s['lo'])}-{_s(s['hi'])}" if s["lo"] is not None and s["hi"] is not None
          else f"from {_s(s['lo'])}" if s["lo"] is not None else "-")
    out = [f"  {name}: n {s['n']} ({s['events']} seen, {s['n'] - s['events']} censored); median {_s(s['median'])} s "
           f"(95% {ci}), quartiles {_s(s['q1'])} / {_s(s['q3'])} s; the longest followed {_s(s['longest'])} s"]
    if table:
        out.append("    share left: " + (", ".join(f"{t:g} s {v:.2f} ({lo:.2f}-{hi:.2f}, {n} followed)"
                                                   for t, v, lo, hi, n in s["left"]) or "-"))
    return out


def report(runs: Runs, model, game: dict | None = None) -> list[str]:
    """The report's lines. `game`: the map's spawn fields from the game data (spawn_fields)."""
    lines = [f"{runs.files} files: {runs.looks} looks placing her ({runs.recorded} saying what they searched, "
             f"{runs.searched} searched for monsters, as recorded or inferred), {runs.kills} kills of hers"
             + (f", {runs.others} deaths of others' (the feed's)" if runs.others else "")]
    if game:
        lines.append(f"game data: {game['spawns']} spawn points; mobTime {game['mob_time']}; mobRate "
                     f"{game['mob_rate']}; createMobInterval {game['create_mob_interval']}")
    lines += ["", "(a) respawn: from a kill that emptied its platform to a monster seen there again"]
    eps = runs.episodes
    lines += _lines("all", [(e.time, e.event) for e in eps])
    lines += _lines(f"lone (no other kill there in the {LONE_S:g} s before)", [(e.time, e.event) for e in eps if e.lone])
    lines += _lines(f"near (her within {NEAR:g} of one of its spawn points)", [(e.time, e.event) for e in eps if e.near])
    lines += _lines("not near", [(e.time, e.event) for e in eps if not e.near])
    widths = sorted(e.width for e in eps if e.event)
    if widths:
        lines.append(f"  the look that saw one back, after the last that saw none: {widths[len(widths) // 2]:.2f} s "
                     f"apart at the median, at most {widths[-1]:.2f} (MAX_GAP {MAX_GAP:g})")
    censored = Counter(e.why for e in eps if not e.event)
    if censored:
        lines.append("  censored: " + ", ".join(f"{n} {why}" for why, n in censored.most_common()))
    if runs.dropped:
        lines.append("  kills that made no episode: " + ", ".join(f"{n} {why}" for why, n in runs.dropped.most_common()))
    spawn = model.spawn_platforms
    for pid in sorted({e.platform for e in eps}):
        lines += _lines(f"platform {pid} ({spawn.get(pid, 0)} spawn points)",
                        [(e.time, e.event) for e in eps if e.platform == pid], table=False)
    if times := sorted(round(e.time, 1) for e in eps if e.event):
        lines.append("  the respawns seen (s): " + " ".join(f"{t:g}" for t in times))
    if cens := sorted(round(e.time, 1) for e in eps if not e.event):
        lines.append("  censored at (s): " + " ".join(f"{t:g}" for t in cens))
    lines += ["", "(b) drop life: from a drop's kill (hers) or first sight (found) until it's gone, not by her"]
    ds = runs.drops
    lines += _lines("all", [(d.time, d.event) for d in ds])
    lines += _lines("hers (from the kill)", [(d.time, d.event) for d in ds if d.hers])
    lines += _lines("found (from first seen: older than that)", [(d.time, d.event) for d in ds if not d.hers])
    lines.append("  how they ended: " + ", ".join(f"{n} {why}" for why, n in Counter(d.why for d in ds).most_common()))
    if gone := sorted(round(d.time) for d in ds if d.event):
        lines.append("  gone at (s): " + " ".join(f"{t:g}" for t in gone))
    if long := sorted((round(d.time) for d in ds if not d.event), reverse=True)[:12]:
        lines.append("  the longest seen lying (censored, s): " + " ".join(f"{t:g}" for t in long))
    lines += ["", "(c) how full: the platforms monsters spawn on, over the looks that covered them"]
    covered = seen_all = points = 0.0
    for pid in sorted(spawn):
        cov, seen, empty = runs.full.get(pid, (0.0, 0.0, 0.0))
        if cov <= 0:
            lines.append(f"  platform {pid} ({spawn[pid]} spawn points): never covered")
            continue
        covered, seen_all, points = covered + cov, seen_all + seen, points + cov * spawn[pid]
        lines.append(f"  platform {pid} ({spawn[pid]} spawn points): covered {cov:.0f} s, {seen / cov:.2f} monsters "
                     f"on average ({seen / cov / spawn[pid]:.0%} of its points), none {empty / cov:.0%} of the time")
    if points:
        lines.append(f"  all of them: {seen_all / points:.0%} of their spawn points had a monster, over {covered:.0f} s "
                     "of platforms covered")
    return lines


def spawn_fields(tree: dict) -> dict:
    """The map image's own spawn fields (its 'life' entries' mobTime, info's mobRate and createMobInterval)."""
    lives = [s for s in (tree.get("life") or {}).values() if isinstance(s, dict) and s.get("type") == "m"]
    info = tree.get("info") or {}
    return {"spawns": len(lives), "mob_time": dict(Counter(s.get("mobTime") for s in lives)),
            "mob_rate": info.get("mobRate"), "create_mob_interval": info.get("createMobInterval", "none")}
