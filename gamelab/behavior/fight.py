"""Fights: clearing the monsters in reach (`clear`, what tours use), killing one monster (`kill`), and
shooting what's in her way while she travels (`FightOnTheWay`, the navigator's escort).

A fight is a loop of looks: each look, attack.stance decides what to do about the monsters around her
(throw, jump-throw, jump back, step away, close in, or wait) and control.actions does it. A kill is her
EXP going up after she started throwing at something: vanishing alone isn't one (hit monsters flinch
and hide behind damage numbers, and other players kill monsters too). A star hits the first monster
in its path, whatever its kind, so a kill is matched against everything in the path of the throw: from
the ground, what the claw or the star's search can pick by each monster's body; from a jump, what the
attack's area takes in as high as she is when it picks its target; and short of any wall of her foothold
group (attack.reach).
Kills are counted in the hunter's Report; their drops go to her Looter (behavior.looting).
"""

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from maplehelper.engine import RIGHT
from maplehelper.game import COMBAT

from .. import directions, paths, telemetry
from ..attack import reach
from ..attack.stance import TOO_CLOSE, act_text, air_gap, decide, open_above, pct, pick_target, stance
from ..localization.mapdata import MapModel, MobBody
from ..localization.monsters import (MONSTER_FLINCH_TOP, MONSTER_TOP, SEARCH_SLACK, TRACK_SLACK, Monster, bodies,
                                     dead_one, seen_again)
from ..localization.scene import Scene
from .looting import DropSpot

if TYPE_CHECKING:
    from .hunter import Hunter

LOST_AFTER = 3.0   # s a target may stay unseen (without an EXP message) before she gives up on it
HIDDEN_FOR = 1.8   # s she keeps throwing at a monster she was throwing at once it can't be seen: a hit
                   # one flinches under its damage numbers and the detector loses it for 1-3 s, and
                   # she used to stop throwing until it showed again
EMPTY_PATIENCE = 0.4  # s a fight waits when no monster is seen at all (not even out of reach)
FADING = 1.2       # s a killed monster may still be seen fading out where it died: not thrown at
UNPLACED_LOOKS = 10  # looks in a row that can't place her before a fight gives up on her
IN_HER_WAY = TOO_CLOSE + 40  # a monster of any kind this close on her platform is in her way: fought too
THROWS = ("throw", "jump_throw", "jump_back")  # the stances that throw a star (each an Actions method)
GUARD_SECONDS = 15  # longest spent shooting a monster out of the way while travelling
GUARD_PATIENCE = 0.3  # s such a fight waits with nothing in reach before she travels on


def kills_text(kills: dict[str, int]) -> str:
    """Kills by kind as text: "2 pig, 1 slime", or "none"."""
    return ", ".join(f"{n} {k}" for k, n in kills.items()) or "none"


def add_kills(kills: dict[str, int], more: dict[str, int]) -> None:
    """Count the kills in `more` (by kind) into `kills`."""
    for kind, n in more.items():
        kills[kind] = kills.get(kind, 0) + n


@dataclass
class Report:
    """What she did in a fight (or a tour stop): kills, EXP gains, throws, loot runs, potions, HP lost."""
    kills: dict[str, int] = field(default_factory=dict)
    loot_runs: int = 0
    potions: int = 0
    throws: int = 0
    exp_kills: int = 0         # times her EXP went up: every kill of hers, target or not
    lost: int = 0              # targets that vanished without her EXP going up (hidden, or killed by others)
    damage_taken: float = 0.0  # fraction of HP lost overall
    stop: str = ""

    def __str__(self):
        return (f"kills: {kills_text(self.kills)} (EXP gains: {self.exp_kills}, targets lost: {self.lost}); "
                f"throws: {self.throws}; loot runs: {self.loot_runs}; potions: {self.potions}; "
                f"HP lost: {self.damage_taken:.0%}; stopped because: {self.stop}")

    def count_kill(self, kind: str) -> None:
        """One more kill of `kind`."""
        self.kills[kind] = self.kills.get(kind, 0) + 1

    def add(self, other: "Report") -> None:
        """Count another report's kills, throws, loot runs, potions, EXP gains, lost targets and HP lost in."""
        add_kills(self.kills, other.kills)
        self.throws += other.throws
        self.loot_runs += other.loot_runs
        self.potions += other.potions
        self.exp_kills += other.exp_kills
        self.lost += other.lost
        self.damage_taken += other.damage_taken


@dataclass
class FightState:
    """What clear() keeps between looks."""
    thrown: list[list] = field(default_factory=list)  # [monster, when last seen]: thrown at, kept while hidden
    dying: list[tuple[Monster, float]] = field(default_factory=list)  # kills, and when: seen fading out a moment
    aimed: list[Monster] = field(default_factory=list)  # in the star's path at the last throw
    flinching: list[Monster] = field(default_factory=list)  # hit lately, so maybe still flinching (just_hit)
    kill_spots: list[DropSpot] = field(default_factory=list)  # where kills on her platform died: swept after
    exp_counted: int = 0  # her EXP gains (Report.exp_kills) already told apart as kills
    unseen: int = 0       # looks in a row that couldn't place her (see UNPLACED_LOOKS)
    start_platform: int | None = None  # the platform she's on; leaving it means she was knocked off
    last_act: tuple | None = None  # the last look's stance ...
    last_hidden: bool = False      # ... and whether monsters she was throwing at were hidden then
    calm_since: float | None = None  # since when she's had nothing to do (see clear's `patience`)

    def forget_faded(self, now: float) -> None:
        """Forgets the kills that died FADING s ago or more: they've faded out by `now`."""
        self.dying = [(d, t) for d, t in self.dying if now - t < FADING]

    def fading(self, monsters: list[Monster]) -> list[Monster]:
        """The `monsters` seen fading out where a kill died (one still in `dying`): not thrown at. A kill
        is still seen for a moment, and throwing at it (and then at its "hidden" ghost) wasted throws
        after most kills."""
        return [m for m in monsters if any(seen_again(d, [m]) is m for d, _ in self.dying)]

    def just_hit(self, visible: list[Monster]) -> list[Monster]:
        """The monsters she threw at whose HP bar shows up or drops in this look (`visible`; before
        `follow`): hit since they were last seen, so maybe still flinching, taller (for FLINCH_S, 0.6 s,
        localization.monsters), when her next attack picks its target."""
        out = []
        for m, _ in self.thrown:
            again = seen_again(m, visible)
            if again is not None and again.bar is not None and (m.bar is None or again.bar < m.bar - 0.02):
                out.append(again)
        return out

    def follow(self, visible: list[Monster], now: float) -> None:
        """The monsters she threw at that are among those `visible` again: where they are, seen `now`."""
        for entry in self.thrown:
            if (again := seen_again(entry[0], visible)) is not None:
                entry[0], entry[1] = again, now

    def forget_hidden(self, now: float) -> None:
        """Forgets the monsters thrown at that haven't been seen for HIDDEN_FOR s by `now`."""
        self.thrown = [e for e in self.thrown if now - e[1] < HIDDEN_FOR]

    def hidden(self, now: float) -> list[Monster]:
        """The monsters thrown at that weren't seen in this look (`now`): still there, hit, hidden under
        their damage numbers."""
        return [e[0] for e in self.thrown if e[1] != now]

    def note_throw(self, now: float) -> None:
        """A throw at the monsters in its path (`aimed`): each is kept (once) among those thrown at, seen `now`."""
        self.thrown += [[m, now] for m in self.aimed if not any(e[0] is m for e in self.thrown)]


# What a step of clear()'s loop tells it to do next (None: go on).
_AGAIN = "again"  # look again now
_OVER = "over"    # the fight is over: report.stop says why
_DONE = "done"    # finished: sweep up what fell on her platform, and stop on the kill count or the time


def clear(h: "Hunter", kinds=None, ground: set[int] | None = None, max_kills: int = 10, seconds: float = 120,
          patience: float = 1.5, approach: bool = True, finish: bool = False, far: bool = False) -> Report:
    """Fight like a player: a star hits the nearest monster in reach on the side she faces, so
    she keeps throwing toward the side with the most monsters in reach, one throw per attack
    cooldown, looking around during the cooldown, never idle while one is in reach. She keeps
    them all on one side: one too close makes her step back (or, cornered, fight from where
    she is), never cross over them. With nothing in reach she walks (if `approach`) just near
    enough to the nearest, picking up as she goes; then sweeps up what fell on her platform.
    Kills are her EXP going up. Targets: monsters (of `kinds`) on the `ground` platforms, or
    anywhere if None, and any on her own platform. Stops after `max_kills`, after `seconds`, or
    when nothing has been in reach for `patience` seconds (`EMPTY_PATIENCE` when none is seen at
    all). A monster she was throwing at that goes out of sight is still thrown at for
    `HIDDEN_FOR` seconds: hit, it flinches under its damage numbers, and the detector loses it.
    With `finish`, reaching `max_kills` doesn't end the fight while monsters are still in reach:
    she finishes those (without walking after others), as stopping halfway gave up a good spot.
    Monsters of other kinds on her platform that come close are in her way, and fought too. With
    `far`, monsters (of `kinds`) on any platform in reach are fought too: her darts reach far, and
    their drops are left to the tour (looter.pending)."""
    report = h.report
    deadline = time.monotonic() + seconds
    exp_at_start = report.exp_kills
    st = FightState(exp_counted=exp_at_start)
    placed = None  # the last look that placed her
    while time.monotonic() < deadline:
        enough = report.exp_kills - exp_at_start >= max_kills
        if enough and not finish:
            break
        s = h.look()
        if (step := _place_and_check(h, st, s)) == _OVER:
            _count_late(h, st, placed)
            return report
        if step == _AGAIN:
            continue
        placed = s
        now = time.monotonic()  # read once a look: hidden ones are told by it
        fight, visible, hidden = _targets(h, st, s, kinds, ground, far, now)
        act = _take_stance(h, st, s, fight, visible, hidden, approach and not enough)
        if (step := _carry_out(h, st, s, act, fight, now, patience, enough)) == _OVER:
            return report
        if step == _DONE:
            break
    _count_late(h, st, placed)
    if st.kill_spots:
        h.looter.sweep_after(st.kill_spots)
    report.stop = "kill limit" if report.exp_kills - exp_at_start >= max_kills else "time limit"
    return report


# --- the steps of a fight's look (clear) ---------------------------------------------------------

def _count_late(h: "Hunter", st: FightState, placed: Scene | None) -> None:
    """At the end of a fight: her EXP gains its own looks didn't tell apart as kills (seen by a sweep's
    looks before it ended at the kill count, or by the look it ended on: knocked off, low HP), told apart
    from its last look that placed her (`placed`). Else they were missing from its kills by kind, what a
    tour's stop sums up."""
    if h.report.exp_kills > st.exp_counted and placed is not None:
        _count_kills(h, st, placed, [], h.report.exp_kills - st.exp_counted, time.monotonic())
        st.exp_counted = h.report.exp_kills


def _place_and_check(h: "Hunter", st: FightState, s: Scene) -> str | None:
    """Whether the fight goes on with look `s`: _AGAIN (after a short wait) if it can't place her, and
    _OVER (report.stop says why) once that's gone on for UNPLACED_LOOKS looks, at low HP
    (h.health_check), or when she's left the platform she started on (knocked off). None: go on."""
    if s.where is None or s.platform is None:
        st.unseen += 1
        if st.unseen >= UNPLACED_LOOKS:
            h.report.stop = "lost track of her"
            return _OVER
        time.sleep(0.2)
        return _AGAIN
    st.unseen = 0
    if stop := h.health_check(s):
        h.report.stop = stop
        return _OVER
    if st.start_platform is None:
        st.start_platform = s.platform.id
    if s.platform.id != st.start_platform:
        h.log(f"  she's on platform {s.platform.id} now, not {st.start_platform} (knocked off?)")
        h.report.stop = f"left platform {st.start_platform}"
        return _OVER
    return None


def _targets(h: "Hunter", st: FightState, s: Scene, kinds, ground: set[int] | None, far: bool,
             now: float) -> tuple[list[Monster], list[Monster], list[Monster]]:
    """What's there to fight: the monsters wanted and seen (not fading out where a kill died), less
    the kills her EXP tells of; plus those she threw at that are hidden now. Returns those, the
    monsters seen (but those fading out), and the hidden ones. Those hit lately (st.flinching: their HP
    bar new or lower, or hidden under their damage numbers) may still flinch when her next attack picks."""
    wanted = _wanted(s, kinds, ground, far)
    st.forget_faded(now)
    fading = st.fading(s.monsters)
    wanted = [m for m in wanted if m not in fading]
    visible = [m for m in s.monsters if m not in fading]
    hit = st.just_hit(visible)
    st.follow(visible, now)
    wanted = _count_kills(h, st, s, wanted, h.report.exp_kills - st.exp_counted, now)
    st.exp_counted = h.report.exp_kills
    st.forget_hidden(now)  # only after _count_kills: this look's kill may still be one of these
    hidden = st.hidden(now)
    st.flinching = hit + hidden
    return wanted + hidden, visible, hidden


def body_sizes(model: MapModel | None) -> tuple[dict[str, float], dict[str, float]]:
    """How tall each kind is at least, and half its width, by kind (localization.monsters.bodies: the map's
    game data): what stance and decide go by."""
    body = bodies(model)
    return {kind: b.top for kind, b in body.items()}, {kind: b.half for kind, b in body.items()}


def _take_stance(h: "Hunter", st: FightState, s: Scene, fight: list[Monster], visible: list[Monster],
                 hidden: list[Monster], approach: bool) -> tuple:
    """What to do about the monsters to `fight` (attack.stance, by each kind's size), walking up to them
    only if `approach`. For a throw, everything in the star's path is kept (st.aimed): a kill is matched
    against it."""
    here = s.platform
    body = bodies(h.model)  # each kind's body (game data)
    tops, halves = body_sizes(h.model)
    act = stance(s.where.x, s.where.y, here, fight, approach, h.actions.facing,
                 can_jump=lambda a, b: open_above(h.model, here, a, b), tops=tops, halves=halves)
    # Logged when it changes, and when monsters she was throwing at have just gone hidden.
    if act[:2] != (st.last_act or ())[:2] or (hidden and act[0] != "wait" and not st.last_hidden):
        h.log(f"  {act_text(act, fight, s.where.x, hidden)} (HP {s.hp:.0%})")
    st.last_act, st.last_hidden = act, bool(hidden)
    throwing = act[0] in THROWS
    if throwing:
        st.aimed = _in_path(s, act[1], visible + hidden, air_gap(act[0]), body, st.flinching)
    if telemetry.recording():
        telemetry.emit("stance", act=act[0], text=act_text(act, fight, s.where.x, hidden),
                       aimed=[{"kind": m.kind, "x": round(m.x, 1), "y": round(m.y, 1)} for m in st.aimed]
                       if throwing else [])
    return act


def _carry_out(h: "Hunter", st: FightState, s: Scene, act: tuple, fight: list[Monster], now: float,
               patience: float, enough: bool) -> str | None:
    """Do what the stance `act` says: throw (at those in the star's path, seen `now`), or walk; with
    nothing to do, see _idle. Returns what _idle does (None after a throw or a walk)."""
    if act[0] in THROWS:
        st.calm_since = None
        st.note_throw(now)
        {"throw": h.actions.throw, "jump_throw": h.actions.jump_throw,
         "jump_back": h.actions.jump_back}[act[0]](act[1])
        h.report.throws += 1
        return None
    if act[0] in ("back", "approach"):
        st.calm_since = None
        h.actions.walk(act[1], act[2])
        return None
    return _idle(h, st, s, fight, patience, enough)


def _idle(h: "Hunter", st: FightState, s: Scene, fight: list[Monster], patience: float,
          enough: bool) -> str | None:
    """Nothing to do here: loot, then wait (`patience`, or EMPTY_PATIENCE when no monster is seen at all)
    for something to come into reach. _DONE once she's finished what was in reach after the kill count
    (`enough`); _OVER when nothing came (report.stop says so); None: look again."""
    if st.kill_spots:  # sweep up what fell on her platform, all in one walk
        h.looter.sweep_through(s, st.kill_spots)
        st.kill_spots = []
        return None
    if h.looter.pending:
        h.looter.loot_pending(s)
    if enough:  # finished what was in reach after the kill count
        return _DONE
    st.calm_since = st.calm_since or time.monotonic()
    wait = patience if fight else min(patience, EMPTY_PATIENCE)  # none seen: move on soon
    if time.monotonic() - st.calm_since >= wait:
        h.report.stop = "nothing left in reach"
        return _OVER
    time.sleep(0.1)
    return None


def _wanted(s: Scene, kinds, ground: set[int] | None, far: bool) -> list[Monster]:
    """The monsters to fight: of `kinds`, on the `ground` platforms (or anywhere), on hers, or (with
    `far`) on any platform in reach; and any monster coming close on her platform, whatever its kind
    (a snail too): it's in her way."""
    here, x, y = s.platform.id, s.where.x, s.where.y
    wanted = [m for m in s.monsters if (kinds is None or m.kind in kinds)
              and (ground is None or m.platform in ground or m.platform == here
                   or (far and reach.can_hit(s.platform, x, y, m)))]
    wanted += [m for m in s.monsters if m not in wanted and m.platform == here and abs(m.x - x) < IN_HER_WAY]
    return wanted


def _count_kills(h: "Hunter", st: FightState, s: Scene, wanted: list[Monster], gains: int,
                 now: float) -> list[Monster]:
    """Her EXP went up `gains` times since the last look: those are the kills, among what she was
    throwing at (the last throw's path, and what she's thrown at lately: a kill can come in after she's
    turned to the next one). Each kill's drops go to the sweep after the fight (on her platform) or to
    the tour. Returns `wanted` without the dead."""
    here = s.platform.id
    for _ in range(gains):
        dead = dead_one(st.aimed + [e[0] for e in st.thrown if not any(e[0] is a for a in st.aimed)],
                        s.monsters, s.where.x)
        if dead is None:
            h.log("  a kill (EXP up) she can't place: its drops are left")
            break
        if dead in st.aimed:
            st.aimed.remove(dead)
        st.thrown = [e for e in st.thrown if e[0] is not dead and seen_again(dead, [e[0]]) is None]
        st.dying.append((dead, now))
        wanted = [m for m in wanted if seen_again(dead, [m]) is not m]
        h.report.count_kill(dead.kind)
        drops = DropSpot(dead.x, dead.y, dead.platform, time.monotonic())
        (st.kill_spots if dead.platform == here else h.looter.pending).append(drops)
        h.log(f"  {dead.kind} down at x {dead.x:.0f} (platform {dead.platform})")
        telemetry.emit("event", kind="kill", text=f"{dead.kind} down on platform {dead.platform}",
                       x=round(dead.x, 1), y=round(dead.y, 1))
    return wanted


def _in_path(s: Scene, side: str, monsters: list[Monster], gap_ms: float | None = None,
             body: dict[str, MobBody] | None = None, flinching: list[Monster] = ()) -> list[Monster]:
    """Everything in a star's path that way, whatever its kind: a snail in front of a far pig takes the
    star (and its EXP isn't the pig's kill). From the ground, what the claw's swing or the star's search
    can pick (reach.ground_pick, by each kind's body walking: reach.body_of, half its width from `body`,
    the map's game data): on her platform and off it, a monster off her level only as close as the
    search picks it. From a jump (the attack keyed `gap_ms` after the jump key), what the attack's area
    takes in from as high as she is when it picks its target (reach.in_area, by how tall each kind is at
    least, `body`'s top): monsters on the platform above too, as far as a star reaches, but none that
    stays under its floor (a snail beside her: an EXP gain then came from one of those above). Those of
    `flinching` (hit lately, FightState.just_hit) as tall as they flinch (`body`'s flinch_top): a pig 60
    below her is picked from 111 units then, from 151 walking. None behind a wall of her foothold group
    (reach.wall_to: from 4, those on 2)."""
    x, y = s.where.x, s.where.y
    body = body or bodies()
    ahead = [m for m in monsters if (m.x > x) == (side == RIGHT)]
    rise, out = (0.0 if gap_ms is None else reach.air_rise(gap_ms)), []
    for m in ahead:
        b = body.get(m.kind)  # its body (a pig's if unknown)
        walking, half = reach.body_of(m.kind, body)
        least = b.top if b is not None else MONSTER_TOP["pig"]
        top = walking if gap_ms is None else least  # what each attack's pick goes by (see above)
        if any(m is f for f in flinching):
            top = max(top, b.flinch_top if b is not None else MONSTER_FLINCH_TOP["pig"])
        if gap_ms is None:
            picked = reach.ground_pick(abs(m.x - x), m.y - y, top, half) is not None
        else:
            picked = reach.in_area(abs(m.x - x), m.y - y, top, rise, half)
        if picked and reach.wall_to(s.platform, x, y, m, rise) is None:
            out.append(m)
    return out


@dataclass
class KillState:
    """What kill() keeps between looks."""
    kind: str
    last: tuple[float, float]  # where the target was last seen ...
    platform: int | None       # ... and on which platform (its drops land there)
    misses: int = 0            # looks in a row without her or it
    throws: int = 0
    gone_since: float | None = None   # since when it hasn't been seen
    first_throw: float | None = None  # when she first threw at it: EXP gains since then can be its kill
    flying: list[dict] = field(default_factory=list)  # throws whose hit isn't confirmed yet (see _score_shots)
    had_bar: bool = False      # its HP bar has been seen (it was hit) ...
    dying_since: float | None = None  # ... and since when it's been seen without it: dying, fading out

    def seen(self, monsters: list[Monster]) -> list[Monster]:
        """The `monsters` that can be the target: of its kind, near where it was last seen (searched for
        further once it's out of sight)."""
        slack = TRACK_SLACK if self.gone_since is None else SEARCH_SLACK
        return [m for m in monsters if m.kind == self.kind
                and abs(m.x - self.last[0]) <= slack and abs(m.y - self.last[1]) <= 40]


def kill(h: "Hunter", target: Monster, seconds: float = 30) -> str:
    """Shoot `target` until it dies; then loot (or remember the drops if they're off her
    platform). Returns how it ended: 'killed', 'lost', 'timeout', 'unreachable', or a reason
    to stop (low HP)."""
    st = KillState(target.kind, (target.x, target.y), target.platform)
    deadline = time.monotonic() + seconds
    h.log(f"target: {st.kind} on platform {st.platform}")
    while time.monotonic() < deadline:
        s = h.look()
        if s.where is None or s.platform is None:
            st.misses += 1
            if st.misses > 8:
                return "lost track of her"
            time.sleep(0.15)
            continue
        if stop := h.health_check(s):
            return stop
        found = st.seen(s.monsters)
        ended = _shoot_at(h, st, s, found) if found else _target_gone(h, st)
        if ended:
            return ended
    return "timeout"


# --- the steps of a kill's look (kill) -----------------------------------------------------------

def _target_gone(h: "Hunter", st: KillState) -> str | None:
    """The target isn't in this look: 'killed' (and looted) if her EXP went up since she started throwing
    at it, 'lost' once it's been out of sight LOST_AFTER s; None: look again, after a short wait."""
    st.misses += 1
    st.gone_since = st.gone_since or time.monotonic()
    # Dead only if her EXP went up since she started throwing at it. Vanishing alone
    # isn't enough: hit monsters flinch and hide behind damage numbers, and other
    # players kill monsters too (about half the "kills" counted by vanishing alone
    # weren't hers).
    if st.first_throw is not None and h.take_exp(st.first_throw):
        # One star killed it: the last due before it died (its bar gone, else before it was gone: a kill is
        # seen fading out for 0.7-1.3 s). Those due before that may have hit it too (a hit one hides under
        # its damage numbers): undecided. Those due after it missed it.
        landed = [sh for sh in st.flying if sh["due"] <= (st.dying_since or st.gone_since) + 0.2]
        kill = max(landed, key=lambda sh: sh["due"], default=None)
        for sh in st.flying:
            _log_shot(sh, True if sh is kill else None if any(sh is o for o in landed) else False)
        h.report.count_kill(st.kind)
        h.log(f"{st.kind} down after {st.throws} throws")
        h.looter.collect(DropSpot(st.last[0], st.last[1], st.platform, time.monotonic()))
        return "killed"
    if time.monotonic() - st.gone_since >= LOST_AFTER:
        if st.throws:
            h.report.lost += 1
            h.log(f"{st.kind} gone without her EXP going up after {st.throws} throws: not counted")
        return "lost"
    time.sleep(0.15)
    return None


def _shoot_at(h: "Hunter", st: KillState, s: Scene, found: list[Monster]) -> str | None:
    """The target is in look `s` (the nearest of those `found`): score the throws in flight by its HP bar,
    and throw at it or walk (attack.stance.decide). 'unreachable' if no spot on her platform reaches it;
    None: look again."""
    st.misses, st.gone_since = 0, None
    # An EXP gain while it's still alive was another monster dying (a star hits the first
    # monster in its way): it isn't this one's.
    h.exp_events = [e for e in h.exp_events if s.t - e < 1.5]
    t = min(found, key=lambda m: abs(m.x - st.last[0]))
    st.last, st.platform = (t.x, t.y), t.platform if t.platform is not None else st.platform
    bar = t.bar
    if bar is not None:
        st.had_bar, st.dying_since = True, None
    elif st.had_bar and st.dying_since is None:
        st.dying_since = time.monotonic()  # its bar gone: it died, and fades out where it died
    _note_pick(st.flying, s, t)
    st.flying = _score_shots(st.flying, bar, st.dying_since is not None)
    threats = [o for o in s.monsters if o.platform == s.platform.id and o is not t]
    act = decide(s.where.x, s.where.y, s.platform, t, threats, body_sizes(h.model)[1])
    h.log(f"  {st.kind} {t.x - s.where.x:+.0f},{t.y - s.where.y:+.0f} (bar {pct(bar)}), "
          f"HP {s.hp:.0%}: {' '.join(str(v if isinstance(v, str) else round(v)) for v in act)}")
    if act[0] == "skip":
        return "unreachable"
    if act[0] == "throw":
        dx, dy = t.x - s.where.x, t.y - s.where.y
        shot = None
        if reach.can_shoot(dx, dy, reach=reach.STAR_REACH, kind=t.kind) and act[1] == directions.toward(dx):
            shot = _shot(s, t, dx, dy, h.model)
        keyed = []  # when the attack key goes down: after her last attack's lock, up to 0.7 s from now

        def on_throw(when: float, side: str) -> None:
            keyed.append(when)
        h.actions.throw_listeners.append(on_throw)
        try:
            h.actions.throw(act[1])
        finally:
            h.actions.throw_listeners.remove(on_throw)
        if shot is not None:
            if keyed:  # its star is due that long after the key
                shot["due"] += keyed[0] - shot["at"]
            st.flying.append(shot)
        h.report.throws += 1
        st.throws += 1
        st.first_throw = st.first_throw or time.monotonic()
    else:
        h.actions.walk(act[1], act[2])
    return None


def _shot(s: Scene, t: Monster, dx: float, dy: float, model: MapModel | None = None) -> dict:
    """A throw at `t` (`dx`, `dy` from her), to be scored: what the shot log keeps, its HP bar before
    the throw, and when the star should hit (`due`: the attack's release, then the star's flight, after `at`,
    when its key goes down: now, until _shoot_at knows better). The log also keeps where she and it were
    (map x), her foothold group, and whether a wall of it stood in the way (`clear`: reach.wall_between,
    whether or not her decisions keep to it), so what stops a star can be checked from it; and every monster
    in front of her then (`front`), and where they were at the next look, about when the attack picked its
    target (`pick`, _note_pick), so which one it could pick can be fitted from the log alone."""
    here = s.platform
    on = here if t.platform == here.id else None  # the target's platform, if known
    if on is None and model is not None and t.platform is not None:
        on = model.platform(t.platform)
    clear = reach.wall_between(here, s.where.x, s.where.y, t.x, target=on) is None
    now = time.monotonic()
    return {"kind": t.kind, "dx": int(round(dx)), "dy": int(round(dy)), "her": here.id,
            "target": t.platform, "bar": t.bar, "seen": bool(s.tag_seen),
            "x": int(round(s.where.x)), "tx": int(round(t.x)),
            "her_group": list(here.group) if here.group is not None else None, "clear": clear,
            "y": int(round(s.where.y)), "front": _front(s, directions.toward(dx), t),
            "at": now, "due": now + COMBAT.attack_release_ms / 1000 + abs(dx) / reach.STAR_SPEED}


def _front(s: Scene, side: str, target: Monster | None = None) -> list[dict]:
    """The monsters in look `s` on her `side`, as the shot log keeps them: kind, map x and y, platform, HP
    bar, and whether it's the one thrown at."""
    x = s.where.x
    return [{"kind": m.kind, "x": int(round(m.x)), "y": int(round(m.y)), "platform": m.platform, "bar": m.bar,
             **({"target": True} if m is target else {})}
            for m in s.monsters if (m.x > x) == (side == RIGHT)]


def _note_pick(flying: list[dict], s: Scene, target: Monster) -> None:
    """The first look after each throw in flight, about when its attack picked its target (the look
    starts once the throw's key plan is done): where she and the monsters in front were then."""
    for sh in flying:
        if "pick" not in sh:
            sh["pick"] = {"x": int(round(s.where.x)), "y": int(round(s.where.y)),
                          "front": _front(s, directions.toward(sh["dx"]) if sh["dx"] else RIGHT, target)}


def _score_shots(flying: list[dict], bar: float | None, dying: bool = False) -> list[dict]:
    """A throw hit when the target's HP bar shows up or drops after the star should have arrived;
    it missed when nothing changed within ~0.9 s of that. While it's `dying` (seen without the bar it had)
    every throw waits for the kill (_target_gone). Returns the throws still undecided."""
    if dying:
        return flying
    now, left = time.monotonic(), []
    for sh in flying:
        if now >= sh["due"] - 0.1 and bar is not None and (sh["bar"] is None or bar < sh["bar"] - 0.02):
            _log_shot(sh, True)
        elif now > sh["due"] + 0.9:
            _log_shot(sh, False)
        else:
            left.append(sh)
    return left


def _log_shot(shot: dict, hit: bool | None) -> None:
    """Keep a record of every throw (lab/shots2.jsonl): the reach of a star is measured from these. `hit`
    None: undecided (a throw in flight before the one that killed it)."""
    try:
        with paths.SHOT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**{k: v for k, v in shot.items() if k not in ("bar", "due", "at")},
                                "hit": None if hit is None else bool(hit),
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except OSError:
        pass


class FightOnTheWay:
    """The navigator's escort on a hunting tour: between moves she shoots what's in reach (a short burst,
    never walking after it), and any monster standing in the way of the next move."""

    def __init__(self, h: "Hunter"):
        self.h = h
        self.kinds: set[str] | None = None  # shoot monsters in reach of these kinds only (None: all); any
                                            # in her way is shot

    def target(self, s: Scene) -> Monster | None:
        """The monster to shoot from where she stands, if any is in reach (pick_target's choice)."""
        in_reach = [m for m in s.monsters if reach.can_hit(s.platform, s.where.x, s.where.y, m)
                    and (self.kinds is None or m.kind in self.kinds)]
        return pick_target(s.where.x, s.where.y, s.platform, in_reach) if in_reach else None

    def shoot(self) -> int:
        """A short fight at what's in reach (up to 3 kills, never walking after them); its kills."""
        before = self.h.report.exp_kills
        self.h.clear(self.kinds, max_kills=3, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE, approach=False)
        return self.h.report.exp_kills - before

    def in_way(self, s: Scene, x: float) -> Monster | None:
        """The monster nearest her standing on her platform between her and map x (where her next move
        starts, or where she's sent to), or too close to her, whatever its kind (None if there's none)."""
        lo, hi = sorted((s.where.x, x))
        in_way = [m for m in s.monsters if m.platform == s.platform.id and lo - TOO_CLOSE <= m.x <= hi + TOO_CLOSE]
        return min(in_way, key=lambda m: abs(m.x - s.where.x)) if in_way else None

    def clear_way(self, s: Scene, x: float) -> int | None:
        """Shoot the monster in her way to map x (in_way). Returns the kills, or None if nothing was in
        the way."""
        if (m := self.in_way(s, x)) is None:
            return None
        self.h.log(f"  {m.kind} in the way at x {m.x:.0f}: shooting it first")
        before = self.h.report.exp_kills
        self.h.clear(max_kills=2, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE, approach=False)
        return self.h.report.exp_kills - before
