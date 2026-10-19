"""Fights: clearing the monsters in reach (`clear`, what tours use), killing one monster (`kill`), and
shooting what's in her way while she travels (`FightOnTheWay`, the navigator's escort).

A fight is a loop of looks: each look, attack.stance decides what to do about the monsters around her
(throw, jump-throw, jump back, step away, close in, or wait) and control.actions does it. A kill is her
EXP going up after she started throwing at something: vanishing alone isn't one (hit monsters flinch
and hide behind damage numbers, and other players kill monsters too). A star hits the first monster
in its path, whatever its kind, so a kill is matched against everything in the path of the throw.
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
from ..attack.stance import TOO_CLOSE, act_text, decide, open_above, pct, pick_target, stance
from ..localization.monsters import SEARCH_SLACK, TRACK_SLACK, Monster, dead_one, seen_again
from ..localization.scene import Scene
from ..motion.route import Move
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
        """Count another report's kills, throws, loot runs, potions, EXP gains and lost targets in."""
        add_kills(self.kills, other.kills)
        self.throws += other.throws
        self.loot_runs += other.loot_runs
        self.potions += other.potions
        self.exp_kills += other.exp_kills
        self.lost += other.lost


@dataclass
class FightState:
    """What clear() keeps between looks."""
    thrown: list[list] = field(default_factory=list)  # [monster, when last seen]: thrown at, kept while hidden
    dying: list[tuple[Monster, float]] = field(default_factory=list)  # kills, and when: seen fading out a moment
    aimed: list[Monster] = field(default_factory=list)  # in the star's path at the last throw
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
    while time.monotonic() < deadline:
        enough = report.exp_kills - exp_at_start >= max_kills
        if enough and not finish:
            break
        s = h.look()
        if (step := _place_and_check(h, st, s)) == _OVER:
            return report
        if step == _AGAIN:
            continue
        now = time.monotonic()  # read once a look: hidden ones are told by it
        fight, visible, hidden = _targets(h, st, s, kinds, ground, far, now)
        act = _take_stance(h, st, s, fight, visible, hidden, approach and not enough)
        if (step := _carry_out(h, st, s, act, fight, now, patience, enough)) == _OVER:
            return report
        if step == _DONE:
            break
    if st.kill_spots:
        h.looter.sweep_after(st.kill_spots)
    report.stop = "kill limit" if report.exp_kills - exp_at_start >= max_kills else "time limit"
    return report


# --- the steps of a fight's look (clear) ---------------------------------------------------------

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
    monsters seen (but those fading out), and the hidden ones."""
    wanted = _wanted(s, kinds, ground, far)
    st.forget_faded(now)
    fading = st.fading(s.monsters)
    wanted = [m for m in wanted if m not in fading]
    visible = [m for m in s.monsters if m not in fading]
    st.follow(visible, now)
    wanted = _count_kills(h, st, s, wanted, h.report.exp_kills - st.exp_counted, now)
    st.exp_counted = h.report.exp_kills
    st.forget_hidden(now)  # only after _count_kills: this look's kill may still be one of these
    hidden = st.hidden(now)
    return wanted + hidden, visible, hidden


def _take_stance(h: "Hunter", st: FightState, s: Scene, fight: list[Monster], visible: list[Monster],
                 hidden: list[Monster], approach: bool) -> tuple:
    """What to do about the monsters to `fight` (attack.stance), walking up to them only if `approach`.
    For a throw, everything in the star's path is kept (st.aimed): a kill is matched against it."""
    here = s.platform
    act = stance(s.where.x, s.where.y, here, fight, approach, h.actions.facing,
                 can_jump=lambda a, b: open_above(h.model, here, a, b))
    # Logged when it changes, and when monsters she was throwing at have just gone hidden.
    if act[:2] != (st.last_act or ())[:2] or (hidden and act[0] != "wait" and not st.last_hidden):
        h.log(f"  {act_text(act, fight, s.where.x, hidden)} (HP {s.hp:.0%})")
    st.last_act, st.last_hidden = act, bool(hidden)
    throwing = act[0] in THROWS
    if throwing:
        st.aimed = _in_path(s, act[1], visible + hidden)
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
                   or (far and reach.can_shoot(m.x - x, m.y - y)))]
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


def _in_path(s: Scene, side: str, monsters: list[Monster]) -> list[Monster]:
    """Everything in a star's path that way, whatever its kind: a snail in front of a far pig takes the
    star (and its EXP isn't the pig's kill)."""
    x, y, here = s.where.x, s.where.y, s.platform.id
    return [m for m in monsters if (m.x > x) == (side == RIGHT)
            and (abs(m.x - x) <= reach.STAR_REACH if m.platform == here
                 else reach.can_shoot(m.x - x, m.y - y, reach.STAR_REACH))]


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
        for sh in st.flying:
            _log_shot(sh, sh["due"] <= st.gone_since + 0.2)
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
    st.flying = _score_shots(st.flying, bar)
    threats = [o for o in s.monsters if o.platform == s.platform.id and o is not t]
    act = decide(s.where.x, s.where.y, s.platform, t, threats)
    h.log(f"  {st.kind} {t.x - s.where.x:+.0f},{t.y - s.where.y:+.0f} (bar {pct(bar)}), "
          f"HP {s.hp:.0%}: {' '.join(str(v if isinstance(v, str) else round(v)) for v in act)}")
    if act[0] == "skip":
        return "unreachable"
    if act[0] == "throw":
        dx, dy = t.x - s.where.x, t.y - s.where.y
        if reach.can_shoot(dx, dy, reach=reach.STAR_REACH) and act[1] == directions.toward(dx):
            st.flying.append(_shot(s, t, dx, dy))
        h.actions.throw(act[1])
        h.report.throws += 1
        st.throws += 1
        st.first_throw = st.first_throw or time.monotonic()
    else:
        h.actions.walk(act[1], act[2])
    return None


def _shot(s: Scene, t: Monster, dx: float, dy: float) -> dict:
    """A throw at `t` (`dx`, `dy` from her), to be scored: what the shot log keeps, its HP bar before
    the throw, and when the star should hit (`due`: the attack's release, then the star's flight)."""
    return {"kind": t.kind, "dx": int(round(dx)), "dy": int(round(dy)), "her": s.platform.id,
            "target": t.platform, "bar": t.bar, "seen": bool(s.tag_seen),
            "due": time.monotonic() + COMBAT.attack_release_ms / 1000 + abs(dx) / reach.STAR_SPEED}


def _score_shots(flying: list[dict], bar: float | None) -> list[dict]:
    """A throw hit when the target's HP bar shows up or drops after the star should have arrived;
    it missed when nothing changed within ~0.9 s of that. Returns the throws still undecided."""
    now, left = time.monotonic(), []
    for sh in flying:
        if now >= sh["due"] - 0.1 and bar is not None and (sh["bar"] is None or bar < sh["bar"] - 0.02):
            _log_shot(sh, True)
        elif now > sh["due"] + 0.9:
            _log_shot(sh, False)
        else:
            left.append(sh)
    return left


def _log_shot(shot: dict, hit: bool) -> None:
    """Keep a record of every throw (lab/shots.jsonl): the reach of a star is measured from these."""
    try:
        with paths.SHOT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**{k: v for k, v in shot.items() if k not in ("bar", "due")}, "hit": bool(hit),
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except OSError:
        pass


class FightOnTheWay:
    """The navigator's escort on a hunting tour: between moves she shoots what's in reach (a short burst,
    never walking after it), and any monster standing in the way of the next move."""

    def __init__(self, h: "Hunter"):
        self.h = h
        self.hunting = True   # shoot monsters in reach between moves (off when going to rest)
        self.kinds: set[str] | None = None  # ... of these kinds only (None: all); any in her way is shot

    def target(self, s: Scene) -> Monster | None:
        """The monster to shoot from where she stands, if any is in reach (pick_target's choice)."""
        in_reach = [m for m in s.monsters if reach.can_shoot(m.x - s.where.x, m.y - s.where.y)
                    and (self.kinds is None or m.kind in self.kinds)]
        return pick_target(s.where.x, s.where.y, s.platform, in_reach) if in_reach else None

    def shoot(self) -> int:
        """A short fight at what's in reach (up to 3 kills, never walking after them); its kills."""
        before = self.h.report.exp_kills
        self.h.clear(self.kinds, max_kills=3, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE, approach=False)
        return self.h.report.exp_kills - before

    def clear_way(self, s: Scene, move: Move) -> int | None:
        """Shoot a monster standing on her platform between her and where the move starts (or too
        close to her), whatever its kind. Returns the kills, or None if nothing was in the way."""
        lo, hi = sorted((s.where.x, move.x))
        in_way = [m for m in s.monsters if m.platform == s.platform.id and lo - TOO_CLOSE <= m.x <= hi + TOO_CLOSE]
        if not in_way:
            return None
        m = min(in_way, key=lambda m: abs(m.x - s.where.x))
        self.h.log(f"  {m.kind} in the way at x {m.x:.0f}: shooting it first")
        before = self.h.report.exp_kills
        self.h.clear(max_kills=2, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE, approach=False)
        return self.h.report.exp_kills - before
