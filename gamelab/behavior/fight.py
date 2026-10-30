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
Kills are counted in the hunter's Report; their drops go to her Looter (behavior.looting). A monster she has
hurt (Hunter.wounds) is finished before a fight ends, from where she is or a short walk along her platform
(`owed`, `finish_act`; the hunt loop's rule 4b too).
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT

from .. import character, directions, paths, telemetry
from ..attack import reach
from ..attack.stance import (AWAY, FLINCH_HOLD_S, MEET_PRICE_S, TOO_CLOSE, TRACK_MATCH, TRACK_SEEN, Pace, act_text,
                             air_gap, decide, first_attack_s, held_step, jump_hits, landing, move_limits, open_above,
                             pct, pick_target, place_to_shoot, plan_of, shooting_place, stance)
from ..control import actions as actions_mod
from ..control.actions import STAR_DELAY_MS
from ..localization.mapdata import MapModel, MobBody
from ..localization.memory import FADE_S
from ..localization.monsters import (DEFAULT_SPEED, MONSTER_FLINCH_TOP, MONSTER_TOP, SEARCH_SLACK, TRACK_SLACK, Monster,
                                     MonsterTrack, bodies, dead_one, seen_again, speeds)
from .ledger import dying
from ..localization.scene import Scene
from ..motion import physics
from .looting import DropSpot

if TYPE_CHECKING:
    from .hunter import Hunter

LOST_AFTER = 3.0   # s a target may stay unseen (without an EXP message) before she gives up on it
HIDDEN_FOR = 1.8   # s she keeps throwing at a monster she was throwing at once it can't be seen: a hit
                   # one flinches under its damage numbers and the detector loses it for 1-3 s, and
                   # she used to stop throwing until it showed again
EMPTY_PATIENCE = 0.4  # s a fight waits when no monster is seen at all (not even out of reach)
FADING = 1.2       # s a killed monster may still be seen fading out where it died (not thrown at), for a kind
                   # whose death animation the game data lacks; else as long as that runs (fade_s)
UNPLACED_LOOKS = 10  # looks in a row that can't place her before a fight gives up on her
IN_HER_WAY = TOO_CLOSE + 40  # a monster of any kind this close on her platform is in her way: fought too
THROWS = ("throw", "jump_throw", "jump_back")  # the stances that throw a star (each an Actions method)
JUMPS = ("jump_throw", "jump_back")  # ... and of those, the ones that jump
MOVES = JUMPS + ("back", "approach")  # the stances that move her (each but a throw from the ground)
# Whether the stances weigh her jumps' attacks (attack.reach.coverage, attack.stance's `jumps`): with nothing in
# reach from the ground, a jump throw at a monster her attack from the air surely picks (on a platform above: up to
# 115 above her up close, 201 at the last band), timed for it; in passing, a jump forward on her way that hits one.
# False: as before (throws from the ground, a jump throw or a jump back only up close, their attack at the fixed
# JUMP_ATTACK_MS and JUMP_BACK_ATTACK_MS).
JUMP_SHOTS = True
STEADY_WAIT_S = 0.05  # s before the next look when a jump or a step waits out a touch's knock (knocked_move)
GUARD_SECONDS = 15  # longest spent shooting a monster out of the way while travelling
GUARD_PATIENCE = 0.3  # s such a fight waits with nothing in reach before she travels on
# Whether a monster that could touch her on the chain ahead (the navigator's _meet: FightOnTheWay.clear_for) is shot
# first where only a jump throw from where she stands takes it (FightOnTheWay.jump_reaches), as the trip's last landing's
# shot first is (shot_place, from 2026-10-28). False: from the ground alone, as before. In the harness (Tree, crowd,
# choice, measured costs, zero walls, near misses, walkers; off against on, paired: D:/MapleHelper-scratch/lshot/mine,
# table-meet-*.md) on, grinding (96 seeds) had its trips' landings within 105 of a monster 13.6% fewer (t -3.8), within
# 60 15.8% (t -3.5), its trips' touches 7.4% fewer (t -2.1), EXP the same (+0.3%, se 1.3%), jump backs 34% more (t 2.2);
# the hunt loop (48) landings within 105 19.4% fewer (t -3.4), trip touches 14.7% fewer (t -2.8), trips 8.6% shorter
# (t -3.6), EXP 3.3% more (t 1.7); no death, fall or stop. Off for the user's call: on, 9 of the 28 Tree goldens part at a
# jump shot first on the way (lshot/mine/golden-meet), and loop_tree_out_of_darts no longer ends her retreat on a safe
# place (knocked off 24's drop to 25 on the way; of seeds 1-12 2 not safe on, 0 off: lshot/mine/darts_seeds.out).
MEET_JUMP_SHOTS = False
# The kinds her trips keep clear of (FightOnTheWay.threat) are those the danger model counts (Hunter.danger,
# attack.danger.DANGER_MIN: a touch taking 5% of her HP or more, by meowdb's formula at her level and HP, config.toml
# [character]): on Tree Dungeon, Forest Up North II at level 24 a Horny Mushroom (16%) and a Green Mushroom (7.2%), not
# a Slime (1.2%); on West Rocky Mountain II at 23 an Axe Stump (10.8%), not a Dark Stump (4.3%); on the first map at 19
# nothing (an Orange Mushroom 2.0%), so her chains there are planned blind as before (though whether the next follows
# on, and the run on past a chain's end, are judged by her box now: motion.bodies). Live on Tree Dungeon on 2026-10-24,
# 74 of 186 touches came on trips (33% of the HP lost), 28 of 48 travel hits a Green Mushroom's, 14 a Horny's.
# A monster she has hurt (Hunter.wounds) is finished before she moves on, from where she is or from a walk this
# long along her platform at most (a second's walk at 149 units/s): in the live hunt loops of 2026-10-21 she set
# off 44 times with a monster's HP bar down and it in reach or a step of 18-99 units away (to a pile, a spot,
# the next ground), 23 of them not killed in the next 20 s; in the simulator's game-like world 13% of the
# monsters she hurt (tour 17%) died to her only 20 s or more after the first hit.
FINISH_WALK = 150
# When a throw's damage can show on its target, for the shot log (_score_shots, _target_gone): from LANDS_FROM_S
# after its attack key (the star leaves STAR_DELAY_MS after it, less a look's slack) to SETTLED_S after it's due
# (`due`: the release, then the flight at reach.STAR_SPEED). So a bar drop is a throw's hit only if no other throw's
# damage can have shown in the same stretch between looks, and a kill only if a single throw can have landed it
# (else undecided). Live on 2026-10-25 the server's damage showed sooner: a kill's gain 0.15-0.46 s after the key, a
# bar's first drop 0.21-0.55 (control.actions.DAMAGE_SHOWS_MS, which the simulator now applies): the shot log, which
# no decision reads, still goes by the release (to redo: README, Her throws at the dead).
LANDS_FROM_S = (STAR_DELAY_MS - 50) / 1000
SETTLED_S = 0.3    # (as probe.SETTLED_S) a target seen this long after its star was due, its bar as before: missed
HURT = 0.02        # a bar this much lower than before is a hit
KEYED_KEEP_S = 2.5  # s her throws are kept with their paths for the kills their gains tell of (FightState.keyed)
# A monster her kill is bound to with no death seen yet (behavior.ledger.Ledger.unsure: drawn alive until her star
# lands, and on the recordings 12 of the 103 bindings on named gains, 21 of 62 on gains of no kind known, were a live
# one, re-found 0.53-1.55 s on) is no target, but one this near her (TOUCH_NEAR across, TOUCH_NEAR_DY up or down) stays
# among the monsters her body keeps clear of (the stance's `touch`: its danger, her jumps' arcs and landings, an
# escape's ways), not among those that take her stars (the drawn dead kept among all of them had her step and jump away
# from them, and on loop_rocky_bridge she fell out of the map so: the review of 2026-10-26). Of those 33 let go, 17
# were this near her when bound (review of 2026-10-27: review-truth binds.py).
TOUCH_NEAR = 150.0
TOUCH_NEAR_DY = 80.0


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
    touches: int = 0           # times a monster touched her (her popups: Hunter._touched) ...
    misses: int = 0            # ... of them, her evasion's MISS (no HP lost)
    stop: str = ""

    def __str__(self):
        return (f"kills: {kills_text(self.kills)} (EXP gains: {self.exp_kills}, targets lost: {self.lost}); "
                f"throws: {self.throws}; loot runs: {self.loot_runs}; potions: {self.potions}; "
                f"touches: {self.touches} ({self.misses} MISS); "
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
        self.touches += other.touches
        self.misses += other.misses


@dataclass
class FightState:
    """What clear() keeps between looks."""
    thrown: list[list] = field(default_factory=list)  # [monster, when last seen]: thrown at, kept while hidden
    dying: list[tuple[Monster, float]] = field(default_factory=list)  # kills, and when: seen fading out a moment
    aimed: list[Monster] = field(default_factory=list)  # in the star's path at the last throw
    touch: list[Monster] = field(default_factory=list)  # her kills' monsters not seen dead yet by her (TOUCH_NEAR)
    # Her throws of the last KEYED_KEEP_S: when each one's attack key went down, and its path (`aimed` then): a gain
    # GAIN_AFTER_KEY after one is its kill (localization.monsters.dead_one)
    keyed: list[tuple[float, list[Monster]]] = field(default_factory=list)
    flinching: list[Monster] = field(default_factory=list)  # hit lately, so maybe still flinching (just_hit)
    kill_spots: list[DropSpot] = field(default_factory=list)  # where kills on her platform died: swept after
    exp_counted: int = 0  # her EXP gains (Report.exp_kills) already told apart as kills
    unseen: int = 0       # looks in a row that couldn't place her (see UNPLACED_LOOKS)
    start_platform: int | None = None  # the platform she's on; leaving it means she was knocked off
    last_act: tuple | None = None  # the last look's stance ...
    last_hidden: bool = False      # ... and whether monsters she was throwing at were hidden then
    calm_since: float | None = None  # since when she's had nothing to do (see clear's `patience`)
    moves: int = 0  # her moves away up close since her last throw (attack.stance.MAX_MOVES: then she throws)

    def forget_faded(self, now: float, body: dict[str, MobBody] | None = None) -> None:
        """Forgets the kills that have faded out by `now`: counted as long ago as their kind's death
        animation lasts, or longer (fade_s, by `body`)."""
        self.dying = [(d, t) for d, t in self.dying if now - t < fade_s(d.kind, body)]

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

    def note_throw(self, now: float, keyed_at: float | None = None) -> None:
        """A throw at the monsters in its path (`aimed`): each is kept (once) among those thrown at, seen `now`; and
        the throw, its attack key down at `keyed_at` (Actions.attack_at), with its path (`keyed`)."""
        self.thrown += [[m, now] for m in self.aimed if not any(e[0] is m for e in self.thrown)]
        if keyed_at is not None:
            self.keyed = [k for k in self.keyed if keyed_at - k[0] <= KEYED_KEEP_S] + [(keyed_at, list(self.aimed))]


def fade_s(kind: str, body: dict[str, MobBody] | None = None) -> float:
    """How long a killed monster of `kind` may still be seen fading out where it died, from when her EXP
    told of its death: its death animation (`body`, the map's game data: MobBody.die_s; a pig 0.66 s, a
    slime 0.84, a snail 1.26), else FADING. The kill is counted at a look after it died, so that's long
    enough; with 1.2 s for every kind, a pig walking up to where one had just died was left alone for half
    a second or more after the dead one was gone."""
    b = (body or bodies()).get(kind)
    return b.die_s if b is not None and b.die_s else FADING


# What a step of clear()'s loop tells it to do next (None: go on).
_AGAIN = "again"  # look again now
_OVER = "over"    # the fight is over: report.stop says why
_DONE = "done"    # finished: sweep up what fell on her platform, and stop on the kill count or the time


def clear(h: "Hunter", kinds=None, ground: set[int] | None = None, max_kills: int = 10, seconds: float = 120,
          patience: float = 1.5, approach: bool = True, finish: bool = False, far: bool = False,
          finish_wounds: bool = True) -> Report:
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
    Nor, `finish` or not, while one she had hurt by then (owed) is in reach or a short walk from it
    (finish_act): she finishes it, closing in on it even without `approach`, before the fight ends,
    and before she waits on nothing (she used to leave monsters half hunted). Only those hurt by the
    look that saw the kill count reached: in the simulator's basin, where respawns never stop, there was
    always another, and the escort's fights in her way to drops ran out the trips. Not without
    `finish_wounds`: a probe's fight off a monster close to her went on to finish the probe's own targets, its
    throws outside the probe's log (behavior.probe._fight: its fights off one close and its clears of her platform).
    Monsters of other kinds on her platform that come close are in her way, and fought too. With
    `far`, monsters (of `kinds`) on any platform in reach are fought too: her darts reach far, and
    their drops are left to the tour (looter.pending)."""
    report = h.report
    deadline = time.monotonic() + seconds
    exp_at_start = report.exp_kills
    st = FightState(exp_counted=exp_at_start)
    placed = None  # the last look that placed her
    owing = False  # at the last look, one she had hurt was in reach or a short walk from it ...
    enough_at = None  # ... hurt by when the kill count was reached (once it was)
    while time.monotonic() < deadline:
        enough = report.exp_kills - exp_at_start >= max_kills
        if enough and not finish and not owing:
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
        targets, closing = fight, approach and not enough
        if enough and enough_at is None:
            enough_at = now
        wounds = owed(h, s, st, enough_at) if finish_wounds else []
        owing = finish_act(h, s, wounds, visible + hidden) is not None
        if owing and ((enough and not finish) or _plain_stance(h, s, fight, visible + hidden, closing)[0] == "wait"):
            targets, closing = finish_targets(s, wounds, visible + hidden), True  # finish hers (finish_act)
        act = _take_stance(h, st, s, targets, visible, hidden, closing, limits=move_room(h, s))
        if knocked_move(h, s, act):
            continue
        if (step := _carry_out(h, st, s, act, fight, now, patience, enough)) == _OVER:
            return report
        if step == _DONE:
            break
    _count_late(h, st, placed)
    if st.kill_spots:
        h.looter.sweep_after(st.kill_spots)
        # Its looks may see another kill: counted, its drops left. Left to the tour, they sent it on loot
        # trip after loot trip (tour_run: 16 instead of 6, 1185 s instead of 708).
        _count_late(h, st, placed, drops=False)
    report.stop = "kill limit" if report.exp_kills - exp_at_start >= max_kills else "time limit"
    return report


# --- the steps of a fight's look (clear) ---------------------------------------------------------

def _count_late(h: "Hunter", st: FightState, placed: Scene | None, drops: bool = True) -> None:
    """At the end of a fight: her EXP gains its own looks didn't tell apart as kills (seen by a sweep's
    looks, before it ended at the kill count or in the sweep after it, or by the look it ended on: knocked
    off, low HP), told apart from its last look that placed her (`placed`). Else they were missing from its
    kills by kind, what a tour's stop sums up. Their drops are kept (see _count_kills) only with `drops`."""
    if h.report.exp_kills > st.exp_counted and placed is not None:
        _count_kills(h, st, placed, [], h.report.exp_kills - st.exp_counted, time.monotonic(), drops)
        st.exp_counted = h.report.exp_kills


def _place_and_check(h: "Hunter", st: FightState, s: Scene) -> str | None:
    """Whether the fight goes on with look `s`: _AGAIN (after a short wait) if it can't place her, and
    _OVER (report.stop says why) once that's gone on for UNPLACED_LOOKS looks, or when she's left the platform she
    started on (knocked off). None: go on. Her potions and the safety stop first (h.health_check: it raises)."""
    if s.where is None or s.platform is None:
        st.unseen += 1
        if st.unseen >= UNPLACED_LOOKS:
            h.report.stop = "lost track of her"
            return _OVER
        time.sleep(0.2)
        return _AGAIN
    st.unseen = 0
    h.health_check(s)
    if st.start_platform is None:
        st.start_platform = s.platform.id
    if s.platform.id != st.start_platform:
        h.log(f"  she's on platform {s.platform.id} now, not {st.start_platform} (knocked off?)")
        h.report.stop = f"left platform {st.start_platform}"
        return _OVER
    return None


def _targets(h: "Hunter", st: FightState, s: Scene, kinds, ground: set[int] | None, far: bool,
             now: float) -> tuple[list[Monster], list[Monster], list[Monster]]:
    """What's there to fight: the monsters wanted and seen (not fading out where a kill died: this fight's, or one
    the ledger placed, ledger.dying), less
    the kills her EXP tells of; plus those she threw at that are hidden now. Returns those, the
    monsters seen (but those fading out), and the hidden ones. Those hit lately (st.flinching: their HP bar new or
    lower, or hidden under their damage numbers) may still flinch when her next attack picks. One the ledger takes for
    dying is taken so by where it is (the nearest of its kind to where a kill was placed): with the dead one unseen, a
    live one beside it is, for up to its death animation. Kept among the monsters about her (the stance's `around`:
    its danger, its jumps' arcs, too close), a dead one drawn there had her step and jump away from the dead, and on
    loop_rocky_bridge she fell out of the map so (review of 2026-10-26): it's left out of them, as before."""
    wanted = _wanted(s, kinds, ground, far)
    body = bodies(h.model)
    st.forget_faded(now, body)
    fading = st.fading(s.monsters)
    fading += [m for m in dying_seen(h, s, now, body) if m not in fading]  # (and by a kill the ledger placed)
    wanted = [m for m in wanted if m not in fading]
    visible = [m for m in s.monsters if m not in fading]
    hit = st.just_hit(visible)
    st.follow(visible, now)
    wanted = _count_kills(h, st, s, wanted, h.report.exp_kills - st.exp_counted, now)
    st.exp_counted = h.report.exp_kills
    st.forget_hidden(now)  # only after _count_kills: this look's kill may still be one of these
    dead = getattr(getattr(h, "ledger", None), "dead", ())  # (a track taken for dead: thrown at no more, hidden or not)
    st.thrown = [e for e in st.thrown if getattr(e[0], "track", None) is None or e[0].track not in dead]
    hidden = st.hidden(now)
    st.flinching = hit + hidden
    st.touch = _unsure_by_her(h, s)
    return wanted + hidden, visible, hidden


def _unsure_by_her(h: "Hunter", s: Scene) -> list[Monster]:
    """The monsters of look `s` her kills are bound to with no death seen yet (Ledger.unsure), within TOUCH_NEAR and
    TOUCH_NEAR_DY of her: the stance's `touch`."""
    unsure = getattr(getattr(h, "ledger", None), "unsure", ())
    if not unsure or s.where is None:
        return []
    x, y = s.where.x, s.where.y
    return [m for m in s.monsters if getattr(m, "track", None) in unsure
            and abs(m.x - x) <= TOUCH_NEAR and abs(m.y - y) <= TOUCH_NEAR_DY]


def dying_seen(h: "Hunter", s: Scene, now: float, body: dict[str, MobBody] | None = None) -> list[Monster]:
    """The monsters of look `s` taken for one dying where a kill the hunter's ledger placed died (ledger.dying, by
    their kinds' death animations: `body`, the map's game data), at `now`: no target."""
    if (ledger := getattr(h, "ledger", None)) is None:
        return []
    die = {k: b.die_s for k, b in (body or bodies(getattr(h, "model", None))).items()}
    return [m for m in s.monsters if dying(ledger.kills, m, now, s.monsters, die, getattr(ledger, "dead", ()))]


def body_sizes(model: MapModel | None) -> tuple[dict[str, float], dict[str, float]]:
    """How tall each kind is at least, and half its width, by kind (localization.monsters.bodies: the map's
    game data): what stance and decide go by."""
    body = bodies(model)
    return {kind: b.top for kind, b in body.items()}, {kind: b.half for kind, b in body.items()}


def walk_on(model: MapModel) -> reach.Where:
    """Where a monster of a look will be t s on (attack.reach.shots' `where`): one seen walking (its pose 'move', facing
    a way) walks on that way at its kind's speed (the game data's: localization.monsters.speeds), turning back at its
    platform's ends, as MonsterTrack.at plans it; any other stays where it was seen. An attack from a jump picks
    0.1-0.5 s after the look: a Horny Mushroom walks 7-35 units meanwhile, a band and more of the star's search."""
    walking = speeds(model)

    def where(m: Monster, t: float) -> float:
        if m.pose != "move" or not m.facing or m.platform is None:
            return m.x
        try:
            p = model.platform(m.platform)
        except (KeyError, IndexError):
            return m.x
        half = reach.body_of(m.kind)[1]
        return MonsterTrack(m.platform, m.x, half=half, vx=m.facing * walking.get(m.kind, DEFAULT_SPEED)).at(t, p)
    return where


def pace_of(h: "Hunter", s: Scene) -> Pace:
    """How soon her next attack picks after look `s`, and how its monsters walk meanwhile (attack.stance.Pace, the
    stance's `pace`: clearance kept ahead of time): `lead`, what's left of her attack lock (Actions.lock_until) and the
    look's age; a monster's walk, its track's speed (the feed's, Scene.tracks: its kind and platform, within
    TRACK_MATCH, seen in TRACK_SEEN frames at least; one seen in fewer, or with no track, not known: where it was seen,
    or with stance.WALK_AT walking at her), else for one the look saw where no feed follows them its pose (walking the
    way it faces, else where it is: as before), and none for one the look didn't see (hidden under its damage numbers,
    or remembered: where it was seen, as before; had walking at her, a fight threw at the ghosts of the dead the camps
    killed); speeds and widths (where one walking at her meets her: stance.PACE_STOP) by the map's game data."""
    now = time.monotonic()
    lock = getattr(getattr(h, "actions", None), "lock_until", 0.0)
    shown = getattr(s, "t", None)  # (a test's stand-in look may have none: no age)
    lead = max(0.0, lock - now) + (0.0 if shown is None else max(0.0, now - shown))
    tracks = [tr for tr in (getattr(s, "tracks", None) or ()) if getattr(tr, "state", None) in ("live", "hidden")]
    fed = bool(getattr(s, "tracks", None))
    seen = {id(m) for m in getattr(s, "monsters", ())}
    model = getattr(h, "model", None)
    walking = speeds(model)

    aim = getattr(h, "aim", None)  # her running attack's target (Aim: throw_act's, _note_aim)
    key = getattr(getattr(h, "actions", None), "attack_at", None)
    hold = None
    counts = getattr(getattr(h, "danger", None), "any", False)  # (where something counts as a danger: not the first map)
    if FLINCH and counts and aim is not None and key is not None and abs(aim.key - key) < 1e-6 and shown is not None \
            and shown - key < FLINCH_HOLD_S:
        shows = key + actions_mod.DAMAGE_SHOWS_MS / 1000  # (her hit's damage shows from then: the bar too)

        def hold(m: Monster) -> tuple[float, float] | None:
            if m.kind != aim.kind or m.platform != aim.platform:
                return None
            # where the hold has it now: walking as its track had it from the decision's look until her hit shows
            if abs(aim.x + aim.walk * max(0.0, min(shown, shows) - aim.seen) - m.x) > HOLD_MATCH:
                return None
            if not HOLD_HIT_SHOWN:
                return shows - shown, key + FLINCH_HOLD_S - shown
            if shown >= shows:  # a look that can show her hit: held only if its bar is lower than at the decision
                if m.bar is None or (aim.bar is not None and m.bar >= aim.bar - 1e-6):
                    return None
                return shows - shown, key + FLINCH_HOLD_S - shown
            return shows - shown, shows + (key + FLINCH_HOLD_S - shows) / 2 - shown  # (till then, half the window)

    def walk(m: Monster) -> float | None:
        best = None
        for tr in tracks:
            if tr.kind == m.kind and tr.platform == m.platform and abs(tr.x - m.x) <= TRACK_MATCH:
                if best is None or abs(tr.x - m.x) < abs(best.x - m.x):
                    best = tr
        if best is not None:
            return best.vx if best.seen >= TRACK_SEEN else None
        if id(m) not in seen:
            return 0.0
        if fed:
            return None
        if m.pose == "move" and m.facing:
            return m.facing * walking.get(m.kind, DEFAULT_SPEED)
        return 0.0
    until = getattr(getattr(h, "senses", None), "invulnerable_until", None)
    window = max(0.0, until() - now) if until is not None else 0.0  # (her invulnerability left: stance's R8)
    return Pace(lead, walk, walking, body_sizes(model)[1] if model is not None else None, hold, window)


# The navigator's price of a landing (FightOnTheWay.landings) by what the looks don't have now: a track not seen within
# STALE_S (a death's fade, localization.memory.FADE_S: after it a remembered or hidden one is as likely a ghost) and the
# spawn points of the platform none of its tracks accounts for (a monster there the looks missed, or a respawn: in the
# simulator's crowd a look misses one on another platform 30% of the time, sim.world.OFF_SEEN), each priced UNSEEN_PRICE
# of one seen. By eye on the live recordings (the reviews of 2026-10-27, D:/MapleHelper-scratch/land/review-numbers/eye),
# 2 of the 9 landings within 105 of a monster by hidden tracks alone had one there (the rest cap drops, another player's
# kill dying, nothing), and 2 of 5 with none near a respawn fading in 80-130 off: a prior, 0.25. In the harness (Tree,
# crowd, the `walkers` world, 48 seeds of 5 minutes, against master's stack: D:/MapleHelper-scratch/land/fix, tree c2)
# grinding (the default) was within 60 of a monster 12% less of the time with it (t -3.7), within 105 at its landings
# 12% less (t -3.3), its EXP a minute 2.6% less (se 2.1%); with 0 (24 seeds, tree c2u) within 60 6% less (t -0.9) and
# 7.7% less EXP (se 3.0%). 0: the tracks seen within STALE_S alone (an older one left out).
STALE_S = FADE_S
UNSEEN_PRICE = 0.25
MID_MEETS = True  # a landing she runs on from priced by its touch alone; False: as the route's last's


def _air() -> bool:
    """Whether her attack is made from a jump too (her profile's attack: character.AttackSpec.air; the thief's Lucky
    Seven yes, the magician's Magic Claw no)."""
    attacks = character.active().attacks
    spec = attacks.get("skill") or next(iter(attacks.values()), None)
    return True if spec is None else bool(spec.air)


def jump_cover(model: MapModel, s: Scene, targets: list[Monster], around: list[Monster]):
    """What her attacks reach from where she stands in look `s` at `targets`, from the ground and from her jumps
    (attack.reach.coverage, by the map's bodies; `around`: every monster about her, each where it walks to by then:
    walk_on), as a callable working it out when first asked (attack.stance's `jumps`): None with JUMP_SHOTS off, or
    for an attack she doesn't make from a jump (_air: the magician's Magic Claw)."""
    if not JUMP_SHOTS or not _air() or s.where is None or s.platform is None:
        return None
    done = []

    def cover() -> reach.Coverage:
        if not done:
            done.append(reach.coverage(model, s.platform, s.where.x, s.where.y, targets, sizes=bodies(model),
                                       others=around, where=walk_on(model)))
        return done[0]
    return cover


def _take_stance(h: "Hunter", st: FightState, s: Scene, fight: list[Monster], visible: list[Monster],
                 hidden: list[Monster], approach: bool, toward: int = 0, jumps=None,
                 goal: float | None = None, edges: tuple[float, float] | None = None, up=None,
                 limits: tuple[float, float] | None = None) -> tuple:
    """What to do about the monsters to `fight` (attack.stance, by each kind's size, with what her jumps' attacks
    reach: jump_cover, or `jumps` if given), walking up to them only if `approach`; `toward`: she's travelling that
    way, to map x `goal` on her platform if given; `edges`, `up`, `limits`: stance's, and her moves away since her last
    throw (st.moves: MAX_MOVES). For a throw, everything in the star's path is kept (st.aimed): a kill is matched against
    it. A jump throw that a stance chose for where its attack reaches is
    logged with its numbers and sent as a telemetry 'jump' record when it's carried out (throw_act: log_jump). Her
    casts going only where her attack choice sends one (castable_of: the magician's), as _cast_or_wait has it: a
    throw her cast would send none with is one the other way, a step off her spot held, or a wait, logged with why
    (act_text's `refused`: "no cast: no effect from here (green-mushroom +120)", not "nothing in reach")."""
    here = s.platform
    body = bodies(h.model)  # each kind's body (game data)
    tops, halves = body_sizes(h.model)
    around = visible + hidden
    castable = castable_of(h)
    act = stance(s.where.x, s.where.y, here, fight, approach, h.actions.facing,
                 can_jump=lambda a, b: open_above(h.model, here, a, b), tops=tops, halves=halves,
                 around=around, danger=h.danger,
                 jumps=jumps if jumps is not None else jump_cover(h.model, s, fight, around), toward=toward,
                 goal=goal, edges=edges, up=up, pace=pace_of(h, s), touch=st.touch, limits=limits,
                 moves=st.moves, castable=castable, **wounds_of(h, s))
    act, refused = _cast_or_wait(h, act, s, castable)
    # Logged when it changes, and when monsters she was throwing at have just gone hidden.
    if act[:2] != (st.last_act or ())[:2] or (hidden and act[0] != "wait" and not st.last_hidden):
        h.log(f"  {act_text(act, fight, s.where.x, hidden, refused)} (HP {s.hp:.0%})")
    st.last_act, st.last_hidden = act, bool(hidden)
    throwing = act[0] in THROWS
    if throwing:
        plan = plan_of(act)
        at = s if plan is None or plan.shot is None else _moved(s, plan.shot.x)
        st.aimed = _in_path(at, act[1], around, attack_gap(h.actions, act, _turn_key(h, s, act)), body,
                            st.flinching)
    if telemetry.recording():
        telemetry.emit("stance", act=act[0], text=act_text(act, fight, s.where.x, hidden, refused),
                       away=moved_away(act),
                       aimed=[{"kind": m.kind, "x": round(m.x, 1), "y": round(m.y, 1)} for m in st.aimed]
                       if throwing else [])
    return act


def attack_gap(actions, act: tuple, turn: int = physics.AIR_TURN_ATTACK_MS) -> float | None:
    """How long after its jump key the attack key of a throwing stance `act` goes as her actions will press it
    (attack.stance.air_gap; None from the ground), `turn` the attack key of a jump that turns her in the air
    (_turn_key): a jump throw in place that turns her has it `turn` after the jump key at the soonest
    (control.actions.Actions.jump_throw, AIR_TURN), whatever the stance's own timing, so she's higher when it picks
    (_turn_of: a plan's turning shot keyed as planned); a jump back without a plan, `turn` (Actions.jump_back, as
    _throw keys it). Its path (`FightState.aimed`) is worked out there: until the review of 2026-10-26 a jump back's
    at physics.JUMP_BACK_ATTACK_MS (85), where Actions.jump_back keyed it at 120, and that path credits the kill of a
    gain soon after the key (FightState.keyed)."""
    gap = air_gap(act)
    plan = plan_of(act)
    if act[0] == "jump_back" and plan is None:
        return max(turn, physics.JUMP_BACK_ATTACK_MS)
    if (act[0] == "jump_throw" and gap is not None and actions_mod.AIR_TURN
            and (plan is None or (not plan.way and plan.air_turn)) and not actions.faces(act[1])):
        return max(gap, _turn_of(plan, turn))
    return gap


def _turn_of(plan, turn: int) -> int:
    """The soonest attack key of a jump throw turning her in the air with JumpPlan `plan` (None: no plan), `turn` that
    of its kind (_turn_key): a plan's shot planned turning (reach.Shot.turn) goes at its own key, which the plan's
    shots worked out for its target's kind (reach.shots: reach.turn_key_ms of its height, a pig's 105 ms), not later."""
    if plan is not None and plan.shot is not None and plan.shot.turn:
        return min(turn, plan.key_ms)
    return turn


def _moved(s: Scene, x: float) -> Scene:
    """Look `s` with her at map x instead (where a jump has her when its attack picks), for _in_path."""
    from dataclasses import replace
    return replace(s, where=replace(s.where, x=x))


def log_jump(h: "Hunter", s: Scene, act: tuple) -> None:
    """A jump throw or jump back a stance chose for what its attack reaches (its JumpPlan has the shot): her log says
    what it's for and the shot's numbers, and a telemetry 'jump' record carries them (`kind`: why, attack.stance's
    JumpPlan.why; the jump's kind, the side, the attack key ms after the jump key and the window of keys that pick the
    same, how high she is then, the target, how far off and above, where she lands), so a live run shows how often
    each fires (sim.harness counts them: attack_kinds' `chosen`) and what came of it (its kills: the ledger's)."""
    plan = plan_of(act)
    if plan is None or plan.shot is None:
        return
    sh, x, y = plan.shot, s.where.x, s.where.y
    t = sh.target
    if hasattr(h, "jump_counts"):
        h.jump_counts[f"{plan.why}: {sh.kind}"] += 1
    h.log(f"    jump ({plan.why}): {sh.text()}, {y - t.y:+.0f} above her; lands at x {sh.land_x:.0f}")
    if telemetry.recording():
        telemetry.emit("jump", kind=plan.why, jump=sh.kind, side=sh.side, key_ms=sh.key_ms, window=list(sh.window),
                       rise=round(sh.rise, 1), target=t.kind, dx=round(t.x - x, 1), up=round(y - t.y, 1),
                       platform=t.platform, x=round(x, 1), land_x=round(sh.land_x, 1), way=plan.way)


def throw_act(h: "Hunter", s: Scene, act: tuple) -> bool:
    """Carry out a throwing stance `act` (THROWS) decided from look `s` with her actions (control.actions.Actions): a
    throw from the ground, a jump throw (in place or forward, its attack timed as its JumpPlan says:
    attack.stance.plan_of), a jump back. One whose plan has its shot is logged (log_jump), and its target goes to the
    attack choice as the throw's (behavior.attacks.Attacks.aimed: its own pick goes by a throw from the ground, which
    has no monster on a platform above in its way). False: dropped at its last look before the key (Actions.still: a
    kill come in since `s`), nothing sent: the caller looks again."""
    plan = plan_of(act)
    attacks = getattr(h, "attacks", None)
    if plan is not None and plan.shot is not None:
        log_jump(h, s, act)
        if attacks is not None:
            attacks.aimed = plan.shot.target
    try:
        thrown = _throw(h.actions, act, plan, _turn_key(h, s, act)) is not False
    finally:
        if attacks is not None:
            attacks.aimed = None  # (its key took it, or no choice was asked: no Lucky Seven key)
    if thrown and FLINCH and getattr(getattr(h, "danger", None), "any", False):
        _note_aim(h, s, act, plan)
    return thrown


# Her running attack's target stands a while in her decisions' pace (pace_of: attack.stance.Pace.hold, FLINCH_HOLD_S): the
# nearest monster of her platform the way she threw (a jump's own target, where its plan has one), as the look saw it and
# walking as its track had it (h.aim); where something counts as a danger (Hunter.danger.any: not on the first map, whose
# decisions are as they were). Held only for a hit: a look from when her hit's damage shows (DAMAGE_SHOWS_MS) holds it
# only if its bar is lower than at the decision, and until such a look half the window. Live (D:/MapleHelper-scratch/land/
# diag-move-away/flinch.out: her throws at walkers 120-300 off that didn't kill) the 28 hurt at +0.3 s came on 17 units by
# 0.6 s and 19 by 0.9, the 38 not hurt 36 and 43, the 24 not read 52 and 69: held whatever, two thirds of the holds had
# it come on 10-45 units less than it did (the reviews of 2026-10-27). Not for a jump back's throw, which rarely takes
# anything. Off, as before.
FLINCH = True
HOLD_HIT_SHOWN = True  # False: every throw's target held the whole window, a jump back's too (as first built)
HOLD_MATCH = TRACK_MATCH + 20  # units: a look's monster is the held one within this of where the hold has it (a guess:
                               # the tracks' match and her hit's push, the simulator's 18 units)


@dataclass(frozen=True)
class Aim:
    """Her running attack's target (_note_aim, pace_of's hold): her attack key's time, its kind, platform, map x and walk
    (units/s) as the decision's look had them, that look's time (`seen`), and its bar then (None: unhurt)."""
    key: float
    kind: str
    platform: int
    x: float
    walk: float
    seen: float
    bar: float | None


def _note_aim(h: "Hunter", s: Scene, act: tuple, plan) -> None:
    """Throw `act` keyed from look `s`: its target kept for pace_of's hold (h.aim, an Aim), or none (a jump back's)."""
    if s.where is None or s.platform is None:
        return
    key = getattr(h.actions, "attack_at", None)
    if act[0] == "jump_back" and HOLD_HIT_SHOWN:
        t = None
    elif plan is not None and plan.shot is not None:
        t = plan.shot.target
    else:
        x, here = s.where.x, s.platform.id
        way = [m for m in s.monsters if m.platform == here and (m.x > x) == (act[1] == RIGHT)]
        t = min(way, key=lambda m: abs(m.x - x)) if way else None
    if t is None or key is None:
        h.aim = None
        return
    v = pace_of(h, s).walk
    h.aim = Aim(key, t.kind, t.platform, t.x, (v(t) if v is not None else 0.0) or 0.0,
                getattr(s, "t", key), t.bar)


def _throw(actions, act: tuple, plan, turn: int = physics.AIR_TURN_ATTACK_MS) -> bool | None:
    """throw_act's keys: the Actions method for `act`, with its JumpPlan's timing (`plan`), if any; what it returns
    (False: dropped). Without one, a jump that turns her in the air has its attack `turn` ms after its jump key
    (attack.reach.turn_key_ms: AIR_TURN_LEAD_MS after the arrow, sooner for a monster too short to pick then)."""
    if act[0] == "throw":
        return actions.throw(act[1])
    if act[0] == "jump_throw":
        if plan is None:
            if turn == physics.AIR_TURN_ATTACK_MS:  # (as callers that don't know of it: the tests' fakes)
                return actions.jump_throw(act[1])
            return actions.jump_throw(act[1], turn_key=turn)
        turn = _turn_of(plan, turn)
        if turn == physics.AIR_TURN_ATTACK_MS:  # (as callers that don't know of it: the tests' fakes)
            return actions.jump_throw(act[1], plan.key_ms, plan.way, plan.air_turn)
        return actions.jump_throw(act[1], plan.key_ms, plan.way, plan.air_turn, turn_key=turn)
    if act[0] == "jump_back":
        if plan is None:
            return actions.jump_back(act[1]) if turn == physics.AIR_TURN_ATTACK_MS else actions.jump_back(act[1], turn)
        return actions.jump_back(act[1], plan.key_ms)
    raise ValueError(f"not a throw: {act!r}")


def _turn_key(h: "Hunter", s: Scene, act: tuple) -> int:
    """The attack key (ms after the jump key) of stance `act`'s jump if it turns her in the air: reach.turn_key_ms for
    the nearest monster on her platform the way it throws (a stance's jump up close is at it), by its kind's height
    (the map's game data); AIR_TURN_ATTACK_MS with none."""
    if act[0] not in JUMPS or s.where is None or s.platform is None:
        return physics.AIR_TURN_ATTACK_MS
    x, here = s.where.x, s.platform.id
    ahead = [m for m in s.monsters if m.platform == here and (m.x > x) == (act[1] == RIGHT)]
    if not ahead:
        return physics.AIR_TURN_ATTACK_MS
    nearest = min(ahead, key=lambda m: abs(m.x - x))
    b = bodies(h.model).get(nearest.kind)
    return reach.turn_key_ms(b.top if b is not None else MONSTER_TOP["pig"])


def knocked_move(h: "Hunter", s: Scene, act: tuple) -> bool:
    """Whether stance `act` is a jump (a jump throw or a jump back) or a step (stepping away, closing in) decided
    from a look a touch's knock may still be moving her in (Hunter.knocked): then it waits a moment for the next
    look. Keyed from such a look, the jump key came while she was still in the air, and did nothing (in loop_run
    21 jump keys were lost so, 9 now: those left come from a touch after the look, in the plan's lead-in), or once
    she had landed and still slid on, from where she no longer was: in the harness (tour, seed 5, both tracks) a
    jump throw keyed from a look taken mid-knock went in after she had landed on 34 sliding at 104 units/s, took
    her off its end up onto 31 (that nothing was overhead was checked where she was at the look, 55 units back)
    and landed her 0.4 units from 31's end, which the next throw's turn walked her off, and out of the map. A step
    so walks her on from where the knock leaves her: on 22, a snail's touch knocked her 40 units toward its end
    and a step back of 106 from where she'd been took her off it (fight_snail_short_platform, seed 4)."""
    if act[0] in MOVES and h.knocked(s):
        if h.knock_waited != h.knock_at:  # logged once a knock
            h.knock_waited = h.knock_at
            h.log(f"    hit: the {'jump' if act[0] in JUMPS else 'step'} waits for the knock to end")
        time.sleep(STEADY_WAIT_S)
        return True
    return False


def _carry_out(h: "Hunter", st: FightState, s: Scene, act: tuple, fight: list[Monster], now: float,
               patience: float, enough: bool) -> str | None:
    """Do what the stance `act` says: throw (at those in the star's path, seen `now`), or walk; with
    nothing to do, see _idle. Returns what _idle does (None after a throw or a walk)."""
    if act[0] in THROWS:
        st.calm_since = None
        if throw_act(h, s, act):  # (dropped at its last look: look again)
            st.note_throw(now, h.actions.attack_at)
            h.report.throws += 1
            st.moves = 0
        return None
    if act[0] in ("back", "approach"):
        st.calm_since = None
        st.moves += moved_away(act)
        h.actions.walk(act[1], act[2])
        return None
    return _idle(h, st, s, fight, patience, enough)


def moved_away(act: tuple) -> bool:
    """Whether stance `act` moves her away from a close monster (a step of attack.stance's close quarters or escape,
    ('back', side, units, AWAY)): what FightState.moves counts toward MAX_MOVES; a step out from under one above
    doesn't."""
    return act[0] == "back" and (act[3:4] == (AWAY,) or not AWAY_ONLY)


AWAY_ONLY = True  # False: every step counted toward MAX_MOVES, out from under a monster above too (as first built)


# Up close her moves away keep to move_limits (attack.stance: WALL_ROOM from a wall end, EDGE_MARGIN from a short drop,
# DROP_ROOM from a long one), where they kept 50 from every end (and DROP_ROOM from a long drop): from 2026-10-27, where
# something counts as a danger, in the hunt loop's fights and the tours' and the escort's (clear). False: as before.
MOVE_LIMITS = True


def move_room(h: "Hunter", s: Scene) -> tuple[float, float] | None:
    """The stance's `limits` for her in look `s`: move_limits where something counts as a danger (MOVE_LIMITS), else
    None (as it was)."""
    d, model = getattr(h, "danger", None), getattr(h, "model", None)
    if not MOVE_LIMITS or d is None or not d.any or s.platform is None or model is None:
        return None
    return move_limits(model, s.platform)


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
                   or (far and reach.can_hit(s.platform, x, y, m, others=s.monsters)))]
    wanted += [m for m in s.monsters if m not in wanted and m.platform == here and abs(m.x - x) < IN_HER_WAY]
    return wanted


def owed(h: "Hunter", s: Scene, st: FightState | None = None, hurt_by: float | None = None) -> list[Monster]:
    """The monsters she has hurt and not killed (Hunter.wounds) around her at look `s` (first hurt by `hurt_by`,
    if given): as seen in it, or unseen in it (hit, under their damage numbers) where last seen, for HIDDEN_FOR at
    most. Not one seen fading out where a kill of the fight `st` died."""
    now = time.monotonic()
    wounds = h.wounds.seen_within(now, HIDDEN_FOR, hurt_by)
    if not wounds:
        return []
    visible = s.monsters if st is None else [m for m in s.monsters if m not in st.fading(s.monsters)]
    out = []
    for w in wounds:
        again = seen_again(w, visible)
        if again is None and any(seen_again(w, [m]) is not None for m in s.monsters):
            continue  # seen fading out where it died
        if (again or w) not in out:
            out.append(again or w)
    return out


def finish_targets(s: Scene, wounds: list[Monster], around: list[Monster]) -> list[Monster]:
    """What she shoots at to finish `wounds` from look `s`: them, and any monster of `around` (every monster
    around her) in her way on her platform (IN_HER_WAY)."""
    here, x = s.platform.id, s.where.x
    return wounds + [m for m in around if m.platform == here and abs(m.x - x) < IN_HER_WAY
                     and not any(m is w for w in wounds)]


def finish_act(h: "Hunter", s: Scene, wounds: list[Monster], around: list[Monster]) -> tuple | None:
    """What finishing `wounds` (owed) takes from where she stands in look `s`: attack.stance at finish_targets,
    closing in if need be (`around`: every monster around her), if that's a throw or a walk of FINISH_WALK at
    most along her platform; None if neither (out of reach from near her: she's free to move on)."""
    if not wounds:
        return None
    act = _plain_stance(h, s, finish_targets(s, wounds, around), around, True)
    if act[0] in THROWS or (act[0] in ("back", "approach") and act[2] <= FINISH_WALK):
        return act
    return None


def _plain_stance(h: "Hunter", s: Scene, targets: list[Monster], around: list[Monster], approach: bool) -> tuple:
    """attack.stance at `targets` from look `s`, by each kind's size, as _take_stance has it (unlogged)."""
    here = s.platform
    tops, halves = body_sizes(h.model)
    castable = castable_of(h)
    return cast_or_wait(h, stance(s.where.x, s.where.y, here, targets, approach, h.actions.facing,
                                  can_jump=lambda a, b: open_above(h.model, here, a, b), tops=tops, halves=halves,
                                  around=around, danger=h.danger, pace=pace_of(h, s), castable=castable,
                                  **wounds_of(h, s)),
                        s, castable)


def cast_or_wait(h: "Hunter", act: tuple, s: Scene | None = None, castable=None) -> tuple:
    """Stance `act` as hunter `h` carries it out from look `s` (_cast_or_wait's; `castable`: the stance's)."""
    return _cast_or_wait(h, act, s, castable)[0]


HELD = ("no effect", "no effect from here")  # (behavior.attacks.Attacks.refused's holds: rule 3)


def _cast_or_wait(h: "Hunter", act: tuple, s: Scene | None = None, castable=None) -> tuple[tuple, tuple | None]:
    """Stance `act` as hunter `h` carries it out from look `s`, and why her cast goes at none (None: it goes, or no
    cast was in it), as (act, (why, the body her search would pick)). Where her attack choice casts only where her
    search picks a body she casts at (castable_of: the magician's), a throw that way with none (Attacks.target: one
    held, likely dead, or no body her search picks first) is, as the review of 2026-10-29 had it: a throw the other way
    where her search picks one she casts at there (facing right, a held one first in her search at +120, a castable one
    at +220 and one at -150: she waited; a throw from the ground only, no jump's plan changed); else a step off her
    spot where it's held (stance.held_step: HELD_STEP away from that side, within where she keeps to, into no body nor
    more danger: her move is the new evidence that lifts the hold, behavior.attacks.Spot); else ('wait',), the fight's
    patience running and the hunt loop's next rules asked, not a key that sends nothing look after look. A stance that
    throws none where `castable` refused one it would have thrown at (Castable.refused) is as it is, with that why, and
    a wait so a step off her spot held too. Each such no cast is counted, logged and recorded (Attacks.no_cast, by
    'stance': choice.THROW_S apart at most, her casts' pace): before, only `key`'s refusals were, and live the stance
    mostly decides first. Anything else as it is (the thief's always: `picks` off)."""
    attacks = getattr(h, "attacks", None)
    if attacks is None or not getattr(attacks, "picks", False):
        return act, None
    if act[0] in ("throw", "jump_throw"):
        side = act[1]
        if attacks.target(side) is not None:
            return act, None
        why, first = attacks.refusal(side) or ("nothing her search picks", None)
        other = directions.opposite(side)
        if act[0] == "throw" and attacks.target(other) is not None:
            return ("throw", other), None
        attacks.no_cast(side, why, first)
        return _held_step(h, s, side, why) or ("wait",), (why, first)
    refused = getattr(castable, "refused", None)
    if not refused:
        return act, None
    m, why = refused[0]
    side = RIGHT if s is None or s.where is None or m.x > s.where.x else LEFT
    attacks.no_cast(side, why, m)
    if act[0] == "wait":
        act = _held_step(h, s, side, why) or act
    return act, (why, m)


def _held_step(h: "Hunter", s: Scene | None, side: str, why: str) -> tuple | None:
    """A step off her spot held that way (stance.held_step), where `why` her cast goes at none is a hold (HELD), from
    look `s`; else None."""
    if why not in HELD or s is None or s.where is None or s.platform is None:
        return None
    model = getattr(h, "model", None)
    return held_step(s.where.x, s.platform, side, s.monsters, body_sizes(model)[1], getattr(h, "danger", None),
                     move_room(h, s) if model is not None else None)


class Castable:
    """The stance's `castable` for her attacks (behavior.attacks.Attacks, `picks`): whether her cast goes at a monster
    (Attacks.refused says no why), each it refused kept with its why (`refused`, in the order the stance asked), for
    _cast_or_wait: a stance that throws none where it would have thrown at one of them is a cast not sent, said."""

    def __init__(self, attacks):
        self.attacks = attacks
        self.refused: list[tuple[Monster, str]] = []

    def __call__(self, m: Monster) -> bool:
        why = self.attacks.refused(m)
        if why is not None and not any(o is m for o, _ in self.refused):
            self.refused.append((m, why))
        return why is None


def wounds_of(h: "Hunter", s: Scene) -> dict:
    """The stance's `hurt` and `hidden_hurt` for hunter `h` at look `s` (behavior.attacks.Attacks.hurt, hidden_hurt:
    her wounds finished first, attack.stance.WOUNDS_FIRST), where her casts go only at a body she sees (`picks`, the
    magician's); none for the thief (as before)."""
    attacks = getattr(h, "attacks", None)
    if attacks is None or not getattr(attacks, "picks", False):
        return {}
    return {"hurt": attacks.hurt, "hidden_hurt": attacks.hidden_hurt(s)}


def castable_of(h: "Hunter") -> Castable | None:
    """The stance's `castable` for hunter `h`, a new one each stance (Castable): whether her attack goes at a monster
    at all, where her attack choice has her skill cast only at a body seen, not dead, likely dead or held
    (behavior.attacks.Attacks.picks, the magician's: Attacks.refused); None for the thief (every one, as before)."""
    attacks = getattr(h, "attacks", None)
    if attacks is None or not getattr(attacks, "picks", False):
        return None
    return Castable(attacks)


def _count_kills(h: "Hunter", st: FightState, s: Scene, wanted: list[Monster], gains: int,
                 now: float, drops: bool = True) -> list[Monster]:
    """Her EXP told of `gains` kills since the last look: those are among what she was throwing at (the last
    throw's path, and what she's thrown at lately: a kill can come in after she's turned to the next one), each of
    the kind its gain's size named when it named one (Hunter.kinds_told); with the feed, the monster the ledger has it
    on (its track: behavior.ledger, the tracker's), the same answer as every other part's. Each kill's drops go to the
    sweep after the fight (on her platform) or to the tour (unless not `drops`). A kill of a kind she wasn't throwing at
    still counts by its kind, with no place and no drops. Returns `wanted` without the dead."""
    here = s.platform.id
    told = h.gains_told(gains) if hasattr(h, "gains_told") else [(None, k) for k in h.kinds_told(gains)]
    for gained, kind in told:
        dead = _tracked_kill(h, st, s, gained)
        if dead is None:
            dead = dead_one(st.aimed + [e[0] for e in st.thrown if not any(e[0] is a for a in st.aimed)],
                            s.monsters, s.where.x, kind, st.keyed, gained)
        if dead is None and kind is not None:
            h.report.count_kill(kind)
            h.log(f"  {kind} down (by her EXP), not one she was throwing at: its drops are left")
            continue
        if dead is None:
            h.log("  a kill (EXP up) she can't place: its drops are left")
            break
        if dead in st.aimed:
            st.aimed.remove(dead)
        st.thrown = [e for e in st.thrown if e[0] is not dead and seen_again(dead, [e[0]]) is None]
        st.dying.append((dead, now))
        wanted = [m for m in wanted if seen_again(dead, [m]) is not m]
        h.report.count_kill(dead.kind)
        if drops:
            spot = DropSpot(dead.x, dead.y, dead.platform, time.monotonic())
            (st.kill_spots if dead.platform == here else h.looter.pending).append(spot)
        h.log(f"  {dead.kind} down at x {dead.x:.0f} (platform {dead.platform})")
        telemetry.emit("event", kind="kill", text=f"{dead.kind} down on platform {dead.platform}",
                       x=round(dead.x, 1), y=round(dead.y, 1))
    return wanted


def _tracked_kill(h: "Hunter", st: FightState, s: Scene, gained: float | None) -> Monster | None:
    """The monster the hunter's ledger has her kill whose gain showed at `gained` on, by its track (ledger.Kill.track:
    the tracker's): as this fight last saw it (in its path or thrown at), or as look `s` has it, or where its track is;
    None without one."""
    ledger = getattr(h, "ledger", None)
    if gained is None or ledger is None:
        return None
    k = next((k for k in reversed(ledger.kills[-8:]) if k.gained is not None and abs(k.gained - gained) < 0.02), None)
    if k is None or not k.track:
        return None
    seen = [*st.aimed, *(e[0] for e in st.thrown), *s.monsters]
    if (m := next((m for m in seen if getattr(m, "track", None) == k.track), None)) is not None:
        return m
    tr = next((tr for tr in getattr(s, "tracks", None) or () if tr.id == k.track), None)
    if tr is None:
        return None
    return Monster(tr.mob, tr.x, tr.y, tr.platform, tr.bar, tr.facing, "", tr.id)


def _in_path(s: Scene, side: str, monsters: list[Monster], gap_ms: float | None = None,
             body: dict[str, MobBody] | None = None, flinching: list[Monster] = ()) -> list[Monster]:
    """Everything in a star's path that way, whatever its kind: a snail in front of a far pig takes the
    star (and its EXP isn't the pig's kill). From the ground, what the claw or the star's search can pick
    (reach.ground_pick, by each kind's body walking: reach.body_of, half its width from `body`, the map's
    game data): on her platform and off it, a monster off her level only as close as the search picks it,
    and every monster in a band, as the client's order within one can't be seen. From a jump (the attack
    keyed `gap_ms` after the jump key), what the attack's area takes in from as high as she is when it
    picks its target (reach.in_area, by how tall each kind is at least, `body`'s top): monsters on the
    platform above too, as far as a star reaches, but none that stays under its floor (a snail beside
    her: an EXP gain then came from one of those above). Those of
    `flinching` (hit lately, FightState.just_hit) as tall as they flinch (`body`'s flinch_top): a pig 60
    below her is picked from 111 units then, from 151 walking. None whose star terrain stops (reach.wall_to:
    from 4, those on 2; one the claw's stance reaches is swiped whatever the terrain), nor any behind one of
    those in the attack's order (the client checks the line to its pick alone, and stopped, no other monster
    takes the star): in a later band of the search from the ground, farther off from a jump."""
    x, y = s.where.x, s.where.y
    body = body or bodies()
    ahead = [m for m in monsters if (m.x > x) == (side == RIGHT)]
    rise, keyed = (0.0 if gap_ms is None else reach.air_rise(gap_ms)), []
    for m in ahead:
        b = body.get(m.kind)  # its body (a pig's if unknown)
        walking, half = reach.body_of(m.kind, body)
        least = b.top if b is not None else MONSTER_TOP["pig"]
        top = walking if gap_ms is None else least  # what each attack's pick goes by (see above)
        if any(m is f for f in flinching):
            top = max(top, b.flinch_top if b is not None else MONSTER_FLINCH_TOP["pig"])
        if gap_ms is None:
            order = reach.ground_pick(abs(m.x - x), m.y - y, top, half)
            order = None if order is None else order[0]  # 0: the claw's; 1 + k: the search's band k
        else:
            order = (abs(m.x - x) if reach.in_area(abs(m.x - x), m.y - y, top, rise, half, m.platform == s.platform.id)
                     else None)
        if order is not None:
            keyed.append((order, m, reach.wall_to(s.platform, x, y, m, rise) is None))
    stopped = [k for k, _, clear in keyed if not clear and (gap_ms is not None or k > 0)]
    first = min(stopped, default=math.inf)  # a star at what comes after it ends at what stops it
    return [m for k, m, clear in keyed if (clear or (gap_ms is None and k == 0)) and k <= first]


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
    flying: list[dict] = field(default_factory=list)  # its throws, for the shot log: those not yet logged have no
                               # "hit" (see _score_shots); the logged ones stay, as their damage may still explain
                               # a later bar drop
    had_bar: bool = False      # its HP bar has been seen (it was hit) ...
    dying_since: float | None = None  # ... and since when it's been seen without it: dying, fading out

    def seen(self, monsters: list[Monster]) -> list[Monster]:
        """The `monsters` that can be the target: of its kind, near where it was last seen (searched for
        further once it's out of sight)."""
        slack = TRACK_SLACK if self.gone_since is None else SEARCH_SLACK
        return [m for m in monsters if m.kind == self.kind
                and abs(m.x - self.last[0]) <= slack and abs(m.y - self.last[1]) <= 40]

    def nearest(self, found: list[Monster]) -> Monster | None:
        """Which of those `found` (seen) is the target: the nearest to where it was last seen."""
        return min(found, key=lambda m: abs(m.x - self.last[0])) if found else None


def kill(h: "Hunter", target: Monster, seconds: float = 30) -> str:
    """Shoot `target` until it dies; then loot (or remember the drops if they're off her
    platform). Returns how it ended: 'killed', 'lost', 'timeout' or 'unreachable' (her HP staying low raises the safety
    stop: h.health_check)."""
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
        h.health_check(s)
        found = st.seen(s.monsters)
        _note_pick(st.flying, s, st.nearest(found))  # seen or not: hit, it's often hidden under its damage numbers
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
    if st.first_throw is not None and h.take_exp(st.first_throw, st.kind):
        # Dead since before it was seen without its bar (else before it was gone: a kill is seen fading out
        # for 0.7-1.3 s): _credit_kill.
        _credit_kill(st.flying, st.dying_since or st.gone_since)
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
    h.exp_events = [e for e in h.exp_events if s.t - e[0] < 1.5]
    t = st.nearest(found)
    st.last, st.platform = (t.x, t.y), t.platform if t.platform is not None else st.platform
    bar = t.bar
    if bar is not None:
        st.had_bar, st.dying_since = True, None
    elif st.had_bar and st.dying_since is None:
        st.dying_since = time.monotonic()  # its bar gone: it died, and fades out where it died
    _score_shots(st.flying, bar, st.dying_since is not None, s.t)
    if (ledger := getattr(h, "ledger", None)) is not None and dying(ledger.kills, t, time.monotonic(), s.monsters,
                                                                   ledger.die, getattr(ledger, "dead", ())):
        time.sleep(0.1)  # a kill of hers placed on it: dying, it fades out (then her EXP ends the kill: _target_gone)
        return None
    threats = [o for o in s.monsters if o.platform == s.platform.id and o is not t]
    act = decide(s.where.x, s.where.y, s.platform, t, threats, body_sizes(h.model)[1], around=s.monsters)
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
            thrown = h.actions.throw(act[1]) is not False
        finally:
            h.actions.throw_listeners.remove(on_throw)
        if not thrown:  # dropped at its last look before the key (Actions.still): look again
            return None
        if shot is not None:
            if keyed:  # its star is due that long after the key
                shot["due"] += keyed[0] - shot["at"]
                shot["key"] = keyed[0]
            st.flying.append(shot)
        h.report.throws += 1
        st.throws += 1
        st.first_throw = st.first_throw or time.monotonic()
    else:
        h.actions.walk(act[1], act[2])
    return None


def _shot(s: Scene, t: Monster, dx: float, dy: float, model: MapModel | None = None) -> dict:
    """A throw at `t` (`dx`, `dy` from her), to be scored: what the shot log keeps, its HP bar before
    the throw (`bar`, as of `before_t`; every look at it from now to its scoring, `looks`), when its attack
    key goes down (`key`) and when the star should hit (`due`: the attack's release, then the star's flight,
    after the key: now, until _shoot_at knows better).
    The log also keeps where she and it were (map x), her foothold group, and whether a wall of it stood in
    the way (`clear`: reach.wall_between, whether or not her decisions keep to it), so what stops a star can
    be checked from it; and every monster in front of her then (`front`), and where they were at the next
    look that placed her, about when the attack picked its target (`pick`, _note_pick), so which one it could
    pick can be fitted from the log alone."""
    here = s.platform
    on = here if t.platform == here.id else None  # the target's platform, if known
    if on is None and model is not None and t.platform is not None:
        on = model.platform(t.platform)
    clear = reach.wall_between(here, s.where.x, s.where.y, t.x, target=on, ty=t.y, body=reach.body_of(t.kind)) is None
    now = time.monotonic()
    return {"kind": t.kind, "dx": int(round(dx)), "dy": int(round(dy)), "her": here.id,
            "target": t.platform, "bar": t.bar, "seen": bool(s.tag_seen),
            "x": int(round(s.where.x)), "tx": int(round(t.x)),
            "her_group": list(here.group) if here.group is not None else None, "clear": clear,
            "y": int(round(s.where.y)), "front": _front(s, directions.toward(dx), t),
            "at": now, "key": now, "before_t": now, "looks": [(now, t.bar)],
            "due": now + COMBAT.attack_release_ms / 1000 + abs(dx) / reach.STAR_SPEED}


def _front(s: Scene, side: str, target: Monster | None = None) -> list[dict]:
    """The monsters in look `s` on her `side`, as the shot log keeps them: kind, map x and y, platform, HP
    bar, and whether it's the one thrown at."""
    x = s.where.x
    return [{"kind": m.kind, "x": int(round(m.x)), "y": int(round(m.y)), "platform": m.platform, "bar": m.bar,
             **({"target": True} if m is target else {})}
            for m in s.monsters if (m.x > x) == (side == RIGHT)]


def _note_pick(flying: list[dict], s: Scene, target: Monster | None) -> None:
    """The first look that places her after each throw, about when its attack picked its target (the look
    starts once the throw's key plan is done), whether the target is seen in it or not: where she and the
    monsters in front were then, and when (`t`: s after the attack key)."""
    if s.where is None:
        return
    for sh in flying:
        if "pick" not in sh:
            sh["pick"] = {"x": int(round(s.where.x)), "y": int(round(s.where.y)), "t": round(s.t - sh["key"], 3),
                          "front": _front(s, directions.toward(sh["dx"]) if sh["dx"] else RIGHT, target)}


def _score_shots(flying: list[dict], bar: float | None, dying: bool = False, now: float | None = None) -> None:
    """Score the throws at the target not logged yet by a look at it (`now`, its HP bar `bar`; None: none
    shown), and log those decided. Until a throw's damage can show (LANDS_FROM_S after its key) a look's bar
    is what its star finds (`bar`, seen at `before_t`). After that, the bar new or lower than that is its hit,
    if no other throw's damage can have shown since then too (else undecided: one drop, two throws that may
    have made it); the bar as before SETTLED_S after it was due, a miss. While it's `dying` (seen without the
    bar it had) every throw waits for the kill (_target_gone)."""
    now = time.monotonic() if now is None else now
    for sh in flying:
        if "hit" not in sh:
            sh.setdefault("looks", []).append((now, bar))
    if dying:
        return
    for sh in flying:
        if "hit" in sh:
            continue
        if now < sh["key"] + LANDS_FROM_S:
            if bar is not None:
                sh["bar"] = bar
            sh["before_t"] = now
        elif bar is not None and (sh["bar"] is None or bar < sh["bar"] - HURT):
            others = [o for o in flying if o is not sh and _may_show(o, sh["before_t"], now)]
            _decide(sh, None if others else True, now)
        elif now >= sh["due"] + SETTLED_S:
            _decide(sh, False, now)


def _may_show(sh: dict, after: float, by: float) -> bool:
    """Whether throw `sh`'s damage can have shown on its target after `after` and by `by`: from LANDS_FROM_S
    after its key to SETTLED_S after it was due, or to the look it was decided in; never if it missed."""
    if sh.get("hit") is False:
        return False
    end = sh["due"] + SETTLED_S if "hit" not in sh else min(sh["due"] + SETTLED_S, sh["decided"])
    return sh["key"] + LANDS_FROM_S <= by and end > after


def _credit_kill(flying: list[dict], died: float) -> None:
    """Log the throws at a target found killed that aren't logged yet, `died` being the look that first saw it
    dying (its bar gone) or gone. One star killed it, the last to land by then. A throw whose damage can't
    have shown by then (LANDS_FROM_S after its key) missed it. Of those that can, the last keyed is the kill
    if it alone can, or if it was surely in by then (due by `died`, 0.2 s to spare); else it may have got
    there after the kill (if the game shows damage when the star gets there), and which one killed it can't
    be told. The others may have hit it too (a hit one hides under its damage numbers): undecided."""
    left = [sh for sh in flying if "hit" not in sh]
    could = [sh for sh in left if sh["key"] + LANDS_FROM_S <= died]
    last = max(could, key=lambda sh: sh["key"], default=None)
    sure = last is not None and (len(could) == 1 or last["due"] <= died + 0.2)
    for sh in left:
        _decide(sh, (True if sh is last and sure else None) if any(sh is o for o in could) else False, died)


def _decide(sh: dict, hit: bool | None, when: float) -> None:
    """Throw `sh` scored at `when`: logged, and kept as decided."""
    sh["hit"], sh["decided"] = hit, when
    _log_shot(sh, hit)


SCORING = ("bar", "due", "at", "key", "before_t", "looks", "hit", "decided")  # a throw's scoring, kept as times


def _log_shot(shot: dict, hit: bool | None) -> None:
    """Keep a record of every throw (lab/shots2.jsonl): the reach of a star is measured from these. `hit`
    None: undecided (a throw in flight before the one that killed it, or one of two whose stars may have
    made the same bar drop). Its times are s after its attack key (`key_t`, time.monotonic: the throws at a
    target in flight together can be put side by side): when its star was due (`due_s`), when it was scored
    (`decided_s`: for a kill, the look that saw the target dying or gone), and its target's HP bar at every
    look at it from the throw on (`bars`), so a scorer that knows when the game shows a star's damage can tell
    what an undecided one did."""
    key = shot.get("key", shot["due"])
    try:
        with paths.of().shot_log.open("a", encoding="utf-8") as f:  # (hers)
            f.write(json.dumps({**{k: v for k, v in shot.items() if k not in SCORING},
                                "key_t": round(key, 3), "due_s": round(shot["due"] - key, 3),
                                "decided_s": round(shot.get("decided", key) - key, 3),
                                "bars": [[round(t - key, 3), b] for t, b in shot.get("looks", ())],
                                "hit": None if hit is None else bool(hit),
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except OSError:
        pass


class FightOnTheWay:
    """The navigator's escort on a hunting tour: between moves she shoots what's in reach (a short burst,
    never walking after it), any monster standing in the way of the next move, and one that could touch her
    anywhere along the chain of moves the navigator plans (a kind whose touch counts: `threat`), if a star reaches it
    from where she stands (`clear_for`)."""

    def __init__(self, h: "Hunter"):
        self.h = h
        self.kinds: set[str] | None = None  # shoot monsters in reach of these kinds only (None: all); any
                                            # in her way is shot

    def target(self, s: Scene) -> Monster | None:
        """The monster to shoot from where she stands, if any is in reach (pick_target's choice), else one she
        has hurt that a short walk brings back into reach (owed, finish_act): the burst (shoot) finishes it."""
        in_reach = [m for m in s.monsters if reach.can_hit(s.platform, s.where.x, s.where.y, m, others=s.monsters)
                    and (self.kinds is None or m.kind in self.kinds)]
        if in_reach:
            return pick_target(s.where.x, s.where.y, s.platform, in_reach, s.monsters)
        wounds = owed(self.h, s)
        if finish_act(self.h, s, wounds, s.monsters) is not None:
            return min(wounds, key=lambda m: abs(m.x - s.where.x))
        return None

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

    def clear_way(self, s: Scene, x: float, finish: bool = True) -> int | None:
        """Shoot the monster in her way to map x (in_way). Returns the kills, or None if nothing was in
        the way. `finish`: the fight finishes what she has hurt (clear's finish_wounds). Not in transit
        (the navigator's TRANSIT: the hunt loop's trips, whose rule 4b finishes what she hurt within its FINISH_S, and
        until 2026-10-26 the trips to rest at low HP): finishing there, a rest trip's fights in the basin, where
        respawns never stop, ran to their GUARD_SECONDS twice as often as before it, and some trips ran out."""
        if (m := self.in_way(s, x)) is None:
            return None
        self.h.log(f"  {m.kind} in the way at x {m.x:.0f}: shooting it first")
        before = self.h.report.exp_kills
        self.h.clear(max_kills=2, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE, approach=False, finish_wounds=finish)
        return self.h.report.exp_kills - before

    def threat(self, kind: str) -> bool:
        """Whether her trips keep clear of monsters of `kind`: the danger model counts its touch (Hunter.danger: a share
        of her HP of DANGER_MIN or more at her level, from the map's game data). A kind the data doesn't have doesn't
        count."""
        return self.h.danger.share(kind) > 0

    def clear_for(self, s: Scene, m: Monster, finish: bool = True, jump: bool = False) -> int | None:
        """Shoot monster `m`, one that could touch her on the chain of moves ahead (the navigator's `in_the_way`), if a
        star reaches it from where she stands (reach.can_hit; with `jump`, or one from a jump the stance would make
        there: `reaches`): a short fight, as clear_way's, at its kind on its platform (and any in reach elsewhere, or
        close on hers), never walking after it. A kill clears the way, and a hit's knockback may (the navigator plans
        again from the next look). Returns the kills, or None if no star reaches it from here. `finish`: as clear_way's.
        `jump`: the navigator's shot first by a trip's last landing (shot_place's); _meet's from the ground alone as
        before, unless MEET_JUMP_SHOTS."""
        if not self.reaches(s, m, jump=jump or MEET_JUMP_SHOTS):
            return None
        self.h.log(f"  {m.kind} on platform {m.platform} at x {m.x:.0f} could touch her on the way: shooting it first")
        before = self.h.report.exp_kills
        self.h.clear({m.kind}, ground={m.platform}, max_kills=1, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE,
                     approach=False, far=True, finish_wounds=finish)
        return self.h.report.exp_kills - before

    def throws(self) -> int:
        """Her throws so far (the hunter's report's: every throw and jump throw a fight carries out): the navigator's
        shot first by a landing tells one that threw from one that didn't (from 2026-10-28: live at 00:44:09 clear_for's
        fight logged "nothing in reach" three times and threw nothing, the Green missing from its own looks)."""
        return self.h.report.throws

    def reaches(self, s: Scene, m: Monster, jump: bool = False) -> bool:
        """Whether her attack from where she stands in look `s` takes monster `m`: the fights' own rule, a throw from
        the ground (reach.can_hit, what her profile's attack geometry names for the thief, character.Geometry
        'star_search': a star's band under and over her, reach.SHOT_DOWN and the search's windows; its line, v83's
        walls and floors; a monster the attack takes first shielding it), clear_for's; with `jump`, or a jump throw the
        stance would make at it from there (jump_reaches: what the hunt loop's Hunt._jump_hits counts). The seam S1's
        aim facade replaces for another character's attack, with stance.place_to_shoot's search along her platform."""
        if s.platform is None or s.where is None:
            return False
        if reach.can_hit(s.platform, s.where.x, s.where.y, m, others=s.monsters):
            return True
        return jump and self.jump_reaches(s, m)

    def dying(self, s: Scene, m: Monster) -> bool:
        """Whether monster `m` of look `s` is taken for dead or dying by the hunter's ledger (ledger.dying, by its kind's
        death animation: as dying_seen), which no fight takes for a target. In nav_tree_landing_shot_first (from
        2026-10-28) _meet's clear_for killed the Horny Mushroom by her landing, and its body, standing there through its
        death, still priced the landing: two shots first at it, each "nothing in reach", 1.3 s, as first built both
        'shot first' (the landing's price itself still counts it: a question for the chains, not changed)."""
        ledger = getattr(self.h, "ledger", None)
        if ledger is None:
            return False
        die = {k: b.die_s for k, b in bodies(getattr(self.h, "model", None)).items()}
        seen = s.monsters if any(o is m for o in s.monsters) else [*s.monsters, m]
        return dying(ledger.kills, m, time.monotonic(), seen, die, getattr(ledger, "dead", ()))

    def jump_reaches(self, s: Scene, m: Monster) -> bool:
        """Whether a jump throw from where she stands in look `s` surely takes monster `m` on the stance's terms
        (stance.jump_hits over jump_cover: the jump landing her back where she keeps to however it goes, her body clear
        of the look's monsters, the danger kept; as Hunt._jump_hits), where her attack is made from a jump at all
        (_air: the thief's Lucky Seven, not the magician's Magic Claw) and JUMP_SHOTS. From 2026-10-28 (the review's):
        from 25 at 140 a Green on 24 at 260 is out of reach of every throw from the ground on 25, and the fight
        jump-throws at it in place (4 of 9 simulated tries killed it with one), where shot_place had said 'from
        nowhere'."""
        if not (JUMP_SHOTS and _air()) or s.platform is None or s.where is None:
            return False
        model = getattr(self.h, "model", None)
        if model is None:
            return False
        around = s.monsters if any(o is m for o in s.monsters) else [*s.monsters, m]
        cover = jump_cover(model, s, [m], around)
        if cover is None or not any(sh.jump is not None for sh in cover().of(m)):
            return False
        actions = getattr(self.h, "actions", None)
        return jump_hits(s.where.x, s.where.y, s.platform, m, cover(), getattr(actions, "facing", None), around,
                         getattr(self.h, "danger", None), body_sizes(model)[1])

    def shot_place(self, s: Scene, m: Monster, most: float) -> tuple[float | None, str]:
        """Where on her platform her attack reaches monster `m` from (the navigator's shot first by a trip's last
        landing): (her x, '') if from where she stands (`reaches` with jumps: clear_for's with `jump`), else (the nearest
        place within `most` units along her platform, '') where a throw from the ground does, SHOOT_SLACK either way of
        it too (stance.place_to_shoot: shoot_from's), else (None, why not). Nothing is sent. Live (2026-10-28 00:38:36)
        the drop 21 -> 23 by a Horny Mushroom logged "shooting it first" and threw nothing: 23 is 117-122 under 21, past
        a star's band (reach.SHOT_DOWN), so from nowhere on 21 does one reach it. Nor at one dying (`dying`: the fights
        take none for a target, so a shot first at it threw nothing: nav_tree_landing_shot_first's two, 1.3 s)."""
        if self.dying(s, m):
            return None, "it's dying (a kill of hers on it): no target"
        if self.reaches(s, m, jump=True):
            return s.where.x, ""
        if s.platform is None or s.where is None or m.platform is None:
            return None, "no look of her on a platform"
        jumps = ", nor one from a jump where she stands" if JUMP_SHOTS and _air() else ""
        if most <= 0:
            return None, f"no throw from the ground takes it from where she stands{jumps} (no walk to shoot it from)"
        place, why = place_to_shoot(s.platform, s.where.x, s.where.y, m, most, s.monsters)
        return place, (why + jumps if place is None else why)

    # (shoot_from) a monster found again after the walk: its kind on its platform within this of where it was seen
    SHOOT_REFIND = TRACK_MATCH + DEFAULT_SPEED * 1.0  # the tracks' match and a second of its kind's walk (a guess)

    def shoot_from(self, s: Scene, m: Monster, most: float, finish: bool = True) -> int | None:
        """Monster `m` shot from along her platform (the navigator's, landings with clearance: one by where her trip's
        last move lands her): a walk of `most` units at most to the nearest place a star reaches it from
        (attack.stance.shooting_place: within where she stands to shoot with monsters about, no monster of her platform
        in the walk's way), then, if it's in reach from where the walk leaves her (the next look's, the same kind on its
        platform within SHOOT_REFIND of where it was: TRACK_MATCH and a second of its kind's walk), clear_for's fight at
        it (from a jump too: clear_for's `jump`). The kills (0 if the walk's look doesn't find it or it's out of reach
        from there, each logged), or None if there's no such place."""
        here = s.platform
        if here is None or s.where is None or m.platform is None:
            return None
        x = s.where.x
        g = shooting_place(here, x, s.where.y, m, most, s.monsters)
        if g is None:
            return None
        side = RIGHT if g > x else LEFT
        self.h.log(f"  {m.kind} on platform {m.platform} at x {m.x:.0f} by where she'd land: walking {side} "
                   f"{abs(g - x):.0f} to shoot it first")
        self.h.actions.walk(side, abs(g - x))
        look = self.h.look()
        if look.where is None or look.platform is None or look.platform.id != here.id:
            self.h.log("    the walk's look doesn't have her on her platform: no throw")
            return 0
        again = [o for o in look.monsters if o.kind == m.kind and o.platform == m.platform
                 and abs(o.x - m.x) <= self.SHOOT_REFIND]
        if not again:
            self.h.log(f"    the walk's look doesn't find the {m.kind} again: no throw")
            return 0
        kills = self.clear_for(look, min(again, key=lambda o: abs(o.x - m.x)), finish, jump=True)
        if kills is None:
            self.h.log(f"    the {m.kind} out of her attack's reach from x {look.where.x:.0f}: no throw")
        return 0 if kills is None else kills

    def landings(self, tracks: list):
        """The navigator's price of a leg's landing against `tracks` (its chain's, MonsterTrack in the plan's time), as
        (s, the track making it so, or None): attack.stance.landing, each track on the platform she lands on where it
        is then and when her first attack there starts (attack.stance.first_attack_s, walking as followed), by the map's
        bodies, her danger's keep-off, no jump in place under a platform overhead (open_above) nor for an attack she
        can't make from a jump (her profile's: character.AttackSpec.air). The tracks seen within STALE_S, and with
        nothing priced among those, the older ones and the platform's spawn points none of them accounts for
        (Danger.free_points: a monster there the looks missed, or a respawn), each priced UNSEEN_PRICE of a seen one's,
        with no track making it so (nothing to shoot first or wait out); a landing she runs on from (`last` False: a leg
        before a chain's last) by a seen one walking into her there alone (MEET_PRICE_S, with MID_MEETS). None where no
        kind counts as a danger (the first map: as before)."""
        h = self.h
        danger, model = getattr(h, "danger", None), getattr(h, "model", None)
        if danger is None or not danger.any or model is None:
            return None
        tops, halves = body_sizes(model)
        air = _air()

        def judge(leg, last: bool = True):
            q, x = leg.end.platform, leg.end.x
            mine = [mo for mo in tracks if mo.platform == q.id]
            t0 = leg.end.t
            t1 = first_attack_s(t0)
            jump = (lambda a, b: open_above(model, q, a, b))
            fresh = [mo for mo in mine if mo.seen_at >= -STALE_S]
            if fresh:
                near = [(mo.kind or "pig", mo.at(t0, q), mo.at(t1, q)) for mo in fresh]
                got = landing(q, x, near, halves, tops, danger, jump, air)
                if not last and MID_MEETS:  # (a landing she runs on from: its touch alone)
                    return (MEET_PRICE_S, None) if got.meets else (0.0, None)
                if got.price_s > 0:
                    return got.price_s, min(fresh, key=lambda mo: abs(mo.at(t1, q) - x))
            if not last and MID_MEETS:
                return 0.0, None
            unseen = [(mo.kind or "pig", mo.at(t0, q), mo.at(t1, q)) for mo in mine if mo.seen_at < -STALE_S] \
                + ([(kind, sx, sx) for sx, kind in danger.free_points(q.id, mine)] if UNSEEN_PRICE > 0 else [])
            if unseen and UNSEEN_PRICE > 0:
                got = landing(q, x, unseen, halves, tops, danger, jump, air)
                if got.price_s > 0:
                    return UNSEEN_PRICE * got.price_s, None
            return 0.0, None
        return judge
