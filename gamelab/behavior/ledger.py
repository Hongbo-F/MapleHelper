"""The hunt's bookkeeping, kept up from every look, whoever takes it (Senses.listeners), and from every
throw (Actions.throw_listeners): the kill ledger and the drop store.

- **The ledger** (`Ledger`): every kill the hunter's EXP gains tell of is a kill (a gain's size names its kind, or
  says it was two kills, or none: localization.gains). It's placed (kind, platform, where) when a star of hers
  explains it: a monster of the kind named (when one was) that can have taken the attack of a throw made in the
  last STAR_WINDOW s (or, with none, that long before the look before, as the gain came after it; any of them:
  which of the claw's stances an attack takes is drawn, and the client's order within a band of the star's search
  can't be seen): with the feed, the monster the perception service's tracker bound the gain to (localization.tracking:
  the one her attack took from where she was at the key, by where the tracks were at the gain; `Kill.track`), followed
  as its track walks on until her star lands, and put on another if the tracker lets it go; else a gain that shows
  GAIN_AFTER_KEY (0.1-0.6 s) after a throw's key is that throw's, the first of its kind in its path, seen or not (the
  server's kill shows before her star leaves); else preferring one that's gone from view (and of those, one with
  another player by it last: PLAYER_KILLS), the newest throw's and the likeliest first; where that monster was seen
  last (a throw's path is followed from look to look). A kill no throw explains (a kill seen only from a look without
  monsters) still counts, credited to where she is, by the kind its gain named, but has no place and so no drops.
  Which monsters are dead (`dead`, `dying`): the feed's tracks a kill or a share is bound to, or found dead, by their
  ids, until long after they're gone (every throw's target, the fights, her marks, wounds and memory go by them);
  without the feed's ids, a monster of a placed kill's kind by where it was placed, while its death animation shows.
- **The drop store** (`DropStore`): every pile she knows of, with when its kill happened (not when it was
  first seen), when it was last seen, whether a kill of hers explains it, what the last look that saw it saw
  in it and what that's worth (the map's looting.Worth: the game data's prices, mesos by what the map's monsters
  drop; one no look has seen yet, its kind's expected drops: looting.expected_worth), and its status. A drop seen is
  in one pile only, the nearest within LOOT_REACH (until 2026-10-30 every pile within LOOT_REACH of it took it: 15-24% of
  the piles' worth listed live was the same drops counted by 2-3 piles). Every look's drops count, and every strip of
  drops perception searched since the look before (the state's rotation, worldstate.WorldState.drops: Scene.strips), at
  its own time. A pile is gone as the Looter forgets drops: never before SETTLE after the kill, and only
  after two empty looks EMPTY_LOOKS apart, because monsters are drawn over drops, each a strip sure of it (worldstate.
  DropStrip.sure: whole, on a frame placed firmly in height); a pile of hers seen gone that a sighting shows drops at
  again is hers again (`_sighting`). Gone piles are dropped from the
  store GONE_KEEP_S later; what the hunt loop counts of them is kept in counters (`gone_tried`). How likely a pile is
  to lie there still is `survival` (by its age, hers or another's: the live Kaplan-Meier), not a fixed life.
- **Her wounds** (`Wounds`): the monsters she has hurt and not killed. A monster in the path of a throw of
  hers (as the ledger has it) whose HP bar shows up or changes in a look within STAR_WINDOW s of the throw is
  hers, and followed look to look until a kill of hers is placed on it, it's out of sight for WOUND_LOST_S,
  or WOUND_S have gone by since it was first hurt (chased too long). The fights and the hunt loop finish
  these before they move on (behavior.fight.owed): she used to leave monsters half hunted.

Step 4 of gamelab/docs/hunt-modes.md: these are observers, kept up whoever looks; the simulator checks the
ledger against its world (tests/test_ledger.py), and the hunt loop reads the drop store and the wounds.
"""

import itertools
import math
from dataclasses import dataclass, field

from maplehelper.engine import RIGHT

from .. import telemetry
from ..attack import reach
from ..localization.monsters import GAIN_AFTER_KEY, TRACK_SLACK, Monster, bodies, is_one, same, seen_again
from ..localization.players import near_rival
from ..localization.scene import Loot, Scene
from . import looting
from .looting import COINS, EMPTY_LOOKS, FIRST, LOOT_REACH, SETTLE, Worth, standing

STAR_WINDOW = 2.5  # s after a throw's attack key that an EXP gain can still be its kill, as the looks see it: the
                   # gain shows 0.15-0.46 s after the key live (the server's: localization.monsters.GAIN_AFTER_KEY),
                   # but a look may come seconds later (a sweep's walks and pick-ups take none)
# A kill of hers placed this recently (Kill.t: the look that saw its gain), and a monster of its kind on its platform
# this near where it was placed (map units), are taken for the monster dying there (`dying`): not a throw's target
# (behavior.attacks), not fought (behavior.fight._targets). Live on 2026-10-25 (the hunt of 23:43) the dying monster
# stayed drawn 0.6-1.3 s after its gain, 14-25 units from where the kill was placed, and was taken from the path again,
# whole: 18 casts of Lucky Seven and 4 regular throws went at one in 11 minutes, a tenth of her MP potions
# (D:/MapleHelper-scratch/analysis-1025/mp-attack-check/atdead.py). How recently: its kind's death animation (its die1
# frames in the game data, MobBody.die_s: a Horny or Green Mushroom 0.84 s; DIE_S for a kind it lacks), from the look
# that placed it, which comes after the gain, as the animation begins. 1.5 s at first, and every one within DYING_DX:
# then, once the dead one was gone, a live one by where it died went unshot for most of a second, and grinding, finding
# nothing to shoot, went on (simulator, Tree Dungeon's crowd, grinding, 8 seeds of 10 minutes: 320 EXP a minute; the
# nearest only, 317; the animation and 0.15 s more, 351; the animation alone, 390, as at 0.8 s; master's stack in the
# branch's world 361).
DIE_S = 0.66
DYING_DX = 40.0
# The feed's tracks taken for dead (`Ledger.dead`): those a look's tracks have bound to a kill or a share of hers, or
# found dead (tracking.Track.killed), kept this long after a look last had them so (a track's id is never used again;
# a dead one goes from the tracker DEAD_KEEP_S after its death, and her marks of it LOST_S after it was last seen)
DEAD_KEEP_S = 15.0
TRACK_GAIN_S = 0.02  # s: a track bound to a gain (tracking.Track.gain) is a kill's whose gain showed within this
FOLLOW_S = 3.0      # s after a kill that its place follows its track (it walks on until her star lands, and the
                    # tracker may put the gain on another until tracking.REBIND_S + CONFIRM_S)
DROP_LANDS = 2.0   # s after a kill that its drops lie on the ground (recorded at 2 fps)
LANDED_S = 0.2     # s past that from which a search of the place with none in it says the kill dropped nothing (`searched_empty`)
# s a kill's drops are its killer's alone (the game's ownership: nobody else can pick them up meanwhile; looting.
# cant_pick has seen mesos she stood on twice stay): another's pile is worth a trip once past it. The simulator's
# (sim.world.Realism.rival_owns), not measured; until 2026-10-30 the hunt loop waited OTHERS_AFTER (30 s, a guess) for
# drops no kill of hers explains, and live 30% of others' drops were gone within those 30 s.
OWN_S = 15.0
# How long drops lie (`survival`): the share of them still there at an age (s since her kill, or since another's was
# first seen). Hers by the live evidence (from 2026-10-31, the loot designs: D:/MapleHelper-scratch/replan/loot/diag,
# live_km.txt, and design): hers alone for OWN_S, then taken by someone else at TAKE_RATE a minute, gone at the expiry
# pass past EXPIRE_S, as the simulator's drop-takers world has them (sim.world.others_life, EXPIRE_PASS_S): the same
# guesses on both sides, so within the branch that world agrees with the planner by construction (the review of
# 2026-10-31); the harness's standard world (one coin a kill, gone at 180 s, nobody taking hers) doesn't, and is where
# the belief is checked against a world it isn't fitted to. The perception service's own, her pets' pick-ups
# (71% of her drops that vanished did so 15 units or more from her) and the detector's misses taken out, 95%, 88% and
# 86% of her drops were still lying at 15, 60 and 90 s: at most about 0.1 a minute left to others. Until then hers were
# the service's raw curve (the thief's live run of 2026-10-30, her picks censored: 55% at 60 s, 26% at 120, 9% at 150),
# which counted those as takings, so with SURVIVE_MIN she gave up on her own piles at 150-165 s, and in 15-minute runs
# of the harness's standard grind 11% of her drops' worth expired lying in her store. Others' as before: that raw curve
# (live, 12.1% of the drops seen on 3 looks or more were seen lying past 90 s; the last sightings' p99 138 s).
TAKE_RATE = 0.05   # a minute: others taking her drops once past OWN_S (a guess inside the live bound)
EXPIRE_S = 180.0   # s her drops lie at least (HeavenMS's 180 s) ...
EXPIRE_PASS_S = 60.0  # ... then gone at the server's next expiry pass, a time no look tells: as likely any time in
                      # this long past EXPIRE_S (the one live pass took four drops 198-228 s old, so its passes come 48 s
                      # apart at least; 60 a guess, sim.world.EXPIRE_PASS_S). Until the review of 2026-10-31 none past
                      # EXPIRE_S, so a pile seen lying past it was given up at once.
SURVIVAL = {True: ((0.0, 1.0), (OWN_S, 1.0))
                  + tuple((t, math.exp(-TAKE_RATE / 60.0 * (t - OWN_S)))
                          for t in (30.0, 60.0, 90.0, 120.0, 150.0, EXPIRE_S))
                  + ((EXPIRE_S + EXPIRE_PASS_S, 0.0),),
            False: ((0.0, 1.0), (5.0, 0.92), (10.0, 0.83), (20.0, 0.75), (30.0, 0.70), (60.0, 0.57), (90.0, 0.31),
                    (120.0, 0.22), (150.0, 0.03), (180.0, 0.0))}
# s a pile seen gone stays in the store before it's dropped. Nothing reads a gone pile but the hunt loop's
# count of the ones she went for (kept in DropStore.gone_tried), so this only bounds the store: kept
# forever, every kill added a pile, and every look went through all of them.
GONE_KEEP_S = 300.0
# s: a pile of hers seen gone that a sighting this soon after shows drops at again was missed, not taken (`_sighting`): a
# strip's search misses drops under a monster or another player (74 of 962 whole strips over the simulator's drops, the
# review of 2026-10-30), and the rotation is back on a strip in 1.2-2.4 s. Later, they may be another's kill's.
REVIVE_S = 30.0
# Whether a pile of hers holding drops another of hers held before keeps that one's age (`_sighting`), from 2026-10-31:
# every drop seen goes to its nearest pile within LOOT_REACH, so a newer kill's pile by her older drops takes them, the
# older pile turns gone, and the drops were priced by the newer's age (on the harness's seed 5, a pile from a kill at
# 118.7 s held drops from kills at 22-57 s). False: each pile its own kill's age, as before. Only a pile holding a drop
# where one of the older's lay when it was last seen (within HELD_DX: drops don't move once down, and a look places
# them to a few units), from the review of 2026-10-31: until then any pile of hers within LOOT_REACH that took drops in
# that sighting, so a newer kill's pile by an older one she had picked up herself took its age, and its drops were
# judged expired up to the gap between the two kills too soon. And only while a look still sees a drop where the older's
# lay (Pile.aged, aged_xs), its own kill's age after (the pile's `since` kept: Pile.born), from the independent check of
# 2026-11-01: until then the whole pile took the older's age for good, so a pile whose own drops lay until 330 s in the
# simulator was given up at 241 s, once the older drop had gone.
HELD_AGE = True
HELD_DX = 10.0
# What a pile no look has seen yet is worth (Pile.value), in mesos: a kill's pile is coins most often (63-80% of
# each kind's piles seen where her kills died on 2026-10-21; looting.COINS), and its item is worth less than they are
# but for the Slime Bubble (looting.PRICES). That's the first map's; on another a bronze coin of its own (Pile.unseen,
# from the drop store's Worth: on Tree Dungeon 33.4, on West Rocky Mountain II 25.9).
PILE_WORTH = COINS["bronze"]
# Whether a gain goes on a monster gone from view by another player (localization.players.near_rival) only when no
# other a throw of hers explains has gone: one gone by them, they likelier killed. The harness can't tell: with her
# gains naming their kinds (localization.gains) it changes no run at all (preset game, hunt loop and tour, 20 seeds of
# 20 minutes, with the preset's other player and with one killing every 8 s: 0 of 80 runs), and the ledger put none of
# her kills on the other player's (0 of 3,569 placed, 98.8% right, on or off, with the busy one). (Before the kinds:
# 9 of about 4,200 kills placed where its kill was without it, 8 with it, EXP a minute 129.93 -> 129.76 on 2 seeds.)
# On because the request was for her kills to go by the other players, and all it does is put last, of the monsters
# of the kind named gone from her throws' paths, one by another player; whether a real one's kills are ever in her
# throws' paths is for the game to say (README: Other players, to check in game).
PLAYER_KILLS = True
# Whether a kill whose gain named a kind (localization.gains) that none of her recent throws' paths holds goes on the
# likeliest monster of any kind (True), or unplaced (False: no pile expected; its kind still counted). Over the three
# basin scenarios' seeds 1-10 (30 runs, 1,111 kills): with the kind placed only on one of that kind, 90% of the kills
# were placed and 96% of those right (kind, platform, within 2.5 s of the world's kill), 86% of all; with any kind as
# well, 99.5% placed and 86% of those right, 85% of all; without the kind (before her EXP number was read), 99.9%
# and 70%.
KIND_ANY = False
# Her wounds (Wounds). A bar new, or this much other than at the throw, is a hit (fight.HURT). A wound unseen this long
# is gone: a hit monster hides under its damage numbers for 1-3 s (fight.HIDDEN_FOR is how long she keeps
# throwing at one unseen; seen again within this, it's still hers), and seen without its bar it's still kept
# (in the live looks of 2026-10-21 a bar seen was gone at the next look of the monster 70 times, 21 of them
# with no kill after). Chased this long from the look that first saw it hurt, it's no longer owed: in the
# simulator's game-like world (20 minutes, 30 seeds) she killed half of the monsters she hurt within 5.4 s of
# the first hit and 84% within 20 s.
WOUND_HURT = 0.02
WOUND_LOST_S = 3.0
WOUND_S = 15.0


@dataclass
class Kill:
    """One kill her EXP told of, and what it was if a throw explains it."""
    t: float                  # the look that saw her EXP go up
    kind: str | None          # None: no throw of hers explains it (no place, no drops)
    platform: int | None      # where the monster died ...
    x: float
    y: float
    ground: int | None        # ... and the platform the kill is credited to (hers, when unplaced)
    named: str | None = None  # the kind her EXP gain's size named (None: not known)
    gained: float | None = None  # when its gain showed (its frame's time; None: not known)
    track: int | None = None  # the feed's track its monster is (tracking: the one the gain is bound to); None: no track
                              # known (placed by its throws' paths), 0: the one it was on was let go, none known since
    pile: "Pile | None" = field(default=None, repr=False, compare=False)  # its pile in the drop store, if any

    @property
    def placed(self) -> bool:
        return self.kind is not None


@dataclass
class _Throw:
    t: float                  # when its attack key went down
    side: str
    path: list[Monster]       # what can have taken its attack then, the likeliest first (from the last look
                              # with monsters)


class Ledger:
    """Her kills for the whole hunt: every EXP gain, placed where a throw of hers explains it."""

    def __init__(self, hunter, store: "DropStore | None" = None):
        self.h = hunter
        self.store = store
        self.kills: list[Kill] = []
        self._accounted = hunter.exp_gains  # gains already in the ledger
        self._throws: list[_Throw] = []
        self._last: Scene | None = None     # the last look that searched for monsters
        self._looked: float | None = None   # when the last look of all was
        self.dead: dict[int, float] = {}    # the feed's tracks taken for dead, by id: when a look last had them so
        self.theirs: set[int] = set()       # ... of those, another player's kills (a death not hers, or a share)
        self.unsure: set[int] = set()       # ... and those bound to a gain with no death seen yet (`_dead_of`)
        model = getattr(hunter, "model", None)  # each kind's death animation (`dying`), the map's game data
        self.die = {k: b.die_s for k, b in bodies(model).items()} if hasattr(model, "mob_bodies") else {}
        # Whether the hunt follows the tracker's corrections (`moved`): where her kills are put on monsters by her
        # search (her Shape's kill credit, the magician's: localization.tracking's band order); the thief's as before
        shape = getattr(getattr(hunter, "senses", None), "shape", None)
        self.follows = getattr(shape, "kill", None) is not None
        # Her kills the tracker put on another monster after the hunt had them (`_follow`, with `follows`): (when, the
        # kill as it is now, the track, platform and x it was on), for her memory (localization.memory: the old one
        # alive again, the new one killed) and the grind's credit by platform (behavior.grind)
        self.moved: list[tuple[float, Kill, int | None, int | None, float]] = []

    @property
    def placed(self) -> list[Kill]:
        return [k for k in self.kills if k.placed]

    def on_throw(self, when: float, side: str) -> None:
        """A throw (Actions.throw_listeners): what's in its path, from the last look with monsters (none dying where a
        kill was placed: `dying`, a star passes it)."""
        s = self._last
        path = [] if s is None or s.where is None or s.platform is None else \
            in_path(s, side, [m for m in s.monsters if not dying(self.kills, m, when, s.monsters, self.die, self.dead)])
        self._throws.append(_Throw(when, side, path))

    def on_look(self, s: Scene) -> None:
        """After every look (after the hunter's own count): each new EXP gain becomes a kill. Throws are kept
        STAR_WINDOW s, or that long before the look before if it's older: a gain seen now came after it."""
        since = s.t if self._looked is None else min(s.t, self._looked)
        self._looked = s.t
        self._throws = [th for th in self._throws if since - th.t <= STAR_WINDOW]
        self._dead_of(s)
        new, self._accounted = self.h.exp_gains - self._accounted, self.h.exp_gains
        told = self.h.gains_told(new) if hasattr(self.h, "gains_told") else [(None, k) for k in self.h.kinds_told(new)]
        for gained, kind in told:
            self.kills.append(self._place(s, kind, gained))
        self._follow(s)
        if s.searched:
            self._last = s
            # What can have taken a throw's attack is followed from look to look (as Wounds follows them), so a
            # kill is placed, and its pile expected, where its monster was seen last, not where it was at the
            # throw: a slime shot from 16 walked 90 units on 17 before it died, and the drops seen where it fell,
            # past the pile's LOOT_REACH, were taken for someone else's (the hunt loop waits for those).
            for th in self._throws:
                th.path = [seen_again(m, s.monsters) or m for m in th.path]

    def _place(self, s: Scene, kind: str | None = None, gained: float | None = None) -> Kill:
        """The kill a gain seen in look `s` was (of `kind`, when its size named one; `gained`: when it showed, if
        known): the first of that kind in the path of the latest throw keyed GAIN_AFTER_KEY before the gain, seen or not
        (its monster is drawn dying then); else a monster of that kind that
        can have taken the attack of a throw made in the last STAR_WINDOW s (its path), preferring one not seen any
        more (the newest throw first, the likeliest of its path first); with none, of a throw made up to STAR_WINDOW
        s before the look before, which the gain came after; else unplaced (or, with KIND_ANY, of any kind). A
        fight's last star often lands after its last look, and the walks and pick-ups of the sweep after it take no
        look for up to 7 s: in the simulator, 2 of tour_hunt_basin's 61 kills went unplaced that way, and 9 of 58 once
        its monsters walked as fast as in the game. Going back that far for every gain placed more kills on the
        wrong monster (458 of 612 right in 10 scenarios, against 437 of 628)."""
        if (bound := _bound(s, gained)) is not None:  # the tracker's: the one her attack took (Kill.track)
            lo, hi = GAIN_AFTER_KEY
            for th in [th for th in reversed(self._throws) if lo <= gained - th.t <= hi][:1]:
                self._throws.remove(th)  # one star, one kill
            here = None if s.platform is None else s.platform.id
            kill = Kill(s.t, bound.kind, bound.platform, bound.x, bound.y,
                        bound.platform if bound.platform is not None else here, kind, gained, bound.id)
            if self.store is not None:
                kill.pile = self.store.on_kill(kill)
            return kill
        firsts = self._candidates(s.t, kind)
        if not firsts and kind is not None and KIND_ANY:
            firsts = self._candidates(s.t, None)
        soon = [] if gained is None else [(th, m) for th, m in firsts
                                          if GAIN_AFTER_KEY[0] <= gained - th.t <= GAIN_AFTER_KEY[1]]
        if soon:  # (firsts: the newest throw's first, each path's likeliest first)
            firsts = soon[:1]
        elif s.searched:
            gone = [(th, m) for th, m in firsts if not any(same(o, m) for o in s.monsters)]
            if PLAYER_KILLS and s.players:  # (a stable sort: the newest throw's and the likeliest first as before)
                gone.sort(key=lambda tm: near_rival(s.players, tm[1].x, tm[1].y) is not None)
            firsts = gone or firsts
        here = None if s.platform is None else s.platform.id
        if not firsts:
            x, y = (s.where.x, s.where.y) if s.where is not None else (0.0, 0.0)
            return Kill(s.t, None, None, x, y, here, kind, gained)
        th, m = firsts[0]
        self._throws.remove(th)  # one star, one kill
        kill = Kill(s.t, m.kind, m.platform, m.x, m.y, m.platform if m.platform is not None else here, kind,
                    gained)
        if self.store is not None:
            kill.pile = self.store.on_kill(kill)
        return kill

    def _dead_of(self, s: Scene) -> None:
        """The feed's tracks look `s` has dead or dying (tracking.Track.killed: a kill or a share bound to it, or found
        dead) into `dead`, another player's (`theirs`: dead by another's hand, tracking's cause "other", or a share)
        and those bound to a gain of hers or a share whose death hasn't been seen or judged yet (`unsure`: drawn alive
        until her star lands, and 12 of 103 bindings on the recordings' named gains were a live neighbour:
        fight._targets keeps those by her in her danger); one it has alive is out of it (let go: another was the
        kill); none kept DEAD_KEEP_S after."""
        for tr in getattr(s, "tracks", None) or ():
            if getattr(tr, "killed", False):
                self.dead[tr.id] = s.t
                if getattr(tr, "cause", None) == "other" or getattr(tr, "share", False):
                    self.theirs.add(tr.id)
                else:  # (hers, or hers after all: tracking._taken_back)
                    self.theirs.discard(tr.id)
                if tr.state == "dead" or getattr(tr, "died_seen", None) is not None:
                    self.unsure.discard(tr.id)
                else:
                    self.unsure.add(tr.id)
            else:
                self.dead.pop(tr.id, None)
                self.unsure.discard(tr.id)
        self.dead = {i: t for i, t in self.dead.items() if s.t - t <= DEAD_KEEP_S}
        self.theirs &= self.dead.keys()
        self.unsure &= self.dead.keys()

    def _follow(self, s: Scene) -> None:
        """The kills of the last FOLLOW_S bound to a track (Kill.track): where it is now (it walks on until her star
        lands), its pile with it; one the tracker put on another, on that one (its gain's track), or on none (0)."""
        tracks = getattr(s, "tracks", None) or ()
        if not tracks:
            return
        for k in reversed(self.kills):
            if s.t - k.t > FOLLOW_S:
                break
            if not k.track:
                continue
            tr = _bound(s, k.gained)
            if tr is None:
                if any(o.id == k.track for o in tracks):  # (let go: alive. Gone from the tracker: where it was last)
                    self._moved(s.t, k, None)
                    k.track = 0
                continue
            if tr.id != k.track:
                self._moved(s.t, k, tr)
            k.track, k.kind, k.platform, k.x, k.y = tr.id, tr.kind, tr.platform, tr.x, tr.y
            if tr.platform is not None:
                k.ground = tr.platform
            if (p := k.pile) is not None and p.status == "expected" and tr.platform is not None:
                p.x, p.y, p.platform = tr.x, tr.y, tr.platform

    def _moved(self, t: float, k: Kill, tr) -> None:
        """Kill `k`, on a track (or let go on none: 0), put by the tracker on track `tr` (None: let go, on none) at t:
        kept in `moved` and told (telemetry event 'kill_moved'), where the hunt follows the tracker's corrections
        (`follows`)."""
        if not self.follows or k.track is None:
            return
        self.moved.append((t, k, k.track, k.platform, k.x))
        if telemetry.recording():
            gained = None if k.gained is None else round(k.gained, 3)
            telemetry.emit("event", kind="kill_moved",
                           text=f"her kill of {gained} from #{k.track} to " + ("none" if tr is None else f"#{tr.id}"),
                           gained=gained, old=k.track, old_platform=k.platform, old_x=round(k.x, 1),
                           new=None if tr is None else tr.id, platform=None if tr is None else tr.platform,
                           x=None if tr is None else round(tr.x, 1))

    def _candidates(self, t: float, kind: str | None) -> list[tuple[_Throw, Monster]]:
        """What a gain seen at t can have been (of `kind`, if not None): the monsters in the paths of the throws of
        the last STAR_WINDOW s, else of those kept (see _place), the newest throw's first, each path's likeliest
        first."""
        ok = (lambda m: True) if kind is None else (lambda m: m.kind == kind)
        recent = [th for th in self._throws if t - th.t <= STAR_WINDOW and any(ok(m) for m in th.path)]
        return [(th, m) for th in reversed(recent or self._throws) for m in th.path if ok(m)]


@dataclass
class Wound:
    """A monster she has hurt and not seen die: as a look last saw it (`m`), when a look first saw it hurt by
    her (`since`), and when one last saw it (`seen`)."""
    m: Monster
    since: float
    seen: float


class Wounds:
    """The monsters she has hurt and not killed (see the module's docstring), kept up from every look that
    searched for monsters and every throw. `ledger`: its kills close the wounds they're placed on."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.open: list[Wound] = []
        self._paths: list[tuple[float, list[Monster]]] = []  # recent throws: when, and their path, followed
        self._last: Scene | None = None   # the last look that searched for monsters
        self._kills = len(ledger.kills)   # the ledger's kills already gone through

    def on_throw(self, when: float, side: str) -> None:
        """A throw (Actions.throw_listeners): what's in its path, from the last look with monsters (in_path)."""
        s = self._last
        if s is not None and s.where is not None and s.platform is not None:
            self._paths.append((when, in_path(s, side, [m for m in s.monsters
                                                        if not dying(self.ledger.kills, m, when, s.monsters,
                                                                     self.ledger.die, self.ledger.dead)])))

    def on_look(self, s: Scene) -> None:
        """After every look (after the ledger's): a monster in a recent throw's path with its bar new or changed
        is hers; her wounds are followed; a kill placed on one closes it (`is_kill`), its track taken for dead does
        (Ledger.dead), and so do WOUND_LOST_S out of sight and WOUND_S since it was first hurt."""
        for k in self.ledger.kills[self._kills:]:
            if k.placed:
                self.open = [w for w in self.open if not is_kill(w.m, k)]
        self._kills = len(self.ledger.kills)
        dead = self.ledger.dead
        self.open = [w for w in self.open if getattr(w.m, "track", None) is None or w.m.track not in dead]
        if not s.searched or s.where is None:
            return
        self._last = s
        self._paths = [(t, path) for t, path in self._paths if s.t - t <= STAR_WINDOW]
        for _, path in self._paths:
            for i, m in enumerate(path):
                if (again := seen_again(m, s.monsters)) is None:
                    continue
                # Its bar new, or other than it was (two of a kind near each other can be taken for each other
                # from look to look: a higher one is still one in her star's path that she has hurt)
                if again.bar is not None and (m.bar is None or abs(again.bar - m.bar) > WOUND_HURT) \
                        and not any(is_one(w.m, again) for w in self.open):
                    self.open.append(Wound(again, s.t, s.t))
                path[i] = again
        hurt = [o for o in s.monsters if o.bar is not None]  # a hurt one shows its bar: those first
        for w in self.open:
            if (again := seen_again(w.m, hurt) or seen_again(w.m, s.monsters)) is not None:
                w.m, w.seen = again, s.t
        self.open = [w for w in self.open if s.t - w.seen < WOUND_LOST_S and s.t - w.since < WOUND_S]

    def seen_within(self, now: float, seconds: float, hurt_by: float | None = None) -> list[Monster]:
        """Her wounds as last seen, those seen within `seconds` of `now` (and first hurt by `hurt_by`, if given)."""
        return [w.m for w in self.open if now - w.seen <= seconds and (hurt_by is None or w.since <= hurt_by)]


def is_kill(m: Monster, k: Kill) -> bool:
    """Whether placed kill `k` is monster `m`'s: its track (Kill.track) where the feed says both (a kill let go, on
    none, is nobody's), else of its kind on its platform within TRACK_SLACK of where it was placed (as a look follows
    one)."""
    if k.track is not None and getattr(m, "track", None) is not None:
        return k.track == m.track
    return m.kind == k.kind and m.platform == k.platform and abs(m.x - k.x) < TRACK_SLACK


def _bound(s: Scene, gained: float | None):
    """The track of look `s` its tracker bound to her gain shown at `gained` (tracking.Track.gain; not a share), or
    None."""
    if gained is None:
        return None
    return next((tr for tr in getattr(s, "tracks", None) or () if getattr(tr, "gain", None) is not None
                 and abs(tr.gain - gained) < TRACK_GAIN_S and not tr.share), None)


def dying(kills: list[Kill], m: Monster, t: float, seen: list[Monster] = (),
          die: dict[str, float] | None = None, dead=()) -> bool:
    """Whether monster `m`, seen at t, is taken for dead or dying: its track (Monster.track) one the feed has dead or
    dying (`dead`: Ledger.dead) or a kill of hers is on (Kill.track); else, by where a kill of hers with no track was
    placed (or for a monster with none): within its kind's death animation (`die`: kind -> MobBody.die_s, the map's game
    data; DIE_S without) of the look that placed it, and of the monsters of its kind on its platform seen with it
    (`seen`), the nearest the kill, within DYING_DX: one only, not a live one beside it."""
    track = getattr(m, "track", None)
    if track is not None and (track in dead or any(k.track == track for k in kills[-8:])):
        return True
    window = (die or {}).get(m.kind, DIE_S)
    near = [k for k in kills[-8:] if k.placed and (k.track is None or track is None) and 0.0 <= t - k.t < window
            and k.kind == m.kind and k.platform == m.platform and abs(k.x - m.x) < DYING_DX]
    if not near:
        return False
    others = [o for o in seen if o is not m and o.kind == m.kind and o.platform == m.platform]
    return any(all(abs(o.x - k.x) >= abs(m.x - k.x) for o in others) for k in near)


def in_path(s: Scene, side: str, monsters: list[Monster], sizes: dict | None = None) -> list[Monster]:
    """The monsters an attack from the ground that way from where she stands in look `s` can take, the
    likeliest first (attack.reach.take_chances, by each kind's body: the nearest the claw's stance reaches,
    else the star's search's first band, any monster in it, whatever the terrain), and of those none whose
    star terrain stops (reach.wall_to: a kill after a throw from 4 isn't put on a monster on 2). The client
    checks the line to its pick alone, and stopped, no other monster takes the star: so with the first band's
    only monster on 2, there's none (one the claw reaches is swiped whatever the terrain). `sizes`: each kind's
    body from the map's game data (localization.monsters.bodies; reach.body_of), else reach's tables (a mushroom as
    a pig, 34 either side where the game has 28): the magician's casts go by the game's (behavior.attacks.Attacks.picks:
    her search's first 50 by a body's far side), the thief's as before."""
    x, y = s.where.x, s.where.y
    ahead = [m for m in monsters if (m.x > x) == (side == RIGHT)]
    bodies = [(abs(m.x - x), m.y - y, *reach.body_of(m.kind, sizes)) for m in ahead]
    chances = reach.take_chances(bodies)
    return [m for c, m, b in sorted(zip(chances, ahead, bodies), key=lambda cmb: -cmb[0])
            if c > 0 and (reach.claw_chance(*b) > 0 or reach.wall_to(s.platform, x, y, m) is None)]


@dataclass
class Pile:
    """Drops lying together (or about to land) somewhere on the map."""
    x: float
    y: float
    platform: int
    since: float              # when its kill happened (hers), or when it was first seen
    hers: bool                # a kill of hers explains it
    mesos: bool = False       # mesos were seen in it
    last_seen: float | None = None
    empty_since: float | None = None  # when a look first saw it empty (see EMPTY_LOOKS)
    status: str = "expected"  # expected (a kill's, not seen yet) | seen | gone (seen empty twice)
    tries: int = 0            # times the hunt loop went for it (DropStore.tried)
    rushed: int = 0           # ... of those, at once for what it's worth (hunt's rule 6a)
    gone_at: float | None = None  # when it was seen gone
    items: tuple[str, ...] = ()   # what the latest look that saw it saw in it (one kind per drop) ...
    worth: float | None = None    # ... what those are worth together (looting.Worth); None until a look sees it ...
    dearest: float | None = None  # ... and the dearest of them on its own
    each: tuple[float, ...] = ()  # ... and what each is worth, one to an item (behavior.hunt: the dearest she can take)
    xs: tuple[float, ...] = ()    # ... and where each lay (map x: the loot sweep's stand covers them)
    unseen: float = PILE_WORTH    # what it's worth before a look sees it: a bronze coin on its map, or (her kill's) its
                                  # kind's expected drops (looting.expected_worth)
    kind: str | None = None       # the kind of the monster whose drops they are (her kill's), if known
    aged: float | None = None     # an older pile of hers' `born`, whose drops it holds (HELD_AGE) while a look sees one
    aged_xs: tuple[float, ...] = ()  # ... where they lie
    none_landed: bool = False     # the target-first planner marked it gone at one look (Grind._none_landed)
    key: int = field(default_factory=lambda: next(_PILE_KEYS), compare=False)  # its own, never another's (the sweep's)

    @property
    def value(self) -> float:
        """What it's worth, in mesos: what was seen in it last, or `unseen` before it's been seen."""
        return self.unseen if self.worth is None else self.worth

    @property
    def born(self) -> float:
        """When its oldest drop fell: its kill (or its first sighting), or an older pile's whose drops it holds
        (`aged`)."""
        return self.since if self.aged is None else min(self.since, self.aged)

    def still(self, now: float, dt: float = 0.0) -> float:
        """The chance it still lies there `dt` s after `now`, given it did when a look last saw it (or, not seen yet, at
        its kill): `survival` at its age (from `born`) then over `survival` at its age when last seen."""
        born = self.born
        ref = 0.0 if self.last_seen is None else max(0.0, self.last_seen - born)
        there = survival(ref, self.hers)
        return 0.0 if there <= 0.0 else min(1.0, survival(now + dt - born, self.hers) / there)


_PILE_KEYS = itertools.count(1)


def survival(age: float, hers: bool = True) -> float:
    """The share of drops still lying `age` s on (SURVIVAL: hers or others'), straight between its points."""
    curve = SURVIVAL[bool(hers)]
    if age <= 0.0:
        return 1.0
    for (t0, s0), (t1, s1) in zip(curve, curve[1:]):
        if age <= t1:
            return s0 + (s1 - s0) * (age - t0) / (t1 - t0)
    return 0.0


@dataclass
class DropStore:
    """Every pile she knows of. Fed by placed kills (Ledger) and by every look that searched for loot."""
    value: Worth = FIRST  # what drops are worth on her map (looting.worth_for)
    piles: list[Pile] = field(default_factory=list)  # the piles not gone, and those gone in the last GONE_KEEP_S
    gone_tried: int = 0  # piles the hunt loop went for that looks have seen gone, ever (dropped ones too)
    revived: int = 0     # piles of hers seen gone that a sighting showed were only missed (`_sighting`), ever
    revive_s: float = REVIVE_S  # how long after it's seen gone a sighting does that (the target-first planner: its whole life)
    _strips: dict = field(default_factory=dict, repr=False)  # each strip of drops' last search taken: (platform, x0) -> t

    @property
    def known(self) -> list[Pile]:
        """The piles not gone."""
        return [p for p in self.piles if p.status != "gone"]

    def tried(self, p: Pile) -> None:
        """The hunt loop went for pile `p` (or gave up on it): one more try. A pile can be tried after a
        look saw it gone (a trip to it then ends), so it's counted in `gone_tried` from its first try."""
        if p.status == "gone" and not p.tries:
            self.gone_tried += 1
        p.tries += 1

    def on_kill(self, kill: Kill) -> "Pile | None":
        """A kill of hers placed: its pile expected where it died (returned; None: no place), worth its kind's expected
        drops (looting.expected_worth) until a look sees it."""
        if kill.platform is None:
            return None
        self.piles.append(Pile(kill.x, kill.y, kill.platform, kill.t, hers=True,
                               unseen=looting.expected_worth(kill.kind, self.value), kind=kill.kind))
        return self.piles[-1]

    def on_look(self, s: Scene) -> None:
        """A look: the strips of drops perception searched since the look before (Scene.strips: the state's rotation,
        each at its own time, and the look's own asked for), then the look's own if it searched for loot (`looked`,
        `loot`): each a sighting (`_sighting`). A strip sees a pile empty only where it's sure (worldstate.DropStrip.sure:
        whole, and its frame's map placed firmly in height: 15 of 49 of her piles were seen gone with their coins there
        on the harness's seed 3 where the rotation searched strips on any frame, the review of 2026-10-30); with the
        state's strips the look's own strips are among them, so the look's sighting confirms what it saw and no more."""
        self.piles = [p for p in self.piles if p.gone_at is None or s.t - p.gone_at <= GONE_KEEP_S]
        ground = not looting.STANDING_ONLY or standing(s)
        strips = getattr(s, "strips", None) or ()
        new = [st for st in strips if st.at is not None
               and st.at > self._strips.get((st.platform, round(st.x0)), -1e18)]
        for st in sorted(new, key=lambda st: st.at):
            self._strips[(st.platform, round(st.x0))] = st.at
            self._sighting(st.at, list(st.loot), lambda pid, x, st=st: pid == st.platform and st.x0 <= x <= st.x1,
                           ground and not st.mesos_only and getattr(st, "sure", True))
        if s.looked:
            self._sighting(s.t, list(s.loot), s.in_view, ground and not any(st.at == s.t for st in strips))

    def searched_empty(self, s: Scene, p: Pile) -> bool:
        """Whether look `s` has a search of pile `p`'s place from when its drops have landed (DROP_LANDS, LANDED_S) that can see
        it empty, as `on_look`'s sightings have one: a strip sure of it, not for mesos alone; without strips the look's own; and
        her standing. Not from SETTLE, the sightings': waiting for that kept each kill's pile open 0.8 s longer and cost her 20
        EXP a minute (dev seeds, t +3.00), and a search too early for drops still showing is undone by a sighting (`revive_s`)."""
        if not standing(s):  # (one look, not two: whatever STANDING_ONLY says of her sightings, none from mid-jump here)
            return False
        after = p.since + DROP_LANDS + LANDED_S
        over = [st for st in getattr(s, "strips", None) or () if st.at is not None and st.platform == p.platform
                and st.x0 <= p.x <= st.x1]
        if over:
            return any(st.at >= after and st.sure and not st.mesos_only for st in over)
        return s.t >= after and bool(s.looked) and s.in_view(p.platform, p.x)

    def _sighting(self, t: float, seen: list[Loot], in_view, ground: bool) -> None:
        """Drops seen at t where `in_view(platform, x)` was searched: each in one pile, the nearest within LOOT_REACH of
        it, which they confirm (or, none, piles of their own, a LOOT_REACH of them from the left: another's, or one no
        kill of hers placed); a pile in view that two sightings EMPTY_LOOKS apart see empty, SETTLE after its kill, is
        gone. `ground`: whether this sighting can see one empty (with looting.STANDING_ONLY only a look showing her
        standing: looting.standing, from mid-jump a look's strips miss the drops on the ground; a strip not sure of it:
        on_look). A strip searched before a pile was last seen says nothing of it. Drops no pile known takes, within
        LOOT_REACH of a pile of hers seen gone REVIVE_S ago at most, are that pile's, seen again: it was missed, not
        taken (until 2026-10-30 they made a pile of another's, a bronze coin's worth, and not a candidate for OWN_S:
        hunt)."""
        known = self.known
        again = [p for p in self.piles if p.status == "gone" and p.hers and p.gone_at is not None
                 and 0.0 <= t - p.gone_at <= self.revive_s]
        mine: dict[int, list[Loot]] = {}
        loose = []
        for d in seen:
            near = [p for p in known if _in_pile(p, d)] or [p for p in again if _in_pile(p, d)]
            if near:
                mine.setdefault(id(min(near, key=lambda p: abs(p.x - d.x))), []).append(d)
            else:
                loose.append(d)
        for p in again:
            if id(p) in mine and t > p.gone_at:
                self.revived += 1
                if telemetry.recording():  # (the first live check of Grind._none_landed: its marks against these)
                    telemetry.emit("event", kind="pile_revived", gone_at=round(p.gone_at, 2), at=round(t, 2),
                                   platform=p.platform, x=round(p.x), none_landed=p.none_landed)
                p.status, p.gone_at, p.empty_since = "seen", None, None  # (missed, not taken: hers again, below)
                known.append(p)
        for p in known:
            if (inside := mine.get(id(p))) is not None:
                if p.last_seen is not None and t < p.last_seen:
                    continue
                p.status, p.last_seen, p.empty_since = "seen", t, None
                p.mesos = p.mesos or any(d.mesos for d in inside)
                p.items, p.worth, p.dearest, p.each = _contents(inside, self.value)
                p.xs = tuple(d.x for d in inside)
                if p.aged is not None:  # (HELD_AGE: the older pile's age while a drop of it is still seen there)
                    p.aged_xs = tuple(x for x in p.aged_xs if any(abs(d.x - x) <= HELD_DX for d in inside))
                    if not p.aged_xs:
                        p.aged = None
            elif ground and t - p.since >= SETTLE and in_view(p.platform, p.x) \
                    and (p.last_seen is None or t > p.last_seen):
                if p.empty_since is None:
                    p.empty_since = t
                elif t - p.empty_since >= EMPTY_LOOKS:
                    p.status, p.gone_at = "gone", t
                    self.gone_tried += p.tries > 0
                    if HELD_AGE and p.hers:  # (its drops another pile of hers by it holds now: theirs, its age)
                        held = [q for q in known if q.hers and id(q) in mine and q.platform == p.platform
                                and abs(q.x - p.x) <= LOOT_REACH
                                and any(abs(d.x - x) <= HELD_DX for d in mine[id(q)] for x in p.xs)]
                        if held:
                            q = min(held, key=lambda q: abs(q.x - p.x))
                            old = tuple(d.x for d in mine[id(q)] if any(abs(d.x - x) <= HELD_DX for x in p.xs))
                            q.aged = p.born if q.aged is None else min(q.aged, p.born)
                            q.aged_xs = q.aged_xs + old
        taken: set[int] = set()
        for d in sorted(loose, key=lambda d: (d.platform, d.x)):
            if id(d) in taken:
                continue
            group = [e for e in loose if id(e) not in taken and e.platform == d.platform
                     and 0.0 <= e.x - d.x <= LOOT_REACH]
            taken.update(id(e) for e in group)
            mid = (min(e.x for e in group) + max(e.x for e in group)) / 2
            self.piles.append(Pile(mid, d.y, d.platform, t, hers=False, mesos=any(e.mesos for e in group),
                                   last_seen=t, status="seen", unseen=self.value.coins["bronze"]))
            pile = self.piles[-1]
            pile.items, pile.worth, pile.dearest, pile.each = _contents(group, self.value)
            pile.xs = tuple(e.x for e in group)


def _in_pile(p: Pile, d: Loot) -> bool:
    return p.platform == d.platform and abs(p.x - d.x) <= LOOT_REACH


def _contents(seen: list[Loot], value: Worth = FIRST) -> tuple[tuple[str, ...], float, float, tuple[float, ...]]:
    """The drops seen in a pile: their kinds, what they're worth together (`value`: the map's looting.Worth), the
    dearest's worth, and each one's."""
    each = tuple(float(value(d.item.kind, d.item.coin)) for d in seen)
    return tuple(d.item.kind for d in seen), float(sum(each)), float(max(each, default=0.0)), each

