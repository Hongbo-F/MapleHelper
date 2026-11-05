"""Her attack a throw (attack.choice: Lucky Seven or the regular attack), chosen as each throw goes out from what she
knows then: the monster it takes, the HP she reckons it has left, and her MP as her bar reads it. Kept up from every
look (Senses.listeners), whoever takes it, and from the throws themselves; every throw goes through Actions, whose
attack key asks `key` (the fights, the hunt loop, grinding and the navigator's fights on the way alike).

- The monster a throw takes: the likeliest in its way from where she stands in the last look that searched for
  monsters (ledger.in_path: the claw's stance, else the star's search's first band) that isn't likelier dead than
  alive by her throws (DEAD_LIKELY: one is, and the star takes the next), else the one she last threw at that way, one
  not likely dead first, else none: then nothing seen is in its way, and with nothing that way that it could take
  there's no throw at all (`_nothing_there`), else it's the regular attack (no MP thrown at nothing). One a look missed
  (those about, `unseen`, and her marks) only while a throw at it is worth its price on the evidence of this look and
  her throws since (`worth_a_throw`: behavior.presence; the magician's casts go at none). Never one dead or dying
  (ledger.dying: with the feed, its track one a kill or a share of hers is bound to, or found dead, Ledger.dead;
  without, of a placed kill's kind on its platform, the nearest it within DYING_DX, while its death animation plays):
  it's still drawn, and no throw at it makes a mark; a mark on one is closed at once (`on_look`). Live on 2026-10-25
  (the hunt of 23:43) the kill was
  placed and its mark dropped, but the dying monster stayed drawn 0.6-1.3 s, was taken from the path again with its
  bar's HP or its whole (a dying monster's bar reads None), got a new mark, and once it was gone the hidden fallback
  followed that mark for 1.8 s with no DEAD_LIKELY check: 18 casts of Lucky Seven and 4 regular throws went at the
  dead in 11 minutes, a tenth of her MP potions (D:/MapleHelper-scratch/analysis-1025/mp-attack-check/atdead.py).
- What it has left (`reckon`: attack.choice.Left, a chance of each amount): its HP in the game data (MapModel.mob_hp),
  less her throws at it. Her throws at a monster are kept with it (`Mark`, followed look to look: monsters.seen_again),
  each taking off its attack's damage as drawn, or nothing (choice.MISS); a look that sees it alive LANDS_S or more
  after a throw rules out what that throw would have killed it with (Left.alive), and one that reads its HP bar
  between empty and full sets it from the bar (the throws that can't have landed by then still taken off). A bar up
  but read empty (at 1366x768 every hurt monster's bar read so until 2026-10-25, perception.hud.hp_bar taking the
  bar's black frame for its fill; the harness's `--world bars-empty`) still says
  her damage shows: not all her throws at it missed (Left.hit), once none is out that could have shown since the
  last look and not landed by LANDS_S (SHOWS_S). Before (review, 2026-10-25) such a bar counted for nothing, and in
  the simulator with the bars drawn empty she reckoned a target's HP 86-98 off (median), throwing no regular attack
  to finish one in 518 throws. Her EXP says it sooner: a look or glance of a frame KILL_SHOWN_S after a throw's key,
  with no gain GAIN_AFTER_KEY after it that can be its target's kill (`heard`: of its kind, a share of one, or one not
  read; not one a kill of hers was placed on elsewhere), rules out what that throw would have killed it with too, seen
  or not (its damage still taken off as drawn). A throw is chosen by what the monster has left with the chance it's dead
  (DEAD_CHOICE: Lucky Seven's throws saved count only if it's alive). A monster she hasn't thrown at, whose bar no
  look reads, has its whole HP. Others' damage to it isn't known: a monster's bar shows its damage to whoever dealt
  it, and the bars on monsters she hadn't thrown at in the live hunts of 2026-10-24 didn't go with other players by
  them (7.2% of sightings with a player on its platform, 9.7% without: her stars that hit another in their path than
  the one she aimed at, and misreads).
- Her MP: the latest look's bar share (Scene.mp) of her MP when full (config.toml [character] mp), less her casts
  that look's frame can't show yet. A cast's MP is taken off as its throw goes out (`on_throw`, told by Actions after
  its key is chosen, at the key's own time); live a cast's MP first showed 0.21-0.53 s after its key, so a bar read
  off a frame shown at t is less the casts keyed after t - MP_FIRST_S (not shown yet), and less those keyed between
  t - MP_SHOWS_S and then (shown or not) where her MP as reckoned says so: between the two her reckoning stands, past
  them the bar is believed so (her pet's drink, her regen, a press that took no MP: `_believed`). Until 2026-10-26 the
  bar was believed from senses.KEY_EFFECT_S, 68 ms, on, and Actions.throw told of a ground throw's cast only at the
  next throw, with that one's key time: harmless while the look before the next key came before the cast showed, 8
  of 10 pairs live, one cast low in the other 2. (Taking the bar as it read, with KEYS_WAIT "moves" in
  loop_tree_mp_runs_out she cast Lucky Seven with 5 MP, nothing thrown: the look after her last cast, shown before
  its key, read the 12 MP it took.) For a day (review, 2026-10-26) a read within MP_SHOWS_S of a cast was believed
  only where it was lower: with a look 0.2-0.34 s after each key, nothing that raised her MP showed while she kept
  casting, and replayed through the live hunts of 23:43 to 01:03 she threw the regular attack for 'MP' 91 times, 45
  of them with a cast's MP surely hers (D:/MapleHelper-scratch/hunt-ab/review-a-correct/mp_replay_code.py); the
  harness 11.75 a run, every one with the MP. Under her pet's MP warning the bar reads 0 for up to 0.6 s after the
  cast that took her there, while she still has the MP for a cast (WARNING_STEPS): such a 0 isn't believed, her MP
  is the last read less her casts since.

The magician's casts (`picks`: a skill that has to pick a body her search finds, her profile's Policy.unseen 'none';
from 2026-10-29, the user watching her grind Tree Dungeon: "Why you were standing on platform 18 and cast magic claw
for nothing, wasting mp?"). Her MP is what her EXP is short of, and a Magic Claw that picks nothing takes its 20 MP all
the same. In the three live grinds of 2026-10-29 01:09-02:03 (745 casts: D:/MapleHelper-scratch/mcasts, classify.py)
half of her casts showed what they did (a kill or EXP of hers, a bar lower or up); the rest went at monsters the look
didn't see then (her memory's, the feed's hidden tracks, the one she last cast at that way within HIDDEN_FOR), at ones
dead by then (a dying body drawn, a kill's ghost), at nothing ('nothing seen in its way': the thief's 'skill' mode
casts whenever the MP pays), at ones seen and unhurt after it, and a few inside her search's first 50 or past it. On
platform 18 at 01:55:17-41, her placed by her minimap dot alone (not seen on screen) and a Green Mushroom 106-150 ahead
in 15 of the 64 looks, 26 casts in 24 s took nothing off it, its bar never up. So for her:
- A body seen (rule 2): her target is one the last look that searched saw, not dying, that her search picks from where
  that look had her (ledger.in_path by the map's bodies, with reach's constants as attack.magic sets them: from 50
  ahead to 300, nothing behind her); never one about that the look missed (`unseen`: the feed's hidden tracks, her
  memory's), nor the one she last cast at that way within HIDDEN_FOR (the thief's fallback for a monster under its
  damage numbers), and with none of them no cast: nothing is cast at nothing (Policy.unseen 'none').
- Not one likely dead (rule 2): her casts reckon it DEAD_LIKELY dead or more, and no look since has seen it alive nor
  her EXP settled it (`reckon`, KILL_SHOWN_S): the next in her search's first band is taken, else none (the thief's
  throw takes the likeliest then); nor, FADE_S after a kill of hers was placed on her mark at it (`_corpses`), a body
  that is it seen again (its track; without the feed, of its kind near where it was): a dying one is drawn (0.84 s a
  mushroom) and fades, and ledger.dying's window ends with the animation (live, 15 casts 0.65-0.76 s after a kill of
  hers went at a body by it).
- Not again where her last cast at it took no effect (rule 3, `Hold`): each cast at a mark is judged at the first look
  LANDS_S to JUDGE_BY_S after its key that sees it again. An effect (its bar PICK_BAR_STEP / 2 lower or one up where
  none was, or so for another monster that look had that way, which her search may have picked; any gain of her EXP
  from SHOWS_S after the key; her attack's MISS popup over it as keycheck reads it: her cast picked it and missed) and
  she casts at it as before; none (its bar as it was, or still none up) holds the mark: no cast at it until a look sees
  new evidence, its bar otherwise (PICK_BAR_STEP), her on another platform or HOLD_MOVED along hers, it on another or
  HOLD_MOVED nearer or farther ahead of her (a band of her search). A held mark is kept HELD_LOST_S unseen (LOST_S
  otherwise) and FORGET_S after her last cast at it, as every mark.
- Nor from where she stands, at any body that way (rule 3, `Spot`; the review of 2026-10-29): the mark's hold alone
  was lost on 18, the case the user saw. It went with the feed's track (a new track for the same body took a cast),
  and with the mushroom's walk of 20 (it walked 106-150 off her), while she stood at one x by her minimap dot (her
  facing or her place, more likely what failed there, are hers, not the mushroom's). So a cast judged no effect holds
  her spot that way too: no cast that way from within HOLD_MOVED of where she cast until her move (HOLD_MOVED along,
  another platform), an effect that way (a bar lower than the look before had it or one up, which only her damage
  makes; any gain of her EXP), her cast the other way (`_turned`: the next one back goes with the arrow tapped, new
  evidence where her keys had her facing wrong; the mark holds that way lifted too), or SPOT_HELD_S (one cast tests
  it). And a mark with casts to judge or a hold that the look doesn't see by its track is carried to a body of its
  kind beside it under a track first seen since (`_carry`: the feed's new one for a body it lost), as a kill's body is
  by place under one first seen after the kill (`_corpse`). WAIT_FIRST (her second cast at a new mark waiting for
  the first's judging) is off: replayed, it held back more hits than it kept casts that showed nothing.
`refused` says which of these keeps her cast off a monster, for the stance (attack.stance's `castable`, behavior.fight's
castable_of: no throw chosen at one she won't cast at, so a fight moves on, or waits, rather than stand pressing a key
that sends nothing), and `key` sends none there whatever the caller (the fights, the hunt loop, the navigator's shot
first), its reason logged the 1st, 10th and 100th time and counted ('no cast: ...', in `summary`; `no_cast`), as the
stance's are (behavior.fight.cast_or_wait: by 'stance', one in choice.THROW_S at most). Replayed through these rules
(D:/MapleHelper-scratch/mcasts/fix, replay2.py: the live looks, kills and EXP, made-up track ids, each logged cast's
key asked at its time), 247 of the 382 casts that showed nothing (65%) would have gone unsent, 4,940 of the 14,900 MP
her casts took, and 67 of the 363 that did something (18%) would have waited (before the review's spot and carry,
235 and 68); on 18 at 01:55:15-42, 3 of the 29 casts sent (a kill, the first at the mushroom and one test SPOT_HELD_S
on) where the mark's hold alone sent 9. The thief's throws are as before (`picks` off).

Her damage a throw, and the share of her throws that take nothing, as the monsters' bars show them (`book`: behavior.
damage.Book, the magician's from 2026-10-29: her profile's prior until the bars say otherwise, learned look by look and
kept in her own file between runs; `_learn_bar`, `_learn_kill`). The thief's attacks are as measured (no book).

Each choice is logged with its numbers, and sent as a telemetry 'attack' record; `counts` keeps them for the run's
report.

Her regular presses are watched for the doubt that its key isn't the game's (behavior.keycheck.RegularWatch, the
Hunter's: `watch`): each one keyed at a target (`key`, or with the regular attack alone her Actions' own key, its target
taken as a throw that way would: `on_throw`), her Lucky Seven casts at one, her kills placed and her EXP's gains, and
every look judges them. Held (a doubt waiting for its check, or after one: RegularWatch.holds), a throw that would be
the regular attack is Lucky Seven while she has the MP for a cast, else no throw (`key` says None).
"""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from ..attack import choice, reach
from . import damage
from ..attack.choice import Attack, Left
from ..localization.mapdata import MapModel
from ..localization.memory import FADE_S
from ..localization.monsters import GAIN_AFTER_KEY, Monster, bodies, same, seen_again
from ..localization.scene import Scene
from ..perception.kinds import kind_of
from .ledger import dying, in_path, is_kill
from . import presence
from .looting import expected_worth, worth_for
from .presence import Evidence

HIDDEN_FOR = 1.8   # s of the doubt's watch (keycheck) and of her hidden wounds (fight.owed): no throw goes by it (`worth_a_throw`)
SEEN_MARK_S = HIDDEN_FOR  # s after her last throw that one goes at a mark a look has, out of its way (`target`)
LANDS_S = 0.6      # s after a throw's attack key by which its damage has shown on its target's bar, if it hit: the
                   # server's (actions.DAMAGE_SHOWS_MS), live a Horny Mushroom's bar first lower 0.21-0.55 s after a
                   # cast's key (p10-p90), a kill's gain by 0.46 s. 0.85 until 2026-10-26, when the star's flight was
                   # taken for when it showed (its release 0.52 s after the key and up to 0.6 s in the air)
SHOWS_S = 0.15     # s after a throw's attack key that its damage may first show: a kill's gain live from 0.15 s on, a
                   # bar's drop from 0.21 (0.47 until 2026-10-26, about when the star leaves)
# s after a throw's attack key by which a frame shows its kill's EXP gain, if it killed: a look or glance of a frame this
# late with no EXP gain GAIN_AFTER_KEY after the key (`heard`, `_unkilled`) rules out what that throw would have killed
# its target with (Left.alive), its damage still taken off as drawn (LANDS_S says when that shows). Live on 2026-10-26 (the grinds of 21:04 and 21:18,
# the service at 30 frames a second) the perception service had the gain of 97.4% of 237 kills by 0.43 s after the
# killing key (its frame about 0.03 s before), 98.4% by 0.5, the median 0.22 (D:/MapleHelper-scratch/akill/diag
# kills.py); last night's service, at about 15 a second, 50-96% by 0.43. Until then a kill was ruled out only by a look
# LANDS_S after its throw that saw its target, and the look a throw was decided on came 0.2-0.34 s after the key before:
# 232 of 652 throws at a target went out reckoned 50% dead or more, 224 of them alive at the key (diag/deadcasts.py),
# and an unseen one's reckoning only grew with her throws (66% -> 89% -> 96% dead, at a Green Mushroom alive). Replayed
# through this code (D:/MapleHelper-scratch/akill/fix/replay.py: the grinds of 21:04-22:28, 2,525 throws at a target
# with the truth known): reckoned 50% dead or more at the key on 72 (28 at a monster dead by then, 11 that killed it, 33
# alive), where master's reckoning replayed had 853 (33, 250, 570), within 0.02 of the live records' on 2,377 of 2,525;
# the hunts of 23:43-01:28 (2,123, the service slower) 253 (158, 19, 76) against 737 (162, 169, 406), master's replayed
# matching live's on 1,352 of 2,120 only (the code that ran then flagged 285): those left mostly had a gain that could be
# its kill when it would show.
KILL_SHOWN_S: float | None = 0.4  # (None: no throw's kill ruled out so, as until 2026-10-26)
GAINS_KEPT_S = 5.0  # s her EXP gains are kept for `heard` (her throws' marks have a pending throw this long at most)
GAIN_SAME_S = 0.02  # s: a placed kill's gain (ledger.Kill.gained) is the one `heard` has within this
LOST_S = 3.0       # s unseen before a monster she threw at is forgotten (ledger.WOUND_LOST_S) ...
FORGET_S = 15.0    # ... and s since her last throw at it (ledger.WOUND_S)
MODES = ("choice", "skill", "regular")  # attack.choice.pick's modes: `tour --attack` (choice by default)
DEAD_LIKELY = 0.5  # a monster in a throw's way this likely dead by her throws (Left.dead) isn't its target: the next is
# Whether a throw's attack is chosen with the chance its target is dead (choice.finishes on the whole reckoning: what
# Lucky Seven saves counts only if it's alive), not by what it has left if alive alone (False: as until 2026-10-26).
# Only with KILL_SHOWN_S: on the stale chances before it, replayed through the live grinds of 2026-10-26 21:04-22:28
# (74 minutes, 2,657 throws) it turned 274 casts at monsters alive at the key into regular stars, 54 of them casts that
# killed, against 11 at monsters dead by then; with it, 3 (1 that killed) against 2, and through the hunts of
# 2026-10-25/26 23:43-01:28 (71 minutes) 9 (4) against 17 (D:/MapleHelper-scratch/akill/fix/replay.py; 18 (3) against 9
# and 12 (4) against 18 before `_unkilled` took a gain only GAIN_AFTER_KEY after its throw's key).
DEAD_CHOICE = True
# s after a cast's attack key that its MP may first show on her bar (MP_FIRST_S), and by when it has (MP_SHOWS_S): live
# on 2026-10-25 (the hunt of 23:43, 1366x768) a cast that went out first showed 0.21-0.53 s after its key, the bar
# falling over 0.1-0.2 s more (isolated casts: D:/MapleHelper-scratch/analysis-1025/mp-attack-check/nodrop.py,
# lowcheck2.py). A read off a frame shown at t is less the casts keyed after t - MP_FIRST_S, and those keyed from
# t - MP_SHOWS_S to then are taken off it as far as her reckoning (on_throw took them off) says (`_believed`).
MP_FIRST_S = 0.21
MP_SHOWS_S = 0.55
# Her pet's MP warning (System Options, 5% steps: sim.world.PET_MP) makes the bar read 0: live on 2026-10-25 every one
# of 26 runs of 0 reads began 0.20-0.46 s after a cast's key and lasted 0.6 s at most, ending in the pet's drink (24)
# or, the potions out, in the true 3 px and 1 px; no look ever read 4 or 5 px, and the bar read 6-9 px before each, so
# the cast took her under 5% (about 20-35 MP: a cast's 14 more), and 11 regular throws went out for 'MP 0' with the MP
# for a cast (mp-attack-check/zeros.py, series.py). Which frame the game draws then isn't known (perception.hud.mp_fill
# finds no blue on its row: a warning flash?). So a 0 read within WARNING_S of a read above 0 isn't believed where that
# read, or her MP as reckoned since (the read less her casts), is WARNING_STEPS or less (of choice.MP_STEPS): her MP is
# the reckoning (and a 0 read longer is believed). By the reckoning too since the review of 2026-10-26: replayed, 2 of
# the live runs of 0 came after a read of 10 px and two casts (5.3 px, 33 MP) and were believed.
WARNING_STEPS = 9
WARNING_S = 2.0
# Her casts judged (`picks`, the magician's: the module's doc, rule 3): at the first look LANDS_S to JUDGE_BY_S after a
# cast's key that sees its target again. The thief's regular presses are judged by 2.0 (keycheck.JUDGE_BY_S) and an
# unhurt one by two looks UNHURT_AGAIN_S apart (a bar a look missed); hers by one look, out to 3.0 s: live on platform
# 18 the looks had her target in 15 of 64 (the feed loses a monster under her hit's 83x94 effect for 570 ms), and a bar
# a look missed holds the mark only until the next look reads it, new evidence. A guess, a cast's 20 MP against her
# waiting a look.
JUDGE_BY_S = 3.0
PICK_BAR_STEP = 0.03  # a bar this much lower is damage (keycheck.BAR_STEP): a no effect's is under half of it lower
HOLD_MOVED = 20.0     # units: her or its move along that is new evidence for a held mark (reach.SEARCH_STEP: a band of
                      # her search; a Green Mushroom walks it in 0.2 s, and live on 18 one walked 106-140 off her)
HELD_LOST_S = 6.0     # s unseen before a held mark is forgotten (LOST_S for one that isn't): its no effect outlives a
                      # look or two that miss it (live, 3.3 s on 18)
# A spot of hers held (Spot: the module's doc, rule 3, after the review of 2026-10-29): this long with nothing new, one
# cast tests it again. A backstop: her move, a turn or an effect that way lifts it sooner. On 18 at 01:55:17-41 she
# stood 24 s at one x by her minimap dot (26 casts); held so, a test every 6 s. A guess, as HELD_LOST_S.
SPOT_HELD_S = 6.0
# Another player this near a monster (map units, along and up or down): nothing learned of her damage there (`book`): a
# bar shows whoever's damage, and another's isn't told from hers.
OTHERS_DX = 400.0
OTHERS_DY = 250.0
# Whether a mark none of her casts has shown an effect on yet (Mark.proven) takes no second cast until the first one is
# judged (rule 3, `refused`'s 'first cast unjudged'): her next throw's look comes about 0.48 s after its key
# (Senses.shows: KILL_SHOWN_S), under LANDS_S, so every new mark took two casts before the first could be judged no
# effect (the review of 2026-10-29, on 18); an effect shown from SHOWS_S on lets the second go at once (a bar's first
# drop live 0.21-0.55 s after the key, p10-p90). Off: replayed through the live grinds of 01:09-02:03 (mcasts/fix,
# replay2.py) it kept 8 more of the 382 casts that showed nothing (255 against 247) and held back 17 more of the 363
# that did (84 against 67), the second cast at a new target mostly waiting where its first had hit; her spot's hold
# (Spot) already stops a stand's casts after the first two.
WAIT_FIRST = False
# Throws at one monster with no HP bar on it and no EXP since the first, whose results should have shown by now: then the
# next throw taps the arrow, as she may not be facing the way she's thought to (the user, 2026-11-04: Actions.retap).
UNLANDED_THROWS = 2


@dataclass(frozen=True)
class _Read:
    """A frame's MP bar as `Attacks._mp_read` takes it (a look's Scene has the same)."""
    t: float
    mp: float | None


@dataclass
class Mark:
    """A monster she has thrown at: as a look last saw it, when, what it has left as of the throws looks have seen
    land (`left`), and her throws at it since (when, which attack)."""
    m: Monster
    seen: float
    left: Left
    pending: list[tuple[float, Attack]] = field(default_factory=list)
    last: float | None = None  # her last throw at it
    whole: bool = True  # `left` is reckoned from its HP before any throw of hers (False: from a bar read since)
    casts: list["Cast"] = field(default_factory=list)  # her casts at it waiting to be judged (`picks`: rule 3)
    bar: float | None = None  # its bar as a look last read it between empty and full (None: none yet)
    held: "Hold | None" = None  # her last cast at it took no effect: none at it until new evidence (`picks`)
    proven: bool = False  # a cast of hers at it has shown an effect since it was marked or its hold lifted (`picks`)
    read: tuple[float, float] | None = None  # its bar as a look last read it between empty and full, and when (`book`)
    keyed: list = field(default_factory=list)  # her throws at it, [key time, Attack] each, oldest first (`book`)
    blind: list = field(default_factory=list)  # her throws at it while no look had it, waiting to be judged (`_judge_blind`)
    blank: int = 0  # how many of those in a row showed nothing, since a look last had it
    shown: bool = False  # ... and whether the last of them showed something

    @property
    def thrown(self) -> float:
        return self.seen if self.last is None else self.last


@dataclass
class Cast:
    """A cast of hers at a mark, waiting to be judged (Attacks.picks: rule 3): its attack key's time (its choice's until
    its throw says), its target's bar before it (None: none up, unhurt), where the look it was chosen on had her (map
    x, platform) and it (`ahead`: how far off along, either way), and the other monsters that look saw the way she cast,
    with their bars (None: none up): her search may pick one of those instead (live at 02:02:56 a Horny Mushroom on
    its window's edge 60 over her was her target, and her casts went to a Green Mushroom 63 under her, killing it)."""
    t: float
    bar: float | None
    x: float
    platform: int | None
    ahead: float
    others: tuple = ()  # ((Monster, its bar), ...)
    side: str = RIGHT  # the way she cast


@dataclass
class Blind:
    """A throw of hers at a mark no look had (`worth_a_throw`), waiting to be judged at the first look LANDS_S after its
    key (`Attacks._judge_blind`): its attack key's time (its choice's until its throw says), the way she threw, the MP
    share it should take (None: no MP), whether it's from the ground, her MP bar before it, whether an effect has shown."""
    t: float
    side: str
    cast: float | None
    ground: bool
    mp: float | None
    shown: bool = False


@dataclass(frozen=True)
class Hold:
    """A held mark (Attacks.picks: rule 3): her cast at it keyed at `cast` took no effect, judged at the look at `t`,
    which had her at map x `x` on `platform` and it `ahead` of her (along) on `its` platform with bar `bar` (None: none
    up), the cast that way (`side`). New evidence, as later looks see it, lifts it (Attacks._lift), and her cast the
    other way (Attacks._turned)."""
    cast: float
    t: float
    x: float
    platform: int | None
    ahead: float
    its: int | None
    bar: float | None
    side: str = RIGHT


@dataclass(frozen=True)
class Spot:
    """A spot of hers held (Attacks.picks: rule 3, the review of 2026-10-29): her cast that way (`side`) from map x `x`
    on `platform`, keyed at `cast`, took no effect, judged at the look at `t`: no cast that way from within HOLD_MOVED
    of it, at any body, until new evidence (Attacks._lift_spots) or SPOT_HELD_S."""
    side: str
    platform: int | None
    x: float
    t: float
    cast: float


class Attacks:
    """Her attack a throw on map `model` with keys `keys`: in `mode` (attack.choice.pick: 'choice', 'skill' or
    'regular'), with MP `max_mp` when full and Lucky Seven at skill `level` (None if she has no Lucky Seven key).
    `kills`: her kills so far (the ledger's), which end the marks they're placed on; `dead`: the feed's tracks taken for
    dead (Ledger.dead), whose marks end too."""
    # (the doubt's: see __init__; here too for an Attacks made without it, as the tests' stand-ins are)
    watch = None
    casts = None
    roped = None
    forced = None
    _target = None
    _regular = False
    _asked = False
    picks = False
    _corpses: list = []
    _judged: Mark | None = None
    _bodies = None
    _spots: dict = {}
    _known: dict = {}
    _keyed_at = -1e18
    _said_at = -1e18
    next_key = None
    worth: dict = {}
    rate = None
    touched = None
    _told: list = []
    _newest = None
    _blinded = None

    def __init__(self, model: MapModel, keys, mode: str = "choice", max_mp: float = 623.0, level: float = 16.0,
                 log: Callable[[str], None] = print, kills: Callable[[], list] | None = None,
                 dead: Callable[[], object] | None = None, skill: Attack | None = None, magic: bool = False,
                 no_regular: bool = False, picks: bool = False, book: "damage.Book | None" = None,
                 her_level: Callable[[], int | None] | None = None):
        """`skill`: her skill if not Lucky Seven at `level` (the magician's Magic Claw: behavior.hunter, her profile's),
        `magic` its damage against the monsters' defence against magic (MapModel.mob_mdd); `no_regular`: her regular
        attack never thrown (the magician's wand: no swing at range, none to finish a monster, none for MP: `key` says
        None, no throw, where it would be the regular attack); `picks`: her skill casts only at a body her search picks
        from where she stands, seen, not likely dead, not held (the magician's: Policy.unseen 'none'; the module's doc:
        no cast at nothing, at a body unseen or dead, nor again where it took no effect); `book`: her damage learned
        from the bars (behavior.damage, the magician's), her skill's taken from it as it learns, and what it learns
        from her looks kept there; `her_level`, her level as perception reads it (the book's reads are kept by it)."""
        if mode not in MODES:
            raise ValueError(f"no such attack mode: {mode!r} (one of {', '.join(MODES)})")
        self.keys, self.mode, self.max_mp, self.log = keys, mode, max_mp, log
        self.skill: Attack | None = (skill if skill is not None else choice.lucky_seven(level)
                                     if keys.lucky_seven is not None else None)
        self.regular = choice.REGULAR
        self.no_regular = no_regular
        self.picks = picks
        self.book = book
        self.her_level = her_level or (lambda: None)
        self._prior = self.skill  # (her skill as her profile has it: the book's fit goes over it)
        self.miss = choice.MISS  # the share of her throws that take nothing (the book's, once it knows)
        self._book_version = -1
        self._take_book()
        self._corpses: list[tuple[float, Monster]] = []  # (picks) her marks a kill of hers was placed on: (when, it)
        self._judged: Mark | None = None  # (picks) the mark `key` just put a cast on, until its throw says when
        self._spots: dict[str, Spot] = {}  # (picks) her spot held each way (rule 3: Spot)
        self._known: dict[int, list[float]] = {}  # (picks) the feed's tracks her looks have had: [first seen, last]
        self._keyed_at = -1e18  # (picks) her last cast's key (on_throw) ...
        self._said_at = -1e18   # ... and her last no cast counted (no_cast)
        defence = getattr(model, "mob_mdd", {}) if magic else model.mob_pdd
        self.stats = {kind_of(name): (hp, defence.get(name, 0.0)) for name, hp in model.mob_hp.items() if hp > 0}
        # each kind's death animation (ledger.dying), the map's game data, and (`picks`) its body: what her search picks
        self._die = {k: b.die_s for k, b in bodies(model).items()} if hasattr(model, "mob_bodies") else {}
        self._bodies = bodies(model) if hasattr(model, "mob_bodies") else None
        self.marks: list[Mark] = []
        self.mp: float | None = None    # her MP bar's share in the latest look, less her casts it can't show (on_look)
        self._cast: float | None = None  # the MP share of the cast `key` chose, until its throw says when it's keyed
        self._casts: list[tuple[float, float]] = []  # her casts' attack keys (on_throw) and their MP shares, from
                                                      # MP_SHOWS_S before the latest look on
        self._read: tuple[float, float] | None = None  # the last bar share read above 0, and when (WARNING_STEPS)
        self._warned = False  # the last bar read was a 0 under her pet's warning (counted once a run of them)
        self.counts: Counter = Counter()  # throws by attack (its key) and by why
        self._heard = -1e18  # the newest frame a look or glance has shown her EXP up to (`heard`) ...
        # ... and her EXP's gains up to then, the last GAINS_KEPT_S: each one's frame time and the kinds of kill it can
        # be (None: any), those a kill of hers was placed on gone
        self._gains: dict[float, frozenset | None] = {}
        self._keyed: tuple[Mark, int, int] | None = None  # the pending throw `key` just made (its mark, its index in
        # the mark's pending and in its keyed), until its throw says when
        self._shows = -1e18  # when a frame shows her last throw at a target's kill, if it killed (`shown_by`)
        self._last: Scene | None = None  # the latest look that searched for monsters, placed
        self._kills = kills or (lambda: [])
        self._dead = dead or (lambda: ())
        self._killed = 0
        # The monster the next throw is at, if the stack knows better than a throw from the ground would (a jump's
        # attack at one on a platform above: behavior.fight.throw_act), taken by that throw's key alone
        self.aimed: Monster | None = None
        # The monsters about her that a look didn't find, which a throw takes as the look's own (target): the hunt
        # loop's (Hunt._unseen_here: those on her platform, the dead ones' tracks left out; hunt.UNSEEN_ATTACK); None:
        # the look's alone. With a monster under her body unfound (live on 2026-10-25, 7-12% of frames found one within
        # 50 units: localization.unseen), a throw at it was 'nothing in its way' and went out as the regular attack.
        self.unseen: Callable[[Scene], list[Monster]] | None = None
        # Out of darts' watch (behavior.darts.DartWatch: the Hunter's), told of her MP bar's every read and every cast
        self.darts = None
        # Whether the throw whose key is being chosen is one from the ground (the Hunter's: Actions.keyed_ground), and
        # whether the cast `key` just chose is at a monster the claw reaches from there (`_swipe`: no MP whatever her
        # darts, so out of darts doesn't count it: DartWatch.cast) and whether it's from the ground (`_ground`: one on
        # her after a touch takes its swipe, DartWatch.touched)
        self.from_ground: Callable[[], bool] | None = None
        self._swipe = False
        self._ground = True
        # The doubt that her regular attack's key isn't the game's (behavior.keycheck.RegularWatch: the Hunter's), her
        # look having her on a rope or at its top (the Hunter's: those presses aren't judged), an attack forced for the
        # throw `key` is asked next ("lucky_seven": her stars' test, behavior.keycheck), and what `key` chose for its
        # throw: its target, whether it's the regular attack, and whether `key` was asked at all (with the regular attack
        # alone her Actions key it themselves)
        self.watch = None
        self.retap: Callable[[], None] | None = None  # (Actions.retap)
        self.gains: Callable[[], int] | None = None   # (her kills her EXP told of, ever: the Hunter's exp_gains)
        self._unlanded: tuple = (None, 0, 0)          # (`_landing`: the track, her throws at it, her gains at the first)
        self.casts = None  # ... and her Lucky Seven casts so (keycheck.RegularWatch, `kind` "casts": the Hunter's)
        self.roped: Callable[[], bool] | None = None
        self.forced: str | None = None
        self._target: Monster | None = None
        self._regular = False
        self._asked = False
        self.thrown: tuple[float, Monster | None] | None = None  # the last throw's key time and what `key` aimed it at
        # What her throws at monsters no look has go by (`worth_a_throw`: behavior.presence): each kind's worth in EXP (its
        # EXP and expected drops), her EXP a second (the Hunter's: fight.made_rate; None: the prior), whether a monster the
        # looks miss has just touched her (the Hunter's: `_nothing_there`), every gain her EXP told lately, kept whoever's
        # kill it was placed on (`_gains` loses those), the feed's tracks as of its newest frame (`newest`), the mark
        # `key` just put a throw at one on, until its throw says when
        self.worth = _worth_of(model)
        self.rate: Callable[[], float] | None = None
        self.touched: Callable[[Scene], bool] | None = None
        self._told: list[tuple[float, frozenset | None]] = []
        self._newest: tuple[float, list] | None = None
        self._blinded: Mark | None = None

    # --- after every look ------------------------------------------------------------------------

    def on_look(self, s: Scene) -> None:
        """Her MP; the marks followed, their bars read, forgotten when unseen LOST_S, untouched FORGET_S or killed."""
        self._newest = None  # (this look's tracks are as new)
        if (read := self._mp_read(s)) is not None:
            self.mp = self._believed(read, s.t)
        if self.darts is not None:
            self.darts.see(s.t, s.mp)
        if self.book is not None and self.book.version != self._book_version:
            self._take_book()
        kills = self._kills()
        for k in kills[self._killed:]:
            if getattr(k, "placed", False):
                for w in (self.watch, self.casts):
                    if w is not None:
                        gained = getattr(k, "gained", None)
                        w.killed(k.t if gained is None else gained, k.kind, k.platform, k.x, getattr(k, "track", None))
                if self.picks:  # (her: its body is drawn dying, then fades, FADE_S: no cast at it seen again)
                    self._corpses += [(s.t, mk.m) for mk in self.marks if is_kill(mk.m, k)]
                if self.book is not None:  # (what the throw that killed it did at least: her damage learned)
                    for mk in self.marks:
                        if is_kill(mk.m, k):
                            self._learn_kill(mk, k, s)
                self.marks = [mk for mk in self.marks if not is_kill(mk.m, k)]
                if (gained := getattr(k, "gained", None)) is not None:  # its gain was that monster's: no other's kill
                    for g in [g for g in self._gains if abs(g - gained) < GAIN_SAME_S][:1]:
                        del self._gains[g]
        self._killed = len(kills)
        dead = self._dead()  # (a mark on one dead or dying: no sighting or bar of it says it's alive)
        self.marks = [mk for mk in self.marks if getattr(mk.m, "track", None) is None or mk.m.track not in dead]
        if not s.searched or s.where is None or s.platform is None:
            return
        prev, self._last = self._last, s
        self._corpses = [(t, m) for t, m in self._corpses if s.t - t < FADE_S]
        found = [seen_again(mk.m, s.monsters) for mk in self.marks]
        if self.picks:  # (her spots' holds lifted by what this look sees, her marks the feed lost carried: rule 3)
            self._lift_spots(s, prev)
            found = self._carry(found, s)
            self._seen_tracks(s)
        else:  # (her throws at the ones no look had, judged: the thief's rule 3)
            self._judge_blind(found, s, prev)
        for mk, again in zip(self.marks, found):
            if again is None:
                continue
            if self.picks:  # (her casts at it judged, and a hold lifted, by what this look sees: rule 3)
                self._lift(mk, again, s)
                self._judge(mk, again, s)
            if self.book is not None and _believable(again.bar):  # (what her throw since the bar before did)
                self._learn_bar(mk, again, s)
            mk.m, mk.seen = again, s.t
            if _believable(again.bar):
                mk.bar = again.bar
                mk.read = (s.t, again.bar)
            landed = [a for t, a in mk.pending if t <= s.t - LANDS_S]
            mk.pending = [(t, a) for t, a in mk.pending if t > s.t - LANDS_S]
            if _believable(again.bar) and (stats := self.stats.get(again.kind)) is not None:
                mk.left, mk.whole = Left.of(again.bar * stats[0], stats[1]), False
                continue
            if landed:  # alive after them: what would have killed it ruled out
                left = mk.left
                for a in landed:
                    left = left.after(a, self.miss)
                mk.left = left.alive() or mk.left  # (none alive: her throws did less than reckoned, taken as nothing)
            if again.bar is not None and mk.whole and not any(t <= s.t - SHOWS_S for t, _ in mk.pending):
                mk.left = mk.left.hit() or mk.left  # her damage shows: not all of her throws reckoned in it missed
        self.marks = [mk for mk in self.marks if s.t - mk.seen < (LOST_S if mk.held is None else HELD_LOST_S)
                      and s.t - mk.thrown < FORGET_S]
        if self.watch is not None:
            self.watch.look(s.t, s.monsters, s)  # (her regular presses judged: the doubt)
        if self.casts is not None:
            self.casts.look(s.t, s.monsters, s)  # (... and her casts)

    def glance(self, t: float, mp: float | None) -> None:
        """Her MP bar's share `mp` off a frame shown at `t` newer than the latest look (None: not read), as on_look
        reads it: the glance before a throw's key (behavior.hunter.Hunter._still). Her pet's drink shows about 0.3 s
        after the cast that brings it, and the look a throw is decided on comes 0.2-0.34 s after the key before (live,
        2026-10-25): without it, a drink shown between the look and the key went unseen, and the throw went the regular
        attack for 'MP' (the harness: 1.25 a run with her MP back)."""
        if (read := self._mp_read(_Read(t, mp))) is not None:
            self.mp = self._believed(read, t)
        if self.darts is not None:
            self.darts.see(t, mp)

    def heard(self, t: float, gains: list[tuple[float, frozenset | None]]) -> None:
        """Her EXP as a look or glance of a frame shown at `t` has it (behavior.hunter: Hunter._vitals, every look's
        changes of her EXP number; Hunter._still, the feed's gains not drained yet at a throw's key): `gains`, the frames
        her EXP went up in since the last, each with the kinds of kill its size names (None: any; a gain no kill of the
        map's explains, a share of one another player hurt too, may be any: hunter.kinds_of). A throw keyed KILL_SHOWN_S
        or more before the newest such frame, with no gain GAIN_AFTER_KEY after its key that can be its target's kill,
        didn't kill (`reckon`); a gain a kill of hers was placed on (ledger.Kill.gained) is that monster's, no other's (on_look)."""
        self._gains = {g: k for g, k in self._gains.items() if t - g < GAINS_KEPT_S}
        self._gains.update((g, k) for g, k in gains if t - g < GAINS_KEPT_S)  # (a look's after a glance's: its meter's)
        self._told = [(g, k) for g, k in self._told if t - g < GAINS_KEPT_S]
        self._told += [(g, k) for g, k in gains if t - g < GAINS_KEPT_S and not any(abs(g - o) < GAIN_SAME_S
                                                                                     for o, _ in self._told)]
        self._heard = max(self._heard, t)
        for w in (self.watch, self.casts):
            if w is not None:
                w.heard(gains)

    def newest(self, t: float, tracks) -> None:
        """The feed's tracks off its newest state, of a frame shown at `t` newer than the latest look (None: none told):
        what her last look before a throw's key knows of a monster the look missed (Hunter._still; `worth_a_throw`)."""
        if tracks:
            self._newest = (t, list(tracks))

    def shown_by(self) -> float:
        """When a frame shows what her last throw at a target did to it, if it killed it: KILL_SHOWN_S after its key
        (Senses.shows: the look a throw after it is decided on waits for that frame), or long ago."""
        return self._shows

    def _unkilled(self, t: float, kind: str) -> bool:
        """Whether a throw keyed at `t` at a monster of `kind` is known not to have killed it: a frame KILL_SHOWN_S after
        it has shown her EXP, with no gain that can be its kill (`heard`) GAIN_AFTER_KEY after its key, when its kill's
        would show. Any gain since counted at first (review, 2026-10-27): replayed through the grind of 21:34, a gain of
        no kind known 0.66 s after a cast kept the one before it, keyed 0.94 s earlier, 72% dead, and the next five
        throws at that Horny Mushroom, alive, went out the regular attack 'finishing it' (4 casts killed it live);
        bounded, over the grinds of 21:04-22:28 and the hunts of 23:43-01:28, 12 casts turned regular at a monster alive
        at the key where there were 30, and 19 at the dead where there were 27 (D:/MapleHelper-scratch/akill/fix/
        replay.py; review-behaviour/cmp_win.py)."""
        lo, hi = GAIN_AFTER_KEY
        return KILL_SHOWN_S is not None and self._heard >= t + KILL_SHOWN_S and not any(
            t + lo <= g <= t + hi and (kinds is None or kind in kinds) for g, kinds in self._gains.items())

    # --- her casts (`picks`: the magician's; the module's doc) -------------------------------------

    def refused(self, m: Monster) -> str | None:
        """Why her cast won't go at monster `m` (None: it may; always None for the thief, `picks` off): 'unseen', not a
        body the last look that searched saw (rule 2: her memory's, the feed's hidden tracks, the fights' hidden ones);
        'dead', dying (ledger.dying) or the body of her mark a kill of hers was placed on (`_corpse`); 'no effect', a
        held mark (rule 3: her last cast at it took none, no new evidence since); 'no effect from here', her spot held
        that way (rule 3: Spot, whatever the body); 'likely dead', her casts reckon it DEAD_LIKELY dead or more (rule
        2); 'first cast unjudged', a mark none of her casts has shown an effect on, its first not judged yet
        (WAIT_FIRST). Whether her search picks it from where she stands is the stance's to ask (attack.stance's
        `castable`), and `target`'s by the look."""
        if not self.picks:
            return None
        s = self._last
        if s is None or not any(o is m for o in s.monsters):
            return "unseen"
        if self._dying(m, time.monotonic(), s.monsters) or self._corpse(m, s.monsters):
            return "dead"
        mk = self._mark_of(m)
        if mk is not None and mk.held is not None:
            return "no effect"
        if self._held_here(RIGHT if m.x > s.where.x else LEFT, s):
            return "no effect from here"
        if (left := self.reckon(m)) is not None and left.dead >= DEAD_LIKELY:
            return "likely dead"
        if WAIT_FIRST and mk is not None and mk.casts and not mk.proven:
            return "first cast unjudged"
        return None

    def hurt(self, m: Monster) -> float | None:
        """(`picks`) The share of its HP monster `m` has left as she reckons it, if she has thrown at it and it's under
        its whole, alive (not likely dead: DEAD_LIKELY) and not held (rule 3): one of her wounds, finished first
        (attack.stance's WOUNDS_FIRST); else None."""
        if not self.picks:
            return None
        mk = self._mark_of(m)
        stats = self.stats.get(m.kind)
        if mk is None or not mk.keyed or mk.held is not None or stats is None:
            return None
        left = self.reckon(m)
        if left is None or left.dead >= DEAD_LIKELY:
            return None
        share = (left.alive() or left).hp / stats[0]
        return share if share < 0.999 else None

    def hidden_hurt(self, s: Scene) -> list[Monster]:
        """(`picks`) Her wounds (`hurt`'s) that look `s` doesn't see, as last seen within attack.stance.WOUND_HIDDEN_S
        (under her hit's effect, 570 ms; the feed's track lost), not dying: waited for rather than turned from."""
        if not self.picks:
            return []
        from ..attack.stance import WOUND_HIDDEN_S
        now = time.monotonic()
        return [mk.m for mk in self.marks
                if mk.seen < s.t and s.t - mk.seen <= WOUND_HIDDEN_S and seen_again(mk.m, s.monsters) is None
                and not self._dying(mk.m, now) and self.hurt(mk.m) is not None]

    def refusal(self, side: str) -> tuple[str, Monster | None] | None:
        """(`picks`) Why her cast that way would send none now, and the first body in her search's way if any
        (`_her_target`'s), or None: it would go (always None for the thief). What behavior.fight.cast_or_wait says."""
        if not self.picks:
            return None
        target, why, first = self._her_target(side)
        return None if target is not None else (why, first)

    def no_cast(self, side: str, why: str, first: Monster | None = None, by: str = "stance") -> bool:
        """(`picks`) A cast of hers that way not sent (`why`, at `first`: the body her search would pick, if any),
        counted ('no cast: <why>', in `summary`), logged the 1st, 10th and 100th time of each why, and sent as a
        telemetry 'no_cast' record (`by`: 'key', `key` refusing her throw's key; 'stance', the stance or
        behavior.fight.cast_or_wait choosing no throw for it, the review of 2026-10-29: those were silent). The stance's
        are asked look after look: one counts only choice.THROW_S (her casts' pace) after her last cast's key and the
        last counted, so the counts are about the casts she didn't send. Whether it counted."""
        now = time.monotonic()
        if by != "key" and now - max(self._keyed_at, self._said_at) < choice.THROW_S:
            return False
        self._said_at = now
        said = f"no cast: {why}"
        self.counts[said] += 1
        if self.counts[said] in (1, 10, 100):
            where = None if self._last is None else self._last.where
            at = "" if first is None or where is None else f" ({first.kind} {first.x - where.x:+.0f})"
            self.log(f"    no cast {side}: {why}{at}; {self.skill.name if self.skill else 'her skill'} goes only at a "
                     f"body seen that her search picks from where she stands"
                     + ("" if self.counts[said] == 1 else f" ({self.counts[said]} times)"))
        if telemetry.recording():
            telemetry.emit("no_cast", why=why, side=side, by=by, **_target_fields(first))
        return True

    def _held_here(self, side: str, s: Scene) -> bool:
        """(`picks`) Whether her spot is held that way at look `s` (Spot: her on its platform within HOLD_MOVED)."""
        sp = self._spots.get(side)
        return (sp is not None and s.platform is not None and s.where is not None and s.platform.id == sp.platform
                and abs(s.where.x - sp.x) < HOLD_MOVED)

    def boxes(self, t: float) -> list[Monster]:
        """The targets of her casts waiting to be judged (`picks`) or of her throws at those no look had (`_judge_blind`),
        whose MISS popup a look at `t` could show (keycheck's MISS_FROM_S to MISS_BY_S after the key): where
        behavior.hunter reads her attack's MISS, as it does over her regular presses' targets (Hunter._miss_points)."""
        from .keycheck import MISS_BY_S, MISS_FROM_S
        return [mk.m for mk in self.marks if any(c.t + MISS_FROM_S <= t <= c.t + MISS_BY_S for c in [*mk.casts, *mk.blind])]

    def _her_target(self, side: str) -> tuple[Monster | None, str, Monster | None]:
        """(`picks`) Her cast's target that way: of the bodies the last look that searched saw, not dying nor a corpse
        of her marks' (they take no cast: the client's search passes over the dead), those her search picks first from
        where the look had her (ledger.in_path: its first band, any of them alike), the first she'd cast at (`refused`:
        not held, not likely dead). With none, (None, why, the first in her search's way if any): 'nothing seen' (no
        look that placed her), 'nothing her search picks' (no live body in her search that way: within its first 50,
        behind her, past its 300, above or under its windows), or why the first band's were refused, 'no effect' or
        'likely dead' (her cast would pick one of them all the same)."""
        s = self._last
        if s is None or s.where is None or s.platform is None:
            return None, "nothing seen", None
        now = time.monotonic()
        live = [m for m in s.monsters if not self._dying(m, now, s.monsters) and not self._corpse(m, s.monsters)]
        path = in_path(s, side, live, self._bodies)
        if not path:
            dead = in_path(s, side, s.monsters, self._bodies)
            return None, "nothing her search picks", (dead[0] if dead else None)
        whys = []
        for m in path:
            if (why := self.refused(m)) is None:
                return m, "", None
            whys.append(why)
        return None, whys[0], path[0]

    def _corpse(self, m: Monster, seen: list[Monster]) -> bool:
        """(`picks`) Whether monster `m`, seen with `seen`, is the body of one of her marks a kill of hers was placed on
        within FADE_S (`_corpses`: drawn dying, 0.84 s a mushroom, then fading): its track where both have one; else of
        its kind on its platform within ledger.DYING_DX of where the mark had it, the nearest of those `seen` there (a
        live one beside it isn't). With the feed, by place too where its own track isn't seen and `m`'s was first seen
        after the kill (`_known`: the feed's new track for the fading body, as `_carry` takes one)."""
        from .ledger import DYING_DX
        for at, c in self._corpses:
            if (track := getattr(c, "track", None)) is not None and getattr(m, "track", None) is not None:
                if track == m.track:
                    return True
                if any(getattr(o, "track", None) == track for o in seen) \
                        or self._known.get(m.track, (at,))[0] < at:
                    continue  # (its own body seen, or `m` a track seen before the kill: another)
            if m.kind != c.kind or m.platform != c.platform or abs(m.x - c.x) >= DYING_DX:
                continue
            if all(abs(o.x - c.x) >= abs(m.x - c.x) for o in seen
                   if o is not m and o.kind == m.kind and o.platform == m.platform):
                return True
        return False

    def _judge(self, mk: Mark, again: Monster, s: Scene) -> None:
        """(`picks`, rule 3) Her casts at mark `mk` judged at look `s`, which sees it again as `again`: an effect
        (`_effect`) of any keyed SHOWS_S or more before it, as soon as it shows, clears those and proves the mark
        (`proven`: WAIT_FIRST); else those keyed LANDS_S to JUDGE_BY_S before it (one older unjudged is dropped), with
        its bar as before (`_none`), hold the mark (Hold: no cast at it until new evidence, `_lift`) and her spot that
        way (Spot: none that way from there at any body, `_lift_spots`; the review of 2026-10-29: on 18 the mark's hold
        went with the feed's track and with its walk, and her casts went on from where she stood)."""
        mk.casts = [c for c in mk.casts if s.t - c.t <= JUDGE_BY_S]
        shown = [c for c in mk.casts if s.t - c.t >= SHOWS_S]
        if any(self._effect(c, again, s) for c in shown):
            mk.proven = True
            mk.casts = [c for c in mk.casts if c.t > s.t - SHOWS_S]
            return
        due = [c for c in shown if s.t - c.t >= LANDS_S]
        if not due:
            return
        if not all(_none(c, again) for c in due):
            return  # (no bar to judge by: one hurt read none, or higher than its own was)
        c = max(due, key=lambda c: c.t)
        last = c.t
        mk.held = Hold(last, s.t, s.where.x, s.platform.id, abs(again.x - s.where.x), again.platform,
                       again.bar if _believable(again.bar) else None, c.side)
        self._spots[c.side] = Spot(c.side, c.platform, c.x, s.t, last)
        mk.casts = [c for c in mk.casts if c.t > last]
        self.counts["held: no effect"] += 1
        if self.counts["held: no effect"] in (1, 10, 100):
            self.log(f"    her cast at the {again.kind} {again.x - s.where.x:+.0f} took no effect (its bar "
                     f"{'none up' if again.bar is None else f'{again.bar:.0%}'} {s.t - last:.1f} s after): none at it "
                     f"until she or it moves {HOLD_MOVED:.0f} or its bar changes")

    def _effect(self, c: Cast, again: Monster, s: Scene) -> bool:
        """(`picks`) Whether look `s` shows an effect of cast `c` at the monster it sees as `again`, or at another the
        look it was chosen on saw that way (`c.others`: her search may have picked it): a bar PICK_BAR_STEP / 2 lower,
        or one up where none was; any gain of her EXP from SHOWS_S after the key (a kill, maybe another's); her attack's
        MISS popup over the target in the look's frame (Scene.misses, keycheck's box: MISS_DX either way, MISS_UP over
        its feet, MISS_DOWN under; MISS_FROM_S to MISS_BY_S after the key): her cast picked it, and missed. A cast that
        hurt anything isn't one that picked nothing."""
        if _hurt(c.bar, again.bar):
            return True
        for o, bar in c.others:
            if (now := seen_again(o, s.monsters)) is not None and now is not again and _hurt(bar, now.bar):
                return True
        if any(c.t + SHOWS_S <= g <= s.t for g in self._gains):
            return True
        if misses := (getattr(s, "misses", None) or ()):
            from .keycheck import MISS_BY_S, MISS_DOWN, MISS_DX, MISS_FROM_S, MISS_UP
            if c.t + MISS_FROM_S <= s.t <= c.t + MISS_BY_S and any(
                    abs(mx - again.x) <= MISS_DX and again.y - MISS_UP <= my <= again.y + MISS_DOWN
                    for mx, my in misses):
                return True
        return False

    def _lift(self, mk: Mark, again: Monster, s: Scene) -> None:
        """(`picks`, rule 3) A held mark's hold lifted at look `s`, which sees it again as `again`, by new evidence: its
        bar otherwise than at the hold (one up where none was, or PICK_BAR_STEP off it), her on another platform or
        HOLD_MOVED along hers from where the hold had her, it on another platform or HOLD_MOVED nearer or farther ahead
        of her than then."""
        h = mk.held
        if h is None:
            return
        bar = (again.bar is not None if h.bar is None
               else _believable(again.bar) and abs(again.bar - h.bar) > PICK_BAR_STEP)
        her = s.platform.id != h.platform or abs(s.where.x - h.x) >= HOLD_MOVED
        its = again.platform != h.its or abs(abs(again.x - s.where.x) - h.ahead) >= HOLD_MOVED
        if bar or her or its:
            mk.held, mk.proven = None, False
            self.counts["held: lifted"] += 1

    def _lift_spots(self, s: Scene, prev: Scene | None) -> None:
        """(`picks`, rule 3) Her spots' holds (Spot) lifted at look `s` (`prev`: the look that searched before it) by
        new evidence: her on another platform or HOLD_MOVED along hers from it; a gain of her EXP from SHOWS_S after the
        held cast's key on (a kill of hers, any), or a kill of hers placed that way since (its gain then that
        monster's, gone from `_gains`); a monster that way whose bar `s` reads lower than `prev` did, or up where it had
        none (`_hurt`: a bar shows her damage alone); SPOT_HELD_S since it was held (one cast tests it). Her cast the
        other way lifts it too (`_turned`). A monster walking, or another coming, isn't new evidence of her: on 18 the
        mark's hold was lifted by its walk of 20 (106-150 off her) and her casts went on."""
        kills = []
        if self._spots:  # (the kills since the oldest held cast: the ledger's are in time order)
            since = min(sp.cast for sp in self._spots.values())
            for k in reversed(self._kills()):
                if k.t < since:
                    break
                kills.append(k)
        for side, sp in list(self._spots.items()):
            moved = s.platform.id != sp.platform or abs(s.where.x - sp.x) >= HOLD_MOVED
            gained = any(g >= sp.cast + SHOWS_S for g in self._gains) or any(
                getattr(k, "placed", False) and k.t >= sp.cast and (k.x > sp.x) == (side == RIGHT) for k in kills)
            hurt = prev is not None and any(
                (now := seen_again(o, s.monsters)) is not None and (now.x > s.where.x) == (side == RIGHT)
                and _hurt(o.bar, now.bar) for o in prev.monsters)
            if moved or gained or hurt or s.t - sp.t >= SPOT_HELD_S:
                del self._spots[side]
                self.counts["held here: lifted"] += 1

    def _turned(self, side: str) -> None:
        """(`picks`, rule 3) Her cast that way (`side`) turns her: her holds the other way lifted, her spot's and her
        marks' (`proven` again to show), as her next cast there goes with the arrow tapped first as a rule
        (control.actions.Actions.throw: by her keys she faces this way then), new evidence where her keys had her
        facing wrong (live on 2026-10-29 her heading by keys disagreed with the frame 9 times in 17 minutes; on 18,
        placed by her dot, nothing but her keys told her heading)."""
        other = LEFT if side == RIGHT else RIGHT
        if self._spots.pop(other, None) is not None:
            self.counts["held here: lifted"] += 1
        for mk in self.marks:
            if mk.held is not None and mk.held.side == other:
                mk.held, mk.proven = None, False
                self.counts["held: lifted"] += 1

    def _carry(self, found: list, s: Scene) -> list:
        """(`picks`, rule 3) Her marks with casts waiting to be judged or a hold that look `s` doesn't see by the feed's
        track (`found`, seen_again's: None) carried to a body of their kind beside where they were (monsters.same, on
        its platform) under a track first seen since (`_known`), the nearest, one no other mark has: the feed's new
        track for a body it lost (under her hit's effect, 570 ms live). Else her hold went with the old track, and so
        did her casts' judging (the review of 2026-10-29: a held Green Mushroom seen again where it was under another
        track took a cast; live on 18 her marks began again at its whole HP at 01:55:17.0, 22.5, 27.1 and 40.2, the
        same one 116-150 ahead)."""
        taken = {id(a) for a in found if a is not None}
        out = list(found)
        for i, mk in enumerate(self.marks):
            track = getattr(mk.m, "track", None)
            if out[i] is not None or track is None or (not mk.casts and mk.held is None):
                continue
            new = [o for o in s.monsters if id(o) not in taken and o.platform == mk.m.platform and same(o, mk.m)
                   and (t := getattr(o, "track", None)) is not None and t != track
                   and self._known.get(t, (s.t,))[0] > mk.seen]
            if new:
                out[i] = min(new, key=lambda o: abs(o.x - mk.m.x))
                taken.add(id(out[i]))
                self.counts["carried: the feed's new track"] += 1
        return out

    def _seen_tracks(self, s: Scene) -> None:
        """(`picks`) The feed's tracks look `s` has, when first and last seen (`_known`), those unseen FORGET_S gone."""
        for o in s.monsters:
            if (t := getattr(o, "track", None)) is not None:
                self._known.setdefault(t, [s.t, s.t])[1] = s.t
        self._known = {t: v for t, v in self._known.items() if s.t - v[1] < FORGET_S}

    # --- at every throw --------------------------------------------------------------------------

    def key(self, side: str) -> str | None:
        """The attack key a throw that way goes out with now (Actions' attack key asks this): its choice logged. Its
        target: `aimed`, if set (and then cleared), else what a throw that way takes (target). The regular attack held
        (`watch`: behavior.keycheck), Lucky Seven while she has the MP for a cast, else None: no throw. Her skill with
        `picks` (the magician's): only at a body her search picks from where the last look had her (`_her_target`; no
        jump's `aimed`, she casts from the ground only), else None, no cast, its reason logged and counted ('no cast:
        ...') and sent as a telemetry 'no_cast' record."""
        now = time.monotonic()
        aimed, self.aimed = self.aimed, None
        seen = [] if self._last is None else self._last.monsters
        if self.picks:
            target, none_why, first = self._her_target(side)
            if target is None:
                self._cast, self._keyed, self._regular, self._target, self._asked = None, None, False, None, False
                self.no_cast(side, none_why, first, by="key")
                return None
        else:
            target = (aimed if aimed is not None and not self._dying(aimed, now, seen) and self.worth_a_throw(aimed)
                      else self.target(side))
            if target is None and self.forced is None and self._nothing_there(side):
                return self._no_throw(side)
        hp, pdd = self.stats.get(target.kind, (0.0, 0.0)) if target is not None else (0.0, 0.0)
        left = None if target is None else self.reckon(target)
        alive = None if left is None else left.alive() or left  # as it is if it's alive when the throw lands
        # Chosen by what it has left with the chance it's dead (choice.finishes on the whole reckoning: Lucky Seven's
        # throws saved count only if it's alive), not by what it has if alive alone: at one likely dead a cast's MP goes
        # for little. Until 2026-10-26 by the latter, the dead chance then mostly stale (KILL_SHOWN_S).
        what = None if target is None else float("inf") if left is None else left if DEAD_CHOICE else alive
        attack, why = choice.pick(what, pdd, self.mp, self.max_mp, self.skill, self.mode, self.regular)
        if attack is self.skill and target is not None and self.forced is None and self._regular_pays(target):
            attack, why = self.regular, "unseen"  # (no look has it: not worth a cast's MP, worth a regular attack's lock)
        skill_held = self.watch is not None and getattr(self.watch, "skill_held", False)
        if attack is self.skill and skill_held and self.forced is None:
            attack, why = self.regular, "Lucky Seven held"  # (out of darts due, its key's check put off: keycheck)
        if self.forced == "lucky_seven" and self.skill is not None:
            attack, why = self.skill, "her stars' test"
        elif attack is self.regular and self.watch is not None and self.watch.holds(target):
            if target is not None and self.skill is not None and not skill_held and (self.mp is None or choice.can_cast(
                    self.mp, self.max_mp, self.skill.mp)):
                attack, why = self.skill, "off" if self.watch.off is not None else "held"
            else:  # (nothing in its way: no cast spent at nothing; no Lucky Seven to throw instead: no throw)
                self._cast, self._keyed, self._regular, self._target, self._asked = None, None, False, None, False
                said = "held: nothing in its way" if target is None else "held: none thrown"
                self.counts[said] += 1
                if self.counts[said] in (1, 10, 100):
                    self.log("    no throw: the regular attack is " + ("off this run" if self.watch.off is not None
                                                                   else "held (its key doubted)") + " and "
                             + ("nothing is in its way (no Lucky Seven cast at nothing)" if target is None else
                                "there's no Lucky Seven to throw instead"
                                + ("" if self.skill is None else " (her MP under a cast)")))
                return None
        if attack is self.regular and self.no_regular:  # (the magician's wand: never at range, nor for MP)
            self._cast, self._keyed, self._regular, self._target, self._asked = None, None, False, None, False
            said = f"no cast: {why}"
            self.counts[said] += 1
            if self.counts[said] in (1, 10, 100):
                self.log(f"    no cast ({why}): her regular attack isn't thrown"
                         + (f" (her MP {self.mp * self.max_mp:.0f} under a cast's {self.skill.mp:g}: her pet's MP "
                            f"potions)" if why == "MP" and self.mp is not None and self.skill is not None else ""))
            if telemetry.recording():  # (every one, from 2026-11-02: the log says the 1st, 10th and 100th alone, so
                # how often her reckoned MP refused a cast couldn't be told, P27 of the recordings' analysis)
                telemetry.emit("no_cast", why=why, side=side, by="key", **_target_fields(target),
                               mp=None if self.mp is None else round(self.mp * self.max_mp),
                               cost=None if self.skill is None else self.skill.mp)
            return None
        self._target, self._regular, self._asked = target, attack is self.regular, True
        self._cast = attack.mp / self.max_mp if attack.mp > 0 and self.max_mp > 0 else None
        self._swipe = self._cast is not None and target is not None and self._clawed(target)
        self._ground = self.from_ground() if self.from_ground is not None else True
        self._keyed = None
        if target is not None and target.kind in self.stats and not self._dying(target, now, seen):
            mk = self._mark_of(target)
            if mk is None:
                mk = Mark(target, now, left, whole=not _believable(target.bar))
                if _believable(target.bar) and self._last is not None:
                    mk.read = (self._last.t, target.bar)  # (its bar as the look it was chosen on read it)
                self.marks.append(mk)
            mk.pending.append((now, attack))
            mk.keyed = mk.keyed[-15:] + [[now, attack]]
            mk.last = now
            self._keyed = (mk, len(mk.pending) - 1, len(mk.keyed) - 1)  # (its time its key's, once the throw says:
            # on_throw)
            if self.picks and self._cast is not None and (s := self._last) is not None:  # (judged: rule 3)
                if _believable(target.bar):
                    mk.bar = target.bar
                way = [o for o in s.monsters if o is not target and (o.x > s.where.x) == (side == RIGHT)]
                mk.casts.append(Cast(now, mk.bar, s.where.x, s.platform.id, abs(target.x - s.where.x),
                                     tuple((o, o.bar) for o in way), side))
                self._judged = mk
                self._turned(side)
            if not self.picks and self._missed(target):  # (no look has it: what this throw shows is judged)
                mk.blind.append(Blind(now, side, self._cast, self._ground, None if self._last is None else self._last.mp))
                self._blinded = mk
        self.counts[attack.key] += 1
        self.counts[why] += 1
        self._say(attack, why, target, left, alive, hp, side)
        self._landing(target)
        return getattr(self.keys, attack.key).name

    def on_throw(self, at: float, side: str) -> None:
        """A throw (Actions.throw_listeners), its attack key going down at `at`: the throw `key` chose for a target is
        pending at its key's time (a jump throw's attack key goes 40-460 ms after `key` is asked: KILL_SHOWN_S and
        LANDS_S count from it), and the look the next is decided on waits for its kill to show (`shown_by`); a cast
        `key` chose takes its MP off now (looks shown before it could show still read it: on_look). Its regular
        attack's press, or its cast at a target, told to the doubt's watch (`watch`)."""
        self.thrown = (at, self._target if self._asked else None)  # (its aim, told to the feed: Hunter._tell_feed)
        if self.watch is not None:
            if not self._asked:  # (the regular attack alone: her Actions keyed it, its target as a throw that way takes)
                self._watch_regular(at, self.target(side))
            elif self._regular:
                self._watch_regular(at, self._target)
            else:
                self.watch.cast(at, self._target)
        if self.casts is not None and self._asked:
            if self._regular:
                self.casts.cast(at, self._target)  # (the other attack: an effect it could have made isn't a cast's)
            elif not self._swipe and self.forced is None:  # (up close the key swipes; her stars' test's are its own)
                self._watch_regular(at, self._target, self.casts)
        self._asked, self._regular = False, False
        if self._blinded is not None:  # (... and her throw at one no look has)
            if self._blinded.blind:
                self._blinded.blind[-1].t = at
            self._blinded = None
        if self._judged is not None:  # (her cast judged from its key's time: rule 3)
            if self._judged.casts:
                self._judged.casts[-1].t = at
            self._judged = None
        if self._keyed is not None:
            mk, i, j = self._keyed
            self._keyed = None
            if j < len(mk.keyed):
                mk.keyed[j][0] = at
            if i < len(mk.pending):
                mk.pending[i] = (at, mk.pending[i][1])
                mk.last = at
                self._shows = at + (KILL_SHOWN_S or 0.0)
        if self._cast is None:
            if self.darts is not None:
                self.darts.regular(at)  # (the regular attack's: watched at a rope's top only, DartWatch.regular)
            return
        if self.mp is not None:
            self.mp = max(0.0, self.mp - self._cast)
        self._casts.append((at, self._cast))
        self._keyed_at = at  # (her no casts counted from it: no_cast)
        self._cast = None
        if self.darts is not None:
            self.darts.cast(at, swipe=self._swipe, ground=self._ground)

    def _watch_regular(self, at: float, m: Monster | None, w=None) -> None:
        """A regular press keyed at `at` at `m` told to the doubt's watch, with what decides whether it's judged: its
        kind's HP known, it not dying, a throw from the ground (`from_ground`), her not on a rope (`roped`), it in her
        regular attack's reach from where the last look had her (the watch's reach_ok), the monsters about that the look
        didn't find (`unseen`, and those the watch saw lately: RegularWatch.hidden) counted with its own (one of them
        nearer that way takes the attack: keycheck._in_front)."""
        s, w = self._last, self.watch if w is None else w
        ground = self.from_ground() if self.from_ground is not None else True
        roped = self.roped() if self.roped is not None else False
        seen = [] if s is None else s.monsters
        about = s
        if s is not None:
            more = ([] if self.unseen is None else list(self.unseen(s))) + w.hidden(at, s.monsters)
            if more:
                about = SimpleNamespace(where=s.where, platform=s.platform, monsters=s.monsters + more)
        ok = m is not None and s is not None and (w.reach_ok is None or w.reach_ok(about, m))
        w.pressed(at, m, known=m is not None and m.kind in self.stats,
                  dying=m is not None and self._dying(m, at, seen), ground=ground, roped=roped, reach_ok=ok)

    def target(self, side: str) -> Monster | None:
        """The monster a throw that way takes, as she knows it: the likeliest in its way in the last look (and those
        about that it didn't find: `unseen`) that isn't likely dead by her throws (DEAD_LIKELY; the likeliest if all
        are), else the one she last threw at that way (unseen: under its damage numbers; one a look still has, out of its
        way, within SEEN_MARK_S), one not likely dead first, else None; of those a look missed, only one a throw is worth
        (`worth_a_throw`), at any age; never one dying where a kill of hers was
        placed (ledger.dying, by all of them: an unseen one nearer the kill is taken for the dying one, not a live one the
        look found beside it). With `picks` (the magician's), `_her_target`'s: a body seen that her search picks, none
        about unseen, no fallback."""
        if self.picks:
            return self._her_target(side)[0]
        s = self._last
        if s is not None and s.where is not None and s.platform is not None:
            now = time.monotonic()
            about = s.monsters + ([] if self.unseen is None else list(self.unseen(s)))
            path = in_path(s, side, [m for m in about if not self._dying(m, now, about)
                                     and self.worth_a_throw(m)])  # (no star: dying; none she doesn't believe in)
            if path:
                return next((m for m in path if (left := self.reckon(m)) is None or left.dead < DEAD_LIKELY), path[0])
            x = s.where.x
            recent = [mk for mk in self.marks if (mk.m.x > x) == (side == RIGHT) and not self._dying(mk.m, now)
                      and self.worth_a_throw(mk.m)
                      and (now - mk.thrown < SEEN_MARK_S or (presence.EVIDENCE and self._missed(mk.m)))]
            alive = [mk for mk in recent if (left := self.reckon(mk.m)) is None or left.dead < DEAD_LIKELY]
            if recent:  # (as the path: one not likely dead first, the likeliest if all are)
                return max(alive or recent, key=lambda mk: mk.thrown).m
        return None

    def _missed(self, m: Monster) -> bool:
        """Whether no look has monster `m`: not of the last look that searched (a stand-in of it isn't), nor seen in the
        feed's newest frame (`newest`), by its track."""
        s = self._last
        if s is None or any(o is m for o in s.monsters):
            return False
        track = getattr(m, "track", None)
        if track is not None and any(getattr(o, "track", None) == track for o in s.monsters):
            return False
        if self._newest is not None and track is not None:
            t, tracks = self._newest
            return not any(tr.id == track and tr.last_seen >= t - GAIN_SAME_S for tr in tracks)
        return True

    def evidence(self, m: Monster, at: float | None = None) -> Evidence:
        """What's known of monster `m`, which no look has, for a throw keyed at `at` (the next she can: her lock, Hunter's
        `next_key`; presence.Evidence): its feed track as of the newest state told (`newest`) or the last look, her
        throws at it since it was seen (its mark), her gains lately."""
        s = self._last
        at = self._key_time() if at is None else at
        track = getattr(m, "track", None)
        tracks = self._newest[1] if self._newest is not None else (getattr(s, "tracks", None) or ())
        tr = None if track is None else next((q for q in tracks if q.id == track), None)
        mk = self._mark_of(m)
        if tr is not None:
            age, state, short, blind = (at - tr.last_seen, tr.state, tr.seen <= presence.SHORT_FRAMES,
                                        getattr(tr, "blind", 0))
        else:
            age = (at - mk.seen) if mk is not None else (at - s.t) if s is not None else 0.0
            state, short, blind = ("gone" if track is not None else None), False, 0  # (a track: there's a feed)
        gain = any(at - presence.GAIN_S <= g <= at and (kinds is None or m.kind in kinds) for g, kinds in self._told)
        spent = 0.0
        if mk is not None and mk.blank > 0:
            left = mk.left
            for _, a in mk.pending:
                left = left.after(a, self.miss)
            spent = left.dead
        return Evidence(age, state, 0 if mk is None else mk.blank, mk is not None and mk.shown, gain, short, spent, blind)

    def _key_time(self) -> float:
        """When her next throw's attack key can go: her lock's end, or now."""
        now = time.monotonic()
        return max(now, self.next_key()) if self.next_key is not None else now

    def chance(self, m: Monster) -> float | None:
        """How likely a throw at monster `m` shows an effect, by what's known now (presence.chance), or None: a look has
        it, as sure as a look makes it."""
        return presence.chance(self.evidence(m)) if self._missed(m) else None

    def seen_net(self, m: Monster) -> float:
        """What a throw at monster `m`, which a look has, is worth in EXP (presence.net at SEEN_CHANCE)."""
        return self._best_net(m, presence.SEEN_CHANCE)[0]

    def worth_a_throw(self, m: Monster, than: float | None = None) -> bool:
        """Whether a throw at monster `m` goes: one a look has, always; else its best attack's net (`_best_net`) over
        nothing, and over `than` if given (what a throw at a monster she sees would be worth, `seen_net`). Always with
        `picks` (the magician's casts have their own checks: `refused`)."""
        if self.picks or self.mode == "regular" or not presence.EVIDENCE:
            return True
        p = self.chance(m)
        if p is None:
            return True
        net = self._best_net(m, p)[0]
        return net >= 0.0 and (than is None or not presence.OVER_SEEN or net >= than)

    def _best_net(self, m: Monster, p: float) -> tuple[float, Attack | None]:
        """The attack she'd do best to throw at monster `m` with chance `p` of landing, and what that's worth in EXP
        (presence.net): Lucky Seven where she has the MP for a cast and her mode throws it, the regular attack unless
        never thrown (`no_regular`, her key held). What it has left as she reckons it, if alive; a kind whose HP isn't
        known, as much as a whole one of three regular stars."""
        left, stats = self.reckon(m), self.stats.get(m.kind)
        hp = (left.alive() or left).hp if left is not None else 3 * self.regular.mean()
        scale = 100.0 / (100.0 + stats[1]) if stats is not None else 1.0
        rate = presence_rate(self.rate)
        lock = presence.THROW_LOCK_S if self.from_ground is None or self.from_ground() else presence.JUMP_LOCK_S
        worth = self.worth.get(m.kind, self.worth.get(None, 20.0))
        out: tuple[float, Attack | None] = (-1e18, None)
        if self.skill is not None and self.mode != "regular" and (
                self.mp is None or choice.can_cast(self.mp, self.max_mp, self.skill.mp)):
            out = (presence.net(p, worth, hp, self.skill.mean() * scale, self.skill.mp, lock, rate), self.skill)
        if not self.no_regular and not (self.watch is not None and self.watch.holds(m)):
            net = presence.net(p, worth, hp, self.regular.mean() * scale, self.regular.mp, lock, rate)
            if net > out[0]:
                out = (net, self.regular)
        return out

    def _regular_pays(self, m: Monster) -> bool:
        """Whether a throw at monster `m`, which no look has, does better as the regular attack than as the cast her
        choice would make: at thin evidence a cast's MP is what's not worth it (presence.net)."""
        if self.picks or self.mode == "regular" or self.no_regular or not presence.EVIDENCE:
            return False
        p = self.chance(m)
        return p is not None and self._best_net(m, p)[1] is self.regular

    def _nothing_there(self, side: str) -> bool:
        """Whether a throw that way has nothing it could take (`key`'s no throw): no monster the last look saw that way
        in a star's reach, none a touch says is on her (`touched`). Always False with `picks` and in her modes that throw
        whatever: what's left is the keys of a throw at a monster she no longer believes in."""
        s = self._last
        if s is None or s.where is None or s.platform is None or self.picks or self.mode != "choice" \
                or not presence.EVIDENCE:
            return False
        if self.touched is not None and self.touched(s):
            return False
        x, y, now = s.where.x, s.where.y, time.monotonic()
        return not any((m.x > x) == (side == RIGHT) and not self._dying(m, now, s.monsters)
                       and reach.can_hit(s.platform, x, y, m, others=s.monsters) for m in s.monsters)

    def nothing_to_throw_at(self, side: str) -> bool:
        """(fight._cast_or_wait) Whether her throw that way would go with no key (`key`'s no throw): no target and nothing
        there."""
        return self.mode == "choice" and not self.picks and self.forced is None and self.target(side) is None \
            and self._nothing_there(side)

    def _no_throw(self, side: str) -> None:
        """`key`'s answer to a throw with nothing to take (`_nothing_there`): none, said the 1st, 10th and 100th time."""
        self._cast, self._keyed, self._regular, self._target, self._asked = None, None, False, None, False
        said = "no throw: nothing there"
        self.counts[said] += 1
        if self.counts[said] in (1, 10, 100):
            self.log(f"    no throw {side}: nothing a look has that way, none worth one unseen"
                     + ("" if self.counts[said] == 1 else f" ({self.counts[said]} times)"))
        if telemetry.recording():
            telemetry.emit("no_throw", side=side)
        return None

    def _judge_blind(self, found: list, s: Scene, prev: Scene | None) -> None:
        """Her throws at marks no look had, judged at look `s` (`found`: seen_again's for each mark; `prev`, the look
        that searched before it): the thief's rule 3 (`picks` has `_judge`). A mark seen again owes nothing, its count
        starts anew. Else each throw shows an effect as soon as a look says so (`_showed`), and LANDS_S after its key
        with none it's one that showed nothing (Mark.blank): what the next throw at that mark goes by."""
        for mk, again in zip(self.marks, found):
            if again is not None:
                mk.blind, mk.blank, mk.shown = [], 0, False
                continue
            if not mk.blind:
                continue
            for b in mk.blind:
                b.shown = b.shown or (s.t >= b.t + SHOWS_S and self._showed(b, mk, s, prev))
            for b in [b for b in mk.blind if b.shown or s.t - b.t >= LANDS_S]:
                mk.blank, mk.shown = (0, True) if b.shown else (mk.blank + 1, False)
                if not b.shown:
                    self.counts["unseen: showed nothing"] += 1
            mk.blind = [b for b in mk.blind if not b.shown and s.t - b.t < LANDS_S]

    def _showed(self, b: Blind, mk: Mark, s: Scene, prev: Scene | None) -> bool:
        """Whether look `s` shows an effect of throw `b` at mark `mk`: any gain of her EXP from SHOWS_S after its key (`_told`:
        a kill, maybe another's); a bar lower or newly up on a monster that way (`_hurt`: a bar shows her damage alone);
        her attack's MISS popup where the mark was (keycheck's box: she picked it, and missed); a cast from the ground
        with its MP not taken, her claw's swipe (a body there, MP_SHOWS_S on)."""
        if any(g >= b.t + SHOWS_S for g, _ in self._told):
            return True
        if prev is not None and any(
                (now := seen_again(o, s.monsters)) is not None and (now.x > s.where.x) == (b.side == RIGHT)
                and _hurt(o.bar, now.bar) for o in prev.monsters):
            return True
        if misses := (getattr(s, "misses", None) or ()):
            from .keycheck import MISS_BY_S, MISS_DOWN, MISS_DX, MISS_FROM_S, MISS_UP
            if b.t + MISS_FROM_S <= s.t <= b.t + MISS_BY_S and any(
                    abs(mx - mk.m.x) <= MISS_DX and mk.m.y - MISS_UP <= my <= mk.m.y + MISS_DOWN for mx, my in misses):
                return True
        return (b.cast is not None and b.ground and b.mp is not None and s.mp is not None and s.mp > 0.0
                and s.t >= b.t + MP_SHOWS_S and b.mp - s.mp < b.cast / 2)

    def _clawed(self, m: Monster) -> bool:
        """Whether the throw whose key is being chosen is one from the ground (`from_ground`) at monster `m` within the
        claw's reach of where she stood in the last look (reach.claw_chance, by its kind's body): the Lucky Seven key
        swipes it as the attack key does, taking no MP whatever her darts (behavior.darts: a swipe says nothing of
        them). Live a Lucky Seven press from the ground at one 20-90 ahead on her platform took MP 54 times in 200."""
        s = self._last
        if self.from_ground is None or not self.from_ground() or s is None or s.where is None:
            return False
        ahead = abs(m.x - s.where.x)
        top, half = reach.body_of(m.kind)  # (as ledger.in_path, the throw's path)
        return reach.claw_chance(ahead, m.y - s.where.y, top, half) > 0

    def in_way(self, side: str, kinds: set) -> str | None:
        """The kind of a monster a throw that way would take (`aimed`, its target, its path in the last look) that
        `kinds` names (None in it: any kind), or None: what drops a throw at its last look before the key (a kill of
        that kind come in since: Hunter._still)."""
        if not kinds:
            return None
        s = self._last
        ways = [m for m in (self.aimed, self.target(side)) if m is not None]
        if s is not None and s.where is not None and s.platform is not None:
            ways += in_path(s, side, s.monsters)
        return next((m.kind for m in ways if None in kinds or m.kind in kinds), None)

    def reckon(self, m: Monster) -> Left | None:
        """What monster `m` has left as she reckons it (see the module's docstring), her throws at it that no look
        has seen land yet taken off, and of those, what the ones whose kill would have shown by now (KILL_SHOWN_S, with
        no EXP gain GAIN_AFTER_KEY after its key) would have killed it with ruled out; None for a kind whose HP isn't
        known."""
        stats = self.stats.get(m.kind)
        if stats is None:
            return None
        mk = self._mark_of(m)
        if mk is None:
            return Left.of(m.bar * stats[0] if _believable(m.bar) else stats[0], stats[1])
        left = mk.left
        for t, a in mk.pending:
            left = left.after(a, self.miss)
            if self._unkilled(t, m.kind):
                left = left.alive() or left  # (none alive: her throws did less than reckoned, taken as nothing)
        return left

    def left(self, m: Monster) -> float:
        """The HP monster `m` has left on average as she reckons it if it's alive (`reckon`); inf for a kind whose HP
        isn't known."""
        left = self.reckon(m)
        if left is None:
            return float("inf")
        return (left.alive() or left).hp

    def kill_s(self, m: Monster) -> float:
        """s she takes to kill monster `m` from what it has left if alive (`reckon`), with the attack she'd throw
        (Lucky Seven while she has the MP for a cast in a mode that throws it), her misses (`miss`), her throws
        THROW_S apart."""
        left = self.reckon(m)
        attack = self.skill if (self.skill is not None and self.mode != "regular" and
                                (self.mp is None or choice.can_cast(self.mp, self.max_mp, self.skill.mp))) \
            else self.regular
        if left is None:  # an unknown kind: three regular stars
            return 3 * choice.REGULAR.mean() / attack.mean() / (1 - self.miss) * choice.THROW_S
        return (left.alive() or left).throws(attack) / (1 - self.miss) * choice.THROW_S

    def summary(self) -> str:
        """Her throws by attack and why, for the end of a run's log ("" without a choice asked: mode 'regular')."""
        c = self.counts
        if not c:
            return ""
        whys = ", ".join(f"{k} {c[k]}" for k in ("finish", "MP", "nothing in its way") if c[k])
        warned = c["MP 0 under the warning"]
        if self.skill is not None and self.skill.key != "lucky_seven":  # (the magician's: her skill, and her casts held)
            held = ", ".join(f"{k} {n}" for k, n in c.items() if k.startswith("no cast: "))
            judged = c["held: no effect"]
            learned = "" if self.book is None else self.book.said(self.skill.key)
            return (f"{self.skill.name} {c[self.skill.key]}, the regular attack {c['attack']}"
                    + (f" ({held})" if held else "")
                    + (f"; her damage {learned}" if learned else "")
                    + (f"; her casts that took no effect {judged} (holds lifted: her marks' {c['held: lifted']}, her "
                       f"spots' {c['held here: lifted']})" if judged else "")
                    + (f"; her MP bar's 0 under her pet's warning not believed {warned} times" if warned else ""))
        return (f"Lucky Seven {c['lucky_seven']}, the regular attack {c['attack']}" + (f" ({whys})" if whys else "")
                + (f"; her MP bar's 0 under her pet's warning not believed {warned} times" if warned else ""))

    # --- her damage learned (`book`: behavior.damage) -------------------------------------------------------------

    def _take_book(self) -> None:
        """Her skill's damage and her misses' share as the book has them now (its fit over her profile's prior)."""
        if self.book is None:
            return
        self._book_version = self.book.version
        if self._prior is not None:
            self.skill = self.book.attack(self._prior)
            self.miss = self.book.miss(self._prior.key, choice.MISS)

    def _since(self, mk: Mark) -> tuple[float, float, bool] | None:
        """Mark `mk`'s bar before (when read, its share, and whether one was read: `barred`) for what her throws since
        did: its last read (Mark.read); with none, full from before her first throw at it where she first threw at it
        with no bar up (Mark.whole); else None."""
        if mk.read is not None:
            return mk.read[0], mk.read[1], True
        if mk.whole:
            return -1e18, 1.0, False
        return None

    def _others_near(self, m: Monster, s: Scene) -> bool:
        """Whether look `s` has another player within OTHERS_DX along and OTHERS_DY up or down of monster `m`."""
        return any(abs(p.x - m.x) < OTHERS_DX and abs(p.y - m.y) < OTHERS_DY for p in getattr(s, "players", ()) or ())

    def _learn_bar(self, mk: Mark, again: Monster, s: Scene) -> None:
        """(`book`) What look `s`'s read of mark `mk`'s bar (`again`'s, between empty and full) says of the one throw of
        hers whose damage surely shows since the bar before and by now (keyed from SHOWS_S before that read to LANDS_S
        before this one), with none keyed where its damage may or may not have shown by either: its damage (the drop
        x its HP before its defence: damage.EXACT, a bar px either way), or nothing (damage.MISS, under MISS_PX px
        lower after a bar read before). None learned with another player near it, or another throw in that time."""
        stats = self.stats.get(again.kind)
        before = self._since(mk)
        if stats is None or before is None or self._others_near(again, s):
            return
        t0, b0, barred = before
        sure = [a for t, a in mk.keyed if t0 - SHOWS_S < t <= s.t - LANDS_S]
        unsure = [a for t, a in mk.keyed if t0 - LANDS_S < t <= t0 - SHOWS_S or s.t - LANDS_S < t <= s.t - SHOWS_S]
        if len(sure) != 1 or unsure:
            return
        hp, defence = stats
        k = (100.0 + defence) / 100.0
        tol = max(damage.TOL_MIN, hp * k / damage.BAR_PX)
        drop = b0 - again.bar
        a = sure[0]
        if drop * damage.BAR_PX < damage.MISS_PX:
            if barred and drop > -PICK_BAR_STEP:  # (higher by a step: another's bar or a misread, nothing learned)
                self.book.add(a.key, a.stars, damage.Read(time.time(), again.kind, damage.MISS, 0.0, tol,
                                                          self.her_level(), True))
            return
        self.book.add(a.key, a.stars, damage.Read(time.time(), again.kind, damage.EXACT, drop * hp * k, tol,
                                                  self.her_level(), barred))

    def _learn_kill(self, mk: Mark, k, s: Scene) -> None:
        """(`book`) A kill of hers placed on mark `mk` (`k`: the ledger's) after exactly one throw of hers at it since
        the bar before (keyed from SHOWS_S before that read to SHOWS_S before its gain), none keyed where its damage
        may or may not have shown by that read: that throw did at least what it had left (damage.AT_LEAST, a bar px
        under it). None learned with another player near it."""
        stats = self.stats.get(mk.m.kind)
        gained = getattr(k, "gained", None)
        gained = k.t if gained is None else gained
        before = self._since(mk)
        if stats is None or before is None or self._others_near(mk.m, s):
            return
        t0, b0, barred = before
        sure = [a for t, a in mk.keyed if t0 - SHOWS_S < t <= gained - SHOWS_S]
        unsure = [a for t, a in mk.keyed if t0 - LANDS_S < t <= t0 - SHOWS_S]
        if len(sure) != 1 or unsure:
            return
        hp, defence = stats
        f = (100.0 + defence) / 100.0
        tol = max(damage.TOL_MIN, hp * f / damage.BAR_PX)
        a = sure[0]
        self.book.add(a.key, a.stars, damage.Read(time.time(), mk.m.kind, damage.AT_LEAST, max(0.0, b0 * hp * f - tol),
                                                  tol, self.her_level(), barred))

    def save(self) -> None:
        """Her damage as learned written to her file (the book's; at a run's end)."""
        if self.book is not None:
            self.book.save()

    def _doubt(self, m: Monster | None) -> dict:
        """What her 'attack' record says of a target no look has (presence.Evidence): `chance`, its seconds since a
        sighting, her throws at it that showed nothing, the track's frames unseen by the exact finder. Nothing for one a
        look has, or with `picks`."""
        if m is None or self.picks or not self._missed(m):
            return {}
        e = self.evidence(m)
        return {"chance": round(presence.chance(e), 3), "unseen_s": round(e.age, 3), "blank": e.blank, "blind": e.blind,
                "state": e.state}

    def _dying(self, m: Monster, t: float, seen: list[Monster] = ()) -> bool:
        """Whether `m` is taken for dead or dying (ledger.dying: its track, or where a kill of hers was placed), at t,
        seen with `seen`."""
        return dying(self._kills(), m, t, seen, self._die, self._dead())

    def _mp_read(self, s: Scene) -> float | None:
        """Her MP bar's share as look `s` read it, or None: not read, or a 0 under her pet's warning (within WARNING_S
        of a read above 0, it or her MP as reckoned WARNING_STEPS px or less), counted once a run of them ('MP 0 under
        the warning': live the bar read 0 on up to 6 looks in a row, 10 a second)."""
        if s.mp is None:
            return None
        if s.mp > 0.0:
            self._read, self._warned = (s.t, s.mp), False
            return s.mp
        if self._read is not None and s.t - self._read[0] <= WARNING_S \
                and min(self._read[1], 1.0 if self.mp is None else self.mp) <= WARNING_STEPS / choice.MP_STEPS + 1e-9:
            if not self._warned:
                self.counts["MP 0 under the warning"] += 1
            self._warned = True
            return None
        self._warned = False
        return s.mp

    def _believed(self, read: float, t: float) -> float:
        """Her MP from bar share `read` off a frame shown at `t`: less her casts keyed after t - MP_FIRST_S (it can't
        show them yet), and her MP as reckoned where that's within what the casts keyed from t - MP_SHOWS_S to then
        (shown or not) leave of it; past those bounds, the bound (a drink, regen or a press that took no MP shows,
        less a cast at most; a cast her reckoning missed, at once)."""
        self._casts = [(at, mp) for at, mp in self._casts if at > t - MP_SHOWS_S]
        hi = max(0.0, read - sum(mp for at, mp in self._casts if at > t - MP_FIRST_S))
        lo = max(0.0, hi - sum(mp for at, mp in self._casts if at <= t - MP_FIRST_S))
        return hi if self.mp is None else min(max(self.mp, lo), hi)

    def _landing(self, target: Monster | None) -> None:
        """A throw at `target`: where UNLANDED_THROWS before it at that monster show nothing (no HP bar on it, no EXP
        since), the next throw taps the arrow (`retap`), and they're counted again from there."""
        track, gains = getattr(target, "track", None), 0 if self.gains is None else self.gains()
        last, n, at = self._unlanded
        if track is None or target.bar is not None:
            self._unlanded = (None, 0, gains)
        elif track != last or gains != at:
            self._unlanded = (track, 1, gains)
        elif n < UNLANDED_THROWS:
            self._unlanded = (track, n + 1, at)
        else:
            self._unlanded = (track, 0, at)
            if self.retap is not None:
                self.retap()
            self.log(f"    {n} throws at {target.kind} #{track} show nothing: the next one taps the arrow first")

    def _mark_of(self, m: Monster) -> Mark | None:
        return next((mk for mk in self.marks if mk.m is m or seen_again(mk.m, [m]) is m), None)

    def _say(self, attack: Attack, why: str, target: Monster | None, left: Left | None, alive: Left | None,
             hp: float, side: str | None = None) -> None:
        mp = None if self.mp is None else max(0.0, choice.mp_points(self.mp, self.max_mp))
        if target is None:
            what = "nothing seen in its way"
        elif left is None or alive is None:
            what = target.kind
        else:
            what = f"{target.kind} ~{alive.hp:.0f}/{hp:.0f} HP" + (f" if alive ({left.dead:.0%} dead)"
                                                                   if left.dead >= 0.05 else "")
        reason = {"MP": f"MP {mp:.0f} < a cast's {self.skill.mp:.0f}" if self.skill is not None and mp is not None
                  else "MP", "finish": "finishing it", "damage": "", "skill": "", "regular": "",
                  "nothing in its way": "", "held": "the regular attack held (its key doubted)",
                  "Lucky Seven held": "Lucky Seven held (out of darts due: its key's check put off)",
                  "off": "the regular attack off this run"}.get(why, why)
        self.log(f"    {attack.name}: {what}" + (f"; {reason}" if reason else "")
                 + ("" if mp is None or why == "MP" else f" (MP {mp:.0f})"))
        if telemetry.recording():  # (where its target was seen too: sim.harness counts the throws at the dead)
            # (which monster it went at and which way, the perception service's track of it among its fields, so a
            # live check of her kill's binding needs no reconstruction: _target_fields, a 'no_cast' record's too)
            telemetry.emit("attack", key=attack.key, why=why, **_target_fields(target),
                           left=None if alive is None else round(alive.hp),
                           dead=None if left is None else round(left.dead, 3), hp=hp,
                           mp=None if mp is None else round(mp), side=side, **self._doubt(target))


def presence_rate(rate: Callable[[], float] | None) -> float:
    """Her EXP a second for a throw's price: `rate`'s (the Hunter's: fight.made_rate), else the prior (fight.RATE_PRIOR)."""
    if rate is not None:
        return rate()
    from .fight import RATE_PRIOR
    return RATE_PRIOR


def _worth_of(model) -> dict:
    """Each kind's worth in EXP, for a landing's price (behavior.presence): its EXP and what its kill is expected to drop
    at the user's reference price (looting.expected_worth, presence.MESO_EXP); None's: the mean of the map's. {} where the
    map doesn't say (tests' stand-ins)."""
    exps = getattr(model, "mob_exp", None)
    if not exps:
        return {}
    try:
        value = worth_for(model)
    except (AttributeError, KeyError, TypeError):
        value = None
    out = {}
    for name, exp in exps.items():
        kind = kind_of(name)
        out[kind] = float(exp) + (presence.MESO_EXP * expected_worth(kind, value) if value is not None else 0.0)
    if out:
        out[None] = sum(out.values()) / len(out)
    return out


def _target_fields(m: Monster | None) -> dict:
    """A throw's target in its 'attack' record (`_say`), and a cast's not sent in its 'no_cast' record (one schema,
    from the integration of 2026-11-02: work/rec-records' records beside work/rec-replay's track): its kind, platform,
    x and the perception service's track (None each, no target)."""
    return {"kind": None if m is None else m.kind, "platform": None if m is None else m.platform,
            "x": None if m is None else round(m.x, 1), "track": None if m is None else getattr(m, "track", None)}


def _believable(bar: float | None) -> bool:
    """Whether an HP bar read is a share to go by: between empty and full (a bar is up only once she has hurt it, and
    one with no green, as a monster dies, reads None: perception.hud.hp_bar)."""
    return bar is not None and 0.0 < bar < 1.0


def _hurt(before: float | None, bar: float | None) -> bool:
    """(`Attacks.picks`) Whether a monster's bar `bar` as a look reads it shows damage since `before` (None: none up):
    one up where none was, or a believable one PICK_BAR_STEP / 2 lower."""
    if before is None:
        return bar is not None
    return _believable(bar) and bar < before - PICK_BAR_STEP / 2


def _none(c: Cast, again: Monster) -> bool:
    """(`Attacks.picks`, rule 3) Whether cast `c` took no effect on its target as a look sees it (`again`): still no
    bar up where it had none, or its bar believable and no more than PICK_BAR_STEP / 2 under the one before (nor
    PICK_BAR_STEP over it: another's bar). A hurt one whose bar a look reads none, as a dying one's, isn't judged."""
    if c.bar is None:
        return again.bar is None
    return _believable(again.bar) and c.bar - PICK_BAR_STEP / 2 <= again.bar <= c.bar + PICK_BAR_STEP
