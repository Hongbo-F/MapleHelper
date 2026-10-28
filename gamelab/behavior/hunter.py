"""The hunter: the agent that runs hunting objectives on its own (kill one monster, clear the monsters in
reach, and the tours built on them).

The decider (a person, or Claude) picks objectives; the hunter makes the small decisions in a loop:
look at the game, keep the target at the far end of a star's reach, face it, throw (the regular attack, or with
`attack` 'choice' Lucky Seven but to finish a monster and while her MP is under a cast: behavior.attacks), step away
from monsters that get close, loot each kill straight away when it fell on her platform (drops elsewhere are
remembered for the tour to collect), drink a potion when HP gets low, and hand control back when the target is gone,
nothing is left in reach, or the time is up. Her HP staying low (her pet's potions not keeping her up) ends the hunt:
the safety stop (behavior.safety). Every key goes through the gamelab player, so the Lie Detector watchdog stops
everything the moment the game's anti-bot check appears.

It's made of the layers below: Senses (perception and localization: every look), Actions (control:
her keys), a Looter (what she picks up and remembers), and the fight loops (behavior.fight). After every
look, whoever took it, the hunter keeps count of her EXP and her HP: a kill is her EXP going up, its number read
when it changes (perception.exp_number), and the gain's size says what kind it was, or that it was several kills,
or none (localization.gains). It also reads which way she faces off the look's frame (perception.heading) for her
keys (Actions.see), and logs where that tells the other way from her last arrow. And every look's monsters go into her
memory of them (localization.memory: what she has seen anywhere, kept across looks), her kills marking the ones they
were.
"""

import copy
import json
import random
import time
from collections import Counter, deque
from collections.abc import Callable

import numpy as np

from maplehelper.config import GameKeys

from .. import character as profiles
from .. import paths, telemetry
from ..attack import danger
from ..attack.reach import LOOK_DOWN, LOOK_UP
from ..control import client
from ..control.actions import Actions
from ..control.mover import STILL, Mover
from ..control.navigator import Navigator, rope_at_top
from ..localization.gains import ExpMeter, Gain
from ..localization.mapdata import MapModel
from ..localization.memory import Memory
from ..localization.monsters import Monster
from ..localization.pose import Where
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..localization.worldstate import HP_GAP_S, reading_of
from ..motion import physics
from ..perception import drops, exp_number, hp_number
from ..perception.eyes import Eyes
from ..perception.heading import Reading
from ..perception.hud import exp_changed
from . import fight
from .attacks import Attacks
from .darts import DartWatch, presses_for
from .fight import Report
from .keycheck import KeyCheck, KeyDoubt, RegularWatch, regular_reach, unquiet
from .ledger import DropStore, Ledger, Wounds
from .looting import Looter, worth_for
from .pet import PetCare
from .safety import HpWatch, TooStrong, Touches, kind_of_touch, reserve_of

EXP_MEMORY = 30     # s her EXP gains are kept (exp_events), for kill() to match to its target
POTION_EVERY = 3    # s at least between two potions
# Her pet (it drinks her potions since 2026-10-23; a hungry pet stops) is fed by its fullness (behavior.pet), read once
# (at a moment no monster is within PET_CLEAR of her: the read takes a few s) and followed in time from there. Until
# 2026-10-24 it was fed blind every 10 minutes, and only while hunting: found at fullness 9.
PET_CLEAR = 250  # map units
HP_ZERO = 0.02        # an HP read this low ...
HP_DOUBT_AFTER = 0.3  # ... right after one this high isn't believed at once (_doubt_hp)
HP_ZERO_SHOTS = 5     # frames of such looks saved, to find out why
HIT_EXACT_S = 1.0     # s: a hit's text has her HP number's touch if it read one this soon before (touches came 1.38 s
                      # apart at the least: behavior.safety)
HP_NUMBER_WAIT_S = 10.0  # s of looks with her HP number never read: said once (the safety stop on her bar then, at her
                         # profile's max HP: the perception service not restarted since 2026-10-28, a client it can't
                         # read at, such as the stretched 1920x1080, where it reads none)
# The safety stop raised at the look or glance it's due at, whoever took it (Senses.checks and glance_checks:
# Hunter._stop_now, _glanced), not only at the next health_check: a mover's settle, a throw's looks and a chain's
# glances ask none, and in the simulator (the review of 2026-10-28: D:/MapleHelper-scratch/safety3/review-safety,
# stops2.py) it came up to 1.4 s late that way, a touch landing in that time in 3 of 42 runs with her pet out of potions
# (one from 183 HP to 61, dead 0.3 s into her retreat). Now 0 s late in all 82 runs of the review's two sets
# (D:/MapleHelper-scratch/safety3/fix/stops3.py); deaths on her way out after it about as many (18 and 7, against 17 and
# 9): what she has left at the stop is what tells (behavior.safety). From her first health_check (a hunt asking after
# her health: `asked`), and not while she retreats (`retreating`: the Lie Detector's way to a safe place isn't cut short
# by it). False: at health_check only, as until then.
STOP_ON_LOOK = True
# s a touch's knock may still be moving her after the look that saw it take HP: it carries her through the air
# (0.22 s in the simulator) and she slides on as she lands (180 units/s braked at 800: 0.22 s more). A jump keyed
# from a look in that time comes while she's in the air (and is lost) or still sliding, from where she no longer is.
KNOCK_S = 0.45
ROPE_SLACK = 12  # map units: off her platform, within this of a rope's x (her dot is good to 11) she's on it, where
                 # only her back shows and her facing isn't read
HEADING_SHOTS_MAX = 200  # crops of her kept in lab/heading where the frame and her last arrow disagreed, at most
# A throw's way is told to the feed as her keys made it (the stance's side: Actions.throw_listeners), and the feed's
# tracker binds the kill its gain tells of to what her attack took that way (localization.tracking._bind). A look whose
# frame shows her facing the other way from her keys, strongly (Actions.see: 'told otherwise'; heading.Reading.strong:
# none of the frames read the wrong way on 2026-10-24 was, and the 5 below were) tells again that way the throws
# keyed since her last arrow, her last hit and the last look whose frame agreed with her keys, RETELL_S back at most
# (`_retell`): nothing has turned her since that anyone saw, and she threw them the way she faces now (her keys' tap
# swallowed). A turn without her keys (her player's own arrow, the simulator's Noise.turned) comes after a look that
# agreed, so the throws before it aren't told again. On the recording of 2026-10-26 00:20 two of 15 kills were bound to
# a live monster on the side she hadn't thrown at so (review of 2026-10-27, D:/MapleHelper-scratch/kplace/review-truth);
# over the four recordings of 2026-10-25/26, 5 of the 25 frames logged so ('heading') came after throws with no arrow
# nor hit between them (the others after an arrow), 6 throws.
RETELL_S = 2.5
TOLD_KEEP = 12  # her last throws told to the feed, kept for `_retell`
# A throw's last look before its key (`_still`: Actions.still), once her lock has been waited out: an EXP gain naming a
# kind in its way shown since the look it was decided on drops it, and the next look decides again. Live on 2026-10-25
# (the hunt of 23:43) she decided each throw on a look taken 0.2-0.34 s after the key before (190 of 229 back-to-back
# pairs had one look between them), then waited out her lock blind: the look a throw was decided on was a median 0.57
# s old at its key, a kill's gain shows 0.15-0.46 s after the key that made it, and 17 throws went out after a gain
# that showed before their key, 6 more before the hunter had read it (21 casts of Lucky Seven: analysis-1025/
# jumps-perception-skeptic/pair.py, looks_between.py, wasted2.py). False: every throw goes as decided.
LAST_LOOK = True


def kinds_of(gains: list[Gain]) -> frozenset | None:
    """The kinds of kill a change of her EXP number can be, by the gains it was (localization.gains): those their sizes
    named, or None (any) where one named none, the number couldn't be read yet (no gain), or no kill of the map's
    explains it (a share of a kill another player hurt too: her throw's target's maybe)."""
    if not gains or any(g.odd or None in g.kinds for g in gains):
        return None
    return frozenset(k for g in gains for k in g.kinds)


class Hunter:
    """The agent: her senses, her keys and her looter, and what she did (`report`); runs the fights."""

    def __init__(self, eyes: Eyes, keys: GameKeys, model: MapModel, *, log: Callable[[str], None] = print,
                 low_hp: float | None = None, potion_hp: float = 0.5, skill: bool = False, feed=None,
                 her: danger.Her | None = None, attack: str | None = None):
        """`low_hp`: the safety stop's threshold (behavior.safety: her HP under it ends the hunt; 1 or under a share of
        her max HP, over 1 HP; None: her reserve, her profile's hits of the map's largest touch, under `potion_hp`'s with
        a potion key of her own); `potion_hp`: her potion key's (drink_if_low; none bound, her pet drinks them). `feed`:
        what her looks and glances come from (localization.feed.Feed: the perception service's client live,
        localization.service; SyncFeed in the simulator; None: the step on the eyes' own frames, feed.InlineFeed), told
        of her throws. `her`: her HP, level and DEF, what a touch is measured against (attack.danger, the safety stop's
        reserve; her profile's stats by default, config.toml [character]'s for the thief: her HP her max HP until the
        HUD's number is read). `attack`: which attack each throw is (behavior.attacks.MODES: 'choice', Lucky Seven but
        to finish a monster and while her MP is under a cast; 'skill', Lucky Seven while she has the MP; 'regular';
        None: 'skill' with `skill`, else 'regular'), with her MP and Lucky Seven's level from config.toml
        [character]."""
        self.model, self.log = model, log
        c = physics.CHARACTER
        self.her = her or danger.Her(c.hp, c.level, c.defence)
        self.danger = danger.of_map(model, self.her)  # where touches would cost her
        self.potion_hp = potion_hp
        # The safety stop (behavior.safety): her reserve on this map, her profile's hits of its largest touch at her DEF
        # and level (the level her EXP number says, once read: _level_check), her own touches as her HP number measured
        # them kept (`touches`, her file); every look's HP watched against it (`hp_watch`), in HP: her HP number's, else
        # her bar's at her max HP (the HUD's number's, else her profile's: `max_hp`).
        profile = profiles.active()
        self.safety_hits, self.pet_warning = profile.safety_hits, profile.pet.hp_warning
        # Her Magic Guard's share of a touch (profile.magic_guard) isn't taken off the reserve: it's a toggle, and
        # nothing reads whether it's up (a cast lost, a press that put it off), so a touch may take all of itself.
        self.guard = 0.0
        self.max_mp = c.mp
        self.level = self.her.level      # her level as last known (her profile's, then her EXP number's)
        self.max_hp_read: int | None = None  # her max HP as the HUD's number says it (None: not read yet)
        self.hp_follow = hp_number.Follow()  # her HP number off her own looks (without a feed: the feed's step reads it)
        self._hp_box_t: float | None = None  # ... the last look it was seen on
        self.touches = Touches(paths.of().touches)  # (hers)
        self._last_touch: tuple[float, int, int] | None = None  # her HP number's last touch: when, HP before, after
        self._first_look: float | None = None  # her first look's time, and whether her HP number not read since ...
        self._unread_said = False              # ... HP_NUMBER_WAIT_S has been said
        self.asked = False       # a hunt has asked after her health (health_check): the stop raised at any look since
        self.retreating = False  # ... but not while she retreats (behavior.retreat.away)
        own = potion_hp if keys.hp_potion is not None else None  # (her own potion key's: under it, the stop's lower)
        self.hp_watch = HpWatch(self._reserve(model), potion_hp=own, low_hp=low_hp, max_hp=self.max_hp)
        log(f"safety stop: {self.hp_watch.describe()}")
        self.senses = Senses(eyes, model, band=(LOOK_UP, LOOK_DOWN), feed=feed)
        self.senses.log = eyes.log = log  # looks that failed their checks, said in her log (localization.senses)
        self.senses.checks.append(self._stop_now)  # (the safety stop at the look it's due at: STOP_ON_LOOK) ...
        self.senses.glance_checks.append(self._glanced)  # ... and at a glance (a chain's watch: no look for seconds)
        mode = attack or ("skill" if skill else "regular")
        self._attack_mode = mode
        self.actions = Actions(keys, skill=mode != "regular")
        self.senses.keyed = lambda: self.actions.last_key  # a look sees what came of her keys (Senses.fresh_after)
        self.worth = worth_for(model)  # what drops are worth on this map (looting.Worth)
        if note := drops.unknown_note(model):  # its monsters whose drops aren't known, said once
            log(note)
        self.looter = Looter(self)
        self.report = Report()  # what she did (a tour starts a new one at every stop)
        # Her kills her EXP told of: when, and the kind its gain's size named (None: not known). The last EXP_MEMORY s
        # of them, for kill() to match to its target (`take_exp`), and the latest few for the fights and the ledger to
        # place (`told`).
        self.exp_events: list[tuple[float, str | None]] = []
        self.told: deque[tuple[float, str | None]] = deque(maxlen=64)
        self.exp_gains = 0  # kills her EXP told of, ever (reports are replaced; this isn't)
        self.meter = ExpMeter(model.mob_exp)  # her EXP as read off its number, and what each gain was
        self._exp: np.ndarray | None = None  # the last look's EXP number (text pixels)
        self._last_hp: float | None = None
        self._hp_doubted = False  # the last look's HP read 0 and wasn't believed (_doubt_hp)
        self._hp_zero_shots = 0
        self._last_where: Where | None = None
        self.hit_at: float | None = None  # the time of the last look that saw her HP drop (a touch)
        self.knock_waited: float | None = None  # ... the hit_at a jump or step last waited out (fight.knocked_move)
        self.aim = None  # her running attack's target, for her pace's hold (fight.Aim: throw_act's _note_aim, pace_of)
        self._last_potion = 0.0
        self.pet = PetCare(self.actions, eyes, log=log)
        # The hunt's bookkeeping from every look and throw (behavior.ledger): the hunt loop reads the drop store,
        # and the fights and the hunt loop her wounds (the monsters she has hurt, finished before she moves on).
        self.drops = DropStore(self.worth)
        self.ledger = Ledger(self, self.drops)
        self.wounds = Wounds(self.ledger)
        # The monsters she has seen anywhere, kept across looks (localization.memory), her kills (the ledger's) marking
        # the ones they were: after the ledger, so a look's kills are in it.
        self.memory = Memory(model, kills=lambda: self.ledger.kills, dead=lambda: self.ledger.dead,
                             theirs=lambda: self.ledger.theirs)
        # Her attack a throw (behavior.attacks): every throw's key is its choice, but in mode 'regular'.
        self.attacks = Attacks(model, keys, mode, max_mp=c.mp, level=c.lucky_seven, log=log,
                               kills=lambda: self.ledger.kills, dead=lambda: self.ledger.dead)
        self.attacks.from_ground = lambda: getattr(self.actions, "keyed_ground", False)
        if mode != "regular":
            self.actions.choose = self.attacks.key
            self.senses.shows = self.attacks.shown_by  # (the look a throw is decided on has the last one's kill)
        # Out of darts (behavior.darts): her Lucky Seven presses and MP bar (the attack choice's), her kills (_gained)
        self.darts = DartWatch(presses_for(model.mob_hp, getattr(model, "mob_pdd", {})))
        self.attacks.darts = self.darts
        # The doubt that her regular attack's key isn't the game's (behavior.keycheck): her regular presses judged by
        # what her looks see of their targets, every mode's (the attack choice's, and the regular attack's alone); on
        # a rope or at its top not judged. In the hunt loop (`doubts_raise`, Hunt.run) a doubt raises KeyDoubt there
        # and the loop runs the check (check_keys); elsewhere her regular presses are held (`hold_regular`).
        self.attacks.watch = RegularWatch(reach_ok=regular_reach)
        self.attacks.roped = lambda: self.darts.roped or self.darts.at_top
        self.doubts_raise = False
        self.key_check: KeyCheck | None = None  # (made at the first check)
        self.refused_dot = None  # her dot (minimap px) at the game's last refusal of her presses on a rope ...
        self.off_tries = 0       # ... and the tries since to get her off it with her dot there
        if LAST_LOOK:
            self.actions.still = self._still
        self._players: dict[int, object] = {}  # the other players on the map as of the last look, by id
        self.facing_seen = Reading(None, why="no look")  # which way the last look's frame showed her facing
        self.told_throws: deque[list] = deque(maxlen=TOLD_KEEP)  # her throws told to the feed: [key, way] (`_retell`)
        self.facing_agreed = -1e18  # ... and the last look whose frame showed her facing as her keys had her
        # Her facing off the frames: looks, told or not (and why), what her keys made of each (Actions.saw), and those
        # that told the other way from her last arrow ('against')
        self.heading_counts: Counter[str] = Counter()
        # The jump throws her stances chose for what their attack reaches, by why and the jump's kind (fight.log_jump)
        self.jump_counts: Counter[str] = Counter()
        shots = paths.of().heading_shots  # (hers)
        self._shots = len(list(shots.glob("*.png"))) if shots.exists() else 0
        self.senses.listeners += [self._vitals, self._heading, self._others, self._publish, self.ledger.on_look,
                                  self.drops.on_look, self.wounds.on_look, self.memory.on_look, self.attacks.on_look]
        self.actions.throw_listeners += [self.ledger.on_throw, self.wounds.on_throw, self.memory.on_throw,
                                         self.attacks.on_throw,  # (a cast's MP)
                                         self._tell_feed]  # (the feed's: whose kills are whose, tracking)
        self._feed_throw = self.senses.feed.throw  # (the feed she was made with, as the listener was before)

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False) -> Scene:
        """A look (Senses.look); drops are looked for as her inventory allows (only mesos when it's full). Not with
        the Lie Detector alarm up (client.LieDetectorAlarm): a hunt that only looks for a while (waiting, standing by)
        stops there too, as at its next key plan, and retreats (behavior.retreat)."""
        client.check_alarm()
        return self.senses.look(mobs, wide, loot, mesos_only=self.looter.inventory_full)

    def clear(self, kinds=None, **kw) -> Report:
        """Fight the monsters in reach (behavior.fight.clear)."""
        return fight.clear(self, kinds, **kw)

    def kill(self, target: Monster, seconds: float = 30) -> str:
        """Shoot one monster dead and loot it (behavior.fight.kill)."""
        return fight.kill(self, target, seconds)

    def navigator(self, fighting: bool = False, rng: random.Random | None = None) -> Navigator:
        """A navigator for her: `fighting`, she shoots monsters on the way (hunting tours) instead of
        avoiding them; its fighting trips keep clear of the monsters her memory has too (Navigator.remembered)."""
        return Navigator(self.senses, self.actions, self.model, log=self.log, health=self.health_check,
                         escort=fight.FightOnTheWay(self) if fighting else None, rng=rng, memory=self.memory)

    def knocked(self, s: Scene) -> bool:
        """Whether a touch's knock may still be moving her at look `s`: a look within KNOCK_S before it (or it)
        saw her HP drop."""
        return self.hit_at is not None and 0 <= s.t - self.hit_at < KNOCK_S

    @property
    def max_hp(self) -> float:
        """Her max HP: the HUD's number's, once read; her profile's until then (`her`)."""
        return float(self.max_hp_read) if self.max_hp_read is not None else float(self.her.hp)

    def _reserve(self, model: MapModel | None = None):
        """Her safety stop's reserve on the map (behavior.safety.reserve_of) at her level as last known."""
        return reserve_of(model or self.model, self.safety_hits, self.her.defence, self.level, self.touches.rows,
                          self.guard)

    def too_strong(self) -> TooStrong | None:
        """Whether the map's monsters are too strong for her (behavior.safety.TooStrong): with no `--low-hp`, her
        reserve at her max HP (her MP full) at or over her pet's warning there, where her pet keeps her HP: the stop would
        end the hunt at once. A command that hunts raises it before it hunts (gamelab.__main__)."""
        w, m = self.hp_watch, self.max_hp
        if w.low_hp is not None:
            return None
        th = w.threshold(m, self.max_mp)
        if th < self.pet_warning * m:
            return None
        return TooStrong(th, m, self.pet_warning, w.words(m, self.max_mp))

    def health_check(self, s: Scene) -> None:
        """Asked at every look of a fight, a trip and a pass of the hunt loop: her potions and her pet's food
        (drink_if_low), and the safety stop: her HP under her reserve on two looks DEBOUNCE_S apart, or under one touch
        two looks in a row (`hp_watch`, which every look feeds) raises behavior.safety.SafetyStop, once. Until 2026-10-26
        under `low_hp` (30% in a tour) it stopped the fight at once, "HP low", and the tour or the hunt loop sent her to
        rest on the chair; until 2026-10-28 the stop was 65% for 2 s over 3 looks. Then out of
        darts (behavior.darts: `darts`, which her attack choice feeds): her Lucky Seven presses taking no MP and
        killing nothing for long raise OutOfDarts, once, with what the pickup log says then; a stretch of them long
        enough has the log read (DartWatch.asks: if the game says she's out, the stop comes sooner). A stretch of 3
        and more has the message bar read (DartWatch.asks_chat), and the game's refusal there starts it again
        (_hear_chat): on a rope, she's got off it. From the first, the safety stop is raised at any look it's due at
        too (`_stop_now`)."""
        self.asked = True
        self.drink_if_low(s)
        if (stop := self.hp_watch.due()) is not None:
            raise stop
        if self.darts.asks_chat():
            self._hear_chat(s)
        if self.darts.asks():
            said = self._messages()
            self.darts.hear(said)
            lines = " / ".join(line for line in said.splitlines() if line.strip()) or "nothing"
            self.log(f"  {self.darts.shown} Lucky Seven presses in a row took no MP: the pickup log reads {lines}")
        if (stop := self.darts.due(time.monotonic())) is not None:
            if stop.said is None:
                stop.heard(self._messages())
            raise stop
        watch = getattr(getattr(self, "attacks", None), "watch", None)  # (her regular attack's key: behavior.keycheck)
        if watch is not None and (doubt := watch.due()) is not None:
            self._doubted(doubt)
        if (watch is not None and watch.deferred is not None and self.doubts_raise and unquiet(self, s) is None
                and (doubt := watch.again(time.monotonic())) is not None):
            self.log("  the key check put off: a quiet moment now")
            raise KeyDoubt(doubt)  # (a check put off: tried again at a quiet look, keycheck.KeyCheck._defer)

    # --- her regular attack's key (behavior.keycheck) --------------------------------------------------------------

    def _doubted(self, doubt) -> None:
        """Her regular attack took no effect DOUBT_PRESSES times in a row (keycheck.RegularWatch.due): said in capitals
        and sent as a telemetry event 'key_doubt'; her regular presses held; in the hunt loop (`doubts_raise`) KeyDoubt
        raised for it to run the check (check_keys), elsewhere whoever is at the PC paged to run `gamelab keys --read`."""
        text = doubt.text() + ": checking her stars, then the game's key bindings"
        self.log(text)
        telemetry.emit("event", kind="key_doubt", text=text, presses=len(doubt.presses), number=doubt.number)
        self.hold_regular()
        if self.doubts_raise:
            raise KeyDoubt(doubt)
        self.log("  (not in the hunt loop: no check here; her regular presses held, Lucky Seven only, for the rest of "
                 "this run)")
        from ..control import alarm
        alarm.page("MapleStory: the attack key", f"gamelab: {doubt.text().lower()}. Its regular presses are held. "
                   "Check the game's Key Bindings (\\), or run `python -m gamelab keys --read`.")

    def check_keys(self, doubt):
        """The check after a doubt (keycheck.KeyCheck.run: her stars, a place, the game's key bindings, what then); its
        outcome. Raises OutOfDarts (her stars' test) and client.LieDetectorAlarm as any of her plans."""
        if self.key_check is None:
            self.key_check = KeyCheck(self)
        return self.key_check.run(doubt)

    def hold_regular(self, on: bool = True) -> None:
        """Her regular presses held (keycheck.RegularWatch.held), or let go again (`on` False); every throw's key asked
        of her attack choice from now on (Actions.routed: with the regular attack alone, or no Lucky Seven key, her
        Actions keyed it themselves), which throws Lucky Seven instead, or nothing."""
        self.attacks.watch.held = on
        self._route()

    def hold_at(self, targets) -> None:
        """No more regular attacks at `targets` (keycheck: a check found the key right, or none ran), in every attack
        mode (her attack choice asked at every throw: _route); her regular presses let go elsewhere."""
        self.attacks.watch.no_regular_at.extend(targets)
        self.attacks.watch.held = False
        self._route()

    def regular_off(self, why: str) -> None:
        """No more regular attacks this run (keycheck: the game has Attack on no key, or the window wasn't read):
        Lucky Seven only, or nothing."""
        self.attacks.watch.off = why
        self.attacks.watch.held = False
        self._route()

    def rekey(self, key: str) -> None:
        """Her regular attack on `key` (config.toml's name) for the rest of the run (keycheck: the game has Attack
        there): her Actions' keys (shared by the navigator, the mover and her pet's care; her retreat's moves are made
        from them as they're needed) and her attack choice's; her regular presses let go again."""
        self.actions.rekey(key)
        self.attacks.keys = self.actions.keys
        self.attacks.watch.held = False
        self._route()

    def _route(self) -> None:
        """Every throw's key asked of her attack choice once anything holds her regular attack (Actions.routed)."""
        w = self.attacks.watch
        holds = bool(w.held or w.off is not None or w.no_regular_at)
        self.actions.routed = holds
        if holds:
            self.actions.choose = self.attacks.key
        elif self._attack_mode == "regular":  # (as it was: her Actions key the regular attack themselves)
            self.actions.choose = None

    def _stop_now(self, s: Scene) -> None:
        """After every look's listeners (Senses.checks; STOP_ON_LOOK): the safety stop raised there if it's due (the
        watch saw the look already: _vitals), once a hunt has asked after her health (`asked`) and not while she
        retreats (`retreating`)."""
        if STOP_ON_LOOK and self.asked and not self.retreating and (stop := self.hp_watch.due()) is not None:
            raise stop

    def _glanced(self, st) -> None:
        """A glance's state (Senses.glance_checks: a chain's watch, glancing at her every 30 ms or so, where no look
        comes for up to seconds: the review of 2026-10-28 had a stop's first look under her reserve come after a
        second touch): her HP to the safety stop's watch as a look's (her HP number's as the state has it, else its
        bar's), and the stop raised there if it's due (`_stop_now`). An HP of HP_ZERO or less right after one of
        HP_DOUBT_AFTER or more isn't taken from a glance (_doubt_hp has the next look tell)."""
        hp, exact = getattr(st, "hp", None), getattr(st, "hp_of", None)
        if hp is None:
            return
        if exact is None and hp <= HP_ZERO and (self.hp_watch.share or 0.0) >= HP_DOUBT_AFTER:
            return
        if exact is not None:
            self._max_hp_seen(exact[1])
        mp = getattr(st, "mp", None)
        self.hp_watch.see(st.t, hp, self.max_hp, None if exact is None else exact[0],
                          None if mp is None else mp * self.max_mp)
        self._stop_now(st)

    def _hear_chat(self, s: Scene) -> None:
        """The message bar read for out of darts (DartWatch.asks_chat): the game's refusal of her presses on it starts
        the stretch again (DartWatch.hear_chat), logged and sent as a telemetry event 'refused'; on a rope ("That
        cannot be used while you're on a rope or ladder."), she's got off it (_off_rope). Her dot where it was at the
        rope's last (`refused_dot`, STILL: she hangs there yet) has it, still up, start the stretch again and her got
        off again (`off_tries`: down the rope every second time, up the first). A line that says she has no stars:
        logged ('no_darts_said'), the stop to come at READ_EVERY presses."""
        n, top = self.darts.shown, self.darts.top_presses
        if (line := self._chat_line()) is None:
            return
        dot, was = None if s is None or s.where is None else s.where.dot, self.refused_dot
        still = dot is not None and was is not None and abs(dot[0] - was[0]) <= STILL and abs(dot[1] - was[1]) <= STILL
        if (why := self.darts.hear_chat(line, line.t or time.monotonic(), still=still)) is None:
            return
        if why == "darts":
            text = f"the message bar says she's out of throwing stars: \"{line.text}\" ({n} in a row that took no MP)"
            self.log(f"  {text}")
            telemetry.emit("event", kind="no_darts_said", text=text)
            return
        self.off_tries = self.off_tries + 1 if still and why == "rope" else 0
        self.refused_dot = dot if why == "rope" else None
        text = (f"the game refused her presses: \"{line.text}\" ("
                + (f"{n} in a row that took no MP: not counted" if n or not top
                   else f"{top} in a row at a rope's top, the regular attack's or the claw's")
                + ")" + (f"; her dot where it was then: try {self.off_tries + 1}" if self.off_tries else ""))
        self.log(f"  {text}")
        telemetry.emit("event", kind="refused", text=text)
        if why == "rope":
            self._off_rope(s, down=self.off_tries % 2 == 1)

    def _chat_line(self):
        """The message bar as the eyes read it now (Eyes.chat_line); None without it (eyes that don't read it), or if
        the read fails (said)."""
        read = getattr(self.senses.eyes, "chat_line", None)
        if read is None:
            return None
        try:
            return read()
        except Exception as e:  # (the game window gone, the service not answering)
            self.log(f"  the message bar not read ({type(e).__name__}: {e})")
            return None

    def _off_rope(self, s: Scene, down: bool = False) -> None:
        """The game says she's on a rope: off it (control.mover.Mover.off_rope; `down`, down it, the way up having
        done nothing), the rope at the top of which look `s` has her standing (her dot snapped onto the platform:
        control.navigator.rope_at_top), or the one she's on (`s` on no platform). Placed on a platform away from any
        rope's top: nothing (a line left up from before)."""
        if s is None or s.where is None:
            return
        rope, at = None, None
        if s.where.platform is not None:
            rope = rope_at_top(self.model, s.where)
            if rope is None:
                self.log(f"  ... but she's on platform {s.where.platform.id} at x {s.where.x:.0f}, by no rope's top: "
                         "not acted on")
                return
            if s.where.dot is not None:
                at = self.senses.localizer.to_map(s.where.dot)
        self.log(f"  on a rope at x {(rope.x if rope else s.where.x):.0f} by the game's word: getting off"
                 + (" down it" if down else ""))
        Mover(self.senses, self.actions, self.model, log=self.log).off_rope(s, rope, at, down=down)

    def _messages(self) -> str:
        """The pickup log as the eyes read it now (Eyes.messages, OCR: "" without it, or if the read fails)."""
        try:
            return self.senses.eyes.messages()
        except Exception as e:  # (the game window gone, the service not answering)
            self.log(f"  the pickup log not read ({type(e).__name__}: {e})")
            return ""

    def _doubt_hp(self, s: Scene) -> None:
        """An HP of HP_ZERO or less a look after HP_DOUBT_AFTER or more isn't believed until the next look reads it too:
        on Tree Dungeon on 2026-10-24 her HP read 0% three times in two minutes right after 82-87%, the bar's row with
        no red at all, while the game showed 592/752; each sent her off to rest. The look keeps the HP before it, and its
        frame is saved (lab/shots/hp0_<time>.png, the first HP_ZERO_SHOTS) to find out why."""
        if s.hp > HP_ZERO:
            self._hp_doubted = False
            return
        if self._last_hp is None or self._last_hp < HP_DOUBT_AFTER or self._hp_doubted:
            return
        self._hp_doubted = True
        self.log(f"HP read {s.hp:.0%} right after {self._last_hp:.0%}: not believed until the next look reads it too")
        if self._hp_zero_shots < HP_ZERO_SHOTS and s.frame is not None:
            self._hp_zero_shots += 1
            try:
                from PIL import Image
                (paths.LAB / "shots").mkdir(parents=True, exist_ok=True)
                Image.fromarray(s.frame).save(paths.LAB / "shots" / f"hp0_{time.strftime('%H%M%S')}.png")
            except Exception:
                pass
        s.hp = self._last_hp

    def drink_if_low(self, s: Scene) -> None:
        """Drink an HP potion below `potion_hp` (one every POTION_EVERY s at most), and feed her pet when it's due
        (feed_pet_if_due). With no potion key (her pet drinks them since 2026-10-23), nothing: it logged "drinking a
        potion" at every look under `potion_hp` and drank none (in the simulator on West Rocky Mountain II, the hunt
        loop from 40% HP with her pet out of potions: 83 times in 3 minutes)."""
        self.feed_pet_if_due(s)
        if self.actions.keys.hp_potion is None:
            return
        if s.hp < self.potion_hp and time.monotonic() - self._last_potion > POTION_EVERY:
            self.log(f"HP {s.hp:.0%}: drinking a potion")
            if self.actions.drink():
                self._last_potion = time.monotonic()
                self.report.potions += 1
                telemetry.emit("event", kind="potion", text=f"potion at {s.hp:.0%} HP")

    def feed_pet_if_due(self, s: Scene | None = None) -> bool:
        """Follow her pet's fullness and give it a food when it's low (behavior.pet.PetCare.tend); never read yet, it's
        read first, but only with no monster within PET_CLEAR of her in look `s` (the read takes a few s). True if
        its fullness is known. Nothing without a pet food key. Not with the Lie Detector alarm up: her pet's care is the
        hunt's, which stops there (client.LieDetectorAlarm; the pet babysitter, `pet --every`, goes on feeding it)."""
        client.check_alarm()
        clear = s is None or s.where is None or not any(
            abs(m.x - s.where.x) < PET_CLEAR and abs(m.y - s.where.y) < 120 for m in s.monsters)
        return self.pet.tend(read_ok=clear) is not None

    def take_exp(self, since: float, kind: str | None = None) -> bool:
        """Use up one kill her EXP told of since `since`, if there is one: of `kind`, or of a kind not known (a gain
        whose size named another kind is another monster's death)."""
        for i, (t, k) in enumerate(self.exp_events):
            if t >= since and (kind is None or k is None or k == kind):
                del self.exp_events[i]
                return True
        return False

    def kinds_told(self, n: int) -> list[str | None]:
        """The kinds of the last `n` kills her EXP told of (None where not known), the oldest first."""
        last = list(self.told)[-n:] if n > 0 else []
        return [None] * (n - len(last)) + [k for _, k in last]

    def gains_told(self, n: int) -> list[tuple[float | None, str | None]]:
        """The last `n` kills her EXP told of, the oldest first: when each showed (None where not known) and its kind
        (None where not known)."""
        last = list(self.told)[-n:] if n > 0 else []
        return [(None, None)] * (n - len(last)) + last

    def _still(self, side: str) -> bool:
        """Whether a throw that way still goes, asked by her Actions once her lock has been waited out (LAST_LOOK): not
        if her EXP has gained a kill of a kind in its way since the look it was decided on. Read off the feed's newest
        state, if of a frame after her keys' effect (Senses.fresh_after; none, the throw goes: nothing is asked of the
        feed, which takes its frames as it would): its events not drained yet (the next look's), their
        EXP readings told apart as the hunter's meter would (a copy of it: the kinds each gain names; one it can't
        read names none, any kind then). Dropped, it's logged and sent as a telemetry event 'dropped'. Its MP bar goes
        to the attack choice first (Attacks.glance: a pet's drink shown since the look)."""
        feed = self.senses.feed
        st = feed.newest(self.senses.fresh_after())
        if st is None:
            return True
        self.attacks.glance(st.t, getattr(st, "mp", None))  # (her MP too: a drink shown since the look)
        events = [e for e in feed.pending_events(st.t) if e.kind in ("exp_gain", "exp_read")]
        if not any(e.kind == "exp_gain" for e in events):
            self.attacks.heard(st.t, [])  # (and her EXP: no kill since the look, Attacks.heard)
            return True
        meter, kinds, heard = copy.deepcopy(self.meter), set(), []
        for e in events:
            gains = meter.update(e.t, reading_of(e.data), e.kind == "exp_gain")
            for g in gains:
                kinds.update(g.kinds)
            if e.kind == "exp_gain":
                heard.append((e.t, kinds_of(gains)))
        self.attacks.heard(st.t, heard)
        if not kinds and meter.unread > self.meter.unread:  # a change it couldn't read: a kill of a kind not known
            kinds = {None}
        if (kind := self.attacks.in_way(side, kinds)) is None:
            return True
        text = f"a kill came in before the key ({kind}): the throw {side} is dropped, decided again"
        self.log(f"    {text}")
        telemetry.emit("event", kind="dropped", text=text)
        return False

    # --- after every look ---------------------------------------------------------------------

    def _vitals(self, s: Scene) -> None:
        """Her EXP number changed since the last look: read, the kills of hers its gain was (one if it can't be
        read; none if no kill explains it: logged). Her HP dropped: a hit (logged with how far it pushed her when no
        key was pressed). With a feed, from what it saw frame by frame (s.events, with the number as read then): two
        gains between two looks are two gains, and the HP lost is every drop's, a potion drunk between them or not."""
        self._doubt_hp(s)
        exact = self._hp_number(s)
        self._unread(s, exact)
        mp = getattr(s, "mp", None)
        self.hp_watch.see(s.t, s.hp, self.max_hp, None if exact is None else exact[0],
                          None if mp is None else mp * self.max_mp)
        self.darts.roped = self.on_rope(s)  # (presses keyed on a rope, which the game refuses, aren't counted)
        self.darts.at_top = (s.where is not None and s.where.platform is not None  # (her regular presses there watched)
                             and rope_at_top(self.model, s.where) is not None)
        if s.events is None:
            reads = []
            changed = exp_changed(self._exp, s.exp)
            if changed or self.meter.wants_reading(s.t):
                reads.append((s.t, exp_number.read(s.exp_box) if s.exp_box is not None else None, changed))
            lost = 0.0 if self._last_hp is None or s.hp >= self._last_hp else self._last_hp - s.hp
        else:
            reads = [(e.t, reading_of(e.data), e.kind == "exp_gain") for e in s.events
                     if e.kind in ("exp_gain", "exp_read")]
            lost = sum(e.data.get("lost", 0.0) for e in s.events if e.kind == "hp_drop")
        heard = []  # her EXP's changes, and the kills each can be: what her throws didn't kill (Attacks.heard)
        for t, reading, changed in reads:
            gains = self.meter.update(t, reading, changed)
            for g in gains:
                self._gained(g, s)
            if changed:
                heard.append((t, kinds_of(gains)))
        self.attacks.heard(s.t, heard)
        for g in self.meter.due(s.t):  # changes it couldn't read that no reading has explained: a kill each
            self._gained(g, s)
        for note in self.meter.notes:
            self.log(f"  {note}")
            telemetry.emit("event", kind="exp", text=note)
        self.meter.notes.clear()
        self.exp_events = [e for e in self.exp_events if s.t - e[0] < EXP_MEMORY]
        self._exp = s.exp
        self._level_check()
        if lost > 0 and self._last_hp is not None:
            self.report.damage_taken += lost
            self.hit_at = s.t
            if (watch := getattr(self.attacks, "watch", None)) is not None:
                watch.touched(s.t)  # (her regular presses about it aren't judged: behavior.keycheck)
            touch = self._last_touch
            text = (f"hit: HP {touch[1]} -> {touch[2]} (-{touch[1] - touch[2]})"
                    if touch is not None and 0 <= s.t - touch[0] < HIT_EXACT_S
                    else f"hit: HP {self._last_hp:.0%} -> {s.hp:.0%}")
            telemetry.emit("event", kind="hit", text=text)
            self._log_hit(s)
        self._last_hp, self._last_where, self.actions.pressed = s.hp, s.where, False

    def _hp_number(self, s: Scene) -> tuple[int, int] | None:
        """Her HP number at look `s`: (cur, max) if it was read at its last change (with a feed, the state's: Scene.hp_of,
        its step reading it at every change; without, read here off the look's box when it changed: `hp_follow`), else
        None. Each read since the last look says her max HP (_max_hp_seen), and each drop from the read before with
        nothing unread between is a touch, exactly (_touched)."""
        reads = []  # (t, cur, max, HP the read before said: None, not known)
        events, box = getattr(s, "events", None), getattr(s, "hp_box", None)
        if events is not None:
            reads = [(e.t, e.data["cur"], e.data["max"], e.data.get("before")) for e in events
                     if e.kind == "hp_read" and "cur" in e.data]
            exact = getattr(s, "hp_of", None)
        elif box is not None:
            if self._hp_box_t is not None and s.t - self._hp_box_t > HP_GAP_S:
                self.hp_follow.see(None)  # (her looks' gap: what it did meanwhile isn't known, as the feed's step has it)
            self._hp_box_t = s.t
            seen = self.hp_follow.see(box, s.geo)
            if seen is not None and seen[0].cur is not None:
                reads.append((s.t, seen[0].cur, seen[0].max, seen[1]))
            exact = self.hp_follow.of
        else:
            exact = getattr(s, "hp_of", None)
        for t, cur, most, before in reads:
            self._max_hp_seen(most)
            if before is not None and cur < before:
                self._touched(t, before, cur, most)
        if exact is not None:
            self._max_hp_seen(exact[1])
        return exact

    def _unread(self, s: Scene, exact: tuple[int, int] | None) -> None:
        """Her HP number not read on any look for HP_NUMBER_WAIT_S since her first: said once (the safety stop goes by
        her bar at her max HP as known then)."""
        if self._unread_said:
            return
        if exact is not None or self.max_hp_read is not None:
            self._unread_said = True  # (read: nothing to say, ever)
            return
        if self._first_look is None:
            self._first_look = s.t
        if s.t - self._first_look < HP_NUMBER_WAIT_S:
            return
        self._unread_said = True
        self.log(f"  her HP number not read in {HP_NUMBER_WAIT_S:g} s of looks (the perception service not restarted "
                 f"since it reads it? a client size it reads none at?): the safety stop goes by her HP bar at her "
                 f"max HP {self.max_hp:g}")

    def _max_hp_seen(self, most: int) -> None:
        """Her HP number said her max HP is `most`: hers from now on (said when it's new, and once if her profile says
        otherwise: config.toml [character] hp for the thief)."""
        if most <= 0 or most == self.max_hp_read:
            return
        was, self.max_hp_read = self.max_hp_read, most
        if was is not None:
            self.log(f"  her max HP {most} by the HUD's number (was {was})")
        elif most != self.her.hp:
            self.log(f"  her max HP {most} by the HUD's number; her profile's (config.toml [character] hp) is "
                     f"{self.her.hp:g}: stale")

    def _touched(self, t: float, before: int, cur: int, most: int) -> None:
        """Her HP number dropped from `before` to `cur` at t: a touch of that size, its kind told by its size
        (behavior.safety.kind_of_touch), kept in her touches (`touches`) at her level as her EXP number says it (None
        until it's read: counted at none). One that raises its kind's touch in her reserve (MARGIN over it: one is
        enough, as a measured touch only ever raises it) has the reserve made again at once, said; one that fits no
        kind of the map's is said too (counted for none)."""
        size = before - cur
        self._last_touch = (t, before, cur)
        level = None if self.meter.at is None else self.meter.at[0]
        kind = kind_of_touch(size, self.model, self.her.defence, self.level)
        self.touches.add({"t": round(time.time(), 3), "map": self.model.id,
                          "level": None if level is None else int(level), "hp": size, "kind": kind, "max": most})
        was = self.hp_watch.reserve
        if kind is None:
            if was.touch is None:  # (a map whose monsters' touches the game data doesn't give: none fits any)
                return
            text = (f"a touch of {size} HP ({before} -> {cur}) fits no kind of the map's: kept, counted for none (the "
                    f"safety stop's {was.touch.said} touch {was.touch.hp})")
        elif level is None or int(level) != int(self.level):
            return
        else:
            now = self._reserve()
            if now.hp(self.max_hp) <= was.hp(self.max_hp):
                return
            self.hp_watch.reserve = now
            text = (f"a touch of {size} HP ({before} -> {cur}, a {kind}'s) raises the safety stop: now "
                    f"{self.hp_watch.describe(self.max_hp)}")
        self.log(f"  {text}")
        telemetry.emit("event", kind="touch", text=text)

    def _level_check(self) -> None:
        """Her EXP number says another level than the one her reserve is for: the reserve made again at it (her
        touches measured at that level, the formula at it), and said."""
        level = None if self.meter.at is None else self.meter.at[0]
        if level is None or level == self.level:
            return
        was, self.level = self.level, level
        self.hp_watch.reserve = self._reserve()
        self.log(f"  her level {level:g} by her EXP number (was {was:g}): the safety stop now "
                 f"{self.hp_watch.describe(self.max_hp)}")

    def _heading(self, s: Scene) -> None:
        """Which way her look's frame shows her facing (Eyes.heading; not asked on a rope), for her keys (Actions.see:
        where it's sure and new enough it's her heading, which her throws' taps go by; after a hit, not sure makes it
        unknown). Where her keys leave no doubt, the look teaches her pictures (Eyes.learn_heading). A frame telling
        the other way from her last arrow is logged as an event 'heading' with how long after the arrow and the last
        hit, and her crop kept in lab/heading (HEADING_SHOTS_MAX at most): how often a knock or anything else turns her
        without a key."""
        a, eyes = self.actions, self.senses.eyes
        if not hasattr(eyes, "heading"):  # (eyes that don't read her picture: the facing is her keys')
            return
        rope = self.on_rope(s)
        reading = Reading(None, why="rope") if rope else eyes.heading(s)
        self.facing_seen = reading
        c = self.heading_counts
        c["looks"] += 1
        c["told" if reading.side is not None else f"not told ({reading.why})"] += 1
        if (sure := a.keys_say(s.t, self.hit_at)) is not None and not rope:
            eyes.learn_heading(s, sure, reading)
        against = a.see(reading.side, s.t, self.hit_at, reading.strong)
        c[a.saw] += 1
        if a.saw == "told otherwise" and reading.strong:
            self._retell(s, reading.side)
        elif a.saw == "as it was":
            self.facing_agreed = s.t
        if against is None:
            return
        c["against"] += 1
        since_arrow = s.t - a.arrows_at
        since_hit = None if self.hit_at is None else s.t - self.hit_at
        text = (f"the frame shows her facing {reading.side}, her last arrow {against} ({since_arrow:.1f} s after it, "
                + ("no hit before" if since_hit is None else f"{since_hit:.1f} s after a hit")
                + f"; matched {reading.score:.2f}, {reading.margin:.2f} ahead of {against})")
        self.log(f"  {text}")
        shot = None
        if self._shots < HEADING_SHOTS_MAX:
            shot = paths.of().heading_shots / f"{time.strftime('%Y%m%d-%H%M%S')}-{c['against']}-keys-{against}.png"
            if eyes.keep_heading(s, shot):
                self._shots += 1
            else:
                shot = None
        telemetry.emit("event", kind="heading", text=text, seen=reading.record(), keys=against,
                       since_arrow=round(since_arrow, 2), since_hit=None if since_hit is None else round(since_hit, 2),
                       **({"shot": shot.name} if shot is not None else {}))

    def _tell_feed(self, when: float, side: str) -> None:
        """A throw (Actions.throw_listeners) told to the feed (localization.tracking: whose kills are whose)."""
        self.told_throws.append([when, side])
        self._feed_throw(when, side)

    def _retell(self, s: Scene, side: str) -> None:
        """Look `s`'s frame shows her facing `side`, otherwise than her keys had her (Actions.see): the throws told
        the other way keyed since her last arrow, her last hit and the last look that agreed with her keys (and RETELL_S
        before the frame at most) are told to the feed again that way (localization.tracking: their kills bound on that
        side)."""
        since = max(self.actions.arrows_at, -1e18 if self.hit_at is None else self.hit_at, self.facing_agreed,
                    s.t - RETELL_S)
        for th in self.told_throws:
            if since < th[0] <= s.t and th[1] != side:
                th[1] = side
                self._feed_throw(th[0], side)
                self.log(f"  her throw of {s.t - th[0]:.2f} s before this look told again: toward {side}, as the "
                         f"frame shows her")
                telemetry.emit("event", kind="retold", text=f"throw told again toward {side}", key=round(th[0], 3),
                               side=side)

    def on_rope(self, s: Scene) -> bool:
        """Whether look `s` has her on a rope: off the platforms, within ROPE_SLACK of a rope's x, between its ends."""
        return s.where is not None and s.where.platform is None and any(
            abs(r.x - s.where.x) <= ROPE_SLACK and r.top - ROPE_SLACK <= s.where.y <= r.bottom + ROPE_SLACK
            for r in self.model.ropes)

    def _gained(self, g: Gain, s: Scene) -> None:
        """A gain of her EXP: its kills counted (exp_events, told, the report), each at its time, or, if no kill
        explains it, logged; with her EXP as read, for the mini-map and the run's record (telemetry 'exp')."""
        for i, kind in enumerate(g.kinds):
            self.darts.kill(g.when(i))  # (her presses did something: out of darts' stretch starts again)
            self.exp_events.append((g.when(i), kind))
            self.told.append((g.when(i), kind))
            self.report.exp_kills += 1
            self.exp_gains += 1
        if g.odd:
            text = ("EXP read at a level that doesn't follow from the last: not counted" if g.exp is None else
                    f"EXP down {-g.exp} (a death?)" if g.exp < 0 else
                    f"EXP +{g.exp}: no kill of this map's monsters gives that (a party share, a quest?)")
            self.log(f"  {text}")
            telemetry.emit("event", kind="exp", text=text)
        if telemetry.recording() and self.meter.at is not None:
            level, exp = self.meter.at
            rate = self.meter.per_minute(s.t)
            telemetry.emit("exp", exp=exp, level=level, gain=g.exp, kinds=list(g.kinds), total=self.meter.total,
                           per_minute=None if rate is None else round(rate, 1))

    def _others(self, s: Scene) -> None:
        """Other players coming onto the map (their minimap dots, confirmed: localization.players) and leaving it,
        logged and sent to the mini-map."""
        now = {p.id: p for p in s.players}
        for pid, p in now.items():
            if pid not in self._players:
                text = f"another player ({p.kind}) on the map: platform {p.platform} at x {p.x:.0f}"
                self.log(f"  {text}")
                telemetry.emit("event", kind="player", text=text)
        for pid, p in self._players.items():
            if pid not in now:
                text = f"another player ({p.kind}) gone from the minimap (last on platform {p.platform} at x {p.x:.0f})"
                self.log(f"  {text}")
                telemetry.emit("event", kind="player", text=text)
        self._players = now

    def _publish(self, s: Scene) -> None:
        """The look and the drops she remembers, for the mini-map (telemetry); with a feed, the deaths it saw that
        weren't hers ('died', as `watch` publishes them: analysis.respawn reads them as others' kills, a platform
        emptied by someone else; hers are the 'kill' events already)."""
        if telemetry.recording():
            record = s.record()
            if record["her"] is not None:  # which way she faces as best known (keys or frame: facing_by), by her keys
                a = self.actions             # alone, and by the frame alone (and whether strongly: only a strong
                # frame takes a throw's tap away, Actions.faces)
                record["her"].update(facing=a.heading, facing_by=a.facing_by, facing_keys=a.facing,
                                     facing_seen=self.facing_seen.side, facing_strong=self.facing_seen.strong)
            telemetry.emit("scene", **record)
            telemetry.emit("drops", inventory_full=self.looter.inventory_full, pending=[
                {"x": round(d.x, 1), "y": round(d.y, 1), "platform": d.platform, "mesos": d.mesos}
                for d in self.looter.pending])
            for e in s.events or ():
                if e.kind == "died" and e.data.get("cause") != "hers":
                    telemetry.emit("event", kind="died", text=e.text(), track=e.track)

    def _log_hit(self, s: Scene) -> None:
        """A hit while no key was pressed since the last look: how far it pushed her (knockback), kept
        in lab/hits.jsonl."""
        before = self._last_where
        if self.actions.pressed or before is None or s.where is None or before.platform is None:
            return
        pushed = s.where.x - before.x
        same = s.platform is not None and s.platform.id == before.platform.id
        self.log(f"  hit standing still: pushed {pushed:+.0f} units" + ("" if same else
                 f", off platform {before.platform.id} onto {s.platform.id if s.platform else 'nothing'}"))
        try:
            with paths.of().hit_log.open("a", encoding="utf-8") as f:  # (hers)
                f.write(json.dumps({"pushed": round(pushed, 1), "dy": round(s.where.y - before.y, 1),
                                    "hp": round(self._last_hp - s.hp, 4), "how": s.how,
                                    "platform": before.platform.id, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
