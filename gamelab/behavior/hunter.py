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

from .. import paths, telemetry
from ..attack import danger
from ..attack.reach import LOOK_DOWN, LOOK_UP
from ..control import client
from ..control.actions import Actions
from ..control.navigator import Navigator
from ..localization.gains import ExpMeter, Gain
from ..localization.mapdata import MapModel
from ..localization.memory import Memory
from ..localization.monsters import Monster
from ..localization.pose import Where
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..localization.worldstate import reading_of
from ..motion import physics
from ..perception import drops, exp_number
from ..perception.eyes import Eyes
from ..perception.heading import Reading
from ..perception.hud import exp_changed
from . import fight
from .attacks import Attacks
from .darts import DartWatch, presses_for
from .fight import Report
from .ledger import DropStore, Ledger, Wounds
from .looting import Looter, worth_for
from .pet import PetCare
from .safety import HpWatch, threshold

EXP_MEMORY = 30     # s her EXP gains are kept (exp_events), for kill() to match to its target
POTION_EVERY = 3    # s at least between two potions
# Her pet (it drinks her potions since 2026-10-23; a hungry pet stops) is fed by its fullness (behavior.pet), read once
# (at a moment no monster is within PET_CLEAR of her: the read takes a few s) and followed in time from there. Until
# 2026-10-24 it was fed blind every 10 minutes, and only while hunting: found at fullness 9.
PET_CLEAR = 250  # map units
HP_ZERO = 0.02        # an HP read this low ...
HP_DOUBT_AFTER = 0.3  # ... right after one this high isn't believed at once (_doubt_hp)
HP_ZERO_SHOTS = 5     # frames of such looks saved, to find out why
# s a touch's knock may still be moving her after the look that saw it take HP: it carries her through the air
# (0.22 s in the simulator) and she slides on as she lands (180 units/s braked at 800: 0.22 s more). A jump keyed
# from a look in that time comes while she's in the air (and is lost) or still sliding, from where she no longer is.
KNOCK_S = 0.45
ROPE_SLACK = 12  # map units: off her platform, within this of a rope's x (her dot is good to 11) she's on it, where
                 # only her back shows and her facing isn't read
HEADING_SHOTS_MAX = 200  # crops of her kept in lab/heading where the frame and her last arrow disagreed, at most
# A throw's last look before its key (`_still`: Actions.still), once her lock has been waited out: an EXP gain naming a
# kind in its way shown since the look it was decided on drops it, and the next look decides again. Live on 2026-10-25
# (the hunt of 23:43) she decided each throw on a look taken 0.2-0.34 s after the key before (190 of 229 back-to-back
# pairs had one look between them), then waited out her lock blind: the look a throw was decided on was a median 0.57
# s old at its key, a kill's gain shows 0.15-0.46 s after the key that made it, and 17 throws went out after a gain
# that showed before their key, 6 more before the hunter had read it (21 casts of Lucky Seven: analysis-1025/
# jumps-perception-skeptic/pair.py, looks_between.py, wasted2.py). False: every throw goes as decided.
LAST_LOOK = True


class Hunter:
    """The agent: her senses, her keys and her looter, and what she did (`report`); runs the fights."""

    def __init__(self, eyes: Eyes, keys: GameKeys, model: MapModel, *, log: Callable[[str], None] = print,
                 low_hp: float | None = None, potion_hp: float = 0.5, skill: bool = False, feed=None,
                 her: danger.Her | None = None, attack: str | None = None):
        """`low_hp`: the safety stop's threshold (behavior.safety: her HP under it for SAFETY_S ends the hunt; None:
        safety.threshold's, SAFETY_HP under her pet's warning, under `potion_hp` with a potion key of her own);
        `potion_hp`: her potion key's (drink_if_low; none bound, her pet drinks them). `feed`: what her
        looks and glances come from (localization.feed.Feed: the perception service's client live,
        localization.service; SyncFeed in the simulator; None: the step on the eyes' own frames, feed.InlineFeed), told
        of her throws. `her`: her HP and level, what a touch is measured against (attack.danger; config.toml [character]
        by default). `attack`: which attack each throw is (behavior.attacks.MODES: 'choice', Lucky Seven but
        to finish a monster and while her MP is under a cast; 'skill', Lucky Seven while she has the MP; 'regular';
        None: 'skill' with `skill`, else 'regular'), with her MP and Lucky Seven's level from config.toml
        [character]."""
        self.model, self.log = model, log
        c = physics.CHARACTER
        self.danger = danger.of_map(model, her or danger.Her(c.hp, c.level))  # where touches would cost her
        self.potion_hp = potion_hp
        own = potion_hp if keys.hp_potion is not None else None  # (her own potion key's: under it, the stop's lower)
        self.hp_watch = HpWatch(threshold(own) if low_hp is None else low_hp)  # the safety stop's: every look's HP
        self.senses = Senses(eyes, model, band=(LOOK_UP, LOOK_DOWN), feed=feed)
        self.senses.log = eyes.log = log  # looks that failed their checks, said in her log (localization.senses)
        mode = attack or ("skill" if skill else "regular")
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
        self._last_potion = 0.0
        self.pet = PetCare(self.actions, eyes, log=log)
        # The hunt's bookkeeping from every look and throw (behavior.ledger): the hunt loop reads the drop store,
        # and the fights and the hunt loop her wounds (the monsters she has hurt, finished before she moves on).
        self.drops = DropStore(self.worth)
        self.ledger = Ledger(self, self.drops)
        self.wounds = Wounds(self.ledger)
        # The monsters she has seen anywhere, kept across looks (localization.memory), her kills (the ledger's) marking
        # the ones they were: after the ledger, so a look's kills are in it.
        self.memory = Memory(model, kills=lambda: self.ledger.kills)
        # Her attack a throw (behavior.attacks): every throw's key is its choice, but in mode 'regular'.
        self.attacks = Attacks(model, keys, mode, max_mp=c.mp, level=c.lucky_seven, log=log,
                               kills=lambda: self.ledger.kills)
        if mode != "regular":
            self.actions.choose = self.attacks.key
        # Out of darts (behavior.darts): her Lucky Seven presses and MP bar (the attack choice's), her kills (_gained)
        self.darts = DartWatch(presses_for(model.mob_hp, getattr(model, "mob_pdd", {})))
        self.attacks.darts = self.darts
        if LAST_LOOK:
            self.actions.still = self._still
        self._players: dict[int, object] = {}  # the other players on the map as of the last look, by id
        self.facing_seen = Reading(None, why="no look")  # which way the last look's frame showed her facing
        # Her facing off the frames: looks, told or not (and why), what her keys made of each (Actions.saw), and those
        # that told the other way from her last arrow ('against')
        self.heading_counts: Counter[str] = Counter()
        # The jump throws her stances chose for what their attack reaches, by why and the jump's kind (fight.log_jump)
        self.jump_counts: Counter[str] = Counter()
        self._shots = len(list(paths.HEADING_SHOTS.glob("*.png"))) if paths.HEADING_SHOTS.exists() else 0
        self.senses.listeners += [self._vitals, self._heading, self._others, self._publish, self.ledger.on_look,
                                  self.drops.on_look, self.wounds.on_look, self.memory.on_look, self.attacks.on_look]
        self.actions.throw_listeners += [self.ledger.on_throw, self.wounds.on_throw, self.memory.on_throw,
                                         self.attacks.on_throw,  # (a cast's MP)
                                         self.senses.feed.throw]  # (the feed's: whose kills are whose, tracking)

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

    def health_check(self, s: Scene) -> None:
        """Asked at every look of a fight, a trip and a pass of the hunt loop: her potions and her pet's food
        (drink_if_low), and the safety stop: her HP under its threshold SAFETY_S over SAFETY_LOOKS looks (`hp_watch`,
        which every look feeds) raises behavior.safety.SafetyStop, once. Until 2026-10-26 under `low_hp` (30% in a tour)
        it stopped the fight at once, "HP low", and the tour or the hunt loop sent her to rest on the chair. Then out of
        darts (behavior.darts: `darts`, which her attack choice feeds): her Lucky Seven presses taking no MP and
        killing nothing for long raise OutOfDarts, once, with what the pickup log says then; a stretch of them long
        enough has the log read (DartWatch.asks: if the game says she's out, the stop comes sooner)."""
        self.drink_if_low(s)
        if (stop := self.hp_watch.due()) is not None:
            raise stop
        if self.darts.asks():
            said = self._messages()
            self.darts.hear(said)
            lines = " / ".join(line for line in said.splitlines() if line.strip()) or "nothing"
            self.log(f"  {self.darts.shown} Lucky Seven presses in a row took no MP: the pickup log reads {lines}")
        if (stop := self.darts.due(time.monotonic())) is not None:
            if stop.said is None:
                stop.heard(self._messages())
            raise stop

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
            return True
        meter, kinds = copy.deepcopy(self.meter), set()
        for e in events:
            for g in meter.update(e.t, reading_of(e.data), e.kind == "exp_gain"):
                kinds.update(g.kinds)
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
        self.hp_watch.see(s.t, s.hp)
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
        for t, reading, changed in reads:
            for g in self.meter.update(t, reading, changed):
                self._gained(g, s)
        for g in self.meter.due(s.t):  # changes it couldn't read that no reading has explained: a kill each
            self._gained(g, s)
        for note in self.meter.notes:
            self.log(f"  {note}")
            telemetry.emit("event", kind="exp", text=note)
        self.meter.notes.clear()
        self.exp_events = [e for e in self.exp_events if s.t - e[0] < EXP_MEMORY]
        self._exp = s.exp
        if lost > 0 and self._last_hp is not None:
            self.report.damage_taken += lost
            self.hit_at = s.t
            telemetry.emit("event", kind="hit", text=f"hit: HP {self._last_hp:.0%} -> {s.hp:.0%}")
            self._log_hit(s)
        self._last_hp, self._last_where, self.actions.pressed = s.hp, s.where, False

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
        rope = s.where is not None and s.where.platform is None and any(
            abs(r.x - s.where.x) <= ROPE_SLACK and r.top - ROPE_SLACK <= s.where.y <= r.bottom + ROPE_SLACK
            for r in self.model.ropes)
        reading = Reading(None, why="rope") if rope else eyes.heading(s)
        self.facing_seen = reading
        c = self.heading_counts
        c["looks"] += 1
        c["told" if reading.side is not None else f"not told ({reading.why})"] += 1
        if (sure := a.keys_say(s.t, self.hit_at)) is not None and not rope:
            eyes.learn_heading(s, sure, reading)
        against = a.see(reading.side, s.t, self.hit_at, reading.strong)
        c[a.saw] += 1
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
            shot = paths.HEADING_SHOTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{c['against']}-keys-{against}.png"
            if eyes.keep_heading(s, shot):
                self._shots += 1
            else:
                shot = None
        telemetry.emit("event", kind="heading", text=text, seen=reading.record(), keys=against,
                       since_arrow=round(since_arrow, 2), since_hit=None if since_hit is None else round(since_hit, 2),
                       **({"shot": shot.name} if shot is not None else {}))

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
            with paths.HIT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pushed": round(pushed, 1), "dy": round(s.where.y - before.y, 1),
                                    "hp": round(self._last_hp - s.hp, 4), "how": s.how,
                                    "platform": before.platform.id, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass
