"""The hunter: the agent that runs hunting objectives on its own (kill one monster, clear the monsters in
reach, and the tours built on them).

The decider (a person, or Claude) picks objectives; the hunter makes the small decisions in a loop:
look at the game, keep the target at the far end of a star's reach, face it, throw a regular attack
(Lucky Seven costs MP and isn't needed on these monsters), step away from monsters that get close, loot
each kill straight away when it fell on her platform (drops elsewhere are remembered for the tour to
collect), drink a potion when HP gets low, and hand control back when HP is too low, the target is gone,
nothing is left in reach, or the time is up. Every key goes through the gamelab player, so the Lie
Detector watchdog stops everything the moment the game's anti-bot check appears.

It's made of the layers below: Senses (perception and localization: every look), Actions (control:
her keys), a Looter (what she picks up and remembers), and the fight loops (behavior.fight). After every
look, whoever took it, the hunter keeps count of her EXP and her HP: a kill is her EXP going up, its number read
when it changes (perception.exp_number), and the gain's size says what kind it was, or that it was several kills,
or none (localization.gains). It also reads which way she faces off the look's frame (perception.heading) for her
keys (Actions.see), and logs where that tells the other way from her last arrow.
"""

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
from .fight import Report
from .ledger import DropStore, Ledger, Wounds
from .looting import Looter, worth_for
from .pet import PetCare

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


class Hunter:
    """The agent: her senses, her keys and her looter, and what she did (`report`); runs the fights."""

    def __init__(self, eyes: Eyes, keys: GameKeys, model: MapModel, *, log: Callable[[str], None] = print,
                 low_hp: float = 0.35, potion_hp: float = 0.5, skill: bool = False, feed=None,
                 her: danger.Her | None = None):
        """`feed`: a continuous feed for her looks and glances (localization.feed), told of her throws. `her`: her HP and
        level, what a touch is measured against (attack.danger; config.toml [character] by default)."""
        self.model, self.log = model, log
        c = physics.CHARACTER
        self.danger = danger.of_map(model, her or danger.Her(c.hp, c.level))  # where touches would cost her
        self.low_hp, self.potion_hp = low_hp, potion_hp
        self.senses = Senses(eyes, model, band=(LOOK_UP, LOOK_DOWN), feed=feed)
        self.senses.log = eyes.log = log  # looks that failed their checks, said in her log (localization.senses)
        self.actions = Actions(keys, skill)
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
        self._players: dict[int, object] = {}  # the other players on the map as of the last look, by id
        self.facing_seen = Reading(None, why="no look")  # which way the last look's frame showed her facing
        # Her facing off the frames: looks, told or not (and why), what her keys made of each (Actions.saw), and those
        # that told the other way from her last arrow ('against')
        self.heading_counts: Counter[str] = Counter()
        self._shots = len(list(paths.HEADING_SHOTS.glob("*.png"))) if paths.HEADING_SHOTS.exists() else 0
        self.senses.listeners += [self._vitals, self._heading, self._others, self._publish, self.ledger.on_look,
                                  self.drops.on_look, self.wounds.on_look]
        self.actions.throw_listeners += [self.ledger.on_throw, self.wounds.on_throw]
        if feed is not None:  # whose kills are whose (localization.tracking)
            self.actions.throw_listeners.append(feed.throw)

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False) -> Scene:
        """A look (Senses.look); drops are looked for as her inventory allows (only mesos when it's full). Not with
        the Lie Detector alarm up (client.LieDetectorAlarm): a hunt that only looks for a while (waiting, resting)
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
        avoiding them."""
        return Navigator(self.senses, self.actions, self.model, log=self.log, health=self.health_check,
                         potions=self.drink_if_low, escort=fight.FightOnTheWay(self) if fighting else None,
                         rng=rng)

    def knocked(self, s: Scene) -> bool:
        """Whether a touch's knock may still be moving her at look `s`: a look within KNOCK_S before it (or it)
        saw her HP drop."""
        return self.hit_at is not None and 0 <= s.t - self.hit_at < KNOCK_S

    def health_check(self, s: Scene) -> str | None:
        """Drink if HP is low; returns a reason to stop if it's too low."""
        if s.hp < self.low_hp:
            return f"HP low ({s.hp:.0%})"
        self.drink_if_low(s)
        return None

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
        (feed_pet_if_due). The health check on the way to rest: low HP is why she's going, not a reason to stop. With
        no potion key (her pet drinks them since 2026-10-23), nothing: it logged "drinking a potion" at every look
        under `potion_hp` and drank none (in the simulator on West Rocky Mountain II, the hunt loop from 40% HP with
        her pet out of potions: 83 times in 3 minutes)."""
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

    # --- after every look ---------------------------------------------------------------------

    def _vitals(self, s: Scene) -> None:
        """Her EXP number changed since the last look: read, the kills of hers its gain was (one if it can't be
        read; none if no kill explains it: logged). Her HP dropped: a hit (logged with how far it pushed her when no
        key was pressed). With a feed, from what it saw frame by frame (s.events, with the number as read then): two
        gains between two looks are two gains, and the HP lost is every drop's, a potion drunk between them or not."""
        self._doubt_hp(s)
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
