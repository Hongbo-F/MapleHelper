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
look, whoever took it, the hunter keeps count of her EXP (a kill is her EXP going up) and her HP.
"""

import json
import random
import time
from collections.abc import Callable

import numpy as np

from maplehelper.config import GameKeys

from .. import paths, telemetry
from ..attack.reach import LOOK_DOWN, LOOK_UP
from ..control.actions import Actions
from ..control.navigator import Navigator
from ..localization.mapdata import MapModel
from ..localization.monsters import Monster
from ..localization.pose import Where
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..perception.eyes import Eyes
from . import fight
from .fight import Report
from .ledger import DropStore, Ledger
from .looting import Looter

EXP_CHANGED = 3     # pixels of her EXP number that must differ from the last look's for an EXP gain
EXP_MEMORY = 30     # s her EXP gains are kept (exp_events), for kill() to match to its target
POTION_EVERY = 3    # s at least between two potions
# s a touch's knock may still be moving her after the look that saw it take HP: it carries her through the air
# (0.22 s in the simulator) and she slides on as she lands (180 units/s braked at 800: 0.22 s more). A jump keyed
# from a look in that time comes while she's in the air (and is lost) or still sliding, from where she no longer is.
KNOCK_S = 0.45


class Hunter:
    """The agent: her senses, her keys and her looter, and what she did (`report`); runs the fights."""

    def __init__(self, eyes: Eyes, keys: GameKeys, model: MapModel, *, log: Callable[[str], None] = print,
                 low_hp: float = 0.35, potion_hp: float = 0.5, skill: bool = False):
        self.model, self.log = model, log
        self.low_hp, self.potion_hp = low_hp, potion_hp
        self.senses = Senses(eyes, model, band=(LOOK_UP, LOOK_DOWN))
        self.actions = Actions(keys, skill)
        self.looter = Looter(self)
        self.report = Report()  # what she did (a tour starts a new one at every stop)
        self.exp_events: list[float] = []  # when her EXP went up (the last EXP_MEMORY s)
        self.exp_gains = 0  # times her EXP went up, ever (reports are replaced; this isn't)
        self._exp: np.ndarray | None = None  # the last look's EXP number (text pixels)
        self._last_hp: float | None = None
        self._last_where: Where | None = None
        self.hit_at: float | None = None  # the time of the last look that saw her HP drop (a touch)
        self.knock_waited: float | None = None  # ... the hit_at a jump last waited out (fight.knocked_jump)
        self._last_potion = 0.0
        # The hunt's bookkeeping from every look and throw (behavior.ledger): read by nothing that decides yet.
        self.drops = DropStore()
        self.ledger = Ledger(self, self.drops)
        self.senses.listeners += [self._vitals, self._publish, self.ledger.on_look, self.drops.on_look]
        self.actions.throw_listeners.append(self.ledger.on_throw)

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False) -> Scene:
        """A look (Senses.look); drops are looked for as her inventory allows (only mesos when it's full)."""
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

    def drink_if_low(self, s: Scene) -> None:
        """Drink an HP potion below `potion_hp` (one every POTION_EVERY s at most). The health check on
        the way to rest: low HP is why she's going, not a reason to stop."""
        if s.hp < self.potion_hp and time.monotonic() - self._last_potion > POTION_EVERY:
            self.log(f"HP {s.hp:.0%}: drinking a potion")
            if self.actions.drink():
                self._last_potion = time.monotonic()
                self.report.potions += 1
                telemetry.emit("event", kind="potion", text=f"potion at {s.hp:.0%} HP")

    def take_exp(self, since: float) -> bool:
        """Use up one EXP gain seen since `since`, if there is one."""
        for i, t in enumerate(self.exp_events):
            if t >= since:
                del self.exp_events[i]
                return True
        return False

    # --- after every look ---------------------------------------------------------------------

    def _vitals(self, s: Scene) -> None:
        """Her EXP number changed since the last look: a kill of hers. Her HP dropped: a hit (logged
        with how far it pushed her when no key was pressed)."""
        if _exp_changed(self._exp, s.exp):
            self.exp_events.append(s.t)
            self.report.exp_kills += 1
            self.exp_gains += 1
        self.exp_events = [t for t in self.exp_events if s.t - t < EXP_MEMORY]
        self._exp = s.exp
        if self._last_hp is not None and s.hp < self._last_hp:
            self.report.damage_taken += self._last_hp - s.hp
            self.hit_at = s.t
            telemetry.emit("event", kind="hit", text=f"hit: HP {self._last_hp:.0%} -> {s.hp:.0%}")
            self._log_hit(s)
        self._last_hp, self._last_where, self.actions.pressed = s.hp, s.where, False

    def _publish(self, s: Scene) -> None:
        """The look and the drops she remembers, for the mini-map (telemetry)."""
        if telemetry.recording():
            record = s.record()
            if record["her"] is not None:
                record["her"]["facing"] = self.actions.facing
            telemetry.emit("scene", **record)
            telemetry.emit("drops", inventory_full=self.looter.inventory_full, pending=[
                {"x": round(d.x, 1), "y": round(d.y, 1), "platform": d.platform, "mesos": d.mesos}
                for d in self.looter.pending])

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


def _exp_changed(before: np.ndarray | None, now: np.ndarray | None) -> bool:
    """Whether her EXP number (its text pixels) changed between two looks: it went up."""
    return (before is not None and now is not None and now.shape == before.shape
            and int((now != before).sum()) >= EXP_CHANGED)
