"""Her actions in the game as key plans for the player: throws (from the ground, jumping, jumping
back), walking, picking up, potions and the chair. Actions keeps which way she faces and when her
attack lock ends; it never decides anything (attack and behavior do).

Every plan goes through `send`, blocking until the player has run it, except the navigator's chains
(control.navigator), which are queued with client.send.
"""

import time

from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT, MOVEMENT

from .. import directions
from ..motion.physics import hold_ms
from . import client

TAP_MS = 40          # a key tap: released this long after it's pressed
TURN_SETTLE_MS = 30  # after the arrow tap that turns her, before the next key
JUMP_ATTACK_MS = 100  # a jump throw's attack key comes this long after the jump key (the macro's F1 combo)
# Margin past her attack lock (COMBAT.attack_lock_ms, counted from when the plan is done) before the
# next attack: after a throw, and after a jump throw.
THROW_LOCK_EXTRA_MS = 60
JUMP_THROW_LOCK_EXTRA_MS = 160

# A jump back: run away a moment, jump, turn toward the monster in mid-air and throw.
JB_RUN_MS = 80     # the arrow away is held this long before the jump, to get her moving
STAR_DELAY_MS = 520  # the star leaves this long after the attack key, in the air or not (recorded at 60
                     # fps: 515-545 ms with the key at 190, 290, 450 and 610 ms into a jump back)
JB_TURN_MS = JB_RUN_MS + 60  # the arrow toward the monster: turning as soon as she's off the ground
JB_THROW_MS = JB_TURN_MS + 25  # the attack key right after, so the star leaves just before she lands
                               # (165 + 520 = 685; she lands 80 + 55 + 555 = 690 ms in): she's as far
                               # off as the jump takes her

# Picking up: one item per press.
PILE_PRESSES = 4      # presses standing on a kill's drops: a pig drops several
PILE_EVERY_MS = 120   # ... this far apart
WALK_PICK_EVERY_MS = 150  # pressed this often while she walks (anything she passes over is looted)
WALK_PICK_AFTER_MS = 200  # ... and on this long after the arrow is let go, while she slides to a stop


def tap(t: int, key: str) -> list[list]:
    """Key events tapping `key` at t ms into a plan."""
    return [[t, key, "down"], [t + TAP_MS, key, "up"]]


class Actions:
    """Her actions as key plans, run one at a time (send blocks until the player is done)."""

    def __init__(self, keys: GameKeys, skill: bool = False):
        self.keys = keys
        self.skill = skill  # throw Lucky Seven instead of the regular attack (costs MP)
        self.facing: str | None = None  # which way she faces (None: unknown), so throws only turn when needed
        self.pressed = False  # keys sent since the last look (then a move isn't a knockback)
        self.lock_until = 0.0  # when her last attack stops locking out the next one
        # Called at every throw with (when its attack key goes down, the way she throws): the kill ledger
        # (behavior.ledger) matches EXP gains to the stars that could have made them.
        self.throw_listeners: list = []

    def play(self, events: list, name: str) -> None:
        """Run a key plan (events in time order) and wait until the player has. Held back first if a jump
        or attack in it would come inside her attack lock (hold_back)."""
        self.hold_back(events)
        client.submit(events, lead_ms=0, focus=True, name=name)
        self.pressed = True

    def hold_back(self, events: list) -> None:
        """Wait just long enough that no jump or attack key in a plan of `events` (starting now) goes down
        before her attack lock ends: pressed inside it, they're ignored. Every plan that can press them
        comes through here: Actions' own, the mover's single moves (via play), and the navigator's chains."""
        locked = {self.keys.jump.name, self.keys.attack.name}
        if self.keys.lucky_seven is not None:
            locked.add(self.keys.lucky_seven.name)
        first = min((e[0] for e in events if len(e) == 3 and e[1] in locked and e[2] == "down"), default=None)
        if first is not None and (left := self.lock_until - (time.monotonic() + first / 1000)) > 0:
            time.sleep(left)

    def arrow(self, direction: str) -> str:
        """The name of the arrow key for `direction`."""
        return (self.keys.left if direction == LEFT else self.keys.right).name

    def wait_lock(self) -> None:
        """Wait out her last attack's lock (keys pressed before it ends are ignored)."""
        if (left := self.lock_until - time.monotonic()) > 0:
            time.sleep(left)

    # --- attacks ------------------------------------------------------------------------------

    def throw(self, direction: str) -> None:
        """Face `direction` (if she isn't already), then throw (regular attack; Lucky Seven if `skill`).
        Turning taps the arrow, which also walks her a few units: tapping before every throw walked
        her off the end of a platform after nine throws the same way."""
        self.wait_lock()
        events, t = self._turn(direction)
        self._thrown(t, direction)
        self.play(events + tap(t, self._attack_key()), "throw")
        self.facing = direction
        self.lock_until = time.monotonic() + (COMBAT.attack_lock_ms + THROW_LOCK_EXTRA_MS) / 1000

    def jump_throw(self, direction: str) -> None:
        """Face `direction`, jump straight up and attack in mid-air: in the air the attack throws a
        star even at point-blank range, where on the ground the claw only swipes (9-11 damage against
        30-70 for a star, seen with slimes crowding her in the basin). The macro's F1 combo, with its
        timing (jump, then the attack 100 ms after the jump key)."""
        self.wait_lock()
        events, t = self._turn(direction)
        key, jump = self._attack_key(), self.keys.jump.name
        events += tap(t, jump) + tap(t + JUMP_ATTACK_MS, key)
        self._thrown(t + JUMP_ATTACK_MS, direction)
        self.play(events, "jump_throw")
        self.facing = direction
        self.lock_until = time.monotonic() + (COMBAT.attack_lock_ms + JUMP_THROW_LOCK_EXTRA_MS) / 1000

    def jump_back(self, direction: str) -> None:
        """At a monster too close to throw at (on `direction`'s side): jump away from it, turn toward it
        in mid-air, and throw so the star leaves just before she lands. Thrown at point-blank range, from
        the ground or straight up, stars often missed; just before landing she's the whole jump away
        (about 85 units), and out of its way. A star leaves about 520 ms after the attack key, so the
        key goes in right after lift-off: pressed later, the star left after she had landed. The caller
        makes sure there's room behind her."""
        self.wait_lock()
        key = self._attack_key()
        away = self.arrow(directions.opposite(direction))
        toward, jump = self.arrow(direction), self.keys.jump.name
        events = [[0, away, "down"], *tap(JB_RUN_MS, jump),
                  [JB_TURN_MS, away, "up"], *tap(JB_TURN_MS, toward),
                  *tap(JB_THROW_MS, key)]
        self._thrown(JB_THROW_MS, direction)
        self.play(sorted(events, key=lambda e: e[0]), "jump_back")  # the player wants them in time order
        self.facing = direction
        # Locked until the attack's lock ends, which is after she lands.
        self.lock_until = time.monotonic() + (JB_THROW_MS + COMBAT.attack_lock_ms) / 1000

    def _thrown(self, key_ms: int, direction: str) -> None:
        """Tell the throw listeners: the attack key goes down `key_ms` into the plan about to be sent."""
        when = time.monotonic() + key_ms / 1000
        for listener in self.throw_listeners:
            listener(when, direction)

    def _attack_key(self) -> str:
        return (self.keys.lucky_seven if self.skill and self.keys.lucky_seven else self.keys.attack).name

    def _turn(self, direction: str) -> tuple[list, int]:
        """The tap that turns her toward `direction` (none if she already faces it), and when it's done."""
        if self.facing == direction:
            return [], 0
        arrow = self.arrow(direction)
        return [[0, arrow, "down"], [MOVEMENT.tap_turn_ms, arrow, "up"]], MOVEMENT.tap_turn_ms + TURN_SETTLE_MS

    # --- moving and looting -------------------------------------------------------------------

    def walk(self, direction: str, units: float) -> None:
        """Walk `units` that way, pressing pick-up all the while (anything she passes over is looted)."""
        self.wait_lock()
        arrow, ms = self.arrow(direction), hold_ms(units)
        events = [[0, arrow, "down"], [ms, arrow, "up"]]
        if self.keys.pick_up is not None:
            events += self.pickups(ms + WALK_PICK_AFTER_MS)
        self.play(sorted(events, key=lambda e: e[0]), "walk")
        self.facing = direction
        time.sleep(MOVEMENT.stop_slide_ms / 1000)

    def pick_here(self, presses: int = PILE_PRESSES) -> None:
        """Stand and press pick-up `presses` times, PILE_EVERY_MS apart: one item per press."""
        if self.keys.pick_up is None:
            return
        self.wait_lock()
        self.play(self.pickups(presses * PILE_EVERY_MS, every=PILE_EVERY_MS), "pick")

    def pickups(self, end: int, every: int = WALK_PICK_EVERY_MS) -> list[list]:
        """Pick-up taps every `every` ms from 0 until `end` (none if no pick-up key is bound)."""
        if self.keys.pick_up is None:
            return []
        name = self.keys.pick_up.name
        return [e for t in range(0, end, every) for e in tap(t, name)]

    # --- other ----------------------------------------------------------------------------------

    def drink(self) -> bool:
        """Drink an HP potion; False if no potion key is bound."""
        if self.keys.hp_potion is None:
            return False
        self.wait_lock()
        self.play(tap(0, self.keys.hp_potion.name), "potion")
        return True

    def sit(self) -> None:
        """Sit on the chair (HP comes back faster)."""
        if self.keys.chair is not None:
            self.play(tap(0, self.keys.chair.name), "chair")

    def stand(self) -> None:
        """Stand up from the chair: tapping an arrow does it (she then faces right)."""
        arrow = self.keys.right.name
        self.play([[0, arrow, "down"], [50, arrow, "up"]], "stand")
        self.facing = RIGHT

    def nudge_mouse(self) -> None:
        """Move the mouse inside the game (moves only, never clicks): a tooltip from another program
        once sat over the minimap until the mouse moved."""
        self.play([[0, "mouse", 1000, 620], [120, "mouse", 960, 600]], "mouse")
