"""Her actions in the game as key plans for the player: throws (from the ground, jumping, jumping
back), walking, picking up and potions. Actions keeps which way she faces and when her
attack lock ends; it never decides anything (attack and behavior do).

Every plan goes through `send`, blocking until the player has run it, except the navigator's chains
(control.navigator), which are queued with client.send. With the Lie Detector alarm up the player takes only two kinds
of plan (control.alarm): her pet's food (feed_pet's), and her retreat's moves (`moves_only`'s, behavior.retreat).

Which way she faces is known two ways: by her keys (`facing`: the way her plans last turned her or threw, unknown
after a move), and off a look's frame (perception.heading, which the hunter passes on: `see`). Whether a throw taps the
arrow to turn her first goes by `heading`: the frame's where it's sure and newer than what her keys last did (after a
move, a knock, her player's own keys, a plan cut short), else her keys'. The frame takes a tap away only where it's
strong (heading.Reading.strong: none of the frames read the wrong way on 2026-10-24 was), and adds one wherever it tells
otherwise (`faces`). Where the frame can't tell, her keys' stands, but after a hit it's unknown: the throw taps first.
The decisions (attack.stance's side she keeps working, the close monster she jumps away from) go by her keys' as
before: going by the frame's too, which knows her facing after a move where her keys don't, the simulator's hunt loop
made 12% less EXP a minute and the tour 3% (README, Her heading).
"""

import dataclasses
import time
from collections.abc import Callable

from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT, MOVEMENT

from .. import directions
from ..motion import physics
from ..motion.physics import (AIR_TURN_ATTACK_MS, AIR_TURN_MS, BLIND_ENDS_MS, FORWARD_JUMP_MS, JUMP_ATTACK_MS,
                              JUMP_BACK_ATTACK_MS, JUMP_THROW_TURN_MS,
                              KICK_JUMPS, hold_ms)
from ..screen import STRETCHED_CLIENT
from . import client
from .alarm import PET_FOOD, RETREAT

TAP_MS = 40          # a key tap: released this long after it's pressed
TURN_SETTLE_MS = 30  # after the arrow tap that turns her, before the next key
# A frame tells which way she faces only from this long after the last arrow a plan pressed (pressed or let go): in
# the old recordings she turned 60-110 ms after an arrow went down (50 ms taps too, and the toward-taps in mid-air of
# the jump backs), never without one; with a frame's own lag and the player's start on top.
HEADING_SETTLE_S = 0.2
# ... and her keys tell it beyond doubt (for her pictures to learn from: perception.heading) from this long after.
LEARN_AFTER_S = 0.25
# A jump throw's attack key comes JUMP_ATTACK_MS after its jump key, and a jump back's JUMP_BACK_ATTACK_MS
# after its own (both in motion.physics: attack.reach reads them too, since how high she is when the attack
# starts decides what it can hit).

# Margin past her attack lock (COMBAT.attack_lock_ms, counted from when the plan is done) before the
# next attack: after a throw, and after a jump throw.
THROW_LOCK_EXTRA_MS = 60
JUMP_THROW_LOCK_EXTRA_MS = 160
# An arrow pressed while her attack plays doesn't move her until it's over: in game on 2026-10-21, of 13 chains
# sent within 0.35 s of the look after a throw (their first arrow about 0.3 s after its attack key), 6 first moved
# her 0.73-0.92 s after they went out (about 1.0 s after the key: the arrow was still held) and 2 never did (a
# step let go before then). From 00:49 a plan's first arrow waited until 0.6 s after the key, and those still set
# off late: timed by when her dot went off the end of 3 (sim.world.ARROW_HELD has how), the 17 walk-offs whose
# first arrow went 0.59-0.62 s after the key were acted on about 0.79 s after it, 0.12 s late, and the 3 whose arrow
# went 0.79-0.82 s after it at once. Of the 10 chains that set off for the jump from 16 up onto 14 right after a
# throw, 8 came down short of 14 and fell to 25 (taking off 15-30 units short where tracked) and one overshot onto
# 17 off a step back gone late, where the 18 sent with no throw in the 1.5 s before all landed. So a plan's first
# arrow waits until this long after the last attack key (hold_back): the game acts on it 66 ms later, about 0.08 s
# after she could first move. Whether an arrow pressed from about 0.65 s on (acted on from 0.72) is swallowed isn't
# measured: none was pressed then (gamelab README, In game: the check to make before going sooner).
ARROW_LOCK_MS = 800

# A jump back: run away a moment, jump, turn toward the monster in mid-air and throw.
# The arrow away is held this long before the jump, to get her moving. Keyed 80 ms after the arrow, the jump
# came in the blind spot (physics.sure_jump_ms): three of the game's 30 ms steps into her run, at 126 units/s,
# she gets no kick, and three of the five recorded at 60 fps left at 121-127 units/s (one at 142, the arrow seen
# late and kicked; one at 146). Keyed 130 ms after it, as it was until the jump sweeps of 2026-10-21, up to a
# quarter leave short by what they allow (of the 4 jumps keyed 120 ms after the arrow there, 3 left at 125-127).
# Keyed at physics.BLIND_ENDS_MS (150) she has had four steps of the arrow but for at most 5% of jumps, runs 14
# units first, and lands 90-94 units away by the engine, the jump 20 ms later than at 130. With KICK_JUMPS it
# is keyed 52 ms after the arrow, the kick's best by the sweeps (one or two steps of the arrow: kicked to full
# speed from 42 or 84), lifting off within 3 units of where she stands. In the simulator's game-like world (40
# seeds of 20 minutes), keyed at 130 she made 3% less EXP a minute than at 80 (124 and 88 against 127 and 91,
# hunt loop and tour: the plan is 50 ms longer) and picked up 2 points more of her drops; keyed at 40 she made
# as much as at 80.
JB_RUN_MS = 52 if KICK_JUMPS else BLIND_ENDS_MS
STAR_DELAY_MS = 520  # the star leaves this long after the attack key, in the air or not (recorded at 60
                     # fps: 515-545 ms with the key at 190, 290, 450 and 610 ms into a jump back; measured again on
                     # 2026-10-25, the swing beside her 524-536 ms after the key in jump_throw_0 and _1 (the key 100 ms
                     # after the jump key) and jb_k165-jb_k450 (85-370 ms), 508 in snail_jt40b (40 ms): wherever in
                     # the jump the key goes)
# The server applies a throw's damage this long after its attack key, before its star leaves: a kill's EXP gain shows
# then and its monster starts dying, a hurt one's HP bar drops. Live on 2026-10-25 (the hunt of 23:43, the perception
# service's frames) a kill's gain showed 0.15-0.46 s after the key of the throw that made it, the median 0.29-0.33,
# whatever the distance (37 gains with a single throw of hers in the 2.5 s before them:
# D:/MapleHelper-scratch/analysis-1025/jumps-perception-skeptic/gain_delay.py), and a Horny Mushroom's bar first lower
# 0.21-0.55 s after a cast's key (p10-p90; the median 0.40 for 85 casts, 0.31 for 16 regular stars:
# D:/MapleHelper-scratch/hunt-ab/a/scripts/bar_delay.py). The simulator applies it so (sim.world), the frame showing
# it then 0-33 ms later.
DAMAGE_SHOWS_MS = 290
JB_TURN_MS = JB_RUN_MS + AIR_TURN_MS  # the arrow toward the monster: turning as soon as she's off the ground
JB_THROW_MS = JB_RUN_MS + AIR_TURN_ATTACK_MS  # the attack key physics.AIR_TURN_LEAD_MS after it (270), the star
                               # leaving as she lands (270 + 520 = 790; she lands 150 + 65 + 555 = 770 ms in). Until
                               # 2026-10-26 25 ms after the arrow (235), the star just before she landed, and both live
                               # jump backs of 2026-10-25 threw the wrong way: the attack picks its target as it starts,
                               # in the air either way


@physics.follows
def _follow() -> None:
    """Her walk as set (physics.set_speed, which rebinds the names taken from it): the jump back's keys made again
    of BLIND_ENDS_MS (the same 150 for any walk from 134 to 168: physics._blind_ends)."""
    global JB_RUN_MS, JB_TURN_MS, JB_THROW_MS
    JB_RUN_MS = 52 if KICK_JUMPS else physics.BLIND_ENDS_MS
    JB_TURN_MS = JB_RUN_MS + AIR_TURN_MS
    JB_THROW_MS = JB_RUN_MS + AIR_TURN_ATTACK_MS


# A jump throw that turns her first: the turn's arrow tap (MOVEMENT.tap_turn_ms) walks her, and a jump that
# lifts off before she has stopped carries that speed into the air (the arrow is let go by then: no kick).
# Keyed TURN_SETTLE_MS (30) after the tap, the jump lifted off with her at 71 units/s and came down 44 units
# toward the monster in the simulator (40 with the lags spread, 69 at p90; in game, snail_jt100b, a 20 ms tap
# then a jump, drifted 20). By the engine a 50 ms tap has her standing still 231 ms after its press
# (body.hold_time), so the jump key goes in once she is, the lags' spread allowed for: 205 ms after the tap,
# lifting off 272 ms after it. She then comes down where the tap walked her, 8 units on (16 at p90 with the
# lags spread). Taps that short are bimodal in game (two of three 50 ms taps didn't move her at all), so
# half of these waits may be for nothing: an in-game look at jump-throws that turn decides (0 puts back the
# jump TURN_SETTLE_MS after the tap).
# (physics.JUMP_THROW_TURN_MS, 205, imported above)
# ... which is now only for an attack that has to come too soon in the jump for her to turn in the air first (an attack
# keyed under physics.AIR_TURN_ATTACK_MS after the jump key: the stance asks for it only at a monster too short for
# a later one to pick, reach.air_hits). Every other jump throw that turns her jumps at once and taps the arrow in the
# air (physics.AIR_TURN_MS after the jump key, as the jump backs do), its attack AIR_TURN_ATTACK_MS (120, the arrow's
# physics.AIR_TURN_LEAD_MS after it; 85 until 2026-10-26, when 36% of such turns went the wrong way live) after the jump
# key at the soonest: 120 ms after the plan starts, not 245; by the engine she comes down 3.8 units on (5.3 so).
# A jump forward (a jump throw with `way`): the arrow that way held from the plan's start, the jump physics.FORWARD_JUMP_MS
# after it (the kick: she leaves at full speed, 149), and let go this long after the jump key, once the game has seen it
# held at the jump's step (it acts on the jump 50-80 ms after its key): in the air it does nothing (it steers toward
# 8.9 units/s at most), and on landing her attack, still playing, stops her. The arrow turns her that way too, so a jump
# forward never taps first.
FORWARD_HOLD_MS = 100
AIR_TURN = True  # whether a jump throw that turns her taps the arrow in the air (above); False: always first, waiting

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

    def __init__(self, keys: GameKeys, skill: bool = False, kind: str | None = None):
        self.keys = keys
        self.before_play: Callable[[], None] | None = None  # asked before a plan is sent; may raise (Hunter._stop_asked)
        self.skill = skill  # throw Lucky Seven instead of the regular attack (costs MP) ...
        # ... or, with `skill`, the attack key each throw is asked of this (the way it's thrown -> a key name): the
        # hunter's choice of attack a throw (behavior.attacks: Lucky Seven, or the regular attack to finish a monster
        # and while her MP is under a cast)
        self.choose: Callable[[str], str | None] | None = None
        # Every throw's key asked of `choose`, a Lucky Seven key or not: her regular attack held at some targets or for
        # the run (behavior.keycheck), which only her attack choice knows; its answer None sends no throw
        self.routed = False
        self.keyed_ground = False  # whether the attack key last chosen (_attack_key) is a throw's from the ground
        # Whether her attack is made from the ground only (the magician's Magic Claw: never cast mid-jump, the user,
        # 2026-10-27): a jump throw goes out as a throw from the ground, a jump back is the jump away alone (no key in
        # the air). False: the thief's, as before.
        self.ground_only = False
        # Whether a throw that way still goes (the way -> False to drop it), asked once her lock has been waited out,
        # before its keys: the hunter's last look before the key (behavior.hunter: a kill shown since the look it was
        # decided on drops it). None: every throw goes.
        self.still: Callable[[str], bool] | None = None
        # Her attack key as the throw listeners hear of it: its jump's kind (physics.jump_of; None: from the ground), which
        # way that jump goes, the key's ms after the jump key, whether a tap on the ground turned her first (Hunter._key_pick)
        self.key_is: tuple[str | None, int, int, bool] = (None, 0, 0, False)
        self.lock_watch: Callable[[], bool] | None = None  # (`_lock_for_attack`)
        self.kind = kind    # what its plans are to the player with the Lie Detector alarm up (alarm.RETREAT: her
                            # retreat's moves); None: refused then
        # Which way she faces (None: unknown). `facing`: by her keys, the way her last plan turned her or threw
        # (None after a move), set at `_keys_at`; `keys_facing`: the way her last arrow turned her (None after a move),
        # at `_arrow_at`; `seen`: the frame's (see), at `seen_at`, and whether that frame was strong (`seen_strong`).
        # `heading` is `seen` where its frame is newer than her last arrow, else `facing`.
        self._facing: str | None = None
        self.keys_facing: str | None = None
        self.seen: str | None = None
        self.seen_at = -1e18
        self.seen_strong = False
        self._keys_at = 0.0
        self._arrow_at = 0.0
        self.arrows_at = -1e18    # when the last arrow a plan pressed goes down or up (time.monotonic)
        self._blind = False       # a plan went out around play (a navigator chain): its arrows aren't known here
        self._doubted = False     # a frame has told the other way from her last arrow (told once: see)
        self.saw = ""             # what came of the last frame `see` was given (for the hunter's counts)
        self._retap = False       # her throws aren't landing: the next throw taps the arrow first, whatever `heading` says
        self.turns = 0            # throws that tapped the arrow to turn her first, and the time it added (ms)
        self.turn_ms = 0
        self.pressed = False  # keys sent since the last look (then a move isn't a knockback)
        self._sent_at = 0.0    # when the last key plan was sent (a look older than that may be out of date)
        # When the last key of her plans went, as far as it's known (time.monotonic): a plan's end as its result is read,
        # less what it ran on after its last key (`ended`). A look from the perception service is of a frame shown
        # senses.KEY_EFFECT_S after it at the soonest, so it sees what came of the keys before it.
        self.last_key = -1e18
        self.of: Actions | None = None  # the actions these are made from (moves_only): told of every key here too
        self.lock_until = 0.0  # when her last attack stops locking out the next one
        self.arrow_until = 0.0  # ... and stops swallowing arrows (ARROW_LOCK_MS after its key)
        self._attack_at = 0.0   # when the last attack key went down (time.monotonic)
        self.keyless = False    # whether the last throw carried out was a jump back that sent no attack key (jump_back)
        # The player's answer to her last plan played here (its 'sent': each key as pressed, ms into the plan; None: none
        # read), and the jump and attack keys of her last jump throw or jump back (key_gap)
        self.sent: list | None = None
        self._jump_keys: tuple[str, str] | None = None
        # Called at every throw with (when its attack key goes down, the way she throws): the kill ledger
        # (behavior.ledger) matches EXP gains to the stars that could have made them.
        self.throw_listeners: list = []

    @property
    def facing(self) -> str | None:
        """Which way she faces by her keys (None: unknown): what the decisions go by."""
        return self._facing

    @facing.setter
    def facing(self, side: str | None) -> None:
        """Her keys' doing: the way a plan's arrow turned her, or None after a move (control.mover, control.navigator).
        A navigator's chains are over by then: frames show what they did from now on (see)."""
        self._keys(side, arrow=True)

    @property
    def heading(self) -> str | None:
        """Which way she faces as best known (None: unknown), what a throw's tap goes by: the frame's (`seen`) where
        it's newer than the last arrow her keys pressed, else her keys' (`facing`). A throw that went without a tap
        leaves her facing as it was, so a frame shown before it still tells: a look that doesn't wait for her keys'
        effect (senses.KEYS_WAIT "moves" or "none") can be of one shown before a throw's key, right after it."""
        return self.seen if self.seen_at >= self._arrow_at else self._facing

    @property
    def facing_by(self) -> str:
        """Who `heading` is from: "frame" or "keys"."""
        return "frame" if self.seen_at >= self._arrow_at else "keys"

    def _keys(self, side: str | None, arrow: bool) -> None:
        """Her keys set her facing: by an arrow (or a move, None), or by a throw that didn't need one."""
        now = time.monotonic()
        self._facing, self._keys_at = side, now
        if arrow:
            self.keys_facing, self._arrow_at, self._doubted, self._retap = side, now, False, False
        if self._blind:
            self._blind = False
            self.arrows_at = max(self.arrows_at, now)

    @property
    def attack_at(self) -> float:
        """When the last attack key of her plans went down, or goes (time.monotonic; a throw's, as it's sent)."""
        return self._attack_at

    @property
    def sent_at(self) -> float:
        """When the last key plan was sent (a look older than that may be out of date)."""
        return self._sent_at

    @sent_at.setter
    def sent_at(self, t: float) -> None:
        """A plan sent around play (the navigator's chains, client.send): which arrows it presses isn't known here, so
        no frame tells her facing until the navigator says it's over (by setting `facing`)."""
        self._sent_at, self._blind = t, True

    def play(self, events: list, name: str, kind: str | None = None) -> None:
        """Run a key plan (events in time order) and wait until the player has. Held back first if a jump
        or attack in it would come inside her attack lock (hold_back). `kind`: what it is to the player with the
        Lie Detector alarm up (client.send's; these actions' own by default)."""
        self.hold_back(events)
        if self.before_play is not None:  # (`gamelab stop`, come since the look it was planned from: nothing sent)
            self.before_play()
        self._sent_at = time.monotonic()
        self.sent = None
        try:
            result = client.submit(events, lead_ms=0, focus=True, name=name, kind=kind or self.kind)
            self.sent = result.get("sent") if isinstance(result, dict) else None
        finally:
            self.ended()  # (refused or cut short, keys may have gone too)
        self.pressed = True

    def key_gap(self) -> float | None:
        """How long after her last jump throw's or jump back's jump key its attack key went down, as the player pressed
        them (ms: its answer's 'sent', control.player: when each key went out; the simulator's player answers with the
        plan's own times), what the jump attack's jitter came to (behavior.fight.throw_act's 'jump_keys' record); None
        if the answer has neither (refused, cut short, no attack key)."""
        if self._jump_keys is None or not self.sent:
            return None
        jump, attack = self._jump_keys
        downs = {}
        for e in self.sent:
            if len(e) >= 3 and e[2] == "down" and e[1] in (jump, attack) and e[1] not in downs:
                downs[e[1]] = e[0]
        if jump not in downs or attack not in downs or jump == attack:
            return None
        return round(downs[attack] - downs[jump], 1)

    def ended(self, tail_s: float = 0.0) -> None:
        """A plan of hers is over, its result just read (or it was cut short: every key let go now, `tail_s` 0): its
        last key went `tail_s` s before now, what it ran on after it (a chain's end_ms past its last key). The actions
        these were made from (`of`: the hunter's, whose last key her looks wait for) are told too."""
        t = time.monotonic() - tail_s
        a = self
        while a is not None:
            a.last_key = max(a.last_key, t)
            a = a.of

    def hold_back(self, events: list) -> None:
        """Wait just long enough that no jump or attack key in a plan of `events` (starting now) goes down
        before her attack lock ends, and no arrow before her attack stops swallowing arrows (ARROW_LOCK_MS):
        pressed inside them, they're ignored. Every plan that can press them comes through here: Actions'
        own, the mover's single moves (via play), and the navigator's chains: when its last arrow goes is kept
        (`arrows_at`), as frames before then and HEADING_SETTLE_S after can't show what it did to her facing."""
        locked = {self.keys.jump.name, self.keys.attack.name}
        if (skill := self._skill_key()) is not None:
            locked.add(skill.name)
        arrows = {self.keys.left.name, self.keys.right.name}
        now, wait = time.monotonic(), 0.0
        first = min((e[0] for e in events if len(e) == 3 and e[1] in locked and e[2] == "down"), default=None)
        if first is not None:
            wait = max(wait, self.lock_until - (now + first / 1000))
        first = min((e[0] for e in events if len(e) == 3 and e[1] in arrows and e[2] == "down"), default=None)
        if first is not None:
            wait = max(wait, self.arrow_until - (now + first / 1000))
        if wait > 0:
            time.sleep(wait)
        last = max((e[0] for e in events if len(e) == 3 and e[1] in arrows), default=None)
        if last is not None:
            self.arrows_at = max(self.arrows_at, time.monotonic() + last / 1000)

    # --- which way she faces --------------------------------------------------------------------

    def see(self, side: str | None, t: float, hit_at: float | None = None, strong: bool = False) -> str | None:
        """What the frame of a look shown at `t` says of her facing (`side`; None: it can't tell). Frames before her
        last arrow could show what it did (HEADING_SETTLE_S after it), or of a navigator's chain still playing, tell
        nothing; nor do those while her attack plays (until `arrow_until`, when she can't turn and her pictures read
        her worst: perception.heading) unless they're `strong` (heading.Reading.strong). A frame that's sure is her
        `heading` where it tells otherwise, or agrees and is strong (one that agrees and isn't changes nothing: a throw
        that way goes without a tap only by her keys or a strong frame, `faces`); one that can't tell leaves it, but
        after a hit since it was last set (`hit_at`: the time of the last look that saw her HP drop), when a knock may
        have turned her, makes it unknown: the next throw taps the arrow first. Returns her last arrow's way the first
        time a frame tells the other (else None), for the hunter to log. What came of the frame is in `saw`."""
        if self._blind:
            self.saw = "a chain playing"
            return None
        if t < self.arrows_at + HEADING_SETTLE_S:
            self.saw = "too soon after an arrow"
            return None
        if t < self.arrow_until and not strong:
            self.saw = "her attack playing"
            return None
        if side is None:
            self.saw = "not told"
            if hit_at is not None and hit_at >= max(self._keys_at, self.seen_at) and self.heading is not None:
                self.seen, self.seen_at, self.seen_strong = None, t, False
                self.saw = "unknown after a hit"
            return None
        against = None
        if self.keys_facing is not None and side != self.keys_facing and not self._doubted:
            against, self._doubted = self.keys_facing, True
        self.saw = "told otherwise" if side != self.heading else "as it was"
        if side != self.heading or strong:
            self.seen, self.seen_at, self.seen_strong = side, t, strong
        return against

    def keys_say(self, t: float, hit_at: float | None = None) -> str | None:
        """Which way her keys say beyond doubt that she faces in a frame shown at `t` (for her pictures to learn
        from), or None: her last arrow LEARN_AFTER_S or more before it, with no hit since, no navigator chain
        playing, and no frame having told otherwise."""
        if self._blind or self.keys_facing is None or self._doubted:
            return None
        if t < self.arrows_at + LEARN_AFTER_S or t < self._arrow_at:
            return None
        if hit_at is not None and hit_at >= self._arrow_at:
            return None
        return self.keys_facing

    def arrow(self, direction: str) -> str:
        """The name of the arrow key for `direction`."""
        return (self.keys.left if direction == LEFT else self.keys.right).name

    def wait_lock(self) -> None:
        """Wait out her last attack's lock (keys pressed before it ends are ignored)."""
        if (left := self.lock_until - time.monotonic()) > 0:
            time.sleep(left)

    def _lock_for_attack(self) -> bool:
        """Wait out her lock before an attack's key: as `wait_lock`, with `lock_watch` frame by frame if she has one
        (behavior.targeting: a monster fading in or coming at her calls her attack off, the next pass decides again): False."""
        if self.lock_watch is not None:
            return self.lock_watch()
        self.wait_lock()
        return True

    # --- attacks ------------------------------------------------------------------------------

    def throw(self, direction: str) -> bool:
        """Face `direction` (the arrow tapped first unless she surely faces it already: `faces`, by her keys or a strong
        frame), then throw (regular attack; with `skill`, Lucky Seven or as `choose` says, a throw at a time).
        Turning taps the arrow, which also walks her a few units: tapping before every throw walked
        her off the end of a platform after nine throws the same way. False: dropped once her lock was waited out
        (`still`), or no attack to throw (her regular attack held and no Lucky Seven to throw instead: `choose`
        says None), no key sent."""
        if not self._lock_for_attack() or not self._still(direction):
            return False
        events, t = self._turn(direction)
        key = self._attack_key(direction, ground=True)  # (chosen first, as jump_throw's: its MP goes with its own key)
        if key is None:
            return False
        self._thrown(t, direction, (None, 0, 0, bool(events)))
        self.play(events + tap(t, key), "throw")
        self._turned(events, direction, t)
        self.lock_until = time.monotonic() + (COMBAT.attack_lock_ms + THROW_LOCK_EXTRA_MS) / 1000
        self.arrow_until = self._attack_at + ARROW_LOCK_MS / 1000
        return True

    def jump_throw(self, direction: str, key_ms: int | None = None, way: int = 0, air_turn: bool = True,
                   turn_key: int = AIR_TURN_ATTACK_MS) -> bool:
        """Face `direction`, jump and attack in mid-air: in the air the attack throws a star even at point-blank range,
        where on the ground the claw only swipes (9-11 damage against 30-70 for a star, seen with slimes crowding her in
        the basin). The attack key comes `key_ms` after the jump key (JUMP_ATTACK_MS, 40, by default: while she's still
        low; the macro's F1 combo times it at 100 ms, and an attack starting that high in the jump went over a snail;
        at the jump's top it starts 77 up, for a monster on a platform above her: attack.reach.shots says which, and
        behavior.fight.jittered moves it a little).
        `way` 0 jumps in place: turning first, she taps the arrow in the air (AIR_TURN_MS after the jump key) and the
        attack comes `turn_key` after the jump key at the soonest (AIR_TURN_ATTACK_MS, AIR_TURN_LEAD_MS after the
        arrow; sooner for a monster too short to pick then: reach.turn_key_ms); with `air_turn` False (an attack that
        must come sooner, at a monster too short for a later one), the tap goes first and the jump waits for its walk to
        stop (JUMP_THROW_TURN_MS). `way` +1/-1 jumps forward that way (the arrow held: FORWARD_JUMP_MS before the jump,
        let go FORWARD_HOLD_MS after it), which faces her that way: a throw that way. False: dropped once her lock was
        waited out (`still`), or no attack to throw (`throw`'s), no key sent. From the ground only (`ground_only`: the
        magician's), a throw from the ground instead."""
        if self.ground_only:
            return self.throw(direction)
        if not self._lock_for_attack() or not self._still(direction):
            return False
        key_ms = JUMP_ATTACK_MS if key_ms is None else key_ms
        key, jump = self._attack_key(direction), self.keys.jump.name
        if key is None:
            return False
        turn, added = [], 0  # the arrow that turns her, and how much later it has her attack come
        self._jump_keys = (jump, key)
        if way:
            arrow = self.arrow(RIGHT if way > 0 else LEFT)
            t = FORWARD_JUMP_MS
            events = [[0, arrow, "down"], *tap(t, jump), [t + FORWARD_HOLD_MS, arrow, "up"], *tap(t + key_ms, key)]
        elif not self.faces(direction) and air_turn and AIR_TURN:
            t, wanted = 0, key_ms
            key_ms = max(key_ms, turn_key)
            turn = tap(AIR_TURN_MS, self.arrow(direction))
            events = [*tap(0, jump), *turn, *tap(key_ms, key)]
            added = key_ms - wanted
        else:
            turn, t = self._turn(direction)
            if turn:  # turned on the ground: the jump waits until the tap's walk has stopped (JUMP_THROW_TURN_MS)
                t = max(t, JUMP_THROW_TURN_MS)
            events = turn + tap(t, jump) + tap(t + key_ms, key)
            added = t
        self._thrown(t + key_ms, direction, ("forward" if way else "up", way, key_ms, bool(turn and t)))
        self.play(sorted(events, key=lambda e: e[0]), "jump_throw")
        if way:
            self.facing = direction  # her arrow turned her (and moved her): no tap, no time added
        else:
            self._turned(turn, direction, added)
        self.lock_until = time.monotonic() + (COMBAT.attack_lock_ms + JUMP_THROW_LOCK_EXTRA_MS) / 1000
        self.arrow_until = self._attack_at + ARROW_LOCK_MS / 1000
        return True

    def jump_back(self, direction: str, key_ms: int | None = None) -> bool:
        """At a monster too close to throw at (on `direction`'s side): jump away from it, turn toward it
        in mid-air, and throw so the star leaves just before she lands. Thrown at point-blank range, from
        the ground or straight up, stars often missed; just before landing she's the whole jump away
        (about 90 units), and out of its way. A star leaves about 520 ms after the attack key, so the
        key goes in right after lift-off: pressed later, the star left after she had landed. The caller
        makes sure there's room behind her. `key_ms`: the attack key that long after the jump key
        (AIR_TURN_ATTACK_MS, 120, by default: AIR_TURN_LEAD_MS after the turn's arrow; sooner for a monster too short
        to pick then, reach.turn_key_ms, JUMP_BACK_ATTACK_MS at the soonest; the top's, reach.TOP_KEY_MS, for a monster
        higher up; jittered: behavior.fight.jittered). With no
        attack to throw (`throw`'s), or her attack from the ground only (`ground_only`: the magician's), the jump away
        alone, turning toward it in the air; `key_ms` 0 (the stance's jump aside) so too, her next keys when she has landed, her
        next throw tapping the arrow: none of 89 jump backs live went with no attack, and on the frames she flips with the attack."""
        self.wait_lock()
        key = None if self.ground_only or key_ms == 0 else self._attack_key(direction)
        away = self.arrow(directions.opposite(direction))
        toward, jump = self.arrow(direction), self.keys.jump.name
        throw_ms = JB_THROW_MS if key_ms is None else JB_RUN_MS + max(key_ms, JUMP_BACK_ATTACK_MS)
        self._jump_keys = None if key is None else (jump, key)
        events = [[0, away, "down"], *tap(JB_RUN_MS, jump),
                  [JB_TURN_MS, away, "up"], *tap(JB_TURN_MS, toward),
                  *([] if key is None else tap(throw_ms, key))]
        if key is not None:
            self._thrown(throw_ms, direction, ("back", -1 if direction == RIGHT else 1, throw_ms - JB_RUN_MS, False))
        else:
            self.keyless = not self.ground_only  # (no attack her choice would send: behavior.fight.keyless. The magician's
            # jump away alone counts as her throw, as it always has)
        self.play(sorted(events, key=lambda e: e[0]), "jump_back")  # the player wants them in time order
        self.facing = direction
        # Locked until the attack's lock ends, which is after she lands: counted, as a jump throw's, from when the plan
        # is done, its last key the attack's (keyed 85-480 ms after the jump key until 2026-11-01, 120-298 since: the
        # soonest or the top's, jittered; 0.89 s on; she lands 0.14-0.54 s
        # after it, 770 ms into the plan). Until 2026-10-25 it counted the attack key's time into the plan again (play
        # waits for the plan): 235 ms more before her next jump or attack, up to 630 with the attack keyed later
        # (review, 2026-10-25).
        self.lock_until = time.monotonic() + (COMBAT.attack_lock_ms + JUMP_THROW_LOCK_EXTRA_MS) / 1000
        self.arrow_until = self._attack_at + ARROW_LOCK_MS / 1000
        if key_ms == 0:
            self.lock_until = self.sent_at + JB_RUN_MS / 1000 + physics.JUMP_LAG + physics.AIR_S  # (no attack: she's down)
            self.retap()  # (the arrow in `facing` cleared it)
        return True  # (a jump back is her way out from a monster too close: never dropped)

    def _still(self, direction: str) -> bool:
        """Whether a throw that way still goes (`still`), once her lock has been waited out."""
        return self.still is None or self.still(direction)

    def _thrown(self, key_ms: int, direction: str, key_is: tuple) -> None:
        """Tell the throw listeners: the attack key goes down `key_ms` into the plan about to be sent (`key_is`: what kind)."""
        when = time.monotonic() + key_ms / 1000
        self._attack_at = when
        self.keyless = False
        self.key_is = key_is
        for listener in self.throw_listeners:
            listener(when, direction)

    def _attack_key(self, direction: str, ground: bool = False) -> str | None:
        """The attack key of a throw that way: the regular attack's; with `skill` (and a Lucky Seven key), Lucky Seven's,
        or whichever `choose` picks (`keyed_ground` says whether it's a throw from the ground: Attacks.key); `routed`,
        whichever `choose` picks whatever her keys (None: no throw)."""
        self.keyed_ground = ground
        skill = self._skill_key()
        if self.choose is not None and (self.routed or (self.skill and skill)):
            return self.choose(direction)
        if self.skill and skill:
            return skill.name
        return self.keys.attack.name

    def _skill_key(self):
        """Her skill's key: Lucky Seven's (the thief's), else Magic Claw's (the magician's), else None."""
        lucky = self.keys.lucky_seven
        return lucky if lucky is not None else getattr(self.keys, "magic_claw", None)

    def rekey(self, attack: str) -> None:
        """Her regular attack on key `attack` (config.toml's name) from now on (behavior.keycheck: the game has it
        there): these actions' keys with it (her moves' copies made from them later too: moves_only)."""
        from maplehelper.keys import parse
        self.follow(dataclasses.replace(self.keys, attack=parse(attack)))

    def follow(self, keys: GameKeys) -> None:
        """Her keys as the game has them from now on (behavior.keycheck, control.keymap.follow): every plan made from
        now reads them (her throws, walks and pick-ups here, the mover's single moves, the navigator's chains through
        its key names, control.navigator.KeyNames, her pet's food, her retreat's moves: moves_only)."""
        self.keys = keys

    def faces(self, direction: str) -> bool:
        """Whether she faces `direction` surely enough for a throw that way to go without the tap that turns her: her
        `heading` is that way by her keys, or by a strong frame (heading.Reading.strong). A frame that isn't strong
        can add a tap (where it tells otherwise: see) but never take one away: every frame read the wrong way on
        2026-10-24 (README, Her heading) was weaker than that, and a star the wrong way costs more than a tap. Not after
        `retap`, until an arrow has gone."""
        return not self._retap and self.heading == direction and (self.facing_by == "keys" or self.seen_strong)

    def retap(self) -> None:
        """Her throws aren't landing (behavior.attacks.Attacks._landing): the next throw taps the arrow first. On
        2026-11-04 at 22:35:08 a strong frame read her facing left, wrongly: no tap, her keys then said left too, and 85
        attacks went into the wall behind her over 71 s, with nothing to tell otherwise."""
        self._retap = True

    def _turn(self, direction: str) -> tuple[list, int]:
        """The tap that turns her toward `direction` (none if she surely faces it already: `faces`), and when it's
        done."""
        if self.faces(direction):
            return [], 0
        arrow = self.arrow(direction)
        return [[0, arrow, "down"], [MOVEMENT.tap_turn_ms, arrow, "up"]], MOVEMENT.tap_turn_ms + TURN_SETTLE_MS

    def _turned(self, tap_events: list, direction: str, key_ms: int) -> None:
        """After a throw her way by her keys is `direction`: its tap (`tap_events`; none if her heading was that way
        already) turned her, putting its attack key `key_ms` later in the plan, counted (`turns`, `turn_ms`)."""
        self._keys(direction, arrow=bool(tap_events))
        if tap_events:
            self.turns += 1
            self.turn_ms += key_ms

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

    def feed_pet(self) -> bool:
        """Feed her pet (it drinks her potions for her); False if no pet food key is bound. A plan of kind
        alarm.PET_FOOD: the player takes it with the Lie Detector alarm up too (a few a minute)."""
        if self.keys.pet_food is None:
            return False
        self.wait_lock()
        self.play(tap(0, self.keys.pet_food.name), "pet_food", kind=PET_FOOD)
        return True

    def nudge_mouse(self, client: tuple[int, int] = STRETCHED_CLIENT) -> None:
        """Move the mouse inside the game (moves only, never clicks), in a client area that size (client px): a
        tooltip from another program once sat over the minimap until the mouse moved. Just below the middle: at
        1920x1080 (1000, 620) then (960, 600). Not in a retreat from the Lie Detector (`kind`): whoever is at the PC
        may be answering it with the mouse."""
        if self.kind is not None:
            return
        w, h = client
        self.play([[0, "mouse", round(1000 * w / 1920), round(620 * h / 1080)],
                   [120, "mouse", round(960 * w / 1920), round(600 * h / 1080)]], "mouse")

    def moves_only(self) -> "Actions":
        """These actions for her retreat from the Lie Detector (behavior.retreat): her moves' keys alone (arrows, jump,
        up and down; no pick-up, chair, potion, pet food or Lucky Seven), every plan of kind alarm.RETREAT, no mouse
        (nudge_mouse), and her attack's locks as they are here (an arrow pressed inside one doesn't move her). Her last
        key is these actions' too, both ways (`of`): the retreat's looks are of frames that show what came of its keys
        (Senses.keyed reads the hunter's: before, the retreat's own keys went unseen there, and on loop_tree_lie_detector's
        seed 1, sighted at 40 s, 4 of its 13 looks were of frames shown less than senses.KEY_EFFECT_S after its last
        key, two of them before it)."""
        keys = dataclasses.replace(self.keys, lucky_seven=None, pick_up=None, chair=None, sit=None, hp_potion=None,
                                   mp_potion=None, pet_food=None, magic_claw=None, magic_guard=None, magic_armor=None,
                                   energy_bolt=None)
        out = Actions(keys, kind=RETREAT)
        out.facing, out.lock_until, out.arrow_until = self.facing, self.lock_until, self.arrow_until
        out._attack_at, out.sent_at = self._attack_at, self.sent_at
        out.last_key, out.of = self.last_key, self
        return out
