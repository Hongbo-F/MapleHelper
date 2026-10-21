"""Reusable in-game moves, built on the measurements in maplehelper.game.

Each move takes a combo Ctx (maplehelper.engine), so the helper's runner and gamelab can both play
it, and the same timing rules apply (e.g. nothing is pressed inside an attack's lock). Distances
are game-window pixels at 1920x1080; see maplehelper/game.py for where the numbers come from.
"""

import math

from .engine import LEFT, Ctx
from .game import MOVEMENT, RECOVERY
from .keys import Key

PICK_UP_GAP_MS = 260  # between pick-up presses; one item per press
# A running jump from standing goes in at least this long after its arrow. The game moves her in steps of
# about 30 ms: after three steps of the arrow she runs at 85% of her walking speed, too fast for the jump's
# kick (which takes her to full speed from below 80% of it) and short of it herself, so a jump that comes then
# leaves at 85% and lands short (a flat jump about 18 px short). In game on 2026-10-21, of 4 jumps keyed 120 ms
# after the arrow 3 left at 84-85%, and of 4 keyed 100 ms after it 3 did; none of the 10 keyed 130-160 ms after
# it. By gamelab's movement engine fitted to those (gamelab/motion/body.py), from 150 ms on at most 5% leave
# short (physics.BLIND_ENDS_MS, which gamelab keys its own jumps from standing after; at 120 ms, 16-44%).
RUN_JUMP_AFTER_MS = 150


def _key(c: Ctx, action: str) -> Key:
    key = getattr(c.keys, action)
    if key is None:
        raise ValueError(f"'{action}' isn't bound in config.toml [keys]")
    return key


def _arrow(c: Ctx, direction: str) -> Key:
    return c.keys.left if direction == LEFT else c.keys.right


def walk_ms(px: float) -> int:
    """How long to hold an arrow so she walks about `px` pixels."""
    if px <= 0:
        return 0
    return max(MOVEMENT.tap_turn_ms, round((px - MOVEMENT.walk_start_px) / MOVEMENT.walk_px_per_ms))


def turn(c: Ctx, direction: str) -> None:
    """Face `direction` without walking (also stands her up from a chair)."""
    c.tap(_arrow(c, direction), MOVEMENT.tap_turn_ms)
    c.wait(MOVEMENT.reaction_ms)


def walk(c: Ctx, direction: str, px: float) -> None:
    """Walk about `px` pixels and come to a stop. Walking off a ledge just drops her (no fall damage)."""
    c.tap(_arrow(c, direction), walk_ms(px))
    c.wait(MOVEMENT.stop_slide_ms)


def jump(c: Ctx, direction: str | None = None) -> None:
    """Jump in place (~104 px up), or a running jump that carries her ~120 px toward `direction`."""
    if direction is None:
        c.tap(c.keys.jump)
        c.wait(MOVEMENT.reaction_ms + MOVEMENT.jump_airtime_ms)
        return
    arrow = _arrow(c, direction)
    c.down(arrow)
    c.wait(max(MOVEMENT.reaction_ms, RUN_JUMP_AFTER_MS))  # already moving at takeoff, so the jump carries her
    c.tap(c.keys.jump)
    c.wait(MOVEMENT.jump_airtime_ms)
    c.up(arrow)
    c.wait(MOVEMENT.reaction_ms)


EDGE_MARGIN_PX = 15  # stay this far back from the end of her own platform


def ledge_takeoff_px(height_px: float) -> float:
    """Farthest a ledge `height_px` above her can be (sideways) for a running jump to land on it.

    Platforms can be jumped through from below, so arriving early is fine; she must still be at
    least `height_px` up when she reaches the edge, i.e. before the falling side of the jump drops
    below it. Jump height follows a parabola peaking at jump_height_px after jump_peak_ms.
    """
    m = MOVEMENT
    if height_px >= m.jump_height_px:
        return 0.0
    last_ms = m.jump_peak_ms * (1 + math.sqrt(1 - max(0.0, height_px) / m.jump_height_px))
    return m.walk_px_per_ms * last_ms


def jump_to_ledge(c: Ctx, direction: str, ledge_px: float, height_px: float, own_edge_px: float) -> None:
    """Land on a ledge `height_px` higher whose near edge is `ledge_px` away (sideways), taking off
    from her own platform, which ends `own_edge_px` away in the same direction.

    She walks to just short of her own edge (walking past it drops her), then makes a running jump.
    Raises ValueError if the ledge is too high or too far to make it from there.

    Not yet confirmed in game: two attempts at ~90 px ledges with ~60 px gaps both fell short (the
    second stopped before jumping, which this version no longer does). Ropes are the sure way up.
    """
    m = MOVEMENT
    if height_px > m.jump_height_px - 5:
        raise ValueError(f"a {height_px:.0f} px ledge is too high; a jump peaks at ~{m.jump_height_px} px")
    takeoff_at_px = max(0.0, own_edge_px - EDGE_MARGIN_PX)
    takeoff_gap = ledge_px - takeoff_at_px
    reach = 0.85 * ledge_takeoff_px(height_px)  # margin for timing jitter
    if takeoff_gap > reach:
        raise ValueError(f"too far: she'd take off {takeoff_gap:.0f} px from the ledge, "
                         f"but can only reach a {height_px:.0f} px ledge from ~{reach:.0f} px")
    # One continuous run: stopping first and jumping from a standstill carries her ~30% less far.
    # The jump key goes down early enough that she takes off (one reaction later) at takeoff_at_px, but
    # not sooner than RUN_JUMP_AFTER_MS after the arrow: keyed at 100 ms (as it was for take-offs within
    # ~21 px) most such jumps leave short (3 of 4 in game).
    arrow = _arrow(c, direction)
    run_ms = m.reaction_ms + max(0.0, takeoff_at_px - m.walk_px_per_ms * m.reaction_ms) / m.walk_px_per_ms
    run_ms = max(run_ms, RUN_JUMP_AFTER_MS)
    c.down(arrow)
    c.wait(round(run_ms))
    c.tap(c.keys.jump)
    c.wait(m.jump_airtime_ms)
    c.up(arrow)
    c.wait(m.reaction_ms)


def climb_rope(c: Ctx, px: float) -> None:
    """Grab a rope or ladder hanging over her and climb about `px` pixels up it.

    Stand still right under it first: she grabs it by jumping with up held (with sideways momentum
    she sails past it). Climbing past its top steps her onto the platform above.
    """
    up = _key(c, "up")
    c.down(up)
    c.tap(c.keys.jump)
    c.wait(MOVEMENT.jump_peak_ms + round(px / MOVEMENT.climb_px_per_ms))
    c.up(up)
    c.wait(MOVEMENT.reaction_ms)


def climb_down(c: Ctx, px: float) -> None:
    """Climb about `px` pixels down the rope she's on (or step down onto one from its top).

    At the bottom end she lets go and drops; holding down on the ground makes her lie down.
    """
    c.tap(_key(c, "down"), round(px / MOVEMENT.climb_px_per_ms))
    c.wait(MOVEMENT.reaction_ms)


def pick_up(c: Ctx, items: int = 1) -> None:
    """Pick up the drops she's standing on, one per press."""
    for _ in range(items):
        c.tap(_key(c, "pick_up"))
        c.wait(PICK_UP_GAP_MS)


def sit_on_chair(c: Ctx) -> None:
    c.tap(_key(c, "chair"))
    c.wait(3 * MOVEMENT.reaction_ms)


def stand_up(c: Ctx) -> None:
    """Get up from the chair without moving: tap the arrow she's already facing."""
    c.tap(c.front, MOVEMENT.tap_turn_ms)
    c.wait(MOVEMENT.reaction_ms)


def rest_ms(missing_hp: int) -> int:
    """How long to sit on the chair to get `missing_hp` back (it heals in 10-second ticks)."""
    return math.ceil(max(0, missing_hp) / RECOVERY.on_chair_hp_per_10s) * 10_000


def rest(c: Ctx, missing_hp: int) -> None:
    """Sit on the chair until about `missing_hp` HP is back, then stand up. Find a safe spot first."""
    sit_on_chair(c)
    c.wait(rest_ms(missing_hp))
    stand_up(c)


def attack(c: Ctx, times: int = 1) -> None:
    """Regular attack: a claw swipe if a monster is right next to her, a thrown star otherwise."""
    for _ in range(times):
        c.tap(c.keys.attack)
        c.wait(c.t.throw_gap)  # throw_gap is set to clear COMBAT.attack_lock_ms


def lucky_seven(c: Ctx, times: int = 1) -> None:
    """Two stars at one enemy in front within range; 8 MP each at level 4."""
    key = _key(c, "lucky_seven")
    for _ in range(times):
        c.tap(key)
        c.wait(c.t.throw_gap)


def drink_hp(c: Ctx) -> None:
    c.tap(_key(c, "hp_potion"))
    c.wait(MOVEMENT.reaction_ms * 2)


def drink_mp(c: Ctx) -> None:
    c.tap(_key(c, "mp_potion"))
    c.wait(MOVEMENT.reaction_ms * 2)


# name -> (function, argument converters), for running moves by name (gamelab `do`).
# Trailing arguments may be left out where the function has a default.
ACTIONS = {
    "turn": (turn, (str,)),
    "walk": (walk, (str, float)),
    "jump": (jump, (str,)),
    "jump_to_ledge": (jump_to_ledge, (str, float, float, float)),
    "climb_rope": (climb_rope, (float,)),
    "climb_down": (climb_down, (float,)),
    "pick_up": (pick_up, (int,)),
    "sit_on_chair": (sit_on_chair, ()),
    "stand_up": (stand_up, ()),
    "rest": (rest, (int,)),
    "attack": (attack, (int,)),
    "lucky_seven": (lucky_seven, (int,)),
    "drink_hp": (drink_hp, ()),
    "drink_mp": (drink_mp, ()),
}
