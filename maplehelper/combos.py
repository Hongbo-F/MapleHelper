"""The combos. Each `say` is one step on the coach overlay. Slow mode pauses at every `say`
unless it's marked pause=False, so a pausing step must start a complete move (never mid-jump)."""

from .engine import Combo, Ctx


def jump_throw(c: Ctx) -> None:
    c.say(f"{c.keys.jump.label}, then {c.keys.attack.label} in mid-air → star")
    c.jump_attack()


def back_off(c: Ctx) -> None:
    c.say(f"Hold {c.back.label} to walk away")
    c.tap(c.back, c.t.walk_away)
    c.say(f"Tap {c.front.label} to turn around")
    c.tap(c.front, c.t.turn_tap)
    c.say(f"{c.throw_label} ×2")
    c.throw(2)


def hop_back(c: Ctx) -> None:
    c.say(f"Hold {c.back.label} + {c.keys.jump.label} to hop away")
    c.down(c.back)
    c.wait(c.t.key_press)  # already moving when she jumps, so the hop carries her away
    c.tap(c.keys.jump)
    c.wait(c.t.hop_hold)
    c.up(c.back)
    c.say(f"Tap {c.front.label} to turn around")
    c.tap(c.front, c.t.turn_tap)
    c.say(f"{c.throw_label} ×2")
    c.throw(2)


def jump_back_shot(c: Ctx) -> None:
    """Jump away from the monster, turn around in the air, and throw at it before landing."""
    c.say(f"Hold {c.back.label} + {c.keys.jump.label} to jump away")
    c.down(c.back)
    c.wait(c.t.key_press)  # already moving when she jumps, so the jump carries her away
    c.tap(c.keys.jump)
    c.wait(c.t.air_delay)
    c.up(c.back)  # the jump keeps carrying her backwards after the arrow is released
    c.say(f"In the air: tap {c.front.label} to face them", pause=False)
    c.tap(c.front, c.t.turn_tap)
    c.say(f"{c.air_throw_label} to throw before landing", pause=False)
    c.air_throw()
    c.wait(c.t.throw_gap)


COMBOS = [
    Combo("jump_throw", "Jump-Throw", jump_throw),
    Combo("back_off", "Back Off & Throw", back_off),
    Combo("hop_back", "Hop Back & Throw", hop_back),
    Combo("jump_back_shot", "Jump-Back Shot", jump_back_shot),
]
