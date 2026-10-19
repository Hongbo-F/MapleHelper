"""The combos. Each `say` is one step on the coach overlay. Slow mode pauses at every `say`
unless it's marked pause=False, so a pausing step must start a complete move (never mid-jump)."""

from .engine import Combo, Ctx


def jump_throw(c: Ctx) -> None:
    c.say(f"{c.keys.jump.label}, then {c.keys.attack.label} in mid-air → star")
    c.jump_attack()
    c.wait(c.t.throw_gap)  # the combo ends when she can act again, like the others


def back_off(c: Ctx) -> None:
    c.say(f"Hold {c.back.label} to walk away")
    c.tap(c.back, c.t.walk_away)
    c.say(f"Tap {c.front.label} to turn around")
    c.tap(c.front, c.t.turn_tap)
    c.say(f"{c.throw_label} ×2")
    c.throw(2)


def hop_back(c: Ctx) -> None:
    """A Jump-Back Shot, then one more throw once she can act again."""
    jump_back_shot(c)
    c.say(f"{c.throw_label} again")
    c.throw()


def jump_back_shot(c: Ctx) -> None:
    """Jump away from the monster, turn around in the air, and throw at it; the star comes out as she lands."""
    c.say(f"Hold {c.back.label} + {c.keys.jump.label} to jump away")
    c.down(c.back)
    c.wait(c.t.run_up)  # already moving when she jumps, so the jump carries her away
    c.tap(c.keys.jump)
    c.wait(c.t.air_delay)  # keep pushing away through takeoff
    c.up(c.back)  # the jump keeps carrying her backwards after the arrow is released
    c.say(f"In the air: hold {c.front.label} to face them", pause=False)
    c.down(c.front)
    # Right after takeoff the game sometimes ignores a turn, so the arrow stays held through the attack.
    c.wait(c.t.turn_lead)
    c.say(f"{c.air_throw_label} while still holding {c.front.label}", pause=False)
    c.air_throw()
    c.up(c.front)
    c.wait(c.t.throw_gap)


COMBOS = [
    Combo("jump_throw", "Jump-Throw", jump_throw),
    Combo("back_off", "Back Off & Throw", back_off),
    Combo("hop_back", "Hop Back & Throw", hop_back),
    Combo("jump_back_shot", "Jump-Back Shot", jump_back_shot),
]
