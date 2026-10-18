"""The combos. Each `say` is one step on the coach overlay. Slow mode pauses at every `say`,
so each step must be a complete move that's safe to pause before (never mid-jump)."""

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


def panic(c: Ctx) -> None:
    k = c.keys
    if k.dark_sight:
        c.say(f"{k.dark_sight.label}: Dark Sight, monsters lose you")
        c.tap(k.dark_sight)
        c.wait(c.t.cast_delay)
    if k.hp_potion:
        c.say(f"{k.hp_potion.label}: drink a potion")
        c.tap(k.hp_potion)
        c.wait(c.t.potion_gap)
    c.say(f"Hold {c.back.label} to walk to safety")
    c.tap(c.back, c.t.panic_walk)
    if k.hp_potion:
        c.say(f"{k.hp_potion.label}: drink again")
        c.tap(k.hp_potion)
        c.wait(c.t.potion_gap)
    c.say(f"Tap {c.front.label} to face them again")
    c.tap(c.front, c.t.turn_tap)


COMBOS = [
    Combo("jump_throw", "Jump-Throw", jump_throw),
    Combo("back_off", "Back Off & Throw", back_off),
    Combo("hop_back", "Hop Back & Throw", hop_back),
    Combo("panic", "Panic: Hide & Heal", panic, urgent=True),
]
