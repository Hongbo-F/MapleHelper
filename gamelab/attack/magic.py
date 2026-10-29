"""The magician's Magic Claw in the attack layer, the quick way: attack.reach's constants (and the few copied out of it)
swapped for her spell's search for the process she plays in (the user, 2026-10-28 ~23:55: "I need it ASAP to start
grinding on magician"). The design's aim facade (D:/MapleHelper-scratch/mage/design.md, S1: aim.can_hit and the rest as
the profile's geometry, the thief's reach looked up at every call) is for later; until then her decisions read reach's
module constants at call time as the thief's do, so setting them once at a command's start (`apply`, from
gamelab.__main__ for the commands that hunt) has every decision go by her search. The thief's process never calls it:
her constants stay as they are (her goldens are the check).

What changes (classes.magician.SEARCH and CAST: the client's search for an aimed spell, settled by the user on
2026-10-27: walls and platforms don't block it, never cast mid-jump):
- the search: from 50 ahead of her (SEARCH_START; its first window 16-40 over her feet) to 300 (STAR_RANGE; 13 bands),
  aimed inside 310 (SHOT_RANGE: the thief's 10 past her range, a monster walking while the cast starts), reach's
  STAR_REACH the range and the widest body's half; can_hit's and can_shoot's default reach with it (their defaults are
  bound at import);
- no claw up close: her key never swipes (CLAW: a box no body reaches), and her decisions keep a monster on her
  platform at least CLAW_FAR (the search's start) past its half and CLAW_MARGIN off before a cast (reach.claw_edge: a
  Horny Mushroom about 90 off), stepping away as the thief does from her claw's reach (attack.stance's close quarters);
- nothing stops it (WALLS_BLOCK off: walls and floors alike), no old ramp below her (RAMP_BELOW off), no flight
  (STAR_SPEED: its hit shows at the cast's release);
- from the ground only: no jump shots planned (stance.PASSING_JUMPS off here; behavior.fight's and the hunt loop's
  jump cover none for an attack her profile doesn't make from a jump, fight._air), and her Actions never key an attack
  in the air (control.actions.Actions.ground_only: a jump throw goes out from the ground, a jump back is the jump away
  alone).
"""

import contextlib

from ..character import Character

_SAVED: list[dict] = []  # the constants as they were before `apply` (one swap at a time)


def _targets():
    from . import reach, spots, stance
    return reach, spots, stance


def swaps(c: Character) -> dict:
    """What `apply` sets for character `c` (a magician's profile: her skill's search), as {(module, name): value}."""
    reach, spots, stance = _targets()
    s = c.attacks["skill"].geometry.search
    start, rng = s.start, s.range
    shot_range = rng + 10  # (the thief's SHOT_RANGE 410 against her STAR_RANGE 400)
    widest = max(reach.MONSTER_HALF.values())
    out = {(reach, "STAR_RANGE"): rng, (reach, "STAR_REACH"): rng + widest, (reach, "SHOT_RANGE"): shot_range,
           (reach, "SEARCH_START"): start, (reach, "SEARCH_STEP"): s.step, (reach, "SEARCH_SPREAD"): s.spread,
           (reach, "SEARCH_FLOOR0"): reach.SHOT_H - start // s.spread,
           (reach, "SEARCH_TOP0"): reach.SHOT_H + start // s.spread,
           (reach, "BANDS"): s.bands, (reach, "AREA_UP"): reach.SHOT_H + start // s.spread,
           (reach, "CLAW_FAR"): start,
           (reach, "CLAW"): reach.Claw((("none", ((-1, -10000), (0, -9999))),), start),  # (no body reaches it)
           (reach, "RAMP_BELOW"): False, (reach, "WALLS_BLOCK"): False, (reach, "STAR_SPEED"): 1e9,
           (spots, "SHOT_RANGE"): shot_range,
           (stance, "PASSING_JUMPS"): False, (stance, "JUMP_THROW_SNAILS"): False}
    return out


def apply(c: Character) -> bool:
    """Set reach's constants (and the copies of them) for character `c` if her skill isn't the thief's star search
    (a magician's Magic Claw); nothing for the thief. Whether anything was set."""
    geo = c.attacks.get("skill").geometry if c.attacks.get("skill") is not None else None
    if c.klass == "thief" or geo is None or geo.kind != "search" or geo.search is None:
        return False
    if _SAVED:
        restore()
    reach, spots, stance = _targets()
    table = swaps(c)
    saved = {key: getattr(key[0], key[1]) for key in table}
    saved[("defaults", "can_shoot")] = reach.can_shoot.__defaults__
    saved[("defaults", "can_hit")] = reach.can_hit.__defaults__
    saved[(reach, "STAR_SURE")] = reach.STAR_SURE
    _SAVED.append(saved)
    for (mod, name), value in table.items():
        setattr(mod, name, value)
    rng = table[(reach, "SHOT_RANGE")]
    reach.can_shoot.__defaults__ = (rng, *reach.can_shoot.__defaults__[1:])
    reach.can_hit.__defaults__ = (rng, *reach.can_hit.__defaults__[1:])
    reach.STAR_SURE = reach.star_sure(reach.MONSTER_HALF["snail"])
    stance._JUMP_REACH.clear()
    return True


def restore() -> None:
    """The constants as they were before `apply` (tests)."""
    if not _SAVED:
        return
    reach, spots, stance = _targets()
    saved = _SAVED.pop()
    for key, value in saved.items():
        if key[0] == "defaults":
            getattr(reach, key[1]).__defaults__ = value
        else:
            setattr(key[0], key[1], value)
    stance._JUMP_REACH.clear()


def applied() -> bool:
    """Whether her constants are set (apply)."""
    return bool(_SAVED)


@contextlib.contextmanager
def swapped(c: Character):
    """Tests: `apply(c)` for the block, then everything back."""
    done = apply(c)
    try:
        yield done
    finally:
        if done:
            restore()
