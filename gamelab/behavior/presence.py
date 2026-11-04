"""How likely a throw at a monster the look missed is to land (`chance`), and what such a throw is worth (`net`): the
thief's throws at what she can't see go by the evidence of each look, not by how long ago she saw it.

The bug (the user, 2026-11-04, watching the live grind: "mobs near the player are not detected by perception and kept for
2s, during which the motion planner will keep attempting attack the mob even tho it may well have died"; their direction
of 2026-11-03: "Motion planning should consider the current game situation and dynamically calculate the best action to
perform"). A monster the look missed stayed a target for fixed times: 1.8 s from her last throw (fight.HIDDEN_FOR,
attacks.HIDDEN_FOR, renewed by each throw), 1.5 s and 2.0 s for those about (localization.unseen: TRACK_S, UNSEEN_S),
ended only by a kill placed on it. Nothing in her decisions asked whether it was still there. In the live hours of
2026-11-02 22:46 and 2026-11-04 00:22 (118.5 minutes; D:/MapleHelper-scratch/unseen/decisions) 430 of her 2,831 throws
went at a target no look had (13-17%), plus 110 regular attacks at nothing; 53-60% of those throws showed nothing
against 23-24% for targets seen; 34-43% of the targets were dead or gone by the key (a floor) against 7-11%; 245 casts,
3,408 MP and 234 s showed nothing.

**The chance** a throw shows an effect (`chance`): log-odds, a weight for each piece of evidence, each read again at every
look and at the key. The weights are a logistic fit on those 401 throws with no seen monster in their path (fitted on one
hour and tried on the other, out of sample logloss 0.611 and 0.655 against 0.684 and 0.668 for a constant; effect shown
5-20% of the time where the other hour's fit said under 0.2, 58-70% where it said 0.6-0.8:
D:/MapleHelper-scratch/unseen/fix/fit3.py). Evidence with no stable sign between the two hours (a hurt bar at the last
sighting, off her platform, how far off) has no weight. "Effect" is her EXP up, a bar down that way or her claw (a ground
cast with no MP): it can hide a real hit, so a chance is a floor on her throws' worth and the refusals an upper bound.

**The price** (`net`, the user's: HP 2 a meso, MP 1 a meso, a meso 0.5 EXP): a landing is worth the share of the monster's
EXP and drops her attack takes off what it has left (`landing`): a Horny Mushroom's 53 EXP-equivalent, 0.65 of it from a
Lucky Seven at a whole one, all of it at one with less left than her attack does; a throw costs her attack's MP and its
lock's time at her EXP a second (`cost`: 13 EXP-equivalent a cast, 5 a regular attack). So a monster nearly dead is
worth a throw on thinner evidence than a whole one, and a cheap regular attack on thinner evidence than a cast: no fixed
window nor rule, the throw goes while chance x landing beats its cost, and beats a throw at a monster she does see
(`SEEN_CHANCE`, attacks.Attacks.seen_net).

**Positioning** isn't this: the monsters about her go on counting for her body (the stance's `around`), and one too close
that isn't worth a throw is still kept off (stance's `unsure`) as long as the chance of its touch outweighs a step's time
(fight._not_thrown_at). The magician's casts have their own checks and never go at a body unseen (attacks.Attacks.picks).
"""

import math
from dataclasses import dataclass

# The weights (log-odds; fit3.py's fit on both hours, the gain's by kind): a monster of the look just missed starts at
# BASE; each second since its last sighting AGE_S; its track gone from the feed (dropped, or let go) GONE; first seen in
# SHORT_FRAMES frames or fewer, a flicker's or a dead walker's new id, SHORT; her last throw at it unseen showed nothing
# BLANK (n 85 of 430: shown 19%, 56% dead or gone; first throw 47% and 35%), and it showed something PROVEN (n 55, 58%
# and 35%); her EXP up, a gain that can be its kind's, in the GAIN_S before GAIN (n 101, 38%, 61% dead or gone; the fit's
# by kind is -0.3 to -1.4 by the hour, any kind's -0.4 to -0.5). Her throws at it reckoned to have killed it with none of
# them having shown anything, SPENT more (n 59: shown 14%, 61% dead or gone, against 26%, 48% for blank throws under it;
# not with no blank: n 27, shown 56%: those are hits).
BASE = 0.26
AGE_S = -0.65
GONE = -0.86
SHORT = -0.45
SHORT_FRAMES = 3
BLANK = -0.72
PROVEN = 0.89
GAIN = -1.31
GAIN_S = 1.2
SPENT = -0.7
SPENT_DEAD = 0.5   # chance dead by her throws, as if her EXP said nothing, from which they count as spent
# Dying or dead by the feed (n 9: all dead by the key; the 33% shown were others', bystanders'). The fit's weight is
# small only for those few bystanders; what it says of the target is that it's gone.
DYING = -2.5
# A frame the exact finder looked at and didn't find it (perception.exact: tracking.Track.blind): on the lossless clip of
# 2026-11-02 22:49 (fix/gaps.py) of 945 frames-after-a-sighting within 100 x 70 of her, a sighting came next on 928, and
# after a miss of 1, 2, 3 and 4 frames 11 of 17, 6 of 11, 1 of 6 and 0 of 1 went on missing, with 6 of those chains ending
# in a death seen: so about even up to BLIND_FREE frames (0.2 s), then BLIND a frame (log-odds: 0.52 to 0.15 at the 4th,
# about 0.03 at the 5th). So absence soon says it's gone, as the search's misses (a spell of 0.3-1 s near her, sim.world
# .NEAR_SPELL_1102) never could. Not in the fit: the two hours ran without the finder.
BLIND_FREE = 3
BLIND = -1.7
# What a throw at a monster seen shows an effect for (live, both hours: 76-77%).
SEEN_CHANCE = 0.76
OVER_SEEN = True  # a throw at one unseen must beat a throw at one in reach that a look has (Attacks.worth_a_throw)
# Whether her throws at what no look has go by all this. False: as until 2026-11-04, every one of them a target for the fixed
# times, and a regular attack at nothing with none (for the harness's sides: fix/README's arms).
EVIDENCE = True
MESO_EXP = 0.5       # a meso in EXP, and an MP point (MP 1 a meso): the user's reference prices of 2026-10-31 (behavior.hunt)
MP_EXP = MESO_EXP
THROW_LOCK_S = 0.86  # s a throw from the ground takes her from all else (live, measured: the diagnosis's) ...
JUMP_LOCK_S = 1.1    # ... and one from a jump
STEP_S = 0.4         # s a step away takes her (a guess: her walk over about 60 units): fight._not_thrown_at's price


@dataclass(frozen=True)
class Evidence:
    """What's known at a moment of a monster the look missed, for `chance`: `age` s since it was last seen as of the key;
    `state` its feed track's (tracking.LIVE, HIDDEN, DYING, DEAD), "gone" with a feed that no longer has it, None with no
    feed; her throws at it since it was seen: the last one showed nothing (`blank`) or something (`proven`); `gain`: her EXP
    up lately, a gain that can be its kind's; `short`: its track seen in SHORT_FRAMES frames or fewer; `spent`: the chance
    her throws at it have killed it as if her EXP said nothing; `blind`: frames the exact finder looked and missed it."""
    age: float
    state: str | None = None
    blank: int = 0
    proven: bool = False
    gain: bool = False
    short: bool = False
    spent: float = 0.0
    blind: int = 0


def log_odds(e: Evidence) -> float:
    """`chance`'s log-odds."""
    return (BASE + AGE_S * max(0.0, e.age)
            + (GONE if e.state == "gone" else DYING if e.state in ("dying", "dead") else 0.0)
            + (SHORT if e.short else 0.0)
            + (BLANK if e.blank > 0 else PROVEN if e.proven else 0.0)
            + (GAIN if e.gain else 0.0)
            + (SPENT if e.blank > 0 and e.spent >= SPENT_DEAD else 0.0)
            + BLIND * max(0, e.blind - BLIND_FREE))


def chance(e: Evidence) -> float:
    """How likely a throw at a monster the look missed shows an effect, on evidence `e`."""
    return 1.0 / (1.0 + math.exp(-log_odds(e)))


def landing(worth: float, left_hp: float, damage: float) -> float:
    """What a landing of an attack doing `damage` on average is worth at a monster worth `worth` (EXP and drops, in EXP)
    with `left_hp` left: the share of it that attack takes."""
    return worth * min(1.0, damage / max(1.0, left_hp))


def cost(mp: float, lock_s: float, rate: float) -> float:
    """What a throw costs in EXP: its `mp` at MP_EXP EXP each, and its lock's `lock_s` at her EXP a second `rate`."""
    return mp * MP_EXP + lock_s * rate


def net(p: float, worth: float, left_hp: float, damage: float, mp: float, lock_s: float, rate: float) -> float:
    """What a throw is worth in EXP, with chance `p` of landing (`landing`, `cost`)."""
    return p * landing(worth, left_hp, damage) - cost(mp, lock_s, rate)
