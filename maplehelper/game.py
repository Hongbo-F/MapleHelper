"""What we know about MapleStory Classic World for her character, measured in the game with gamelab.

Measured 2026-10-18 on BugBaby (Rogue, level 19, claw, Lucky Seven level 4) in Victoria Road:
Thicket Around the Beach II, with the game window at 1920x1080. Distances are game-window pixels;
times are milliseconds from when the key goes down. Re-measure with gamelab after gear, level,
speed buffs or window size change (see gamelab/README.md).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Movement:
    reaction_ms: int = 100      # key down -> she visibly starts moving (input + frame latency)
    walk_px_per_ms: float = 0.209  # ~209 px/s once moving
    walk_start_px: int = 12     # holding an arrow for t ms moves her ~ walk_start_px + walk_px_per_ms * t
                                # (300 ms -> 75 px, 600 ms -> 146 px, 1000 ms -> 221 px)
    stop_slide_ms: int = 200    # she keeps sliding this long after the arrow is released
    tap_turn_ms: int = 50       # an arrow tap this short turns her (or stands her up) without walking
    jump_airtime_ms: int = 530  # takeoff to landing on flat ground
    jump_peak_ms: int = 260     # takeoff to the top of the jump
    jump_height_px: int = 104   # top of a jump above takeoff
    jump_length_px: int = 120   # sideways distance of a jump taken while walking
    # Ledges: platforms can be jumped through from below. To land on one h px higher she must still
    # be h px up when she reaches its edge (see actions.ledge_takeoff_px); from ~140 px away an
    # 86 px ledge was missed, and walking toward a ledge can walk her off her own platform first.
    climb_px_per_ms: float = 0.165  # rope/ladder climbing, up or down (approximate: the camera scrolled)
    fall_damage: bool = False   # walking off a ledge just drops her to the ground below


@dataclass(frozen=True)
class Combat:
    attack_release_ms: int = 530  # attack key -> star leaves / claw swings (regular attack and Lucky Seven)
    attack_lock_ms: int = 730     # attack key -> next jump or attack is accepted; presses before are ignored
                                  # (not queued), and holding a key through the lock doesn't help either
    melee_when_close: bool = True  # regular attack swipes with the claw if a monster is right next to her,
                                   # and throws a star otherwise; Lucky Seven always throws
    hit_swallows_input: bool = True  # while monsters keep hitting her, presses get lost: 4 Lucky Sevens
                                     # used no MP, and a rope grab was knocked off. Get clear first.
    lucky_seven_mp: int = 8       # level 4 (level 5: 9 MP, 76%)
    lucky_seven_damage_pct: int = 72
    lucky_seven_range: int = 200  # game units (tooltip): hits one enemy in front within this range
    lucky_seven_reach_px: int = 375  # observed: killed a snail ~375 window px away, level with her,
                                     # with a platform in between (stars aren't blocked by platforms)
    lucky_seven_stars: int = 2


@dataclass(frozen=True)
class Recovery:
    chair_bonus_hp_per_10s: int = 30  # The Green Relaxer's tooltip: "an additional 30 HP every 10 seconds"
    on_chair_hp_per_10s: int = 50     # in total while sitting on it (reported by her player; not measured)
    meat_hp: int = 160            # Meat (R): "Restores around 160 HP"
    blue_potion_mp: int = 200     # Blue Potion (T): "Restores 200 MP"


MOVEMENT = Movement()
COMBAT = Combat()
RECOVERY = Recovery()


# Her in-game key bindings, read from the Key Bindings window (press \ in game), preset 1.
KEY_BINDINGS = {
    "left": "move left", "right": "move right", "up": "move up (climb)", "down": "move down (climb / lie down)",
    "alt": "jump", "f": "attack", "g": "skill: Lucky Seven", "z": "pick up", "x": "sit",
    "space": "NPC chat",
    "d": "item: The Green Relaxer (chair)", "r": "item: Meat", "t": "item: Blue Potion", "u": "item: Pet Food",
    "esc": "menu", "\\": "key bindings", "q": "quest", "w": "world map", "e": "equipment", "i": "item inventory",
    "s": "ability", "k": "skills", "l": "crafting", "m": "mini map", "o": "buddy", "p": "party",
    "h": "whisper", "[": "maple chat", "]": "quick slot", ";": "mailbox", "'": "chat+", ",": "channel",
    "8": "chat to buddy", "9": "chat to party", "0": "chat to all",
    "f1": "emote", "f2": "emote", "f3": "emote", "f4": "emote", "f5": "emote", "f6": "emote", "f7": "emote",
    "f9": "cash shop", "f10": "shortcut", "f11": "capture mode", "scrolllock": "screenshot",
}


@dataclass(frozen=True)
class Monster:
    name: str
    level: int | None        # from its "Lv.N Name" label (shown with an HP bar once it's hit)
    exp: int | None          # EXP she gained per kill
    observed: str            # what fighting it was like
    drops_seen: tuple[str, ...] = ()


MONSTERS = {
    "snail": Monster("Snail", None, 2, "died to 3 regular attacks or fewer (claw swipe up close)"),
    "slime": Monster("Slime", 6, 12, "died to ~3 Lucky Sevens, or a few hits plus 2 claw swipes; it backs off "
                     "when hit and comes back", ("Squishy Liquid", "Red Potion", "mesos (13)")),
    "pig": Monster("Pig", 7, 13, "bumps into her for 10-16 damage every few seconds; 2-3 Lucky Sevens kill one; "
                   "several at once pinned her, and they wander into her path and knock her off ropes"),
}

# The Lie Detector: the game's anti-bot check. After ~10 minutes of gamelab driving her (fighting and
# looting on the lower ground), a "LIE DETECTOR" window appeared: "13 sec until the Lie Detector
# mini-game starts. Please find the beautiful Violetta wearing make-up... if you do not enter any
# answers, you will fail the test!" A person has to answer it; gamelab watches for it, stops all
# input, pages, and shuts down (gamelab/alarm.py). Keep automated sessions short and supervised.

# Reading the screen:
# - Violet numbers over her are damage she took; a violet MISS means a monster missed her.
# - Monsters show "Lv.N Name" and an HP bar once damaged.
# - The bottom-right log reports EXP ("You received EXP (+12)") and loot ("Red Potion x1 earned (Use)").

# How things work, as observed:
# - Ropes: stand still right under one and jump with up held; she grabs it mid-air (with sideways
#   momentum she sails past it). Keep holding up to climb; at the top she steps onto the platform.
#   Hold down to climb down; at the bottom end she lets go and drops. Down on the ground = lie down.
# - Loot: walk onto the drop and press pick up (z) once per item. The bottom-right log confirms it
#   ("Squishy Liquid x1 earned (Etc)", "You have gained mesos (+13)").
# - Chair (d): she sits on The Green Relaxer; tapping an arrow stands her up without moving her.
# - Kills show up as EXP on the HUD (e.g. 423 -> 435 for a Green Slime).
# - Moving between platforms (gamelab/route.py, gamelab/moves.py; map layouts come from the game data):
#   jumping straight up passes through a platform overhead and lands on it (a jump rises 77 map
#   units); down + jump drops through the platform she stands on; down at a rope's top climbs down it.
#   Running jumps need a run-up: from 12 units before a platform's end she walked off before the jump
#   started, from ~35-40 back they land. A platform's real end is ~24 screen px past where its drawn
#   grass seems to end. Monsters stay on the platform they spawn on, so platforms without spawns are
#   safe to rest on.
# - The screen draws the map 1.40625x (1080/768) at a 1920x1080 window; the minimap is the map at
#   1/16, so one minimap pixel is 11.4 map units.
