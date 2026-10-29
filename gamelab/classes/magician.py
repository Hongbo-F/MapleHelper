"""The magician: the user's second character (from 2026-10-27), to grind Tree Dungeon with Magic Claw (2001003).

What the game data says [data] (skills.json; Skill/200.img in Packs/Skill_00000.ms; String_000.wz Skill.img):
Magic Claw is info/type 1 with magicDamage 1, no ball, no lt/rb and no range node; each level 2 hits (attackCount) on 1
monster (mobCount), mad 45-65 and mpCon 10-20 (MAGIC_CLAW_MP); its tooltip "Consumes MP to attack a single enemy within
300 range in front twice"; its action swingO1-O3, 800 ms of frames each; its hit over the target 77-83 x 72-94 game px
for 570 ms; it needs Energy Bolt at level 1 (req). Magic Guard (2001000) moves x % of a touch to MP (MAGIC_GUARD_X), a
toggle ("Press the key again to deactivate"); Magic Armor (2001001) lasts 300-600 s (MAGIC_ARMOR_S).

What the client does with it (D:/MapleHelper-scratch/mage/range/magic-claw-range.md, from the v83 port RustMS and two
readings of this game's client, meowdb and ohmi69; it supersedes design.md 4.1's box): the client's search for an
aimed spell (SEARCH), from 50 ahead of her to 300 in bands 20 wide, the nearest band any monster's body overlaps
picking, never behind her nor in its first 50 ahead; nothing stops it, walls nor floors; from the ground only. The
user, 2026-10-27: walls and platforms don't block it, and it can't be cast mid-jump (settled, not to measure). L1
(a supervised attack window) only confirms the search.

Everything else is a PLACEHOLDER, each marked with what replaces it: the user's answer (her stat and skill windows,
her Key Config, her name, her pet, her MP policy: design.md section 7) or a live measurement (L0: her walk, jump, climb,
calibration and pet; L1: her lock, delays, MP a cast; design.md 6.1). Until then nothing hunts with her profile.
"""

from dataclasses import replace

from maplehelper import config
from maplehelper.config import GameKeys
from maplehelper.keys import parse as parse_key

from ..character import AttackSpec, Box, Buff, Character, Geometry, Perception, Pet, Policy, Priors, Search, Shape

# [data] skills.json's tables, by skill level from 1 (tests/test_character.py checks them against it)
MAGIC_CLAW_MP = (10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18, 19, 20)   # 2001003 mpCon
MAGIC_GUARD_X = (30, 33, 36, 39, 42, 49, 52, 55, 58, 61, 68, 71, 74, 77, 80)  # 2001000 x: % of a touch to her MP
MAGIC_GUARD_MP = (8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12)      # 2001000 mpCon
MAGIC_ARMOR_S = (300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 600)
MAGIC_ARMOR_MP = (8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 13, 13, 13, 13, 13, 16, 16, 16, 16, 16)  # 2001001 mpCon
ALERT_MS = 600  # a buff's cast plays alert2, 600 ms of frames [data: actions.json]; its lock: L0's casts

FIXED_STATS = config.Character(
    speed=100.0,        # her SPEED: her stat window, read live 2026-10-28 (lab/characters/magician/her_stats.json)
    jump=100.0,         # her JUMP: her stat window, the same read
    mp=1007.0,          # her MP when full: her stat window's MP 667/1007 at level 27 (the same read)
    lucky_seven=0.0,    # (not hers: a thief's skill)
                        # (her HP, level and W.DEF: perception's, as the thief's: behavior.bootstrap)
    avoid=18.0,         # her AVOID: her stat window's EVA 18 (the same read)
    climb_speed=117.0,  # PLACEHOLDER (the thief's, physics.CLIMB_SPEED): one climb in L0
    mp_regen=0.32)      # PLACEHOLDER (the thief's, sim.world.MP_REGEN): her idle MP bar in L0, or L1's quiet stretches
SKILLS = {"magic_claw": 20,   # PLACEHOLDER (its Master Level): her skill window -- the user's answer
          "energy_bolt": 1,   # PLACEHOLDER (Magic Claw's req [data]): her skill window
          "magic_guard": 0,   # PLACEHOLDER (not learned): her skill window
          "magic_armor": 0}   # PLACEHOLDER (not learned): her skill window

# Magic Claw's hit (each of its 2), before the monster's defence: a PRIOR by the v83 magic formula on her stat window of
# 2026-10-28 (level 27: INT 113, MAGIC 97; her_stats.json) with Magic Claw's mad at level 20 (65 [data]): the most
# ((MAGIC^2 / 1000 + MAGIC) / 30 + INT / 200) x mad = 267, the least with 10% mastery 76. What it's used for: what a
# monster has left as she reckons it (behavior.attacks: a monster likely dead isn't cast at, her bars and EXP set it
# right), never which attack goes (she casts Magic Claw whenever her MP pays for it).
MAGIC_CLAW_HIT = (76, 267)
# Her keys: the game's Key Bindings preset 2 as the user gave them (memory magician-facts, 2026-10-28): the thief's
# but Shift is Magic Claw, A Energy Bolt, her chair (The Green Relaxer) '-', S her stats ([characters.magician.keys] in
# config.toml replaces them; the binding read off the game's window, `gamelab keys --read`, is followed over them)
KEYS = GameKeys(left=parse_key("left"), right=parse_key("right"), jump=parse_key("alt"), attack=parse_key("ctrl"),
                up=parse_key("up"), down=parse_key("down"), pick_up=parse_key("z"), chair=parse_key("-"),
                sit=parse_key("x"), pet_food=parse_key("u"), magic_claw=parse_key("shift"),
                energy_bolt=parse_key("a"))
NAME = "RiceBaby"  # her name as it shows (her name tag: lab/characters/magician/tag_template_1366x768.npy)

# Magic Claw (an aimed spell: RustMS ShotPath.h:36-65, its test tests/shot_path.cpp:76 "so is Magic Claw"; no lt/rb,
# range or ball of its own [data]): the client's search with a spell's start. 13 bands, the last 290-300; band k's window
# 28 -/+ (50 + 20 k) // 4 above her feet: 16..40 at 50-70 ahead, -44..100 at 290-300 (range doc 1.3). The order inside
# a band isn't settled (map-entry order in the port): any in the first band alike, as reach.take_chances has stars.
SEARCH = Search(start=50,   # ShotPath.h:33 SPELL_REACH; meowdb starts.magic 50 ("STARTS 50 PX AHEAD")
                range=300)  # ShotPath.h:34 SPELL_RANGE; meowdb rc 300 "Game default"; ohmi69 client_default 300; the
                            # String tooltip's "within 300 range in front" [data]
CAST = Geometry("search", search=SEARCH,
                air=None,     # from the ground only: settled by the user, 2026-10-27 (meowdb "Cast position: Ground
                              # only"); never keyed in the air
                line=None,    # nothing stops it, walls nor floors: settled by the user, 2026-10-27 (no ball: the port
                              # checks a line only for a bullet, Combat.cpp:1382; meowdb "Ignored")
                order="band", mob_count=1)
# Her regular attack swings her wand up close [GK]: a box from her weapon's afterimage (Character/Afterimage/
# <info.afterImage>.img, swingO1-O3's lt/rb, as attack.reach.CLAW_BOXES came from swordOL.img) once her weapon is known
# (mace.img's swingO1-O3 go 94 ahead at most: range doc 1.1).
WAND_BOX = Box(near=0.0, far=65.0, down=10.0, up=60.0)  # PLACEHOLDER: her weapon (the user's answer), its afterimage
_TIMES = dict(lock_ms=730,              # PLACEHOLDER (the thief's measured 730 with the same swingO1-3 frames [data];
                                        # the port's 800, meowdb's 810 shown) -- L1's `lock magic_claw`
              lock_extra_ms=(60, 160),  # PLACEHOLDER (the thief's margins, control.actions) -- L1
              arrow_lock_ms=800,        # PLACEHOLDER (the thief's ARROW_LOCK_MS) -- L1's walk-off after a cast
              release_ms=515,           # PLACEHOLDER: no flight, so the hit itself (lands_ms) -- L1's 60 fps recording
              damage_ms=290,            # PLACEHOLDER (the server's, likely the same [GK]) -- perceive's EXP gain and
                                        # bar drop against the key (gain_delay.py, bar_delay.py)
              lands_ms=515,             # PLACEHOLDER (the port's rule: its damage 450 ms after the attack starts, at
                                        # mace.img's frame '2', and the attack about 65 ms after the key, as stars;
                                        # range doc 5): key to the hit on the target -- L1's 60 fps recording
              speed=None,               # no flight (no ball [data])
              cycle_s=0.79,             # PLACEHOLDER: the lock and its 60 ms margin -- L1's lock
              miss=0.275,               # PLACEHOLDER (the thief's star miss): her accuracy against each kind's eva and
                                        # level (mobs.json) -- L1/L2's casts with no bar drop
              hidden_s=1.8,             # PLACEHOLDER (the thief's): under her bigger hit (83x94, 570 ms) -- L1
              air=False)                # from the ground only (CAST.air: settled)
MAGIC_CLAW = AttackSpec("magic_claw", "magic_claw", "Magic Claw", "casting", 2001003, level=SKILLS["magic_claw"],
                        hits=2,                 # attackCount [data]: both on the one monster, together
                        lo=MAGIC_CLAW_HIT[0], hi=MAGIC_CLAW_HIT[1],  # PLACEHOLDER (the formula's prior: MAGIC_CLAW_HIT)
                                                # -- fit on her live HP bars (L2)
                        defence="mdd",          # magicDamage [data]: against the monster's MDDamage
                        mp=float(MAGIC_CLAW_MP[SKILLS["magic_claw"] - 1]),  # mpCon at her level [data]
                        ammo=0, geometry=CAST, **_TIMES)
WAND = AttackSpec("wand", "attack", "the regular attack", "swinging", None, level=None, hits=1,
                  lo=None, hi=None,             # PLACEHOLDER: her weapon -- the user's answer, a fit
                  defence="pdd", mp=0.0, ammo=0,
                  geometry=Geometry("melee", box=WAND_BOX, air=None, line=None, order="nearest"), **_TIMES)

# Her Shape (what localization takes of her attacks), from the search's outer bound with the thief's slack: a monster's
# middle from 6 ahead of hers (its far side past 50, the widest body's half 34 back from it, and the thief's 10 behind for
# what the tracker gets wrong) to 374 (its near side short of 300, the half, and the thief's 40 ahead); its feet up to 110
# over hers (short of the windows' 100 roof, strict, and 10 of slack: tracking.THROW_DY's over the star's 60) and 119
# under them (its top over the 44 floor: the tallest body's 65, an Orange Mushroom flinching, and the 10).
WIDEST_HALF = 34.0  # localization.monsters.MONSTER_HALF's widest (a pig's)
TALLEST_TOP = 65.0  # monsters.MONSTER_FLINCH_TOP's tallest (an Orange Mushroom hit): what reaches up into the floor
SHAPE = Shape(near=SEARCH.start - WIDEST_HALF - 10.0,           # 6
              ahead=SEARCH.range + WIDEST_HALF + 40.0,           # 374
              up=SEARCH.roof + 10.0,                             # 110
              down=-SEARCH.floor + TALLEST_TOP + 10.0,           # 119
              hit_after_s=MAGIC_CLAW.lands_ms / 1000 + 0.2,      # PLACEHOLDER 0.715 (no flight; 0.2 of slack) -- L1
              window_s=1.5,                                      # PLACEHOLDER (design.md: about 1-1.5 s) -- L1
              hide_s=3.5,                                        # PLACEHOLDER (the thief's) -- L1: the detector under
                                                                 # her hit
              gain_after_key=(0.1, 0.6),                         # PLACEHOLDER (the server's, likely the same) --
                                                                 # perceive's gains
              pick_s=0.065,                                      # PLACEHOLDER (the stars' key-to-pick, the same input
                                                                 # path: range doc 5) -- L1
              order="band",
              look_band=(SEARCH.roof + 80, -SEARCH.floor + TALLEST_TOP + 15),  # (180, 124): attack.reach.LOOK_UP's 80
                                                                 # and LOOK_DOWN's 15 over the feet it picks
              shows_wait_s=0.45)                                 # PLACEHOLDER (the thief's: the same lock prior) --
                                                                 # L1's lock
POLICY = Policy(default_mode="skill",     # Magic Claw whenever her MP pays for a cast (design.md 4.6)
                finish_with_regular=False,  # her regular attack is a swing up close [GK], not a star at range
                regular_needs_box=True,   # PLACEHOLDER: the wand only at a monster in its box -- the user's answer
                                          # (ever, only at a monster on her, or never)
                unseen="none",            # nothing cast at nothing: her MP is her attack
                close_quarters="touch",   # no point-blank failure: one too close isn't picked (the search's first 50),
                                          # the danger's escape only (range doc 7)
                mp_floor=0.0)             # PLACEHOLDER: the MP she keeps back, and what under it -- the user's MP policy
PET = Pet(hp_warning=0.55, mp_warning=0.05, hp_heal=250.0, mp_heal=185.0)  # hp_warning live (2026-10-29: her pet drank
                                          # at 359 and 335 of 628, 57% and 53%); the rest PLACEHOLDER (the thief's pet): her pet,
                                          # its auto-potion thresholds and potions -- the user's answer; `pet --read` (L0)
PRIORS = Priors(kill_s_per_exp=0.085, danger_hp_s=0.0175, hp_price=100.0, ground_danger=2.0, hp_exp=250.0,
                seen_share=0.3, threat_s=1.7, keep_danger=0.08, meet_s=8.0)  # PLACEHOLDER (the thief's): re-tuned in
                                          # the simulator with her HP and DEF (S7), then her live kill times (L2)
# The safety stop's reserve: three of the map's largest touches (behavior.safety: worked out live, at her level and DEF as
# perception reads them, over the map's kinds). The user, 2026-10-28: "Well for thief with high evasion this should be 2
# hits, 3 for magician". Where that's at or over her pet's warning at her max HP (her HUD's), a command that hunts
# refuses the map (TooStrong). For now 2 (the user, 2026-10-29 ~00:01, asked whether the stop or her pet's threshold
# should change, her pet drinking only at ~55% and 3 Horny touches 65% of her 628: "Stop at 2 touches for now").
SAFETY_HITS = 2
PLACEHOLDERS = ("skill levels", "lock and delays", "damage", "miss", "the wand", "pet", "priors")


def _at(table: tuple, level: int) -> float:
    return float(table[min(max(int(level), 1), len(table)) - 1])


def profile(cid: str = "magician") -> Character:
    """A magician as the design's priors have her: her files under lab/characters/<cid>/ (paths.of)."""
    return Character(id=cid, klass="magician", stats=FIXED_STATS, fixed_stats=FIXED_STATS, skills=dict(SKILLS),
                     keys=KEYS,           # the user's preset 2 ([characters.<id>.keys] over it)
                     timing=None, ammo=None, stops=("out_of_mp",), buffs=(), magic_guard=0.0,
                     attacks={"skill": MAGIC_CLAW, "regular": WAND}, pet=PET, policy=POLICY, priors=PRIORS,
                     shape=SHAPE,
                     perception=Perception(files=cid,
                                           job="Magician",  # PLACEHOLDER: her bottom bar's job line -- the user's
                                           name=NAME,
                                           heading_box=(50, 8, 16),  # PLACEHOLDER (the thief's): L0's frames of her,
                                           attack_score=0.4,         # widened if her swing leaves it; refit on her
                                           attack_margin=0.2,        # cast frames (L1)
                                           hit_effect=(83, 94, 570)),  # Magic Claw's hit [data]
                     placeholders=PLACEHOLDERS, safety_hits=SAFETY_HITS)


def configured(c: Character, table: config.CharacterConfig) -> Character:
    """Her [characters.<id>] tables over her class's: her lines, keys, timing, skill levels, pet, and the buffs kept up
    (Magic Guard's share of a touch to her MP at her level while it's kept up). Her stats are gamelab.character.load's."""
    skills = {**c.skills, **table.skills}
    claw = c.attacks["skill"]
    claw = replace(claw, level=skills["magic_claw"], mp=_at(MAGIC_CLAW_MP, skills["magic_claw"]))
    buffs = []
    if table.buffs.get("magic_guard") and skills.get("magic_guard"):
        lv = skills["magic_guard"]
        buffs.append(Buff("magic_guard", "magic_guard", 2001000, lv, _at(MAGIC_GUARD_MP, lv), None, ALERT_MS))
    if table.buffs.get("magic_armor") and skills.get("magic_armor"):
        lv = skills["magic_armor"]
        buffs.append(Buff("magic_armor", "magic_armor", 2001001, lv, _at(MAGIC_ARMOR_MP, lv), _at(MAGIC_ARMOR_S, lv),
                          ALERT_MS))
    guard = _at(MAGIC_GUARD_X, skills["magic_guard"]) / 100 if any(b.id == "magic_guard" for b in buffs) else 0.0
    pet = replace(c.pet, **table.pet)
    return replace(c, skills=skills, keys=table.keys or c.keys, timing=table.timing or c.timing, buffs=tuple(buffs),
                   magic_guard=guard, attacks={**c.attacks, "skill": claw}, pet=pet,
                   perception=replace(c.perception, job=table.job or c.perception.job,
                                      name=table.name or c.perception.name))
