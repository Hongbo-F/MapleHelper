"""The magician: the user's second character (from 2026-10-27), to grind Tree Dungeon with Magic Claw (2001003).

What the game data says [data] (skills.json; Skill/200.img in Packs/Skill_00000.ms; String_000.wz Skill.img):
Magic Claw is info/type 1 with magicDamage 1, no ball, no lt/rb and no range node; each level 2 hits (attackCount) on 1
monster (mobCount), mad 45-65 and mpCon 10-20 (MAGIC_CLAW_MP); its tooltip "Consumes MP to attack a single enemy within
300 range in front twice"; its action swingO1-O3, 800 ms of frames each; its hit over the target 77-83 x 72-94 game px
for 570 ms; it needs Energy Bolt at level 1 (req). Magic Guard (2001000) moves x % of a touch to MP (MAGIC_GUARD_X), a
toggle ("Press the key again to deactivate"); Magic Armor (2001001) lasts 300-600 s (MAGIC_ARMOR_S).

Everything else is a PLACEHOLDER, each marked with what replaces it: the user's answer (her stat and skill windows,
her Key Config, her name, her pet, her MP policy: design.md section 7) or a live measurement (L0: her walk, jump, climb,
calibration and pet; L1, a supervised attack window: her lock, delays, MP a cast and the box's edges; design.md 6.1).
Until then her profile is the design's priors (design.md 4.1-4.6), and nothing hunts with it yet.
"""

from dataclasses import replace

from maplehelper import config

from ..character import AttackSpec, Box, Buff, Character, Geometry, Perception, Pet, Policy, Priors, Shape

# [data] skills.json's tables, by skill level from 1 (tests/test_character.py checks them against it)
MAGIC_CLAW_MP = (10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18, 19, 20)   # 2001003 mpCon
MAGIC_GUARD_X = (30, 33, 36, 39, 42, 49, 52, 55, 58, 61, 68, 71, 74, 77, 80)  # 2001000 x: % of a touch to her MP
MAGIC_GUARD_MP = (8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12)      # 2001000 mpCon
MAGIC_ARMOR_S = (300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 600)
MAGIC_ARMOR_MP = (8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 13, 13, 13, 13, 13, 16, 16, 16, 16, 16)  # 2001001 mpCon
ALERT_MS = 600  # a buff's cast plays alert2, 600 ms of frames [data: actions.json]; its lock: L0's casts

FIXED_STATS = config.Character(
    speed=100.0,        # PLACEHOLDER (the game's 100, no gear or Haste): her SPEED -- her stat window; L0's walk
    jump=100.0,         # PLACEHOLDER: her JUMP -- her stat window; L0's jump (77 units at 100)
    hp=400.0,           # PLACEHOLDER: her HP when full -- her HP bar's number
    level=25.0,         # PLACEHOLDER (design.md 4.7's example): her level -- the user's answer
    mp=500.0,           # PLACEHOLDER: her MP when full -- her MP bar's number
    lucky_seven=0.0,    # (not hers: a thief's skill)
    defence=75.0,       # PLACEHOLDER (the thief's fit, attack.danger.DEFENCE): her W.DEF -- her stat window, or a fit
                        # from her touches (L1, L2)
    avoid=27.0,         # PLACEHOLDER (the thief's fit, sim.world.EVADE_ACC): her AVOID -- her stat window
    climb_speed=117.0,  # PLACEHOLDER (the thief's, physics.CLIMB_SPEED): one climb in L0
    mp_regen=0.32)      # PLACEHOLDER (the thief's, sim.world.MP_REGEN): her idle MP bar in L0, or L1's quiet stretches
SKILLS = {"magic_claw": 20,   # PLACEHOLDER (its Master Level): her skill window -- the user's answer
          "energy_bolt": 1,   # PLACEHOLDER (Magic Claw's req [data]): her skill window
          "magic_guard": 0,   # PLACEHOLDER (not learned): her skill window
          "magic_armor": 0}   # PLACEHOLDER (not learned): her skill window

# Magic Claw picks the nearest monster whose body overlaps a box in front of her [GK: HeavenClient's find_closest; the
# data has no ball, unlike Energy Bolt's], hit twice; no star flies, no claw swipes up close.
BOX = Box(near=-10.0,   # PLACEHOLDER [GK]: point blank and a body over her are hit -- L1's near-edge probe
          far=300.0,    # its tooltip's 300 [data]; whether to her middle or the body's near edge -- L1's far-edge probe
          down=50.0,    # PLACEHOLDER (Thunder Bolt's lt/rb (-150,-50)-(150,50), the one magician box in this data):
          up=50.0)      # ... how far under and over her feet -- L1's probes at the platforms below and above
CAST = Geometry("box", box=BOX,
                air=None,     # PLACEHOLDER: the prior is "same" (the box at her pose at the pick, design.md 4.3); no
                              # jump casts until L1's (keyed 40, 100 and 200 ms after the jump key) show it
                line=None,    # PLACEHOLDER [GK]: walls and floors stop nothing (magicDamage, no ball) -- L1's casts
                              # across a tower wall and onto the platform under her ("v83" if they stop it)
                order="nearest", mob_count=1)
# Her regular attack swings her wand up close [GK]: a box from her weapon's afterimage (Character/Afterimage/
# <info.afterImage>.img, swingO1-O3's lt/rb, as attack.reach.CLAW_BOXES came from swordOL.img) once her weapon is known.
WAND_BOX = Box(near=0.0, far=65.0, down=10.0, up=60.0)  # PLACEHOLDER: her weapon (the user's answer), its afterimage
_TIMES = dict(lock_ms=730,              # PLACEHOLDER (the thief's measured 730: the same swingO1-3 frames [data]) --
                                        # L1's `lock magic_claw`
              lock_extra_ms=(60, 160),  # PLACEHOLDER (the thief's margins, control.actions) -- L1
              arrow_lock_ms=800,        # PLACEHOLDER (the thief's ARROW_LOCK_MS) -- L1's walk-off after a cast
              release_ms=530,           # PLACEHOLDER (the thief's: when her claw swings) -- L1's 60 fps recording
              damage_ms=290,            # PLACEHOLDER (the server's, likely the same [GK]) -- perceive's EXP gain and
                                        # bar drop against the key (gain_delay.py, bar_delay.py)
              lands_ms=520,             # PLACEHOLDER (about the thief's, no flight after it): key to the hit on the
                                        # target -- L1's 60 fps recording
              speed=None,               # no flight (no ball [data])
              cycle_s=0.79,             # PLACEHOLDER: the lock and its 60 ms margin -- L1's lock
              miss=0.275,               # PLACEHOLDER (the thief's star miss): her accuracy against each kind's eva and
                                        # level (mobs.json) -- L1/L2's casts with no bar drop
              hidden_s=1.8,             # PLACEHOLDER (the thief's): under her bigger hit (83x94, 570 ms) -- L1
              air=False)                # PLACEHOLDER: as CAST.air
MAGIC_CLAW = AttackSpec("magic_claw", "magic_claw", "Magic Claw", "casting", 2001003, level=SKILLS["magic_claw"],
                        hits=2,                 # attackCount [data]
                        lo=None, hi=None,       # PLACEHOLDER (not known): by the magic formula from her INT and magic
                                                # attack (the user's answer) and mad (45-65 [data]); fit on L1's HP bars
                        defence="mdd",          # magicDamage [data]: against the monster's MDDamage
                        mp=float(MAGIC_CLAW_MP[SKILLS["magic_claw"] - 1]),  # mpCon at her level [data]
                        ammo=0, geometry=CAST, **_TIMES)
WAND = AttackSpec("wand", "attack", "the regular attack", "swinging", None, level=None, hits=1,
                  lo=None, hi=None,             # PLACEHOLDER: her weapon -- the user's answer, a fit
                  defence="pdd", mp=0.0, ammo=0,
                  geometry=Geometry("melee", box=WAND_BOX, air=None, line=None, order="nearest"), **_TIMES)

WIDEST_HALF = 34.0  # localization.monsters.MONSTER_HALF's widest (a pig's): a body's middle past the box's edge
SHAPE = Shape(ahead=BOX.far + WIDEST_HALF + 40.0,       # 374: the box, the widest body's half, and the thief's 40 for
                                                       # what she and it move between the frame and the pick
              behind=-BOX.near + WIDEST_HALF,           # 44
              dy=max(BOX.up, BOX.down) + 10.0,          # 60: tracking.THROW_DY's 10 of slack over the box
              hit_after_s=MAGIC_CLAW.lands_ms / 1000 + 0.2,  # PLACEHOLDER 0.72 (no flight; 0.2 of slack) -- L1
              window_s=1.5,                             # PLACEHOLDER (design.md: about 1-1.5 s, no flight) -- L1
              hide_s=3.5,                               # PLACEHOLDER (the thief's) -- L1: the detector under her hit
              gain_after_key=(0.1, 0.6),                # PLACEHOLDER (the server's, likely the same) -- perceive's gains
              order="nearest",
              look_band=(BOX.up + 80, BOX.down + 15),   # (130, 65): attack.reach.LOOK_UP/DOWN's slack over the box
              shows_wait_s=0.45)                        # PLACEHOLDER (the thief's: the same lock prior) -- L1's lock
POLICY = Policy(default_mode="skill",     # Magic Claw whenever her MP pays for a cast (design.md 4.6)
                finish_with_regular=False,  # her regular attack is a swing up close [GK], not a star at range
                regular_needs_box=True,   # PLACEHOLDER: the wand only at a monster in its box -- the user's answer
                                          # (ever, only at a monster on her, or never)
                unseen="none",            # nothing cast at nothing: her MP is her attack
                close_quarters="touch",   # no point-blank failure: the danger's escape only (design.md 4.1)
                mp_floor=0.0)             # PLACEHOLDER: the MP she keeps back, and what under it -- the user's MP policy
PET = Pet(hp_warning=0.70, mp_warning=0.05, hp_heal=250.0, mp_heal=185.0)  # PLACEHOLDER (the thief's pet): her pet,
                                          # its auto-potion thresholds and potions -- the user's answer; `pet --read` (L0)
PRIORS = Priors(kill_s_per_exp=0.085, danger_hp_s=0.0175, hp_price=100.0, ground_danger=2.0, hp_exp=250.0,
                seen_share=0.3, threat_s=1.7, keep_danger=0.08, meet_s=8.0)  # PLACEHOLDER (the thief's): re-tuned in
                                          # the simulator with her HP and DEF (S7), then her live kill times (L2)
PLACEHOLDERS = ("stats", "skill levels", "keys", "name", "the box's edges", "air", "walls", "lock and delays",
                "damage", "miss", "pet", "MP policy", "priors")


def _at(table: tuple, level: int) -> float:
    return float(table[min(max(int(level), 1), len(table)) - 1])


def profile(cid: str = "magician") -> Character:
    """A magician as the design's priors have her: her files under lab/characters/<cid>/ (paths.of)."""
    return Character(id=cid, klass="magician", stats=FIXED_STATS, fixed_stats=FIXED_STATS, skills=dict(SKILLS),
                     keys=None,           # PLACEHOLDER: her Key Config -- [characters.<id>.keys], the user's answer
                     timing=None, ammo=None, stops=("out_of_mp",), buffs=(), magic_guard=0.0,
                     attacks={"skill": MAGIC_CLAW, "regular": WAND}, pet=PET, policy=POLICY, priors=PRIORS,
                     shape=SHAPE,
                     perception=Perception(files=cid,
                                           job="Magician",  # PLACEHOLDER: her bottom bar's job line -- the user's
                                           name=None,       # PLACEHOLDER: her name as it shows -- the user's answer
                                           heading_box=(50, 8, 16),  # PLACEHOLDER (the thief's): L0's frames of her,
                                           attack_score=0.4,         # widened if her swing leaves it; refit on her
                                           attack_margin=0.2,        # cast frames (L1)
                                           hit_effect=(83, 94, 570)),  # Magic Claw's hit [data]
                     placeholders=PLACEHOLDERS)


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
