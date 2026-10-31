"""The thief: BugBaby, a Rogue, with Lucky Seven and her claw's throwing stars (her HP, level and DEF are
perception's: behavior.bootstrap).

Every value here is a module constant of the stack's, the place it's measured and used: her code reads it there, as it
did before profiles. Here it describes her, and tests/test_character.py checks each against the constant named beside
it, so the two can't drift (moving a constant into the profile is a later change of its own). SHAPE is the one the
localization trackers take (THIEF_SHAPE, their default): equal to their constants, so they act as before.
"""

from dataclasses import replace

from maplehelper import config

from ..character import AttackSpec, Character, Geometry, Perception, Pet, Policy, Priors, Search, Shape

FIXED_STATS = config.Character()  # her stats as measured (config.Character's defaults), what FIXED_STATS pins

# Her star search, as attack.reach has it: its bands (order), the v83 line (reach.LINE: walls and floors stop it), its
# air search (reach.AIR_MODEL); the regular attack swipes with the claw up close (reach.CLAW), Lucky Seven always throws
# (maplehelper.game.COMBAT.melee_when_close).
STAR_SEARCH = Geometry("star_search",
                       search=Search(start=65,    # attack.reach.SEARCH_START (its step, spread and row SEARCH_STEP,
                                     range=400),  # SEARCH_SPREAD, SHOT_H); STAR_RANGE
                       air="search", line="v83", order="band")
_TIMES = dict(lock_ms=730,                  # maplehelper.game.COMBAT.attack_lock_ms
              lock_extra_ms=(60, 160),      # control.actions.THROW_LOCK_EXTRA_MS, JUMP_THROW_LOCK_EXTRA_MS
              arrow_lock_ms=800,            # control.actions.ARROW_LOCK_MS
              release_ms=530,               # COMBAT.attack_release_ms
              damage_ms=290,                # control.actions.DAMAGE_SHOWS_MS
              lands_ms=520,                 # control.actions.STAR_DELAY_MS (the star leaves; it flies on)
              speed=690.0,                  # attack.reach.STAR_SPEED
              cycle_s=0.84,                 # attack.choice.THROW_S (behavior.patrol.THROW_S)
              miss=0.275,                   # attack.choice.MISS
              hidden_s=1.8,                 # behavior.fight.HIDDEN_FOR (behavior.attacks.HIDDEN_FOR)
              air=True)
LUCKY_SEVEN = AttackSpec("lucky_seven", "lucky_seven", "Lucky Seven", "throwing", 4001003,
                         level=16,          # attack.choice.SKILL_LEVEL (config.toml [character] lucky_seven)
                         hits=2,            # choice.SKILL_STARS
                         lo=96, hi=192,     # choice.SKILL_STAR
                         defence="pdd",
                         mp=14.0,           # choice.cast_mp(16) (LUCKY_SEVEN_MP: skills.json's 4001003)
                         ammo=2, geometry=STAR_SEARCH, **_TIMES)
# [data] skills.json's 4001003 by level from 1 (attack.choice.LUCKY_SEVEN_MP, LUCKY_SEVEN_DAMAGE): its MP a cast, and its
# damage (%), which scales the stars measured at level 16 (choice.lucky_seven)
LUCKY_SEVEN_MP = (8, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16)
LUCKY_SEVEN_DAMAGE = (60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 140)
STAR = AttackSpec("star", "attack", "the regular attack", "throwing", None, level=None,
                  hits=1, lo=20, hi=140,    # choice.STAR (choice.REGULAR)
                  defence="pdd", mp=0.0, ammo=1, geometry=replace(STAR_SEARCH, point_blank="claw"), **_TIMES)

SHAPE = Shape(near=-10.0,                  # tracking.MonsterTracker._in_path's -10 (memory's 0)
              ahead=434.0 + 40.0,           # localization.tracking.THROW_REACH
              up=70.0,                      # tracking.THROW_DY, above her feet ...
              down=70.0,                    # ... and below
              hit_after_s=1.2,              # tracking.HIT_AFTER_S
              window_s=2.5,                 # tracking.STAR_WINDOW (behavior.ledger.STAR_WINDOW)
              hide_s=3.5,                   # tracking.HIDE_S
              gain_after_key=(0.1, 0.6),    # localization.monsters.GAIN_AFTER_KEY
              pick_s=0.065,                 # motion.physics.JUMP_LAG (an attack picks when it starts, about that after
                                            # its key)
              order="line",
              look_band=(140, 75),          # attack.reach.LOOK_UP, LOOK_DOWN
              shows_wait_s=0.45)            # localization.senses.SHOWS_WAIT_S
PERCEPTION = Perception(files="",          # paths' own names (paths.of)
                        job="Rogue", name="BugBaby",  # her bottom bar (behavior.pet.NAME_BOX's comment)
                        heading_box=(50, 8, 16),      # perception.heading.UP, DOWN, HALF
                        attack_score=0.4, attack_margin=0.2,  # heading.ATTACK_SCORE, ATTACK_MARGIN
                        hit_effect=(44, 42, None))    # Lucky Seven's hit over a monster, game px (its time not read)
POLICY = Policy(default_mode="choice",      # behavior.hunter's: Lucky Seven or the regular attack (attack.choice)
                finish_with_regular=True,   # the user's rule (memory: regular-attack-only)
                regular_needs_box=False,
                unseen="regular",           # behavior.hunt.UNSEEN_ATTACK
                close_quarters="claw",      # attack.stance's claw-edge rules
                mp_floor=0.0)               # a cast whenever it's paid for (attack.choice.can_cast)
# The safety stop's reserve: this many of the map's largest touches (behavior.safety: worked out live, at her level and
# DEF as perception reads them, over the map's kinds). The user's rule, 2026-10-28: "Well for thief with high evasion
# this should be 2 hits, 3 for magician". Three for now, the user's decision of 2026-10-28 ~15:30 ("3 hits for now"):
# her retreat after a stop doesn't dodge yet (her way to a safe place keeps clear of monsters only by waiting them out:
# the simulator with her pet out of potions lost her on it in 7-8 of 40 runs as live at 2 hits, 2-5 at 3; behavior.
# safety). Back to 2 once her retreat dodges (another workflow's).
SAFETY_HITS = 3
PET = Pet(hp_warning=0.70,                  # sim.world.PET_HP (behavior.safety: her pet at 70%)
          mp_warning=0.05,                  # sim.world.PET_MP
          hp_heal=250.0,                    # sim.world.HP_POTION
          mp_heal=185.0)                    # sim.world.MP_POTION
PRIORS = Priors(kill_s_per_exp=0.085,       # behavior.patrol.KILL_S_PER_EXP
                danger_hp_s=0.0175,         # patrol.DANGER_HP_S
                hp_price=250.0,             # patrol.HP_PRICE
                ground_danger=2.0,          # behavior.hunt.GROUND_DANGER
                hp_exp=250.0,               # hunt.HP_EXP
                seen_share=0.3,             # behavior.tour.SEEN_SHARE
                threat_s=1.7,               # attack.danger.THREAT_S
                keep_danger=0.08,           # danger.KEEP_DANGER
                meet_s=8.0)                 # control.navigator.MEET_S


def profile(cid: str = config.THIEF) -> Character:
    """The thief as the stack has her (her stats are config.toml's [character]: gamelab.character.load)."""
    return Character(id=cid, klass="thief", stats=FIXED_STATS, fixed_stats=FIXED_STATS,
                     skills={"lucky_seven": int(FIXED_STATS.lucky_seven)}, keys=None, timing=None,
                     ammo="throwing stars", stops=("out_of_darts",), buffs=(), magic_guard=0.0,
                     attacks={"skill": LUCKY_SEVEN, "regular": STAR}, pet=PET, policy=POLICY, priors=PRIORS,
                     shape=SHAPE, perception=PERCEPTION,
                     safety_hits=SAFETY_HITS)


def _at(table: tuple, level: float) -> float:
    return float(table[min(max(int(level), 1), len(table)) - 1])


def lucky_seven(level: float) -> AttackSpec:
    """Lucky Seven at skill `level` as attack.choice.lucky_seven has it: its MP a cast there, and its stars as measured
    at LUCKY_SEVEN.level (SKILL_STAR at SKILL_LEVEL) scaled by its damage there (past the tables' ends, their nearest)."""
    k = _at(LUCKY_SEVEN_DAMAGE, level) / _at(LUCKY_SEVEN_DAMAGE, LUCKY_SEVEN.level)
    return replace(LUCKY_SEVEN, level=int(level), lo=round(LUCKY_SEVEN.lo * k), hi=round(LUCKY_SEVEN.hi * k),
                   mp=_at(LUCKY_SEVEN_MP, level))


def leveled(c: Character, level: float) -> Character:
    """The thief with Lucky Seven at `level` (config.toml [character] lucky_seven, the stack's Attacks(level=...))."""
    return replace(c, skills={**c.skills, "lucky_seven": int(level)}, attacks={**c.attacks, "skill": lucky_seven(level)})


def configured(c: Character, table: config.CharacterConfig) -> Character:
    """[characters.thief]: her bottom bar's lines, if given (the rest of her is the flat tables)."""
    return replace(c, perception=replace(c.perception, job=table.job or c.perception.job,
                                         name=table.name or c.perception.name))
