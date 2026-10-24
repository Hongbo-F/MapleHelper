"""What a map's monsters drop: the items the loot detector looks for on that map (`items_for`, perception.loot), and
what the rest of the stack knows of each monster's drops (the mesos a kill drops, behavior.looting; reference rates
for the simulator's drops, sim.world.game_drops).

The source is meowdb's MapleStory Classic pages (this game: https://meowdb.com/msclassic/monsters/<id>, read on
2026-10-24, their drops through the page's own /msclassic/api/drops and /api/mesos-reports). Each monster's page has
three lists, and says of each:
- "Drops (MS Classic)", community sourced: "Items players have personally seen drop in-game. Upvote what you've seen,
  downvote what you haven't." `MonsterDrops.items` are its Etc and Use items with more votes up than down, but for
  scrolls (all of them three pictures, a shop pays 1 for one: not looked for, README Limits), quest items (Animal and
  Plant Fossil, the Omok pieces and table, the Very Dry Branch: they drop only for a player on the quest, and a shop
  pays 1) and arrows (Arrows for Bows and for Crossbows: a shop pays nothing for them, and a thief throws stars). Left
  out as voted down as often as up or more: a Red Snail's Iron Ore (5.25 up, 7.25 down) and Amethyst Ore (2, 2), an
  Axe Stump's Wooden Top (3, 14.5) and a Dark Stump's "Lemon (Alt)" (1.25, 1; no item of that name in the game data).
- OSMLib's reference rates: "These rates may not represent MapleStory Classic World rates." Kept as `rates` (for the
  simulator only): on the first map, in game, her slimes' piles held Squishy Liquid 65% of the time, a Slime Bubble
  10% and a Red Potion 4% (behavior.looting), where OSMLib gives 40%, 3% and 1%.
- An old MapleSEA list: "historical reference only" ("Classic World appears to draw from the pre-Big-Bang MapleSEA
  drop tables"). Not used.
Mesos are the players' reports on each page (`mesos`, the lowest and highest reported; none for a Slime or a Red
Snail: the first map's, seen in game, for the Slime).

Every map's drops are its monsters' and the potions (POTIONS, as on the first map: monsters at this level drop them),
and mesos (perception.loot.MESOS, the coins). The first map's are pinned as they were (PINNED): what was seen where her
kills died there, and the potions; meowdb's slimes drop Orihalcon and Sapphire Ore too, which aren't looked for there.
A monster this table doesn't know adds nothing: its drops aren't looked for (`unknown` names it).

Each item's picture (`iconRaw`) and price (`info/price`) are the game data's, by its name (perception.loot.item_ids,
behavior.looting.PRICES); its kind is its name as a kind (perception.kinds), which is also its template's file name,
so a name can't hold a character Windows keeps out of file names (`safe`: "Omok Piece: Slime" would be written as an
alternate data stream of a file "omok-piece", which no folder listing shows).
"""

import re
from dataclasses import dataclass, field

from .kinds import kind_of

FIRST_MAP = 10000012  # Victoria Road: Thicket Around the Beach II
# The potions monsters this level drop (Use items: their pictures are in Item/Consume), looked for on every map as on
# the first: the community lists name the Red, Orange and Blue Potions for these maps' monsters, the White for none.
POTIONS = ("Red Potion", "Orange Potion", "White Potion", "Blue Potion")


@dataclass(frozen=True)
class MonsterDrops:
    """One monster's drops (see the module's docstring for where each list comes from)."""
    items: tuple[str, ...]              # Etc and Use items it drops, by name (the community list's order: most votes)
    mesos: tuple[int, int] | None       # mesos a kill drops, lowest and highest reported; None: none reported
    rates: dict[str, float] = field(default_factory=dict)  # OSMLib's reference rates (a share of kills), where given
    source: str = "meowdb"


# By the monster's name in the game (MapModel.spawns' names). Read 2026-10-24; votes up/down in each comment.
MONSTERS = {
    # Tree Dungeon, Forest Up North II (10002073)
    "Horny Mushroom": MonsterDrops(  # meowdb monster 19
        ("Horny Mushroom Cap",   # 71.75 / 0
         "Orange Potion",        # 70.5 / 0
         "Bronze Ore",           # 65.25 / 2
         "Blue Potion",          # 63.5 / 0
         "Magic Potion",         # 58 / 0
         "Topaz Ore",            # 48.25 / 0
         "Sapphire Ore"),        # 16.25 / 2
        (40, 53),                # 45-53 and 40-48 reported
        {"Horny Mushroom Cap": 0.40, "Orange Potion": 0.01, "Blue Potion": 0.01, "Bronze Ore": 0.003,
         "Topaz Ore": 0.002}),
    "Green Mushroom": MonsterDrops(  # 13
        ("Fragment of Magic",    # 71.25 / 0.25
         "Green Mushroom Cap",   # 68.25 / 0
         "Silver Ore",           # 51.5 / 0.25
         "Blue Potion",          # 47.25 / 1
         "Red Potion",           # 41.5 / 0
         "Emerald Ore",          # 38.5 / 0
         "Orihalcon Ore"),       # 18.25 / 0
        (25, 30),
        {"Green Mushroom Cap": 0.40, "Red Potion": 0.02, "Blue Potion": 0.01, "Silver Ore": 0.003,
         "Emerald Ore": 0.002}),
    "Slime": MonsterDrops(  # 7; on the first map too
        ("Slime Bubble",         # 53.5 / 0.25
         "Squishy Liquid",       # 50.5 / 0
         "Orihalcon Ore",        # 39 / 0
         "Sapphire Ore",         # 33.5 / 0
         "Red Potion"),          # 31.5 / 0
        (10, 20),                # none reported on meowdb: the first map's monsters' 10-20, 13 seen in game
        {"Squishy Liquid": 0.40, "Slime Bubble": 0.03, "Red Potion": 0.01, "Sapphire Ore": 0.003,
         "Orihalcon Ore": 0.002}),
    # West Rocky Mountain II (10004021)
    "Stump": MonsterDrops(  # 6
        ("Bronze Ore",           # 30.5 / 3
         "Tree Branch",          # 24.75 / 0
         "Emerald Ore",          # 18.25 / 0
         "Red Potion",           # 10.5 / 0
         "Adamantium Ore"),      # 2.25 / 0
        (1, 49),                 # one report
        {"Tree Branch": 0.40, "Red Potion": 0.01, "Adamantium Ore": 0.003, "Emerald Ore": 0.002}),
    "Dark Stump": MonsterDrops(  # 11
        ("Iron Ore",             # 76.25 / 4.25
         "Leaf",                 # 57.5 / 0
         "Garnet Ore",           # 57.5 / 6.75
         "Tree Branch",          # 32 / 1
         "Red Potion",           # 22 / 0
         "Lemon"),               # 10.75 / 0
        (19, 23),
        {"Leaf": 0.40, "Red Potion": 0.02, "Iron Ore": 0.003, "Garnet Ore": 0.002}),
    "Axe Stump": MonsterDrops(  # 15
        ("Firewood",             # 31.75 / 0
         "Orihalcon Ore",        # 16.75 / 0
         "Blue Potion",          # 14.5 / 0
         "Bronze Ore",           # 14.5 / 1
         "Red Potion",           # 11.25 / 0
         "Opal Ore"),            # 8 / 0
        (30, 41),                # 30-35 and 35-41 reported
        {"Firewood": 0.40, "Red Potion": 0.02, "Blue Potion": 0.01, "Orihalcon Ore": 0.003, "Opal Ore": 0.002}),
    "Red Snail": MonsterDrops(  # 5
        ("Red Snail Shell",      # 33.25 / 0
         "Silver Ore",           # 28.5 / 0
         "Red Potion",           # 20.25 / 0
         "Opal Ore"),            # 16 / 0.25
        None,
        {"Red Snail Shell": 0.40, "Red Potion": 0.01, "Opal Ore": 0.003, "Silver Ore": 0.002}),
}

# The first map's drops, as they were: the slime's green bubble and its liquid, the pig's leather (and its head and
# ribbon), the Orange Mushroom's cap, the snail's shell (snails in her way get shot), all seen where her kills died
# there, and the potions.
FIRST_ITEMS = ("Squishy Liquid", "Slime Bubble", "Leather", "Pig's Head", "Pig's Ribbon", "Orange Mushroom Cap",
               "Snail Shell") + POTIONS
PINNED = {FIRST_MAP: FIRST_ITEMS}

_NOT_IN_FILE_NAMES = re.compile(r'[<>:"/\\|?*]')


def items_for(model) -> tuple[str, ...]:
    """The items looked for on map `model` (a MapModel, or anything with its `id` and `spawns`), by name: PINNED's for
    the first map; else every drop of each of its monsters MONSTERS knows (in their order), then POTIONS."""
    pinned = PINNED.get(getattr(model, "id", None))
    if pinned is not None:
        return pinned
    out: list[str] = []
    for name in _names(model):
        for item in MONSTERS[name].items if name in MONSTERS else ():
            if item not in out:
                out.append(item)
    return tuple(out) + tuple(p for p in POTIONS if p not in out)


def unknown(model) -> list[str]:
    """The monsters of map `model` MONSTERS doesn't know (their drops aren't looked for), but on a pinned map."""
    return [] if getattr(model, "id", None) in PINNED else [n for n in _names(model) if n not in MONSTERS]


def monsters_of(model) -> dict[str, int]:
    """Map `model`'s monsters by name, and their spawn points, in the order they first spawn."""
    out: dict[str, int] = {}
    for s in model.spawns:
        if s.name:
            out[s.name] = out.get(s.name, 0) + 1
    return out


def safe(name: str) -> bool:
    """Whether an item's kind can be a file's name (see the module's docstring)."""
    return not _NOT_IN_FILE_NAMES.search(kind_of(name))


def _names(model) -> list[str]:
    return list(monsters_of(model))
