"""Looting: picking up what her kills drop, and remembering drops she can't pick up yet (the Looter).

Drops land about 2 s after a kill (the death animation, then they pop up and fall), spread about
LOOT_REACH around where the monster died, and monsters are drawn in front of them. So after a fight
she sweeps where each kill died, then looks again and goes back for what's left (`mop_up`). Drops on
other platforms are remembered (`pending`), for the tour to collect. A drop she has stood on twice,
pressing pick-up, that's still there is someone else's (drops belong to the killer for a while), or her
inventory is full: the game says so in the pickup log, and then she only takes mesos.
"""

import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from maplehelper.ocr.item_names import ITEM_NAMES
from maplehelper.ocr.parse import match_prefix, parse_loot

from .. import directions, telemetry
from ..attack.reach import EDGE_MARGIN
from ..attack.stance import TOO_CLOSE
from ..control.navigator import MIN_GROUND_PX
from ..localization.monsters import Monster
from ..localization.players import near_rival
from ..localization.scene import Loot, Scene
from ..perception import drops
from ..perception.kinds import kind_of
from ..perception.loot import MESOS, says_full

if TYPE_CHECKING:
    from .hunter import Hunter

LOOT_REACH = 60    # drops spread about this far around where the monster died
PILE_GAP = 15      # drops closer together than this are one pile: picked up standing in one place
STOOD_ON = 15      # she stood on a drop when she stopped within this of it (walks stop ~10 units off)
MOP_UP = 3         # looks after a sweep for drops still on her platform, going back for them
SETTLE = 3.0       # s a kill's drops take to land and show: recorded at 2 fps, they lay on the ground about
                   # 2 s after her EXP went up (the death animation, then they pop up and fall)
EMPTY_LOOKS = 1.5  # s between two looks that must both see a remembered drop spot empty before it's
                   # forgotten: monsters are drawn over drops, and a pig walking over coins hid them
UNPICKABLE_FOR = 90  # s drops she failed to pick up are left alone (they vanish after a while anyway)
FAILED_PICKS = 3   # items in a row that stayed under her with no item picked up: their tabs are full
# Her inventory's tabs (from 2026-10-31, the user: "we also need to watch for backpack full scenario"): the one a drop's
# kind goes into (`tab_of`: its item id's in the game data, perception.loot.item_ids, 2xxxxxx Use, 4xxxxxx Etc; mesos
# none; a kind the game data doesn't name, Use for a potion and Etc for the rest, as the simulator's pickup log has
# them), and which are full (Looter.full), learned where items stay under her after two stands (`cant_pick`): the pickup
# log's full message ("You can't get anymore items.", which names no tab) marks the tabs of the items stuck there,
# FAILED_PICKS failures in a row without it too; an "earned. (Tab)" line in the same read (hers or her pets') says that
# tab has room. A full tab's items are left and not gone for, mesos and the other tab's still taken (Looter.can_take:
# behavior.hunt's piles priced by what she can take), perception looking for mesos alone only once every item tab is
# full (`items_full`); a full tab is taken as having room again TAB_RETRY_S on (she may have sold or dropped things),
# learned again at the next failure. Until then one flag for the whole run (`inventory_full`, now: any tab full): every
# item left once it was set, nothing ever clearing it. Live a full inventory came once (2026-10-19, the first map's Etc
# tab, a Snail Shell), never on Tree Dungeon.
# From the review of 2026-10-31: the log's lines in their order (the game adds a line at the bottom and the OCR reads
# them top to bottom, oldest first; the simulator's log the same), so an "earned" line before the full message (the item
# that filled the tab, then the next refused) says nothing of room now, one after it does; and a full tab takes more of a
# kind she holds a part-filled stack of (v83: no free slot, but a stack below its most): the kinds she has picked up this
# run (`held`: stood on while their tab had room and gone from the pile by the looks after, `stood_on` and `on_look`, or
# earned by the log's word, her pets' too) are still taken from a full tab until one stays under her (`refused`: its
# stack full too), a kind she doesn't hold needs a slot and isn't. (Taken as hers on the stand alone, as first built, a
# tab full from the start had every kind she stood on before she learned it taken for hers: 3.0 of her 5.2 stands a run
# for nothing came after she had, in `--world etc-full`.) Until then any "earned" line of a tab in the read kept it
# from being marked full, and one refused kind marked every kind of its tab full for TAB_RETRY_S.
# From the independent check of 2026-11-01: the "earned" lines' items named by maplehelper's own loot-line parser
# against the game's item names (`_earned`: maplehelper.ocr.parse.parse_loot, ITEM_NAMES; a cut name the parser leaves
# open, "Orange Mushro...", the one priced drop it starts, PRICES), where a regex of the name before "x1" read the
# game's cut names ("Horny Mushroo... x1 earned. (Etc)", live on 2026-10-26: tests/fixtures/
# pickup_log_lines_2026-10-26.json) as kinds of their own ('horny-mushroo...'), never one she held: her cap top-up into
# a full Etc tab cleared it. And an "earned" line after the last refusal says its tab has room only for a kind that
# didn't fit (stuck under her now, or `refused`): one more of a kind she holds a stack of, or may hold from before the
# run, needs no slot (until then any such line in the read of a tab being learned kept it from being marked full, so
# her stacks' top-ups, hers or her pets', stalled the learning). FAILED_PICKS' row of items stuck under her is broken by
# an item of a stand of hers the looks show gone from its pile (`on_look`: the hunt loop's stands, which no mop-up
# follows) and starts afresh once it marks a tab full or a tab is tried again (`_retry`): until then only a mop-up's
# item gone broke it, so in the grind three stuck items in a run, however far apart, marked every later one's tab full.
ITEM_TABS = ("Use", "Etc")
TAB_RETRY_S = 600.0  # guess
PICK_SEEN_S = 10.0   # s after a stand the looks have to show a kind gone from the pile she stood on (`on_look`)
EARNED = re.compile(r"earned\W*\(\s*(equip|use|set-?up|etc|cash)\s*\)", re.I)  # an "earned" line's tab
DROPS_LAST = 90    # s remembered drops are worth going back for (they vanish, and others may take them)
MIN_WALK = 10      # a sweep's stops nearer to her than this aren't walked to: she's there
# Whether only a look that shows her standing can see a drop spot empty (`standing`). From mid-jump a look's loot
# strips miss the drops on the ground, and two such looks forget a spot with its drops still there: in simulated
# hunt loops (preset game, seed 2), 357 of the 359 piles forgotten with drops on them were forgotten that way,
# and most of the loop's lost drops (its drops share fell 0.84 -> 0.72 on this branch, as jump backs and jump
# throws grew) lay in the basin under such piles. Switched on, the loop's drops share is 0.92, but its EXP a
# minute 7% lower (8 seeds; two runs fell out of the map): it then goes for piles in the basin that monsters
# always guard, and in seed 4 seven of its ten pile trips there ran out their 60 s. Off until the hunt loop's
# pile rules weigh that. (In game it's the same: in mid-jump her y is put on the platform under her, so the
# strips sit as high above its ground as she is.)
STANDING_ONLY = False

# What a drop is worth, in mesos (`worth`, a map's `Worth`; the drop store's Pile.value, which the hunt loop ranks
# piles by and goes for the valuable ones by at once, hunt.VALUABLE). An item's is what a shop pays for it, the game
# data's own price: `info/price` of each item in Item/Etc/Etc_000.wz and Item/Consume/Consume_000.wz, read by item id
# (the ids by name from String, perception.loot.item_ids; the first map's on 2026-10-21, the new maps' on 2026-10-24):
# every drop of every map (perception.drops), the same on all of them. The potions with what they restore (their
# `spec`), which doesn't add to it: her HP key is bound to Meat and her MP key to the Blue Potion, which the regular
# attack never needs (maplehelper/game.py; on the new maps her pet drinks her potions), so a potion picked up is one
# to sell. Of the drops seen where her kills died in the hunts of 2026-10-21, a Slime Bubble lay in 5 of the 52 slimes'
# piles seen (Squishy Liquid in 34), and no White Potion was seen.
PRICES = {
    # the first map's
    "squishy-liquid": 6,       # Etc 4000006 (slimes)
    "slime-bubble": 50,        # Etc 4000007 (slimes)
    "leather": 10,             # Etc 4000049 (pigs)
    "pig's-head": 7,           # Etc 4000008 (pigs)
    "pig's-ribbon": 10,        # Etc 4000010 (pigs)
    "orange-mushroom-cap": 8,  # Etc 4000009
    "snail-shell": 1,          # Etc 4000001
    "red-potion": 5,           # Use 2000000: 100 HP
    "orange-potion": 15,       # Use 2000001: 250 HP
    "white-potion": 35,        # Use 2000002: 500 HP
    "blue-potion": 22,         # Use 2000003: 200 MP
    # Tree Dungeon, Forest Up North II's
    "horny-mushroom-cap": 22,  # Etc 4000019
    "green-mushroom-cap": 14,  # Etc 4000013
    "fragment-of-magic": 10,   # Etc 4000063 (Green Mushrooms)
    "bronze-ore": 20,          # Etc 4010000 (Horny Mushrooms; Stumps, Axe Stumps)
    "silver-ore": 80,          # Etc 4010004 (Green Mushrooms; Red Snails)
    "orihalcon-ore": 100,      # Etc 4010005 (Green Mushrooms, slimes; Axe Stumps)
    "emerald-ore": 80,         # Etc 4020003 (Green Mushrooms; Stumps)
    "sapphire-ore": 120,       # Etc 4020005 (Horny Mushrooms, slimes)
    "topaz-ore": 80,           # Etc 4020006 (Horny Mushrooms)
    "magic-potion": 250,       # Use 2002001: magic attack +10 for 600 s (Horny Mushrooms)
    # West Rocky Mountain II's
    "tree-branch": 5,          # Etc 4000005 (Stumps, Dark Stumps)
    "leaf": 11,                # Etc 4000011 (Dark Stumps)
    "firewood": 17,            # Etc 4000015 (Axe Stumps)
    "red-snail-shell": 4,      # Etc 4000004
    "iron-ore": 20,            # Etc 4010001 (Dark Stumps)
    "adamantium-ore": 60,      # Etc 4010003 (Stumps)
    "garnet-ore": 40,          # Etc 4020000 (Dark Stumps)
    "opal-ore": 160,           # Etc 4020004 (Axe Stumps, Red Snails)
    "lemon": 15,               # Use 2010005: 150 MP (Dark Stumps)
}
# Mesos by the coin they're drawn as (perception.loot: Item.coin), which their amount picks: the game data has a
# bronze coin, a gold one, a bundle of bills and a sack (Item/Special 0900000k); by the v83 client (HeavenClient's
# MapDrops; not checked for Classic) a bronze coin under 50 mesos, gold to 99. The first map's monsters drop 10-20 (13
# seen in game: maplehelper/game.py), always bronze coins; gold ones are worth more, and bills and sacks aren't looked
# for (perception.loot.MESO_KINDS). A coin of no known kind counts as bronze. These are the first map's; another map's
# go by what its monsters drop (`coins_for`).
COINS = {"bronze": 15, "gold": 75}
# A drop the table doesn't know (none: every map's drops are in PRICES, and equipment and scrolls aren't looked for,
# README Limits): at this level those are worth the most, so as much as the first map's dearest drop, a Slime Bubble.
# (It had followed the dearest in PRICES, which is now the Magic Potion's 250: every drop not known would have been
# valuable, hunt's rule 6a.)
UNKNOWN_WORTH = 50
# A drop worth this many times a bronze coin on its own is valuable (`Worth.valuable`: hunt's rule 6a goes for it at
# once, and PLAYER_PILES keeps it): on the first map 30 mesos (hunt.VALUABLE), a Slime Bubble (50), a White Potion (35)
# and a gold coin (75), 2-8 times a pile's coins there (15); on Tree Dungeon 66.8 (the Magic Potion, the ores but
# Bronze, its 33.4 a coin twice), on West Rocky Mountain II 51.8 (the Opal, Orihalcon, Silver, Emerald and Adamantium
# Ores, and a gold coin). Not weighed by the simulator on those maps: it doesn't run them yet.
VALUABLE_COINS = 2.0
# Whether drops seen by another player (localization.players.near_rival) are left to them unless one is valuable (the
# map's Worth.valuable: on the first map a Slime Bubble, a White Potion): not remembered for the tour to come back for
# (`remember`). Her own kills' drops are swept where they died, whoever stands by them. Off: the tour can't tell whose
# drops it sees, and some by the other player were hers. In the simulator's tour (preset game, 20 seeds of 20 minutes)
# it changed 11 seeds' runs, 7 for the better (the worst 7% down): 91.4 -> 92.8 EXP a minute, HP lost 3.00 -> 2.88
# bars, but her drops picked up 94.2% -> 93.9% (seeds 1-10: 94.7% -> 92.6%), below the harness's bar; with the other
# player killing every 8 s, 15 changed, 11 for the better, 85.39 -> 88.47 (+4.1%, standard error 2.3%; the worst 19%
# down), drops 94.2% -> 93.4%. (The hunt loop's drop store knows which piles her kills explain: hunt.PLAYER_PILES.)
PLAYER_PILES = False


@dataclass(frozen=True)
class Worth:
    """What drops are worth on one map, in mesos: an item what a shop pays for it (PRICES, the same on every map);
    mesos by their coin (`coins`: on the first map COINS, on another what its monsters drop, `coins_for`)."""
    coins: dict = field(default_factory=lambda: dict(COINS))

    def __call__(self, kind: str, coin: str | None = None) -> float:
        """What a drop of this kind (perception.loot.Item.kind; mesos by their coin) is worth."""
        if kind == MESOS:
            return self.coins.get(coin or "bronze", self.coins["bronze"])
        return PRICES.get(kind, UNKNOWN_WORTH)

    @property
    def valuable(self) -> float:
        """What a drop must be worth on its own to be valuable on this map: VALUABLE_COINS bronze coins."""
        return VALUABLE_COINS * self.coins["bronze"]


FIRST = Worth(COINS)  # the first map's
_BY_KIND = {kind_of(name): known for name, known in drops.MONSTERS.items()}  # perception.drops' monsters by kind


def worth(kind: str, coin: str | None = None) -> float:
    """What a drop of this kind (perception.loot.Item.kind; mesos by their coin) is worth on the first map, in mesos."""
    return FIRST(kind, coin)


def worth_for(model) -> Worth:
    """What drops are worth on map `model` (a MapModel, or anything with its `id` and `spawns`)."""
    return FIRST if getattr(model, "id", None) == drops.FIRST_MAP else Worth(coins_for(model))


# What a kill of hers is expected to drop before a look sees its pile (`expected_worth`: the drop store's Pile.unseen):
# live on 2026-10-30 (the thief's run as the perception service saw it, the loot design's expcount.py) her kills gave
# 0.64 drops seen a kill, 57% of them mesos. Each mesos drop is worth the middle of its kind's amounts (perception.drops),
# each item the price of its kind's items weighed by their reference rates (OSMLib's; the rarest's where it gives none):
# a Horny Mushroom's pile 23.5 mesos, a Green's 14.1, a slime's 8.3. A kind none of that knows: a bronze coin of the
# map's, as every pile no look had seen was until then.
EXPECTED_DROPS = 0.64
EXPECTED_MESOS = 0.57


def expected_worth(kind: str | None, value: "Worth", can: Callable[[str], bool] | None = None) -> float:
    """What a pile of a kill of a monster of `kind` is worth before a look sees it, on the map whose drops are worth
    `value` (see EXPECTED_DROPS); with `can` (Looter.can_take: her inventory by tab), its items she can't take worth
    nothing."""
    known = _BY_KIND.get(kind) if kind is not None else None
    if known is None or known.mesos is None:
        return value.coins["bronze"]
    rates = [(i, known.rates.get(i, min(known.rates.values(), default=0.002))) for i in known.items]
    total = sum(r for _, r in rates)
    item = sum(r * value(kind_of(i)) for i, r in rates if can is None or can(kind_of(i))) / total if total \
        else value.coins["bronze"]
    return EXPECTED_DROPS * (EXPECTED_MESOS * sum(known.mesos) / 2 + (1 - EXPECTED_MESOS) * item)


def coins_for(model) -> dict[str, float]:
    """What a bronze and a gold coin are worth on map `model`: the mean of the amounts each stands for (a bronze coin
    under 50 mesos, gold 50-99) that its monsters drop (perception.drops: what players report, each amount in the range
    taken as likely as another), weighed by the monster's spawn points and the share of its amounts in that coin. A
    monster none is reported for counts for nothing; a coin none of the map's drops is worth what it is on the first
    map (COINS). Tree Dungeon: a bronze coin 33.4 (a Horny Mushroom's 40-49, a Green's 25-30, a slime's 10-20), gold
    51.5 (a Horny's 50-53); West Rocky Mountain II: bronze 25.9 (a Stump's 1-49, a Dark Stump's 19-23, an Axe Stump's
    30-41; none reported for a Red Snail), gold 75."""
    spawns = drops.monsters_of(model)
    sums = {coin: [0.0, 0.0] for coin in COINS}  # coin -> [weight, weight x mean amount]
    for name, n in spawns.items():
        known = drops.MONSTERS.get(name)
        if known is None or known.mesos is None:
            continue
        lo, hi = known.mesos
        for coin, (a, b) in (("bronze", (lo, min(hi, 49))), ("gold", (max(lo, 50), min(hi, 99)))):
            if a <= b:
                w = n * (b - a + 1) / (hi - lo + 1)
                sums[coin][0] += w
                sums[coin][1] += w * (a + b) / 2
    return {coin: round(s / w, 1) if w else COINS[coin] for coin, (w, s) in sums.items()}


@dataclass
class DropSpot:
    """Where drops lie or will land, to be picked up: where a kill of hers died, or drops seen on the
    ground."""
    x: float
    y: float
    platform: int | None
    since: float
    mesos: bool = False      # mesos were seen there (still worth a trip with her inventory full)
    on_ground: bool = False  # seen on the ground (else: where a kill of hers died)
    empty_since: float | None = None  # when a look first saw the spot empty (see EMPTY_LOOKS)


class Looter:
    """What she picks up, what she remembers for later (`pending`), and what she can't pick up."""

    def __init__(self, hunter: "Hunter"):
        self.h = hunter
        self.pending: list[DropSpot] = []  # drops off her platform, left for later (the tour collects them)
        self.full: dict[str, tuple[float, str]] = {}  # tab -> (when she learned it's full, why): its items left
        self.held: set[str] = set()  # item kinds she has a stack of (see ITEM_TABS): taken from a full tab still ...
        self.refused: dict[str, float] = {}  # ... but these: kind -> when one stayed under her with its tab full
        self._stands: list[tuple] = []  # (pile, kinds, when): her stands whose picks the looks after confirm (`on_look`)
        self.unpickable: list[tuple[float, int, float]] = []  # (x, platform, when) of drops she couldn't take
        self._failed_picks = 0  # items in a row that stayed under her (see FAILED_PICKS)
        # Whether a look's monsters call her off picking up (collect, mop_up), leaving the drops for later: the hunt
        # loop's EXP before loot (behavior.hunt: a wanted monster in reach); None: never
        self.yield_to: Callable[[Scene], bool] | None = None
        # The monsters about her that a look didn't find, which guard drops as the look's own do (collect, mop_up): the
        # hunt loop's (Hunt._unseen: the feed's hidden tracks, her memory's of the last moments); None: the look's
        # alone. Live on 2026-10-25 her pickup walk on Tree Dungeon's 23 went through a Horny Mushroom two looks had
        # missed (localization.unseen).
        self.unseen: Callable[[Scene], list[Monster]] | None = None

    @property
    def inventory_full(self) -> bool:
        """Whether a tab of hers is full (the 'drops' record's, analysis.respawn's)."""
        return bool(self.full)

    @inventory_full.setter
    def inventory_full(self, full: bool) -> None:
        """Every item tab full (True, as the one flag before the tabs had it), or none."""
        now = time.monotonic()
        self.full = {tab: (now, "set") for tab in ITEM_TABS} if full else {}

    @property
    def items_full(self) -> bool:
        """Whether every item tab is full: drops are looked for as mesos alone (Hunter.look)."""
        self._retry()
        return all(tab in self.full for tab in ITEM_TABS)

    def can_take(self, kind: str) -> bool:
        """Whether a drop of `kind` fits: mesos always, an item while its tab isn't known to be full, or one of a full
        tab she holds a stack of (`held`) that hasn't stayed under her since (`refused`)."""
        if kind == MESOS:
            return True
        self._retry()
        return tab_of(kind) not in self.full or kind in self.held and kind not in self.refused

    def stood_on(self, kinds, pile=None) -> None:
        """She stood on drops of these kinds pressing pick-up, those whose tab has room: she holds a stack of them now
        (`held`) once the looks after show them gone from `pile` (ledger.Pile: `on_look`), at once with no pile; one
        that stays under her is `cant_pick`'s."""
        kinds = {k for k in kinds if k != MESOS and tab_of(k) not in self.full}
        if pile is None:
            self.held.update(kinds)
        elif kinds:
            self._stands.append((pile, kinds, time.monotonic()))

    def on_look(self, s) -> None:
        """After a look (once the drop store has seen it): her stands' kinds the pile she stood on no longer holds, or
        that the store has seen gone, she picked up (`held`), and FAILED_PICKS' row of items stuck under her is broken
        (the hunt loop's stands have no mop-up to break it: `mop_up`); those still in it as seen since, she didn't."""
        now, kept, took = time.monotonic(), [], set()
        for pile, kinds, at in self._stands:
            if pile.status == "gone":
                took.update(kinds)
            elif pile.last_seen is not None and pile.last_seen > at:
                took.update(k for k in kinds if k not in pile.items)
            elif now - at < PICK_SEEN_S:
                kept.append((pile, kinds, at))
        self._stands = kept
        if took:
            self.held.update(took)
            self._failed_picks = 0

    def _retry(self) -> None:
        """A tab full for TAB_RETRY_S taken as having room again (learned again at the next failure)."""
        now = time.monotonic()
        for tab in [tab for tab, (since, _) in self.full.items() if now - since >= TAB_RETRY_S]:
            del self.full[tab]
            for k in [k for k in self.refused if tab_of(k) == tab]:
                del self.refused[k]
            self._failed_picks = 0  # (a row of items stuck counted afresh)
            self._tell(tab, False, f"{TAB_RETRY_S / 60:.0f} minutes since it was: trying its items again")

    def _tell(self, tab: str, full: bool, why: str, kinds: tuple[str, ...] = ()) -> None:
        """A tab's state changed: her log, and a telemetry event 'inventory'."""
        if full:
            others = [t for t in ITEM_TABS if t != tab and t not in self.full]
            stacks = sorted(k for k in self.held if tab_of(k) == tab and k not in self.refused)
            self.h.log(f"  her {tab} tab is full ({why}): leaving {tab} items"
                       + (f" but {', '.join(stacks)} (stacks of hers)" if stacks else "") + ", still taking mesos"
                       + "".join(f" and {t} items" for t in others))
        else:
            self.h.log(f"  her {tab} tab has room ({why})")
        telemetry.emit("event", kind="inventory", tab=tab, full=full, why=why, kinds=list(kinds))

    def wanted(self, loot: list[Loot]) -> list[Loot]:
        """The drops worth going for: not ones she failed to pick up, and no items of a full tab."""
        now = time.monotonic()
        self.unpickable = [u for u in self.unpickable if now - u[2] < UNPICKABLE_FOR]
        return [d for d in loot if (d.mesos or self.can_take(d.item.kind))
                and not any(p == d.platform and abs(x - d.x) <= STOOD_ON for x, p, _ in self.unpickable)]

    def remember(self, s: Scene) -> None:
        """Note what a look (with `loot`) saw, for the tour to collect: drops seen are remembered (a pile per
        LOOT_REACH). A remembered spot is forgotten (taken by someone, or by her on the way) once its drops
        have had time to land (SETTLE) and two looks EMPTY_LOOKS apart saw nothing there: one look can
        miss drops a monster stands in front of."""
        now = time.monotonic()
        seen = self.wanted(s.loot)
        kept = []
        for d in self.pending:
            if now - d.since < SETTLE or not s.in_view(d.platform, d.x) or any(_in_pile(d, loot) for loot in seen):
                d.empty_since = None  # still landing, out of view, or drops seen there
            elif STANDING_ONLY and not standing(s):
                pass                  # in the air: blind to the drops on the ground (standing)
            elif d.empty_since is None:
                d.empty_since = now   # seen empty once
            elif now - d.empty_since >= EMPTY_LOOKS:
                continue              # seen empty twice: forgotten
            kept.append(d)
        if gone := len(self.pending) - len(kept):
            self.h.log(f"  {gone} remembered drop spot(s) seen empty twice: forgotten")
        self.pending = kept
        if PLAYER_PILES and s.players:
            value = self.h.worth
            seen = [d for d in seen if near_rival(s.players, d.x, d.y) is None
                    or value(d.item.kind, d.item.coin) >= value.valuable]
        self.note(seen)

    def note(self, seen: list[Loot]) -> None:
        """Remember drops seen, a pile per LOOT_REACH, for later (the tour collects them)."""
        for loot in seen:
            pile = next((d for d in self.pending if _in_pile(d, loot)), None)
            if pile is None:
                self.pending.append(DropSpot(loot.x, loot.y, loot.platform, time.monotonic(), loot.mesos,
                                             on_ground=True))
            else:
                pile.mesos = pile.mesos or loot.mesos

    def collect(self, drops: DropSpot) -> bool:
        """Pick up the drops now if they're on her platform and no monster stands by them; otherwise
        remember them (pending) for later. Drops seen around the spot are walked to directly. Where
        a kill died she sweeps even when nothing shows: drops can still be falling, or hidden behind a
        monster, and the trip there cost far more. Drops seen earlier that are gone are done with.
        Returns True when done with them."""
        s = self.h.look(loot=True)
        if s.where is None or s.platform is None or drops.platform != s.platform.id:
            self.pending.append(drops)
            return False
        if self.yield_to is not None and self.yield_to(s):
            self.h.log("  something in reach first: the drops later")
            self.pending.append(drops)
            return False
        if _guarded(drops.x, drops.platform, s.monsters + self._about(s)):
            self.h.log("  drops guarded by a monster: later")
            self.pending.append(drops)
            return False
        seen = [d for d in self.wanted(s.loot) if d.platform == drops.platform
                and abs(d.x - drops.x) <= 2 * LOOT_REACH]
        if seen:
            self.sweep_through(s, [DropSpot(d.x, d.y, d.platform, s.t) for d in seen], margin=0)
        elif drops.on_ground and time.monotonic() - drops.since >= SETTLE and s.in_view(drops.platform, drops.x) \
                and (not STANDING_ONLY or standing(s)):
            self.h.log(f"  the drops seen at x {drops.x:.0f} are gone")
        else:
            self.sweep_through(s, [drops])
        return True

    def loot_pending(self, s: Scene | None = None) -> None:
        """Sweep remembered drops that are on her platform (loot() re-queues any still guarded); the
        rest stay remembered. `s` is a recent look, to tell which ones are on her platform."""
        s = s or self.h.look(mobs=False)
        here = None if s.platform is None else s.platform.id
        fresh = [d for d in self.pending if time.monotonic() - d.since < DROPS_LAST]  # drops vanish
        self.pending = [d for d in fresh if d.platform != here]
        for d in fresh:
            if d.platform == here:
                self.collect(d)

    def sweep_after(self, spots: list[DropSpot]) -> None:
        """Sweep up a fight's drops once it's over. The fight often ends mid-jump (a jump-throw), where
        a look finds her on no platform, and the sweep used to be skipped: those drops were left. Looks
        again until she has landed; drops on another platform than hers are left to the tour."""
        for _ in range(8):
            s = self.h.look(mobs=False)
            if s.where is not None and s.platform is not None:
                self.pending += [d for d in spots if d.platform != s.platform.id]
                self.sweep_through(s, spots)
                return
            time.sleep(0.15)
        self.pending += spots

    def sweep_through(self, s: Scene, spots: list[DropSpot], margin: float = LOOT_REACH) -> None:
        """Pick up the drops on her platform (`walk_piles`), then look again for any still there and go
        back for them (`mop_up`)."""
        self.mop_up(self.walk_piles(s, spots, margin))

    def walk_piles(self, s: Scene, spots: list[DropSpot], margin: float) -> list[float]:
        """Walk over the drops on her platform: from the nearer end of them to the farther one (and
        `margin` on past each end: a kill's drops spread around where it died), pressing pick-up all
        the way, and standing on each pile for a few presses. A pick-up press takes one item, a pig
        drops several, and walking past at full speed got one or two. Returns where she stood."""
        p = s.platform

        def clamp(v: float) -> float:  # never past her platform's ends
            return min(max(v, p.left + EDGE_MARGIN), p.right - EDGE_MARGIN)

        piles = _piles([clamp(d.x) for d in spots if d.platform == p.id])
        if not piles:
            return []
        x = s.where.x
        if abs(x - piles[-1]) < abs(x - piles[0]):
            piles.reverse()
        first, last = piles[0], piles[-1]
        sign = 1 if last >= first else -1
        stops = ([(clamp(first - sign * margin), False)] + [(g, True) for g in piles]
                 + [(clamp(last + sign * margin), False)])  # (where, a pile to stand on)
        for goal, pile in stops:
            dx = goal - x
            if abs(dx) >= MIN_WALK:
                self.h.actions.walk(directions.toward(dx), abs(dx))
                x = goal
            if pile:
                self.h.actions.pick_here()
        self.h.report.loot_runs += 1
        return piles

    def mop_up(self, stood: list[float] | None = None) -> None:
        """Look again after a sweep for drops still on her platform, and go back for them: walking past
        at full speed, or stopping a little off a pile, left some. A drop she has stood on twice that's
        still there can't be picked up (`cant_pick`); drops with a monster by them wait for later.
        `stood`: where the sweep just had her stand picking up."""
        stood, here, tried = list(stood or []), None, []  # tried: items she went back for last round
        for _ in range(MOP_UP):
            time.sleep(0.3)  # a picked-up drop flies to her for a moment
            s = self.h.look(loot=True)
            if s.where is None or s.platform is None or (here is not None and s.platform.id != here):
                return  # in the air, or knocked off: whatever is left, the tour sees
            here = s.platform.id
            seen = [d for d in s.loot if d.platform == here]
            if self.yield_to is not None and self.yield_to(s) and self.wanted(seen):
                self.h.log("  something in reach first: the rest of the drops later")
                return
            if any(_picked_up(t, seen) for t in tried):
                self._failed_picks = 0  # an item she went back for is gone: she picked it up
            left = self.wanted(seen)
            stuck = [d for d in left if sum(abs(x - d.x) <= STOOD_ON for x in stood) >= 2]
            if stuck:
                self.cant_pick(stuck)
                left = [d for d in left if d not in stuck]
            about = s.monsters + self._about(s)
            guarded = [d for d in left if _guarded(d.x, here, about)]
            if guarded:
                self.h.log(f"  {len(guarded)} drop(s) by a monster: later")
                self.note(guarded)
                left = [d for d in left if d not in guarded]
            if not left:
                return
            self.h.log(f"  {_summary(left)} still on platform {here}: back for them")
            tried = [d for d in left if not d.mesos]
            stood += self.walk_piles(s, [DropSpot(d.x, d.y, here, s.t) for d in left], margin=0)

    def _about(self, s: Scene) -> list[Monster]:
        """The monsters about her that look `s` didn't find (`unseen`), [] without that."""
        return [] if self.unseen is None else list(self.unseen(s))

    def cant_pick(self, stuck: list[Loot]) -> None:
        """Drops she stood on twice, pressing pick-up, that are still there: left alone for a while. Mesos
        always fit, so those are someone else's (drops belong to the killer for a while). Items may be
        too, unless the pickup log says her inventory is full, or items keep staying under her
        (FAILED_PICKS times in a row, with none picked up in between): then their tabs are full (see the top), but a
        tab the same read has an item that didn't fit earned into after the refusal, and their items are left, mesos
        and the rest taken."""
        now = time.monotonic()
        self.unpickable += [(d.x, d.platform, now) for d in stuck]
        self.h.log(f"  can't pick up {_summary(stuck)} at x {', '.join(f'{d.x:.0f}' for d in stuck)}: left")
        didnt = {d.item.kind for d in stuck if not d.mesos} | set(self.refused)  # (the kinds that didn't fit)
        kinds = tuple(sorted({d.item.kind for d in stuck if not d.mesos} - set(self.refused)))
        stacked = [k for k in kinds if tab_of(k) in self.full]  # (a kind she holds, of a tab full: its stack full too)
        for k in stacked:
            self.refused[k] = now
            self.h.log(f"  her {k} stack is full too: leaving {k} with her {tab_of(k)} tab full")
            telemetry.emit("event", kind="inventory", tab=tab_of(k), full=True, why="its stack full too", kinds=[k])
        kinds = tuple(k for k in kinds if k not in stacked)
        tabs = {tab_of(k) for k in kinds}
        if not tabs:
            return
        self._failed_picks += 1
        lines = self.h.senses.eyes.messages().splitlines()  # (oldest first: see ITEM_TABS)
        refusal = max((i for i, line in enumerate(lines) if says_full(line)), default=-1)
        room = set()
        for i, line in enumerate(lines):
            if (got := _earned(line)) is None:
                continue
            tab, kind = got
            if i > refusal and kind in didnt:
                room.add(tab)  # (one that didn't fit earned after the last refusal: a slot for it then)
            if kind is not None:
                self.held.add(kind)  # (a stack of it with room, hers or her pets')
                self.refused.pop(kind, None)
        for tab in sorted(room & set(self.full)):  # (one that didn't fit earned into it: it has room)
            del self.full[tab]
            for k in [k for k in self.refused if tab_of(k) == tab]:
                del self.refused[k]
            self._tell(tab, False, "an item earned into it")
        if refusal >= 0:
            why = f"the game says so: {' / '.join(line for line in lines if says_full(line))}"
        elif self._failed_picks >= FAILED_PICKS:
            why = f"items stayed under her {self._failed_picks} times in a row"
        else:
            return
        for tab in sorted(tabs - room):
            self.full[tab] = (now, why)
            for k in kinds:
                if tab_of(k) == tab:
                    self.refused[k] = now
            self._failed_picks = 0  # (the row that marked it: the next counted afresh)
            self._tell(tab, True, why, tuple(k for k in kinds if tab_of(k) == tab))


def _earned(line: str) -> tuple[str, str | None] | None:
    """A pickup log line that says an item was earned: (its tab, the item's kind, None where the line doesn't tell it),
    None for any other line. The name by maplehelper's loot-line parser against the game's item names (parse_loot,
    ITEM_NAMES), a cut one it leaves open the one priced drop it starts (PRICES: see ITEM_TABS); the tab as the line
    says it, else the kind's (`tab_of`: OCR's "(Ete)")."""
    tab = EARNED.search(line)
    got = parse_loot(line, ITEM_NAMES)
    name = got.item if got is not None and got.kind == "item" else None
    if name and got.cut:
        name = match_prefix(name, _PRICED)
    kind = kind_of(name) if name else None
    if tab is not None:
        return tab.group(1).replace("-", "").capitalize(), kind
    return (tab_of(kind), kind) if kind is not None else None


_PRICED = tuple(k.replace("-", " ") for k in PRICES)  # (the priced drops' names: `_earned`'s cut names)
_TABS: dict[str, str] | None = None


def tab_of(kind: str) -> str | None:
    """The inventory tab a drop of `kind` goes into (see ITEM_TABS): None for mesos."""
    global _TABS
    if kind == MESOS:
        return None
    if _TABS is None:
        try:
            from ..perception.loot import item_ids
            _TABS = {kind_of(name): "Use" if ids[0] // 1_000_000 == 2 else "Etc" for name, ids in item_ids().items()}
        except (OSError, ImportError, KeyError, ValueError):  # (no game data here: by the rule alone)
            _TABS = {}
    return _TABS.get(kind) or ("Use" if kind.endswith("-potion") else "Etc")


def standing(s: Scene) -> bool:
    """Whether look `s` shows her standing: ground under her feet on screen (control.navigator's test, but not
    with her name tag hidden). With STANDING_ONLY, only such a look can see a drop spot empty: in the air or on a
    rope there's no ground under her, and a look's loot strips, put just above her platform's ground as if she
    stood on it, lie above the drops by as high as she is, so it finds none. (MIN_GROUND_PX is in half-size px of
    the 1920x1080 window: the look's geometry converts it, 8.5 px at 1366x768.)"""
    return s.ground is not None and s.ground[1] - s.ground[0] >= s.geo.fpx(MIN_GROUND_PX)


def _in_pile(spot: DropSpot, loot: Loot) -> bool:
    """Whether drops seen (`loot`) are in a remembered spot's pile: on its platform, within LOOT_REACH."""
    return spot.platform == loot.platform and abs(spot.x - loot.x) <= LOOT_REACH


def _guarded(x: float, platform: int, monsters: list[Monster]) -> bool:
    """Whether a monster stands by drops at map x on `platform` (nearer than TOO_CLOSE): they wait."""
    return any(m.platform == platform and abs(m.x - x) < TOO_CLOSE for m in monsters)


def _picked_up(item: Loot, seen: list[Loot]) -> bool:
    """Whether an item she went back for is gone: no item is seen where it lay any more."""
    return not any(abs(d.x - item.x) <= STOOD_ON for d in seen if not d.mesos)


def _piles(xs: list[float]) -> list[float]:
    """Drop spots closer together than PILE_GAP made one pile (their middle), in order."""
    piles: list[list[float]] = []
    for x in sorted(xs):
        if piles and x - piles[-1][-1] < PILE_GAP:
            piles[-1].append(x)
        else:
            piles.append([x])
    return [round(sum(p) / len(p)) for p in piles]


def _summary(loot: list[Loot]) -> str:
    """Drops by kind: "2 mesos, 1 leather"."""
    return ", ".join(f"{n} {k}" for k, n in Counter(d.item.kind for d in loot).items())
