"""Other players on the map, from their dots on the minimap (perception.minimap.other_dots): each placed on the
map as her own dot is (pose.Localizer, snapped to the platform under it), and followed from look to look.

A dot is a player once it's been seen in CONFIRM looks in a row (a marker read wrong for a look isn't anyone), and
one is kept until its dot has been missed in MISSES looks in a row and for MISS_S (her own marker passing over
theirs doesn't make them go: the continuous feed looks 30 times a second). A look that didn't see her own dot
(something over the minimap) says nothing of theirs either. Dots are matched to the players of the look before,
nearest first, the same kind, within MATCH_PX and what a player walks meanwhile. A look from the feed brings the
players its own frames followed (`adopt`), so a look taken here after it goes on from them, ids and all.

The minimap shows the whole map, so every look sees every player on it: the attribution of kills and the hunt loop's
piles go by them (localization.tracking, behavior.ledger, behavior.hunt), and its choice of ground can
(hunt.PLAYER_DISCOUNT, off).
"""

from dataclasses import dataclass, replace

from ..perception.minimap import OTHER_SHIFT, Marker
from .pose import Localizer

CONFIRM = 2       # looks in a row a new dot must be seen in to be a player
MISSES = 2        # looks in a row a player's dot must be missing for the player to be gone ...
# ... and for this long (s): with the feed 2 looks are 67 ms, and her marker drawn over theirs as she walks past them
# (11 px a second, a marker 7 px across from the core: the game draws hers last, or theirs, which isn't known) hides it
# for about half a second. A player who goes stays on the map this much longer.
MISS_S = 1.0
LOCAL_IDS = 1_000_001  # where the ids of the players a look taken here finds begin, when a feed's looks bring theirs
MATCH_PX = 4.0    # full-size px a player's dot may move between two looks and be the same one ...
MATCH_PX_S = 15.0  # ... and this many px a second between them (a player walking 125 units/s, 11 px a second, and
                   # jumping) ...
MATCH_MAX_PX = 40.0  # ... up to this; farther, it's a new dot (it went through a portal or teleported), and the old
                     # one goes after MISSES looks
# The kinds of player who hunt against her: anyone not in her party. Her party's kills near her give her a share of
# their EXP, and in the saved frames of 2026-10-21 a leader of her party stood on the safe platform 32 for the five
# minutes between two of them (lie_detector_012504.png, calibrate.png).
RIVALS = ("stranger", "guild", "buddy")
# How far from another player a monster they kill, or a pile of theirs, can be: about a star's reach across (434,
# attack.reach.STAR_REACH; the simulator's other player kills within 400) and a step up or down. A guess: what other
# players' classes reach isn't known.
REACH_X = 400.0
REACH_DY = 100.0


@dataclass
class Player:
    """Another player, as their minimap dot places them."""
    id: int
    kind: str                    # Marker.kind: stranger, party, guild, buddy
    x: float                     # map units (snapped to the platform under the dot, as hers is)
    y: float
    platform: int | None         # the platform under them (None: in the air, on a rope, unsure)
    dot: tuple[float, float]     # their dot's core (full-size px)
    first_seen: float
    last_seen: float
    seen: int = 1                # looks in a row their dot was seen in (to CONFIRM)
    missed: int = 0              # ... and missing in (to MISSES)

    @property
    def rival(self) -> bool:
        """Whether they hunt against her (RIVALS: not her party)."""
        return self.kind in RIVALS

    def near(self, x: float, y: float, dx: float, dy: float) -> bool:
        """Whether they're within `dx` map units across and `dy` up or down of (x, y)."""
        return abs(self.x - x) <= dx and abs(self.y - y) <= dy

    def record(self) -> dict:
        """As telemetry data (a scene's 'players')."""
        return {"id": self.id, "kind": self.kind, "x": round(self.x, 1), "y": round(self.y, 1),
                "platform": self.platform}


class Players:
    """The other players on the map, kept up from every look's minimap (`update`)."""

    def __init__(self, localizer: Localizer, first_id: int = 1):
        self.localizer = localizer
        self.tracks: list[Player] = []  # confirmed and not yet
        self.next_id = first_id
        self._t: float | None = None    # the last look's time

    @property
    def current(self) -> list[Player]:
        """The players as of the last look (confirmed, and not missed MISSES looks in a row): copies, which the
        looks after don't change."""
        return [replace(p) for p in self.tracks if p.seen >= CONFIRM]

    def update(self, t: float, markers: list[Marker]) -> list[Player]:
        """A look at t saw these other players' dots: the players there are now (`current`)."""
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        reach = min(MATCH_MAX_PX, MATCH_PX + MATCH_PX_S * dt)
        pairs = sorted((abs(m.x - p.dot[0]) + abs(m.y - p.dot[1]), i, j)
                       for i, m in enumerate(markers) for j, p in enumerate(self.tracks)
                       if m.kind == p.kind and abs(m.x - p.dot[0]) <= reach and abs(m.y - p.dot[1]) <= reach)
        taken: dict[int, int] = {}  # marker -> track
        for _, i, j in pairs:
            if i not in taken and j not in taken.values():
                taken[i] = j
        kept, seen = [], set(taken.values())
        for j, p in enumerate(self.tracks):
            if j not in seen:
                p.missed += 1
                if p.seen < CONFIRM or (p.missed >= MISSES and t - p.last_seen >= MISS_S):  # a dot seen once, or a
                    continue                                                                   # player gone
            kept.append(p)
        for i, m in enumerate(markers):
            j = taken.get(i)
            if j is None:
                kept.append(self._place(Player(self.next_id, m.kind, 0.0, 0.0, None, (m.x, m.y), t, t), m))
                self.next_id += 1
                continue
            p = self._place(self.tracks[j], m)
            p.last_seen, p.missed = t, 0
            p.seen += 1
        self.tracks = kept
        return self.current

    def adopt(self, t: float, players: list["Player"]) -> None:
        """The players of a look the feed gave at t (as its own frames followed them): the players now, so a look
        taken here after it goes on from them."""
        self.tracks = [replace(p, seen=max(p.seen, CONFIRM), missed=0) for p in players]
        self._t = t

    def _place(self, p: Player, m: Marker) -> Player:
        """Player `p` where marker `m` puts them: their dot as hers would be at the same place (OTHER_SHIFT), placed
        and snapped to a platform as hers is."""
        where = self.localizer.where_dot((m.x - OTHER_SHIFT, m.y - OTHER_SHIFT))
        p.x, p.y, p.dot = where.x, where.y, (m.x, m.y)
        p.platform = None if where.platform is None else where.platform.id
        return p


def rivals(players: list[Player]) -> list[Player]:
    """Those of `players` who hunt against her (Player.rival)."""
    return [p for p in players if p.rival]


def near_rival(players: list[Player], x: float, y: float, dx: float = REACH_X, dy: float = REACH_DY) -> Player | None:
    """Of `players`, the nearest who hunts against her within `dx` map units across and `dy` up or down of (x, y)
    (by default REACH_X and REACH_DY: one who can have killed a monster there), or None."""
    close = [p for p in players if p.rival and p.near(x, y, dx, dy)]
    return min(close, key=lambda p: abs(p.x - x)) if close else None
