"""A movement-only walk: random trips over the map, one after another, in one navigator that avoids monsters (no attack,
no escort). The live QA walk of 2026-10-28 (D:/MapleHelper-scratch/qa/walk.py: `nav` started again for each trip, its
platforms 7-33 of Tree Dungeon) as one process (`python -m gamelab nav --walk MINUTES`: the navigator's tracker, memory and
trips' state kept from trip to trip, as a hunt keeps them), and the harness's `walk` policy (sim.harness._walk), the
same trips drawn the same way.

Each trip: a platform drawn from PLATFORMS on Tree Dungeon (elsewhere any MIN_WIDTH wide), an x on it EDGE in from its
ends, TRIP_S to get there. `save()` every SAVE_S (the live walk's rolling recording, save-recent); `nav --walk` parks her
at the end (behavior.retreat.park).
"""

import random
import time
from collections.abc import Callable

TREE = 10002073
PLATFORMS = range(7, 34)  # (Tree Dungeon: the live QA walk's)
MIN_WIDTH = 60.0
EDGE = 15.0
TRIP_S = 40.0
SAVE_S = 270.0


def places(model) -> list:
    """The platforms a walk's trips go to on `model`'s map."""
    return [p for p in model.places if (p.id in PLATFORMS if model.id == TREE else p.right - p.left >= MIN_WIDTH)]


def walk(nav, model, rng: random.Random, minutes: float | None = None, trip_s: float = TRIP_S,
         log: Callable[[str], None] | None = None, save: Callable[[], None] | None = None,
         save_every: float = SAVE_S) -> list:
    """Trips of navigator `nav` (one that avoids monsters) to random places (`places`, drawn with `rng`), `trip_s` each,
    until `minutes` have passed (None: until something stops it, as the harness's horizon does); `save()` every
    `save_every` s, if given. Each trip said (`log`, if given). Returns the trips (navigator.Trip)."""
    targets = places(model)
    end = None if minutes is None else time.monotonic() + 60 * minutes
    saved = time.monotonic()
    trips = []
    while end is None or time.monotonic() < end:
        p = rng.choice(targets)
        x = rng.uniform(p.left + EDGE, p.right - EDGE)
        if log is not None:
            log(f"walk: trip {len(trips) + 1} to platform {p.id} at x {x:.0f}")
        trip = nav.go(p.id, x=x, seconds=trip_s)
        trips.append(trip)
        if log is not None:
            log(f"walk: trip {len(trips)} {trip.result} ({len(trip.moves)} moves, {trip.hits} touched on the way, "
                f"{trip.waits} waits)")
        if save is not None and time.monotonic() - saved >= save_every:
            save()
            saved = time.monotonic()
    return trips
