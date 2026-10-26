"""The monsters she knows are about that a look didn't find (`unseen`): the perception service's tracks out of sight on
screen (localization.tracking, HIDDEN: hit under her damage numbers, under her own body, or just missed, kept HIDE_S),
and her memory's of the last moments (localization.memory, seen within UNSEEN_S), each where it likeliest is by the
look's time. Her decisions take them as the look's own: what she shoots and keeps off in a fight, where she lands for a
pile, the monsters her trips keep clear of (behavior.hunt, control.navigator).

Why: the detector is nearly blind near her. Live on 2026-10-25 on Tree Dungeon (23:43-23:54, the service's 10 fps
records, lab/telemetry/20261025-210747-perceive.jsonl; D:/MapleHelper-scratch/analysis-1025/damage-skeptic occl.py):
of the frames in which the service followed a monster on her platform, 7-12% found it within 50 units of her, 20-34% at
50-100, 44-55% beyond (sim.world.NEAR_SEEN), and none of the hunter's 1317 searched looks had one within 25 units. 22
of her 34 touches came from a monster no look had in the 0.6 s before; 26 of 34 had one of its kind seen within 150
units in the 3 s before. One Horny Mushroom (the service's track 130) caused 6 of them and a fall to 29: the looks
missed it at 120-150 units for 1.3 s, her trip's plan dropped her memory of it (23 was in view and searched), she
jumped onto 23 beside it, and two looks later her pickup walk went through it.

Where a track is: walked on at its velocity as followed (its kind's speed at most; under STILL_VX it stands), turning
back at its platform's ends (MonsterTrack.at). Where a memory entry is: where it was seen (its walking isn't known; a
trip's chains take it wherever it can have walked since: control.navigator). One the look found (of its kind on its
platform, within what it can have walked since: memory.MATCH_SLACK, MATCH_SPEED) is that one, not another.
"""

import dataclasses

from .mapdata import MapModel
from .memory import MATCH_SLACK, MATCH_SPEED, MATCH_WALK_S, SURE
from .monsters import DEFAULT_SPEED, Monster, MonsterTrack, bodies, speeds
from .scene import Scene
from .tracking import HIDDEN, HIDE_S

# s: her memory's monsters seen this long ago at most count as about (a guess, as navigator.TRIP_MEMORY_S: where one
# can be by then is 200 units either way; longer shuts most of a platform for little). The service's own tracks keep
# theirs HIDE_S (3.5 s), with which way they walked.
UNSEEN_S = 2.0
STILL_VX = 15.0  # units/s: a track slower than this stands where it was last seen (navigator.STILL_VX)


def unseen(s: Scene, memory, model: MapModel, within: float = UNSEEN_S, platforms=None) -> list[Monster]:
    """The monsters about that look `s` didn't find (see the top), on `platforms` (None: any), as Monsters placed where
    each likeliest is at the look's time: the service's hidden tracks (Scene.tracks), then her memory's entries
    (`memory`, a localization.memory.Memory, at memory.SURE; None: none) seen within `within` s, none that a monster
    of the look's (Scene.monsters and Scene.swept) or one before it here is."""
    if s.where is None:
        return []
    found = [m for m in [*s.monsters, *s.swept] if m.platform is not None]
    walking, body = speeds(model), bodies(model)
    out: list[Monster] = []
    for tr in getattr(s, "tracks", ()) or ():
        if tr.state != HIDDEN or tr.platform is None or (platforms is not None and tr.platform not in platforms):
            continue
        age = s.t - tr.last_seen
        if not 0 <= age <= HIDE_S:
            continue
        speed = walking.get(tr.kind, DEFAULT_SPEED)
        if _is_one(tr.kind, tr.platform, tr.x, age, speed, found):
            continue
        p = model.platform(tr.platform)
        b = body.get(tr.kind)
        vx = max(-speed, min(speed, tr.vx)) if abs(tr.vx) > STILL_VX else 0.0
        x = MonsterTrack(tr.platform, tr.x, speed, half=b.half if b else 28.0, vx=vx).at(age, p)
        out.append(Monster(tr.mob, x, p.y_at(x), tr.platform, tr.bar, tr.facing, ""))
    if memory is None:
        return out
    for e in memory.alive(s.t, SURE):
        age = s.t - e.seen
        if not 0 < age <= within or (platforms is not None and e.platform not in platforms):
            continue
        speed = walking.get(e.kind, DEFAULT_SPEED)
        if _is_one(e.kind, e.platform, e.x, age, speed, found + out):
            continue
        out.append(dataclasses.replace(e.monster, pose=""))
    return out


def _is_one(kind: str, platform: int, x: float, age: float, speed: float, found: list[Monster]) -> bool:
    """Whether a monster of `kind` on `platform` last at map x `age` s ago is one of `found` (as memory matches a
    sighting: within MATCH_SLACK and what it can have walked since, MATCH_SPEED x its speed)."""
    slack = MATCH_SLACK + MATCH_SPEED * speed * min(age, MATCH_WALK_S)
    return any(m.kind == kind and m.platform == platform and abs(m.x - x) <= slack for m in found)
