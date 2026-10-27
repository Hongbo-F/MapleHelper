"""The monsters she knows are about that a look didn't find (`unseen`): the perception service's tracks out of sight on
screen (localization.tracking, HIDDEN: hit under her damage numbers, under her own body, or just missed; `hidden`),
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

Not every track out of sight is a monster there (`hidden`; review of 2026-10-26): the feed keeps one HIDDEN while it
waits FADE_S + MISSING_S (1.8 s) to put her EXP gain on it, and one someone else killed, or that it lost, up to HIDE_S
(3.5 s). Live on 2026-10-25 (23:43-23:55, the service's 'tracks' records; D:/MapleHelper-scratch/hunt-ab/B/fix/
hidden_age.py) 45% of its hidden track records on a platform ended with the monster seen again, 12% dead by her, 13%
dead by others and 26% dropped without a death; in the simulator three quarters of the hidden tracks taken here had
no monster there (D:/MapleHelper-scratch/hunt-ab/review-b-correct/phantoms.py). So a track a kill of hers is on
(`kills`: the ledger's, placed: _hers) isn't about, nor one out of sight longer than TRACK_S.
"""

import dataclasses

from .mapdata import MapModel
from .memory import MATCH_SLACK, MATCH_SPEED, MATCH_WALK_S, SURE
from .monsters import DEFAULT_SPEED, Monster, MonsterTrack, bodies, speeds
from .scene import Scene
from .tracking import FADE_S, HIDDEN, LIVE, MISSING_S, STAR_WINDOW

# s: her memory's monsters seen this long ago at most count as about (a guess, as navigator.TRIP_MEMORY_S: where one
# can be by then is 200 units either way; longer shuts most of a platform for little). The service's own tracks count
# TRACK_S, with which way they walked.
UNSEEN_S = 2.0
STILL_VX = 15.0  # units/s: a track slower than this stands where it was last seen (navigator.STILL_VX)
# s: a track out of sight this long at most counts as about (`hidden`). Of the hidden track records on a platform of
# the live hunt of 2026-10-25 (above) with no kill of hers on them, by the time since the track's last record that had
# it seen (records 0.53 s apart: 0-0.5 s more than since it was seen), 79% ended with the monster seen again at
# 0.5-1.0 s, 65% at 1.0-1.5, 53% at 1.5-2.0, 38% at 2.0-2.5, 29% at 2.5-3.0, 16% at 3.0-3.5 (the rest dead by others or
# dropped): up to where one was still about as likely there as not. One hidden under her own damage numbers longer is the
# fight's (fight.HIDDEN_FOR) and her memory's (UNSEEN_S; its covered misses don't count while she throws its way).
TRACK_S = 1.5
# units: a kill of hers is on the track of its kind on its platform nearest where the ledger placed it (where a look
# last saw its monster, followed throw to throw) within this: the frames saw it up to a look later (0.2-0.5 s, 20-50
# units at a mushroom's walk), and its last hit knocked it 18-25 units (a guess past those)
KILL_NEAR = 80.0


def unseen(s: Scene, memory, model: MapModel, within: float = UNSEEN_S, platforms=None, kills=()) -> list[Monster]:
    """The monsters about that look `s` didn't find (see the top), on `platforms` (None: any), as Monsters placed where
    each likeliest is at the look's time: the service's hidden tracks that count (`hidden`: none that a kill of hers of
    `kills` is on), then her memory's entries (`memory`, a localization.memory.Memory, at memory.SURE; None: none) seen
    within `within` s, none that a monster of the look's (Scene.monsters and Scene.swept) or one before it here is."""
    if s.where is None:
        return []
    found = [m for m in [*s.monsters, *s.swept] if m.platform is not None]
    walking, body = speeds(model), bodies(model)
    out: list[Monster] = []
    for tr in hidden(s, kills, platforms):
        age = s.t - tr.last_seen
        speed = walking.get(tr.kind, DEFAULT_SPEED)
        if _is_one(tr.kind, tr.platform, tr.x, age, speed, found):
            continue
        p = model.platform(tr.platform)
        b = body.get(tr.kind)
        vx = max(-speed, min(speed, tr.vx)) if abs(tr.vx) > STILL_VX else 0.0
        x = MonsterTrack(tr.platform, tr.x, speed, half=b.half if b else 28.0, vx=vx).at(age, p)
        out.append(Monster(tr.mob, x, p.y_at(x), tr.platform, tr.bar, tr.facing, "", tr.id))
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


def hidden(s: Scene, kills=(), platforms=None) -> list:
    """The feed's tracks out of sight on screen as of look `s` (Scene.tracks, tracking.Track) that count as a monster
    about (see the top), on `platforms` (None: any): HIDDEN on a platform, last seen within TRACK_S, none that a kill of
    hers (`kills`: the ledger's, behavior.ledger.Kill, the newest last) is on (_hers)."""
    tracks = getattr(s, "tracks", ()) or ()
    if not tracks:
        return []
    dead = _hers(tracks, kills, s.t)
    return [tr for tr in tracks if tr.state == HIDDEN and tr.platform is not None and tr.id not in dead
            and not getattr(tr, "owed", False) and (platforms is None or tr.platform in platforms)
            and 0 <= s.t - tr.last_seen <= TRACK_S]


def _hers(tracks, kills, t: float) -> set[int]:
    """The ids of `tracks` that kills of hers are on: for each one placed (`kills`, the newest last) recently enough for
    its track to count still (STAR_WINDOW, FADE_S + MISSING_S and TRACK_S before `t`), its own (behavior.ledger.Kill.
    track: the tracker's), else the track of its kind on its platform nearest where it was placed, within KILL_NEAR,
    seen about when it was killed (from STAR_WINDOW before the look that saw her EXP go up to FADE_S + MISSING_S after
    it: it fades where it died), one not seen live first (fading, out of sight, or found dead). One kill, one track."""
    out: set[int] = set()
    since = t - (STAR_WINDOW + FADE_S + MISSING_S + TRACK_S)
    for k in reversed(kills or ()):
        if k.t < since:
            break
        if (own := getattr(k, "track", None)) is not None:
            if own:
                out.add(own)
            continue
        if k.kind is None or k.platform is None:
            continue
        near = [tr for tr in tracks if tr.id not in out and tr.kind == k.kind and tr.platform == k.platform
                and abs(tr.x - k.x) <= KILL_NEAR and k.t - STAR_WINDOW <= tr.last_seen <= k.t + FADE_S + MISSING_S]
        if near:
            out.add(min(near, key=lambda tr: (tr.state == LIVE, abs(tr.x - k.x))).id)
    return out


def _is_one(kind: str, platform: int, x: float, age: float, speed: float, found: list[Monster]) -> bool:
    """Whether a monster of `kind` on `platform` last at map x `age` s ago is one of `found` (as memory matches a
    sighting: within MATCH_SLACK and what it can have walked since, MATCH_SPEED x its speed)."""
    slack = MATCH_SLACK + MATCH_SPEED * speed * min(age, MATCH_WALK_S)
    return any(m.kind == kind and m.platform == platform and abs(m.x - x) <= slack for m in found)
